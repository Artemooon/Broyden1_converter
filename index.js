const {SVD} = require('svd-js');
const {transpose, conj, qr, inv, norm, lusolve, dot} = require('mathjs');
const { identity, zeros, array, dot: num_dot, reshape, NdArray, add} = require('numjs');
const blas = require('blas');

function convertToNdArray(arr) {
  if (arr instanceof NdArray) {
    return arr;
  }
  if (!arr) {
    return [];
  }
  else if (arr.length === 0) {
    return arr;
  }
  else {
    return array(arr);
  }
}

function getNdArrayData(arr){
  if(arr instanceof NdArray) {
    return arr.selection.data;
  }
  return arr;
}

function arraysZip(...arrays) {
  const minLength = Math.min(...arrays.map(array => array.length));
  const zipped = [];
  for (let i = 0; i < minLength; i++) {
    const innerArray = arrays.map(array => array[i]);
    zipped.push(innerArray);
  }
  return zipped;
}

class TerminationCondition {
  constructor(
      f_tol = Number.EPSILON ** (1/3),
      f_rtol = Number.POSITIVE_INFINITY,
      x_tol = Number.POSITIVE_INFINITY,
      x_rtol = Number.POSITIVE_INFINITY,
      iter = null,
      norm = maxnorm
  ) {
    this.x_tol = x_tol;
    this.x_rtol = x_rtol;
    this.f_tol = f_tol;
    this.f_rtol = f_rtol;

    this.norm = norm;

    this.iter = iter;

    this.f0_norm = null;
    this.iteration = 0;
  }

  check(f, x, dx) {
    this.iteration++;
    let f_norm = this.norm(f);
    let x_norm = this.norm(x);
    let dx_norm = this.norm(dx);

    if (this.f0_norm === null) {
      this.f0_norm = f_norm;
    }

    if (f_norm === 0) {
      return 1;
    }

    if (this.iter !== null) {
      return 2 * (this.iteration > this.iter);
    }

    return Number(
        (f_norm <= this.f_tol && f_norm/this.f_rtol <= this.f0_norm) &&
        (dx_norm <= this.x_tol && dx_norm/this.x_rtol <= x_norm)
    );
  }
}

class LowRankMatrix {
  constructor(alpha, n, dtype) {
    this.alpha = alpha;
    this.cs = [];
    this.ds = [];
    this.n = n;
    this.dtype = dtype;
    this.collapsed = null;
  }

  static _matvec(v, alpha, cs, ds) {
    let w = array(v).multiply(alpha);
    console.log("v in _matvec", v);
    console.log("alpha in _matvec", alpha);

    ds.map(item => console.log("ds item", item.selection.data))
    cs.map(item => console.log("cs item", item.selection.data))

      arraysZip(cs, ds).map(item => {
        let [c, d] = item;
        const n_ddot = d.size;
        let a = blas.ddot(n_ddot, getNdArrayData(d), 1, 0, v, 1, 0);
        // w = a * w * c;
        // daxpy = (int *n, d *da, d *dx, int *incx, d *dy, int *incy) in Python
        w = blas.daxpy(c.size, a, getNdArrayData(c), 1, 0, getNdArrayData(w), 1, 0);
      })
    console.log('www', w);
    return getNdArrayData(w);
  }

  static _solve(v, alpha, cs, ds) {
    let w;
    if (cs.length === 0 || (cs.size && cs.size === 0)) {
      w = array(v).divide(alpha);
      return w.selection.data;
    }
    // (B + C D^H)^-1 = B^-1 - B^-1 C (I + D^H B^-1 C)^-1 D^H B^-1

    let c0 = cs[0];
    let A = identity(cs.length, typeof (c0)).multiply(alpha);
    for (const [i, d] of ds.etries()) {
      for (const [j, c] in cs.entries()) {
        // n_ddot, d, 1, 0, v, 1, 0
        A.set(i, j, A.add(blas.ddot(d.length, d, 1, 0, c, 1, 0)));
      }
    }
    let q = zeros(cs.length, typeof (c0));
    for (const [j, d] in ds.entries()) {
      q.set(i, null, q.assign(blas.ddot(d.length, d, 1, 0, v, 1, 0)));
    }
    q = q.divide(alpha);
    q = lusolve(A, q.selection.data);
    w = array(v).divide(alpha);
    for (const [c, qc] of arraysZip(cs, q)) {
      // w.size, a, c, 1, 0, w, 1, 0
      return blas.daxpy(w.size, -qc, c, 1, 0, w, 1, 0);
    }
  }

  matvec(v) {
    // Evaluate w = M v
    if (this.collapsed !== null) {
      console.log("this.collapsed", this.collapsed);
      console.log("v", v);
      return num_dot(this.collapsed, v);
    }
    console.log("this.cs, this.ds in matvec", this.cs, this.ds);
    return LowRankMatrix._matvec(v, this.alpha, this.cs, this.ds);
  }

  rmatvec(v) {
    // Evaluate w = M^H v
    if (this.collapsed) {
      console.log("transpose(conj(getNdArrayData(this.collapsed)))", transpose(conj(getNdArrayData(this.collapsed))));
      return num_dot(transpose(conj(getNdArrayData(this.collapsed))), v);
    }
    return LowRankMatrix._matvec(v, conj(this.alpha), this.ds, this.cs);
  }

  solve(v, tol = 0) {
    // Evaluate w = M^-1 v
    if (this.collapsed) {
      return lusolve(this.collapsed, v);
    }
    return LowRankMatrix._solve(v, this.alpha, this.cs, this.ds);
  }

  rsolve(v, tol = 0) {
    // Evaluate w = M^-H v
    if (this.collapsed) {
      return lusolve(transpose(conj(this.collapsed)), v);
    }
    return LowRankMatrix._solve(v, conj(this.alpha), this.ds, this.cs);
  }

  append(c, d) {
    if (this.collapsed) {
      const reshaped_c = c.reshape(c.size, 1);
      const reshaped_d = d.reshape(1, d.size);
      this.collapsed = convertToNdArray(this.collapsed).add(reshaped_c.multiply(conj(getNdArrayData(reshaped_d))));
      return;
    }
    // concatenate two arrays!
    this.cs.push(c);
    this.ds.push(d);
    const cs_length = this.cs.length;
    if (cs_length > c.size) {
      this.collapse();
    }
  }

  collapseLowRankMatrixToNdArray() {
    if (this.collapsed) {
      return this.collapsed;
    }

    let Gm = identity(this.n).multiply(this.alpha);

    let loc_counter = 0;
    for (const [c, d] of arraysZip(this.cs, this.ds)) {
      console.log("arraysZip(this.cs, this.ds)", arraysZip(this.cs, this.ds))
      loc_counter += 1;
      console.log("loc_counter", loc_counter);
      console.log("c", c);
      console.log("d", d);
      console.log("c.reshape([0, c.size], null))", c.reshape(c.size, 1));

      const reshaped_c = c.reshape(c.size, 1);
      const reshaped_d = d.reshape(1, d.size);
      const c_matrix = this.creatreMatrixFromArray(getNdArrayData(reshaped_c), reshaped_c.selection.shape[0], reshaped_c.selection.shape[1])
      const d_matrix = this.creatreMatrixFromArray(getNdArrayData(reshaped_d), reshaped_d.selection.shape[0], reshaped_d.selection.shape[1])
      console.log("c_matrix", c_matrix);
      console.log("d_matrix", conj(d_matrix));
      // console.log("conj(d_matrix)", conj(d_matrix));
      const matricies_mult = this.multiplyMatricies(c_matrix, conj(d_matrix));
      console.log("Error after it!");
      console.log("GM", Gm);
      console.log("matricies_mult", matricies_mult)
      Gm = array(Gm).add(array(matricies_mult));
      Gm = this.creatreMatrixFromArray(getNdArrayData(Gm),  Gm.selection.shape[0], Gm.selection.shape[1]);
      console.log("Gm final matrix!!!", Gm);
    }
    return Gm
  }

  multiplyMatricies(matrix1, matrix2) {
    const result_arr = []
    console.log("matrix1[0]", matrix1);
    matrix1.map((arr1_item) => {
      const temp_arr = []
      matrix2[0].map(arr2_item => {
        temp_arr.push(arr1_item[0]*arr2_item);
      })
      result_arr.push(temp_arr);
    })
    return result_arr;
  }


  collapse() {
    // Collapse the low-rank matrix to a full-rank one.

    this.collapsed = this.collapseLowRankMatrixToNdArray();
    this.cs = null;
    this.ds = null;
    this.alpha = null;
  }

  restart_reduce(rank) {
    // Reduce the rank of the matrix by dropping all vectors.
    if (this.collapsed !== null) {
      return null;
    }
    console.assert(rank > 0);
    console.log("this.cs.size", this.cs.size);
    console.log("this.ds", this.ds);
    if (this.cs.size > rank) {
      this.cs.splice(0, this.cs.length);
      this.ds.splice(0, this.ds.length);
    }
  }

  simple_reduce(rank) {
    // Reduce the rank of the matrix by dropping oldest vectors.
    if (this.collapsed !== null) {
      return null;
    }
    console.assert(rank > 0);
    while (this.cs.size > rank) {
      this.cs.splice(0, 1);
      this.ds.splice(0, 1);
    }
  }


  svd_reduce(max_rank, to_retain = null) {

    if (this.collapsed !== null) {
      return null;
    }

    let p = max_rank;
    let a;

    if (to_retain !== null) {
      a = to_retain;
    } else {
      a = p - 2;
    }

    if (this.cs.size) {
      p = Math.min(p, this.cs.get(0));
    }
    a = Math.max(0, Math.min(a, p - 1));

    let m = this.cs.size;

    if (m < p) {
      // nothing to do
      return;
    }

    let C = transpose(this.cs);
    let D = transpose(this.ds);

    let { Q, R } = qr(D);
    const RConj = conj(transpose(R));

    C = dot(C, RConj);

    C = this.creatreMatrixFromArray(C.selection.data, C.selection.shape[0], C.selection.shape[1]);
    let { u, q, v } = SVD(C);


    C = this.creatreMatrixFromArray(array(C).selection.data, array(C).selection.shape[0], array(C).selection.shape[1]);
    v = this.creatreMatrixFromArray(array(v).selection.data, array(v).selection.shape[0], array(v).selection.shape[1]);

    C = dot(C, inv(v));
    D = this.creatreMatrixFromArray(array(Q).selection.data, array(Q).selection.shape[0], array(Q).selection.shape[1]);
    D = dot(D, conj(transpose(v)));


    C = this.creatreMatrixFromArray(array(C).selection.data, array(C).selection.shape[0], array(C).selection.shape[1]);
    D = this.creatreMatrixFromArray(array(D).selection.data, array(D).selection.shape[0], array(D).selection.shape[1]);


    for (let k = 0; k < a; k++) {
      this.cs[k] = C.map(col => col[k]).slice();
      this.ds[k] = D.map(col => col[k]).slice();
    }

    this.cs.splice(0, a);
    this.ds.splice(0, a);
  }

  creatreMatrixFromArray(array, rows, columns) {
    const matrix = [];
    const elementsPerRow = columns;
    for (let i = 0; i < rows * columns; i += elementsPerRow) {
      matrix.push(array.slice(i, i + elementsPerRow));
    }
    return matrix;
  }
}

class Jacobian {
  constructor(options) {
    this.shape = null;
    this.func = null;
    this.dtype = null;


    const names = ["solve", "update", "matvec", "rmatvec", "rsolve", "matmat", "todense", "shape", "dtype"];

    if (options?.length) {
      for (let name in options) {
        if (!names.includes(name)) {
          throw new Error(`Unknown keyword argument ${name}`);
        } else {
          this[name] = options[name];
        }
      }
    }

  }

  solve(v, tol = 0) {
    throw new Error("NotImplementedError");
  }

  update() {
    // pass
  }

  setup(x, F, func) {
    this.func = func;
    this.shape = [F.size, x.size];
    this.dtype = F.dtype;
    if (Object.is(this.setup.constructor.name, Jacobian.setup)) {
    // Call on the first point unless overridden
      this.update(x, F)
    }
    return this;
  }
}


//===================================================================================================================


class GenericBroyden extends Jacobian {

  constructor() {
    super();
    this.last_f = null;
    this.last_x = null;

  }

  setupGeneric(x0, f0, func) {
    this.setup(x0, f0, func);
    this.last_f = f0;
    this.last_x = x0;

    if (!this.alpha) {
      // Autoscale the initial Jacobian parameter
      // unless we have already guessed the solution.
      const normf0 = norm(f0.selection.data);
      if (normf0) {
        this.alpha = 0.5 * Math.max(norm(x0.selection.data), 1) / normf0;
      } else {
        this.alpha = 1.0
      }
    }
    return this;
  }


  calculateMatrix(f, s, d = 1) {
    return f.map((i, idx) => i + d*s[idx])
  }

  update(x, f) {
    console.log('this.last_x', this.last_x)
    console.log('this.last_f', this.last_f)

    const df = this.calculateMatrix(f, this.last_f.selection.data, -1)
    const dx = this.calculateMatrix(x, this.last_x.selection.data, -1)

    this._update(x, f, dx, df, norm(dx), norm(df))

    this.last_x = array(x)
    this.last_f = array(f)
  }
}

function maxnorm(x) {
  return Math.max(x)
}


function asjacobian(J) {
  if (J instanceof Jacobian) {
    return J;
  } else if (typeof J === "string") {
    return {
      broyden1: BroydenFirst,
      // broyden2: BroydenSecond,
      // anderson: Anderson,
      // diagbroyden: DiagBroyden,
      // linearmixing: LinearMixing,
      // excitingmixing: ExcitingMixing,
      // krylov: KrylovJacobian,
    }[J](); }
  else {
    throw new TypeError("Cannot convert object to a Jacobian");
  }
}


class BroydenFirst extends GenericBroyden {
  constructor(alpha = null, reduction_method = "restart", max_rank) {
    super();
    this.alpha = alpha;
    this.Gm = null;

    if (!max_rank) {
      max_rank = Infinity;
    }
    this.max_rank = max_rank;

    let reduce_params = []

    if (typeof reduction_method === "string") {
      reduce_params = [];
    } else {
      reduce_params = reduction_method.slice(1);
      reduction_method = reduction_method[0];
    }
    reduce_params = [this.max_rank - 1, ...reduce_params];

    if (reduction_method === "svd") {
      this._reduce = () => this.Gm.svd_reduce(...reduce_params);
    } else if (reduction_method === "simple") {
      this._reduce = () => this.Gm.simple_reduce(...reduce_params);
    } else if (reduction_method === "restart") {
      this._reduce = () => this.Gm.restart_reduce(...reduce_params);
    } else {
      throw new Error("Unknown rank reduction method: " + reduction_method);
    }
  }


  setupBroydenFirst(x, F, func) {
    this.setupGeneric(x, F, func);
    //console.log("this.alpha", this.alpha);
    // const alpha = this.alpha.selection.data.map(el => +`-${el}`)
    const alpha = this.alpha;
    //console.log(alpha, 'alfa')
    this.Gm = new LowRankMatrix(-alpha, this.shape[0], this.dtype);
    //console.log(this.Gm, 'this.gm')
    //return this;
  }

  solve(f, tol = 0) {
    let r = this.Gm.matvec(f.selection.data);
    if ([Infinity].includes(r)) {
      console.log("in Infinity")
      // singular; reset the Jacobian approximation
      this.setupBroydenFirst(this.last_x, this.last_f, this.func);
      console.log("this.matvec(f)", this.Gm.matvec(f));
      return this.Gm.matvec(f);
    }
    console.log(r, "solve FINISH");
    return r;
  }

  matvec(f) {
    return this.Gm.solve(f);
  }


  rsolve(f) {
    return this.Gm.rmatvec(f);
  }


  rmatvec(f) {
    return this.Gm.rsolve(f);
  }


  _update(x, f, dx, df, dx_norm, df_norm) {

    this._reduce() // reduce first to preserve secant condition
    let v = this.Gm.rmatvec(dx);
    const gmMatvecData = this.Gm.matvec(df);
    // const c = dx.map((item, idx) => item - gmMatvecData[idx])
    const c = array(dx).subtract(gmMatvecData);
    const d = array(v).divide(num_dot(df, v).selection.data[0]);
    console.log("Before hard divide!!");

    console.log('VVV', v, c, d)

    // this.concat(c, d);

    this.Gm.append(c, d);
  }

}

function _as_inexact(x) {
  const res = array(x)
  return res
}

function _safe_norm(v) {
  if ([Infinity].includes(v)) {
    return Array().fill(Infinity);
  }
  return  norm(v)
}


function _nonlin_line_search(func, x, Fx, dx, search_type='armijo', rdiff=1e-8, smin=1e-2) {
  const tmp_s = [0]
  const tmp_Fx = [Fx]
  const tmp_phi = [norm(getNdArrayData(Fx))**2]
  if (x.selection) {
    x = getNdArrayData(x)
  }
  const s_norm = norm(x)/norm(dx)

  function phi(s, store= true) {
    if (s === tmp_s[0]) {
      return tmp_phi[0]
    }

    const sdx = dx.map(item => item*s)
    const xt = x.map((item, idx) => item + sdx[idx])

    const v = func(xt)
    const p = _safe_norm(v.selection.data)**2

    if (store) {
      tmp_s[0] = s
      tmp_phi[0] = p
      tmp_Fx[0] = v
    }
    return p
  }

  let s
  let phi1

  if (search_type === 'armijo') {
    let [s, phi1] = scalar_search_armijo(phi, tmp_phi[0], -tmp_phi[0], 1e-4, 1, smin)
  }

  if (!s) {
    s = 1.0
  }

  const sdx = array(dx).multiply(s);
  x = array(x).add(sdx);
  x = getNdArrayData(x);

  if (s === tmp_s[0]) {
    Fx = tmp_Fx[0]
  } else {
    Fx = func(x)
  }

  const Fx_norm = norm(getNdArrayData(Fx))

  return [s, x, Fx, Fx_norm]
}


function _array_like(x, x0) {
  // console.log(x, x0, '_array_like START')
  x = reshape(x, x0.shape)
  const res = array(x)
  //  console.log(res, '_array_like FINISH')
  return res.selection.data
}


function nonlin_solve(F, x0, jacobian='broyden1', iter=null, verbose=false,
                      maxiter=null, f_tol=null, f_rtol=null, x_tol=null, x_rtol=null,
                      tol_norm=null, line_search='armijo', callback=null,
                      full_output=true, raise_exception=true) {
  //let tol_norm = tol_norm || maxnorm;
  tol_norm = tol_norm ? tol_norm : maxnorm;
  let condition = new TerminationCondition(f_tol, f_rtol, x_tol, x_rtol, iter, tol_norm);
  x0 = _as_inexact(x0);

  let func = (z) => _as_inexact(F(_array_like(z, x0)));
  let x = x0;
  let dx = Array(x.length).fill(Infinity);
  let Fx = func(x);
  let Fx_norm = norm(Fx.selection.data);
  jacobian = asjacobian(jacobian);
  jacobian.setupBroydenFirst(x.slice(), Fx, func);

  maxiter = maxiter || (iter ? iter + 1 : 100 * (x.size + 1));

  if (![null, 'armijo'].includes(line_search)) { //'wolfe'].includes(line_search)) {
    throw new Error("Invalid line search");
  }
  let gamma = 0.9;
  let eta_max = 0.9999;
  let eta_treshold = 0.1;
  let eta = 0.001;


  let status_failed = false;
  let status;
  let counter = 0;
  for (let n = 0; n < maxiter; n++) {
    counter += 1;
    console.log("Iter n", counter);
    status = condition.check(Fx, x, dx);
    if (status) {
      status_failed = true;
      break;
    }

    let tol = Math.min(eta, eta * Fx_norm);


    dx = getNdArrayData(jacobian.solve(Fx, tol));
    dx = dx.map(function(element) {
      return -element;
    });
    console.log('DX, ', dx)

    if (norm(dx) === 0) {
      throw new Error("Jacobian inversion yielded zero vector. This indicates a bug in the Jacobian approximation.");
    }
    //console.log("Before Fx_norm_new");
    let Fx_norm_new;
    let s;
    if (line_search) {
      console.log("Data before res in line_search", func, x, Fx, dx, line_search);
      const res = _nonlin_line_search(func, x, Fx, dx, line_search);
      s = res[0]
      x = res[1]
      Fx = res[2]
      Fx_norm_new = res[3]
      console.log("Data after res in line_search", res[0], res[1], res[2], res[3]);
    } else {
      s = 1.0;
      x = [...x, ...dx];
      Fx = func(x);
      Fx_norm_new = norm(Fx);
    }

    jacobian.update(x.slice(), getNdArrayData(Fx));

    if (callback) {
        callback(x, getNdArrayData(Fx));
    }

    // Adjust forcing parameters for inexact methods
    let eta_A = gamma * Fx_norm_new ** 2 / Fx_norm ** 2;
    if (gamma * eta ** 2 < eta_treshold) {
      eta = Math.min(eta_max, eta_A)
    } else {
      eta = Math.min(eta_max, Math.max(eta_A, gamma * eta ** 2))
    }

    Fx_norm = Fx_norm_new

    console.log('END', eta_A, gamma, eta, eta_max, Fx_norm, eta_treshold)

    // Print status
    if (verbose) {
        console.log(n, tol_norm(Fx), s, "VERBOSEEE")
    }
  }
  if (status_failed) {
      if (raise_exception) {
          throw new Error(_array_like(x, x0));
      } else {
          status = 2;
      }
  }

  if (full_output) {
    info = {
      'nit': condition.iteration,
      'fun': Fx,
      'status': status,
      'success': status === 1,
      'message': {
        1: 'A solution was found at the specified tolerance.',
        2: 'The maximum number of iterations allowed has been reached.'
      }[status]
    }
    console.log("we are there!!", x, x0);
    return [_array_like(x, x0), info]
  } else {
    return _array_like(x, x0)
  }
}


function scalar_search_armijo(phi, phi0, derphi0, c1=1e-4, alpha0=1, amin=0) {

  // Minimize over alpha, the function ``phi(alpha)``.
  // Uses the interpolation algorithm (Armijo backtracking) as suggested by
  // Wright and Nocedal in 'Numerical Optimization', 1999, pp. 56-57
  // alpha > 0 is assumed to be a descent direction.
  // Returns
  // -------
  // alpha
  // phi1

  let phi_a0 = phi(alpha0);

  if (phi_a0 <= phi0 + c1 * alpha0 * derphi0) {
    return [alpha0, phi_a0];
  }

  // Otherwise, compute the minimizer of a quadratic interpolant:

  let alpha1 = -(derphi0) * alpha0**2 / 2 / (phi_a0 - phi0 - derphi0 * alpha0);
  let phi_a1 = phi(alpha1);

  if (phi_a1 <= phi0 + c1 * alpha1 * derphi0) {
    return [alpha1, phi_a1];
  }

  // Otherwise, loop with cubic interpolation until we find an alpha which
  // satisfies the first Wolfe condition (since we are backtracking, we will
  // assume that the value of alpha is not too small and satisfies the second
  // condition.

  while (alpha1 > amin) {
    let factor = alpha0**2 * alpha1**2 * (alpha1-alpha0);
    let a = alpha0**2 * (phi_a1 - phi0 - derphi0*alpha1) - alpha1**2 * (phi_a0 - phi0 - derphi0*alpha0);
    a = a / factor;
    let b = (-alpha0)**3 * (phi_a1 - phi0 - derphi0*alpha1) + alpha1**3 * (phi_a0 - phi0 - derphi0*alpha0);
    b = b / factor;

    let alpha2 = (-b + Math.sqrt(Math.abs(b**2 - 3 * a * derphi0))) / (3 * a);
    let phi_a2 = phi(alpha2);

    if (phi_a2 <= phi0 + c1 * alpha2 * derphi0) {
      return [alpha2, phi_a2];
    }

    if ((alpha1 - alpha2) > alpha1 / 2 || (1 - alpha2/alpha1) < 0.96) {
      alpha2 = alpha1 / 2;
    }

    alpha0 = alpha1;
    alpha1 = alpha2;
    phi_a0 = phi_a1;
    phi_a1 = phi_a2;
  }
  return [null, phi_a1];
}

// TEST
function broyden1(F, xin, iter=null, alpha=null, reduction_method='restart',
                  max_rank=null, verbose=false, maxiter=null, f_tol=null, f_rtol=null, x_tol=null, x_rtol=null,
                  tol_norm=null, line_search='armijo', callback=null, options) {
  const jac = new BroydenFirst(alpha, reduction_method, max_rank);
  return nonlin_solve(F, xin, jac, iter, verbose, maxiter,
      f_tol, f_rtol, x_tol, x_rtol, tol_norm, line_search,
      callback)
}

function func(x) {
  const res = array([x[0]**2 + x[1]**2 - 1, x[0] - x[1] - 1])
  return res
}

const x0 = [1.5, 1.5];
const root = broyden1(func, x0);


console.log("root", root);