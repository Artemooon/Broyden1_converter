function dcstep(stx, fx, dx, sty, fy, dy, stp, fp, dp, brackt, stpmin, stpmax) {
  const sgnd = dp * (dx / Math.abs(dx));
  const theta = 3.0 * (fx - fp) / (stp - stx) + dx + dp;
  const s = Math.max(Math.abs(theta), Math.abs(dx), Math.abs(dp));
  let gamma = s * Math.sqrt((theta / s) ** 2 - (dx / s) * (dp / s));

  if (fp > fx) {
    if (stp < stx) {
      gamma = -gamma;
    }
    const p = (gamma - dx) + theta;
    const q = ((gamma - dx) + gamma) + dp;
    const r = p / q;
    let stpc = stx + r * (stp - stx);
    const stpq = stx + (dx / ((fx - fp) / (stp - stx) + dx)) / 2.0 * (stp - stx);
    if (Math.abs(stpc - stx) < Math.abs(stpq - stx)) {
      stpf = stpc;
    } else {
      stpf = stpc + (stpq - stpc) / 2.0;
    }
    brackt = true;
  } else if (sgnd < 0) {
    if (stp > stx) {
      gamma = -gamma;
    }
    const p = (gamma - dp) + theta;
    const q = ((gamma - dp) + gamma) + dx;
    const r = p / q;
    let stpc = stp + r * (stx - stp);
    const stpq = stp + (dp / (dp - dx)) * (stx - stp);
    if (Math.abs(stpc - stp) > Math.abs(stpq - stp)) {
      stpf = stpc;
    } else {
      stpf = stpq;
    }
    brackt = true;
  } else if (Math.abs(dp) < Math.abs(dx)) {
    let stpc;
    if (r < 0 && gamma != 0) {
      stpc = stp + r * (stx - stp);
    } else if (stp > stx) {
      stpc = stpmax;
    } else {
      stpc = stpmin;
    }
    const stpq = stp + (dp / (dp - dx)) * (stx - stp);
  }
}