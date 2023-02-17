import broyden1 from "./index.js";

function func(x) {
  return [(x-3)**2]
}

const x0 = 2;
const root = broyden1(func, x0, null, null, 'restart');


console.log("root", root);
