const broyden1 = require("./index");

function func(x) {
  return [x[0]**2 + x[1]**2 - 1, x[0] - x[1] - 1]
}

const x0 = [1, 1];
const root = broyden1(func, x0, null, null, 'simple');


console.log("root", root);
