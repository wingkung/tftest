var tf = require('@tensorflow/tfjs');

var xs = [];
var ys = [];
for (var i = 0; i < 100; i++) {
    x  = Math.random();
    y = x * 0.43;
    xs.push(x);
    ys.push(y);
}
xs = tf.tensor1d(xs);
ys = tf.tensor1d(ys);
const a = tf.scalar(Math.random()).variable();
const f = x => a.mul(x);
const loss = (pred, label) => pred.sub(label).square().mean();

const optimizer = tf.train.sgd(0.5);

for (let i = 0; i < 201; i++) {
    optimizer.minimize(() => loss(f(xs), ys));
}

const preds = f(xs).dataSync();
preds.forEach((pred, i) => {
    console.log(`x: ${i}, pred: ${pred}, a: ${a.dataSync()}`);
});