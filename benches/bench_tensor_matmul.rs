use adtensor::tensor::{Shape, Tensor, Matmul};
use criterion::*;
use ndarray::{Array2, Array4, Array5};

const K: usize = 28*28;
const M: usize = 64;
const N: usize = 10;
const W: usize = 28;
const S: usize = 3; 

fn criterion_benchmark(c: &mut Criterion) {
  c.bench_function(&format!("tensor mul [{} {}] x [{} {}]", M, K, K, N), |b| {
    let x = Tensor::<f32>::with_fn(Shape::from([M, K]), &mut || 0.1);
    let y = Tensor::<f32>::with_fn(Shape::from([K, N]), &mut || 0.1);
    b.iter(|| x.mm(&y));  
  });
  c.bench_function(&format!("array mul [{} {}] x [{} {}]", M, K, K, N), |b| {
    let x = Array2::<f32>::from_shape_fn((M, K), &mut |_| 0.1);
    let y = Array2::<f32>::from_shape_fn((K, N), &mut |_| 0.1);
    b.iter(|| x.dot(&y));  
  });
}





criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
