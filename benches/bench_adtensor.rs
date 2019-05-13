use adtensor::core::{Shape, Tensor, Matmul};
use criterion::*;
use ndarray::{Array1, Array2};

const K: usize = 28*28;
const M: usize = 64;
const N: usize = 10;
const W: usize = 28;
const S: usize = 3; 

fn criterion_benchmark(c: &mut Criterion) {
  c.bench_function(&format!("tensor [{} {}] mm [{} {}]", M, K, K, N), |b| {
    let x = Tensor::shape_elem(vec![M, K], 0.1f32);
    let y = Tensor::shape_elem(vec![K, N], 0.1f32);
    b.iter(|| x.mm(&y));  
  });
  c.bench_function(&format!("array [{} {}] dot [{} {}]", M, K, K, N), |b| {
    let x = Array2::from_shape_fn((M, K), &mut |_| 0.1f32);
    let y = Array2::from_shape_fn((K, N), &mut |_| 0.1f32);
    b.iter(|| x.dot(&y));  
  });
  c.bench_function("Tensor [100 10] + [1 10]", |b| {
    let x = Tensor::shape_elem(vec![100, 10], 0.1);
    let y = Tensor::shape_elem(vec![1, 10], 0.1);
    b.iter(|| &x + &y)
  }); 
  c.bench_function("Tensor [1000] + [1000]", |b| {
    let x = Tensor::shape_elem(vec![1000], 0.1);
    let y = Tensor::shape_elem(vec![1000], 0.1);
    b.iter(|| &x + &y)
  }); 
  c.bench_function("Array [1000] + [1000]", |b| {
    let x = Array1::from_elem(1000, 0.1);
    b.iter(|| &x + &x)
  });
  c.bench_function("Tensor [100] map x + 1", |b| {
    let x = Tensor::<f32>::ones(vec![100]);
    b.iter(|| x.map(|x| x + 1.))
  });
  c.bench_function("Array [100] map x + 1", |b| {
    let x = Array1::from_elem(100, 1f32);
    b.iter(|| x.map(|x| x + 1.))
  });
}





criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
