use adtensor::tensor::{Tensor, Shape};
use std::iter::{FromIterator};
use criterion::*;
use rand::distributions::{Distribution, StandardNormal};
use ndarray::{Array1};

const N: usize = 100;

fn criterion_benchmark(c: &mut Criterion) {
  c.bench_function(&format!("Tensor [{}] + [{}]", N, N), |b| {
    let x = Tensor::with_elem(Shape::from([N]), 0.1);
    b.iter(|| &x + &x)
  }); 
  c.bench_function(&format!("Array [{}] + [{}]", N, N), |b| {
    let x = Array1::from_elem(N, 0.1);
    b.iter(|| &x + &x)
  }); 
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
