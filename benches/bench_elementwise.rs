use adtensor::adscalar::ADScalar;
use adtensor::scalar::Scalar;
use std::iter::{FromIterator};
use criterion::*;
use rand::distributions::{Distribution, StandardNormal};
use ndarray::{Array1};

const N: usize = 100;

fn criterion_benchmark(c: &mut Criterion) {
  c.bench_function("Scalar", |b| {
    let x = Array1::<f32>::from_shape_fn([N], |_| StandardNormal.sample(&mut rand::thread_rng()) as f32);
    let x = Vec::from_iter(x.into_iter().map(|&x| Scalar::new(x)));
    b.iter(|| Vec::from_iter(x.iter().map(|&x| (x + Scalar::from(1.)) * Scalar::from(2.))));
  });
  c.bench_function("ADScalar", |b| {
    let x = Array1::<f32>::from_shape_fn([N], |_| StandardNormal.sample(&mut rand::thread_rng()) as f32);
    let x = Vec::from_iter(x.into_iter().map(|&x| ADScalar::new(x)));
    b.iter(|| Vec::from_iter(x.iter().map(|&x| (x + ADScalar::from(1.)) * ADScalar::from(2.))));
  });
  c.bench_function("ADScalar with Scalar", |b| {
    let x = Array1::<f32>::from_shape_fn([N], |_| StandardNormal.sample(&mut rand::thread_rng()) as f32);
    let x = Vec::from_iter(x.into_iter().map(|&x| ADScalar::new(x)));
    b.iter(|| Vec::from_iter(x.iter().map(|&x| (x + Scalar::from(1.)) * Scalar::from(2.))));
  });
  c.bench_function("Array1", |b| {
    let x = Array1::<f32>::from_shape_fn([N], |_| StandardNormal.sample(&mut rand::thread_rng()) as f32);
    b.iter(|| x.map(|&x| (x + 1.) * 2.));
  });
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
