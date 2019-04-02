use criterion::*;
use adtensor::*;
use std::ops::{Add, Deref};
use std::mem;
use std::iter::{FromIterator, Iterator, IntoIterator, repeat};
use typenum::*;
use generic_array::{GenericArray, ArrayLength};
use ndarray::{Array1};
use nalgebra as na;

fn vec_zeros<T, N>() -> Vec<T>
  where T: Default + Clone,
        N: Unsigned {
  let mut v = Vec::<T>::default();
  v.resize(<N as Unsigned>::to_usize(), T::default());
  v
}

fn array1_zeros<T, N>() -> Array1<T>
  where T: Default + Clone,
        N: Unsigned {
  Array1::<T>::from_vec(vec_zeros::<T, N>())
}
 
fn criterion_benchmark(c: &mut Criterion) {
  type T = f32;
  type N = U100;
  type M = na::U100;
  c.bench_function("dot", |b| b.iter(|| {
    let x = &[0.1; 100];
    let y = &[0.1; 100];
    dot::<T, N>(x, y)
  }));
  c.bench_function("vector_dot", |b| b.iter(|| {
    let x = Vector::<T, N>::from_iter(repeat(0.1));
    let y = Vector::<T, N>::from_iter(repeat(0.1));
    x.dot(&y)
  }));
  /*c.bench_function("na_arr_dot", |b| b.iter(|| {
    let x = na::Matrix::<T, M, na::U1, na::ArrayStorage<T, M, na::U1>>::from_iterator(repeat(0.1));
    let y = na::Matrix::<T, M, na::U1, na::ArrayStorage<T, M, na::U1>>::from_iterator(repeat(0.1));
    x.dot(&y)
  }));
  c.bench_function("array1_dot", |b| b.iter(|| {
    let x = Array1::from_iter(repeat(0.1).take(N::to_usize()));
    let y = Array1::from_iter(repeat(0.1).take(N::to_usize()));
    x.dot(&y)
  }));
  c.bench_function("vector_map", |b| b.iter(|| {
    let x = Vector::<T, N>::from_iter(repeat(0.1));
    let w = x.map(|t| (t+1.0)/(t-1.0));
    w[0]
  }));
  c.bench_function("vector_apply", |b| b.iter(|| {
    let mut x = Vector::<T, N>::from_iter(repeat(0.1));
    x.apply(|t| (t+1.0)/(t-1.0));
    x[0]
  }));
  c.bench_function("na_apply", |b| b.iter(|| {
    let mut x = na::Matrix::<T, M, na::U1, na::ArrayStorage<T, M, na::U1>>::from_iterator(repeat(0.1));
    x.apply(|t| (t+1.0)/(t-1.0));
    x[0]
  }));
  c.bench_function("array1_apply", |b| b.iter(|| {
    let mut x = Array1::from_iter(repeat(0.1).take(N::to_usize()));
    for i in 0..x.len() {
      x[i] = (x[i]+1.0)/(x[i]-1.0);
    }
    x[0]
  }));*/
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches); 
