use crate::{Shape, Tensor};
use std::marker::PhantomData;
use rand::distributions::{Distribution, Normal};
use num_traits::{Float};
use num_traits::cast::{NumCast, ToPrimitive};

pub trait Initializer<T> {
  fn init(&self, s: Shape) -> Tensor<T>;
}

#[derive(Debug, Default, Clone, Copy)]
pub struct ElemInit<T> {
  t: T
}

impl<T> ElemInit<T> {
  pub fn new(t: T) -> Self {
    Self{t}
  }
}

impl<T> Initializer<T> for ElemInit<T>
  where T: Copy {
  fn init(&self, s: Shape) -> Tensor<T> {
    Tensor::shape_elem(s, self.t)
  }
}

#[derive(Debug, Default, Clone, Copy)]
pub struct HeInit<T> {
  _p: PhantomData<T>
}

impl<T> HeInit<T> {
  pub fn new() -> Self {
    Self{_p: PhantomData::default()}
  }
}

impl<T> Initializer<T> for HeInit<T>
  where T: Float {
  fn init(&self, s: Shape) -> Tensor<T> {
    let d0 = 
      if s.len() > 0 {s[0]}
      else {1};
    let d1 = 
      if s.len() > 1 {s[1]}
      else {1};
    // https://medium.com/usf-msds/deep-learning-best-practices-1-weight-initialization-14e5c0295b94
    let n = Normal::new(0., (2./(d0 + d1).to_f64().unwrap()).sqrt());
    Tensor::shape_fn(s, |_| T::from(n.sample(&mut rand::thread_rng())).unwrap())
  }
}

#[derive(Debug, Default, Clone, Copy)]
pub struct XavierInit<T> {
  _p: PhantomData<T>
}

impl<T> XavierInit<T> {
  pub fn new() -> Self {
    Self{_p: PhantomData::default()}
  }
}

impl<T> Initializer<T> for XavierInit<T>
  where T: Float {
  fn init(&self, s: Shape) -> Tensor<T> {
    let d1 = 
      if s.len() > 1 {s[1]}
      else {1};
    // https://medium.com/usf-msds/deep-learning-best-practices-1-weight-initialization-14e5c0295b94
    let n = Normal::new(0., (1./d1.to_f64().unwrap()).sqrt());
    Tensor::shape_fn(s, |_| T::from(n.sample(&mut rand::thread_rng())).unwrap())
  }
}
