use crate::{Shape, Tensor};
use crate::autograd::{Initializer, Parameter};
use std::marker::PhantomData;
use std::fmt::{Debug, Formatter, Result};
use rand::distributions::{Distribution, Normal};
use num_traits::{Zero, Float};
use num_traits::cast::{NumCast, ToPrimitive};

#[derive(Default)]
pub struct Zeros<T> {
  _m: PhantomData<T>
} 

impl<T> Debug for Zeros<T> {
  fn fmt(&self, f: &mut Formatter) -> Result {
    write!(f, "Zeros")
  }
}

impl<T> Initializer<T> for Zeros<T>
  where T: Zero + Clone {
  fn tensor(&self, s: Shape) -> Tensor<T> {
    Tensor::zeros(s)
  }
}
/*
impl<T, M> Parameter<T, M> {
  pub fn zeros<S>(s: S) -> Self
    where Shape: From<S>,
          T: Zero + Clone + Default + 'static,
          M: Default {
    Self::new(Tensor::shape(s), Zeros::default())
  }
}*/

#[derive(Default)]
pub struct He<T> {
  _m: PhantomData<T>
}

impl<T> Debug for He<T> {
  fn fmt(&self, f: &mut Formatter) -> Result {
    write!(f, "He")
  }
}

impl<T> Initializer<T> for He<T>
  where T: Float {
  fn tensor(&self, s: Shape) -> Tensor<T> {
    let d0 = 
      if s.len() > 0 {s[0]}
      else {1};
    let d1 = 
      if s.len() > 1 {s[1]}
      else {1};
    // https://medium.com/usf-msds/deep-learning-best-practices-1-weight-initialization-14e5c0295b94
    let n = Normal::new(0., (2./(d0 + d1).to_f64().unwrap()).sqrt());
    Tensor::shape_fn(s, || T::from(n.sample(&mut rand::thread_rng())).unwrap())
  }
}
/*
impl<T, M> Parameter<T, M> {
  pub fn he<S>(s: S) -> Self
    where Shape: From<S>,
          T: Float + Default + 'static,
          M: Default {
    Self::new(Tensor::shape(s), He::default())
  }
}*/

#[derive(Default)]
pub struct Xavier<T> {
  _m: PhantomData<T>
}

impl<T> Debug for Xavier<T> {
  fn fmt(&self, f: &mut Formatter) -> Result {
    write!(f, "Xavier")
  }
}

impl<T> Initializer<T> for Xavier<T>
  where T: Float {
  fn tensor(&self, s: Shape) -> Tensor<T> {
    let d1 = 
      if s.len() > 1 {s[1]}
      else {1};
    // https://medium.com/usf-msds/deep-learning-best-practices-1-weight-initialization-14e5c0295b94
    let n = Normal::new(0., (1./d1.to_f64().unwrap()).sqrt());
    Tensor::shape_fn(s, || T::from(n.sample(&mut rand::thread_rng())).unwrap())
  }
}
/*
impl<T, M> Parameter<T, M> {
  pub fn xavier<S>(s: S) -> Self
    where Shape: From<S>,
          T: Float + Default + 'static,
          M: Default {
    Self::new(Tensor::shape(s), Xavier::default())
  }
}*/
