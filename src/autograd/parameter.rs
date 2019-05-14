use crate::{Shape, Tensor};
use std::fmt::Debug;

pub trait Initializer<T>: Debug {
  fn tensor(&self, s: Shape) -> Tensor<T>;
}

pub trait Optimizer<'p, T, M> {
  fn step(&self, p: &'p mut Tensor<T>, m: &'p mut M, dp: Tensor<T>);
}

#[derive(Debug)]
pub struct Parameter<T, M> {
  t: Tensor<T>,
  i: Box<Initializer<T>>,
  m: M
}

impl<T, M> Parameter<T, M> {
  #[inline]
  pub fn new<I>(t: Tensor<T>, i: I) -> Self
    where I: Initializer<T> + 'static,
          M: Default {
    Self{t, i: Box::new(i), m: M::default()}
  }
  #[inline]
  pub fn shape_init<S, I>(s: S, i: I) -> Self
    where Shape: From<S>,
          I: Initializer<T> + 'static,
          M: Default {
    Self::new(Tensor::shape(s), i)
  }
  #[inline]
  pub fn init(&mut self, s: Shape)
    where M: Default {
    self.t = (*self.i).tensor(s);
    self.m = M::default();
  }
  #[inline]
  pub fn t(&self) -> &Tensor<T> {
    &self.t
  }
  #[inline]
  pub fn step<'p, O>(&'p mut self, dy: Tensor<T>, opt: O)
    where O: Optimizer<'p, T, M> {
    opt.step(&mut self.t, &mut self.m, dy)
  }
} 






