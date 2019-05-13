use crate::core::{Shape, Tensor};
use crate::autograd::{ADTensor};
use crate::net::{Function, Initializer};
use crate::net::initializer::{BasicInit, XavierInit};
use num_traits::{Float};

pub trait Activation<T>: Initializer<T> + Function<T> {
  fn is_identity() -> bool { false }
}

#[derive(Default)]
pub struct Identity<T> {
  i: BasicInit<T>
}

impl<T> Identity<T> {
  pub fn new() -> Self {
    Self{i: BasicInit::new()}
  }
}

impl<T> Initializer<T> for Identity<T>
  where T: Float {
  fn init(&self, s: Shape) -> Tensor<T> {
    self.i.init(s)
  }
}

impl<T> Function<T> for Identity<T> {
  fn forward<'b, 'g, 'p>(&self, x: &'b ADTensor<'g, 'p, T>) -> ADTensor<'g, 'p, T> {
    x.clone()
  }
  fn eval<'b>(&self, x: &'b Tensor<T>) -> Tensor<T> {
    x.clone()
  }
}

impl<T> Activation<T> for Identity<T>
  where T: Float {
  fn is_identity() -> bool { true }
}
  
/*
#[derive(Default)]
pub struct Sigmoid<T> {
  i: XavierInit<T>
}

impl<T> Sigmoid<T> {
  pub fn new() -> Self {
    Self{i: XavierInit::new()}
  }
}

impl<T> Initializer<T> for Sigmoid<T>
  where T: Float {
  fn init(&self, s: Shape) -> Tensor<T> {
    self.i.init(s)
  }
}

impl<'p, T: 'p> Layer<'p, T> for Sigmoid<T> {
  type ParamIter = Empty<&'p Tensor<T>>;
  fn params(&self) -> Empty<&'p Tensor<T>> { empty() }
  fn build<'b>(&mut self, s: &'b Shape) -> Shape {
    panic!();
  }
  fn forward<'b, 'g>(&self, x: &'b ADTensor<'g, 'p, T>) -> ADTensor<'g, 'p, T> {
    panic!();
  }
  fn eval<'b>(&self, x: &'b Tensor<T>) -> Tensor<T> {
    panic!();
  }
}*/
