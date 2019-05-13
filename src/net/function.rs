use crate::core::Tensor;
use std::iter::IntoIterator;

pub trait GetParams<'p, T> {
  fn params(&'p self, p: &mut Vec<&'p Tensor<T>>) {}
  fn params_mut(&'p mut self, p: &mut Vec<&'p mut Tensor<T>>) {}
}

pub trait Function<'p, T, X>: GetParams<'p, T> {
  type Output;
  fn build(&'p mut self, x: X, rebuild: bool) -> Self::Output;
  fn eval(&'p self, x: X) -> Self::Output;
}
