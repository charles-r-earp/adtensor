use crate::Tensor;
use crate::autograd::Parameter;

pub trait Optimizer<'p, T> {
  type Meta;
  fn step<I>(&self, p: &'p mut Parameter<T, Self::Meta>, dp: Tensor<T>) 
}
