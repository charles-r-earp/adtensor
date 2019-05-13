use crate::core::Tensor;
use crate::autograd::Loss;

pub trait Optimizer<T> {
  fn step<'p>(&mut self, p: Vec<&'p mut Tensor<T>>, loss: Loss<T>);
}
