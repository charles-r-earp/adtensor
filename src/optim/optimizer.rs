use crate::Tensor;

pub trait Optimizer<'p, 'dy, T> {
  fn step(&mut self, p: &'p mut Tensor<T>, dy: &'dy Tensor<T>);
}
