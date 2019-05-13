use crate::Tensor;
use crate::optim::Optimizer;
use std::ops::{SubAssign, Mul};

#[derive(Debug, Clone, Copy)]
pub struct SGD<T> {
  pub lr: T
}

impl<T> SGD<T> {
  pub fn new(lr: T) -> Self {
    Self{lr}
  }
}

impl<'p, 'dy, T> Optimizer<'p, 'dy, T> for SGD<T>
  where T: SubAssign<T> + Mul<T, Output=T> + Copy {
  fn step(&mut self, p: &'p mut Tensor<T>, dy: &'dy Tensor<T>) {
    p.iter_mut().zip(dy.iter())
                .for_each(|(p, &dy)| {
      *p -= self.lr * dy;
    });
  }
}

