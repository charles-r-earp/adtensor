use crate::Tensor;
use crate::autograd::Optimizer;
use std::ops::{Deref, DerefMut, SubAssign, Mul};

#[derive(Debug)]
pub struct SGD<T> {
  pub lr: T
}

#[derive(Debug, Default)]
pub struct SGDParam<T> {
  p: Tensor<T>
}

impl<T> Deref for SGDParam {
  p: 

impl<'p, 'dp, T> Optimizer<'p, T> for SGD<T>
  where T: SubAssign<T> + Mul<T, Output=T> + Copy {
  fn step(&mut self, p: &'p mut Tensor<T>, dp: Tensor<T>) {
    p.iter_mut().zip(dp.iter())
                .for_each(|(p, &dp)| {
      *p -= self.lr * dp;
    });
  }
}

