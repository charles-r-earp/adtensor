use crate::core::Tensor;
use crate::autograd::Loss;
use crate::optim::Optimizer;
use num_traits::Float;

#[derive(Debug)]
pub struct SGD<T> {
  pub lr: T
}

impl<T> SGD<T> {
  pub fn new(lr: T) -> Self {
    Self{lr}
  }
}

impl<T> Optimizer<T> for SGD<T>
  where T: Float {
  fn step<'p>(&mut self, p: Vec<&'p mut Tensor<T>>, loss: Loss<T>) {
    p.into_iter().zip(loss.g.into_iter())
                .for_each(|(p, g)| {
      *p += &g.map(|d| -self.lr * d)
    });
  }
}

