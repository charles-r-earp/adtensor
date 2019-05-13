use crate::{Shape, Shaped, Tensor, Matmul};
use crate::nn::Initializer;
use crate::autograd::Forward;
use std::marker::PhantomData;
use num_traits::Float;


#[derive(Debug)]
pub struct Weight<T, I, O> {
  w: Tensor<T>,
  init: I,
  opt: O
}

impl<T, I, O> Weight<T, I, O> {
  pub fn new<S>(s: S, init: I, opt: O) -> Self
    where Shape: From<S> {
    Self{w: Tensor::shape(s),
         init,
         opt}
  }
} 

impl<'w, X, T, I, O> Forward<'w, X> for Weight<T, I, O>
  where T: 'w, 
        X: Matmul<&'w Tensor<T>, Output=Tensor<T>> + Shaped,
        I: Initializer<T> {
  type Y = Tensor<T>;
  fn forward(&'w mut self, x: X) -> Tensor<T> {
    if !x.s().can_broadcast_mm(self.w.s()) {
      let s = Shape::from(vec![x.s()[0], self.w.s()[0]]);
      self.w = self.init.init(s);
    }
    x.mm(&self.w)
  }
}

#[derive(Debug)]


