use crate::{Shape, Shaped, Tensor, Matmul, seq};
use crate::nn::{Initializer, ElemInit, HeInit, Seq};
use crate::optim::{SGD};
use crate::autograd::Forward;
use std::ops::Add;
use std::marker::PhantomData;
use num_traits::{Zero, Float};


#[derive(Debug)]
pub struct Weight<T, I, O> {
  pub w: Tensor<T>,
  pub init: I,
  pub opt: O
}

impl<T> Weight<T, HeInit<T>, SGD<T>> {
  pub fn c(c: usize) -> Self
    where T: Float {
    Self{w: Tensor::shape(vec![c]),
         init: HeInit::new(),
         opt: SGD::new(T::from(0.01).unwrap())}
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
pub struct Bias<T, I, O> {
  pub b: Tensor<T>,
  pub init: I,
  pub opt: O
}

impl<T, I, O> Weight<T, I, O> {
  pub fn bias(self) -> Seq<Self, Bias<T, ElemInit<T>, O>>
    where T: Zero,
          O: Copy {
    let b = Bias{b: Tensor::new(),
                 init: ElemInit::new(T::zero()),
                 opt: self.opt};
    seq![self, b]
  }
}

impl<'b, X, T, I, O> Forward<'b, X> for Bias<T, I, O>
  where T: 'b, 
        X: Add<&'b Tensor<T>, Output=Tensor<T>> + Shaped,
        I: Initializer<T> {
  type Y = Tensor<T>;
  fn forward(&'b mut self, x: X) -> Tensor<T> {
    if !x.s().can_broadcast(self.b.s()) {
      let s = Shape::from(vec![1, x.s()[0]]);
      self.b = self.init.init(s);
    }
    x + &self.b
  }
}
  

