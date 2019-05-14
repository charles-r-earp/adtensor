use crate::{Shape, Shaped, Tensor, Matmul};
use crate::autograd::{Parameter, Initializer, Forward};
use crate::nn::init::{Zeros, He};
use std::ops::AddAssign;
use num_traits::Float;

#[derive(Debug)]
pub struct Linear<T, M> {
  x: Option<Tensor<T>>,
  w: Parameter<T, M>,
  b: Option<Parameter<T, M>>
}

impl<T, M> Linear<T, M> {
  #[inline]
  pub fn new(w: Parameter<T, M>, b: Option<Parameter<T, M>>) -> Self {
    Self{x: None, w, b}
  }
}

impl<'a, T, M> Forward<'a, Tensor<T>> for Linear<T, M>
  where &'a Tensor<T>: Matmul<&'a Tensor<T>, Output=Tensor<T>>, 
        Tensor<T>: AddAssign<&'a Tensor<T>>,
        T: 'a,
        M: Default {
  type Y = Tensor<T>;
  fn forward(&'a mut self, x: Tensor<T>) -> Tensor<T> {
    if !x.s().can_broadcast_mm(self.w.t().s()) {
      self.w.init(Shape::from(vec![x.s()[0], self.w.t().s()[0]]));
    } 
    self.x = Some(x);
    let mut y = if let Some(ref x) = self.x {
      x.mm(self.w.t())
    }
    else {
      unreachable!();
    };
    if let Some(ref mut b) = self.b {
      if !b.t().s().can_broadcast(y.s()) {
        b.init(Shape::from(vec![1, y.s()[0]]));
      }
      y += b.t();
    }
    y
  }
}

/*
#[derive(Debug)]
pub struct Weight<T, I, O> {
  pub w: Tensor<T>,
  x: Tensor<T>,
  init: I,
  pub opt: O
}

impl<T> Weight<T, HeInit<T>, SGD<T>> {
  #[inline]
  pub fn c(c: usize) -> Self
    where T: Float {
    Self{w: Tensor::shape(vec![c]),
         x: Tensor::new(),
         init: HeInit::new(),
         opt: SGD::new(T::from(0.01).unwrap())}
  }
} 

impl<'w, T, I, O> Forward<'w, Tensor<T>> for Weight<T, I, O>
  where T: 'w, 
        &'w Tensor<T>: Matmul<&'w Tensor<T>, Output=Tensor<T>>,
        I: Initializer<T> {
  type Y = Tensor<T>;
  #[inline]
  fn forward(&'w mut self, x: Tensor<T>) -> Tensor<T> {
    if !x.s().can_broadcast_mm(self.w.s()) {
      let s = Shape::from(vec![x.s()[0], self.w.s()[0]]);
      self.w = self.init.init(s);
    }
    self.x = x;
    self.x.mm(&self.w)
  }
}

impl<'w, DY, T, I, O> Backward<'w, DY> for Weight<T, I, O>
  where T: 'w, 
        DY: Matmul<&'w Tensor<T>, Output=Tensor<T>> + Shaped,
        O: Optimizer<'w, T> {
  type DX = Tensor<T>;
  fn backward(&'w mut self, dy: DY) -> Tensor<T> {
    let dw = dy.mm(&self.x);
    let dx = dy.mm(&self.w);
    self.x = Tensor::new();
    self.opt.step(&mut self.w, dw);
    dx
  }
}

#[derive(Debug)]
pub struct Bias<T, I, O> {
  pub b: Tensor<T>,
  init: I,
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
}*/
  

