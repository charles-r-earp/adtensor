
/*
use std::ops::Deref;

pub type Ix1 = usize;
pub type Ix2 = (usize, usize);

pub trait Shape {
  fn rmaj_strides(&self) -> Self;
  fn product(&self) -> usize;
 
}

impl Shape for Ix1 {
  fn rmaj_strides(&self) -> Self {
    *self
  }
  fn product(&self) -> usize {
    *self
  }
}

impl Shape for Ix2 {
  fn rmaj_strides(&self) -> Self { 
    (self.1, 1)
  }
  fn product(&self) -> usize {
    self.0 * self.1
  }
}

#[derive(Default, Debug)]
pub struct Tensor<S, D> {
  shape: S,
  strides: S,
  data: D
}*/

    
