use crate::core::Tensor;
use std::ops::{Add, AddAssign};

#[derive(Debug)]
pub struct Loss<T> {
  pub v: Tensor<T>,
  pub g: Vec<Tensor<T>>
}

impl<T> Loss<T> {
  pub fn new() -> Self {
    Self{v: Tensor::new(), g: Vec::new()}
  }
}

impl<'a, 'b, T> Add<&'b Loss<T>> for &'a Loss<T> 
  where Tensor<T>: AddAssign<&'b Tensor<T>> + Clone ,
        &'a Tensor<T>: Add<&'b Tensor<T>, Output=Tensor<T>> {
  type Output = Loss<T>;
  fn add(self, rhs: &'b Loss<T>) -> Loss<T> {
    let v = if self.v.len() == 0 {
      rhs.v.clone()
    }
    else {
      &self.v + &rhs.v
    };
    let g = if self.g.len() == 0 {
      rhs.g.clone()
    }
    else {
      let mut g = Vec::with_capacity(rhs.g.len());
      g.resize(g.capacity(), Tensor::new());
      g.iter_mut().zip(self.g.iter()
                             .zip(rhs.g.iter()))
                  .for_each(|(c, (a, b))| {
        *c = a + b;
      });
      g
    };
    Loss{v, g}
  }
}
        
      
