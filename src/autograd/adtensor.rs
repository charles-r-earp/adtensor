use crate::core::{Shape, Tensor, Matmul};
use crate::autograd::{ADScalar};
use crate::autograd::graph::{Graph, Idx, PartialGrad};
use std::ops::{Deref, Add};
use std::mem;
use num_traits::{Zero, One};

#[derive(Debug)]
pub struct ADTensor<'g, 'p, T> {
  v: Tensor<T>,
  i: Option<Idx>,
  g: &'g Graph<'p, T>
}

impl<'g, 'p, T> ADTensor<'g, 'p, T> {
  #[inline]
  pub fn map<F>(&self, mut f: F) -> Self
    where F: FnMut(ADScalar<T>)->ADScalar<T>,
          T: Copy + Zero + One {
    if let Some(vx) = self.i {
      let n = self.v.len();
      let mut v = Tensor{s: self.v.s.clone(),
                         v: Vec::with_capacity(n)};
      let mut dv = Tensor{s: self.v.s.clone(),
                         v: Vec::with_capacity(n)};
      unsafe {
        v.set_len(n);
        dv.set_len(n);
      }
      self.iter()
          .zip(v.iter_mut()
                .zip(dv.iter_mut()))
          .for_each(|(&x, (y, dy))| {
        let out = f(ADScalar::new(x));
        *y = out.v;
        *dy = out.d;
      });
      self.g.push(PartialGrad::from(vec![(vx, dv)]));
      Self{v, i: Some(Idx::Var(self.g.len()-1)), g: self.g}
    }
    else {
      Self{v: self.v.map(|x| f(x.into()).v), i: None, g: self.g}
    }
  }
}


impl<'g, 'p, T> Deref for ADTensor<'g, 'p, T> {
  type Target = [T];
  #[inline]
  fn deref(&self) -> &[T] {
    &*self.v
  }
}

macro_rules! impl_adtensor_mm {
  ($t:ty) => {
    impl<'a, 'b, 'g, 'p> Matmul<&'b Tensor<$t>> for &'a ADTensor<'g, 'p, $t> {
      type Output = ADTensor<'g, 'p, $t>;
      #[inline]
      fn mm(self, rhs: &'b Tensor<$t>) -> ADTensor<'g, 'p, $t> {
        let v = self.v.mm(rhs);
        if let Some(vx) = self.i {
          if let Some(px) = self.g.find_param(rhs) {
            self.g.push(PartialGrad::from(
              vec![(vx, rhs.clone()), (px, self.v.clone())]
            ));
          }
          else {
            self.g.push(PartialGrad::from(
              vec![(vx, rhs.clone())]
            ));
          }
          ADTensor{v, i: Some(Idx::Var(self.g.len()-1)), g: self.g}
        }
        else if let Some(px) = self.g.find_param(rhs) {
          self.g.push(PartialGrad::from(
            vec![(px, self.v.clone())]
          ));
          ADTensor{v, i: Some(Idx::Var(self.g.len()-1)), g: self.g}
        }
        else {
          ADTensor{v, i: None, g: self.g}
        }
      }
    }
  }
}    

impl_adtensor_mm!(f32);
impl_adtensor_mm!(f64);  

impl<'a, 'b, 'g, 'p, T> Add<&'b Tensor<T>> for &'a ADTensor<'g, 'p, T>
  where &'a Tensor<T>: Add<&'b Tensor<T>, Output=Tensor<T>>,
    T: One + Clone {
  type Output = ADTensor<'g, 'p, T>;
  #[inline]
  fn add(self, rhs: &'b Tensor<T>) -> ADTensor<'g, 'p, T> {
    let v = &self.v + rhs;
    if let Some(vx) = self.i {
      if let Some(px) = self.g.find_param(rhs) {
        self.g.push(PartialGrad::from(
          vec![(vx, Tensor::ones(rhs.s.clone())),
               (px, Tensor::ones(self.v.s.clone()))]
        ));
      }
      else {
        self.g.push(PartialGrad::from(
          vec![(vx, Tensor::ones(rhs.s.clone()))]
        ));
      }
      ADTensor{v, i: Some(Idx::Var(self.g.len()-1)), g: self.g}
    }
    else if let Some(px) = self.g.find_param(rhs) {
      self.g.push(PartialGrad::from(
        vec![(px, Tensor::ones(self.v.s.clone()))]
      ));
      ADTensor{v, i: Some(Idx::Var(self.g.len()-1)), g: self.g}
    }
    else {
      ADTensor{v, i: None, g: self.g}
    }
  }
}



 
        
        
  

