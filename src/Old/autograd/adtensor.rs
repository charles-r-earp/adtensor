use crate::core::{Tensor, Matmul, Shape, Shaped};
use crate::autograd::{Graph, graph::{Vtx, Backward}, Loss};
use std::ops::Deref;

#[derive(Debug, Clone)]
pub struct ADTensor<'g, 'a, 'p, T> {
  v: Tensor<T>,
  i: Option<Vtx>,
  g: &'g Graph<'a, 'p, T>
}

impl<'g, 'a, 'p, T> From<ADTensor<'g, 'a, 'p, T>> for Loss<T>
  where Graph<'a, 'p, T>: Backward<T> {
  fn from(y: ADTensor<'g, 'a, 'p, T>) -> Loss<T> {
    let g = if let Some(Vtx::Var(i)) = y.i {
      y.g.truncate(i+1);
      y.g.backward()
    }
    else {
      Vec::new()
    };
    Loss{v: y.v, g}
  }
}

impl<'a, 'p, T> Graph<'a, 'p, T> {
  #[inline]
  pub fn input<'g>(&'g mut self, v: Tensor<T>) -> ADTensor<'g, 'a, 'p, T> {
    ADTensor{v, i: None, g: self}
  }
} 
/*
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
}*/


impl<'g, 'a, 'p, T> Deref for ADTensor<'g, 'a, 'p, T> {
  type Target = [T];
  #[inline]
  fn deref(&self) -> &[T] {
    &*self.v
  }
}

impl<'g, 'a, 'p, T> Shaped for ADTensor<'g, 'a, 'p, T> {
  fn s(&self) -> &Shape {
    &self.v.s()
  }
}

macro_rules! impl_adtensor_mm {
  ($t:ty) => {
    impl<'t, 'a, 'g, 'p> Matmul<&'p Tensor<$t>> for &'t ADTensor<'g, 'a, 'p, $t> {
      type Output = ADTensor<'g, 'a, 'p, $t>;
      #[inline]
      fn mm(self, rhs: &'p Tensor<$t>) -> ADTensor<'g, 'a, 'p, $t> {
        let v = self.v.mm(rhs as &Tensor<$t>);
        let dv = [
          if let Some(vx) = self.i {
            Some((vx, Tensor::ones(rhs.s().clone())))
          }
          else { None },
          if let Some(px) = self.g.param_vtx(rhs) {
            Some((px, Tensor::ones(self.s().clone())))
          }
          else { None }];
        let i = 
          if dv[0].is_some() || dv[1].is_some() {
            Some(self.g.var_vtx(dv))
          }
          else { None };
        ADTensor{v, i: i, g: self.g}
      }
    }
  }
}    

impl_adtensor_mm!(f32);
impl_adtensor_mm!(f64);  
/*
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
*/


 
        
        
  

