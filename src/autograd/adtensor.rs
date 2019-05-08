use crate::shape::Shape;
use crate::tensor::{Tensor, Matmul};
use crate::adscalar::{ADScalar};
use std::cell::UnsafeCell;
use std::fmt::{Debug, Formatter, Result};
use std::ops::{Deref, Add};
use std::mem;
use num_traits::{Zero, One};

#[derive(Debug)]
pub struct ADTensor<'g, T> {
  v: Tensor<T>,
  i: Option<Idx>,
  g: &'g Graph<T>
}

impl<'g, T> ADTensor<'g, T> {
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
        *y = out.v();
        *dy = out.d();
      });
      self.g.push(PartialGrad::from(vec![(vx, dv)]));
      Self{v, i: Some(Idx::Var(self.g.len()-1)), g: self.g}
    }
    else {
      Self{v: self.v.map(|x| f(x.into()).v()), i: None, g: self.g}
    }
  }
}
    
impl<'g, T> Deref for ADTensor<'g, T> {
  type Target = [T];
  #[inline]
  fn deref(&self) -> &[T] {
    &*self.v
  }
}

macro_rules! impl_adtensor_mm {
  ($t:ty) => {
    impl<'a, 'b, 'g> Matmul<&'b Tensor<$t>> for &'a ADTensor<'g, $t> {
      type Output = ADTensor<'g, $t>;
      #[inline]
      fn mm(self, rhs: &'b Tensor<$t>) -> ADTensor<'g, $t> {
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

impl<'a, 'b, 'g, T> Add<&'b Tensor<T>> for &'a ADTensor<'g, T>
  where &'a Tensor<T>: Add<&'b Tensor<T>, Output=Tensor<T>>,
    T: One {
  type Output = ADTensor<'g, T>;
  #[inline]
  fn add(self, rhs: &'b Tensor<T>) -> ADTensor<'g, T> {
    let v = &self.v + rhs;
    if let Some(vx) = self.i {
      if let Some(px) = self.g.find_param(rhs) {
        self.g.push(PartialGrad::from(
          vec![(vx, Tensor::shape(rhs.s.clone()).init_fn(|| T::one())),
               (px, Tensor::shape(self.v.s.clone()).init_fn(|| T::one()))]
        ));
      }
      else {
        self.g.push(PartialGrad::from(
          vec![(vx, Tensor::shape(rhs.s.clone()).init_fn(|| T::one()))]
        ));
      }
      ADTensor{v, i: Some(Idx::Var(self.g.len()-1)), g: self.g}
    }
    else if let Some(px) = self.g.find_param(rhs) {
      self.g.push(PartialGrad::from(
        vec![(px, Tensor::shape(self.v.s.clone()).init_fn(|| T::one()))]
      ));
      ADTensor{v, i: Some(Idx::Var(self.g.len()-1)), g: self.g}
    }
    else {
      ADTensor{v, i: None, g: self.g}
    }
  }
}

impl<T> Tensor<T> {
  #[inline]
  pub(crate) fn param_key(&self) -> usize {
    self as *const Self as usize
  }
}

#[derive(Debug, Clone, Copy)]
enum Idx {
  Param(usize),
  Var(usize),
}

#[derive(Debug)]
pub struct PartialGrad<T> {
  g: Vec<(Idx, Tensor<T>)>
}

impl<T> PartialGrad<T> {
  #[inline]
  fn new() -> Self {
    Self{g: Vec::new()}
  }
}  

impl<T> From<Vec<(Idx, Tensor<T>)>> for PartialGrad<T> {
  #[inline]
  fn from(g: Vec<(Idx, Tensor<T>)>) -> Self {
    Self{g}
  }
}

pub struct Graph<T> {
  p: Vec<usize>,
  v: UnsafeCell<Vec<PartialGrad<T>>>
}

impl<T> Graph<T> {
  #[inline]
  pub fn params<'p>(p: Vec<&'p Tensor<T>>) -> Self {    
    Self{p: p.iter().map(|p| p.param_key()).collect(), v: UnsafeCell::new(Vec::new())}
  }
  #[inline]
  pub fn input<'g>(&'g mut self, v: Tensor<T>) -> ADTensor<'g, T> {
    ADTensor::<'g, T>{v, i: None, g: self}
  }
  #[inline]
  fn push(&self, p: PartialGrad<T>) {
    unsafe { &mut *self.v.get() }.push(p);
  }
  #[inline]
  fn len(&self) -> usize {
    unsafe { &*self.v.get() }.len() 
  }
  #[inline]
  fn find_param<'b>(&self, p: &'b Tensor<T>) -> Option<Idx> {
    if let Ok(i) = self.p.binary_search(&p.param_key()) {
      Some(Idx::Param(i))
    }
    else {
      None
    }
  }
  #[inline]
  fn grad_index(&self, i: Idx) -> usize {
    match i {
      Idx::Param(i) => i,
      Idx::Var(i) => i + self.p.len()
    }
  }
}

macro_rules! impl_graph_backward {
  ($t:ty) => {
    impl Graph<$t> {
      #[inline]
      pub fn backward<'g, 'a, 'b>(&'g self) -> Vec<Tensor<$t>> {
        let plen = self.p.len();
        let vlen = unsafe { &*self.v.get() }.len();
        let n = plen + vlen;
        let g = UnsafeCell::new(Vec::with_capacity(n));
        unsafe { &mut *g.get() }.resize(n, Tensor::new());
        let grad = unsafe { &*g.get() };
        let grad_mut = unsafe { &mut *g.get() };
        grad_mut[n-1] = Tensor::from(vec![1.]);
        grad.iter().skip(plen)
            .zip(unsafe { &*self.v.get() }.iter())
            .enumerate()
            .rev()
            .for_each(|(u, (dx, pg))| {
          pg.g.iter()
              .for_each(|p| {
            let grad_mut = unsafe { &mut *g.get() };
            grad_mut[self.grad_index(p.0)] += &dx.mm(&p.1);
          });
        });
        grad_mut.truncate(plen);
        let mut grad = Vec::new();
        mem::swap(grad_mut, &mut grad);
        grad
      }  
    }
  }
}

impl_graph_backward!(f32);
impl_graph_backward!(f64);

impl<T> Debug for Graph<T>
  where T: Debug {
  #[inline]
  fn fmt(&self, f: &mut Formatter) -> Result {
    write!(f, "Graph {{p: {:?}, v: {:?}}}", &self.p, unsafe { &*self.v.get() })
  }
}

 
        
        
  

