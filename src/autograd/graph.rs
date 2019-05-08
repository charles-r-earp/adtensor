use crate::core::{Tensor, Matmul};
use crate::optim::Optimizer;
use std::mem;
use std::ops::AddAssign;
use std::iter::IntoIterator;
use std::cell::UnsafeCell;
use std::fmt::{Debug, Formatter, Result};
use itertools::{Itertools, sorted};

impl<T> Tensor<T> {
  #[inline]
  fn param_key(&self) -> *const Tensor<T> {
    self as *const Self
  }
  #[inline]
  fn step<'b, O>(&mut self, dx: &'b Tensor<T>, opt: &mut O)
    where T: AddAssign + Copy,
          O: Optimizer<T> {
    debug_assert!(self.s == dx.s, 
                  format!("Tensor.step self.s != dx.s! {:?} != {:?}", &self.s, &dx.s)); 
    self.iter_mut()
        .zip(dx.iter())
        .for_each(|(x, &dx)| {
      *x += opt.step(dx);
    });
  }
}

#[derive(Debug, Clone, Copy)]
pub(crate) enum Idx {
  Param(usize),
  Var(usize),
}

#[derive(Debug)]
pub struct PartialGrad<T> {
  pub(crate) g: Vec<(Idx, Tensor<T>)>
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

pub struct Graph<'p, T> {
  pub(crate) p: Vec<&'p Tensor<T>>,
  pub(crate) v: UnsafeCell<Vec<PartialGrad<T>>>
}

impl<'p, T> Graph<'p, T> {
  #[inline]
  pub fn params<P, I>(p: P) -> Self
    where T: 'p,
          P: IntoIterator<Item=&'p Tensor<T>, IntoIter=I>,
          I: Iterator<Item=&'p Tensor<T>> {    
    Self{p: p.into_iter().sorted_by_key(|p| p.param_key()).collect(),
         v: UnsafeCell::new(Vec::new())}
  }
  /*#[inline]
  pub fn input<'g>(&'g mut self, v: Tensor<T>) -> ADTensor<'g, T> {
    ADTensor::<'g, T>{v, i: None, g: self}
  }*/
  #[inline]
  pub(crate) fn push(&self, p: PartialGrad<T>) {
    unsafe { &mut *self.v.get() }.push(p);
  }
  #[inline]
  pub(crate) fn len(&self) -> usize {
    unsafe { &*self.v.get() }.len() 
  }
  #[inline]
  pub(crate) fn find_param<'b>(&self, p: &'b Tensor<T>) -> Option<Idx> {
    if let Ok(i) = self.p.binary_search_by_key(&p.param_key(), |p| p.param_key()) {
      Some(Idx::Param(i))
    }
    else {
      None
    }
  }
  #[inline]
  pub(crate) fn grad_index(&self, i: Idx) -> usize {
    match i {
      Idx::Param(i) => i,
      Idx::Var(i) => i + self.p.len()
    }
  }
}

impl<'p, T> Debug for Graph<'p, T>
  where T: Debug {
  #[inline]
  fn fmt(&self, f: &mut Formatter) -> Result {
    write!(f, "Graph {{p: {:?}, v: {:?}}}", &self.p, unsafe { &*self.v.get() })
  }
}

macro_rules! impl_graph_backward {
  ($t:ty) => {
    impl<'p> Graph<'p, $t> {
      #[inline]
      pub fn backward<'b, O>(&mut self, opt: &'b mut O)
        where O: Optimizer<$t> {
        let plen = self.p.len();
        let vlen = unsafe { &*self.v.get() }.len();
        let n = plen + vlen;
        let g = UnsafeCell::new(Vec::with_capacity(n));
        unsafe { &mut *g.get() }.resize(n, Tensor::new());
        let grad = unsafe { &*g.get() };
        let grad_mut = unsafe { &mut *g.get() };
        grad_mut[n-1] = Tensor::ones(vec![1, 1]);
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
        self.p.iter()
              .zip(grad.iter())
              .for_each(|(&p, dx)| {
          unsafe {
            (*(p as *const Tensor<$t> as *mut Tensor<$t>))
            .step(&dx, opt)
          };
        });
      }  
    }
  }
}

impl_graph_backward!(f32);
impl_graph_backward!(f64);

