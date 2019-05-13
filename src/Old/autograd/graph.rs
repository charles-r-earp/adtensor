use crate::core::{Tensor, Matmul};
use std::mem;
use std::ops::AddAssign;
use std::iter::IntoIterator;
use std::cell::UnsafeCell;
use std::fmt::{Debug, Formatter, Result};
use itertools::{Itertools, sorted};

#[derive(Debug, Clone, Copy)]
pub(crate) enum Vtx {
  Param(usize),
  Var(usize),
}

pub struct Graph<'a, 'p, T> {
  pub(crate) p: &'a Vec<&'p Tensor<T>>,
  pub(crate) v: UnsafeCell<Vec<[Option<(Vtx, Tensor<T>)>; 2]>>
}

impl<'a, 'p, T> Graph<'a, 'p, T> {
  #[inline]
  pub fn params(p: &'a Vec<&'p Tensor<T>>) -> Self {
    Self{p, v: UnsafeCell::new(Vec::new())}
  }
  #[inline]
  pub(crate) fn param_vtx(&self, p: &'p Tensor<T>) -> Option<Vtx> {
    let k = p.param_key();
    let r = self.p.binary_search_by_key(&k, |p| p.param_key());
    match r {
      Ok(i) => Some(Vtx::Param(i)),
      Err(i) => None
    }
  }
  #[inline]
  pub(crate) fn var_vtx(&self, v: [Option<(Vtx, Tensor<T>)>; 2]) -> Vtx {
    let vars = unsafe { &mut *self.v.get() };
    let i = vars.len();
    vars.push(v);
    Vtx::Var(i)
  }
  #[inline]
  pub(crate) fn truncate(&self, n: usize) {
    unsafe { &mut *self.v.get() }.truncate(n);
  }
}

impl<'a, 'p, T> Debug for Graph<'a, 'p, T>
  where T: Debug {
  #[inline]
  fn fmt(&self, f: &mut Formatter) -> Result {
    write!(f, 
           "Graph {{p: {:?}, v: {:?}}}", 
           self.p, 
           unsafe { &*self.v.get() })
  }
}

pub trait Backward<T> {
  fn backward(&self) -> Vec<Tensor<T>>;
}

macro_rules! impl_graph_backward {
  ($t:ty) => {
    impl<'a, 'p> Backward<$t> for Graph<'a, 'p, $t> {
      #[inline]
      fn backward(&self) -> Vec<Tensor<$t>> {
        let plen = self.p.len();
        let vlen = unsafe { &*self.v.get() }.len();
        let n = plen + vlen;
        let g = UnsafeCell::new(Vec::with_capacity(n));
        unsafe { &mut *g.get() }.resize(n, Tensor::new());
        let grad = unsafe { &*g.get() };
        let grad_mut = unsafe { &mut *g.get() };
        grad_mut[n-1] = Tensor::ones(vec![1, 1]);
        use crate::core::{Shape, Shaped};
        println!("grad.s(): {:?}", grad.iter().map(|x| x.s()).collect::<Vec<&Shape>>());
        grad.iter().skip(plen)
            .zip(unsafe { &*self.v.get() }.iter())
            .enumerate()
            .rev()
            .for_each(|(u, (dx, pg))| {
          pg.into_iter()
            .for_each(|p| {
              if let Some((i, p)) = p {
                let grad_mut = unsafe { &mut *g.get() };
                let i = match i {
                  Vtx::Var(i) => i + plen,
                  Vtx::Param(i) => *i
                };
                let grad = unsafe { &*g.get() };
                println!("dx: {:?} p: {:?} grad[{}]: {:?}", dx.s(), p.s(), i, grad[i]);
                grad_mut[i] += &dx.mm(p);
                let grad = unsafe { &*g.get() };
                println!("grad[{}]: {:?}", i, grad[i]);
              }
          });
        });
        grad_mut.truncate(plen);
        let mut g = Vec::new();
        mem::swap(&mut g, grad_mut);
        g
      }  
    }
  }
}

impl_graph_backward!(f32);
//impl_graph_backward!(f64);

