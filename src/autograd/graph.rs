use crate::tensor::Tensor;
use std::cell::{RefCell, UnsafeCell};
use std::fmt::{Debug, Formatter, Result};
use std::ops::{Deref, Add, Sub, Mul, Div};
use std::iter::IntoIterator;
use std::mem;
use num_traits::{Zero, One};
use itertools::{Itertools};

#[derive(Debug)]
pub struct Partial<T> {
  i: usize,
  g: Tensor<T>
}

#[derive(Debug)]
pub struct Vertex<T> {
  p: Vec<Partial<T>>
}

impl<T> Vertex<T> {
  fn new() -> Self {
    Self{p: Vec::new()}
  }
  fn push(&mut self, p: Partial<T>) {
    self.p.push(p)
  } 
}

pub trait Optimizer<T> {
  fn step(&mut self, d: T) -> T;
} 

impl<F, T> Optimizer<T> for F
  where F: FnMut(T)->T {
  fn step(&mut self, d: T) -> T {
    (self)(d)
  }
}

impl<T> From<Tensor<T>> for Param<T> {
  fn from(v: Tensor<T>) -> Self {
    Self{i: None, v, on: true}
  }
}

pub struct Graph<T> {
  v: RefCell<Vec<Vertex<T>>>
}

impl<T> Graph<T> {
  #[inline]
  pub fn new() -> Self {
    Self{v: RefCell::new(Vec::new())}
  }
}
/*
impl<T> Graph<T> {
  #[inline]
  pub fn var<'g>(&'g self, v: Tensor<T>) -> Var<'g, T> {
    let mut vts = self.v.borrow_mut();
    vts.push(Vertex::new());
    Var{g: self, i: vts.len() - 1, v, on: true}
  } 
}*/
/*
macro_rules! impl_graph_backward {
  ($t:ty) => {
    impl Graph<$t> {
      #[inline]
      pub fn backward<'p, 'o, 'g, P>(&'g self, params: P, opt: &'o mut Optimizer<$t>)
        where P: 'p + IntoIterator<Item=&'p mut Param<$t>> {
        let mut params = params.into_iter()
                               .filter(|p| p.i.is_some())
                               .sorted_by(|a, b| a.i.unwrap().cmp(&b.i.unwrap()))
                               .rev();
        let mut par = params.next();
        let mut vts = self.v.borrow_mut();
        let mut g = UnsafeCell::new(Vec::<Tensor<$t>>::with_capacity(vts.len()));
        let mut grad_mut = unsafe { &mut *g.get() };
        unsafe { grad_mut.set_len(grad_mut.capacity()) };
        let grad = unsafe { &mut *g.get() };
        grad_mut[grad.len() - 1] = Tensor::from(vec![1.]);
        vts.iter_mut()
           .enumerate()
           .rev()
           .for_each(|(i, v)| {
          let mut p = Vec::new();
          mem::swap(&mut p, &mut v.p);
          p.iter().for_each(|p| {
            if grad[p.i].len() != 0 {
              grad_mut[p.i] = &grad[p.i] + &(&grad[i] * &p.g);
            }
            else {
              grad_mut[p.i] = &grad[i] * &p.g;
            }
          });
          //println!("grad[{}]: {:?}", &i, &grad[i]);
          if let Some(ref mut p) = par {
            debug_assert!(grad[i].s == p.v.s);
            p.v.iter_mut()
               .zip(grad[i].iter())
               .for_each(|(v, &d)| { 
              *v += opt.step(d);
            });
            p.i = None;
            par = params.next();
          }
        });
        vts.clear();
      }
    }
  }
}

impl_graph_backward!(f32);
impl_graph_backward!(f64);*/

impl<T> Debug for Graph<T>
  where T: Debug {
  fn fmt(&self, f: &mut Formatter) -> Result {
    write!(f, "Graph {{{:?}}}", &self.v.borrow())
  }
}

pub struct Var<'g, T> {
  g: &'g Graph<T>,
  i: usize,
  v: Tensor<T>,
  on: bool
}
/*
impl<'g, T> Var<'g, T> {
  #[inline]
  pub fn no_grad(self) -> Self {
    Self{g: self.g, i: self.i, v: self.v, on: false}
  }
  #[inline]
  pub fn with_grad(self) -> Self {
    Self{g: self.g, i: self.i, v: self.v, on: true}
  }
  #[inline]
  pub fn is_grad_on(&self) -> bool {
    !self.on
  }
  #[inline]
  pub fn register<'p, P>(& self, p: P)
    where P: IntoIterator<Item=&'p mut Param<T>>,
          T: 'p + Debug {
    if self.on {
      let mut vts = self.g.v.borrow_mut();
      p.into_iter()
       .filter(|p| !(p.i.is_some() || !p.on))
       .for_each(|p| {
         p.i = Some(vts.len());
         vts.push(Vertex::new());
      });
    }
  } 
}  

impl<'g, T> Debug for Var<'g, T>
  where T: Debug {
  fn fmt(&self, f: &mut Formatter) -> Result {
    write!(f, "Var {{i: {}, v: {:?}, on: {}}}", &self.i, &self.v, &self.on)
  }
}

impl<'g, T> Deref for Var<'g, T> {
  type Target = Tensor<T>;
  fn deref(&self) -> &Tensor<T> {
    &self.v
  }
}

macro_rules! impl_var_addsub_op {
  ($op:tt, $optrait:ident, $func:ident) => {
    impl<'a, 'b, 'g, T> $optrait<&'b Var<'g, T>> for &'a Var<'g, T>
      where &'a Tensor<T>: $optrait<&'b Tensor<T>, Output=Tensor<T>>,
            T: Zero + One + $optrait<Output=T> {
      type Output = Var<'g, T>;
      fn $func(self, rhs: &'b Var<'g, T>) -> Var<'g, T> {
        let mut vts = self.g.v.borrow_mut();
        let i = vts.len();
        let mut v = Vertex::new();
        if self.on {
          v.push(Partial{i: self.i, g: Tensor::shape(rhs.v.s.clone()).init_fn(|| T::one())});
        }
        if rhs.on {
          v.push(Partial{i: rhs.i, g: Tensor::shape(self.v.s.clone()).init_fn(|| T::zero() $op T::one())});
        }
        vts.push(v);
        Var{g: self.g, i, v: &self.v $op &rhs.v, on: self.on || rhs.on}
      }
    }
    impl<'a, 'b, 'g, T> $optrait<&'b Param<T>> for &'a Var<'g, T>
      where &'a Tensor<T>: $optrait<&'b Tensor<T>, Output=Tensor<T>>,
            T: Zero + One + $optrait<Output=T> {
      type Output = Var<'g, T>;
      fn $func(self, rhs: &'b Param<T>) -> Var<'g, T> {
        let mut vts = self.g.v.borrow_mut();
        let i = vts.len();
        let mut v = Vertex::new();
        if self.on {
          v.push(Partial{i: self.i, 
                         g: Tensor::shape(rhs.v.s.clone()).init_fn(|| T::one())});
          if rhs.on {
            if let Some(i) = rhs.i {
              v.push(Partial{i, 
                             g: Tensor::shape(self.v.s.clone()).init_fn(|| T::zero() $op T::one())});
            }
          }
        }
        vts.push(v);
        Var{g: self.g, i, v: &self.v $op &rhs.v, on: self.on}
      }
    }
    impl<'a, 'b, 'g, T> $optrait<&'b Var<'g, T>> for &'a Param<T>
      where &'a Tensor<T>: $optrait<&'b Tensor<T>, Output=Tensor<T>>,
            T: Zero + One + $optrait<Output=T> {
      type Output = Var<'g, T>;
      fn $func(self, rhs: &'b Var<'g, T>) -> Var<'g, T> {
        let mut vts = rhs.g.v.borrow_mut();
        let i = vts.len();
        let mut v = Vertex::new();
        if rhs.on {
          if self.on {
            if let Some(i) = self.i {
              v.push(Partial{i, 
                             g: Tensor::shape(rhs.v.s.clone()).init_fn(|| T::one())});
            }
          }
          v.push(Partial{i: rhs.i, 
                         g: Tensor::shape(self.v.s.clone()).init_fn(|| T::zero() $op T::one())});
        }
        vts.push(v);
        Var{g: rhs.g, i, v: &self.v $op &rhs.v, on: rhs.on}
      }
    }
    impl<'a, 'b, 'g, T> $optrait<&'b Tensor<T>> for &'a Var<'g, T>
      where &'a Tensor<T>: $optrait<&'b Tensor<T>, Output=Tensor<T>>,
            T: Zero + One + $optrait<Output=T> {
      type Output = Var<'g, T>;
      fn $func(self, rhs: &'b Tensor<T>) -> Var<'g, T> {
        let mut vts = self.g.v.borrow_mut();
        let i = vts.len();
        let mut v = Vertex::new();
        if self.on {
          v.push(Partial{i: self.i, g: Tensor::shape(rhs.s.clone()).init_fn(|| T::one())});
        }
        vts.push(v);
        Var{g: self.g, i, v: &self.v $op &rhs, on: self.on}
      }
    }
    impl<'a, 'b, 'g, T> $optrait<&'b Var<'g, T>> for &'a Tensor<T>
      where &'a Tensor<T>: $optrait<&'b Tensor<T>, Output=Tensor<T>>,
            T: Zero + One + $optrait<Output=T> {
      type Output = Var<'g, T>;
      fn $func(self, rhs: &'b Var<'g, T>) -> Var<'g, T> {
        let mut vts = rhs.g.v.borrow_mut();
        let i = vts.len();
        let mut v = Vertex::new();
        if rhs.on {
          v.push(Partial{i: rhs.i, g: Tensor::shape(self.s.clone()).init_fn(|| T::zero() $op T::one())});
        }
        vts.push(v);
        Var{g: rhs.g, i, v: self $op &rhs.v, on: rhs.on}
      }
    }
  }  
}

impl_var_addsub_op!(+, Add, add);
impl_var_addsub_op!(-, Sub, sub);


impl<'a, 'b, 'g, T> Mul<&'b Var<'g, T>> for &'a Var<'g, T>
  where &'a Tensor<T>: Mul<&'b Tensor<T>, Output=Tensor<T>>,
        T: Zero + One + Mul<Output=T> + Clone {
  type Output = Var<'g, T>;
  fn mul(self, rhs: &'b Var<'g, T>) -> Var<'g, T> {
    let mut vts = self.g.v.borrow_mut();
    let i = vts.len();
    let mut v = Vertex::new();
    if self.on {
      v.push(Partial{i: self.i, g: rhs.v.clone()});
    }
    if rhs.on {
      v.push(Partial{i: rhs.i, g: self.v.clone()});
    }
    vts.push(v);
    Var{g: self.g, i, v: &self.v * &rhs.v, on: self.on || rhs.on}
  }
}

impl<'a, 'b, 'g, T> Mul<&'b Param<T>> for &'a Var<'g, T>
  where &'a Tensor<T>: Mul<&'b Tensor<T>, Output=Tensor<T>>,
        T: Zero + One + Mul<Output=T> + Clone {
  type Output = Var<'g, T>;
  fn mul(self, rhs: &'b Param<T>) -> Var<'g, T> {
    let mut vts = self.g.v.borrow_mut();
    let i = vts.len();
    let mut v = Vertex::new();
    if self.on {
      v.push(Partial{i: self.i, 
                     g: rhs.v.clone()});
      if rhs.on {
        if let Some(i) = rhs.i {
          v.push(Partial{i, 
                         g: self.v.clone()});
        }
      }
    }
    vts.push(v);
    Var{g: self.g, i, v: &self.v * &rhs.v, on: self.on}
  }
}
*/
