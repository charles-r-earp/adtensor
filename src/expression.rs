use crate::tensor::TensorLike;
use crate::scalar::ScalarLike;
use std::marker::PhantomData;
use std::cell::RefCell;
use std::ops::{Deref, DerefMut};
use num_traits::{Zero};

pub trait Expression<T> { 
  fn eval<'a, 'b>(&self, lhs: &'a T, rhs: &'b T) -> (T, T);   
}

pub struct Function<F, T> {
  f: F,
  _t: PhantomData<(T)>
}

impl<F, T> From<F> for Function<F, T> {
  fn from(f: F) -> Self {
    Self{f, _t: PhantomData::<T>::default()}
  }
}

impl<F, T> Expression<T> for Function<F, T>
  where T: Default,
        F: Fn(&T)->T {
  fn eval<'a, 'b>(&self, lhs: &'a T, rhs: &'b T) -> (T, T) {
    ((self.f)(lhs), T::default())
  }
} 

#[derive(Debug, Default, Clone)]
pub struct Identity<T> {
  _t: PhantomData<T>
}

impl<T> Identity<T> {
  pub fn new() -> Self {
    Self{_t: PhantomData::<T>::default()}
  }
}

impl<T> Expression<T> for Identity<T>
  where T: Clone {
  fn eval<'a, 'b>(&self, lhs: &'a T, rhs: &'b T) -> (T, T) {
    (lhs.clone(), rhs.clone())
  }
}
    
pub struct Expr<T> {
  p: (usize, usize),
  e: Box<Expression<T>>,
  t: T
}

impl<T> Expression<T> for Expr<T> {
  fn eval<'a, 'b>(&self, lhs: &'a T, rhs: &'b T) -> (T, T) {
    self.e.eval(lhs, rhs)
  }
}

pub struct ExprGraph<T> {
  es: RefCell<Vec<Expr<T>>>
}

pub struct Var<'e, T> {
  e: &'e ExprGraph<T>,
  i: usize
}

impl<T> ExprGraph<T> {
  #[inline]
  pub fn new() -> Self {
    Self{es: RefCell::new(Vec::<Expr<T>>::new())}
  }
  #[inline]
  pub fn len(&self) -> usize {
    self.es.borrow().len()
  }
  #[inline]
  fn push<'e>(&'e self, e: Expr<T>) -> Var<'e, T> {
    let v = Var{e: self, i: self.len()};
    let mut es = self.es.borrow_mut();
    es.push(e);
    v
  }
  #[inline]
  pub fn var<'e>(&'e self, t: T) -> Var<'e, T>
    where T: 'static + Clone {
    let i = self.len();
    self.push(Expr{p: (i, i), e: Box::new(Identity::<T>::new()), t})
  } 
}

impl<'e, T> Var<'e, T> {
  pub fn func<F>(self, f: F) -> Self
    where T: 'static + Default,
          F: 'static + Fn(&T)->T {
    let ref t = self.e.es.borrow()[self.i].t;
    let p = (self.i, self.e.len());
    let t = f(t);
    let e = Function::<F, T>::from(f);
    self.e.push(Expr{p, e: Box::new(e), t})
  }
}

