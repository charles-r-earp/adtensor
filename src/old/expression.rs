

//use std::fmt::{Debug, Formatter, Result};
use std::cell::RefCell;
use std::ops::{Add};
use num_traits::{Zero};

pub struct Node<T> {
  p: [usize; 2],
  d: [T; 2]
}

impl<T> Node<T> {
  fn new(p: [usize; 2], d: [T; 2]) -> Self {
    Self{p, d}
  }
}

#[derive(Default)]
pub struct Expr<T> {
  n: RefCell<Vec<Node<T>>>
}

impl<T> Expr<T> {
  pub fn new() -> Self {
    Self::default()
  } 
  fn unary(&self, p: usize, d: T) -> usize
  where T: Zero {
    let mut n = self.n.borrow_mut();
    let i = n.len();
    n.push(Node::new([p, i], [d, T::zero()]));
    i
  }
}

pub struct Const<T> {
  t: T
}

impl<T> From<T> for Const<T> {
  fn from(t: T) -> Self {
    Self{t}
  }
}

#[derive(Clone)] 
pub struct Var<'e, T> {
  e: &'e Expr<T>,
  i: usize,
  t: T
}

impl<T> Expr<T> {
  pub fn var<'e>(&'e self, t: T) -> Var<'e, T> {
    let i = self.n.len()
    let v = Var{e, i, t};
    self.n.push(Node

impl<'e, T> Var<'e, T> {
  fn unary(&self, t: T, d: T) -> Self
    where T: Zero {
    let i = self.e.unary(self.i, d);
    Self{e: self.e, i, t}
  }
}

impl<'a, 'b, 'e, T> Add<&'b Const<T>> for &'a Var<'e, T>
  where T: Zero,
        &'a T: Add<&'b T, Output=T> {
  type Output = Var<'e, T>;
  fn add(self, rhs: &'b Const<T>) -> Self::Output {
    self.unary(&self.t + &rhs.t, T::zero())
  }
}




 

