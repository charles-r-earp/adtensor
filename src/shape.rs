use std::fmt::{Debug, Formatter, Result};
use std::ops::{Index, IndexMut};
use std::iter::{Rev};
use std::slice;

#[derive(Default, Clone)]
pub struct Shape {
  s: Vec<usize>
}

impl Shape {
  #[inline]
  pub fn new() -> Self {
    Self::default()
  }
  #[inline]
  pub fn len(&self) -> usize {
    self.s.len()
  }
  #[inline]
  pub fn iter<'a>(&'a self) -> Rev<slice::Iter<'a, usize>> {
    self.s.iter().rev()
  }
  #[inline]
  pub fn iter_mut<'a>(&'a mut self) -> Rev<slice::IterMut<'a, usize>> {
    self.s.iter_mut().rev()
  }
  #[inline]
  pub fn product(&self) -> usize {
    self.s.iter().product::<usize>()
  }
  #[inline]
  pub fn broadcast<'b>(&self, rhs: &'b Self) -> (usize, usize, Self) {
    let n1 = self.product();
    let n2 = rhs.product();
    debug_assert!({
      let mut i = self.iter()
              .zip(rhs.iter());
      !i.clone().any(|(a, b)| a < b)
      || !i.any(|(a, b)| a > b)
    }, format!("Can't broadcast shapes {:?} and {:?}!", &self, &rhs));
    (n1, n2, if n1 > n2 {self.clone()} else { if self.len() > rhs.len() {self.clone()} else {rhs.clone()} })
  }
  #[inline]
  pub fn broadcast_mm<'b>(&self, rhs: &'b Self) -> Self {
    let mut s = self.clone();
    if rhs.len() != 1 || rhs[0] > 1 {
      debug_assert!(!self.iter()
                         .zip(rhs.iter())
                         .skip(2)
                         .any(|(a, b)| a < b) &&
                    self.len() >= rhs.len() && self[0] == if rhs.len() > 1 {rhs[1]} else {rhs[0]}, 
                    "{}", 
                    format!("Can't matmul shapes {:?} and {:?}!", &self, &rhs));
      s[0] = rhs[0];
    }
    s
  }
}

impl<S> From<S> for Shape
  where Vec<usize>: From<S> {
  fn from(s: S) -> Self {
    Self{s: s.into()}
  }
}

impl Index<usize> for Shape {
  type Output = usize;
  #[inline]
  fn index(&self, i: usize) -> &usize {
    &self.s[self.len() - (1 + i)]
  }
}

impl IndexMut<usize> for Shape {
  #[inline]
  fn index_mut(&mut self, i: usize) -> &mut usize {
    let n = self.len();
    &mut self.s[n - (1 + i)]
  }
}

impl Debug for Shape {
  #[inline]
  fn fmt(&self, f: &mut Formatter) -> Result {
    self.s.fmt(f)
  }
}
