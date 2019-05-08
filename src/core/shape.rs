use std::fmt::{Debug, Formatter, Result};
use std::ops::{Index, IndexMut};
use std::iter::{Rev, repeat};
use std::slice;
use std::cmp::max;

#[derive(Default, Clone, PartialEq)]
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
  fn can_broadcast<'b>(&self, rhs: &'b Self) -> bool {
    let mut d = 0;
    self.iter().chain(repeat(&1))
        .zip(rhs.iter().chain(repeat(&1)))
        .take(max(self.len(), rhs.len()))
        .all(|(&a, &b)| {
      if a == b {
        true
      }
      else if a == 1 && d != -1 {
        d = 1;
        true
      }
      else if b == 1 && d != 1 {
        d = -1;
        true
      }
      else { false }
    })   
  }             
  #[inline]
  pub fn broadcast<'b>(&self, rhs: &'b Self) -> (usize, usize, Self) {
    let n1 = self.product();
    let n2 = rhs.product();
    debug_assert!(self.can_broadcast(&rhs),
                  format!("Can't broadcast shapes {:?} and {:?}!", &self, &rhs));
    (n1, n2, if n1 > n2 {self.clone()} else { if self.len() > rhs.len() {self.clone()} else {rhs.clone()} })
  }
  #[inline]
  pub fn broadcast_mm<'b>(&self, rhs: &'b Self) -> Self {
    let mut s = self.clone();
    debug_assert!(self.len() >= 2
                  && rhs.len() >= 2
                  && self.len() >= rhs.len()
                  && (self[0] == rhs[1] || self[0] == 1 || rhs[1] == 1)
                  && !self.iter()
                          .zip(rhs.iter())
                          .skip(2)
                          .any(|(a, b)| a < b), 
                  "{}", 
                  format!("Can't matmul shapes {:?} and {:?}!", &self, &rhs));
    s[0] = rhs[0];
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

#[cfg(test)]
mod tests {
  #[test]
  fn test_broadcast() {
    use crate::core::shape::Shape;
    assert!(Shape::from(vec![1, 2]).can_broadcast(&Shape::from(vec![3, 1, 2])));
    assert!(Shape::from(vec![1, 2]).can_broadcast(&Shape::from(vec![1, 1])));
    assert!(!Shape::from(vec![2, 1, 1]).can_broadcast(&Shape::from(vec![1, 3])));
  }
}
