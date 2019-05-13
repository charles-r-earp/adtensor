use std::fmt::{Debug, Formatter, Result};
use std::ops::{Index, IndexMut};
use std::iter::{Rev, repeat};
use std::slice;
use std::cmp::max;

pub trait Shaped {
  fn s(&self) -> &Shape;
}

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
  pub fn can_broadcast<'b>(&self, rhs: &'b Self) -> bool {
    if self.len() == rhs.len() {
      if self.s[1..] == rhs.s[1..] {
        return self.s[0] == 1 
          || rhs.s[0] == 1
          || self.s[0] == rhs.s[0];
      }   
    }
    false
  }             
  #[inline]
  pub fn broadcast<'b>(&self, rhs: &'b Self) -> Self {
    debug_assert!(self.can_broadcast(&rhs),
                  format!("Can't broadcast shapes {:?} and {:?}!", &self, &rhs));
    if rhs.s[0] != 1 {rhs.clone()} else {self.clone()}
  }
  #[inline]
  pub fn can_broadcast_mm<'b>(&self, rhs: &'b Self) -> bool {
    if self.len() == rhs.len() {
      if self.len() > 2 {
        let n = self.len();
        if self.s[1..n-2] != rhs.s[1..n-2] {
          return false;
        }
        if !(self.s[0] == 1
          || rhs.s[0] == 1
          || self.s[0] == rhs.s[0]) {
          return false;
        }
      }  
      return self[0] == rhs[1];
    }
    false
  }
  #[inline]
  pub fn broadcast_mm<'b>(&self, rhs: &'b Self) -> Self {
    debug_assert!(self.can_broadcast_mm(&rhs),
                  format!("Can't matmul shapes {:?} and {:?}!", &self, &rhs));
    let mut s = self.clone();
    s[0] = rhs[0];
    if self.s[0] == 1 {
      s.s[0] = rhs.s[0];
    }
    s
  }
}

impl<S> From<S> for Shape
  where Vec<usize>: From<S> {
  #[inline]
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
  use crate::core::shape::Shape;
  #[test]
  fn test_broadcast() {
    assert_eq!(Shape::from(vec![1, 1, 2]).broadcast(&Shape::from(vec![3, 1, 2])),
               Shape::from(vec![3, 1, 2]));
    assert_eq!(Shape::from(vec![1, 2]).broadcast(&Shape::from(vec![2, 2])), 
               Shape::from(vec![2, 2]));
    assert!(!Shape::from(vec![2, 1, 1]).can_broadcast(&Shape::from(vec![2, 1, 3])));
  }
  #[test]
  fn test_broadcast_mm() {
    assert_eq!(Shape::from(vec![2, 3]).broadcast_mm(&Shape::from(vec![3, 4])),
               Shape::from(vec![2, 4]));
    assert_eq!(Shape::from(vec![2, 2, 3]).broadcast_mm(&Shape::from(vec![1, 3, 4])),
               Shape::from(vec![2, 2, 4]));
    assert!(!Shape::from(vec![2, 1, 2, 3]).can_broadcast_mm(&Shape::from(vec![1, 2, 3, 4])));
  }
}
