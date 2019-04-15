use crate::shape::Shape;
use crate::small_arrayvec::SmallArrayVec; 
use crate::tensor_expr::TensorExpr;
use std::ops::{Deref, DerefMut};
use std::slice;
use std::iter::{IntoIterator};
use smallvec::{SmallVec};
use rand::distributions::{Distribution};

type SVec<T> = SmallArrayVec<[T; 128]>;

#[derive(Default, Debug, Clone, PartialEq)]
pub struct Tensor<T> {
  s: Shape,
  a: SVec<T>
}

impl<T> Deref for Tensor<T> {
  type Target = SVec<T>;
  #[inline]
  fn deref(&self) -> &Self::Target {
    &self.a
  }
}

impl<T> DerefMut for Tensor<T> {
  #[inline]
  fn deref_mut(&mut self) -> &mut Self::Target {
    &mut self.a
  }
}

/*impl<'a, T> IntoIterator for &'a Tensor<T> {
  type Item = T;
  type IntoIter = slice::Iter<'a, T>;
  #[inline]
  fn into_iter(self) -> Self::IntoIter {
    self.a.iter()
  }
}*/

impl<T> IntoIterator for Tensor<T> {
  type Item = T;
  type IntoIter = <SVec<T> as IntoIterator>::IntoIter;
  #[inline]
  fn into_iter(self) -> Self::IntoIter {
    self.a.into_iter()
  }
}

impl<T> Tensor<T> {
  pub fn new<D, V>(d: D, v: V) -> Self
    where T: Copy,
          D: AsRef<[usize]>,
          V: AsRef<[T]> {
    let v = v.as_ref();
    let s = Shape::new(d);
    debug_assert_eq!(v.len(), s.size());
    let a = SVec::from_slice(v.as_ref());
    Self{s, a}
  } 
  pub fn fill<D>(d: D, v: T) -> Self
    where T: Clone,
          D: AsRef<[usize]> {
    let s = Shape::new(d);
    let a = SVec::from_elem(v, s.size());
    Self{s, a}
  }  
  pub fn rand<D, R, U>(d: D, r: R) -> Self
    where D: AsRef<[usize]>,
          R: Distribution<U>,
          T: From<U> {
    let s = Shape::new(d);
    let n = s.size();
    let mut a = SVec::<T>::with_capacity(n);
    (0..n).for_each(|_| a.push(r.sample(&mut rand::thread_rng()).into()));
    Self{s, a}
  }
  pub fn from<E>(t: TensorExpr<E>) -> Self
    where TensorExpr<E>: IntoIterator<Item=T> {
    let s = t.shape().to_owned();
    let mut a = SVec::<T>::with_capacity(s.size());
    a.insert_many(0, t);
    Self{s, a}
  }
  pub fn shape<'a>(&'a self) -> &'a Shape {
    &self.s
  }
}


