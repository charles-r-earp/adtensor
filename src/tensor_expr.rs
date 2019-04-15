use crate::shape::Shape;
use crate::tensor::Tensor;
use std::ops::{Add};
use std::iter::{IntoIterator, Iterator, Zip, Cycle, Take};
use std::slice;

pub struct TensorExpr<E> {
  s: Shape,
  e: E
}

impl<E> TensorExpr<E> {
  fn new(s: Shape, e: E) -> Self {
    Self{s, e}
  }
  pub fn shape<'a>(&'a self) -> &'a Shape {
    &self.s
  }
} 

pub trait Reshape {
  type Output;
  fn reshape<D>(d: D) -> Self::Output
    where D: AsRef<[usize]>;
}

impl<E> IntoIterator for TensorExpr<E>
  where E: IntoIterator {
  type Item = E::Item;
  type IntoIter = E::IntoIter;
  fn into_iter(self) -> Self::IntoIter {
    self.e.into_iter()
  }
} 

type ItItem<I> = <I as Iterator>::Item;

macro_rules! impl_binary_op {
  ($optrait:ident, $ex:ident, $func:tt, $op:tt) => {
    pub struct $ex<A, B> {
      z: Zip<A, B>
    }
    impl<A, B> $ex<A, B> {
      pub fn new(a: A, b: B) -> Self
        where A: Iterator,
              B: Iterator {
        Self{z: a.zip(b)}
      }
    }
    impl<A, B> Iterator for $ex<A, B>
      where A: Iterator,
            B: Iterator,
            ItItem<A>: $optrait<ItItem<B>>,
            Zip<A, B>: Iterator<Item=(ItItem<A>, ItItem<B>)> {
      type Item = <ItItem<A> as $optrait<ItItem<B>>>::Output;
      fn next(&mut self) -> Option<Self::Item> {
        self.z.next().and_then(|(a, b)| Some(a $op b))
      }
    } 
    impl<'a, 'b, T> $optrait<&'b Tensor<T>> for &'a Tensor<T>
      where $ex<Cycle<slice::Iter<'a, T>>, Cycle<slice::Iter<'b, T>>>: Iterator {
      type Output = TensorExpr<Take<$ex<Cycle<slice::Iter<'a, T>>, Cycle<slice::Iter<'b, T>>>>>;
      #[inline]
      fn $func(self, rhs: &'b Tensor<T>) -> Self::Output {
        let s = Shape::broadcast(&*self.shape(), &*rhs.shape());
        let n = s.size();
        let e = AddExpr::new(self.iter().cycle(), rhs.iter().cycle()).take(n);
        TensorExpr{s, e}
      }
    }
    impl<'a, T, E> $optrait<TensorExpr<E>> for &'a Tensor<T>
      where E: IntoIterator,
            E::IntoIter: Clone,
            $ex<Cycle<slice::Iter<'a, T>>, Cycle<E::IntoIter>>: Iterator {
      type Output = TensorExpr<Take<$ex<Cycle<slice::Iter<'a, T>>, Cycle<E::IntoIter>>>>;
      #[inline]
      fn $func(self, rhs: TensorExpr<E>) -> Self::Output {
        let s = Shape::broadcast(&*self.shape(), &*rhs.shape());
        let n = s.size();
        let e = AddExpr::new(self.iter().cycle(), rhs.into_iter().cycle()).take(n);
        TensorExpr{s, e}
      }
    }
    impl<'b, T, E> $optrait<&'b Tensor<T>> for TensorExpr<E>
      where E: IntoIterator,
            E::IntoIter: Clone,
            $ex<Cycle<E::IntoIter>, Cycle<slice::Iter<'b, T>>>: Iterator {
      type Output = TensorExpr<Take<$ex<Cycle<E::IntoIter>, Cycle<slice::Iter<'b, T>>>>>;
      #[inline]
      fn $func(self, rhs: &'b Tensor<T>) -> Self::Output {
        let s = Shape::broadcast(&*self.shape(), &*rhs.shape());
        let n = s.size();
        let e = AddExpr::new(self.into_iter().cycle(), rhs.iter().cycle()).take(n);
        TensorExpr{s, e}
      }
    }
  }
}

impl_binary_op!(Add, AddExpr, add, +);
    



