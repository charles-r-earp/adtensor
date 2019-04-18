use crate::shape::Shape;
use crate::tensor::Tensor;
use std::ops::{Add, Sub, Mul, Div};
use std::iter::{IntoIterator, Iterator, Zip, Cycle, Take};
use std::slice;

#[derive(Debug, Clone)]
pub struct TensorExpr<E> {
  s: Shape,
  e: E
}

impl<E> TensorExpr<E> {
  #[inline]
  fn new(s: Shape, e: E) -> Self {
    Self{s, e}
  }
  #[inline]
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
  #[inline]
  fn into_iter(self) -> Self::IntoIter {
    self.e.into_iter()
  }
} 

type ItItem<I> = <I as Iterator>::Item;

macro_rules! impl_binary_op {
  ($optrait:ident, $ex:ident, $func:tt, $op:tt) => {
    #[derive(Debug, Clone)]
    pub struct $ex<A, B> {
      z: Zip<A, B>
    }
    impl<A, B> $ex<A, B> {
      #[inline]
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
      #[inline]
      fn next(&mut self) -> Option<Self::Item> {
        self.z.next().and_then(|(a, b)| Some(a $op b))
      }
    } 
    impl<'a, 'b, T> $optrait<&'b Tensor<T>> for &'a Tensor<T>
      where $ex<slice::Iter<'a, T>, slice::Iter<'b, T>>: Iterator {
      type Output = TensorExpr<$ex<slice::Iter<'a, T>, slice::Iter<'b, T>>>;
      #[inline]
      fn $func(self, rhs: &'b Tensor<T>) -> Self::Output {
        let s = Shape::broadcast(&*self.shape(), &*rhs.shape());
        let n = s.size();
        let e = $ex::new(self.iter(), rhs.iter());
        TensorExpr{s, e}
      }
    }
    /*impl<'a, 'b, T> $optrait<&'b Tensor<T>> for &'a Tensor<T>
      where $ex<Cycle<slice::Iter<'a, T>>, Cycle<slice::Iter<'b, T>>>: Iterator {
      type Output = TensorExpr<Take<$ex<Cycle<slice::Iter<'a, T>>, Cycle<slice::Iter<'b, T>>>>>;
      #[inline]
      fn $func(self, rhs: &'b Tensor<T>) -> Self::Output {
        let s = Shape::broadcast(&*self.shape(), &*rhs.shape());
        let n = s.size();
        let e = $ex::new(self.iter().cycle(), rhs.iter().cycle()).take(n);
        TensorExpr{s, e}
      }
    }*/
    impl<'a, T, E> $optrait<TensorExpr<E>> for &'a Tensor<T>
      where E: IntoIterator,
            E::IntoIter: Clone,
            $ex<Cycle<slice::Iter<'a, T>>, Cycle<E::IntoIter>>: Iterator {
      type Output = TensorExpr<Take<$ex<Cycle<slice::Iter<'a, T>>, Cycle<E::IntoIter>>>>;
      #[inline]
      fn $func(self, rhs: TensorExpr<E>) -> Self::Output {
        let s = Shape::broadcast(&*self.shape(), &*rhs.shape());
        let n = s.size();
        let e = $ex::new(self.iter().cycle(), rhs.into_iter().cycle()).take(n);
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
        let e = $ex::new(self.into_iter().cycle(), rhs.iter().cycle()).take(n);
        TensorExpr{s, e}
      }
    }
    impl<A, B> $optrait<TensorExpr<B>> for TensorExpr<A>
      where A: IntoIterator,
            A::IntoIter: Clone,
            B: IntoIterator,
            B::IntoIter: Clone,
            $ex<Cycle<A::IntoIter>, Cycle<B::IntoIter>>: Iterator {
      type Output = TensorExpr<Take<$ex<Cycle<A::IntoIter>, Cycle<B::IntoIter>>>>;
      #[inline]
      fn $func(self, rhs: TensorExpr<B>) -> Self::Output {
        let s = Shape::broadcast(&*self.shape(), &*rhs.shape());
        let n = s.size();
        let e = $ex::new(self.into_iter().cycle(), rhs.into_iter().cycle()).take(n);
        TensorExpr{s, e}
      }
    }
  }
}

impl_binary_op!(Add, AddExpr, add, +);
impl_binary_op!(Sub, SubExpr, sub, -);
impl_binary_op!(Mul, MulExpr, mul, *);
impl_binary_op!(Div, DivExpr, div, /);
    



