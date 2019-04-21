use crate::scalar::{Scalar, ScalarLike};
use std::ops::{Deref, DerefMut, Index, IndexMut, Add, Sub, Mul, Div};
use std::iter::{Iterator};
use std::cmp::{min, max};
use std::mem;
use std::fmt::{Debug, Formatter, Result};
use matrixmultiply::{sgemm, dgemm};

#[derive(Default, Clone)]
pub struct Shape {
  d: Vec<usize>
}

impl Deref for Shape {
  type Target = [usize];
  #[inline]
  fn deref(&self) -> &Self::Target {
    self.d.deref()
  }
}

impl DerefMut for Shape {
  #[inline]
  fn deref_mut(&mut self) -> &mut Self::Target {
    self.d.deref_mut()
  }
}

impl Index<isize> for Shape {
  type Output = usize;
  #[inline]
  fn index(&self, i: isize) -> &usize {
    let n = self.len() as isize;
    if i < 0 {
      &self.d[-(1+i) as usize]
    }
    else {
      &self.d[(n - (1 + i)) as usize]
    }
  }
}

impl IndexMut<isize> for Shape {
  #[inline]
  fn index_mut(&mut self, i: isize) -> &mut usize {
    let n = self.len() as isize;
    if i < 0 {
      &mut self.d[-(1+i) as usize]
    }
    else {
      &mut self.d[(n - (1 + i)) as usize]
    }
  }
}

impl Shape {
  #[inline]
  pub fn new(d: Vec<usize>) -> Self {
    Self{d}
  }
  #[inline]
  pub fn from<D>(d: D) -> Self
    where D: AsRef<[usize]> {
    Self::new(d.as_ref().into())
  }
  #[inline]
  pub fn zeros(n: usize) -> Self {
    let mut s = Self::default();
    s.d.resize(n, 0);
    s
  }
  #[inline]
  unsafe fn with_len(n: usize) -> Self {
    let mut s = Self::default();
    s.d.reserve_exact(n);
    s.d.set_len(n);
    s
  }
  #[inline]
  pub fn len(&self) -> usize {
    self.d.len()
  }
  #[inline]
  pub fn product(&self) -> usize {
    self.iter().product::<usize>()
  } 
  #[inline]
  pub fn broadcast<'b>(&self, rhs: &'b Self) -> (usize, usize, Self) {
    let n1 = self.product();
    let n2 = rhs.product();
    debug_assert!({
      !self.iter().rev()
          .zip(rhs.iter().rev())
          .any(|(a, b)| {
        if n1 > n2 {a < b}
        else {a > b}
      })
    }, format!("Can't broadcast shapes {:?} and {:?}!", &self, &rhs));
    (n1, n2, if n1 >= n2 {self.clone()} else {rhs.clone()})
  }
  #[inline]
  pub fn broadcast_matmul<'b>(&self, rhs: &'b Self) -> Self {
    debug_assert_eq!(self[0], rhs[1], "{}", format!("Can't matmul shapes {:?} and {:?}!", &self, &rhs));
    let mut s = self.clone();
    s[0] = rhs[0];
    s
  }
}

impl Debug for Shape {
  #[inline]
  fn fmt(&self, f: &mut Formatter) -> Result {
    self.d.fmt(f)
  }
}
       
#[derive(Default, Debug, Clone)] 
pub struct Tensor<T> {
  s: Shape,
  v: Vec<T>
}

impl<T> Deref for Tensor<T> {
  type Target = [T];
  #[inline]
  fn deref(&self) -> &Self::Target {
    self.v.deref()
  }
} 

impl<T> DerefMut for Tensor<T> {
  #[inline]
  fn deref_mut(&mut self) -> &mut Self::Target {
    self.v.deref_mut()
  }
} 

impl<T> Tensor<T> {
  #[inline]
  pub fn new(s: Shape, v: Vec<T>) -> Self {
    debug_assert_eq!(s.product(), v.len(), "Tensor shape doesn't match data vec!");
    Self{s, v}
  }
  #[inline]
  pub fn with_shape(s: Shape) -> Self {
    Self{s, v: Vec::<T>::new()}
  } 
  #[inline]
  pub fn with_elem(s: Shape, x: T) -> Self
    where T: Copy {
    let n = s.product();
    let mut t = Self{s, v: Vec::with_capacity(n)};
    t.v.resize(n, x);
    t
  }
  #[inline]
  pub fn with_fn<F>(s: Shape, f: &mut F) -> Self
    where F: FnMut()->T {
    let n = s.product();
    let mut t = Self{s, v: Vec::with_capacity(n)};
    t.v.resize_with(n, f);
    t
  }
  #[inline]
  unsafe fn uninitialized(s: Shape) -> Self {
    let n = s.product();
    let mut t = Self{s, v: Vec::with_capacity(n)};
    t.v.set_len(t.v.capacity());
    t
  }
  #[inline]
  pub fn init_fn<F>(&mut self, f: &mut F) -> &mut Self
    where F: FnMut()->T {
    if self.is_empty() {
      let n = self.s.product();
      self.v.reserve_exact(n);
      self.v.resize_with(n, f);
    }
    else {
      debug_assert_eq!(self.s.product(), self.v.len());
      self.iter_mut().for_each(|x| *x = f());
    }
    self
  }
  #[inline]
  fn as_ptr(&self) -> *const T {
    self.v.as_ptr()
  }
  #[inline]
  fn as_mut_ptr(&mut self) -> *mut T {
    self.v.as_mut_ptr()
  }
  #[inline]
  pub fn s<'a>(&'a self) -> &'a Shape {
    &self.s
  }  
  #[inline]
  pub fn len(&self) -> usize {
    self.v.len()
  }
  #[inline]
  pub fn is_empty(&self) -> bool {
    self.v.is_empty()
  }
  #[inline]
  pub fn clear(&mut self) {
    self.v.clear()
  }
}

macro_rules! impl_tensor_op {
  ($op:tt, $optrait:ident, $func:ident) => {
    impl<'a, 'b, T> $optrait<&'b Tensor<T>> for &'a Tensor<T>
      where T: Copy + $optrait<Output=T> {
      type Output = Tensor<T>;
      #[inline]
      fn $func(self, rhs: &'b Tensor<T>) -> Self::Output {
        let (n1, n2, s) = self.s.broadcast(&rhs.s); 
        let mut t = unsafe { Tensor::<T>::uninitialized(s) };
        if n1 > n2 {
          self.v.chunks_exact(n2)
              .zip(t.chunks_exact_mut(n2))
              .for_each(|(a, c)| {
          a.iter().zip(rhs.iter())
                  .zip(c.iter_mut())
                  .for_each(|((&a, &b), c)| *c = a $op b);
          });
        }
        else if n1 < n2 {
          rhs.v.chunks_exact(n1)
              .zip(t.chunks_exact_mut(n1))
              .for_each(|(b, c)| {
          self.iter().zip(b.iter())
                  .zip(c.iter_mut())
                  .for_each(|((&a, &b), c)| *c = a $op b);
          });
        }
        else {
          self.iter().zip(rhs.iter())
                     .zip(t.iter_mut())
                     .for_each(|((&a, &b), c)| *c = a $op b);
        }
        t
      }
    }
    impl<'b, T> $optrait<&'b Tensor<T>> for Tensor<T>
      where T: Copy + $optrait<Output=T> {
      type Output = Tensor<T>;
      #[inline]
      fn $func(self, rhs: &'b Tensor<T>) -> Self::Output {
        &self $op rhs
      }
    }
    impl<'a, T> $optrait<Tensor<T>> for &'a Tensor<T>
      where T: Copy + $optrait<Output=T> {
      type Output = Tensor<T>;
      #[inline]
      fn $func(self, rhs: Tensor<T>) -> Self::Output {
        self $op &rhs
      }
    }
    impl<T> $optrait<Tensor<T>> for Tensor<T>
      where T: Copy + $optrait<Output=T> {
      type Output = Tensor<T>;
      #[inline]
      fn $func(self, rhs: Tensor<T>) -> Self::Output {
        &self $op &rhs
      }
    }
  }
}

impl_tensor_op!(+, Add, add);
impl_tensor_op!(-, Sub, sub);
impl_tensor_op!(*, Mul, mul);
impl_tensor_op!(/, Div, div);

pub trait Matmul<R> {
  type Output;
  fn mm(self, rhs: R) -> Self::Output;
}

macro_rules! impl_tensor_mm {
  ($t:ty, $mm:ident) => {
    impl<'a, 'b> Matmul<&'b Tensor<$t>> for &'a Tensor<$t> {
      type Output = Tensor<$t>;
      #[inline]
      fn mm(self, rhs: &'b Tensor<$t>) -> Self::Output {
        let mut t = unsafe { Tensor::<$t>::uninitialized(self.s.broadcast_matmul(&rhs.s)) };
        let m = self.s[1];
        let k = self.s[0];
        let n = rhs.s[0];
        self.v.chunks_exact(m * k).cycle()
              .zip(rhs.v.chunks_exact(k * n).cycle())
              .zip(t.v.chunks_exact_mut(m * n))
              .for_each(|((a, b), c)| {
          unsafe {
            $mm(
                m,
                k,
                n,
                1.,
                a.as_ptr(),
                k as isize,
                1,
                b.as_ptr(),
                n as isize,
                1,
                0.,
                c.as_mut_ptr(),
                n as isize,
                1
              )
            };
        });
        t
      }
    }
  }
}

impl_tensor_mm!(f32, sgemm);
impl_tensor_mm!(f64, dgemm);
  /*          
pub trait TensorLike<T> {
  fn map<F, X>(&self, f: &mut F) -> Self
    where F: FnMut(Scalar<T>)->Scalar<T>;
}

impl<T> TensorLike<T> for Tensor<T>
  where T: Copy {
  #[inline]
  fn map<F>(&self, f: &mut F) -> Self
    where F: FnMut(Scalar<T>)->Scalar<T> {
    let mut t = unsafe { Self::uninitialized(self.s.clone()) };
    t.iter_mut()
     .zip(self.iter())
     .for_each(|(y, &x)| *y = f(Scalar::from(x)).v());
    t
  }
}
*/
