use crate::{Shape, Shaped};
use std::ops::{Deref, DerefMut, Add, AddAssign, Sub, Mul, Div};
use std::mem;
use matrixmultiply::{sgemm, dgemm};
use num_traits::{Zero, One};

#[derive(Default, Debug, Clone, PartialEq)]
pub struct Tensor<T> {
  s: Shape,
  v: Vec<T>
}

impl<T> Shaped for Tensor<T> {
  fn s(&self) -> &Shape {
    &self.s
  }
}

impl<T> Deref for Tensor<T> {
  type Target = Vec<T>;
  #[inline]
  fn deref(&self) -> &Self::Target {
    &self.v
  }
} 

impl<V> DerefMut for Tensor<V> {
  #[inline]
  fn deref_mut(&mut self) -> &mut Self::Target {
    &mut self.v
  }
}

impl<T> Tensor<T> {
  #[inline]
  pub fn new() -> Self {
    Self{s: Shape::new(), v: Vec::new()} 
  }
  #[inline]
  pub fn shape<S>(s: S) -> Self
    where Shape: From<S> {
    Self{s: s.into(), v: Vec::new()}
  }
  #[inline]
  pub unsafe fn shape_uninit<S>(s: S) -> Self
    where Shape: From<S> {
    let s = Shape::from(s);
    let n = s.product();
    let mut t = Tensor{s, v: Vec::with_capacity(n)};
    t.v.set_len(n);
    t
  }
  #[inline]
  pub fn shape_fn<S, F>(s: S, mut f: F) -> Self
    where Shape: From<S>,
          F: FnMut()->T {
    let mut t = Self::shape(s);
    t.v.resize_with(t.s.product(), || f());
    t
  }
  #[inline]
  pub fn shape_elem<S>(s: S, x: T) -> Self
    where Shape: From<S>,
          T: Clone {
    let mut t = Self::shape(s);
    t.v.resize(t.s.product(), x);
    t
  }
  #[inline]
  pub fn zeros<S>(s: S) -> Self
    where Shape: From<S>,
          T: Zero + Clone {
    Self::shape_elem(s, T::zero())
  }
  #[inline]
  pub fn ones<S>(s: S) -> Self
    where Shape: From<S>,
          T: One + Clone {
    Self::shape_elem(s, T::one())
  }
  #[inline]
  pub fn len(&self) -> usize {
    self.v.len()
  }
  #[inline]
  pub fn reshape<S>(self, s: S) -> Tensor<T>
    where Shape: From<S> {
    let t = Tensor{s: s.into(), v: self.v};
    debug_assert!({
      assert!(
        t.len() == t.s.product(), 
        format!(
          "Cannot reshape tensor with len {} to shape {:?} with product {}!",
          &t.len(), &t.s, &t.s.product()
        )
      );
      true              
    });
    t
  }
}
/*
impl<T> From<Vec<T>> for Tensor<T> {
  #[inline]
  fn from(v: Vec<T>) -> Self {
    Self{s: vec![1, v.len()].into(), v}
  }
}*/

pub trait TensorMap<A, B> {
  type Output;
  fn map<F>(&self, f: F) -> Self::Output
    where F: FnMut(A)->B;
}

impl<A, B> TensorMap<A, B> for Tensor<A>
  where A: Copy {
  type Output = Tensor<B>;
  #[inline]
  fn map<F>(&self, mut f: F) -> Tensor<B>
    where F: FnMut(A)->B {
    let mut x = unsafe { Tensor::shape_uninit(self.s.clone()) };
    self.iter().zip(x.iter_mut())
        .for_each(|(&t, x)| *x = f(t));
    x
  }
}

macro_rules! impl_tensor_op {
  ($op:tt, $optrait:ident, $func:ident) => {
    impl<'a, 'b, T> $optrait<&'b Tensor<T>> for &'a Tensor<T>
      where T: Copy + $optrait<Output=T> {
      type Output = Tensor<T>;
      #[inline]
      fn $func(self, rhs: &'b Tensor<T>) -> Self::Output {
        let s = self.s.broadcast(&rhs.s); 
        let mut t = unsafe { Tensor::shape_uninit(s) };
        self.v.chunks_exact(rhs.len())
            .zip(t.chunks_exact_mut(rhs.len()))
            .for_each(|(a, c)| {
          a.iter().zip(rhs.iter())
                  .zip(c.iter_mut())
                  .for_each(|((&a, &b), c)| *c = a $op b);
        });
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
  }
}

impl_tensor_op!(+, Add, add);
impl_tensor_op!(-, Sub, sub);
impl_tensor_op!(*, Mul, mul);
impl_tensor_op!(/, Div, div);

macro_rules! impl_tensor_inplace_op {
  ($op:tt, $optrait:ident, $func:ident) => {
    impl<'a, 'b, T> $optrait<&'b Tensor<T>> for Tensor<T>
      where T: Copy + $optrait<T> {
      #[inline]
      fn $func(&mut self, rhs: &'b Tensor<T>) {
        if self.len() == 0 {
          mem::swap(self, &mut rhs.clone());
        }
        else {
          debug_assert!(
            self.s.can_broadcast(&rhs.s),
            format!("Cannot perform inplace op with shapes {:?} and {:?}!", &self.s, &rhs.s)
          );
          self.v.chunks_exact_mut(rhs.len())
              .for_each(|a| {
            a.iter_mut().zip(rhs.iter())
                        .for_each(|(a, &b)| *a $op b);
          });
        }
      }
    }
    /*impl<'b, T> $optrait<Tensor<T>> for Tensor<T>
      where T: Copy + $optrait<Output=T> {
      type Output = Tensor<T>;
      #[inline]
      fn $func(self, rhs: &'b Tensor<T>) -> Self::Output {
        (&self).add(rhs)
      }
    }*/
  }
}

impl_tensor_inplace_op!(+=, AddAssign, add_assign);
//impl_tensor_op!(-, Sub, sub);
//impl_tensor_op!(*, Mul, mul);
//impl_tensor_op!(/, Div, div);

pub trait Matmul<R> {
  type Output;
  fn mm(self, rhs: R) -> Self::Output;
}

macro_rules! impl_tensor_mm {
  ($t:ty, $mm:ident) => {
    impl<'a, 'b> Matmul<&'b Tensor<$t>> for &'a Tensor<$t> {
      type Output = Tensor<$t>;
      #[inline]
      fn mm(self, rhs: &'b Tensor<$t>) -> Tensor<$t> {
        let s = self.s.broadcast_mm(&rhs.s);
        let mut t = unsafe { Tensor::shape_uninit(s) };
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
    impl<'b> Matmul<&'b Tensor<$t>> for Tensor<$t> {
      type Output = Tensor<$t>;
      fn mm(self, rhs: &'b Tensor<$t>) -> Tensor<$t> {
        (&self).mm(rhs)
      }
    }
  }
}

impl_tensor_mm!(f32, sgemm);
impl_tensor_mm!(f64, dgemm);

#[cfg(test)]
mod tests {
  use crate::core::shape::Shape;
  use crate::core::tensor::Tensor;
  #[test]
  fn test_ones() {
    let x = Tensor::<f32>::ones(vec![1, 2]);
    assert_eq!(x.v, vec![1., 1.]); 
  }
  #[test]
  fn test_reshape() {
    let x = Tensor::from(vec![1., 2.]);
    let y = x.reshape(vec![2, 1]);
    assert_eq!(y.s, Shape::from(vec![2, 1]));
  }
  #[test]
  fn test_add() {
    let x = Tensor::<f32>{s: vec![2, 2].into(), v: vec![1., 2., 3., 4.]}; 
    let y = Tensor::from(vec![1., 2.]);
    let z = Tensor{s: vec![2, 2].into(), v: vec![2., 4., 4., 6.]};
    assert_eq!(&x + &y, z);
  }
  #[test]
  fn test_mm() {
    use crate::core::tensor::Matmul;
    let x = Tensor::<f32>{s: vec![1, 2, 2].into(), v: vec![1., 2., 3., 4.]};
    let w = Tensor::ones(vec![2, 2, 3]);
    let y = Tensor{s: vec![2, 2, 3].into(), v: vec![3., 3., 3., 7., 7., 7.,
                                                    3., 3., 3., 7., 7., 7.]};
    assert_eq!(x.mm(&w), y);
  }
}
 
