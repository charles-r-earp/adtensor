use crate::core::shape::Shape;
use std::ops::{Deref, DerefMut, Add, AddAssign, Sub, Mul, Div};
use std::mem;
use matrixmultiply::{sgemm, dgemm};
use num_traits::{Zero, One};

#[derive(Default, Debug, Clone, PartialEq)]
pub struct Tensor<T> {
  pub s: Shape,
  pub v: Vec<T>
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
  pub fn shape_fn<S, F>(s: S, f: F) -> Self
    where Shape: From<S>,
          F: FnMut()->T {
    let s = Shape::from(s);
    let n = s.product();
    let mut t = Tensor{s, v: Vec::with_capacity(n)};
    t.v.resize_with(n, f);
    t
  }
  #[inline]
  pub fn shape_elem<S>(s: S, x: T) -> Self
    where Shape: From<S>,
          T: Clone {
    let s = Shape::from(s);
    let n = s.product();
    let mut t = Tensor{s, v: Vec::with_capacity(n)};
    t.v.resize(n, x);
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
  /*#[inline]
  pub fn init_fn<F>(self, f: F) -> Self
    where F: FnMut()->T {
    let mut t = Self{s: self.s, v: self.v};
    let n = t.s.product();
    t.v.reserve_exact(n);
    unsafe { t.v.set_len(0) };
    t.v.resize_with(n, f);
    t
  }*/
  #[inline]
  pub fn map<F>(&self, mut f: F) -> Self
    where F: FnMut(T)->T,
          T: Copy {
    let mut x = unsafe { Tensor::shape_uninit(self.s.clone()) };
    self.iter().zip(x.iter_mut())
        .for_each(|(&t, x)| *x = f(t));
    x
  }
}

impl<T> From<Vec<T>> for Tensor<T> {
  #[inline]
  fn from(v: Vec<T>) -> Self {
    Self{s: vec![1, v.len()].into(), v}
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
  }
}

impl_tensor_op!(+, Add, add);
impl_tensor_op!(-, Sub, sub);
impl_tensor_op!(*, Mul, mul);
impl_tensor_op!(/, Div, div);

impl<'b, T> AddAssign<&'b Tensor<T>> for Tensor<T>
  where T: Copy + Add<Output=T> {
  fn add_assign(&mut self, rhs: &'b Tensor<T>) {
    if self.len() == 0 {
      let mut tmp = rhs.clone();
      mem::swap(self, &mut tmp);
    }
    else {
      let mut tmp = self as &Self + rhs;
      mem::swap(self, &mut tmp);  
    }
  }
}

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
 
