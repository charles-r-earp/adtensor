use crate::shape::{Shape};
use std::ops::{Deref, DerefMut, Add, Sub, Mul, Div};
use matrixmultiply::{sgemm, dgemm};

#[derive(Default, Debug, Clone)]
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
  pub fn len(&self) -> usize {
    self.v.len()
  }
  #[inline]
  pub fn reshape<S>(self, s: S) -> Tensor<T>
    where Shape: From<S> {
    let t = Tensor{s: s.into(), v: self.v};
    debug_assert!({
      assert!(
        t.v.len() == 0 || t.len() == t.s.product(), 
        format!(
          "Can not reshape tensor with len {} to shape {:?} with product {}!",
          &t.len(), &t.s, &t.s.product()
        )
      );
      true              
    });
    t
  }
  #[inline]
  pub fn init_fn<F>(self, f: F) -> Self
    where F: FnMut()->T {
    let mut t = Self{s: self.s, v: self.v};
    let n = t.s.product();
    t.v.reserve_exact(n);
    unsafe { t.v.set_len(0) };
    t.v.resize_with(n, f);
    t
  }
  /*#[inline]
  pub fn map<F>(&self, f: F) -> Self
    where F: FnMut(&T)->T,
          T: Copy {
    let n = self.s.product();
    let mut x = Tensor{s: self.s.clone(), v: Vec::with_capacity(n)};
    unsafe { x.v.set_len(n) };
    self.iter().zip(x.iter_mut())
        .for_each(|(t, x)| *x = *t);
    x
  }*/
}

impl<T> From<Vec<T>> for Tensor<T> {
  fn from(v: Vec<T>) -> Self {
    Self{s: vec![v.len()].into(), v}
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
        let n = s.product();
        let mut t = Tensor{s, v: Vec::with_capacity(n)};
        unsafe { t.v.set_len(n) };
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
    /*impl<'b, T> $optrait<&'b Tensor<T>> for Tensor<T>
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
    }*/
  }
}

impl_tensor_op!(+, Add, add);
impl_tensor_op!(-, Sub, sub);
impl_tensor_op!(*, Mul, mul);
impl_tensor_op!(/, Div, div);


macro_rules! impl_tensor_mm {
  ($t:ty, $mm:ident) => {
    impl Tensor<$t> {
      #[inline]
      pub fn mm<'a, 'b>(&'a self, rhs: &'b Tensor<$t>) -> Tensor<$t> {
        let s = self.s.broadcast_mm(&rhs.s);
        let n = s.product();
        let mut t = Tensor{s, v: Vec::with_capacity(n)};
        unsafe { t.v.set_len(n) };
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

    

 
