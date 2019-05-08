use std::ops::{Deref, Add, Sub, Mul, Div, Neg};
use num_traits::{Zero, Float};

#[derive(Default, Debug, Clone, Copy, PartialEq)] 
pub struct Scalar<T> {
  pub v: T
}

impl<T> Scalar<T> {
  pub fn new(v: T) -> Self {
    Self{v}
  }
}

impl<T> From<T> for Scalar<T> {
  fn from(v: T) -> Self {
    Self{v}
  }
}


macro_rules! impl_scalar_op {
  ($op:tt, $optrait:ident, $func:ident) => {
    impl<T> $optrait<Scalar<T>> for Scalar<T>
      where T: $optrait<Output=T> + Copy {
      type Output = Self;
      #[inline]
      fn $func(self, rhs: Self) -> Self {
        Self{v: self.v $op rhs.v}
      }
    }
  }
}

impl_scalar_op!(+, Add, add);
impl_scalar_op!(-, Sub, sub);
impl_scalar_op!(*, Mul, mul);
impl_scalar_op!(/, Div, div);

impl<T> Neg for Scalar<T>
  where T: Neg<Output=T> {
  type Output = Self;
  #[inline]
  fn neg(self) -> Self {
    Self{v: -self.v}
  }
}

impl<T> Zero for Scalar<T>
  where T: Zero + Copy {
  #[inline]
  fn zero() -> Self {
    Self::from(T::zero())
  }
  #[inline]
  fn is_zero(&self) -> bool {
    self.v.is_zero()
  }
}
