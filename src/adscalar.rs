use crate::scalar::{Scalar};
use num_traits::{Zero, One, Float};
use std::ops::{Deref, Add, Sub, Mul, Div, Neg};

#[derive(Default, Debug, Clone, Copy)] 
pub struct ADScalar<T> {
  v: T,
  d: T
}

impl<T> ADScalar<T> {
  #[inline]
  pub fn new(v: T) -> Self
    where T: One {
    Self{v, d: T::one()}
  }
  #[inline]
  pub fn v(self) -> T
    where T: Copy {
    self.v
  }
  #[inline]
  pub fn d(self) -> T
    where T: Copy {
    self.d
  }
}

impl<T> From<T> for ADScalar<T>
  where T: Zero {
  #[inline]
  fn from(v: T) -> Self {
    Self{v, d: T::zero()}
  }
}

impl<T> From<Scalar<T>> for ADScalar<T>
  where Self: From<T>,
        T: Copy {
  #[inline]
  fn from(x: Scalar<T>) -> Self {
    Self::from(x.v())
  }
}

macro_rules! impl_addsub_op {
  ($op:tt, $optrait:ident, $func:ident) => {
    impl<T> $optrait<ADScalar<T>> for ADScalar<T>
      where T: $optrait<Output=T> + Copy {
      type Output = Self;
      #[inline]
      fn $func(self, rhs: Self) -> Self {
        Self{v: self.v $op rhs.v, d: self.d $op rhs.d}
      }
    }
    impl<T> $optrait<Scalar<T>> for ADScalar<T>
      where T: $optrait<Output=T> + Copy {
      type Output = Self;
      #[inline]
      fn $func(self, rhs: Scalar<T>) -> Self {
        Self{v: self.v $op rhs.v(), d: self.d}
      }
    }
    impl<T> $optrait<ADScalar<T>> for Scalar<T>
      where T: $optrait<Output=T> + Copy {
      type Output = ADScalar<T>;
      #[inline]
      fn $func(self, rhs: ADScalar<T>) -> Self::Output {
        Self::Output{v: self.v() $op rhs.v, d: rhs.d}
      }
    }
  }
}

impl_addsub_op!(+, Add, add);
impl_addsub_op!(-, Sub, sub);


impl<T> Mul<ADScalar<T>> for ADScalar<T>
  where T: Add<Output=T> + Mul<Output=T> + Copy {
  type Output = Self;
  #[inline]
  fn mul(self, rhs: Self) -> Self {
    Self{v: self.v * rhs.v, d: self.d * rhs.v + self.v * rhs.d}
  }
} 

impl<T> Mul<Scalar<T>> for ADScalar<T>
  where T: Add<Output=T> + Mul<Output=T> + Copy {
  type Output = Self;
  #[inline]
  fn mul(self, rhs: Scalar<T>) -> Self {
    Self{v: self.v * rhs.v(), d: self.d * rhs.v()}
  }
} 

impl<T> Mul<ADScalar<T>> for Scalar<T>
  where T: Add<Output=T> + Mul<Output=T> + Copy {
  type Output = ADScalar<T>;
  #[inline]
  fn mul(self, rhs: ADScalar<T>) -> Self::Output {
    Self::Output{v: self.v() * rhs.v, d: self.v() * rhs.d}
  }
} 

impl<T> Div for ADScalar<T>
  where T: Sub<Output=T> + Mul<Output=T> + Div<Output=T> + Copy {
  type Output = Self;
  #[inline]
  fn div(self, rhs: Self) -> Self {
    Self{v: self.v / rhs.v, d: (self.d * rhs.v - self.v * rhs.d)/(rhs.v * rhs.v)}
  }
} 

impl<T> Div<Scalar<T>> for ADScalar<T>
  where T: Sub<Output=T> + Mul<Output=T> + Div<Output=T> + Copy {
  type Output = Self;
  #[inline]
  fn div(self, rhs: Scalar<T>) -> Self {
    Self{v: self.v / rhs.v(), d: self.d / rhs.v()}
  }
} 

impl<T> Div<ADScalar<T>> for Scalar<T>
  where T: Sub<Output=T> + Mul<Output=T> + Div<Output=T> + Neg<Output=T> + Copy {
  type Output = ADScalar<T>;
  #[inline]
  fn div(self, rhs: ADScalar<T>) -> Self::Output {
    Self::Output{v: self.v() / rhs.v, d: - (self.v() * rhs.d)/(rhs.v * rhs.v)}
  }
} 

impl<T> Neg for ADScalar<T>
  where T: Neg<Output=T> {
  type Output = Self;
  fn neg(self) -> Self {
    Self{v: -self.v, d: -self.d}
  }
}

impl<T> Zero for ADScalar<T>
  where T: Zero + Copy {
  fn zero() -> Self {
    Self::from(T::zero())
  }
  fn is_zero(&self) -> bool {
    self.v.is_zero()
  }
}

