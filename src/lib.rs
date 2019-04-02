pub mod vector;
pub use vector::*;
pub mod matrix;
pub use matrix::*;
pub mod vector_ops;
pub use vector_ops::*;
pub mod tensor;
pub use tensor::*;
/*
use std::ops::{Add, Sub, Mul, Div};
use std::iter::{IntoIterator, Iterator, repeat};
use std::fmt::{Debug, Formatter, Result};
use std::marker::{PhantomData};
use typenum::{Unsigned, TArr, ATerm, Prod, Max, Diff, Same, NonZero};
use generic_array::{GenericArray, ArrayLength, sequence::Concat};

type Maxed<A, B> = <A as Max<B>>::Output;

pub trait ReduceMul {
  type Output;
  fn reducemul(&self) -> Self::Output;
}

pub type ReduceProd<A> = <A as ReduceMul>::Output;

impl<V> ReduceMul for TArr<V, ATerm> {
  type Output = V;
  fn reducemul(&self) -> Self::Output {
    unsafe { ::core::mem::uninitialized() }
  }
}

impl<V, A> ReduceMul for TArr<V, A>
  where A: ReduceMul,
        V: Mul<ReduceProd<A>> {
  type Output = Prod<V, ReduceProd<A>>;
  fn reducemul(&self) -> Self::Output {
    unsafe { ::core::mem::uninitialized() }
  }
}

pub trait CanReshape<S> {}

impl<V, A, S> CanReshape<S> for TArr<V, A>
  where TArr<V, A>: ReduceMul,
        S: ReduceMul,
        ReduceProd<TArr<V, A>>: Same<ReduceProd<TArr<V, A>>>,
        <ReduceProd<TArr<V, A>> as Same<ReduceProd<TArr<V, A>>>>::Output: NonZero {}

/*pub trait TCats<A> {
  type Output;
}

pub type TCat<A, B> = <A as TCats<B>>::Output;

impl<A> TCats<A> for ATerm
  where A: TypeArray {
  type Output = A;
}

impl<V, A> TCats<ATerm> for TArr<V, A> {
  type Output = Self;
}

impl<V, W, A, B> TCats<TArr<W, B>> for TArr<V, A>
  where TArr<W, TArr<V, A>>: TCats<B> {
  type Output = TCat<TArr<W, TArr<V, A>>, B>;
}*/

pub trait UnsignedName {
  fn name() -> String;  
}

impl<U> UnsignedName for U
  where U: Unsigned {
  fn name() -> String {
    format!("U{}", U::to_usize())
  }
}

pub trait ShapeName {
  fn name() -> String;
}

impl<V> ShapeName for TArr<V, ATerm>
  where V: UnsignedName {
  fn name() -> String { V::name() }
}

impl<V, A> ShapeName for TArr<V, A>
  where V: UnsignedName,
        A: ShapeName {
  fn name() -> String { 
    format!("{}, {}", V::name(), A::name()) 
  }
} 

pub trait One {
  fn one() -> Self;
}

impl One for f32 {
  fn one() -> Self { 1.0 }
}

impl One for f64 {
  fn one() -> Self { 1.0 }
}*/
/*
pub struct Scalar<T> {
  pub v: T
}

pub fn scalar<T>(v: T) -> Scalar<T> {
  Scalar::<T>{v}
}

impl<T> Default for Scalar<T>
  where T: Default {
  fn default() -> Self {
    scalar(T::default())
  }
}

impl<T> Clone for Scalar<T>
  where T: Clone {
  fn clone(&self) -> Self {
    scalar(self.v.clone())
  }
}

impl<T> Copy for Scalar<T>
  where T: Copy {}

impl<T> Debug for Scalar<T> 
  where T: Debug {
  fn fmt(&self, f: &mut Formatter) -> Result {
    write!(f, "({:?})", &self.v)
  }
} 

impl<T> Add<Scalar<T>> for Scalar<T>
  where T: Add<T, Output=T> {
  type Output = Self;
  fn add(self, rhs: Self) -> Self {
    scalar(self.v + rhs.v)
  }
}

impl<T> Sub<Scalar<T>> for Scalar<T>
  where T: Sub<T, Output=T> {
  type Output = Self;
  fn sub(self, rhs: Self) -> Self {
    scalar(self.v - rhs.v)
  }
}

impl<T> Mul<Scalar<T>> for Scalar<T>
  where T: Mul<T, Output=T> {
  type Output = Self;
  fn mul(self, rhs: Self) -> Self {
    scalar(self.v * rhs.v)
  }
}

impl<T> Div<Scalar<T>> for Scalar<T>
  where T: Div<T, Output=T> {
  type Output = Self;
  fn div(self, rhs: Self) -> Self {
    scalar(self.v / rhs.v)
  }
}

pub struct ADScalar<T, P>
  where P: ArrayLength<T> {
  pub v: T,
  pub g: GenericArray<T, P>
}

pub fn adscalar<T, P>(v: T, g: GenericArray<T, P>) -> ADScalar<T, P>
  where P: ArrayLength<T> {
  ADScalar::<T, P>{v, g}
}

impl<T, P> Default for ADScalar<T, P>
  where P: ArrayLength<T>,
        T: Default {
  fn default() -> Self {
    adscalar(T::default(), GenericArray::<T, P>::default())
  }
}

impl<T, P> Clone for ADScalar<T, P>
  where P: ArrayLength<T>,
        T: Clone {
  fn clone(&self) -> Self {
    adscalar(self.v.clone(), self.g.clone())
  }
}

impl<T, P> Copy for ADScalar<T, P> 
  where P: ArrayLength<T>, T: Copy, GenericArray<T, P>: Copy {}

impl<T, P> Debug for ADScalar<T, P> 
  where P: ArrayLength<T>, T: Debug {
  fn fmt(&self, f: &mut Formatter) -> Result {
    write!(f, "({:?}, {:?})", &self.v, &self.g)
  }
} 


impl<T, P> Add<Scalar<T>> for ADScalar<T, P>
  where P: ArrayLength<T>,
        T: Add<T, Output=T> {
  type Output = Self;
  fn add(self, rhs: Scalar<T>) -> Self {
    adscalar(self.v + rhs.v, self.g)
  }
} 

impl<T, P> Add<ADScalar<T, P>> for Scalar<T>
  where P: ArrayLength<T>,
        T: Add<T, Output=T> {
  type Output = ADScalar<T, P>;
  fn add(self, rhs: ADScalar<T, P>) -> Self::Output {
    adscalar(self.v + rhs.v, rhs.g)
  }
} 

impl<T, P> Sub<Scalar<T>> for ADScalar<T, P>
  where P: ArrayLength<T>,
        T: Sub<T, Output=T> {
  type Output = Self;
  fn sub(self, rhs: Scalar<T>) -> Self {
    adscalar(self.v - rhs.v, self.g)
  }
} 

impl<T, P> Sub<ADScalar<T, P>> for Scalar<T>
  where P: ArrayLength<T>,
        T: Sub<T, Output=T> {
  type Output = ADScalar<T, P>;
  fn sub(self, rhs: ADScalar<T, P>) -> Self::Output {
    adscalar(self.v - rhs.v, rhs.g)
  }
} 

impl<T, P> Mul<Scalar<T>> for ADScalar<T, P>
  where P: ArrayLength<T>,
        T: Mul<T, Output=T> + Clone {
  type Output = Self;
  fn mul(self, rhs: Scalar<T>) -> Self {
    adscalar(self.v * rhs.v.clone(), self.g.map(|d| d * rhs.v.clone()))
  }
} 

impl<T, P> Div<ADScalar<T, P>> for Scalar<T>
  where P: ArrayLength<T>,
        T: Div<T, Output=T> + Clone {
  type Output = ADScalar<T, P>;
  fn div(self, rhs: ADScalar<T, P>) -> Self::Output {
    adscalar(self.v.clone() / rhs.v, rhs.g.map(|d| self.v.clone() / d))
  }
} 

impl<T, P1> ADScalar<T, P1> 
  where P1: Unsigned + ArrayLength<T> {
  pub fn grad<P2, F>(self, rhs: GenericArray<T, P2>, f: F) -> GenericArray<T, Maxed<P1, P2>> 
    where P1: Max<P2>,
          P2: Unsigned + ArrayLength<T>,
          Maxed<P1, P2>: ArrayLength<T>,
          F: Fn(T, T) -> T,
          T: Default + Clone {
    let p1 = <P1 as Unsigned>::to_usize();
    let p2 = <P2 as Unsigned>::to_usize();
    if p1 == p2 {
      return self.g.into_iter()
                   .zip(rhs.into_iter())
                   .map(|(d1, d2)| f(d1, d2))
                   .collect();
    }
    else if p1 < p2 {
      return self.g.into_iter()
                 .chain(repeat(T::default()))
                 .zip(rhs.into_iter())
                 .map(|(d1, d2)| f(d1, d2))
                 .collect();
    }
    else {
      return rhs.into_iter()
             .chain(repeat(T::default()))
             .zip(self.g.into_iter())
             .map(|(d1, d2)| f(d1, d2))
             .collect();
    }
  }
}

impl<T, P1, P2> Add<ADScalar<T, P2>> for ADScalar<T, P1>
  where P1: Unsigned + Max<P2> + ArrayLength<T>,
        P2: Unsigned + ArrayLength<T>,
        Maxed<P1, P2>: ArrayLength<T>,
        ADScalar<T, Maxed<P1, P2>>: Default,
        T: Add<T, Output=T> + Default + Clone {
  type Output = ADScalar<T, Maxed<P1, P2>>;
  fn add(self, rhs: ADScalar<T, P2>) -> Self::Output {
    adscalar(self.v.clone() + rhs.v.clone(), self.grad(rhs.g, |d1, d2| d1 + d2))
  }
}

impl<T, P1, P2> Sub<ADScalar<T, P2>> for ADScalar<T, P1>
  where P1: Unsigned + Max<P2> + ArrayLength<T>,
        P2: Unsigned + ArrayLength<T>,
        Maxed<P1, P2>: ArrayLength<T>,
        ADScalar<T, Maxed<P1, P2>>: Default,
        T: Sub<T, Output=T> + Default + Clone {
  type Output = ADScalar<T, Maxed<P1, P2>>;
  fn sub(self, rhs: ADScalar<T, P2>) -> Self::Output {
    adscalar(self.v.clone() - rhs.v.clone(), self.grad(rhs.g, |d1, d2| d1 - d2))
  }
}

impl<T, P1, P2> Mul<ADScalar<T, P2>> for ADScalar<T, P1>
  where P1: Unsigned + Max<P2> + ArrayLength<T>,
        P2: Unsigned + ArrayLength<T>,
        Maxed<P1, P2>: ArrayLength<T>,
        ADScalar<T, Maxed<P1, P2>>: Default,
        T: Add<T, Output=T> + Mul<T, Output=T> + Default + Clone {
  type Output = ADScalar<T, Maxed<P1, P2>>;
  fn mul(self, rhs: ADScalar<T, P2>) -> Self::Output {
    let v1 = self.v.clone();
    let v2 = rhs.v.clone();
    adscalar(v1.clone() * v2.clone(), self.grad(rhs.g, |d1, d2| d1 * v2.clone() + v1.clone() * d2))
  }
}

impl<T, P1, P2> Div<ADScalar<T, P2>> for ADScalar<T, P1>
  where P1: Unsigned + Max<P2> + ArrayLength<T>,
        P2: Unsigned + ArrayLength<T>,
        Maxed<P1, P2>: ArrayLength<T>,
        ADScalar<T, Maxed<P1, P2>>: Default,
        T: Sub<T, Output=T> + Div<T, Output=T> + Mul<T, Output=T> + Default + Clone + One {
  type Output = ADScalar<T, Maxed<P1, P2>>;
  fn div(self, rhs: ADScalar<T, P2>) -> Self::Output {
    let v1 = self.v.clone();
    let v2 = rhs.v.clone();
    let s = T::one() / (v2.clone() * v2.clone());
    adscalar(v1.clone() / v2.clone(), self.grad(rhs.g, |d1, d2| (d1 * v2.clone() - v1.clone() * d2) * s.clone()))
  }
}

pub struct ADParam<T, P> {
  pub v: T,
  pub i: usize,
  _p: PhantomData<P>
}

pub fn adparam<T, P>(v: T, i: usize) -> ADParam<T, P> {
  ADParam::<T, P>{v, i, _p: PhantomData::<P>::default()}
}

impl<T, P1> ADParam<T, P1> {
  pub fn grad<P2, F>(self, rhs: GenericArray<T, P2>, f: F) -> GenericArray<T, Maxed<P1, P2>>
    where F: Fn(T) -> T,
          P1: Max<P2> + Unsigned,
          P2: Sub<P1> + Unsigned + ArrayLength<T>,
          Maxed<P1, P2>: Unsigned + ArrayLength<T>,
          T: Default,
          GenericArray<T, P1>: Concat<T, Diff<P2, P1>> {
    let p2 = <P2 as Unsigned>::to_usize();
    let p3 = <Maxed<P1, P2> as Unsigned>::to_usize();
    let d = f(rhs.g.clone());
    let g = rhs.concat(GenericArray::<T, Diff<P2, P1>>::default());
    g[self.i] = d;
    g
  }
}
      
*/    

/*impl<T, P1, P2> Add<ADParam<T, P2>> for ADScalar<T, P1>
  where T: Add<T, Output=T>,
        P1: Max<P2> {
  type Output = ADScalar<T, Maxed<P1, P2>>;
  fn add(self, rhs: ADParam<T, P2>) -> Self::Output {
    adscalar(self.v + rhs.v, rhs.*/

/*pub trait ScalarType {
  type Type;
}

type ToScalar<T> = <T as ScalarType>::Type;

impl ScalarType for f32 {
  type Type = Scalar<f32>;
}

impl ScalarType for f64 {
  type Type = Scalar<f64>;
}

impl<T> ScalarType for Scalar<T> {
  type Type = Self;
}

impl<T, P> ScalarType for ADScalar<T, P>
  where P: ArrayLength<T> {
  type Type = Self;
}

pub struct TensorImpl<A, S> {
  a: A,
  s: PhantomData<S>
}

pub type Tensor<T, S> = TensorImpl<GenericArray<Scalar<T>, ReduceProd<S>>, S>;
pub type ADTensor<T, P, S> = TensorImpl<GenericArray<ADScalar<T, P>, ReduceProd<S>>, S>;

fn tensorimpl<A, S>(a: A) -> TensorImpl<A, S> {
  TensorImpl{a, s: PhantomData::<S>::default()}
}

impl<A, S> Default for TensorImpl<A, S>
  where A: Default {
  fn default() -> Self {
    tensorimpl::<A, S>(A::default())
  }
}

impl<A, S> Clone for TensorImpl<A, S>
  where A: Clone {
  fn clone(&self) -> Self {
    tensorimpl::<A, S>(self.a.clone())
  }
}

impl<A, S> Copy for TensorImpl<A, S>
  where A: Copy {}

impl<A, S> Debug for TensorImpl<A, S>
  where A: Debug,
        S: ShapeName {
  fn fmt(&self, f: &mut Formatter) -> Result {
    write!(f, "<{}>{:?}", S::name(), &self.a)
  }
}

*/


