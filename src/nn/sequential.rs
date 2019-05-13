use crate::autograd::Forward;
use std::borrow::Borrow;

#[derive(Debug)]
pub struct Seq<A, B> {
  pub a: A,
  pub b: B
}

#[macro_export]
macro_rules! seq {
  ($a:expr, $b:expr) => { Seq{a: $a, b: $b} };
  ($a:expr, $b:expr $(, $c:expr)*) => { seq![Seq{$a, $b}, $c, *] }
} 

impl<'a, A, B, X, Y1, Y2> Forward<'a, X> for Seq<A, B>
  where A: Forward<'a, X, Y=Y1>,
        B: Forward<'a, Y1, Y=Y2> {
  type Y = Y2;
  fn forward(&'a mut self, x: X) -> Self::Y {
    self.b.forward(self.a.forward(x))
  }
} 
