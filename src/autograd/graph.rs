pub trait Forward<'a, X> {
  type Y;
  fn forward(&'a mut self, x: X) -> Self::Y; 
}

pub trait Backward<'a, DY, O> {
  type DX;
  fn backward<'o>(&'a mut self, dy: DY, opt: Option<&'o O>) -> Self::DX;
}

pub trait Evaluate<'a, 'x, X> {
  type Y;
  fn eval(&'a self, x: &'x X) -> Self::Y;
}
