pub trait Forward<'a, X> {
  type Y;
  fn forward(&'a mut self, x: X) -> Self::Y; 
}

pub trait Backward<'a, DY> {
  type DX;
  fn backward(&'a mut self, dy: DY) -> Self::DX;
}
