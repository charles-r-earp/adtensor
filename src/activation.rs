trait Activate<F> {
  type Output;
  fn act(self, f: F) -> Self::Output;
}

trait Relu {
  type Output; 
  fn relu(self) -> Self::Output;
}

trait Sigmoid {
  type Output;
  fn sigmoid(self) -> Self::Output;
}
