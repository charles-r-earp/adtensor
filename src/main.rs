use adtensor::{shape::Shape, tensor::Tensor};
use rand::distributions::{StandardNormal};

fn main() {
  let a = Shape::new([1, 2]);
  let b = Shape::new([3]);
  let c = Shape::broadcast(&a, &b);
  println!("{:?} and {:?} make {:?}", &a, &b, &c);
}
