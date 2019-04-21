use adtensor::tensor::{Shape, Tensor};

fn main() {
  let x = Tensor::new(Shape::from([2]), vec![1, 2]);
  let y = Tensor::new(Shape::from([4]), vec![1, 2, 3, 4]);
  let mut z = [0, 0, 0, 0];
  z.iter_mut()
   .zip(x.iter().cycle().zip(y.iter()))
   .for_each(|(c, (&a, &b))| *c = a + b);
  println!("{:?}", &z);
  let w = &x + y * &x;
  println!("{:?}", &w);
}
