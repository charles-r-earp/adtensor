use adtensor::adscalar::{ADScalar};


fn main() {
  let x = vec![ADScalar::new(1.); 10];
  let y: Vec<ADScalar<f32>> = x.iter().map(|&x| if x.v > 0. {x} else {ADScalar::from(0.)}).collect();
  println!("x = {:?}, y = {:?}", &x, &y); 
}
