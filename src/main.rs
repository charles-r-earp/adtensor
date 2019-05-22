use adtensor::{Tensor, TensorOwned};
use timeit::{timeit, timeit_loops};

fn main() {
  let n = 10000000;
  let x = Tensor::shape_elem(n, 1f32);
  let y = Tensor::shape_elem(n, 2f32);
  let mut z = Tensor::shape_elem(n, 0f32);
  println!("Default sequential...");
  timeit!({
    z = TensorOwned{shape: x.shape,
                    strides: x.strides,
                    data: x.view()
                           .iter()
                           .zip(y.view().iter())
                           .map(|(x, y)| x * y)
                           .collect()}
  });
  println!("{:?}", z[0]);
  z = Tensor::shape_elem(n, 0f32);
  println!("Opencl parallel...");
  timeit!({z = x.view() * y.view()});
  println!("{:?}", z[0]);
}
