
pub struct OpenCL {
  context: ocl::Context,
  program: ocl::Program
}

impl OpenCL {
  pub fn src(src: String) -> Self {
    let context = ocl::Context::builder()
      .build()
      .unwrap();
    let program = ocl::Program::builder()
      .src(src)
      .build(&context)
      .unwrap();
    Self{context, program}
  }
  pub fn context(&self) -> &ocl::Context { &self.context }
  pub fn queue(&self) -> ocl::Queue {
    ocl::Queue::new(&self.context, self.context.devices()[0], None).unwrap()
  }
}

#[derive(Debug, Default)]
pub struct Graph {
  fw: Vec<ocl::Kernel>,
  bw: Vec<ocl::Kernel>
}

impl Graph {
  pub fn push_fw(&mut self, kernel: ocl::Kernel) {
    self.fw.push(kernel);
  }
  pub fn push_bw(&mut self, kernel: ocl::Kernel) {
    self.bw.push(kernel);
  }
  pub fn exec<'b>(&mut self, queue: ocl::Queue) {
    self.fw.iter()
      .chain(self.bw.iter().rev())
      .for_each(|k| { 
      unsafe {
        k.cmd()
          .queue(&queue)
          .enq()
          .unwrap();
      }
    });
    self.fw.clear();
    self.bw.clear();
  }
}

#[derive(Debug)]
pub struct Tensor<T: ocl::OclPrm, D: ndarray::Dimension> {
  buffer: ocl::Buffer<T>,
  shape: ndarray::Shape<D>
}

impl<T: ocl::OclPrm, D: ndarray::Dimension> Tensor<T, D> {
  fn new<S: ndarray::ShapeBuilder<Dim=D>>(shape: S) -> Self {
    let shape = shape.into_shape();
    let buffer = ocl::Buffer::builder()
      .len(shape.size())
      .build()
      .unwrap();
    Self{buffer, shape}
  }
  pub fn as_array(&self, queue: ocl::Queue) -> ndarray::Array<T, D> {
    let mut array = unsafe { ndarray::Array::uninitialized(self.shape.clone()) };
    self.buffer.read(array.as_slice_mut().unwrap())
      .queue(&queue)
      .enq()
      .unwrap();
    array
  }
} 

pub trait IntoTensor {
  type Tensor;
  fn into_tensor<'b>(self, backend: &'b OpenCL) -> Self::Tensor;
}

impl<T: ocl::OclPrm, S: ndarray::Data<Elem=T>, D: ndarray::Dimension> IntoTensor for ndarray::ArrayBase<S, D> {
  type Tensor = Tensor<T, D>;
  fn into_tensor<'b>(self, backend: &'b OpenCL) -> Self::Tensor {
    use ndarray::ShapeBuilder;
    let shape = self.dim().into_shape();
    let buffer = ocl::Buffer::builder()
      .context(backend.context())
      .len(shape.size())
      .build()
      .unwrap();
    buffer.write(self.as_slice().unwrap())
      .queue(&backend.queue())
      .enq()
      .unwrap();
    Tensor{buffer, shape}
  }
}

#[cfg(test)]
mod tests {
  use super::*;
  #[test]
  fn test_tensor() {
    let backend = OpenCL::src("".to_string());
    let mut graph = Graph::default();
    let x = ndarray::arr1(&[1, 2]);
    let y = x.view().into_tensor(&backend)
      .as_array(backend.queue());
    assert_eq!(x, y);
  }
}


  
