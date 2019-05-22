use std::ops::{Deref, Add, Sub, Mul, Div};
use ocl;

pub type Ix1 = usize;
pub type Ix2 = (usize, usize);

pub trait Shape: Copy + Eq + std::fmt::Debug {
  fn rmaj_strides(&self) -> Self;
  fn product(&self) -> usize;
 
}

impl Shape for Ix1 {
  fn rmaj_strides(&self) -> Self {
    *self
  }
  fn product(&self) -> usize {
    *self
  }
}

impl Shape for Ix2 {
  fn rmaj_strides(&self) -> Self { 
    (self.1, 1)
  }
  fn product(&self) -> usize {
    self.0 * self.1
  }
}

#[derive(Default, Debug)]
pub struct Tensor<S, D> {
  pub shape: S,
  pub strides: S,
  pub data: D
}

pub type TensorOwned<S, T> = Tensor<S, Vec<T>>;
pub type TensorView<'a, S, T> = Tensor<S, &'a [T]>; 

impl<S, T> TensorOwned<S, T>
  where S: Shape {
  pub fn shape_uninit(shape: S) -> Self {
    let strides = shape.rmaj_strides();
    let n = shape.product();
    let mut data = Vec::with_capacity(n);
    unsafe { data.set_len(n) };
    Self{shape, strides, data}
  }
  pub fn shape_elem(shape: S, elem: T) -> Self
    where T: Clone {
    let strides = shape.rmaj_strides();
    let n = shape.product();
    let mut data = Vec::with_capacity(n);
    data.resize(n, elem); 
    Self{shape, strides, data}
  }
  pub fn view<'a>(&'a self) -> TensorView<'a, S, T> {
    TensorView{shape: self.shape,
               strides: self.strides,
               data: self.data.deref()}
  } 
}   

impl<S, D, T> Deref for Tensor<S, D>
  where D: Deref<Target=[T]> {
  type Target = [T];
  fn deref(&self) -> &[T] {
    &*self.data
  }
}  

pub trait CLType {
  fn name() -> &'static str;
} 

impl CLType for f32 {
  fn name() -> &'static str { &"float" }
}

impl CLType for f64 {
  fn name() -> &'static str { &"double" }
}

impl CLType for i32 {
  fn name() -> &'static str { &"int" }
}

macro_rules! impl_tensor_op {
  ($op_trait:ident, $func:ident, $op_str:expr) => {
    impl<S, D1, D2, T> $op_trait<Tensor<S, D2>> for Tensor<S, D1>
      where S: Shape,
            D1: Deref<Target=[T]>,
            D2: Deref<Target=[T]>,
            T: Clone + $op_trait<T, Output=T> + ocl::OclPrm + CLType {
      type Output = TensorOwned<S, T>;
      #[inline]
      fn $func(self, rhs: Tensor<S, D2>) -> Self::Output {
        debug_assert_eq!(self.shape, rhs.shape); 
        let mut out = TensorOwned::shape_uninit(self.shape);
        let src = format!(
          "__kernel void add(__global {} const* const a, __global {} const* const b, __global {}* const c) {{
            int idx = get_global_id(0);
            c[idx] = a[idx] {} b[idx];
          }}",
          T::name(),
          T::name(),
          T::name(),
          $op_str
        );  
        let pro_que = ocl::ProQue::builder()
          .src(src)
          .dims(self.shape.product())
          .build()
          .unwrap();
        let a = unsafe {
          pro_que.buffer_builder()
                 .flags(ocl::MemFlags::new().read_only())
                 .use_host_slice(self.deref())
                 .build()
                 .unwrap()
        }; 
        let b = unsafe {
          pro_que.buffer_builder()
                 .flags(ocl::MemFlags::new().read_only())
                 .use_host_slice(rhs.deref())
                 .build()
                 .unwrap()
        }; 
        let c = unsafe {
          pro_que.buffer_builder()
                 .flags(ocl::MemFlags::new().write_only())
                 .use_host_slice(rhs.deref())
                 .build()
                 .unwrap()
        }; 
        let kernel = pro_que.kernel_builder("add")
          .arg(&a)
          .arg(&b)
          .arg(&c)
          .build()
          .unwrap();
        unsafe { 
          kernel.enq()
                .unwrap(); 
        }
        c.read(&mut out.data)
         .enq()
         .unwrap();
        out
      }
    }
  }
}

impl_tensor_op!(Add, add, "+");
impl_tensor_op!(Sub, sub, "-");
impl_tensor_op!(Mul, mul, "*");
impl_tensor_op!(Div, div, "/");

    
