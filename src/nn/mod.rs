pub mod initializer;
pub use initializer::{Initializer, ElemInit, HeInit, XavierInit};
pub mod activation;
pub use activation::{Sigmoid};
pub mod linear;
pub use linear::{Weight, Bias};
pub mod sequential;
pub use sequential::Seq;
