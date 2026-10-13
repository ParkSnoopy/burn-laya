//! Request encoding, local model execution, and typed answer decoding.

mod agent;
mod attention;
pub mod input;
pub mod model;
pub mod output;
#[cfg(feature = "prepare")]
pub mod prepare;

pub use agent::{Agent, Prediction, Usage};
