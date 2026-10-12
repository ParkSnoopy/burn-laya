//! Request encoding, local model execution, and typed answer decoding.

mod agent;
pub mod input;
pub mod model;
pub mod output;

pub use agent::{Agent, Prediction, Usage};
