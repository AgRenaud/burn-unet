pub mod model;

#[cfg(feature = "dataset")]
pub mod dataset;

#[cfg(feature = "training")]
pub mod training;

pub use model::UNet;
pub use model::UNetConfig;

#[cfg(feature = "dataset")]
pub use dataset::{InputMode, SegmentationConfig, SegmentationMode};

#[cfg(feature = "training")]
pub use training::{IoULoss, IoULossConfig, IoUMetric, SegmentationOutput};

pub const VERSION: &str = env!("CARGO_PKG_VERSION");
