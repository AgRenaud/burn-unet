#[cfg(feature = "models")]
pub mod models;

#[cfg(feature = "dataset")]
pub mod dataset;

#[cfg(feature = "training")]
pub mod training;

#[cfg(feature = "dataset")]
pub use dataset::{
    InputMode, SegmentationBatch, SegmentationBatcher, SegmentationConfig, SegmentationImageItem,
    SegmentationImageItemRaw, SegmentationMode,
};

#[cfg(feature = "training")]
pub use training::{IoULoss, IoULossConfig, IoUMetric, SegmentationOutput};

pub const VERSION: &str = env!("CARGO_PKG_VERSION");
