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

pub const VERSION: &str = env!("CARGO_PKG_VERSION");
