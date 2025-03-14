use std::path::{Path, PathBuf};

use burn::data::dataset::transform::{Mapper, MapperDataset};
use burn::data::dataset::vision::PixelDepth;
use burn::data::dataset::{Dataset, InMemDataset};
use burn::{data::dataloader::batcher::Batcher, prelude::*};
use image::ColorType;
use thiserror::Error;

#[derive(Config, Debug)]
pub enum SegmentationMode {
    Binary,
    Multiclass { num_classes: usize },
}

#[derive(Config, Debug)]
pub enum InputMode {
    Grayscale,
    RGB,
}

impl InputMode {
    pub fn channels(&self) -> usize {
        match self {
            InputMode::Grayscale => 1,
            InputMode::RGB => 3,
        }
    }
}

#[derive(Config, Debug)]
pub struct SegmentationConfig {
    pub mode: SegmentationMode,
    pub input_mode: InputMode,
    pub image_size: [usize; 2],
    pub class_names: Option<Vec<String>>,
}

impl Default for SegmentationConfig {
    fn default() -> Self {
        Self {
            mode: SegmentationMode::Binary,
            input_mode: InputMode::RGB,
            image_size: [512, 512],
            class_names: None,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct SegmentationImageItem {
    pub image: Vec<PixelDepth>,
    pub mask: Vec<usize>,
    pub fov_mask: Option<Vec<bool>>,
}

#[derive(Debug, Clone)]
pub struct SegmentationImageItemRaw {
    pub image_path: PathBuf,
    pub mask_path: PathBuf,
    pub fov_mask_path: Option<PathBuf>,
}

#[derive(Clone, Debug)]
pub struct SegmentationBatch<B: Backend> {
    pub images: Tensor<B, 4, Float>,
    pub masks: Tensor<B, 4, Int>,
    pub fov_masks: Option<Tensor<B, 4, Int>>,
}

#[derive(Clone)]
pub struct SegmentationBatcher<B: Backend> {
    device: B::Device,
    config: crate::dataset::SegmentationConfig,
}

impl<B: Backend> SegmentationBatcher<B> {
    pub fn new(device: B::Device, config: crate::dataset::SegmentationConfig) -> Self {
        Self { device, config }
    }
}

impl<B: Backend> Batcher<SegmentationImageItem, SegmentationBatch<B>> for SegmentationBatcher<B> {
    fn batch(&self, items: Vec<SegmentationImageItem>) -> SegmentationBatch<B> {
        let batch_size = items.len();
        let [height, width] = self.config.image_size;
        let _input_channels = self.config.input_mode.channels();

        let mut images = Vec::with_capacity(batch_size);
        let mut masks = Vec::with_capacity(batch_size);
        let mut has_fov_masks = false;
        let mut fov_masks = Vec::with_capacity(batch_size);

        for item in items {
            let image_tensor = match self.config.input_mode {
                InputMode::RGB => {
                    let mut image_data = Vec::with_capacity(3 * height * width);
                    for c in 0..3 {
                        for y in 0..height {
                            for x in 0..width {
                                let idx = (y * width + x) * 3 + c;
                                let val = if idx < item.image.len() {
                                    match &item.image[idx] {
                                        PixelDepth::U8(v) => *v as f32 / 255.0,
                                        PixelDepth::U16(v) => *v as f32 / 65535.0,
                                        PixelDepth::F32(v) => *v,
                                    }
                                } else {
                                    0.0
                                };
                                image_data.push(val);
                            }
                        }
                    }
                    Tensor::<B, 3>::from_data(
                        TensorData::new(image_data, Shape::new([3, height, width]))
                            .convert::<B::FloatElem>(),
                        &self.device,
                    )
                }
                InputMode::Grayscale => {
                    let mut image_data = Vec::with_capacity(height * width);
                    for y in 0..height {
                        for x in 0..width {
                            let idx = y * width + x;
                            let val = if idx < item.image.len() {
                                match &item.image[idx] {
                                    PixelDepth::U8(v) => *v as f32 / 255.0,
                                    PixelDepth::U16(v) => *v as f32 / 65535.0,
                                    PixelDepth::F32(v) => *v,
                                }
                            } else {
                                0.0 // Padding if needed
                            };
                            image_data.push(val);
                        }
                    }
                    Tensor::<B, 3>::from_data(
                        TensorData::new(image_data, Shape::new([1, height, width]))
                            .convert::<B::FloatElem>(),
                        &self.device,
                    )
                }
            };

            let mask_data: Vec<i32> = item.mask.iter().map(|&x| x as i32).collect();
            let mask_tensor = Tensor::<B, 3, Int>::from_data(
                TensorData::new(mask_data, Shape::new([1, height, width])).convert::<B::IntElem>(),
                &self.device,
            );

            if let Some(fov_mask) = &item.fov_mask {
                has_fov_masks = true;
                let fov_data: Vec<i32> = fov_mask.iter().map(|&x| if x { 1 } else { 0 }).collect();
                let fov_tensor = Tensor::<B, 3, Int>::from_data(
                    TensorData::new(fov_data, Shape::new([1, height, width]))
                        .convert::<B::IntElem>(),
                    &self.device,
                );
                fov_masks.push(fov_tensor);
            }

            images.push(image_tensor);
            masks.push(mask_tensor);
        }

        let images = Tensor::stack::<4>(images, 0);
        let masks = Tensor::stack::<4>(masks, 0);
        let fov_masks = if has_fov_masks {
            Some(Tensor::stack::<4>(fov_masks, 0))
        } else {
            None
        };

        SegmentationBatch {
            images,
            masks,
            fov_masks,
        }
    }
}
