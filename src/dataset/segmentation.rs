use burn::data::dataset::vision::{Annotation, ImageDatasetItem, PixelDepth};
use burn::{data::dataloader::batcher::Batcher, prelude::*};

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

#[derive(Clone)]
pub struct SegmentationBatcher<B: Backend> {
    device: B::Device,
    config: SegmentationConfig,
}

impl<B: Backend> SegmentationBatcher<B> {
    pub fn new(device: B::Device, config: SegmentationConfig) -> Self {
        Self { device, config }
    }
}

#[derive(Clone, Debug)]
pub struct SegmentationBatch<B: Backend> {
    pub images: Tensor<B, 4, Float>,
    pub masks: Tensor<B, 4, Int>,
}

impl<B: Backend> Batcher<ImageDatasetItem, SegmentationBatch<B>> for SegmentationBatcher<B> {
    fn batch(&self, items: Vec<ImageDatasetItem>) -> SegmentationBatch<B> {
        let batch_size = items.len();
        let [height, width] = self.config.image_size;

        let mut images = Vec::with_capacity(batch_size);
        let mut masks = Vec::with_capacity(batch_size);

        for item in items {
            let image_tensor: Tensor<B, 3> = match self.config.input_mode {
                InputMode::RGB => {
                    let mut image_data = Vec::with_capacity(3 * height * width);

                    for c in 0..3 {
                        for y in 0..height {
                            for x in 0..width {
                                let idx = (y * width + x) * 3 + c;
                                let val = match item.image.get(idx) {
                                    Some(pixel) => match pixel {
                                        PixelDepth::U8(v) => *v as f32 / 255.0,
                                        PixelDepth::U16(v) => *v as f32 / 65535.0,
                                        PixelDepth::F32(v) => *v,
                                    },
                                    None => 0.0,
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
                            let idx = (y * width + x) * 3; // RGB format in the dataset
                            let val = match item.image.get(idx) {
                                Some(pixel) => match pixel {
                                    PixelDepth::U8(v) => *v as f32 / 255.0,
                                    PixelDepth::U16(v) => *v as f32 / 65535.0,
                                    PixelDepth::F32(v) => *v,
                                },
                                None => 0.0, // Handle potential index errors gracefully
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

            let mask_tensor: Tensor<B, 3, Int> = match &item.annotation {
                Annotation::SegmentationMask(mask) => match self.config.mode {
                    SegmentationMode::Binary => {
                        let bool_mask: Vec<bool> = mask.mask.iter().map(|&x| x > 0).collect();

                        Tensor::<B, 3, Int>::from_data(
                            TensorData::new(bool_mask, Shape::new([1, height, width]))
                                .convert::<B::BoolElem>(),
                            &self.device,
                        )
                    }
                    SegmentationMode::Multiclass { .. } => {
                        let int_mask: Vec<i32> = mask.mask.iter().map(|&x| x as i32).collect();

                        Tensor::<B, 3, Int>::from_data(
                            TensorData::new(int_mask, Shape::new([1, height, width]))
                                .convert::<B::IntElem>(),
                            &self.device,
                        )
                    }
                },
                _ => {
                    println!("Warning: Item does not contain segmentation mask annotation");
                    match self.config.mode {
                        SegmentationMode::Binary => {
                            let bool_mask = vec![false; height * width];
                            Tensor::<B, 3, Int>::from_data(
                                TensorData::new(bool_mask, Shape::new([1, height, width]))
                                    .convert::<B::BoolElem>(),
                                &self.device,
                            )
                        }
                        SegmentationMode::Multiclass { .. } => {
                            let zeros_mask = vec![0i32; height * width];
                            Tensor::<B, 3, Int>::from_data(
                                TensorData::new(zeros_mask, Shape::new([1, height, width]))
                                    .convert::<B::IntElem>(),
                                &self.device,
                            )
                        }
                    }
                }
            };

            images.push(image_tensor);
            masks.push(mask_tensor);
        }

        let images: Tensor<B, 4> = Tensor::stack::<4>(images.to_vec(), 0);
        let masks: Tensor<B, 4, Int> = Tensor::stack::<4>(masks.to_vec(), 0);

        SegmentationBatch { images, masks }
    }
}
