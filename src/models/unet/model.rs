use burn::{
    nn::conv::{Conv2d, Conv2dConfig},
    prelude::*,
    tensor::{activation::softmax, backend::AutodiffBackend},
};

#[cfg(feature = "training")]
use crate::{
    dataset::SegmentationBatch,
    training::{
        SegmentationOutput,
        loss::{IoULossConfig, SegmentationCrossEntropyLossConfig},
    },
};
#[cfg(feature = "training")]
use burn::train::{TrainOutput, TrainStep, ValidStep};

use nn::{PaddingConfig2d, Sigmoid};

use super::blocks::{
    ConvBlock, ConvBlockConfig, DecoderBlock, DecoderBlockConfig, EncoderBlock, EncoderBlockConfig,
};

#[derive(Module, Debug)]
pub struct UNet<B: Backend> {
    encoder_block_1: EncoderBlock<B>,
    encoder_block_2: EncoderBlock<B>,
    encoder_block_3: EncoderBlock<B>,
    encoder_block_4: EncoderBlock<B>,
    bottleneck: ConvBlock<B>,
    decoder_block_1: DecoderBlock<B>,
    decoder_block_2: DecoderBlock<B>,
    decoder_block_3: DecoderBlock<B>,
    decoder_block_4: DecoderBlock<B>,
    conv: Conv2d<B>,
    conv_1x1: Conv2d<B>,
    use_softmax: bool,

    num_classes: usize,
}

#[derive(Config, Debug)]
pub struct UNetConfig {
    input_size: [usize; 2],
    #[config(default = "64")]
    base_channels: usize,
    #[config(default = "1")]
    num_classes: usize,
    #[config(default = "true")]
    use_softmax: bool,
}

impl UNetConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> UNet<B> {
        UNet {
            encoder_block_1: EncoderBlockConfig::new(ConvBlockConfig::new(1, self.base_channels))
                .init(device),
            encoder_block_2: EncoderBlockConfig::new(ConvBlockConfig::new(
                self.base_channels,
                self.base_channels * 2,
            ))
            .init(device),
            encoder_block_3: EncoderBlockConfig::new(ConvBlockConfig::new(
                self.base_channels * 2,
                self.base_channels * 4,
            ))
            .init(device),
            encoder_block_4: EncoderBlockConfig::new(ConvBlockConfig::new(
                self.base_channels * 4,
                self.base_channels * 8,
            ))
            .init(device),
            bottleneck: ConvBlockConfig::new(self.base_channels * 8, self.base_channels * 16)
                .init(device),
            decoder_block_1: DecoderBlockConfig::new(
                self.base_channels * 16,
                self.base_channels * 8,
                ConvBlockConfig::new(self.base_channels * 16, self.base_channels * 8),
            )
            .init(device),
            decoder_block_2: DecoderBlockConfig::new(
                self.base_channels * 8,
                self.base_channels * 4,
                ConvBlockConfig::new(self.base_channels * 8, self.base_channels * 4),
            )
            .init(device),
            decoder_block_3: DecoderBlockConfig::new(
                self.base_channels * 4,
                self.base_channels * 2,
                ConvBlockConfig::new(self.base_channels * 4, self.base_channels * 2),
            )
            .init(device),
            decoder_block_4: DecoderBlockConfig::new(
                self.base_channels * 2,
                self.base_channels,
                ConvBlockConfig::new(self.base_channels * 2, self.base_channels),
            )
            .init(device),
            conv: Conv2dConfig::new([self.base_channels, self.base_channels], [3, 3])
                .with_padding(PaddingConfig2d::Same)
                .init(device),
            conv_1x1: Conv2dConfig::new([self.base_channels, self.num_classes], [1, 1])
                .init(device),
            use_softmax: self.use_softmax,
            num_classes: self.num_classes,
        }
    }
}

impl<B: Backend> UNet<B> {
    pub fn forward(&self, images: Tensor<B, 4>) -> Tensor<B, 4> {
        let x = images;

        let (x, skip_features_1) = self.encoder_block_1.forward(x);
        let (x, skip_features_2) = self.encoder_block_2.forward(x);
        let (x, skip_features_3) = self.encoder_block_3.forward(x);
        let (x, skip_features_4) = self.encoder_block_4.forward(x);

        let x = self.bottleneck.forward(x);

        let x = self.decoder_block_1.forward(x, skip_features_4);
        let x = self.decoder_block_2.forward(x, skip_features_3);
        let x = self.decoder_block_3.forward(x, skip_features_2);
        let x = self.decoder_block_4.forward(x, skip_features_1);

        let x = self.conv.forward(x);

        let x = self.conv_1x1.forward(x);

        if self.use_softmax {
            softmax(x, 1)
        } else {
            Sigmoid::new().forward(x)
        }
    }

    #[cfg(feature = "training")]
    pub fn forward_segmentation(&self, item: SegmentationBatch<B>) -> SegmentationOutput<B> {
        let targets = item.masks;
        let output = self.forward(item.images);
        let masks = item.fov_masks.unwrap_or(output.ones_like().bool());

        // let loss = IoULossConfig::new()
        //     .with_num_classes(self.num_classes)
        //     .init(&output.device())
        //     .forward(output.clone(), targets.clone());

        let loss = SegmentationCrossEntropyLossConfig::new()
            .init(&output.device())
            .forward(output.clone(), targets.clone(), masks.clone());

        SegmentationOutput {
            loss,
            output,
            targets,
        }
    }
}

#[cfg(feature = "training")]
impl<B: AutodiffBackend> TrainStep<SegmentationBatch<B>, SegmentationOutput<B>> for UNet<B> {
    fn step(&self, batch: SegmentationBatch<B>) -> TrainOutput<SegmentationOutput<B>> {
        let item = self.forward_segmentation(batch);
        TrainOutput::new(self, item.loss.backward(), item)
    }
}

#[cfg(feature = "training")]
impl<B: Backend> ValidStep<SegmentationBatch<B>, SegmentationOutput<B>> for UNet<B> {
    fn step(&self, batch: SegmentationBatch<B>) -> SegmentationOutput<B> {
        self.forward_segmentation(batch)
    }
}
