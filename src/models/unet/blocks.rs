use burn::{
    nn::{
        Dropout, DropoutConfig, Relu,
        conv::{Conv2d, Conv2dConfig, ConvTranspose2d, ConvTranspose2dConfig},
        pool::MaxPool2d,
    },
    prelude::*,
};
use nn::{PaddingConfig2d, pool::MaxPool2dConfig};

#[derive(Module, Debug)]
pub struct ConvBlock<B: Backend> {
    conv1: Conv2d<B>,
    conv2: Conv2d<B>,
    activation: Relu,
    dropout: Dropout,
}

impl<B: Backend> ConvBlock<B> {
    pub fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        let x = self.conv1.forward(x);
        let x = self.activation.forward(x);
        let x = self.dropout.forward(x);
        let x = self.conv2.forward(x);

        self.activation.forward(x)
    }
}

#[derive(Config, Debug)]
pub struct ConvBlockConfig {
    input_channels: usize,
    num_filters: usize,
    #[config(default = "0.2")]
    dropout: f64,
}

impl ConvBlockConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> ConvBlock<B> {
        ConvBlock {
            conv1: Conv2dConfig::new([self.input_channels, self.num_filters], [3, 3])
                .with_padding(PaddingConfig2d::Same)
                .init(device),
            conv2: Conv2dConfig::new([self.num_filters, self.num_filters], [3, 3])
                .with_padding(PaddingConfig2d::Same)
                .init(device),
            activation: Relu::new(),
            dropout: DropoutConfig::new(self.dropout).init(),
        }
    }
}

#[derive(Module, Debug)]
pub struct EncoderBlock<B: Backend> {
    conv_block: ConvBlock<B>,
    max_pool: MaxPool2d,
}

impl<B: Backend> EncoderBlock<B> {
    pub fn forward(&self, x: Tensor<B, 4>) -> (Tensor<B, 4>, Tensor<B, 4>) {
        let skip_features = self.conv_block.forward(x);
        let x = self.max_pool.forward(skip_features.clone());

        (x, skip_features)
    }
}

#[derive(Config, Debug)]
pub struct EncoderBlockConfig {
    conv_block: ConvBlockConfig,
}

impl EncoderBlockConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> EncoderBlock<B> {
        EncoderBlock {
            conv_block: self.conv_block.init(device),
            max_pool: MaxPool2dConfig::new([2, 2]).with_strides([2, 2]).init(),
        }
    }
}

#[derive(Module, Debug)]
pub struct DecoderBlock<B: Backend> {
    conv_transpose: ConvTranspose2d<B>,
    conv_block: ConvBlock<B>,
}

impl<B: Backend> DecoderBlock<B> {
    pub fn forward(&self, x: Tensor<B, 4>, skip_features: Tensor<B, 4>) -> Tensor<B, 4> {
        let x = self.conv_transpose.forward(x);
        let x = Tensor::cat(vec![x, skip_features], 1);

        self.conv_block.forward(x)
    }
}

#[derive(Config, Debug)]
pub struct DecoderBlockConfig {
    input_channels: usize,
    num_filters: usize,
    conv_block: ConvBlockConfig,
}

impl DecoderBlockConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> DecoderBlock<B> {
        DecoderBlock {
            conv_transpose: ConvTranspose2dConfig::new(
                [self.input_channels, self.num_filters],
                [2, 2],
            )
            .with_stride([2, 2])
            .init(device),
            conv_block: self.conv_block.init(device),
        }
    }
}
