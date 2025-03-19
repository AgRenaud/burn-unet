use std::marker::PhantomData;

use burn::{
    prelude::*,
    tensor::activation::{sigmoid, softmax},
};

#[derive(Config, Debug)]
pub struct IoULossConfig {
    #[config(default = 1e-5)]
    pub smooth: f32,

    #[config(default = true)]
    pub reduction: bool,

    #[config(default = 2)]
    pub num_classes: usize,

    #[config(default = false)]
    pub apply_activation: bool,
}

impl IoULossConfig {
    pub fn init<B: Backend>(&self, _device: &B::Device) -> IoULoss<B> {
        self.assertions();
        IoULoss {
            smooth: self.smooth,
            reduction: self.reduction,
            num_classes: self.num_classes,
            apply_activation: self.apply_activation,
            _b: PhantomData,
        }
    }

    fn assertions(&self) {
        assert!(
            self.smooth >= 0.,
            "Smoothing factor must be non-negative. Got {}",
            self.smooth
        );

        assert!(
            self.num_classes >= 2,
            "Number of classes must be at least 2 (for binary segmentation). Got {}",
            self.num_classes
        );
    }
}

#[derive(Module, Debug)]
pub struct IoULoss<B: Backend> {
    pub smooth: f32,
    pub reduction: bool,
    pub num_classes: usize,
    pub apply_activation: bool,
    _b: PhantomData<B>,
}

impl<B: Backend> IoULoss<B> {
    pub fn forward(&self, inputs: Tensor<B, 4>, targets: Tensor<B, 4, Int>) -> Tensor<B, 1> {
        self.assertions(&inputs, &targets);

        let _batch_size = inputs.dims()[0];
        let input_channels = inputs.dims()[1];

        let _device = &targets.device();

        let probs = if self.apply_activation {
            if self.num_classes == 2 && input_channels == 1 {
                sigmoid(inputs)
            } else {
                softmax(inputs, 1)
            }
        } else {
            inputs
        };

        if self.num_classes == 2 && input_channels == 1 {
            return self.binary_iou_loss(probs, targets);
        }

        self.multiclass_iou_loss(probs, targets)
    }

    fn binary_iou_loss(&self, probs: Tensor<B, 4>, targets: Tensor<B, 4, Int>) -> Tensor<B, 1> {
        let device = &targets.device();
        let targets_float = targets.float();

        // Calculate intersection and union
        let intersection = (probs.clone() * targets_float.clone()).sum();
        let pred_sum = probs.sum();
        let target_sum = targets_float.sum();

        // IoU = (intersection + smooth) / (union + smooth)
        // where union = pred_sum + target_sum - intersection
        let union = pred_sum + target_sum - intersection.clone();

        let iou = (intersection + self.smooth) / (union + self.smooth);

        let ones = Tensor::<B, 1>::ones([1], device);
        ones - iou
    }

    fn multiclass_iou_loss(&self, probs: Tensor<B, 4>, targets: Tensor<B, 4, Int>) -> Tensor<B, 1> {
        let device = &targets.device();
        let _batch_size = probs.dims()[0];
        let num_classes = probs.dims()[1];
        let _height = probs.dims()[2];
        let _width = probs.dims()[3];

        let mut class_ious = Vec::new();

        for class_idx in 0..num_classes {
            let class_probs = probs.clone().narrow(1, class_idx, 1);

            let class_targets = targets.clone().equal_elem(class_idx as i64).float();

            let intersection = (class_probs.clone() * class_targets.clone()).sum();
            let pred_sum = class_probs.sum();
            let target_sum = class_targets.sum();

            let union = pred_sum + target_sum - intersection.clone();

            let class_iou = (intersection + self.smooth) / (union + self.smooth);
            class_ious.push(class_iou);
        }

        let stacked_ious = Tensor::stack::<2>(class_ious.to_vec(), 0);

        let mean_iou = if self.reduction {
            stacked_ious.mean()
        } else {
            stacked_ious.sum()
        };

        // Convert to loss (1 - IoU)
        let ones = Tensor::<B, 1>::ones([1], device);
        ones - mean_iou
    }

    fn assertions(&self, inputs: &Tensor<B, 4>, targets: &Tensor<B, 4, Int>) {
        let input_dims = inputs.dims();
        let target_dims = targets.dims();

        assert!(
            input_dims[0] == target_dims[0],
            "Batch size mismatch: inputs ({}) vs targets ({})",
            input_dims[0],
            target_dims[0]
        );

        assert!(
            input_dims[2] == target_dims[2] && input_dims[3] == target_dims[3],
            "Spatial dimensions mismatch: inputs ({},{}) vs targets ({},{})",
            input_dims[2],
            input_dims[3],
            target_dims[2],
            target_dims[3]
        );

        if self.num_classes == 2 && inputs.dims()[1] == 1 {
            assert!(
                target_dims[1] == 1,
                "For binary segmentation with single channel output, targets should have 1 channel, got {}",
                target_dims[1]
            );
        } else {
            assert!(
                inputs.dims()[1] == self.num_classes,
                "For multi-class segmentation, inputs should have num_classes ({}) channels, got {}",
                self.num_classes,
                inputs.dims()[1]
            );

            assert!(
                target_dims[1] == 1,
                "For multi-class segmentation, targets should have 1 channel with class indices, got {}",
                target_dims[1]
            );
        }
    }
}
