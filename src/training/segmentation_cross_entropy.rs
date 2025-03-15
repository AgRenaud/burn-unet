//! Segmentation Cross Entropy Loss
//!
//! This implementation is inspired by and extends the original CrossEntropyLoss from
//! the Burn crate. It adds support for image segmentation tasks by accepting an
//! additional mask tensor and handling spatial dimensions appropriately.
//!
//! The original implementation can be found in the Burn crate's cross_entropy.rs file.
//! https://github.com/tracel-ai/burn/blob/v0.16.0/crates/burn-core/src/nn/loss/cross_entropy.rs

use burn::{
    module::{Content, DisplaySettings, ModuleDisplay},
    prelude::*,
    tensor::activation::log_softmax,
};

/// Configuration to create a [Segmentation Cross-entropy loss](SegmentationCrossEntropyLoss) using the [init function](SegmentationCrossEntropyLossConfig::init).
#[derive(Config, Debug)]
pub struct SegmentationCrossEntropyLossConfig {
    /// Create cross-entropy with label smoothing.
    ///
    /// Hard labels {0, 1} will be changed to y_smoothed = y(1 - a) + a / nr_classes.
    /// Alpha = 0 would be the same as default.
    pub smoothing: Option<f32>,

    /// Create weighted cross-entropy.
    ///
    /// The loss of a specific sample will be multiplied by the weight corresponding to the class label.
    ///
    /// # Pre-conditions
    ///   - The order of the weight vector should correspond to the label integer assignment.
    ///   - Targets assigned negative Int's will not be allowed.
    pub weights: Option<Vec<f32>>,

    /// Create cross-entropy with probabilities as input instead of logits.
    ///
    #[config(default = true)]
    pub logits: bool,

    /// Ignore pad labels in the loss calculation.
    /// Usefull to ignore background or boundary classes
    pub ignore_index: Option<usize>,
}

impl SegmentationCrossEntropyLossConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> SegmentationCrossEntropyLoss<B> {
        self.assertions();
        SegmentationCrossEntropyLoss {
            weights: self
                .weights
                .as_ref()
                .map(|e| Tensor::<B, 1>::from_floats(e.as_slice(), device)),
            smoothing: self.smoothing,
            logits: self.logits,
            ignore_index: self.ignore_index,
        }
    }

    fn assertions(&self) {
        if let Some(alpha) = self.smoothing {
            assert!(
                (0.0..=1.).contains(&alpha),
                "Alpha of Cross-entropy loss with smoothed labels should be in interval [0, 1]. Got {}",
                alpha
            );
        };
        if let Some(weights) = self.weights.as_ref() {
            assert!(
                weights.iter().all(|e| e > &0.),
                "Weights of cross-entropy have to be positive."
            );
        }
    }
}

/// Calculate the segmentation cross entropy loss from the input logits, targets, and mask.
///
/// Should be created using [SegmentationCrossEntropyLossConfig]
#[derive(Module, Debug)]
#[module(custom_display)]
pub struct SegmentationCrossEntropyLoss<B: Backend> {
    /// Weights for cross-entropy.
    pub weights: Option<Tensor<B, 1>>,
    /// Label smoothing factor.
    pub smoothing: Option<f32>,
    /// Use logits as input.
    pub logits: bool,
    /// Ignore specific index during loss calculation.
    pub ignore_index: Option<usize>,
}

impl<B: Backend> ModuleDisplay for SegmentationCrossEntropyLoss<B> {
    fn custom_settings(&self) -> Option<DisplaySettings> {
        DisplaySettings::new()
            .with_new_line_after_attribute(false)
            .optional()
    }

    fn custom_content(&self, content: Content) -> Option<Content> {
        content
            .add("weights", &self.weights)
            .add("smoothing", &self.smoothing)
            .add("logits", &self.logits)
            .add("ignore_index", &self.ignore_index)
            .optional()
    }
}

impl<B: Backend> SegmentationCrossEntropyLoss<B> {
    /// Compute the criterion on the input tensor.
    ///
    /// # Shapes
    ///
    /// - predictions: `[batch_size, num_classes, height, width]`
    /// - targets: `[batch_size, 1, height, width]` (integer class indices)
    /// - mask: `[batch_size, 1, height, width]` (boolean mask where True indicates valid pixels)
    pub fn forward(
        &self,
        predictions: Tensor<B, 4>,
        targets: Tensor<B, 4, Int>,
        mask: Tensor<B, 4, Bool>,
    ) -> Tensor<B, 1> {
        Self::assertions(&predictions, &targets, &mask);

        let [batch_size, num_classes, height, width] = predictions.dims();
        let device = &predictions.device();

        let predictions_2d = predictions
            .reshape([batch_size, num_classes, height * width])
            .permute([0, 2, 1])
            .reshape([batch_size * height * width, num_classes]);

        let targets_2d = targets
            .reshape([batch_size, 1, height * width])
            .reshape([batch_size * height * width]);

        let mask_2d = mask
            .reshape([batch_size, 1, height * width])
            .reshape([batch_size * height * width]);

        let ignore_mask = if let Some(ignore_idx) = self.ignore_index {
            targets_2d.clone().not_equal_elem(ignore_idx as i32)
        } else {
            targets_2d.clone().ones_like().bool()
        };

        let combined_mask = mask_2d.int().add(ignore_mask.int()).bool();

        match self.smoothing {
            Some(alpha) => self.forward_smoothed(
                predictions_2d,
                targets_2d,
                combined_mask,
                alpha,
                batch_size,
                num_classes,
            ),
            _ => self.forward_default(predictions_2d, targets_2d, combined_mask),
        }
    }

    fn forward_smoothed(
        &self,
        predictions: Tensor<B, 2>,
        targets: Tensor<B, 1, Int>,
        mask: Tensor<B, 1, Bool>,
        alpha: f32,
        batch_size: usize,
        num_classes: usize,
    ) -> Tensor<B, 1> {
        let tensor = if self.logits {
            log_softmax(predictions, 1)
        } else {
            predictions.log()
        };

        let total_elements = tensor.dims()[0];
        let tensor = tensor
            * Self::compute_smoothed_targets([total_elements, num_classes], targets.clone(), alpha);

        match &self.weights {
            Some(weights) => {
                let tensor = tensor
                    * weights
                        .clone()
                        .reshape([1, num_classes])
                        .repeat_dim(0, total_elements);

                let tensor = Self::apply_mask_2d(tensor, mask.clone());

                let weighted_mask = mask.unsqueeze().repeat_dim(1, num_classes).int().float()
                    * weights
                        .clone()
                        .gather(0, targets.clone())
                        .unsqueeze()
                        .repeat_dim(1, num_classes);

                let valid_sum = weighted_mask.sum();

                let valid_count = mask.int().float().sum();

                let valid_sum_mask = valid_sum.greater_elem(0.0);
                let valid_count_mask = valid_sum_mask.bool_not();

                let sum_term = valid_sum * valid_sum_mask.int().float();
                let count_term = valid_count * valid_count_mask.int().float();
                let denominator = sum_term + count_term;

                let denominator = denominator.clamp_max(1.0);

                tensor.sum().neg() / denominator
            }
            None => {
                let tensor = Self::apply_mask_2d(tensor, mask.clone());
                let valid_count = mask.int().float().sum().clamp_max(1.0);
                tensor.sum().neg() / valid_count
            }
        }
    }

    fn forward_default(
        &self,
        predictions: Tensor<B, 2>,
        targets: Tensor<B, 1, Int>,
        mask: Tensor<B, 1, Bool>,
    ) -> Tensor<B, 1> {
        let [total_elements, num_classes] = predictions.dims();

        let tensor = if self.logits {
            log_softmax(predictions, 1)
        } else {
            predictions.log()
        };

        let tensor = tensor
            .gather(1, targets.clone().reshape([total_elements, 1]))
            .reshape([total_elements]);

        match &self.weights {
            Some(weights) => {
                let weights = weights.clone().gather(0, targets);
                let tensor = tensor * weights.clone();
                let tensor = Self::apply_mask_1d(tensor, mask.clone());

                // Compute weighted sum
                let weighted_mask = mask.int().float() * weights;

                let valid_sum = weighted_mask.sum();

                let valid_count = mask.int().float().sum();

                let valid_sum_mask = valid_sum.greater_elem(0.0);
                let valid_count_mask = valid_sum_mask.bool_not();

                let sum_term = valid_sum * valid_sum_mask.int().float();
                let count_term = valid_count * valid_count_mask.int().float();
                let denominator = sum_term + count_term;

                let denominator = denominator.clamp_max(1.0);
                tensor.sum().neg() / denominator
            }
            None => {
                let tensor = Self::apply_mask_1d(tensor, mask.clone());
                let valid_count = mask.int().float().sum().clamp_max(1.0);
                tensor.sum().neg() / valid_count
            }
        }
    }

    fn compute_smoothed_targets(
        shape: [usize; 2],
        targets: Tensor<B, 1, Int>,
        alpha: f32,
    ) -> Tensor<B, 2> {
        let [batch_size, nr_classes] = shape;
        let device = &targets.device();
        let targets_matrix = Tensor::<B, 2>::zeros(shape, device).scatter(
            1,
            targets.reshape([batch_size, 1]),
            Tensor::ones([batch_size, 1], device),
        );
        targets_matrix * (1. - alpha) + alpha / nr_classes as f32
    }

    fn apply_mask_1d(tensor: Tensor<B, 1>, mask: Tensor<B, 1, Bool>) -> Tensor<B, 1> {
        tensor.mask_fill(mask.bool_not(), 0.0)
    }

    fn apply_mask_2d(tensor: Tensor<B, 2>, mask: Tensor<B, 1, Bool>) -> Tensor<B, 2> {
        let [batch_size, nr_classes] = tensor.dims();
        tensor.mask_fill(
            mask.bool_not()
                .reshape([batch_size, 1])
                .repeat_dim(1, nr_classes),
            0.0,
        )
    }

    fn assertions(
        predictions: &Tensor<B, 4>,
        targets: &Tensor<B, 4, Int>,
        mask: &Tensor<B, 4, Bool>,
    ) {
        let [pred_batch, _pred_classes, pred_height, pred_width] = predictions.dims();
        let [target_batch, target_channels, target_height, target_width] = targets.dims();
        let [mask_batch, mask_channels, mask_height, mask_width] = mask.dims();

        assert_eq!(
            pred_batch, target_batch,
            "Batch size mismatch: predictions ({}) vs targets ({})",
            pred_batch, target_batch
        );

        assert_eq!(
            pred_batch, mask_batch,
            "Batch size mismatch: predictions ({}) vs mask ({})",
            pred_batch, mask_batch
        );

        assert_eq!(
            target_channels, 1,
            "Target should have exactly 1 channel, got {}",
            target_channels
        );

        assert_eq!(
            mask_channels, 1,
            "Mask should have exactly 1 channel, got {}",
            mask_channels
        );

        assert_eq!(
            pred_height, target_height,
            "Height mismatch: predictions ({}) vs targets ({})",
            pred_height, target_height
        );

        assert_eq!(
            pred_width, target_width,
            "Width mismatch: predictions ({}) vs targets ({})",
            pred_width, target_width
        );

        assert_eq!(
            pred_height, mask_height,
            "Height mismatch: predictions ({}) vs mask ({})",
            pred_height, mask_height
        );

        assert_eq!(
            pred_width, mask_width,
            "Width mismatch: predictions ({}) vs mask ({})",
            pred_width, mask_width
        );
    }
}
