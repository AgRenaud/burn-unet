//! Segmentation Cross Entropy Loss
//!
//! This implementation is inspired by and extends the original CrossEntropyLoss from
//! the Burn crate. It adds support for image segmentation tasks by accepting an
//! additional mask tensor and handling spatial dimensions appropriately.
//!
//! The original implementation can be found in the Burn crate's cross_entropy.rs file.
//! https://github.com/tracel-ai/burn/blob/v0.16.0/crates/burn-core/src/nn/loss/cross_entropy.rs

use std::f64::EPSILON;

use burn::{
    module::{Content, DisplaySettings, ModuleDisplay},
    prelude::*,
    tensor::{activation::log_softmax, cast::ToElement},
};

/// Configuration to create a [SegmentationCrossEntropyLoss] instance.
///
/// This struct allows you to customize the behavior of the segmentation cross-entropy loss,
/// including label smoothing, class weighting, input handling, and class index ignoring.
///
/// # Example
///
/// ```rust
/// let config = SegmentationCrossEntropyLossConfig::new()
///     .with_smoothing(0.1)
///     .with_weights(vec![1.0, 2.0, 3.0])
///     .with_ignore_indices(vec![0]);
///
/// let loss_fn = config.init(&device);
/// ```
#[derive(Config, Debug)]
pub struct SegmentationCrossEntropyLossConfig {
    /// Optional label smoothing factor (between 0.0 and 1.0).
    ///
    /// Hard labels {0, 1} will be changed to y_smoothed = y(1 - α) + α / nr_classes.
    /// This helps prevent the model from becoming overconfident and improves generalization.
    ///
    /// Default: None (no smoothing)
    pub smoothing: Option<f32>,

    /// Optional class weights for handling class imbalance.
    ///
    /// The loss of a specific sample will be multiplied by the weight corresponding to the class label.
    /// This is particularly useful in segmentation tasks where certain classes may be rare but important.
    ///
    /// The order of the weight vector must correspond to the label integer assignment.
    /// All weights must be positive values.
    ///
    /// Default: None (all classes weighted equally)
    pub weights: Option<Vec<f32>>,

    /// Whether input predictions are logits (true) or probabilities (false).
    ///
    /// - If true, a softmax function will be applied internally to convert logits to probabilities.
    /// - If false, inputs are assumed to already be probabilities (should sum to 1).
    ///
    /// Default: true (expects logits)
    #[config(default = true)]
    pub logits: bool,

    /// Class indices to ignore in the loss calculation.
    ///
    /// Pixels assigned to these class indices will not contribute to the loss or gradient.
    /// This is useful for ignoring background, boundary classes, or "don't care" regions.
    ///
    /// Default: Empty vector (no classes ignored)
    pub ignore_indices: Option<Vec<usize>>,
}

impl SegmentationCrossEntropyLossConfig {
    /// Initialize a new [SegmentationCrossEntropyLoss] instance from this configuration.
    ///
    /// # Arguments
    ///
    /// * `device` - The device where the tensors will be allocated
    ///
    /// # Returns
    ///
    /// A new [SegmentationCrossEntropyLoss] instance configured according to this configuration.
    pub fn init<B: Backend>(&self, device: &B::Device) -> SegmentationCrossEntropyLoss<B> {
        self.assertions();
        let ignore_indices = self.ignore_indices.clone().unwrap_or(Vec::new());
        SegmentationCrossEntropyLoss {
            weights: self
                .weights
                .as_ref()
                .map(|e| Tensor::<B, 1>::from_floats(e.as_slice(), device)),
            smoothing: self.smoothing,
            logits: self.logits,
            ignore_indices,
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

/// Segmentation Cross Entropy Loss for image segmentation tasks.
///
/// This loss function extends the standard cross-entropy to handle the specific needs
/// of image segmentation, including:
/// - Spatial dimensions
/// - Field-of-view masks to focus on valid regions
/// - Ignored class indices
/// - Label smoothing
/// - Class weighting
///
/// # Input Tensor Shapes
///
/// - predictions: `[batch_size, num_classes, height, width]` - Model predictions
/// - targets: `[batch_size, 1, height, width]` - Ground truth labels with integer class indices
/// - masks: `[batch_size, 1, height, width]` - Boolean mask where true = valid pixel
///
/// Should be created using [SegmentationCrossEntropyLossConfig].
#[derive(Module, Debug)]
#[module(custom_display)]
pub struct SegmentationCrossEntropyLoss<B: Backend> {
    /// Weights for cross-entropy.
    pub weights: Option<Tensor<B, 1>>,
    /// Label smoothing factor.
    pub smoothing: Option<f32>,
    /// Use logits as input.
    pub logits: bool,
    /// Ignore specific indices during loss calculation.
    pub ignore_indices: Vec<usize>,
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
            .add("ignore_index", &self.ignore_indices)
            .optional()
    }
}

impl<B: Backend> SegmentationCrossEntropyLoss<B> {
    /// Compute the segmentation cross-entropy loss.
    ///
    /// # Arguments
    ///
    /// * `predictions` - Model predictions with shape `[batch_size, num_classes, height, width]`
    /// * `targets` - Ground truth class indices with shape `[batch_size, 1, height, width]`
    /// * `masks` - Boolean mask indicating valid pixels, shape `[batch_size, 1, height, width]`
    ///
    /// # Returns
    ///
    /// A scalar tensor representing the average loss over all valid pixels.
    pub fn forward(
        &self,
        predictions: Tensor<B, 4>,
        targets: Tensor<B, 4, Int>,
        masks: Tensor<B, 4, Bool>,
    ) -> Tensor<B, 1> {
        Self::assertions(&predictions, &targets, &masks);

        let device = predictions.device().clone();

        let [batch_size, num_classes, height, width] = predictions.dims();

        let predictions: Tensor<B, 2> = predictions
            .reshape([batch_size, num_classes, height * width])
            .permute([0, 2, 1])
            .reshape([batch_size * height * width, num_classes]);
        let targets: Tensor<B, 1, Int> = targets.reshape([batch_size * height * width]);
        let masks: Tensor<B, 1, Bool> = masks.reshape([batch_size * height * width]);

        tracing::info!("Predictions: {:?}", predictions.dims());
        tracing::info!("Targets: {:?}", targets.dims());
        tracing::info!("Masks: {:?}", masks.dims());

        tracing::info!("Compute combined Masks");
        // Combined masks with ignored classes
        let mut combined_mask = masks.int();
        for ignore_idx in &self.ignore_indices {
            let ignore_mask: Tensor<B, 1, Bool> =
                targets.clone().not_equal_elem(ignore_idx.clone() as i32);
            combined_mask = combined_mask.mask_fill(ignore_mask, 1);
        }
        let masks: Tensor<B, 1, Bool> = combined_mask.bool();
        tracing::info!("Masks: {:?}", masks.dims());

        tracing::info!("Appy logits");
        let mut tensor: Tensor<B, 2> = if self.logits {
            log_softmax(predictions, 1)
        } else {
            predictions.log()
        };

        // Apply smoothing
        tracing::info!("Apply smoothing");
        let smoothed_targets = Self::compute_smoothed_targets(
            num_classes,
            batch_size,
            targets.clone(),
            self.smoothing.unwrap_or(0.0),
        );

        // Apply weights
        tracing::info!("Apply weights");
        if let Some(weights) = &self.weights {
            tensor = tensor.clone()
                * weights
                    .clone()
                    .reshape([1, num_classes])
                    .repeat_dim(0, batch_size * height * width);
        }

        // Ensure we have no log(0) later
        tensor = tensor.clamp(EPSILON, 1f64 - EPSILON);
        let neg_log_likehood = (smoothed_targets * tensor).sum_dim(1).neg();

        let [elems, _] = neg_log_likehood.clone().dims();
        let masked_neg_log_likehood = neg_log_likehood.mask_fill(
            masks
                .clone()
                .bool_not()
                .reshape([elems, 1])
                .repeat_dim(1, num_classes),
            0.0,
        );

        let loss = masked_neg_log_likehood.sum();

        let valid_pixels = masks.int().sum().into_scalar().to_u32();

        if valid_pixels > 0 {
            tracing::info!("Valid pixel for loss {:?} / {:?}", loss, valid_pixels);
            loss / valid_pixels
        } else {
            tracing::info!("Default loss (0)");
            Tensor::from_data([0.0], &device)
        }
    }

    fn compute_smoothed_targets(
        nr_classes: usize,
        batch_size: usize,
        targets: Tensor<B, 1, Int>,
        alpha: f32,
    ) -> Tensor<B, 2> {
        tracing::info!("Compute smoothed targets");
        let device = &targets.device();
        let [n_elems] = targets.dims();
        let mut targets_matrix = Tensor::<B, 2>::zeros([n_elems, nr_classes], device);
        let indices = targets.reshape([n_elems, 1]);
        let values = Tensor::ones([n_elems, 1], device);

        targets_matrix = targets_matrix.scatter(1, indices, values);
        targets_matrix * (1. - alpha) + alpha / nr_classes as f32
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
