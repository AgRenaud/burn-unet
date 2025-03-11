use std::marker::PhantomData;

use burn::prelude::*;
use burn::train::metric::state::{FormatOptions, NumericMetricState};
use burn::train::metric::{Metric, MetricEntry, MetricMetadata, Numeric};
use derive_new::new;

#[derive(Default)]
pub struct IoUMetric<B: Backend> {
    state: NumericMetricState,
    pad_token: Option<usize>,
    _b: PhantomData<B>,
}

#[derive(new)]
pub struct IoUInput<B: Backend> {
    outputs: Tensor<B, 2>,
    targets: Tensor<B, 1, Int>,
}

impl<B: Backend> IoUMetric<B> {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_pad_token(mut self, index: usize) -> Self {
        self.pad_token = Some(index);
        self
    }
}

impl<B: Backend> Metric for IoUMetric<B> {
    type Input = IoUInput<B>;
    const NAME: &'static str = "IoU";

    fn update(&mut self, input: &IoUInput<B>, _metadata: &MetricMetadata) -> MetricEntry {
        let targets = input.targets.clone();
        let outputs = input.outputs.clone();

        let [batch_size, n_classes] = outputs.dims();

        let predictions = outputs.argmax(1).reshape([batch_size]);

        let iou = match self.pad_token {
            Some(pad_token) => {
                let mask = targets.clone().equal_elem(pad_token as i64);

                let mut total_iou = 0.0;
                let mut valid_samples = 0;

                for class_idx in 0..n_classes {
                    if class_idx == pad_token {
                        continue;
                    }

                    let target_mask = targets.clone().equal_elem(class_idx as i64).float();
                    let pred_mask = predictions.clone().equal_elem(class_idx as i64).float();

                    let valid_mask = mask.clone().bool_not().float();
                    let target_mask = target_mask * valid_mask.clone();
                    let pred_mask = pred_mask * valid_mask;

                    let intersection = (target_mask.clone() * pred_mask.clone())
                        .sum()
                        .into_scalar()
                        .elem::<f64>();
                    let union = (target_mask.clone() + pred_mask.clone()
                        - (target_mask.clone() * pred_mask.clone()))
                    .sum()
                    .into_scalar()
                    .elem::<f64>();

                    if union > 0.0 {
                        total_iou += intersection / union;
                        valid_samples += 1;
                    }
                }

                if valid_samples > 0 {
                    total_iou / valid_samples as f64
                } else {
                    0.0
                }
            }
            None => {
                let mut total_iou = 0.0;
                let mut valid_samples = 0;

                for class_idx in 0..n_classes {
                    let target_mask = targets.clone().equal_elem(class_idx as i64).float();
                    let pred_mask = predictions.clone().equal_elem(class_idx as i64).float();

                    let intersection = (target_mask.clone() * pred_mask.clone())
                        .sum()
                        .into_scalar()
                        .elem::<f64>();
                    let union = (target_mask.clone() + pred_mask.clone()
                        - (target_mask.clone() * pred_mask.clone()))
                    .sum()
                    .into_scalar()
                    .elem::<f64>();

                    if union > 0.0 {
                        total_iou += intersection / union;
                        valid_samples += 1;
                    }
                }

                if valid_samples > 0 {
                    total_iou / valid_samples as f64
                } else {
                    0.0
                }
            }
        };

        self.state.update(
            100.0 * iou,
            batch_size,
            FormatOptions::new(Self::NAME).unit("%").precision(2),
        )
    }

    fn clear(&mut self) {
        self.state.reset()
    }
}

impl<B: Backend> Numeric for IoUMetric<B> {
    fn value(&self) -> f64 {
        self.state.value()
    }
}
