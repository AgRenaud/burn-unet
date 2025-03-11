use burn::{
    backend::NdArray,
    prelude::*,
    tensor::{Int, Transaction},
    train::metric::{Adaptor, ItemLazy, LossInput},
};
use derive_new::new;

#[derive(new)]
pub struct SegmentationOutput<B: Backend> {
    pub loss: Tensor<B, 1>,
    pub output: Tensor<B, 4>,
    pub targets: Tensor<B, 4, Int>,
}

impl<B: Backend> ItemLazy for SegmentationOutput<B> {
    type ItemSync = SegmentationOutput<NdArray>;

    fn sync(self) -> Self::ItemSync {
        let [output, loss, targets] = Transaction::default()
            .register(self.output)
            .register(self.loss)
            .register(self.targets)
            .execute()
            .try_into()
            .expect("Correct amount of tensor data");

        let device = &Default::default();

        SegmentationOutput {
            output: Tensor::from_data(output, device),
            loss: Tensor::from_data(loss, device),
            targets: Tensor::from_data(targets, device),
        }
    }
}

impl<B: Backend> Adaptor<LossInput<B>> for SegmentationOutput<B> {
    fn adapt(&self) -> LossInput<B> {
        LossInput::new(self.loss.clone())
    }
}
