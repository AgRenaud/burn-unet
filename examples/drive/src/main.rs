mod drive_dataset;

use anyhow::Result;
use clap::Parser;
use std::path::{Path, PathBuf};

use burn::{
    backend::{Autodiff, Wgpu, wgpu::WgpuDevice},
    data::dataloader::DataLoaderBuilder,
    optim::AdamConfig,
    prelude::*,
    record::CompactRecorder,
    train::{
        LearnerBuilder,
        metric::{CpuMemory, CpuTemperature, CpuUse, LossMetric},
    },
};

use crate::drive_dataset::DriveDataset;
use burn::data::dataloader::Dataset;
use burn_unet::{
    InputMode, SegmentationConfig, SegmentationMode, UNetConfig,
    dataset::SegmentationBatcher,
    training::{IoULoss, IoUMetric},
};

#[derive(Parser, Debug)]
pub struct TrainArgs {
    #[arg(short, long)]
    pub data_dir: PathBuf,

    #[arg(short, long, default_value_t = 10)]
    pub epochs: usize,

    #[arg(short, long, default_value_t = 8)]
    pub batch_size: usize,

    #[arg(short, long, default_value_t = 0.0003)]
    pub lr: f64,

    #[arg(long, default_value_t = 4)]
    pub num_workers: usize,

    #[arg(long, default_value_t = true)]
    pub save_checkpoints: bool,

    #[arg(short, long, default_value = "artifacts")]
    pub artifact_dir: PathBuf,

    #[arg(long, default_value_t = 64)]
    pub base_channels: usize,

    #[arg(long, default_value_t = 42)]
    pub seed: u64,

    #[arg(long, default_value_t = 640)]
    pub image_size: usize,

    #[arg(long, short, action, default_value = "false")]
    pub grayscale: bool,
}

fn create_artifact_dir(artifact_dir: &str) {
    std::fs::remove_dir_all(artifact_dir).ok();
    std::fs::create_dir_all(artifact_dir).ok();
}

pub fn main() -> Result<()> {
    type MyBackend = Wgpu<f32, i32>;
    type MyAutodiffBackend = Autodiff<MyBackend>;

    let args = TrainArgs::parse();

    let artifact_dir = args.artifact_dir.to_str().expect("Can't find artifact dir");
    create_artifact_dir(artifact_dir);

    println!("Initializing device...");
    let device = WgpuDevice::default();

    MyAutodiffBackend::seed(args.seed);

    let seg_config = SegmentationConfig::new(
        SegmentationMode::Binary,
        InputMode::RGB,
        [args.image_size, args.image_size],
    );

    println!("Loading datasets...");
    let train_dir = args.data_dir.join("train");
    let valid_dir = args.data_dir.join("val");

    // Load training dataset with triplets (image, groundtruth, FOV mask)
    println!("Loading training dataset from {}...", train_dir.display());
    let train_dataset = match DriveDataset::new_from_folders(&train_dir) {
        Ok(dataset) => {
            println!("Loaded {} samples (training dataset)", dataset.len());
            dataset
        }
        Err(e) => {
            return Err(anyhow::anyhow!("Failed to load training dataset: {}", e));
        }
    };

    // Load validation dataset with triplets
    println!("Loading validation dataset from {}...", valid_dir.display());
    let valid_dataset = match DriveDataset::new_from_folders(&valid_dir) {
        Ok(dataset) => {
            println!("Loaded {} samples (validation dataset)", dataset.len());
            dataset
        }
        Err(e) => {
            return Err(anyhow::anyhow!("Failed to load validation dataset: {}", e));
        }
    };

    println!("Creating data batchers...");
    let batcher_train =
        SegmentationBatcher::<MyAutodiffBackend>::new(device.clone(), seg_config.clone());

    let batcher_valid = SegmentationBatcher::<MyBackend>::new(device.clone(), seg_config.clone());

    println!(
        "Building dataloaders with batch size {}...",
        args.batch_size
    );
    let dataloader_train = DataLoaderBuilder::new(batcher_train)
        .batch_size(args.batch_size)
        .num_workers(args.num_workers)
        .shuffle(args.seed)
        .build(train_dataset);

    let dataloader_valid = DataLoaderBuilder::new(batcher_valid)
        .shuffle(args.seed)
        .build(valid_dataset);

    println!(
        "Creating U-Net model with {} base channels...",
        args.base_channels
    );
    let model = UNetConfig::new([args.image_size, args.image_size])
        .with_base_channels(args.base_channels)
        .with_num_classes(2) // Vessels and background
        .init(&device);

    println!(
        "Initializing Adam optimizer with learning rate {}...",
        args.lr
    );
    let optimizer = AdamConfig::new().init();

    let checkpoint_dir = PathBuf::from("checkpoints");
    if !checkpoint_dir.exists() {
        std::fs::create_dir(&checkpoint_dir)?;
        println!("Created checkpoint directory: {}", checkpoint_dir.display());
    }

    println!("Building learner...");
    let mut learner = LearnerBuilder::new(artifact_dir)
        // Model metrics
        .metric_train_numeric(LossMetric::new())
        .metric_valid_numeric(LossMetric::new())
        // Add IoU metric for evaluation
        // .metric_train_numeric(IoUMetric::new())
        // .metric_valid_numeric(IoUMetric::new())
        // System metrics
        .metric_train_numeric(CpuUse::new())
        .metric_valid_numeric(CpuUse::new())
        .metric_train_numeric(CpuMemory::new())
        .metric_valid_numeric(CpuMemory::new())
        .metric_train_numeric(CpuTemperature::new())
        .metric_valid_numeric(CpuTemperature::new())
        .devices(vec![device.clone()])
        .num_epochs(args.epochs)
        .summary();

    if args.save_checkpoints {
        learner = learner.with_file_checkpointer(CompactRecorder::new())
    }

    let learner = learner.build(model, optimizer, args.lr);

    let model_trained = learner.fit(dataloader_train, dataloader_valid);

    if args.save_checkpoints {
        println!("Saving model checkpoint...");
        model_trained
            .save_file(format!("{artifact_dir}/model"), &CompactRecorder::new())
            .expect("Trained model should be saved successfully");
    }

    println!("Training completed successfully!");
    Ok(())
}
