use std::path::{Path, PathBuf};

use anyhow::Result;
use burn::data::dataloader::Dataset;
use burn::data::dataset::vision::ImageFolderDataset;
use burn::train::metric::LossMetric;
use burn::{
    backend::{Autodiff, Wgpu, wgpu::WgpuDevice},
    data::dataloader::DataLoaderBuilder,
    optim::AdamConfig,
    prelude::*,
    train::LearnerBuilder,
};
use clap::Args;
use rust_unet::{
    InputMode, SegmentationConfig, SegmentationMode, UNetConfig, dataset::SegmentationBatcher,
};

#[derive(Args)]
pub struct TrainArgs {
    #[arg(short, long)]
    pub train_data_dir: PathBuf,

    #[arg(short, long)]
    pub valid_data_dir: PathBuf,

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

    #[arg(long, default_value_t = 2)]
    pub num_classes: usize,

    #[arg(long, default_value_t = 0.8)]
    pub train_ratio: f64,

    #[arg(long, default_value_t = 42)]
    pub seed: u64,
}

fn create_artifact_dir(artifact_dir: &str) {
    std::fs::remove_dir_all(artifact_dir).ok();
    std::fs::create_dir_all(artifact_dir).ok();
}

pub fn load_segmentation_dataset(
    root: &Path,
    config: &SegmentationConfig,
) -> Result<ImageFolderDataset, Box<dyn std::error::Error>> {
    let root_path = Path::new(root);
    let images_dir = root_path.join("images");
    let masks_dir = root_path.join("masks");

    if !images_dir.exists() || !images_dir.is_dir() {
        return Err(format!("Images directory does not exist: {:?}", images_dir).into());
    }

    if !masks_dir.exists() || !masks_dir.is_dir() {
        return Err(format!("Masks directory does not exist: {:?}", masks_dir).into());
    }

    let mut image_mask_pairs = Vec::new();
    let img_extensions = ["jpg", "jpeg", "png"];

    for entry in std::fs::read_dir(&images_dir)? {
        let entry = entry?;
        let path = entry.path();

        if path.is_file()
            && path
                .extension()
                .and_then(|ext| ext.to_str())
                .is_some_and(|ext| {
                    img_extensions
                        .iter()
                        .any(|&valid_ext| valid_ext.eq_ignore_ascii_case(ext))
                })
        {
            if let Some(stem) = path.file_stem() {
                let mask_path = masks_dir.join(format!("{}.png", stem.to_string_lossy()));

                if mask_path.exists() {
                    image_mask_pairs.push((path, mask_path));
                }
            }
        }
    }

    let class_names = match &config.mode {
        SegmentationMode::Binary => config
            .class_names
            .clone()
            .unwrap_or_else(|| vec!["class_0".to_string(), "class_1".to_string()]),
        SegmentationMode::Multiclass { num_classes } => config
            .class_names
            .clone()
            .unwrap_or_else(|| (0..*num_classes).map(|i| format!("class_{}", i)).collect()),
    };

    ImageFolderDataset::new_segmentation_with_items(image_mask_pairs, &class_names)
        .map_err(|e| e.into())
}

pub fn run(args: &TrainArgs) -> Result<()> {
    type MyBackend = Wgpu<f32, i32>;
    type MyAutodiffBackend = Autodiff<MyBackend>;

    create_artifact_dir(args.artifact_dir.to_str().expect(""));

    println!("Initializing device...");
    let device = WgpuDevice::default();

    MyAutodiffBackend::seed(args.seed);

    let seg_mode = if args.num_classes == 1 {
        SegmentationMode::Binary
    } else {
        SegmentationMode::Multiclass {
            num_classes: args.num_classes,
        }
    };

    let seg_config = SegmentationConfig::new(
        seg_mode,
        InputMode::RGB,
        [256, 256], // fixed size for now
    );

    println!(
        "Loading training dataset from {}...",
        args.train_data_dir.display()
    );
    let train_dataset = match load_segmentation_dataset(&args.train_data_dir, &seg_config) {
        Ok(dataset) => {
            println!("Loaded {} samples (training dataset)", dataset.len());
            dataset
        }
        Err(e) => {
            return Err(anyhow::anyhow!("Failed to load dataset: {}", e));
        }
    };

    let valid_dataset = match load_segmentation_dataset(&args.valid_data_dir, &seg_config) {
        Ok(dataset) => {
            println!("Loaded {} samples (valid dataset)", dataset.len());
            dataset
        }
        Err(e) => {
            return Err(anyhow::anyhow!("Failed to load dataset: {}", e));
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
    let model = UNetConfig::new([256, 256])
        .with_base_channels(args.base_channels)
        .with_num_classes(args.num_classes)
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
    let learner = LearnerBuilder::new(&args.artifact_dir)
        .metric_train_numeric(LossMetric::new())
        .metric_valid_numeric(LossMetric::new())
        .num_epochs(args.epochs)
        .build(model, optimizer, args.lr);

    let model_trained = learner.fit(dataloader_train, dataloader_valid);

    if args.save_checkpoints {
        println!("Saving model checkpoint...");
    }

    println!("Training completed successfully!");
    Ok(())
}
