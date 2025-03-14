use image::ColorType;
use std::path::{Path, PathBuf};
use thiserror::Error;

use burn::data::dataset::transform::{Mapper, MapperDataset};
use burn::data::dataset::vision::PixelDepth;
use burn::data::dataset::{Dataset, InMemDataset};

use burn_unet::{SegmentationImageItem, SegmentationImageItemRaw};

pub fn load_mask_to_vec_usize(mask_path: &PathBuf) -> Vec<usize> {
    let image = image::open(mask_path).unwrap();

    match image.color() {
        ColorType::L8 => image.into_luma8().iter().map(|&x| x as usize).collect(),
        ColorType::L16 => image.into_luma16().iter().map(|&x| x as usize).collect(),
        ColorType::Rgb8 => image
            .into_rgb8()
            .iter()
            .step_by(3)
            .map(|&x| x as usize)
            .collect(),
        ColorType::Rgb16 => image
            .into_rgb16()
            .iter()
            .step_by(3)
            .map(|&x| x as usize)
            .collect(),
        _ => panic!("Unrecognized image color type"),
    }
}

pub fn load_fov_mask_to_vec_bool(mask_path: &PathBuf) -> Vec<bool> {
    let image = image::open(mask_path).unwrap();

    match image.color() {
        ColorType::L8 => image.into_luma8().iter().map(|&x| x > 0).collect(),
        ColorType::L16 => image.into_luma16().iter().map(|&x| x > 0).collect(),
        ColorType::Rgb8 => image
            .into_rgb8()
            .iter()
            .step_by(3)
            .map(|&x| x > 0)
            .collect(),
        ColorType::Rgb16 => image
            .into_rgb16()
            .iter()
            .step_by(3)
            .map(|&x| x > 0)
            .collect(),
        _ => panic!("Unrecognized image color type"),
    }
}

struct PathToSegmentationItem;

impl Mapper<SegmentationImageItemRaw, SegmentationImageItem> for PathToSegmentationItem {
    fn map(&self, item: &SegmentationImageItemRaw) -> SegmentationImageItem {
        let mask = load_mask_to_vec_usize(&item.mask_path);

        let fov_mask = item
            .fov_mask_path
            .as_ref()
            .map(load_fov_mask_to_vec_bool);

        let image = image::open(&item.image_path).unwrap();

        let img_vec = match image.color() {
            ColorType::L8 => image
                .into_luma8()
                .iter()
                .map(|&x| PixelDepth::U8(x))
                .collect(),
            ColorType::La8 => image
                .into_luma_alpha8()
                .iter()
                .map(|&x| PixelDepth::U8(x))
                .collect(),
            ColorType::L16 => image
                .into_luma16()
                .iter()
                .map(|&x| PixelDepth::U16(x))
                .collect(),
            ColorType::La16 => image
                .into_luma_alpha16()
                .iter()
                .map(|&x| PixelDepth::U16(x))
                .collect(),
            ColorType::Rgb8 => image
                .into_rgb8()
                .iter()
                .map(|&x| PixelDepth::U8(x))
                .collect(),
            ColorType::Rgba8 => image
                .into_rgba8()
                .iter()
                .map(|&x| PixelDepth::U8(x))
                .collect(),
            ColorType::Rgb16 => image
                .into_rgb16()
                .iter()
                .map(|&x| PixelDepth::U16(x))
                .collect(),
            ColorType::Rgba16 => image
                .into_rgba16()
                .iter()
                .map(|&x| PixelDepth::U16(x))
                .collect(),
            ColorType::Rgb32F => image
                .into_rgb32f()
                .iter()
                .map(|&x| PixelDepth::F32(x))
                .collect(),
            ColorType::Rgba32F => image
                .into_rgba32f()
                .iter()
                .map(|&x| PixelDepth::F32(x))
                .collect(),
            _ => panic!("Unrecognized image color type"),
        };

        SegmentationImageItem {
            image: img_vec,
            mask,
            fov_mask,
        }
    }
}

#[derive(Error, Debug)]
pub enum ImageDatasetError {
    #[error("I/O error: `{0}`")]
    IOError(String),

    #[error("Invalid file extension: `{0}`")]
    InvalidFileExtensionError(String),
}

const SUPPORTED_FILES: [&str; 4] = ["bmp", "jpg", "jpeg", "png"];

type DriveDatasetMapper = MapperDataset<
    InMemDataset<SegmentationImageItemRaw>,
    PathToSegmentationItem,
    SegmentationImageItemRaw,
>;

pub struct DriveDataset {
    dataset: DriveDatasetMapper,
}

impl Dataset<SegmentationImageItem> for DriveDataset {
    fn get(&self, index: usize) -> Option<SegmentationImageItem> {
        self.dataset.get(index)
    }

    fn len(&self) -> usize {
        self.dataset.len()
    }
}

impl DriveDataset {
    /// Create a segmentation dataset with specified triplets.
    ///
    /// # Arguments
    ///
    /// * `items` - List of dataset items, each represented by a tuple (image path, groundtruth path, optional fov mask path)
    ///
    /// # Returns
    /// A new dataset instance
    pub fn new_with_triplets<P: AsRef<Path>>(
        items: Vec<(P, P, Option<P>)>,
    ) -> Result<Self, ImageDatasetError> {
        let items = items
            .into_iter()
            .map(|(image_path, gt_path, fov_path)| {
                let image_path = image_path.as_ref().to_path_buf();
                let mask_path = gt_path.as_ref().to_path_buf();
                let fov_mask_path = fov_path.map(|fov_mask_path| fov_mask_path.as_ref().to_path_buf());

                Self::check_extension(
                    &image_path
                        .extension()
                        .and_then(|ext| ext.to_str())
                        .unwrap_or(""),
                )?;

                Ok(SegmentationImageItemRaw {
                    image_path,
                    mask_path,
                    fov_mask_path,
                })
            })
            .collect::<Result<Vec<_>, _>>()?;

        Self::with_items(items)
    }

    /// Create a segmentation dataset from the root folder with triplets of images.
    ///
    /// # Arguments
    ///
    /// * `root` - Dataset root folder. Should contain 'images', 'groundtruth', and optionally 'masks' subfolders.
    ///
    /// # Returns
    /// A new dataset instance.
    pub fn new_from_folders<P: AsRef<Path>>(root: P) -> Result<Self, ImageDatasetError> {
        let root_path = root.as_ref();
        let images_dir = root_path.join("images");
        let groundtruth_dir = root_path.join("groundtruth");
        let masks_dir = root_path.join("masks");

        if !images_dir.exists() || !images_dir.is_dir() {
            return Err(ImageDatasetError::IOError(format!(
                "Images directory does not exist: {:?}",
                images_dir
            )));
        }

        if !groundtruth_dir.exists() || !groundtruth_dir.is_dir() {
            return Err(ImageDatasetError::IOError(format!(
                "Groundtruth directory does not exist: {:?}",
                groundtruth_dir
            )));
        }

        let has_masks = masks_dir.exists() && masks_dir.is_dir();

        let mut image_gt_mask_triplets = Vec::new();

        for entry in
            std::fs::read_dir(&images_dir).expect("Unable to read directory images directory")
        {
            if let Ok(entry) = entry {
                let path = entry.path();

                if path.is_file()
                    && path
                        .extension()
                        .and_then(|ext| ext.to_str())
                        .is_some_and(|ext| {
                            SUPPORTED_FILES
                                .iter()
                                .any(|&valid_ext| valid_ext.eq_ignore_ascii_case(ext))
                        })
                {
                    if let Some(file_stem) = path.file_stem() {
                        let file_name = file_stem.to_string_lossy();

                        let mut gt_path = None;
                        for ext in SUPPORTED_FILES {
                            let potential_gt =
                                groundtruth_dir.join(format!("{}.{}", file_name, ext));
                            if potential_gt.exists() {
                                gt_path = Some(potential_gt);
                                break;
                            }
                        }

                        let mut mask_path = None;
                        if has_masks {
                            for ext in SUPPORTED_FILES {
                                let potential_mask =
                                    masks_dir.join(format!("{}.{}", file_name, ext));
                                if potential_mask.exists() {
                                    mask_path = Some(potential_mask);
                                    break;
                                }
                            }
                        }
                        if let Some(gt_path) = gt_path {
                            image_gt_mask_triplets.push((path, gt_path, mask_path));
                        }
                    }
                }
            }
        }

        if image_gt_mask_triplets.is_empty() {
            return Err(ImageDatasetError::IOError(
                "No valid image-groundtruth pairs found".to_string(),
            ));
        }

        Self::new_with_triplets(image_gt_mask_triplets)
    }

    fn with_items(items: Vec<SegmentationImageItemRaw>) -> Result<Self, ImageDatasetError> {
        let dataset = InMemDataset::new(items);
        let mapper = PathToSegmentationItem;
        let dataset = MapperDataset::new(dataset, mapper);

        Ok(Self { dataset })
    }

    fn check_extension<S: AsRef<str>>(extension: &S) -> Result<String, ImageDatasetError> {
        let extension = extension.as_ref();
        if !SUPPORTED_FILES.contains(&extension) && !extension.is_empty() {
            Err(ImageDatasetError::InvalidFileExtensionError(
                extension.to_string(),
            ))
        } else {
            Ok(extension.to_string())
        }
    }
}
