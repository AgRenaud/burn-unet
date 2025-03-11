use anyhow::Result;
use clap::{Parser, Subcommand};

mod tasks;

#[derive(Parser)]
#[command(
    name = "rust-unet",
    about = "U-Net image segmentation toolkit",
    author,
    version
)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    Train(tasks::train::TrainArgs),
}

fn main() -> Result<()> {
    let cli = Cli::parse();

    match &cli.command {
        Commands::Train(args) => tasks::train::run(args),
        _ => panic!("Command not found"),
    }
}
