"""
YOLO Livestock Detection CLI with Automatic Dataset Compilation
================================================================

This is the main entry point for the YOLO livestock detection system. It provides
a unified interface for dataset compilation, model training, and evaluation.

Key Features:
- Automatic dataset compilation before training (optional)
- Unified CLI interface for all operations
- Smart dataset handling and validation
- Comprehensive error handling and user feedback
"""
#  python main.py train --compile-inputs ./my_test/inputs/chicken1 ./my_test/inputs/chicken2 ./my_test/inputs/chicken3 ./my_test/inputs/cow1 ./my_test/inputs/cow2 ./my_test/inputs/sheep_3 ./my_test/inputs/uni1 ./my_test/inputs/uni2  --data ./unified --epochs 100 --dataset_shuffle
import argparse
from pathlib import Path
import sys, os
# Import compilation functions from the existing compile.py
# This allows us to reuse the existing dataset compilation logic
sys.path.append(str(Path(__file__).parent / "dataset"))

from utils.arg_helper import prepare_dataset
from utils.ui_helper import show_banner, validate_compile_args, validate_train_args, validate_eval_args, validate_labels_args, validate_augment_args, print_augment_config
from augment import setup_augmentation_environment, run_augmentation_pipeline, get_cuda_status
from utils.directory_helper import get_base_directory
from utils.modify_helper import modify_dataset_labels, validate_dataset_structure
from utils.hardware_info import cpu_info, ram_info, gpu_info

# Import inference functions
from inference import evaluate_model

from train import train_from_args


def build_parser():
    parser = argparse.ArgumentParser(
        description="YOLO Livestock Detection CLI with Automatic Dataset Compilation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
# Simple training with existing dataset
python main.py train --data ./my_dataset --epochs 50

# Compile multiple datasets then train
python main.py train --compile-inputs ./cow_data ./sheep_data --data ./unified --epochs 100

# Infer your images/videos with a model
python main.py eval --data ./my_dataset --output ./runs/eval --model ./model/default.pt
        """
    )

    subparsers = parser.add_subparsers(dest="mode", required=False)

    # ------------------- compile -------------------
    compile_parser = subparsers.add_parser(
        "compile",
        help="Compile dataset",
        description="Compile dataset"
    )
    cg = compile_parser.add_argument_group("Dataset Compilation Options")
    cg.add_argument("--compile-inputs", nargs="*", help="Datasets to compile")
    cg.add_argument("--compile-out", help="Output directory for compiled dataset")
    cg.add_argument("--dataset-shuffle", "--dataset_shuffle", dest="dataset_shuffle",
                    action="store_true", default=False,
                    help="Shuffle each source into 80/10/10 before training.")
    cg.add_argument("--shuffle-seed", type=int, default=None,
                    help="Seed for deterministic per-source reshuffle (when --dataset-shuffle)")
    cg.add_argument("--names", help="Class mapping JSON/YAML or file path")
    cg.add_argument("--data", help="Path to dataset (or output if compiling)")
    

    # ------------------- help -------------------
    subparsers.add_parser("help", help="Helper functions")
    
    # ------------------- recommend -------------------
    subparsers.add_parser("recommend", help="Hardware configuration recommendations")

    # ------------------- train -------------------
    train_parser = subparsers.add_parser("train", help="Train YOLO model")
    tg_c = train_parser.add_argument_group("Dataset Compilation Options")
    tg_c.add_argument("--compile-inputs", nargs="*", help="Datasets to compile")
    tg_c.add_argument("--compile-out", help="Output directory for compiled dataset")
    tg_c.add_argument("--dataset-shuffle", "--dataset_shuffle", dest="dataset_shuffle",
                      action="store_true", default=False,
                      help="Shuffle into 80/10/10 before training")
    
    tg_c.add_argument("--shuffle-seed", type=int, default=None,
                      help="Seed for deterministic per-source reshuffle (when --dataset-shuffle)")
    # tg_c.add_argument("--names", help="Class mapping JSON/YAML or file path")

    tg = train_parser.add_argument_group("Training Parameters")
    tg.add_argument("--data", help="Dataset path (or output if compiling)")
    tg.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    # tg.add_argument("--batch-size", type=int, default=16, help="Batch size (default: 16)")
    tg.add_argument("--augment", nargs="+", default=[], help="Data augmentations")
    tg.add_argument("--output", default="./runs/train", help="Output directory")
    tg.add_argument("--size", choices=["s", "m", "x"], default="s")
    tg.add_argument("--weights", type=str, default=None)
    tg.add_argument("--names", type=str, default=None, help="comma-separated names if --data is a directory")
    tg.add_argument("--imgsz", type=int, default=640)
    tg.add_argument("--lr", type=float, default=None)
    tg.add_argument("--optimizer", type=str, default=None)
    tg.add_argument("--device", type=str, default=None)
    tg.add_argument("--seed", type=int, default=42)
    tg.add_argument("--preset", choices=["quick", "balanced", "strong"], default=None)
    tg.add_argument("--resume", action="store_true")
    tg.add_argument("--save_json", action="store_true")
    tg.add_argument("--conf", type=float, default=None)
    tg.add_argument("--iou", type=float, default=None)
    tg.add_argument("--project", type=str, default="train")
    tg.add_argument("--name", type=str, default=None)
    tg.add_argument("--batch", type=int, default=16, help="(alias of --batch-size)")

    # ------------------- eval -------------------
    eval_parser = subparsers.add_parser("eval", help="Infer your images/videos with a model")
    eval_parser.add_argument("--data", help="Path to your dataset")
    eval_parser.add_argument("--output", default="./runs/eval", help="Output directory")
    eval_parser.add_argument("--model", default="./model/default.pt", help="YOLO model path")
    eval_parser.add_argument("--fps", type=int, default=15, help="Frames per second for video inference (ignored in image inference)")
    eval_parser.add_argument("--tracker", default="bt", help="Tracker type for video (ignored in image inference)")
    eval_parser.add_argument("--conf", type=float, default=None, help="Minimum detection confidence (uses YOLO default when omitted)")

    # ------------------- labels -------------------
    # Usage examples (replace [REPLACEABLE] placeholders with your values):
    # - Non-destructive copy:
    #   python main.py labels --data [REPLACEABLE:dataset_root] --old-label [REPLACEABLE:old_name] --new-label [REPLACEABLE:new_name]
    # - Explicit output directory:
    #   python main.py labels --data [REPLACEABLE:dataset_root] --old-label [REPLACEABLE:old_name] --new-label [REPLACEABLE:new_name] --out [REPLACEABLE:output_dir]
    # - In-place (destructive):
    #   python main.py labels --data [REPLACEABLE:dataset_root] --old-label [REPLACEABLE:old_name] --new-label [REPLACEABLE:new_name] --in-place
    # - Example with output (commented):
    #   # Command
    #   python user_system/main.py labels --data user_system/tests/cow2 --old-label cow --new-label cattle
    #   # Example console output (truncated):
    #   🧩 LABEL MODIFICATION MODE ACTIVATED
    #
    #   🔧 Relabel configuration:
    #         - data   : user_system/tests/cow2
    #         - old    : cow
    #         - new    : cattle
    #         - output : user_system/tests/cow2_modified
    #         - mode   : copy
    #
    #   📁 Copying dataset from user_system/tests/cow2 to user_system/tests/cow2_modified
    #   📝 Found 101 label files to process
    #   ✅ Updated dataset configuration:
    #      Old label 'cow' (index 0) → 'cattle' (index 1)
    #      Total classes: 2
    #
    #   ✅ Label modification complete. Stats: {'files_processed': 101, 'labels_changed': 142, 'classes_updated': 1}
    labels_parser = subparsers.add_parser("labels", help="Modify dataset class labels")
    labels_parser.add_argument("--data", help="Path to dataset root (contains data.yaml or dataset.yaml)")
    labels_parser.add_argument("--old-label", help="Existing class name to remap from")
    labels_parser.add_argument("--new-label", help="New class name to remap to")
    labels_parser.add_argument("--out", default=None, help="Output directory (default: <data>_modified if not --in-place)")
    labels_parser.add_argument("--in-place", action="store_true", help="Modify the dataset in place (destructive)")

    # ------------------- augment -------------------
    # Config-driven augmentation. The YAML controls modes and parameters;
    # CLI only provides input dataset roots, config path, and worker count.
    # Usage examples (replace [REPLACEABLE] placeholders with your values):
    #   python main.py augment --inputs [REPLACEABLE:dataset_root]
    #   python main.py augment --inputs [REPLACEABLE:ds1] [REPLACEABLE:ds2] --config user_system/configs/augment.yaml --workers 4
    # Note: Options like seed, out, in_place, counts/multiplier, prob_default,
    # and per-mode knobs (e.g., rotate.max_deg) are defined in augment.yaml.
    augment_parser = subparsers.add_parser("augment", help="Generate an augmented YOLO dataset")
    ag = augment_parser.add_argument_group("Augmentation Options")
    ag.add_argument("--inputs", nargs="*", help="One or more dataset roots (each contains data.yaml or dataset.yaml)")
    ag.add_argument("--config", default="user_system/configs/augment.yaml", help="Path to augmentation YAML (modes and parameters)")
    ag.add_argument("--workers", type=int, default=0, help="Number of parallel workers for I/O")

    return parser

def run_compile(args):
    print("🎯 COMPILE MODE ACTIVATED")
    if not validate_compile_args(args):
        return 0
    final_root = prepare_dataset(args)
    print(f"\n✅ Compilation complete.\n📁 Output: {final_root}")

def run_train(args):
    print("🎯 TRAINING MODE ACTIVATED")
    if not validate_train_args(args):
        return
    final_root = prepare_dataset(args)
    args.data = str(final_root)
    if getattr(args, "batch", None) is None and getattr(args, "batch_size", None) is not None:
        args.batch = args.batch_size

    args._cli_flags = {t for t in sys.argv[1:] if t.startswith("--")}

    print("\n🚀 TRAINING PHASE")
    print("=" * 30)
    print("📊 Training Configuration:")
    print(f"   Data path : {args.data}")
    print(f"   Epochs    : {args.epochs}")
    print(f"   Batch     : {getattr(args, 'batch', None)}")
    print(f"   Img size  : {args.imgsz}")
    print(f"   Preset    : {args.preset or '-'}")
    print(f"   Output    : {args.output}")

    print("🔧 Final hyperparams after preset & CLI merge:")
    print(f"   epochs={args.epochs}, batch={args.batch}, imgsz={args.imgsz}, "
          f"lr={args.lr}, optimizer={args.optimizer or '-'}, device={args.device or '-'}")

    run_dir = train_from_args(args)

    print(f"\n✅ Training finished. Artifacts saved to: {run_dir}")
    print("   - args.yaml / metrics.json (if --save_json) / weights/{best.pt,last.pt}")
    print("   > artifacts:")
    print(f"     - {run_dir / 'args.yaml'}")
    print(f"     - {run_dir / 'weights' / 'best.pt'} (when available)")
    print(f"     - {run_dir / 'metrics.json'} (if --save_json)")


def run_eval(args):
    print("📊 INFERENCE MODE ACTIVATED")
    print("=" * 30)
    print()
    data_path = validate_eval_args(args)
    if data_path is None:
        return

    try:
        results = evaluate_model(
            data_path=str(data_path),
            output_dir=args.output,
            model_name=args.model,
            target_fps=args.fps,
            tracker=args.tracker,
            confidence=args.conf
        )
    except Exception as e:
        print(f"❌ Error: {e}")
        print(f"\n💡 Troubleshooting tips:")
        print("   - Make sure the data path points to valid images or image directory")
        print("   - Check that you have write permissions for output directory")
        print("   - Ensure required dependencies are installed (ultralytics, opencv-python, matplotlib)")
        print("   - Try with different image formats if current ones fail")


def run_augment(args):
    print("🎯 AUGMENTATION MODE ACTIVATED")
    validated = validate_augment_args(args)
    if validated is None:
        return
    data_paths, config_path, cfg = validated
    print("\n✅ Input validation passed")
    print(f"   Inputs : {', '.join(str(p) for p in data_paths)}")
    print(f"   Config : {config_path}")
    print(f"   Workers: {getattr(args, 'workers', 0)}")
    
    # Display configuration in a nice format
    print_augment_config(cfg)
    
    # Set up augmentation environment
    base_dir = get_base_directory()
    created_datasets = setup_augmentation_environment(data_paths, cfg, base_dir)
    
    if created_datasets is None:
        print("\n❌ Failed to set up augmentation environment")
        return
    
    print(f"\n✅ Augmentation environment ready!")
    print(f"   Created {len(created_datasets)} dataset structures")
    for dataset_path in created_datasets:
        print(f"   - {dataset_path}")
    
    # Run the augmentation pipeline with parallel processing
    workers = getattr(args, 'workers', None)
    augmentation_stats = run_augmentation_pipeline(data_paths, created_datasets, cfg, max_workers=workers)
    
    print(f"\n🎉 AUGMENTATION COMPLETE!")
    print(f"   📊 Final Statistics:")
    print(f"   - Total images processed: {augmentation_stats['total_processed']}")
    print(f"   - Total augmented pairs created: {augmentation_stats['total_created']}")
    print(f"   - Dataset expansion: {augmentation_stats['expansion_ratio']:.2f}x")
    
def run_labels(args):
    print("🧩 LABEL MODIFICATION MODE ACTIVATED")
    data_path = validate_labels_args(args)
    if data_path is None:
        return
    if args.in_place:
        output_path = data_path
    else:
        output_path = Path(args.out) if args.out else data_path.parent / (data_path.name + "_modified")

    print("\n🔧 Relabel configuration:\n      - data   : {}\n      - old    : {}\n      - new    : {}\n      - output : {}\n      - mode   : {}".format(
        data_path, args.old_label, args.new_label, output_path, "in-place" if args.in_place else "copy"))

    try:
        stats = modify_dataset_labels(
            dataset_path=data_path,
            old_label=args.old_label,
            new_label=args.new_label,
            output_path=output_path
        )
    except Exception as e:
        print(f"\n❌ LABEL MODIFICATION FAILED")
        print(f"   Error: {e}")
        return
    print(f"\n✅ Label modification complete. Stats: {stats}")
    
def run_recommend(args):
    """
    Display Steam-style hardware configuration recommendations for YOLO training presets.
    Shows current system specs and recommended configurations for each preset.
    """
    print("💻 HARDWARE CONFIGURATION RECOMMENDATIONS")
    print("=" * 50)
    
    # Display current system information
    print("\n🔍 YOUR CURRENT SYSTEM:")
    print("-" * 25)
    try:
        print(f"   {cpu_info()}")
        print(f"   {ram_info()}")
        gpu_list = gpu_info()
        if gpu_list:
            for gpu in gpu_list:
                print(f"   {gpu}")
        else:
            print("   GPU: Not detected")
    except Exception as e:
        print(f"   ⚠️  Hardware detection failed: {e}")
        print("   Please check your system manually")
    
    # Quick Preset - Minimum Requirements
    print("\n🟢 QUICK PRESET - Minimum Requirements")
    print("-" * 40)
    print("   ⚡ Best for: Quick testing, small datasets (<1000 images)")
    print("   ⏱️  Training time: ~10-30 minutes")
    print("   📊 Configuration:")
    print("      • Epochs: 10")
    print("      • Batch size: 16") 
    print("      • Image size: 640x640")
    print("   💾 MINIMUM SYSTEM REQUIREMENTS:")
    print("      • CPU: 4+ cores")
    print("      • RAM: 8 GB")
    print("      • GPU: 4 GB VRAM (GTX 1650, RTX 3050)")
    print("      • Storage: 5 GB free space")
    print("   🎯 Command: python main.py train --preset quick --data ./dataset")
    
    # Balanced Preset - Recommended Requirements  
    print("\n🟡 BALANCED PRESET - Recommended Configuration")
    print("-" * 45)
    print("   ⚡ Best for: Production training, medium datasets (1K-10K images)")
    print("   ⏱️  Training time: ~1-3 hours")
    print("   📊 Configuration:")
    print("      • Epochs: 50")
    print("      • Batch size: 16")
    print("      • Image size: 640x640") 
    print("   💾 RECOMMENDED SYSTEM REQUIREMENTS:")
    print("      • CPU: 8+ cores")
    print("      • RAM: 16 GB")
    print("      • GPU: 6 GB VRAM (GTX 1660 Ti, RTX 3060)")
    print("      • Storage: 15 GB free space")
    print("   🎯 Command: python main.py train --preset balanced --data ./dataset")
    
    # Strong Preset - High-end Requirements
    print("\n🔴 STRONG PRESET - High Performance")
    print("-" * 35)
    print("   ⚡ Best for: Large datasets (10K+ images), maximum accuracy")
    print("   ⏱️  Training time: ~3-8 hours")
    print("   📊 Configuration:")
    print("      • Epochs: 100")
    print("      • Batch size: 32")
    print("      • Image size: 960x960")
    print("   💾 HIGH-END SYSTEM REQUIREMENTS:")
    print("      • CPU: 12+ cores")
    print("      • RAM: 32 GB")
    print("      • GPU: 8+ GB VRAM (RTX 3070, RTX 4060 Ti)")
    print("      • Storage: 50 GB free space")
    print("   🎯 Command: python main.py train --preset strong --data ./dataset")
    
    # Auto-recommendation based on detected hardware
    print("\n🎯 RECOMMENDATION FOR YOUR SYSTEM:")
    print("-" * 35)
    try:
        # Simple heuristic based on RAM detection
        ram_str = ram_info()
        if "GB" in ram_str:
            ram_gb = float(ram_str.split()[1])
            gpu_list = gpu_info()
            gpu_vram = 0
            
            # Try to extract VRAM from GPU info
            for gpu in gpu_list:
                if "VRAM:" in gpu and "GB" in gpu:
                    try:
                        vram_part = gpu.split("VRAM:")[1].strip()
                        if vram_part != "0)" and "GB" in vram_part:
                            gpu_vram = float(vram_part.split()[0])
                            break
                    except:
                        continue
            
            if ram_gb >= 32 and gpu_vram >= 8:
                recommendation = "🔴 STRONG preset recommended"
            elif ram_gb >= 16 and gpu_vram >= 6:
                recommendation = "🟡 BALANCED preset recommended"
            else:
                recommendation = "🟢 QUICK preset recommended"
                
            print(f"   Based on your hardware: {recommendation}")
            print(f"   Detected: {ram_gb:.0f}GB RAM, {gpu_vram:.0f}GB VRAM")
        else:
            print("   ⚠️  Unable to auto-recommend (hardware detection incomplete)")
            print("   💡 Start with QUICK preset and upgrade if performance allows")
            
    except Exception as e:
        print("   ⚠️  Auto-recommendation failed")
        print("   💡 Start with QUICK preset and upgrade based on your experience")
    
    print("\n" + "=" * 50)


def main():
    """
    Main function that handles command-line argument parsing and routes to appropriate modes.
    
    The CLI supports three main modes:
    1. train: Train a YOLO model with optional dataset compilation
    2. eval: Evaluate a trained model
    3. (future modes can be added here)
    """
    
    # Print CUDA status once at program start
    print(get_cuda_status())
    
    # Create the main argument parser
    parser = build_parser()


    # Parse command line arguments
    if len(sys.argv) == 1:
        show_banner()
        parser.print_help()
        return
    
    try:
        args, unknown_flags = parser.parse_known_args()
    except SystemExit:
        # This includes --help flag - argparse prints help and exits cleanly
        return

    # Gracefully report unrecognized flags without exiting
    if 'unknown_flags' in locals() and unknown_flags:
        print("\n⚠️  Ignoring unrecognized flags:")
        for uf in unknown_flags:
            print(f"   - {uf}")
    
    if not args.mode:
        print("\n❌ Please choose a mode: compile | train | eval | help | augment | labels")
        print("\n💡 Available commands:")
        print("   • python main.py compile - Combine multiple datasets")
        print("   • python main.py train  - Train YOLO model")
        print("   • python main.py eval   - Run inference")
        print("   • python main.py augment - Generate augmented datasets")
        print("   • python main.py labels - Modify class labels")
        print("   • python main.py help   - Show detailed examples")
        return
    # ============================================================================
    # TRAINING MODE EXECUTION
    # ============================================================================
    if args.mode == "help":
        show_banner()
        print("\n🎯 DETAILED HELP - System Modules & Examples")
        print("=" * 80)
        print("\nThe system consists of five main modules:\n")
        
        print("📦 1. DATASET COMPILER (python main.py compile)")
        print("   Combines multiple YOLO datasets into a unified, deduplicated dataset.")
        print("   Example:")
        print("   python main.py compile --compile-inputs ./cow_data ./sheep_data ./chicken_data \\")
        print("     --compile-out ./unified_dataset --dataset-shuffle --shuffle-seed 42\n")
        
        print("🎨 2. DATASET AUGMENTATOR (python main.py augment)")
        print("   Generates augmented versions of datasets using configurable transformations.")
        print("   Configure settings in: user_system/configs/augment.yaml")
        print("   Example:")
        print("   python main.py augment --inputs ./dataset1 ./dataset2 \\")
        print("     --config ./configs/augment.yaml --workers 4\n")
        
        print("🏷️  3. LABEL MAPPER (python main.py labels)")
        print("   Modifies class labels in existing datasets (non-destructive or in-place).")
        print("   Example:")
        print("   python main.py labels --data ./dataset --old-label cow --new-label cattle \\")
        print("     --out ./dataset_modified\n")
        
        print("🚂 4. TRAINER (python main.py train)")
        print("   Train YOLO models on livestock detection data with optional dataset compilation.")
        print("   Example:")
        print("   python main.py train --data ./unified_dataset --epochs 100 --batch 16 \\")
        print("     --size s --output ./runs/train\n")
        
        print("🔍 5. INFERENCE MODULE (python main.py eval)")
        print("   Run object detection on images, videos, or batch directories.")
        print("   Example:")
        print("   python main.py eval --data ./test_images --output ./runs/eval \\")
        print("     --model ./model/default.pt --conf 0.25 --tracker bt\n")
        
        print("💡 For hardware recommendations:")
        print("   python main.py recommend")
        
    elif args.mode == "recommend":
        run_recommend(args)
        
    elif args.mode == "compile":
        run_compile(args)


    elif args.mode == "train":
        run_train(args)

    elif args.mode == "eval":
        run_eval(args)
    elif args.mode == "labels":
        run_labels(args)

    elif args.mode == "augment":
        run_augment(args)

if __name__ == "__main__":
    # Entry point of the script
    main()
