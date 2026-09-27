# NEW: pretty banner & header
from pathlib import Path
import sys
from typing import Optional, Tuple, Any
try:
    # Local import to avoid circulars at module import time in some contexts
    from .modify_helper import validate_dataset_structure
except Exception:
    validate_dataset_structure = None  # Fallback; functions will guard

def show_banner():
    print("\n" + "="*80)
    print("🚀 YOLO Livestock Detection System")
    print("="*80)
    print("\n   Use 'python main.py help' for detailed usage examples")
    print("\n" + "="*80)


def print_mode_header(args):
    print(f"\n🔍 Mode: {args.mode}")
    print(f"📋 Arguments: {args}")
    print("-" * 60)
# NEW: validations (return values signal success/failure)

def validate_compile_args(args):
    try:
        temp = args.compile_out
    except Exception as e:
        temp = None

    try:
        # Check if compile_inputs is None, empty list, or contains empty strings
        compile_inputs = getattr(args, 'compile_inputs', None)
        if not compile_inputs or (isinstance(compile_inputs, list) and len(compile_inputs) == 0) or \
           (isinstance(compile_inputs, list) and all(not inp.strip() for inp in compile_inputs)):
            print("\n❌ Missing required info for compilation.")
            print("   💡 Provide --compile-inputs with at least one directory and either --compile-out or --data (as output root).")
            print("   Example: python main.py compile --compile-inputs ./cow ./sheep --data ./unified")
            return False

        # Validate that each provided input path exists; print missing ones gracefully
        missing = []
        for p in compile_inputs:
            try:
                sp = str(p or '').strip()
                if not sp:
                    continue
                if not Path(sp).exists():
                    missing.append(sp)
            except Exception:
                # If path formatting throws, treat as missing
                missing.append(str(p))
        if missing:
            print("\n❌ Some input paths do not exist:")
            for m in missing:
                print(f"   - {m}")
            print("   💡 Please check the paths and try again.")
            return False
        
        if not (temp or args.data):
            print("\n❌ Missing output directory for compilation.")
            print("   💡 Provide either --compile-out or --data (as output root).")
            print("   Example: python main.py compile --compile-inputs ./cow ./sheep --data ./unified")
            return False
        return True
    except Exception as e:
        return False


def validate_train_args(args):
    # compile+train path
    try:
        # Check if compile_inputs is provided and not empty
        compile_inputs = getattr(args, 'compile_inputs', None)
        has_valid_compile_inputs = (compile_inputs and 
                                  isinstance(compile_inputs, list) and 
                                  len(compile_inputs) > 0 and 
                                  not all(not inp.strip() for inp in compile_inputs))
        
        if has_valid_compile_inputs:
            # Gracefully validate each input path exists
            missing = []
            for p in compile_inputs:
                try:
                    sp = str(p or '').strip()
                    if not sp:
                        continue
                    if not Path(sp).exists():
                        missing.append(sp)
                except Exception:
                    missing.append(str(p))
            if missing:
                print("\n❌ Some input paths for compilation do not exist:")
                for m in missing:
                    print(f"   - {m}")
                print("   💡 Please correct these paths or remove them before training.")
                return False

            try:
                temp = args.compile_out
            except Exception as e:
                temp = None
            if not (temp or args.data):
                print("\n❌ When using --compile-inputs, also specify an output via --compile-out or --data.")
                print("   Example: python main.py train --compile-inputs ./cow ./sheep --data ./unified --epochs 100")
                return False
            return True
        else:
            # If compile_inputs was provided but is empty, show specific error
            if compile_inputs is not None:
                print("\n❌ --compile-inputs was provided but no directories were specified.")
                print("   💡 Provide at least one directory with --compile-inputs.")
                print("   Example: python main.py train --compile-inputs ./cow ./sheep --data ./unified --epochs 100")
                return False
            
            # No compile_inputs provided, check for data
            if not args.data:
                print("\n❌ Please provide --data for training (or add --compile-inputs to build a dataset first).")
                print("   Example: python main.py train --data ./my_dataset --epochs 50")
                return False
            return True
    except Exception as e:
        print("\n❌ Please provide --data for training (or add --compile-inputs to build a dataset first).")
        print("   Example: python main.py train --data ./my_dataset --epochs 50")
        return False


def validate_eval_args(args):
    args_data = getattr(args, 'data', None)  
    if not args_data: 
        print(f"\n❌ Missing required argument: --data")
        print("   Example: python main.py eval --data ./test_images --output ./run/eval --model ./model/default.pt")
        return None
    
    data_path = Path(args_data)
    if not data_path.exists():
        print(f"\n❌ Test data not found: {data_path}")
        print("   💡 Ensure the path exists and is accessible.")
        return None
    return data_path


def validate_labels_args(args) -> Optional[Path]:
    """Validate args for label modification. Returns dataset Path or None.
    Requirements:
      --data must exist and be a YOLO dataset (yaml + labels present)
      --old-label and --new-label must be provided (argparse enforces required)
    """
    # --data
    data_arg = getattr(args, "data", None)
    if not data_arg:
        print("\n❌ Label modification requires --data")
        return None
    # --old-label / --new-label
    if not getattr(args, "old_label", None):
        print("\n❌ Label modification requires --old-label")
        return None
    if not getattr(args, "new_label", None):
        print("\n❌ Label modification requires --new-label")
        return None
    data_path = Path(str(data_arg))
    if not data_path.exists():
        print(f"\n❌ Dataset path not found: {data_path}")
        return None
    if validate_dataset_structure is None:
        print("\n❌ Internal error: dataset validator unavailable")
        return None
    if not validate_dataset_structure(data_path):
        print(f"\n❌ Invalid dataset structure at: {data_path}")
        print("   - Expected a data.yaml or dataset.yaml and at least one labels file")
        return None
    return data_path


def validate_augment_args(args) -> Optional[Tuple[list[Path], Path, Any]]:
    """Validate args for augmentation.
    
    Args:
        args: Command line arguments containing inputs and config parameters
        
    Returns:
        On success: Tuple containing:
            - list[Path]: List of validated dataset paths from --inputs
            - Path: Path to the configuration file (from --config or default)
            - Any: Loaded configuration dictionary from YAML file
        On failure: None
        
    Validation checks:
        - At least one input dataset provided via --inputs
        - All input paths exist and are valid YOLO datasets
        - Configuration file exists and is valid YAML
        - Configuration contains required 'modes' key
    """
    # --inputs
    inputs = getattr(args, "inputs", None) or []
    if not inputs:
        print("\n❌ Augmentation requires at least one dataset via --inputs")
        return None
    data_paths: list[Path] = []
    missing: list[str] = []
    invalid: list[str] = []
    for raw in inputs:
        p = Path(str(raw))
        if not p.exists():
            missing.append(str(p))
            continue
        if validate_dataset_structure is None or not validate_dataset_structure(p):
            invalid.append(str(p))
            continue
        data_paths.append(p)
    if missing:
        print("\n❌ Some inputs do not exist:")
        for m in missing:
            print(f"   - {m}")
        return None
    if invalid:
        print("\n❌ Some inputs are not valid YOLO datasets (expect data.yaml and labels):")
        for m in invalid:
            print(f"   - {m}")
        return None
    # Resolve config
    # Resolve config path with robust relative handling
    # Base directory relative to the running entrypoint (main.py)
    try:
        main_file = Path(sys.modules['__main__'].__file__).resolve()
        base_dir = main_file.parent  # typically .../user_system
    except Exception:
        base_dir = Path(__file__).resolve().parents[1]
    default_cfg = base_dir / "configs" / "augment.yaml"
    cfg_arg = getattr(args, "config", None)
    candidates = []
    if cfg_arg:
        p = Path(str(cfg_arg))
        # If absolute or exists as-given, use it; otherwise try CWD-relative and user_root-relative
        if p.is_absolute() and p.exists():
            candidates = [p]
        else:
            candidates = [p, Path.cwd() / p, base_dir / p]
    else:
        candidates = [default_cfg]

    config_path = None
    for cand in candidates:
        try:
            if cand.exists():
                config_path = cand
                break
        except Exception:
            continue
    if config_path is None:
        # Final fallback: default inside base_dir/configs (relative to main.py)
        if default_cfg.exists():
            config_path = default_cfg
        else:
            print("\n❌ Augmentation config not found at any expected location.")
            print(f"   Tried: {', '.join(str(c) for c in candidates + [default_cfg])}")
            print("   - Create it or pass --config to an existing YAML")
            return None
    # Load YAML
    try:
        import yaml  # lazy import
        with config_path.open("r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
    except Exception as e:
        print(f"\n❌ Failed to load augmentation config: {config_path}")
        print(f"   Error: {e}")
        return None
    if "modes" not in cfg:
        print("\n❌ Invalid config: missing 'modes' key")
        print("   - Ensure your YAML has top-level keys and 'modes' section")
        return None
    return (data_paths, config_path, cfg)


def print_augment_config(cfg):
    """Display augmentation configuration in a nice formatted way."""
    print("\n🎨 AUGMENTATION CONFIGURATION")
    print("=" * 50)
    
    # General settings
    print("📋 General Settings:")
    general_keys = ['output', 'in_place', 'seed', 'keep_original', 'counts', 'multiplier', 'prob_default', 'deterministic', 'use_cuda', 'cuda_device_id']
    for key in general_keys:
        if key in cfg:
            value = cfg[key]
            if value is not None:
                print(f"   {key:12}: {value}")
    
    # Modes section
    modes = cfg.get('modes', {})
    if modes:
        print("\n🔧 Augmentation Modes:")
        enabled_modes = []
        disabled_modes = []
        
        for mode_name, mode_config in modes.items():
            if isinstance(mode_config, dict):
                is_enabled = mode_config.get('enabled', False)
                if is_enabled:
                    enabled_modes.append((mode_name, mode_config))
                else:
                    disabled_modes.append(mode_name)
            else:
                # Simple boolean config
                if mode_config:
                    enabled_modes.append((mode_name, {}))
                else:
                    disabled_modes.append(mode_name)
        
        # Show enabled modes with details
        if enabled_modes:
            print("   ✅ ENABLED:")
            for mode_name, mode_config in enabled_modes:
                prob = mode_config.get('prob', cfg.get('prob_default', 'default'))
                print(f"      • {mode_name:12} (prob: {prob})", end="")
                
                # Add specific parameters for each mode
                params = []
                if 'max_deg' in mode_config:
                    params.append(f"max_deg: {mode_config['max_deg']}")
                if 'max_frac' in mode_config:
                    params.append(f"max_frac: {mode_config['max_frac']}")
                if 'range' in mode_config:
                    params.append(f"range: {mode_config['range']}")
                if 'ksize' in mode_config:
                    params.append(f"ksize: {mode_config['ksize']}")
                if 'h' in mode_config or 's' in mode_config or 'v' in mode_config:
                    hsv_params = []
                    for hsv_key in ['h', 's', 'v']:
                        if hsv_key in mode_config:
                            hsv_params.append(f"{hsv_key}: {mode_config[hsv_key]}")
                    if hsv_params:
                        params.append(f"HSV({', '.join(hsv_params)})")
                
                if params:
                    print(f" - {', '.join(params)}")
                else:
                    print()
        
        # Show disabled modes
        if disabled_modes:
            print("   ❌ DISABLED:")
            for mode_name in disabled_modes:
                print(f"      • {mode_name}")
    
    print("=" * 50)
