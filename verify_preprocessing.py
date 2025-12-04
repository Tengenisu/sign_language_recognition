import numpy as np
import os
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm

def verify_npz_file(npz_path, verbose=False):
    """
    Verify a single NPZ file and return its properties.
    
    Args:
        npz_path: Path to the NPZ file
        verbose: Print detailed information
    
    Returns:
        Dictionary with file properties or None if invalid
    """
    try:
        data = np.load(npz_path)
        
        info = {
            'filename': os.path.basename(npz_path),
            'path': npz_path,
            'keys': list(data.keys()),
            'file_size_mb': os.path.getsize(npz_path) / (1024 * 1024),
            'arrays': {}
        }
        
        # Get info about each array in the NPZ
        for key in data.keys():
            arr = data[key]
            info['arrays'][key] = {
                'shape': arr.shape,
                'dtype': str(arr.dtype),
                'size_mb': arr.nbytes / (1024 * 1024),
                'min': float(np.min(arr)) if arr.size > 0 else None,
                'max': float(np.max(arr)) if arr.size > 0 else None,
                'mean': float(np.mean(arr)) if arr.size > 0 else None
            }
        
        data.close()
        
        if verbose:
            print(f"\n✓ {info['filename']}")
            print(f"  File size: {info['file_size_mb']:.2f} MB")
            print(f"  Keys: {', '.join(info['keys'])}")
            for key, arr_info in info['arrays'].items():
                print(f"\n  Array '{key}':")
                print(f"    Shape: {arr_info['shape']}")
                print(f"    Dtype: {arr_info['dtype']}")
                print(f"    Size: {arr_info['size_mb']:.2f} MB")
                print(f"    Range: [{arr_info['min']:.4f}, {arr_info['max']:.4f}]")
                print(f"    Mean: {arr_info['mean']:.4f}")
        
        return info
        
    except Exception as e:
        print(f"✗ Error loading {npz_path}: {str(e)}")
        return None

def verify_split(split_dir, expected_keys=None, check_shape_consistency=True, verbose=False):
    """
    Verify all NPZ files in a single split directory (train/test/val).
    
    Args:
        split_dir: Path to split directory
        expected_keys: List of expected keys in each NPZ file
        check_shape_consistency: Check if all files have consistent shapes
        verbose: Print detailed information for each file
    
    Returns:
        Tuple of (valid_files, invalid_files, all_infos)
    """
    npz_files = list(Path(split_dir).glob('*.npz'))
    
    if not npz_files:
        print(f"  No NPZ files found in {split_dir}")
        return [], [], []
    
    valid_files = []
    invalid_files = []
    all_infos = []
    
    # Progress bar for file verification
    for npz_file in tqdm(npz_files, desc=f"  Verifying", unit="file"):
        info = verify_npz_file(str(npz_file), verbose=verbose)
        
        if info is None:
            invalid_files.append(str(npz_file))
            continue
        
        # Check expected keys
        if expected_keys:
            missing_keys = set(expected_keys) - set(info['keys'])
            extra_keys = set(info['keys']) - set(expected_keys)
            
            if missing_keys or extra_keys:
                print(f"\n  ⚠ Warning for {info['filename']}:")
                if missing_keys:
                    print(f"    Missing keys: {missing_keys}")
                if extra_keys:
                    print(f"    Extra keys: {extra_keys}")
        
        valid_files.append(str(npz_file))
        all_infos.append(info)
    
    return valid_files, invalid_files, all_infos

def verify_autsl_dataset(base_dir='processed_autsl', expected_keys=['data'], 
                        check_shape_consistency=True, verbose=False):
    """
    Verify the entire AUTSL dataset with train/test/val splits.
    
    Args:
        base_dir: Base directory containing train/test/val folders
        expected_keys: List of expected keys in each NPZ file (default: ['data'])
        check_shape_consistency: Check if all files have consistent shapes within each split
        verbose: Print detailed information for each file
    
    Returns:
        Dictionary with verification results for each split
    """
    base_path = Path(base_dir)
    
    if not base_path.exists():
        print(f"Error: Directory {base_dir} does not exist")
        return None
    
    splits = ['train', 'test', 'val']
    results = {}
    
    print("=" * 70)
    print(f"VERIFYING AUTSL DATASET: {base_dir}")
    print("=" * 70)
    
    for split in splits:
        split_path = base_path / split
        
        if not split_path.exists():
            print(f"\n⚠ Warning: {split} directory does not exist")
            continue
        
        print(f"\n{'='*70}")
        print(f"SPLIT: {split.upper()}")
        print(f"{'='*70}")
        
        valid_files, invalid_files, all_infos = verify_split(
            split_path, 
            expected_keys=expected_keys,
            check_shape_consistency=check_shape_consistency,
            verbose=verbose
        )
        
        # Summary for this split
        print(f"\n{split.upper()} Summary:")
        print(f"  Valid files: {len(valid_files)}")
        print(f"  Invalid files: {len(invalid_files)}")
        
        if all_infos:
            total_size = sum(info['file_size_mb'] for info in all_infos)
            print(f"  Total size: {total_size:.2f} MB")
            
            # Check shape consistency within split
            if check_shape_consistency and len(all_infos) > 1:
                print(f"\n  Shape consistency check:")
                
                keys_set = set()
                for info in all_infos:
                    keys_set.update(info['keys'])
                
                for key in sorted(keys_set):
                    shapes = [info['arrays'][key]['shape'] for info in all_infos if key in info['arrays']]
                    unique_shapes = list(set(map(str, shapes)))
                    
                    if len(unique_shapes) > 1:
                        print(f"    ⚠ Array '{key}' has inconsistent shapes:")
                        for shape_str in unique_shapes:
                            count = sum(1 for s in shapes if str(s) == shape_str)
                            print(f"      {shape_str}: {count} files")
                    else:
                        print(f"    ✓ Array '{key}': {shapes[0]} (consistent)")
        
        if invalid_files:
            print(f"\n  Invalid files:")
            for f in invalid_files:
                print(f"    - {os.path.basename(f)}")
        
        results[split] = {
            'valid_files': valid_files,
            'invalid_files': invalid_files,
            'infos': all_infos
        }
    
    # Overall summary
    print(f"\n{'='*70}")
    print("OVERALL SUMMARY")
    print(f"{'='*70}")
    
    total_valid = sum(len(results[s]['valid_files']) for s in results)
    total_invalid = sum(len(results[s]['invalid_files']) for s in results)
    total_size = sum(
        sum(info['file_size_mb'] for info in results[s]['infos'])
        for s in results
    )
    
    print(f"Total valid files: {total_valid}")
    print(f"Total invalid files: {total_invalid}")
    print(f"Total dataset size: {total_size:.2f} MB ({total_size/1024:.2f} GB)")
    
    for split in splits:
        if split in results and results[split]['valid_files']:
            count = len(results[split]['valid_files'])
            print(f"  {split}: {count} files")
    
    return results


# Example usage
if __name__ == "__main__":
    # Basic verification for AUTSL sign language dataset
    # Expected: NPZ files with 'data' key containing (64, 56, 3) arrays
    # 64 frames, 56 landmarks (14 pose + 21 left hand + 21 right hand), 3 coords (x,y,z)
    
    results = verify_autsl_dataset(
        base_dir='processed_autsl',
        expected_keys=['data'],  # Your preprocessing saves data under 'data' key
        check_shape_consistency=True,
        verbose=False  # Set to True for detailed per-file output
    )
    
    # Access results programmatically
    if results:
        print(f"\n{'='*70}")
        print("DETAILED BREAKDOWN")
        print(f"{'='*70}")
        for split in ['train', 'test', 'val']:
            if split in results and results[split]['valid_files']:
                print(f"\n{split.upper()}:")
                print(f"  Files: {len(results[split]['valid_files'])}")
                
                # Check expected shape
                first_info = results[split]['infos'][0]
                data_shape = first_info['arrays']['data']['shape']
                print(f"  Expected shape: (64, 56, 3)")
                print(f"  Actual shape: {data_shape}")
                
                if data_shape == (64, 56, 3):
                    print(f"  ✓ Shape is correct!")
                else:
                    print(f"  ⚠ Shape mismatch!")