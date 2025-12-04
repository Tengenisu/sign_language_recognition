"""
check_dataset_joints.py - Verify actual number of joints in your dataset
"""

import torch
from dataset import create_dataloader

def check_joints():
    print("="*60)
    print("CHECKING DATASET JOINT COUNT")
    print("="*60)
    
    # Load a sample from training data
    train_loader = create_dataloader(
        'processed_autsl', 
        'train', 
        batch_size=1, 
        num_workers=0, 
        augment=False
    )
    
    sequences, labels, joint_types = next(iter(train_loader))
    
    print(f"\nDataloader returns:")
    print(f"  sequences shape: {sequences.shape}")
    print(f"  labels shape: {labels.shape}")
    print(f"  joint_types shape: {joint_types.shape}")
    
    B, T, J, F = sequences.shape
    
    print(f"\nParsed dimensions:")
    print(f"  Batch size (B): {B}")
    print(f"  Sequence length (T): {T}")
    print(f"  Number of joints (J): {J}")
    print(f"  Features per joint (F): {F}")
    
    print(f"\n{'='*60}")
    print(f"ACTUAL NUMBER OF JOINTS: {J}")
    print(f"{'='*60}")
    
    # Check if there's any preprocessing that reduces joints
    print("\nChecking joint_types:")
    print(f"  joint_types shape: {joint_types.shape}")
    print(f"  Unique joint types: {torch.unique(joint_types)}")
    print(f"  Number of unique joints: {len(torch.unique(joint_types))}")
    
    return J

if __name__ == '__main__':
    actual_joints = check_joints()
    print(f"\n✅ Your dataset has {actual_joints} joints per frame")
    print(f"\n⚠️  You need to train with num_joints={actual_joints}!")