import torch
from model import PointCloudNet
from depth_estimator import MiDaSDepthEstimator
import numpy as np

def test_model_architecture():
    """Test the updated model architecture"""
    print("Testing model architecture...")
    
    # Create model
    model = PointCloudNet(num_views=1, point_cloud_size=1024, num_heads=4, dim_feedforward=2048)
    model.eval()
    
    # Create dummy inputs
    batch_size = 1
    rgb_input = torch.randn(batch_size, 1, 3, 224, 224)  # RGB input
    depth_input = torch.randn(batch_size, 1, 1, 224, 224)  # Depth input
    
    print(f"RGB input shape: {rgb_input.shape}")
    print(f"Depth input shape: {depth_input.shape}")
    
    # Test RGB-only forward pass
    with torch.no_grad():
        output_rgb_only = model(rgb_input)
        print(f"RGB-only output shape: {output_rgb_only.shape}")
        print(f"RGB-only output range: [{output_rgb_only.min().item():.4f}, {output_rgb_only.max().item():.4f}]")
    
    # Test RGB + Depth forward pass
    with torch.no_grad():
        output_with_depth = model(rgb_input, depth_input)
        print(f"RGB+Depth output shape: {output_with_depth.shape}")
        print(f"RGB+Depth output range: [{output_with_depth.min().item():.4f}, {output_with_depth.max().item():.4f}]")
    
    # Check if outputs are different (they should be)
    diff = torch.abs(output_rgb_only - output_with_depth).mean().item()
    print(f"Difference between RGB-only and RGB+Depth outputs: {diff:.6f}")
    
    if diff > 1e-6:
        print("✓ Model architecture is working correctly - outputs differ with depth input")
    else:
        print("⚠ Warning: Outputs are identical - depth input may not be affecting the model")
    
    return model

def test_depth_estimator():
    """Test the depth estimator"""
    print("\nTesting depth estimator...")
    
    try:
        depth_estimator = MiDaSDepthEstimator()
        print("✓ Depth estimator initialized successfully")
        
        # Create a dummy image (you can replace this with a real image path)
        dummy_depth = np.random.rand(224, 224)
        depth_features = depth_estimator.get_depth_features(dummy_depth)
        print(f"✓ Depth features shape: {depth_features.shape}")
        
        return depth_estimator
    except Exception as e:
        print(f"✗ Depth estimator failed: {e}")
        return None

def test_model_loading():
    """Test loading the pretrained model"""
    print("\nTesting model loading...")
    
    try:
        model = PointCloudNet(num_views=1, point_cloud_size=1024, num_heads=4, dim_feedforward=2048)
        checkpoint = torch.load("trained_models/pc1024_three.pth", map_location='cpu')
        
        # Check what keys are in the checkpoint
        print(f"Checkpoint keys: {list(checkpoint.keys())}")
        if 'model' in checkpoint:
            print(f"Model state dict keys: {list(checkpoint['model'].keys())[:10]}...")  # Show first 10 keys
        
        # Try to load the model
        model.load_state_dict(checkpoint["model"])
        print("✓ Model loaded successfully")
        
        return model
    except Exception as e:
        print(f"✗ Model loading failed: {e}")
        return None

if __name__ == "__main__":
    print("=== RGB2Point Model Testing ===\n")
    
    # Test 1: Model architecture
    model = test_model_architecture()
    
    # Test 2: Depth estimator
    depth_estimator = test_depth_estimator()
    
    # Test 3: Model loading
    loaded_model = test_model_loading()
    
    print("\n=== Test Summary ===")
    if model is not None:
        print("✓ Model architecture test passed")
    else:
        print("✗ Model architecture test failed")
    
    if depth_estimator is not None:
        print("✓ Depth estimator test passed")
    else:
        print("✗ Depth estimator test failed")
    
    if loaded_model is not None:
        print("✓ Model loading test passed")
    else:
        print("✗ Model loading test failed") 