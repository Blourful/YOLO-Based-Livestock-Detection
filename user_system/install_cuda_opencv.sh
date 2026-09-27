#!/bin/bash

# Installation script for CUDA-enabled OpenCV
# This script installs OpenCV with CUDA support for GPU acceleration

echo "🎮 Installing CUDA-enabled OpenCV..."

# Remove existing OpenCV installations
echo "📦 Removing existing OpenCV installations..."
pip uninstall -y opencv-python opencv-contrib-python opencv-python-headless opencv-contrib-python-headless

# Install CUDA-enabled OpenCV (headless version)
echo "⚡ Installing opencv-contrib-python-headless with CUDA support..."
pip install opencv-contrib-python-headless>=4.8.0

# Verify installation
echo "🔍 Verifying CUDA support..."
python -c "
import cv2
print(f'OpenCV version: {cv2.__version__}')
try:
    cuda_devices = cv2.cuda.getCudaEnabledDeviceCount()
    print(f'CUDA devices detected: {cuda_devices}')
    if cuda_devices > 0:
        print('✅ CUDA support is available!')
        # Test basic CUDA operation
        try:
            test_mat = cv2.cuda_GpuMat()
            print('✅ CUDA operations are functional!')
        except Exception as e:
            print(f'⚠️  CUDA operations limited: {e}')
    else:
        print('❌ No CUDA devices detected (CPU fallback will be used)')
        print('💡 Ensure you have:')
        print('   - CUDA-capable GPU')
        print('   - NVIDIA drivers installed')
        print('   - CUDA toolkit installed')
except AttributeError:
    print('❌ CUDA module not available in this OpenCV build')
    print('💡 Try building OpenCV from source with CUDA flags')
except Exception as e:
    print(f'❌ CUDA detection failed: {e}')
"

echo ""
echo "🎯 Installation complete!"
echo "💡 If CUDA is not detected:"
echo "   1. Install NVIDIA drivers"
echo "   2. Install CUDA toolkit (11.8 or 12.x)"
echo "   3. Restart system"
echo "   4. Run this script again"
echo ""
echo "🚀 For guaranteed CUDA support, consider:"
echo "   - Building OpenCV from source with CUDA flags"
echo "   - Using Docker with nvidia/cuda base images"
echo "   - Using cloud instances with pre-configured CUDA"
