#!/bin/bash
set -e

# === CONFIG ===
OPENPOSE_DIR=~/openpose
PYTHON_BIN=$(which python3)
NUM_THREADS=$(nproc)

echo "🚀 Installing OpenPose with CUDA-only (no cuDNN)..."

# === Update and install system dependencies ===
sudo apt update && sudo apt install -y \
    build-essential cmake git libopencv-dev \
    libopenblas-dev liblapack-dev \
    libatlas-base-dev libboost-all-dev \
    libhdf5-serial-dev libprotobuf-dev protobuf-compiler \
    python3-dev python3-pip \
    libgoogle-glog-dev libgflags-dev \
    liblmdb-dev libsnappy-dev libopencv-core-dev \
    libopencv-highgui-dev libopencv-imgproc-dev \
    libopencv-video-dev libopencv-calib3d-dev \
    libopencv-objdetect-dev libopencv-contrib-dev \
    libboost-python-dev libyaml-cpp-dev libgtk-3-dev

# === Python dependencies from requirements.txt ===
echo "📦 Installing Python packages from requirements.txt..."
pip3 install -r requirements.txt

# === Clone OpenPose ===
cd ~
if [ ! -d "$OPENPOSE_DIR" ]; then
    echo "📥 Cloning OpenPose..."
    git clone https://github.com/CMU-Perceptual-Computing-Lab/openpose.git
fi
cd $OPENPOSE_DIR

# === Patch io.cpp for newer Protobuf compatibility ===
PATCH_FILE="$OPENPOSE_DIR/3rdparty/caffe/src/caffe/util/io.cpp"
if grep -q "SetTotalBytesLimit(kProtoReadBytesLimit, 536870912);" "$PATCH_FILE"; then
    echo "🔧 Patching io.cpp to support modern Protobuf..."
    sed -i 's/SetTotalBytesLimit(kProtoReadBytesLimit, 536870912)/SetTotalBytesLimit(kProtoReadBytesLimit)/g' "$PATCH_FILE"
fi

# === Build OpenPose ===
mkdir -p build && cd build

echo "⚙️ Configuring with CMake..."
cmake -DBUILD_PYTHON=ON \
      -DPYTHON_EXECUTABLE=$PYTHON_BIN \
      -DUSE_CUDNN=OFF \
      ..

echo "🔨 Compiling OpenPose..."
make -j$NUM_THREADS

# === Add Python API to path ===
BASHRC_UPDATE="export PYTHONPATH=\$PYTHONPATH:$OPENPOSE_DIR/build/python"
if ! grep -Fxq "$BASHRC_UPDATE" ~/.bashrc; then
    echo "$BASHRC_UPDATE" >> ~/.bashrc
    echo "🔧 PYTHONPATH updated in ~/.bashrc"
fi
export PYTHONPATH=$PYTHONPATH:$OPENPOSE_DIR/build/python

# === Test Python API ===
echo "🧪 Testing OpenPose Python API..."
cd $OPENPOSE_DIR/build/python/openpose
python3 -c 'import pyopenpose as op; print("✅ OpenPose Python API is working!")'

echo "🎉 Done! OpenPose with CUDA-only is fully installed and functional."

