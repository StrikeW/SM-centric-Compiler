#!/usr/bin/env bash

ROOT=$(pwd)

sudo apt update
sudo apt install -y build-essential
sudo apt install -y subversion cmake ninja-build

mkdir llvm && cd llvm
svn co http://llvm.org/svn/llvm-project/llvm/trunk llvm
cd llvm/tools
svn co http://llvm.org/svn/llvm-project/cfe/trunk clang
cd clang/tools
svn co http://llvm.org/svn/llvm-project/clang-tools-extra/trunk extra


cd ${ROOT}/llvm && mkdir build && cd build
cmake -G Ninja -DCMAKE_BUILD_TYPE=Release ../llvm

ninja
ninja install



