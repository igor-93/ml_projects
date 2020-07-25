# OpenCV Installation Guide:

[Official guide](https://docs.opencv.org/master/d7/d9f/tutorial_linux_install.html)
Below command is the adjusted version of [this](https://gist.github.com/raulqf/f42c718a658cddc16f9df07ecc627be7)

inside of opencv folder:
`mkdir build & cd build`

Cmake command to run in build folder:
```
cmake -D CMAKE_BUILD_TYPE=RELEASE \
        -D CMAKE_C_COMPILER=/usr/bin/gcc-8 \
-D CMAKE_INSTALL_PREFIX=/usr/local \
-D INSTALL_PYTHON_EXAMPLES=ON \
-D INSTALL_C_EXAMPLES=OFF \
-D WITH_TBB=ON \
-D WITH_CUDA=ON \
-D BUILD_opencv_cudacodec=OFF \
-D ENABLE_FAST_MATH=1 \
-D CUDA_FAST_MATH=1 \
-D WITH_CUBLAS=1 \
-D WITH_V4L=ON \
-D WITH_QT=OFF \
-D WITH_OPENGL=ON \
-D WITH_GSTREAMER=ON \
-D OPENCV_GENERATE_PKGCONFIG=ON \
-D OPENCV_PC_FILE_NAME=opencv.pc \
-D OPENCV_ENABLE_NONFREE=ON \
-D OPENCV_PYTHON3_INSTALL_PATH=/home/igor/.conda/envs/mlenv/lib/python3.7/site-packages \
-D OPENCV_EXTRA_MODULES_PATH=~/software/opencv_contrib/modules \
-D PYTHON3_EXECUTABLE=/home/igor/.conda/envs/mlenv/bin/python \
-D PYTHON3_INCLUDE_DIR=/home/igor/.conda/envs/mlenv/include/python3.7m \
-D PYTHON3_LIBRARY=/home/igor/.conda/envs/mlenv/lib \
-D PYTHON3_NUMPY_INCLUDE_DIRS=/home/igor/.conda/envs/mlenv/lib/python3.7/site-packages/numpy/core/include \
-D PYTHON_DEFAULT_EXECUTABLE=/home/igor/.conda/envs/mlenv/bin/python \
-D BUILD_EXAMPLES=ON ..```