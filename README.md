# MLC-Nota

## Introduction
This repository is crafted with reference to the implementations of [mlc_llm](https://github.com/mlc-ai/mlc-llm) and [mlc_imp](https://github.com/MILVLG/mlc-imp). It supports the conversion of models such as Llama, Gemma, and also **LLaVA**, and includes the necessary implementations for processing these models. Future updates will include support for a broader range of foundational models.

## Demo
You can get the apk file in [Google Drive](https://drive.google.com/file/d/1OeIHYpr44zZNk-N7n18mhUeB3v6FQJrB/view?usp=sharing) link.
  
<img src="https://github.com/nota-github/mlc_mobile_fm/assets/86578246/dc240362-bc5f-4b68-b2fe-9690e14d268e" width="300px">


## Get Started

1. Please follow the [official link](https://llm.mlc.ai/docs/deploy/android.html#prerequisite) and install the prerequisites(Rust, Android Studio, JDK).

2. Clone the repo
```bash
git clone --recursive https://github.com/nota-github/mlc_mobile_fm.git
cd mlc-nota
```

3. Add environment variables at `~/.zshrc`
```
export TVM_HOME="absolute/path/to/mlc_mobile_fm/3rdparty/tvm"
export PYTHONPATH=$TVM_HOME/python:${PYTHONPATH}
```

4. Create virtual environment
```bash
conda env remove -n mlc-chat-venv -y
conda create -n mlc-chat-venv -c conda-forge \
    "llvmdev>=15" \
    "cmake>=3.24" \
    git \
    python=3.11 -y
conda activate mlc-chat-venv
```

5. Build TVM from scratch
```bash
cd 3rdparty/tvm
mkdir build && cd build
cp ../cmake/config.cmake .
echo "set(USE_LLVM \"llvm-config --ignore-libllvm --link-static\")" >> config.cmake
echo "set(HIDE_PRIVATE_SYMBOLS ON)" >> config.cmake
echo "set(CMAKE_BUILD_TYPE RelWithDebInfo)" >> config.cmake
echo "set(USE_METAL  ON)" >> config.cmake
cmake .. && cmake --build . --parallel 4 && cd ../../../
```
For more detailed information about the options, please refer to the *Option 2. Build from source - Details* section of the provided [link](https://llm.mlc.ai/docs/install/tvm.html).

6. Build MLC LLM module from scratch
```bash
mkdir build && cd build
python ../cmake/gen_cmake_config.py
cmake .. && cmake --build . --parallel 4 && cd ..
cd python
pip install -e .
cd ..
```

7. Package the model for android
```bash
chmod +x package_from_scratch.sh
bash package_from_scratch.sh
```

8. Open `Android` directory in Android Studio and run the app.

