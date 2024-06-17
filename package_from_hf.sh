set -e

MODEL_NAME=llava-1.5-7b-hf
QUANTIZATION=q4f16_1

# 1. Download converted weights from HF
git lfs install
git clone https://huggingface.co/jykim310/${MODEL_NAME}-${QUANTIZATION}-MLC \
          ./dist/${MODEL_NAME}-${QUANTIZATION}-MLC
          
# 2. compile model library with specification in mlc-chat-config.json
mlc_llm compile ./dist/${MODEL_NAME}-${QUANTIZATION}-MLC/mlc-chat-config.json \
    --device android -o ./dist/${MODEL_NAME}-${QUANTIZATION}-MLC/${MODEL_NAME}-${QUANTIZATION}-android.tar

# 3. Generate model execution logics
cd android/library
chmod +x prepare_libs.sh
bash prepare_libs.sh