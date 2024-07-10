set -e

MODEL_NAME=phiva-4b-hf
QUANTIZATION=q4f16_1

# 1. Download weights from HF
git lfs install
git clone https://huggingface.co/nota-ai/$MODEL_NAME \
          ./dist/models/$MODEL_NAME
          
# 2. convert weights
mlc_llm convert_weight ./dist/models/$MODEL_NAME/ --quantization $QUANTIZATION -o dist/$MODEL_NAME-$QUANTIZATION-MLC/

# 3. create mlc-chat-config.json
mlc_llm gen_config ./dist/models/$MODEL_NAME/ --quantization $QUANTIZATION \
  --conv-template phiva --context-window-size 768 -o dist/${MODEL_NAME}-${QUANTIZATION}-MLC/

# 4. compile model library with specification in mlc-chat-config.json
mlc_llm compile ./dist/${MODEL_NAME}-${QUANTIZATION}-MLC/mlc-chat-config.json \
    --device android -o ./dist/${MODEL_NAME}-${QUANTIZATION}-MLC/${MODEL_NAME}-${QUANTIZATION}-android.tar

# # 5. Generate model execution logics
cd android/library
chmod +x prepare_libs.sh
bash prepare_libs.sh