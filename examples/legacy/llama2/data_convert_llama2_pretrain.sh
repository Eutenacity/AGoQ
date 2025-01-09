# 请按照您的真实环境修改 set_env.sh 路径
# source /usr/local/Ascend/ascend-toolkit/set_env.sh
# mkdir ./dataset

python ./preprocess_data.py \
   --input /workspace/project2/dataset/20231101.en \
   --tokenizer-name-or-path /workspace/project2/MindSpeed-LLM/Llama-2-7b-hf \
   --output-prefix ./dataset/enwiki \
   --workers 64 \
   --log-interval 10000  \
   --tokenizer-type PretrainedFromHF