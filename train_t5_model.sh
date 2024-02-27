## Setup
# Check that your disk is visible and get its name
#lsblk
# Mount disk in the data folder
#sudo mkfs.ext4 -m 0 -E lazy_itable_init=0,lazy_journal_init=0,discard /dev/sdb
#sudo mkdir -p data2
#sudo mount -o discard,defaults /dev/sdb data2
#sudo chmod a+w data2

# Permanently export the disk as the new path for HF dataset caching
export HF_HOME=~/t5/data/misc
export HF_DATASETS_CACHE=~/t5/data/datasets
export TRANSFORMERS_CACHE=~/t5/data/models

sudo apt update -y
sudo apt-get install python3.8-venv -y
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install "jax[tpu]>=0.2.16" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
pip install -r requirements.txt
sudo apt-get install git-lfs
#git lfs install

# Train Tokenizer
export HF_PROJECT="t5-base-it"

# Variables for training the tokenizer and creating the config
export VOCAB_SIZE="31000"
export N_INPUT_SENTENCES="100000" # Num of sentences to train the tokenizer
export DATASET_OSCAR="oscar" # Name of the dataset in the Huggingface Hub
export DATASET_WIKIPEDIA="wikipedia" # Name of the dataset in the Huggingface Hub
export DATASET_CONFIG_OSCAR="unshuffled_deduplicated_de" # Config of the dataset in the Huggingface Hub
export DATASET_CONFIG_WIKIPEDIA="20220301.de" # Config of the dataset in the Huggingface Hub
export DATASET_SPLIT="train" # Split to use for training tokenizer and model
export TEXT_FIELD="text" # Field containing the text to be used for training
export CONFIG_TYPE="GermanT5/t5-efficient-gc4-german-base-nl36" # Config that our model will use
export MODEL_PATH="data/${HF_PROJECT}" # Path to the model, e.g. here inside the mount

# Create the tokenizer and the config
python3 create_tokenizer_cfg.py \
    --model_dir $MODEL_PATH \
    --dataset $DATASET_OSCAR \
    --dataset_config $DATASET_CONFIG_OSCAR \
    --dataset_split $DATASET_SPLIT \
    --text_field $TEXT_FIELD \
    --vocab_size $VOCAB_SIZE \
    --input_sentence_size $N_INPUT_SENTENCES \
    --config_type $CONFIG_TYPE\
    --cache_dir $HF_DATASETS_CACHE

# Pretraining
python3 run_t5_mlm_flax.py \
    --output_dir=$MODEL_PATH \
    --model_name_or_path=$MODEL_PATH2 \
    --tokenizer_name=$MODEL_PATH2 \
    --preprocessing_num_workers="96" \
    --train_file="data/verwaltung/vector-data-bw.jsonl" \
    --max_seq_length="512" \
    --per_device_train_batch_size="16" \
    --per_device_eval_batch_size="16" \
    --adafactor \
    --do_train=True \
    --do_eval=True \
    --learning_rate="0.004" \
    --weight_decay="0.001" \
    --warmup_steps="100" \
    --overwrite_output_dir \
    --logging_steps="1000" \
    --save_steps="1000" \
    --eval_steps="300" \
    --max_steps=4000000 \
    --validation_split_count="15000" \
    --cache_dir=$HF_DATASETS_CACHE \
    --num_train_epochs=10

# Setup For Finetuning
pip install -r requirements_finetuning.txt

# Finetuning on German Quad
python3 finetune_t5_quad_closed_book.py --model_path='data/t5-german' --num_train_epochs=3 --batch_size=8 --push_to_hub=False --output_dir='dwolpers/german_T5_Large_Quad'
python3 finetune_t5_quad2_closed_book.py --model_path='dwolpers/german_T5_Large_Quad' --num_train_epochs=3 --batch_size=8 --push_to_hub=False --output_dir='dwolpers/german_T5_Large_Closed'
python3 finetune_t5_quad_open_book.py --model_path='dwolpers/german_T5_Large_Quad' --num_train_epochs=3 --batch_size=8 --push_to_hub=False --output_dir='dwolpers/german_T5_Large_Open'
python3 finetune_t5_classification.py  --model_path='dwolpers/german_T5_Large_Closed' --num_train_epochs=3 --batch_size=8 --push_to_hub=False --output_dir='dwolpers/german_T5_Large_Closed_Class'
python3 finetune_t5_classification.py  --model_path='dwolpers/german_T5_Large_Open' --num_train_epochs=3 --batch_size=8 --push_to_hub=False --output_dir='dwolpers/german_T5_Large_Open_Class'