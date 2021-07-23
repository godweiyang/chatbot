## install
```shell
git clone https://github.com/google/sentencepiece.git & cd sentencepiece
mkdir build & cd build
cmake ..
make -j $(nproc)
sudo make install
sudo ldconfig -v

pip3 install neurst
```

## sentencepiece
```shell
spm_train --input=./data/train/train.src,./data/train/train.trg \
    --model_prefix=./data/spm \
    --vocab_size=32000 \
    --character_coverage=0.9995
```

## tfrecords
```shell
python3 -m neurst.cli.create_tfrecords \
    --config_paths configs/task_args.yml \
    --dataset ParallelTextDataset \
    --src_file ./data/train/train.src \
    --trg_file ./data/train/train.trg \
    --processor_id 0 \
    --num_processors 1 \
    --num_output_shards 32 \
    --output_range_begin 0 \
    --output_range_end 32 \
    --output_template ./data/tfrecords/train.tfrecords-%5.5d-of-%5.5d
```

## train & valid
```shell
python3 -m neurst.cli.run_exp \
    --entry trainer \
    --task translation \
    --hparams_set transformer_big \
    --model_dir ./models/transformer_big \
    --config_paths ./configs/task_args.yml,./configs/train_args.yml,./configs/valid_args.yml
    --distribution_strategy horovod \
    --enable_xla
```

## predict
```shell
python3 -m neurst.cli.run_exp \
    --entry predict \
    --model_dir ./models/transformer_big \
    --config_paths ./configs/predict_args.yml \
    --output output.txt
```
