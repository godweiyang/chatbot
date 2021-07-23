## 安装环境
需要安装SentencePiece的命令行版本和python版本，还需要安装NeurST来训练模型，安装LightSeq来加速模型推理。

```shell
git clone https://github.com/google/sentencepiece.git & cd sentencepiece
mkdir build & cd build
cmake ..
make -j $(nproc)
sudo make install
sudo ldconfig -v

pip3 install lightseq neurst sentencepiece
```

## 生成词表
这里使用的是SentencePiece来生成大小为32k的词表。

```shell
spm_train --input=./data/train/train.src,./data/train/train.trg \
    --model_prefix=./data/spm \
    --vocab_size=32000 \
    --character_coverage=0.9995
```

## 生成TFRecord
这是为了加快训练时数据处理的速度。

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

## 模型训练
这里开启了XLA优化，使用Horovod分布式训练。如果报错，去掉这两行。

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

## 模型预测
这里需要指定测试集，而不是用户交互输入。这一步可以跳过，直接进行下面步骤。

```shell
python3 -m neurst.cli.run_exp \
    --entry predict \
    --model_dir ./models/transformer_big \
    --config_paths ./configs/predict_args.yml \
    --output output.txt
```

## 模型导出为PB格式
这是为了后续导入到LightSeq中进行推理加速。

```shell
python3 export/export.py \
    --model_dir ./models/transformer_big \
    --output_file ./models/transformer_big/model.pb \
    --beam_size 4 \
    --length_penalty 0.6
```

## 开始交互式聊天！

```shell
python3 chat.py \
    --spm_model ./data/spm.model \
    --model_file ./models/transformer_big/model.pb
```
