# 智能聊天机器人，你的AI女友

只用训练半天，你就可以得到一个属于你的聊天机器人，我把它命名为“杨超越”。随你提问什么，她都能对答如流！

首先来看看回复的效果怎么样：

```text
聊天开始！（按q退出）
我：很高兴认识你
杨超越：我也很开心哦
我：我喜欢你
杨超越：我也喜欢你
我：做我女朋友好不好？
杨超越：哈哈,可以呢!
我：我会一直支持超越妹妹的！
杨超越：谢谢我们会一直努力的
我：我什么时候有女朋友？
杨超越：女朋友。。
我：我什么时候脱单？
杨超越：脱你妹啊!!
我：q
聊天结束！
```

可以看到超越妹妹的回复还是非常流畅的，那她究竟是怎么诞生的呢？

## 快速体验:yum:
我提供了训练好的词表和模型文件，分别是`demo/spm.model`和`demo/model.pb`。由于大小限制，我这里只提供了训练好的小模型，词表大小也只有10k。如果你想要更好的回复效果，请参照后续教程自己训练一个Transformer-big模型，同时词表扩大到32k。

只需要直接运行下面命令就能开始聊天：

```shell
pip3 install lightseq sentencepiece

python3 chat.py \
    --spm_model ./demo/spm.model \
    --model_file ./demo/model.pb
```

## 介绍
这里我才用的是网上公开的小黄鸡聊天语料，大概有100万条左右，但是质量不是很高，都放在了`data`目录下。

模型采用标准的Transformer-big模型，输入你的提问句子，预测超越妹妹回复的句子，`config`目录下是训练和预测的配置文件。

模型训练采用NeurST训练库，主要基于TensorFlow，也支持PyTorch训练。模型快速推理采用LightSeq，可加速推理10倍以上，同时还能加速NeurST的训练，最高加速3倍。两者都是字节跳动AI Lab自研的，都已开源。

## 安装环境
我们需要安装三样东西：
* SentencePiece的命令行版本和python版本，用来对句子进行分词。
* NeurST深度学习训练库，用来训练Transformer模型。
* LightSeq，用来加速模型推理。

安装命令都很简单：

```shell
git clone https://github.com/google/sentencepiece.git && cd sentencepiece
mkdir build && cd build
cmake .. && make -j $(nproc) && sudo make install
sudo ldconfig -v

pip3 install lightseq neurst sentencepiece
```

## 开始养成
### 生成词表
首先我们需要从训练语料库中抽取出词表，为了方便，直接用SentencePiece来分词，生成大小为32k的词表。

```shell
spm_train --input=./data/train/train.src,./data/train/train.trg \
    --model_prefix=./data/spm \
    --vocab_size=32000 \
    --character_coverage=0.9995
```

这里需要指定训练语料路径`--input`、词表保存的路径前缀`--model_prefix`和词表大小`--vocab_size`。运行结束后会在`data`目录下生成`spm.model`和`spm.vocab`两个词表文件。一个是训练好的分词模型，一个是词表。


### 生成TFRecord
为了加快TensorFlow的训练速度，可以预先将训练语料用上面的词表处理成id，然后保存为TFRecord格式。这样模型训练时就可以直接读取id进行训练了，不需要做前面的分词操作。能大大加快训练速度，提升显卡利用率。

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

这里主要需要指定训练集的路径`--src_file`和`--trg_file`，其它参数保持默认即可。生成完毕后会在`data/tfrecords`下面生成32个二进制文件，这就是处理好的训练数据了。

### 模型训练
有了词表，有了处理好的训练数据，接下来就是训练模型了。这里开启了XLA优化，加快训练速度。

```shell
python3 -m neurst.cli.run_exp \
    --entry trainer \
    --task translation \
    --hparams_set transformer_big \
    --model_dir ./models \
    --config_paths ./configs/task_args.yml,./configs/train_args.yml,./configs/valid_args.yml \
    --enable_xla
```

这里需要指定的参数就是模型保存路径`model_dir`，其他都保持默认。训练好的模型会保存在`models`下，里面还细分为了`best`、`best_avg`等文件夹，用来存最好的模型、模型的平均值等等。

我在8张V100 32G显卡上训练了8个小时左右，如果你们自己训练的话还是比较耗时的。

### 模型预测
训练好的模型会保存在`models`目录下，然后我们就可以开始预测啦。

```shell
python3 -m neurst.cli.run_exp \
    --entry predict \
    --model_dir ./models \
    --config_paths ./configs/predict_args.yml \
    --output output.txt
```

但是这时候还没有交互功能，只能指定一个测试集文件，写在了模型预测的配置文件里`configs/predict_args.yml`。还可以指定`--output`，将回复结果输出到文件中。

**如果想直接体验交互式的对话聊天，可以跳过这一步。**

### 模型导出为PB格式
如果直接用TensorFlow进行推理的话，速度非常慢，你就会感觉你和超越妹妹之间存在延时。所以可以将训练得到的ckpt模型导出为PB格式，然后就可以用LightSeq训练加速引擎进行快速推理了。

```shell
python3 export/export.py \
    --model_dir ./models \
    --output_file ./models/model.pb \
    --generation_method topk \
    --topk 4 \
    --length_penalty 0.6 \
    --beam_size 4
```

这里需要指定模型路径`--model_dir`和导出PB文件的路径`--output_file`，其它参数保持默认。最后会得到`models/model.pb`这个PB文件。

### 开始交互式聊天！
有了PB模型文件，就可以和超越妹妹开始聊天啦！

```shell
python3 chat.py \
    --spm_model ./data/spm.model \
    --model_file ./models/model.pb
```

这里需要指定两个路径。一是最开始训练好的分词模型`--spm_model`，用来将你输入的句子变成整数id。二是`--model_file`，也就是上一步中的PB格式模型文件。

聊天过程中随时可以按q退出聊天，你每说一句话，超越妹妹就会回复你一句。

## 欢迎关注
这次用到的NeurST训练库和LightSeq加速库都非常好用，从上面使用教程中也可以看出，几乎不需要你写什么代码就能使用起来。

**NeurST训练库：**
[https://github.com/bytedance/neurst](https://github.com/bytedance/neurst)

**LightSeq加速库：**
[https://github.com/bytedance/lightseq](https://github.com/bytedance/lightseq)
