import argparse

import sentencepiece as spm
import lightseq.inference as lsi


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--spm_model", type=str, default="./data/spm.model")
    parser.add_argument(
        "--model_file", type=str, default="./models/transformer_big/model.pb"
    )
    args = parser.parse_args()

    s = spm.SentencePieceProcessor()
    s.Init(model_file=args.spm_model)
    model = lsi.Transformer(args.model_file, 128)
    eos_id = s.vocab_size() + 2

    print("聊天开始！（按q退出）")
    while True:
        text = str(input("我："))
        if text == "q":
            break
        ids = [s.Encode(text) + [eos_id]]
        res = model.infer(ids)[0][0][0].tolist()
        while eos_id in res:
            res.remove(eos_id)
        res = s.Decode(res)
        print("杨超越：" + res)
    print("聊天结束！")
