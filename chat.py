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

    try:
        print("聊天开始！")
        while True:
            text = str(input())
            ids = [s.Encode(text) + [eos_id]]
            res = list(model.infer(ids)[0][0][0][:-1])
            for i in range(len(res)):
                if res[i] == eos_id:
                    res = res[:i]
            print(res)
            res = s.Decode(res)
            print(res)
    except:
        print("聊天结束！")
