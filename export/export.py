import tensorflow as tf
from absl import app, logging

import neurst.utils.flags_core as flags_core
from neurst.layers.decoders.transformer_decoder import TransformerDecoder
from neurst.layers.encoders.transformer_encoder import TransformerEncoder
from neurst.models.encoder_decoder_model import EncoderDecoderModel
from neurst.models.model import BaseModel
from neurst.models.transformer import Transformer
from neurst.utils.configurable import ModelConfigs
from neurst.utils.registry import REGISTRIED_CLS2ALIAS
from transformer_proto_utils import transfer_ckpt2pb

FLAG_LIST = [
    flags_core.Flag(
        "model_dir",
        dtype=flags_core.Flag.TYPE.STRING,
        help="The path to the well-trained checkpoint.",
    ),
    flags_core.Flag(
        "output_file",
        dtype=flags_core.Flag.TYPE.STRING,
        default=None,
        help="The path to saving the extracted weights.",
    ),
    flags_core.Flag(
        "generation_method",
        dtype=flags_core.Flag.TYPE.STRING,
        default="beam_search",
        choices=["beam_search", "topk", "topp"],
        help="The generation method: beam_search (default), topk or topp.",
    ),
    flags_core.Flag(
        "beam_size",
        dtype=flags_core.Flag.TYPE.INTEGER,
        default=None,
        help="The beam width of sequence generation for Transformer model.",
    ),
    flags_core.Flag(
        "length_penalty",
        dtype=flags_core.Flag.TYPE.FLOAT,
        default=-1,
        help="The length penalty of sequence generation for Transformer model.",
    ),
    flags_core.Flag(
        "max_seq_len",
        dtype=flags_core.Flag.TYPE.INTEGER,
        default=256,
        help="The maximum sequence length.",
    ),
    flags_core.Flag(
        "extra_decode_length",
        dtype=flags_core.Flag.TYPE.INTEGER,
        default=50,
        help="The extra decoding length of sequence generation for Transformer model.",
    ),
    flags_core.Flag(
        "topk",
        dtype=flags_core.Flag.TYPE.INTEGER,
        default=4,
        help="topk for sampling, only 1,2,4,...,32 are valid ",
    ),
    flags_core.Flag(
        "topp", dtype=flags_core.Flag.TYPE.FLOAT, default=0.75, help="topp for sampling"
    ),
    flags_core.Flag(
        "use_aligned_penalty",
        dtype=flags_core.Flag.TYPE.BOOLEAN,
        default=False,
        help="Whether to use aligned length penalty",
    ),
    flags_core.Flag(
        "align_len_ratio",
        dtype=flags_core.Flag.TYPE.FLOAT,
        default=1.0,
        help="The length ratio against source sequence",
    ),
    flags_core.Flag(
        "align_len_bias",
        dtype=flags_core.Flag.TYPE.FLOAT,
        default=3.0,
        help="The length bias",
    ),
    flags_core.Flag(
        "align_len_alpha",
        dtype=flags_core.Flag.TYPE.FLOAT,
        default=1.0,
        help="The exponent of length penalty",
    ),
]


def _is_standard_transformer(model_cls, model_params):
    if (
        model_cls
        in REGISTRIED_CLS2ALIAS["tf"][BaseModel.REGISTRY_NAME][Transformer.__name__]
    ):
        return True
    if (
        model_cls in REGISTRIED_CLS2ALIAS["tf"]["model"][EncoderDecoderModel.__name__]
        and model_params["encoder.class"]
        in REGISTRIED_CLS2ALIAS["tf"]["encoder"][TransformerEncoder.__name__]
        and model_params["decoder.class"]
        in REGISTRIED_CLS2ALIAS["tf"]["decoder"][TransformerDecoder.__name__]
    ):
        return True
    return False


def export_lightseq_model(model_dir, output_file, **generation_options):
    """Extracts weights from `model_dir` and saves to `output_file`."""
    assert model_dir, "Must provide `model_dir`."
    assert output_file, "Must provide `output_file`."
    if tf.io.gfile.isdir(output_file):
        raise ValueError("`output_file` should be a file path, not a directory.")
    logging.info(f"Extracting weights from {model_dir} to {output_file}")
    model_args = ModelConfigs.load(model_dir)
    if _is_standard_transformer(model_args["model.class"], model_args["model.params"]):
        gen_method = generation_options["generation_method"]
        logging.info(
            f"Extract and export Transformer weights using method: {gen_method}."
        )
        if gen_method == "beam_search":
            assert generation_options[
                "beam_size"
            ], "`beam_size` must be provided for exporting Transformer."
        transfer_ckpt2pb(
            output_file=output_file,
            model_dir=model_dir,
            model_params=model_args["model.params"],
            generation_method=gen_method,
            max_seq_len=generation_options["max_seq_len"],
            extra_decode_length=generation_options["extra_decode_length"],
            beam_size=generation_options["beam_size"],
            length_penalty=generation_options["length_penalty"],
            topk=generation_options["topk"],
            topp=generation_options["topp"],
            target_bos=model_args["task.params"].get("target_begin_of_sentence", "bos"),
            use_aligned_penalty=generation_options["use_aligned_penalty"],
            align_len_ratio=generation_options["align_len_ratio"],
            align_len_bias=generation_options["align_len_bias"],
            align_len_alpha=generation_options["align_len_alpha"],
        )
    else:
        raise NotImplementedError(
            "No matched proto for export. Now we only support standard Transformer."
        )


def _main(_):
    arg_parser = flags_core.define_flags(FLAG_LIST, with_config_file=True)
    args, remaining_argv = flags_core.intelligent_parse_flags(FLAG_LIST, arg_parser)
    flags_core.verbose_flags(FLAG_LIST, args, remaining_argv)
    export_lightseq_model(
        model_dir=args.pop("model_dir"), output_file=args.pop("output_file"), **args
    )


if __name__ == "__main__":
    logging.set_verbosity(logging.INFO)
    app.run(_main, argv=["pseudo.py"])
