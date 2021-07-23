import importlib
import logging
import math
import os
from collections import OrderedDict

import numpy
import tensorflow as tf

from neurst.utils.compat import wrapper_var_name
from neurst.utils.configurable import ModelConfigs

enc_layer_mapping_dict = OrderedDict(
    {
        "multihead_norm_scale": "self_attention_prepost_wrapper gamma",
        "multihead_norm_bias": "self_attention_prepost_wrapper beta",
        "multihead_project_kernel_qkv": "qkv_transform kernel",
        "multihead_project_bias_qkv": "qkv_transform bias",
        "multihead_project_kernel_output": "output_transform kernel",
        "multihead_project_bias_output": "output_transform bias",
        "ffn_norm_scale": "ffn_prepost_wrapper gamma",
        "ffn_norm_bias": "ffn_prepost_wrapper beta",
        "ffn_first_kernel": "dense1 kernel",
        "ffn_first_bias": "dense1 bias",
        "ffn_second_kernel": "dense2 kernel",
        "ffn_second_bias": "dense2 bias",
    }
)

dec_layer_mapping_dict = OrderedDict(
    {
        "self_norm_scale": "self_attention_prepost_wrapper gamma",
        "self_norm_bias": "self_attention_prepost_wrapper beta",
        "self_project_kernel_qkv": "self_attention qkv_transform kernel",
        "self_project_bias_qkv": "self_attention qkv_transform bias",
        "self_project_kernel_output": "self_attention_prepost_wrapper output_transform kernel",
        "self_project_bias_output": "self_attention_prepost_wrapper output_transform bias",
        "encdec_norm_scale": "encdec_attention_prepost_wrapper gamma",
        "encdec_norm_bias": "encdec_attention_prepost_wrapper beta",
        "encdec_project_kernel_q": "encdec_attention q_transform kernel",
        "encdec_project_bias_q": "encdec_attention q_transform bias",
        "encdec_project_kernel_output": "encdec_attention output_transform kernel",
        "encdec_project_bias_output": "encdec_attention output_transform bias",
        "ffn_norm_scale": "ffn_prepost_wrapper gamma",
        "ffn_norm_bias": "ffn_prepost_wrapper beta",
        "ffn_first_kernel": "dense1 kernel",
        "ffn_first_bias": "dense1 bias",
        "ffn_second_kernel": "dense2 kernel",
        "ffn_second_bias": "dense2 bias",
    }
)

src_emb_mapping_dict = OrderedDict(
    {
        "norm_scale": "TransformerEncoder gamma-ffn_prepost_wrapper self_attention_prepost_wrapper",
        "norm_bias": "TransformerEncoder beta-ffn_prepost_wrapper self_attention_prepost_wrapper",
    }
)

trg_emb_mapping_dict = OrderedDict(
    {
        "norm_scale": "TransformerDecoder gamma-self_attention_prepost_wrapper "
        "ffn_prepost_wrapper encdec_attention_prepost_wrapper",
        "norm_bias": "TransformerDecoder beta-self_attention_prepost_wrapper "
        "ffn_prepost_wrapper encdec_attention_prepost_wrapper",
        "shared_bias": "target_symbol_modality bias",
    }
)

shared_trg_emb_mapping_dict = OrderedDict(
    {
        "norm_scale": "TransformerDecoder gamma-self_attention_prepost_wrapper "
        "ffn_prepost_wrapper encdec_attention_prepost_wrapper",
        "norm_bias": "TransformerDecoder beta-self_attention_prepost_wrapper "
        "ffn_prepost_wrapper encdec_attention_prepost_wrapper",
        "shared_bias": "shared_symbol_modality bias",
    }
)


def _gather_encORdec_layer_tensor_name(tensor_names, pattern):
    layer_ids = set()
    layer_tensor_names = []
    for _ in range(1000):
        layer_tensor_names.append([])
    for tn in tensor_names:
        if pattern not in tn.split("/"):
            continue
        if "layer_" not in tn:
            continue
        lid = int(tn.split("layer_")[1].split("/")[0])
        layer_ids.add(lid)
        layer_tensor_names[lid].append(tn)
    layer_ids = sorted(list(layer_ids))
    required_ids = list(range(len(layer_ids)))
    assert (
        layer_ids and layer_ids == required_ids
    ), "layer_ids: %s shoule be equal to %s and not empty" % (layer_ids, required_ids)
    return layer_tensor_names[: len(layer_ids)]


def _check_rule(tensor_name, rule):
    if "Adam" in tensor_name or "adam" in tensor_name:
        return False
    assert isinstance(rule, str) and rule
    rule = rule.split("-")
    assert len(rule) < 3
    if len(rule) == 2:
        white, black = rule[0].split(" "), rule[1].split(" ")
    else:
        white, black = rule[0].split(" "), []
    for b in black:
        if b in tensor_name.split("/"):
            return False
    for w in white:
        if w not in tensor_name.split("/"):
            return False
    return True


def _fill_layer(tensor_names, name2var_dict, layer, mapping_dict):
    for proto_name, ckpt_rule in mapping_dict.items():
        expression = [
            ele for ele in ckpt_rule.split(":") if ele.startswith("expression_")
        ]
        assert len(expression) < 2
        expression = "" if not expression else expression[0].split("_")[1]
        ckpt_rule = [
            ele for ele in ckpt_rule.split(":") if not ele.startswith("expression_")
        ]
        target_tn = []
        for cr in ckpt_rule:
            tmp = []
            for tn in tensor_names:
                if _check_rule(tn, cr):
                    tmp.append(tn)
            # print(cr)
            # print(tmp)
            assert len(tmp) == 1
            target_tn.extend(tmp)
        target_tensor = [name2var_dict[name] for name in target_tn]
        tt = {}
        exec("tt['save'] = [ele%s for ele in target_tensor]" % expression)

        target_tensor = numpy.concatenate(tt["save"], axis=-1)
        logging.info(
            "ckpt_tensor_name: %s, proto_name: %s, shape: %s"
            % (target_tn, proto_name, target_tensor.shape)
        )
        exec("layer.%s[:]=target_tensor.flatten().tolist()" % proto_name)


def _get_encode_output_mapping_dict(dec_layer_num):
    encode_output_kernel_pattern = [
        "TransformerDecoder layer_{} encdec_attention kv_transform kernel".format(ele)
        for ele in range(dec_layer_num)
    ]
    encode_output_bias_pattern = [
        "TransformerDecoder layer_{} encdec_attention kv_transform bias".format(ele)
        for ele in range(dec_layer_num)
    ]
    return {
        "encode_output_project_kernel_kv": ":".join(encode_output_kernel_pattern),
        "encode_output_project_bias_kv": ":".join(encode_output_bias_pattern),
    }


def _get_position_encoding(length, hidden_size, min_timescale=1.0, max_timescale=1.0e4):
    """Return positional encoding.

    Calculates the position encoding as a mix of sine and cosine functions with
    geometrically increasing wavelengths.
    Defined and formulized in Attention is All You Need, section 3.5.

    Args:
      length: Sequence length.
      hidden_size: Size of the
      min_timescale: Minimum scale that will be applied at each position
      max_timescale: Maximum scale that will be applied at each position

    Returns:
      Tensor with shape [length, hidden_size]
    """
    position = tf.cast(tf.range(length), tf.float32)
    num_timescales = hidden_size // 2
    log_timescale_increment = math.log(float(max_timescale) / float(min_timescale)) / (
        tf.cast(num_timescales, tf.float32) - 1
    )
    inv_timescales = min_timescale * tf.exp(
        tf.cast(tf.range(num_timescales), tf.float32) * -log_timescale_increment
    )
    scaled_time = tf.expand_dims(position, 1) * tf.expand_dims(inv_timescales, 0)
    signal = tf.concat([tf.math.sin(scaled_time), tf.math.cos(scaled_time)], axis=1)
    return signal


def _gather_token_embedding(tensor_names, name2var_dict, tn_pattern):
    """use pattern to diff source and target."""
    target_tn = []
    for tn in tensor_names:
        if (tn_pattern in tn) and ("emb/weights" in tn or "shared/weights" in tn):
            target_tn.append(tn)
    # target_tn = sorted(target_tn, key=lambda x: int(x.split('_')[-1]))
    target_tensor = [name2var_dict[name] for name in target_tn]
    target_tensor = numpy.concatenate(target_tensor, axis=0)
    target_tensor = target_tensor * (target_tensor.shape[1] ** 0.5)
    logging.info(
        "token embedding shape is %s, scaled by %s"
        % (target_tensor.shape, target_tensor.shape[1] ** 0.5)
    )
    return target_tensor


def transfer_ckpt2pb(
    output_file,
    model_dir,
    model_params,
    generation_method,
    extra_decode_length=50,
    max_seq_len=256,
    beam_size=4,
    length_penalty=-1,
    topk=4,
    topp=0.75,
    target_bos="bos",
    use_aligned_penalty=False,
    align_len_ratio=1.0,
    align_len_bias=3.0,
    align_len_alpha=1.0,
):
    # create transformer proto object
    this_dir = os.path.dirname(__file__)
    transformer_proto_path = os.path.join(this_dir, "transformer.proto")
    logging.info("Using {}".format(transformer_proto_path))
    if not os.path.exists(os.path.join(this_dir, "transformer_pb2.py")):
        os.system(
            "protoc -I={} --python_out={} {}".format(
                this_dir, this_dir, transformer_proto_path
            )
        )
    this_package = __name__
    new_module = importlib.import_module("transformer_pb2", package=this_package)
    transformer = getattr(new_module, "Transformer")()
    # load var names
    var_name_shape_list = tf.train.list_variables(model_dir)
    name2var = dict()
    for name, _ in var_name_shape_list:
        name2var[wrapper_var_name(name)] = tf.train.load_variable(model_dir, name)
    var_name_list = list(name2var.keys())

    # fill in encoder weights
    enc_tensor_names = _gather_encORdec_layer_tensor_name(
        var_name_list, "TransformerEncoder"
    )
    for layer_tensor_names in enc_tensor_names:
        _fill_layer(
            layer_tensor_names,
            name2var,
            transformer.encoder_stack.add(),
            enc_layer_mapping_dict,
        )

    # fill in decoder weights
    dec_tensor_names = _gather_encORdec_layer_tensor_name(
        var_name_list, "TransformerDecoder"
    )
    for layer_tensor_names in dec_tensor_names:
        _fill_layer(
            layer_tensor_names,
            name2var,
            transformer.decoder_stack.add(),
            dec_layer_mapping_dict,
        )

    # fill in embedding weights
    _fill_layer(
        var_name_list, name2var, transformer.src_embedding, src_emb_mapping_dict
    )
    encode_output_mapping_dict = _get_encode_output_mapping_dict(len(dec_tensor_names))
    if model_params["modality.share_source_target_embedding"]:
        shared_trg_emb_mapping_dict.update(encode_output_mapping_dict)
        _fill_layer(
            var_name_list,
            name2var,
            transformer.trg_embedding,
            shared_trg_emb_mapping_dict,
        )
    else:
        trg_emb_mapping_dict.update(encode_output_mapping_dict)
        _fill_layer(
            var_name_list, name2var, transformer.trg_embedding, trg_emb_mapping_dict
        )

    # fill in position embedding weights
    pos_emb_tensor = _get_position_encoding(
        max_seq_len, len(transformer.src_embedding.norm_scale)
    )
    pos_emb_list = pos_emb_tensor.numpy().reshape([-1]).tolist()
    if isinstance(model_params["modality.source.timing"], dict) and model_params[
        "modality.source.timing"
    ].get("sinusoids_as_variable", False):
        for _n, _v in name2var.items():
            if "input_symbol_modality" in _n and "position_embeddings/weights" in _n:
                transformer.src_embedding.position_embedding[:] = _v.reshape(
                    [-1]
                ).tolist()
                break
    else:
        transformer.src_embedding.position_embedding[:] = pos_emb_list
    if isinstance(model_params["modality.target.timing"], dict) and model_params[
        "modality.target.timing"
    ].get("sinusoids_as_variable", False):
        for _n, _v in name2var.items():
            if "target_symbol_modality" in _n and "position_embeddings/weights" in _n:
                transformer.trg_embedding.position_embedding[:] = _v.reshape(
                    [-1]
                ).tolist()
                break
    else:
        transformer.trg_embedding.position_embedding[:] = pos_emb_list

    # gather token embedding
    if model_params["modality.share_source_target_embedding"]:
        src_tb = _gather_token_embedding(
            var_name_list, name2var, "shared_symbol_modality"
        )
        transformer.src_embedding.token_embedding[:] = src_tb.flatten().tolist()
        trg_tb = _gather_token_embedding(
            var_name_list, name2var, "shared_symbol_modality"
        )
        transformer.trg_embedding.token_embedding[:] = (
            trg_tb.transpose().flatten().tolist()
        )
    else:
        src_tb = _gather_token_embedding(
            var_name_list, name2var, "input_symbol_modality"
        )
        transformer.src_embedding.token_embedding[:] = src_tb.flatten().tolist()
        trg_tb = _gather_token_embedding(
            var_name_list, name2var, "target_symbol_modality"
        )
        transformer.trg_embedding.token_embedding[:] = (
            trg_tb.transpose().flatten().tolist()
        )

    # fill in conf
    if "encoder.num_attention_heads" in model_params:
        assert (
            model_params["encoder.num_attention_heads"]
            == model_params["decoder.num_attention_heads"]
        ), "Now only support Transformer with the same heads in encoder and decoder."
        transformer.model_conf.head_num = model_params["encoder.num_attention_heads"]
    elif "encoder.params" in model_params:
        assert (
            model_params["encoder.params"]["num_attention_heads"]
            == model_params["decoder.params"]["num_attention_heads"]
        ), "Now only support Transformer with the same heads in encoder and decoder."
        transformer.model_conf.head_num = model_params["encoder.params"][
            "num_attention_heads"
        ]
    else:
        raise NotImplementedError

    transformer.model_conf.beam_size = beam_size
    transformer.model_conf.length_penalty = length_penalty

    transformer.model_conf.extra_decode_length = extra_decode_length
    transformer.model_conf.src_padding_id = src_tb.shape[0]

    if target_bos == "bos":
        transformer.model_conf.trg_start_id = trg_tb.shape[0] - 2
    elif target_bos == "eos":  # compat with fairseq
        transformer.model_conf.trg_start_id = trg_tb.shape[0] - 1
    else:
        raise ValueError(f"Unknown value of target bos: {target_bos}")

    transformer.model_conf.sampling_method = generation_method
    transformer.model_conf.topk = topk
    transformer.model_conf.topp = topp
    transformer.model_conf.use_aligned_penalty = use_aligned_penalty
    transformer.model_conf.align_len_ratio = align_len_ratio
    transformer.model_conf.align_len_bias = align_len_bias
    transformer.model_conf.align_len_alpha = align_len_alpha

    if "lang_mask" in name2var:
        # For multi lingual
        logging.info("Load multi lingual model config...")
        transformer.model_conf.is_multilingual = True

        # Fill in target vocab mask
        lang_mask = name2var["lang_mask"]
        transformer.trg_embedding.trg_vocab_mask[:] = lang_mask.flatten().tolist()

        # Fill in language emb
        lang_emb_kernel = None
        for vname, var in name2var.items():
            if "position_emb_offset" in vname:
                lang_emb_kernel = name2var[vname].astype("float32")
        assert lang_emb_kernel is not None, "No lang emb kernel"

        lang_emb = (
            trg_tb[: lang_mask.shape[0], :] / float(trg_tb.shape[1] ** 0.5)
        ).astype("float32")
        src_lang_emb = numpy.dot(lang_emb, lang_emb_kernel)
        trg_lang_emb = (
            src_lang_emb + trg_tb[: lang_mask.shape[0], :] + pos_emb_tensor.numpy()[1]
        )

        transformer.src_embedding.lang_emb[:] = src_lang_emb.flatten().tolist()
        transformer.trg_embedding.lang_emb[:] = trg_lang_emb.flatten().tolist()

    with tf.io.gfile.GFile(output_file, "wb") as fout:
        fout.write(transformer.SerializeToString())

    transformer = getattr(new_module, "Transformer")()
    with tf.io.gfile.GFile(output_file, "rb") as fin:
        transformer.ParseFromString(fin.read())
    logging.info("============ pb model conf ============")
    logging.info(transformer.model_conf)


def transfer_pb2ckpt(pb_file, model_dir):
    # create transformer proto object
    this_dir = os.path.dirname(__file__)
    transformer_proto_path = os.path.join(this_dir, "transformer.proto")
    logging.info("Using {}".format(transformer_proto_path))
    os.system(
        "protoc -I={} --python_out={} {}".format(
            this_dir, this_dir, transformer_proto_path
        )
    )
    this_package = __name__[: __name__.rindex(".")]
    new_module = importlib.import_module(".transformer_pb2", package=this_package)
    transformer = getattr(new_module, "Transformer")()

    with open(pb_file, "rb") as f:
        transformer.ParseFromString(f.read())

    num_heads = transformer.model_conf.head_num
    dmodel = len(transformer.trg_embedding.norm_bias)
    model_vars = {}
    # target embedding
    name = "SequenceToSequence/target_symbol_modality_posenc_wrapper/target_symbol_modality/shared/"
    vocab_size = len(transformer.trg_embedding.token_embedding) // dmodel
    trg_emb = numpy.reshape(
        transformer.trg_embedding.token_embedding, [dmodel, vocab_size]
    )
    trg_emb /= dmodel ** 0.5
    model_vars[name + "weights"] = trg_emb.transpose()
    model_vars[name + "bias"] = transformer.trg_embedding.shared_bias

    # source embedding
    name = "SequenceToSequence/input_symbol_modality_posenc_wrapper/input_symbol_modality/emb/weights"
    vocab_size = len(transformer.src_embedding.token_embedding) // dmodel
    src_emb = numpy.reshape(
        transformer.src_embedding.token_embedding, [vocab_size, dmodel]
    )
    src_emb /= dmodel ** 0.5
    model_vars[name] = src_emb

    # encoder
    encprefix = "SequenceToSequence/TransformerEncoder"
    for lid, layer in enumerate(transformer.encoder_stack):
        name = encprefix + f"/layer_{lid}/"
        model_vars[
            name + "self_attention_prepost_wrapper/ln/gamma"
        ] = layer.multihead_norm_scale
        model_vars[
            name + "self_attention_prepost_wrapper/ln/beta"
        ] = layer.multihead_norm_bias
        model_vars[
            name
            + "self_attention_prepost_wrapper/self_attention/output_transform/kernel"
        ] = numpy.reshape(layer.multihead_project_kernel_output, [dmodel, dmodel])
        model_vars[
            name + "self_attention_prepost_wrapper/self_attention/output_transform/bias"
        ] = layer.multihead_project_bias_output
        model_vars[
            name + "self_attention_prepost_wrapper/self_attention/qkv_transform/kernel"
        ] = numpy.reshape(layer.multihead_project_kernel_qkv, [dmodel, dmodel * 3])
        model_vars[
            name + "self_attention_prepost_wrapper/self_attention/qkv_transform/bias"
        ] = layer.multihead_project_bias_qkv
        model_vars[name + "ffn_prepost_wrapper/ffn/dense1/kernel"] = numpy.reshape(
            layer.ffn_first_kernel, [dmodel, -1]
        )
        model_vars[name + "ffn_prepost_wrapper/ffn/dense1/bias"] = layer.ffn_first_bias
        model_vars[name + "ffn_prepost_wrapper/ffn/dense2/kernel"] = numpy.reshape(
            layer.ffn_second_kernel, [-1, dmodel]
        )
        model_vars[name + "ffn_prepost_wrapper/ffn/dense2/bias"] = layer.ffn_second_bias
        model_vars[name + "ffn_prepost_wrapper/ln/gamma"] = layer.ffn_norm_scale
        model_vars[name + "ffn_prepost_wrapper/ln/beta"] = layer.ffn_norm_bias
        model_vars[
            encprefix + "/output_ln/gamma"
        ] = transformer.src_embedding.norm_scale
        model_vars[encprefix + "/output_ln/beta"] = transformer.src_embedding.norm_bias

    # decoder
    dlayers = len(transformer.decoder_stack)
    decprefix = "SequenceToSequence/TransformerDecoder"
    for lid, layer in enumerate(transformer.decoder_stack):
        name = decprefix + f"/layer_{lid}/"
        model_vars[
            name + "self_attention_prepost_wrapper/ln/gamma"
        ] = layer.self_norm_scale
        model_vars[
            name + "self_attention_prepost_wrapper/ln/beta"
        ] = layer.self_norm_bias
        model_vars[
            name
            + "self_attention_prepost_wrapper/self_attention/output_transform/kernel"
        ] = numpy.reshape(layer.self_project_kernel_output, [dmodel, dmodel])
        model_vars[
            name + "self_attention_prepost_wrapper/self_attention/output_transform/bias"
        ] = layer.self_project_bias_output
        model_vars[
            name + "self_attention_prepost_wrapper/self_attention/qkv_transform/kernel"
        ] = numpy.reshape(layer.self_project_kernel_qkv, [dmodel, dmodel * 3])
        model_vars[
            name + "self_attention_prepost_wrapper/self_attention/qkv_transform/bias"
        ] = layer.self_project_bias_qkv
        model_vars[name + "ffn_prepost_wrapper/ffn/dense1/kernel"] = numpy.reshape(
            layer.ffn_first_kernel, [dmodel, -1]
        )
        model_vars[name + "ffn_prepost_wrapper/ffn/dense1/bias"] = layer.ffn_first_bias
        model_vars[name + "ffn_prepost_wrapper/ffn/dense2/kernel"] = numpy.reshape(
            layer.ffn_second_kernel, [-1, dmodel]
        )
        model_vars[name + "ffn_prepost_wrapper/ffn/dense2/bias"] = layer.ffn_second_bias
        model_vars[name + "ffn_prepost_wrapper/ln/gamma"] = layer.ffn_norm_scale
        model_vars[name + "ffn_prepost_wrapper/ln/beta"] = layer.ffn_norm_bias
        model_vars[
            name
            + "encdec_attention_prepost_wrapper/encdec_attention/output_transform/kernel"
        ] = numpy.reshape(layer.encdec_project_kernel_output, [dmodel, dmodel])
        model_vars[
            name
            + "encdec_attention_prepost_wrapper/encdec_attention/output_transform/bias"
        ] = layer.encdec_project_bias_output
        model_vars[
            name
            + "encdec_attention_prepost_wrapper/encdec_attention/q_transform/kernel"
        ] = numpy.reshape(layer.encdec_project_kernel_q, [dmodel, dmodel])
        model_vars[
            name + "encdec_attention_prepost_wrapper/encdec_attention/q_transform/bias"
        ] = layer.encdec_project_bias_q
        model_vars[
            name + "encdec_attention_prepost_wrapper/ln/gamma"
        ] = layer.encdec_norm_scale
        model_vars[
            name + "encdec_attention_prepost_wrapper/ln/beta"
        ] = layer.encdec_norm_bias
    model_vars[decprefix + "/output_ln/gamma"] = transformer.trg_embedding.norm_scale
    model_vars[decprefix + "/output_ln/beta"] = transformer.trg_embedding.norm_bias
    kv_kernels = numpy.split(
        numpy.reshape(
            transformer.trg_embedding.encode_output_project_kernel_kv, [dmodel, -1]
        ),
        dlayers,
        axis=-1,
    )
    kv_biases = numpy.split(
        numpy.array(transformer.trg_embedding.encode_output_project_bias_kv),
        dlayers,
        axis=-1,
    )
    for lid in range(dlayers):
        name = decprefix + f"/layer_{lid}/"
        model_vars[
            name
            + "encdec_attention_prepost_wrapper/encdec_attention/kv_transform/kernel"
        ] = kv_kernels[lid]
        model_vars[
            name + "encdec_attention_prepost_wrapper/encdec_attention/kv_transform/bias"
        ] = kv_biases[lid]

    tf_vars = {}
    for name, arr in model_vars.items():
        tf_vars[name] = tf.Variable(arr, dtype=tf.float32, name=name)
    tf.train.Checkpoint(**tf_vars).save(os.path.join(model_dir, "ckpt"))
    model_configs = {
        "model.class": "Transformer",
        "model.params": {
            "modality.share_source_target_embedding": False,
            "modality.share_embedding_and_softmax_weights": True,
            "modality.dim": dmodel,
            "modality.timing": "sinusoids",
            "encoder.num_layers": len(transformer.encoder_stack),
            "encoder.hidden_size": dmodel,
            "encoder.num_attention_heads": num_heads,
            "encoder.filter_size": dmodel * 4,
            "encoder.attention_dropout_rate": 0.1,
            "encoder.attention_type": "dot_product",
            "encoder.ffn_activation": "relu",
            "encoder.ffn_dropout_rate": 0.1,
            "encoder.layer_postprocess_dropout_rate": 0.1,
            "decoder.num_layers": len(transformer.decoder_stack),
            "decoder.hidden_size": dmodel,
            "decoder.num_attention_heads": num_heads,
            "decoder.filter_size": dmodel * 4,
            "decoder.attention_dropout_rate": 0.1,
            "decoder.attention_type": "dot_product",
            "decoder.ffn_activation": "relu",
            "decoder.ffn_dropout_rate": 0.1,
            "decoder.layer_postprocess_dropout_rate": 0.1,
        },
    }
    ModelConfigs.dump(model_configs, model_dir)
