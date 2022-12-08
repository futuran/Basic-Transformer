import math
import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import Transformer
from typing import List, Optional

import copy
from torch.nn.modules.container import ModuleList


class PositionalEncoding(nn.Module):
    # helper Module that adds positional encoding to the token embedding to introduce a notion of word order.
    def __init__(self, emb_size: int, dropout: float, maxlen: int = 5000):
        super(PositionalEncoding, self).__init__()
        den = torch.exp(- torch.arange(0, emb_size, 2)
                        * math.log(10000) / emb_size)
        pos = torch.arange(0, maxlen).reshape(maxlen, 1)
        pos_embedding = torch.zeros((maxlen, emb_size))
        pos_embedding[:, 0::2] = torch.sin(pos * den)
        pos_embedding[:, 1::2] = torch.cos(pos * den)
        pos_embedding = pos_embedding.unsqueeze(-2)

        self.dropout = nn.Dropout(dropout)
        self.register_buffer('pos_embedding', pos_embedding)

    def forward(self, token_embedding: Tensor):
        return self.dropout(token_embedding + self.pos_embedding[:token_embedding.size(0), :])

# 未完成
class PositionalEncodingWithLanguageLabel(nn.Module):
    # helper Module that adds positional encoding to the token embedding to introduce a notion of word order.
    def __init__(self, emb_size: int, dropout: float, maxlen: int = 5000):
        super(PositionalEncoding, self).__init__()
        den = torch.exp(- torch.arange(0, emb_size, 2)
                        * math.log(10000) / emb_size)
        pos = torch.arange(0, maxlen).reshape(maxlen, 1)
        pos_embedding = torch.zeros((maxlen, emb_size))
        pos_embedding[:, 0::2] = torch.sin(pos * den)
        pos_embedding[:, 1::2] = torch.cos(pos * den)
        pos_embedding = pos_embedding.unsqueeze(-2)

        self.dropout = nn.Dropout(dropout)
        self.register_buffer('pos_embedding', pos_embedding)

        self.language_embedding = nn.Linear(2, emb_size)

    def forward(self, token_embedding: Tensor, language_ids: Tensor):
        return self.dropout(token_embedding + self.pos_embedding[:token_embedding.size(0), :])


class TokenEmbedding(nn.Module):
    # helper Module to convert tensor of input indices into corresponding tensor of token embeddings
    def __init__(self, vocab_size: int, emb_size):
        super(TokenEmbedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, emb_size)
        self.emb_size = emb_size

    def forward(self, tokens: Tensor):
        return self.embedding(tokens.long()) * math.sqrt(self.emb_size)


class OriginalTransformer(nn.Module):
    """
    ---------------------------------
    Seq2Seq Network using Transformer
    ---------------------------------

    Transformer is a Seq2Seq model introduced in `“Attention is all you
    need” <https://papers.nips.cc/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf>`__
    paper for solving machine translation tasks.
    Below, we will create a Seq2Seq network that uses Transformer. The network
    consists of three parts. First part is the embedding layer. This layer converts tensor of input indices
    into corresponding tensor of input embeddings. These embedding are further augmented with positional
    encodings to provide position information of input tokens to the model. The second part is the
    actual `Transformer <https://pytorch.org/docs/stable/generated/torch.nn.Transformer.html>`__ model.
    Finally, the output of Transformer model is passed through linear layer
    that give un-normalized probabilities for each token in the target language.
    """

    def __init__(self,
                 num_encoder_layers: int,
                 num_decoder_layers: int,
                 emb_size: int,
                 nhead: int,
                 src_vocab_size: int,
                 tgt_vocab_size: int,
                 dim_feedforward: int = 512,
                 dropout: float = 0.1):
        super(OriginalTransformer, self).__init__()
        self.transformer = Transformer(d_model=emb_size,
                                       nhead=nhead,
                                       num_encoder_layers=num_encoder_layers,
                                       num_decoder_layers=num_decoder_layers,
                                       dim_feedforward=dim_feedforward,
                                       dropout=dropout,
                                       layer_norm_eps=1e-6,
                                       norm_first=True)
        self.generator = nn.Linear(emb_size, tgt_vocab_size)
        self.src_tok_emb = TokenEmbedding(src_vocab_size, emb_size)
        # self.tgt_tok_emb = TokenEmbedding(tgt_vocab_size, emb_size)
        # for share embedding
        self.tgt_tok_emb = self.src_tok_emb

        self.positional_encoding = PositionalEncoding(
            emb_size, dropout=dropout)

        self.softmax = nn.Softmax()

    def forward(self,
                src: Tensor,
                tgt: Tensor,
                src_mask: Tensor,
                tgt_mask: Tensor,
                src_padding_mask: Tensor,
                tgt_padding_mask: Tensor,
                memory_key_padding_mask: Tensor):
        src_emb = self.positional_encoding(self.src_tok_emb(src))
        tgt_emb = self.positional_encoding(self.src_tok_emb(tgt))
        outs = self.transformer(src_emb, tgt_emb, src_mask, tgt_mask, None,
                                src_padding_mask, tgt_padding_mask, memory_key_padding_mask)
        return self.generator(outs)

    def encode(self, src: Tensor, src_mask: Tensor):
        return self.transformer.encoder(self.positional_encoding(
            self.src_tok_emb(src)), src_mask)

    def decode(self, tgt: Tensor, memory: Tensor, tgt_mask: Tensor):
        return self.transformer.decoder(self.positional_encoding(
            self.tgt_tok_emb(tgt)), memory,
            tgt_mask)


class MyTransformerEncoder(nn.Module):
    r"""TransformerEncoder is a stack of N encoder layers. Users can build the
    BERT(https://arxiv.org/abs/1810.04805) model with corresponding parameters.
    Args:
        encoder_layer: an instance of the TransformerEncoderLayer() class (required).
        num_layers: the number of sub-encoder-layers in the encoder (required).
        norm: the layer normalization component (optional).
        enable_nested_tensor: if True, input will automatically convert to nested tensor
            (and convert back on output). This will improve the overall performance of
            TransformerEncoder when padding rate is high. Default: ``True`` (enabled).
    Examples::
        >>> encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8)
        >>> transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=6)
        >>> src = torch.rand(10, 32, 512)
        >>> out = transformer_encoder(src)
    """
    __constants__ = ['norm']

    def __init__(self, encoder_layer, num_layers, norm=None, enable_nested_tensor=True, mask_check=True):
        super(MyTransformerEncoder, self).__init__()
        torch._C._log_api_usage_once(f"torch.nn.modules.{self.__class__.__name__}")
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.enable_nested_tensor = enable_nested_tensor
        self.mask_check = mask_check

    def forward(self, src: Tensor, mask: Optional[Tensor] = None, src_key_padding_mask: Optional[Tensor] = None) -> Tensor:
        r"""Pass the input through the encoder layers in turn.
        Args:
            src: the sequence to the encoder (required).
            mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).
        Shape:
            see the docs in Transformer class.
        """
        if src_key_padding_mask is not None:
            _skpm_dtype = src_key_padding_mask.dtype
            if _skpm_dtype != torch.bool and not torch.is_floating_point(src_key_padding_mask):
                raise AssertionError(
                    "only bool and floating types of key_padding_mask are supported")
        output = src
        convert_to_nested = False
        first_layer = self.layers[0]
        src_key_padding_mask_for_layers = src_key_padding_mask
        why_not_sparsity_fast_path = ''
        str_first_layer = "self.layers[0]"
        if not isinstance(first_layer, torch.nn.TransformerEncoderLayer):
            why_not_sparsity_fast_path = f"{str_first_layer} was not TransformerEncoderLayer"
        elif first_layer.norm_first :
            why_not_sparsity_fast_path = f"{str_first_layer}.norm_first was True"
        elif first_layer.training:
            why_not_sparsity_fast_path = f"{str_first_layer} was in training mode"
        elif not first_layer.self_attn.batch_first:
            why_not_sparsity_fast_path = f" {str_first_layer}.self_attn.batch_first was not True"
        elif not first_layer.self_attn._qkv_same_embed_dim:
            why_not_sparsity_fast_path = f"{str_first_layer}.self_attn._qkv_same_embed_dim was not True"
        elif not first_layer.activation_relu_or_gelu:
            why_not_sparsity_fast_path = f" {str_first_layer}.activation_relu_or_gelu was not True"
        elif not (first_layer.norm1.eps == first_layer.norm2.eps) :
            why_not_sparsity_fast_path = f"{str_first_layer}.norm1.eps was not equal to {str_first_layer}.norm2.eps"
        elif not src.dim() == 3:
            why_not_sparsity_fast_path = f"input not batched; expected src.dim() of 3 but got {src.dim()}"
        elif not self.enable_nested_tensor:
            why_not_sparsity_fast_path = "enable_nested_tensor was not True"
        elif src_key_padding_mask is None:
            why_not_sparsity_fast_path = "src_key_padding_mask was None"
        elif (((not hasattr(self, "mask_check")) or self.mask_check)
                and not torch._nested_tensor_from_mask_left_aligned(src, src_key_padding_mask.logical_not())):
            why_not_sparsity_fast_path = "mask_check enabled, and src and src_key_padding_mask was not left aligned"
        elif output.is_nested:
            why_not_sparsity_fast_path = "NestedTensor input is not supported"
        elif mask is not None:
            why_not_sparsity_fast_path = "src_key_padding_mask and mask were both supplied"
        elif first_layer.self_attn.num_heads % 2 == 1:
            why_not_sparsity_fast_path = "num_head is odd"
        elif torch.is_autocast_enabled():
            why_not_sparsity_fast_path = "autocast is enabled"

        if not why_not_sparsity_fast_path:
            tensor_args = (
                src,
                first_layer.self_attn.in_proj_weight,
                first_layer.self_attn.in_proj_bias,
                first_layer.self_attn.out_proj.weight,
                first_layer.self_attn.out_proj.bias,
                first_layer.norm1.weight,
                first_layer.norm1.bias,
                first_layer.norm2.weight,
                first_layer.norm2.bias,
                first_layer.linear1.weight,
                first_layer.linear1.bias,
                first_layer.linear2.weight,
                first_layer.linear2.bias,
            )

            if torch.overrides.has_torch_function(tensor_args):
                why_not_sparsity_fast_path = "some Tensor argument has_torch_function"
            elif not (src.is_cuda or 'cpu' in str(src.device)):
                why_not_sparsity_fast_path = "src is neither CUDA nor CPU"
            elif torch.is_grad_enabled() and any(x.requires_grad for x in tensor_args):
                why_not_sparsity_fast_path = ("grad is enabled and at least one of query or the "
                                              "input/output projection weights or biases requires_grad")

            if (not why_not_sparsity_fast_path) and (src_key_padding_mask is not None):
                convert_to_nested = True
                output = torch._nested_tensor_from_mask(output, src_key_padding_mask.logical_not(), mask_check=False)
                src_key_padding_mask_for_layers = None

        output_list = []
        for mod in self.layers:
            output = mod(output, src_mask=mask, src_key_padding_mask=src_key_padding_mask_for_layers)
            output_list.append(output)

        if convert_to_nested:
            output = output.to_padded_tensor(0.)

        if self.norm is not None:
            output = self.norm(output)

        return output_list


class MyTransformerDecoder(nn.Module):
    r"""TransformerDecoder is a stack of N decoder layers
    Args:
        decoder_layer: an instance of the TransformerDecoderLayer() class (required).
        num_layers: the number of sub-decoder-layers in the decoder (required).
        norm: the layer normalization component (optional).
    Examples::
        >>> decoder_layer = nn.TransformerDecoderLayer(d_model=512, nhead=8)
        >>> transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=6)
        >>> memory = torch.rand(10, 32, 512)
        >>> tgt = torch.rand(20, 32, 512)
        >>> out = transformer_decoder(tgt, memory)
    """
    __constants__ = ['norm']

    def __init__(self, decoder_layer, num_layers, norm=None):
        super(MyTransformerDecoder, self).__init__()
        torch._C._log_api_usage_once(f"torch.nn.modules.{self.__class__.__name__}")
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, tgt: Tensor, memory_list: List, tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None, tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None) -> Tensor:
        r"""Pass the inputs (and mask) through the decoder layer in turn.
        Args:
            tgt: the sequence to the decoder (required).
            memory: the sequence from the last layer of the encoder (required).
            tgt_mask: the mask for the tgt sequence (optional).
            memory_mask: the mask for the memory sequence (optional).
            tgt_key_padding_mask: the mask for the tgt keys per batch (optional).
            memory_key_padding_mask: the mask for the memory keys per batch (optional).
        Shape:
            see the docs in Transformer class.
        """
        output = tgt

        for mod, memory in zip(self.layers, memory_list):
            output = mod(output, memory, tgt_mask=tgt_mask,
                         memory_mask=memory_mask,
                         tgt_key_padding_mask=tgt_key_padding_mask,
                         memory_key_padding_mask=memory_key_padding_mask)

        if self.norm is not None:
            output = self.norm(output)

        return output


def _get_clones(module, N):
    return ModuleList([copy.deepcopy(module) for i in range(N)])


class MyTransformer(nn.Module):
    def __init__(self,
                 num_jirei: int,
                 num_encoder_layers: int,
                 num_decoder_layers: int,
                 emb_size: int,
                 nhead: int,
                 src_vocab_size: int,
                 tgt_vocab_size: int,
                 dim_feedforward: int = 512,
                 dropout: float = 0.1):
        super(MyTransformer, self).__init__()

        self.src_tok_emb = TokenEmbedding(src_vocab_size, emb_size)
        self.tgt_tok_emb = self.src_tok_emb

        self.positional_encoding = PositionalEncoding(emb_size, dropout=dropout)

        encoder_norm = nn.LayerNorm(emb_size, eps=1e-6)
        encoder_layer = nn.TransformerEncoderLayer(d_model=emb_size, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout, layer_norm_eps=1e-6, norm_first=True)
        self.encoder = MyTransformerEncoder(encoder_layer, num_layers=num_encoder_layers, norm=encoder_norm)

        decoder_norm = nn.LayerNorm(emb_size, eps=1e-6)
        decoder_layer = nn.TransformerDecoderLayer(d_model=emb_size, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout, layer_norm_eps=1e-6, norm_first=True)
        self.decoder = MyTransformerDecoder(decoder_layer, num_layers=num_decoder_layers, norm=decoder_norm)

        self.generator = nn.Linear(emb_size, tgt_vocab_size)

        self.softmax = nn.Softmax(dim=1)
        self.log_softmax = nn.LogSoftmax(dim=2)


    def forward(self,
                src: Tensor,
                tgt: Tensor,
                src_type_label: List,
                src_length_mask: Tensor,
                src_mask: Tensor,
                tgt_mask: Tensor,
                src_padding_mask: Tensor,
                tgt_padding_mask: Tensor,
                memory_key_padding_mask: Tensor):
        src_emb = self.positional_encoding(self.src_tok_emb(src))
        tgt_emb = self.positional_encoding(self.tgt_tok_emb(tgt))
        
        # Language Embedding
        src_length_mask = src_length_mask.unsqueeze(dim=2).expand(-1,-1, src_emb.shape[2])  # 142*96*512
        # src_emb = src_emb + (1-src_length_mask)*1024


        is_batched = src_emb.dim() == 3

        memory_list = self.encoder(src_emb, mask=src_mask, src_key_padding_mask=src_padding_mask)
        memory_en_last = torch.mul(memory_list[-1], src_length_mask)        # Encoder最終層の出力をマスクして追加


        new_memory_list = []
        for i in range(6):
            memory_de_i = torch.mul(memory_list[6-i-1], 1-src_length_mask)
            new_memory_list.append(memory_en_last + memory_de_i)

        outs = self.decoder(tgt_emb, new_memory_list, tgt_mask=tgt_mask, memory_mask=None,
                            tgt_key_padding_mask=tgt_padding_mask,
                            memory_key_padding_mask=memory_key_padding_mask)
        return self.generator(outs)

    def encode_pred(self, src: Tensor, src_mask: Tensor, src_length_mask):
        src_emb = self.positional_encoding(self.src_tok_emb(src))
        src_length_mask = src_length_mask.unsqueeze(dim=2).expand(-1,-1, src_emb.shape[2])
        memory_list = self.encoder(src_emb, src_mask)
        memory_en_last = torch.mul(memory_list[-1], src_length_mask)        # Encoder最終層の出力をマスクして追加

        new_memory_list = []
        for i in range(6):
            memory_de_i = torch.mul(memory_list[6-i-1], 1-src_length_mask)
            new_memory_list.append(memory_en_last + memory_de_i)
        return new_memory_list



    def decode_pred(self, tgt: Tensor, memory: Tensor, tgt_mask: Tensor):
        tgt_emb = self.positional_encoding(self.tgt_tok_emb(tgt))
        outs = self.decoder(tgt_emb, memory, tgt_mask)
        return outs

    # def encode_with_mask_for_training(self, src: Tensor, src_mask: Tensor, src_padding_mask: Tensor, src_length_mask: Tensor):
    #     src_emb = self.positional_encoding(self.src_tok_emb(src))
    #     memory = self.transformer.encoder(src_emb, mask=src_mask, src_key_padding_mask=src_padding_mask)
    #     src_length_mask = src_length_mask.unsqueeze(dim=2).expand(-1,-1, src_emb.shape[2])
    #     memory = torch.mul(memory, src_length_mask)
    #     return memory

    # def encode_with_mask_for_prediction(self, src: Tensor, src_mask: Tensor, src_length_mask: Tensor):
    #     src_emb = self.positional_encoding(self.src_tok_emb(src))
    #     memory = self.transformer.encoder(src_emb, mask=src_mask)
    #     src_length_mask = src_length_mask.unsqueeze(dim=2).expand(-1,-1, src_emb.shape[2])
    #     memory = torch.mul(memory, src_length_mask)
    #     return memory

    # def encode_without_mask_for_training(self, src: Tensor, src_mask: Tensor, src_padding_mask: Tensor):
    #     src_emb = self.positional_encoding(self.src_tok_emb(src))
    #     memory = self.transformer.encoder(src_emb, mask=src_mask, src_key_padding_mask=src_padding_mask)
    #     return memory

    # def encode_without_mask_for_prediction(self, src: Tensor, src_mask: Tensor):
    #     src_emb = self.positional_encoding(self.src_tok_emb(src))
    #     memory = self.transformer.encoder(src_emb, mask=src_mask)
    #     return memory

    # def decode_for_training(self, tgt: Tensor, memory: Tensor, tgt_mask: Tensor, tgt_padding_mask: Tensor, memory_key_padding_mask: Tensor):
    #     tgt_emb = self.positional_encoding(self.tgt_tok_emb(tgt))
    #     outs = self.transformer.decoder(tgt_emb, memory, tgt_mask=tgt_mask, memory_mask=None,
    #                                     tgt_key_padding_mask=tgt_padding_mask,
    #                                     memory_key_padding_mask=memory_key_padding_mask)
    #     return self.generator(outs)
    
    # def decode_for_prediction(self, tgt: Tensor, memory: Tensor, tgt_mask: Tensor):
    #     tgt_emb = self.positional_encoding(self.tgt_tok_emb(tgt))
    #     outs = self.transformer.decoder(tgt_emb, memory, tgt_mask)
    #     return outs