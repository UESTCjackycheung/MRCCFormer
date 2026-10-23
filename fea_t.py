# FEA-T, written by Yantao Chen.
# Abandon locality: Frame-wise Embedding Aided Transformer for Automatic Modulation Recognition.
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as Func
from torch.nn.init import constant_, xavier_normal_, xavier_uniform_
from timm.layers import trunc_normal_

"\n《Abandon locality: Frame-wise embedding aided transformer for automatic modulation recognition》\n"


class MHSA_block(nn.Module):
    def __init__(
        self,
        d_model=64,
        d_fix_qk=16,
        d_fix_v=16,
        n_head_qk=4,
        n_head_v=4,
        dropout=0.0,
        bias=False,
        talking=True,
        attn_res=False,
    ):
        super(MHSA_block, self).__init__()
        self.dm = d_model
        self.df_qk = d_fix_qk
        self.df_v = d_fix_v
        self.h_qk = n_head_qk
        self.h_v = n_head_v
        self.attn_res = attn_res
        self.register_buffer(
            "scale", torch.tensor(float(d_fix_qk)).sqrt(), persistent=False
        )
        self.to_q = nn.Linear(d_model, d_fix_qk * n_head_qk, bias=bias)
        self.to_k = nn.Linear(d_model, d_fix_qk * n_head_qk, bias=bias)
        self.to_v = nn.Linear(d_model, d_fix_v * n_head_v, bias=bias)
        self.proj_bf = (
            nn.Conv2d(n_head_qk, n_head_qk, (1, 1), bias=False)
            if talking
            else nn.Identity()
        )
        self.proj_v = nn.Linear(d_fix_v * n_head_v, d_model, bias=bias)
        self.softmax = nn.Softmax(dim=-1)
        self.Dp_attn = nn.Dropout(dropout)
        self.Dp_v = nn.Dropout(dropout)

    def forward(self, x, attn_=None, return_attn=False):
        B, P, D = x.shape
        attn_ = attn_ if attn_ is not None else 0.0
        q = (
            self.to_q(x).reshape(B, P, self.h_qk, self.df_qk).permute(0, 2, 1, 3)
            / self.scale
        )
        k = self.to_k(x).reshape(B, P, self.h_qk, self.df_qk).permute(0, 2, 1, 3)
        v = self.to_v(x).reshape(B, P, self.h_v, self.df_v).permute(0, 2, 1, 3)
        attn = q @ k.transpose(-2, -1)
        attn = self.proj_bf(attn)
        attn = self.softmax(attn + attn_)
        attn = self.Dp_attn(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, P, self.df_v * self.h_v)
        x = self.Dp_v(self.proj_v(x))
        if return_attn or self.attn_res:
            return (x, attn)
        return x


class MLP_block(nn.Module):
    def __init__(self, d_model=64, dim_feedforward=256, dropout=0.0, activate="relu"):
        super(MLP_block, self).__init__()
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        if activate == "relu":
            self.Act = nn.ReLU()
        elif activate == "elu":
            self.Act = nn.ELU()
        else:
            self.Act = nn.GELU()

    def forward(self, x):
        return self.linear2(self.dropout(self.Act(self.linear1(x))))


class GLU_block(nn.Module):
    def __init__(self, d_model=64, dim_feedforward=128, dropout=0.0, activate="relu"):
        super(GLU_block, self).__init__()
        self.dim_f = dim_feedforward // 2
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward // 2, d_model)
        if activate == "relu":
            self.Act = nn.ReLU()
        elif activate == "sigmoid":
            self.Act = nn.Sigmoid()
        else:
            self.Act = nn.GELU()

    def forward(self, x):
        x_ = self.linear1(x)
        x, x_gate = (x_[..., : self.dim_f], x_[..., self.dim_f :])
        return self.linear2(self.dropout(x * self.Act(x_gate)))


class DB_GLU_block(nn.Module):
    def __init__(
        self,
        d_model=64,
        dim_feedforward=128,
        dropout=0.0,
        activate="relu",
        merge_method="add",
    ):
        super(DB_GLU_block, self).__init__()
        self.merge_method = merge_method
        self.dim_f = dim_feedforward // 2
        dim_linear2 = dim_feedforward // 2 if merge_method != "cat" else dim_feedforward
        if merge_method == "cross_add" and dropout == 0.0:
            dropout = 0.1
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.dropout_2 = (
            nn.Dropout(dropout) if merge_method == "cross_add" else nn.Identity()
        )
        self.linear2 = nn.Linear(dim_linear2, d_model)
        if activate == "relu":
            self.Act = nn.ReLU()
        elif activate == "sigmoid":
            self.Act = nn.Sigmoid()
        else:
            self.Act = nn.GELU()

    def forward(self, x):
        x_ = self.linear1(x)
        x_1, x_2 = (x_[..., : self.dim_f], x_[..., self.dim_f :])
        if self.merge_method == "add":
            return self.linear2(self.dropout(x_1 * self.Act(x_2) + x_2 * self.Act(x_1)))
        elif self.merge_method == "cross_add":
            return self.linear2(
                self.dropout(x_1 * self.Act(x_2)) + self.dropout_2(x_2 * self.Act(x_1))
            )
        else:
            return self.linear2(
                self.dropout(
                    torch.cat((x_1 * self.Act(x_2), x_2 * self.Act(x_1)), dim=-1)
                )
            )


class Frame(nn.Module):
    def __init__(self, PatchSize=32, n_patch=32, overlap=0.5, bias=False):
        super(Frame, self).__init__()
        self.Stride = int(PatchSize * overlap)
        self.PatchSize = PatchSize
        self.FrameNum = n_patch - 1
        self.Embeddinbg = nn.Linear(PatchSize * 2, PatchSize * 2)

    def forward(self, x):
        frames = x.unfold(dimension=1, size=self.PatchSize, step=self.Stride)
        if frames.size(1) != self.FrameNum:
            raise ValueError(
                f"Expected {self.FrameNum} frames, obtained {frames.size(1)}"
            )
        input_feature = torch.cat((frames[:, :, 0], frames[:, :, 1]), dim=-1)
        return self.Embeddinbg(input_feature)


class Transformer_layer(nn.Module):
    def __init__(
        self,
        d_model=64,
        d_fix_qk=64,
        d_fix_v=16,
        d_mid=256,
        n_head_qk=4,
        n_head_v=4,
        dropout=0.0,
        bias=False,
        talking=True,
        real_former=True,
        activate="relu",
        ffn_type="GLU",
    ):
        super(Transformer_layer, self).__init__()
        self.attn_res = real_former
        self.mhsa = MHSA_block(
            d_model=d_model,
            d_fix_qk=d_fix_qk,
            d_fix_v=d_fix_v,
            n_head_qk=n_head_qk,
            n_head_v=n_head_v,
            dropout=dropout,
            bias=bias,
            talking=talking,
            attn_res=real_former,
        )
        if ffn_type == "MLP":
            self.ffn = MLP_block(
                d_model=d_model,
                dim_feedforward=d_mid,
                dropout=dropout,
                activate=activate,
            )
        elif ffn_type == "GLU":
            self.ffn = GLU_block(
                d_model=d_model,
                dim_feedforward=d_mid,
                dropout=dropout,
                activate=activate,
            )
        else:
            self.ffn = DB_GLU_block(
                d_model=d_model,
                dim_feedforward=d_mid,
                dropout=dropout,
                activate=activate,
                merge_method=ffn_type[7:],
            )
        self.norm_1 = nn.LayerNorm(d_model, eps=1e-05)
        self.norm_2 = nn.LayerNorm(d_model, eps=1e-05)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            xavier_uniform_(m.weight)
        if (
            isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d)
        ) and m.bias is not None:
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x, attn_=None):
        if self.attn_res:
            x_sa, attn_ = self.mhsa(x, attn_)
        else:
            x_sa = self.mhsa(x)
        x = self.norm_1(x + x_sa)
        x = self.norm_2(x + self.ffn(x))
        return (x, attn_)


class FEA_T(nn.Module):
    def __init__(
        self,
        patch_size=16,
        d_fix_qk=16,
        d_fix_v=16,
        seq_length=128,
        hidden_features=64 * 4,
        n_head_qk=4,
        n_head_v=4,
        overlap=0.2,
        dropout=0.0,
        layer_num=8,
        num_class=11,
        pos_emb=True,
        bias=False,
        talking=False,
        real_former=False,
        ffn_type="DB_GLU_add",
        activation="gelu",
        feature_mode="IQ",
    ):
        super(FEA_T, self).__init__()
        self.if_pos_emb = pos_emb
        self.feature_mode = feature_mode
        n_patch = int((seq_length - patch_size) / int(patch_size * overlap) + 2)
        in_features = patch_size * 2
        self.cls_token = nn.Parameter(torch.zeros(1, 1, patch_size * 2))
        self.pos_embed = nn.Parameter(torch.zeros(1, n_patch, in_features))
        self.Embedding = Frame(patch_size, n_patch, bias=False, overlap=overlap)
        self.Enc = nn.ModuleList([])
        for i in range(layer_num):
            self.Enc.append(
                Transformer_layer(
                    d_model=in_features,
                    d_fix_qk=d_fix_qk,
                    d_fix_v=d_fix_v,
                    d_mid=hidden_features,
                    n_head_qk=n_head_qk,
                    n_head_v=n_head_v,
                    dropout=dropout,
                    bias=bias,
                    talking=talking,
                    real_former=real_former,
                    activate=activation,
                    ffn_type=ffn_type,
                )
            )
        self.classifier = nn.Linear(in_features, num_class)
        trunc_normal_(self.cls_token, std=0.02)
        trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x, state=None):
        x = torch.transpose(x, 1, 2)
        B = x.shape[0]
        dlY = self.Embedding(x)
        cls_token = self.cls_token.expand(B, -1, -1)
        dlY = torch.cat((dlY, cls_token), dim=1)
        dlY = dlY + self.pos_embed if self.if_pos_emb else dlY
        attn_res = None
        for i in range(len(self.Enc)):
            dlY, attn_res = self.Enc[i](dlY, attn_res)
        return self.classifier(dlY[:, -1])


def fea_t_128(num_classes: int, sig_size: int):
    model = FEA_T(
        patch_size=16, seq_length=sig_size, overlap=0.2, num_class=num_classes
    )
    return model


class _MHSA_block1024(nn.Module):
    def __init__(
        self,
        d_model=64,
        d_fix_qk=16,
        d_fix_v=16,
        n_head_qk=4,
        n_head_v=4,
        dropout=0.0,
        bias=False,
        talking=True,
        attn_res=False,
    ):
        super(_MHSA_block1024, self).__init__()
        self.dm = d_model
        self.df_qk = d_fix_qk
        self.df_v = d_fix_v
        self.h_qk = n_head_qk
        self.h_v = n_head_v
        self.attn_res = attn_res
        self.scale = float(d_fix_qk) ** 0.5
        self.to_q = nn.Linear(d_model, d_fix_qk * n_head_qk, bias=bias)
        self.to_k = nn.Linear(d_model, d_fix_qk * n_head_qk, bias=bias)
        self.to_v = nn.Linear(d_model, d_fix_v * n_head_v, bias=bias)
        self.proj_bf = (
            nn.Conv2d(n_head_qk, n_head_qk, (1, 1), bias=False)
            if talking
            else nn.Identity()
        )
        self.proj_v = nn.Linear(d_fix_v * n_head_v, d_model, bias=bias)
        self.softmax = nn.Softmax(dim=-1)
        self.Dp_attn = nn.Dropout(dropout)
        self.Dp_v = nn.Dropout(dropout)

    def forward(self, x, attn_=None, return_attn=False):
        B, P, D = x.shape
        attn_ = attn_ if attn_ is not None else 0.0
        q = (
            self.to_q(x).reshape(B, P, self.h_qk, self.df_qk).permute(0, 2, 1, 3)
            / self.scale
        )
        k = self.to_k(x).reshape(B, P, self.h_qk, self.df_qk).permute(0, 2, 1, 3)
        v = self.to_v(x).reshape(B, P, self.h_v, self.df_v).permute(0, 2, 1, 3)
        attn = q @ k.transpose(-2, -1)
        attn = self.proj_bf(attn)
        attn = self.softmax(attn + attn_)
        attn = self.Dp_attn(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, P, self.df_v * self.h_v)
        x = self.Dp_v(self.proj_v(x))
        if return_attn or self.attn_res:
            return (x, attn)
        return x


class _Transformer_layer1024(nn.Module):
    def __init__(
        self,
        d_model=64,
        d_fix_qk=64,
        d_fix_v=16,
        d_mid=256,
        n_head_qk=4,
        n_head_v=4,
        dropout=0.0,
        bias=False,
        talking=True,
        real_former=True,
        activate="relu",
        ffn_type="GLU",
    ):
        super(_Transformer_layer1024, self).__init__()
        self.attn_res = real_former
        self.mhsa = _MHSA_block1024(
            d_model=d_model,
            d_fix_qk=d_fix_qk,
            d_fix_v=d_fix_v,
            n_head_qk=n_head_qk,
            n_head_v=n_head_v,
            dropout=dropout,
            bias=bias,
            talking=talking,
            attn_res=real_former,
        )
        if ffn_type == "MLP":
            self.ffn = MLP_block(
                d_model=d_model,
                dim_feedforward=d_mid,
                dropout=dropout,
                activate=activate,
            )
        elif ffn_type == "GLU":
            self.ffn = GLU_block(
                d_model=d_model,
                dim_feedforward=d_mid,
                dropout=dropout,
                activate=activate,
            )
        else:
            self.ffn = DB_GLU_block(
                d_model=d_model,
                dim_feedforward=d_mid,
                dropout=dropout,
                activate=activate,
                merge_method=ffn_type[7:],
            )
        self.norm_1 = nn.LayerNorm(d_model, eps=1e-05)
        self.norm_2 = nn.LayerNorm(d_model, eps=1e-05)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            xavier_uniform_(m.weight)
        if (
            isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d)
        ) and m.bias is not None:
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x, attn_=None):
        if self.attn_res:
            x_sa, attn_ = self.mhsa(x, attn_)
        else:
            x_sa = self.mhsa(x)
        x = self.norm_1(x + x_sa)
        x = self.norm_2(x + self.ffn(x))
        return (x, attn_)


class _FEA_T1024(nn.Module):
    def __init__(
        self,
        patch_size=32,
        d_fix_qk=16,
        d_fix_v=16,
        seq_length=1024,
        hidden_features=64 * 4,
        n_head_qk=4,
        n_head_v=4,
        overlap=0.5,
        dropout=0.0,
        layer_num=8,
        num_class=24,
        pos_emb=True,
        bias=False,
        talking=False,
        real_former=False,
        ffn_type="DB_GLU_add",
        activation="gelu",
        feature_mode="IQ",
    ):
        super(_FEA_T1024, self).__init__()
        self.if_pos_emb = pos_emb
        self.feature_mode = feature_mode
        n_patch = int((seq_length - patch_size) / int(patch_size * overlap) + 2)
        in_features = patch_size * 2
        self.cls_token = nn.Parameter(torch.zeros(1, 1, patch_size * 2))
        self.pos_embed = nn.Parameter(torch.zeros(1, n_patch, in_features))
        self.Embedding = Frame(patch_size, n_patch, bias=False, overlap=overlap)
        self.Enc = nn.ModuleList([])
        for i in range(layer_num):
            self.Enc.append(
                _Transformer_layer1024(
                    d_model=in_features,
                    d_fix_qk=d_fix_qk,
                    d_fix_v=d_fix_v,
                    d_mid=hidden_features,
                    n_head_qk=n_head_qk,
                    n_head_v=n_head_v,
                    dropout=dropout,
                    bias=bias,
                    talking=talking,
                    real_former=real_former,
                    activate=activation,
                    ffn_type=ffn_type,
                )
            )
        self.classifier = nn.Linear(in_features, num_class)
        trunc_normal_(self.cls_token, std=0.02)
        trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x, state=None):
        x_in = torch.transpose(x, 1, 2)
        B = x_in.shape[0]
        dlY = self.Embedding(x_in)
        cls_token = self.cls_token.expand(B, -1, -1)
        dlY = torch.cat((dlY, cls_token), dim=1)
        dlY = dlY + self.pos_embed if self.if_pos_emb else dlY
        attn_res = None
        for i in range(len(self.Enc)):
            dlY, attn_res = self.Enc[i](dlY, attn_res)
        return self.classifier(dlY[:, -1])


def numParams(net):
    num = 0
    for param in net.parameters():
        if param.requires_grad:
            num += int(np.prod(param.size()))
    return num


def fea_t_1024(num_classes: int, sig_size: int) -> _FEA_T1024:
    if sig_size != 1024:
        raise ValueError("The HisarMod FEA-T configuration expects 1024 samples.")
    return _FEA_T1024(
        patch_size=32, seq_length=sig_size, overlap=0.5, num_class=num_classes
    )


def build_feat(num_classes, signal_length):
    if signal_length == 128:
        return fea_t_128(num_classes, signal_length)
    if signal_length == 1024:
        return fea_t_1024(num_classes, signal_length)
    raise ValueError("FEA-T supports input lengths 128 and 1024")
