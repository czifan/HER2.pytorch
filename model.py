import torch.nn as nn 
import torch 
import torchvision.models as models 
import os 
from torch.nn import functional as F
import numpy as np

def split_last(x, shape):
    "split the last dimension to given shape"
    shape = list(shape)
    assert shape.count(-1) <= 1
    if -1 in shape:
        shape[shape.index(-1)] = int(x.size(-1) / -np.prod(shape))
    return x.view(*x.size()[:-1], *shape)

def merge_last(x, n_dims):
    "merge the last n_dims to a dimension"
    s = x.size()
    assert n_dims > 1 and n_dims < len(s)
    return x.view(*s[:-n_dims], -1)

class FeedForward(nn.Module):
    def __init__(self, dim, out_dim, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, out_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(out_dim, out_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)

class MultiHeadedSelfAttention(nn.Module):
    """Multi-Headed Dot Product Attention"""
    def __init__(self, feat_dim, dim, num_heads, dropout):
        super().__init__()
        self.proj_q = nn.Linear(feat_dim, dim)
        self.proj_k = nn.Linear(feat_dim, dim)
        self.proj_v = nn.Linear(feat_dim, dim)
        self.drop = nn.Dropout(dropout)
        self.n_heads = num_heads
        self.scores = None # for visualization

    def forward(self, x, mask):
        """
        x, q(query), k(key), v(value) : (B(batch_size), S(seq_len), D(dim))
        mask : (B(batch_size) x S(seq_len))
        * split D(dim) into (H(n_heads), W(width of head)) ; D = H * W
        """
        # (B, S, D) -proj-> (B, S, D) -split-> (B, S, H, W) -trans-> (B, H, S, W)
        q, k, v = self.proj_q(x), self.proj_k(x), self.proj_v(x)
        q, k, v = (split_last(x, (self.n_heads, -1)).transpose(1, 2) for x in [q, k, v])
        # (B, H, S, W) @ (B, H, W, S) -> (B, H, S, S) -softmax-> (B, H, S, S)
        scores = q @ k.transpose(-2, -1) / np.sqrt(k.size(-1))
        #scores = torch.cosine_similarity(q[:, :, :, None, :], q[:, :, None, :, :], dim=-1)
        if mask is not None:
            mask = mask[:, None, None, :].float()
            scores -= 10000.0 * (1.0 - mask)
        attn_map = F.softmax(scores, dim=-1)
        scores = self.drop(attn_map)
        # (B, H, S, S) @ (B, H, S, W) -> (B, H, S, W) -trans-> (B, S, H, W)
        h = (scores @ v).transpose(1, 2).contiguous()
        # -merge-> (B, S, D)
        h = merge_last(h, 2)
        self.scores = attn_map
        return h

class Block(nn.Module):
    """Transformer Block"""
    def __init__(self, dim, num_heads, dropout):
        super().__init__()
        self.attn = MultiHeadedSelfAttention(dim, dim, num_heads, dropout)
        self.attn_norm = nn.LayerNorm(dim)
        self.mlp = nn.Linear(dim, dim)
        self.mlp_norm = nn.LayerNorm(dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, feat_x, mask):
        attn_x = self.attn(feat_x, mask)
        attn_x = self.attn_norm(attn_x)
        attn_x = attn_x + feat_x 
        mlp_x  = self.mlp(attn_x)
        mlp_x  = self.mlp_norm(mlp_x)
        mlp_x  = self.drop(mlp_x)
        out_x  = mlp_x + attn_x
        return out_x

class MyTransformer(nn.Module):
    def __init__(self,
                args,
                in_dim,
                num_head,
                dropout,
                num_attn,
                merge_token=False):
        super().__init__()
        self.args = args 
        self.merge_token = merge_token
        if self.merge_token:
            self.token = nn.Parameter(torch.zeros(1, 1, in_dim).float())
            self.pe_token = nn.Parameter(torch.zeros(1, 1, in_dim).float())
        else:
            self.weight_fc = nn.Linear(in_dim, 1, bias=True)
            self.weight = None

        self.attn_layer_lst = nn.ModuleList([
            Block(in_dim, num_head, dropout) for _ in range(num_attn)
        ])

        self.init_weights()

    @torch.no_grad()
    def init_weights(self):
        def _init(m):
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)  # _trunc_normal(m.weight, std=0.02)  # from .initialization import _trunc_normal
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.normal_(m.bias, std=1e-6)  # nn.init.constant(m.bias, 0)
        self.apply(_init)
        if self.merge_token:
            nn.init.constant_(self.token, 0)
            nn.init.constant_(self.pe_token, 0)

    def forward(self, x, mask, pe):
        # x: (B, T, C)
        # mask: (B, T)

        # if flag:
        #     mask = mask.unsqueeze(dim=-1) # (B, T, 1)
        #     x = x * mask # (B, T, C)
        #     x = torch.sum(x, dim=1) # (B, C)
        #     mask = torch.sum(mask, dim=1) # (B, 1)
        #     x = x / mask
        #     return x

        if self.merge_token:
            x = torch.cat([self.token.expand(x.shape[0], 1, -1).to(x.device), x], dim=1)
            mask = torch.cat([torch.ones(mask.shape[0], 1).float().to(mask.device), mask], dim=1)
            if pe is not None:
                pe = torch.cat([self.pe_token.expand(pe.shape[0], 1, -1).to(pe.device), pe], dim=1)
        for attn_layer in self.attn_layer_lst:
            if pe is not None:
                x = x + pe
            x = attn_layer(x, mask)
        if self.merge_token:
            return x[:, 0]
        else:
            weight = torch.softmax(self.weight_fc(x), dim=1)
            x = torch.sum(x * weight, dim=1)
            self.weight = weight.squeeze(dim=-1)
            return x
    
from ibnnet import *
class MyModel(nn.Module):
    def __init__(self,
                args,
                pretrained=True):
        super().__init__()

        def get_cnn_channel(backbone):
            if 'resnet18' in backbone:
                return 512
            elif 'resnet34' in backbone:
                return 512
            elif 'resnet50' in backbone:
                return 2048
            elif 'resnet101' in backbone:
                return 2048
            else:
                raise NotImplementedError

        resnet = eval(f'models.{args.backbone}')(pretrained=pretrained)
        self.cnn = nn.Sequential(*list(resnet.children())[:-1])

        #self.t_cnn = nn.Sequential(*list(eval(f'models.{args.backbone}')(pretrained=pretrained).children())[:-1])
        #self.s_cnn = nn.Sequential(*list(eval(f'models.{args.backbone}')(pretrained=pretrained).children())[:-1])

        # self.cnn_fc = nn.Sequential(nn.Linear(get_cnn_channel(args.backbone), args.d_model),
        #                             nn.ReLU(inplace=True),
        #                             nn.Dropout(p=args.dropout))

        #self.cnn = resnet
        #self.cnn.fc = nn.Linear(2048, 512)
        self.ttrans = MyTransformer(args, args.d_model, args.num_head, args.dropout, 2, merge_token=True)
        self.ltrans = MyTransformer(args, args.d_model, args.num_head, args.dropout, 1, merge_token=True)
        self.classifier = nn.Sequential(
            nn.LayerNorm(args.d_model),
            nn.Linear(args.d_model, args.d_model),
            nn.Linear(args.d_model, args.num_classes)
        )

        self.dropout = nn.Dropout(args.dropout)

        self.d_model = args.d_model

        # self.freeze_cnn()

    #     self.init_weights()

    # @torch.no_grad()
    # def init_weights(self):
    #     def _init(m):
    #         if isinstance(m, nn.Linear):
    #             nn.init.xavier_uniform_(m.weight)  # _trunc_normal(m.weight, std=0.02)  # from .initialization import _trunc_normal
    #             if hasattr(m, 'bias') and m.bias is not None:
    #                 nn.init.normal_(m.bias, std=1e-6)  # nn.init.constant(m.bias, 0)
    #     self.apply(_init)

    def freeze_cnn(self):
        for child in self.cnn.children():
            for param in child.parameters():
                param.requires_grad = False

    def forward(self, x, mask, source, time_pe, lesion_pe):
        # x: (B, T, N, 3, H, W)
        # mask: (B, T, N)
        # source: (B, T, N)
        # time_pe: (B, T,)
        # lesion_pe: (B, N,)
        B, T, N, C, H, W = x.shape 
        x = x.view(B*T*N, C, H, W)
        
        # t_cnn_feat = self.t_cnn(x).view(x.shape[0], self.d_model)
        # s_cnn_feat = self.s_cnn(x).view(x.shape[0], self.d_model)
        # s = source.view(B*T*N, 1)
        # cnn_feat = (t_cnn_feat * (1.0 - s)) + (s_cnn_feat * s)

        cnn_feat = self.cnn(x).view(B*T*N, -1)
        cnn_feat = self.dropout(cnn_feat)
        cnn_feat = cnn_feat.view(B, T, N, -1)
        cnn_feat = cnn_feat.permute(0, 2, 1, 3).contiguous() # (B, N, T, 2048)
        cnn_feat = cnn_feat.view(B*N, T, -1)
        cnn_mask = mask.permute(0, 2, 1).contiguous() # (B, N, T)
        cnn_mask = cnn_mask.view(B*N, T)
        time_pe = torch.stack([time_pe for _ in range(N)], dim=1).view(B*N, T, -1)
        ttrans_feat = self.ttrans(cnn_feat, cnn_mask, time_pe).view(B, N, -1) # (B, N, 512)
        # ttrans_feat = self.ttrans(cnn_feat, cnn_mask, None).view(B, N, -1) # (B, N, 512)
        # ltrans_feat = self.ltrans(ttrans_feat, mask.max(dim=1).values, lesion_pe) # (B, 512)
        ltrans_feat = self.ltrans(ttrans_feat, mask.max(dim=1).values, None) # (B, 512)
        pred = self.classifier(self.dropout(ltrans_feat))
        return pred
    
class ClinicalModel(nn.Module):
    def __init__(self,
                args,
                pretrained=True):
        super().__init__()
        
        self.in_fc = nn.ModuleList([
            nn.Linear(1, args.d_model) for _ in range(args.num_clinical)
        ])
        
        self.ttrans = MyTransformer(args, args.d_model, args.num_head, args.dropout, 2, merge_token=True)
        self.ctrans = MyTransformer(args, args.d_model, args.num_head, args.dropout, 1, merge_token=True)
        self.classifier = nn.Sequential(
            nn.LayerNorm(args.d_model),
            nn.Linear(args.d_model, args.d_model),
            nn.Linear(args.d_model, args.num_classes)
        )

        self.dropout = nn.Dropout(args.dropout)
        self.d_model = args.d_model

    def forward(self, x, mask, source, time_pe, clinical_pe):
        # x: (B, T, N)
        # mask: (B, T, N)
        # time_pe: (B, T,)
        
        B, T, N = x.shape 
        x = x.unsqueeze(dim=-1) # (B, T, N, 1)

        feat = []
        for n in range(N):
            feat.append(self.in_fc[n](x[:, :, n, :]))
        feat = torch.stack(feat, dim=2) # (B, T, N, C)
        feat = feat.view(B*T*N, -1)
        feat = self.dropout(feat)
        feat = feat.view(B, T, N, -1)
        feat = feat.permute(0, 2, 1, 3).contiguous() # (B, N, T, C)
        feat = feat.view(B*N, T, -1)
        t_mask = mask.permute(0, 2, 1).contiguous() # (B, N, T)
        t_mask = t_mask.view(B*N, T)
        time_pe = torch.stack([time_pe for _ in range(N)], dim=1).view(B*N, T, -1)
        ttrans_feat = self.ttrans(feat, t_mask, time_pe).view(B, N, -1)
        ctrans_feat = self.ctrans(ttrans_feat, mask.max(dim=1).values, None)
        pred = self.classifier(self.dropout(ctrans_feat))
        return pred

if __name__ == '__main__':
    X = torch.Tensor(2, 3, 4, 3, 224, 224).float().cuda()
    M = torch.ones(2, 3, 4).float().cuda()
    model = MyModel(None, 2, True).cuda()
    pred = model(X, M)
    print(pred.shape)
