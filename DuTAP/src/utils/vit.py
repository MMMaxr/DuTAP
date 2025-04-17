import torch
from torch import nn, einsum
import torch.nn.functional as F

from einops import rearrange, repeat
from einops.layers.torch import Rearrange
def pair(t):
    return t if isinstance(t, tuple) else (t, t)


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)



class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)



class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim=-1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout),
        ) if project_out else nn.Identity()

    def forward(self, x):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), qkv)
        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale
        attn = self.attend(dots)
        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x


class ViT(nn.Module):
    def __init__(self, num_patch, patch_dim, dim, depth, heads, mlp_dim, pool='cls',  dim_head=64, dropout=0., emb_dropout=0.):
        super().__init__()


        num_patches = num_patch
        patch_dim = patch_dim
        assert pool in {'cls', 'mean'}

        self.to_patch_embedding = nn.Sequential(
            nn.Linear(patch_dim, dim)
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches+1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))					# nn.Parameter()定义可学习参数
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.pool = pool
        self.to_latent = nn.Identity()


    def forward(self, img):
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape
        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b=b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n+1)]
        x = self.dropout(x)
        x = self.transformer(x)
        return x


class TFormer(nn.Module):
    def __init__(self, num_patch, patch_dim, dim, depth, heads, mlp_dim, pool='cls',  dim_head=64, dropout=0., emb_dropout=0.):
        super().__init__()


        num_patches = num_patch
        patch_dim = patch_dim
        assert pool in {'cls', 'mean'}

        self.to_patch_embedding = nn.Sequential(
            nn.Linear(patch_dim, dim)
        )
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches+1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.pool = pool
    def forward(self, img,cls):
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape
        cls = cls.unsqueeze(1)
        x = torch.cat((cls, x), dim=1)
        x += self.pos_embedding[:, :(n+1)]
        x = self.dropout(x)
        x = self.transformer(x)
        return x


if __name__ == '__main__':
    model = ViT(
        num_patch=108,
        patch_dim=224,
        dim = 512,
        depth = 6,
        heads = 8,
        mlp_dim = 1024,
        dim_head = 64,
        dropout = 0.1,
        emb_dropout = 0.1
    )

    img = torch.randn(8, 108, 224)
    preds = model(img)
    print(preds.shape)
    print(preds)
    print('Done')