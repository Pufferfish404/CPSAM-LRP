import torch
from segment_anything import sam_model_registry
torch.backends.cuda.matmul.allow_tf32 = True
from torch import nn 
import torch.nn.functional as F
from einops import rearrange
from CPSAM-LRP.modules.layers_ours import *

class Conv2dWithRelprop(Conv2d):
    def relprop(self, cam, **kwargs):
        weight = self.weight.clamp(min=0)
        return F.conv_transpose2d(cam, weight, bias=None, stride=self.stride, padding=self.padding)

class AttentionWithRelprop(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = True,
        use_rel_pos: bool = False,
        rel_pos_zero_init: bool = True,
        input_size: Optional[Tuple[int, int]] = None,
    ) -> None:
        super().__init__()
        # A = Q*K^T
        self.matmul1 = einsum('bhid,bhjd->bhij')
        # attn = A*V
        self.matmul2 = einsum('bhij,bhjd->bhid')
        self.softmax = Softmax(dim=-1)
        self.attn_cam = None
        self.attn = None
        self.v = None
        self.v_cam = None
        self.attn_gradients = None
      
    def get_attn(self):
        return self.attn
  
    def save_attn(self, attn):
        self.attn = attn
  
    def save_attn_cam(self, cam):
        self.attn_cam = cam
  
    def get_attn_cam(self):
        return self.attn_cam
  
    def get_v(self):
        return self.v
  
    def save_v(self, v):
        self.v = v
  
    def save_v_cam(self, cam):
        self.v_cam = cam
  
    def get_v_cam(self):
        return self.v_cam
  
    def save_attn_gradients(self, attn_gradients):
        self.attn_gradients = attn_gradients
  
    def get_attn_gradients(self):
        return self.attn_gradients
  
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, H, W, _ = x.shape
        # qkv with shape (3, B, nHead, H * W, C)
        qkv = self.qkv(x).reshape(B, H * W, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        # q, k, v with shape (B * nHead, H * W, C)
        q, k, v = qkv.reshape(3, B * self.num_heads, H * W, -1).unbind(0)
        self.save_v(v)
  
        dots = self.matmul1([q, k]) * self.scale
        attn = self.softmax(dots)
        self.save_attn(attn)
        attn.register_hook(self.save_attn_gradients)
  
        if self.use_rel_pos:
            attn = add_decomposed_rel_pos(attn, q, self.rel_pos_h, self.rel_pos_w, (H, W), (H, W))
  
        attn = attn.softmax(dim=-1)
        x = (attn @ v).view(B, self.num_heads, H, W, -1).permute(0, 2, 3, 1, 4).reshape(B, H, W, -1)
        x = self.proj(x)
  
        return x
    
    def relprop(self, cam, **kwargs):
        cam = self.proj.relprop(cam, **kwargs)
        cam = rearrange(cam, 'b n (h d) -> b h n d', h=self.num_heads)
    
        # attn = A*V
        (cam1, cam_v)= self.matmul2.relprop(cam, **kwargs)
        cam1 /= 2
        cam_v /= 2
    
        self.save_v_cam(cam_v)
        self.save_attn_cam(cam1)
    
        cam1 = self.softmax.relprop(cam1, **kwargs)
    
        # A = Q*K^T
        (cam_q, cam_k) = self.matmul1.relprop(cam1, **kwargs)
        cam_q /= 2
        cam_k /= 2
    
        cam_qkv = rearrange([cam_q, cam_k, cam_v], 'qkv b h n d -> b n (qkv h d)', qkv=3, h=self.num_heads)
    
        return self.qkv.relprop(cam_qkv, **kwargs)

class PatchEmbeddingWithRelprop(nn.Module):
    def __init__(
        self,
        kernel_size: Tuple[int, int] = (16, 16),
        stride: Tuple[int, int] = (16, 16),
        padding: Tuple[int, int] = (0, 0),
        in_chans: int = 3,
        embed_dim: int = 768,
    ) -> None:
        super().__init__()
        self.proj = Conv2d(
            in_chans, embed_dim, kernel_size=kernel_size, stride=stride, padding=padding
        )

    def relprop(self, cam, **kwargs):
        cam = cam.permute(0, 3, 1, 2)
        cam = self.proj.relprop(cam, **kwargs)
        return cam
      
class MLPWithRelprop():
    def __init__(
        self,
        embedding_dim: int,
        mlp_dim: int,
        act: Type[nn.Module] = GELU(),
    ) -> None:
        super().__init__()
        self.lin1 = Linear(embedding_dim, mlp_dim)
        self.lin2 = Linear(mlp_dim, embedding_dim)
        self.act = act()
      
    def relprop(self, cam: torch.Tensor, **kwargs) -> torch.Tensor:
        cam = self.lin2.relprop(cam, **kwargs)
        cam = self.act.relprop(cam, **kwargs)
        cam = self.lin1.relprop(cam, **kwargs)
        return cam

class LayerNorm2dWithRelprop():
    def __init__(self, num_channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.mu = None
        self.variance = None

   def get_mu(self):
        return self.mu
  
    def save_mu(self, mu):
        self.mu = mu

   def get_variance(self):
        return self.variance
  
    def save_variance(self, variance):
        self.variance = variance

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x.mean(1, keepdim=True)
        self.save_mu(u)
        s = (x - u).pow(2).mean(1, keepdim=True)
        self.save_variance(s)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x
        
    def relprop(self, cam, **kwargs):
        weight = self.weight[:, None, None]
        boolmask = weight != 0
        x = torch.zeros_like(cam)
        x[boolmask] = (cam[boolmask] - self.bias[:, None, None][boolmask]) / (self.weight)[boolmask]
        std = torch.sqrt(self.variance + self.eps)
        x = x * std
        x = x + self.mu
        return x

class BlockWithRelprop():
    def relprop(self, cam, **kwargs):
