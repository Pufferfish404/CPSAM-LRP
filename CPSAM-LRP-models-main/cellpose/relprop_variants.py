import torch
from segment_anything import sam_model_registry
torch.backends.cuda.matmul.allow_tf32 = True
from torch import nn 
import torch.nn.functional as F
from einops import rearrange


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
      super().__init__(
          dim=dim,
          num_heads=num_heads,
          qkv_bias=qkv_bias,
          use_rel_pos=use_rel_pos,
          rel_pos_zero_init=rel_pos_zero_init,
          input_size=input_size,
      )
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
      cam = self.proj_drop.relprop(cam, **kwargs)
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

class MLPWithRelprop():
    def relprop(self, cam, **kwargs):

class ResidualBlockWithRelprop():
    def relprop(self, cam, **kwargs):

class LayerNormWithRelprop():
    def relprop(self, cam, **kwargs):
    
