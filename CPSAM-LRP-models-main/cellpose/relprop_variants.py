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

  def save_v_cam(self, cam):
      self.v_cam = cam
  
  def relprop(self, cam, **kwargs):
    if not self.matmul1 and not self.matmul2:
      # A = Q*K^T
      self.matmul1 = einsum('bhid,bhjd->bhij')
      # attn = A*V
      self.matmul2 = einsum('bhij,bhjd->bhid')
    
      cam = self.proj_drop.relprop(cam, **kwargs)
      cam = self.proj.relprop(cam, **kwargs)
      cam = rearrange(cam, 'b n (h d) -> b h n d', h=self.num_heads)

      # attn = A*V
      (cam1, cam_v)= self.matmul2.relprop(cam, **kwargs)
      cam1 /= 2
      cam_v /= 2

      self.save_v_cam(cam_v)
      self.save_attn_cam(cam1)

      cam1 = self.attn_drop.relprop(cam1, **kwargs)
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
    
