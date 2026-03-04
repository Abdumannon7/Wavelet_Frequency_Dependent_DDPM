import torch as t
import torch.nn as nn

def get_time_embedding(timestep,t_emb_dim):
    factor = 10000**((t.arange(0,t_emb_dim//2,device=timestep.device))/(t_emb_dim//2))
    scaled_time=timestep.unsqueeze(-1)/factor  #to broadcast and match the martix division so that every timestep has the factor divided  (shape alignment right to left)
    sin_component=t.sin(scaled_time)
    cos_component=t.cos(scaled_time)

    t_emb=t.cat([sin_component,cos_component],dim=-1)
    return t_emb

class DownBlock(nn.Module):
    def __init__(self, in_channel,out_channel,t_emb_dim,down_sample,num_heads):
        super().__init__()
        self.down_sample=down_sample
        self.resnet_1 = nn.Sequential(

            nn.GroupNorm(8,in_channel),
            nn.SiLU(),
            nn.Conv2d(in_channel,out_channel,kernel_size=3,stride=1,padding=1)
        )

        self.t_emb_layer = nn.Sequential(
            nn.SiLU(),
            nn.Linear(t_emb_dim,out_channel)
        )

        self.resnet_2=nn.Sequential(
            nn.GroupNorm(8,out_channel),
            nn.SiLU(),
            nn.Conv2d(out_channel,out_channel,kernel_size=3,stride=1,padding=1)

        )

        self.attention_norm = nn.GroupNorm(8,out_channel)
        self.attention=nn.MultiheadAttention(out_channel,num_heads,batch_first=True)
        self.residual_input_conv = nn.Conv2d(in_channel,out_channel,kernel_size=1)
        self.down_sample_conv= nn.Conv2d(out_channel,out_channel,kernel_size=4,stride=2,padding=1) if self.down_sample else nn.Identity()

    def forward (self,x,t_emb):
        out=x

        #resnet block
        resnet_input=out
        out=self.resnet_1(out)
        out=out+self.t_emb_layer(t_emb)[:,:,None,None]
        out=self.resnet_2(out)
        out=out+ self.residual_input_conv(resnet_input)

        #attention block
        batch_size,channel,h,w = out.shape
        in_attention = out.reshape(batch_size,channel,h*w)
        in_attention=self.attention_norm(in_attention)
        in_attention=in_attention.transpose(1,2)  #(B, C, L) to (B, L, C) is the required format of multiheadattention in pytorch
        # removed duplicate attention call with undefined 'ne' that would crash at runtime
        out_attention,_=self.attention(in_attention,in_attention,in_attention)
        out_attention=out_attention.transpose(1,2).reshape(batch_size,channel,h,w)
        out=out+out_attention

        out=self.down_sample_conv(out)
        return out



class mid_block(nn.Module):
    def __init__(self, in_channel,out_channel,t_emb_dim,num_heads):
        super().__init__()
        # first resnet block: in_channel -> out_channel
        self.resnet_1 = nn.Sequential(

            nn.GroupNorm(8,in_channel),
            nn.SiLU(),
            nn.Conv2d(in_channel,out_channel,kernel_size=3,stride=1,padding=1)
        )

        self.t_emb_layer_1 = nn.Sequential(
            nn.SiLU(),
            nn.Linear(t_emb_dim,out_channel)
        )

        self.resnet_2=nn.Sequential(
            nn.GroupNorm(8,out_channel),
            nn.SiLU(),
            nn.Conv2d(out_channel,out_channel,kernel_size=3,stride=1,padding=1)

        )

        self.attention_norm = nn.GroupNorm(8,out_channel)
        self.attention=nn.MultiheadAttention(out_channel,num_heads,batch_first=True)
        self.residual_input_conv = nn.Conv2d(in_channel,out_channel,kernel_size=1)

        # second resnet block needs dedicated layers to avoid weight sharing with first block
        self.resnet_3 = nn.Sequential(
            nn.GroupNorm(8,out_channel),
            nn.SiLU(),
            nn.Conv2d(out_channel,out_channel,kernel_size=3,stride=1,padding=1)
        )

        self.t_emb_layer_2 = nn.Sequential(
            nn.SiLU(),
            nn.Linear(t_emb_dim,out_channel)
        )

        self.resnet_4 = nn.Sequential(
            nn.GroupNorm(8,out_channel),
            nn.SiLU(),
            nn.Conv2d(out_channel,out_channel,kernel_size=3,stride=1,padding=1)
        )

    def forward(self,x,t_emb):
        out=x

        #resnet block 1
        resnet_input=out
        out=self.resnet_1(out)
        out=out+self.t_emb_layer_1(t_emb)[:,:,None,None]
        out=self.resnet_2(out)
        # restored residual connection for gradient flow through first block
        out=out+ self.residual_input_conv(resnet_input)

        #attention block
        batch_size,channel,h,w = out.shape
        in_attention = out.reshape(batch_size,channel,h*w)
        in_attention=self.attention_norm(in_attention)
        in_attention=in_attention.transpose(1,2)  #(B, C, L) -> (B, L, C) is the required format of multiheadattention in pytorch
        # removed duplicate attention call with undefined 'ne' that would crash at runtime
        out_attention,_=self.attention(in_attention,in_attention,in_attention)
        out_attention=out_attention.transpose(1,2).reshape(batch_size,channel,h,w)
        out=out+out_attention

        #resnet block 2 using dedicated layers so weights are not shared with block 1
        resnet_input=out
        out=self.resnet_3(out)
        out=out+self.t_emb_layer_2(t_emb)[:,:,None,None]
        out=self.resnet_4(out)
        # identity residual since in/out channels match for second block
        out=out+ resnet_input

        return out


class UpBlock(nn.Module):
    def __init__(self, in_channel,out_channel,t_emb_dim,up_sample,num_heads):
        super().__init__()
        self.up_sample=up_sample
        self.resnet_1 = nn.Sequential(

            nn.GroupNorm(8,in_channel),
            nn.SiLU(),
            nn.Conv2d(in_channel,out_channel,kernel_size=3,stride=1,padding=1)
        )

        self.t_emb_layer = nn.Sequential(
            nn.SiLU(),
            nn.Linear(t_emb_dim,out_channel)
        )

        self.resnet_2=nn.Sequential(
            nn.GroupNorm(8,out_channel),
            nn.SiLU(),
            nn.Conv2d(out_channel,out_channel,kernel_size=3,stride=1,padding=1)

        )

        self.attention_norm = nn.GroupNorm(8,out_channel)
        self.attention=nn.MultiheadAttention(out_channel,num_heads,batch_first=True)
        self.residual_input_conv = nn.Conv2d(in_channel,out_channel,kernel_size=1)
        # upsample operates on pre-concat channels (in_channel//2) to double spatial dims
        self.up_sample_conv= nn.ConvTranspose2d(in_channel//2,in_channel//2,kernel_size=4,stride=2,padding=1) if self.up_sample else nn.Identity()

    def forward(self,x,t_emb,out_down):
        # upsample BEFORE concat so spatial dims match the skip connection
        x=self.up_sample_conv(x)
        x=t.cat([x,out_down],dim=1)

        out=x

        #resnet block 1
        resnet_input=out
        out=self.resnet_1(out)
        out=out+self.t_emb_layer(t_emb)[:,:,None,None]
        out=self.resnet_2(out)
        # added missing residual connection for proper gradient flow
        out=out+ self.residual_input_conv(resnet_input)

        #attention block
        batch_size,channel,h,w = out.shape
        in_attention = out.reshape(batch_size,channel,h*w)
        in_attention=self.attention_norm(in_attention)
        in_attention=in_attention.transpose(1,2)  #(B, C, L) -> (B, L, C) is the required format of multiheadattention in pytorch
        # removed duplicate attention call with undefined 'ne' that would crash at runtime
        out_attention,_=self.attention(in_attention,in_attention,in_attention)
        out_attention=out_attention.transpose(1,2).reshape(batch_size,channel,h,w)
        out=out+out_attention

        return out


class Unet(nn.Module):
    # accepts config dict since train_model.py passes model_config
    def __init__(self, config):
        super().__init__()
        in_channel = config['image_channels']
        self.down_channels=[32,64,128,256]
        self.mid_channels=[256,256,128]
        self.t_emb_dim=128
        self.down_sample = [True,True,False]

        self.t_proj=nn.Sequential(

            nn.Linear(self.t_emb_dim,self.t_emb_dim),
            nn.SiLU(),
            nn.Linear(self.t_emb_dim,self.t_emb_dim)

        )   #to get initial time step representation

        # [True,True,False] so up_sample[i] mirrors down_sample[i] when iterated in reverse
        self.up_sample = [True,True,False]

        # in_channel*2 to accept concatenated conditioning input (noisy HF + LL band)
        self.conv_in = nn.Conv2d(in_channel * 2,self.down_channels[0],kernel_size=3,padding=1)

        self.downs = nn.ModuleList([])
        for i in range(len(self.down_channels)-1):
            self.downs.append(DownBlock(self.down_channels[i],self.down_channels[i+1],self.t_emb_dim,down_sample=self.down_sample[i],num_heads=4))

        self.mids=nn.ModuleList([])
        for i in range(len(self.mid_channels)-1):
            self.mids.append(mid_block(self.mid_channels[i],self.mid_channels[i+1],self.t_emb_dim,num_heads=4))

        # track actual output channels through the up path to fix dimension mismatch
        self.ups=nn.ModuleList([])
        prev_out_ch = self.mid_channels[-1]
        for i in reversed(range(len(self.down_channels)-1)):
            skip_ch = self.down_channels[i]
            in_ch = prev_out_ch + skip_ch
            out_ch = self.down_channels[i-1] if i!=0 else 16
            self.ups.append(UpBlock(in_ch, out_ch, self.t_emb_dim, up_sample=self.up_sample[i], num_heads=4))
            prev_out_ch = out_ch


        self.norm_out=nn.GroupNorm(8,16)
        self.conv_out=nn.Conv2d(16,in_channel,kernel_size=3,padding=1)


    def forward(self,x,t):
        out=self.conv_in(x)
        t_emb=get_time_embedding(t,self.t_emb_dim)
        t_emb = self.t_proj(t_emb)

        down_outs=[]
        for down in self.downs:
            down_outs.append(out)
            out=down(out,t_emb)
        for mid in self.mids:
            out=mid(out,t_emb)

        for up in self.ups:
            down_out = down_outs.pop()
            # fixed: argument order was swapped (down_out and t_emb were switched)
            out=up(out,t_emb,down_out)

        out=self.norm_out(out)
        out=nn.SiLU()(out)
        out=self.conv_out(out)
        return out
