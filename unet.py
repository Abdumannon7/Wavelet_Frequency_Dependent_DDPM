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
        in_attention=in_attention.transpose(1,2)  #(B, C, L) -> (B, L, C) is the required format of multiheadattention in pytorch
        out_attention,_=self.attention(in_attention,in_attention,in_attention) # creates 3 matrix Q K V  fromt eh input we gave and then return the output tensor and weights tensor ((B, L, L)) how much a value is relevant for another value / we can also set needs_weight as false
        out_attention,_=self.attention(in_attention,in_attention,in_attention,ne)
        out_attention=out_attention.transpose(1,2).reshape(batch_size,channel,h,w)
        out=out+out_attention

        out=self.down_sample_conv(out)
        return out
    


class mid_block(nn.Module):
    def __init__(self, in_channel,out_channel,t_emb_dim,num_heads):
        super().__init__()
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

    def forward(self,x,t_emb):
        out=x

        #resnet block 1
        resnet_input=out
        out=self.resnet_1(out)
        out=out+self.t_emb_layer(t_emb)[:,:,None,None]
        out=self.resnet_2(out)
        # out=out+ self.residual_input_conv(resnet_input)

        #attention block
        batch_size,channel,h,w = out.shape
        in_attention = out.reshape(batch_size,channel,h*w)
        in_attention=self.attention_norm(in_attention)
        in_attention=in_attention.transpose(1,2)  #(B, C, L) -> (B, L, C) is the required format of multiheadattention in pytorch
        out_attention,_=self.attention(in_attention,in_attention,in_attention) # creates 3 matrix Q K V  fromt eh input we gave and then return the output tensor and weights tensor ((B, L, L)) how much a value is relevant for another value / we can also set needs_weight as false
        out_attention,_=self.attention(in_attention,in_attention,in_attention,ne)
        out_attention=out_attention.transpose(1,2).reshape(batch_size,channel,h,w)
        out=out+out_attention

        #resnet block 2
        resnet_input=out
        out=self.resnet_2(out)
        out=out+self.t_emb_layer(t_emb)[:,:,None,None]
        out=self.resnet_2(out)
        out=out+ self.residual_input_conv(resnet_input)

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
        self.up_sample_conv= nn.ConvTranspose2d(in_channel//2,in_channel//2,kernel_size=4,stride=2,padding=1) if self.up_sample else nn.Identity()   #cuz due to skip connections the channels have doubled in the begining of the up block and we processed with twice number of channels to extract more info but in the end we have to half the chanels to make it identical will down block input, we could have written out_channel directly but to maintain the systematic logic that its a concat and twice amout of input channel hence to represneet properly we divide by 2 and how it 

    

    def forward(self,x,t_emb,out_down):
        # x=self.up_sample_conv(x)
        x=t.cat([x,out_down],dim=1)

        out=x

        #resnet block 1
        resnet_input=out
        out=self.resnet_1(out)
        out=out+self.t_emb_layer(t_emb)[:,:,None,None]
        out=self.resnet_2(out)



        #attention block
        batch_size,channel,h,w = out.shape
        in_attention = out.reshape(batch_size,channel,h*w)
        in_attention=self.attention_norm(in_attention)
        in_attention=in_attention.transpose(1,2)  #(B, C, L) -> (B, L, C) is the required format of multiheadattention in pytorch
        out_attention,_=self.attention(in_attention,in_attention,in_attention) # creates 3 matrix Q K V  fromt eh input we gave and then return the output tensor and weights tensor ((B, L, L)) how much a value is relevant for another value / we can also set needs_weight as false
        out_attention,_=self.attention(in_attention,in_attention,in_attention,ne)
        out_attention=out_attention.transpose(1,2).reshape(batch_size,channel,h,w)
        out=out+out_attention 

        out=self.up_sample_conv(out)

        return out


class Unet(nn.Module):
    def __init__(self, in_channel):
        super().__init__()
        self.down_channels=[32,64,128,256]
        self.mid_channels=[256,256,128]
        self.t_emb_dim=128
        self.down_sample = [True,True,False]

        self.t_proj=nn.Sequential(

            nn.Linear(self.t_emb_dim,self.t_emb_dim),
            nn.SiLU(),
            nn.Linear(self.t_emb_dim,self.t_emb_dim)

        )   #to get initial time step representation

        self.up_sample = [False,True,True]
    
        self.conv_in =  t.conv2d(in_channel,self.down_channels[0],kernel=3,padding=1)

        self.downs = nn.ModuleList([])
        for i in range(len(self.down_channels)-1):
            self.downs.append(DownBlock(self.down_channels[i],self.down_channels[i+1],self.t_emb_dim,down_sample=self.down_sample[i],num_heads=4))

        self.mids=nn.ModuleList([])
        for i in range(len(self.mid_channels)-1):
            self.mids.append(mid_block(self.mid_channels[i],self.mid_channels[i+1],self.t_emb_dim,num_heads=4))    


        self.ups=nn.ModuleList([])
        for i in reversed(range(len(self.down_channels)-1)):
            self.ups.append(UpBlock(self.down_channels[i]*2,self.down_channels[i-1] if i!=0 else 16,self.t_emb_dim,up_sample=self.up_sample[i],num_heads=4 ))    

        
        self.norm_out=nn.GroupNorm(8,16)
        self.conv_out=nn.Conv2d(16,in_channel,kernel_size=3,padding=1)


    def forward(self,x,t):
        out=self.conv_in(x)
        t_emb=get_time_embedding(t,self.t_emb_dim)
        t_emb = self.t_proj(t_emb)

        down_outs=[]
        for down in self.downs:
            print(out.shape)
            down_outs.append(out)
            out=down(out,t_emb)
        for mid in self.mids:
            print(out.shape)
            out=mid(out,t_emb)

        for up in self.ups:
            down_out =  down_outs.pop()
            print(out,self.down_out.shape)
            out=up(out,down_out,t_emb)

        out=self.norm_out(out)
        out=nn.SiLU()(out)
        out=self.conv_out(out)
        return out    






