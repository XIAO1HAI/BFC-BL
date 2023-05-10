
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from model.base.aslayer import AttentiveSqueezeLayer


class AttentionLearner(nn.Module):
    def __init__(self, inch, way):
        super(AttentionLearner, self).__init__()
        self.way = way

        def make_building_attentive_block(in_channel, out_channels, kernel_sizes, spt_strides, pool_kv=False):
            building_block_layers = []
            for idx, (outch, ksz, stride) in enumerate(zip(out_channels, kernel_sizes, spt_strides)):
                inch = in_channel if idx == 0 else out_channels[idx - 1]
                padding = ksz // 2 if ksz > 2 else 0
                building_block_layers.append(AttentiveSqueezeLayer(inch, outch, ksz, stride, padding, pool_kv=pool_kv))

            return nn.Sequential(*building_block_layers)

        self.encoder_layer4 = make_building_attentive_block(inch[0], [32, 128], [5, 3], [4, 2])
        self.p4_up = nn.Conv2d(inch[0],inch[1],[3, 3],[1, 1],padding=1)
        self.encoder_layer4_up3 = make_building_attentive_block(inch[1], [32, 128], [5, 5], [3, 3], pool_kv=True)
        self.encoder_layer3 = make_building_attentive_block(inch[1], [32, 128], [5, 5], [4, 4], pool_kv=True)
        self.p3_up = nn.Conv2d(128, inch[2], [3, 3], [1, 1], padding=1)
        self.encoder_layer3_up2 = make_building_attentive_block(inch[2], [32, 128], [5, 5], [3, 3], pool_kv=True)
        self.encoder_layer2 = make_building_attentive_block(inch[2], [32, 128], [5, 5], [4, 4], pool_kv=True)

        self.encoder_layer4to3 = make_building_attentive_block(128, [128, 128], [1, 2], [1, 1])
        self.encoder_layer3to2 = make_building_attentive_block(128, [128, 128], [1, 2], [1, 1])

        # Decoder layers
        self.decoder1 = nn.Sequential(nn.Conv2d(128, 128, (3, 3), padding=(1, 1), bias=True),
                                      nn.ReLU(),
                                      nn.Conv2d(128, 64, (3, 3), padding=(1, 1), bias=True),
                                      nn.ReLU())

        self.decoder2 = nn.Sequential(nn.Conv2d(64, 64, (3, 3), padding=(1, 1), bias=True),
                                      nn.ReLU(),
                                      nn.Conv2d(64, 2, (3, 3), padding=(1, 1), bias=True))

    def interpolate_query_dims(self, hypercorr, spatial_size):# spatial_size = 25
        bsz, ch, ha, wa, hb, wb = hypercorr.size()
        hypercorr = rearrange(hypercorr, 'b c d t h w -> (b h w) c d t')# hypercorr = （1,128,13,13）
        # (B H W) C D T -> (B H W) C * spatial_size
        hypercorr = F.interpolate(hypercorr, spatial_size, mode='bilinear', align_corners=True)# hypercorr = （1,128,25,25）
        return rearrange(hypercorr, '(b h w) c d t -> b c d t h w', b=bsz, h=hb, w=wb)

    def forward(self, hypercorr_pyramid, support_mask):
        # support_mask = （1,400,400）
        bsz, ch, ha, wa, hb, wb = hypercorr_pyramid[0].size()
        hypercorr_sqz4 = self.encoder_layer4((hypercorr_pyramid[0], support_mask))[0]# hypercorr_sqz4 = （1,128,13,13,2,2）
        p4_ = rearrange(hypercorr_pyramid[0],'b c d t h w -> b c (d t) (h w)')#(1,3,169,169)
        p4_up = self.p4_up(p4_)#(1,6,169,169)
        # print(p4_up.size)
        p4_up = rearrange(p4_up, 'b c (d t) (h w) -> b c d t h w',d=ha, t=wa, h=hb, w=wb)#(1,6,13,13,13,13)
        # print(p4_up)
        hypercorr_p4_up = self.encoder_layer4_up3((p4_up, support_mask))[0]##(1,6,13,13,2,2)
        hypercorr_sqz3 = self.encoder_layer3((hypercorr_pyramid[1], support_mask))[0]  # hypercorr_sqz3 = （1,128,25,25,2,2）
        hypercorr_p4_up = self.interpolate_query_dims(hypercorr_p4_up,hypercorr_sqz3.size()[-4:-2])
        mix_p43 = hypercorr_p4_up + hypercorr_sqz3#（1,128,25,25,2,2）

        hypercorr_mix_p43 = self.encoder_layer4to3((mix_p43, support_mask))[0]
        hypercorr_sqz4 = self.interpolate_query_dims(hypercorr_sqz4, hypercorr_mix_p43.size()[-4:-2])
        mix_p43 = hypercorr_mix_p43 + hypercorr_sqz4#(1,128,25,25,2,2)
        mix_p43 = self.encoder_layer4to3((mix_p43,support_mask))[0]


        bsz, ch, ha, wa, hb, wb = mix_p43.size()
        p43_ = rearrange(mix_p43,'b c d t h w -> b c (d t) (h w)')
        p3_up = self.p3_up(p43_)
        p3_up = rearrange(p3_up, 'b c (d t) (h w) -> b c d t h w',d=ha, t=wa, h=hb, w=wb)
        hypercorr_sqz2 = self.encoder_layer2((hypercorr_pyramid[2], support_mask))[0]  # hypercorr_sqz2 = （1,128,50,50,2,2）
        hypercorr_p3_up = self.encoder_layer3_up2((p3_up, support_mask))[0]
        hypercorr_p3_up = self.interpolate_query_dims(hypercorr_p3_up, hypercorr_sqz2.size()[-4:-2])
        mix_p32 = hypercorr_sqz2 + hypercorr_p3_up

        mix_p432 = self.interpolate_query_dims(mix_p43, mix_p32.size()[-4:-2])
        mix_p432 = mix_p432 + mix_p32

        hypercorr_p432 = self.encoder_layer3to2((mix_p432, support_mask))[0]

        bsz, ch, ha, wa, hb, wb = hypercorr_p432.size()
        hypercorr_encoded = hypercorr_p432.view(bsz, ch, ha, wa, -1).squeeze(-1)




        # hypercorr_sqz4 = hypercorr_sqz4.mean(dim=[-1, -2], keepdim=True)# hypercorr_sqz4 = （1,128,13,13,1,1）
        # hypercorr_sqz4 = self.interpolate_query_dims(hypercorr_sqz4, hypercorr_sqz3.size()[-4:-2])# # hypercorr_sqz4 = （1,128,25,25,1,1）
        # hypercorr_mix43 = hypercorr_sqz4 + hypercorr_sqz3# hypercorr_sqz43 = （1,128,25,25,2,2）
        # hypercorr_mix43 = self.encoder_layer4to3((hypercorr_mix43, support_mask))[0]# hypercorr_sqz43 = （1,128,25,25,1,1）

        # hypercorr_mix43 = self.interpolate_query_dims(hypercorr_mix43, hypercorr_sqz2.size()[-4:-2])# hypercorr_sqz43 = （1,128,50,50,1,1）
        # hypercorr_mix432 = hypercorr_mix43 + hypercorr_sqz2# hypercorr_sqz432 = （1,128,50,50,2,2）
        # hypercorr_mix432 = self.encoder_layer3to2((hypercorr_mix432, support_mask))[0]# hypercorr_sqz432 = （1,128,50,50,1,1）

        # bsz, ch, ha, wa, hb, wb = hypercorr_mix432.size()
        # hypercorr_encoded = hypercorr_mix432.view(bsz, ch, ha, wa, -1).squeeze(-1)# hypercorr_encoded = （1,128,50,50）

        # hypercorr_decoded = self.decoder1(hypercorr_encoded)# hypercorr_decoded = （1,64,50,50）
        # upsample_size = (hypercorr_decoded.size(-1) * 2,) * 2
        # hypercorr_decoded = F.interpolate(hypercorr_decoded, upsample_size, mode='bilinear', align_corners=True)# hypercorr_decoded = （1,64,100,100）
        # logit_mask = self.decoder2(hypercorr_decoded)# logit_mask = （1,2,100,100）
        #
        # logit_mask = logit_mask.view(-1, self.way, *logit_mask.shape[1:])# logit_mask = （1,1,2,100,100）

        hypercorr_decoded = self.decoder1(hypercorr_encoded)  # hypercorr_decoded =
        upsample_size = (hypercorr_decoded.size(-1) * 2,) * 2
        hypercorr_decoded = F.interpolate(hypercorr_decoded, upsample_size, mode='bilinear',
                                          align_corners=True)  # hypercorr_decoded =
        logit_mask = self.decoder2(hypercorr_decoded)  # logit_mask =

        logit_mask = logit_mask.view(-1, self.way, *logit_mask.shape[1:])  # logit_mask =

        # B, N, 2, H, W-->foreground maps
        return logit_mask
