# 参考IGEV，构建代价体，从代价体中回归得到初始视差图，用于更新
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import time
import matplotlib.pyplot as plt

from PIL import Image

from core.update import BasicMultiUpdateBlock
from core.extractor import MultiBasicEncoder, Feature
from core.geometry import Combined_Geo_Encoding_Volume
from core.submodule import *

from core.model.backbone.twins import TwinsFeatureEncoder
from core.model.backbone.waftv2_dav2 import DepthAnythingFeature
from core.model.backbone.dinov3 import DinoV3Feature
from core.model.backbone.vit import VisionTransformer, MODEL_CONFIGS

from core.utils.utils import coords_grid, Padder, bilinear_sampler, bilinear_sampler_2d

try:
    autocast = torch.cuda.amp.autocast
except:
    class autocast:
        def __init__(self, enabled):
            pass
        def __enter__(self):
            pass
        def __exit__(self, *args):
            pass

class hourglass(nn.Module):
    def __init__(self, in_channels):
        super(hourglass, self).__init__()

        self.conv1 = nn.Sequential(BasicConv(in_channels, in_channels*2, is_3d=True, bn=True, relu=True, kernel_size=3,
                                             padding=1, stride=2, dilation=1),
                                   BasicConv(in_channels*2, in_channels*2, is_3d=True, bn=True, relu=True, kernel_size=3,
                                             padding=1, stride=1, dilation=1))
                                    
        self.conv2 = nn.Sequential(BasicConv(in_channels*2, in_channels*4, is_3d=True, bn=True, relu=True, kernel_size=3,
                                             padding=1, stride=2, dilation=1),
                                   BasicConv(in_channels*4, in_channels*4, is_3d=True, bn=True, relu=True, kernel_size=3,
                                             padding=1, stride=1, dilation=1))                             

        self.conv3 = nn.Sequential(BasicConv(in_channels*4, in_channels*6, is_3d=True, bn=True, relu=True, kernel_size=3,
                                             padding=1, stride=2, dilation=1),
                                   BasicConv(in_channels*6, in_channels*6, is_3d=True, bn=True, relu=True, kernel_size=3,
                                             padding=1, stride=1, dilation=1)) 


        self.conv3_up = BasicConv(in_channels*6, in_channels*4, deconv=True, is_3d=True, bn=True,
                                  relu=True, kernel_size=(4, 4, 4), padding=(1, 1, 1), stride=(2, 2, 2))

        self.conv2_up = BasicConv(in_channels*4, in_channels*2, deconv=True, is_3d=True, bn=True,
                                  relu=True, kernel_size=(4, 4, 4), padding=(1, 1, 1), stride=(2, 2, 2))

        self.conv1_up = BasicConv(in_channels*2, 8, deconv=True, is_3d=True, bn=False,
                                  relu=False, kernel_size=(4, 4, 4), padding=(1, 1, 1), stride=(2, 2, 2))

        self.agg_0 = nn.Sequential(BasicConv(in_channels*8, in_channels*4, is_3d=True, kernel_size=1, padding=0, stride=1),
                                   BasicConv(in_channels*4, in_channels*4, is_3d=True, kernel_size=3, padding=1, stride=1),
                                   BasicConv(in_channels*4, in_channels*4, is_3d=True, kernel_size=3, padding=1, stride=1),)

        self.agg_1 = nn.Sequential(BasicConv(in_channels*4, in_channels*2, is_3d=True, kernel_size=1, padding=0, stride=1),
                                   BasicConv(in_channels*2, in_channels*2, is_3d=True, kernel_size=3, padding=1, stride=1),
                                   BasicConv(in_channels*2, in_channels*2, is_3d=True, kernel_size=3, padding=1, stride=1))



        self.feature_att_8 = FeatureAtt(in_channels*2, 64)
        self.feature_att_16 = FeatureAtt(in_channels*4, 192)
        self.feature_att_32 = FeatureAtt(in_channels*6, 160)
        self.feature_att_up_16 = FeatureAtt(in_channels*4, 192)
        self.feature_att_up_8 = FeatureAtt(in_channels*2, 64)

    def forward(self, x, features):
        # x:[B,8,maxdisp//4,H/4,W/4],features的尺寸
        # [B,16,maxdisp//8,H/8,W/8]
        conv1 = self.conv1(x)
        conv1 = self.feature_att_8(conv1, features[1])
        # [B,32,maxdisp//16,H/16,W/16]
        conv2 = self.conv2(conv1)
        conv2 = self.feature_att_16(conv2, features[2])
        # [B,48,maxdisp//32,H/32,W/32]
        conv3 = self.conv3(conv2)
        conv3 = self.feature_att_32(conv3, features[3])
        # [B,32,maxdisp//16,H/16,W/16]
        conv3_up = self.conv3_up(conv3)
        conv2 = torch.cat((conv3_up, conv2), dim=1)
        conv2 = self.agg_0(conv2)
        conv2 = self.feature_att_up_16(conv2, features[2])
        # [B,16,maxdisp//8,H/8,W/8]
        conv2_up = self.conv2_up(conv2)
        conv1 = torch.cat((conv2_up, conv1), dim=1)
        conv1 = self.agg_1(conv1)
        conv1 = self.feature_att_up_8(conv1, features[1])
        # [B,8,maxdisp//4,H/4,W/4]     
        conv = self.conv1_up(conv1)

        return conv

class resconv(nn.Module):
    def __init__(self, inp, oup, k=3, s=1):
        super(resconv, self).__init__()
        self.conv = nn.Sequential(
            nn.GELU(),
            nn.Conv2d(inp, oup, kernel_size=k, stride=s, padding=k//2, bias=True),
            nn.GELU(),
            nn.Conv2d(oup, oup, kernel_size=3, stride=1, padding=1, bias=True),
        )
        if inp != oup or s != 1:
            self.skip_conv = nn.Conv2d(inp, oup, kernel_size=1, stride=s, padding=0, bias=True)
        else:
            self.skip_conv = nn.Identity()

    def forward(self, x):
        return self.conv(x) + self.skip_conv(x)

class ResNet18Deconv(nn.Module):
    def __init__(self, inp, oup):
        super(ResNet18Deconv, self).__init__()
        self.feature_dims = [oup, 64, 128, 256, 512,]
        self.ds1 = resconv(inp, 32, k=7, s=2)
        self.conv1 = resconv(32, 64, k=3, s=2)
        self.conv2 = resconv(64, 128, k=3, s=2)
        self.conv3 = resconv(128, 256, k=3, s=2)
        self.conv4 = resconv(256, 512, k=3, s=2)
        self.up_4 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2, padding=0, bias=True)
        self.proj_3 = resconv(256, 256, k=3, s=1)
        self.up_3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2, padding=0, bias=True)
        self.proj_2 = resconv(128, 128, k=3, s=1)
        self.up_2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2, padding=0, bias=True)
        self.proj_1 = resconv(64, 64, k=3, s=1)
        self.up_1 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2, padding=0, bias=True)
        self.proj_0 = resconv(32, oup, k=3, s=1)

    def forward(self, x):
        out_0 = self.ds1(x) #H/2
        out_1 = self.conv1(out_0) #H/4
        out_2 = self.conv2(out_1) #H/8
        out_3 = self.conv3(out_2) #H/16
        out_4 = self.conv4(out_3) #H/32
        out_3 = self.proj_3(out_3 + self.up_4(out_4)) # H/16
        out_2 = self.proj_2(out_2 + self.up_3(out_3)) # H/8
        out_1 = self.proj_1(out_1 + self.up_2(out_2)) # H/4
        out_0 = self.proj_0(out_0 + self.up_1(out_1)) # H/2
        # 输出[B,oup=64,H/2,W/2],[B,64,H/4,W/4],[B,128,H/8,W/8],[B,256,H/16,W/16],[B,512,H/32,W/32],
        return [out_0, out_1, out_2, out_3, out_4]

class Feat_transfer(nn.Module):
    def __init__(self, dim_encoder):
        super(Feat_transfer, self).__init__()
        self.conv4x = nn.Sequential(
            nn.Conv2d(in_channels=int(48+dim_encoder), out_channels=48, kernel_size=5, stride=1, padding=2),
            nn.InstanceNorm2d(48), nn.ReLU()
            )
        self.conv8x = nn.Sequential(
            nn.Conv2d(in_channels=int(64+dim_encoder), out_channels=64, kernel_size=5, stride=1, padding=2),
            nn.InstanceNorm2d(64), nn.ReLU()
            )
        self.conv16x = nn.Sequential(
            nn.Conv2d(in_channels=int(192+dim_encoder), out_channels=192, kernel_size=5, stride=1, padding=2),
            nn.InstanceNorm2d(192), nn.ReLU()
            )
        self.conv32x = nn.Sequential(
            nn.Conv2d(in_channels=dim_encoder, out_channels=160, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm2d(160), nn.ReLU()
            )
        self.conv_up_32x = nn.ConvTranspose2d(160,
                                192,
                                kernel_size=3,
                                padding=1,
                                output_padding=1,
                                stride=2,
                                bias=False)
        self.conv_up_16x = nn.ConvTranspose2d(192,
                                64,
                                kernel_size=3,
                                padding=1,
                                output_padding=1,
                                stride=2,
                                bias=False)
        self.conv_up_8x = nn.ConvTranspose2d(64,
                                48,
                                kernel_size=3,
                                padding=1,
                                output_padding=1,
                                stride=2,
                                bias=False)
        
        self.res_16x = nn.Conv2d(dim_encoder, 192, kernel_size=1, padding=0, stride=1)
        self.res_8x = nn.Conv2d(dim_encoder, 64, kernel_size=1, padding=0, stride=1)
        self.res_4x = nn.Conv2d(dim_encoder, 48, kernel_size=1, padding=0, stride=1)



    # features_left_4x: (B, 64, H/4, W/4) 0
    # features_left_8x: (B, 64, H/8, W/8) 1
    # features_left_16x:(B, 64, H/16, W/16) 2
    # features_left_32x:(B, 64, H/32, W/32) 3
    def forward(self, features):
        features_mono_list = []
        # (B, 160, H/32, W/32)
        feat_32x = self.conv32x(features[3])
        # (B, 192, H/16, W/16)
        feat_32x_up = self.conv_up_32x(feat_32x)
        # (B, 192, H/16, W/16)
        feat_16x = self.conv16x(torch.cat((features[2], feat_32x_up), 1)) + self.res_16x(features[2])
        # (B, 64, H/8, W/8)
        feat_16x_up = self.conv_up_16x(feat_16x)
        # (B, 64, H/8, W/8)
        feat_8x = self.conv8x(torch.cat((features[1], feat_16x_up), 1)) + self.res_8x(features[1])
        # (B, 48, H/4, W/4)
        feat_8x_up = self.conv_up_8x(feat_8x)
        # (B, 48, H/4, W/4)
        feat_4x = self.conv4x(torch.cat((features[0], feat_8x_up), 1)) + self.res_4x(features[0])
        features_mono_list.append(feat_4x)
        features_mono_list.append(feat_8x)
        features_mono_list.append(feat_16x)
        features_mono_list.append(feat_32x)
        return features_mono_list

class WAFTv2(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        if args.feature_encoder == 'twins':
            self.encoder = TwinsFeatureEncoder(frozen=True)
            self.factor = 32
        elif args.feature_encoder == 'dav2':
            self.encoder = DepthAnythingFeature(model_name="vits", pretrained=True, lvl=-3)
            self.factor = 112
        elif args.feature_encoder == 'dinov3':
            self.encoder = DinoV3Feature(model_name="vits", lvl=-2)
            self.factor = 32
        else:
            raise ValueError(f"Unknown feature encoder: {args.feature_encoder}")

        self.pretrain_dim = self.encoder.output_dim
        self.fnet = ResNet18Deconv(3, self.pretrain_dim)
        self.iter_dim = MODEL_CONFIGS[args.iterative_module]['features']
        self.refine_net = VisionTransformer(args.iterative_module, self.iter_dim, patch_size=8)
        self.fmap_conv = nn.Conv2d(self.pretrain_dim+48, self.iter_dim, kernel_size=1, stride=1, padding=0, bias=True)
        self.hidden_conv = nn.Conv2d(self.iter_dim*2, self.iter_dim, kernel_size=1, stride=1, padding=0, bias=True)
        self.warp_linear = nn.Conv2d(3*self.iter_dim+1, self.iter_dim, 1, 1, 0, bias=True)
        self.refine_transform = nn.Conv2d(self.iter_dim//2*3, self.iter_dim, 1, 1, 0, bias=True)
        self.upsample_weight = nn.Sequential(
            # convex combination of 3x3 patches
            nn.Conv2d(self.iter_dim, 2*self.iter_dim, 3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(2*self.iter_dim, 16*9, 1, padding=0, bias=True)
        )
        self.disp_head = nn.Sequential(
            # flow(2) + weight(2) + log_b(2)
            nn.Conv2d(self.iter_dim, 2*self.iter_dim, 3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(2*self.iter_dim, 1, 1, padding=0, bias=True)
        )
        # IGEV的mask头,从隐藏状态回归上采样mask
        self.mask_head = nn.Sequential(
            nn.Conv2d(self.iter_dim, 32, 3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.spx_1_gru = nn.Sequential(nn.Conv2d(self.pretrain_dim, 32, 3, padding=1),)
        self.spx_2_gru = Conv2x(32, 32, True)
        self.spx_gru = nn.Sequential(nn.ConvTranspose2d(2*32, 9, kernel_size=4, stride=2, padding=1),)

        self.feat_transfer = Feat_transfer(self.pretrain_dim)
        self.max_disp=192
        self.corr_stem = BasicConv(8, 8, is_3d=True, kernel_size=3, stride=1, padding=1)
        self.corr_feature_att = FeatureAtt(8, 112)
        self.cost_agg = hourglass(8)
        self.classifier = nn.Conv3d(8, 1, 3, 1, 1, bias=False)

        self.spx = nn.Sequential(nn.ConvTranspose2d(2*32, 9, kernel_size=4, stride=2, padding=1),)
        self.spx_2 = Conv2x_IN(24, 32, True)
        self.spx_4 = nn.Sequential(
            BasicConv_IN(112, 24, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(24, 24, 3, 1, 1, bias=False),
            nn.InstanceNorm2d(24), nn.ReLU()
            )

    def upsample_data(self, flow, info, mask):
        # WAFT原设计,从[B,1,H/4,W/4]的视差或者[B,2,H/4,W/4]的光流,利用[B,4,H/4,W/4]的info,[N,144,H/4,W/4]的mask权重
        N, C, H, W = info.shape
        mask = mask.view(N, 1, 9, 4, 4, H, W)
        mask = torch.softmax(mask, dim=2)

        up_flow = F.unfold(4 * flow, [3, 3], padding=1)
        up_flow = up_flow.view(N, 1, 9, 1, 1, H, W)
        up_info = F.unfold(info, [3, 3], padding=1)
        up_info = up_info.view(N, C, 9, 1, 1, H, W)

        up_flow = torch.sum(mask * up_flow, dim=2)
        up_flow = up_flow.permute(0, 1, 4, 2, 5, 3)
        up_info = torch.sum(mask * up_info, dim=2)
        up_info = up_info.permute(0, 1, 4, 2, 5, 3)
        
        return up_flow.reshape(N, 1, 4*H, 4*W), up_info.reshape(N, C, 4*H, 4*W)

    def upsample_disp(self, disp, mask_feat_4, stem_2x):
        # 参考IGEV,利用1/4尺度下通道数为32的mask和1/2尺寸下通道数为32的特征图上采样
        # with autocast(enabled=self.args.mixed_precision, dtype=getattr(torch, self.args.precision_dtype, torch.float16)):
        stem_2x = self.spx_1_gru(stem_2x)
        xspx = self.spx_2_gru(mask_feat_4, stem_2x)
        spx_pred = self.spx_gru(xspx)
        spx_pred = F.softmax(spx_pred, 1)
        up_disp = context_upsample(disp*4., spx_pred).unsqueeze(1)

        return up_disp

    def normalize_image(self, img):
        '''
        @img: (B,C,H,W) in range 0-255, RGB order
        '''
        tf = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], inplace=False)
        return tf(img/255.0).contiguous()

    def forward(self, image1, image2, iters=None, flow_gt=None):
        """ Estimate disp of a picture """
        if iters is None:
            iters = self.args.train_iters
        # [B,3,H_in,W_in]
        image1 = self.normalize_image(image1)
        image2 = self.normalize_image(image2)
        padder = Padder(image1.shape, factor=self.factor)
        # [B,3,H,W]
        image1 = padder.pad(image1)
        image2 = padder.pad(image2)
        disp_predictions = []
        N, _, H, W = image1.shape
        # encoder是预训练模型，输出[B,64,H/4,W/4],[B,64,H/8,W/8][B,64,H/16,W/16],[B,64,H/32,W/32],
        fmap1_encoder_list = self.encoder(image1)
        fmap2_encoder_list = self.encoder(image2)
        # 可训练的残差头，输出一组特征图，[1,48,H/4,W/4],[1,64,H/8,W/8],[1,192,H/16,W/16],[1,160,H/32,W/32]
        fmap1_pretrain_list = self.feat_transfer(fmap1_encoder_list)
        fmap2_pretrain_list = self.feat_transfer(fmap2_encoder_list)
        # 残差块，输出[B,oup=64,H/2,W/2],[B,64,H/4,W/4],[B,128,H/8,W/8],[B,256,H/16,W/16],[B,512,H/32,W/32],
        fmap1_img_list = self.fnet(image1)
        fmap2_img_list = self.fnet(image2)
        # 用于初始视差图的上采样[B,32,H/2,W/2]
        fmap1_img_2x = self.spx_1_gru(fmap1_img_list[0])
        # 单层卷积，融合两个特征提取的结果，映射到指定尺寸，输出为[B, D_iter=32, H//4, W//4]
        fmap1_4x = self.fmap_conv(torch.cat([fmap1_pretrain_list[0], fmap1_img_list[1]], dim=1))
        fmap2_4x = self.fmap_conv(torch.cat([fmap2_pretrain_list[0], fmap2_img_list[1]], dim=1))

        # 构建代价体，用于得到初始视差图,gwc_volume形状为[B,8,maxdisp//4,H/4,W/4]
        gwc_volume = build_gwc_volume(fmap1_4x, fmap2_4x, self.max_disp//4, 8)
        # 卷积聚合,[B,8,maxdisp//4,H/4,W/4]
        gwc_volume = self.corr_stem(gwc_volume)
        # 融合特征图,从左特征图得到注意力图，加权到代价体上，[B,8,maxdisp//4,H/4,W/4]
        gwc_volume = self.corr_feature_att(gwc_volume, torch.cat([fmap1_pretrain_list[0], fmap1_img_list[1]], dim=1))
        # 沙漏聚合，[B,8,maxdisp//4,H/4,W/4]
        geo_encoding_volume = self.cost_agg(gwc_volume, fmap1_pretrain_list)
        # Init disp from geometry encoding volume [B,maxdisp//4,H/4,W/4]
        prob = F.softmax(self.classifier(geo_encoding_volume).squeeze(1), dim=1)
        # [B,1,H/4,W/4]
        init_disp = disparity_regression(prob, self.max_disp//4)
        
        # 指导初始视差图上采样
        # [B,24,H/4,W/4]
        xspx = self.spx_4(torch.cat([fmap1_pretrain_list[0], fmap1_img_list[1]], dim=1))
        # 输入的两个形状：[B,24,H/4,W/4]，[B,32,H/2,W/2]
        xspx = self.spx_2(xspx, fmap1_img_2x)
        # [B,9,H,W]
        spx_pred = self.spx(xspx)
        # 回归为概率[B,9,H,W]
        spx_pred = F.softmax(spx_pred, 1)
        
        # 单层卷积，融合左右图，输出net为[B, D_iter=32, H//4, W//4]
        net = self.hidden_conv(torch.cat([fmap1_4x, fmap2_4x], dim=1))
        disp_4x = init_disp
        for itr in range(iters):
            disp_4x = disp_4x.detach()
            # coords为坐标网格，尺寸为[B,2,H/4,W/4],使用当前估计的视差来偏移
            coords = coords_grid(N, H//4, W//4, device=image1.device)
            coords_x = coords[:, :1, :, :] + disp_4x
            coords_y = coords[:, 1:, :, :]
            coords2 = torch.cat([coords_x, coords_y], dim=1)
            # warp操作，和IGEV warp函数一致，从右特征图恢复得到左图特征，尺寸为[B,D_iter=32,H/4,W/4]
            warp_4x = bilinear_sampler_2d(fmap2_4x, coords2.permute(0, 2, 3, 1))
            # 融合特征，得到输出尺寸为[B,D_iter=32,H/4,W/4]
            refine_inp = self.warp_linear(torch.cat([fmap1_4x, warp_4x, net, disp_4x], dim=1))
            # 送入ViT中进行迭代，refine_outs['out']形状为[B, D_iter/2=16, H/4, W/4]
            refine_outs = self.refine_net(refine_inp)
            # 更新隐藏状态net，net形状保持不变[B, D_iter, H//4, W//4]
            net = self.refine_transform(torch.cat([refine_outs['out'], net], dim=1))
            # 从更新后的隐藏状态中预测视差更新量，尺寸为[B,5,H/4,W/4]
            disp_update = self.disp_head(net)
            mask = self.mask_head(net)
            disp_4x = disp_4x + disp_update
            # 利用mask和1/2特征图上采样
            disp_up = self.upsample_disp(disp_4x, mask, fmap1_img_list[0])
            disp_predictions.append(disp_up)

        for i in range(len(disp_predictions)):
             # 去除填充，得到原始形状[B,2,H_in,W_in]
            disp_predictions[i] = padder.unpad(disp_predictions[i])
         
        # init_disp = padder.unpad(torch.zeros((N,1,H,W)))
        init_disp = context_upsample(init_disp*4., spx_pred.float()).unsqueeze(1)
        return init_disp, disp_predictions
    
if __name__ == "__main__":
    # encoder使用Dinov3的vits,迭代器使用Vit的vitt
    class Args:
        def __init__(self):
            self.feature_encoder = "dinov3"
            self.iterative_module = next(iter(MODEL_CONFIGS))
            self.train_iters = 5


    @torch.no_grad()
    def test_waftv2_forward_smoke():
        args = Args()
        model = WAFTv2(args).eval()
        image1 = torch.randn(1, 3, 128, 128)
        image2 = torch.randn(1, 3, 128, 128)

        init_disp, disp_predictions = model(image1, image2)
        print(f"the first step output is {disp_predictions[0].shape}")
        print(f"iter steps is {len(disp_predictions)}")

    # ...existing code...
    @torch.no_grad()
    def test_waftv2_forward_smoke2():
        args = Args()
        model = WAFTv2(args).eval()

        left = Image.open("demo-imgs/Motorcycle/im0.png").convert("RGB")
        right = Image.open("demo-imgs/Motorcycle/im0.png").convert("RGB")
        to_tensor = torchvision.transforms.ToTensor()
        image1 = to_tensor(left).unsqueeze(0) * 255.0
        image2 = to_tensor(right).unsqueeze(0) * 255.0

        init_disp, disp_predictions = model(image1, image2)
        disp = disp_predictions[-1].squeeze().cpu().numpy()
        plt.imshow(disp, cmap="plasma")
        plt.colorbar()
        plt.savefig("demo-imgs/Motorcycle/disp4x_cost_volume.png", dpi=200)
        plt.show()

    test_waftv2_forward_smoke2()