import torch
import torch.nn as nn
import torch.nn.functional as F

from nets.resnet import resnet50
from nets.vgg import VGG16

# 🔥 导入注意力机制
from atention import CAA, EMA, EfficientAdditiveAttnetion, AFGCAttention, DualDomainSelectionMechanism, AttentionTSSA
from module.ECA import ECA_layer
from atention import C2f_IEL

# 🔥 导入CAA_HSFPN模块
from simplified_block import CAA_HSFPN

# 定义UNet的上采样模块
class unetUp(nn.Module):
    def __init__(self, in_size, out_size, attention_type='none'):
        super(unetUp, self).__init__()
        
        # 第一个卷积层
        self.conv1 = nn.Conv2d(in_size, out_size, kernel_size=3, padding=1)
        # 第二个卷积层
        self.conv2 = nn.Conv2d(out_size, out_size, kernel_size=3, padding=1)
        # 上采样操作，放大特征图
        self.up = nn.UpsamplingBilinear2d(scale_factor=2)
        self.relu = nn.ReLU(inplace=True)
        
        # 🔥 正确计算注意力模块的输入通道数
        if in_size == 3072:  # up_concat4
            skip_channels = 1024  # feat4
            up_channels = 2048    # feat5_up
        elif in_size == 1024:  # up_concat3
            skip_channels = 512   # feat3
            up_channels = 512     # up4
        elif in_size == 512:   # up_concat2
            skip_channels = 256   # feat2
            up_channels = 256     # up3
        elif in_size == 192:   # up_concat1
            skip_channels = 64    # feat1
            up_channels = 128     # up2
        else:
            # 🔥 通用计算方式
            up_channels = out_size * 2
            skip_channels = in_size - up_channels
            
            if skip_channels <= 0:
                skip_channels = out_size
                up_channels = in_size - skip_channels
        
        # 🔥 添加注意力模块
        if attention_type != 'none':
            self.attention = DecoderAttentionModule(
                skip_channels=skip_channels,
                up_channels=up_channels,
                attention_type=attention_type
            )
        else:
            self.attention = None
            
        print(f"🔧 unetUp模块: in_size={in_size}, out_size={out_size}, skip_channels={skip_channels}, up_channels={up_channels}, attention={attention_type}")

    def forward(self, inputs1, inputs2):
        # inputs1: 跳跃连接特征
        # inputs2: 来自下层的特征，需要上采样
        
        # 🔥 上采样inputs2
        up_feat = self.up(inputs2)
        
        # 🔥 应用注意力机制（如果有）
        if self.attention is not None:
            skip_enhanced, up_enhanced = self.attention(inputs1, up_feat)
        else:
            skip_enhanced = inputs1
            up_enhanced = up_feat
        
        # 🔥 拼接增强后的特征
        outputs = torch.cat([skip_enhanced, up_enhanced], 1)
        
        # 🔥 卷积处理
        outputs = self.conv1(outputs)
        outputs = self.relu(outputs)
        outputs = self.conv2(outputs)
        outputs = self.relu(outputs)
        
        return outputs

# 🔥 新增：特征融合模块，将CAA_HSFPN集成到编码器和解码器之间
class EncoderDecoderBridge(nn.Module):
    """编码器-解码器桥接模块，使用CAA_HSFPN进行特征融合"""
    
    def __init__(self, backbone='resnet50', use_caa_hsfpn=True):
        super(EncoderDecoderBridge, self).__init__()
        
        self.use_caa_hsfpn = use_caa_hsfpn
        self.backbone = backbone
        
        if use_caa_hsfpn:
            if backbone == 'resnet50':
                # ResNet50的各层特征通道数：feat1(64), feat2(256), feat3(512), feat4(1024), feat5(2048)
                self.caa_hsfpn_feat1 = CAA_HSFPN(ch=64, flag=True)    # 最浅层特征
                self.caa_hsfpn_feat2 = CAA_HSFPN(ch=256, flag=True)   # 第二层特征
                self.caa_hsfpn_feat3 = CAA_HSFPN(ch=512, flag=True)   # 第三层特征
                self.caa_hsfpn_feat4 = CAA_HSFPN(ch=1024, flag=True)  # 第四层特征
                self.caa_hsfpn_feat5 = CAA_HSFPN(ch=2048, flag=True)  # 最深层特征（瓶颈层）
                
            elif backbone == 'vgg':
                # VGG16的各层特征通道数（根据实际VGG实现调整）
                self.caa_hsfpn_feat1 = CAA_HSFPN(ch=64, flag=True)
                self.caa_hsfpn_feat2 = CAA_HSFPN(ch=128, flag=True)
                self.caa_hsfpn_feat3 = CAA_HSFPN(ch=256, flag=True)
                self.caa_hsfpn_feat4 = CAA_HSFPN(ch=512, flag=True)
                self.caa_hsfpn_feat5 = CAA_HSFPN(ch=512, flag=True)
            
            print(f"🔥 CAA_HSFPN桥接模块已启用 - 骨干网络: {backbone}")
            print(f"   - 将对所有编码器特征进行空间坐标注意力增强")
        else:
            print(f"⚠️ CAA_HSFPN桥接模块已禁用")
    
    def forward(self, encoder_features):
        """
        对编码器特征应用CAA_HSFPN增强
        Args:
            encoder_features: [feat1, feat2, feat3, feat4, feat5] 编码器输出的5层特征
        Returns:
            enhanced_features: 增强后的特征列表
        """
        feat1, feat2, feat3, feat4, feat5 = encoder_features
        
        if self.use_caa_hsfpn:
            # 🔥 对每层特征应用CAA_HSFPN空间坐标注意力增强
            feat1_enhanced = self.caa_hsfpn_feat1(feat1)  # 增强浅层特征的空间细节
            feat2_enhanced = self.caa_hsfpn_feat2(feat2)  # 增强第二层特征
            feat3_enhanced = self.caa_hsfpn_feat3(feat3)  # 增强第三层特征
            feat4_enhanced = self.caa_hsfpn_feat4(feat4)  # 增强第四层特征
            feat5_enhanced = self.caa_hsfpn_feat5(feat5)  # 增强瓶颈层特征的语义表达
            
            return [feat1_enhanced, feat2_enhanced, feat3_enhanced, feat4_enhanced, feat5_enhanced]
        else:
            # 不使用CAA_HSFPN，直接返回原特征
            return [feat1, feat2, feat3, feat4, feat5]

# 定义UNet主干网络
class Unet(nn.Module):
    def __init__(self, num_classes=9, pretrained=False, backbone='resnet50', 
                 attention_type='caa', layer_attentions=None, use_c2f_iel=True, 
                 use_caa_hsfpn=True, use_transmamba=False):  # 🔥 新增参数
        super(Unet, self).__init__()
        
        # 🔥 仅添加这两行保存参数状态
        self.use_caa_hsfpn = use_caa_hsfpn
        self.use_c2f_iel = use_c2f_iel
        self.use_transmamba = use_transmamba
        # 🔥 处理多层注意力配置
        if layer_attentions is None:
            self.layer_attentions = {
                'up_concat4': 'caa',
                'up_concat3': 'eca',# 'eca'
                'up_concat2': 'none',
                'up_concat1': 'none'
            }
        else:
            self.layer_attentions = {}
            for layer in ['up_concat4', 'up_concat3', 'up_concat2', 'up_concat1']:
                self.layer_attentions[layer] = layer_attentions.get(layer, attention_type)
        
        print(f"\n🔥 构建增强版多层注意力UNet:")
        print(f"   骨干网络: {backbone}")
        print(f"   CAA_HSFPN桥接: {'启用' if use_caa_hsfpn else '禁用'}")
        print(f"   C2f_IEL增强: {'启用' if use_c2f_iel else '禁用'}")
        print(f"   TransMamba处理: {'启用' if use_transmamba else '禁用'}")  # 🔥 新增
        print(f"   解码器注意力配置:")
        for layer, att_type in self.layer_attentions.items():
            print(f"     {layer}: {att_type}")
        
        # 选择backbone
        if backbone == 'vgg':
            self.vgg = VGG16(pretrained=pretrained)
            in_filters = [192, 384, 768, 1024]  # 各层输入通道数
        elif backbone == "resnet50":
            self.resnet = resnet50(pretrained=pretrained, use_transmamba=use_transmamba)  # 🔥 传递参数
            in_filters = [192, 512, 1024, 3072]
        else:
            raise ValueError('Unsupported backbone - `{}`, Use vgg, resnet50.'.format(backbone))
        
        out_filters = [64, 128, 256, 512]  # 各层输出通道数

        # 🔥 添加编码器-解码器桥接模块（CAA_HSFPN）
        self.encoder_decoder_bridge = EncoderDecoderBridge(
            backbone=backbone, 
            use_caa_hsfpn=use_caa_hsfpn
        )

        # 🔥 定义四个带注意力的上采样模块
        self.up_concat4 = unetUp(
            in_filters[3], out_filters[3], 
            attention_type=self.layer_attentions['up_concat4']
        )
        self.up_concat3 = unetUp(
            in_filters[2], out_filters[2], 
            attention_type=self.layer_attentions['up_concat3']
        )
        self.up_concat2 = unetUp(
            in_filters[1], out_filters[1], 
            attention_type=self.layer_attentions['up_concat2']
        )
        self.up_concat1 = unetUp(
            in_filters[0], out_filters[0], 
            attention_type=self.layer_attentions['up_concat1']
        )

        # resnet50 backbone下的额外上采样卷积
        if backbone == 'resnet50':
            self.up_conv = nn.Sequential(
                nn.UpsamplingBilinear2d(scale_factor=2), 
                nn.Conv2d(out_filters[0], out_filters[0], kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv2d(out_filters[0], out_filters[0], kernel_size=3, padding=1),
                nn.ReLU(),
            )
        else:
            self.up_conv = None

        # 最后一层1x1卷积输出类别数
        self.final = nn.Conv2d(out_filters[0], num_classes, 1)

        self.backbone = backbone

        # 🔥 添加C2f_IEL特征增强模块（在CAA_HSFPN之后进一步增强）
        if use_c2f_iel:  # 🔥 改为条件初始化
            print("🔧 初始化C2f_IEL模块...")
            if backbone == "resnet50":
                # ResNet50的特征通道数：feat1(64), feat2(256), feat3(512), feat4(1024)
                self.c2f_iel_feat1 = C2f_IEL(c1=64, c2=64, n=1, shortcut=False, e=0.5)
                self.c2f_iel_feat2 = C2f_IEL(c1=256, c2=256, n=2, shortcut=False, e=0.5)  
                self.c2f_iel_feat3 = C2f_IEL(c1=512, c2=512, n=3, shortcut=False, e=0.5)
                self.c2f_iel_feat4 = C2f_IEL(c1=1024, c2=1024, n=4, shortcut=False, e=0.5)
            elif backbone == "vgg":
                self.c2f_iel_feat1 = C2f_IEL(c1=64, c2=64, n=1, shortcut=False, e=0.5)
                self.c2f_iel_feat2 = C2f_IEL(c1=128, c2=128, n=1, shortcut=False, e=0.5)  
                self.c2f_iel_feat3 = C2f_IEL(c1=256, c2=256, n=1, shortcut=False, e=0.5)
                self.c2f_iel_feat4 = C2f_IEL(c1=512, c2=512, n=1, shortcut=False, e=0.5)
            print("✅ C2f_IEL模块初始化完成")
        else:
            print("⚠️ C2f_IEL模块已禁用")
            # 🔥 设置为None避免调用错误
            self.c2f_iel_feat1 = None
            self.c2f_iel_feat2 = None
            self.c2f_iel_feat3 = None
            self.c2f_iel_feat4 = None
        
        self.use_c2f_iel = use_c2f_iel
    
    def forward(self, inputs):
        # 🔥 步骤1: 编码器特征提取
        if self.backbone == "vgg":
            encoder_features = self.vgg.forward(inputs)
        elif self.backbone == "resnet50":
            encoder_features = self.resnet.forward(inputs)

        # 🔥 步骤2: 编码器-解码器桥接（CAA_HSFPN空间坐标注意力增强）
        enhanced_features = self.encoder_decoder_bridge(encoder_features)
        feat1_bridge, feat2_bridge, feat3_bridge, feat4_bridge, feat5_bridge = enhanced_features

        # 🔥 步骤3: C2f_IEL进一步特征增强（在前四层）
        if self.use_c2f_iel:
            feat1_enhanced = self.c2f_iel_feat1(feat1_bridge)  # 双重增强feat1
            feat2_enhanced = self.c2f_iel_feat2(feat2_bridge)  # 双重增强feat2  
            feat3_enhanced = self.c2f_iel_feat3(feat3_bridge)  # 双重增强feat3
            feat4_enhanced = self.c2f_iel_feat4(feat4_bridge)  # 双重增强feat4
            feat5_final = feat5_bridge  # feat5仅使用CAA_HSFPN增强
        else:
            # 仅使用CAA_HSFPN增强的特征
            feat1_enhanced = feat1_bridge
            feat2_enhanced = feat2_bridge
            feat3_enhanced = feat3_bridge
            feat4_enhanced = feat4_bridge
            feat5_final = feat5_bridge

        # 🔥 步骤4: 解码器阶段（使用双重增强后的特征）
        up4 = self.up_concat4(feat4_enhanced, feat5_final)  # 使用双重增强的feat4
        up3 = self.up_concat3(feat3_enhanced, up4)          # 使用双重增强的feat3
        up2 = self.up_concat2(feat2_enhanced, up3)          # 使用双重增强的feat2
        up1 = self.up_concat1(feat1_enhanced, up2)          # 使用双重增强的feat1

        # resnet50下再上采样一次
        if self.up_conv != None:
            up1 = self.up_conv(up1)

        # 输出分割结果
        final = self.final(up1)
        
        return final

    # 冻结backbone参数，不参与训练
    def freeze_backbone(self):
        if self.backbone == "vgg":
            for param in self.vgg.parameters():
                param.requires_grad = False
        elif self.backbone == "resnet50":
            for param in self.resnet.parameters():
                param.requires_grad = False

    # 解冻backbone参数，参与训练
    def unfreeze_backbone(self):
        if self.backbone == "vgg":
            for param in self.vgg.parameters():
                param.requires_grad = True
        elif self.backbone == "resnet50":
            for param in self.resnet.parameters():
                param.requires_grad = True
    
    def get_attention_summary(self):
        """获取注意力配置摘要"""
        summary = self.layer_attentions.copy()
        summary['caa_hsfpn_bridge'] = hasattr(self, 'encoder_decoder_bridge') and self.encoder_decoder_bridge.use_caa_hsfpn
        summary['c2f_iel_enhancement'] = self.use_c2f_iel
        return summary

# 🔥 解码器注意力模块保持不变
class DecoderAttentionModule(nn.Module):
    """解码器注意力模块，用于跳跃连接和上采样特征的融合"""
    
    def __init__(self, skip_channels, up_channels, attention_type='caa'):
        super(DecoderAttentionModule, self).__init__()
        
        self.attention_type = attention_type
        self.skip_channels = skip_channels
        self.up_channels = up_channels
        
        if attention_type == 'caa':
            self.skip_attention = CAA(ch=skip_channels)
            self.up_attention = CAA(ch=up_channels)
            
        elif attention_type == 'eca':
            self.skip_attention = ECA_layer(skip_channels)
            self.up_attention = ECA_layer(up_channels)
            
        elif attention_type == 'ema':
            self.skip_attention = EMA(channels=skip_channels)
            self.up_attention = EMA(channels=up_channels)
            
        elif attention_type == 'spatial':
            self.skip_spatial_att = nn.Sequential(
                nn.Conv2d(skip_channels, max(skip_channels//8, 1), 1),
                nn.ReLU(inplace=True),
                nn.Conv2d(max(skip_channels//8, 1), 1, 1),
                nn.Sigmoid()
            )
            self.up_spatial_att = nn.Sequential(
                nn.Conv2d(up_channels, max(up_channels//8, 1), 1),
                nn.ReLU(inplace=True),
                nn.Conv2d(max(up_channels//8, 1), 1, 1),
                nn.Sigmoid()
            )
            
        elif attention_type == 'channel':
            self.skip_channel_att = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(skip_channels, max(skip_channels//16, 1), 1),
                nn.ReLU(inplace=True),
                nn.Conv2d(max(skip_channels//16, 1), skip_channels, 1),
                nn.Sigmoid()
            )
            self.up_channel_att = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(up_channels, max(up_channels//16, 1), 1),
                nn.ReLU(inplace=True),
                nn.Conv2d(max(up_channels//16, 1), up_channels, 1),
                nn.Sigmoid()
            )
            
        elif attention_type == 'none':
            pass
            
        else:
            print(f"⚠️ 警告: 不支持的注意力类型 '{attention_type}'，将使用无注意力模式")
            self.attention_type = 'none'
        
        if attention_type != 'none':
            print(f"✅ 解码器注意力模块: {attention_type} (skip:{skip_channels}, up:{up_channels})")
    
    def forward(self, skip_feat, up_feat):
        if self.attention_type == 'caa':
            skip_enhanced = self.skip_attention(skip_feat)
            up_enhanced = self.up_attention(up_feat)
            
        elif self.attention_type == 'eca':
            skip_enhanced = self.skip_attention(skip_feat)
            up_enhanced = self.up_attention(up_feat)
            
        elif self.attention_type == 'ema':
            skip_enhanced = self.skip_attention(skip_feat)
            up_enhanced = self.up_attention(up_feat)
            
        elif self.attention_type == 'spatial':
            skip_att = self.skip_spatial_att(skip_feat)
            up_att = self.up_spatial_att(up_feat)
            skip_enhanced = skip_feat * skip_att
            up_enhanced = up_feat * up_att
            
        elif self.attention_type == 'channel':
            skip_att = self.skip_channel_att(skip_feat)
            up_att = self.up_channel_att(up_feat)
            skip_enhanced = skip_feat * skip_att
            up_enhanced = up_feat * up_att
            
        else:
            skip_enhanced = skip_feat
            up_enhanced = up_feat
        
        return skip_enhanced, up_enhanced
