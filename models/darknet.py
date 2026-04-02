import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# --- 1. 基础卷积块 (ConvBlock) ---
class ConvBlock(nn.Module):
    """
    基础卷积块：Conv2d + BatchNorm2d + LeakyReLU
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.leaky_relu = nn.LeakyReLU(0.1, inplace=True)

    def forward(self, x):
        return self.leaky_relu(self.bn(self.conv(x)))

# --- 2. CoordConv 辅助模块 ---
class AddCoords(nn.Module):
    """
    为输入特征图增加标准化的 (x, y) 坐标通道，取值范围 [-1, 1]。
    增强网络对绝对位置的感知能力。
    """
    def forward(self, x):
        batch_size, _, height, width = x.size()
        xx_ones = torch.ones([batch_size, 1, 1, width], dtype=x.dtype, device=x.device)
        xx_range = torch.arange(width, dtype=x.dtype, device=x.device).unsqueeze(0).unsqueeze(0).unsqueeze(0)
        xx_channel = (xx_range / (width - 1)) * 2 - 1
        xx_channel = xx_channel.repeat(batch_size, 1, height, 1)

        yy_ones = torch.ones([batch_size, 1, height, 1], dtype=x.dtype, device=x.device)
        yy_range = torch.arange(height, dtype=x.dtype, device=x.device).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
        yy_channel = (yy_range / (height - 1)) * 2 - 1
        yy_channel = yy_channel.repeat(batch_size, 1, 1, width)

        return torch.cat([x, xx_channel, yy_channel], dim=1)

# --- 3. 2D Sine-Cosine 位置编码辅助函数 ---
def get_2d_sincos_pos_embed(embed_dim, grid_size_h, grid_size_w):
    """
    生成 2D 正余弦位置编码。
    grid_size_h, grid_size_w: 特征图的高和宽
    embed_dim: 编码总维度
    """
    grid_h = torch.arange(grid_size_h, dtype=torch.float32)
    grid_w = torch.arange(grid_size_w, dtype=torch.float32)
    grid = torch.meshgrid(grid_h, grid_w, indexing='ij')
    grid = torch.stack(grid, dim=0) # [2, H, W]

    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    return pos_embed

def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 4 == 0
    # 使用一半维度编码高度 grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    # 使用另一半维度编码宽度 grid_w
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)
    return torch.cat([emb_h, emb_w], dim=1) # (H*W, D)

def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    assert embed_dim % 2 == 0
    omega = torch.arange(embed_dim // 2, dtype=torch.float32)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = torch.einsum('m,d->md', pos, omega)  # (M, D/2)

    emb_sin = torch.sin(out) # (M, D/2)
    emb_cos = torch.cos(out) # (M, D/2)

    return torch.cat([emb_sin, emb_cos], dim=1)  # (M, D)

# --- 4. 增强型 Transformer 全局感知模块 (Neck) ---
class EnhancedTransformerNeck(nn.Module):
    """
    接收 [B, 1024, 18, 25] 特征图，增加坐标信息和位置编码，
    通过 2 层 Transformer 进行全局建模。
    """
    def __init__(self, in_channels=1024, d_model=256, nhead=8, num_layers=2):
        super(EnhancedTransformerNeck, self).__init__()
        self.add_coords = AddCoords()
        # 降维：因为增加了 2 个坐标通道，in_channels + 2
        self.reduce_conv = ConvBlock(in_channels + 2, d_model, kernel_size=1)
        
        # Transformer 编码器层
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            dim_feedforward=d_model * 4, 
            batch_first=True,
            dropout=0.1
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.d_model = d_model

    def forward(self, x):
        # x shape: [B, 1024, 18, 25]
        B, C, H, W = x.shape
        
        # 1. 增加坐标通道并压缩 [B, 256, 18, 25]
        x = self.add_coords(x)
        x = self.reduce_conv(x)
        
        # 2. 序列化 [B, 450, 256]
        x_seq = x.flatten(2).transpose(1, 2)
        
        # 3. 注入 2D 正余弦位置编码
        pos_embed = get_2d_sincos_pos_embed(self.d_model, H, W).to(x.device)
        x_seq = x_seq + pos_embed.unsqueeze(0)
        
        # 4. 全局自注意力计算
        x_enhanced = self.transformer(x_seq)
        
        # 5. 重新恢复特征图空间维度 [B, 256, 18, 25]
        x_out = x_enhanced.transpose(1, 2).reshape(B, -1, H, W)
        
        return x_out

# --- 5. 改进后的 Darknet (方案 A: 全局回归) ---
class Darknet(nn.Module):
    """
    标准的 Darknet-19 Backbone + 增强版 Transformer Neck + 全局解耦 Head。
    输出维度固定为 [B, 22]，直接适配单目标回归训练。
    """
    def __init__(self, num_outputs=22):
        super(Darknet, self).__init__()
        
        # --- Backbone: Darknet-19 ---
        # Block 1
        self.conv1 = ConvBlock(3, 32, 3, padding=1)
        self.pool1 = nn.MaxPool2d(2, 2)
        
        # Block 2
        self.conv2 = ConvBlock(32, 64, 3, padding=1)
        self.pool2 = nn.MaxPool2d(2, 2)
        
        # Block 3
        self.conv3 = ConvBlock(64, 128, 3, padding=1)
        self.conv4 = ConvBlock(128, 64, 1, padding=0)
        self.conv5 = ConvBlock(64, 128, 3, padding=1)
        self.pool3 = nn.MaxPool2d(2, 2)
        
        # Block 4
        self.conv6 = ConvBlock(128, 256, 3, padding=1)
        self.conv7 = ConvBlock(256, 128, 1, padding=0)
        self.conv8 = ConvBlock(128, 256, 3, padding=1)
        self.pool4 = nn.MaxPool2d(2, 2)
        
        # Block 5
        self.conv9 = ConvBlock(256, 512, 3, padding=1)
        self.conv10 = ConvBlock(512, 256, 1, padding=0)
        self.conv11 = ConvBlock(256, 512, 3, padding=1)
        self.conv12 = ConvBlock(512, 256, 1, padding=0)
        self.conv13 = ConvBlock(256, 512, 3, padding=1)
        self.pool5 = nn.MaxPool2d(2, 2)
        
        # Block 6
        self.conv14 = ConvBlock(512, 1024, 3, padding=1)
        self.conv15 = ConvBlock(1024, 512, 1, padding=0)
        self.conv16 = ConvBlock(512, 1024, 3, padding=1)
        self.conv17 = ConvBlock(1024, 512, 1, padding=0)
        self.conv18 = ConvBlock(512, 1024, 3, padding=1) # [B, 1024, 18, 25]
        
        # --- Neck: 增强版 Transformer 全局增强 ---
        self.neck = EnhancedTransformerNeck(in_channels=1024, d_model=256, num_layers=2)
        
        # --- Head: 全局回归解耦头 (Decoupled Global Head) ---
        # 1. 全局池化：将 18x25 压缩到 1x1
        self.gap = nn.AdaptiveAvgPool2d(1)
        
        # 2. 回归分支：预测 21 维 (2D中心 + 8顶点偏移 + 3D尺寸)
        self.reg_branch = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(256, 21)
        )
        
        # 3. 置信度分支：预测 1 维 (Logits)
        self.conf_branch = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        # 1. Backbone 前向传播
        x = self.pool1(self.conv1(x))
        x = self.pool2(self.conv2(x))
        x = self.conv3(x); x = self.conv4(x); x = self.pool3(self.conv5(x))
        x = self.conv6(x); x = self.conv7(x); x = self.pool4(self.conv8(x))
        x = self.conv9(x); x = self.conv10(x); x = self.conv11(x); x = self.conv12(x)
        x = self.pool5(self.conv13(x))
        x = self.conv14(x); x = self.conv15(x); x = self.conv16(x); x = self.conv17(x)
        x = self.conv18(x) # [B, 1024, 18, 25]
        
        # 2. Neck 全局感知增强
        x = self.neck(x) # [B, 256, 18, 25]
        
        # 3. 全局平均池化
        x = self.gap(x) # [B, 256, 1, 1]
        x = torch.flatten(x, 1) # [B, 256]
        
        # 4. 解耦回归
        reg_out = self.reg_branch(x)   # [B, 21]
        conf_out = self.conf_branch(x) # [B, 1]
        
        # 5. 合并输出
        out = torch.cat([reg_out, conf_out], dim=1) # [B, 22]
        
        return out

# 为了保持导入兼容性
class Darknet19(Darknet):
    def __init__(self, num_outputs=22):
        super(Darknet19, self).__init__(num_outputs=num_outputs)

# --- 6. 测试代码 ---
if __name__ == '__main__':
    # 模拟输入：BatchSize=2, 通道=3, 高=600, 宽=800
    dummy_input = torch.randn(2, 3, 600, 800)
    
    # 初始化模型
    model = Darknet()
    
    # 打印模型参数量
    total_params = sum(p.numel() for p in model.parameters())
    print(f"模型总参数量: {total_params / 1e6:.2f} M")
    
    # 前向传播测试
    print("开始测试前向传播...")
    with torch.no_grad():
        output = model(dummy_input)
    
    print(f"输入张量维度: {dummy_input.shape}")
    print(f"输出张量维度: {output.shape} <-- 匹配 [B, 22]！")
    
    if output.shape == (2, 22):
        print("测试通过！网络架构符合全局回归设计要求。")
    else:
        print("测试失败！请检查网络层参数配置。")
