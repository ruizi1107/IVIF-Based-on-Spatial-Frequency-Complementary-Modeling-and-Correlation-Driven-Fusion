import os, sys, time, datetime
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import kornia

from net import Restormer_Encoder, Restormer_Decoder, BaseFeatureExtraction, DetailFeatureExtraction
from utils.dataset import H5Dataset
from utils.loss import Fusionloss, cc

# ------------------------------
# 训练配置
# ------------------------------
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
device = 'cuda' if torch.cuda.is_available() else 'cpu'

num_epochs = 120         
epoch_gap  = 40           
batch_size = 8
lr = 1e-4
weight_decay = 0.0
optim_step  = 20
optim_gamma = 0.5


alpha_5 = 2.0  
coeff_mse_loss_VF = 1. # alpha1
coeff_mse_loss_IF = 1.
coeff_decomp = 2.      # alpha2 and alpha4
coeff_tv = 5.
decomp_delta = 1.001
clip_grad_norm_value = 0.01

# ------------------------------
# 构建模型
# ------------------------------
DIDF_Encoder = nn.DataParallel(Restormer_Encoder()).to(device)
DIDF_Decoder = nn.DataParallel(Restormer_Decoder()).to(device)
BaseFuseLayer   = nn.DataParallel(BaseFeatureExtraction(dim=64)).to(device)
DetailFuseLayer = nn.DataParallel(DetailFeatureExtraction(in_c=64, out_c=64)).to(device)


# 优化器 & 调度器
optimizer1 = torch.optim.Adam(DIDF_Encoder.parameters(), lr=lr, weight_decay=weight_decay)
optimizer2 = torch.optim.Adam(DIDF_Decoder.parameters(), lr=lr, weight_decay=weight_decay)
optimizer3 = torch.optim.Adam(BaseFuseLayer.parameters(), lr=lr, weight_decay=weight_decay)
optimizer4 = torch.optim.Adam(DetailFuseLayer.parameters(), lr=lr, weight_decay=weight_decay)

scheduler1 = torch.optim.lr_scheduler.StepLR(optimizer1, step_size=optim_step, gamma=optim_gamma)
scheduler2 = torch.optim.lr_scheduler.StepLR(optimizer2, step_size=optim_step, gamma=optim_gamma)
scheduler3 = torch.optim.lr_scheduler.StepLR(optimizer3, step_size=optim_step, gamma=optim_gamma)
scheduler4 = torch.optim.lr_scheduler.StepLR(optimizer4, step_size=optim_step, gamma=optim_gamma)

# ------------------------------
# 构建损失
# ------------------------------
MSELoss = nn.MSELoss()  
L1Loss = nn.L1Loss()
Loss_ssim = kornia.losses.SSIMLoss(11, reduction='mean')
FusionLossStage2 = Fusionloss()  


train_h5 = "data/MSRS_train_imgsize_128_stride_200.h5"
assert os.path.isfile(train_h5), f"找不到训练数据：{train_h5}，请先运行 dataprocessing.py 生成 h5"
trainloader = DataLoader(H5Dataset(train_h5), batch_size=batch_size, shuffle=True, num_workers=0)
loader = {'train': trainloader}

# ------------------------------
# 训练循环
# ------------------------------
timestamp = datetime.datetime.now().strftime("%m-%d-%H-%M")
torch.backends.cudnn.benchmark = True
prev_time = time.time()

for epoch in range(num_epochs):
    DIDF_Encoder.train(); DIDF_Decoder.train(); BaseFuseLayer.train(); DetailFuseLayer.train()
    for i, (data_VIS, data_IR) in enumerate(loader['train']):
        data_VIS = data_VIS.to(device)
        data_IR  = data_IR.to(device)

        optimizer1.zero_grad(set_to_none=True)
        optimizer2.zero_grad(set_to_none=True)
        optimizer3.zero_grad(set_to_none=True)
        optimizer4.zero_grad(set_to_none=True)

        # ============== Stage-I（重建）================
        if epoch < epoch_gap:
            feature_V_B, feature_V_D, _ = DIDF_Encoder(data_VIS)
            feature_I_B, feature_I_D, _ = DIDF_Encoder(data_IR)
            data_VIS_hat, _ = DIDF_Decoder(data_VIS, feature_V_B, feature_V_D)
            data_IR_hat, _ = DIDF_Decoder(data_IR, feature_I_B, feature_I_D)

            cc_loss_B = cc(feature_V_B, feature_I_B)
            cc_loss_D = cc(feature_V_D, feature_I_D)
            mse_loss_V = 5 * Loss_ssim(data_VIS, data_VIS_hat) + MSELoss(data_VIS, data_VIS_hat)
            mse_loss_I = 5 * Loss_ssim(data_IR, data_IR_hat) + MSELoss(data_IR, data_IR_hat)

            Gradient_loss = L1Loss(kornia.filters.SpatialGradient()(data_VIS),
                                   kornia.filters.SpatialGradient()(data_VIS_hat))
            loss_decomp =  (cc_loss_D) ** 2/ (1.01 + cc_loss_B)  

            loss = coeff_mse_loss_VF * mse_loss_V + coeff_mse_loss_IF * mse_loss_I + \
                    coeff_decomp * loss_decomp + coeff_tv * Gradient_loss 
            
             
            loss.backward()
            nn.utils.clip_grad_norm_(DIDF_Encoder.parameters(), max_norm=clip_grad_norm_value)
            nn.utils.clip_grad_norm_(DIDF_Decoder.parameters(), max_norm=clip_grad_norm_value)
            optimizer1.step(); optimizer2.step()

        # ============== Stage-II（融合）================
        else:
            feat_V_B, feat_V_D, _ = DIDF_Encoder(data_VIS)
            feat_I_B, feat_I_D, _ = DIDF_Encoder(data_IR)
            fuse_B = BaseFuseLayer(feat_V_B + feat_I_B)
            fuse_D = DetailFuseLayer(feat_V_D + feat_I_D)

            fused, _ = DIDF_Decoder(data_VIS, fuse_B, fuse_D) 

            fusionloss, loss_int, loss_grad = FusionLossStage2(data_VIS, data_IR, fused) 
            
            cc_B = cc(feat_V_B, feat_I_B)
            cc_D = cc(feat_V_D, feat_I_D)
            loss_decomp = (cc_D ** 2) / (cc_B + decomp_delta) 

            loss = fusionloss + alpha_5*loss_decomp
            

            loss.backward()
            for m in [DIDF_Encoder, DIDF_Decoder, BaseFuseLayer, DetailFuseLayer]:
                nn.utils.clip_grad_norm_(m.parameters(), max_norm=clip_grad_norm_value)

            optimizer1.step(); optimizer2.step(); optimizer3.step(); optimizer4.step()

        # ---- ETA 打印 ----
        batches_done = epoch * len(loader['train']) + i
        batches_left = num_epochs * len(loader['train']) - batches_done
        time_left = datetime.timedelta(seconds=batches_left * (time.time() - prev_time))
        prev_time = time.time()
        sys.stdout.write(
            "\r[Epoch %d/%d] [Batch %d/%d] [loss: %.6f] ETA: %s"
            % (epoch, num_epochs, i, len(loader['train']), float(loss.item()), str(time_left))
        )

    # ---- 调度器步进 ----
    scheduler1.step(); scheduler2.step()
    if epoch >= epoch_gap:
        scheduler3.step(); scheduler4.step()

    # 防止 LR 过低
    for opt in [optimizer1, optimizer2, optimizer3, optimizer4]:
        if opt.param_groups[0]['lr'] <= 1e-6:
            opt.param_groups[0]['lr'] = 1e-6

# ---- 保存权重 ----
os.makedirs("models", exist_ok=True)
ckpt = {
    'DIDF_Encoder':   DIDF_Encoder.state_dict(),
    'DIDF_Decoder':   DIDF_Decoder.state_dict(),
    'BaseFuseLayer':  BaseFuseLayer.state_dict(),
    'DetailFuseLayer':DetailFuseLayer.state_dict(),
}
torch.save(ckpt, os.path.join("models", f"TSFI_on_CDDFuse_{timestamp}.pth"))
print("\n[OK] Saved:", os.path.join("models", f"TSFI_on_CDDFuse_{timestamp}.pth"))
