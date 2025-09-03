import numpy as np
import torch
from .base_model import BaseModel, VGGNet
from . import networks
from .patchnce import PatchNCELoss
import util.util as util


## midas 'depth-concat PatchGAN' discriminator integration 
import torch.nn as nn
import torch.nn.functional as F
# -------- MiDaS official wrapper (isl-org/MiDaS) --------
class MiDaSOfficial(nn.Module):
    """
    Uses torch.hub('isl-org/MiDaS', 'DPT_Large') and official transforms.dpt_transform.
    Expects input as RGB [-1,1] torch tensor [B,3,H,W].
    """
    def __init__(self, model_type='DPT_Large', img_size=384, device='cuda'):
        super().__init__()
        self.device = device
        self.img_size = img_size

        # Try new org first, then legacy hub path for robustness.
        try:
            self.midas = torch.hub.load('isl-org/MiDaS', model_type).to(device).eval()
            self.transforms = torch.hub.load('isl-org/MiDaS', 'transforms')
        except Exception:
            self.midas = torch.hub.load('intel-isl/MiDaS', model_type).to(device).eval()
            self.transforms = torch.hub.load('intel-isl/MiDaS', 'transforms')

        # Official transform for DPT models
        self.transform = self.transforms.dpt_transform

        # Freeze
        for p in self.midas.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def forward(self, x_rgb_m11: torch.Tensor) -> torch.Tensor:
        """
        x_rgb_m11: [-1,1] RGB, [B,3,H,W]
        Return: depth [B,1,H,W], min-max normalized per-sample to [0,1]
        Uses official transform per image (loop).
        """
        B, _, H, W = x_rgb_m11.shape
        # [-1,1] -> [0,1], RGB
        x01 = (x_rgb_m11 + 1.0) * 0.5
        x01 = x01.clamp(0, 1)

        # build input batch via official transform (expects HWC uint/float RGB np)
        inp = []
        for b in range(B):
            img = (x01[b].permute(1,2,0).detach().cpu().numpy() * 255.0).astype(np.uint8)  # RGB uint8
            # official: returns [1,3,384,384] torch.Float
            tin = self.transform(img)  # already resized/normalized for DPT
            inp.append(tin)

        input_batch = torch.cat(inp, dim=0).to(self.device)  # [B,3,384,384]

        # predict depth
        pred = self.midas(input_batch)  # [B,384,384] or [B,1,384,384]
        if pred.dim() == 3:
            pred = pred.unsqueeze(1)     # -> [B,1,h,w]

        depth = F.interpolate(pred, size=(H, W), mode='bicubic', align_corners=False)

        # scale-invariant min-max normalize per sample (to [0,1])
        d_min = depth.amin(dim=(2,3), keepdim=True)
        d_max = depth.amax(dim=(2,3), keepdim=True)
        depth_01 = (depth - d_min) / (d_max - d_min + 1e-6)
        return depth_01

# -------------------------------
# (A) MiDaS 커스텀 forward 추출기
# -------------------------------
class MidasDPTExtractor(nn.Module):
    """
    DPT-Large(dpt_large_384) 공식 transform을 사용하여:
    - 최종 depth (H,W) 1ch  [0,1] 정규화
    - 중간 refine pyramid features (p1..p4) 반환
    를 동시에 수행한다.
    """
    def __init__(self, model_type='DPT_Large', img_size=384, device='cuda', local_ckpt_path=''):
        super().__init__()
        self.device = device
        self.img_size = img_size

        # 1) 모델/트랜스폼 로드 (허브 우선, 실패시 intel-isl로 폴백)
        if local_ckpt_path:
            # ---- 로컬 대안 (옵션) ----
            # 클론을 PYTHONPATH에 추가 후, 아래 import가 성공해야 함:
            # from midas.dpt_depth import DPTDepthModel
            # from midas.transforms import Resize, NormalizeImage, PrepareForNet, Compose
            # 여기서는 간결성을 위해 torch.hub 경로를 권장. (필요시 내가 로컬 버전 코드도 만들어줄게)
            raise NotImplementedError("For local ckpt use, ping me to drop-in a local loader variant.")
        else:
            try:
                self.model = torch.hub.load('isl-org/MiDaS', model_type).to(device).eval()
                self.transforms = torch.hub.load('isl-org/MiDaS', 'transforms')
            except Exception:
                self.model = torch.hub.load('intel-isl/MiDaS', model_type).to(device).eval()
                self.transforms = torch.hub.load('intel-isl/MiDaS', 'transforms')

        # DPT 공식 transform (입력은 numpy HWC RGB 기대)
        self.transform = self.transforms.dpt_transform

        # 동결
        for p in self.model.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def forward(self, x_rgb_m11: torch.Tensor):
        """
        x_rgb_m11: [-1,1] RGB, [B,3,H,W]
        returns:
          depth_01: [B,1,H,W]  (min-max per-sample normalized)
          feats: dict with keys {'p1','p2','p3','p4'} (refine outputs), each [B,C,h,w]
        """
        B, _, H, W = x_rgb_m11.shape

        # [-1,1] -> [0,1], HWC uint8 for official transform
        x01 = (x_rgb_m11 + 1.0) * 0.5
        x01 = x01.clamp(0, 1)

        # build transform batch
        batch = []
        for b in range(B):
            img = (x01[b].permute(1,2,0).detach().cpu().numpy() * 255.0).astype(np.uint8)  # RGB uint8
            tin = self.transform(img)  # [1,3,384,384] float
            batch.append(tin)
        inp = torch.cat(batch, dim=0).to(self.device)  # [B,3,384,384]

        # ----- DPT 내부 경로를 명시적으로 재현 -----
        try:
            from midas.blocks import forward_vit
        except ImportError:
            import sys
            sys.path.append("/data/jhpark/MiDaS")  # 직접 clone 한 루트로 
            from midas.blocks import forward_vit
        
        # 1) encoder (pretrained): 4개 스테이지 이미지형 피처
        l1, l2, l3, l4 = forward_vit(self.model.pretrained, inp)   # shapes: ~1/4, 1/8, 1/16, 1/32

        # 2) reassemble to scratch channels
        s = self.model.scratch
        p4 = s.layer4_rn(l4)       # 가장 깊은 단계
        p3 = s.layer3_rn(l3)
        p2 = s.layer2_rn(l2)
        p1 = s.layer1_rn(l1)

        # 3) refine pyramid (top-down progressive fusion)
        p4 = s.refinenet4(p4)              # 1/32-ish
        p3 = s.refinenet3(p3, p4)          # 1/16-ish
        p2 = s.refinenet2(p2, p3)          # 1/8-ish
        p1 = s.refinenet1(p1, p2)          # 1/4-ish (decoder 최종 feature)

        ## 점검부 
        # print(l1.shape, l2.shape, l3.shape, l4.shape)   # 대략 1/4, 1/8, 1/16, 1/32 해상도
        # print(p1.shape, p2.shape, p3.shape, p4.shape)   # refinenet 경로, 주로 256ch

        out = s.output_conv(p1)            # [B,1,h,w]  (h,w≈inp/2)
        # DPT는 보통 입력의 절반 해상도로 출력하므로, 원 해상도로 업샘플
        depth = F.interpolate(out, size=(H, W), mode='bicubic', align_corners=False)

        # per-sample min-max 정규화
        d_min = depth.amin(dim=(2,3), keepdim=True)
        d_max = depth.amax(dim=(2,3), keepdim=True)
        depth_01 = (depth - d_min) / (d_max - d_min + 1e-6)

        feats = {'p1': p1, 'p2': p2, 'p3': p3, 'p4': p4}  # decoder refine pyramid
        return depth_01, feats

# --------------------------------------
# (B) Depth FPN: p1..p4 -> [B,Cd,H,W]
# --------------------------------------
class DepthFPNFuse(nn.Module):
    """
    DPT refine pyramid(p1..p4)를 FPN 방식으로 융합.
    - 각 p_k를 1x1 Conv -> 채널 정렬
    - top-down 업샘플 + sum
    - 최종 3x3 Conv로 Cd 채널 출력
    """
    def __init__(self, in_ch=256, out_ch=16):
        super().__init__()
        # DPT scratch refinenet 출력 채널은 통상 256 (버전에 따라 다르면 아래 1x1을 조정)
        self.l1 = nn.Conv2d(in_ch, out_ch, 1, 1, 0)
        self.l2 = nn.Conv2d(in_ch, out_ch, 1, 1, 0)
        self.l3 = nn.Conv2d(in_ch, out_ch, 1, 1, 0)
        self.l4 = nn.Conv2d(in_ch, out_ch, 1, 1, 0)
        self.proj = nn.Sequential(
            nn.Conv2d(out_ch, out_ch, 3, 1, 1),
            nn.InstanceNorm2d(out_ch),
            nn.ReLU(True)
        )

    def forward(self, feats: dict, target_hw: tuple):
        """
        feats: {'p1','p2','p3','p4'}
        target_hw: (H,W)  최종 해상도
        returns: depth_feat [B,out_ch,H,W]
        """
        p1, p2, p3, p4 = feats['p1'], feats['p2'], feats['p3'], feats['p4']  # p1가 가장 고해상도(≈1/4)

        # 1x1 정렬
        f1 = self.l1(p1)
        f2 = self.l2(p2)
        f3 = self.l3(p3)
        f4 = self.l4(p4)

        # top-down: f4 -> f3 -> f2 -> f1
        def up(x, like):
            return F.interpolate(x, size=like.shape[-2:], mode='bilinear', align_corners=False)

        u3 = f3 + up(f4, f3)
        u2 = f2 + up(u3, f2)
        u1 = f1 + up(u2, f1)

        # 최종 target(H,W)로 업샘플 + 정리
        H, W = target_hw
        u1 = F.interpolate(u1, size=(H, W), mode='bilinear', align_corners=False)
        out = self.proj(u1)
        return out  # [B,out_ch,H,W]

# ---------------------------------------------------
# (C) DepthFeatConcatD: depth_feat과 RGB concat하는 D
# ---------------------------------------------------
class DepthFeatConcatD(nn.Module):
    """ precomputed depth_feat과 RGB를 concat해서 PatchGAN에 투입 """
    def __init__(self, base_D):
        super().__init__()
        self.D = base_D

    def forward(self, img, depth_feat):
        x = torch.cat([img, depth_feat], dim=1)
        return self.D(x)

# -------- Depth feature aggregator: depth map -> C channels --------
class DepthAgg(nn.Module):
    """ depth map [B,1,H,W] -> depth feature [B,Cd,H,W] """
    def __init__(self, out_ch=16):
        super().__init__()
        c = max(16, out_ch)
        self.enc = nn.Sequential(
            nn.Conv2d(1, c, 3, 1, 1), nn.InstanceNorm2d(c), nn.ReLU(True),
            nn.Conv2d(c, out_ch, 3, 1, 1), nn.InstanceNorm2d(out_ch), nn.ReLU(True),
        )

    def forward(self, d01):
        return self.enc(d01)  # [B,out_ch,H,W]


# -------- D^cat wrapper: concat depth feature then run PatchGAN --------
class DepthConcatD(nn.Module):
    """
    Wraps a PatchGAN D to take concat([img, depth_feat]) as input.
    """
    def __init__(self, base_D, depth_agg):
        super().__init__()
        self.D = base_D
        self.depth_agg = depth_agg

    def forward(self, img, depth01):
        d_feat = self.depth_agg(depth01)
        x = torch.cat([img, d_feat], dim=1)
        return self.D(x)

class UCLModel(BaseModel):
    """ This class implements UCL-Dehaze model

    The code borrows heavily from the PyTorch implementation of CycleGAN, CUT and CWR
    https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix
    https://github.com/taesungp/contrastive-unpaired-translation
    https://github.com/JunlinHan/CWR
    """
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """  Configures options specific for CUT model
        """
        parser.add_argument('--lambda_GAN', type=float, default=1.0, help='weight for GAN loss：GAN(G(X))')
        parser.add_argument('--lambda_NCE', type=float, default=1.0, help='weight for NCE loss: NCE(G(X), X)')
        parser.add_argument('--lambda_IDT', type=float, default=5.0, help='weight for NCE loss: IDT(G(Y), Y)')
        parser.add_argument('--lambda_SCP', type=float, default=0.0002, help='weight for SCP loss: vgg-based pixel-wise perceptual contrastive loss')
        parser.add_argument('--nce_idt', type=util.str2bool, nargs='?', const=True, default=True, help='use NCE loss for identity mapping: NCE(G(Y), Y))')
        parser.add_argument('--nce_layers', type=str, default='0,5,9,13,17', help='compute NCE loss on which layers')
        parser.add_argument('--nce_includes_all_negatives_from_minibatch',
                            type=util.str2bool, nargs='?', const=True, default=False,
                            help='(used for single image translation) If True, include the negatives from the other samples of the minibatch when computing the contrastive loss. Please see models/patchnce.py for more details.')
        parser.add_argument('--netF', type=str, default='mlp_sample', choices=['sample', 'reshape', 'mlp_sample'], help='how to downsample the feature map')
        parser.add_argument('--netF_nc', type=int, default=256)
        parser.add_argument('--nce_T', type=float, default=0.07, help='temperature for NCE loss')
        parser.add_argument('--num_patches', type=int, default=256, help='number of patches per layer')
        parser.add_argument('--flip_equivariance',
                            type=util.str2bool, nargs='?', const=True, default=False,
                            help="Enforce flip-equivariance as additional regularization. It's used by FastCUT, but not UCL-Dehaze")

        ## midas 'depth-concat PatchGAN' discriminator integration 
        ## midas depth model options
        parser.add_argument('--use_depth_cat', type=util.str2bool, nargs='?', const=True, default=False,
                            help='enable Depth-Concat auxiliary discriminator (D^cat)')
        parser.add_argument('--use_depth_cat_intermediate', type=util.str2bool, nargs='?', const=True, default=False,
                            help='use DPT-Large middle features instead of depth map for D^cat')
        parser.add_argument('--lambda_cat', type=float, default=0.3,
                            help='weight for D^cat GAN loss in G update')
        parser.add_argument('--depth_cat_channels', type=int, default=16,
                            help='#channels of aggregated depth features fed to D^cat')
        parser.add_argument('--midas_model', type=str, default='DPT_Large',
                            help='MiDaS model type: DPT_Large (dpt_large_384)')
        parser.add_argument('--midas_resize', type=int, default=384,
                            help='shorter side for MiDaS transform (official dpt is 384)')
        parser.add_argument('--midas_ckpt', type=str, default='',
                            help='(optional) local .pt weight for offline use; if empty, use torch.hub')
        parser.add_argument('--depth_cache_dir', type=str, default='',
                            help='(optional) preload depth (.pt/.npy) if you want to skip on-the-fly MiDaS')

        parser.set_defaults(pool_size=0)  # no image pooling

        opt, _ = parser.parse_known_args()

        # Why 하드코딩? 
        # parser.set_defaults(nce_idt=True, lambda_NCE=1.0)

        return parser

    def __init__(self, opt):
        BaseModel.__init__(self, opt)

        # VGG for perceptual (SCP) loss: device-safe & frozen
        self.vgg = VGGNet().to(self.device).eval()
        for p in self.vgg.parameters():
            p.requires_grad_(False)

        # specify the training losses you want to print out.
        # The training/test scripts will call <BaseModel.get_current_losses>
        # self.loss_names = ['G_GAN', 'D_real', 'D_fake', 'G', 'NCE', 'idt', 'perceptual']
        self.loss_names = ['D_real', 'D_fake', 'G', 'G_GAN', 'perceptual',
                            'scp_raw', 'scp_num', 'scp_den', 'idt', 'NCE']  # loss 로깅 수정 (SCP 분해 로깅 및 순서 변경) 
        self.visual_names = ['real_A', 'fake_B', 'real_B']
        self.nce_layers = [int(i) for i in self.opt.nce_layers.split(',')]

        if opt.nce_idt and self.isTrain:
            self.loss_names += ['NCE_Y']
            self.visual_names += ['idt_B']

        if self.isTrain:
            self.model_names = ['G', 'F', 'D']
        else:  # during test time, only load G
            self.model_names = ['G']

        # define networks (both generator and discriminator)
        self.netG = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.normG, not opt.no_dropout, opt.init_type, opt.init_gain, opt.no_antialias, opt.no_antialias_up, self.gpu_ids, opt)
        self.netF = networks.define_F(opt.input_nc, opt.netF, opt.normG, not opt.no_dropout, opt.init_type, opt.init_gain, opt.no_antialias, self.gpu_ids, opt)

        if self.isTrain:
            self.netD = networks.define_D(opt.output_nc, opt.ndf, opt.netD, opt.n_layers_D, opt.normD, opt.init_type, opt.init_gain, opt.no_antialias, self.gpu_ids, opt)

            # define loss functions
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)
            self.criterionNCE = []

            for nce_layer in self.nce_layers:
                self.criterionNCE.append(PatchNCELoss(opt).to(self.device))

            self.criterionIdt = torch.nn.L1Loss().to(self.device)
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, opt.beta2))
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=opt.lr, betas=(opt.beta1, opt.beta2))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)

            ## midas 'depth-concat PatchGAN' discriminator integration 
            if self.isTrain and getattr(opt, 'use_depth_cat', False):
                # 1) MiDaS official (DPT_Large = dpt_large_384)
                self.midas = MiDaSOfficial(model_type=opt.midas_model, img_size=opt.midas_resize, device=self.device)

                # 2) D^cat: PatchGAN with input_nc = output_nc(=3) + depth_feat_channels
                self.netD_cat = DepthConcatD(
                    base_D = networks.define_D(
                        opt.output_nc + opt.depth_cat_channels, opt.ndf, opt.netD, opt.n_layers_D,
                        opt.normD, opt.init_type, opt.init_gain, opt.no_antialias, self.gpu_ids, opt
                    ).to(self.device),
                    depth_agg = DepthAgg(out_ch=opt.depth_cat_channels).to(self.device)
                ).to(self.device)

                # 3) Optimizer for D^cat
                self.optimizer_D_cat = torch.optim.Adam(self.netD_cat.parameters(), lr=opt.lr, betas=(opt.beta1, opt.beta2))
                self.optimizers.append(self.optimizer_D_cat)

                # 4) logging
                self.loss_names += ['D_cat_real', 'D_cat_fake', 'G_GAN_cat']
                self.model_names.append('D_cat')

            # ---- D^cat(중간 feature 기반) 구성 ----
            if self.isTrain and getattr(opt, 'use_depth_cat_intermediate', False):
                # 1) 커스텀 extractor (DPT-Large 공식 transform + 중간 feat 반환)
                self.midas_extractor = MidasDPTExtractor(
                    model_type=opt.midas_model,
                    img_size=opt.midas_resize,
                    device=self.device,
                    local_ckpt_path=opt.midas_ckpt  # 빈 문자열이면 torch.hub 사용
                )

                # 2) FPN 융합기: p1..p4 -> [B,Cd,H,W]
                self.depth_fpn = DepthFPNFuse(in_ch=256, out_ch=opt.depth_cat_channels).to(self.device)

                # 3) 보조 D (RGB+depth_feat concat)
                #    input_nc = RGB(3) + Cd
                self.netD_cat = DepthFeatConcatD(
                    base_D = networks.define_D(
                        opt.output_nc + opt.depth_cat_channels, opt.ndf, opt.netD, opt.n_layers_D,
                        opt.normD, opt.init_type, opt.init_gain, opt.no_antialias, self.gpu_ids, opt
                    ).to(self.device)
                ).to(self.device)

                # 4) 옵티마이저 + 로깅
                self.optimizer_D_cat = torch.optim.Adam(self.netD_cat.parameters(), lr=opt.lr, betas=(opt.beta1, opt.beta2))
                self.optimizers.append(self.optimizer_D_cat)
                self.loss_names += ['D_cat_real', 'D_cat_fake', 'G_GAN_cat']
                self.model_names.append('D_cat')
    
    ## midas 'depth-concat PatchGAN' discriminator integration 
    ## MiDaS 입력 준비 헬퍼
    def _rgb_m11_to_01(self, t_rgbm11: torch.Tensor) -> torch.Tensor:
        return (t_rgbm11 + 1.0) * 0.5

    def data_dependent_initialize(self, data):
        """
        The feature network netF is defined in terms of the shape of the intermediate, extracted
        features of the encoder portion of netG. Because of this, the weights of netF are
        initialized at the first feedforward pass with some input images.
        Please also see PatchSampleF.create_mlp(), which is called at the first forward() call.
        """
        self.set_input(data)
        # bs_per_gpu = self.real_A.size(0) // max(len(self.opt.gpu_ids), 1)
        # 배치=1, GPU=2이면 0이 되어 이후 텐서가 공집합으로 변하는 현상 방지 
        world = max(len(self.opt.gpu_ids), 1)
        bs_total = self.real_A.size(0)
        bs_per_gpu = max(1, bs_total // world)  # 최소 1 보장

        self.real_A = self.real_A[:bs_per_gpu]
        self.real_B = self.real_B[:bs_per_gpu]
        self.forward()                     # compute fake images: G(A)
        if self.opt.isTrain:
            self.compute_D_loss().backward()                  # calculate gradients for D
            self.compute_G_loss().backward()                   # calculate graidents for G
            if self.opt.lambda_NCE > 0.0:
                self.optimizer_F = torch.optim.Adam(self.netF.parameters(), lr=self.opt.lr, betas=(self.opt.beta1, self.opt.beta2))
                self.optimizers.append(self.optimizer_F)

    def optimize_parameters(self):
        # forward
        self.forward()

        # ---- update D (RGB + optional D^cat) ----
        ## final depth map 만 활용 /또는/ 중간 depth feature map 기반 /모두 공통/ 
        # update D
        self.set_requires_grad(self.netD, True)

        ## midas 'depth-concat PatchGAN' discriminator integration 
        if getattr(self, 'netD_cat', None) is not None:
            self.set_requires_grad(self.netD_cat, True)

        self.optimizer_D.zero_grad()

        ## midas 'depth-concat PatchGAN' discriminator integration 
        if getattr(self, 'optimizer_D_cat', None) is not None:
            self.optimizer_D_cat.zero_grad()

        self.loss_D = self.compute_D_loss()
        self.loss_D.backward()
        self.optimizer_D.step()

        ## midas 'depth-concat PatchGAN' discriminator integration 
        if getattr(self, 'optimizer_D_cat', None) is not None:
            self.optimizer_D_cat.step()

        # ---- update G ----
        self.set_requires_grad(self.netD, False)

        ## midas 'depth-concat PatchGAN' discriminator integration 
        if getattr(self, 'netD_cat', None) is not None:
            self.set_requires_grad(self.netD_cat, False)

        self.optimizer_G.zero_grad()
        # if self.opt.netF == 'mlp_sample':
        # 호출 조건 강화
        if self.opt.netF == 'mlp_sample' and self.opt.lambda_NCE > 0.0 and hasattr(self, 'optimizer_F'):
            self.optimizer_F.zero_grad()

        self.loss_G = self.compute_G_loss()
        self.loss_G.backward()
        self.optimizer_G.step()
        # if self.opt.netF == 'mlp_sample':
        # 호출 조건 강화
        if self.opt.netF == 'mlp_sample' and self.opt.lambda_NCE > 0.0 and hasattr(self, 'optimizer_F'):
            self.optimizer_F.step()

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.
        Parameters:
            input (dict): include the data itself and its metadata information.
        The option 'direction' can be used to swap domain A and domain B.
        """
        AtoB = self.opt.direction == 'AtoB'
        self.real_A = input['A' if AtoB else 'B'].to(self.device)
        self.real_B = input['B' if AtoB else 'A'].to(self.device)
        self.image_paths = input['A_paths' if AtoB else 'B_paths']

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        self.real = torch.cat((self.real_A, self.real_B), dim=0) if self.opt.nce_idt and self.opt.isTrain else self.real_A
        if self.opt.flip_equivariance:
            self.flipped_for_equivariance = self.opt.isTrain and (np.random.random() < 0.5)
            if self.flipped_for_equivariance:
                self.real = torch.flip(self.real, [3])

        self.fake = self.netG(self.real)
        self.fake_B = self.fake[:self.real_A.size(0)]
        if self.opt.nce_idt:
            self.idt_B = self.fake[self.real_A.size(0):]


    def compute_D_loss(self):
        """Calculate GAN loss for the discriminator"""
        fake = self.fake_B.detach()

        # ---- main RGB D ----
        # Fake; stop backprop to the generator by detaching fake_B
        pred_fake = self.netD(fake)
        self.loss_D_fake = self.criterionGAN(pred_fake, False).mean()      # LS_GAN MSE loss
        # Real
        self.pred_real = self.netD(self.real_B)
        loss_D_real = self.criterionGAN(self.pred_real, True)
        self.loss_D_real = loss_D_real.mean()

        # combine loss and calculate gradients
        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5

        ## midas 'depth-concat PatchGAN' discriminator integration 
        # ---- auxiliary D^cat ----
        if getattr(self.opt, 'use_depth_cat', False):
            with torch.no_grad():
                d_fake = self.midas(fake)       # official transform inside
                d_real = self.midas(self.real_B)

            pred_fake_cat = self.netD_cat(fake, d_fake)
            pred_real_cat = self.netD_cat(self.real_B, d_real)
            self.loss_D_cat_fake = self.criterionGAN(pred_fake_cat, False).mean()
            self.loss_D_cat_real = self.criterionGAN(pred_real_cat,  True).mean()
            self.loss_D_cat = 0.5 * (self.loss_D_cat_fake + self.loss_D_cat_real)

            self.loss_D = self.loss_D + self.loss_D_cat

        # ---- 보조 D^cat (MiDaS 중간 feature 기반) ----
        if getattr(self.opt, 'use_depth_cat_intermediate', False):
            with torch.no_grad():
                # 중간 feat 추출 (공식 transform 포함)
                _, feats_fake = self.midas_extractor(self.fake_B.detach())
                _, feats_real = self.midas_extractor(self.real_B)

                # p1..p4 -> [B,Cd,H,W]
                H, W = self.real_B.shape[-2:]
                dfeat_fake = self.depth_fpn(feats_fake, target_hw=(H,W))
                dfeat_real = self.depth_fpn(feats_real, target_hw=(H,W))

            pred_fake_cat = self.netD_cat(self.fake_B.detach(), dfeat_fake.detach())
            pred_real_cat = self.netD_cat(self.real_B,          dfeat_real)

            self.loss_D_cat_fake = self.criterionGAN(pred_fake_cat, False).mean()
            self.loss_D_cat_real = self.criterionGAN(pred_real_cat,  True).mean()
            self.loss_D_cat = 0.5 * (self.loss_D_cat_fake + self.loss_D_cat_real)

            self.loss_D = self.loss_D + self.loss_D_cat

        return self.loss_D

    def compute_G_loss(self):
        """Calculate GAN and NCE loss for the generator"""
        fake = self.fake_B

        # ---- main RGB GAN ----
        ## First, G(A) should fake the discriminator
        if self.opt.lambda_GAN > 0.0:
            pred_fake = self.netD(fake)
            self.loss_G_GAN = self.criterionGAN(pred_fake, True).mean() * self.opt.lambda_GAN
        else:
            self.loss_G_GAN = 0.0

        ## NCE (X->G(X))
        if self.opt.lambda_NCE > 0.0:
            self.loss_NCE = self.calculate_NCE_loss(self.real_A, self.fake_B)    # input & generated
        else:
            self.loss_NCE, self.loss_NCE_bd = 0.0, 0.0

        ## NCE_Y and IDT
        if self.opt.nce_idt and self.opt.lambda_NCE > 0.0:
            # self.loss_NCE_Y = 0
            # 0813 수정사항 
            self.loss_NCE_Y = self.calculate_NCE_loss(self.real_B, self.idt_B)

            loss_NCE_both = (self.loss_NCE + self.loss_NCE_Y)
            self.loss_idt = self.criterionIdt(self.idt_B, self.real_B) * self.opt.lambda_IDT       # G(Y) & clear w.CUT/ w.o. FastCUT
        else:
            loss_NCE_both = self.loss_NCE

            # 안전 가드: nce_idt 비활성 시 idt/NCE_Y는 0으로 정의
            self.loss_idt = torch.tensor(0.0, device=self.device)
            if not hasattr(self, 'loss_NCE_Y'):
                self.loss_NCE_Y = torch.tensor(0.0, device=self.device)

        ## SCP
        # self.loss_perceptual = self.perceptual_loss(self.real_A, self.fake_B, self.real_B) * 0.0002
        self.loss_perceptual = self.perceptual_loss(self.real_A, self.fake_B, self.real_B) * self.opt.lambda_SCP

        # ---- D^cat: encourage fake to be real under depth-concat D ----
        ## midas 'depth-concat PatchGAN' discriminator integration 
        self.loss_G_GAN_cat = torch.tensor(0.0, device=self.device)
        if getattr(self.opt, 'use_depth_cat', False):
            with torch.no_grad():
                d_fake = self.midas(self.fake_B)  # official transform inside
            pred_fake_cat_g = self.netD_cat(self.fake_B, d_fake)
            self.loss_G_GAN_cat = self.criterionGAN(pred_fake_cat_g, True).mean() * self.opt.lambda_cat

        if getattr(self.opt, 'use_depth_cat_intermediate', False):
            with torch.no_grad():
                _, feats_fake = self.midas_extractor(self.fake_B)
                H, W = self.fake_B.shape[-2:]
                dfeat_fake = self.depth_fpn(feats_fake, target_hw=(H,W))

            pred_fake_cat_g = self.netD_cat(self.fake_B, dfeat_fake)
            self.loss_G_GAN_cat = self.criterionGAN(pred_fake_cat_g, True).mean() * self.opt.lambda_cat

        ## midas 'depth-concat PatchGAN' discriminator integration
        # self.loss_G = self.loss_G_GAN + loss_NCE_both + self.loss_idt + self.loss_perceptual
        self.loss_G = self.loss_G_GAN + self.loss_G_GAN_cat + loss_NCE_both + self.loss_idt + self.loss_perceptual
        return self.loss_G

    def perceptual_loss(self, x, y, z):
        """
        x: hazy (real_A), y: dehazed (fake_B), z: clear (real_B)
        """
        # c = torch.nn.MSELoss()
        c = torch.nn.L1Loss()

        # 0813 수정사항 
        # [SCP 안정화]
        # - VGG 입력 정규화 보장: perceptual 계산 직전 x,y,z를 [0,1]로 unnormalize 후 ImageNet mean/std로 normalize하여 VGG에 투입 ?
        # - 분모가 너무 작을 때 ε 안정화: eps = 1e-6
        # - 필요 시 self.loss_perceptual *= 5~10 배 상향(예: 0.001~0.002 수준 목표) → 논문 표 7에서 SCP가 성능에 크게 기여함이 확인됨.

        eps = 1e-6

        # VGG 입력 정규화
        x_vgg = self._prep_for_vgg(x)
        y_vgg = self._prep_for_vgg(y)
        z_vgg = self._prep_for_vgg(z)

        # x,z는 상수 취급 (vgg 파라미터는 동결이므로 메모리 절감 가능)
        with torch.no_grad():
            fx1, fx2, fx3 = self.vgg(x_vgg)   # hazy features (Rh)
            fz1, fz2, fz3 = self.vgg(z_vgg)   # clear features (Rc)

        # y는 G로부터 gradient가 흘러야 하므로 no_grad 금지
        fy1, fy2, fy3 = self.vgg(y_vgg)      # dehazed features (G(x))

        # 비율형 대조 지표 (+ 분모 안정화)
        # m1 = c(fz1, fy1) / (c(fx1, fy1) + eps)
        # m2 = c(fz2, fy2) / (c(fx2, fy2) + eps)
        # m3 = c(fz3, fy3) / (c(fx3, fy3) + eps)

        # loss = 0.4 * m1 + 0.6 * m2 + m3
        # return loss
        num1, den1 = c(fz1, fy1), c(fx1, fy1)
        num2, den2 = c(fz2, fy2), c(fx2, fy2)
        num3, den3 = c(fz3, fy3), c(fx3, fy3)

        # 스케일 가중합 (논문: 0.4/0.6/1.0)
        num = 0.4 * num1 + 0.6 * num2 + 1.0 * num3
        den = 0.4 * den1 + 0.6 * den2 + 1.0 * den3
        raw = num / (den + eps)   # <-- 가중 전(raw) SCP 값

        # ------- 로깅용 값 보관(학습 그래프 분리) -------
        # 'loss_' prefix 사용 → 자동 프린트/저장
        self.loss_scp_raw = raw.detach()
        self.loss_scp_num = num.detach()
        self.loss_scp_den = den.detach()
        # ---------------------------------------------

        return raw


    def _prep_for_vgg(self, t: torch.Tensor) -> torch.Tensor:
        """
        Input:  t in [-1, 1], shape [B,3,H,W], RGB
        Output: normalized for ImageNet VGG, shape [B,3,H,W]
        """
        # [-1,1] -> [0,1]
        t = (t + 1.0) * 0.5
        t = t.clamp(0.0, 1.0)

        # ImageNet mean/std (RGB)
        mean = torch.tensor([0.485, 0.456, 0.406], device=t.device, dtype=t.dtype).view(1, 3, 1, 1)
        std  = torch.tensor([0.229, 0.224, 0.225], device=t.device, dtype=t.dtype).view(1, 3, 1, 1)

        # [0,1] -> ImageNet-normalized
        t = (t - mean) / std
        return t

    def calculate_NCE_loss(self, src, tgt):
        n_layers = len(self.nce_layers)     # CUT: RGB,1,2(downsampling),1,5(resblock)
        feat_q = self.netG(tgt, self.nce_layers, encode_only=True)

        if self.opt.flip_equivariance and self.flipped_for_equivariance:      # Fast CUT (Cycle-GAN default)
            feat_q = [torch.flip(fq, [3]) for fq in feat_q]

        feat_k = self.netG(src, self.nce_layers, encode_only=True)               # feature
        feat_k_pool, sample_ids = self.netF(feat_k, self.opt.num_patches, None)  # MLP 256 patch
        feat_q_pool, _ = self.netF(feat_q, self.opt.num_patches, sample_ids)     # MLP

        total_nce_loss = 0.0
        for f_q, f_k, crit, nce_layer in zip(feat_q_pool, feat_k_pool, self.criterionNCE, self.nce_layers):
            loss = crit(f_q, f_k) * self.opt.lambda_NCE
            total_nce_loss += loss.mean()

        return total_nce_loss / n_layers



