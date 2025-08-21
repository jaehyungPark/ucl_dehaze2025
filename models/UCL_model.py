import numpy as np
import torch
from .base_model import BaseModel, VGGNet
from . import networks
from .patchnce import PatchNCELoss
import util.util as util

# pretrained VGG16 module set in evaluation mode for feature extraction
# vgg = VGGNet().cuda().eval()
    # moved into UCLModel.__init__ with proper device & freezing


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

        # update D
        self.set_requires_grad(self.netD, True)
        self.optimizer_D.zero_grad()
        self.loss_D = self.compute_D_loss()
        self.loss_D.backward()
        self.optimizer_D.step()

        # update G
        self.set_requires_grad(self.netD, False)
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
        # Fake; stop backprop to the generator by detaching fake_B
        pred_fake = self.netD(fake)
        self.loss_D_fake = self.criterionGAN(pred_fake, False).mean()      # LS_GAN MSE loss
        # Real
        self.pred_real = self.netD(self.real_B)
        loss_D_real = self.criterionGAN(self.pred_real, True)
        self.loss_D_real = loss_D_real.mean()

        # combine loss and calculate gradients
        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5
        return self.loss_D

    def compute_G_loss(self):
        """Calculate GAN and NCE loss for the generator"""
        fake = self.fake_B
        # First, G(A) should fake the discriminator
        if self.opt.lambda_GAN > 0.0:
            pred_fake = self.netD(fake)
            self.loss_G_GAN = self.criterionGAN(pred_fake, True).mean() * self.opt.lambda_GAN
        else:
            self.loss_G_GAN = 0.0

        if self.opt.lambda_NCE > 0.0:
            self.loss_NCE = self.calculate_NCE_loss(self.real_A, self.fake_B)    # input & generated
        else:
            self.loss_NCE, self.loss_NCE_bd = 0.0, 0.0

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

        # self.loss_perceptual = self.perceptual_loss(self.real_A, self.fake_B, self.real_B) * 0.0002
        self.loss_perceptual = self.perceptual_loss(self.real_A, self.fake_B, self.real_B) * self.opt.lambda_SCP

        self.loss_G = self.loss_G_GAN + loss_NCE_both + self.loss_idt + self.loss_perceptual
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



