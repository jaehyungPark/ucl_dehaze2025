# UCL-Dehaze: Towards Real-world Image Dehazing via Unsupervised Contrastive Learning (TIP'2024)

Authors: Yongzhen Wang, Xuefeng Yan, Fu Lee Wang, Haoran Xie, Wenhan Yang, Xiao-Ping Zhang, Jing Qin, and Mingqiang Wei

[[Paper Link]](https://ieeexplore.ieee.org/document/10431709)

### Abstract

While the wisdom of training an image dehazing model on synthetic hazy data can alleviate the difficulty of collecting real-world hazy/clean image pairs, it brings the well-known domain shift problem. From a different yet new perspective, this paper explores contrastive learning with an adversarial training effort to leverage unpaired real-world hazy and clean images, thus alleviating the domain shift problem and enhancing the network’s generalization ability in real-world scenarios. We propose an effective unsupervised contrastive learning paradigm for image dehazing, dubbed UCL-Dehaze. Unpaired real-world clean and hazy images are easily captured, and will serve as the important positive and negative samples respectively when training our UCL-Dehaze network. To train the network more effectively, we formulate a new self-contrastive perceptual loss function, which encourages the restored images to approach the positive samples and keep away from the negative samples in the embedding space. Besides the overall network architecture of UCL-Dehaze, adversarial training is utilized to align the distributions between the positive samples and the dehazed images. Compared with recent image dehazing works, UCL-Dehaze does not require paired data during training and utilizes unpaired positive/negative data to better enhance the dehazing performance. We conduct comprehensive experiments to evaluate our UCL-Dehaze and demonstrate its superiority over the state-of-the-arts, even only 1,800 unpaired real-world images are used to train our network. Source code is publicly available at https://github.com/yz-wang/UCL-Dehaze.

#### If you find the resource useful, please cite the following :- )

```
@article{Wang_2024_TIP,
  author={Wang, Yongzhen and Yan, Xuefeng and Wang, Fu Lee and Xie, Haoran and Yang, Wenhan and Zhang, Xiao-Ping and Qin, Jing and Wei, Mingqiang},
  journal={IEEE Transactions on Image Processing}, 
  title={UCL-Dehaze: Toward Real-World Image Dehazing via Unsupervised Contrastive Learning}, 
  year={2024},
  volume={33},
  number={},
  pages={1361-1374},
  doi={10.1109/TIP.2024.3362153}}
```  

## Prerequisites
Python 3.6 or above.

For packages, see requirements.txt.

### Getting started


- Install PyTorch 1.6 or above and other dependencies (e.g., torchvision, visdom, dominate, gputil).

  For pip users, please type the command `pip install -r requirements.txt`.

  For Conda users,  you can create a new Conda environment using `conda env create -f environment.yml`.
  
### UCL-Dehaze Training and Test

- A one image train/test example is provided.

- To view training results and loss plots, run `python -m visdom.server` and click the URL http://localhost:8097.

- Train the UCL-Dehaze model:
```bash
python train.py --dataroot ./datasets/hazy2clear --name dehaze
```
The checkpoints will be stored at `./checkpoints/dehaze/web`.

- Test the UCL-Dehaze model:
```bash
python test.py --dataroot ./datasets/hazy2clear --name dehaze --preprocess scale_width
```
The test results will be saved to an html file here: `./results/dehaze/latest_test/index.html`.


### Acknowledgments
Our code is developed based on [pytorch-CycleGAN-and-pix2pix](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix) , [CWR](https://github.com/JunlinHan/CWR) and [CUT](http://taesung.me/ContrastiveUnpairedTranslation/). We thank the awesome work provided by CycleGAN, CWR and CUT.
And great thanks to the anonymous reviewers for their helpful feedback.

