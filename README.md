
# UCL-Dehaze: Towards Real-world Image Dehazing via Unsupervised Contrastive Learning

We provide our PyTorch implementation for paper "UCL-Dehaze: Towards Real-world Image Dehazing via Unsupervised Contrastive Learning". 


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

