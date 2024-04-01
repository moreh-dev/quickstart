import glob

import clip
import torch
import tqdm
from torchmetrics.image.fid import FrechetInceptionDistance

from .img_utils import img_to_tensor

IMAGE_EXTENSIONS = {'bmp', 'jpg', 'jpeg', 'pgm', 'png', 'ppm', 'tif', 'tiff', 'webp'}
TEXT_EXTENSIONS = {'txt'}


class TextToImageEvalulator():

    def __init__(self, output_dir, real_dir=None, img_size=(64, 64), use_clip_transform=True):
        self.output_dir = output_dir
        self.real_dir = real_dir
        self.img_size = img_size
        self.clip_model = None

        self.output_img_tensors = None
        self.real_img_tensors = None
        self.use_clip_transform = use_clip_transform

    def eval_func(self, func):
        self.update_img_tensors()
        func()
        #some drawing functions will be added below?..

    @staticmethod
    def read_img_tensors_from_dir(dir, exts=IMAGE_EXTENSIONS, transform=None):
        img_tensors = [img_to_tensor(path, transform=transform)\
                    for path in glob.glob(f"{dir}/*")\
                    if path.rsplit('.',1)[-1] in exts]
        img_tensor = torch.vstack(img_tensors)
        return img_tensor

    def update_img_tensors(self):
        if self.output_img_tensors is None:
            self.output_img_tensors = TextToImageEvalulator.read_img_tensors_from_dir(self.output_dir)
        if self.real_img_tensors is None:
            self.real_img_tensors = TextToImageEvalulator.read_img_tensors_from_dir(self.real_dir)

    @eval_func
    def FID_score(self, feature=2048):
        fid = FrechetInceptionDistance(feature=feature)
        fid.update(self.output_img_tensors, real=False)
        fid.update(self.real_img_tensors, real=True)
        return fid.compute()

    @eval_func
    def CLIP_score(
        self,
        text_img_loader,
        clip_model='ViT-B/32',
    ):
        if self.clip_model is None:
            print('Loading CLIP model: {}'.format(clip_model))
            self.clip_model, self.clip_preprocess = clip.load(clip_model, device='cuda')

        score_acc = 0.
        sample_num = 0.
        logit_scale = self.clip_model.logit_scale.exp()

        for batch_data in tqdm(text_img_loader):
            img, text = batch_data
            img_features = self.clip_model.encode_image(img.cuda())
            text_features = self.clip_model.encode_text(text.cuda(), 'txt')

            # normalize features
            img_features = img_features / img_features.norm(dim=1, keepdim=True).to(torch.float32)
            text_features = text_features / text_features.norm(dim=1, keepdim=True).to(torch.float32)

            # calculate scores
            score = logit_scale * (img_features * text_features).sum()
            score_acc += score
            sample_num += img.shape[0]

        return (score_acc / sample_num).cpu().item()

    # TBD
    def get_clip_loader(self):
        raise NotImplementedError
