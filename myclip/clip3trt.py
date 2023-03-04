# https://github.com/OFA-Sys/Chinese-CLIP
import torch
import cn_clip.clip as clip
from cn_clip.deploy.tensorrt_utils import TensorRTModel
from cn_clip.clip.utils import _MODEL_INFO, image_transform


img_trt_model_path="./models/vit-b-16.img.fp16.trt"
txt_trt_model_path="./models/vit-b-16.txt.fp16.trt"
model_arch = "ViT-B-16"


def setup(labels):
    global text_features, preprocess, img_trt_model
    
    # 文本模型
    txt_trt_model = TensorRTModel(txt_trt_model_path)
    # 计算文本特征
    text = clip.tokenize(labels).cuda()
    text_features = []
    for i in range(len(text)):
        text_feature = txt_trt_model(inputs={'text': torch.unsqueeze(text[i], dim=0)})['unnorm_text_features']
        text_features.append(text_feature)
    text_features = torch.squeeze(torch.stack(text_features), dim=1)
    text_features = text_features / text_features.norm(dim=1, keepdim=True) # 归一化

    # 图像模型
    preprocess = image_transform(_MODEL_INFO[model_arch]['input_resolution'])
    img_trt_model = TensorRTModel(img_trt_model_path)


def predict(img):

    image = preprocess(img).unsqueeze(0).cuda()
    
    with torch.no_grad(), torch.cuda.amp.autocast():
        image_features = img_trt_model(inputs={'image': image})['unnorm_image_features']
        image_features /= image_features.norm(dim=1, keepdim=True)

        logits_per_image = 100 * image_features @ text_features.t()
        probs = logits_per_image.softmax(dim=-1).cpu().tolist()[0]

    max_p = max(probs)
    max_i = probs.index(max_p)

    return max_i, max_p
