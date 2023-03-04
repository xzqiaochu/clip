# https://github.com/openai/CLIP
import torch
import clip


device = "cuda" if torch.cuda.is_available() else "cpu"


def setup(labels):
    loadModel()
    calcText(labels)


def loadModel():
    global model, preprocess
    model, preprocess = clip.load("ViT-B/32", download_root='./models/')


def calcText(labels):
    global text_features

    # 使用英文标签
    labels = [x[2] for x in labels]

    text = clip.tokenize(labels).to(device)
    
    with torch.no_grad():
        text_features = model.encode_text(text)
        text_features /= text_features.norm(dim=1, keepdim=True)


def predict(img):

    image = preprocess(img).unsqueeze(0).to(device)

    with torch.no_grad(), torch.cuda.amp.autocast():

        image_features = model.encode_image(image)
        image_features /= image_features.norm(dim=1, keepdim=True)

        logit_scale = model.logit_scale.exp()
        logits_per_image = logit_scale * image_features @ text_features.t()

        probs = logits_per_image.softmax(dim=-1).cpu().tolist()[0]

    max_p = max(probs)
    max_i = probs.index(max_p)

    return max_i, max_p
