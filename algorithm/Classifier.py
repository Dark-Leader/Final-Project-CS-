import torch
from torch import nn
import numpy as np


class Classifier:

    def __init__(self, model, classes, device):
        self.class_name_to_idx = classes
        self.idx_to_name = {value: key for key, value in classes.items()}
        self.model = model
        self.device = device

    def detect(self, images_loader):
        res = []
        self.model.eval()
        with torch.no_grad(): # not updating weights
            for data, target in images_loader:
                data = data.to(self.device)
                output = self.model(data)
                probs = nn.functional.softmax(output, dim=1).cpu().numpy()[0]
                prediction = output.max(1, keepdim=True)[1] # get predictions
                prob = probs[prediction]
                if prob < 0.5: # in case we get a symbol which the model didn't train on we want to ignore it.
                    res.append(np.array([[-1]], dtype=np.int64))
                else:
                    res.append(prediction.cpu().numpy()) # move predictions from GPU memory to CPU.

        res = np.concatenate(np.asarray(res, dtype=object)).ravel() # flatten the predictions to a 1d array
        return self.translate(res)

    def translate(self, predictions):
        res = [self.idx_to_name[prediction] for prediction in predictions]
        return res