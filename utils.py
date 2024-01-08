from PIL import Image
import numpy as np
import torch
from torch.autograd import Function

######################################## learning-related ########################################

def set_requires_grad(model, requires_grad=True):
    for param in model.parameters():
        param.requires_grad = requires_grad


def loop_iterable(iterable):
    while True:
        yield from iterable


class GrayscaleToRgb:
    """Convert a grayscale image to rgb"""
    def __call__(self, image):
        image = np.array(image)
        image = np.dstack([image, image, image])
        return Image.fromarray(image)


class GradientReversalFunction(Function):
    """
    Gradient Reversal Layer from:
    Unsupervised Domain Adaptation by Backpropagation (Ganin & Lempitsky, 2015)

    Forward pass is the identity function. In the backward pass,
    the upstream gradients are multiplied by -lambda (i.e. gradient is reversed)
    """

    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.clone()

    @staticmethod
    def backward(ctx, grads):
        lambda_ = ctx.lambda_
        lambda_ = grads.new_tensor(lambda_)
        dx = -lambda_ * grads
        return dx, None

class GradientReversal(torch.nn.Module):
    def __init__(self, lambda_=1):
        super(GradientReversal, self).__init__()
        self.lambda_ = lambda_

    def forward(self, x):
        return GradientReversalFunction.apply(x, self.lambda_)


######################################## resnet-related ########################################

# for resnet only
def feature_extractor_resnet(model,inputs):
    conv_features = None
    for name, layer in model.named_children():
        # print(name)
        if name == 'fc':  # Stop before the classifier
            break
        inputs = layer(inputs)
        if name == 'avgpool':  # Save the output of the last convolutional layer
            conv_features = inputs   
    features = conv_features.view(conv_features.size(0),-1)
    return features

from torchvision.models.resnet import resnet18
# for resnet18 only
def feature_extractor_resnet18(model:resnet18,inputs):
    inputs = model.conv1(inputs)
    inputs = model.bn1(inputs)
    inputs = model.relu(inputs)
    inputs = model.maxpool(inputs)
    inputs = model.layer1(inputs)
    inputs = model.layer2(inputs)
    inputs = model.layer3(inputs)
    inputs = model.layer4(inputs)
    inputs = model.avgpool(inputs)
    features = inputs.view(inputs.size(0),-1)
    return features

# for resnet only
def classifier(model,features):
    outputs = model.fc(features)
    _,predicted = torch.max(outputs,axis = 1)
    return predicted

######################################## plot-related ########################################

import seaborn as sns
from sklearn.metrics import confusion_matrix
from matplotlib import pyplot as plt

def plot_cm(labels, pre, savepath='./temp_cm'):
    conf_numpy = confusion_matrix(labels, pre)
    conf_numpy = conf_numpy.astype('float') / conf_numpy.sum(axis=1)
    conf_numpy_norm = np.around(conf_numpy, decimals=2)
    plt.figure(figsize=(8, 7))
    sns.heatmap(conf_numpy_norm, annot=True, cmap="Blues")
    plt.title('confusion matrix', fontsize=15)
    plt.ylabel('True labels', fontsize=14)
    plt.xlabel('Predict labels', fontsize=14)
    plt.tight_layout()
    plt.savefig(savepath+'.png')
    plt.savefig(savepath+'.eps')

import itertools
def plot_confusion_matrix(labels, pre, classes, savepath='./temp_cm', normalize=False, title='Confusion matrix', cmap=plt.cm.Blues,fontsize=20):
    conf_numpy = confusion_matrix(labels, pre)
    if normalize:
        conf_numpy = conf_numpy.astype('float') / conf_numpy.sum(axis = 1)
        conf_numpy = np.around(conf_numpy,decimals=2)
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')
    print(conf_numpy)

    plt.figure(figsize=(8, 7))
    plt.imshow(conf_numpy, interpolation='nearest', cmap=cmap)
    plt.title(title, fontsize=fontsize)
    cbar = plt.colorbar()
    cbar.ax.tick_params(labelsize=fontsize)

    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45, fontsize=fontsize)
    plt.yticks(tick_marks, classes, fontsize=fontsize)

    fmt = '.2f' if normalize else 'd'
    thresh = conf_numpy.max() / 2.
    for i, j in itertools.product(range(conf_numpy.shape[0]), range(conf_numpy.shape[1])):
        plt.text(j, i, format(conf_numpy[i, j], fmt),
                 horizontalalignment="center",
                 fontsize=fontsize,
                 color="white" if conf_numpy[i, j] > thresh else "black")

    plt.ylabel('True label', fontsize=fontsize)
    plt.xlabel('Predicted label', fontsize=fontsize)
    plt.tight_layout()
    plt.savefig(savepath+'.png')
    plt.savefig(savepath+'.eps')

def plot_tsne(tsne_result, labels, classes, savepath='./temp_tsne',title = 't-SNE Visualization',legend=True):
    plt.figure(figsize=(8, 7))
    unique_labels = np.unique(labels)
    scatter = plt.scatter(tsne_result[:, 0], tsne_result[:, 1], c=labels, cmap='viridis')

    if legend:
        class_colors = plt.cm.viridis(np.linspace(0, 1, len(unique_labels)))
        handles = [plt.Line2D([0], [0], marker='o', linestyle='None', color=class_colors[i], markerfacecolor=class_colors[i], markersize=10, label=classes[i]) for i in unique_labels]
        plt.legend(handles=handles, title='Classes')

    plt.title('t-SNE Visualization')
    plt.tight_layout()
    plt.savefig(savepath+'.png')
    plt.savefig(savepath+'.eps')

def plot_tsne_v2(tsne_result_sim, labels_sim, tsne_result_real, labels_real, classes,  savepath='./temp_tsne', title = 't-SNE Visualization', legend=True):
    plt.figure(figsize=(8, 7))
    unique_labels = np.unique(labels_sim+labels_real)
    plt.scatter(tsne_result_sim[:, 0], tsne_result_sim[:, 1], c=labels_sim, cmap='viridis',marker='*')
    plt.scatter(tsne_result_real[:, 0], tsne_result_real[:, 1], c=labels_real, cmap='viridis',marker='o')

    if legend:
        class_colors = plt.cm.viridis(np.linspace(0, 1, len(unique_labels)))
        handles1 = [plt.Line2D([0], [0], marker='*', linestyle='None', color=class_colors[i], markerfacecolor=class_colors[i], markersize=10, label=classes[i]+" (sim)") for i in unique_labels]
        handels2 = [plt.Line2D([0], [0], marker='o', linestyle='None', color=class_colors[i], markerfacecolor=class_colors[i], markersize=10, label=classes[i]+" (real)") for i in unique_labels]
        handles=handles1+handels2
        plt.legend(handles=handles, title='Classes')

    plt.title(title)
    plt.tight_layout()
    plt.savefig(savepath+'.png')
    plt.savefig(savepath+'.eps')