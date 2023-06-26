#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 10 00:28:47 2022

@author: k4cp3rskiii
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

import warnings

warnings.filterwarnings("ignore")

from sklearn.metrics import classification_report

from termcolor import cprint
import os

from PIL import Image
# Pytorch imports
import torch
import torch.nn as nn
import torch.nn.functional as F
from prettytable.colortable import ColorTable, Theme

from torch.autograd import Variable
from torchvision.datasets import KMNIST
from torchvision.transforms import ToTensor
import torch.optim as optim
from torch.utils.data import random_split

from torchvision import transforms, utils, datasets
import pathlib
import time
from tqdm import tqdm
# from tqdm.gui import tqdm

import Hubbard_aux as aux
from Loaders import Importer
from Models import CNN_Upgrade
from ThermEncoding import data_to_therm


# %%
class SaveBestModel:
    """
    Class to save the best model while training. If the current epoch's
    validation loss is less than the previous least less, then save the
    model state.
    """

    def __init__(
            self, best_valid_loss=float('inf')
    ):
        self.best_valid_loss = best_valid_loss

    def __call__(
            self, current_valid_loss,
            model, verb=False,
            checkpoint_best="upgraded_model_x"
    ):
        if current_valid_loss < self.best_valid_loss:
            self.best_valid_loss = current_valid_loss
            if verb:
                cprint("[INFO]", "magenta", end=" ")
                print(f"\nBest validation loss: {self.best_valid_loss}")
                cprint("[INFO]", "magenta", end=" ")
                print(f"\nSaving new best model\n")
            torch.save(model.state_dict(), f'Models/{checkpoint_best}_checkpoint.dict')


class EarlyStopper:
    def __init__(self, patience=1, min_delta=0, warmup=100):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = np.inf
        self.warmup = warmup

    def early_stop(self, validation_loss, epoch):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience and epoch >= self.warmup:
                return True
        return False


def count_parameters(model, device):
    th = Theme(
        default_color="36",
        vertical_color="32",
        horizontal_color="32",
        junction_color="36",
    )
    th_2 = Theme(
        default_color="92",
        vertical_color="32",
        horizontal_color="32",
        junction_color="36",
    )
    table = ColorTable(["Modules", "Parameters"], theme=th)
    table_2 = ColorTable(["Additional stats", ""], theme=th_2)
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad: continue
        params = parameter.numel()
        table.add_row([name, f"{params:,}"])
        total_params += params
    print(table)

    # total parameters and trainable parameters
    total_params = sum(p.numel() for p in model.parameters())
    total_trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad)
    table_2.add_row(["Total parameters", f"{total_params:,}"])
    table_2.add_row(["Trainable parameters", f"{total_trainable_params:,}"])
    table_2.add_row(["Computation device", device])
    print(table_2)
    return total_params


def train_model(model,
                epoch_count,
                trainDataLoader,
                valDataLoader,
                optimizer,
                batch_size,
                classes_no,
                device,
                print_all=True,
                therm=False,
                therm_levels=100,
                checkpoint_best=None,
                early_stopping=None,
                scheduler=None,
                ):
    """
    CNN training function

    Parameters
    ----------
    model : TYPE
        DESCRIPTION.
    epoch_count : TYPE
        DESCRIPTION.
    trainDataLoader : TYPE
        DESCRIPTION.
    valDataLoader : TYPE
        DESCRIPTION.
    optimizer : TYPE
        DESCRIPTION.
    batch_size : TYPE
        DESCRIPTION.
    classes_no : TYPE
        DESCRIPTION.
    print_all : TYPE, optional
        DESCRIPTION. The default is True.
    therm : bool, optional
        Should the model use Thermometer encoding, as defined in [1].
        If set to True, then therm_levels defines the thermometer encoding
        bin count. The default is False.
    therm_levels : int, optional
        Sets the number of bins for thermometer encoding. The default is 100.
    checkpoint_best : str, optional
        If not None, then it should be a filename of the checkpointed model.
        The naming convention is `Models/{checkpoint_best}_checkpoint.dict.
        The default is None.
    early_stopping : dict, optional
        If not `None`, then the training will stop if after early_stopping['patience']
        number of epochs there is no improvement in validation loss,
        with tolerance range defined by early_stopping['tolerance'].
        The default is None.
    scheduler : torch.optim.lr_scheduler, optional
        If not 'None', then the learning rate will be adjusted dynamically, according to its configuration.
        
    References
    ----------
    .. [1] Buckman, Jacob, Aurko Roy, Colin Raffel, and Ian Goodfellow.
    "Thermometer encoding: One hot way to resist adversarial examples."
    In International Conference on Learning Representations. 2018.


    Returns
    -------
    H : dict
        Training history dictionary.

    """
    # initialize a dictionary to store training history
    H = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": []
    }

    trainSteps = len(trainDataLoader.dataset) // batch_size
    valSteps = len(valDataLoader.dataset) // batch_size

    cprint("\n[INFO]", "magenta", end=" ")
    print("Training the network...")
    startTime = time.time()

    cprint("\n======================================", "green")
    cprint("          Nerd stats section\n", "green")
    count_parameters(model, device)
    cprint("\n======================================\n", "green")

    if checkpoint_best is not None:
        # initialize SaveBestModel class
        save_best_model = SaveBestModel()

    if early_stopping is not None:
        early_stopper = EarlyStopper(
            patience=early_stopping['patience'],
            min_delta=early_stopping['tolerance'],
            warmup=early_stopping['warmup']
            )

    for e in tqdm(range(0, epoch_count), colour="green", unit="epoch"):
        # set the training mode in the model
        model.train()

        # Initialize total traininig and validation loss
        totalTrainLoss = 0
        totalValLoss = 0

        # Initialize the number of correct predictons in the training and validation step
        trainCorrect = 0
        valCorrect = 0

        # Loop over the training set
        for batch_idx, (x, y) in enumerate(trainDataLoader):
            y = F.one_hot(y, num_classes=classes_no)
            y = y.float()

            (x, y) = (x.to(device), y.to(device))

            # Perform a forward pass and calculate the training loss
            pred = model(x)
            loss = nn.BCELoss()(pred, y)

            # Zero out the gradients, perform the backprop step, and update the weights
            # optimizer.zero_grad() is supposedly slower than this loop:
            for param in model.parameters():
                param.grad = None
            loss.backward()
            optimizer.step()

            # Add the loss to the total training loss so far and calculate the number of correct predictions
            totalTrainLoss += loss
            trainCorrect += (pred.argmax(1) == y.argmax(1)).type(torch.float).sum().item()

            # if batch_idx % log_interval == 0:
            #     print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
            #         e, batch_idx * len(data['is_insulator']), len(trainloader.dataset),
            #         100. * batch_idx / len(trainloader), loss.item()))

        # Validation

        # Switch off autograd for evaluation
        with torch.no_grad():
            #  Set model in evaluation mode
            model.eval()

            #  Loop over the validation set
            for (x, y) in valDataLoader:
                # These three lines are essential to apply for our data!!!!!!
                y = F.one_hot(y, num_classes=classes_no)
                y = y.float()

                # Send the input to the device
                (x, y) = (x.to(device), y.to(device))

                # Make the predictions and calulate the validation loss
                pred = model(x)
                totalValLoss += nn.BCELoss()(pred, y)

                # Calculate the number of correct predictions
                valCorrect += (pred.argmax(1) == y.argmax(1)).type(torch.float).sum().item()

        # Calculate the average training and validation loss
        avgTrainLoss = totalTrainLoss / trainSteps
        avgValLoss = totalValLoss / valSteps

        # Calculate the training and validation accuracy
        trainCorrect = trainCorrect / len(trainDataLoader.dataset)
        valCorrect = valCorrect / len(valDataLoader.dataset)

        # save the best model till now if we have the least loss in the current epoch
        if checkpoint_best is not None:
            save_best_model(
                avgValLoss, model, verb=False, checkpoint_best=checkpoint_best
            )

        # Update the training history
        H["train_loss"].append(avgTrainLoss.cpu().detach().numpy())
        H["train_acc"].append(trainCorrect)
        H["val_loss"].append(avgValLoss.cpu().detach().numpy())
        H["val_acc"].append(valCorrect)

        if early_stopping is not None:
            if early_stopper.early_stop(avgValLoss, e):
                break

        if scheduler is not None:
            scheduler.step(avgValLoss)

        # print the model training and validation information
        if print_all:
            cprint("\n[INFO]", "magenta", end=" ")
            cprint(f"Current time : {time.strftime('%H:%M:%S', time.localtime())}", end=" ")
            cprint("EPOCH:", "white", end=" ")
            print("{}/{}".format(e + 1, epoch_count))
            print("Train loss: {:.6f}, Train accuracy: {:.4f}".format(
                avgTrainLoss, trainCorrect))
            print(" Val  loss: {:.6f},  Val  accuracy: {:.4f}\n".format(
                avgValLoss, valCorrect))
        else:
            pass
            # if (e+1) % (epoch_count//5) == 0:
            #     cprint("[INFO]", "magenta", end=" ")
            #     cprint("EPOCH:", "white", end=" ")
            #     print("{}/{}".format(e + 1, epoch_count))

    # finish measuring how long training took
    endTime = time.time()
    cprint("\n[INFO]", "magenta", end=" ")
    print("Total time taken to train the model: {:.2f}s".format(
        endTime - startTime))

    return H


def test_model(model,
               testDataLoader,
               class_names,
               device,
               verbose=True,
               therm=False,
               therm_levels=100,
               ):
    # we can now evaluate the network on the test set
    if verbose:
        cprint("[INFO]", "magenta", end=" ")
        print("Evaluating network...")
    # turn off autograd for testing evaluation
    with torch.no_grad():
        # set the model in evaluation mode
        model.eval()

        # initialize a list to store our predictions
        preds = []
        # loop over the test set
        for (x, y) in testDataLoader:
            y = F.one_hot(y, num_classes=len(class_names))
            y = y.float()

            # send the input to the device
            x = x.to(device)
            # make the predictions and add them to the list
            pred = model(x)
            preds.extend(pred.argmax(axis=1).cpu().numpy())
    # generate a classification report
    if verbose:
        print(classification_report(testDataLoader.dataset.Y.cpu().numpy(),
                                    np.array(preds), target_names=class_names))
    return classification_report(testDataLoader.dataset.Y.cpu().numpy(),
                                 np.array(preds), target_names=class_names, output_dict=True)


def get_preds_from_model(model,
                         testDataLoader,
                         class_names,
                         device,
                         therm=False,
                         therm_levels=100,
                         ):
    with torch.no_grad():
        # set the model in evaluation mode
        model.eval()

        # initialize a list to store our predictions
        preds = []
        # loop over the test set
        for (x, y) in testDataLoader:
            # These three lines are essential to apply for our data!!!!!!
            y = F.one_hot(y, num_classes=len(class_names))
            y = y.float()

            # send the input to the device
            x = x.to(device)
            # make the predictions and add them to the list
            pred = model(x)
            preds.extend(pred.argmax(axis=1).cpu().numpy())

    return np.array(preds)


class Hook():
    def __init__(self, m):
        self.hook = m.register_forward_hook(self.hook_func)

    def hook_func(self, m, i, o):
        self.stored = o.detach().clone()

    def __enter__(self, *args): return self

    def __exit__(self, *args):
        self.hook.remove()


def plot_training_hist(H):
    plt.style.use("ggplot")
    fig, ax = plt.subplots(1, 2, figsize=(20, 8))
    ax[0].plot(H["train_loss"], label="train_loss")
    ax[0].plot(H["val_loss"], label="val_loss")
    ax[1].plot(H["train_acc"], label="train_acc")
    ax[1].plot(H["val_acc"], label="val_acc")
    fig.suptitle("Training Loss and Accuracy on Dataset", size=25)

    ax[0].set_xlabel("Epoch #")
    ax[1].set_xlabel("Epoch #")
    ax[0].set_ylabel("Loss/Accuracy")
    ax[1].set_ylabel("Loss/Accuracy")

    ax[0].legend(loc="upper left")
    ax[1].legend(loc="lower left")




def hook_feature(module, input, output):
    features_blobs.append(output.data.cpu().numpy())


def get_saliency_map(img_tensor,
                     model,
                     M,
                     device,
                     ):
    saliency_arr = []
    model.eval()
    for x in img_tensor:
        x = x.reshape([1, -1, M, M])
        x = x.to(device)
        x.requires_grad_()
        output = model(x)

        # Catch the output
        output_idx = output.argmax()
        output_max = output[0, output_idx]

        # Do backpropagation to get the derivative of the output based on the image
        output_max.backward()

        # Retireve the saliency map and also pick the maximum value from channels on each pixel.
        # In this case, we look at dim=1. Recall the shape (batch_size, channel, width, height)
        saliency, _ = torch.max(x.grad.data.abs(), dim=1)
        saliency = saliency.reshape(M, M)
        saliency_arr.append(saliency.cpu().numpy())
    return np.array(saliency_arr)


def get_labels_comparison_fig(fig, ax, ds, preds):
    fig.suptitle(f"Model :  | W = {np.unique(ds.W_tab)[0]}", fontsize=20)
    plt.tight_layout()
    ax.plot(ds.v_tab, ds.test_labels, lw=0, marker=6,
            label=
            "True label")
    ax.plot(ds.v_tab, preds, lw=0, marker=7,
            label=
            "Predicted label")
    ax.set_ylabel("Winding Number $\mathcal{W}$", fontsize=18)
    ax.set_xlabel("$\mathcal{v}$", fontsize=18)
    ax.set_yticks([0, 1])
    ax.axvline(x=1.0, ls="dotted")
    plt.legend(loc='lower left', fontsize=25)
    return fig

def returnCAM(feature_conv,
              weight_softmax,
              class_idx):
    size_upsample = (50, 50)
    bz, nc, h, w = feature_conv.shape
    output_cam = []
    for idx in class_idx:
        cam = weight_softmax[idx].dot(feature_conv.reshape((nc, h * w)))
        cam = cam.reshape(h, w)
        cam = cam - np.min(cam)
        cam_img = cam / np.max(cam)
        cam_img = np.uint8(255 * cam_img)
        cam_img = Image.fromarray(cam_img)
        cam_img = cam_img.resize(size_upsample)
        output_cam.append(cam_img)
    return output_cam

def get_plain_CAM(model, ds, device, cam_img_num=0, last_conv_name="cnn_layers_2", therm=False, therm_levels=100):
    features_blobs = []

    def hook_feature(module, input, output):
        features_blobs.append(output.data.cpu().numpy())

    model._modules.get(last_conv_name).register_forward_hook(hook_feature)

    # Tutaj wyciągamy wagi, które pojawiają się podczas klasyfikacji, przy softmaxie (po stronie GAPa, stąd 'params[-2]', czyli druga od końca)
    params = list(model.parameters())
    weight_softmax = np.squeeze(params[-2].cpu().data.numpy())
    # print(weight_softmax)

    # %%

    transform = transforms.Compose([ToTensor()])
    if therm:
        img_tensor = (
            transform(ds.test_data[cam_img_num])
            .float()
            .reshape(-1, 1, 50, 50)
        )
        img_tensor = data_to_therm(img_tensor, therm_levels)
        img_tensor = img_tensor.to(device)
    else:
        img_tensor = (
            transform(ds.test_data[cam_img_num])
            .float()
            .reshape(-1, 1, 50, 50)
            .to(device)
        )
    img_variable = Variable(img_tensor)
    # # Teraz predykcja! Hak przyczepiony do ostatniej warstwy konwolucyjnej
    # # zapamięta do czego skolapsował obrazek.
    logit = model(img_variable)
    # %%

    # Puszczamy przez softmax:
    h_x = F.softmax(logit, dim=1).data.squeeze()

    # Sortujemy prawdopodobieństwa wraz z odpowiadającymi im kategoriami
    probs, idx = h_x.cpu().sort(0, True)
    probs = probs.numpy()
    idx = idx.numpy()

    # Wreszcie możemy wygenerować CAM dla dowolnej kategorii.
    # Zacznijmy od klasy z największym prawdopodobieństwem, czyli top1: idx[0] (bo sortowaliśmy względem prawdopodobieństw, pamiętacie?)
    CAMs_topo = returnCAM(features_blobs[0], weight_softmax, [1])
    CAMs_trivial = returnCAM(features_blobs[0], weight_softmax, [0])
    return CAMs_topo, CAMs_trivial

# def returnCAM(feature_conv, weight_softmax, class_idx):
#     size_upsample = (50, 50)
#     bz, nc, h, w = feature_conv.shape
#     output_cam = []
#     for idx in class_idx:
#         cam = weight_softmax[idx] * feature_conv.reshape((nc, h * w))
#         cam = np.sum(cam, axis=0)
#         cam = cam - np.min(cam)
#         cam_img = cam / np.max(cam)
#         cam_img = np.uint8(255 * cam_img)
#         cam_img = Image.fromarray(cam_img)
#         cam_img = cam_img.resize(size_upsample)
#         output_cam.append(cam_img)
#     return output_cam

def upsample_CAM(cam):
    size_upsample = (50, 50)
    cam_img = cam / np.max(cam)
    cam_img = np.uint8(255 * cam_img)
    cam_img = Image.fromarray(cam_img)
    cam_img = cam_img.resize(size_upsample)
    return cam_img

# Grad-CAM implementation from ChatGPT
def get_grad_CAM(model, ds, device, cam_img_num=0, last_conv_name="cnn_layers_2", therm=False, therm_levels=100):
    features_blobs = []
    gradients = []

    def hook_feature(module, input, output):
        features_blobs.append(output)

    def hook_grad(module, grad_input, grad_output):
        gradients.append(grad_output[0])

    model._modules.get(last_conv_name).register_forward_hook(hook_feature)
    model._modules.get(last_conv_name).register_backward_hook(hook_grad)

    transform = transforms.Compose([ToTensor()])
    if therm:
        img_tensor = (
            transform(ds.test_data[cam_img_num])
            .float()
            .reshape(1, 1, 50, 50)
        )
        img_tensor = data_to_therm(img_tensor, therm_levels)
        img_tensor = img_tensor.to(device)
    else:
        img_tensor = (
            transform(ds.test_data[cam_img_num])
            .float()
            .reshape(1, 1, 50, 50)
            .to(device)
        )
    img_variable = Variable(img_tensor, requires_grad=True)

    model.eval()  # Set model to evaluation mode
    logit = model(img_variable)
    model.zero_grad()

    # Perform backpropagation
    one_hot = torch.zeros_like(logit, requires_grad=False)
    one_hot[0, torch.argmax(logit)] = 1
    logit.backward(gradient=one_hot)

    # Calculate the gradients and features
    gradients = gradients[0]
    features = features_blobs[0][0]

    # Calculate the weights using global average pooling
    # weights = torch.mean(gradients, dim=(1, 2), keepdim=True)
    pooled_gradients = torch.mean(gradients, dim=[0, 2, 3])

    # Generate the class activation map
    # CAMs_topo = torch.sum(weights.squeeze() * features, axis=1).detach().numpy()[0]
    for i in range(features.size()[0]):
        features[i, :, :] *= pooled_gradients[i]

    CAMs_topo = torch.mean(features, dim=0).detach().numpy()
    CAMs_topo = np.maximum(CAMs_topo, 0)  # ReLU activation
    CAMs_topo = CAMs_topo / np.max(CAMs_topo)  # Normalize

    CAMs_topo = upsample_CAM(CAMs_topo)
    
    # # Generate the class activation map
    # CAMs_trivial = torch.sum(weights.squeeze() * features, axis=1).detach().numpy()[0]
    # CAMs_trivial = np.maximum(CAMs_trivial, 0)  # ReLU activation
    # CAMs_trivial = CAMs_trivial / np.max(CAMs_trivial)  # Normalize

    return CAMs_topo#,CAMs_trivial

def get_CAM(model, ds, device, cam_img_num=0, last_conv_name="cnn_layers_2", therm=False, therm_levels=100, grad_cam=False):
    if grad_cam:
        return get_grad_CAM(model, ds, device, cam_img_num=0, last_conv_name="cnn_layers_2", therm=False, therm_levels=100)
    else:
        return get_plain_CAM(model, ds, device, cam_img_num=0, last_conv_name="cnn_layers_2", therm=False, therm_levels=100)

def CAMs_viz(fig, ax, CAMs_tab, ds, preds, class_names, cam_img_num=0, M=50):
    CAMs_topo, CAMs_trivial = CAMs_tab
    test_loader = ds.get_test_loader()

    map_img = cm.coolwarm

    img = ds.test_data[cam_img_num]
    img_cmap = np.uint8(map_img(img) * 255)

    extent = 0, M, 0, M

    fig.suptitle(
        f"Img_num = {cam_img_num}\n Predicted label = {preds[cam_img_num]} | True label = {test_loader.dataset.Y[cam_img_num]}\n {class_names[preds[cam_img_num]]} | {class_names[test_loader.dataset.Y[cam_img_num]]}",
        size=25)
    cbar2 = ax[0, 0].imshow(img_cmap, alpha=1, extent=extent, cmap=map_img)
    cbar = ax[0, 1].imshow(CAMs_topo[0], alpha=1, extent=extent, cmap=map_img)
    ax[0, 2].imshow(CAMs_trivial[0], alpha=1, extent=extent, cmap=map_img)
    ax[0, 0].set_title("Image", size=20)
    ax[0, 1].set_title("CAM - Topological", size=20)
    ax[0, 2].set_title("CAM - Trivial", size=20)

    ax[1, 0].imshow(img_cmap, alpha=1, extent=extent, cmap=map_img)
    ax[1, 1].imshow(img_cmap, alpha=.65, extent=extent, cmap=map_img)
    ax[1, 1].imshow(CAMs_topo[0], alpha=.35, extent=extent, cmap=map_img)
    ax[1, 2].imshow(img_cmap, alpha=.65, extent=extent, cmap=map_img)
    ax[1, 2].imshow(CAMs_trivial[0], alpha=.35, extent=extent, cmap=map_img)

    cax = plt.axes([1.02, 0.01, 0.075, 0.95])
    cax2 = plt.axes([-0.1, 0.01, 0.075, 0.95])
    plt.colorbar(cbar, cax=cax)
    plt.colorbar(cbar2, cax=cax2)

    fig.tight_layout()
    for axs in ax.flat:
        axs.grid(visible=False)
        axs.label_outer()

    plt.show()


# %%
if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else
                          'mps' if torch.backends.mps.is_available() else
                          "cpu")
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # ds_path = pathlib.Path("./Datasets/M=50/1000-200-100/")
    # ds_path = pathlib.Path("./Dataset s/M=50/2000-400-200/")
    ds_path = pathlib.Path("./Datasets/M=50/5000-1000-500/")
    # ds_clean = pathlib.Path("./Datasets/M=50/1000-200-100/")
    # ds_clean = pathlib.Path("./Datasets/M=50/2000-400-200/")
    ds_clean = pathlib.Path("./Datasets/M=50/5000-1000-500/")
    ds_w_001 = pathlib.Path("./Datasets/M=50/W=0.01-500-samples/")
    ds_w_005 = pathlib.Path("./Datasets/M=50/W=0.05-500-samples/")
    ds_w_015 = pathlib.Path("./Datasets/M=50/W=0.15-500-samples/")
    ds_w_100 = pathlib.Path("./Datasets/M=50/W=1-500-samples/")
    ds_w_300 = pathlib.Path("./Datasets/M=50/W=3-500-samples/")
    def_batch_size = 500
    epochs = 5000

    lr = 0.0001
    moment = 0.9
    weight_decay = 0.1
    noise_pow = 0
    es_dict = {
        "patience" : 20,
        "tolerance" : 0.00001,
        'warmup':1000
        }
    # ==============================================================================
    # Here we only have two classes, defined by Winding number:
    # Winding number = 0
    # Winding number = 1
    # ==============================================================================
    num_classes = 2
    class_names = ["Trivial", "Topological"]

    # %%
    perturb_list = [ds_w_001, ds_w_005]
    ds = Importer(ds_path, def_batch_size, device, noise_pow,
                  perturbative_ds=perturb_list)
    train_loader = ds.get_train_loader(seed=2137)
    val_loader = ds.get_val_loader()
    test_loader = ds.get_test_loader()

    model = CNN_Upgrade()

    model = model.float()
    model.to(device);
    optimizer = optim.SGD(model.parameters(),
                          lr=lr,
                          momentum=moment,
                          weight_decay=weight_decay)

    train_history = train_model(model, 
                                epochs, 
                                train_loader, 
                                val_loader,
                                optimizer, 
                                def_batch_size, 
                                num_classes, 
                                device,
                                print_all=False, 
                                therm=False, 
                                therm_levels=100,
                                early_stopping=es_dict,
                                )
    test_dict = test_model(model, test_loader, class_names, device, therm=False, therm_levels=100)
    plot_training_hist(train_history)

    # %%
    # serialize the model to disk
    fname = 'upgraded_model_x'
    torch.save(model.state_dict(), f"Models/{fname}.dict")
    torch.save(train_history, f"Models/{fname}_history.pickle")
    """
    # %%
    ds_path = pathlib.Path("./Datasets/M=50/5000-1000-500/")

    cam_img_num = 30
    # ==============================================================================
    # Here we only have two classes, defined by Winding number:
    # Winding number = 0
    # Winding number = 1
    # ==============================================================================
    num_classes = 2
    class_names = ["Trivial", "Topological"]

    # %%

    ds = Importer(ds_path, def_batch_size, device)

    test_loader = ds.get_test_loader()
    preds = get_preds_from_model(model, test_loader, class_names, device, therm=False, therm_levels=100)

    # with plt.xkcd():
    fig, ax = plt.subplots(1, 1, figsize=(16, 8))
    get_labels_comparison_fig(fig, ax, ds, preds)
    fig.show()

    features_blobs = []

    model._modules.get('cnn_layers_2').register_forward_hook(hook_feature)

    # Tutaj wyciągamy wagi, które pojawiają się podczas klasyfikacji, przy softmaxie (po stronie GAPa, stąd 'params[-2]', czyli druga od końca)
    params = list(model.parameters())
    weight_softmax = np.squeeze(params[-2].cpu().data.numpy())
    # print(weight_softmax)

    transform = transforms.Compose([ToTensor()])
    img_tensor = transform(ds.test_data[cam_img_num]).float().reshape(-1, 1, 50, 50).to(device)
    img_variable = Variable(img_tensor)
    # # Teraz predykcja! Hak przyczepiony do ostatniej warstwy konwolucyjnej
    # # zapamięta do czego skolapsował obrazek.
    logit = model(img_variable)
    logit

    # Puszczamy przez softmax:
    h_x = F.softmax(logit, dim=1).data.squeeze()

    # Sortujemy prawdopodobieństwa wraz z odpowiadającymi im kategoriami
    probs, idx = h_x.cpu().sort(0, True)
    probs = probs.numpy()
    idx = idx.numpy()

    # Wreszcie możemy wygenerować CAM dla dowolnej kategorii.
    # Zacznijmy od klasy z największym prawdopodobieństwem, czyli top1: idx[0] (bo sortowaliśmy względem prawdopodobieństw, pamiętacie?)
    CAMs = returnCAM(features_blobs[0], weight_softmax, [idx[0]])

    img = ds.test_data[cam_img_num]
    img_cmap = np.uint8(cm.jet(img) * 255)
    # img_pil = Image.fromarray(img_cmap)
    # img_pil = img_pil.resize((50, 50))

    cam_tmp = CAMs[0]
    heatmap = np.uint8(cm.jet(cam_tmp) * 255)
    # hmap_pil = Image.fromarray(heatmap)
    # hmap_pil = hmap_pil.resize((50, 50))

    # result = Image.blend(img_pil, hmap_pil, 0.3)

    extent = 0, 50, 0, 50

    fig, ax = plt.subplots(1, 2, figsize=(10, 10))
    ax[0].grid(False)
    ax[1].grid(False)
    ax[0].imshow(img_cmap, alpha=1, extent=extent, cmap=cm.viridis)
    cbar = ax[1].imshow(CAMs[0], alpha=1, extent=extent, cmap=cm.magma)
    ax[0].set_title("")
    cax = plt.axes([0.95, 0.1, 0.075, 0.8])
    plt.colorbar(cbar, cax=cax)
    # plt.axis('off')

    plt.show()
    """
