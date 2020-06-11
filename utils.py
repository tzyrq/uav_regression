import os
import torch
import pickle
import torchvision
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_curve, average_precision_score


def read_pickle(filename):
    infile = open(filename, 'rb')
    data = pickle.load(infile)
    infile.close()
    return data


def dump_pickle(filename, data):
    outfile = open(filename, "wb")
    pickle.dump(data, filename)
    outfile.close()


def visualize_sum_testing_result(path, init, prediction, sub_prediction, label, batch_id, epoch, batch_size):
    assert prediction.shape[0] == label.shape[0], "prediction size and label size is not identical"
    if not os.path.exists(path):
        os.mkdir(path)
    if not os.path.exists(path + "/epoch_" + str(epoch)):
        os.mkdir(path + "/epoch_" + str(epoch))
    if not os.path.exists(path + "/epoch_" + str(epoch) + "/sum"):
        os.mkdir(path + "/epoch_" + str(epoch) + "/sum")

    for idx, _ in enumerate(prediction):
        init_output = init[idx].cpu().detach()
        init_output = torch.squeeze(init_output)

        prediction_output = prediction[idx].cpu().detach()
        prediction_output = torch.squeeze(prediction_output)

        # print("sub_prediction.shape ", sub_prediction.shape)
        # 4, 1, 60, 100, 100
        sub_prediction_output = sub_prediction[idx][40].cpu().detach()
        sub_prediction_output = torch.squeeze(sub_prediction_output)
        label_output = label[idx].cpu().detach()
        label_output = torch.squeeze(label_output)

        torchvision.utils.save_image(init_output, path + "/epoch_" + str( \
            epoch) + "/sum" + "/" + str(idx + batch_id * batch_size) + "_init.png")
        torchvision.utils.save_image(sub_prediction_output, path + "/epoch_" + str(
            epoch) + "/sum" + "/" + str(idx + batch_id * batch_size) + "_sub_prediction.png")
        torchvision.utils.save_image(prediction_output, path + "/epoch_" + str(epoch) + "/sum" + "/" + str(
            idx + batch_id * batch_size) + "_prediction.png")
        torchvision.utils.save_image(label_output, path + "/epoch_" + str(epoch) + "/sum" + "/" + str(
            idx + batch_id * batch_size) + "_label.png")


def visualize_sum_training_result(init, prediction, sub_prediction, label, batch_id, epoch, batch_size):
    assert prediction.shape[0] == label.shape[0], "prediction size and label size is not identical"
    if not os.path.exists("/home/share_uav/ruiz/data/uav_regression/training_result/epoch_" + str(epoch)):
        os.mkdir("/home/share_uav/ruiz/data/uav_regression/training_result/epoch_" + str(epoch))
    if not os.path.exists("/home/share_uav/ruiz/data/uav_regression/training_result/epoch_" + str(epoch) + "/sum"):
        os.mkdir("/home/share_uav/ruiz/data/uav_regression/training_result/epoch_" + str(epoch) + "/sum")

    for idx, _ in enumerate(prediction):
        init_output = init[idx].cpu().detach()
        init_output = torch.squeeze(init_output)

        prediction_output = prediction[idx].cpu().detach()
        prediction_output = torch.squeeze(prediction_output)

        # print("sub_prediction.shape ", sub_prediction.shape)
        sub_prediction_output = sub_prediction[idx][0][1].cpu().detach()
        sub_prediction_output = torch.squeeze(sub_prediction_output)
        # print("ub_prediction_output.shape ", sub_prediction_output.shape)
        label_output = label[idx].cpu().detach()
        label_output = torch.squeeze(label_output)

        torchvision.utils.save_image(init_output,
                                     "/home/share_uav/ruiz/data/uav_regression/training_result/epoch_" + str( \
                                         epoch) + "/sum" + "/" + str(idx + batch_id * batch_size) + "_init.png")
        torchvision.utils.save_image(sub_prediction_output,
                                     "/home/share_uav/ruiz/data/uav_regression/training_result/epoch_" + str(
                                         epoch) + "/sum" + "/" + str(
                                         idx + batch_id * batch_size) + "_sub_prediction.png")
        torchvision.utils.save_image(prediction_output,
                                     "/home/share_uav/ruiz/data/uav_regression/training_result/epoch_" + str(
                                         epoch) + "/sum" + "/" + str(idx + batch_id * batch_size) + "_prediction.png")
        torchvision.utils.save_image(label_output,
                                     "/home/share_uav/ruiz/data/uav_regression/training_result/epoch_" + str(
                                         epoch) + "/sum" + "/" + str(idx + batch_id * batch_size) + "_label.png")


def pltMaxHeatMap(path, epoch, batch_id, batch_size, initMax, labelMax, predMax):
    if not os.path.exists(path):
        os.mkdir(path)
    if not os.path.exists(path + "/epoch_" + str(epoch)):
        os.mkdir(path + "/epoch_" + str(epoch))
    if not os.path.exists(path + "/epoch_" + str(epoch) + "/heat_map"):
        os.mkdir(path + "/epoch_" + str(epoch) + "/heat_map")

    for idx, _ in enumerate(initMax):
        eInitMax = initMax[idx].cpu().detach().numpy()
        eInitMax = np.squeeze(eInitMax)
        eLabelMax = labelMax[idx].cpu().detach().numpy()
        eLabelMax = np.squeeze(eLabelMax)
        ePredMax = predMax[idx].cpu().detach().numpy()
        ePredMax = np.squeeze(ePredMax)

        fig, ax = plt.subplots(1, 3, figsize=(10, 15))
        # print(eInitMax.shape)
        ax[0].imshow(eInitMax, cmap="jet")
        ax[0].set_title("initial max map", fontsize=15)

        ax[1].imshow(ePredMax, cmap="jet")
        ax[1].set_title("predicted max map", fontsize=15)

        ax[2].imshow(eLabelMax, cmap="jet")
        ax[2].set_title("label max map", fontsize=15)

        plt.savefig(path + "/epoch_" + str(epoch) + "/heat_map"
                    + "/" + str(idx + batch_id * batch_size)
                    + ".png", dpi=150)

        plt.close(fig)


def pltMinMaxHeatMap(path, epoch, batch_id, batch_size, initMin, labelMin, predMin, initMax, labelMax, predMax):
    if not os.path.exists(path):
        os.mkdir(path)
    if not os.path.exists(path + "/epoch_" + str(epoch)):
        os.mkdir(path + "/epoch_" + str(epoch))
    if not os.path.exists(path + "/epoch_" + str(epoch) + "/heat_map"):
        os.mkdir(path + "/epoch_" + str(epoch) + "/heat_map")

    for idx, _ in enumerate(initMin):
        eInitMin = initMin[idx].cpu().detach().numpy()
        eInitMin = np.squeeze(eInitMin)
        eLabelMin = labelMin[idx].cpu().detach().numpy()
        eLabelMin = np.squeeze(eLabelMin)
        ePredMin = predMin[idx].cpu().detach().numpy()
        ePredMin = np.squeeze(ePredMin)
        eInitMax = initMax[idx].cpu().detach().numpy()
        eInitMax = np.squeeze(eInitMax)
        eLabelMax = labelMax[idx].cpu().detach().numpy()
        eLabelMax = np.squeeze(eLabelMax)
        ePredMax = predMax[idx].cpu().detach().numpy()
        ePredMax = np.squeeze(ePredMax)

        fig, ax = plt.subplots(2, 3, figsize=(10, 15))
        # print(eInitMax.shape)
        ax[0, 0].imshow(eInitMin)
        ax[0, 0].set_title("initial min map", fontsize=15)

        ax[0, 1].imshow(ePredMin)
        ax[0, 1].set_title("predicted min map", fontsize=15)

        ax[0, 2].imshow(eLabelMin)
        ax[0, 2].set_title("label min map", fontsize=15)

        ax[1, 0].imshow(eInitMax)
        ax[1, 0].set_title("initial max map", fontsize=15)

        ax[1, 1].imshow(ePredMax)
        ax[1, 1].set_title("predicted max map", fontsize=15)

        ax[1, 2].imshow(eLabelMax)
        ax[1, 2].set_title("label max map", fontsize=15)

        plt.savefig(path + "/epoch_" + str(epoch) + "/heat_map"
                    + "/" + str(idx + batch_id * batch_size)
                    + ".png", dpi=150)

        plt.close(fig)

