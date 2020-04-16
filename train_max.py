import argparse
import numpy as np
from maxDataset import MaxDatasetTuple
from model_max import MaxModel
import torch
import torch.nn as nn
import os
from tqdm import tqdm
from utils import pltMaxHeatMap
import torch.optim as optim
from torch.optim import lr_scheduler
from correlation import Correlation

image_saving_dir = '/home/share_uav/ruiz/data/max/img/'

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

init_min_cor = Correlation()
pred_min_cor = Correlation()
init_max_cor = Correlation()
pred_max_cor = Correlation()


def train(model: MaxModel, train_loader, device, optimizer, criterion, epoch):
    # set 为 train模式
    model.train()

    sum_running_loss = 0.0
    num_images = 0

    for batch_idx, data in enumerate(tqdm(train_loader)):
        # 初始化梯度为0
        optimizer.zero_grad()

        initMax = data['init_max'].to(device).float()
        labelMax = data['label_max'].to(device).float()
        inputs = data['input'].to(device).float()

        predictionMax = model(initMax=initMax, x=inputs)

        lossMax = criterion(predictionMax, labelMax.data)

        lossMax.backward()

        optimizer.step()

        if lossMax != 0.0:
            sum_running_loss += lossMax * initMax.size(0)
        num_images += initMax.size(0)

        if batch_idx % 50 == 0 or batch_idx == len(train_loader) - 1:
            sum_epoch_loss = sum_running_loss / num_images
            print('\nTraining phase: epoch: {} batch:{} Loss: {:.4f}\n'.format(epoch, batch_idx, sum_epoch_loss))


def val(model, path, test_loader, device, criterion, epoch, batch_size):
    model.eval()
    sum_running_loss = 0.0

    with torch.no_grad():
        for batch_idx, data in enumerate(tqdm(test_loader)):
            initMax = data['init_max'].to(device).float()
            labelMax = data['label_max'].to(device).float()
            inputs = data['input'].to(device).float()

            predictionMax = model(initMax=initMax, x=inputs)

            lossMax = criterion(predictionMax, labelMax.data)

            sum_running_loss += lossMax.item() * initMax.size(0)

            # TODO: visualization
            pltMaxHeatMap(path, epoch, batch_idx, batch_size,
                          initMax, labelMax, predictionMax)

            if batch_idx == 0:
                outPredictionMax = predictionMax.cpu().detach().numpy()
                outLabelMax = labelMax.cpu().detach().numpy()
                outInitMax = initMax.cpu().detach().numpy()
            else:
                outPredictionMax = np.append(predictionMax.cpu().detach().numpy(),
                                             outPredictionMax, axis=0)
                outLabelMax = np.append(labelMax.cpu().detach().numpy(),
                                        outLabelMax, axis=0)
                outInitMax = np.append(initMax.cpu().detach().numpy(),
                                       outInitMax, axis=0)

    sum_loss = sum_running_loss / len(test_loader.dataset)
    print('\nTesting phase: epoch: {} Loss: {:.4f}\n'.format(epoch, sum_loss))

    # auc_path = os.path.join(path, "min_epoch_" + str(epoch))
    # auc(['flow'], [2, 4, 10, 100], [[outLabelMin, outPredictionMin]], auc_path)
    # auc_path = os.path.join(path, "max_epoch_" + str(epoch))
    # auc(['flow'], [2, 4, 10, 100], [[outLabelMax, outPredictionMax]], auc_path)

    return sum_loss, outPredictionMax, outLabelMax, outInitMax


def loadData(init_max_path: str, label_max_path: str,
             input_path: str, splitRatio: str, batch_size):
    all_dataset = MaxDatasetTuple(input_path=input_path, init_max_path=init_max_path, label_max_path=label_max_path)

    trainSize = int(splitRatio * len(all_dataset))
    testSize = len(all_dataset) - trainSize
    trainDataset, testDataset = torch.utils.data.random_split(all_dataset, [trainSize, testSize])
    print("Total image tuples for train: ", len(trainDataset))
    print("Total image tuples for test: ", len(testDataset))

    # print("\nUsing #", torch.cuda.current_device(), "GPU!\n")

    trainDataLoader = torch.utils.data.DataLoader(trainDataset, batch_size=batch_size, shuffle=True,
                                                  num_workers=30,
                                                  drop_last=True)
    testDataLoader = torch.utils.data.DataLoader(testDataset, batch_size=batch_size, shuffle=False, num_workers=30,
                                                 drop_last=True)
    return trainDataLoader, testDataLoader


def setParser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", help="input path", required=True, type=str)
    parser.add_argument("--init_max_path", help="init max path", required=True, type=str)
    parser.add_argument("--label_max_path", help="label max path", required=True, type=str)
    parser.add_argument("--lr", help="learning rate", required=True, type=float)
    parser.add_argument("--momentum", help="momentum", required=True, type=float)
    parser.add_argument("--weight_decay", help="weight decay", required=True, type=float)
    parser.add_argument("--batch_size", help="batch size", required=True, type=int)
    parser.add_argument("--num_epochs", help="num_epochs", required=True, type=int)
    parser.add_argument("--split_ratio", help="training/testing split ratio", required=True, type=float)
    parser.add_argument("--checkpoint_dir", help="checkpoint_dir", required=True, type=str)
    parser.add_argument("--model_checkpoint_name", help="model checkpoint name", required=True, type=str)
    parser.add_argument("--load_from_main_checkpoint", type=str)
    parser.add_argument("--image_save_folder", type=str, required=True)
    parser.add_argument("--eval_only", dest='eval_only', action='store_true')
    args, unknown = parser.parse_known_args()
    return args


def save_model(checkpoint_dir, model_checkpoint_name, model):
    model_save_path = '{}/{}'.format(checkpoint_dir, model_checkpoint_name)
    print('save model to: \n{}'.format(model_save_path))
    torch.save(model.state_dict(), model_save_path)


def main():
    torch.manual_seed(0)

    print('=' * 50 + "parse commandline parameters" + '=' * 50)
    args = setParser()
    print('=' * 50 + "ending parsing commandline parameters" + '=' * 50)

    image_saving_path = image_saving_dir + args.image_save_folder

    if not os.path.exists(args.checkpoint_dir):
        os.mkdir(args.checkpoint_dir)
    if not os.path.exists(args.checkpoint_dir + "/" + args.model_checkpoint_name):
        os.mkdir(args.checkpoint_dir + "/" + args.model_checkpoint_name)

    device = torch.device('cuda')

    print('=' * 50 + "loading data and instantiating data loader" + '=' * 50)
    trainDataLoader, testDataLoader = loadData(init_max_path=args.init_max_path, label_max_path=args.label_max_path,
                                               input_path=args.input_path, splitRatio=args.split_ratio,
                                               batch_size=args.batch_size)
    print('=' * 50 + "ending loading data and instantiating data loader" + '=' * 50)

    print('=' * 50 + "instantiate model" + '=' * 50)
    model = MaxModel()

    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)
    model = model.to(device)
    print('=' * 50 + "ending instantiating model" + '=' * 50)

    if args.load_from_main_checkpoint:
        print('=' * 50 + "load check point" + '=' * 50)
        chkpt_mainmodel_path = args.load_from_main_checkpoint
        print("Loading ", chkpt_mainmodel_path)
        model.load_state_dict(torch.load(chkpt_mainmodel_path, map_location=device))
        print('=' * 50 + "end loading check point" + '=' * 50)

    print('=' * 50 + "setting loss criterion, optimizer, lr decay" + '=' * 50)
    criterion = nn.MSELoss(reduction='sum')
    optimizer_ft = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=20, gamma=0.1)
    print('=' * 50 + "ending setting loss criterion, optimizer, lr decay" + '=' * 50)

    correlation_path = image_saving_path

    if args.eval_only:
        print("evaluate only")
        for epoch in range(1):
            loss, outPreMax, outLabelMax, outInitMax = \
                val(model, path=image_saving_path, test_loader=testDataLoader,
                    device=device, criterion=criterion, epoch=epoch, batch_size=args.batch_size)
            cor_path = os.path.join(correlation_path, "epoch_" + str(epoch))

            max_coef_pre_label = pred_max_cor.corrcoef(prediction=outPreMax, label=outLabelMax,
                                                       path=cor_path, name="max_correlation_{0}.png".format(epoch))
            max_coef_init_label = init_max_cor.corrcoef(outInitMax, outLabelMax,
                                                        cor_path, "max_correlation_init_label{0}.png".format(epoch))
            print('correlation coefficient max:{0}\n'.format(max_coef_pre_label))
            print('correlation_init_label coefficient max:{0}\n'
                  .format(max_coef_init_label))
        print('=' * 50 + "ending evaluating only" + '=' * 50)
        return True

    best_loss = np.inf
    print('=' * 50 + "train and evaluating" + '=' * 50)
    for epoch in range(args.num_epochs):
        print('Epoch {}/{}'.format(epoch, args.num_epochs - 1))
        print('-' * 80)

        train(model, trainDataLoader, device, optimizer_ft, criterion, epoch)
        exp_lr_scheduler.step()
        cor_path = os.path.join(correlation_path, "epoch_" + str(epoch))
        if not os.path.exists(cor_path):
            os.mkdir(cor_path)

        loss, outPreMax, outLabelMax, outInitMax = \
            val(model, path=image_saving_path, test_loader=testDataLoader,
                device=device, criterion=criterion, epoch=epoch, batch_size=args.batch_size)

        if loss < best_loss:
            save_model(checkpoint_dir=args.checkpoint_dir + "/" + args.model_checkpoint_name,
                       model_checkpoint_name=args.model_checkpoint_name + "_epoch_" + str(epoch) + '_' + str(loss),
                       model=model)
            best_loss = loss

        max_coef_pre_label = pred_max_cor.corrcoef(prediction=outPreMax, label=outLabelMax,
                                                   path=cor_path, name="max_correlation_{0}.png".format(epoch))
        max_coef_init_label = init_max_cor.corrcoef(outInitMax, outLabelMax,
                                                    cor_path, "max_correlation_init_label{0}.png".format(epoch))
        print('correlation coefficient max:{0}\n'.format(max_coef_pre_label))
        print('correlation_init_label coefficient max:{0}\n'.format(max_coef_init_label))
    print('=' * 50 + "ending training and evaluating" + '=' * 50)


if __name__ == "__main__":
    main()
