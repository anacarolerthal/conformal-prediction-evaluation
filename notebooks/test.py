import pandas
import torchvision
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
path = sys.path.insert(1,"C:/PythonEnviroments/pytorch_resnet_cifar10")
from resnet import resnet110
from torchvision import transforms
import torchmetrics
import dill
import numpy as np
import torchvision.transforms as transforms
from temp_scale import ModelWithTemperature
from conformal_logits import conformal, evaluate_conformal, create_fake_logits


# if name main
if __name__ == '__main__':
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    transform = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])

    train_set = torchvision.datasets.CIFAR10(
        root="./data", train=True, download=True,
        transform=transform,
    )
    test_set = torchvision.datasets.CIFAR10(
        root="./data", train=False, download=True,
        transform=transform
    )

    val_set = torchvision.datasets.CIFAR10(
        root="./data", train=False, download=True,
        transform=transform
    )



    # the network architecture coresponding to the checkpoint

    model = resnet110()

    # remember to set map_location
    check_point = torch.load('../models/resnet110-1d1ed7c2.pth', map_location='cuda:0')

    # cause the model are saved from Parallel, we need to wrap it
    model = torch.nn.DataParallel(model)
    model.load_state_dict(check_point['state_dict'])

    # pay attention to .module! without this, if you load the model, it will be attached with [Parallel.module]
    # that will lead to some trouble!
    torch.save(model.module, 'resnet110.pth', pickle_module=dill)

    # load the converted pretrained model
    model = torch.load('resnet110.pth', map_location='cuda:0')

    epoch = 0
    while True:
        #randomly sample the test and validation set
        val_set_size = np.random.choice([1000, 1500, 2000, 2500, 3000])
        test_set_random = torch.utils.data.Subset(test_set, np.random.choice(len(test_set), 5000, replace=False))
        val_set_random = torch.utils.data.Subset(val_set, np.random.choice(len(val_set), val_set_size, replace=False))

        test_loader = torch.utils.data.DataLoader(
            test_set_random, batch_size=128, shuffle=True, num_workers=2)
        val_loader = torch.utils.data.DataLoader(
            val_set_random, batch_size=128, shuffle=True, num_workers=2)


        epoch += 1
        model.eval()
        predictions = torch.tensor([])
        targets = torch.tensor([])
        predictions_val = torch.tensor([])
        targets_val = torch.tensor([])
        with torch.no_grad():
            for i, (x, y) in enumerate(test_loader):
                x = x.cuda()
                y_hat = model(x)
                predictions = torch.cat((predictions, y_hat.cpu()))
                targets = torch.cat((targets, y.cpu()))
            for i, (x, y) in enumerate(val_loader):
                x = x.cuda()
                y_hat = model(x)
                predictions_val = torch.cat((predictions_val, y_hat.cpu()))
                targets_val = torch.cat((targets_val, y.cpu()))

        ece1 = torchmetrics.functional.classification.multiclass_calibration_error(predictions, targets, n_bins=15, num_classes=10)
        nll1 = F.cross_entropy(predictions, targets.type(torch.long))

        temp_calib_model = ModelWithTemperature(model)
        temp_calib_model = temp_calib_model.set_temperature(val_loader)

        temp_calib_model.eval()
        ts_predictions = torch.tensor([])
        ts_targets = torch.tensor([])
        with torch.no_grad():
            for i, (x, y) in enumerate(test_loader):
                x = x.cuda()
                y_hat = temp_calib_model(x)
                ts_predictions = torch.cat((ts_predictions, y_hat.cpu()))
                ts_targets = torch.cat((ts_targets, y.cpu()))

        ece2 = torchmetrics.functional.classification.multiclass_calibration_error(ts_predictions, ts_targets, n_bins=15, num_classes=10)
        nll2 = F.cross_entropy(ts_predictions, ts_targets.type(torch.long))

        smx = nn.Softmax(dim=1)(predictions)
        ts_smx = nn.Softmax(dim=1)(ts_predictions)
        val_smx = nn.Softmax(dim=1)(predictions_val)

        smx = smx.numpy()
        ts_smx = ts_smx.numpy()
        val_smx = val_smx.numpy()

        targets = targets.numpy().astype(int)
        ts_targets = ts_targets.numpy().astype(int)
        targets_val = targets_val.numpy().astype(int)

        prediction_sets, alphas, actual_labels = conformal(val_smx, targets_val,targets, smx, len(val_smx))

        conformal_logits = create_fake_logits(alphas)

        conformal_logits = torch.tensor(conformal_logits)
        y_pred = torch.argmax(torch.tensor(smx), dim=1)
        new_logits = torch.tensor(predictions)

        for i in range(len(new_logits)):
            for j in range(len(new_logits[i])):
                if j != y_pred[i]:
                    new_logits[i][j] = 0
                else:
                    new_logits[i][y_pred[i]] = conformal_logits[i]

        ece3 = torchmetrics.functional.classification.multiclass_calibration_error(new_logits, torch.tensor(targets), n_bins=15, num_classes=10)

        #we will generate predictive intervals by adding the softmax probabilities

        smx = nn.Softmax(dim=1)(predictions)
        smx_ts = nn.Softmax(dim=1)(ts_predictions)

        alpha = np.random.choice([0.02, 0.05, 0.1, 0.15, 0.2, 0.25])
        pred_sets = []
        for i in range(len(smx)):
            temp = 1
            pred_set =[]
            while temp > alpha:
                temp = smx[i].max()
                pred_set.append(smx[i].argmax())
                smx[i][smx[i].argmax()] = 0
            pred_sets.append(pred_set)

        pred_sets_ts = []
        for i in range(len(smx_ts)):
            temp = 1
            pred_set =[]
            while temp > alpha:
                temp = smx_ts[i].max()
                pred_set.append(smx_ts[i].argmax())
                smx_ts[i][smx_ts[i].argmax()] = 0
            pred_sets_ts.append(pred_set)

        total = 0
        correct = 0
        for label in targets:
            if label in pred_sets[total]:
                correct += 1
            total += 1

        cov1 = correct / total
        mean_pred_set_size = 0
        for pred_set in pred_sets:
            mean_pred_set_size += len(pred_set)
        mean_pred_set_size = mean_pred_set_size / len(pred_sets)
        mean_pred_set_size1 = mean_pred_set_size

        total = 0
        correct = 0
        for label in ts_targets:
            if label in pred_sets_ts[total]:
                correct += 1
            total += 1

        cov2 = correct / total

        mean_pred_set_size = 0
        for pred_set in pred_sets_ts:
            mean_pred_set_size += len(pred_set)
        mean_pred_set_size = mean_pred_set_size / len(pred_sets_ts)
        mean_pred_set_size2 = mean_pred_set_size

        cal_smx = nn.Softmax(dim=1)(predictions_val).numpy()
        #cal_targets = cal_targets.numpy()
        cal_targets = targets_val.astype(int)

        val_smx = nn.Softmax(dim=1)(predictions).numpy()
        val_targets = targets.astype(int)

        cal_scores = 1 - cal_smx[np.arange(len(cal_smx)), cal_targets]

        q_level = np.ceil((len(cal_smx) + 1) * (1 - alpha))/len(cal_smx)
        q_hat = np.quantile(cal_scores, q_level, interpolation='higher')

        pred_sets = val_smx >= (1 - q_hat)

        empirical_coverage = pred_sets[np.arange(pred_sets.shape[0]),val_targets].mean()
        cov3 = empirical_coverage

        #now we calculate the average size of the predictive sets
        mean_pred_set_size = 0
        for pred_set in pred_sets:
            for i in pred_set:
                mean_pred_set_size += i
        mean_pred_set_size = mean_pred_set_size / len(pred_sets)
        mean_pred_set_size3 = mean_pred_set_size


        #write to file
        with open('resultsCIFAR.txt', 'a') as f:
            f.write(f'{epoch}, {val_set_size}, {alpha}: {ece1} , {nll1} , {cov1} , {mean_pred_set_size1} , {ece2} , {nll2} , {cov2} , {mean_pred_set_size2} , {ece3} , {cov3} , {mean_pred_set_size3}\n')