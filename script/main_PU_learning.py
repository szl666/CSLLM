import argparse
import os
import shutil
import sys
import time
import warnings
from random import sample

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import pickle
import csv
import gc
from tqdm import tqdm
from sklearn import metrics
from torch.autograd import Variable
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import Dataset, DataLoader

from cgcnn.data_PU_learning import CIFData
from cgcnn.data_PU_learning import collate_pool, get_train_val_test_loader, split_bagging, bootstrap_aggregating
from cgcnn.model_PU_learning import CrystalGraphConvNet

parser = argparse.ArgumentParser(description='Synthesizability prediction by PU-learning using CGCNN classifier')
parser.add_argument('--disable-cuda', action='store_true',
                    help='Disable CUDA')
parser.add_argument('-j', '--workers', default=0, type=int, metavar='N',
                    help='number of data loading workers (default: 0)')
parser.add_argument('--epochs', default=30, type=int, metavar='N',
                    help='number of total epochs to run (default: 30)')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--lr', '--learning-rate', default=0.01, type=float,
                    metavar='LR', help='initial learning rate (default: '
                                       '0.01)')
parser.add_argument('--lr-milestones', default=[100], nargs='+', type=int,
                    metavar='N', help='milestones for scheduler (default: '
                                      '[100])')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=0, type=float,
                    metavar='W', help='weight decay (default: 0)')
parser.add_argument('--print-freq', '-p', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')

parser.add_argument('--optim', default='SGD', type=str, metavar='SGD',
                    help='choose an optimizer, SGD or Adam, (default: SGD)')
parser.add_argument('--atom-fea-len', default=64, type=int, metavar='N',
                    help='number of hidden atom features in conv layers')
parser.add_argument('--h-fea-len', default=128, type=int, metavar='N',
                    help='number of hidden features after pooling')
parser.add_argument('--n-conv', default=3, type=int, metavar='N',
                    help='number of conv layers')
parser.add_argument('--n-h', default=1, type=int, metavar='N',
                    help='number of hidden layers after pooling')

# Bagging size for PU Learning
parser.add_argument('--bag', default=50, type=int, metavar='N',
                    help='Bagging size of PU training')
parser.add_argument('--split', default='./saved_splits', type=str, metavar='N',
                    help='Saving directory of data-splits for PU-learning')
parser.add_argument('--graph', type=str, metavar='N',
                    help='Folder name for preloaded crystal graph files')
parser.add_argument('--cifs', type=str, metavar='N',
                    help='Folder name containing cif files and id_prop.csv file')
parser.add_argument('--restart', default=0, type=int, metavar='N',
                    help='Set restart point of bagging #')


args = parser.parse_args(sys.argv[1:])

args.cuda = not args.disable_cuda and torch.cuda.is_available()

best_mae_error = 0.


def preload(preload_folder, id_prop_file):
    data = []
    with open(id_prop_file) as g:
        reader = csv.reader(g)
        cif_list = [row[0] for row in reader]

    for cif_id in tqdm(cif_list):
        with open(preload_folder+'/'+cif_id+'.pickle', 'rb') as f:
            data.append(pickle.load(f))

    return data


def main():
    global args, best_mae_error

    graph_dir = args.graph
    '''
    Valid: Out-of-sample of positive and negatively labeled data for each iteration, the out-of-sample positive data were fixed during training for estimating model performance.
    Test: Remaining unlabeled data for predicting which not included in training set as negative labeled data.
    '''

    split_bagging(os.path.join(args.cifs, 'id_prop.csv'), args.restart, args.bag, args.split), 


    # Train/Valid/Test for all bagging loop
    for bagging in range(args.restart, args.restart+args.bag):

        initial_time = time.time()
        best_mae_error = 0
        collate_fn = collate_pool

        # Load Train/Valid/Test crystal graph data 
        dataset_train = preload(preload_folder = graph_dir, id_prop_file = os.path.join(args.split, 'id_prop_bag_'+str(bagging+1)+'_train.csv'))
        dataset_valid = preload(preload_folder = graph_dir, id_prop_file = os.path.join(args.split, 'id_prop_bag_'+str(bagging+1)+'_valid.csv'))
        dataset_test = preload(preload_folder = graph_dir, id_prop_file = os.path.join(args.split, 'id_prop_bag_'+str(bagging+1)+'_test-unlabeled.csv'))

        train_loader = DataLoader(dataset_train, batch_size=args.batch_size,shuffle=True,
                                  num_workers=args.workers,
                                  collate_fn=collate_fn, pin_memory=args.cuda)

        val_loader = DataLoader(dataset_valid, batch_size=args.batch_size,shuffle=True,
                                num_workers=args.workers,
                                collate_fn=collate_fn, pin_memory=args.cuda)

        test_loader = DataLoader(dataset_test, batch_size=args.batch_size,shuffle=True,
                                 num_workers=args.workers,
                                 collate_fn=collate_fn, pin_memory=args.cuda)

        preload_time = time.time()-initial_time
        print("--------------------------------------------------Data loaded in bagging #%d, Time: %f" % (bagging+1, preload_time))

        # obtain target value normalizer
        normalizer = Normalizer(torch.zeros(2))
        normalizer.load_state_dict({'mean': 0., 'std': 1.})

        # build model
        structures, _, _ = dataset_train[0]
        orig_atom_fea_len = structures[0].shape[-1]
        nbr_fea_len = structures[1].shape[-1]
        model = CrystalGraphConvNet(orig_atom_fea_len, nbr_fea_len,
                                    atom_fea_len=args.atom_fea_len,
                                    n_conv=args.n_conv,
                                    h_fea_len=args.h_fea_len,
                                    n_h=args.n_h,
                                    classification=True)
        if args.cuda:
            model.cuda()

        # define loss func and optimizer
        criterion = nn.NLLLoss()
        if args.optim == 'SGD':
            optimizer = optim.SGD(model.parameters(), args.lr,
                                  momentum=args.momentum,
                                  weight_decay=args.weight_decay)
        elif args.optim == 'Adam':
            optimizer = optim.Adam(model.parameters(), args.lr,
                                  weight_decay=args.weight_decay)
        else:
            raise NameError('Only SGD or Adam is allowed as --optim')

        # optionally resume from a checkpoint
        if args.resume:
            if os.path.isfile(args.resume):
                print("=> loading checkpoint '{}'".format(args.resume))
                checkpoint = torch.load(args.resume)
                args.start_epoch = checkpoint['epoch']
                best_mae_error = checkpoint['best_mae_error']
                model.load_state_dict(checkpoint['state_dict'])
                optimizer.load_state_dict(checkpoint['optimizer'])
                normalizer.load_state_dict(checkpoint['normalizer'])
                print("=> loaded checkpoint '{}' (epoch {})"
                      .format(args.resume, checkpoint['epoch']))
            else:
                print("=> no checkpoint found at '{}'".format(args.resume))

        scheduler = MultiStepLR(optimizer, milestones=args.lr_milestones,
                                gamma=0.1)

        print("Train/Val/Test in Bagging %d started... ----------------------------------------------------------------------------------------------------------" % (bagging+1))

        for epoch in range(args.start_epoch, args.epochs):
            # train for one epoch
            train(train_loader, model, criterion, optimizer, epoch, normalizer)

            # evaluate on validation set
            mae_error, recall_value = validate(val_loader, model, criterion, normalizer, bagging+1)

            if mae_error != mae_error:
                print('Exit due to NaN')
                sys.exit(1)

            scheduler.step()

            # remember the best mae_eror and save checkpoint
            is_best = mae_error > best_mae_error

            if is_best:
                best_epoch = epoch + 1
                best_recall = recall_value

            best_mae_error = max(mae_error, best_mae_error)
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'best_mae_error': best_mae_error,
                'optimizer': optimizer.state_dict(),
                'normalizer': normalizer.state_dict(),
                'args': vars(args)
            }, is_best, bagging)

        # Test final epoch model
        print('In bagging %d, Best epoch = %d, Best AUC = %f, Best recall = %f' % (bagging+1, best_epoch, best_mae_error, best_recall))
        print('---------Evaluate Model on Test-unlabeled Set---------------')
        best_checkpoint = torch.load('checkpoint_bag_'+str(bagging+1)+'.pth.tar')
        model.load_state_dict(best_checkpoint['state_dict'])
        _, bestrecall_bagging = validate(test_loader, model, criterion, normalizer, bagging+1, test=True)

        # Memory Reflush
        gc.collect()

    bootstrap_aggregating(args.bag, prediction=False)



def train(train_loader, model, criterion, optimizer, epoch, normalizer):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    accuracies = AverageMeter()
    precisions = AverageMeter()
    recalls = AverageMeter()
    fscores = AverageMeter()
    auc_scores = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target, _) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        if args.cuda:
            input_var = (Variable(input[0].cuda(non_blocking=True)),
                         Variable(input[1].cuda(non_blocking=True)),
                         input[2].cuda(non_blocking=True),
                         [crys_idx.cuda(non_blocking=True) for crys_idx in input[3]])
        else:
            input_var = (Variable(input[0]),
                         Variable(input[1]),
                         input[2],
                         input[3])
        # normalize target
        target_normed = target.view(-1).long()

        if args.cuda:
            target_var = Variable(target_normed.cuda(non_blocking=True))
        else:
            target_var = Variable(target_normed)

        # compute output
        output = model(*input_var)
        loss = criterion(output, target_var)

        # measure accuracy and record loss
        accuracy, precision, recall, fscore, auc_score = \
            class_eval(output.data.cpu(), target, test=False)
        losses.update(loss.data.cpu().item(), target.size(0))
        accuracies.update(accuracy, target.size(0))
        precisions.update(precision, target.size(0))
        recalls.update(recall, target.size(0))
        fscores.update(fscore, target.size(0))
        auc_scores.update(auc_score, target.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Accu {accu.val:.3f} ({accu.avg:.3f})\t'
                  'Precision {prec.val:.3f} ({prec.avg:.3f})\t'
                  'Recall {recall.val:.3f} ({recall.avg:.3f})\t'
                  'F1 {f1.val:.3f} ({f1.avg:.3f})\t'
                  'AUC {auc.val:.3f} ({auc.avg:.3f})'.format(
                epoch, i, len(train_loader), batch_time=batch_time,
                data_time=data_time, loss=losses, accu=accuracies,
                prec=precisions, recall=recalls, f1=fscores,
                auc=auc_scores)
            )


def validate(val_loader, model, criterion, normalizer, bagging, test=False):
    batch_time = AverageMeter()
    losses = AverageMeter()

    accuracies = AverageMeter()
    precisions = AverageMeter()
    recalls = AverageMeter()
    fscores = AverageMeter()
    auc_scores = AverageMeter()

    if test:
        test_targets = []
        test_preds = []
        test_cif_ids = []

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, (input, target, batch_cif_ids) in enumerate(val_loader):
        if args.cuda:
            with torch.no_grad():
                input_var = (Variable(input[0].cuda(non_blocking=True)),
                             Variable(input[1].cuda(non_blocking=True)),
                             input[2].cuda(non_blocking=True),
                             [crys_idx.cuda(non_blocking=True) for crys_idx in input[3]])
        else:
            with torch.no_grad():
                input_var = (Variable(input[0]),
                             Variable(input[1]),
                             input[2],
                             input[3])
        target_normed = target.view(-1).long()
        if args.cuda:
            with torch.no_grad():
                target_var = Variable(target_normed.cuda(non_blocking=True))
        else:
            with torch.no_grad():
                target_var = Variable(target_normed)

        # compute output
        output = model(*input_var)
        loss = criterion(output, target_var)

        # measure accuracy and record loss
        if test:
            accuracy, precision, recall, fscore = \
                class_eval(output.data.cpu(), target, test=True)
            losses.update(loss.data.cpu().item(), target.size(0))
            accuracies.update(accuracy, target.size(0))
            precisions.update(precision, target.size(0))
            recalls.update(recall, target.size(0))
            fscores.update(fscore, target.size(0))
            #auc_scores.update(auc_score, target.size(0))
            test_pred = torch.exp(output.data.cpu())
            test_target = target
            assert test_pred.shape[1] == 2
            test_preds += test_pred[:, 1].tolist()
            test_targets += test_target.view(-1).tolist()
            test_cif_ids += batch_cif_ids

        else:
            accuracy, precision, recall, fscore, auc_score = \
                class_eval(output.data.cpu(), target, test=False)
            losses.update(loss.data.cpu().item(), target.size(0))
            accuracies.update(accuracy, target.size(0))
            precisions.update(precision, target.size(0))
            recalls.update(recall, target.size(0))
            fscores.update(fscore, target.size(0))
            auc_scores.update(auc_score, target.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Accu {accu.val:.3f} ({accu.avg:.3f})\t'
                  'Precision {prec.val:.3f} ({prec.avg:.3f})\t'
                  'Recall {recall.val:.3f} ({recall.avg:.3f})\t'
                  'F1 {f1.val:.3f} ({f1.avg:.3f})\t'
                  'AUC {auc.val:.3f} ({auc.avg:.3f})'.format(
                i, len(val_loader), batch_time=batch_time, loss=losses,
                accu=accuracies, prec=precisions, recall=recalls,
                f1=fscores, auc=auc_scores))

    if test:
        star_label = '**'
        import csv
        with open('test_results_bag_'+str(bagging)+'.csv', 'w') as f:
            writer = csv.writer(f)
            for cif_id, target, pred in zip(test_cif_ids, test_targets,
                                            test_preds):
                writer.writerow((cif_id, target, pred))
    else:
        star_label = '*'
        print(' {star} Recall(TPR) {recall.avg:.3f}'.format(star=star_label,
                                                 recall=recalls))
        print(' {star} AUC {auc.avg:.3f}'.format(star=star_label,
                                                 auc=auc_scores))
    return auc_scores.avg, recalls.avg


class Normalizer(object):
    """Normalize a Tensor and restore it later. """

    def __init__(self, tensor):
        """tensor is taken as a sample to calculate the mean and std"""
        self.mean = torch.mean(tensor)
        self.std = torch.std(tensor)

    def norm(self, tensor):
        return (tensor - self.mean) / self.std

    def denorm(self, normed_tensor):
        return normed_tensor * self.std + self.mean

    def state_dict(self):
        return {'mean': self.mean,
                'std': self.std}

    def load_state_dict(self, state_dict):
        self.mean = state_dict['mean']
        self.std = state_dict['std']


def mae(prediction, target):
    """
    Computes the mean absolute error between prediction and target

    Parameters
    ----------

    prediction: torch.Tensor (N, 1)
    target: torch.Tensor (N, 1)
    """
    return torch.mean(torch.abs(target - prediction))


def class_eval(prediction, target, test):
    prediction = np.exp(prediction.numpy())
    target = target.numpy()
    pred_label = np.argmax(prediction, axis=1)
    target_label = np.squeeze(target)
    if not target_label.shape:
        target_label = np.asarray([target_label])
    if prediction.shape[1] == 2:
        precision, recall, fscore, _ = metrics.precision_recall_fscore_support(
            target_label, pred_label, average='binary')
        if not test:
            auc_score = metrics.roc_auc_score(target_label, prediction[:, 1])
        accuracy = metrics.accuracy_score(target_label, pred_label)
    else:
        raise NotImplementedError
    if test:
        return accuracy, precision, recall, fscore
    else:
        return accuracy, precision, recall, fscore, auc_score


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def save_checkpoint(state, is_best, bagging):
    filename = 'checkpoint_bag_'+str(bagging+1)+'.pth.tar'
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_highest_AUC_bag_'+str(bagging+1)+'.pth.tar')


def adjust_learning_rate(optimizer, epoch, k):
    """Sets the learning rate to the initial LR decayed by 10 every k epochs"""
    assert type(k) is int
    lr = args.lr * (0.1 ** (epoch // k))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


if __name__ == '__main__':
    main()
