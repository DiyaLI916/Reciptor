import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.backends.cudnn as cudnn
from sklearn.metrics import precision_score, recall_score, f1_score

from data_loader_foodcom import TextLoader
from args import get_parser
from net import CategoryClassification

# =============================================================================
parser = get_parser()
opts = parser.parse_args()
# =============================================================================

if not (torch.cuda.device_count()):
    device = torch.device(*('cpu', 0))
else:
    torch.cuda.manual_seed(opts.seed)
    device = torch.device(*('cuda', 1))

device = ("cuda:1" if torch.cuda.is_available() else "cpu")

def main():
    print('dataset:', opts.full_data_path)
    print('batch size', opts.batch_size)
    model = CategoryClassification()
    model.to(device)

    weights_class = torch.Tensor(opts.numClasses).fill_(1)
    weights_class[0] = 0  # the background class is set to 0, i.e. ignore

    # CrossEntropyLoss combines LogSoftMax and NLLLoss in one single class
    class_crit = nn.CrossEntropyLoss(weight=weights_class).to(device)
    criterion = class_crit

    base_params = model.parameters()
    # optimizer - with lr initialized accordingly
    optimizer = torch.optim.AdamW([
        {'params': base_params}
    ], lr=opts.lr)

    if opts.resume:
        if os.path.isfile(opts.resume):
            checkpoint = torch.load(opts.resume)
            opts.start_epoch = checkpoint['epoch']
            best_val = checkpoint['best_val']
            model.load_state_dict(checkpoint['state_dict'])

            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(opts.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(opts.resume))
            best_val = float('inf')
    else:
        best_val = float('inf')
    valtrack = 0

    print('==There are %d parameter groups' % len(optimizer.param_groups))
    print('==Initial base params lr: %f' % optimizer.param_groups[0]['lr'])

    cudnn.benchmark = True

    if opts.do_test:
        test_loader = torch.utils.data.DataLoader(
            TextLoader(pretrained_embed_path=opts.pretrained_embed_path, full_data=opts.full_data_path, \
                       data_path=opts.data_path, partition='test'),
            batch_size=opts.batch_size, shuffle=True,
            num_workers=opts.workers, pin_memory=True)
        print('Test loader prepared.')
        evaluate(test_loader, model, criterion)
        exit()

    train_loader = torch.utils.data.DataLoader(
        TextLoader(pretrained_embed_path=opts.pretrained_embed_path, full_data=opts.full_data_path, \
                   data_path=opts.data_path, partition='train'),
        batch_size=opts.batch_size, shuffle=True,
        num_workers=opts.workers, pin_memory=True)
    print('Training loader prepared.')

    eval_loader = torch.utils.data.DataLoader(
        TextLoader(pretrained_embed_path=opts.pretrained_embed_path, full_data=opts.full_data_path, \
                   data_path=opts.data_path, partition='val'),
        batch_size=opts.batch_size, shuffle=False,
        num_workers=opts.workers, pin_memory=True)
    print('Evaluation loader prepared.')

    # run epochs
    for epoch in range(opts.start_epoch, opts.epochs):
        print('start training epoch {}:'.format(epoch))
        loss = train(train_loader, model, criterion, optimizer, epoch)

        # evaluate on validation set on val freq
        if (epoch + 1) % opts.valfreq == 0 and epoch != 0:
            results = evaluate(eval_loader, model, criterion)
            val_loss = results['loss']
            # check patience
            if val_loss >= best_val:
                valtrack += 1
            else:
                valtrack = 0

            if valtrack >= opts.patience:
                # change the learning rate accordingly
                adjust_learning_rate(optimizer, epoch, opts)
                valtrack = 0

            # save the best model
            is_best = val_loss < best_val
            best_val = min(val_loss, best_val)

            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'best_val': best_val,
                'optimizer': optimizer.state_dict(),
                'curr_val': val_loss,
            }, is_best)

            print('** Validation: %f (best) - %d (valtrack)' % (best_val, valtrack))

def train(train_loader, model, criterion, optimizer, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    class_losses = AverageMeter()

    # switch to train mode
    model.train()
    end = time.time()
    train_start = time.time()
    for i, (input, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        # print('check foodcom input', input.shape, target)
        input = input.to(device)
        classes = target[0].to(device)
        output = model(input)

        loss = criterion(output, classes)
        class_losses.update(loss.data, input.size(0))
        # print('total loss', loss.data)
        # compute gradient and do Adam step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        # print('1 batch time', time.time() - end)
        end = time.time()

    print('1 epoch time', time.time() - train_start)
    print('Epoch: {0}\t'
          'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
          'lr ({recipeLR})\t'.format(epoch, loss=class_losses,
        recipeLR=optimizer.param_groups[0]['lr']))

    # print('loss', loss.cpu().data.numpy())
    return loss.cpu().data.numpy()

def evaluate(eval_loader, model, criterion):
    model.eval()
    class_eval_losses = AverageMeter()

    preds = None
    out_label_ids = None
    for i, (input, target) in enumerate(eval_loader):
        input = input.to(device)
        classes = target[0].to(device)

        with torch.no_grad():
            logits = model(input)
            loss = criterion(logits, classes)
            class_eval_losses.update(loss.data, input.size(0))

        if preds is None:
            preds = logits.cpu().numpy()
            out_label_ids = classes.detach().cpu().numpy()
        else:
            preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
            out_label_ids = np.append(out_label_ids, classes.detach().cpu().numpy(), axis=0)

    preds = np.argmax(preds, axis=1)

    print(out_label_ids[:], preds[:])
    # exit()

    result = {
        "loss": class_eval_losses.avg.cpu().numpy(),
        "precision": precision_score(out_label_ids, preds, labels=[1, 2, 3, 4, 5, 6, 7, 8], average='micro'),
        "recall": recall_score(out_label_ids, preds, labels=[1, 2, 3, 4, 5, 6, 7, 8], average='micro'),
        "f1": f1_score(out_label_ids, preds, labels=[1, 2, 3, 4, 5, 6, 7, 8], average='micro')
    }

    print('Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(loss=class_eval_losses))
    print(result)

    return result

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    filename = opts.snapshots + 'model_e%03d_v-%.3f.pth' % (state['epoch'], state['best_val'])
    if is_best:
        torch.save(state, filename)
        print('save checkpoint %s' % filename)

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

def adjust_learning_rate(optimizer, epoch, opts):
    """Switching between modalities"""
    # parameters corresponding to the rest of the network
    optimizer.param_groups[0]['lr'] = opts.lr * opts.freeRecipe
    print('Initial base params lr: %f' % optimizer.param_groups[0]['lr'])
    # after first modality change we set patience to 3
    opts.patience = 3

if __name__ == '__main__':
    main()
