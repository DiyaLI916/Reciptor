import os
import time
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.backends.cudnn as cudnn

from data_loader_foodcom import TripletLoader, FullTextLoader
from args import get_parser
from net import ingre2recipe, Reciptor
import pickle

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
    if opts.model_type == 'reciptor':
        model = Reciptor()
        print('run reciptor model, use cosine and triplet loss ...')
    elif opts.model_type == 'jm':
        model = ingre2recipe()
        print('run joint model, use cosine and triplet loss ...')
    elif opts.model_type == 'sjm':
        model = ingre2recipe()
        print('run shallow joint model, use cosine loss only ...')
    else:
        raise Exception('please specify a model type [reciptor|jm|sjm]')

    model.to(device)

    # .module problem
    # if torch.cuda.device_count() > 1:
    #     print("Let's use", torch.cuda.device_count(), "GPUs!")
    #     # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
    #     model = nn.DataParallel(model, device_ids=[1, 2, 3]).cuda()

    # print('cuda current device', torch.cuda.current_device())

    # define loss function (criterion) and optimizer
    cosine_crit = nn.CosineEmbeddingLoss(0.05, reduction='sum').to(device)

    if opts.triplet_loss:
        print('use triplet loss and cosine loss')
        triplet_loss = nn.TripletMarginLoss(margin=0.05, p=2, reduction='sum').to(device)
        criterion = [cosine_crit, triplet_loss]
    else:
        print('use cosine loss solely')
        criterion = cosine_crit

    base_params = model.parameters()

    # optimizer - with lr initialized accordingly
    optimizer = torch.optim.Adam([
        {'params': base_params}
    ], lr=opts.lr * opts.freeRecipe)

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

    if opts.save_tuned_embed:
        data_loader = torch.utils.data.DataLoader(
            FullTextLoader(data_path=opts.data_path, full_data=opts.full_data_path, save_tuned_embed=opts.save_tuned_embed),
            batch_size=opts.batch_size, shuffle=False,
            num_workers=opts.workers, pin_memory=True)

        write2tensorboard(data_loader, model)
        exit()

    if opts.triplet_loss:
        if opts.batch_size % 3 != 0:
            raise Exception('Invalid batch size, try 3*N ...')
        train_loader = torch.utils.data.DataLoader(
            TripletLoader(data_path=opts.data_path, triplet_path=opts.triplet_path, full_data=opts.full_data_path),
            batch_size=opts.batch_size, shuffle=False)
            # num_workers=opts.workers, pin_memory=True)
        print('Triplet training loader prepared.')
    else:
        train_loader = torch.utils.data.DataLoader(
            FullTextLoader(data_path=opts.data_path, full_data=opts.full_data_path, save_tuned_embed=opts.save_tuned_embed),
            batch_size=opts.batch_size, shuffle=True,
            num_workers=opts.workers, pin_memory=True)
        # don't use pin memory
        print('Training loader prepared.')

    # run epochs
    for epoch in range(opts.start_epoch, opts.epochs):
        print('start training epoch {}:'.format(epoch))
        loss = train(train_loader, model, criterion, optimizer, epoch)

        # evaluate on validation set on val freq
        if (epoch + 1) % opts.valfreq == 0 and epoch != 0:
            val_loss = loss
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
    cos_losses = AverageMeter()
    if opts.triplet_loss:
        tri_losses = AverageMeter()

    # if opts.semantic_reg:
    #     rec_losses = AverageMeter()

    # switch to train mode
    model.train()
    end = time.time()
    train_start = time.time()
    for i, (input, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        # print('check foodcom input', target[0], target[2][:10])
        # exit()

        input_var = list()
        for j in range(len(input)):
            input_var.append(input[j].to(device))

        target_var = list()
        for j in range(len(target) - 1):
            target_var.append(target[j].to(device))

        # compute output
        output = model(input_var[0], input_var[1], input_var[2], input_var[3])
        [skip_emb, ingre_emb] = output
        # print('output', output)

        if i % 10 == 0:
            cos = nn.CosineSimilarity(dim=1, eps=1e-6)
            output = cos(skip_emb[:8, :], skip_emb[8:16, :])
            print('skip_emb[:8, :], skip_emb[8:16, :]\n', output.cpu().data)
            output = cos(ingre_emb[:8, :], ingre_emb[8:16, :])
            print('ingre_emb[:8, :], ingre_emb[8:16, :]\n', output.cpu().data)

        if opts.triplet_loss:
            anchor_batch = ingre_emb[0::3, :]
            pos_batch = ingre_emb[1::3, :]
            negative_batch = ingre_emb[2::3, :]
            # print(anchor_batch.shape, pos_batch.shape, negative_batch.shape)
            # exit()

            cos_loss = criterion[0](skip_emb, ingre_emb, target_var[0].float())
            tri_loss = criterion[1](anchor_batch, pos_batch, negative_batch)
            # print('cos, tri', cos_loss.cpu().data, tri_loss.cpu().data)

            # combined loss
            loss = (1 - opts.tri_weight) * cos_loss + opts.tri_weight * tri_loss * 3
            # print('total loss', loss.data)
            # exit()

            # measure performance and record losses
            cos_losses.update(cos_loss.data, input[0].size(0))
            tri_losses.update(tri_loss.data, input[0].size(0) / 3)

        else:
            loss = criterion(skip_emb, ingre_emb, target_var[0].float())
            cos_losses.update(loss.data, input[0].size(0))
            print('total loss', loss.data)

        # compute gradient and do Adam step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        # print('1 batch time', time.time() - end)
        end = time.time()

    print('1 epoch time', time.time() - train_start)

    if opts.triplet_loss:
        print('Epoch: {0}\t'
              'cos loss {cos_loss.val:.4f} ({cos_loss.avg:.4f})\t'
              'triplet loss {tri_loss.val:.4f} ({tri_loss.avg:.4f})\t'
              'recipe ({recipeLR})\t'.format(epoch, cos_loss=cos_losses,
            tri_loss=tri_losses, recipeLR=optimizer.param_groups[0]['lr']))
    else:
        print('Epoch: {0}\t'
              'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
              'lr ({recipeLR})\t'.format(epoch, loss=cos_losses,
            recipeLR=optimizer.param_groups[0]['lr']))

    print('loss', loss.cpu().data.numpy())

    return loss.cpu().data.numpy()

# save the trained embeddings
def write2tensorboard(data_loader, model):
    # switch to evaluate mode
    model.eval()

    print('start save the tuned embeddings')
    for i, (input, target) in enumerate(data_loader):
        input_var = list()
        for j in range(len(input)):
            input_var.append(input[j].to(device))
        # print(target)
        # exit()

        # compute output
        output = model(input_var[0], input_var[1], input_var[2], input_var[3])
        # now: [skip_emb, ingre_emb, ingre_sem]

        # [target, rec_class, rec_id]
        if i == 0:
            skip_emb = output[0].data.cpu().numpy()
            ingre_emb = output[1].data.cpu().numpy()
            id = target[-1]
            recipe_class = target[-2]
        else:
            skip_emb = np.concatenate((skip_emb, output[0].data.cpu().numpy()), axis=0)
            ingre_emb = np.concatenate((ingre_emb, output[1].data.cpu().numpy()), axis=0)
            id = np.concatenate((id, target[-1]), axis=0)
            recipe_class = np.concatenate((recipe_class, target[-2]), axis=0)

    with open('tuned_embed/new2_{}_{}_tuned_emb.pkl'.format(opts.model_type, id.size), 'wb') as f:
        pickle.dump(skip_emb, f)
        pickle.dump(ingre_emb, f)
        pickle.dump(id, f)
        pickle.dump(recipe_class, f)
        print('created tuned_embed/new2_{}_{}_tuned_emb.pkl'.format(opts.model_type, id.size))
    return 0

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
