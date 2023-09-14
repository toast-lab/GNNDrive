import argparse
import os
import time
import torch
import torch.nn.functional as F
from tqdm import tqdm
import threading
from queue import Queue
from sage import SAGE,GAT,GCN
import math

from lib.data import *
from lib.neighbor_sampler import MMAPNeighborSampler
from lib.utils import *
from lib.offload import *

# Parse arguments
argparser = argparse.ArgumentParser()
argparser.add_argument('--gpu', type=int, default=0)
argparser.add_argument('--num-epochs', type=int, default=5)
argparser.add_argument('--batch-size', type=int, default=1000)
argparser.add_argument('--num-workers', type=int, default=32)
argparser.add_argument('--num-hiddens', type=int, default=256)
argparser.add_argument('--lr', type=float, default=0.003)
argparser.add_argument('--model', type=str, default="sage")
argparser.add_argument('--dataset', type=str, default='ogbn-papers100M')
argparser.add_argument('--sizes', type=str, default='10,10,10')
argparser.add_argument('--train-only', dest='train_only', default=False, action='store_true')
argparser.add_argument('--features', type=int, default=128)
argparser.add_argument('--compute-type', type=str, default="gpu")
argparser.add_argument('--buffer-size', type=float, default=1)
argparser.add_argument('--fallback', type=int, default=0)
args = argparser.parse_args()

# Set environment and path
dataset_path = os.path.join('./data/dataset', args.dataset + '-ginex')
split_idx_path = os.path.join(dataset_path, 'split_idx.pth')

# Prepare dataset
features_path = os.path.join(dataset_path, 'features-' + str(args.features) + '.dat')
sizes = [int(size) for size in args.sizes.split(',')]

sample_worker_num = 2
loading_worker_num = 2
executing_worker_num = 1
releasing_worker_num = 1

fallback_mode = bool(args.fallback)

sample_q_size = sample_worker_num + 2
loading_q_size = loading_worker_num

each_batch = args.batch_size
if each_batch < 1000:
    each_batch = 1000

cache_size = each_batch
for i in sizes:
    cache_size *= i

stage_size = cache_size * loading_worker_num

cache_size = int(cache_size * (loading_q_size + executing_worker_num) * args.buffer_size)

indptr, indices, y, num_features, num_classes, num_nodes, train_idx, valid_idx, test_idx = get_mmap_dataset_async(
    path=dataset_path, split_idx_path=split_idx_path, num_features=args.features)


# Define model
if args.compute_type == 'cpu':
    device = torch.device('cpu')
    offloader = offload.Offloader(features_path, num_nodes, num_features, cache_size, 'cpu', 0, 0)
else:
    device = torch.device('cuda:%d' % args.gpu)
    torch.cuda.set_device(device)
    if (fallback_mode):
        offloader = offload.Offloader(features_path, num_nodes, num_features, cache_size, 'cpu', args.gpu, 0)
    else:
        offloader = offload.Offloader(features_path, num_nodes, num_features, cache_size, 'gpu', args.gpu, stage_size)

x = offloader.get_tensor()

if x.numel() == 0:
    exit(-1)

if fallback_mode:
    y = y.long()
else:
    y = y.to(device).long()

if args.model == 'sage':
    model = SAGE(num_features, args.num_hiddens, num_classes, num_layers=len(sizes))
elif args.model == 'gcn':
    model = GCN(num_features, args.num_hiddens, num_classes, num_layers=len(sizes), 
                norm=True)
elif args.model == 'gat':
    model = GAT(num_features, args.num_hiddens, num_classes, num_layers=len(sizes),
                heads=4)
else:
    raise NotImplementedError

model = model.to(device)



def sampling(res_list, sampling_q, adjs_map, t_id, batch_size, 
             indptr, indices, sizes, node_idx, num_workers):
    train_loader = MMAPNeighborSampler(indptr, indices, node_idx=node_idx,
                               sizes=sizes, batch_size=batch_size,
                               shuffle=True, num_workers=num_workers)
    for step, (batch_size, ids, adjs) in enumerate(train_loader):
        key = t_id * 10000 + step
        # GPU
        if args.compute_type == 'gpu' and not fallback_mode:
            adjs = [adj.to(device) for adj in adjs]
        adjs_map[key] = (batch_size, ids, adjs)
        sampling_q.put((key, ids))
    res_list[t_id] = len(train_loader)


def loading(loading_q, sampling_q, loader, t_id, t_total):
    while True:
        # print("loading queue size", sampling_q.qsize(), loading_q.qsize())
        key, ids = sampling_q.get()
        if key < 0:
            break
        remap_ids = loader.async_load(ids, t_id, t_total)
        if remap_ids.numel() == 0:
            print("loading error")
            exit(-1)
        # print("loading", remap_ids[0], ids[0], x[remap_ids[0]][0], real[ids[0]][0])
        loading_q.put((key, remap_ids))


def executing(loading_q, releasing_q, adjs_map, t_id, pbar, total_list):
    total_loss = total_correct = 0

    while True:
        # print("executing queue size", loading_q.qsize())
        key, remap_ids = loading_q.get()
        if key < 0:
            break

        batch_size, ids, adjs = adjs_map.pop(key, (0, 0, 0))
        if batch_size == 0:
            print("executing error", t_id, key)
            continue

        # Forward
        if fallback_mode:
            cuda_x = x[remap_ids].to(device)
            cuda_y = y[ids[:batch_size]].to(device)
            cuda_adjs = [adj.to(device) for adj in adjs]
            out = model(cuda_x, cuda_adjs)
            loss = F.nll_loss(out, cuda_y)
        else:
            out = model(x[remap_ids], adjs)
            loss = F.nll_loss(out, y[ids[:batch_size]])

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        releasing_q.put(ids)

        # Free
        total_loss += float(loss)
        if fallback_mode:
            total_correct += int(out.argmax(dim=-1).eq(cuda_y).sum())
        else:
            total_correct += int(out.argmax(dim=-1).eq(y[ids[:batch_size]].long()).sum())

        if args.compute_type == 'gpu':
            del(adjs)
            if fallback_mode:
                del(cuda_x)
                del(cuda_y)
                del(cuda_adjs)
                torch.cuda.empty_cache()

        pbar.update(batch_size)

    total_list[0] = total_loss
    total_list[1] = total_correct


def releasing(releasing_q, loader):
    while True:
        # print("releasing queue size", releasing_q.qsize())
        ids = releasing_q.get()
        if len(ids) == 0:
            break
        loader.release(ids)


def train(epoch):
    model.train()

    pbar = tqdm(total=train_idx.size(0))
    pbar.set_description(f'Epoch {epoch:02d}')

    
    sample_workers = []
    loading_workers = []
    releasing_workers = []

    # Queues for parallel execution of CPU & GPU operations via multi-threading
    adjs_map = {}
    res_list = []

    sampling_q = Queue(maxsize=sample_q_size)
    loading_q = Queue(maxsize=loading_q_size) 
    releasing_q = Queue()

    last_train_id = 0
    each_train_num = math.ceil(train_idx.size(0) / sample_worker_num)
    for i in range(sample_worker_num):
        if i == sample_worker_num - 1:
            t_train_idx = train_idx[last_train_id:]
        else:
            t_train_idx = train_idx[last_train_id:last_train_id+each_train_num]
            last_train_id += each_train_num
        sample_workers.append(
            threading.Thread(target=sampling, 
                             args=(res_list, sampling_q, adjs_map, i, args.batch_size, 
                                   indptr, indices, sizes, t_train_idx, 
                                   math.ceil(args.num_workers / sample_worker_num)), 
                                   daemon=True))
        res_list.append(0)
        sample_workers[i].start()

    for i in range(loading_worker_num):
        loading_workers.append(
            threading.Thread(target=loading, 
                             args=(loading_q, sampling_q, offloader, i, loading_worker_num), daemon=True))
        loading_workers[i].start()

    total_list = []
    executing_workers = []

    for i in range(executing_worker_num):
        total_list.append([0, 0])
        executing_workers.append(
            threading.Thread(target=executing, 
                        args=(loading_q, releasing_q, adjs_map, i, pbar, total_list[i]), 
                        daemon=True))
        executing_workers[i].start()

    for i in range(releasing_worker_num):
        releasing_workers.append(
            threading.Thread(target=releasing, 
                             args=(releasing_q, offloader), daemon=True))
        releasing_workers[i].start()

    for i in sample_workers:
        i.join()

    # print('sample finish')
    for i in range(loading_worker_num):
        sampling_q.put((-1, 0))

    for i in loading_workers:
        i.join()

    # print('loading finish')
    for i in range(executing_worker_num):
        loading_q.put((-1, 0))

    for i in executing_workers:
        i.join()

    # print('executing finish')
    for i in range(releasing_worker_num):
        releasing_q.put([])

    for i in releasing_workers:
        i.join()

    pbar.close()

    all_total = [0, 0]

    for i in total_list:
        all_total[0] += i[0]
        all_total[1] += i[1]

    loss = all_total[0] / sum(res_list)
    approx_acc = all_total[1] / train_idx.size(0)

    return loss, approx_acc


@torch.no_grad()
def inference(mode='test'):
    model.eval()

    if mode == 'test':
        pbar = tqdm(total=test_idx.size(0))
    elif mode == 'valid':
        pbar = tqdm(total=valid_idx.size(0))
    pbar.set_description('Evaluating')

    total_loss = total_correct = 0

    if mode == 'test':
        inference_loader = MMAPNeighborSampler(indptr, indices, 
                            node_idx=test_idx,
                            sizes=sizes, batch_size=args.batch_size,
                            shuffle=False, num_workers=args.num_workers)
    elif mode == 'valid':
        inference_loader = MMAPNeighborSampler(indptr, indices, 
                            node_idx=valid_idx,
                            sizes=sizes, batch_size=args.batch_size,
                            shuffle=False, num_workers=args.num_workers)

    mmap_x = get_mmap_x(path=dataset_path, split_idx_path=split_idx_path, num_features=args.features)
    # Sample
    for step, (batch_size, ids, adjs) in enumerate(inference_loader):
        # Gather
        batch_inputs = gather_mmap(mmap_x, ids)
        batch_labels = y[ids[:batch_size]]

        # Transfer
        batch_inputs_cuda = batch_inputs.to(device)
        batch_labels_cuda = batch_labels.to(device)
        adjs = [adj.to(device) for adj in adjs]

        # Forward
        out = model(batch_inputs_cuda, adjs)
        loss = F.nll_loss(out, batch_labels_cuda.long())
        tensor_free(batch_inputs)

        torch.cuda.synchronize()
        total_loss += float(loss)
        total_correct += int(out.argmax(dim=-1).eq(batch_labels_cuda.long()).sum())
        pbar.update(batch_size)

    pbar.close()

    loss = total_loss / len(inference_loader)
    if mode == 'test':
        approx_acc = total_correct / test_idx.size(0)
    elif mode == 'valid':
        approx_acc = total_correct / valid_idx.size(0)

    return loss, approx_acc


if __name__=='__main__':
    model.reset_parameters()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_val_acc = final_test_acc = 0
    for epoch in range(args.num_epochs):
        start = time.time()
        loss, acc = train(epoch)
        end = time.time()
        print(f'Epoch {epoch:02d}, Loss: {loss:.4f}, Approx. Train: {acc:.4f}')
        print('Epoch time: {:.4f} ms'.format((end - start) * 1000))

        if epoch > 3 and not args.train_only:
            val_loss, val_acc = inference(mode='valid')
            test_loss, test_acc = inference(mode='test')
            print ('Valid loss: {0:.4f}, Valid acc: {1:.4f}, Test loss: {2:.4f}, Test acc: {3:.4f},'.format(val_loss, val_acc, test_loss, test_acc))

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                final_test_acc = test_acc

    if not args.train_only:
        tqdm.write('Final Test acc: {:.4f}'.format(final_test_acc))