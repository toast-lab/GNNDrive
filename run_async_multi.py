import os
import argparse
import time
import threading
import math

from queue import Queue

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel
from tqdm import tqdm
from sage import SAGE,GAT,GCN

from lib.data import *
from lib.neighbor_sampler import MMAPNeighborSampler
from lib.utils import *
from lib.offload_cpu import *
from lib.offload_gpu import *

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
argparser.add_argument('--train-only', dest='train_only', default=True, action='store_true')
argparser.add_argument('--features', type=int, default=128)
argparser.add_argument('--compute-type', type=str, default="gpu")
argparser.add_argument('--world-size', type=int, default=2)
args = argparser.parse_args()

# Set environment and path
dataset_path = os.path.join('./data/dataset', args.dataset + '-ginex')
split_idx_path = os.path.join(dataset_path, 'split_idx.pth')

# Prepare dataset
features_path = os.path.join(dataset_path, 'features-' + str(args.features) + '.dat')
sizes = [int(size) for size in args.sizes.split(',')]

def sampling(res_list, sampling_q, adjs_map, t_id, batch_size, 
             indptr, indices, sizes, node_idx, num_workers, device = None):
    train_loader = MMAPNeighborSampler(indptr, indices, node_idx=node_idx,
                               sizes=sizes, batch_size=batch_size,
                               shuffle=True, num_workers=1)
    for step, (batch_size, ids, adjs) in enumerate(train_loader):
        key = t_id * 10000 + step
        # GPU
        if device is not None:
            adjs = [adj.to(device) for adj in adjs]
        adjs_map[key] = (batch_size, ids, adjs)
        sampling_q.put((key, ids))
    res_list[t_id] = len(train_loader)


def loading(loading_q, sampling_q, loader, rank):
    while True:
        # print("loading queue size", rank, sampling_q.qsize(), loading_q.qsize())
        key, ids = sampling_q.get()
        if key < 0:
            break
        remap_ids = loader.async_load(ids)
        if remap_ids.numel() == 0:
            print("loading error")
            exit(-1)
        # print("loading", remap_ids[0], ids[0], x[remap_ids[0]][0], real[ids[0]][0])
        loading_q.put((key, remap_ids))


def executing(model, optimizer, x, y, loading_q, releasing_q, adjs_map, rank, total_list, pbar):
    total_loss = total_correct = 0

    while True:
        # print("executing queue size", rank, loading_q.qsize())
        key, remap_ids = loading_q.get()
        if key < 0:
            break

        batch_size, ids, adjs = adjs_map.pop(key, (0, 0, 0))
        if batch_size == 0:
            print("executing error", rank, key)
            continue

        # Forward
        out = model(x[remap_ids], adjs)
        loss = F.nll_loss(out, y[ids[:batch_size]])

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        releasing_q.put(ids)

        # Free
        total_loss += float(loss)
        total_correct += int(out.argmax(dim=-1).eq(y[ids[:batch_size]].long()).sum())

        if pbar is not None:
            pbar.update(batch_size)

    total_list[rank*3] = total_loss
    total_list[rank*3+1] = total_correct


def releasing(releasing_q, loader):
    while True:
        # print("releasing queue size", releasing_q.qsize())
        ids = releasing_q.get()
        if len(ids) == 0:
            break
        loader.release(ids)


def train(model, optimizer, x, y, 
          sample_worker_num, sample_q_size, 
          loading_worker_num, loading_q_size, releasing_worker_num, 
          train_idx, indptr, indices, 
          offloader, rank, total_list, world_size, pbar, device = None):

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
                                   math.ceil(args.num_workers / sample_worker_num / world_size), device), 
                                   daemon=True))
        res_list.append(0)
        sample_workers[i].start()

    for i in range(loading_worker_num):
        loading_workers.append(
            threading.Thread(target=loading, 
                             args=(loading_q, sampling_q, offloader, rank), daemon=True))
        loading_workers[i].start()

    executing_worker = threading.Thread(target=executing, 
                        args=(model, optimizer, x, y, loading_q, releasing_q, 
                              adjs_map, rank, total_list, pbar), 
                        daemon=True)
    executing_worker.start()

    for i in range(releasing_worker_num):
        releasing_workers.append(
            threading.Thread(target=releasing, 
                             args=(releasing_q, offloader), daemon=True))
        releasing_workers[i].start()

    for i in sample_workers:
        i.join()

    # print('sample finish', rank)
    for i in range(loading_worker_num):
        sampling_q.put((-1, 0))

    for i in loading_workers:
        i.join()

    # print('loading finish', rank)
    
    loading_q.put((-1, 0))

    executing_worker.join()

    # print('executing finish', rank)
    for i in range(releasing_worker_num):
        releasing_q.put([])

    for i in releasing_workers:
        i.join()

    total_list[rank*3+2] = sum(res_list)



def run(rank, world_size, total_list, num_features, num_classes, compute_type):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '22355'

    if compute_type == 'cpu':
        dist.init_process_group('gloo', rank=rank, world_size=world_size)
    else:
        dist.init_process_group('nccl', rank=rank, world_size=world_size)

    my_rank = dist.get_rank()
    my_size = dist.get_world_size()
    print("my rank = %d  my size = %d" % (my_rank, my_size))

    torch.manual_seed(12345)
    print(num_features, args.num_hiddens, num_classes, len(sizes))
    if args.model == 'sage':
        model = SAGE(num_features, args.num_hiddens, num_classes,
                     num_layers=len(sizes))
    elif args.model == 'gcn':
        model = GCN(num_features, args.num_hiddens, num_classes, 
                    num_layers=len(sizes), norm=True)
    elif args.model == 'gat':
        model = GAT(num_features, args.num_hiddens, num_classes, 
                    num_layers=len(sizes), heads=4)
    else:
        raise NotImplementedError

    if compute_type == 'cpu':
        model = DistributedDataParallel(model)
    else:
        device_num = torch.cuda.device_count()
        device_id = my_rank % device_num
        model = model.to(device_id)
        model = DistributedDataParallel(model, device_ids=[device_id])
    
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    sample_worker_num = 1
    loading_worker_num = 2
    executing_worker_num = 1
    releasing_worker_num = 1


    sample_q_size = sample_worker_num + 2
    loading_q_size = loading_worker_num
    
    if loading_q_size < 2:
        loading_q_size = 2 

    cache_size = args.batch_size * (1 + executing_worker_num)
    for i in sizes:
        cache_size *= i

    if compute_type == 'cpu':
        cache_size = cache_size * world_size

    indptr, indices, num_nodes, train_idx, valid_idx, test_idx = get_mmap_dataset_share2(
        path=dataset_path, split_idx_path=split_idx_path, num_features=args.features)
    
    total_train_size = train_idx.size(0)
    
    train_idx = train_idx.split(total_train_size // world_size)[rank]
    
    if compute_type == 'cpu':
        device = torch.device('cpu')
        offloader = offloadCPU.CPUOffloader(features_path, num_nodes, num_features, cache_size, rank, world_size)
        device_in = None
    else:
        device = torch.device('cuda:%d' % device_id)
        torch.cuda.set_device(device)
        offloader = offloadGPU.GPUOffloader(features_path, num_nodes, num_features, cache_size, rank, world_size, device_id)
        device_in = device

    x = offloader.get_tensor()

    if x.numel() == 0:
        exit(-1)

    y = get_labels(dataset_path, args.features).to(device).long() # TODO .share_memory_()

    for epoch in range(args.num_epochs):
        model.train()
        pbar = None
        dist.barrier()
        if rank == 0:
            start = time.time()
            pbar = tqdm(total=train_idx.size(0))
            pbar.set_description(f'Epoch {epoch:02d}')

        train(model, optimizer, x, y, 
              sample_worker_num, sample_q_size, 
              loading_worker_num, loading_q_size, releasing_worker_num, 
              train_idx, indptr, indices, 
              offloader, rank, total_list, world_size, pbar, device_in)

        dist.barrier()

        if rank == 0:
            pbar.close()

            loss, acc, total_sampling = 0, 0, 0
            for i in range(len(total_list)):
                if i % 3 == 0:
                    loss += total_list[i]
                elif i % 3 == 1:
                    acc += total_list[i]
                else:
                    total_sampling += total_list[i]
            loss /= total_sampling
            acc /= total_train_size

            end = time.time()
            print(f'Epoch {epoch:02d}, Loss: {loss:.4f}, Approx. Train: {acc:.4f}')
            print('Epoch time: {:.4f} ms'.format((end - start) * 1000))

        # if rank == 0 and epoch % 5 == 4:  # We evaluate on a single GPU for now
        #     # eval
        #     print("eval")

        dist.barrier()

    dist.destroy_process_group()


if __name__ == '__main__':
    world_size = args.world_size
    print('Let\'s use', world_size, args.compute_type)

    ctx = mp.get_context('spawn')
    total = ctx.Array('d', 3*world_size)

    num_features, _ = get_feature_info(dataset_path, args.features)
    num_classes = get_num_classes(dataset_path, args.features)
    mp.spawn(run, 
             args=(world_size, total, num_features, num_classes, args.compute_type), 
             nprocs=world_size, join=True)
