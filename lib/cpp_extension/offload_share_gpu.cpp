#include <stdio.h>
#include <ATen/ATen.h>
#include <torch/extension.h>
#include <unistd.h>
#include <fcntl.h>
#include <string>
#include <stdexcept>
#include <unordered_map>
#include <list>
#include <error.h>
#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <sys/uio.h>
#include <omp.h>
#include <atomic>
#include <liburing.h>
#include <cuda_runtime.h>
#include <cstring>

#include <sys/ipc.h>
#include <sys/shm.h>

#define ALIGNMENT 512
#define ASYNC_ENYRY_NUM 80

#define SHARE_NAME "data/sharedfile"
#define SHARE_KEY 8

enum class AsyncType {
    CPU,
    GPU,
    GDS,
    None
};

typedef struct map_info_s
{
    int64_t index;
    int32_t ref;
    int32_t valid;
} map_info;

class GPUOffloader
{
public:
    GPUOffloader(const std::string &filename, const int64_t node_num, 
            const int64_t dim, const int64_t buffer_size, 
            int rank, int world_size, int device_id);
    ~GPUOffloader();

    torch::Tensor get_tensor();

    torch::Tensor async_load(torch::Tensor &idx);

    void release(torch::Tensor &idx);

private:
    AsyncType async_type = AsyncType::GPU;

    const std::string filename;
    int fd;

    char *shared_mem = NULL;
    int shmid;
    int rank = 0;
    int world_size = 0;
    float *cache_data;
    int64_t *host_map_table;
    int64_t *host_back_index;

    torch::Tensor feature_tensor;
    float *device_cache;
    int64_t feature_dim;
    int64_t cache_size;
    std::vector<int64_t> back_index;

    std::mutex update_mutex;
    int64_t node_size;
    std::vector<map_info> map_table;
    // free table
    int group_size;
    int64_t free_index_size;
    std::list<int64_t> free_lru_list;
    std::unordered_map<int64_t, std::list<int64_t>::iterator> free_map_table;

    int64_t get_free_index() {
        if (this->free_lru_list.empty()) {
            return -1;
        }

        int64_t index = this->free_lru_list.front();
        this->free_lru_list.pop_front();
        this->free_map_table.erase(index);
        int64_t orignal_key = this->back_index[index];
        if (orignal_key >= 0)
        {
            this->map_table[orignal_key].valid = 0;
        }
        return index;
    }

    void put_free_index(int64_t index) {
        this->free_lru_list.push_back(index);
        auto it = this->free_lru_list.end();
        it--;
        this->free_map_table.insert({index, it});
    }

    bool reuse_free_index(int64_t index){
        auto it = this->free_map_table.find(index);
        if (it != this->free_map_table.end()) {
            this->free_lru_list.erase(it->second);
            this->free_map_table.erase(it);
            return true;
        }
        return false;
    }
    
    torch::Tensor gpu_async_load(torch::Tensor &idx);

    void load_callback(int key, int host_index, cudaStream_t& cuda_read_stream);
    
};


GPUOffloader::GPUOffloader(const std::string &filename, const int64_t node_num, 
    const int64_t dim, const int64_t buffer_size, 
    int rank, int world_size, int device_id) 
    : filename(filename), node_size(node_num), feature_dim(dim), cache_size(buffer_size), rank(rank), world_size(world_size)
{
    this->group_size = ALIGNMENT / (this->feature_dim * sizeof(float));
    if (this->group_size < 1) {
        this->group_size = 1;
    }

    this->free_index_size = this->cache_size;
    this->cache_size = this->cache_size * group_size;

    // shared memory
    size_t cache_data_size = this->cache_size * feature_dim * sizeof(float);
    if (cache_data_size % ALIGNMENT)
        cache_data_size = (cache_data_size / ALIGNMENT + 1) * ALIGNMENT;

    size_t map_table_size = node_size * sizeof(int64_t);

    size_t back_index_size = this->free_index_size * sizeof(int64_t);

    size_t mem_size = cache_data_size + map_table_size + back_index_size;

    // if (mem_size % 4096)
    //     mem_size = (mem_size / 4096 + 1) * 4096;

    key_t key = ftok(SHARE_NAME, SHARE_KEY);
    if(key==-1)
        fprintf(stderr, "ftok error %d: %d %s\n", key, errno, strerror(errno));
  
    this->shmid = shmget(key, mem_size, 0666 | IPC_CREAT);
    if(this->shmid==-1)
        fprintf(stderr, "shmget error %d: %llu %d %s\n", key, mem_size, errno, strerror(errno));

    this->shared_mem = (char *)shmat(this->shmid, NULL, 0);
    if (this->shared_mem ==  (char *) -1)
        fprintf(stderr, "shmat error %d: %d %s\n", key, errno, strerror(errno));

    this->cache_data = (float *)this->shared_mem;
    this->host_map_table = (int64_t *)(this->shared_mem + cache_data_size);
    this->host_back_index = (int64_t *)(this->shared_mem + cache_data_size + map_table_size);

    if (this->rank == 0) {
        memset(this->shared_mem, 0, cache_data_size);
        memset(this->host_map_table, -1, map_table_size);
        memset(this->host_back_index, -1, back_index_size);
    }

    this->fd = open(filename.c_str(), O_RDONLY | O_DIRECT);
    if (this->fd < 0)
    {
        fprintf(stderr, "open file %s failed %s\n", filename.c_str(), strerror(errno));
    }

    cudaSetDevice(device_id);

    cudaMalloc(&this->device_cache, cache_data_size);

    auto options = torch::TensorOptions()
        .dtype(torch::kFloat32)
        .layout(torch::kStrided)
        .device(torch::kCUDA, device_id)
        .requires_grad(false);
    
    this->feature_tensor = torch::from_blob(this->device_cache, 
            {this->cache_size, this->feature_dim}, options);

    this->map_table.resize(this->node_size);
    this->back_index.resize(this->cache_size);
    this->back_index.assign(this->cache_size, -1);

    for (int64_t i = 0; i < this->free_index_size; i++)
        put_free_index(i);

    printf("GPUOffloader init done %s %ld %ld %ld %d %llu %llu %llu\n", filename.c_str(), 
        this->node_size, this->feature_dim, this->cache_size, this->rank, mem_size, mem_size % 4096, cache_data_size);
}

GPUOffloader::~GPUOffloader()
{
    if(shmdt(this->shared_mem)==-1)
        perror("detach error ");

    if (this->rank == 0)
        shmctl(this->shmid, IPC_RMID, NULL);

    cudaFree(this->device_cache);

    close(this->fd);
}

torch::Tensor GPUOffloader::get_tensor()
{
    if (this->fd > 0)
        return this->feature_tensor;
    else
        return torch::zeros(0);
}

torch::Tensor GPUOffloader::async_load(torch::Tensor &idx) 
{
    switch (this->async_type)
    {
    case AsyncType::CPU:
        // return cpu_async_load(idx);
        fprintf(stderr, "Not support: %d\n", this->async_type);
        break;
    case AsyncType::GPU:
        return gpu_async_load(idx);
    case AsyncType::GDS:
        // return gds_async_load(idx);
    default:
        fprintf(stderr, "Not support: %d\n", this->async_type);
        break;
    }
}

void GPUOffloader::load_callback(int key, int host_index, cudaStream_t& cuda_read_stream)
{
    this->host_back_index[host_index] = key;

    int index = this->map_table[key].index;

    this->map_table[key].valid = 1;

    float *host_buffer;
    host_buffer = this->cache_data + host_index * this->group_size * this->feature_dim;
    float *dev_buffer;
    dev_buffer = this->device_cache + index * this->group_size * this->feature_dim;
    unsigned cuda_nbytes = this->feature_dim * sizeof(float);
    if (cuda_nbytes < ALIGNMENT)
        cuda_nbytes = ALIGNMENT;
    cudaMemcpyAsync(dev_buffer, host_buffer, cuda_nbytes,
                    cudaMemcpyHostToDevice, cuda_read_stream);
}

// ssd -> host mem -> gpu mem
torch::Tensor GPUOffloader::gpu_async_load(torch::Tensor &idx) 
{
    omp_lock_t lock;
    omp_init_lock(&lock);
    bool need_load = false;
    std::unordered_set<int64_t> need_wait;

    torch::Tensor remap_idx = torch::zeros_like(idx);
    int64_t num_idx = idx.numel();
    auto idx_data = idx.data_ptr<int64_t>();
    auto remap_data = remap_idx.data_ptr<int64_t>();

    this->update_mutex.lock();

    for (int64_t n = 0; n < num_idx; n++) {
        int64_t key = idx_data[n];
        int64_t offset = 0;
        if (this->group_size > 1) {
            offset = key % this->group_size;
            key = key / this->group_size * this->group_size;
        }
        if (key > this->map_table.size())
            printf("key %lld\n", key);
        if (this->map_table[key].valid > 0) {
            remap_data[n] = this->map_table[key].index * this->group_size + offset;
            if (this->map_table[key].ref == 0) {
                omp_set_lock(&lock);
                reuse_free_index(this->map_table[key].index);
                omp_unset_lock(&lock);
            }
        } else {
            if (this->map_table[key].ref > 0) {
                remap_data[n] = this->map_table[key].index * this->group_size + offset;
                omp_set_lock(&lock);
                need_wait.insert(key);
                omp_unset_lock(&lock);
            } else {
                remap_data[n] = -1;
                need_load = true;
            }
        }
        this->map_table[key].ref += 1;
    }

    if (need_load > 0) {
        io_uring ring;
        int64_t finished = 0;
        int64_t async_loading = 0;
        cudaStream_t read_stream;
        cudaStreamCreate(&read_stream);

        int ret = io_uring_queue_init(ASYNC_ENYRY_NUM, &ring, 0);
        if (ret)
        {
            fprintf(stderr, "Unable to setup io_uring: %s\n", strerror(-ret));
            goto err_lock;
        }

        for (int64_t n = 0; n < num_idx; n++)
        {
            // data in device mem
            if (remap_data[n] >= 0)
                continue;

            int64_t key = idx_data[n];
            int64_t offset = 0;
            if (this->group_size > 1) {
                // loaded before
                offset = key % this->group_size;
                key = key / this->group_size * this->group_size;
                if (this->back_index[this->map_table[key].index] == key) {
                    remap_data[n] = this->map_table[key].index * this->group_size + offset;
                    continue;
                }
            }

            // data in shared mem (may conflict with other process)
            if (this->host_map_table[key] >= 0)
            {
                int host_index = this->host_map_table[key];
                if (this->host_back_index[host_index] == key)
                {
                    int64_t index = get_free_index();
                    if (index < 0)
                    {
                        fprintf(stderr, "No free table in gpu");
                        goto err_lock;
                    }
                    remap_data[n] = index * this->group_size + offset;
                    this->map_table[key].index = index;
                    this->back_index[index] = key;
                    load_callback(key, host_index, read_stream);
                    continue;
                }
            }

            // data in disk
            io_uring_sqe *sqe = io_uring_get_sqe(&ring);
            if (!sqe)
            {
                fprintf(stderr, "Could not get SQE.\n");
                goto err_lock;
            }

            int64_t host_index = this->free_index_size / this->world_size * this->rank + n;
            if (host_index >= this->free_index_size)
            {
                fprintf(stderr, "No free table in host. %d %d %d\n", this->free_index_size, host_index, n);
                goto err_lock;
            } else {
                this->host_map_table[key] = host_index;
            }

            int64_t index = get_free_index();
            if (index < 0)
            {
                fprintf(stderr, "No free table in gpu.\n");
                goto err_lock;
            }
            remap_data[n] = index * this->group_size + offset;
            this->map_table[key].index = index;            
            this->back_index[index] = key;

            float *f_buffer;
            f_buffer = this->cache_data + host_index * this->group_size * this->feature_dim;
            unsigned f_nbytes = this->feature_dim * sizeof(float);
            if (f_nbytes < ALIGNMENT)
                f_nbytes = ALIGNMENT;
            __u64 f_offset = key * this->feature_dim * sizeof(float);
            io_uring_prep_read(sqe, this->fd, f_buffer, f_nbytes, f_offset);
            sqe->user_data = static_cast<uint64_t>(key);
            io_uring_submit(&ring);
            async_loading += 1;

            io_uring_cqe *cqe;
            ret = io_uring_peek_cqe(&ring, &cqe);
            if (ret == 0)
            {
                int64_t cqe_key = static_cast<int64_t>(cqe->user_data);
                if (cqe->res < 0)
                {
                    fprintf(stderr, "Error in async operation: %s %d\n", strerror(-cqe->res), cqe_key);
                }
                io_uring_cqe_seen(&ring, cqe);
                finished += 1;
                load_callback(cqe_key, this->host_map_table[cqe_key], read_stream);
            }
        }
        this->update_mutex.unlock();

        while (finished < async_loading)
        {
            io_uring_cqe *cqe;
            ret = io_uring_wait_cqe(&ring, &cqe);
            if (ret < 0)
            {
                fprintf(stderr, "Error waiting for completion: %s\n", strerror(-ret));
                goto err;
            }
            int64_t cqe_key = static_cast<int64_t>(cqe->user_data);
            if (cqe->res < 0)
            {
                fprintf(stderr, "Error in async operation: %s %d\n", strerror(-cqe->res), cqe_key);
            }
            io_uring_cqe_seen(&ring, cqe);
            finished += 1;
            load_callback(cqe_key, this->host_map_table[cqe_key], read_stream);
        }
        io_uring_queue_exit(&ring);

        cudaStreamSynchronize(read_stream);
        cudaStreamDestroy(read_stream);
    } else {
        this->update_mutex.unlock();
    }

    for (int64_t key : need_wait) {
        while (this->map_table[key].valid == 0)
        {
            continue;
        }
    }
    return remap_idx;

err_lock:
    this->update_mutex.unlock();
err:
    return torch::zeros(0);
}


void GPUOffloader::release(torch::Tensor &idx)
{
    omp_lock_t lock;
    omp_init_lock(&lock);

    int64_t num_idx = idx.numel();
    auto idx_data = idx.data_ptr<int64_t>();

    this->update_mutex.lock();

    for (int64_t n = 0; n < num_idx; n++) {
        int64_t key = idx_data[n];
        if (this->group_size > 1) {
            key = key / this->group_size * this->group_size;
        }
        this->map_table[key].ref -= 1;
        if (this->map_table[key].ref == 0) {
            omp_set_lock(&lock);
            put_free_index(this->map_table[key].index);
            omp_unset_lock(&lock);
        }
    }

    this->update_mutex.unlock();
}


namespace py = pybind11;

PYBIND11_MODULE(offloadGPU, m)
{
    py::class_<GPUOffloader>(m, "GPUOffloader")
        .def(py::init<const std::string &, const int64_t, const int64_t, 
             const int64_t, int, int, int>(),
             py::arg("filename"), py::arg("node_num"), py::arg("dim"), 
             py::arg("buffer_size"), py::arg("rank"), py::arg("world_size"), py::arg("device_id"))
        .def("async_load", &GPUOffloader::async_load, py::arg("tensor"))
        .def("release", &GPUOffloader::release, py::arg("tensor"))
        .def("get_tensor", &GPUOffloader::get_tensor);
}

