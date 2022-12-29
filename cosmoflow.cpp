// C++/MPI proxy • CosmoFlow model 
// Distributed training (hybrid of model x data parallelism)

#include <mpi.h>
#include <unistd.h>
#include <stdio.h>
#include <string>
#include <time.h>
#include <stdlib.h>
#include <assert.h>

#define WARM_UP 8
#define RUNS 128

#define NUM_LAYERS 8


int fwd_rt_per_layer[NUM_LAYERS] = {6567, 13135, 6567, 3283, 1641, 5, 3, 1};
int bwd_rt_per_layer[NUM_LAYERS] = {2, 6, 10, 3283, 6567, 13135, 26270, 13135};

#define NUM_CONV_LAYERS 5
int conv_fwd_halo_sizes[NUM_CONV_LAYERS-1] = {2097152, 1048576, 524288, 262144};
int conv_bwd_halo_sizes[NUM_CONV_LAYERS-1] = {131072, 262144, 524288, 1048576};

#define NUM_DENSE_LAYERS 3
int dense_fwd_allgather_sizes[NUM_DENSE_LAYERS] = {65536, 256, 128};
int dense_bwd_reduce_scatter_sizes[NUM_DENSE_LAYERS] = {128, 256, 65536};

int allreduce_sizes[NUM_LAYERS-2] = {1050737, 3539456, 884992, 221312, 55360, 3488};


// Func declarations
void initialize_mpi(int* rank, int* world_size, int* dense_allreduce_group_rank, int* mp_group_rank, int* dense_allreduce_group_size, int* mp_group_size, MPI_Comm* dense_allreduce_comm, MPI_Comm* model_parallel_comm);
void allocate_buffers(float*** fwd_halo_send_buff0_ptrs, float*** fwd_halo_send_buff1_ptrs, float*** fwd_halo_recv_buff0_ptrs, float*** fwd_halo_recv_buff1_ptrs, float*** bwd_halo_send_buff0_ptrs, float*** bwd_halo_send_buff1_ptrs, float*** bwd_halo_recv_buff0_ptrs, float*** bwd_halo_recv_buff1_ptrs, float*** dense_fwd_allgather_sbuff_ptrs, float*** dense_fwd_allgather_rbuff_ptrs, float*** dense_bwd_rs_sbuff_ptrs, float*** dense_bwd_rs_rbuff_ptrs, float*** grad_ptrs, float*** sum_grad_ptrs);
void free_buffers(float** fwd_halo_send_buff0_ptrs, float** fwd_halo_send_buff1_ptrs, float** fwd_halo_recv_buff0_ptrs, float** fwd_halo_recv_buff1_ptrs, float** bwd_halo_send_buff0_ptrs, float** bwd_halo_send_buff1_ptrs, float** bwd_halo_recv_buff0_ptrs, float** bwd_halo_recv_buff1_ptrs, float** dense_fwd_allgather_sbuff_ptrs, float** dense_fwd_allgather_rbuff_ptrs, float** dense_bwd_rs_sbuff_ptrs, float** dense_bwd_rs_rbuff_ptrs, float** grad_ptrs, float** sum_grad_ptrs);
void warmup(int num_warmup, float** fwd_halo_send_buff0_ptrs, float** fwd_halo_send_buff1_ptrs, float** fwd_halo_recv_buff0_ptrs, float** fwd_halo_recv_buff1_ptrs, float** bwd_halo_send_buff0_ptrs, float** bwd_halo_send_buff1_ptrs, float** bwd_halo_recv_buff0_ptrs, float** bwd_halo_recv_buff1_ptrs, float** dense_fwd_allgather_sbuff_ptrs, float** dense_fwd_allgather_rbuff_ptrs, float** dense_bwd_rs_sbuff_ptrs, float** dense_bwd_rs_rbuff_ptrs, float** grad_ptrs, float** sum_grad_ptrs, MPI_Comm model_parallel_comm, MPI_Comm dense_allreduce_comm);
void run_measurement(int num_runs, float** fwd_halo_send_buff0_ptrs, float** fwd_halo_send_buff1_ptrs, float** fwd_halo_recv_buff0_ptrs, float** fwd_halo_recv_buff1_ptrs, float** bwd_halo_send_buff0_ptrs, float** bwd_halo_send_buff1_ptrs, float** bwd_halo_recv_buff0_ptrs, float** bwd_halo_recv_buff1_ptrs, float** dense_fwd_allgather_sbuff_ptrs, float** dense_fwd_allgather_rbuff_ptrs, float** dense_bwd_rs_sbuff_ptrs, float** dense_bwd_rs_rbuff_ptrs, float** grad_ptrs, float** sum_grad_ptrs, MPI_Comm model_parallel_comm, MPI_Comm dense_allreduce_comm);
void output_results(int rank, int world_size, int mp_group_size, int dense_allreduce_group_size, int total_params);
void cleanup_mpi(MPI_Comm dense_allreduce_comm, MPI_Comm model_parallel_comm);

int main(int argc, char *argv[]){
    int rank, world_size;
    int dense_allreduce_group_rank, mp_group_rank;
    int dense_allreduce_group_size, mp_group_size;

    MPI_Comm dense_allreduce_comm, model_parallel_comm;

    initialize_mpi(&rank, &world_size, &dense_allreduce_group_rank, &mp_group_rank, &dense_allreduce_group_size, &mp_group_size, &dense_allreduce_comm, &model_parallel_comm);

    float** fwd_halo_send_buff0_ptrs;
    float** fwd_halo_send_buff1_ptrs;
    float** fwd_halo_recv_buff0_ptrs;
    float** fwd_halo_recv_buff1_ptrs;
    float** bwd_halo_send_buff0_ptrs;
    float** bwd_halo_send_buff1_ptrs;
    float** bwd_halo_recv_buff0_ptrs;
    float** bwd_halo_recv_buff1_ptrs;
    float** dense_fwd_allgather_sbuff_ptrs;
    float** dense_fwd_allgather_rbuff_ptrs;
    float** dense_bwd_rs_sbuff_ptrs;
    float** dense_bwd_rs_rbuff_ptrs;
    float** grad_ptrs;
    float** sum_grad_ptrs;

    allocate_buffers(&fwd_halo_send_buff0_ptrs, &fwd_halo_send_buff1_ptrs, &fwd_halo_recv_buff0_ptrs, &fwd_halo_recv_buff1_ptrs, &bwd_halo_send_buff0_ptrs, &bwd_halo_send_buff1_ptrs, &bwd_halo_recv_buff0_ptrs, &bwd_halo_recv_buff1_ptrs, &dense_fwd_allgather_sbuff_ptrs, &dense_fwd_allgather_rbuff_ptrs, &dense_bwd_rs_sbuff_ptrs, &dense_bwd_rs_rbuff_ptrs, &grad_ptrs, &sum_grad_ptrs);

    MPI_Barrier(MPI_COMM_WORLD);

    warmup(WARM_UP, fwd_halo_send_buff0_ptrs, fwd_halo_send_buff1_ptrs, fwd_halo_recv_buff0_ptrs, fwd_halo_recv_buff1_ptrs, bwd_halo_send_buff0_ptrs, bwd_halo_send_buff1_ptrs, bwd_halo_recv_buff0_ptrs, bwd_halo_recv_buff1_ptrs, dense_fwd_allgather_sbuff_ptrs, dense_fwd_allgather_rbuff_ptrs, dense_bwd_rs_sbuff_ptrs, dense_bwd_rs_rbuff_ptrs, grad_ptrs, sum_grad_ptrs, model_parallel_comm, dense_allreduce_comm);

    run_measurement(RUNS, fwd_halo_send_buff0_ptrs, fwd_halo_send_buff1_ptrs, fwd_halo_recv_buff0_ptrs, fwd_halo_recv_buff1_ptrs, bwd_halo_send_buff0_ptrs, bwd_halo_send_buff1_ptrs, bwd_halo_recv_buff0_ptrs, bwd_halo_recv_buff1_ptrs, dense_fwd_allgather_sbuff_ptrs, dense_fwd_allgather_rbuff_ptrs, dense_bwd_rs_sbuff_ptrs, dense_bwd_rs_rbuff_ptrs, grad_ptrs, sum_grad_ptrs, model_parallel_comm, dense_allreduce_comm);

    output_results(rank, world_size, mp_group_size, dense_allreduce_group_size, total_params);

    free_buffers(fwd_halo_send_buff0_ptrs, fwd_halo_send_buff1_ptrs, fwd_halo_recv_buff0_ptrs, fwd_halo_recv_buff1_ptrs, bwd_halo_send_buff0_ptrs, bwd_halo_send_buff1_ptrs, bwd_halo_recv_buff0_ptrs, bwd_halo_recv_buff1_ptrs, dense_fwd_allgather_sbuff_ptrs, dense_fwd_allgather_rbuff_ptrs, dense_bwd_rs_sbuff_ptrs, dense_bwd_rs_rbuff_ptrs, grad_ptrs, sum_grad_ptrs);

    cleanup_mpi(dense_allreduce_comm, model_parallel_comm);

    return 0;
}