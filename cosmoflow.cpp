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

// 2x2 2D spatial decomposition for 3D tensors. Each worker has two neighbors in 2D decomposition!

// Conv layer halo exchange message sizes in forward
int conv_fwd_halo_sizes[NUM_CONV_LAYERS-1] = {2097152, 1048576, 524288, 262144};

// Conv layer halo exchange message sizes in backward
int conv_bwd_halo_sizes[NUM_CONV_LAYERS-1] = {131072, 262144, 524288, 1048576};

#define NUM_DENSE_LAYERS 3

// Dense layer allgather msg sizes in forward
int dense_fwd_allgather_sizes[NUM_DENSE_LAYERS] = {65536, 256, 128};

// Dense layer reduce_scatter msg sizes in backward
int dense_bwd_reduce_scatter_sizes[NUM_DENSE_LAYERS] = {128, 256, 65536};

// Allreduce sizes for gradients with message aggregation
// Aggregate all dense layers: Dense2-0 Conv4 Conv3 Conv2 Conv1 Conv0
int allreduce_sizes[NUM_LAYERS-2] = {1050737, 3539456, 884992, 221312, 55360, 3488};



int run_parallel_model(float** fwd_send_buff0_ptrs,
                        float** fwd_send_buff1_ptrs,
                        float** fwd_recv_buff0_ptrs,
                        float** fwd_recv_buff1_ptrs,
                        float** bwd_send_buff0_ptrs,
                        float** bwd_send_buff1_ptrs,
                        float** bwd_recv_buff0_ptrs,
                        float** bwd_recv_buff1_ptrs,
                        float** dense_fwd_allgather_sbuff_ptrs,
                        float** dense_fwd_allgather_rbuff_ptrs,
                        float** dense_bwd_rs_sbuff_ptrs,
                        float** dense_bwd_rs_rbuff_ptrs,
                        float** grad_ptrs,
                        float** sum_grad_ptrs,
                        MPI_Comm model_comm,
                        MPI_Comm dense_comm){

    // forward (fwd)
    int model_group_rank;
    MPI_Comm_rank(model_comm, &model_group_rank);
    for(int i=0; i<NUM_LAYERS; i++){
        if(i>=1 && i<NUM_CONV_LAYERS){ // Halo exchange for conv layers
            int msg_idx = i-1;
            MPI_Request requests[4];
            MPI_Isend(fwd_send_buff0_ptrs[msg_idx], conv_fwd_halo_sizes[msg_idx], MPI_FLOAT, model_group_rank^1, i, model_comm, &requests[0]);
            MPI_Isend(fwd_send_buff1_ptrs[msg_idx], conv_fwd_halo_sizes[msg_idx], MPI_FLOAT, model_group_rank^2, i, model_comm, &requests[1]);
            MPI_Irecv(fwd_recv_buff0_ptrs[msg_idx], conv_fwd_halo_sizes[msg_idx], MPI_FLOAT, model_group_rank^1, i, model_comm, &requests[2]);
            MPI_Irecv(fwd_recv_buff1_ptrs[msg_idx], conv_fwd_halo_sizes[msg_idx], MPI_FLOAT, model_group_rank^2, i, model_comm, &requests[3]);
            MPI_Waitall(4, requests, MPI_STATUSES_IGNORE);
        }
        else if(i>=NUM_CONV_LAYERS){ // All gather for dense layers
            int msg_idx = i-NUM_CONV_LAYERS;
            MPI_Allgather(dense_fwd_allgather_sbuff_ptrs[msg_idx], dense_fwd_allgather_sizes[msg_idx], MPI_FLOAT, dense_fwd_allgather_rbuff_ptrs[msg_idx], dense_fwd_allgather_sizes[msg_idx], MPI_FLOAT, model_comm);
        }

        usleep(fwd_rt_per_layer[i]); // Compute
    }

    // backward (bwd)
    MPI_Request grad_allreduce_reqs[NUM_CONV_LAYERS+1];
    for(int i=0; i<NUM_CONV_LAYERS+1; i++)
        grad_allreduce_reqs[i] = MPI_REQUEST_NULL;

    int index, flag;
    for(int i=0; i<NUM_LAYERS; i++){
        if(i > NUM_DENSE_L)
            MPI_Testany(NUM_CONV_LAYERS+1, grad_allreduce_reqs, &index, &flag, MPI_STATUSES_IGNORE); // Advance MPI in the background

        usleep(bwd_rt_per_layer[i]); // Compute

        if(i < NUM_DENSE_L){ // Dense layers
            MPI_Reduce_scatter_block(dense_bwd_rs_sbuff_ptrs[i], dense_bwd_rs_rbuff_ptrs[i], dense_bwd_reduce_scatter_sizes[i], MPI_FLOAT, MPI_SUM, model_comm);
        }
        else if(i < NUM_LAYERS-1){ // Conv layers
            int msg_idx = i-NUM_DENSE_L;
            MPI_Request requests[4];
            MPI_Isend(bwd_send_buff0_ptrs[msg_idx], conv_bwd_halo_sizes[msg_idx], MPI_FLOAT, model_group_rank^1, i, model_comm, &requests[0]);
            MPI_Isend(bwd_send_buff1_ptrs[msg_idx], conv_bwd_halo_sizes[msg_idx], MPI_FLOAT, model_group_rank^2, i, model_comm, &requests[1]);
            MPI_Irecv(bwd_recv_buff0_ptrs[msg_idx], conv_bwd_halo_sizes[msg_idx], MPI_FLOAT, model_group_rank^1, i, model_comm, &requests[2]);
            MPI_Irecv(bwd_recv_buff1_ptrs[msg_idx], conv_bwd_halo_sizes[msg_idx], MPI_FLOAT, model_group_rank^2, i, model_comm, &requests[3]);
            MPI_Waitall(4, requests, MPI_STATUSES_IGNORE);
        }

        if(i == NUM_DENSE_L-1){
            MPI_Iallreduce(grad_ptrs[0], sum_grad_ptrs[0], allreduce_sizes[0], MPI_FLOAT, MPI_SUM, dense_comm, &grad_allreduce_reqs[0]);
        }
        else if(i > NUM_DENSE_L-1){
            MPI_Iallreduce(grad_ptrs[i-NUM_DENSE_L+1], sum_grad_ptrs[i-NUM_DENSE_L+1], allreduce_sizes[i-NUM_DENSE_L+1], MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD, &grad_allreduce_reqs[i-NUM_DENSE_L+1]);
        }
    }

    MPI_Waitall(NUM_CONV_LAYERS+1, grad_allreduce_reqs, MPI_STATUSES_IGNORE);
    return 0;
}

int main(int argc, char *argv[]){
    int rank, world_size;
 
 int model_shards = 4; // Do not change this

    MPI_Init(&argc,&argv);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    int dense_comm_rank, model_group_rank;
    int dense_comm_size, model_group_size;

    // The number of processes should be a multiple of model_shards = 4
    assert(world_size % model_shards == 0);
    int dense_comm_color = rank % model_shards;

    MPI_Comm dense_comm;
    MPI_Comm_split(MPI_COMM_WORLD, dense_comm_color, rank, &dense_comm);

    MPI_Comm_rank(dense_comm, &dense_comm_rank);
    MPI_Comm_size(dense_comm, &dense_comm_size);

    MPI_Comm model_comm;
    MPI_Comm_split(MPI_COMM_WORLD, dense_comm_rank, rank, &model_comm);
    MPI_Comm_rank(model_comm, &model_group_rank);
    MPI_Comm_size(model_comm, &model_group_size);

    assert(dense_comm_color == model_group_rank);
    assert(model_shards == model_group_size);

    float* fwd_send_buff0_ptrs[NUM_CONV_LAYERS-1];
    float* fwd_send_buff1_ptrs[NUM_CONV_LAYERS-1];
    float* fwd_recv_buff0_ptrs[NUM_CONV_LAYERS-1];
    float* fwd_recv_buff1_ptrs[NUM_CONV_LAYERS-1];

    float* bwd_send_buff0_ptrs[NUM_CONV_LAYERS-1];
    float* bwd_send_buff1_ptrs[NUM_CONV_LAYERS-1];
    float* bwd_recv_buff0_ptrs[NUM_CONV_LAYERS-1];
    float* bwd_recv_buff1_ptrs[NUM_CONV_LAYERS-1];
    for(int i=0; i<NUM_CONV_LAYERS-1; i++){
        fwd_send_buff0_ptrs[i] = (float *)calloc(conv_fwd_halo_sizes[i], sizeof(float));
        fwd_send_buff1_ptrs[i] = (float *)calloc(conv_fwd_halo_sizes[i], sizeof(float));
        fwd_recv_buff0_ptrs[i] = (float *)calloc(conv_fwd_halo_sizes[i], sizeof(float));
        fwd_recv_buff1_ptrs[i] = (float *)calloc(conv_fwd_halo_sizes[i], sizeof(float));

        bwd_send_buff0_ptrs[i] = (float *)calloc(conv_bwd_halo_sizes[i], sizeof(float));
        bwd_send_buff1_ptrs[i] = (float *)calloc(conv_bwd_halo_sizes[i], sizeof(float));
        bwd_recv_buff0_ptrs[i] = (float *)calloc(conv_bwd_halo_sizes[i], sizeof(float));
        bwd_recv_buff1_ptrs[i] = (float *)calloc(conv_bwd_halo_sizes[i], sizeof(float));
    }

    float* dense_fwd_allgather_sbuff_ptrs[NUM_DENSE_LAYERS];
    float* dense_fwd_allgather_rbuff_ptrs[NUM_DENSE_LAYERS];
    float* dense_bwd_rs_sbuff_ptrs[NUM_DENSE_LAYERS];
    float* dense_bwd_rs_rbuff_ptrs[NUM_DENSE_LAYERS];
    for(int i=0; i<NUM_DENSE_LAYERS; i++){
        dense_fwd_allgather_sbuff_ptrs[i] = (float *)calloc(dense_fwd_allgather_sizes[i], sizeof(float));
        dense_fwd_allgather_rbuff_ptrs[i] = (float *)calloc(dense_fwd_allgather_sizes[i]*model_shards, sizeof(float));
        dense_bwd_rs_sbuff_ptrs[i] = (float *)calloc(dense_bwd_reduce_scatter_sizes[i]*model_shards, sizeof(float));
        dense_bwd_rs_rbuff_ptrs[i] = (float *)calloc(dense_bwd_reduce_scatter_sizes[i], sizeof(float));
    }

    float* grad_ptrs[NUM_LAYERS-2];
    float* sum_grad_ptrs[NUM_LAYERS-2];
    for(int i=0; i<NUM_LAYERS-2; i++){
        grad_ptrs[i] = (float *)calloc(allreduce_sizes[i], sizeof(float));
        sum_grad_ptrs[i] = (float *)calloc(allreduce_sizes[i], sizeof(float));
    }

    MPI_Barrier(MPI_COMM_WORLD);

   
   // Warm-up
    for(int wmp = 0; wmp < WARM_UP; wmp++){
        run_parallel_model(fwd_send_buff0_ptrs,
                           fwd_send_buff1_ptrs,
                           fwd_recv_buff0_ptrs,
                           fwd_recv_buff1_ptrs,
                           bwd_send_buff0_ptrs,
                           bwd_send_buff1_ptrs,
                           bwd_recv_buff0_ptrs,
                           bwd_recv_buff1_ptrs,
                           dense_fwd_allgather_sbuff_ptrs,
                           dense_fwd_allgather_rbuff_ptrs,
                           dense_bwd_rs_sbuff_ptrs,
                           dense_bwd_rs_rbuff_ptrs,
                           grad_ptrs,
                           sum_grad_ptrs,
                           model_comm,
                           dense_comm);
    }

    double begin, elapse;
    begin = MPI_Wtime();
    for(int iter = 0; iter < RUNS; iter++){
        run_parallel_model(fwd_send_buff0_ptrs,
                           fwd_send_buff1_ptrs,
                           fwd_recv_buff0_ptrs,
                           fwd_recv_buff1_ptrs,
                           bwd_send_buff0_ptrs,
                           bwd_send_buff1_ptrs,
                           bwd_recv_buff0_ptrs,
                           bwd_recv_buff1_ptrs,
                           dense_fwd_allgather_sbuff_ptrs,
                           dense_fwd_allgather_rbuff_ptrs,
                           dense_bwd_rs_sbuff_ptrs,
                           dense_bwd_rs_rbuff_ptrs,
                           grad_ptrs,
                           sum_grad_ptrs,
                           model_comm,
                           dense_comm);
    }
    elapse = (MPI_Wtime()-begin)/RUNS;

    int total_params;
    for(int i=0; i<NUM_LAYERS-2; i++){
        if(i == 0)
            total_params = allreduce_sizes[i] * model_shards;
        else
            total_params += allreduce_sizes[i];
    }

    if(rank == 0){
        printf("Rank = %d, world_size = %d, model_shards = %d, data_shards = %d, total_params = %d, global_batch_size = %d. \n", rank, world_size, model_group_size, dense_comm_size, total_params, 8*dense_comm_size);
        printf("CosmoFlow model • a hybrid of model x data parallelism runtime, per iteration = %f s.\n", elapse);
    }

    MPI_Finalize();
}