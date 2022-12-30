
/* 
 * C++/MPI proxy • CosmoFlow model 
 * Distributed training (hybrid of model x data parallelism)
 *-----------------------------------------------------------
    *  python -m torch.distributed.launch --nproc_per_node=4 dlrm_s_pytorch.py 
    *  --arch-embedding-size="80000-80000-80000-80000" --arch-sparse-feature-size=128 
    *  --arch-mlp-bot="128-128-128-128" --arch-mlp-top="512-512-512-256-1" 
    *  --max-ind-range=40000000 --data-generation=random --loss-function=bce 
    *  --round-targets=True --learning-rate=1.0 --mini-batch-size=2048 --print-freq=2 
    *  --print-time --test-freq=16 --test-mini-batch-size=1024 
    *  --memory-map --use-gpu --num-batches=32 --dist-backend=nccl  
 */

#include <mpi.h>
#include <unistd.h>
#include <stdio.h>
#include <string>
#include <time.h>
#include <stdlib.h>
#include <assert.h>

#define NUM_RUNS 1
#define WARMUP_ITERATIONS 0

#define MLP_BOTTOM_SIZE 49536
#define MLP_TOP_SIZE 728065 
#define ALL2ALL_EMB_SIZE 262144

#define FORWARD_BOTTOM_MLP 341
#define FORWARD_TOP_MLP 455
#define FORWARD_INTER 209
#define FORWARD_EMB 95

void run_custom_dlrm(int num_procs,
                    float *top_gradient,
                    float *sum_top_gradient,
                    float *bottom_gradient,
                    float *sum_bottom_gradient,
                    float *fwd_alltoall_send,
                    float *fwd_alltoall_recv,
                    float *bwd_alltoall_send,
                    float *bwd_alltoall_recv) {

    MPI_Request gradient_allreduce_requests[2];
    usleep(FORWARD_EMB); // Forward pass
    MPI_Alltoall(fwd_alltoall_send, ALL2ALL_EMB_SIZE/num_procs, MPI_FLOAT, fwd_alltoall_recv, ALL2ALL_EMB_SIZE/num_procs, MPI_FLOAT, MPI_COMM_WORLD);

    usleep(FORWARD_BOTTOM_MLP); // Forward pass
    usleep(FORWARD_INTER); // Forward pass

    usleep(FORWARD_TOP_MLP); // Forward pass

    usleep(FORWARD_TOP_MLP * 2); // Backward pass
    MPI_Iallreduce(top_gradient, sum_top_gradient, MLP_TOP_SIZE, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD, &gradient_allreduce_requests[0]);

    usleep(FORWARD_INTER); // Backward pass
    usleep(FORWARD_BOTTOM_MLP * 2); // Backward pass
    MPI_Iallreduce(bottom_gradient, sum_bottom_gradient, MLP_BOTTOM_SIZE, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD, &gradient_allreduce_requests[1]);

    MPI_Alltoall(bwd_alltoall_send, ALL2ALL_EMB_SIZE/num_procs, MPI_FLOAT, bwd_alltoall_recv, ALL2ALL_EMB_SIZE/num_procs, MPI_FLOAT, MPI_COMM_WORLD);
    usleep(FORWARD_EMB * 2); // Backward pass

    MPI_Waitall(2, gradient_allreduce_requests, MPI_STATUSES_IGNORE);
}

int main(int argc, char *argv[]) {
    int process_rank, total_processes;
    double start_time, elapsed_time;

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &total_processes);
    MPI_Comm_rank(MPI_COMM_WORLD, &process_rank);

    float *top_gradient = (float *)calloc(MLP_TOP_SIZE, sizeof(float));
    float *sum_top_gradient = (float *)calloc(MLP_TOP_SIZE, sizeof(float));
    float *bottom_gradient = (float *)calloc(MLP_BOTTOM_SIZE, sizeof(float));
    float *sum_bottom_gradient = (float *)calloc(MLP_BOTTOM_SIZE, sizeof(float));
     
    float *fwd_alltoall_send = (float *)calloc(ALL2ALL_EMB_SIZE, sizeof(float));
    float *fwd_alltoall_recv = (float *)calloc(ALL2ALL_EMB_SIZE, sizeof(float));
    float *bwd_alltoall_send = (float *)calloc(ALL2ALL_EMB_SIZE, sizeof(float));
    float *bwd_alltoall_recv = (float *)calloc(ALL2ALL_EMB_SIZE, sizeof(float));

    MPI_Barrier(MPI_COMM_WORLD);

    // Warm-up
    for(int warmup_iter = 0; warmup_iter < WARMUP_ITERATIONS; warmup_iter++) {
        run_custom_dlrm(total_processes,
                        top_gradient,
                        sum_top_gradient,
                        bottom_gradient,
                        sum_bottom_gradient,
                        fwd_alltoall_send,
                        fwd_alltoall_recv,
                        bwd_alltoall_send,
                        bwd_alltoall_recv);
    }

    start_time = MPI_Wtime();
    for(int iteration = 0; iteration < NUM_RUNS; iteration++) {
        run_custom_dlrm(total_processes,
                        top_gradient,
                        sum_top_gradient,
                        bottom_gradient,
                        sum_bottom_gradient,
                        fwd_alltoall_send,
                        fwd_alltoall_recv,
                        bwd_alltoall_send,
                        bwd_alltoall_recv);
    }
    elapsed_time = (MPI_Wtime() - start_time) / NUM_RUNS;

    if (process_rank == 0)
        printf("Performance Metrics: Rank = %d, Total Processes = %d, Global Batch Size = %d, DLRM Runtime per Iteration = %f seconds\n", process_rank, total_processes, 2048, elapsed_time);

    MPI_Finalize();
}