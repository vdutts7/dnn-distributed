// C++/MPI proxy • GPT2-large model 
// Distributed training (hybrid pipeline x data parallelism)


#include <mpi.h>
#include <unistd.h>
#include <stdio.h>
#include <string>
#include <time.h>
#include <stdlib.h>
#include <assert.h>

#define RUN_COUNT 256
#define WARM_UP_ITERATIONS 10

//p2p msg size for GPT-2 with micro-batch size=1 and seq_length=632
#define P2P_MESSAGE_SIZE 808960

#define BEGINNING_SIZE 85317120
#define INTERMEDIATE_SIZE 19677440
#define ENDING_SIZE 84008960

#define MESSAGE_AGGREGATION 1

#ifdef MESSAGE_AGGREGATION
//message aggregation
#define BEGINNING_NUM 1
#define INTERMEDIATE_NUM 1
#define ENDING_NUM 1
int first_layer_grad_sizes[BEGINNING_NUM] = {BEGINNING_SIZE};
int intermediate_layer_grad_sizes[INTERMEDIATE_NUM] = {INTERMEDIATE_SIZE};
int end_layer_grad_sizes[ENDING_NUM] = {ENDING_SIZE};

#else
#define BEGINNING_NUM 14
#define INTERMEDIATE_NUM 12
#define ENDING_NUM 15
//sizes for the gradients per layer of gpt-2
int first_layer_grad_sizes[BEGINNING_NUM] = {64328960, 1310720, 1280, 4915200, 1638400, 1280, 6553600, 6553600, 1280, 3840, 1280, 1280, 5120, 1280};
int intermediate_layer_grad_sizes[INTERMEDIATE_NUM] = {1280, 4915200, 1638400, 1280, 6553600, 6553600, 1280, 3840, 1280, 1280, 5120, 1280};
int end_layer_grad_sizes[ENDING_NUM] = {1280, 4915200, 1638400, 1280, 6553600, 6553600, 1280, 64328960, 1280, 3840, 1280, 1280, 5120, 1280, 1280};

#endif

int run_gpt2_training(int grad_accumulation_steps, int stage_number, int num_grad_per_stage, 
		 int total_stages, int allreduce_group_size, 
		 float **start_stage_grad_ptrs,
		 float **sum_start_stage_grad_ptrs,
		 float **finish_stage_grad_ptrs,
		 float **sum_finish_stage_grad_ptrs,
		 float **intermediate_stage_grad_ptrs,
		 float **sum_intermediate_stage_grad_ptrs,
		 int *stage_grad_sizes,
		 MPI_Comm p2p_comm, MPI_Comm allreduce_comm){

    float *send_buffer = (float *)calloc(P2P_MESSAGE_SIZE, sizeof(float));
    float *recv_buffer = (float *)calloc(P2P_MESSAGE_SIZE, sizeof(float));

    //p2p forward
    for(int i=0; i<grad_accumulation_steps; i++){
        if(stage_number == 0){
            MPI_Request request;
            MPI_Isend(send_buffer, P2P_MESSAGE_SIZE, MPI_FLOAT, stage_number+1, i, p2p_comm, &request);
            MPI_Wait(&request, MPI_STATUS_IGNORE);
        }
        else if(stage_number == total_stages-1){
            MPI_Request request;
            MPI_Irecv(recv_buffer, P2P_MESSAGE_SIZE, MPI_FLOAT, stage_number-1, i, p2p_comm, &request);
            MPI_Wait(&request, MPI_STATUS_IGNORE);
        }
        else{
            MPI_Request requests[2];
            MPI_Isend(send_buffer, P2P_MESSAGE_SIZE, MPI_FLOAT, stage_number+1, i, p2p_comm, &requests[0]);
            MPI_Irecv(recv_buffer, P2P_MESSAGE_SIZE, MPI_FLOAT, stage_number-1, i, p2p_comm, &requests[1]);
            MPI_Waitall(2, requests, MPI_STATUSES_IGNORE);
        }
    }

    //p2p backward
    for(int i=0; i<grad_accumulation_steps; i++){
        if(stage_number == 0){
            MPI_Request request;
            MPI_Irecv(recv_buffer, P2P_MESSAGE_SIZE, MPI_FLOAT, stage_number+1, i, p2p_comm, &request);
            MPI_Wait(&request, MPI_STATUS_IGNORE);
        }
        else if(stage_number == total_stages-1){
            MPI_Request request;
            MPI_Isend(send_buffer, P2P_MESSAGE_SIZE, MPI_FLOAT, stage_number-1, i, p2p_comm, &request);
            MPI_Wait(&request, MPI_STATUS_IGNORE);
        }
        else{
            MPI_Request requests[2];
            MPI_Isend(send_buffer, P2P_MESSAGE_SIZE, MPI_FLOAT, stage_number-1, i, p2p_comm, &requests[0]);
            MPI_Irecv(recv_buffer, P2P_MESSAGE_SIZE, MPI_FLOAT, stage_number+1, i, p2p_comm, &requests[1]);
            MPI_Waitall(2, requests, MPI_STATUSES_IGNORE);
        }
    }

    //allreduce on gradients
    if(allreduce_group_size > 1){
        if(stage_number == 0){
            for(int i=0; i<num_grad_per_stage; i++){
                MPI_Allreduce(start_stage_grad_ptrs[i], sum_start_stage_grad_ptrs[i], stage_grad_sizes[i], MPI_FLOAT, MPI_SUM, allreduce_comm);
            }
        }
        else if(stage_number == total_stages-1){
            for(int i=0; i<num_grad_per_stage; i++){
                MPI_Allreduce(finish_stage_grad_ptrs[i], sum_finish_stage_grad_ptrs[i], stage_grad_sizes[i], MPI_FLOAT, MPI_SUM, allreduce_comm);
            }
        }
        else{
            for(int i=0; i<num_grad_per_stage; i++){
                MPI_Allreduce(intermediate_stage_grad_ptrs[i], sum_intermediate_stage_grad_ptrs[i], stage_grad_sizes[i], MPI_FLOAT, MPI_SUM, allreduce_comm);
            }
        }
    }
    return 0;
}

int main(int argc, char *argv[]){
    int rank, world_size;
    double start_time, elapsed_time;

    //number of basic Transformer layers
    int num_layers = 48;
    //number of pipeline stages
    int num_stages = 4;
    //number of micro-batches in an iteration
    int grad_accumulation_steps = 8;

    if(argc == 3){
        num_layers = atoi(argv[1]);
        num_stages = atoi(argv[2]);
    }
    if(argc == 4){
        num_layers = atoi(argv[1]);
        num_stages = atoi(argv[2]);
        grad_accumulation_steps = atoi(argv[3]);
    }

    MPI_Init(&argc,&argv);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm allreduce_comm;
    MPI_Comm p2p_comm;
    int allreduce_group_rank, p2p_group_rank;
    int allreduce_group_size, p2p_group_size;

    //the number of processes should be a multiple of num_stages
    assert(world_size % num_stages == 0);
    int allreduce_group_color = rank % num_stages;

    MPI_Comm_split(MPI_COMM_WORLD, allreduce_group_color, rank, &allreduce_comm);
    MPI_Comm_rank(allreduce_comm, &allreduce_group_rank);
    MPI_Comm_size(allreduce_comm, &allreduce_group_size);

    MPI_Comm_split(MPI_COMM_WORLD, allreduce_group_rank, rank, &p2p_comm);
    MPI_Comm_rank(p2p_comm, &p2p_group_rank);
    MPI_Comm_size(p2p_comm, &p2p_group_size);

    int stage_number = allreduce_group_color;
    assert(allreduce_group_color == p2p_group_rank);


    int num_layers_per_stage = (int)(num_layers/num_stages);

    int start_stage_grad_count = BEGINNING_NUM + (num_layers_per_stage - 1)*INTERMEDIATE_NUM;
    int finish_stage_grad_count = ENDING_NUM + (num_layers_per_stage - 1)*INTERMEDIATE_NUM;
    int intermediate_stage_grad_count = num_layers_per_stage * INTERMEDIATE_NUM;

    //pointers of the send/receive buffers
    float* start_stage_grad_ptrs[start_stage_grad_count];
    float* sum_start_stage_grad_ptrs[start_stage_grad_count];

    float* intermediate_stage_grad_ptrs[intermediate_stage_grad_count];
    float* sum_intermediate_stage_grad_ptrs[intermediate_stage_grad_count];

    float* finish_stage_grad_ptrs[finish_stage_grad_count];
    float* sum_finish_stage_grad_ptrs[finish_stage_grad_count];

    int num_grad_per_stage = -1;
    if(stage_number == 0){
        num_grad_per_stage = BEGINNING_NUM + (num_layers_per_stage-1) * INTERMEDIATE_NUM;
    }
    else if(stage_number == num_stages-1){
        num_grad_per_stage = ENDING_NUM + (num_layers_per_stage-1) * INTERMEDIATE_NUM;
    }
    else{
        num_grad_per_stage = num_layers_per_stage * INTERMEDIATE_NUM;
    }

    int stage_grad_sizes[num_grad_per_stage]; 

    if(stage_number == 0){
        for(int i=0; i<BEGINNING_NUM; i++){
            start_stage_grad_ptrs[i] = (float *)calloc(first_layer_grad_sizes[i], sizeof(float));
            sum_start_stage_grad_ptrs[i] = (float *)calloc(first_layer_grad_sizes[i], sizeof(float));
            stage_grad_sizes[i] = first_layer_grad_sizes[i];
        }
        for(int j=0; j<num_layers_per_stage-1; j++){
            for(int k=0; k<INTERMEDIATE_NUM; k++){
                start_stage_grad_ptrs[INTERMEDIATE_NUM*j+k+BEGINNING_NUM] = (float *)calloc(intermediate_layer_grad_sizes[k], sizeof(float));
                sum_start_stage_grad_ptrs[INTERMEDIATE_NUM*j+k+BEGINNING_NUM] = (float *)calloc(intermediate_layer_grad_sizes[k], sizeof(float));
                stage_grad_sizes[INTERMEDIATE_NUM*j+k+BEGINNING_NUM] = intermediate_layer_grad_sizes[k];                
            }
        }
    }
    else if(stage_number == num_stages-1){
        for(int j=0; j<num_layers_per_stage-1; j++){
            for(int k=0; k<INTERMEDIATE_NUM; k++){
                finish_stage_grad_ptrs[INTERMEDIATE_NUM*j+k] = (float *)calloc(intermediate_layer_grad_sizes[k], sizeof(float));
                sum_finish_stage_grad_ptrs[INTERMEDIATE_NUM*j+k] = (float *)calloc(intermediate_layer_grad_sizes[k], sizeof(float));
                stage_grad_sizes[INTERMEDIATE_NUM*j+k] = intermediate_layer_grad_sizes[k];                
            }
        }
        for(int i=0; i<ENDING_NUM; i++){
            finish_stage_grad_ptrs[INTERMEDIATE_NUM*(num_layers_per_stage-1)+i] = (float *)calloc(end_layer_grad_sizes[i], sizeof(float));
            sum_finish_stage_grad_ptrs[INTERMEDIATE_NUM*(num_layers_per_stage-1)+i] = (float *)calloc(end_layer_grad_sizes[i], sizeof(float));
            stage_grad_sizes[INTERMEDIATE_NUM*(num_layers_per_stage-1)+i] = end_layer_grad_sizes[i]; 
        }
    }
    else{
        for(int j=0; j<num_layers_per_stage; j++){
            for(int k=0; k<INTERMEDIATE_NUM; k++){
                intermediate_stage_grad_ptrs[INTERMEDIATE_NUM*j+k] = (float *)calloc(intermediate_layer_grad_sizes[k], sizeof(float));
                sum_intermediate_stage_grad_ptrs[INTERMEDIATE_NUM*j+k] = (float *)calloc(intermediate_layer_grad_sizes[k], sizeof(float));
                stage_grad_sizes[INTERMEDIATE_NUM*j+k] = intermediate_layer_grad_sizes[k];                
            }
        }
    }
    MPI_Barrier(MPI_COMM_WORLD);

    //warmup
    for(int wmp = 0; wmp < WARM_UP_ITERATIONS; wmp++){
        run_gpt2_training(grad_accumulation_steps, stage_number, num_grad_per_stage,
		     num_stages, allreduce_group_size, 
            	     start_stage_grad_ptrs,
            	     sum_start_stage_grad_ptrs,
            	     finish_stage_grad_ptrs,
            	     sum_finish_stage_grad_ptrs,
            	     intermediate_stage_grad_ptrs,
            	     sum_intermediate_stage_grad_ptrs,
            	     stage_grad_sizes,
            	     p2p_comm, allreduce_comm);
    }

    start_time = MPI_Wtime();
    for(int iter = 0; iter < RUN_COUNT; iter++){
        run_gpt2_training(grad_accumulation_steps, stage_number, num_grad_per_stage,
		     num_stages, allreduce_group_size, 
            	     start_stage_grad_ptrs,
            	     sum_start_stage_grad_ptrs,
            	     finish_stage_grad_ptrs,
            	     sum_finish_stage_grad_ptrs,
            	     intermediate_stage_grad_ptrs,
            	     sum_intermediate_stage_grad_ptrs,
            	     stage_grad_sizes,
            	     p2p_comm, allreduce_comm);
    }
    elapsed_time = (MPI_Wtime()-start_time)/RUN_COUNT;

    printf("Rank = %d, world_size = %d, layers = %d, stages = %d, total_params = %d, GPT2-large pipeline and data parallelism runtime for each iteration = %f s\n", rank, world_size, num_layers, num_stages, BEGINNING_SIZE+ENDING_SIZE+INTERMEDIATE_SIZE*(num_layers-2), elapsed_time);

    MPI_Finalize();
}