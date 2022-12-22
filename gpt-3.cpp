#include <mpi.h>
#include <unistd.h>
#include <stdio.h>
#include <string>
#include <time.h>
#include <stdlib.h>
#include <assert.h>

#define MODEL_PARALLEL_SIZE 96
#define DATA_PARALLEL_SIZE 4
#define P2P_BUFFER_SIZE 25165824
#define FORWARD_COMPUTE_TIME 15915
#define BACKWARD_COMPUTE_TIME 31830

#define RUNS 128
#define WARM_UP 8
#define NUM_L 96
#define ACC_STEP_SCALE 2
#define MODEL_SHARDS 4

// Function declarations
void run_forward_pass(int steps_for_accumulation, int stage_index, int total_pipeline_stages,
                       float *send_buffer_fwd, float *recv_buffer_fwd,
                       float **buffers_fwd_mp, float **buffers_fwd_mp_reduced,
                       MPI_Comm comm_pp, MPI_Comm comm_mp);

void run_backward_pass(int steps_for_accumulation, int stage_index, int total_pipeline_stages,
                        float *send_buffer_bwd, float *recv_buffer_bwd,
                        float **buffers_bwd_mp, float **buffers_bwd_mp_reduced,
                        MPI_Comm comm_pp);

void aggregate_gradients(float *grad_buffer, float *aggregated_grad_buffer,
                        MPI_Comm comm_dp);

int main() {
    // Define message sizes and runtime constants
    #define MP_ALLREDUCE_SIZE   25165824
    #define MOE_ALL2ALL_SIZE    25165824
    #define DP_ALLREDUCE_SIZE   452984832
    #define FWD_RT              15915
    #define BWD_RT              31830
    #define BWD_RT_GPIPE        47745

    // Define MPI communicators
    MPI_Comm comm_dp, comm_mp, comm_pp;
    // Initialize MPI communicators

    // Allocate buffers and arrays
    float grad_buffer[DP_ALLREDUCE_SIZE];
    float aggregated_grad_buffer[DATA_PARALLEL_SIZE];
    float send_buffer_fwd[P2P_BUFFER_SIZE], recv_buffer_fwd[P2P_BUFFER_SIZE];
    float send_buffer_bwd[P2P_BUFFER_SIZE], recv_buffer_bwd[P2P_BUFFER_SIZE];
    float *buffers_fwd_mp[2], *buffers_fwd_mp_reduced[2];
    float *buffers_bwd_mp[2], *buffers_bwd_mp_reduced[2];

    for (int i = 0; i < 2; i++) {
        buffers_fwd_mp[i] = new float[MODEL_PARALLEL_SIZE];
        buffers_fwd_mp_reduced[i] = new float[MODEL_PARALLEL_SIZE];
        buffers_bwd_mp[i] = new float[MODEL_PARALLEL_SIZE];
        buffers_bwd_mp_reduced[i] = new float[MODEL_PARALLEL_SIZE];
    }

    // Run the pipeline stage
    int steps_for_accumulation = 10;
    int stage_index = 2;
    int total_pipeline_stages = 4;

    run_forward_pass(steps_for_accumulation, stage_index, total_pipeline_stages,
                     send_buffer_fwd, recv_buffer_fwd,
                     buffers_fwd_mp, buffers_fwd_mp_reduced,
                     comm_pp, comm_mp);

    run_backward_pass(steps_for_accumulation, stage_index, total_pipeline_stages,
                      send_buffer_bwd, recv_buffer_bwd,
                      buffers_bwd_mp, buffers_bwd_mp_reduced,
                      comm_pp);

    aggregate_gradients(grad_buffer, aggregated_grad_buffer, comm_dp);

    // Deallocate buffers
    for (int i = 0; i < 2; i++) {
        delete[] buffers_fwd_mp[i];
        delete[] buffers_fwd_mp_reduced[i];
        delete[] buffers_bwd_mp[i];
        delete[] buffers_bwd_mp_reduced[i];
    }

    return 0;
}

void run_forward_pass(int steps_for_accumulation, int stage_index, int total_pipeline_stages,
                      float *send_buffer_fwd, float *recv_buffer_fwd,
                      float **buffers_fwd_mp, float **buffers_fwd_mp_reduced,
                      MPI_Comm comm_pp, MPI_Comm comm_mp) {

    MPI_Request reqs_fwd[2];

    for (int i = 0; i < 2; i++) {
        reqs_fwd[i] = MPI_REQUEST_NULL;
    }

    for (int step = 0; step < steps_for_accumulation; step++) {
        if (stage_index == 0) {
            MPI_Wait(&reqs_fwd[0], MPI_STATUS_IGNORE);
            usleep(FORWARD_COMPUTE_TIME); // Emulate computation time
            MPI_Isend(send_buffer_fwd, P2P_BUFFER_SIZE, MPI_FLOAT, stage_index + 1, step, comm_pp, &reqs_fwd[0]);
        } else if (stage_index == total_pipeline_stages - 1) {
            MPI_Irecv(recv_buffer_fwd, P2P_BUFFER_SIZE, MPI_FLOAT, stage_index - 1, step, comm_pp, &reqs_fwd[1]);
            MPI_Wait(&reqs_fwd[1], MPI_STATUS_IGNORE);
            usleep(FORWARD_COMPUTE_TIME); // Emulate computation time
        } else {
            MPI_Irecv(recv_buffer_fwd, P2P_BUFFER_SIZE, MPI_FLOAT, stage_index - 1, step, comm_pp, &reqs_fwd[1]);
            MPI_Wait(&reqs_fwd[1], MPI_STATUS_IGNORE);
            usleep(FORWARD_COMPUTE_TIME); // Emulate computation time
            MPI_Isend(send_buffer_fwd, P2P_BUFFER_SIZE, MPI_FLOAT, stage_index + 1, step, comm_pp, &reqs_fwd[0]);
        }

        for (int j = 0; j < 2; j++) {
            MPI_Allreduce(buffers_fwd_mp[j], buffers_fwd_mp_reduced[j], MODEL_PARALLEL_SIZE, MPI_FLOAT, MPI_SUM, comm_mp);
        }
    }
}