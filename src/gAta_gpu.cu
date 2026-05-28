/*
 * gAta_gpu.cu
 *
 * GPU version of gata_algorithm (uniform alltoall).
 *
 * Assumptions:
 *   - sendbuf and recvbuf are CUDA device pointers.
 *   - The MPI implementation is CUDA-aware (device pointers can be passed
 *     directly to MPI_Isend / MPI_Irecv / MPI_Sendrecv).
 */

#include <cuda_runtime.h>
#include <mpi.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>

#include "gAta.h"

//Default macro used to catch CUDA GPU failures
//Anytime we run any CUDA API calls like cudaMemcpy, etc.
//Run it inside CUDA_CHECK(cudaMemcpy), to print out error
#define CUDA_CHECK(stmt) do {                                          \
    cudaError_t err = (stmt);                                          \
    if (err != cudaSuccess) {                                          \
        std::cerr << "CUDA error " << cudaGetErrorString(err)          \
                  << " at " << __FILE__ << ":" << __LINE__ << '\n';   \
        return 1;                                                      \
    }                                                                  \
} while (0)

//====================== MAIN FUNCTION ===============================//
int gata_gpu_algorithm(int r, int b,
                       char *sendbuf, int sendcount, MPI_Datatype sendtype,
                       char *recvbuf, int recvcount, MPI_Datatype recvtype,
                       MPI_Comm comm) {
    //Bounds checking for radix
    //Less than 2, set to 2 for standard Bruck's algo
    if (r < 2) r = 2;

    //Standard variables declaration
    int rank, nprocs, typesize;

    //MPI API calls to initialize variables
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &nprocs);
    MPI_Type_size(sendtype, &typesize);

    if (r > nprocs - 1) r = nprocs - 1;
    if (b <= 0 || b > nprocs) b = nprocs;

    int w, max_rank, nlpow, d, K, i, num_reqs;
    int rotate_index_array[nprocs];
    w = 0; nlpow = 1; max_rank = nprocs - 1;

    while (max_rank) { w++; max_rank /= r; }
    for (i = 0; i < w - 1; i++) nlpow *= r;
    d = (nlpow * r - nprocs) / nlpow;
    K = w * (r - 1) - d;

    int rem1 = K + 1, rem2 = r + 1;
    char *extra_buffer = nullptr, *temp_recv_buffer = nullptr, *temp_send_buffer = nullptr;
    int extra_ids[nprocs - rem2];
    memset(extra_ids, -1, sizeof(extra_ids));
    int spoint = 1, distance = 1, next_distance = distance * r, di = 0;

    const size_t block_size = (size_t)sendcount * typesize;

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    if (K < nprocs - 1) {
        for (i = 0; i < nprocs; i++) {
            rotate_index_array[i] = (2 * rank - i + nprocs) % nprocs;
        }

        CUDA_CHECK(cudaMalloc((void**)&extra_buffer,     block_size * (nprocs - rem1)));
        CUDA_CHECK(cudaMalloc((void**)&temp_recv_buffer, block_size * nprocs));
        CUDA_CHECK(cudaMalloc((void**)&temp_send_buffer, block_size * nprocs));

        for (int x = 0; x < w; x++) {
            for (int z = 1; z < r; z++) {
                spoint = z * distance;
                if (spoint > nprocs) break;
                int end = (spoint + distance > nprocs) ? nprocs : spoint + distance;
                for (i = spoint + 1; i < end; i++) {
                    extra_ids[i - rem2] = di++;
                }
            }
            distance *= r;
        }
    }

    // copy data destined for self (device-to-device)
    CUDA_CHECK(cudaMemcpyAsync(&recvbuf[rank * block_size],
                               &sendbuf[rank * block_size],
                               block_size,
                               cudaMemcpyDeviceToDevice, stream));

    int sent_blocks[r - 1][nlpow];
    int nc, rem, ns, ze, ss;
    spoint = 1; distance = 1; next_distance = distance * r;

    MPI_Request *reqs  = (MPI_Request*) malloc(2 * r * sizeof(MPI_Request));
    MPI_Status  *stats = (MPI_Status*)  malloc(2 * r * sizeof(MPI_Status));
    if (!reqs || !stats) {
        std::cerr << "MPI_Request/Status allocation failed!\n";
        return 1;
    }

    for (int x = 0; x < w; x++) {
        ze = (x == w - 1) ? r - d : r;
        int zoffset = 0, zc = ze - 1;
        int zns[zc];

        for (int k = 1; k < ze; k += b) {
            ss = ze - k < b ? ze - k : b;
            num_reqs = 0;
            size_t send_zoffset = 0;

            for (int s = 0; s < ss; s++) {
                int z = k + s;

                spoint = z * distance;
                nc = nprocs / next_distance * distance;
                rem = nprocs % next_distance - spoint;
                if (rem < 0) rem = 0;
                ns = (rem > distance) ? (nc + distance) : (nc + rem);
                zns[z - 1] = ns;

                int recvrank = (rank + spoint) % nprocs;
                int sendrank = (rank - spoint + nprocs) % nprocs;

                if (ns == 1) {
                    MPI_Irecv(&recvbuf[recvrank * block_size], block_size, MPI_CHAR,
                              recvrank, 1, comm, &reqs[num_reqs++]);
                    MPI_Isend(&sendbuf[sendrank * block_size], block_size, MPI_CHAR,
                              sendrank, 1, comm, &reqs[num_reqs++]);
                } else {
                    di = 0;
                    for (int ii = spoint; ii < nprocs; ii += next_distance) {
                        int j_end = (ii + distance > nprocs) ? nprocs : ii + distance;
                        for (int j = ii; j < j_end; j++) {
                            int id = (j + rank) % nprocs;
                            sent_blocks[z - 1][di++] = id;
                        }
                    }

                    // prepare send data on device (block sizes uniform, no metadata exchange)
                    for (int ii = 0; ii < di; ii++) {
                        int send_index = rotate_index_array[sent_blocks[z - 1][ii]];
                        int o = (sent_blocks[z - 1][ii] - rank + nprocs) % nprocs - rem2;
                        size_t pos = send_zoffset + (size_t)ii * block_size;

                        if (ii % distance == 0) {
                            CUDA_CHECK(cudaMemcpyAsync(
                                &temp_send_buffer[pos],
                                &sendbuf[send_index * block_size],
                                block_size, cudaMemcpyDeviceToDevice, stream));
                        } else {
                            CUDA_CHECK(cudaMemcpyAsync(
                                &temp_send_buffer[pos],
                                &extra_buffer[(size_t)extra_ids[o] * block_size],
                                block_size, cudaMemcpyDeviceToDevice, stream));
                        }
                    }

                    // make sure all device copies for this z are done before MPI sees the buffer
                    CUDA_CHECK(cudaStreamSynchronize(stream));

                    size_t total = (size_t)di * block_size;
                    MPI_Irecv(&temp_recv_buffer[zoffset], total, MPI_CHAR,
                              recvrank, recvrank + z, comm, &reqs[num_reqs++]);
                    MPI_Isend(&temp_send_buffer[send_zoffset], total, MPI_CHAR,
                              sendrank, rank + z, comm, &reqs[num_reqs++]);

                    zoffset      += total;
                    send_zoffset += total;
                }
            }

            MPI_Waitall(num_reqs, reqs, stats);
            for (int ii = 0; ii < num_reqs; ii++) {
                if (stats[ii].MPI_ERROR != MPI_SUCCESS) {
                    std::cerr << "Request " << ii << " error: "
                              << stats[ii].MPI_ERROR << '\n';
                }
            }
        }

        if (K < nprocs - 1) {
            size_t offset = 0;
            for (int ii = 0; ii < zc; ii++) {
                for (int j = 0; j < zns[ii]; j++) {
                    if (zns[ii] > 1) {
                        int o = (sent_blocks[ii][j] - rank + nprocs) % nprocs - rem2;

                        if (j < distance) {
                            CUDA_CHECK(cudaMemcpyAsync(
                                &recvbuf[sent_blocks[ii][j] * block_size],
                                &temp_recv_buffer[offset],
                                block_size, cudaMemcpyDeviceToDevice, stream));
                        } else {
                            CUDA_CHECK(cudaMemcpyAsync(
                                &extra_buffer[(size_t)extra_ids[o] * block_size],
                                &temp_recv_buffer[offset],
                                block_size, cudaMemcpyDeviceToDevice, stream));
                        }
                        offset += block_size;
                    }
                }
            }
            CUDA_CHECK(cudaStreamSynchronize(stream));
        }

        distance *= r;
        next_distance *= r;
    }

    if (K < nprocs - 1) {
        cudaFree(extra_buffer);
        cudaFree(temp_recv_buffer);
        cudaFree(temp_send_buffer);
    }
    free(reqs);
    free(stats);
    cudaStreamDestroy(stream);

    return 0;
}
