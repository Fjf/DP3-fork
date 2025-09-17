// Copyright (C) 2023 ASTRON (Netherlands Institute for Radio Astronomy)
// SPDX-License-Identifier: GPL-3.0-or-later

#include "IterativeScalar.h"

#include <cuComplex.h>
#include <math_constants.h>

#include "Common.h"
#include "Complex.h"
#include "MatrixComplex2x2.h"

#include <iostream>

#define BLOCK_SIZE 256

#define cudaCheckError() {                                      \
 cudaError_t e=cudaGetLastError();                                 \
 if(e!=cudaSuccess) {                                              \
   printf("Cuda failure %s:%d: '%s'\n",__FILE__,__LINE__,cudaGetErrorString(e));           \
   exit(0); \
 }                                                                 \
}

template <bool Add>
__device__ void AddOrSubtractScalar(size_t vis_index, size_t n_solutions, size_t n_antenna,
                              const unsigned int* solution_map,
                              const cuDoubleComplex* solutions,
                              const cuFloatComplex* model,
                              const cuFloatComplex* residual_in,
                              cuFloatComplex* residual_out) {
  // Compute triangular index for the antenna pair
  // The pattern is: for each antenna_2, iterate through antenna_1 < antenna
  const size_t local_vis_index = vis_index % (n_antenna * (n_antenna - 1) / 2);
  const size_t antenna_2 = static_cast<uint32_t>((1.0 + std::sqrt(1.0 + 8.0 * local_vis_index)) / 2.0);
  const size_t antenna_1 = local_vis_index - (antenna_2 * (antenna_2 - 1)) / 2;  

  const size_t solution_index = solution_map[vis_index];
  const cuDoubleComplex solution_1 =
      solutions[antenna_1 * n_solutions + solution_index];
  const cuDoubleComplex solution_2 =
      solutions[antenna_2 * n_solutions + solution_index];

  const cuFloatComplex solution_1_val = cuComplexDoubleToFloat(solution_1);
  const cuFloatComplex solution_2_conj =
      cuComplexDoubleToFloat(cuConj(solution_2));

  const cuFloatComplex contribution = cuCmulf(cuCmulf(model[vis_index], solution_1_val), solution_2_conj);

  if (Add) {
    residual_out[vis_index] = cuCaddf(residual_in[vis_index], contribution);
  } else {
    residual_out[vis_index] = cuCsubf(residual_in[vis_index], contribution);
  }
}

__device__ void SolveScalarDirection(size_t ch_block, size_t vis_index, size_t n_visibilities,
                               size_t n_direction_solutions, size_t n_solutions, size_t n_antenna,
                               const unsigned int* solution_map,
                               const cuDoubleComplex* solutions,
                               const cuFloatComplex* model,
                               const cuFloatComplex* residual,
                               cuFloatComplex* numerator, float* denominator) {
  // Load correct variables to compute on.
  // Derive antenna indices from vis_index (for verification)
  // The pattern is: for each antenna_2, iterate through antenna_1 < antenna_2
  // vis_index 0: (0,1), vis_index 1: (0,2), vis_index 2: (1,2), vis_index 3: (0,3), etc.
  const size_t local_vis_index = vis_index % (n_antenna * (n_antenna - 1) / 2);
  const size_t antenna_2 = static_cast<uint32_t>((1.0 + std::sqrt(1.0 + 8.0 * local_vis_index)) / 2.0);
  const size_t antenna_1 = local_vis_index - (antenna_2 * (antenna_2 - 1)) / 2;

  const size_t solution_index = solution_map[vis_index];


  const cuDoubleComplex solution_antenna_1 =
      solutions[antenna_1 * n_solutions + solution_index];
  const cuDoubleComplex solution_antenna_2 =
      solutions[antenna_2 * n_solutions + solution_index];

  const size_t rel_solution_index = solution_index - solution_map[0];

    printf("%f %f %f\n", solution_antenna_1.x, solution_antenna_2.x);


  // Calculate the contribution of this baseline for both antennas
  // For antenna2,
  // data_ba = data_ab^H, etc., therefore, numerator and denominator
  // become:
  // - num = data_ab^H * solutions_a * model_ab
  // - den = norm(model_ab^H * solutions_a)
  for (size_t i = 0; i < 2; i++) {
    // const size_t antenna = antenna_pairs[vis_index * 2 + i];
    const size_t antenna = (i == 0) ? antenna_1 : antenna_2;

    cuFloatComplex result;
    cuFloatComplex changed_model;

    if (i == 0) {
      const cuFloatComplex solution_val =
          make_cuFloatComplex(solution_antenna_2.x, solution_antenna_2.y);

      // For scalar solving, we can directly access the model data
      const cuFloatComplex scalar_model = model[vis_index];
      const cuFloatComplex scalar_conj_model = cuConjf(scalar_model);
      const cuFloatComplex scalar_result = cuCmulf(solution_val, scalar_conj_model);

      changed_model = scalar_result;

      result = cuCmulf(residual[vis_index], scalar_result);

    } else {
      const cuFloatComplex solution_val =
          make_cuFloatComplex(solution_antenna_1.x, solution_antenna_1.y);

      // For scalar solving, we can directly access the model data
      const cuFloatComplex scalar_model = model[vis_index];
      const cuFloatComplex scalar_result = cuCmulf(solution_val, scalar_model);

      changed_model = scalar_result;

       result = cuCmulf(cuConjf(residual[vis_index]), scalar_result);

    }

    const size_t full_solution_index =
        antenna * n_direction_solutions + rel_solution_index;
    atomicAdd(&numerator[full_solution_index].x, result.x);
    atomicAdd(&numerator[full_solution_index].y, result.y);
    atomicAdd(&denominator[full_solution_index],
              cuCabsf(changed_model) * cuCabsf(changed_model));
  }
}

__global__ void SolveScalarDirectionKernel(
    size_t n_visibilities, size_t n_direction_solutions, size_t n_solutions, size_t n_antenna,
    const unsigned int* solution_map, const cuDoubleComplex* solutions, const cuFloatComplex* model,
    const cuFloatComplex* residual_in, cuFloatComplex* residual_temp,
    cuFloatComplex* numerator, float* denominator, struct sizes sizes) {
  const size_t vis_index = blockIdx.y * blockDim.x + threadIdx.x;
  const size_t ch_block = blockIdx.x;

  // printf("sizes: %lu %lu %lu %lu vis_index: %lu, channel_block: %lu\n", 
  //       (unsigned long) sizes.solution_map, (unsigned long) sizes.solutions, 
  //       (unsigned long) sizes.next_solutions, (unsigned long) sizes.model, 
  //       (unsigned long) vis_index, (unsigned long) ch_block);

  if (vis_index >= n_visibilities) {
    return;
  }

  // Calculate offsets for this channel block (don't modify the original pointers)
  const size_t residual_offset = ch_block * sizes.residual / sizeof(cuFloatComplex);
  const size_t solutions_offset = ch_block * sizes.solutions / sizeof(cuDoubleComplex);
  const size_t model_offset = ch_block * sizes.model / sizeof(cuFloatComplex);

  // Use the direction-specific pointers (solution_map and model are already offset for the current direction)
  AddOrSubtractScalar<true>(vis_index, n_solutions, n_antenna, solution_map,
                      solutions + solutions_offset, model + model_offset, residual_in + residual_offset, residual_temp + residual_offset);
  SolveScalarDirection(ch_block, vis_index, n_visibilities, n_direction_solutions, n_solutions, n_antenna,
                solution_map, solutions + solutions_offset, model + model_offset, residual_temp,
                 numerator, denominator);

}

void LaunchScalarSolveDirectionKernel(
    cudaStream_t stream, size_t n_visibilities, size_t n_direction_solutions,
    size_t n_solutions, size_t n_antenna, size_t n_parallel_channel_blocks, size_t direction,
    cu::DeviceMemory& solution_map, cu::DeviceMemory& solutions,
    cu::DeviceMemory& model, cu::DeviceMemory& residual_in,
    cu::DeviceMemory& residual_temp, cu::DeviceMemory& numerator,
    cu::DeviceMemory& denominator, struct sizes sizes) {
  const size_t block_dim = BLOCK_SIZE;
  // const size_t grid_dim = (n_visibilities + block_dim) / block_dim;
  const dim3 grid_dim(n_parallel_channel_blocks, (n_visibilities + block_dim) / block_dim);

  const size_t direction_offset = direction * n_visibilities;
  const unsigned int* solution_map_direction =
      Cast<const unsigned int>(solution_map) + direction_offset;
  const cuFloatComplex* model_direction =
      Cast<const cuFloatComplex>(model) + direction_offset;

  
  SolveScalarDirectionKernel<<<grid_dim, block_dim, 0, stream>>>(
      n_visibilities, n_direction_solutions, n_solutions, n_antenna, solution_map_direction,
      Cast<const cuDoubleComplex>(solutions), model_direction,
      Cast<const cuFloatComplex>(residual_in),
      Cast<cuFloatComplex>(residual_temp), Cast<cuFloatComplex>(numerator),
      Cast<float>(denominator), sizes);

  cudaCheckError();
}

__global__ void SubtractScalarKernel(size_t n_directions, size_t n_visibilities,
                               size_t n_solutions, size_t n_antenna,
                               const unsigned int* solution_map,
                               const cuDoubleComplex* solutions,
                               const cuFloatComplex* model,
                               cuFloatComplex* residual, struct sizes sizes) {
  const size_t vis_index = blockIdx.y * blockDim.x + threadIdx.x;
  const size_t ch_block = blockIdx.x;
  // const size_t n_channel_blocks = blockDim.x;

  // printf("sizes: %lu %lu %lu %lu vis_index: %lu, channel_block: %lu\n", (unsigned long) sizes.solution_map, (unsigned long) sizes.solutions, (unsigned long) sizes.next_solutions, (unsigned long) sizes.model, (unsigned long) vis_index, (unsigned long) ch_block);

  if (vis_index >= n_visibilities) {
    return;
  };
  
  // Calculate offsets for this channel block (don't modify the original pointers)
  const size_t residual_offset = ch_block * sizes.residual / sizeof(cuFloatComplex);
  const size_t solutions_offset = ch_block * sizes.solutions / sizeof(cuDoubleComplex);
  
  for (size_t direction = 0; direction < n_directions; direction++) {
    const size_t direction_offset = direction * n_visibilities;
    const unsigned int* solution_map_direction =
        solution_map + direction_offset + ch_block * sizes.solution_map / sizeof(unsigned int);
    const cuFloatComplex* model_direction = model + direction_offset + ch_block * sizes.model / sizeof(cuFloatComplex);
    // printf("model_direction: %p, sizes: %p\n", model_direction, residual);
    AddOrSubtractScalar<false>(
        vis_index, n_solutions, n_antenna, solution_map_direction,
        solutions + solutions_offset, model_direction,
        residual + residual_offset, residual + residual_offset);  // in-place
  }

}

void LaunchScalarSubtractKernel(cudaStream_t stream, size_t n_directions,
                          size_t n_visibilities, size_t n_solutions, size_t n_antenna, size_t n_parallel_channel_blocks,
                          cu::DeviceMemory& solution_map,
                          cu::DeviceMemory& solutions, cu::DeviceMemory& model,
                          cu::DeviceMemory& residual, struct sizes sizes) {
  const size_t block_dim = BLOCK_SIZE;
  const dim3 grid_dim(n_parallel_channel_blocks, (n_visibilities + block_dim) / block_dim);



  SubtractScalarKernel<<<grid_dim, block_dim, 0, stream>>>(
      n_directions, n_visibilities, n_solutions, n_antenna,
      Cast<const unsigned int>(solution_map),
      Cast<const cuDoubleComplex>(solutions), Cast<const cuFloatComplex>(model),
      Cast<cuFloatComplex>(residual), sizes);
  cudaCheckError();
}

__global__ void SolveNextScalarSolutionKernel(unsigned int n_antennas,
                                        unsigned int n_direction_solutions,
                                        const unsigned int n_solutions,
                                        const unsigned int* solution_map,
                                        const cuFloatComplex* numerator,
                                        const float* denominator,
                                        cuDoubleComplex* next_solutions) {
  const size_t ch_block = blockIdx.x;
  const size_t antenna = blockIdx.y * blockDim.x + threadIdx.x;

  const size_t n_visibilities = n_direction_solutions * n_antennas;


  if (antenna >= n_antennas) {
    return;
  }

  for (size_t relative_solution = 0; relative_solution < n_direction_solutions;
       relative_solution++) {
    const size_t solution_index = relative_solution + solution_map[0];


    const size_t dest_idx = (ch_block * n_visibilities) + antenna * n_solutions + solution_index;


    const size_t index = (antenna * n_direction_solutions + relative_solution);

    // Print values being used
    if (denominator[index] == 0.0) {
      next_solutions[dest_idx] = {CUDART_NAN, CUDART_NAN};
    } else {
      next_solutions[dest_idx] = {
          numerator[index].x / denominator[index],
          numerator[index].y / denominator[index]};
    }
  }
}

void LaunchScalarSolveNextSolutionKernel(
    cudaStream_t stream, size_t n_antennas, size_t n_visibilities,
    size_t n_direction_solutions, size_t n_solutions, size_t n_channel_blocks, size_t direction,
    cu::DeviceMemory& solution_map, cu::DeviceMemory& next_solutions,
    cu::DeviceMemory& numerator, cu::DeviceMemory& denominator) {

  const size_t block_dim = BLOCK_SIZE;
  const dim3 grid_dim(n_channel_blocks, (n_visibilities + block_dim) / block_dim);

  const size_t direction_offset = direction * n_visibilities;

  const unsigned int* solution_map_direction =
      Cast<const unsigned int>(solution_map) + direction_offset;

  SolveNextScalarSolutionKernel<<<grid_dim, block_dim, 0, stream>>>(
      n_antennas, n_direction_solutions, n_solutions, solution_map_direction,
      Cast<const cuFloatComplex>(numerator), Cast<const float>(denominator),
      Cast<cuDoubleComplex>(next_solutions));
  cudaCheckError();
}

__global__ void StepScalarKernel(const size_t n_visibilities,
                           const cuDoubleComplex* solutions,
                           cuDoubleComplex* next_solutions, bool phase_only,
                           double step_size) {
  const size_t ch_block = blockIdx.x;
  const size_t vis = blockIdx.y * blockDim.x + threadIdx.x;

  if (vis >= n_visibilities) {
    return;
  }

  const size_t vis_index = (ch_block * n_visibilities) + vis;

  if (phase_only) {
    // In phase only mode, a step is made along the complex circle,
    // towards the shortest direction.
    double phase_from = cuCarg(solutions[vis_index]);
    double distance = cuCarg(next_solutions[vis_index]) - phase_from;
    if (distance > CUDART_PI)
      distance = distance - 2.0 * CUDART_PI;
    else if (distance < -CUDART_PI)
      distance = distance + 2.0 * CUDART_PI;

    next_solutions[vis_index] =
        cuCpolar(1.0, phase_from + step_size * distance);
  } else {
    next_solutions[vis_index] =
        cuCadd(cuCmul(solutions[vis_index], (1.0 - step_size)),
               cuCmul(next_solutions[vis_index], step_size));
  }
}

void LaunchScalarStepKernel(cudaStream_t stream, size_t n_visibilities, size_t n_channel_blocks,
                      cu::DeviceMemory& solutions,
                      cu::DeviceMemory& next_solutions, bool phase_only,
                      double step_size) {
  const size_t block_dim = BLOCK_SIZE;
  const dim3 grid_dim(n_channel_blocks, (n_visibilities + block_dim) / block_dim);

  StepScalarKernel<<<grid_dim, block_dim, 0, stream>>>(
      n_visibilities, Cast<const cuDoubleComplex>(solutions),
      Cast<cuDoubleComplex>(next_solutions), phase_only, step_size);
  cudaCheckError();
}
