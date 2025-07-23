// Copyright (C) 2023 ASTRON (Netherlands Institute for Radio Astronomy)
// SPDX-License-Identifier: GPL-3.0-or-later

#include "IterativeScalar.h"

#include <cuComplex.h>
#include <math_constants.h>

#include "Common.h"
#include "Complex.h"
#include "MatrixComplex2x2.h"

#define BLOCK_SIZE 1

#define cudaCheckError() {                                      \
 cudaError_t e=cudaGetLastError();                                 \
 printf("testing cudaCheckError %s:%d\n",__FILE__,__LINE__); \
 if(e!=cudaSuccess) {                                              \
   printf("Cuda failure %s:%d: '%s'\n",__FILE__,__LINE__,cudaGetErrorString(e));           \
   exit(0); \
 } \
 else {                                                         \
   printf("Cuda success %s:%d: '%s'\n",__FILE__,__LINE__,cudaGetErrorString(e));           \
} \
}

template <bool Add>
__device__ void AddOrSubtractScalar(size_t vis_index, size_t n_solutions,
                              const unsigned int* antenna_pairs,
                              const unsigned int* solution_map,
                              const cuDoubleComplex* solutions,
                              const cuFloatComplex* model,
                              const cuFloatComplex* residual_in,
                              cuFloatComplex* residual_out) {
  const uint32_t antenna_1 = antenna_pairs[vis_index * 2 + 0];
  const uint32_t antenna_2 = antenna_pairs[vis_index * 2 + 1];
  const size_t solution_index = solution_map[vis_index];
  const cuDoubleComplex solution_1 =
      solutions[antenna_1 * n_solutions + solution_index];
  const cuDoubleComplex solution_2 =
      solutions[antenna_2 * n_solutions + solution_index];

  const cuFloatComplex solution_1_val = cuComplexDoubleToFloat(solution_1);
  const cuFloatComplex solution_2_conj =
      cuComplexDoubleToFloat(cuConj(solution_2));

  // printf("DEBUG GPU kernel vis_index=%lu,residual_in=[{%.6f,%.6f}]\n",
  //        (unsigned long)vis_index,
  //        residual_out[vis_index].x, residual_out[vis_index].y);

  const cuFloatComplex contribution = cuCmulf(cuCmulf(model[vis_index], solution_1_val), solution_2_conj);
  
  // Debug: Add bounds checking and error detection
  // printf("DEBUG GPU kernel vis_index=%lu, antenna_1=%u, antenna_2=%u, sol_idx=%lu, model=(%f,%f), contrib=(%f,%f)\n",
  //         (unsigned long)vis_index, antenna_1, antenna_2, (unsigned long)solution_index,
  //         model[vis_index].x, model[vis_index].y, contribution.x, contribution.y);
  if (Add) {
    residual_out[vis_index] = cuCaddf(residual_in[vis_index], contribution);
  } else {
    residual_out[vis_index] = cuCsubf(residual_in[vis_index], contribution);
  }
}

template <bool Add>
__device__ void AddOrSubtractScalar2(size_t vis_index, size_t n_solutions,
                              const unsigned int* antenna_pairs,
                              const unsigned int* solution_map,
                              const cuDoubleComplex* solutions,
                              const cuFloatComplex* model,
                              const cuFloatComplex* residual_in,
                              cuFloatComplex* residual_out) {
  const uint32_t antenna_1 = antenna_pairs[vis_index * 2 + 0];
  const uint32_t antenna_2 = antenna_pairs[vis_index * 2 + 1];
  const size_t solution_index = solution_map[vis_index];
  const cuDoubleComplex solution_1 =
      solutions[antenna_1 * n_solutions + solution_index];
  const cuDoubleComplex solution_2 =
      solutions[antenna_2 * n_solutions + solution_index];

  const cuFloatComplex solution_1_val = cuComplexDoubleToFloat(solution_1);
  const cuFloatComplex solution_2_conj =
      cuComplexDoubleToFloat(cuConj(solution_2));

  // printf("DEBUG GPU kernel vis_index=%lu,residual_in=[{%.6f,%.6f}]\n",
  //        (unsigned long)vis_index,
  //        residual_out[vis_index].x, residual_out[vis_index].y);

  const cuFloatComplex contribution = cuCmulf(cuCmulf(model[vis_index], solution_1_val), solution_2_conj);
  
  // Debug: Add bounds checking and error detection
  // printf("DEBUG GPU kernel vis_index=%lu, antenna_1=%u, antenna_2=%u, sol_idx=%lu, model=(%f,%f), contrib=(%f,%f)\n",
  //         (unsigned long)vis_index, antenna_1, antenna_2, (unsigned long)solution_index,
  //         model[vis_index].x, model[vis_index].y, contribution.x, contribution.y);
  if (Add) {
    residual_out[vis_index] = cuCaddf(residual_in[vis_index], contribution);
  } else {
    residual_out[vis_index] = cuCsubf(residual_in[vis_index], contribution);
  }
}

__device__ void SolveScalarDirection(size_t vis_index, size_t n_visibilities,
                               size_t n_direction_solutions, size_t n_solutions,
                               const unsigned int* antenna_pairs,
                               const unsigned int* solution_map,
                               const cuDoubleComplex* solutions,
                               const cuFloatComplex* model,
                               const cuFloatComplex* residual,
                               cuFloatComplex* numerator, float* denominator) {
  // Load correct variables to compute on.
  const size_t antenna_1 = antenna_pairs[vis_index * 2];
  const size_t antenna_2 = antenna_pairs[vis_index * 2 + 1];

  const size_t solution_index = solution_map[vis_index];


  const cuDoubleComplex solution_antenna_1 =
      solutions[antenna_1 * n_solutions + solution_index];
  const cuDoubleComplex solution_antenna_2 =
      solutions[antenna_2 * n_solutions + solution_index];

  const size_t rel_solution_index = solution_index - solution_map[0];



  // Calculate the contribution of this baseline for both antennas
  // For antenna2,
  // data_ba = data_ab^H, etc., therefore, numerator and denominator
  // become:
  // - num = data_ab^H * solutions_a * model_ab
  // - den = norm(model_ab^H * solutions_a)
  for (size_t i = 0; i < 2; i++) {
    const size_t antenna = antenna_pairs[vis_index * 2 + i];

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
      
      // Debug: print model values being used for first few visibilities or non-zero values

      
      result = cuCmulf(residual[vis_index], scalar_result);
      // if ((vis_index < 5) || (cuCabsf(scalar_model) > 1e-10)) {
      //   printf("DEBUG GPU kernel vis_index=%lu: scalar_model={%.6f,%.6f}, sol_val={%.6f,%.6f}, result={%.6f,%.6f}, norm=%.6f\n",
      //          (unsigned long)vis_index, 
      //          scalar_model.x, scalar_model.y,
      //          solution_val.x, solution_val.y,
      //          scalar_result.x, scalar_result.y,
      //          cuCabsf(scalar_model));
      // }

    } else {
      const cuFloatComplex solution_val = 
          make_cuFloatComplex(solution_antenna_1.x, solution_antenna_1.y);
      
      // For scalar solving, we can directly access the model data
      const cuFloatComplex scalar_model = model[vis_index];
      const cuFloatComplex scalar_result = cuCmulf(solution_val, scalar_model);
      
      changed_model = scalar_result;
      

      
      // printf("  i=%lu: changed_model=[{%.1f,%.1f}, {%.1f,%.1f}, {%.1f,%.1f}, {%.1f,%.1f}]\n",
      //        (unsigned long)i, changed_model[0].x, changed_model[0].y, changed_model[1].x, changed_model[1].y,
      //        changed_model[2].x, changed_model[2].y, changed_model[3].x, changed_model[3].y);
      result = cuCmulf(cuConjf(residual[vis_index]), scalar_result);
            // Debug: print model values for first few visibilities or non-zero values in second antenna case  
      // if ((vis_index < 5) || (cuCabsf(scalar_model) > 1e-10)) {
        //   printf("DEBUG GPU kernel vis_index=%lu (i=1): scalar_model={%.6f,%.6f}, sol_val={%.6f,%.6f}, changed_model ={%.6f,%.6f, %.6f, %.6f}, result={%.6f,%.6f, %.6f, %.6f}, norm=%.6f\n, residual={%.6f,%.6f, %.6f, %.6f, %.6f, %.6f, %.6f, %.6f}\n",
        //          (unsigned long)vis_index, 
        //          scalar_model.x, scalar_model.y,
        //          solution_val.x, solution_val.y,
        //          changed_model[0].x, changed_model[0].y,
        //          changed_model[3].x, changed_model[3].y,
        //          result[0].x, result[0].y,
        //          result[3].x, result[3].y,
        //          cuCabsf(scalar_model)),
        //          residual[vis_index][0].x, residual[vis_index][0].y, residual[vis_index][1].x, residual[vis_index][1].y, residual[vis_index][2].x, residual[vis_index][2].y, residual[vis_index][3].x, residual[vis_index][3].y;
        // }
    }

    const size_t full_solution_index =
        antenna * n_direction_solutions + rel_solution_index;
    // printf("DEBUG GPU kernel vis_index=%lu, antenna=%lu, full_solution_index=%lu, rel_solution_index=%lu, solution_index=%lu, n_direction_solutions=%lu, solution_map=%lu, result={%.6f,%.6f}, changed_model={%.6f,%.6f}\n",
    //        (unsigned long)vis_index, (unsigned long)antenna,
    //        (unsigned long)full_solution_index, (unsigned long)rel_solution_index, (unsigned long)solution_index, (unsigned long)n_direction_solutions,
    //        (unsigned int)solution_map, result.x, result.y, changed_model.x, changed_model.y);

    // Atomic reduction into global memory - for scalar solver, we only need one value
    atomicAdd(&numerator[full_solution_index * 2].x, result.x);
    atomicAdd(&numerator[full_solution_index * 2].y, result.y);
    
    atomicAdd(&denominator[full_solution_index * 2],
              cuCabsf(changed_model) * cuCabsf(changed_model));
  }
}

__global__ void SolveScalarDirectionKernel(
    size_t n_visibilities, size_t n_direction_solutions, size_t n_solutions,
    const unsigned int* antenna_pairs, const unsigned int* solution_map,
    const cuDoubleComplex* solutions, const cuFloatComplex* model,
    const cuFloatComplex* residual_in, cuFloatComplex* residual_temp,
    cuFloatComplex* numerator, float* denominator) {
  const size_t vis_index = blockIdx.x * blockDim.x + threadIdx.x;

  if (vis_index >= n_visibilities) {
    return;
  }

  // Use the direction-specific pointers (solution_map and model are already offset for the current direction)
  AddOrSubtractScalar<true>(vis_index, n_solutions, antenna_pairs, solution_map,
                      solutions, model, residual_in, residual_temp);
  // printf("DEBUG GPU kernel vis_index=%lu: residual_temp=[{%.1f,%.1f}, {%.1f,%.1f}, {%.1f,%.1f}, {%.1f,%.1f}]\n",
  //        (unsigned long)vis_index, 
  //        residual_in[vis_index][0].x, residual_in[vis_index][0].y,
  //        residual_in[vis_index][1].x, residual_in[vis_index][1].y,
  //        residual_in[vis_index][2].x, residual_in[vis_index][2].y,
  //        residual_in[vis_index][3].x, residual_in[vis_index][3].y);

  SolveScalarDirection(vis_index, n_visibilities, n_direction_solutions, n_solutions,
                 antenna_pairs, solution_map, solutions, model, residual_temp,
                 numerator, denominator);
}

void LaunchScalarSolveDirectionKernel(
    cudaStream_t stream, size_t n_visibilities, size_t n_direction_solutions,
    size_t n_solutions, size_t direction, cu::DeviceMemory& antenna_pairs,
    cu::DeviceMemory& solution_map, cu::DeviceMemory& solutions,
    cu::DeviceMemory& model, cu::DeviceMemory& residual_in,
    cu::DeviceMemory& residual_temp, cu::DeviceMemory& numerator,
    cu::DeviceMemory& denominator) {
  const size_t block_dim = BLOCK_SIZE;
  const size_t grid_dim = (n_visibilities + block_dim) / block_dim;

  const size_t direction_offset = direction * n_visibilities;
  const unsigned int* solution_map_direction =
      Cast<const unsigned int>(solution_map) + direction_offset;
  const cuFloatComplex* model_direction =
      Cast<const cuFloatComplex>(model) + direction_offset;
  SolveScalarDirectionKernel<<<grid_dim, block_dim, 0, stream>>>(
      n_visibilities, n_direction_solutions, n_solutions,
      Cast<const unsigned int>(antenna_pairs), solution_map_direction,
      Cast<const cuDoubleComplex>(solutions), model_direction,
      Cast<const cuFloatComplex>(residual_in),
      Cast<cuFloatComplex>(residual_temp), Cast<cuFloatComplex>(numerator),
      Cast<float>(denominator));

  cudaCheckError();
  cudaDeviceSynchronize(); // Ensure printf output is flushed
}

__global__ void SubtractScalarKernel(size_t n_directions, size_t n_visibilities,
                               size_t n_solutions,
                               const unsigned int* antenna_pairs,
                               const unsigned int* solution_map,
                               const cuDoubleComplex* solutions,
                               const cuFloatComplex* model,
                               cuFloatComplex* residual) {
  const size_t vis_index = blockIdx.x * blockDim.x + threadIdx.x;

  if (vis_index >= n_visibilities) {
    return;
  }

  for (size_t direction = 0; direction < n_directions; direction++) {
    const size_t direction_offset = direction * n_visibilities;
    const unsigned int* solution_map_direction =
        solution_map + direction_offset;
    const cuFloatComplex* model_direction = model + direction_offset;
    AddOrSubtractScalar<false>(
        vis_index, n_solutions, antenna_pairs, solution_map_direction,
        solutions, model_direction,
        residual, residual);  // in-place
  }
}

__global__ void SubtractScalarKernel2(size_t n_directions, size_t n_visibilities,
                               size_t n_solutions,
                               const unsigned int* antenna_pairs,
                               const unsigned int* solution_map,
                               const cuDoubleComplex* solutions,
                               const cuFloatComplex* model,
                               cuFloatComplex* residual) {
  const size_t vis_index = blockIdx.x * blockDim.x + threadIdx.x;

  if (vis_index >= n_visibilities) {
    return;
  }

  for (size_t direction = 0; direction < n_directions; direction++) {
    const size_t direction_offset = direction * n_visibilities;
    const unsigned int* solution_map_direction =
        solution_map + direction_offset;
    const cuFloatComplex* model_direction = model + direction_offset;
    AddOrSubtractScalar2<false>(
        vis_index, n_solutions, antenna_pairs, solution_map_direction,
        solutions, model_direction,
        residual, residual);  // in-place
  }
}

// TODO: Error is here, fix please
void LaunchScalarSubtractKernel(cudaStream_t stream, size_t n_directions,
                          size_t n_visibilities, size_t n_solutions,
                          cu::DeviceMemory& antenna_pairs,
                          cu::DeviceMemory& solution_map,
                          cu::DeviceMemory& solutions, cu::DeviceMemory& model,
                          cu::DeviceMemory& residual) {
  const size_t block_dim = BLOCK_SIZE;
  const size_t grid_dim = (n_visibilities + block_dim) / block_dim;
  
  printf("DEBUG LaunchScalarSubtractKernel: n_directions=%zu, n_visibilities=%zu, n_solutions=%zu\n",
         n_directions, n_visibilities, n_solutions);
  printf("DEBUG LaunchScalarSubtractKernel: block_dim=%zu, grid_dim=%zu\n", block_dim, grid_dim);
  
  cudaCheckError();
  
  SubtractScalarKernel<<<grid_dim, block_dim, 0, stream>>>(
      n_directions, n_visibilities, n_solutions,
      Cast<const unsigned int>(antenna_pairs),
      Cast<const unsigned int>(solution_map),
      Cast<const cuDoubleComplex>(solutions), Cast<const cuFloatComplex>(model),
      Cast<cuFloatComplex>(residual));
  
  printf("DEBUG LaunchScalarSubtractKernel: After kernel launch\n");
  cudaCheckError();
  printf("DEBUG LaunchScalarSubtractKernel: After cudaCheckError\n");
}

void LaunchScalarSubtractKernel2(cudaStream_t stream, size_t n_directions,
                          size_t n_visibilities, size_t n_solutions,
                          cu::DeviceMemory& antenna_pairs,
                          cu::DeviceMemory& solution_map,
                          cu::DeviceMemory& solutions, cu::DeviceMemory& model,
                          cu::DeviceMemory& residual) {
  const size_t block_dim = BLOCK_SIZE;
  const size_t grid_dim = (n_visibilities + block_dim) / block_dim;
  
  printf("DEBUG LaunchScalarSubtractKernel: n_directions=%zu, n_visibilities=%zu, n_solutions=%zu\n",
         n_directions, n_visibilities, n_solutions);
  printf("DEBUG LaunchScalarSubtractKernel: block_dim=%zu, grid_dim=%zu\n", block_dim, grid_dim);
  
  cudaCheckError();
  
  SubtractScalarKernel2<<<grid_dim, block_dim, 0, stream>>>(
      n_directions, n_visibilities, n_solutions,
      Cast<const unsigned int>(antenna_pairs),
      Cast<const unsigned int>(solution_map),
      Cast<const cuDoubleComplex>(solutions), Cast<const cuFloatComplex>(model),
      Cast<cuFloatComplex>(residual));
  
  printf("DEBUG LaunchScalarSubtractKernel: After kernel launch\n");
  cudaCheckError();
  printf("DEBUG LaunchScalarSubtractKernel: After cudaCheckError\n");
}

__global__ void SolveNextScalarSolutionKernel(unsigned int n_antennas,
                                        unsigned int n_direction_solutions,
                                        const unsigned int n_solutions,
                                        const unsigned int* solution_map,
                                        const cuFloatComplex* numerator,
                                        const float* denominator,
                                        cuDoubleComplex* next_solutions) {
  const size_t antenna = blockIdx.x * blockDim.x + threadIdx.x;

  if (antenna >= n_antennas) {
    return;
  }

  // printf("\nDEBUG Thread info (antenna %lu):\n", (unsigned long)antenna);
  // printf("  blockIdx.x: %d, blockDim.x: %d, threadIdx.x: %d\n", 
  //        blockIdx.x, blockDim.x, threadIdx.x);

  for (size_t relative_solution = 0; relative_solution < n_direction_solutions;
       relative_solution++) {
    const size_t solution_index = relative_solution + solution_map[0];

    
    const size_t dest_idx = antenna * n_solutions + solution_index;

    
    const size_t index = antenna * n_direction_solutions + relative_solution;
    

    // Print values being used

    if (denominator[index * 2] == 0.0) {
      next_solutions[dest_idx] = {CUDART_NAN, CUDART_NAN};
      // printf("  Setting NaN due to zero denominator\n");
    } else {
      next_solutions[dest_idx] = {
          numerator[index * 2].x / denominator[index * 2],
          numerator[index * 2].y / denominator[index * 2]};
      // printf("  Result = {%f, %f}\n", 
      //        numerator[index * 2].x / denominator[index * 2],
      //        numerator[index * 2].y / denominator[index * 2]);
    }
  }
}

void LaunchScalarSolveNextSolutionKernel(
    cudaStream_t stream, size_t n_antennas, size_t n_visibilities,
    size_t n_direction_solutions, size_t n_solutions, size_t direction,
    cu::DeviceMemory& antenna_pairs, cu::DeviceMemory& solution_map,
    cu::DeviceMemory& next_solutions, cu::DeviceMemory& numerator,
    cu::DeviceMemory& denominator) {
  
  const size_t block_dim = BLOCK_SIZE;
  const size_t grid_dim = (n_antennas + block_dim - 1) / block_dim;

  // printf("\nDEBUG LaunchScalarSolveNextSolutionKernel:\n");
  // printf("  n_antennas: %lu\n", (unsigned long)n_antennas);
  // printf("  n_visibilities: %lu\n", (unsigned long)n_visibilities);
  // printf("  n_direction_solutions: %lu\n", (unsigned long)n_direction_solutions);
  // printf("  n_solutions: %lu\n", (unsigned long)n_solutions);
  // printf("  direction: %lu\n", (unsigned long)direction);
  // printf("  grid_dim: %lu\n", (unsigned long)grid_dim);
  // printf("  block_dim: %lu\n", (unsigned long)block_dim);

  const size_t direction_offset = direction * n_visibilities;
  // printf("  direction_offset: %lu\n", (unsigned long)direction_offset);

  const unsigned int* solution_map_direction =
      Cast<const unsigned int>(solution_map) + direction_offset;

  // Print memory sizes
  // printf("\nDEBUG Memory sizes and offsets:\n");
  // printf("  solution_map offset: %lu bytes\n", direction_offset * sizeof(unsigned int));
  // printf("  numerator size: %lu bytes\n", n_antennas * n_direction_solutions * 2 * sizeof(cuFloatComplex));
  // printf("  denominator size: %lu bytes\n", n_antennas * n_direction_solutions * 2 * sizeof(float));
  // printf("  next_solutions size: %lu bytes\n", n_antennas * n_solutions * sizeof(cuDoubleComplex));

  cudaCheckError();
  
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
  const size_t vis_index = blockIdx.x * blockDim.x + threadIdx.x;

  if (vis_index >= n_visibilities) {
    return;
  }

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

void LaunchScalarStepKernel(cudaStream_t stream, size_t n_visibilities,
                      cu::DeviceMemory& solutions,
                      cu::DeviceMemory& next_solutions, bool phase_only,
                      double step_size) {
  const size_t block_dim = BLOCK_SIZE;
  const size_t grid_dim = (n_visibilities + block_dim) / block_dim;
  cudaCheckError();

  StepScalarKernel<<<grid_dim, block_dim, 0, stream>>>(
      n_visibilities, Cast<const cuDoubleComplex>(solutions),
      Cast<cuDoubleComplex>(next_solutions), phase_only, step_size);
  cudaCheckError();
}
