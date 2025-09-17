// Copyright (C) 2023 ASTRON (Netherlands Institute for Radio Astronomy)
// SPDX-License-Identifier: GPL-3.0-or-later

#ifndef DP3_DDECAL_GAIN_SOLVERS_KERNELS_ITERATIVESCALAR_H_
#define DP3_DDECAL_GAIN_SOLVERS_KERNELS_ITERATIVESCALAR_H_

#include <complex>
#include <cuda_runtime.h>

#include <cudawrappers/cu.hpp>
struct sizes {
  size_t solution_map;
  size_t solutions;
  size_t next_solutions;
  size_t model;
  size_t residual;
  size_t numerator;
  size_t denominator;
};

void LaunchScalarSubtractKernel(cudaStream_t stream, size_t n_directions,
                                size_t n_visibilities, size_t n_solutions,
                                size_t n_antenna, size_t n_parallel_channel_blocks,
                                cu::DeviceMemory& solution_map,
                                cu::DeviceMemory& solutions,
                                cu::DeviceMemory& model,
                                cu::DeviceMemory& residual, struct sizes sizes);

void LaunchScalarSolveNextSolutionKernel(
    cudaStream_t stream, size_t n_antennas, size_t n_visibilities,
    size_t n_direction_solutions, size_t n_solutions, size_t n_channel_blocks,
    size_t direction, cu::DeviceMemory& solution_map,
    cu::DeviceMemory& next_solutions, cu::DeviceMemory& numerator,
    cu::DeviceMemory& denominator);

void LaunchScalarSolveDirectionKernel(
    cudaStream_t stream, size_t n_visibilities, size_t n_direction_solutions,
    size_t n_solutions, size_t n_antenna, size_t n_parallel_channel_blocks,
    size_t direction, cu::DeviceMemory& solution_map,
    cu::DeviceMemory& solutions, cu::DeviceMemory& model,
    cu::DeviceMemory& residual_in, cu::DeviceMemory& residual_temp,
    cu::DeviceMemory& numerator, cu::DeviceMemory& denominator,
    struct sizes sizes);

void LaunchScalarStepKernel(cudaStream_t stream, size_t n_visibilities,
                            size_t n_channel_blocks,
                            cu::DeviceMemory& solutions,
                            cu::DeviceMemory& next_solutions, bool phase_only,
                            double step_size);

#endif  // DP3_DDECAL_GAIN_SOLVERS_KERNELS_ITERATIVESCALAR_H_