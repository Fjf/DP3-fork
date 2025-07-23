// Copyright (C) 2023 ASTRON (Netherlands Institute for Radio Astronomy)
// SPDX-License-Identifier: GPL-3.0-or-later

#include "IterativeScalarSolverCuda.h"

#include <algorithm>
#include <iostream>
#include <vector>
#include <numeric>
#include <fstream>
#include <iomanip>

#include <cuda_runtime.h>
#include <nvToolsExt.h>

#include <aocommon/matrix2x2.h>
#include <aocommon/matrix2x2diag.h>

#include "kernels/IterativeScalar.h"
#include "kernels/IterativeDiagonal.h"

using aocommon::MC2x2;
using aocommon::MC2x2F;

void DumpSolutionsToFile(const std::vector<std::vector<std::complex<double>>>& solutions, 
                         const std::string& filename, size_t iteration = 0) {
  std::ofstream file(filename);
  if (!file.is_open()) {
    std::cerr << "ERROR: Could not open file " << filename << " for writing" << std::endl;
    return;
  }
  
  file << std::scientific << std::setprecision(15);
  
  // Write header with metadata
  file << "# DP3 CUDA Solver Solutions Dump" << std::endl;
  file << "# Iteration: " << iteration << std::endl;
  file << "# Number of channel blocks: " << solutions.size() << std::endl;
  file << "# Format: channel_block antenna_index real_part imaginary_part magnitude phase" << std::endl;
  file << "# " << std::endl;
  
  for (size_t ch_block = 0; ch_block < solutions.size(); ++ch_block) {
    const auto& ch_solutions = solutions[ch_block];
    file << "# Channel block " << ch_block << " has " << ch_solutions.size() << " solutions" << std::endl;
    
    for (size_t ant_idx = 0; ant_idx < ch_solutions.size(); ++ant_idx) {
      const auto& solution = ch_solutions[ant_idx];
      double magnitude = std::abs(solution);
      double phase = std::arg(solution);
      
      file << ch_block << " " 
           << ant_idx << " " 
           << solution.real() << " " 
           << solution.imag() << " " 
           << magnitude << " " 
           << phase << std::endl;
    }
    file << std::endl; // Blank line between channel blocks
  }
  
  file.close();
  std::cout << "Solutions dumped to file: " << filename << std::endl;
  std::cout << "Total channel blocks: " << solutions.size() << std::endl;
  
  // Print summary statistics
  size_t total_solutions = 0;
  size_t nan_count = 0;
  size_t inf_count = 0;
  double max_magnitude = 0.0;
  double min_magnitude = std::numeric_limits<double>::max();
  
  for (const auto& ch_solutions : solutions) {
    for (const auto& solution : ch_solutions) {
      total_solutions++;
      double mag = std::abs(solution);
      if (std::isnan(mag)) {
        nan_count++;
      } else if (std::isinf(mag)) {
        inf_count++;
      } else {
        max_magnitude = std::max(max_magnitude, mag);
        min_magnitude = std::min(min_magnitude, mag);
      }
    }
  }
  
  std::cout << "Solutions summary:" << std::endl;
  std::cout << "  Total solutions: " << total_solutions << std::endl;
  std::cout << "  NaN solutions: " << nan_count << std::endl;
  std::cout << "  Inf solutions: " << inf_count << std::endl;
  std::cout << "  Min magnitude: " << min_magnitude << std::endl;
  std::cout << "  Max magnitude: " << max_magnitude << std::endl;
}

namespace {

inline float Norm(std::complex<float> value) { return std::norm(value); }
inline double Norm(std::complex<double> value) { return std::norm(value); }

#include <type_traits>

/// Helper for scalar types
template <typename T>
typename std::enable_if<!std::is_class<T>::value, void>::type
PrintVectorSummary(const std::vector<T>& vec, const std::string& name) {
  if (vec.empty()) {
    std::cout << name << " is empty." << std::endl;
    return;
  }
  std::cout << name << " first: " << vec.front() << std::endl;
  std::cout << name << " last: " << vec.back() << std::endl;
  T sum = std::accumulate(vec.begin(), vec.end(), T{});
  auto mean = sum / static_cast<float>(vec.size());
  std::cout << name << " mean: " << mean << std::endl;
  double sq_sum = 0.0;
  for (const auto& v : vec) {
    auto diff = v - mean;
    sq_sum += std::norm(diff);
  }
  double stddev = std::sqrt(sq_sum / vec.size());
  std::cout << name << " std: " << stddev << std::endl;

  std::cout << name << " size: " << vec.size() << std::endl;
  //  print max and min
  auto minmax = std::minmax_element(vec.begin(), vec.end());
  std::cout << name << " min: " << *minmax.first << std::endl;
  std::cout << name << " max: " << *minmax.second << "\n" << std::endl;
}

// Helper for matrix types (prints norm summary)
template <typename T>
typename std::enable_if<std::is_class<T>::value, void>::type PrintVectorSummary(
    const std::vector<T>& vec, const std::string& name) {
  if (vec.empty()) {
    std::cout << name << " is empty." << std::endl;
    return;
  }
  auto norm = [](const T& m) { return Norm(m); };
  std::cout << name << " first norm: " << norm(vec.front()) << std::endl;
  std::cout << name << " last norm: " << norm(vec.back()) << std::endl;

  double sum = 0.0;
  size_t valid_count = 0;
  size_t nan_count = 0;

  for (const auto& v : vec) {
    double n = norm(v);
    if (std::isnan(n)) {
      nan_count++;
      // std::cout << "norm: nan" << std::endl;
    } else if (std::isinf(n)) {
      // std::cout << "norm: inf" << std::endl;
    } else {
      sum += n;
      valid_count++;
      // std::cout << "norm: " << n << std::endl;
    }
  }

  std::cout << name << " valid elements: " << valid_count << std::endl;
  std::cout << name << " nan elements: " << nan_count << std::endl;

  if (valid_count > 0) {
    double mean = sum / valid_count;
    std::cout << name << " mean norm: " << mean << std::endl;

    double sq_sum = 0.0;
    for (const auto& v : vec) {
      double n = norm(v);
      if (!std::isnan(n) && !std::isinf(n)) {
        double diff = n - mean;
        sq_sum += diff * diff;
      }
    }
    double stddev = std::sqrt(sq_sum / valid_count);
    std::cout << name << " std norm: " << stddev << std::endl;

    // Find min/max excluding NaN and inf values
    double min_norm = std::numeric_limits<double>::max();
    double max_norm = std::numeric_limits<double>::lowest();

    for (const auto& v : vec) {
      double n = norm(v);
      if (!std::isnan(n) && !std::isinf(n)) {
        min_norm = std::min(min_norm, n);
        max_norm = std::max(max_norm, n);
      }
    }

    std::cout << name << " min norm: " << min_norm << std::endl;
    std::cout << name << " max norm: " << max_norm << std::endl;
  } else {
    std::cout << name << " mean norm: all values are nan/inf" << std::endl;
    std::cout << name << " std norm: all values are nan/inf" << std::endl;
    std::cout << name << " min norm: all values are nan/inf" << std::endl;
    std::cout << name << " max norm: all values are nan/inf" << std::endl;
  }

  std::cout << name << " size: " << vec.size() << "\n" << std::endl;
}

// Helper function to analyze SolutionTensor
void PrintSolutionTensorSummary(
    const xt::xtensor<std::complex<double>, 4>& next_solutions,
    const std::string& name, size_t ch_block = 0) {
  if (next_solutions.size() == 0) {
    std::cout << name << " tensor is empty" << std::endl;
    return;
  }

  // Extract solutions for the specific channel block
  std::vector<std::complex<double>> flat_solutions;
  auto shape = next_solutions.shape();
  if (ch_block < shape[0]) {
    for (size_t ant = 0; ant < shape[1]; ++ant) {
      for (size_t sol = 0; sol < shape[2]; ++sol) {
        for (size_t pol = 0; pol < shape[3]; ++pol) {
          flat_solutions.push_back(next_solutions(ch_block, ant, sol, pol));
        }
      }
    }
  }

  if (!flat_solutions.empty()) {
    PrintVectorSummary(flat_solutions, name + "_ch" + std::to_string(ch_block));
  } else {
    std::cout << name << " has no data for ch_block " << ch_block << std::endl;
  }
}

template <typename VisMatrix>
size_t SizeOfModel(size_t n_directions, size_t n_visibilities) {
  return n_directions * n_visibilities * sizeof(VisMatrix);  // 8
}

template <typename VisMatrix>
size_t SizeOfResidual(size_t n_visibilities) {
  return n_visibilities * sizeof(std::complex<float>);
}

size_t SizeOfSolutions(size_t n_directions, size_t n_antennas, size_t n_subsol, size_t n_pol) {
  return n_directions * n_antennas * n_subsol * n_pol * sizeof(std::complex<double>);
}

size_t SizeOfAntennaPairs(size_t n_visibilities) {
  return n_visibilities * 2 * sizeof(uint32_t);
}

size_t SizeOfSolutionMap(size_t n_directions, size_t n_visibilities) {
  return n_directions * n_visibilities * sizeof(uint32_t);
}

size_t SizeOfNextSolutions(size_t n_channel_blocks, size_t n_antennas, size_t n_subsol, size_t n_pol) {
  return n_channel_blocks * n_antennas * n_subsol * n_pol * sizeof(std::complex<double>);
}

size_t SizeOfNumerator(size_t n_antennas, size_t n_direction_solutions) {
  return n_antennas * n_direction_solutions * sizeof(std::complex<float>);
}

size_t SizeOfDenominator(size_t n_antennas, size_t n_direction_solutions) {
  return n_antennas * n_direction_solutions * sizeof(float);
}

template <typename VisMatrix>
using ChannelBlockData =
    typename dp3::ddecal::SolveData<VisMatrix>::ChannelBlockData;

template <typename VisMatrix>
void SolveDirection(const ChannelBlockData<VisMatrix>& channel_block_data,
                    cu::Stream& stream, size_t n_antennas, size_t n_solutions,
                    size_t direction, cu::DeviceMemory& device_residual_in,
                    cu::DeviceMemory& device_residual_temp,
                    cu::DeviceMemory& device_solution_map,
                    cu::DeviceMemory& device_solutions,
                    cu::DeviceMemory& device_model,
                    cu::DeviceMemory& device_next_solutions,
                    cu::DeviceMemory& device_antenna_pairs,
                    cu::DeviceMemory& device_numerator,
                    cu::DeviceMemory& device_denominator) {
  // Calculate this equation, given ant a:
  //
  //          sum_b data_ab * solutions_b * model_ab^*
  // sol_a =  ----------------------------------------
  //             sum_b norm(model_ab * solutions_b)
  const size_t n_direction_solutions =
      channel_block_data.NSolutionsForDirection(direction);
  const size_t n_visibilities = channel_block_data.NVisibilities();

  // Initialize values to 0
  stream.zero(device_numerator,
              SizeOfNumerator(n_antennas, n_direction_solutions));

  stream.zero(device_denominator,
              SizeOfDenominator(n_antennas, n_direction_solutions));

  stream.memcpyDtoDAsync(device_residual_temp, device_residual_in,
                         SizeOfResidual<VisMatrix>(n_visibilities));

  LaunchScalarSolveDirectionKernel(
      stream, n_visibilities, n_direction_solutions, n_solutions, direction,
      device_antenna_pairs, device_solution_map, device_solutions, device_model,
      device_residual_in, device_residual_temp, device_numerator,
      device_denominator);

  // Ensure the direction kernel completes before starting next solution kernel
  // stream.synchronize();

  LaunchScalarSolveNextSolutionKernel(
      stream, n_antennas, n_visibilities, n_direction_solutions, n_solutions,
      direction, device_antenna_pairs, device_solution_map,
      device_next_solutions, device_numerator, device_denominator);
}

template <typename VisMatrix>
void PerformIteration(
    bool phase_only, double step_size,
    const ChannelBlockData<VisMatrix>& channel_block_data, cu::Stream& stream,
    size_t n_antennas, size_t n_solutions, size_t n_directions,
    cu::DeviceMemory& device_solution_map, cu::DeviceMemory& device_solutions,
    cu::DeviceMemory& device_next_solutions, cu::DeviceMemory& device_residual,
    cu::DeviceMemory& device_residual_temp, cu::DeviceMemory& device_model,
    cu::DeviceMemory& device_antenna_pairs, cu::DeviceMemory& device_numerator,
    cu::DeviceMemory& device_denominator) {
  const size_t n_visibilities = channel_block_data.NVisibilities();

  // Copy visibility data to residual buffer first
  // std::vector<VisMatrix> residual_data(n_visibilities);
  // std::copy(&channel_block_data.Visibility(0),
  //           &channel_block_data.Visibility(0) + n_visibilities,
  //           residual_data.begin());

  // print residual for debugging
  // PrintVectorSummary(residual_data, "residual_pre_kernel");

  // Copy to GPU device
  // stream.memcpyHtoDAsync(device_residual, residual_data.data(),
  //                        SizeOfResidual<VisMatrix>(n_visibilities));
  // // Synchronize the streams to ensure the residual is ready
  // stream.synchronize();

  // Subtract all directions with their current solutions
  // In-place: residual -> residual
  LaunchScalarSubtractKernel(stream, n_directions, n_visibilities, n_solutions,
                             device_antenna_pairs, device_solution_map,
                             device_solutions, device_model, device_residual);

  // // Copy result back from GPU and print for debugging
  // stream.memcpyDtoHAsync(residual_data.data(), device_residual,
  //                        SizeOfResidual<VisMatrix>(n_visibilities));
  // stream.synchronize();
  // // PrintVectorSummary(residual_data, "v_residual_post_kernel");

  // // Copy result back from GPU and print for debugging
  // stream.memcpyDtoHAsync(residual_data.data(), device_residual,
  //                        SizeOfResidual<VisMatrix>(n_visibilities));
  // stream.synchronize();
  // // PrintVectorSummary(residual_data, "v_residual_post_kernel");
  // // exit(0);

  for (size_t direction = 0; direction != n_directions; direction++) {
    // Be aware that we purposely still use the subtraction with 'old'
    // solutions, because the new solutions have not been constrained yet. Add
    // this direction back before solving

    // Out-of-place: residual -> residual_temp

    SolveDirection<VisMatrix>(
        channel_block_data, stream, n_antennas, n_solutions, direction,
        device_residual, device_residual_temp, device_solution_map,
        device_solutions, device_model, device_next_solutions,
        device_antenna_pairs, device_numerator, device_denominator);

    // Print channel_block_data information (CPU side, before GPU data transfer)
    // std::cout << "channel_block_data info:" << std::endl;
    // std::cout << "  NVisibilities: " << channel_block_data.NVisibilities()
    //           << std::endl;
    // std::cout << "  NDirections: " << channel_block_data.NDirections()
    //           << std::endl;
    // std::cout << "  NSolutionsForDirection(" << direction
    //           << "): " << channel_block_data.NSolutionsForDirection(direction)
    //           << std::endl;
    // std::cout << "  NSubSolutions (total): "
    //           << channel_block_data.NSubSolutions() << std::endl;
    //
    // // Print some antenna pair information for first few visibilities
    // const size_t max_vis_to_show =
    //     std::min(static_cast<size_t>(5), channel_block_data.NVisibilities());
    // std::cout << "  First " << max_vis_to_show
    //           << " antenna pairs:" << std::endl;
    // for (size_t i = 0; i < max_vis_to_show; ++i) {
    //   std::cout << "    vis[" << i
    //             << "]: ant1=" << channel_block_data.Antenna1Index(i)
    //             << ", ant2=" << channel_block_data.Antenna2Index(i)
    //             << ", sol_idx="
    //             << channel_block_data.SolutionIndex(direction, i) << std::endl;
    // }

    // Print some model visibility norms for this direction
    // std::vector<double> model_norms;
    // for (size_t i = 0; i < max_vis_to_show; ++i) {
    //   const auto& model_vis = channel_block_data.ModelVisibility(direction, i);
    //   model_norms.push_back(Norm(model_vis));
    // }
    // if (!model_norms.empty()) {
    //   std::cout << "  First " << max_vis_to_show
    //             << " model visibility norms for direction " << direction << ":"
    //             << std::endl;
    //   for (size_t i = 0; i < model_norms.size(); ++i) {
    //     std::cout << "    model_vis[" << i << "] norm: " << model_norms[i]
    //               << std::endl;
    //   }
    // }

    // Count and show actual model values for comparison with CPU version
    // size_t total_model_values = 0;
    // size_t non_zero_model_values = 0;
    // std::cout << "  CPU model values for direction " << direction << " (first "
    //           << max_vis_to_show << "):" << std::endl;
    // for (size_t i = 0; i < max_vis_to_show; ++i) {
    //   const auto& model_vis = channel_block_data.ModelVisibility(direction, i);
    //   std::cout << "    CPU model_vis[" << i << "]: " << model_vis << std::endl;
    // }
    //
    // // Count all non-zero model values like the CPU version does
    // for (size_t i = 0; i < n_visibilities; ++i) {
    //   const auto& model_vis = channel_block_data.ModelVisibility(direction, i);
    //   total_model_values++;
    //   if (Norm(model_vis) > 1e-12) {
    //     non_zero_model_values++;
    //   }
    // }
    // std::cout << "  Count non-zero model values: " << non_zero_model_values
    //           << " out of " << total_model_values << " total" << std::endl;

    // Verify GPU model data by copying back from device
    // std::cout << "DEBUG: Verifying GPU model data for direction " <<
    // direction << std::endl;
    // const size_t model_sample_size =
    //     std::min(static_cast<size_t>(10), n_visibilities);
    // size_t gpu_non_zero_count = 0;
    // if constexpr (std::is_same_v<VisMatrix, std::complex<float>>) {
    //   std::vector<std::complex<float>> gpu_model_full(n_directions *
    //                                                   n_visibilities);
    //   stream.memcpyDtoHAsync(
    //       gpu_model_full.data(), device_model,
    //       n_directions * n_visibilities * sizeof(std::complex<float>));
    //   stream.synchronize();
    //   std::cout << "  GPU model values for direction " << direction
    //             << " (first " << model_sample_size << "):" << std::endl;
    //   const size_t offset = direction * n_visibilities;
    //   for (size_t i = 0; i < model_sample_size; ++i) {
    //     std::cout << "    GPU model_vis[" << i
    //               << "]: " << gpu_model_full[offset + i] << std::endl;
    //   }
    //
    //   // Count all non-zero values in GPU data
    //   for (size_t i = 0; i < n_visibilities; ++i) {
    //     if (std::abs(gpu_model_full[offset + i]) > 1e-12) {
    //       gpu_non_zero_count++;
    //     }
    //   }
    // } else if constexpr (std::is_same_v<VisMatrix, std::complex<double>>) {
    //   std::vector<std::complex<double>> gpu_model_full(n_directions *
    //                                                    n_visibilities);
    //   stream.memcpyDtoHAsync(
    //       gpu_model_full.data(), device_model,
    //       n_directions * n_visibilities * sizeof(std::complex<double>));
    //   stream.synchronize();
    //   std::cout << "  GPU model values for direction " << direction
    //             << " (first " << model_sample_size << "):" << std::endl;
    //   const size_t offset = direction * n_visibilities;
    //   for (size_t i = 0; i < model_sample_size; ++i) {
    //     std::cout << "    GPU model_vis[" << i
    //               << "]: " << gpu_model_full[offset + i] << std::endl;
    //   }
    //
    //   // Count all non-zero values in GPU data
    //   for (size_t i = 0; i < n_visibilities; ++i) {
    //     if (std::abs(gpu_model_full[offset + i]) > 1e-12) {
    //       gpu_non_zero_count++;
    //     }
    //   }
    // }
    // std::cout << "  GPU non-zero model values: " << gpu_non_zero_count
    //           << " out of " << n_visibilities << std::endl;

    // stream.memcpyDtoHAsync(residual_data.data(), device_residual_temp,
    //                        SizeOfResidual<VisMatrix>(n_visibilities));
    // stream.synchronize();
    // PrintVectorSummary(residual_data, "v_residual_post_kernel_2");
    // exit(0);

    // Move device_solutions to host (CPU) for analysis - use separate buffers
    // std::vector<std::complex<double>> current_solutions_data(n_antennas *
    //                                                          n_solutions);
    // std::vector<std::complex<double>> next_solutions_data(n_antennas *
    //                                                       n_solutions);

    // stream.memcpyDtoHAsync(
    //     current_solutions_data.data(),
    //     device_solutions,
    //     SizeOfSolutions(n_directions, n_antennas, n_solutions, 1)); // For scalar solver, polarizations = 1
    // stream.memcpyDtoHAsync(
    //     next_solutions_data.data(),
    //     device_next_solutions,
    //     SizeOfSolutions(n_directions, n_antennas, n_solutions, 1)); // For scalar solver, polarizations = 1
    // stream.synchronize();

    // PrintVectorSummary(current_solutions_data,
    //                    "solutions_post_solve_direction");
    // PrintVectorSummary(next_solutions_data,
    //                    "next_solutions_post_solve_direction");

    // exit(0);

    // if (direction == 0) exit(0);  // Exit after first direction only
  }

  // std::cout << "DEBUG: Iteration completed for channel block with "
  //           << n_directions << " directions." << std::endl;

  LaunchStepKernel(stream, n_visibilities, device_solutions,
                   device_next_solutions, phase_only, step_size);
  // stream.synchronize();
  // std::cout << "DEBUG: Step kernel launched." << std::endl;
}

template <typename VisMatrix>
std::tuple<size_t, size_t, size_t> ComputeArrayDimensions(
    const dp3::ddecal::SolveData<VisMatrix>& data) {
  size_t max_n_direction_solutions = 0;
  size_t max_n_visibilities = 0;
  size_t max_n_directions = 0;

  for (size_t ch_block = 0; ch_block < data.NChannelBlocks(); ch_block++) {
    const ChannelBlockData<VisMatrix>& channel_block_data =
        data.ChannelBlock(ch_block);
    max_n_visibilities =
        std::max(max_n_visibilities, channel_block_data.NVisibilities());
    max_n_directions =
        std::max(max_n_directions, channel_block_data.NDirections());
    for (size_t direction = 0; direction < channel_block_data.NDirections();
         direction++) {
      max_n_direction_solutions =
          std::max(max_n_direction_solutions,
                   static_cast<size_t>(
                       channel_block_data.NSolutionsForDirection(direction)));
    }
  }

  return std::make_tuple(max_n_direction_solutions, max_n_visibilities,
                         max_n_directions);
}
}  // namespace

namespace dp3 {
namespace ddecal {

template <typename VisMatrix>
IterativeScalarSolverCuda<VisMatrix>::IterativeScalarSolverCuda(
    bool keep_buffers)
    : SolverBase(), keep_buffers_{keep_buffers} {
  cu::init();
  device_ = std::make_unique<cu::Device>(0);
  context_ = std::make_unique<cu::Context>(0, *device_);
  context_->setCurrent();
  execute_stream_ = std::make_unique<cu::Stream>();
  host_to_device_stream_ = std::make_unique<cu::Stream>();
  device_to_host_stream_ = std::make_unique<cu::Stream>();
}

template <typename VisMatrix>
void IterativeScalarSolverCuda<VisMatrix>::AllocateGPUBuffers(
    const SolveData<VisMatrix>& data) {
   size_t max_n_direction_solutions = 0;
  size_t max_n_visibilities = 0;
  size_t max_n_directions = 0;
  std::tie(max_n_direction_solutions, max_n_visibilities, max_n_directions) =
      ComputeArrayDimensions(data);

  gpu_buffers_.numerator = std::make_unique<cu::DeviceMemory>(
      SizeOfNumerator(NAntennas(), max_n_direction_solutions));
  gpu_buffers_.denominator = std::make_unique<cu::DeviceMemory>(
      SizeOfDenominator(NAntennas(), max_n_direction_solutions));
  // Allocating two buffers allows double buffering.
  for (size_t i = 0; i < 2; i++) {
    gpu_buffers_.antenna_pairs.emplace_back(
        SizeOfAntennaPairs(max_n_visibilities));
    gpu_buffers_.solution_map.emplace_back(
        SizeOfSolutionMap(max_n_directions, max_n_visibilities));
    gpu_buffers_.solutions.emplace_back(SizeOfSolutions(NDirections(), NAntennas(), NSubSolutions(), NSolutionPolarizations()));
    gpu_buffers_.next_solutions.emplace_back(
        SizeOfNextSolutions(NDirections(), NAntennas(), NSubSolutions(), NSolutionPolarizations()));
    gpu_buffers_.model.emplace_back(
        SizeOfModel<VisMatrix>(max_n_directions, max_n_visibilities));
  }

  // We need two buffers for residual like above to facilitate double-buffering,
  // the third buffer is used for the per-direction add/subtract.
  for (size_t i = 0; i < 3; i++) {
    gpu_buffers_.residual.emplace_back(
        SizeOfResidual<VisMatrix>(max_n_visibilities));
  }

  try {
    // Verify that device memory allocations succeeded
    for (const auto& mem : gpu_buffers_.antenna_pairs) {
      if (!mem) throw std::runtime_error("antenna_pairs buffer allocation failed");
    }
    for (const auto& mem : gpu_buffers_.solution_map) {
      if (!mem) throw std::runtime_error("solution_map buffer allocation failed");
    }
    for (const auto& mem : gpu_buffers_.solutions) {
      if (!mem) throw std::runtime_error("solutions buffer allocation failed");
    }
    for (const auto& mem : gpu_buffers_.next_solutions) {
      if (!mem) throw std::runtime_error("next_solutions buffer allocation failed");
    }
    for (const auto& mem : gpu_buffers_.model) {
      if (!mem) throw std::runtime_error("model buffer allocation failed");
    }
    for (const auto& mem : gpu_buffers_.residual) {
      if (!mem) throw std::runtime_error("residual buffer allocation failed");
    }

    // Verify unique_ptr managed buffers
    if (!gpu_buffers_.numerator) {
      throw std::runtime_error("numerator buffer allocation failed");
    }
    if (!gpu_buffers_.denominator) {
      throw std::runtime_error("denominator buffer allocation failed");
    }

    // Verify CUDA device has enough memory
    size_t free, total;
    cudaMemGetInfo(&free, &total);
    if (free < total * 0.1) { // Less than 10% free memory
      throw std::runtime_error("Insufficient GPU memory available");
    }

  } catch (const std::exception& e) {
    // Clean up any allocated buffers
    gpu_buffers_.antenna_pairs.clear();
    gpu_buffers_.solution_map.clear();
    gpu_buffers_.solutions.clear();
    gpu_buffers_.next_solutions.clear();
    gpu_buffers_.model.clear();
    gpu_buffers_.residual.clear();
    gpu_buffers_.numerator.reset();
    gpu_buffers_.denominator.reset();

    throw std::runtime_error(std::string("GPU buffer allocation failed: ") + e.what());
  }
}

template <typename VisMatrix>
void IterativeScalarSolverCuda<VisMatrix>::AllocateHostBuffers(
    const SolveData<VisMatrix>& data) {
  // For scalar solver, we need one solution per antenna per polarization
  const size_t total_solutions =
      NAntennas() * NSubSolutions() * NSolutionPolarizations();
  // std::cout << "DEBUG: Allocating host_buffers_.next_solutions" << std::endl;
  // std::cout << "DEBUG: Total solutions: " << total_solutions << std::endl;
  const size_t next_solutions_size = SizeOfNextSolutions(
      NChannelBlocks(), NAntennas(), NSubSolutions(), NSolutionPolarizations());
  // std::cout << "DEBUG: Next solutions buffer size: " << next_solutions_size
  // << " bytes" << std::endl;

  try {
    host_buffers_.next_solutions =
        std::make_unique<cu::HostMemory>(next_solutions_size);
    // std::cout << "DEBUG: host_buffers_.next_solutions allocated successfully"
    // << std::endl; std::cout << "DEBUG: Allocated pointer: " << std::hex <<
    // static_cast<void*>(*host_buffers_.next_solutions) << std::dec <<
    // std::endl;
  } catch (const std::exception& e) {
    std::cerr << "ERROR allocating host_buffers_.next_solutions: " << e.what()
              << std::endl;
    throw;
  }
  for (size_t ch_block = 0; ch_block < NChannelBlocks(); ch_block++) {
    const ChannelBlockData<VisMatrix>& channel_block_data =
        data.ChannelBlock(ch_block);
    const size_t n_directions = channel_block_data.NDirections();
    const size_t n_visibilities = channel_block_data.NVisibilities();
    host_buffers_.model.emplace_back(
        SizeOfModel<VisMatrix>(n_directions, n_visibilities));
    host_buffers_.residual.emplace_back(
        SizeOfResidual<VisMatrix>(n_visibilities));
    host_buffers_.solutions.emplace_back(
        SizeOfSolutions(NDirections(), NAntennas(), NSubSolutions(), NSolutionPolarizations()));
    host_buffers_.antenna_pairs.emplace_back(
        SizeOfAntennaPairs(n_visibilities));
    host_buffers_.solution_map.emplace_back(
        SizeOfSolutionMap(n_directions, n_visibilities));
    uint32_t* antenna_pairs =
        static_cast<uint32_t*>(host_buffers_.antenna_pairs[ch_block]);
    for (size_t visibility_index = 0; visibility_index < n_visibilities;
         visibility_index++) {
      antenna_pairs[visibility_index * 2 + 0] =
          channel_block_data.Antenna1Index(visibility_index);
      antenna_pairs[visibility_index * 2 + 1] =
          channel_block_data.Antenna2Index(visibility_index);
    }
  }
}

template <typename VisMatrix>
void IterativeScalarSolverCuda<VisMatrix>::DeallocateHostBuffers() {
  host_buffers_.next_solutions.reset();
  host_buffers_.model.clear();
  host_buffers_.residual.clear();
  host_buffers_.solutions.clear();
  host_buffers_.antenna_pairs.clear();
  host_buffers_.solution_map.clear();
  host_buffers_initialized_ = false;
}
template <typename VisMatrix>
void IterativeScalarSolverCuda<VisMatrix>::CopyHostToHost(
    size_t ch_block, bool first_iteration, const SolveData<VisMatrix>& data,
    const std::vector<std::complex<double>>& solutions, cu::Stream& stream) {
  const ChannelBlockData<VisMatrix>& channel_block_data =
      data.ChannelBlock(ch_block);
  const size_t n_directions = channel_block_data.NDirections();
  const size_t n_visibilities = channel_block_data.NVisibilities();

  cu::HostMemory& host_model = host_buffers_.model[ch_block];
  cu::HostMemory& host_solutions = host_buffers_.solutions[ch_block];

  stream.memcpyHtoHAsync(host_model, &channel_block_data.ModelVisibility(0, 0),
                         SizeOfModel<VisMatrix>(n_directions, n_visibilities));
  stream.memcpyHtoHAsync(host_solutions, solutions.data(),
                         SizeOfSolutions(NDirections(), NAntennas(), NSubSolutions(), NSolutionPolarizations()));
  if (first_iteration) {
    cu::HostMemory& host_residual = host_buffers_.residual[ch_block];
    cu::HostMemory& host_solution_map = host_buffers_.solution_map[ch_block];
    stream.memcpyHtoHAsync(host_residual, &channel_block_data.Visibility(0),
                           SizeOfResidual<VisMatrix>(n_visibilities));
    stream.memcpyHtoHAsync(host_solution_map,
                           channel_block_data.SolutionMapData(),
                           SizeOfSolutionMap(n_directions, n_visibilities));
  }
}

template <typename VisMatrix>
void IterativeScalarSolverCuda<VisMatrix>::CopyHostToDevice(
    size_t ch_block, size_t buffer_id, cu::Stream& stream, cu::Event& event,
    const SolveData<VisMatrix>& data) {
  const ChannelBlockData<VisMatrix>& channel_block_data =
      data.ChannelBlock(ch_block);

  const size_t n_directions = channel_block_data.NDirections();
  const size_t n_visibilities = channel_block_data.NVisibilities();

  cu::HostMemory& host_solution_map = host_buffers_.solution_map[ch_block];
  cu::HostMemory& host_antenna_pairs = host_buffers_.antenna_pairs[ch_block];
  cu::HostMemory& host_model = host_buffers_.model[ch_block];
  cu::HostMemory& host_residual = host_buffers_.residual[ch_block];
  cu::HostMemory& host_solutions = host_buffers_.solutions[ch_block];


  cu::DeviceMemory& device_solution_map = gpu_buffers_.solution_map[buffer_id];
  cu::DeviceMemory& device_antenna_pairs = gpu_buffers_.antenna_pairs[buffer_id];
  cu::DeviceMemory& device_model = gpu_buffers_.model[buffer_id];
  cu::DeviceMemory& device_residual = gpu_buffers_.residual[buffer_id];
  cu::DeviceMemory& device_solutions = gpu_buffers_.solutions[buffer_id];

  // std::cout << "Copying host to device for buffer id=" << buffer_id << std::endl;
  stream.memcpyHtoDAsync(device_solution_map, host_solution_map,
                         SizeOfSolutionMap(n_directions, n_visibilities));
  stream.memcpyHtoDAsync(device_model, host_model,
                         SizeOfModel<VisMatrix>(n_directions, n_visibilities));
  stream.memcpyHtoDAsync(device_residual, host_residual,
                         SizeOfResidual<VisMatrix>(n_visibilities));
  stream.memcpyHtoDAsync(device_antenna_pairs, host_antenna_pairs,
                         SizeOfAntennaPairs(n_visibilities));
  // std::cout << "\tantenna pairs[0] on host =" << ((uint32_t*)host_antenna_pairs)[0] << std::endl;

  stream.memcpyHtoDAsync(device_solutions, host_solutions,
                         SizeOfSolutions(NDirections(), NAntennas(), NSubSolutions(), NSolutionPolarizations()));

  // stream.synchronize();
  event.record(stream);
}
template <typename VisMatrix>
void IterativeScalarSolverCuda<VisMatrix>::PostProcessing(
    size_t& iteration, double time, bool has_previously_converged,
    bool& has_converged, bool& constraints_satisfied, bool& done,
    SolverBase::SolveResult& result,
    std::vector<std::vector<std::complex<double>>>& solutions,
    SolutionSpan& next_solutions, std::vector<double>& step_magnitudes,
    std::ostream* stat_stream) {
  constraints_satisfied =
      ApplyConstraints(iteration, time, has_previously_converged, result,
                       next_solutions, stat_stream);

  double avg_squared_diff;
  has_converged =
      AssignSolutions(solutions, next_solutions, !constraints_satisfied,
                      avg_squared_diff, step_magnitudes);
  iteration++;

  has_previously_converged = has_converged || has_previously_converged;

  done = ReachedStoppingCriterion(iteration, has_converged,
                                  constraints_satisfied, step_magnitudes);
}

template <typename VisMatrix>
SolverBase::SolveResult IterativeScalarSolverCuda<VisMatrix>::Solve(
    const SolveData<VisMatrix>& data,
    std::vector<std::vector<DComplex>>& solutions, double time,
    std::ostream* stat_stream) {
  try {
    PrepareConstraints();
    context_->setCurrent();

    // Validate CUDA context and device
    if (!device_ || !context_) {
      throw std::runtime_error("CUDA device or context not initialized");
    }

    const bool phase_only = GetPhaseOnly();
    const double step_size = GetStepSize();

    SolveResult result;

    /*
     * Allocate buffers
     */
    if (!host_buffers_initialized_) {
      AllocateHostBuffers(data);
      if (!host_buffers_.next_solutions) {
        throw std::runtime_error(
            "Failed to allocate host next_solutions buffer");
      }
      host_buffers_initialized_ = true;
    }

    if (!gpu_buffers_initialized_) {
      AllocateGPUBuffers(data);
      if (!gpu_buffers_.numerator || !gpu_buffers_.denominator) {
        throw std::runtime_error(
            "Failed to allocate GPU numerator/denominator buffers");
      }
      gpu_buffers_initialized_ = true;
    }

    // Validate essential buffers are allocated
    if (host_buffers_.model.empty() || host_buffers_.residual.empty() ||
        host_buffers_.solutions.empty() ||
        host_buffers_.antenna_pairs.empty()) {
      throw std::runtime_error("Host buffer vectors not properly allocated");
    }

    if (gpu_buffers_.antenna_pairs.empty() ||
        gpu_buffers_.solution_map.empty() || gpu_buffers_.solutions.empty() ||
        gpu_buffers_.next_solutions.empty() || gpu_buffers_.model.empty() ||
        gpu_buffers_.residual.empty()) {
      throw std::runtime_error("GPU buffer vectors not properly allocated");
    }

    const std::array<size_t, 4> next_solutions_shape = {
        NChannelBlocks(), NAntennas(), NSubSolutions(),
        NSolutionPolarizations()};

    // Validate solution shape dimensions
    if (next_solutions_shape[0] == 0 || next_solutions_shape[1] == 0 ||
        next_solutions_shape[2] == 0 || next_solutions_shape[3] == 0) {
      throw std::runtime_error("Invalid solution shape dimensions");
    }

    // Validate next_solutions pointer and create span
    std::complex<double>* next_solutions_ptr = *(host_buffers_.next_solutions);
    if (!next_solutions_ptr) {
      throw std::runtime_error("next_solutions_ptr is null");
    }

    // Validate pointer alignment for complex<double>
    if (reinterpret_cast<std::uintptr_t>(next_solutions_ptr) %
            alignof(std::complex<double>) !=
        0) {
      throw std::runtime_error("next_solutions_ptr is not properly aligned");
    }

    // Calculate total size needed for the solution span
    size_t total_elements = 1;
    for (size_t dim : next_solutions_shape) {
      // Check for overflow
      if (dim > std::numeric_limits<size_t>::max() / total_elements) {
        throw std::runtime_error(
            "Solution span size calculation would overflow");
      }
      total_elements *= dim;
    }

    // Verify calculated size matches allocated buffer size
    size_t buffer_size = SizeOfNextSolutions(NChannelBlocks(), NAntennas(), NSubSolutions(), NSolutionPolarizations());
    size_t required_size = total_elements * sizeof(std::complex<double>);
    if (buffer_size < required_size) {
      throw std::runtime_error("Buffer size mismatch: allocated " +
                               std::to_string(buffer_size) + " bytes, need " +
                               std::to_string(required_size) + " bytes");
    }

    // Define buffer type before use
    using buffer_type =
        xt::xbuffer_adaptor<std::complex<double>*, xt::no_ownership>;
    buffer_type buffer(next_solutions_ptr, total_elements);

    // Create span with RAII and proper validation
    SolutionSpan next_solutions(buffer, next_solutions_shape);

    // Validate the created span
    if (!next_solutions.data()) {
      throw std::runtime_error("Solution span data pointer is null");
    }

    size_t span_total_elements = std::accumulate(
        next_solutions.shape().begin(), next_solutions.shape().end(), static_cast<size_t>(1),
        std::multiplies<size_t>());

    if (span_total_elements != total_elements) {
      std::ostringstream oss;
      oss << "Solution span size mismatch: got " << span_total_elements
          << " elements, expected " << total_elements;
      throw std::runtime_error(oss.str());
    }

    // Verify shape dimensions
    auto shape = next_solutions.shape();
    if (shape[0] != NChannelBlocks() || shape[1] != NAntennas() ||
        shape[2] != NSubSolutions() || shape[3] != NSolutionPolarizations()) {
      std::ostringstream oss;
      oss << "Solution span shape mismatch: expected [" << NChannelBlocks()
          << "," << NAntennas() << "," << NSubSolutions() << ","
          << NSolutionPolarizations() << "]"
          << " but got [" << shape[0] << "," << shape[1] << "," << shape[2]
          << "," << shape[3] << "]";
      throw std::runtime_error(oss.str());
    }

    /*
     * Allocate events for each channel block
     */
    std::vector<cu::Event> input_copied_events(NChannelBlocks());
    std::vector<cu::Event> compute_finished_events(NChannelBlocks());
    std::vector<cu::Event> output_copied_events(NChannelBlocks());

    /*
     * Start iterating
     */
    size_t iteration = 0;
    bool has_converged = false;
    bool has_previously_converged = false;
    bool constraints_satisfied = false;
    bool done = false;

    std::vector<double> step_magnitudes;
    step_magnitudes.reserve(GetMaxIterations());
    // std::cout << "DEBUG: Step magnitudes reserved for max iterations: "
    //           << GetMaxIterations() << std::endl;
    do {
      MakeSolutionsFinite1Pol(solutions);

      nvtxRangeId_t nvts_range_gpu = nvtxRangeStart("GPU");
        // std::cout << "Iteration: " << iteration << std::endl;

      for (size_t ch_block = 0; ch_block < NChannelBlocks(); ch_block++) {
        const ChannelBlockData<VisMatrix>& channel_block_data =
            data.ChannelBlock(ch_block);

        const int buffer_id = ch_block % 2;  // Use double buffering
        // Copy input data for first channel block
        // if (ch_block == 0) {
        if (ch_block > 1) {
          // std::cout << "DEBUG: Waiting for previous channel block to finish "
          //           << "before copying next channel block." << std::endl;
          // host_to_device_stream_->wait(output_copied_events[ch_block - 1]);
          // host_to_device_stream_->wait(input_copied_events[ch_block - 1]);
          host_to_device_stream_->wait(compute_finished_events[ch_block - 2]);
        }
        // std::cout << "Copying first channel block" << std::endl;
        CopyHostToHost(ch_block, iteration == 0, data, solutions[ch_block],
                        *host_to_device_stream_);

        CopyHostToDevice(ch_block, buffer_id, *host_to_device_stream_,
                          input_copied_events[ch_block], data);
          // std::cout << "Done copying first channel block" << std::endl;
        // }

        // As soon as input_copied_events[0] is triggered, the input data is
        // copied to the GPU and the host buffers could theoretically be reused.
        // However, since the size of these buffers may differ, every channel
        // block has its own set of host buffers anyway.
        // Before starting kernel execution for the current channel block (on a
        // different stream), the copy of data for the next channel block (if
        // any) is scheduled using a second set of GPU buffers.
        // if (ch_block < NChannelBlocks() - 1) {
        //   if (ch_block > 0) {
        //     std::cout << "DEBUG: Waiting for previous channel block to finish "
        //               << "before copying next channel block." << std::endl;
        //     host_to_device_stream_->wait(output_copied_events[ch_block - 1]);
        //     host_to_device_stream_->wait(input_copied_events[ch_block - 1]);
        //     host_to_device_stream_->wait(compute_finished_events[ch_block - 1]);
        //   }
        //   CopyHostToHost(ch_block + 1, iteration == 0, data,
        //                  solutions[ch_block + 1], *host_to_device_stream_);
        
        //   // Since the computation of channel block <n> and <n + 2> share the
        //   // same set of GPU buffers, wait for the compute_finished event to be
        //   // triggered before overwriting their contents.
        //   if (ch_block > 1) {
        //     std::cout << "DEBUG: Waiting for compute finished event for channel block "
        //               << (ch_block - 2) << std::endl;
        //     host_to_device_stream_->wait(compute_finished_events[ch_block - 1]);
        //     std::cout << "DEBUG: Copying host to device for channel block "
        //               << (ch_block + 1) << std::endl;
        //   }
        //   // std::cout << "DEBUG: Copying host to device for channel block "
        //   // << (ch_block + 1) << std::endl;
        //   CopyHostToDevice(ch_block + 1, buffer_id,
        //                    *host_to_device_stream_,
        //                    input_copied_events[ch_block + 1], data);
        // }

        // Wait for input of the current channel block to be copied
        // std::cout << "DEBUG: Waiting for input data to be copied for channel block "
        //           << ch_block << std::endl;
        execute_stream_->wait(input_copied_events[ch_block]);
        // std::cout << "DEBUG: Starting computation for channel block " << std::endl;
        // Wait for output buffer to be free
        if (ch_block > 1) {
          execute_stream_->wait(output_copied_events[ch_block - 2]);
        }
        // std::cout << "DEBUG: Copying input data for channel block "
        //           << ch_block << " to GPU buffers." << std::endl;

        // Start iteration (dtod copies and kernel execution only)
        // std::cout << "DEBUG: Starting iteration: " << iteration
        //           << " for channel block " << ch_block << std::endl;
        // cudaDeviceSynchronize();


        PerformIteration<VisMatrix>(
            phase_only, step_size, channel_block_data, *execute_stream_,
            NAntennas(), NSubSolutions(), NDirections(),
            gpu_buffers_.solution_map[buffer_id],
            gpu_buffers_.solutions[buffer_id],
            gpu_buffers_.next_solutions[buffer_id],
            gpu_buffers_.residual[buffer_id], gpu_buffers_.residual[2],
            gpu_buffers_.model[buffer_id],
            gpu_buffers_.antenna_pairs[buffer_id], *gpu_buffers_.numerator,
            *gpu_buffers_.denominator);

        // cudaDeviceSynchronize();
        // std::cout << "DEBUG: Finished iteration: " << iteration
        //           << " for channel block " << ch_block << " out of:  " <<
        //           NChannelBlocks() << std::endl;

        execute_stream_->record(compute_finished_events[ch_block]);
        // Wait for the computation to finish
        device_to_host_stream_->wait(compute_finished_events[ch_block]);

        // if (ch_block > 0) {
        //     std::cout << "DEBUG: Waiting for previous channel block to finish "
        //               << "before copying next channel block." << std::endl;
        //     device_to_host_stream_->wait(output_copied_events[ch_block - 1]);
        //     device_to_host_stream_->wait(input_copied_events[ch_block - 1]);
        //     device_to_host_stream_->wait(compute_finished_events[ch_block - 1]);
        //   }
        // Copy next solutions back to host
        device_to_host_stream_->memcpyDtoHAsync(
            &next_solutions(ch_block, 0, 0, 0),
            gpu_buffers_.next_solutions[buffer_id],
            SizeOfNextSolutions(1, NAntennas(), NSubSolutions(), NSolutionPolarizations()));
        // std::cout << "DEBUG: Copied next solutions back to host for channel " << ch_block << std::endl;

        // Record that the output is copied
        device_to_host_stream_->record(output_copied_events[ch_block]);
        // std::cout << "DEBUG: Output copied for channel block " << ch_block
        //           << std::endl;
      }  // end for ch_block

      // Wait for next solutions to be copied
      device_to_host_stream_->synchronize();

      nvtxRangeEnd(nvts_range_gpu);

      // CPU-only postprocessing
      nvtxRangeId_t nvtx_range_cpu = nvtxRangeStart("CPU");
      PostProcessing(iteration, time, has_previously_converged, has_converged,
                     constraints_satisfied, done, result, solutions,
                     next_solutions, step_magnitudes, stat_stream);
      // DumpSolutionsToFile(solutions, "solutions_dump_GPU.txt", iteration);
      // exit(0);  // Debugging exit point
      nvtxRangeEnd(nvtx_range_cpu);
    } while (!done);
exit(0);  // Debugging exit point
    // When we have not converged yet, we set the nr of iterations to the max+1,
    // so that non-converged iterations can be distinguished from converged
    // ones.
    if (has_converged && constraints_satisfied) {
      result.iterations = iteration;
    } else {
      result.iterations = iteration + 1;
    }

    if (!keep_buffers_) DeallocateHostBuffers();
    // exit(0);
    return result;
  } catch (const std::exception& e) {
    // Clean up partially allocated resources on error
    DeallocateHostBuffers();

    throw;
    // throw std::runtime_error(std::string("Error during Solve: ") + e.what());
  }
}


}  // namespace ddecal
}  // namespace dp3
template class dp3::ddecal::IterativeScalarSolverCuda<std::complex<float>>;
