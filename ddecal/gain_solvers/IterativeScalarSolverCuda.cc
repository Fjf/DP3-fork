// Copyright (C) 2023 ASTRON (Netherlands Institute for Radio Astronomy)
// SPDX-License-Identifier: GPL-3.0-or-later

#include "IterativeScalarSolverCuda.h"

#include <algorithm>
#include <iostream>
#include <vector>
#include <numeric>

#include <cuda_runtime.h>
#include <nvToolsExt.h>

#include <aocommon/matrix2x2.h>
#include <aocommon/matrix2x2diag.h>

#include "kernels/IterativeScalar.h"
#include "kernels/IterativeDiagonal.h"

using aocommon::MC2x2;
using aocommon::MC2x2F;

namespace {

inline float Norm(std::complex<float> value) { return std::norm(value); }
inline double Norm(std::complex<double> value) { return std::norm(value); }

#include <type_traits>

/// Helper for scalar types
template<typename T>
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
  std::cout << name << " max: " << *minmax.second << "\n" <<  std::endl;
}

// Helper for matrix types (prints norm summary)
template<typename T>
typename std::enable_if<std::is_class<T>::value, void>::type
PrintVectorSummary(const std::vector<T>& vec, const std::string& name) {
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
void PrintSolutionTensorSummary(const xt::xtensor<std::complex<double>, 4>& next_solutions,
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
  if constexpr (std::is_same_v<VisMatrix, std::complex<float>>) {
    return n_directions * n_visibilities * sizeof(std::complex<float>);
  } else if constexpr (std::is_same_v<VisMatrix, std::complex<double>>) {
    return n_directions * n_visibilities * sizeof(std::complex<double>);
  } else if constexpr (std::is_same_v<VisMatrix, aocommon::MC2x2F>) {
    return n_directions * n_visibilities * sizeof(aocommon::MC2x2F);
  } else if constexpr (std::is_same_v<VisMatrix, aocommon::MC2x2FDiag>) {
    return n_directions * n_visibilities * sizeof(aocommon::MC2x2FDiag);
  } else {
    // For other matrix types, the CUDA kernels expect cuM2x2FloatComplex
    return n_directions * n_visibilities * sizeof(aocommon::MC2x2F);
  }
}

// template <typename VisMatrix>
// size_t SizeOfResidual(size_t n_visibilities) {
//   if constexpr (std::is_same_v<VisMatrix, std::complex<float>>) {
//     std::cout << "SizeOfResidual: using std::complex<float>" << std::endl;
//     return n_visibilities * sizeof(std::complex<float>);
//   } else if constexpr (std::is_same_v<VisMatrix, std::complex<double>>) {
//     std::cout << "SizeOfResidual: using std::complex<double>" << std::endl;
//     return n_visibilities * sizeof(std::complex<double>);
//   } else if constexpr (std::is_same_v<VisMatrix, aocommon::MC2x2F>) {
//     std::cout << "SizeOfResidual: using aocommon::MC2x2F" << std::endl;
//     return n_visibilities * sizeof(aocommon::MC2x2F);
//   } else if constexpr (std::is_same_v<VisMatrix, aocommon::MC2x2FDiag>) {
//     std::cout << "SizeOfResidual: using aocommon::MC2x2FDiag" << std::endl;
//     return n_visibilities * sizeof(aocommon::MC2x2FDiag);
//   } else {
//     std::cout << "SizeOfResidual: using aocommon::MC2x2F" << std::endl;
//     // For other matrix types, the CUDA kernels expect cuM2x2FloatComplex
//     return n_visibilities * sizeof(aocommon::MC2x2F);
//   }
// }

template <typename VisMatrix>
size_t SizeOfResidual(size_t n_visibilities) {
  return n_visibilities * sizeof(std::complex<float>);
}

size_t SizeOfSolutions(size_t n_antennas, size_t n_subsol, size_t n_pol) {
  return n_antennas * n_subsol * n_pol * sizeof(std::complex<double>);
}

size_t SizeOfAntennaPairs(size_t n_visibilities) {
  return n_visibilities * 2 * sizeof(uint32_t);
}

size_t SizeOfSolutionMap(size_t n_directions, size_t n_visibilities) {
  return n_directions * n_visibilities * sizeof(uint32_t);
}

size_t SizeOfNextSolutions(size_t n_antennas, size_t n_subsol, size_t n_pol) {
  return n_antennas * n_subsol * n_pol * sizeof(std::complex<double>);
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
  stream.synchronize();

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
  std::vector<VisMatrix> residual_data(n_visibilities);
  std::copy(&channel_block_data.Visibility(0), &channel_block_data.Visibility(0) + n_visibilities, residual_data.begin());
  
  // print residual for debugging
  PrintVectorSummary(residual_data, "residual_pre_kernel");
  
  // Copy to GPU device
  stream.memcpyHtoDAsync(
      device_residual, residual_data.data(),
      SizeOfResidual<VisMatrix>(n_visibilities));
  // Synchronize the streams to ensure the residual is ready
  stream.synchronize();
  
  // Subtract all directions with their current solutions
  // In-place: residual -> residual
  LaunchScalarSubtractKernel(stream, n_directions, n_visibilities, n_solutions,
                       device_antenna_pairs, device_solution_map,
                       device_solutions, device_model, device_residual);

  // Copy result back from GPU and print for debugging
  stream.memcpyDtoHAsync(residual_data.data(), device_residual,
                         SizeOfResidual<VisMatrix>(n_visibilities));
  stream.synchronize();
  PrintVectorSummary(residual_data, "v_residual_post_kernel");
  

  // Copy result back from GPU and print for debugging
  stream.memcpyDtoHAsync(residual_data.data(), device_residual,
                         SizeOfResidual<VisMatrix>(n_visibilities));
  stream.synchronize();
  PrintVectorSummary(residual_data, "v_residual_post_kernel");
  // exit(0);

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
    std::cout << "channel_block_data info:" << std::endl;
    std::cout << "  NVisibilities: " << channel_block_data.NVisibilities() << std::endl;
    std::cout << "  NDirections: " << channel_block_data.NDirections() << std::endl;
    std::cout << "  NSolutionsForDirection(" << direction << "): " 
              << channel_block_data.NSolutionsForDirection(direction) << std::endl;
    std::cout << "  NSubSolutions (total): " << channel_block_data.NSubSolutions() << std::endl;
    
    // Print some antenna pair information for first few visibilities
    const size_t max_vis_to_show = std::min(static_cast<size_t>(5), channel_block_data.NVisibilities());
    std::cout << "  First " << max_vis_to_show << " antenna pairs:" << std::endl;
    for (size_t i = 0; i < max_vis_to_show; ++i) {
      std::cout << "    vis[" << i << "]: ant1=" << channel_block_data.Antenna1Index(i) 
                << ", ant2=" << channel_block_data.Antenna2Index(i) 
                << ", sol_idx=" << channel_block_data.SolutionIndex(direction, i) << std::endl;
    }
    
    // Print some model visibility norms for this direction
    std::vector<double> model_norms;
    for (size_t i = 0; i < max_vis_to_show; ++i) {
      const auto& model_vis = channel_block_data.ModelVisibility(direction, i);
      model_norms.push_back(Norm(model_vis));
    }
    if (!model_norms.empty()) {
      std::cout << "  First " << max_vis_to_show << " model visibility norms for direction " << direction << ":" << std::endl;
      for (size_t i = 0; i < model_norms.size(); ++i) {
        std::cout << "    model_vis[" << i << "] norm: " << model_norms[i] << std::endl;
      }
    }

    // Count and show actual model values for comparison with CPU version
    size_t total_model_values = 0;
    size_t non_zero_model_values = 0;
    std::cout << "  CPU model values for direction " << direction << " (first " << max_vis_to_show << "):" << std::endl;
    for (size_t i = 0; i < max_vis_to_show; ++i) {
      const auto& model_vis = channel_block_data.ModelVisibility(direction, i);
      std::cout << "    CPU model_vis[" << i << "]: " << model_vis << std::endl;
    }
    
    // Count all non-zero model values like the CPU version does
    for (size_t i = 0; i < n_visibilities; ++i) {
      const auto& model_vis = channel_block_data.ModelVisibility(direction, i);
      total_model_values++;
      if (Norm(model_vis) > 1e-12) {
        non_zero_model_values++;
      }
    }
    std::cout << "  Count non-zero model values: " << non_zero_model_values 
              << " out of " << total_model_values << " total" << std::endl;

    // Verify GPU model data by copying back from device
    // std::cout << "DEBUG: Verifying GPU model data for direction " << direction << std::endl;
    const size_t model_sample_size = std::min(static_cast<size_t>(10), n_visibilities);
    size_t gpu_non_zero_count = 0;
    if constexpr (std::is_same_v<VisMatrix, std::complex<float>>) {
      std::vector<std::complex<float>> gpu_model_full(n_directions * n_visibilities);
      stream.memcpyDtoHAsync(gpu_model_full.data(), device_model,
                           n_directions * n_visibilities * sizeof(std::complex<float>));
      stream.synchronize();
      std::cout << "  GPU model values for direction " << direction << " (first " << model_sample_size << "):" << std::endl;
      const size_t offset = direction * n_visibilities;
      for (size_t i = 0; i < model_sample_size; ++i) {
        std::cout << "    GPU model_vis[" << i << "]: " << gpu_model_full[offset + i] << std::endl;
      }
      
      // Count all non-zero values in GPU data
      for (size_t i = 0; i < n_visibilities; ++i) {
        if (std::abs(gpu_model_full[offset + i]) > 1e-12) {
          gpu_non_zero_count++;
        }
      }
    } else if constexpr (std::is_same_v<VisMatrix, std::complex<double>>) {
      std::vector<std::complex<double>> gpu_model_full(n_directions * n_visibilities);
      stream.memcpyDtoHAsync(gpu_model_full.data(), device_model,
                           n_directions * n_visibilities * sizeof(std::complex<double>));
      stream.synchronize();
      std::cout << "  GPU model values for direction " << direction << " (first " << model_sample_size << "):" << std::endl;
      const size_t offset = direction * n_visibilities;
      for (size_t i = 0; i < model_sample_size; ++i) {
        std::cout << "    GPU model_vis[" << i << "]: " << gpu_model_full[offset + i] << std::endl;
      }
      
      // Count all non-zero values in GPU data
      for (size_t i = 0; i < n_visibilities; ++i) {
        if (std::abs(gpu_model_full[offset + i]) > 1e-12) {
          gpu_non_zero_count++;
        }
      }
    }
    std::cout << "  GPU non-zero model values: " << gpu_non_zero_count << " out of " << n_visibilities << std::endl;

    stream.memcpyDtoHAsync(residual_data.data(), device_residual_temp,
                         SizeOfResidual<VisMatrix>(n_visibilities));
    stream.synchronize();
    PrintVectorSummary(residual_data, "v_residual_post_kernel_2");
    // exit(0);
    
    // Move device_solutions to host (CPU) for analysis - use separate buffers
    std::vector<std::complex<double>> current_solutions_data(n_antennas * n_solutions);
    std::vector<std::complex<double>> next_solutions_data(n_antennas * n_solutions);
    
    stream.memcpyDtoHAsync(
        current_solutions_data.data(),
        device_solutions,
        SizeOfSolutions(n_antennas, n_solutions, 1)); // For scalar solver, polarizations = 1
    stream.memcpyDtoHAsync(
        next_solutions_data.data(),
        device_next_solutions,
        SizeOfSolutions(n_antennas, n_solutions, 1)); // For scalar solver, polarizations = 1
    stream.synchronize();
    
    PrintVectorSummary(current_solutions_data, "solutions_post_solve_direction");
    PrintVectorSummary(next_solutions_data, "next_solutions_post_solve_direction");
  
    // exit(0);
    
    // if (direction == 0) exit(0);  // Exit after first direction only
  }





  std::cout << "DEBUG: Iteration completed for channel block with "
            << n_directions << " directions." << std::endl;

  LaunchStepKernel(stream, n_visibilities, device_solutions,
                   device_next_solutions, phase_only, step_size);
  stream.synchronize();
  std::cout << "DEBUG: Step kernel launched." << std::endl;
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
  // Add type debugging at the start
  std::cout << "DEBUG: VisMatrix type info:" << std::endl
            << "  typeid name: " << typeid(VisMatrix).name() << std::endl
            << "  is_same<complex<float>>: " << std::is_same_v<VisMatrix, std::complex<float>> << std::endl
            << "  is_same<complex<double>>: " << std::is_same_v<VisMatrix, std::complex<double>> << std::endl
            << "  is_same<MC2x2F>: " << std::is_same_v<VisMatrix, aocommon::MC2x2F> << std::endl
            << "  is_same<MC2x2FDiag>: " << std::is_same_v<VisMatrix, aocommon::MC2x2FDiag> << std::endl;
            
  size_t max_n_direction_solutions = 0;
  size_t max_n_visibilities = 0;
  size_t max_n_directions = 0;
  std::tie(max_n_direction_solutions, max_n_visibilities, max_n_directions) =
      ComputeArrayDimensions(data);

  // std::cout << "DEBUG: GPU Buffer dimensions:" << std::endl
  //           << "  max_n_direction_solutions: " << max_n_direction_solutions << std::endl
  //           << "  max_n_visibilities: " << max_n_visibilities << std::endl
  //           << "  max_n_directions: " << max_n_directions << std::endl
  //           << "  NAntennas: " << NAntennas() << std::endl
  //           << "  NSolutionPolarizations: " << NSolutionPolarizations() << std::endl;

  // std::cout << "DEBUG: Allocating GPU buffers with sizes:" << std::endl;
  // std::cout << "  numerator: " << SizeOfNumerator(NAntennas(), max_n_direction_solutions) << " bytes" << std::endl;
  // std::cout << "  denominator: " << SizeOfDenominator(NAntennas(), max_n_direction_solutions) << " bytes" << std::endl;
  // std::cout << "  antenna_pairs: " << SizeOfAntennaPairs(max_n_visibilities) << " bytes" << std::endl;
  // std::cout << "  solution_map: " << SizeOfSolutionMap(max_n_directions, max_n_visibilities) << " bytes" << std::endl;
  // std::cout << "  solutions: " << SizeOfSolutions(NAntennas(), NSubSolutions(), NSolutionPolarizations()) << " bytes" << std::endl;
  // std::cout << "  next_solutions: " << SizeOfNextSolutions(NAntennas(), NSubSolutions(), NSolutionPolarizations()) << " bytes" << std::endl;
  // std::cout << "  model: " << SizeOfModel<VisMatrix>(max_n_directions, max_n_visibilities) << " bytes" << std::endl;
  // std::cout << "  residual: " << SizeOfResidual<VisMatrix>(max_n_visibilities) << " bytes" << std::endl;

  try {
    size_t free_mem = 0, total_mem = 0;
    cudaMemGetInfo(&free_mem, &total_mem);
    // std::cout << "DEBUG: GPU memory before allocation:" << std::endl
    //           << "  Free: " << free_mem << " bytes" << std::endl
    //           << "  Total: " << total_mem << " bytes" << std::endl;

    // std::cout << "DEBUG: Allocating numerator" << std::endl;
    try {
      // For scalar solutions, we need 2 complex values per antenna per direction solution
      const size_t numerator_size = NAntennas() * max_n_direction_solutions * 2 * sizeof(std::complex<float>);
      // std::cout << "DEBUG: Numerator size: " << numerator_size << " bytes" << std::endl;
      gpu_buffers_.numerator = std::make_unique<cu::DeviceMemory>(numerator_size);
      // std::cout << "DEBUG: numerator allocated successfully" << std::endl;
    } catch (const std::exception& e) {
      std::cerr << "ERROR allocating numerator: " << e.what() << std::endl;
      throw;
    }
        
    // std::cout << "DEBUG: Allocating denominator" << std::endl;
    try {
      // For scalar solutions, we need 2 float values per antenna per direction solution
      const size_t denominator_size = NAntennas() * max_n_direction_solutions * 2 * sizeof(float);
      // std::cout << "DEBUG: Denominator size: " << denominator_size << " bytes" << std::endl;
      gpu_buffers_.denominator = std::make_unique<cu::DeviceMemory>(denominator_size);
      // std::cout << "DEBUG: denominator allocated successfully" << std::endl;
    } catch (const std::exception& e) {
      std::cerr << "ERROR allocating denominator: " << e.what() << std::endl;
      throw;
    }

    // Allocating two buffers allows double buffering.
    for (size_t i = 0; i < 2; i++) {
      // std::cout << "DEBUG: Allocating double buffer set " << i << std::endl;
      
      // std::cout << "DEBUG: Allocating antenna_pairs[" << i << "]" << std::endl;
      try {
        gpu_buffers_.antenna_pairs.emplace_back(
            SizeOfAntennaPairs(max_n_visibilities));
        // std::cout << "DEBUG: antenna_pairs[" << i << "] allocated successfully" << std::endl;
      } catch (const std::exception& e) {
        std::cerr << "ERROR allocating antenna_pairs[" << i << "]: " << e.what() << std::endl;
        throw;
      }
          
      // std::cout << "DEBUG: Allocating solution_map[" << i << "]" << std::endl;
      try {
        gpu_buffers_.solution_map.emplace_back(
            SizeOfSolutionMap(max_n_directions, max_n_visibilities));
        // std::cout << "DEBUG: solution_map[" << i << "] allocated successfully" << std::endl;
      } catch (const std::exception& e) {
        std::cerr << "ERROR allocating solution_map[" << i << "]: " << e.what() << std::endl;
        throw;
      }
          
      // std::cout << "DEBUG: Allocating solutions[" << i << "]" << std::endl;
      try {
        // For scalar solutions, we need one complex value per antenna per solution
        const size_t solutions_size = NAntennas() * NSubSolutions() * sizeof(std::complex<double>);
        // std::cout << "DEBUG: Solutions size: " << solutions_size << " bytes" << std::endl;
        gpu_buffers_.solutions.emplace_back(solutions_size);
        // std::cout << "DEBUG: solutions[" << i << "] allocated successfully" << std::endl;
      } catch (const std::exception& e) {
        std::cerr << "ERROR allocating solutions[" << i << "]: " << e.what() << std::endl;
        throw;
      }
          
      // std::cout << "DEBUG: Allocating next_solutions[" << i << "]" << std::endl;
      try {
        // For scalar solutions, we need one complex value per antenna per solution
        const size_t next_solutions_size = NAntennas() * NSubSolutions() * sizeof(std::complex<double>);
        // std::cout << "DEBUG: Next solutions size: " << next_solutions_size << " bytes" << std::endl;
        gpu_buffers_.next_solutions.emplace_back(next_solutions_size);
        // std::cout << "DEBUG: next_solutions[" << i << "] allocated successfully" << std::endl;
      } catch (const std::exception& e) {
        std::cerr << "ERROR allocating next_solutions[" << i << "]: " << e.what() << std::endl;
        throw;
      }
          
      // std::cout << "DEBUG: Allocating model[" << i << "]" << std::endl;
      try {
        gpu_buffers_.model.emplace_back(
            SizeOfModel<VisMatrix>(max_n_directions, max_n_visibilities));
        // std::cout << "DEBUG: model[" << i << "] allocated successfully" << std::endl;
      } catch (const std::exception& e) {
        std::cerr << "ERROR allocating model[" << i << "]: " << e.what() << std::endl;
        throw;
      }
    }

    // We need two buffers for residual like above to facilitate double-buffering,
    // the third buffer is used for the per-direction add/subtract.
    for (size_t i = 0; i < 3; i++) {
      // std::cout << "DEBUG: Allocating residual[" << i << "]" << std::endl;
      try {
        gpu_buffers_.residual.emplace_back(
            SizeOfResidual<VisMatrix>(max_n_visibilities));
        // std::cout << "DEBUG: residual[" << i << "] allocated successfully" << std::endl;
      } catch (const std::exception& e) {
        std::cerr << "ERROR allocating residual[" << i << "]: " << e.what() << std::endl;
        throw;
      }
    }

    cudaMemGetInfo(&free_mem, &total_mem);
    // std::cout << "DEBUG: GPU memory after allocation:" << std::endl
    //           << "  Free: " << free_mem << " bytes" << std::endl
    //           << "  Total: " << total_mem << " bytes" << std::endl;

    // std::cout << "DEBUG: Finished allocating all GPU buffers" << std::endl;
  } catch (const std::exception& e) {
    std::cerr << "ERROR during GPU buffer allocation: " << e.what() << std::endl;
    throw;
  }
}

template <typename VisMatrix>
void IterativeScalarSolverCuda<VisMatrix>::AllocateHostBuffers(
    const SolveData<VisMatrix>& data) {
  // For scalar solver, we need one solution per antenna per polarization
  const size_t total_solutions = NAntennas() * NSubSolutions() * NSolutionPolarizations();
  // std::cout << "DEBUG: Allocating host_buffers_.next_solutions" << std::endl;
  // std::cout << "DEBUG: Total solutions: " << total_solutions << std::endl;
  const size_t next_solutions_size = SizeOfNextSolutions(NAntennas(), NSubSolutions(), NSolutionPolarizations());
  // std::cout << "DEBUG: Next solutions buffer size: " << next_solutions_size << " bytes" << std::endl;
  
  try {
    host_buffers_.next_solutions = std::make_unique<cu::HostMemory>(next_solutions_size);
    // std::cout << "DEBUG: host_buffers_.next_solutions allocated successfully" << std::endl;
    // std::cout << "DEBUG: Allocated pointer: " << std::hex << static_cast<void*>(*host_buffers_.next_solutions) << std::dec << std::endl;
  } catch (const std::exception& e) {
    std::cerr << "ERROR allocating host_buffers_.next_solutions: " << e.what() << std::endl;
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
        SizeOfSolutions(NAntennas(), NSubSolutions(), NSolutionPolarizations()));
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
  
  // Debug: Print some model data after host-to-host copy
  std::cout << "DEBUG: Host-to-host model copy for ch_block " << ch_block << std::endl;
  std::cout << "  Model size: " << SizeOfModel<VisMatrix>(n_directions, n_visibilities) << " bytes" << std::endl;
  std::cout << "  n_directions: " << n_directions << ", n_visibilities: " << n_visibilities << std::endl;
  
  // Sample some model visibility values from the source - check more values
  std::cout << "  Checking model data source:" << std::endl;
  size_t non_zero_count = 0;
  for (size_t dir = 0; dir < n_directions; ++dir) {
    for (size_t vis = 0; vis < std::min(n_visibilities, static_cast<size_t>(50)); ++vis) {
      const auto& model_vis = channel_block_data.ModelVisibility(dir, vis);
      // Use appropriate norm function for both scalar and matrix types
      float vis_norm;
      if constexpr (std::is_same_v<VisMatrix, std::complex<float>>) {
        vis_norm = std::abs(model_vis);
      } else {
        vis_norm = Norm(model_vis);
      }
      if (vis_norm > 1e-10) {
        non_zero_count++;
        if (non_zero_count <= 10) { // Print first 10 non-zero values
          std::cout << "    Source model[" << dir << "][" << vis << "]: " << model_vis 
                    << " (norm=" << vis_norm << ")" << std::endl;
        }
      }
    }
  }
  std::cout << "  Total non-zero model values found in first 50: " << non_zero_count << std::endl;
  
  // Always print first few values regardless
  for (size_t dir = 0; dir < std::min(n_directions, static_cast<size_t>(1)); ++dir) {
    for (size_t vis = 0; vis < std::min(n_visibilities, static_cast<size_t>(5)); ++vis) {
      const auto& model_vis = channel_block_data.ModelVisibility(dir, vis);
      std::cout << "    First model[" << dir << "][" << vis << "]: " << model_vis << std::endl;
    }
  }
  
  // Also check the specific visibility index 811 that's causing issues
  const size_t debug_vis_idx = 811;
  if (debug_vis_idx < n_visibilities) {
    for (size_t dir = 0; dir < n_directions; ++dir) {
      const auto& model_vis = channel_block_data.ModelVisibility(dir, debug_vis_idx);
      std::cout << "    DEBUG vis 811 model[" << dir << "][" << debug_vis_idx << "]: " << model_vis << std::endl;
    }
  }
  
  // Wait for the copy to complete then verify copied data
  stream.synchronize();
  
  // Verify the host buffer contains the same data
  if constexpr (std::is_same_v<VisMatrix, std::complex<float>>) {
    std::complex<float>* host_model_ptr = static_cast<std::complex<float>*>(host_model);
    for (size_t dir = 0; dir < std::min(n_directions, static_cast<size_t>(2)); ++dir) {
      for (size_t vis = 0; vis < std::min(n_visibilities, static_cast<size_t>(3)); ++vis) {
        size_t idx = dir * n_visibilities + vis;
        std::cout << "  Host buffer model[" << dir << "][" << vis << "]: " << host_model_ptr[idx] << std::endl;
      }
    }
    
    // Check the specific debug visibility index 811
    const size_t debug_vis_idx = 811;
    if (debug_vis_idx < n_visibilities) {
      for (size_t dir = 0; dir < n_directions; ++dir) {
        size_t idx = dir * n_visibilities + debug_vis_idx;
        std::cout << "  Host buffer model[" << dir << "][" << debug_vis_idx << "]: " << host_model_ptr[idx] << std::endl;
      }
    }
  } else if constexpr (std::is_same_v<VisMatrix, std::complex<double>>) {
    std::complex<double>* host_model_ptr = static_cast<std::complex<double>*>(host_model);
    for (size_t dir = 0; dir < std::min(n_directions, static_cast<size_t>(2)); ++dir) {
      for (size_t vis = 0; vis < std::min(n_visibilities, static_cast<size_t>(3)); ++vis) {
        size_t idx = dir * n_visibilities + vis;
        std::cout << "  Host buffer model[" << dir << "][" << vis << "]: " << host_model_ptr[idx] << std::endl;
      }
    }
    
    // Check the specific debug visibility index 811
    const size_t debug_vis_idx = 811;
    if (debug_vis_idx < n_visibilities) {
      for (size_t dir = 0; dir < n_directions; ++dir) {
        size_t idx = dir * n_visibilities + debug_vis_idx;
        std::cout << "  Host buffer model[" << dir << "][" << debug_vis_idx << "]: " << host_model_ptr[idx] << std::endl;
      }
    }
  }
  
  stream.memcpyHtoHAsync(host_solutions, solutions.data(),
                         SizeOfSolutions(NAntennas(), NSubSolutions(), NSolutionPolarizations()));
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
  cu::DeviceMemory& device_antenna_pairs =
      gpu_buffers_.antenna_pairs[buffer_id];
  cu::DeviceMemory& device_model = gpu_buffers_.model[buffer_id];
  cu::DeviceMemory& device_residual = gpu_buffers_.residual[buffer_id];
  cu::DeviceMemory& device_solutions = gpu_buffers_.solutions[buffer_id];

  stream.memcpyHtoDAsync(device_solution_map, host_solution_map,
                         SizeOfSolutionMap(n_directions, n_visibilities));
  
  // // Debug: Verify host buffer before copying to device
  // std::cout << "DEBUG: Host-to-device model copy for ch_block " << ch_block << ", buffer_id " << buffer_id << std::endl;
  // std::cout << "  Copying " << SizeOfModel<VisMatrix>(n_directions, n_visibilities) << " bytes from host to device" << std::endl;
  // std::cout << "  Host model buffer address: " << std::hex << static_cast<void*>(host_model) << std::dec << std::endl;
  // std::cout << "  Device model buffer address: " << std::hex << static_cast<void*>(device_model) << std::dec << std::endl;
  
  // // Verify the host buffer still contains correct data before copying to device
  // if constexpr (std::is_same_v<VisMatrix, std::complex<float>>) {
  //   std::complex<float>* host_model_ptr = static_cast<std::complex<float>*>(host_model);
  //   for (size_t dir = 0; dir < std::min(n_directions, static_cast<size_t>(2)); ++dir) {
  //     for (size_t vis = 0; vis < std::min(n_visibilities, static_cast<size_t>(3)); ++vis) {
  //       size_t idx = dir * n_visibilities + vis;
  //       std::cout << "  Pre-copy host buffer model[" << dir << "][" << vis << "]: " << host_model_ptr[idx] << std::endl;
  //     }
  //   }
    
  //   // Check the specific debug visibility index 811
  //   const size_t debug_vis_idx = 811;
  //   if (debug_vis_idx < n_visibilities) {
  //     for (size_t dir = 0; dir < n_directions; ++dir) {
  //       size_t idx = dir * n_visibilities + debug_vis_idx;
  //       std::cout << "  Pre-copy host buffer model[" << dir << "][" << debug_vis_idx << "]: " << host_model_ptr[idx] << std::endl;
  //     }
  //   }
  // } else if constexpr (std::is_same_v<VisMatrix, std::complex<double>>) {
  //   std::complex<double>* host_model_ptr = static_cast<std::complex<double>*>(host_model);
  //   for (size_t dir = 0; dir < std::min(n_directions, static_cast<size_t>(2)); ++dir) {
  //     for (size_t vis = 0; vis < std::min(n_visibilities, static_cast<size_t>(3)); ++vis) {
  //       size_t idx = dir * n_visibilities + vis;
  //       std::cout << "  Pre-copy host buffer model[" << dir << "][" << vis << "]: " << host_model_ptr[idx] << std::endl;
  //     }
  //   }
    
  //   // Check the specific debug visibility index 811
  //   const size_t debug_vis_idx = 811;
  //   if (debug_vis_idx < n_visibilities) {
  //     for (size_t dir = 0; dir < n_directions; ++dir) {
  //       size_t idx = dir * n_visibilities + debug_vis_idx;
  //       std::cout << "  Pre-copy host buffer model[" << dir << "][" << debug_vis_idx << "]: " << host_model_ptr[idx] << std::endl;
  //     }
  //   }
  // }
  
  stream.memcpyHtoDAsync(device_model, host_model,
                         SizeOfModel<VisMatrix>(n_directions, n_visibilities));
  
  // Immediately verify what got copied to GPU
  stream.synchronize();

  stream.memcpyHtoDAsync(device_residual, host_residual,
                         SizeOfResidual<VisMatrix>(n_visibilities));
  stream.memcpyHtoDAsync(device_antenna_pairs, host_antenna_pairs,
                         SizeOfAntennaPairs(n_visibilities));
  stream.memcpyHtoDAsync(device_solutions, host_solutions,
                         SizeOfSolutions(NAntennas(), NSubSolutions(), NSolutionPolarizations()));

  stream.record(event);
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
  // std::cout << "DEBUG: Entering Solve" << std::endl;
  PrepareConstraints();
  context_->setCurrent();

  const bool phase_only = GetPhaseOnly();
  const double step_size = GetStepSize();

  SolveResult result;

  /*
   * Allocate buffers
   */
  // std::cout << "DEBUG: Allocating buffers. Host buffers initialized: " 
  //           << host_buffers_initialized_ << ", GPU buffers initialized: " 
  //           << gpu_buffers_initialized_ << std::endl;
            
  if (!host_buffers_initialized_) {
    // std::cout << "DEBUG: About to allocate host buffers" << std::endl;
    AllocateHostBuffers(data);
    host_buffers_initialized_ = true;
  }
  if (!gpu_buffers_initialized_) {
    // std::cout << "DEBUG: About to allocate GPU buffers" << std::endl;
    AllocateGPUBuffers(data);
    gpu_buffers_initialized_ = true;
  }

  // std::cout << "DEBUG: Creating solution span array" << std::endl;
  const std::array<size_t, 4> next_solutions_shape = {
      NChannelBlocks(), NAntennas(), NSubSolutions(), NSolutionPolarizations()};
  // std::cout << "DEBUG: Solution span shape:" << std::endl
  //           << "  NChannelBlocks: " << NChannelBlocks() << std::endl
  //           << "  NAntennas: " << NAntennas() << std::endl
  //           << "  NSubSolutions: " << NSubSolutions() << std::endl
  //           << "  NSolutionPolarizations: " << NSolutionPolarizations() << std::endl;

  // Detailed memory debug info
  // std::cout << "DEBUG: Memory allocation info:" << std::endl;
  std::cout << "  host_buffers_.next_solutions valid: " 
            << (host_buffers_.next_solutions != nullptr) << std::endl;
  std::cout << "  host_buffers_.next_solutions base: " 
            << (void*)host_buffers_.next_solutions.get() << std::endl;

  // std::cout << "DEBUG: Getting next_solutions pointer" << std::endl;
  // std::cout << "DEBUG: host_buffers_.next_solutions pointer valid: " 
            // << (host_buffers_.next_solutions != nullptr) << std::endl;
  // std::cout << "DEBUG: next_solutions_ptr pointer: " 
            // << std::hex << static_cast<void*>(*host_buffers_.next_solutions) << std::dec << std::endl;
            
  void* raw_ptr = *host_buffers_.next_solutions;
  // std::cout << "DEBUG: Raw dereferenced pointer: " << std::hex << raw_ptr << std::dec << std::endl;
  
  std::complex<double>* next_solutions_ptr = static_cast<std::complex<double>*>(raw_ptr);
  // std::cout << "DEBUG: Cast pointer value: " << std::hex << (void*)next_solutions_ptr << std::dec << std::endl;
  
  if (!next_solutions_ptr) {
    std::cerr << "ERROR: next_solutions_ptr is null" << std::endl;
    throw std::runtime_error("next_solutions_ptr is null");
  }
  // std::cout << "DEBUG: Creating solution span" << std::endl;
  SolutionSpan next_solutions =
      aocommon::xt::CreateSpan(next_solutions_ptr, next_solutions_shape);
  // std::cout << "DEBUG: Solution span created successfully" << std::endl;

  /*
   * Allocate events for each channel block
   */
  std::vector<cu::Event> input_copied_events(NChannelBlocks());
  // std::cout << "DEBUG: Solution span created successfully" << std::endl;
  std::vector<cu::Event> compute_finished_events(NChannelBlocks());
  // std::cout << "DEBUG: Creating compute_finished_events" << std::endl;
  std::vector<cu::Event> output_copied_events(NChannelBlocks());
  // std::cout << "DEBUG: Creating output_copied_events" << std::endl;

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
    // std::cout << "DEBUG: Solutions made finite for iteration " << iteration << std::endl;

    nvtxRangeId_t nvts_range_gpu = nvtxRangeStart("GPU");

    for (size_t ch_block = 0; ch_block < NChannelBlocks(); ch_block++) {
      // std::cout << "\nDEBUG: Processing channel block " << ch_block << std::endl;
      const ChannelBlockData<VisMatrix>& channel_block_data =
          data.ChannelBlock(ch_block);
            
      // std::cout << "DEBUG: Channel block dimensions:" << std::endl
      //           << "  NAntennas: " << NAntennas() << std::endl
      //           << "  NDirections: " << channel_block_data.NDirections() << std::endl
      //           << "  NVisibilities: " << channel_block_data.NVisibilities() << std::endl
      //           << "  NSolutions: " << NSubSolutions() << std::endl
      //           << "  NSolutionPolarizations: " << NSolutionPolarizations() << std::endl;
                  
      const int buffer_id = ch_block % 2;
      // std::cout << "DEBUG: Using buffer_id: " << buffer_id << std::endl;
      // Copy input data for first channel block
      if (ch_block == 0) {
        // std::cout << "DEBUG: Copying host to host for channel block " << ch_block << std::endl;
        CopyHostToHost(ch_block, iteration == 0, data, solutions[ch_block],
                       *host_to_device_stream_);

        CopyHostToDevice(ch_block, buffer_id, *host_to_device_stream_,
                         input_copied_events[0], data);
      }

      // As soon as input_copied_events[0] is triggered, the input data is
      // copied to the GPU and the host buffers could theoretically be reused.
      // However, since the size of these buffers may differ, every channel
      // block has its own set of host buffers anyway.
      // Before starting kernel execution for the current channel block (on a
      // different stream), the copy of data for the next channel block (if any)
      // is scheduled using a second set of GPU buffers.
      if (ch_block < NChannelBlocks() - 1) {
        CopyHostToHost(ch_block + 1, iteration == 0, data,
                       solutions[ch_block + 1], *host_to_device_stream_);

        // Since the computation of channel block <n> and <n + 2> share the same
        // set of GPU buffers, wait for the compute_finished event to be
        // triggered before overwriting their contents.
        if (ch_block > 1) {
          host_to_device_stream_->wait(compute_finished_events[ch_block - 2]);
        }
        // std::cout << "DEBUG: Copying host to device for channel block "
                  // << (ch_block + 1) << std::endl;
        CopyHostToDevice(ch_block + 1, (ch_block + 1) % 2,
                         *host_to_device_stream_,
                         input_copied_events[ch_block + 1], data);
      }

      // Wait for input of the current channel block to be copied
      execute_stream_->wait(input_copied_events[ch_block]);

      // Wait for output buffer to be free
      if (ch_block > 1) {
        execute_stream_->wait(output_copied_events[ch_block - 2]);
      }

      // Start iteration (dtod copies and kernel execution only)
      // std::cout << "DEBUG: Starting iteration: " << iteration
      //           << " for channel block " << ch_block << std::endl;
      PerformIteration<VisMatrix>(
          phase_only, step_size, channel_block_data, *execute_stream_,
          NAntennas(), NSubSolutions(), NDirections(),
          gpu_buffers_.solution_map[buffer_id],
          gpu_buffers_.solutions[buffer_id],
          gpu_buffers_.next_solutions[buffer_id],
          gpu_buffers_.residual[buffer_id], gpu_buffers_.residual[2],
          gpu_buffers_.model[buffer_id], gpu_buffers_.antenna_pairs[buffer_id],
          *gpu_buffers_.numerator, *gpu_buffers_.denominator);
      // std::cout << "DEBUG: Finished iteration: " << iteration
      //           << " for channel block " << ch_block << " out of:  " << NChannelBlocks() << std::endl;
      execute_stream_->record(compute_finished_events[ch_block]);

      // Wait for the computation to finish
      device_to_host_stream_->wait(compute_finished_events[ch_block]);

      // Copy next solutions back to host
      device_to_host_stream_->memcpyDtoHAsync(
          &next_solutions(ch_block, 0, 0, 0),
          gpu_buffers_.next_solutions[buffer_id],
          SizeOfNextSolutions(NAntennas(), NSubSolutions(), NSolutionPolarizations()));

      // Record that the output is copied
      device_to_host_stream_->record(output_copied_events[ch_block]);
    }  // end for ch_block

    // Wait for next solutions to be copied
    device_to_host_stream_->synchronize();

    nvtxRangeEnd(nvts_range_gpu);

    // CPU-only postprocessing
    nvtxRangeId_t nvtx_range_cpu = nvtxRangeStart("CPU");
    PostProcessing(iteration, time, has_previously_converged, has_converged,
                   constraints_satisfied, done, result, solutions,
                   next_solutions, step_magnitudes, stat_stream);
    nvtxRangeEnd(nvtx_range_cpu);
  } while (!done);

  // When we have not converged yet, we set the nr of iterations to the max+1,
  // so that non-converged iterations can be distinguished from converged ones.
  if (has_converged && constraints_satisfied) {
    result.iterations = iteration;
  } else {
    result.iterations = iteration + 1;
  }

  if (!keep_buffers_) DeallocateHostBuffers();
  return result;
}

template class IterativeScalarSolverCuda<aocommon::MC2x2F>;
template class IterativeScalarSolverCuda<aocommon::MC2x2FDiag>;
template class IterativeScalarSolverCuda<std::complex<float>>;

}  // namespace ddecal
}  // namespace dp3
