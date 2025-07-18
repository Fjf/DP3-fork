// Copyright (C) 2020 ASTRON (Netherlands Institute for Radio Astronomy)
// SPDX-License-Identifier: GPL-3.0-or-later

#include "IterativeScalarSolver.h"

#include <algorithm>
#include <optional>
#include <iostream>
#include <cmath>
#include <limits>

#include <aocommon/matrix2x2.h>
#include <aocommon/matrix2x2diag.h>
#include <aocommon/staticfor.h>
#include <xtensor/xtensor.hpp>

using aocommon::MC2x2;
using aocommon::MC2x2F;

namespace dp3 {
namespace ddecal {

inline std::complex<float> HermTranspose(std::complex<float> value) {
  return std::conj(value);
}
inline std::complex<float> Trace(std::complex<float> value) {
  // Calculate the Trace of a 1x1 matrix
  return value;
}
inline float Norm(std::complex<float> value) { return std::norm(value); }
inline double Norm(std::complex<double> value) { return std::norm(value); }


#include <type_traits>

// Helper for scalar types
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
typename IterativeScalarSolver<VisMatrix>::SolveResult
IterativeScalarSolver<VisMatrix>::Solve(
    const SolveData<VisMatrix>& data,
    std::vector<std::vector<DComplex>>& solutions, double time,
    std::ostream* stat_stream) {
  PrepareConstraints();

  SolutionTensor next_solutions({NChannelBlocks(), NAntennas(), NSubSolutions(),
                                 NSolutionPolarizations()});

  SolveResult result;

  // Visibility vector v_residual[cb][vis] of size NChannelBlocks() x
  // n_visibilities
  std::vector<std::vector<VisMatrix>> v_residual(NChannelBlocks());
  // The following loop allocates all structures
  for (size_t ch_block = 0; ch_block != NChannelBlocks(); ++ch_block) {
    v_residual[ch_block].resize(data.ChannelBlock(ch_block).NVisibilities());
  }

  ///
  /// Start iterating
  ///
  size_t iteration = 0;
  bool has_converged = false;
  bool has_previously_converged = false;
  bool constraints_satisfied = false;

  std::vector<double> step_magnitudes;
  step_magnitudes.reserve(GetMaxIterations());

  std::unique_ptr<aocommon::RecursiveFor> recursive_for =
      MakeOptionalRecursiveFor();
  do {
    MakeSolutionsFinite1Pol(solutions);

    // Sequential processing (parallelization disabled)
    // for (size_t ch_block = 0; ch_block < NChannelBlocks(); ++ch_block) {
    //   PerformIteration(ch_block, data.ChannelBlock(ch_block),
    //                    v_residual[ch_block], solutions[ch_block],
    //                    next_solutions);
    // }

    aocommon::RunStaticFor<size_t>(
    0, NChannelBlocks(), [&](size_t ch_block, size_t end_index) {
      for (; ch_block < end_index; ++ch_block) {
        PerformIteration(ch_block, data.ChannelBlock(ch_block),
                          v_residual[ch_block], solutions[ch_block],
                          next_solutions);
      }
    });

    Step(solutions, next_solutions);

    constraints_satisfied =
        ApplyConstraints(iteration, time, has_previously_converged, result,
                         next_solutions, stat_stream);

    double avg_squared_diff;
    has_converged =
        AssignSolutions(solutions, next_solutions, !constraints_satisfied,
                        avg_squared_diff, step_magnitudes);
    iteration++;

    has_previously_converged = has_converged || has_previously_converged;

  } while (!ReachedStoppingCriterion(iteration, has_converged,
                                     constraints_satisfied, step_magnitudes));

  // When we have not converged yet, we set the nr of iterations to the max+1,
  // so that non-converged iterations can be distinguished from converged ones.
  if (has_converged && constraints_satisfied)
    result.iterations = iteration;
  else
    result.iterations = iteration + 1;
  return result;
}

template <typename VisMatrix>
void IterativeScalarSolver<VisMatrix>::PerformIteration(
    size_t ch_block, const ChannelBlockData& cb_data,
    std::vector<VisMatrix>& v_residual, const std::vector<DComplex>& solutions,
    SolutionTensor& next_solutions) {
  // Fill v_residual
  std::copy(cb_data.DataBegin(), cb_data.DataEnd(), v_residual.begin());

  PrintVectorSummary(v_residual, "residual_pre_kernel");

  // Subtract all directions with their current solutions
  for (size_t direction = 0; direction != NDirections(); ++direction)
    AddOrSubtractDirection<false>(cb_data, v_residual, direction, solutions);



  PrintVectorSummary(v_residual, "v_residual_post_kernel");


  const std::vector<VisMatrix> v_copy = v_residual;

  for (size_t direction = 0; direction != NDirections(); ++direction) {
    // Be aware that we purposely still use the subtraction with 'old'
    // solutions, because the new solutions have not been constrained yet. Add
    // this direction back before solving
    if (direction != 0) v_residual = v_copy;
    // PrintVectorSummary(v_residual, "v_residual_post_kernel");
    AddOrSubtractDirection<true>(cb_data, v_residual, direction, solutions);
    
    
    // Analyze solutions before solving direction
    // PrintVectorSummary(solutions, "solutions_pre_solve_direction");
    // PrintSolutionTensorSummary(next_solutions, "next_solutions_pre_solve_direction", ch_block);
    
    // exit(0); // Debugging exit point
    SolveDirection(ch_block, cb_data, v_residual, direction, solutions,
                   next_solutions);
    
    // Print cb_data information
    std::cout << "cb_data info:" << std::endl;
    std::cout << "  NVisibilities: " << cb_data.NVisibilities() << std::endl;
    std::cout << "  NDirections: " << cb_data.NDirections() << std::endl;
    std::cout << "  NSolutionsForDirection(" << direction << "): " 
              << cb_data.NSolutionsForDirection(direction) << std::endl;
    std::cout << "  NSubSolutions (total): " << cb_data.NSubSolutions() << std::endl;
    
    // Print some antenna pair information for first few visibilities
    const size_t max_vis_to_show = std::min(static_cast<size_t>(5), cb_data.NVisibilities());
    std::cout << "  First " << max_vis_to_show << " antenna pairs:" << std::endl;
    for (size_t i = 0; i < max_vis_to_show; ++i) {
      std::cout << "    vis[" << i << "]: ant1=" << cb_data.Antenna1Index(i) 
                << ", ant2=" << cb_data.Antenna2Index(i) 
                << ", sol_idx=" << cb_data.SolutionIndex(direction, i) << std::endl;
    }
    
    // Print some model visibility norms for this direction
    std::vector<double> model_norms;
    for (size_t i = 0; i < max_vis_to_show; ++i) {
      const auto& model_vis = cb_data.ModelVisibility(direction, i);
      model_norms.push_back(Norm(model_vis));
    }
    if (!model_norms.empty()) {
      std::cout << "  First " << max_vis_to_show << " model visibility norms for direction " << direction << ":" << std::endl;
      for (size_t i = 0; i < model_norms.size(); ++i) {
        std::cout << "    model_vis[" << i << "] norm: " << model_norms[i] << std::endl;
      }
    }
    
    PrintVectorSummary(v_residual, "v_residual_post_kernel_2");
    PrintVectorSummary(solutions, "solutions_post_solve_direction");
    
    // Analyze next_solutions after solving direction
    PrintSolutionTensorSummary(next_solutions, "next_solutions_post_solve_direction", ch_block);
    
    // Add count of non-zero model values
    size_t total_model_values = 0;
    size_t non_zero_model_values = 0;
    for (size_t i = 0; i < cb_data.NVisibilities(); ++i) {
      const auto& model_vis = cb_data.ModelVisibility(direction, i);
      total_model_values++;
      if (Norm(model_vis) > 1e-12) {
        non_zero_model_values++;
      }
    }
    std::cout << "Count non-zero model values: " << non_zero_model_values 
              << " out of " << total_model_values << " total" << std::endl;
    
    if (direction == 0) exit(0);  // Exit after first direction only

  }
}

template <typename VisMatrix>
void IterativeScalarSolver<VisMatrix>::SolveDirection(
    size_t ch_block, const ChannelBlockData& cb_data,
    const std::vector<VisMatrix>& v_residual, size_t direction,
    const std::vector<DComplex>& solutions, SolutionTensor& next_solutions) {
  // Calculate this equation, given ant a:
  //
  //          sum_b data_ab * solutions_b * model_ab^*
  // sol_a =  ----------------------------------------
  //             sum_b norm(model_ab * solutions_b)

  const uint32_t n_dir_solutions = cb_data.NSolutionsForDirection(direction);
  std::vector<std::complex<double>> numerator(NAntennas() * n_dir_solutions,
                                              0.0);
  std::vector<double> denominator(NAntennas() * n_dir_solutions, 0.0);

  // Iterate over all data
  const size_t n_visibilities = cb_data.NVisibilities();
  const uint32_t solution_index0 = cb_data.SolutionIndex(direction, 0);

  std::mutex mutex;
  aocommon::RunConstrainedStaticFor<size_t>(
      0u, n_visibilities, NSubThreads(),
      [&](size_t start_vis_index, size_t end_vis_index) {
        std::vector<std::complex<double>> local_numerator(
            NAntennas() * n_dir_solutions, 0.0);
        std::vector<double> local_denominator(NAntennas() * n_dir_solutions,
                                              0.0);
        for (size_t vis_index = start_vis_index; vis_index != end_vis_index;
             ++vis_index) {
          
          const uint32_t antenna_1 = cb_data.Antenna1Index(vis_index);
          const uint32_t antenna_2 = cb_data.Antenna2Index(vis_index);
          
          const uint32_t solution_index =
              cb_data.SolutionIndex(direction, vis_index);
          
          const Complex solution_ant_1(
              solutions[antenna_1 * NSubSolutions() + solution_index]);
          const Complex solution_ant_2(
              solutions[antenna_2 * NSubSolutions() + solution_index]);
          const VisMatrix& data = v_residual[vis_index];
          const VisMatrix& model =
              cb_data.ModelVisibility(direction, vis_index);


          const uint32_t rel_solution_index = solution_index - solution_index0;
          // std::cout << "DEBUG: vis_index: " << vis_index << " ant_1: " << antenna_1 << " ant_2: " << antenna_2 << " sol_idx: " << solution_index << 
          //           " rel_sol_idx: " << rel_solution_index << " sol_ant_1: " << solution_ant_1 << " sol_ant_2: " << solution_ant_2 << std::endl;
          // Calculate the contribution of this baseline for antenna_1
          const VisMatrix cor_model_herm_1(HermTranspose(model) *
                                           solution_ant_2);
          // std::cout << "DEBUG: cor_model_herm_1: " << cor_model_herm_1 << std::endl;
          const uint32_t full_solution_1_index =
              antenna_1 * n_dir_solutions + rel_solution_index;
          local_numerator[full_solution_1_index] +=
              Trace(data * cor_model_herm_1);
          local_denominator[full_solution_1_index] += Norm(cor_model_herm_1);

          // Calculate the contribution of this baseline for antenna2
          const VisMatrix cor_model_2(model * solution_ant_1);
          // std::cout << "DEBUG: cor_model_2: " << cor_model_2 << std::endl;
          const uint32_t full_solution_2_index =
              antenna_2 * n_dir_solutions + rel_solution_index;
          local_numerator[full_solution_2_index] +=
              Trace(HermTranspose(data) * cor_model_2);
          // Printing Trace(HermTranspose(data) * cor_model_2);
          // std::cout << "DEBUG: Trace(HermTranspose(data) * cor_model_2): "
          //           << Trace(HermTranspose(data) * cor_model_2) << std::endl;
          // std::cout << "DEBUG: data: " << data << std::endl;
          local_denominator[full_solution_2_index] += Norm(cor_model_2);
          // std::cout << "local_numerator[" << full_solution_1_index << "]: "
          //           << local_numerator[full_solution_1_index] << std::endl;
          // std::cout << "local_denominator[" << full_solution_1_index << "]: "
          //           << local_denominator[full_solution_1_index] << std::endl;
        }
        std::scoped_lock lock(mutex);
        for (size_t i = 0; i != numerator.size(); ++i)
          numerator[i] += local_numerator[i];
        for (size_t i = 0; i != denominator.size(); ++i)
          denominator[i] += local_denominator[i];
      //   std::cout << "DEBUG: ch_block: " << ch_block
      //             << " direction: " << direction
      //             << " start_vis_index: " << start_vis_index
      //             << " end_vis_index: " << end_vis_index
      //             << " local_numerator: " << local_numerator[0]
      //             << " local_denominator: " << local_denominator[0] << std::endl;
      });

  for (size_t ant = 0; ant != NAntennas(); ++ant) {
    for (uint32_t rel_sol = 0; rel_sol != n_dir_solutions; ++rel_sol) {
      const uint32_t solution_index = rel_sol + solution_index0;
      DComplex& destination = next_solutions(ch_block, ant, solution_index, 0);
      const uint32_t index = ant * n_dir_solutions + rel_sol;
      if (denominator[index] == 0.0) {
        destination = std::numeric_limits<float>::quiet_NaN();
      } else {
        destination = numerator[index] / denominator[index];
        // if (ant < 3) {  // Only print first few antennas to avoid spam
        //   std::cout << "DEBUG CPU: ant=" << ant << " sol=" << solution_index 
        //             << " numerator=" << numerator[index] 
        //             << " denominator=" << denominator[index] 
        //             << " solution=" << destination << std::endl;
        // }
      }
    }
  }
}

template <typename VisMatrix>
template <bool Add>
void IterativeScalarSolver<VisMatrix>::AddOrSubtractDirection(
    const ChannelBlockData& cb_data, std::vector<VisMatrix>& v_residual,
    size_t direction, const std::vector<DComplex>& solutions) {
  const size_t n_visibilities = cb_data.NVisibilities();
  aocommon::StaticFor<size_t> loop;
  loop.ConstrainedRun(
      0, n_visibilities, NSubThreads(),
      [&](size_t start_vis_index, size_t end_vis_index) {
        for (size_t vis_index = start_vis_index; vis_index != end_vis_index;
             ++vis_index) {
          const uint32_t antenna_1 = cb_data.Antenna1Index(vis_index);
          const uint32_t antenna_2 = cb_data.Antenna2Index(vis_index);
          const uint32_t solution_index =
              cb_data.SolutionIndex(direction, vis_index);
          const Complex solution_1(
              solutions[antenna_1 * NSubSolutions() + solution_index]);
          const Complex solution_2_conj = std::conj(
              Complex(solutions[antenna_2 * NSubSolutions() + solution_index]));
          VisMatrix& data = v_residual[vis_index];
          const VisMatrix& model =
              cb_data.ModelVisibility(direction, vis_index);
          const VisMatrix corrected_model =
              model * solution_1 * solution_2_conj;
          std::cout << "DEBUG AddOrSubtract: vis_index: " << vis_index 
                    // << " antenna_1: " << antenna_1 
                    // << " antenna_2: " << antenna_2 
                    // << " solution_index: " << solution_index 
                    // << " solution_1: " << solution_1 
                    // << " solution_2_conj: " << solution_2_conj 
                    // << " model: " << model 
                    << " corrected_model: " << corrected_model << " data" << data << std::endl;
          if (Add) {
            data += corrected_model;
          } else {
            data -= corrected_model;
          }
        }
      });
}

template class IterativeScalarSolver<std::complex<float>>;
template class IterativeScalarSolver<aocommon::MC2x2F>;
template class IterativeScalarSolver<aocommon::MC2x2FDiag>;

}  // namespace ddecal
}  // namespace dp3
