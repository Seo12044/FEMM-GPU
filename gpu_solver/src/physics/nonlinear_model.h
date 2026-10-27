#pragma once

#include "core/types.h"
#include "physics/linear_model.h"
#include "solvers/options.h"
#include "signed_node_constraints.h"

#include <cstdint>
#include <limits>
#include <string>
#include <vector>

namespace gpu_femm
{

struct NonlinearBhEvaluation {
  double h_a_per_m = 0.0;
  double dh_db = 0.0;
  double reluctivity_m_per_h = 0.0;
  double dv_db2 = 0.0;
};

struct NonlinearBhCurve {
  std::vector<double> b_t;
  std::vector<double> h_a_per_m;
  std::vector<double> slope;
  double cold_secant_reluctivity_m_per_h = 0.0;
  int smoothing_passes = 0;
};

bool BuildNonlinearBhSpline(NonlinearBhCurve *curve);

NonlinearBhEvaluation EvaluateNonlinearBh(const NonlinearBhCurve &curve, double field_b);

bool ReadNonlinearBhTable(const std::string &path, NonlinearBhCurve *curve);

// Each element explicitly names a material; an omitted label is a hard error.
struct NonlinearMaterial {
  // v(B) = v0 * (1 + alpha * |B|^2).  alpha==0 is a linear material.
  double reluctivity_zero_m_per_h = 0.0;
  double alpha_per_t2 = 0.0;
  double source_j_per_a = 0.0;
  double h_c_a_per_m = 0.0;
  double magnetization_deg = 0.0;
  int32_t bh_curve_index = -1;
  // -1 denotes a non-driven region.  Driven labels point into the ordered
  // circuit-current vector owned by NonlinearModel.
  int32_t circuit_index = -1;
  // FEMM block-label group, retained for group-based postprocessing.
  int32_t group_number = -1;
};

struct NonlinearTriangle {
  int32_t node[3];
  int32_t material = -1;
};

enum class MagneticProblemType {
  kPlanar,
  kAxisymmetric,
};

// FEMM's native Air Gap Element (AGE) represents the annular air band without
// re-triangulating it.  The four node/weight pairs map each annulus endpoint
// to the independently meshed rotor and stator rings.
struct AirGapQuadPoint {
  int32_t node[4] = {};
  double weight[4] = {};
};

struct AirGapElement {
  bool antiperiodic = false;
  double center_x_m = 0.0;
  double center_y_m = 0.0;
  double inner_radius_m = 0.0;
  double outer_radius_m = 0.0;
  double arc_length_deg = 0.0;
  double inner_shift = 0.0;
  double outer_shift = 0.0;
  std::vector<AirGapQuadPoint> quad_points;
};

struct NonlinearModel {
  std::vector<Node> nodes;
  std::vector<NonlinearMaterial> materials;
  std::vector<NonlinearBhCurve> bh_curves;
  std::vector<NonlinearTriangle> triangles;
  std::vector<AirGapElement> air_gap_elements;
  std::vector<int32_t> dirichlet_nodes;
  std::vector<double> dirichlet_a_wb_per_m;
  std::vector<SignedNodeConstraint> node_constraints;
  // PBC self-pairs at the rotational apex are zero only for an
  // antiperiodic magnetic field. They are tracked separately from physical
  // Dirichlet boundaries so the p and p+1 operators can switch signs exactly
  // and the scalar WST mask does not mistake them for an outer boundary.
  std::vector<int32_t> symmetry_apex_nodes;
  int32_t circuit_count = 1;
  double depth_m = 0.0;
  MagneticProblemType problem_type = MagneticProblemType::kPlanar;
};

struct NonlinearOptions {
  double relative_tolerance = 1e-12;
  int max_newton_iterations = 32;
  int max_linear_iterations = 128;
  double linear_relative_tolerance = 1e-13;
  bool recover_nonconvergence = false;
  bool adaptive_linear_tolerance = false;
};

double NewtonLinearTolerance(const NonlinearOptions &options, double previous_step);

double BatchNewtonLinearTolerance(const NonlinearOptions &options,
                                  const std::vector<double> &previous_steps,
                                  const std::vector<size_t> &slots);

bool NewtonStepConverged(const NonlinearOptions &options, int iteration, double relative_step,
                         double linear_tolerance);

constexpr double kMu0 = 4.0e-7 * 3.141592653589793238462643383279502884;
// fkn's AGE matrix is expressed against relative permeability; P1 assembly
// is SI and therefore needs this free-space reluctivity multiplier.
constexpr double kNativeFemmAgeToSiReluctivity = 1.0 / kMu0;

// The final PCG solve verifies ||Ax-b|| / ||b|| at 1e-10. Earlier Newton
// corrections may use a looser tolerance; the iteration cap stays fixed.
constexpr int kMotorMaxLinearIterations = 16384;
constexpr double kMotorLinearRelativeTolerance = 1e-10;

NonlinearOptions ProductionSampleOptions();

struct NonlinearSolveResult {
  SolveInfo info;
  std::vector<double> a_wb_per_m;
  std::vector<double> bx_t;
  std::vector<double> by_t;
  std::vector<double> residual_history;
  int linear_iterations = 0;
  double last_linear_tolerance = std::numeric_limits<double>::quiet_NaN();
  double verified_equation_residual_relative = std::numeric_limits<double>::infinity();
  std::vector<double> circuit_currents_a;
  std::vector<double> circuit_flux_linkage_wb;
  // Compatibility field for frozen single-circuit callers only.
  double flux_linkage_wb = std::numeric_limits<double>::quiet_NaN();
};

// First-order response about one converged nonlinear operating point.  The
// nodal/element arrays are derivatives per unit command current, not a second
// finite-current operating point.
struct TangentSolveResult {
  SolveInfo info;
  std::string name;
  std::vector<double> circuit_current_derivative_a_per_a;
  NonlinearSolveResult field_derivative;
  SharedOperatorBatchEvidence operator_evidence;
};

struct NonlinearElementTerms {
  double area_m2 = 0.0;
  double b_x[3] = {}; // curl basis x component: dN/dy
  double b_y[3] = {}; // curl basis y component: -dN/dx
};

bool BuildElementTerms(const Node p[3], NonlinearElementTerms *terms);

Status ValidateNonlinearModel(const NonlinearModel &model);

Status ReduceSignedSystem(const NonlinearModel &model, const SparseRows &full_matrix,
                          const std::vector<double> &full_rhs, Assembly *assembly);

std::vector<double> ExpandAssemblySolution(const Assembly &assembly,
                                           const std::vector<double> &free_solution);

std::vector<double> ExpandAssemblyDerivative(const Assembly &assembly,
                                             const std::vector<double> &free_solution);

bool EvaluateNonlinearMaterial(const NonlinearModel &model, const NonlinearMaterial &material,
                               double b_magnitude, bool use_newton, double *reluctivity,
                               double *reluctivity_derivative);

Status AssembleNonlinearNewton(const NonlinearModel &model, const std::vector<double> &a_old,
                               const std::vector<double> &circuit_currents_a, bool use_newton,
                               Assembly *assembly);

Status AssembleNonlinearNewton(const NonlinearModel &model, const std::vector<double> &a_old,
                               double current_a, bool use_newton, Assembly *assembly);

} // namespace gpu_femm
