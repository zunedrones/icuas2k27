// Trajectory predictor: the LUS planning-manager trajectory generation as a service.
//
// For a fly-through waypoint mission it produces exactly the reference the LUS planning manager
// (laser_uav_managers@277a3c9, src/planning_manager_node.cpp) would generate, with the same
// backend and parameter files:
//   * pmm   - pmm_planner::PmmPlanner (same calls); its console diagnostics are captured,
//             because when its first optimisation stage fails the pinned PMM still returns a
//             trajectory but no longer deterministically (`clean = false`);
//   * minco - generate_minco/sample_minco reproduced from the planning manager (private static
//             members there), on the MINCO-S4NU solver exported by super_planner.
// Watchdog: the pinned PMM can loop forever on rare inputs (seen once in a race: the service
// never answered again). A prediction running longer than `watchdog_timeout` seconds ends the
// process; the launch respawns it and the race manager keeps flying the current mission.
//
// The race manager validates the prediction (arena, gate meshes, thrust direction) before
// sending the mission, and publishes it for RViz.
//
// MINCO code adapted from laser_uav_managers (BSD-3-Clause, LASER-Robotics).

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include <Eigen/Dense>
#include <rclcpp/rclcpp.hpp>
#include <traj_opt/minco.h>

#include <geometry_msgs/msg/point.hpp>
#include <geometry_msgs/msg/vector3.hpp>
#include <icuas27_msgs/srv/predict_trajectory.hpp>
#include <laser_msgs/msg/pose_with_heading.hpp>
#include <laser_msgs/msg/reference_state.hpp>
#include <pmm_planner/pmm_planner.hpp>

namespace
{
using Predict = icuas27_msgs::srv::PredictTrajectory;
using Goal = laser_msgs::msg::PoseWithHeading;
using Reference = laser_msgs::msg::ReferenceState;

constexpr double kGravity = 9.81;  // same constant as the PMM planner and the planning manager

double distance(const geometry_msgs::msg::Point & a, const geometry_msgs::msg::Point & b)
{
  return std::hypot(std::hypot(a.x - b.x, a.y - b.y), a.z - b.z);
}

double yaw_of(const geometry_msgs::msg::Quaternion & q)
{
  // Eigen::Quaterniond takes (w, x, y, z).
  const Eigen::Vector3d heading =
    Eigen::Quaterniond(q.w, q.x, q.y, q.z).normalized() * Eigen::Vector3d::UnitX();
  return std::atan2(heading.y(), heading.x());
}

bool finite_state(const Reference & r)
{
  const auto & p = r.pose.position;
  const auto & q = r.pose.orientation;
  const auto & v = r.twist.linear;
  const auto & w = r.twist.angular;
  for (double x : {p.x, p.y, p.z, q.w, q.x, q.y, q.z, v.x, v.y, v.z, w.x, w.y, w.z}) {
    if (!std::isfinite(x)) {
      return false;
    }
  }
  return Eigen::Quaterniond(q.w, q.x, q.y, q.z).norm() > 1e-9;
}

// ------------------------------------------------------------------------------------------
// MINCO (from PlanningManagerNode, laser_uav_managers@277a3c9)

// Bernstein control points bound the whole curve, including peaks between samples.
double derivative_bound(
  Eigen::Matrix<double, 3, Eigen::Dynamic> control, double & lower, int depth = 0)
{
  const int degree = static_cast<int>(control.cols()) - 1;
  const double upper = control.colwise().norm().maxCoeff();
  lower = std::max({lower, control.col(0).norm(), control.col(degree).norm()});
  if (!std::isfinite(upper)) {
    throw std::runtime_error("MINCO returned a non-finite derivative bound");
  }
  if (depth == 16 || upper - lower <= 1e-7 * std::max(1.0, upper)) {
    return upper;
  }
  Eigen::Matrix<double, 3, Eigen::Dynamic> left(3, degree + 1), right(3, degree + 1);
  left.col(0) = control.col(0);
  right.col(degree) = control.col(degree);
  for (int level = 1; level <= degree; ++level) {
    for (int j = 0; j <= degree - level; ++j) {
      control.col(j) = 0.5 * (control.col(j) + control.col(j + 1));
    }
    left.col(level) = control.col(0);
    right.col(degree - level) = control.col(degree - level);
  }
  const double left_bound = derivative_bound(left, lower, depth + 1);
  return std::max(left_bound, derivative_bound(right, lower, depth + 1));
}

double minco_derivative_bound(const geometry_utils::Piece & piece, int derivative)
{
  const int degree = piece.getDegree() - derivative;
  Eigen::Matrix<double, 3, Eigen::Dynamic> power(3, degree + 1);
  const auto & coefficients = piece.getCoeffMat();
  double time_power = 1.0;
  for (int k = 0; k <= degree; ++k) {
    double factor = time_power;
    for (int d = 1; d <= derivative; ++d) {
      factor *= k + d;
    }
    power.col(k) = coefficients.col(piece.getDegree() - k - derivative) * factor;
    time_power *= piece.getDuration();
  }
  Eigen::Matrix<double, 3, Eigen::Dynamic> control = power;
  for (int i = 0; i <= degree; ++i) {
    control.col(i) = power.col(0);
    double ratio = 1.0;
    for (int k = 1; k <= i; ++k) {
      ratio *= static_cast<double>(i - k + 1) / (degree - k + 1);
      control.col(i) += ratio * power.col(k);
    }
  }
  double lower = 0.0;
  return derivative_bound(control, lower);
}

struct MincoPlan
{
  geometry_utils::Trajectory position, yaw;
  double sampling_step{0.01};
  std::vector<double> times;
};

MincoPlan generate_minco(
  const Reference & start, const std::vector<Goal> & goals,
  double speed, double acceleration, double sampling_step, double yaw_rate)
{
  // Collinear knots keep long legs from amplifying derivatives at short neighboring legs.
  const double max_length = std::max(0.1 * speed, speed * speed / (2.0 * acceleration));
  const Eigen::Vector3d initial_velocity = start.use_linear_velocity ?
    Eigen::Vector3d(start.twist.linear.x, start.twist.linear.y, start.twist.linear.z) :
    Eigen::Vector3d::Zero();
  std::vector<Eigen::Vector3d> positions;
  std::vector<double> headings;
  Eigen::Vector3d previous(start.pose.position.x, start.pose.position.y, start.pose.position.z);
  double previous_yaw = yaw_of(start.pose.orientation);
  for (const auto & goal : goals) {
    const Eigen::Vector3d position(goal.position.x, goal.position.y, goal.position.z);
    const double yaw = previous_yaw + std::remainder(goal.heading - previous_yaw, 2.0 * M_PI);
    // Leave room to brake when the initial reference is moving away from the first goal.
    const bool braking_start = positions.empty() &&
      (initial_velocity.dot(position - previous) < 0.0 ||
      (start.use_angular_velocity && start.twist.angular.z * (yaw - previous_yaw) < 0.0));
    const double subdivisions = braking_start ? 1.0 :
      std::max(1.0, std::ceil((position - previous).norm() / max_length));
    if (!std::isfinite(subdivisions) || subdivisions + positions.size() > 16384) {
      throw std::runtime_error("MINCO trajectory exceeds the polynomial piece limit");
    }
    const int count = static_cast<int>(subdivisions);
    for (int j = 1; j <= count; ++j) {
      const double u = static_cast<double>(j) / count;
      positions.push_back(previous + u * (position - previous));
      headings.push_back(previous_yaw + u * (yaw - previous_yaw));
    }
    previous = position;
    previous_yaw = yaw;
  }
  const int pieces = static_cast<int>(positions.size());
  Eigen::Matrix<double, 3, 4> head = Eigen::Matrix<double, 3, 4>::Zero();
  Eigen::Matrix<double, 3, 4> tail = head;
  Eigen::Matrix<double, 3, 4> yaw_head = head;
  Eigen::Matrix<double, 3, 4> yaw_tail = head;
  head.col(0) << start.pose.position.x, start.pose.position.y, start.pose.position.z;
  if (start.use_linear_velocity) {
    head.col(1) << start.twist.linear.x, start.twist.linear.y, start.twist.linear.z;
  }
  yaw_head(0, 0) = yaw_of(start.pose.orientation);
  yaw_head(0, 1) = start.use_angular_velocity ? start.twist.angular.z : 0.0;
  Eigen::Matrix3Xd points(3, pieces - 1), yaw_points(3, pieces - 1);
  yaw_points.setZero();
  Eigen::VectorXd times(pieces);
  previous = head.col(0);
  previous_yaw = yaw_head(0, 0);
  for (int i = 0; i < pieces; ++i) {
    times[i] = std::max(
      {0.1, (positions[i] - previous).norm() / speed,
        2.2 * std::abs(headings[i] - previous_yaw) / yaw_rate});
    if (i + 1 < pieces) {
      points.col(i) = positions[i];
      yaw_points(0, i) = headings[i];
    }
    previous = positions[i];
    previous_yaw = headings[i];
  }
  tail.col(0) = previous;
  yaw_tail(0, 0) = previous_yaw;
  traj_opt::MINCO_S4NU position_solver, yaw_solver;
  position_solver.setConditions(head, tail, pieces);
  yaw_solver.setConditions(yaw_head, yaw_tail, pieces);
  geometry_utils::Trajectory trajectory, yaw_trajectory;
  const double velocity_limit = std::max(speed, head.col(1).norm()) * (1.0 + 1e-6);
  const double yaw_limit = std::max(yaw_rate, std::abs(yaw_head(0, 1))) * (1.0 + 1e-6);
  bool feasible = false;
  for (int attempt = 0; attempt < 40; ++attempt) {
    if (!times.allFinite() || !std::isfinite(times.sum())) {
      throw std::runtime_error("MINCO returned non-finite segment times");
    }
    position_solver.setParameters(points, times);
    yaw_solver.setParameters(yaw_points, times);
    if (!position_solver.getCoeffs().allFinite() || !yaw_solver.getCoeffs().allFinite()) {
      throw std::runtime_error("MINCO returned non-finite polynomial coefficients");
    }
    position_solver.getTrajectory(trajectory);
    yaw_solver.getTrajectory(yaw_trajectory);
    feasible = true;
    Eigen::VectorXd scales = Eigen::VectorXd::Ones(pieces);
    for (int i = 0; i < pieces; ++i) {
      const double scale = std::max(
        {minco_derivative_bound(trajectory[i], 1) / velocity_limit,
          std::sqrt(minco_derivative_bound(trajectory[i], 2) / acceleration),
          minco_derivative_bound(yaw_trajectory[i], 1) / yaw_limit});
      if (scale > 1.0) {
        scales[i] = std::min(1.25, 1.01 * scale);
        feasible = false;
      }
    }
    if (feasible) {
      break;
    }
    // Slow adjacent pieces together so local retiming does not create a new time imbalance.
    for (int i = 0; i < pieces; ++i) {
      times[i] *= std::max(
        {scales[i], scales[std::max(0, i - 1)],
          scales[std::min(pieces - 1, i + 1)]});
    }
  }
  if (!feasible) {
    throw std::runtime_error("MINCO could not satisfy velocity, acceleration and yaw-rate limits");
  }
  MincoPlan result;
  result.position = std::move(trajectory);
  result.yaw = std::move(yaw_trajectory);
  result.sampling_step = sampling_step;
  result.times.push_back(0.0);
  for (int i = 0; i < pieces; ++i) {
    result.times.push_back(result.times.back() + times[i]);
  }
  return result;
}

Reference sample_minco(const MincoPlan & plan, double time, Eigen::Vector3d & acceleration)
{
  time = std::clamp(time, 0.0, plan.times.back());
  const auto upper = std::upper_bound(plan.times.begin(), plan.times.end(), time);
  const auto index = std::min<std::size_t>(upper - plan.times.begin() - 1, plan.times.size() - 2);
  const auto & position_piece = plan.position[static_cast<int>(index)];
  const auto & yaw_piece = plan.yaw[static_cast<int>(index)];
  const double local_time = std::clamp(time - plan.times[index], 0.0, position_piece.getDuration());
  const Eigen::Vector3d position = position_piece.getPos(local_time);
  const Eigen::Vector3d velocity = position_piece.getVel(local_time);
  const double yaw = yaw_piece.getPos(local_time).x();
  acceleration = position_piece.getAcc(local_time);
  const Eigen::Vector3d thrust = acceleration + Eigen::Vector3d(0.0, 0.0, kGravity);
  const Eigen::Vector3d z = thrust.norm() > 1e-9 ?
    Eigen::Vector3d(thrust.normalized()) : Eigen::Vector3d::UnitZ();
  Eigen::Vector3d x = Eigen::Vector3d(-std::sin(yaw), std::cos(yaw), 0.0).cross(z);
  x = x.norm() > 1e-9 ? Eigen::Vector3d(x.normalized()) : z.unitOrthogonal();
  Eigen::Matrix3d attitude;
  attitude.col(0) = x;
  attitude.col(1) = z.cross(x);
  attitude.col(2) = z;
  const Eigen::Quaterniond q(attitude);
  Reference ref;
  ref.pose.position.x = position.x();
  ref.pose.position.y = position.y();
  ref.pose.position.z = position.z();
  ref.pose.orientation.w = q.w();
  ref.pose.orientation.x = q.x();
  ref.pose.orientation.y = q.y();
  ref.pose.orientation.z = q.z();
  ref.twist.linear.x = velocity.x();
  ref.twist.linear.y = velocity.y();
  ref.twist.linear.z = velocity.z();
  ref.twist.angular.z = yaw_piece.getVel(local_time).x();
  ref.use_position = ref.use_orientation = true;
  ref.use_linear_velocity = ref.use_angular_velocity = true;
  if (time >= plan.times.back()) {
    ref.twist = geometry_msgs::msg::Twist();
    acceleration.setZero();
  }
  return ref;
}

geometry_msgs::msg::Vector3 to_vec(const Eigen::Vector3d & v)
{
  geometry_msgs::msg::Vector3 out;
  out.x = v.x();
  out.y = v.y();
  out.z = v.z();
  return out;
}
}  // namespace

class TrajectoryPredictorNode : public rclcpp::Node
{
public:
  TrajectoryPredictorNode()
  : rclcpp::Node("trajectory_predictor")
  {
    // Parameter names and defaults copied from PlanningManagerNode so the same YAML files
    // (planning manager, uav parameters, pmm_planner/<UAV_TYPE>.yaml) configure both nodes.
    declare_parameter<std::string>("planner", "minco");
    declare_parameter("minco.max_velocity", 2.0);
    declare_parameter("minco.max_acceleration", 2.0);
    declare_parameter("minco.sampling_step", 0.01);
    declare_parameter("multirotor_parameters.mass", 0.0);
    declare_parameter("multirotor_parameters.n_motors", 0);
    declare_parameter("multirotor_parameters.G1", std::vector<double>{});
    declare_parameter("multirotor_parameters.inertia", std::vector<double>{});
    declare_parameter("multirotor_parameters.omega_max", std::vector<double>{});
    for (const auto & entry : std::vector<std::pair<std::string, double>>{
        {"multirotor_parameters.max_accel", 22.0}, {"multirotor_parameters.max_vel", 16.0},
        {"multirotor_parameters.default_vel", 1.0}, {"ltd_opt.thrust_decomp_acc_precision", 0.01},
        {"first_vel_opt.alpha", 10.0}, {"first_vel_opt.alpha_reduction_factor", 0.2},
        {"first_vel_opt.alpha_min_threshold", 0.001}, {"second_vel_opt.alpha", 35.0},
        {"second_vel_opt.alpha_reduction_factor", 0.1},
        {"second_vel_opt.alpha_min_threshold", 0.01}, {"time.dt_precision", 0.001},
        {"time.sampling_step", 0.01}})
    {
      declare_parameter("pmm_planner." + entry.first, entry.second);
    }
    declare_parameter("pmm_planner.ltd_opt.use_drag", false);
    declare_parameter("pmm_planner.ltd_opt.thrust_decomp_max_iter", 20);
    declare_parameter("pmm_planner.first_vel_opt.max_iter", 30);
    declare_parameter("pmm_planner.second_vel_opt.run", true);
    declare_parameter("pmm_planner.second_vel_opt.max_iter", 10);
    declare_parameter("watchdog_timeout", 3.0);

    read_parameters();
    watchdog_timeout_ = get_parameter("watchdog_timeout").as_double();
    if (watchdog_timeout_ > 0.0) {
      watchdog_ = std::thread([this] {watchdog_loop();});
    }

    service_ = create_service<Predict>(
      "~/predict",
      [this](const std::shared_ptr<Predict::Request> request,
      std::shared_ptr<Predict::Response> response) {predict(*request, *response);});

    RCLCPP_INFO(
      get_logger(), "Trajectory predictor ready | planner=%s | pmm max_accel=%.2f max_vel=%.2f | "
      "minco max_velocity=%.2f max_acceleration=%.2f", planner_.c_str(),
      parameters_.max_accel_norm, parameters_.max_vel_norm, minco_max_velocity_,
      minco_max_acceleration_);
  }

  ~TrajectoryPredictorNode() override
  {
    stop_watchdog_ = true;
    if (watchdog_.joinable()) {
      watchdog_.join();
    }
  }

private:
  static std::int64_t now_ns()
  {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
      std::chrono::steady_clock::now().time_since_epoch()).count();
  }

  void watchdog_loop()
  {
    while (!stop_watchdog_) {
      std::this_thread::sleep_for(std::chrono::milliseconds(100));
      const auto since = busy_since_ns_.load();
      if (since == 0) {
        continue;
      }
      const double busy = 1e-9 * static_cast<double>(now_ns() - since);
      if (busy > watchdog_timeout_) {
        RCLCPP_FATAL(
          get_logger(), "%s prediction still running after %.1f s (planner stuck): exiting so "
          "the launch respawns the predictor", planner_.c_str(), busy);
        std::cerr.flush();
        std::_Exit(75);
      }
    }
  }

  void read_parameters()
  {
    planner_ = get_parameter("planner").as_string();
    if (planner_ != "pmm" && planner_ != "minco") {
      throw std::invalid_argument("planner must be 'pmm' or 'minco' (SUPER is not predictable)");
    }
    minco_max_velocity_ = get_parameter("minco.max_velocity").as_double();
    minco_max_acceleration_ = get_parameter("minco.max_acceleration").as_double();
    minco_sampling_step_ = get_parameter("minco.sampling_step").as_double();
    physical_.mass = get_parameter("multirotor_parameters.mass").as_double();
    physical_.n_motors =
      static_cast<int>(get_parameter("multirotor_parameters.n_motors").as_int());
    const auto allocation = get_parameter("multirotor_parameters.G1").as_double_array();
    const auto inertia = get_parameter("multirotor_parameters.inertia").as_double_array();
    const auto rates = get_parameter("multirotor_parameters.omega_max").as_double_array();
    if (physical_.mass <= 0.0 || physical_.n_motors != 4 ||
      allocation.size() != static_cast<std::size_t>(4 * physical_.n_motors) ||
      inertia.size() != 3 || rates.size() != 3)
    {
      throw std::invalid_argument(
              "trajectory_predictor needs the multirotor_parameters of the UAV (mass, 4 motors, "
              "G1, inertia, omega_max)");
    }
    // G1 is stored row-major in the YAML, exactly as the planning manager maps it.
    physical_.G1 = Eigen::Map<const Eigen::Matrix<double, 4, Eigen::Dynamic, Eigen::RowMajor>>(
      allocation.data(), 4, physical_.n_motors);
    if (std::abs(physical_.G1.determinant()) < 1e-12) {
      throw std::invalid_argument("PMM requires an invertible four-motor allocation");
    }
    physical_.inertia_matrix = Eigen::Map<const Eigen::Vector3d>(inertia.data()).asDiagonal();
    yaw_rate_limit_ = rates[2];

    parameters_.max_accel_norm =
      get_parameter("pmm_planner.multirotor_parameters.max_accel").as_double();
    parameters_.max_vel_norm =
      get_parameter("pmm_planner.multirotor_parameters.max_vel").as_double();
    parameters_.default_vel_norm =
      get_parameter("pmm_planner.multirotor_parameters.default_vel").as_double();
    parameters_.use_drag = get_parameter("pmm_planner.ltd_opt.use_drag").as_bool();
    parameters_.thrust_decomp_acc_precision =
      get_parameter("pmm_planner.ltd_opt.thrust_decomp_acc_precision").as_double();
    parameters_.thrust_decomp_max_iter =
      static_cast<int>(get_parameter("pmm_planner.ltd_opt.thrust_decomp_max_iter").as_int());
    parameters_.first_run_alpha = get_parameter("pmm_planner.first_vel_opt.alpha").as_double();
    parameters_.first_run_alpha_reduction_factor =
      get_parameter("pmm_planner.first_vel_opt.alpha_reduction_factor").as_double();
    parameters_.first_run_alpha_min_threshold =
      get_parameter("pmm_planner.first_vel_opt.alpha_min_threshold").as_double();
    parameters_.first_run_max_iter =
      static_cast<int>(get_parameter("pmm_planner.first_vel_opt.max_iter").as_int());
    parameters_.run_second_opt = get_parameter("pmm_planner.second_vel_opt.run").as_bool();
    parameters_.second_run_alpha = get_parameter("pmm_planner.second_vel_opt.alpha").as_double();
    parameters_.second_run_alpha_reduction_factor =
      get_parameter("pmm_planner.second_vel_opt.alpha_reduction_factor").as_double();
    parameters_.second_run_alpha_min_threshold =
      get_parameter("pmm_planner.second_vel_opt.alpha_min_threshold").as_double();
    parameters_.second_run_max_iter =
      static_cast<int>(get_parameter("pmm_planner.second_vel_opt.max_iter").as_int());
    parameters_.dt_precision = get_parameter("pmm_planner.time.dt_precision").as_double();
    parameters_.sampling_step = get_parameter("pmm_planner.time.sampling_step").as_double();

    for (double value : {minco_max_velocity_, minco_max_acceleration_, minco_sampling_step_,
        yaw_rate_limit_, parameters_.max_vel_norm, parameters_.sampling_step,
        parameters_.dt_precision})
    {
      if (!std::isfinite(value) || value <= 0.0) {
        throw std::invalid_argument("limits and sampling steps must be positive and finite");
      }
    }
    if (parameters_.max_accel_norm <= kGravity) {
      throw std::invalid_argument("PMM max_accel must exceed gravity");
    }
  }

  void predict(const Predict::Request & request, Predict::Response & response)
  {
    const auto began = std::chrono::steady_clock::now();
    busy_since_ns_ = now_ns();
    struct BusyGuard
    {
      std::atomic<std::int64_t> & flag;
      ~BusyGuard() {flag = 0;}
    } guard{busy_since_ns_};
    response.success = false;
    response.clean = true;
    response.planner = planner_;
    try {
      const auto & goals = request.waypoints;
      // Same acceptance rules as PlanningManagerNode::start_mission (fly-through mission).
      if (goals.empty() || goals.size() > 256) {
        throw std::invalid_argument("waypoint count must be in [1, 256]");
      }
      if (!std::isfinite(request.speed) || request.speed < 0.0) {
        throw std::invalid_argument("speed must be finite and >= 0");
      }
      if (planner_ == "minco" && request.speed > minco_max_velocity_) {
        throw std::invalid_argument("requested speed exceeds minco.max_velocity");
      }
      if (!finite_state(request.start)) {
        throw std::invalid_argument("start reference is not finite");
      }
      for (std::size_t i = 0; i < goals.size(); ++i) {
        const auto & p = goals[i].position;
        if (!std::isfinite(p.x) || !std::isfinite(p.y) || !std::isfinite(p.z) ||
          !std::isfinite(goals[i].heading) ||
          (i > 0 && distance(goals[i - 1].position, p) < 1e-8))
        {
          throw std::invalid_argument("non-finite or coincident fly-through waypoints");
        }
      }
      if (planner_ == "minco") {
        predict_minco(request, response);
      } else {
        predict_pmm(request, response);
      }
      response.duration = response.times.back() - response.times.front();
      response.success = true;
      response.message = response.clean ? "ok" : "planner reported internal errors";
    } catch (const std::exception & e) {
      response.message = e.what();
      RCLCPP_WARN(get_logger(), "Prediction failed: %s", e.what());
    }
    response.planning_ms = std::chrono::duration<double, std::milli>(
      std::chrono::steady_clock::now() - began).count();
  }

  void predict_minco(const Predict::Request & request, Predict::Response & response)
  {
    const double speed = request.speed > 0.0 ? request.speed : minco_max_velocity_;
    const MincoPlan plan = generate_minco(
      request.start, request.waypoints, speed, minco_max_acceleration_, minco_sampling_step_,
      yaw_rate_limit_);
    const double total = plan.times.back();
    const int count = static_cast<int>(std::ceil(total / minco_sampling_step_));
    Eigen::Vector3d acc;
    for (int i = 0; i <= count; ++i) {
      const double t = std::min(i * minco_sampling_step_, total);
      const Reference ref = sample_minco(plan, t, acc);
      if (!finite_state(ref)) {
        throw std::runtime_error("MINCO returned a non-finite reference");
      }
      push_sample(response, t, ref, acc);
      if (t >= total) {
        break;
      }
    }
  }

  void predict_pmm(const Predict::Request & request, Predict::Response & response)
  {
    // The PMM library reports failed optimisation stages on std::cout only.
    std::stringstream captured;
    std::streambuf * previous = std::cout.rdbuf(captured.rdbuf());
    std::vector<double> times;
    std::vector<Reference> states;
    try {
      pmm_planner::PmmPlanner planner(physical_, parameters_);
      const auto & goals = request.waypoints;
      const float speed = static_cast<float>(request.speed);
      if (goals.size() == 1) {
        planner.generateTrajectory(request.start, goals.front(), speed, request.speed > 0.0);
      } else {
        planner.generateTrajectory(request.start, goals, speed);
      }
      times = planner.getTrajectoryTimes();
      states = planner.getTrajectory();
    } catch (...) {
      std::cout.rdbuf(previous);
      throw;
    }
    std::cout.rdbuf(previous);
    response.planner_log = captured.str();
    response.clean = response.planner_log.find("ERROR") == std::string::npos;
    if (times.size() != states.size() || times.size() < 2) {
      throw std::runtime_error("PMM returned an empty or mismatched trajectory");
    }
    // Same de-duplication of equal sample times as the planning manager.
    std::vector<double> t_out;
    std::vector<Reference> s_out;
    for (std::size_t i = 0; i < times.size(); ++i) {
      if (!std::isfinite(times[i]) || times[i] < 0 ||
        (!t_out.empty() && times[i] < t_out.back() - 1e-9))
      {
        throw std::runtime_error("PMM returned invalid sample times");
      }
      if (!t_out.empty() && std::abs(times[i] - t_out.back()) <= 1e-9) {
        s_out.back() = states[i];
      } else {
        t_out.push_back(times[i]);
        s_out.push_back(states[i]);
      }
    }
    s_out.back().twist = geometry_msgs::msg::Twist();
    const Eigen::Vector3d gravity(0.0, 0.0, -kGravity);
    for (std::size_t i = 0; i < s_out.size(); ++i) {
      const auto & s = s_out[i];
      if (!finite_state(s)) {
        throw std::runtime_error("PMM returned a non-finite reference");
      }
      // PmmPlanner stores thrust = mass * |a - g| / n per motor and the attitude whose body z
      // axis is (a - g) normalised, so the commanded acceleration is recovered exactly.
      double thrust = 0.0;
      for (double f : s.individual_thrust.data) {
        thrust += f;
      }
      const auto & q = s.pose.orientation;
      const Eigen::Vector3d z_body =
        Eigen::Quaterniond(q.w, q.x, q.y, q.z).normalized() * Eigen::Vector3d::UnitZ();
      push_sample(response, t_out[i], s, (thrust / physical_.mass) * z_body + gravity);
    }
  }

  static void push_sample(
    Predict::Response & response, double t, const Reference & ref, const Eigen::Vector3d & acc)
  {
    response.times.push_back(t);
    response.positions.push_back(ref.pose.position);
    geometry_msgs::msg::Vector3 v;
    v.x = ref.twist.linear.x;
    v.y = ref.twist.linear.y;
    v.z = ref.twist.linear.z;
    response.velocities.push_back(v);
    response.accelerations.push_back(to_vec(acc));
    response.headings.push_back(yaw_of(ref.pose.orientation));
  }

  std::string planner_;
  double minco_max_velocity_{2.0};
  double minco_max_acceleration_{2.0};
  double minco_sampling_step_{0.01};
  double yaw_rate_limit_{1.0};
  pmm_planner::multirotor_t physical_{};
  pmm_planner::pmm_t parameters_{};
  rclcpp::Service<Predict>::SharedPtr service_;
  double watchdog_timeout_{3.0};
  std::atomic<std::int64_t> busy_since_ns_{0};
  std::atomic<bool> stop_watchdog_{false};
  std::thread watchdog_;
};

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  try {
    rclcpp::spin(std::make_shared<TrajectoryPredictorNode>());
  } catch (const std::exception & e) {
    RCLCPP_FATAL(rclcpp::get_logger("trajectory_predictor"), "%s", e.what());
    rclcpp::shutdown();
    return 1;
  }
  rclcpp::shutdown();
  return 0;
}
