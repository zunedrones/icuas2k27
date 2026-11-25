# ICUAS 2027 UAV Competition — autonomous drone racing (zunedrones)

First version of our solution for the [ICUAS 2027 UAV Competition](http://laser.ci.ufpb.br)
(rulebook v1.0, 20/10/2026): a Crazyflie flies **7 laps through 7 gates** of the
`crazyflie_map` course in the LASER UAV System (LUS) simulation, without human input, and lands
on the takeoff base. Everything runs in Docker (ROS 2 Humble, Ubuntu 22.04, Gazebo Classic 11,
PX4 SITL 1.15).

What it does, end to end:

1. **Course** — the 7 gate poses are extracted from the competition mesh
   (`extract_course`) and stored in [`crazyflie_map.yaml`](ros2_ws/src/icuas27_bringup/config/courses/crazyflie_map.yaml).
2. **Race plan** — each gate passage is a pre-gate, gate-centre and post-gate waypoint (49
   passages, then a hover point above the base). The race is flown as a **receding horizon** of
   short fly-through missions (9 gates each, replaced every 3 gates) sent to the LUS planning
   manager, which generates a **PMM minimum-time trajectory** tracked by the LUS NMPC (angular
   rates + thrust) and the PX4 rate loop.
3. **Validation** — every mission's exact PMM trajectory is predicted with the same library and
   parameters (`trajectory_predictor`) and checked against the arena walls and the real gate
   meshes before it is dispatched (and the whole race once before takeoff, for information).
4. **Race manager** — start command → arm → takeoff → race → landing, with abort/landing on any
   failure and automatic **retry of a missed gate** (flies around the frame and re-enters it).
5. **Referee** — an independent scorer that applies the rulebook to the simulator ground truth:
   gate order and direction, contacts with the real gate geometry, flight envelope, crashes,
   landing on the base, `T_adj = T_elapsed + 5 s × (hits + missed entries)`.
6. **Visualisation and analytics** — live Gazebo GUI + RViz (course, drone, speed-coloured path,
   gate states, HUD), a rosbag of every run, a replay launch, and a self-contained **HTML report**
   per run (replay animation, ~15 figures, lap/gate tables, every metric) plus a multi-run
   comparison. With `camera:=true` the AI-deck camera is simulated and `camera_view` draws the
   gates projected into the drone's image (`/race/camera/overlay`) and logs how visible the next
   gate is — the reference for the future camera gate detector.

## Contents

- [Quick start](#quick-start)
- [Outputs of a run](#outputs-of-a-run)
- [Architecture](#architecture)
- [Current results](#current-results)
- [Rulebook interpretation and assumptions](#rulebook-interpretation-and-assumptions)
- [Configuration and tuning](#configuration-and-tuning)
- [Frames and conventions](#frames-and-conventions)
- [Findings about the competition files](#findings-about-the-competition-files)
- [Submission checklist (Phase 1)](#submission-checklist-phase-1)
- [Development](#development)
- [Roadmap](#roadmap)

## Quick start

Requirements: Docker with compose v2, ~25 GB of disk, x86_64, 8 GB+ RAM (16 GB recommended for the
first build). No GPU is needed (software rendering); no host ROS installation.

```bash
git clone https://github.com/zunedrones/icuas2k27.git && cd icuas2k27
docker compose build            # first build: fetches and compiles LUS + PX4 (~45-90 min)
```

Run one race with the Gazebo GUI and RViz (the race starts automatically when the stack is ready):

```bash
./scripts/race.sh
```

Headless race (evaluation mode), three headless races with a comparison report, replay and report:

```bash
./scripts/race.sh --headless
./scripts/batch.sh 3
./scripts/replay.sh results/<run> 0.5        # replay at half speed in RViz
./scripts/report.sh results/<run>            # rebuild the HTML report
```

Inside the container (`docker compose run --rm sim`), the single automated entry point is

```bash
ros2 launch icuas27_bringup race_sim.launch.py              # gui:=false rviz:=false headless:=true ...
ros2 launch icuas27_bringup race_sim.launch.py auto_start:=false
ros2 service call /race/start std_srvs/srv/Trigger {}       # explicit start command
```

`ros2 launch icuas27_bringup race_sim.launch.py --show-args` lists every option (course, parameter
files, camera, odometry source, recording, ...).

## Outputs of a run

Each run writes `results/<timestamp>[_name]/`:

| Path | Content |
| --- | --- |
| `report/report.html` | self-contained report: result cards, replay animation, figures, lap/gate/event tables, all metrics |
| `report/metrics.json`, `report/figures/*.png` | metrics and figures |
| `referee/summary.json` | official-style result: state, `T_elapsed`, penalties, `T_adj`, passes, lap times, contacts, events |
| `referee/trajectory.csv` | ground truth at 100 Hz in the world frame (+ obstacle clearance) |
| `referee/events.csv`, `passes.csv`, `laps.csv` | referee events, gate passes (crossing offsets, speed, clearance), lap times |
| `plan/plan_*.json`, `plan/missions.csv` | waypoints, predicted PMM trajectory, validation, frame alignment, dispatch times |
| `telemetry/*.csv` | LUS signals: commanded reference, NMPC solve time, motor thrusts, estimation, planning states, PX4 status |
| `bag/` | rosbag2 of the run (replayable with `replay.launch.py`) |
| `config/`, `run.json`, `launch.log` | exact parameter files, metadata (git commit, arguments), console log |

Report contents: top view and 3D path coloured by speed with the planned trajectory; speed,
height and acceleration vs time (actual vs planned); lap times and the laps × gates split matrix;
gate crossing accuracy per gate (where in the opening the drone passed, by lap); hull clearance to
the gate structures; commanded-reference tracking error; attitude and body rates; motor commands
and NMPC timing; speed distribution; plan validation; event log; and a canvas replay with a
time slider. `race_compare` aggregates runs (success rate, best/mean `T_adj`, lap-time spread).

## Architecture

```
                         ┌───────────────────────── icuas27 (this repository) ─────────────────────────┐
 Gazebo Classic          │                                                                            │
 crazyflie_map  ── /uav1/ground_truth ──► gt_odometry_bridge ──► /uav1/race/gt_odometry ─┐            │
 Crazyflie + PX4 SITL    │       │                                                         │            │
        ▲                │       ├──► referee ──► /race/referee/{status,events}, results   │            │
        │ motors         │       ├──► visualizer ──► /race/markers (RViz)                  │            │
        │                │       └──► race_manager ◄─ /race/start                          │            │
        │                │              │  ▲ predict + validate (trajectory_predictor)        │            │
        │                │              │  │                                               │            │
        │                └──────────────┼──┼───────────────────────────────────────────────┼────────────┘
 ┌──────┴──────────── LASER UAV System ─┼──┼───────────────────────────────────────────────┼──────┐
 │ px4_api ◄── control_manager (NMPC rates) ◄─ planning_manager (PMM) ◄── trajectory_path ─┘      │
 │                     ▲         ▲               ▲                                                │
 │                     └── estimation_manager (MEKF) ◄── odometry_in (ground truth) ◄─────────────┘
 └────────────────────────────────────────────────────────────────────────────────────────────────┘
```

Packages (`ros2_ws/src`):

| Package | Type | Role |
| --- | --- | --- |
| `icuas27_msgs` | interfaces | `RaceEvent`, `RaceStatus`, `GateEvent`, `RefereeStatus`, `PredictTrajectory` |
| `icuas27_trajectory` | C++ | `trajectory_predictor`: the LUS planning-manager trajectory generation (PMM or MINCO) as a service, same code and parameter files |
| `icuas27_race` | Python | race manager, referee, ground-truth bridge, visualiser, telemetry logger, analytics (`race_report`, `race_compare`), tools (`extract_course`, `plan_course`) |
| `icuas27_bringup` | launch/config | `race_sim.launch.py` (everything), `sim.launch.py` (LUS stack), `race.launch.py` (race nodes), `replay.launch.py`, course, parameters, RViz |

The pure-Python core (`geometry`, `mesh`, `course`, `planner`, `referee_core`, `analytics`) has no
ROS dependency and is unit tested.

### LASER UAV System version

[`docker/lus_pins.txt`](docker/lus_pins.txt) pins every LUS repository to the snapshot the
competition overlay [`laser_crazyflie@0944388`](https://github.com/laser-robotics-research/laser_crazyflie)
was validated against (the five commits its README lists; the other packages at their branch
heads of that date; PX4 `release/1.15`). [`docker/fetch_lus.sh`](docker/fetch_lus.sh) and
[`docker/build_lus.sh`](docker/build_lus.sh) reproduce `laser_uav_system/install.sh` (simulation,
PX4) non-interactively, with the same `~/git/laser_uav_system` layout the LUS scripts expect, and
apply the overlay with its own `install_crazyflie.sh`. Nothing from LUS is modified; our launch
only passes parameter files and, for the ground-truth estimation, remaps the estimation-manager
odometry input.

## Current results

Simulation, `crazyflie_map`, default configuration (race speed 3 m/s, PMM `max_accel` 12 m/s²),
2026-11-25, on an 8-core laptop CPU with software rendering:

| Run | Mode | Result | Gates | Hits / missed | `T_adj` |
| --- | --- | --- | --- | --- | --- |
| rh4 | headless | FINISHED | 49/49 | 0 / 0 | 64.59 s |
| rh5 | headless | FINISHED | 49/49 | 0 / 0 | 62.90 s |
| rh6 | headless | INVALID (time limit: the PMM predictor got stuck, fixed by the watchdog) | 13/49 | 0 / 0 | - |
| rh7 | headless | FINISHED | 49/49 | 0 / 0 | 64.29 s |
| gui1 | Gazebo GUI + RViz | FINISHED | 49/49 | 0 / 0 | 62.94 s |

Laps take ~8 s; the gate openings are crossed within a few centimetres of their centre.
RViz may abort (`exit code -6`) when the launch shuts it down after the race; that is an rviz2
Humble exit crash, after all results are written.

## Rulebook interpretation and assumptions

The rulebook leaves several points [TBD]. The choices below are the most likely ones and are all
configurable:

| Topic | Choice |
| --- | --- |
| Gate order | Mesh object order `Cube, Cube.001, …, Cube.006` — a clockwise loop (seen from above) starting at the gate next to the takeoff base (`order:` in the course file). |
| Gate direction | The plane normal pointing from the previous gate to the next one (the start position precedes gate 1). |
| Laps / finish | 7 laps (section 3.2); the run ends when the drone lands on the base after the last gate of lap 7. |
| Start event | The `START` race event (automatic when ready, or `/race/start`); takeoff counts in the race time. |
| State estimation | Ground truth in simulation (section 3.1): the P3D ground truth feeds the LUS estimation manager with the hardware-API conventions. `odometry_source:=px4` keeps the stock PX4-EKF path. |
| Penalties | A hit is one contact episode with a gate structure; crossing a gate while touching it is a hit **and** a missed entry (the gate does not count); passing beside the expected gate is a missed entry that must be retried; 5 s each. |
| Invalid run | Leaving the 5 × 5 × 2 m envelope, a crash (on the floor outside the base, or attitude lost), the time limit, or never retrying a missed gate. |
| Drone hull | Disk of 7 cm radius (rotor tips) and 5 cm thickness for contact checks. |

The official timing and scoring come from the organisers; our referee is a faithful estimate for
development.

## Configuration and tuning

| File | What to tune |
| --- | --- |
| [`config/race.yaml`](ros2_ws/src/icuas27_bringup/config/race.yaml) | race speed, approach/exit distances, heading mode, final hover height, validation margins, recovery, timeouts, referee constants |
| [`config/lus/pmm_crazyflie.yaml`](ros2_ws/src/icuas27_bringup/config/lus/pmm_crazyflie.yaml) | PMM `max_accel` (thrust acceleration, 12 m/s² of the 14.3 available) and `max_vel` |
| [`config/lus/control_manager.yaml`](ros2_ws/src/icuas27_bringup/config/lus/control_manager.yaml) | takeoff height/speed, landing speed, goal tolerance, safe area (written per run from the course) |
| [`config/courses/*.yaml`](ros2_ws/src/icuas27_bringup/config/courses) | gates, order, laps, arena, landing base |

Offline sweep of plan parameters with the exact PMM trajectory (no simulator):

```bash
ros2 run icuas27_race plan_course --start-predictor --speed 2.5 3.0 --approach 0.25 0.35 --exit 0.25 0.35
```

Faster laps come from `max_accel`/`max_vel`; robustness from the approach/exit distances (straight
segment through each gate) and the validation margins. Compare settings with `./scripts/batch.sh`.

### Why these LUS settings

Measured in this LUS snapshot (details in the commit history and run reports):

- **Control mode `rates`** (`sim.launch.py control_mode:=rates`, default): the Crazyflie NMPC +
  INDI (`individual_thrust`) diverges in attitude right after takeoff, with ground-truth and with
  PX4 estimation alike. The NMPC `angular_rates_and_thrust` model with the PX4 rate controller
  flies the course reliably. `control_mode:=individual_thrust` keeps the stock configuration.
- **Receding horizon**: the planning manager times a new trajectory from the moment its
  generation starts, so the 148-waypoint full-race PMM (~0.5 s to generate) makes the reference
  jump ahead of the drone at the start (gate 1 hit). Short missions generate in ~0.1 s. Each new
  segment starts from the commanded reference at dispatch; the race manager measures its own
  planning latency, extrapolates the reference by it, drops the waypoints it would already have
  passed, and re-checks just before dispatch, so a segment never starts behind its first waypoint
  (which would fly back through the gate exit).
- **PMM caveats**: when its first optimisation stage fails the pinned PMM falls back to a
  non-deterministic path (detected from its log; such segments are rejected), and its thrust
  model allows inverted thrust. The validation checks the thrust acceleration (≤ 14 m/s²) along the
  predicted trajectory. `planning_manager.yaml` can switch to MINCO. It can also, rarely, loop
  forever on an input: the predictor has a watchdog (`watchdog_timeout`, 3 s) that ends the
  process, the launch respawns it, and the race manager keeps flying the current segment. The
  planning manager has no such protection (LUS code); each plan JSON stores the exact predictor
  input (`start_odom`, `mission_odom`) to reproduce a failure offline.
- **Planning latency**: plan validation runs in the race manager next to high-rate ROS
  callbacks; with Python's default 5 ms GIL switch interval each numpy call could wait that long
  (validation took 0.2-1.7 s instead of 50 ms), so the node sets a 0.1 ms switch interval.
- **Alignment and landing use the ground truth's stillness**: the MEKF velocity carries a
  ~0.26 m/s bias on the ground, so "static" and "landed" are decided from the simulator ground
  truth (or the referee's FINISHED state).

## Frames and conventions

- `world`: Gazebo world, ENU, metres/radians; the course is defined here.
- `T_a_b` is the pose of frame `b` in frame `a` (`p_a = T_a_b p_b`); quaternions are `xyzw`.
- The LUS estimation frame (`uav1/odometry`, root `uav1/cm_odometry_debug` in TF) has its origin
  at the drone resting on the base. The race manager measures `T_world_odom` from synchronised
  static ground-truth/estimation samples before takeoff, converts the course into the estimation
  frame, and publishes `world → uav1/cm_odometry_debug` so every LUS topic is visible in RViz.
- The LASER P3D ground truth is reported relative to the spawn pose (`xyz_offset = -spawn`) with the
  linear twist in the body frame; `GroundTruthAdapter` undoes both.
- Gate frame: origin at the opening centre, `x` = direction of travel, `y` = left, `z` = up.

## Findings about the competition files

- **Gate 7 has no collision geometry.** `obstacles_crazyrun.stl` (the collision mesh of the
  obstacles model) contains only 6 gates (1296 triangles = 6 × 216); `Cube.006` exists only in
  the visual OBJ. In Gazebo the drone can fly through gate 7's frame without a physical contact.
  Our referee and plan validation use the OBJ (all 7 gates). Worth reporting to the organisers.
- The simulated gates are 0.5 × 0.5 m openings in 0.6 × 0.6 m frames (the rulebook describes
  1 × 1 m openings for the real arena); the takeoff base is a 0.564 m square, 5.8 cm high.
- The LUS stable branches moved to PX4 1.17 on 2026-11-17, after the Crazyflie overlay; we stay on
  the overlay's snapshot until the organisers publish the pinned evaluation revision.

## Submission checklist (Phase 1)

- Source, build manifest and dependencies: this repository + `Dockerfile` (`docker/lus_pins.txt`
  for LUS). Third-party components: LASER UAV System (BSD-3) and its dependencies (PX4 BSD-3,
  acados BSD-2, Gazebo Apache-2.0, ROS 2 Apache-2.0).
- Automated, non-interactive entry point: `ros2 launch icuas27_bringup race_sim.launch.py headless:=true shutdown_on_finish:=true`
  (or `scripts/run_race.sh --headless` inside the image).
- Parameter files: `ros2_ws/src/icuas27_bringup/config`. No trained models.
- Computational requirements: one CPU core for the race nodes (+ the LUS/Gazebo stack, real time
  on a 4-core laptop CPU); no GPU.
- External services / network access: none at run time (the online Gazebo model database is
  disabled). The build downloads the pinned repositories.

## Development

```bash
docker compose run --rm sim                     # shell; ~/ros2_ws is this repo's ros2_ws
cd ~/ros2_ws && colcon build --symlink-install  # after changing C++/interfaces
python3 -m pytest ros2_ws/src/icuas27_race/test # pure-Python core: geometry, referee, planner, report
ros2 launch icuas27_bringup sim.launch.py       # LUS stack only (manual flights)
ros2 run icuas27_race extract_course <obstacles.obj>   # regenerate gates from a new course mesh
```

## Roadmap

- Perception with the AI-deck camera (gate detection and pose refinement) for maps that differ
  from the configured course, and for Phase 2.
- Phase 2 (real Crazyflie + motion capture): `alignment_mode:=identity`, `odometry_source` from
  mocap, re-tuned limits and safety envelope.
- Smarter time-optimal plans (per-gate approach distances, heading alignment for the camera).
