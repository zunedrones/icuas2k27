# Live state of the referee (/race/referee/status).

uint8 WAITING=0     # waiting for the start command
uint8 RUNNING=1
uint8 FINISHED=2    # complete and valid run
uint8 INVALID=3     # run invalidated (crash, out of bounds, timeout, aborted)

std_msgs/Header header
uint8 state
float64 race_time         # elapsed time since the start command (s)
int32 next_gate_index     # 0-based index of the expected gate in the race sequence
int32 next_gate_id
int32 lap                 # 1-based current lap
int32 laps_total
int32 gates_passed        # legal passes so far
int32 gates_total
int32 hits                # contacts with obstacles (penalties)
int32 missed_entries      # missed gate entries (penalties)
int32 wrong_way           # gates entered through the exit (do not count)
float64 penalty_time      # (hits + missed_entries) * penalty per incident
float64 adjusted_time     # race_time + penalty_time
float64 speed             # current speed (m/s)
float64 min_clearance     # smallest hull-to-obstacle distance of the run so far (m)
string detail
