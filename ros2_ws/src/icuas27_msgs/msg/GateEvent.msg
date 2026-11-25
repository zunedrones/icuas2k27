# Event detected by the referee from the simulator ground truth (/race/referee/events).

uint8 PASS=0           # legal crossing of the expected gate (correct order and direction, no contact)
uint8 HIT=1            # contact with a gate structure (time penalty)
uint8 MISSED=2         # expected gate not entered legally (passed beside it, or crossed it while
                         # touching); a legal retry is required and the missed entry is penalised
uint8 WRONG_WAY=3      # a gate was entered through its exit
uint8 OUT_OF_ORDER=4   # legal-direction crossing of a gate that was not the expected one
uint8 OUT_OF_BOUNDS=5  # the drone left the flight envelope (run invalid)
uint8 CRASH=6          # severe collision / crash (run invalid)
uint8 FINISH=7         # landed on the landing base after the last gate of the last lap
uint8 TIMEOUT=8        # the run exceeded the time limit (run invalid)

std_msgs/Header header
uint8 type
int32 gate_id            # course gate id, -1 when not applicable
int32 gate_index         # 0-based index in the race sequence (laps x gates), -1 when not applicable
int32 lap                # 1-based lap of the expected gate
float64 race_time        # seconds since the start command
geometry_msgs/Point position  # world position where the event happened
float64 offset_y         # crossing point in the gate frame (m): horizontal, +left of travel
float64 offset_z         # crossing point in the gate frame (m): vertical, +up
float64 speed            # m/s at the event
float64 clearance        # distance between the drone hull and the closest obstacle (m), <0 = contact
string detail
