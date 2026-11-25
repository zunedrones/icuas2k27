# Periodic state of the race manager (/race/status).

std_msgs/Header header
string state               # INIT, WAITING_SYSTEM, ALIGNING, READY, ARMING, TAKING_OFF, RACING,
                           # LANDING, FINISHED, ABORTED
float64 elapsed            # seconds since START (0 before the start command)
uint64 mission_id          # planning-manager mission carrying the race trajectory
float64 planned_duration   # predicted duration of the race trajectory (s), 0 if unknown
uint32 recoveries          # number of re-plans sent to retry missed gates
string detail
