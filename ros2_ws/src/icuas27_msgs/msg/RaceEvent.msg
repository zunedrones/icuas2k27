# Milestones of a race run, published by the race manager on /race/events
# (reliable, transient local, so late subscribers such as the referee get the history).

uint8 START=0          # start command accepted: the race clock starts at header.stamp
uint8 ARMED=1
uint8 TAKEOFF=2        # takeoff requested
uint8 TAKEOFF_DONE=3   # control manager ready for tracking
uint8 RACE_SENT=4      # race trajectory sent to the planning manager
uint8 RECOVERY=5       # a new trajectory was sent to retry a missed gate
uint8 LANDING=6        # landing requested
uint8 LANDED=7
uint8 ABORTED=8

std_msgs/Header header   # stamp = event time (ROS/simulation clock)
uint8 type
string detail
