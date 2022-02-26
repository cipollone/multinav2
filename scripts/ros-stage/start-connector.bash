#!/bin/bash

# Call this script as:
#  start-compose.bash {1, 2, 3, ..}
#  Where integers select one of the action sets available (try 1).

docker exec -it navigation bash -ci "cd /home/robot/src/stage_controls/ && ./start.bash $1"
