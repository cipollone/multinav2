#!/bin/bash

# Use --watch to see what happens, omit, for a fast simulation.

# This variable is read by the docker environment
if [ "$1" == "--watch" ]; then
	export GUIOPTION=""
else
	export GUIOPTION="--no_gui"
fi

# Set these variables based on the location of these repositories
#bitbucket.org:iocchi/marrtino_apps.git
export MARTINO_APPS_PATH=$HOME/repos/marrtino_apps
export STAGE_ENVIRONMENTS=$HOME/repos/stage_environments
export STAGE_CONTROLS=$HOME/repos/stage-controls

# These are constants
export ROS_IP=127.0.0.1
export ROBOT_TYPE=stage

# Start image and server
docker-compose -f ./environment up &
sleep 3

# Start simulation
if [ "$1" == "--watch" ]; then
	echo 'montreal_cp;marrtino' | netcat -w 1 localhost 9235
else
	docker exec -it stage bash -ci "rosrun stage_environments start_simulation.py --no_gui montreal_cp marrtino"
fi
echo '@gbn' | netcat -w 1 localhost 9238

# Go faster?
sleep 2
if [ "$1" != "--watch" ]; then
	docker exec -it navigation bash -ci "rostopic pub --once /stageGUIRequest std_msgs/String 'data: speedup_4'"
fi
