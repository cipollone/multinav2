#!/bin/bash

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

# Start
docker-compose -f ./environment up &
sleep 3
echo 'montreal_cp;marrtino' | netcat -w 1 localhost 9235
echo '@gbn' | netcat -w 1 localhost 9238

sleep 2
if [ "$1" != "--watch" ]; then
	# Go faster
	rostopic pub /stageGUIRequest std_msgs/String "data: 'speedup_4'"
fi
