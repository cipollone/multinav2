#!/bin/bash

if [[ $# -eq 0 ]] ; then
	echo 'Arguments: image-name [option]'

else
	name=${1//\//-}-1
	docker run -dit --rm \
		--user=$UID:`id -g` \
		-v /home/cipollor:/home/cipollor \
		--name=$name \
		${@:2} $1

	sleep 3s
	docker exec -it $name bash -c "tmux attach -t main"
fi
