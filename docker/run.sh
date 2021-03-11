#!/bin/bash

if [[ $# -eq 0 ]] ; then
	echo 'Arguments: image-name [option]'

else
	docker run -it --rm \
		--user=$UID:`id -g` \
		-v /home/cipollor:/home/cipollor \
		${@:2} $1
fi
