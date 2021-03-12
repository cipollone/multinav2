#!/bin/bash
if [[ $# -eq 0 ]] ; then
	echo 'Arguments: running-container-name'

else
	docker exec -it $1 bash -c "tmux attach -t main"
fi
