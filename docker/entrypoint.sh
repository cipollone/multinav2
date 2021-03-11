#!/bin/bash
tmux new-session -s main -d

# The user is expected to attach to this same session
#   with tmux attach -t main
# This allows me to keep this process open until
#   tmux is not completely closed

until ! tmux has-session -t main
do
	sleep 10s
done
