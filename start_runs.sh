#!/usr/bin/bash

# leakfile="../configs/leakage/nt_leakage_hw.json"
# setfile="../configs/settings/nt_hw.json"
# start=0
# end=9
# threads=19

leakfile="../$1"
setfile="../$2"
start=$3
end=$4
step=$5
threads=$6
sname=$7

tmux new-session -d -s "$sname"

for (( i=$start; i<$end; i++ ))
do
    s=$((i*step))
    e=$((s+step))
    w=$((i+1))

    tmux new-window -t "$sname:$w"

    # tmux send-keys -t "$sname:$w" "cd kyber-noiseterm" C-m
    tmux send-keys -t "$sname:$w" "source .env/bin/activate" C-m
    tmux send-keys -t "$sname:$w" "cd src" C-m
    tmux send-keys -t "$sname:$w" "python eval.py --threads $threads --parts $s $e $leakfile $setfile"
done

for (( i=$start; i<$end; i++ ))
do
    s=$((i*3))
    e=$((s+3))
    w=$((i+1))
    tmux send-keys -t "$sname:$w" C-m
done
