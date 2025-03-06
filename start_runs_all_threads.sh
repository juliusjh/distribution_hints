#!/usr/bin/bash

# leakfile="../configs/leakage/nt_leakage_hw.json"
# setfile="../configs/settings/nt_hw.json"
# start=0
# end=9
# threads=19

leakfile="../configs/leakage/$1"
setfile="../configs/settings/$2"
count=$3
sname=$4
threads=$5
th=$(((threads + count - 1)/count))

tmux new-session -d -s "$sname"

for (( i=0; i<$count; i++ ))
do
    e=$((i+1))

    tmux new-window -t "$sname:$e"

    tmux send-keys -t "$sname:$e" "source .env/bin/activate" C-m
    tmux send-keys -t "$sname:$e" "cd src" C-m
    tmux send-keys -t "$sname:$e" "python eval.py --threads $th --parts $i $e $leakfile $setfile"
done

for (( i=0; i<$count; i++ ))
do
    e=$((i+1))
    tmux send-keys -t "$sname:$e" C-m
done
