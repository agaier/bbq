#!/bin/bash
for id in {0..3}
do
    for rep in {0..5}
    do
    python3 launch_instance.py --id=$id --rep=$rep &
    done
done