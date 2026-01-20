#!/bin/bash

# Define your parameter lists
lrs=(3e-4)
bss=(8)
rds=(0.5 0.3)
hds=(0.5 0.3)
stack_options=("--stack_rgb" "") 

for lr in "${lrs[@]}"; do
    for bs in "${bss[@]}"; do
        for rd in "${rds[@]}"; do
            for hd in "${hds[@]}"; do
                for stack in "${stack_options[@]}"; do
                    
                    # The --export flag sends these variables INTO the template
                    sbatch --export=ALL,LR=$lr,BS=$bs,RD=$rd,HD=$hd,STACK="$stack" sbatch_template.sh
                    
                    echo "Submitted: LR=$lr BS=$bs RD=$rd HD=$hd $stack"
                    sleep 0.2
                done
            done
        done
    done
done