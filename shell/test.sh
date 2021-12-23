#!/bin/bash
#SBATCH -J sleep                  
#SBATCH -p bme_gpu               
#SBATCH --time=1               
#SBATCH -N 2                       
#SBATCH -n 2                        
#SBATCH -o out/%j.sleep   
#SBATCH -e out/%j.sleep   
 
echo ${SLURM_JOB_NODELIST}   
echo  start on $(date)                      
sleep 100                                          
echo end on $(date)                       
