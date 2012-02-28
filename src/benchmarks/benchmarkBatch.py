#!/usr/bin/env python
def generateBatchScript(num, procs, time, *args):
  body = '''\
#!/bin/bash
#PBS -N ex%d_GPU_test
#PBS -l walltime=%02d:%02d:%02d
#PBS -l nodes=%d:ppn=1
#PBS -j oe
cd $PBS_O_WORKDIR
echo Master process running on `hostname`
echo Directory is `pwd`
echo PBS has allocated the following nodes:
echo `cat $PBS_NODEFILE`
echo Starting execution at `date`
NPROCS=`wc -l < $PBS_NODEFILE`
echo This job has allocated $NPROCS CPUs
# execute an MPI program
echo Executing mpirun -np $NPROCS ex%d %s
mpirun -np $NPROCS ex%d %s
''' % (num, (time%86400)/3600, (time%3600)/60, time%60, procs, num, ' '.join(args), num, ' '.join(args))
  namePattern = 'ex%d_%03d.batch'
  for n in range(1000):
    try:
      filename = namePattern % (num, n)
      f = file(filename)
      f.close()
      n += 1
    except IOError, e:
      if e.errno == 2:
        break
      else:
        raise e
  with file(filename, 'w') as f:
    f.write(body)
  return filename

if __name__ == '__main__':
  # Waiting for argparse in 2.7
  import sys
  num   = int(sys.argv[1])
  time  = int(sys.argv[2]) # in seconds
  procs = int(sys.argv[3])
  #args  = ['-da_grid_x 800', '-da_grid_y 800', '-log_summary',  '-log_summary_python']
  generateBatchScript(num, procs, time, *sys.argv[4:])
