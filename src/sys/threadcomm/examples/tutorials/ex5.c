static char help[] = "Micro-benchmark kernel times.\n\n";

/*
  Include "petscthreadcomm.h" so that we can use the PetscThreadComm interface.
*/
#include <petscthreadcomm.h>
#include <petscvec.h>
#if defined(PETSC_HAVE_OPENMP)
#  include <omp.h>
#endif

static PetscErrorCode CounterInit_kernel(PetscInt trank,PetscInt **counters)
{
  counters[trank] = malloc(sizeof(PetscInt)); /* Separate allocation per thread */
  *counters[trank] = 0;                      /* Initialize memory to fault it */
  return 0;
}

static PetscErrorCode CounterIncrement_kernel(PetscInt trank,PetscInt **counters)
{
  (*counters[trank])++;
  return 0;
}

static PetscErrorCode CounterFree_kernel(PetscInt trank,PetscInt **counters)
{
  free(counters[trank]);
  return 0;
}

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc,char **argv)
{
  PetscErrorCode ierr;
  PetscInt       i,j,N=100,**counters,tsize;

  PetscInitialize(&argc,&argv,(char *)0,help);

#if defined(PETSC_THREADCOMM_ACTIVE)
  ierr = PetscThreadCommView(PETSC_COMM_WORLD,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
#endif
  ierr = PetscOptionsGetInt(PETSC_NULL,"-N",&N,PETSC_NULL);CHKERRQ(ierr);

  ierr = PetscThreadCommGetNThreads(PETSC_COMM_WORLD,&tsize);CHKERRQ(ierr);
  ierr = PetscMalloc(tsize*sizeof *counters,&counters);CHKERRQ(ierr);
  ierr = PetscThreadCommRunKernel(PETSC_COMM_WORLD,(PetscThreadKernel)CounterInit_kernel,1,counters);CHKERRQ(ierr);

  for (i=0; i<10; i++) {
    PetscReal t0,t1;
    ierr = PetscThreadCommBarrier(PETSC_COMM_WORLD);CHKERRQ(ierr);
    ierr = PetscGetTime(&t0);CHKERRQ(ierr);
    for (j=0; j<N; j++) {
      ierr = PetscThreadCommRunKernel(PETSC_COMM_WORLD,(PetscThreadKernel)CounterIncrement_kernel,1,counters);CHKERRQ(ierr);
    }
    ierr = PetscThreadCommBarrier(PETSC_COMM_WORLD);CHKERRQ(ierr);
    ierr = PetscGetTime(&t1);CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Time per kernel: %g us\n",1e6*(t1-t0)/N);CHKERRQ(ierr);
  }

  for (i=0; i<10; i++) {
    PetscReal t0,t1;
    ierr = PetscThreadCommBarrier(PETSC_COMM_WORLD);CHKERRQ(ierr);
    ierr = PetscGetTime(&t0);CHKERRQ(ierr);
    for (j=0; j<N; j++) {
#pragma omp parallel num_threads(tsize)
      {
        int trank = omp_get_thread_num();
        (*counters[trank])++;
      }
    }
    ierr = PetscThreadCommBarrier(PETSC_COMM_WORLD);CHKERRQ(ierr);
    ierr = PetscGetTime(&t1);CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_WORLD,"OpenMP inline time per kernel: %g us\n",1e6*(t1-t0)/N);CHKERRQ(ierr);
  }

    for (i=0; i<10; i++) {
    PetscReal t0,t1;
    ierr = PetscGetTime(&t0);CHKERRQ(ierr);
    for (j=0; j<N; j++) {
      (*(volatile PetscInt*)counters[0])++;
    }
    ierr = PetscGetTime(&t1);CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Serial inline time per kernel: %g us\n",1e6*(t1-t0)/N);CHKERRQ(ierr);
  }

  ierr = PetscThreadCommRunKernel(PETSC_COMM_WORLD,(PetscThreadKernel)CounterFree_kernel,1,counters);CHKERRQ(ierr);
  ierr = PetscFree(counters);CHKERRQ(ierr);
  PetscFinalize();
  return 0;
}
