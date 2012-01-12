
#ifndef __MPIVECPTHREADIMPL
#define __MPIVECPTHREADIMPL

#include <private/vecimpl.h>
#include <../src/vec/vec/impls/mpi/pvecimpl.h>
#include <../src/vec/vec/impls/seq/seqpthread/vecpthreadimpl.h>

typedef struct {
  VECHEADER
  PetscInt nthreads;      /* Number of threads */
  PetscInt *arrindex;     /* Starting array indices for each thread */
  PetscInt *nelem;        /* Number of array elements assigned to each thread */
  PetscInt *cpu_affinity; /* CPU affinities */

  MPI_Request *send_waits,*recv_waits;  /* for communication during VecAssembly() */
  PetscInt    nsends,nrecvs;
  PetscScalar *svalues,*rvalues;
  PetscInt    rmax;
  
  PetscInt    nghost;                   /* length of local portion including ghost padding */
  
  Vec         localrep;                 /* local representation of vector */
  VecScatter  localupdate;              /* scatter to update ghost values */
} Vec_MPIPthread;

#endif
