/*
    Code for tracing mistakes in MPI usage. For example, sends that are never received,
  nonblocking messages that are not correctly waited for, etc.
*/

#include <petscsys.h> /*I "petscsys.h" I*/

#if defined(PETSC_USE_LOG) && !defined(PETSC_HAVE_MPIUNI)

/*@C
  PetscMPIDump - Dumps a listing of incomplete MPI operations, such as sends that
  have never been received, etc.

  Collective on `PETSC_COMM_WORLD`

  Input Parameter:
. fd - file pointer.  If fp is `NULL`, `stdout` is assumed.

  Options Database Key:
. -mpidump - Dumps MPI incompleteness during call to PetscFinalize()

  Level: developer

.seealso: `PetscMallocDump()`
 @*/
PetscErrorCode PetscMPIDump(FILE *fd)
{
  PetscMPIInt rank;
  double      tsends, trecvs, work;

  PetscFunctionBegin;
  PetscCallMPI(MPI_Comm_rank(PETSC_COMM_WORLD, &rank));
  if (!fd) fd = PETSC_STDOUT;

  /* Did we wait on all the non-blocking sends and receives? */
  PetscCall(PetscSequentialPhaseBegin(PETSC_COMM_WORLD, 1));
  if (petsc_irecv_ct + petsc_isend_ct != petsc_sum_of_waits_ct) {
    PetscCall(PetscFPrintf(PETSC_COMM_SELF, fd, "[%d]You have not waited on all non-blocking sends and receives", rank));
    PetscCall(PetscFPrintf(PETSC_COMM_SELF, fd, "[%d]Number non-blocking sends %g receives %g number of waits %g\n", rank, petsc_isend_ct, petsc_irecv_ct, petsc_sum_of_waits_ct));
    PetscCall(PetscFFlush(fd));
  }
  PetscCall(PetscSequentialPhaseEnd(PETSC_COMM_WORLD, 1));
  /* Did we receive all the messages that we sent? */
  work = petsc_irecv_ct + petsc_recv_ct;
  PetscCallMPI(MPI_Reduce(&work, &trecvs, 1, MPI_DOUBLE, MPI_SUM, 0, PETSC_COMM_WORLD));
  work = petsc_isend_ct + petsc_send_ct;
  PetscCallMPI(MPI_Reduce(&work, &tsends, 1, MPI_DOUBLE, MPI_SUM, 0, PETSC_COMM_WORLD));
  if (rank == 0 && tsends != trecvs) {
    PetscCall(PetscFPrintf(PETSC_COMM_SELF, fd, "Total number sends %g not equal receives %g\n", tsends, trecvs));
    PetscCall(PetscFFlush(fd));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

#else

PetscErrorCode PetscMPIDump(FILE *fd)
{
  PetscFunctionBegin;
  PetscFunctionReturn(PETSC_SUCCESS);
}

#endif
