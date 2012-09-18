
/*
    Code for tracing mistakes in MPI usage. For example, sends that are never received,
  nonblocking messages that are not correctly waited for, etc.
*/

#include <petscsys.h>           /*I "petscsys.h" I*/

#if defined(PETSC_USE_LOG) && !defined(__MPIUNI_H)

#undef __FUNCT__
#define __FUNCT__ "PetscMPIDump"
/*@C
   PetscMPIDump - Dumps a listing of incomplete MPI operations, such as sends that
   have never been received, etc.

   Collective on PETSC_COMM_WORLD

   Input Parameter:
.  fp - file pointer.  If fp is NULL, stdout is assumed.

   Options Database Key:
.  -mpidump - Dumps MPI incompleteness during call to PetscFinalize()

    Level: developer

.seealso:  PetscMallocDump()
 @*/
PetscErrorCode  PetscMPIDump(FILE *fd)
{
  PetscErrorCode ierr;
  PetscMPIInt    rank;
  double         tsends,trecvs,work;
  int            err;

  PetscFunctionBegin;
  ierr = MPI_Comm_rank(PETSC_COMM_WORLD,&rank);CHKERRQ(ierr);
  if (!fd) fd = PETSC_STDOUT;

  /* Did we wait on all the non-blocking sends and receives? */
  ierr = PetscSequentialPhaseBegin(PETSC_COMM_WORLD,1);CHKERRQ(ierr);
  if (petsc_irecv_ct + petsc_isend_ct != petsc_sum_of_waits_ct) {
    ierr = PetscFPrintf(PETSC_COMM_SELF,fd,"[%d]You have not waited on all non-blocking sends and receives",rank);CHKERRQ(ierr);
    ierr = PetscFPrintf(PETSC_COMM_SELF,fd,"[%d]Number non-blocking sends %g receives %g number of waits %g\n",rank,petsc_isend_ct,petsc_irecv_ct,petsc_sum_of_waits_ct);CHKERRQ(ierr);
    err = fflush(fd);
    if (err) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SYS,"fflush() failed on file");
  }
  ierr = PetscSequentialPhaseEnd(PETSC_COMM_WORLD,1);CHKERRQ(ierr);
  /* Did we receive all the messages that we sent? */
  work = petsc_irecv_ct + petsc_recv_ct;
  ierr = MPI_Reduce(&work,&trecvs,1,MPI_DOUBLE,MPI_SUM,0,PETSC_COMM_WORLD);CHKERRQ(ierr);
  work = petsc_isend_ct + petsc_send_ct;
  ierr = MPI_Reduce(&work,&tsends,1,MPI_DOUBLE,MPI_SUM,0,PETSC_COMM_WORLD);CHKERRQ(ierr);
  if (!rank && tsends != trecvs) {
    ierr = PetscFPrintf(PETSC_COMM_SELF,fd,"Total number sends %g not equal receives %g\n",tsends,trecvs);CHKERRQ(ierr);
    err = fflush(fd);
    if (err) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SYS,"fflush() failed on file");
  }
  PetscFunctionReturn(0);
}

#else

#undef __FUNCT__
#define __FUNCT__ "PetscMPIDump"
PetscErrorCode  PetscMPIDump(FILE *fd)
{
  PetscFunctionBegin;
  PetscFunctionReturn(0);
}

#endif









