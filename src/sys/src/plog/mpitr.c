/*$Id: mpitr.c,v 1.29 2001/03/23 23:20:50 balay Exp $*/

/*
    Code for tracing mistakes in MPI usage. For example, sends that are never received,
  nonblocking messages that are not correctly waited for, etc.
*/

#include "petscconfig.h"
#include "petsc.h"           /*I "petsc.h" I*/

#if defined(PETSC_USE_LOG) && !defined(PETSC_HAVE_MPI_UNI)

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

.seealso:  PetscTrDump()
 @*/
int PetscMPIDump(FILE *fd)
{
  int    rank,ierr;
  double tsends,trecvs,work;

  PetscFunctionBegin;
  ierr = MPI_Comm_rank(PETSC_COMM_WORLD,&rank);CHKERRQ(ierr);
  if (!fd) fd = stdout;
   
  /* Did we wait on all the non-blocking sends and receives? */
  ierr = PetscSequentialPhaseBegin(PETSC_COMM_WORLD,1);CHKERRQ(ierr);
  if (irecv_ct + isend_ct != sum_of_waits_ct) {
    fprintf(fd,"[%d]You have not waited on all non-blocking sends and receives",rank);
    fprintf(fd,"[%d]Number non-blocking sends %g receives %g number of waits %g\n",rank,isend_ct,
            irecv_ct,sum_of_waits_ct);
    fflush(fd);
  }
  ierr = PetscSequentialPhaseEnd(PETSC_COMM_WORLD,1);CHKERRQ(ierr);
  /* Did we receive all the messages that we sent? */
  work = irecv_ct + recv_ct;
  ierr = MPI_Reduce(&work,&trecvs,1,MPI_DOUBLE,MPI_SUM,0,PETSC_COMM_WORLD);CHKERRQ(ierr);
  work = isend_ct + send_ct;
  ierr = MPI_Reduce(&work,&tsends,1,MPI_DOUBLE,MPI_SUM,0,PETSC_COMM_WORLD);CHKERRQ(ierr);
  if (!rank && tsends != trecvs) {
    fprintf(fd,"Total number sends %g not equal receives %g\n",tsends,trecvs);
    fflush(fd);
  }
  PetscFunctionReturn(0);
}

#else

#undef __FUNCT__  
#define __FUNCT__ "PetscMPIDump"
int PetscMPIDump(FILE *fd)
{
  PetscFunctionBegin;
  PetscFunctionReturn(0);
}

#endif









