
#ifndef lint
static char vcid[] = "$Id: mpitr.c,v 1.2 1996/07/15 03:43:57 bsmith Exp bsmith $";
#endif

/*
    Code for tracing mistakes in MPI usage. For example, sends that are never received,
  nonblocking messages that are not correctly waited for, etc.
*/

#include <stdio.h>
#include "petsc.h"           /*I "petsc.h" I*/

#if defined(PETSC_LOG) && !defined(PETSC_USING_MPIUNI)
/*@C
   PetscMPIDump - Dumps a listing of incomplete MPI operations. Sends never received, etc

   Input Parameter:
.  fp  - file pointer.  If fp is NULL, stderr is assumed.

   Options Database Key:
$  -mpidump : dumps MPI incompleteness during call to PetscFinalize()

.keywords: MPI errors

.seealso:  PetscTrDump()
 @*/
int PetscMPIDump(FILE *fd)
{
  int    rank;
  double tsends,trecvs,work;

  MPI_Comm_rank(MPI_COMM_WORLD,&rank);
  if (fd == 0) fd = stderr;
   
  /* Did we wait on all the non-blocking sends and receives? */
  PetscSequentialPhaseBegin(MPI_COMM_WORLD,1 );
  if (irecv_ct + isend_ct != sum_of_waits_ct) {
    fprintf(fd,"[%d]You have not waited on all non-blocking sends and receives",rank);
    fprintf(fd,"[%d]Number non-blocking sends %g receives %g number of waits %g\n",rank,isend_ct,
            irecv_ct,sum_of_waits_ct);
    fflush(fd);
  }
  PetscSequentialPhaseEnd(MPI_COMM_WORLD,1 );
  /* Did we receive all the messages that we sent? */
  work = irecv_ct + recv_ct;
  MPI_Reduce(&work,&trecvs,1,MPI_DOUBLE,MPI_SUM,0,MPI_COMM_WORLD);
  work = isend_ct + send_ct;
  MPI_Reduce(&work,&tsends,1,MPI_DOUBLE,MPI_SUM,0,MPI_COMM_WORLD);
  if (!rank && tsends != trecvs) {
    fprintf(fd,"Total number sends %g not equal receives %g\n",tsends,trecvs);
    fflush(fd);
  }
  return 0;
}

#else

int PetscMPIDump(FILE *fd)
{
  return 0;
}

#endif









