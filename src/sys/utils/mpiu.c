
#include <petscsys.h>        /*I  "petscsys.h"  I*/
#include <petsc/private/petscimpl.h>
/*
    Note that tag of 0 is ok because comm is a private communicator
  generated below just for these routines.
*/

PETSC_INTERN PetscErrorCode PetscSequentialPhaseBegin_Private(MPI_Comm comm,int ng)
{
  PetscMPIInt    rank,size,tag = 0;
  MPI_Status     status;

  PetscFunctionBegin;
  CHKERRMPI(MPI_Comm_size(comm,&size));
  if (size == 1) PetscFunctionReturn(0);
  CHKERRMPI(MPI_Comm_rank(comm,&rank));
  if (rank) {
    CHKERRMPI(MPI_Recv(NULL,0,MPI_INT,rank-1,tag,comm,&status));
  }
  /* Send to the next process in the group unless we are the last process */
  if ((rank % ng) < ng - 1 && rank != size - 1) {
    CHKERRMPI(MPI_Send(NULL,0,MPI_INT,rank + 1,tag,comm));
  }
  PetscFunctionReturn(0);
}

PETSC_INTERN PetscErrorCode PetscSequentialPhaseEnd_Private(MPI_Comm comm,int ng)
{
  PetscMPIInt    rank,size,tag = 0;
  MPI_Status     status;

  PetscFunctionBegin;
  CHKERRMPI(MPI_Comm_rank(comm,&rank));
  CHKERRMPI(MPI_Comm_size(comm,&size));
  if (size == 1) PetscFunctionReturn(0);

  /* Send to the first process in the next group */
  if ((rank % ng) == ng - 1 || rank == size - 1) {
    CHKERRMPI(MPI_Send(NULL,0,MPI_INT,(rank + 1) % size,tag,comm));
  }
  if (rank == 0) {
    CHKERRMPI(MPI_Recv(NULL,0,MPI_INT,size-1,tag,comm,&status));
  }
  PetscFunctionReturn(0);
}

/* ---------------------------------------------------------------------*/
/*
    The variable Petsc_Seq_keyval is used to indicate an MPI attribute that
  is attached to a communicator that manages the sequential phase code below.
*/
PetscMPIInt Petsc_Seq_keyval = MPI_KEYVAL_INVALID;

/*@
   PetscSequentialPhaseBegin - Begins a sequential section of code.

   Collective

   Input Parameters:
+  comm - Communicator to sequentialize.
-  ng   - Number in processor group.  This many processes are allowed to execute
   at the same time (usually 1)

   Level: intermediate

   Notes:
   PetscSequentialPhaseBegin() and PetscSequentialPhaseEnd() provide a
   way to force a section of code to be executed by the processes in
   rank order.  Typically, this is done with
.vb
      PetscSequentialPhaseBegin(comm, 1);
      <code to be executed sequentially>
      PetscSequentialPhaseEnd(comm, 1);
.ve

   Often, the sequential code contains output statements (e.g., printf) to
   be executed.  Note that you may need to flush the I/O buffers before
   calling PetscSequentialPhaseEnd().  Also, note that some systems do
   not propagate I/O in any order to the controling terminal (in other words,
   even if you flush the output, you may not get the data in the order
   that you want).

.seealso: PetscSequentialPhaseEnd()

@*/
PetscErrorCode  PetscSequentialPhaseBegin(MPI_Comm comm,int ng)
{
  PetscMPIInt    size;
  MPI_Comm       local_comm,*addr_local_comm;

  PetscFunctionBegin;
  CHKERRQ(PetscSysInitializePackage());
  CHKERRMPI(MPI_Comm_size(comm,&size));
  if (size == 1) PetscFunctionReturn(0);

  /* Get the private communicator for the sequential operations */
  if (Petsc_Seq_keyval == MPI_KEYVAL_INVALID) {
    CHKERRMPI(MPI_Comm_create_keyval(MPI_COMM_NULL_COPY_FN,MPI_COMM_NULL_DELETE_FN,&Petsc_Seq_keyval,NULL));
  }

  CHKERRMPI(MPI_Comm_dup(comm,&local_comm));
  CHKERRQ(PetscMalloc1(1,&addr_local_comm));

  *addr_local_comm = local_comm;

  CHKERRMPI(MPI_Comm_set_attr(comm,Petsc_Seq_keyval,(void*)addr_local_comm));
  CHKERRQ(PetscSequentialPhaseBegin_Private(local_comm,ng));
  PetscFunctionReturn(0);
}

/*@
   PetscSequentialPhaseEnd - Ends a sequential section of code.

   Collective

   Input Parameters:
+  comm - Communicator to sequentialize.
-  ng   - Number in processor group.  This many processes are allowed to execute
   at the same time (usually 1)

   Level: intermediate

   Notes:
   See PetscSequentialPhaseBegin() for more details.

.seealso: PetscSequentialPhaseBegin()

@*/
PetscErrorCode  PetscSequentialPhaseEnd(MPI_Comm comm,int ng)
{
  PetscMPIInt    size,flag;
  MPI_Comm       local_comm,*addr_local_comm;

  PetscFunctionBegin;
  CHKERRMPI(MPI_Comm_size(comm,&size));
  if (size == 1) PetscFunctionReturn(0);

  CHKERRMPI(MPI_Comm_get_attr(comm,Petsc_Seq_keyval,(void**)&addr_local_comm,&flag));
  PetscCheckFalse(!flag,PETSC_COMM_SELF,PETSC_ERR_ARG_INCOMP,"Wrong MPI communicator; must pass in one used with PetscSequentialPhaseBegin()");
  local_comm = *addr_local_comm;

  CHKERRQ(PetscSequentialPhaseEnd_Private(local_comm,ng));

  CHKERRQ(PetscFree(addr_local_comm));
  CHKERRMPI(MPI_Comm_free(&local_comm));
  CHKERRMPI(MPI_Comm_delete_attr(comm,Petsc_Seq_keyval));
  PetscFunctionReturn(0);
}

/*@C
  PetscGlobalMinMaxInt - Get the global min/max from local min/max input

  Collective

  Input Parameter:
. minMaxVal - An array with the local min and max

  Output Parameter:
. minMaxValGlobal - An array with the global min and max

  Level: beginner

.seealso: PetscSplitOwnership()
@*/
PetscErrorCode PetscGlobalMinMaxInt(MPI_Comm comm, const PetscInt minMaxVal[2], PetscInt minMaxValGlobal[2])
{
  PetscInt       sendbuf[3],recvbuf[3];

  PetscFunctionBegin;
  sendbuf[0] = -minMaxVal[0]; /* Note that -PETSC_MIN_INT = PETSC_MIN_INT */
  sendbuf[1] = minMaxVal[1];
  sendbuf[2] = (minMaxVal[0] == PETSC_MIN_INT) ? 1 : 0; /* Are there PETSC_MIN_INT in minMaxVal[0]? */
  CHKERRMPI(MPI_Allreduce(sendbuf, recvbuf, 3, MPIU_INT, MPI_MAX, comm));
  minMaxValGlobal[0] = recvbuf[2] ? PETSC_MIN_INT : -recvbuf[0];
  minMaxValGlobal[1] = recvbuf[1];
  PetscFunctionReturn(0);
}

/*@C
  PetscGlobalMinMaxReal - Get the global min/max from local min/max input

  Collective

  Input Parameter:
. minMaxVal - An array with the local min and max

  Output Parameter:
. minMaxValGlobal - An array with the global min and max

  Level: beginner

.seealso: PetscSplitOwnership()
@*/
PetscErrorCode PetscGlobalMinMaxReal(MPI_Comm comm, const PetscReal minMaxVal[2], PetscReal minMaxValGlobal[2])
{
  PetscReal      sendbuf[2];

  PetscFunctionBegin;
  sendbuf[0] = -minMaxVal[0];
  sendbuf[1] = minMaxVal[1];
  CHKERRMPI(MPIU_Allreduce(sendbuf,minMaxValGlobal,2,MPIU_REAL,MPIU_MAX,comm));
  minMaxValGlobal[0] = -minMaxValGlobal[0];
  PetscFunctionReturn(0);
}
