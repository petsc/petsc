
#include <petscsys.h>        /*I  "petscsys.h"  I*/
/*
    Note that tag of 0 is ok because comm is a private communicator
  generated below just for these routines.
*/

#undef __FUNCT__
#define __FUNCT__ "PetscSequentialPhaseBegin_Private"
PetscErrorCode PetscSequentialPhaseBegin_Private(MPI_Comm comm,int ng)
{
  PetscErrorCode ierr;
  PetscMPIInt    rank,size,tag = 0;
  MPI_Status     status;

  PetscFunctionBegin;
  ierr = MPI_Comm_size(comm,&size);CHKERRQ(ierr);
  if (size == 1) PetscFunctionReturn(0);
  ierr = MPI_Comm_rank(comm,&rank);CHKERRQ(ierr);
  if (rank) {
    ierr = MPI_Recv(0,0,MPI_INT,rank-1,tag,comm,&status);CHKERRQ(ierr);
  }
  /* Send to the next process in the group unless we are the last process */
  if ((rank % ng) < ng - 1 && rank != size - 1) {
    ierr = MPI_Send(0,0,MPI_INT,rank + 1,tag,comm);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscSequentialPhaseEnd_Private"
PetscErrorCode PetscSequentialPhaseEnd_Private(MPI_Comm comm,int ng)
{
  PetscErrorCode ierr;
  PetscMPIInt    rank,size,tag = 0;
  MPI_Status     status;

  PetscFunctionBegin;
  ierr = MPI_Comm_rank(comm,&rank);CHKERRQ(ierr);
  ierr = MPI_Comm_size(comm,&size);CHKERRQ(ierr);
  if (size == 1) PetscFunctionReturn(0);

  /* Send to the first process in the next group */
  if ((rank % ng) == ng - 1 || rank == size - 1) {
    ierr = MPI_Send(0,0,MPI_INT,(rank + 1) % size,tag,comm);CHKERRQ(ierr);
  }
  if (!rank) {
    ierr = MPI_Recv(0,0,MPI_INT,size-1,tag,comm,&status);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

/* ---------------------------------------------------------------------*/
/*
    The variable Petsc_Seq_keyval is used to indicate an MPI attribute that
  is attached to a communicator that manages the sequential phase code below.
*/
static int Petsc_Seq_keyval = MPI_KEYVAL_INVALID;

#undef __FUNCT__
#define __FUNCT__ "PetscSequentialPhaseBegin"
/*@
   PetscSequentialPhaseBegin - Begins a sequential section of code.

   Collective on MPI_Comm

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

   Concepts: sequential stage

@*/
PetscErrorCode  PetscSequentialPhaseBegin(MPI_Comm comm,int ng)
{
  PetscErrorCode ierr;
  PetscMPIInt    size;
  MPI_Comm       local_comm,*addr_local_comm;

  PetscFunctionBegin;
  ierr = MPI_Comm_size(comm,&size);CHKERRQ(ierr);
  if (size == 1) PetscFunctionReturn(0);

  /* Get the private communicator for the sequential operations */
  if (Petsc_Seq_keyval == MPI_KEYVAL_INVALID) {
    ierr = MPI_Keyval_create(MPI_NULL_COPY_FN,MPI_NULL_DELETE_FN,&Petsc_Seq_keyval,0);CHKERRQ(ierr);
  }

  ierr = MPI_Comm_dup(comm,&local_comm);CHKERRQ(ierr);
  ierr = PetscMalloc(sizeof(MPI_Comm),&addr_local_comm);CHKERRQ(ierr);
  *addr_local_comm = local_comm;
  ierr = MPI_Attr_put(comm,Petsc_Seq_keyval,(void*)addr_local_comm);CHKERRQ(ierr);
  ierr = PetscSequentialPhaseBegin_Private(local_comm,ng);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscSequentialPhaseEnd"
/*@
   PetscSequentialPhaseEnd - Ends a sequential section of code.

   Collective on MPI_Comm

   Input Parameters:
+  comm - Communicator to sequentialize.
-  ng   - Number in processor group.  This many processes are allowed to execute
   at the same time (usually 1)

   Level: intermediate

   Notes:
   See PetscSequentialPhaseBegin() for more details.

.seealso: PetscSequentialPhaseBegin()

   Concepts: sequential stage

@*/
PetscErrorCode  PetscSequentialPhaseEnd(MPI_Comm comm,int ng)
{
  PetscErrorCode ierr;
  PetscMPIInt    size,flag;
  MPI_Comm       local_comm,*addr_local_comm;

  PetscFunctionBegin;
  ierr = MPI_Comm_size(comm,&size);CHKERRQ(ierr);
  if (size == 1) PetscFunctionReturn(0);

  ierr = MPI_Attr_get(comm,Petsc_Seq_keyval,(void **)&addr_local_comm,&flag);CHKERRQ(ierr);
  if (!flag) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_INCOMP,"Wrong MPI communicator; must pass in one used with PetscSequentialPhaseBegin()");
  local_comm = *addr_local_comm;

  ierr = PetscSequentialPhaseEnd_Private(local_comm,ng);CHKERRQ(ierr);

  ierr = PetscFree(addr_local_comm);CHKERRQ(ierr);
  ierr = MPI_Comm_free(&local_comm);CHKERRQ(ierr);
  ierr = MPI_Attr_delete(comm,Petsc_Seq_keyval);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
