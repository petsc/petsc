/*$Id: mpimesg.c,v 1.99 2001/01/15 21:44:00 bsmith Exp $*/

#include "petsc.h"        /*I  "petsc.h"  I*/

/*
  PetscGatherMessageLengths - Computes info about messages that a MPI-node will receive, 
  including (from-id,length) pairs for each message.

  Collective on MPI_Comm

  Input Parameters:
+ comm     - Communicator
- ilengths - an array of integers of length 'size of comm' 
             a non zero ilenghts[i] represent a message to i of length ilenghts[i] 

  Output Parameters:
+ nrecvs    - number of messages received
. onodes    - list of node-ids from which messages are expected
- olengths  - corresponding message lengths

  Notes:
  With this info, the correct MPI_Irecv() can be posted with the correct
  from-id, with a buffer with the right amount of memory required.

*/
#undef __FUNC__  
#define __FUNC__ "PetscGatherMessageLengths"
int PetscGatherMessageLengths(MPI_Comm comm,int *ilengths, int *nrecvs, int *onodes, int *olengths)
{
  int *tmp,size,rank,i,j,nsends,tag,ierr;
  MPI_Request *s_waits,*r_waits;
  MPI_Status  *s_status,*r_status;

  PetscFunctionBegin;
  ierr = MPI_Comm_size(comm,&size);CHKERRQ(ierr); /* eliminate? */
  ierr = MPI_Comm_rank(comm,&rank);CHKERRQ(ierr); /* eliminate? */
  ierr = PetscCommGetNewTag(comm,&tag);CHKERRQ(ierr); /* eliminate? */
  for (nsends=0,i=0; i<size; i++) { if (ilengths[i]) nsends++; } /*eliminate? */
  
  ierr = PetscMalloc(size*sizeof(int),&tmp);CHKERRQ(ierr);
  
  /* Now post an allreduce to determine the numer of messages the current node will receive */
  ierr    = MPI_Allreduce(ilengths,tmp,size,MPI_INT,MPI_SUM,comm);CHKERRQ(ierr);
  *nrecvs = tmp[rank];
  
  /* Now post the Irecv to get the message length-info */
  ierr = PetscMalloc((*nrecvs+1)*sizeof(int),&olengths);CHKERRQ(ierr);
  ierr = PetscMalloc((*nrecvs+1)*sizeof(MPI_Request),&r_waits);CHKERRQ(ierr);
  for (i=0; i<*nrecvs; i++) {
    ierr = MPI_Irecv(olengths+i,1,MPI_INT,MPI_ANY_SOURCE,tag,comm,r_waits+i);CHKERRQ(ierr);
  }

  /* Now post the Isends with the message lenght-info */
  ierr = PetscMalloc((nsends+1)*sizeof(MPI_Request),&s_waits);CHKERRQ(ierr);
  for (i=0,j=0; i<size; ++i) {
    if (ilengths[i]) {
      ierr = MPI_Isend(ilengths+i,1,MPI_INT,i,tag,comm,s_waits+j);CHKERRQ(ierr);
      j++;
    }
  }
  
  /* Now post waits on Irecv */
  ierr = PetscMalloc((*nrecvs+1)*sizeof(MPI_Status),&r_status);CHKERRQ(ierr);
  ierr = MPI_Waitall(*nrecvs,r_waits,r_status);CHKERRQ(ierr);
  
  /* Now post waits on Isends */
  ierr = PetscMalloc((nsends+1)*sizeof(MPI_Status),&s_status);CHKERRQ(ierr);
  ierr = MPI_Waitall(nsends,s_waits,s_status);CHKERRQ(ierr);
  
  /* Now pack up the received data */
  ierr = PetscMalloc((*nrecvs+1)*sizeof(int),&onodes);CHKERRQ(ierr);
  for (i=0; i<*nrecvs; ++i) {
    onodes[i] = r_status[i].MPI_SOURCE;
  }

  ierr = PetscFree(tmp);CHKERRQ(ierr);
  ierr = PetscFree(r_waits);CHKERRQ(ierr);
  ierr = PetscFree(s_waits);CHKERRQ(ierr);
  ierr = PetscFree(r_status);CHKERRQ(ierr);
  ierr = PetscFree(s_status);CHKERRQ(ierr);


  
  PetscFunctionReturn(0);
}

