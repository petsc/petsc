/*$Id: mpimesg.c,v 1.2 2001/02/08 20:59:31 balay Exp balay $*/

#include "petsc.h"        /*I  "petsc.h"  I*/


/*@C
  PetscGatherNoOfMessages -  Computes the number of messages a node expects to receive

  Collective on MPI_Comm

  Input Parameters:
+ comm     - Communicator
. nsends   - no of messages that are to be sent. Optionally PETSC_DETERMINE
- iflags   - an array of integers of length sizeof(comm). A '1' in ilenghts[i] represent a 
             message from current node to ith node. Optionally PETSC_NULL
. ilengths - Non zero ilenghts[i] represent a message to i of length ilenghts[i].
             Optionally PETSC_NULL.

  Output Parameters:
+ nrecvs    - number of messages received

  Level: developer

  Concepts: mpi utility

  Notes:
  With this info, the correct message lengths can be determined using
  PetscGatherMessageLengths()

  Either iflags or ilengths should be provided.  If iflags is not
  provided (PETSC_NULL) it can be computed from ilengths. If iflags is
  provided, ilengths is not required. if nsends is not provided, it
  will be computed from iflags.

.seealso: PetscGatherMessageLengths()
@*/
#undef __FUNC__  
#define __FUNC__ "PetscGatherNoogMessages"
int PetscGatherNoOfMessages(MPI_Comm comm,int nsends,int *iflags,int *ilengths,int *nrecvs)
{
  int *recv_buf,size,rank,i,ierr,nsends_local,*iflags_local;

  PetscFunctionBegin;

  ierr = MPI_Comm_size(comm,&size);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(comm,&rank);CHKERRQ(ierr);


  /* If iflags not provided, compute iflags from ilengths */
  if (!iflags) {
    if (!ilengths) SETERRQ(PETSC_ERR_ARG_WRONGSTATE,"Either iflags or ilengths should be provided");
    ierr = PetscMalloc(size*sizeof(int),&iflags_local);CHKERRQ(ierr);
    for (i=0; i<size; i++) { 
      if (ilengths[i])  iflags_local[i] = 1;
      else iflags_local[i] = 0;
    }
  } else {
    iflags_local = iflags;
  }

  /* If nsends is not provided, compute it from iflags */
  if (nsends == PETSC_DETERMINE) {
    for (nsends_local=0,i=0; i<size; i++) nsends_local += iflags_local[i];
  } else {
    nsends_local = nsends;
  }


  ierr = PetscMalloc(size*sizeof(int),&recv_buf);CHKERRQ(ierr);

  /* Now post an allreduce to determine the numer of messages the current node will receive */
  ierr    = MPI_Allreduce(iflags_local,recv_buf,size,MPI_INT,MPI_SUM,comm);CHKERRQ(ierr);
  *nrecvs = recv_buf[rank];

  if (!iflags) {
    ierr = PetscFree(iflags_local);CHKERRQ(ierr);
  }
  ierr = PetscFree(recv_buf);CHKERRQ(ierr);
  
  PetscFunctionReturn(0);
}


/*@C
  PetscGatherMessageLengths - Computes info about messages that a MPI-node will receive, 
  including (from-id,length) pairs for each message.

  Collective on MPI_Comm

  Input Parameters:
+ comm      - Communicator
. nsends    - no of messages that are to be sent.
. nrecvs    - number of messages being received
- ilengths  - an array of integers of length sizeof(comm)
              a non zero ilenghts[i] represent a message to i of length ilenghts[i] 


  Output Parameters:
+ onodes    - list of node-ids from which messages are expected
- olengths  - corresponding message lengths

  Level: developer

  Concepts: mpi utility

  Notes:
  With this info, the correct MPI_Irecv() can be posted with the correct
  from-id, with a buffer with the right amount of memory required.

  To determine nrecevs, one can use PetscGatherNoOfMessages()

.seealso: PetscGatherNoOfMessages()
@*/
#undef __FUNC__  
#define __FUNC__ "PetscGatherMessageLengths"
int PetscGatherMessageLengths(MPI_Comm comm,int nsends,int nrecvs,int *ilengths,int **onodes,int **olengths)
{
  int size,i,j,tag,ierr;
  MPI_Request *s_waits,*r_waits;
  MPI_Status  *w_status;

  PetscFunctionBegin;

  ierr = MPI_Comm_size(comm,&size);CHKERRQ(ierr);
  ierr = PetscCommGetNewTag(comm,&tag);CHKERRQ(ierr);

  ierr = PetscMalloc((nrecvs+nsends+1)*sizeof(MPI_Request),&r_waits);CHKERRQ(ierr);
  s_waits = r_waits + nrecvs;

  /* Now post the Irecv to get the message length-info */
  ierr = PetscMalloc((nrecvs+1)*sizeof(int),olengths);CHKERRQ(ierr);
  for (i=0; i<nrecvs; i++) {
    ierr = MPI_Irecv((*olengths)+i,1,MPI_INT,MPI_ANY_SOURCE,tag,comm,r_waits+i);CHKERRQ(ierr);
  }

  /* Now post the Isends with the message lenght-info */
  for (i=0,j=0; i<size; ++i) {
    if (ilengths[i]) {
      ierr = MPI_Isend(ilengths+i,1,MPI_INT,i,tag,comm,s_waits+j);CHKERRQ(ierr);
      j++;
    }
  }
  
  /* Now post waits on sends and receivs */
  ierr = PetscMalloc((nrecvs+nsends+1)*sizeof(MPI_Status),&w_status);CHKERRQ(ierr);
  ierr = MPI_Waitall(nrecvs+nsends,r_waits,w_status);CHKERRQ(ierr);

  
  /* Now pack up the received data */
  ierr = PetscMalloc((nrecvs+1)*sizeof(int),onodes);CHKERRQ(ierr);
  for (i=0; i<nrecvs; ++i) {
    (*onodes)[i] = w_status[i].MPI_SOURCE;
  }

  ierr = PetscFree(r_waits);CHKERRQ(ierr);
  ierr = PetscFree(w_status);CHKERRQ(ierr);
  
  PetscFunctionReturn(0);
}


/*
  PetscGatherMessageLengthsOrig - Computes info about messages that a MPI-node will receive, 
  including (from-id,length) pairs for each message.

  Collective on MPI_Comm

  Input Parameters:
+ comm     - Communicator
. nsends   - no of messages that are to be sent
             Optionally PETSC_DETERMINE
. ilengths - an array of integers of length sizeof(comm)
             a non zero ilenghts[i] represent a message to i of length ilenghts[i] 
- iflags   - a '1' in ilenghts[i] represent a message to i
             Optionally PETSC_NULL

  Output Parameters:
+ nrecvs    - number of messages received
. onodes    - list of node-ids from which messages are expected
- olengths  - corresponding message lengths

  Notes:
  With this info, the correct MPI_Irecv() can be posted with the correct
  from-id, with a buffer with the right amount of memory required.

  nsends and iflags can be computed from ilengths, but if it is already
  avilable, it can be used.

*/
#undef __FUNC__  
#define __FUNC__ "PetscGatherMessageLengthsOrig"
int PetscGatherMessageLengthsOrig(MPI_Comm comm,int nsends,int *ilengths,int *iflags,int *nrecvs,int **onodes,int **olengths)
{
  int *tmp,size,rank,i,j,tag,ierr,nsends_local,*iflags_local;
  MPI_Request *s_waits,*r_waits;
  MPI_Status  *s_status,*r_status;

  PetscFunctionBegin;

  ierr = MPI_Comm_size(comm,&size);CHKERRQ(ierr); /* eliminate? */
  ierr = MPI_Comm_rank(comm,&rank);CHKERRQ(ierr); /* eliminate? */
  ierr = PetscCommGetNewTag(comm,&tag);CHKERRQ(ierr); /* eliminate? */

  if (nsends == PETSC_DETERMINE) {
    for (nsends_local=0,i=0; i<size; i++) { if (ilengths[i]) nsends_local++; }
  } else {
    nsends_local = nsends;
  }

  if (!iflags) {
    ierr = PetscMalloc(size*sizeof(int),&iflags_local);CHKERRQ(ierr);
    ierr = PetscMemzero(iflags_local,size*sizeof(int));CHKERRQ(ierr);
    for (i=0; i<size; i++) { 
      if (ilengths[i])  iflags_local[i] = 1;
      else iflags_local[i] = 0;
    }
  } else {
    iflags_local = iflags;
  }

  ierr = PetscMalloc(size*sizeof(int),&tmp);CHKERRQ(ierr);

  /* Now post an allreduce to determine the numer of messages the current node will receive */
  ierr    = MPI_Allreduce(iflags_local,tmp,size,MPI_INT,MPI_SUM,comm);CHKERRQ(ierr);
  *nrecvs = tmp[rank];
  
  /* Now post the Irecv to get the message length-info */
  ierr = PetscMalloc((*nrecvs+1)*sizeof(int),olengths);CHKERRQ(ierr);
  ierr = PetscMalloc((*nrecvs+1)*sizeof(MPI_Request),&r_waits);CHKERRQ(ierr);
  for (i=0; i<*nrecvs; i++) {
    ierr = MPI_Irecv((*olengths)+i,1,MPI_INT,MPI_ANY_SOURCE,tag,comm,r_waits+i);CHKERRQ(ierr);
  }

  /* Now post the Isends with the message lenght-info */
  ierr = PetscMalloc((nsends_local+1)*sizeof(MPI_Request),&s_waits);CHKERRQ(ierr);
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
  ierr = PetscMalloc((nsends_local+1)*sizeof(MPI_Status),&s_status);CHKERRQ(ierr);
  ierr = MPI_Waitall(nsends_local,s_waits,s_status);CHKERRQ(ierr);
  
  /* Now pack up the received data */
  ierr = PetscMalloc((*nrecvs+1)*sizeof(int),onodes);CHKERRQ(ierr);
  for (i=0; i<*nrecvs; ++i) {
    (*onodes)[i] = r_status[i].MPI_SOURCE;
  }

  if (!iflags) {
    ierr = PetscFree(iflags_local);CHKERRQ(ierr);
  }
  ierr = PetscFree(tmp);CHKERRQ(ierr);
  ierr = PetscFree(r_waits);CHKERRQ(ierr);
  ierr = PetscFree(s_waits);CHKERRQ(ierr);
  ierr = PetscFree(r_status);CHKERRQ(ierr);
  ierr = PetscFree(s_status);CHKERRQ(ierr);
  
  PetscFunctionReturn(0);
}

