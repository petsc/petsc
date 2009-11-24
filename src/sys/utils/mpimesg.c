#define PETSC_DLL

#include "petscsys.h"        /*I  "petscsys.h"  I*/


#undef __FUNCT__  
#define __FUNCT__ "PetscGatherNumberOfMessages"
/*@C
  PetscGatherNumberOfMessages -  Computes the number of messages a node expects to receive

  Collective on MPI_Comm

  Input Parameters:
+ comm     - Communicator
. iflags   - an array of integers of length sizeof(comm). A '1' in ilengths[i] represent a 
             message from current node to ith node. Optionally PETSC_NULL
- ilengths - Non zero ilengths[i] represent a message to i of length ilengths[i].
             Optionally PETSC_NULL.

  Output Parameters:
. nrecvs    - number of messages received

  Level: developer

  Concepts: mpi utility

  Notes:
  With this info, the correct message lengths can be determined using
  PetscGatherMessageLengths()

  Either iflags or ilengths should be provided.  If iflags is not
  provided (PETSC_NULL) it can be computed from ilengths. If iflags is
  provided, ilengths is not required.

.seealso: PetscGatherMessageLengths()
@*/
PetscErrorCode PETSC_DLLEXPORT PetscGatherNumberOfMessages(MPI_Comm comm,const PetscMPIInt iflags[],const PetscMPIInt ilengths[],PetscMPIInt *nrecvs)
{
  PetscMPIInt    size,rank,*recv_buf,i,*iflags_local = PETSC_NULL,*iflags_localm = PETSC_NULL;
  PetscErrorCode ierr;

  PetscFunctionBegin;

  ierr = MPI_Comm_size(comm,&size);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(comm,&rank);CHKERRQ(ierr);

  ierr = PetscMalloc2(size,PetscMPIInt,&recv_buf,size,PetscMPIInt,&iflags_localm);CHKERRQ(ierr);

  /* If iflags not provided, compute iflags from ilengths */
  if (!iflags) {
    if (!ilengths) SETERRQ(PETSC_ERR_ARG_WRONGSTATE,"Either iflags or ilengths should be provided");
    iflags_local = iflags_localm;
    for (i=0; i<size; i++) { 
      if (ilengths[i])  iflags_local[i] = 1;
      else iflags_local[i] = 0;
    }
  } else {
    iflags_local = (PetscMPIInt *) iflags;
  }

  /* Post an allreduce to determine the numer of messages the current node will receive */
  ierr    = MPI_Allreduce(iflags_local,recv_buf,size,MPI_INT,MPI_SUM,comm);CHKERRQ(ierr);
  *nrecvs = recv_buf[rank];

  ierr = PetscFree2(recv_buf,iflags_localm);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


#undef __FUNCT__  
#define __FUNCT__ "PetscGatherMessageLengths"
/*@C
  PetscGatherMessageLengths - Computes info about messages that a MPI-node will receive, 
  including (from-id,length) pairs for each message.

  Collective on MPI_Comm

  Input Parameters:
+ comm      - Communicator
. nsends    - number of messages that are to be sent.
. nrecvs    - number of messages being received
- ilengths  - an array of integers of length sizeof(comm)
              a non zero ilengths[i] represent a message to i of length ilengths[i] 


  Output Parameters:
+ onodes    - list of node-ids from which messages are expected
- olengths  - corresponding message lengths

  Level: developer

  Concepts: mpi utility

  Notes:
  With this info, the correct MPI_Irecv() can be posted with the correct
  from-id, with a buffer with the right amount of memory required.

  The calling function deallocates the memory in onodes and olengths

  To determine nrecevs, one can use PetscGatherNumberOfMessages()

.seealso: PetscGatherNumberOfMessages()
@*/
PetscErrorCode PETSC_DLLEXPORT PetscGatherMessageLengths(MPI_Comm comm,PetscMPIInt nsends,PetscMPIInt nrecvs,const PetscMPIInt ilengths[],PetscMPIInt **onodes,PetscMPIInt **olengths)
{
  PetscErrorCode ierr;
  PetscMPIInt    size,tag,i,j;
  MPI_Request    *s_waits = PETSC_NULL,*r_waits = PETSC_NULL;
  MPI_Status     *w_status = PETSC_NULL;

  PetscFunctionBegin;
  ierr = MPI_Comm_size(comm,&size);CHKERRQ(ierr);
  ierr = PetscCommGetNewTag(comm,&tag);CHKERRQ(ierr);

  /* cannot use PetscMalloc3() here because in the call to MPI_Waitall() they MUST be contiguous */
  ierr    = PetscMalloc2(nrecvs+nsends,MPI_Request,&r_waits,nrecvs+nsends,MPI_Status,&w_status);CHKERRQ(ierr);
  s_waits = r_waits+nrecvs;

  /* Post the Irecv to get the message length-info */
  ierr = PetscMalloc(nrecvs*sizeof(PetscMPIInt),olengths);CHKERRQ(ierr);
  for (i=0; i<nrecvs; i++) {
    ierr = MPI_Irecv((*olengths)+i,1,MPI_INT,MPI_ANY_SOURCE,tag,comm,r_waits+i);CHKERRQ(ierr);
  }

  /* Post the Isends with the message length-info */
  for (i=0,j=0; i<size; ++i) {
    if (ilengths[i]) {
      ierr = MPI_Isend((void*)(ilengths+i),1,MPI_INT,i,tag,comm,s_waits+j);CHKERRQ(ierr);
      j++;
    }
  }

  /* Post waits on sends and receivs */
  if (nrecvs+nsends) {ierr = MPI_Waitall(nrecvs+nsends,r_waits,w_status);CHKERRQ(ierr);}
  
  /* Pack up the received data */
  ierr = PetscMalloc(nrecvs*sizeof(PetscMPIInt),onodes);CHKERRQ(ierr);
  for (i=0; i<nrecvs; ++i) {
    (*onodes)[i] = w_status[i].MPI_SOURCE;
  }
  ierr = PetscFree2(r_waits,w_status);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PetscGatherMessageLengths2"
/*@C
  PetscGatherMessageLengths2 - Computes info about messages that a MPI-node will receive, 
  including (from-id,length) pairs for each message. Same functionality as PetscGatherMessageLengths()
  except it takes TWO ilenths and output TWO olengths.

  Collective on MPI_Comm

  Input Parameters:
+ comm      - Communicator
. nsends    - number of messages that are to be sent.
. nrecvs    - number of messages being received
- ilengths1, ilengths2 - array of integers of length sizeof(comm)
              a non zero ilengths[i] represent a message to i of length ilengths[i] 

  Output Parameters:
+ onodes    - list of node-ids from which messages are expected
- olengths1, olengths2 - corresponding message lengths

  Level: developer

  Concepts: mpi utility

  Notes:
  With this info, the correct MPI_Irecv() can be posted with the correct
  from-id, with a buffer with the right amount of memory required.

  The calling function deallocates the memory in onodes and olengths

  To determine nrecevs, one can use PetscGatherNumberOfMessages()

.seealso: PetscGatherMessageLengths() and PetscGatherNumberOfMessages()
@*/
PetscErrorCode PETSC_DLLEXPORT PetscGatherMessageLengths2(MPI_Comm comm,PetscMPIInt nsends,PetscMPIInt nrecvs,const PetscMPIInt ilengths1[],const PetscMPIInt ilengths2[],PetscMPIInt **onodes,PetscMPIInt **olengths1,PetscMPIInt **olengths2)
{
  PetscErrorCode ierr;
  PetscMPIInt    size,tag,i,j,*buf_s = PETSC_NULL,*buf_r = PETSC_NULL,*buf_j = PETSC_NULL;
  MPI_Request    *s_waits = PETSC_NULL,*r_waits = PETSC_NULL;
  MPI_Status     *w_status = PETSC_NULL;

  PetscFunctionBegin;
  ierr = MPI_Comm_size(comm,&size);CHKERRQ(ierr);
  ierr = PetscCommGetNewTag(comm,&tag);CHKERRQ(ierr);

  /* cannot use PetscMalloc5() because r_waits and s_waits must be contiquous for the call to MPI_Waitall() */
  ierr = PetscMalloc4(nrecvs+nsends,MPI_Request,&r_waits,2*nrecvs,PetscMPIInt,&buf_r,2*nsends,PetscMPIInt,&buf_s,nrecvs+nsends,MPI_Status,&w_status);CHKERRQ(ierr);
  s_waits = r_waits + nrecvs;

  /* Post the Irecv to get the message length-info */
  ierr = PetscMalloc((nrecvs+1)*sizeof(PetscMPIInt),olengths1);CHKERRQ(ierr);
  ierr = PetscMalloc((nrecvs+1)*sizeof(PetscMPIInt),olengths2);CHKERRQ(ierr);
  for (i=0; i<nrecvs; i++) {
    buf_j = buf_r + (2*i);
    ierr = MPI_Irecv(buf_j,2,MPI_INT,MPI_ANY_SOURCE,tag,comm,r_waits+i);CHKERRQ(ierr);
  }

  /* Post the Isends with the message length-info */
  for (i=0,j=0; i<size; ++i) {
    if (ilengths1[i]) {
      buf_j = buf_s + (2*j);
      buf_j[0] = *(ilengths1+i);
      buf_j[1] = *(ilengths2+i);
      ierr = MPI_Isend(buf_j,2,MPI_INT,i,tag,comm,s_waits+j);CHKERRQ(ierr);
      j++;
    }
  }
  
  /* Post waits on sends and receivs */
  if (nrecvs+nsends) {ierr = MPI_Waitall(nrecvs+nsends,r_waits,w_status);CHKERRQ(ierr);}

  
  /* Pack up the received data */
  ierr = PetscMalloc((nrecvs+1)*sizeof(PetscMPIInt),onodes);CHKERRQ(ierr);
  for (i=0; i<nrecvs; ++i) {
    (*onodes)[i] = w_status[i].MPI_SOURCE;
    buf_j = buf_r + (2*i);
    (*olengths1)[i] = buf_j[0];
    (*olengths2)[i] = buf_j[1];
  }

  ierr = PetscFree4(r_waits,buf_r,buf_s,w_status);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*

  Allocate a bufffer sufficient to hold messages of size specified in olengths.
  And post Irecvs on these buffers using node info from onodes
  
 */
#undef __FUNCT__  
#define __FUNCT__ "PetscPostIrecvInt"
PetscErrorCode PETSC_DLLEXPORT PetscPostIrecvInt(MPI_Comm comm,PetscMPIInt tag,PetscMPIInt nrecvs,const PetscMPIInt onodes[],const PetscMPIInt olengths[],PetscInt ***rbuf,MPI_Request **r_waits)
{
  PetscErrorCode ierr;
  PetscInt       **rbuf_t,i,len = 0;
  MPI_Request    *r_waits_t;

  PetscFunctionBegin;
  /* compute memory required for recv buffers */
  for (i=0; i<nrecvs; i++) len += olengths[i];  /* each message length */

  /* allocate memory for recv buffers */
  ierr = PetscMalloc((nrecvs+1)*sizeof(PetscInt*),&rbuf_t);CHKERRQ(ierr);
  ierr = PetscMalloc(len*sizeof(PetscInt),&rbuf_t[0]);CHKERRQ(ierr);
  for (i=1; i<nrecvs; ++i) rbuf_t[i] = rbuf_t[i-1] + olengths[i-1];

  /* Post the receives */
  ierr = PetscMalloc(nrecvs*sizeof(MPI_Request),&r_waits_t);CHKERRQ(ierr);
  for (i=0; i<nrecvs; ++i) {
    ierr = MPI_Irecv(rbuf_t[i],olengths[i],MPIU_INT,onodes[i],tag,comm,r_waits_t+i);CHKERRQ(ierr);
  }

  *rbuf    = rbuf_t;
  *r_waits = r_waits_t;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PetscPostIrecvScalar"
PetscErrorCode PETSC_DLLEXPORT PetscPostIrecvScalar(MPI_Comm comm,PetscMPIInt tag,PetscMPIInt nrecvs,const PetscMPIInt onodes[],const PetscMPIInt olengths[],PetscScalar ***rbuf,MPI_Request **r_waits)
{
  PetscErrorCode ierr;
  PetscMPIInt    i;
  PetscScalar    **rbuf_t;
  MPI_Request    *r_waits_t;
  PetscInt       len = 0;

  PetscFunctionBegin;
  /* compute memory required for recv buffers */
  for (i=0; i<nrecvs; i++) len += olengths[i];  /* each message length */

  /* allocate memory for recv buffers */
  ierr    = PetscMalloc((nrecvs+1)*sizeof(PetscScalar*),&rbuf_t);CHKERRQ(ierr);
  ierr    = PetscMalloc(len*sizeof(PetscScalar),&rbuf_t[0]);CHKERRQ(ierr);
  for (i=1; i<nrecvs; ++i) rbuf_t[i] = rbuf_t[i-1] + olengths[i-1];

  /* Post the receives */
  ierr = PetscMalloc(nrecvs*sizeof(MPI_Request),&r_waits_t);CHKERRQ(ierr);
  for (i=0; i<nrecvs; ++i) {
    ierr = MPI_Irecv(rbuf_t[i],olengths[i],MPIU_SCALAR,onodes[i],tag,comm,r_waits_t+i);CHKERRQ(ierr);
  }

  *rbuf    = rbuf_t;
  *r_waits = r_waits_t;
  PetscFunctionReturn(0);
}
