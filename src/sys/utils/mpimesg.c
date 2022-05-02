
#include <petscsys.h>        /*I  "petscsys.h"  I*/
#include <petsc/private/mpiutils.h>

/*@C
  PetscGatherNumberOfMessages -  Computes the number of messages a node expects to receive

  Collective

  Input Parameters:
+ comm     - Communicator
. iflags   - an array of integers of length sizeof(comm). A '1' in ilengths[i] represent a
             message from current node to ith node. Optionally NULL
- ilengths - Non zero ilengths[i] represent a message to i of length ilengths[i].
             Optionally NULL.

  Output Parameters:
. nrecvs    - number of messages received

  Level: developer

  Notes:
  With this info, the correct message lengths can be determined using
  PetscGatherMessageLengths()

  Either iflags or ilengths should be provided.  If iflags is not
  provided (NULL) it can be computed from ilengths. If iflags is
  provided, ilengths is not required.

.seealso: `PetscGatherMessageLengths()`
@*/
PetscErrorCode  PetscGatherNumberOfMessages(MPI_Comm comm,const PetscMPIInt iflags[],const PetscMPIInt ilengths[],PetscMPIInt *nrecvs)
{
  PetscMPIInt    size,rank,*recv_buf,i,*iflags_local = NULL,*iflags_localm = NULL;

  PetscFunctionBegin;
  PetscCallMPI(MPI_Comm_size(comm,&size));
  PetscCallMPI(MPI_Comm_rank(comm,&rank));

  PetscCall(PetscMalloc2(size,&recv_buf,size,&iflags_localm));

  /* If iflags not provided, compute iflags from ilengths */
  if (!iflags) {
    PetscCheck(ilengths,PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"Either iflags or ilengths should be provided");
    iflags_local = iflags_localm;
    for (i=0; i<size; i++) {
      if (ilengths[i]) iflags_local[i] = 1;
      else iflags_local[i] = 0;
    }
  } else iflags_local = (PetscMPIInt*) iflags;

  /* Post an allreduce to determine the numer of messages the current node will receive */
  PetscCall(MPIU_Allreduce(iflags_local,recv_buf,size,MPI_INT,MPI_SUM,comm));
  *nrecvs = recv_buf[rank];

  PetscCall(PetscFree2(recv_buf,iflags_localm));
  PetscFunctionReturn(0);
}

/*@C
  PetscGatherMessageLengths - Computes info about messages that a MPI-node will receive,
  including (from-id,length) pairs for each message.

  Collective

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

  Notes:
  With this info, the correct MPI_Irecv() can be posted with the correct
  from-id, with a buffer with the right amount of memory required.

  The calling function deallocates the memory in onodes and olengths

  To determine nrecvs, one can use PetscGatherNumberOfMessages()

.seealso: `PetscGatherNumberOfMessages()`
@*/
PetscErrorCode  PetscGatherMessageLengths(MPI_Comm comm,PetscMPIInt nsends,PetscMPIInt nrecvs,const PetscMPIInt ilengths[],PetscMPIInt **onodes,PetscMPIInt **olengths)
{
  PetscMPIInt    size,rank,tag,i,j;
  MPI_Request    *s_waits  = NULL,*r_waits = NULL;
  MPI_Status     *w_status = NULL;

  PetscFunctionBegin;
  PetscCallMPI(MPI_Comm_size(comm,&size));
  PetscCallMPI(MPI_Comm_rank(comm,&rank));
  PetscCall(PetscCommGetNewTag(comm,&tag));

  /* cannot use PetscMalloc3() here because in the call to MPI_Waitall() they MUST be contiguous */
  PetscCall(PetscMalloc2(nrecvs+nsends,&r_waits,nrecvs+nsends,&w_status));
  s_waits = r_waits+nrecvs;

  /* Post the Irecv to get the message length-info */
  PetscCall(PetscMalloc1(nrecvs,olengths));
  for (i=0; i<nrecvs; i++) {
    PetscCallMPI(MPI_Irecv((*olengths)+i,1,MPI_INT,MPI_ANY_SOURCE,tag,comm,r_waits+i));
  }

  /* Post the Isends with the message length-info */
  for (i=0,j=0; i<size; ++i) {
    if (ilengths[i]) {
      PetscCallMPI(MPI_Isend((void*)(ilengths+i),1,MPI_INT,i,tag,comm,s_waits+j));
      j++;
    }
  }

  /* Post waits on sends and receivs */
  if (nrecvs+nsends) PetscCallMPI(MPI_Waitall(nrecvs+nsends,r_waits,w_status));

  /* Pack up the received data */
  PetscCall(PetscMalloc1(nrecvs,onodes));
  for (i=0; i<nrecvs; ++i) {
    (*onodes)[i] = w_status[i].MPI_SOURCE;
#if defined(PETSC_HAVE_OMPI_MAJOR_VERSION)
    /* This line is a workaround for a bug in OpenMPI-2.1.1 distributed by Ubuntu-18.04.2 LTS.
       It happens in self-to-self MPI_Send/Recv using MPI_ANY_SOURCE for message matching. OpenMPI
       does not put correct value in recv buffer. See also
       https://lists.mcs.anl.gov/pipermail/petsc-dev/2019-July/024803.html
       https://www.mail-archive.com/users@lists.open-mpi.org//msg33383.html
     */
    if (w_status[i].MPI_SOURCE == rank) (*olengths)[i] = ilengths[rank];
#endif
  }
  PetscCall(PetscFree2(r_waits,w_status));
  PetscFunctionReturn(0);
}

/* Same as PetscGatherNumberOfMessages(), except using PetscInt for ilengths[] */
PetscErrorCode  PetscGatherNumberOfMessages_Private(MPI_Comm comm,const PetscMPIInt iflags[],const PetscInt ilengths[],PetscMPIInt *nrecvs)
{
  PetscMPIInt    size,rank,*recv_buf,i,*iflags_local = NULL,*iflags_localm = NULL;

  PetscFunctionBegin;
  PetscCallMPI(MPI_Comm_size(comm,&size));
  PetscCallMPI(MPI_Comm_rank(comm,&rank));

  PetscCall(PetscMalloc2(size,&recv_buf,size,&iflags_localm));

  /* If iflags not provided, compute iflags from ilengths */
  if (!iflags) {
    PetscCheck(ilengths,PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"Either iflags or ilengths should be provided");
    iflags_local = iflags_localm;
    for (i=0; i<size; i++) {
      if (ilengths[i]) iflags_local[i] = 1;
      else iflags_local[i] = 0;
    }
  } else iflags_local = (PetscMPIInt*) iflags;

  /* Post an allreduce to determine the numer of messages the current node will receive */
  PetscCall(MPIU_Allreduce(iflags_local,recv_buf,size,MPI_INT,MPI_SUM,comm));
  *nrecvs = recv_buf[rank];

  PetscCall(PetscFree2(recv_buf,iflags_localm));
  PetscFunctionReturn(0);
}

/* Same as PetscGatherMessageLengths(), except using PetscInt for message lengths */
PetscErrorCode  PetscGatherMessageLengths_Private(MPI_Comm comm,PetscMPIInt nsends,PetscMPIInt nrecvs,const PetscInt ilengths[],PetscMPIInt **onodes,PetscInt **olengths)
{
  PetscMPIInt    size,rank,tag,i,j;
  MPI_Request    *s_waits  = NULL,*r_waits = NULL;
  MPI_Status     *w_status = NULL;

  PetscFunctionBegin;
  PetscCallMPI(MPI_Comm_size(comm,&size));
  PetscCallMPI(MPI_Comm_rank(comm,&rank));
  PetscCall(PetscCommGetNewTag(comm,&tag));

  /* cannot use PetscMalloc3() here because in the call to MPI_Waitall() they MUST be contiguous */
  PetscCall(PetscMalloc2(nrecvs+nsends,&r_waits,nrecvs+nsends,&w_status));
  s_waits = r_waits+nrecvs;

  /* Post the Irecv to get the message length-info */
  PetscCall(PetscMalloc1(nrecvs,olengths));
  for (i=0; i<nrecvs; i++) {
    PetscCallMPI(MPI_Irecv((*olengths)+i,1,MPIU_INT,MPI_ANY_SOURCE,tag,comm,r_waits+i));
  }

  /* Post the Isends with the message length-info */
  for (i=0,j=0; i<size; ++i) {
    if (ilengths[i]) {
      PetscCallMPI(MPI_Isend((void*)(ilengths+i),1,MPIU_INT,i,tag,comm,s_waits+j));
      j++;
    }
  }

  /* Post waits on sends and receivs */
  if (nrecvs+nsends) PetscCallMPI(MPI_Waitall(nrecvs+nsends,r_waits,w_status));

  /* Pack up the received data */
  PetscCall(PetscMalloc1(nrecvs,onodes));
  for (i=0; i<nrecvs; ++i) {
    (*onodes)[i] = w_status[i].MPI_SOURCE;
    if (w_status[i].MPI_SOURCE == rank) (*olengths)[i] = ilengths[rank]; /* See comments in PetscGatherMessageLengths */
  }
  PetscCall(PetscFree2(r_waits,w_status));
  PetscFunctionReturn(0);
}

/*@C
  PetscGatherMessageLengths2 - Computes info about messages that a MPI-node will receive,
  including (from-id,length) pairs for each message. Same functionality as PetscGatherMessageLengths()
  except it takes TWO ilenths and output TWO olengths.

  Collective

  Input Parameters:
+ comm      - Communicator
. nsends    - number of messages that are to be sent.
. nrecvs    - number of messages being received
. ilengths1 - first array of integers of length sizeof(comm)
- ilengths2 - second array of integers of length sizeof(comm)

  Output Parameters:
+ onodes    - list of node-ids from which messages are expected
. olengths1 - first corresponding message lengths
- olengths2 - second  message lengths

  Level: developer

  Notes:
  With this info, the correct MPI_Irecv() can be posted with the correct
  from-id, with a buffer with the right amount of memory required.

  The calling function deallocates the memory in onodes and olengths

  To determine nrecvs, one can use PetscGatherNumberOfMessages()

.seealso: `PetscGatherMessageLengths()` `and` `PetscGatherNumberOfMessages()`
@*/
PetscErrorCode  PetscGatherMessageLengths2(MPI_Comm comm,PetscMPIInt nsends,PetscMPIInt nrecvs,const PetscMPIInt ilengths1[],const PetscMPIInt ilengths2[],PetscMPIInt **onodes,PetscMPIInt **olengths1,PetscMPIInt **olengths2)
{
  PetscMPIInt    size,tag,i,j,*buf_s = NULL,*buf_r = NULL,*buf_j = NULL;
  MPI_Request    *s_waits  = NULL,*r_waits = NULL;
  MPI_Status     *w_status = NULL;

  PetscFunctionBegin;
  PetscCallMPI(MPI_Comm_size(comm,&size));
  PetscCall(PetscCommGetNewTag(comm,&tag));

  /* cannot use PetscMalloc5() because r_waits and s_waits must be contiguous for the call to MPI_Waitall() */
  PetscCall(PetscMalloc4(nrecvs+nsends,&r_waits,2*nrecvs,&buf_r,2*nsends,&buf_s,nrecvs+nsends,&w_status));
  s_waits = r_waits + nrecvs;

  /* Post the Irecv to get the message length-info */
  PetscCall(PetscMalloc1(nrecvs+1,olengths1));
  PetscCall(PetscMalloc1(nrecvs+1,olengths2));
  for (i=0; i<nrecvs; i++) {
    buf_j = buf_r + (2*i);
    PetscCallMPI(MPI_Irecv(buf_j,2,MPI_INT,MPI_ANY_SOURCE,tag,comm,r_waits+i));
  }

  /* Post the Isends with the message length-info */
  for (i=0,j=0; i<size; ++i) {
    if (ilengths1[i]) {
      buf_j    = buf_s + (2*j);
      buf_j[0] = *(ilengths1+i);
      buf_j[1] = *(ilengths2+i);
      PetscCallMPI(MPI_Isend(buf_j,2,MPI_INT,i,tag,comm,s_waits+j));
      j++;
    }
  }
  PetscCheck(j == nsends,PETSC_COMM_SELF,PETSC_ERR_PLIB,"j %d not equal to expected number of sends %d",j,nsends);

  /* Post waits on sends and receivs */
  if (nrecvs+nsends) PetscCallMPI(MPI_Waitall(nrecvs+nsends,r_waits,w_status));

  /* Pack up the received data */
  PetscCall(PetscMalloc1(nrecvs+1,onodes));
  for (i=0; i<nrecvs; ++i) {
    (*onodes)[i]    = w_status[i].MPI_SOURCE;
    buf_j           = buf_r + (2*i);
    (*olengths1)[i] = buf_j[0];
    (*olengths2)[i] = buf_j[1];
  }

  PetscCall(PetscFree4(r_waits,buf_r,buf_s,w_status));
  PetscFunctionReturn(0);
}

/*

  Allocate a buffer sufficient to hold messages of size specified in olengths.
  And post Irecvs on these buffers using node info from onodes

 */
PetscErrorCode  PetscPostIrecvInt(MPI_Comm comm,PetscMPIInt tag,PetscMPIInt nrecvs,const PetscMPIInt onodes[],const PetscMPIInt olengths[],PetscInt ***rbuf,MPI_Request **r_waits)
{
  PetscInt       **rbuf_t,i,len = 0;
  MPI_Request    *r_waits_t;

  PetscFunctionBegin;
  /* compute memory required for recv buffers */
  for (i=0; i<nrecvs; i++) len += olengths[i];  /* each message length */

  /* allocate memory for recv buffers */
  PetscCall(PetscMalloc1(nrecvs+1,&rbuf_t));
  PetscCall(PetscMalloc1(len,&rbuf_t[0]));
  for (i=1; i<nrecvs; ++i) rbuf_t[i] = rbuf_t[i-1] + olengths[i-1];

  /* Post the receives */
  PetscCall(PetscMalloc1(nrecvs,&r_waits_t));
  for (i=0; i<nrecvs; ++i) {
    PetscCallMPI(MPI_Irecv(rbuf_t[i],olengths[i],MPIU_INT,onodes[i],tag,comm,r_waits_t+i));
  }

  *rbuf    = rbuf_t;
  *r_waits = r_waits_t;
  PetscFunctionReturn(0);
}

PetscErrorCode  PetscPostIrecvScalar(MPI_Comm comm,PetscMPIInt tag,PetscMPIInt nrecvs,const PetscMPIInt onodes[],const PetscMPIInt olengths[],PetscScalar ***rbuf,MPI_Request **r_waits)
{
  PetscMPIInt    i;
  PetscScalar    **rbuf_t;
  MPI_Request    *r_waits_t;
  PetscInt       len = 0;

  PetscFunctionBegin;
  /* compute memory required for recv buffers */
  for (i=0; i<nrecvs; i++) len += olengths[i];  /* each message length */

  /* allocate memory for recv buffers */
  PetscCall(PetscMalloc1(nrecvs+1,&rbuf_t));
  PetscCall(PetscMalloc1(len,&rbuf_t[0]));
  for (i=1; i<nrecvs; ++i) rbuf_t[i] = rbuf_t[i-1] + olengths[i-1];

  /* Post the receives */
  PetscCall(PetscMalloc1(nrecvs,&r_waits_t));
  for (i=0; i<nrecvs; ++i) {
    PetscCallMPI(MPI_Irecv(rbuf_t[i],olengths[i],MPIU_SCALAR,onodes[i],tag,comm,r_waits_t+i));
  }

  *rbuf    = rbuf_t;
  *r_waits = r_waits_t;
  PetscFunctionReturn(0);
}
