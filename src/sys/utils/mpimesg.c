
#include <petscsys.h>        /*I  "petscsys.h"  I*/

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

.seealso: PetscGatherMessageLengths()
@*/
PetscErrorCode  PetscGatherNumberOfMessages(MPI_Comm comm,const PetscMPIInt iflags[],const PetscMPIInt ilengths[],PetscMPIInt *nrecvs)
{
  PetscMPIInt    size,rank,*recv_buf,i,*iflags_local = NULL,*iflags_localm = NULL;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MPI_Comm_size(comm,&size);CHKERRMPI(ierr);
  ierr = MPI_Comm_rank(comm,&rank);CHKERRMPI(ierr);

  ierr = PetscMalloc2(size,&recv_buf,size,&iflags_localm);CHKERRQ(ierr);

  /* If iflags not provided, compute iflags from ilengths */
  if (!iflags) {
    if (!ilengths) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"Either iflags or ilengths should be provided");
    iflags_local = iflags_localm;
    for (i=0; i<size; i++) {
      if (ilengths[i]) iflags_local[i] = 1;
      else iflags_local[i] = 0;
    }
  } else iflags_local = (PetscMPIInt*) iflags;

  /* Post an allreduce to determine the numer of messages the current node will receive */
  ierr    = MPIU_Allreduce(iflags_local,recv_buf,size,MPI_INT,MPI_SUM,comm);CHKERRMPI(ierr);
  *nrecvs = recv_buf[rank];

  ierr = PetscFree2(recv_buf,iflags_localm);CHKERRQ(ierr);
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

.seealso: PetscGatherNumberOfMessages()
@*/
PetscErrorCode  PetscGatherMessageLengths(MPI_Comm comm,PetscMPIInt nsends,PetscMPIInt nrecvs,const PetscMPIInt ilengths[],PetscMPIInt **onodes,PetscMPIInt **olengths)
{
  PetscErrorCode ierr;
  PetscMPIInt    size,rank,tag,i,j;
  MPI_Request    *s_waits  = NULL,*r_waits = NULL;
  MPI_Status     *w_status = NULL;

  PetscFunctionBegin;
  ierr = MPI_Comm_size(comm,&size);CHKERRMPI(ierr);
  ierr = MPI_Comm_rank(comm,&rank);CHKERRMPI(ierr);
  ierr = PetscCommGetNewTag(comm,&tag);CHKERRQ(ierr);

  /* cannot use PetscMalloc3() here because in the call to MPI_Waitall() they MUST be contiguous */
  ierr    = PetscMalloc2(nrecvs+nsends,&r_waits,nrecvs+nsends,&w_status);CHKERRQ(ierr);
  s_waits = r_waits+nrecvs;

  /* Post the Irecv to get the message length-info */
  ierr = PetscMalloc1(nrecvs,olengths);CHKERRQ(ierr);
  for (i=0; i<nrecvs; i++) {
    ierr = MPI_Irecv((*olengths)+i,1,MPI_INT,MPI_ANY_SOURCE,tag,comm,r_waits+i);CHKERRMPI(ierr);
  }

  /* Post the Isends with the message length-info */
  for (i=0,j=0; i<size; ++i) {
    if (ilengths[i]) {
      ierr = MPI_Isend((void*)(ilengths+i),1,MPI_INT,i,tag,comm,s_waits+j);CHKERRMPI(ierr);
      j++;
    }
  }

  /* Post waits on sends and receivs */
  if (nrecvs+nsends) {ierr = MPI_Waitall(nrecvs+nsends,r_waits,w_status);CHKERRMPI(ierr);}

  /* Pack up the received data */
  ierr = PetscMalloc1(nrecvs,onodes);CHKERRQ(ierr);
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
  ierr = PetscFree2(r_waits,w_status);CHKERRQ(ierr);
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
- ilengths1, ilengths2 - array of integers of length sizeof(comm)
              a non zero ilengths[i] represent a message to i of length ilengths[i]

  Output Parameters:
+ onodes    - list of node-ids from which messages are expected
- olengths1, olengths2 - corresponding message lengths

  Level: developer

  Notes:
  With this info, the correct MPI_Irecv() can be posted with the correct
  from-id, with a buffer with the right amount of memory required.

  The calling function deallocates the memory in onodes and olengths

  To determine nrecvs, one can use PetscGatherNumberOfMessages()

.seealso: PetscGatherMessageLengths() and PetscGatherNumberOfMessages()
@*/
PetscErrorCode  PetscGatherMessageLengths2(MPI_Comm comm,PetscMPIInt nsends,PetscMPIInt nrecvs,const PetscMPIInt ilengths1[],const PetscMPIInt ilengths2[],PetscMPIInt **onodes,PetscMPIInt **olengths1,PetscMPIInt **olengths2)
{
  PetscErrorCode ierr;
  PetscMPIInt    size,tag,i,j,*buf_s = NULL,*buf_r = NULL,*buf_j = NULL;
  MPI_Request    *s_waits  = NULL,*r_waits = NULL;
  MPI_Status     *w_status = NULL;

  PetscFunctionBegin;
  ierr = MPI_Comm_size(comm,&size);CHKERRMPI(ierr);
  ierr = PetscCommGetNewTag(comm,&tag);CHKERRQ(ierr);

  /* cannot use PetscMalloc5() because r_waits and s_waits must be contiguous for the call to MPI_Waitall() */
  ierr = PetscMalloc4(nrecvs+nsends,&r_waits,2*nrecvs,&buf_r,2*nsends,&buf_s,nrecvs+nsends,&w_status);CHKERRQ(ierr);
  s_waits = r_waits + nrecvs;

  /* Post the Irecv to get the message length-info */
  ierr = PetscMalloc1(nrecvs+1,olengths1);CHKERRQ(ierr);
  ierr = PetscMalloc1(nrecvs+1,olengths2);CHKERRQ(ierr);
  for (i=0; i<nrecvs; i++) {
    buf_j = buf_r + (2*i);
    ierr  = MPI_Irecv(buf_j,2,MPI_INT,MPI_ANY_SOURCE,tag,comm,r_waits+i);CHKERRMPI(ierr);
  }

  /* Post the Isends with the message length-info */
  for (i=0,j=0; i<size; ++i) {
    if (ilengths1[i]) {
      buf_j    = buf_s + (2*j);
      buf_j[0] = *(ilengths1+i);
      buf_j[1] = *(ilengths2+i);
      ierr = MPI_Isend(buf_j,2,MPI_INT,i,tag,comm,s_waits+j);CHKERRMPI(ierr);
      j++;
    }
  }
  if (j != nsends) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_PLIB,"j %d not equal to expected number of sends %d\n",j,nsends);

  /* Post waits on sends and receivs */
  if (nrecvs+nsends) {ierr = MPI_Waitall(nrecvs+nsends,r_waits,w_status);CHKERRMPI(ierr);}

  /* Pack up the received data */
  ierr = PetscMalloc1(nrecvs+1,onodes);CHKERRQ(ierr);
  for (i=0; i<nrecvs; ++i) {
    (*onodes)[i]    = w_status[i].MPI_SOURCE;
    buf_j           = buf_r + (2*i);
    (*olengths1)[i] = buf_j[0];
    (*olengths2)[i] = buf_j[1];
  }

  ierr = PetscFree4(r_waits,buf_r,buf_s,w_status);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*

  Allocate a buffer sufficient to hold messages of size specified in olengths.
  And post Irecvs on these buffers using node info from onodes

 */
PetscErrorCode  PetscPostIrecvInt(MPI_Comm comm,PetscMPIInt tag,PetscMPIInt nrecvs,const PetscMPIInt onodes[],const PetscMPIInt olengths[],PetscInt ***rbuf,MPI_Request **r_waits)
{
  PetscErrorCode ierr;
  PetscInt       **rbuf_t,i,len = 0;
  MPI_Request    *r_waits_t;

  PetscFunctionBegin;
  /* compute memory required for recv buffers */
  for (i=0; i<nrecvs; i++) len += olengths[i];  /* each message length */

  /* allocate memory for recv buffers */
  ierr = PetscMalloc1(nrecvs+1,&rbuf_t);CHKERRQ(ierr);
  ierr = PetscMalloc1(len,&rbuf_t[0]);CHKERRQ(ierr);
  for (i=1; i<nrecvs; ++i) rbuf_t[i] = rbuf_t[i-1] + olengths[i-1];

  /* Post the receives */
  ierr = PetscMalloc1(nrecvs,&r_waits_t);CHKERRQ(ierr);
  for (i=0; i<nrecvs; ++i) {
    ierr = MPI_Irecv(rbuf_t[i],olengths[i],MPIU_INT,onodes[i],tag,comm,r_waits_t+i);CHKERRMPI(ierr);
  }

  *rbuf    = rbuf_t;
  *r_waits = r_waits_t;
  PetscFunctionReturn(0);
}

PetscErrorCode  PetscPostIrecvScalar(MPI_Comm comm,PetscMPIInt tag,PetscMPIInt nrecvs,const PetscMPIInt onodes[],const PetscMPIInt olengths[],PetscScalar ***rbuf,MPI_Request **r_waits)
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
  ierr = PetscMalloc1(nrecvs+1,&rbuf_t);CHKERRQ(ierr);
  ierr = PetscMalloc1(len,&rbuf_t[0]);CHKERRQ(ierr);
  for (i=1; i<nrecvs; ++i) rbuf_t[i] = rbuf_t[i-1] + olengths[i-1];

  /* Post the receives */
  ierr = PetscMalloc1(nrecvs,&r_waits_t);CHKERRQ(ierr);
  for (i=0; i<nrecvs; ++i) {
    ierr = MPI_Irecv(rbuf_t[i],olengths[i],MPIU_SCALAR,onodes[i],tag,comm,r_waits_t+i);CHKERRMPI(ierr);
  }

  *rbuf    = rbuf_t;
  *r_waits = r_waits_t;
  PetscFunctionReturn(0);
}
