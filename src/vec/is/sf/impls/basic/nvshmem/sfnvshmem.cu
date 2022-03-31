#include <petsc/private/cudavecimpl.h>
#include <../src/vec/is/sf/impls/basic/sfpack.h>
#include <mpi.h>
#include <nvshmem.h>
#include <nvshmemx.h>

PetscErrorCode PetscNvshmemInitializeCheck(void)
{
  PetscFunctionBegin;
  if (!PetscNvshmemInitialized) { /* Note NVSHMEM does not provide a routine to check whether it is initialized */
    nvshmemx_init_attr_t attr;
    attr.mpi_comm = &PETSC_COMM_WORLD;
    PetscCall(PetscDeviceInitialize(PETSC_DEVICE_CUDA));
    PetscCall(nvshmemx_init_attr(NVSHMEMX_INIT_WITH_MPI_COMM,&attr));
    PetscNvshmemInitialized = PETSC_TRUE;
    PetscBeganNvshmem       = PETSC_TRUE;
  }
  PetscFunctionReturn(0);
}

PetscErrorCode PetscNvshmemMalloc(size_t size, void** ptr)
{
  PetscFunctionBegin;
  PetscCall(PetscNvshmemInitializeCheck());
  *ptr = nvshmem_malloc(size);
  PetscCheckFalse(!*ptr,PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"nvshmem_malloc() failed to allocate %zu bytes",size);
  PetscFunctionReturn(0);
}

PetscErrorCode PetscNvshmemCalloc(size_t size, void**ptr)
{
  PetscFunctionBegin;
  PetscCall(PetscNvshmemInitializeCheck());
  *ptr = nvshmem_calloc(size,1);
  PetscCheckFalse(!*ptr,PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"nvshmem_calloc() failed to allocate %zu bytes",size);
  PetscFunctionReturn(0);
}

PetscErrorCode PetscNvshmemFree_Private(void* ptr)
{
  PetscFunctionBegin;
  nvshmem_free(ptr);
  PetscFunctionReturn(0);
}

PetscErrorCode PetscNvshmemFinalize(void)
{
  PetscFunctionBegin;
  nvshmem_finalize();
  PetscFunctionReturn(0);
}

/* Free nvshmem related fields in the SF */
PetscErrorCode PetscSFReset_Basic_NVSHMEM(PetscSF sf)
{
  PetscSF_Basic     *bas = (PetscSF_Basic*)sf->data;

  PetscFunctionBegin;
  PetscCall(PetscFree2(bas->leafsigdisp,bas->leafbufdisp));
  PetscCall(PetscSFFree(sf,PETSC_MEMTYPE_CUDA,bas->leafbufdisp_d));
  PetscCall(PetscSFFree(sf,PETSC_MEMTYPE_CUDA,bas->leafsigdisp_d));
  PetscCall(PetscSFFree(sf,PETSC_MEMTYPE_CUDA,bas->iranks_d));
  PetscCall(PetscSFFree(sf,PETSC_MEMTYPE_CUDA,bas->ioffset_d));

  PetscCall(PetscFree2(sf->rootsigdisp,sf->rootbufdisp));
  PetscCall(PetscSFFree(sf,PETSC_MEMTYPE_CUDA,sf->rootbufdisp_d));
  PetscCall(PetscSFFree(sf,PETSC_MEMTYPE_CUDA,sf->rootsigdisp_d));
  PetscCall(PetscSFFree(sf,PETSC_MEMTYPE_CUDA,sf->ranks_d));
  PetscCall(PetscSFFree(sf,PETSC_MEMTYPE_CUDA,sf->roffset_d));
  PetscFunctionReturn(0);
}

/* Set up NVSHMEM related fields for an SF of type SFBASIC (only after PetscSFSetup_Basic() already set up dependant fields */
static PetscErrorCode PetscSFSetUp_Basic_NVSHMEM(PetscSF sf)
{
  cudaError_t    cerr;
  PetscSF_Basic  *bas = (PetscSF_Basic*)sf->data;
  PetscInt       i,nRemoteRootRanks,nRemoteLeafRanks;
  PetscMPIInt    tag;
  MPI_Comm       comm;
  MPI_Request    *rootreqs,*leafreqs;
  PetscInt       tmp,stmp[4],rtmp[4]; /* tmps for send/recv buffers */

  PetscFunctionBegin;
  PetscCall(PetscObjectGetComm((PetscObject)sf,&comm));
  PetscCall(PetscObjectGetNewTag((PetscObject)sf,&tag));

  nRemoteRootRanks      = sf->nranks-sf->ndranks;
  nRemoteLeafRanks      = bas->niranks-bas->ndiranks;
  sf->nRemoteRootRanks  = nRemoteRootRanks;
  bas->nRemoteLeafRanks = nRemoteLeafRanks;

  PetscCall(PetscMalloc2(nRemoteLeafRanks,&rootreqs,nRemoteRootRanks,&leafreqs));

  stmp[0] = nRemoteRootRanks;
  stmp[1] = sf->leafbuflen[PETSCSF_REMOTE];
  stmp[2] = nRemoteLeafRanks;
  stmp[3] = bas->rootbuflen[PETSCSF_REMOTE];

  PetscCall(MPIU_Allreduce(stmp,rtmp,4,MPIU_INT,MPI_MAX,comm));

  sf->nRemoteRootRanksMax   = rtmp[0];
  sf->leafbuflen_rmax       = rtmp[1];
  bas->nRemoteLeafRanksMax  = rtmp[2];
  bas->rootbuflen_rmax      = rtmp[3];

  /* Total four rounds of MPI communications to set up the nvshmem fields */

  /* Root ranks to leaf ranks: send info about rootsigdisp[] and rootbufdisp[] */
  PetscCall(PetscMalloc2(nRemoteRootRanks,&sf->rootsigdisp,nRemoteRootRanks,&sf->rootbufdisp));
  for (i=0; i<nRemoteRootRanks; i++) PetscCallMPI(MPI_Irecv(&sf->rootsigdisp[i],1,MPIU_INT,sf->ranks[i+sf->ndranks],tag,comm,&leafreqs[i])); /* Leaves recv */
  for (i=0; i<nRemoteLeafRanks; i++) PetscCallMPI(MPI_Send(&i,1,MPIU_INT,bas->iranks[i+bas->ndiranks],tag,comm)); /* Roots send. Note i changes, so we use MPI_Send. */
  PetscCallMPI(MPI_Waitall(nRemoteRootRanks,leafreqs,MPI_STATUSES_IGNORE));

  for (i=0; i<nRemoteRootRanks; i++) PetscCallMPI(MPI_Irecv(&sf->rootbufdisp[i],1,MPIU_INT,sf->ranks[i+sf->ndranks],tag,comm,&leafreqs[i])); /* Leaves recv */
  for (i=0; i<nRemoteLeafRanks; i++) {
    tmp  = bas->ioffset[i+bas->ndiranks] - bas->ioffset[bas->ndiranks];
    PetscCallMPI(MPI_Send(&tmp,1,MPIU_INT,bas->iranks[i+bas->ndiranks],tag,comm));  /* Roots send. Note tmp changes, so we use MPI_Send. */
  }
  PetscCallMPI(MPI_Waitall(nRemoteRootRanks,leafreqs,MPI_STATUSES_IGNORE));

  PetscCallCUDA(cudaMalloc((void**)&sf->rootbufdisp_d,nRemoteRootRanks*sizeof(PetscInt)));
  PetscCallCUDA(cudaMalloc((void**)&sf->rootsigdisp_d,nRemoteRootRanks*sizeof(PetscInt)));
  PetscCallCUDA(cudaMalloc((void**)&sf->ranks_d,nRemoteRootRanks*sizeof(PetscMPIInt)));
  PetscCallCUDA(cudaMalloc((void**)&sf->roffset_d,(nRemoteRootRanks+1)*sizeof(PetscInt)));

  PetscCallCUDA(cudaMemcpyAsync(sf->rootbufdisp_d,sf->rootbufdisp,nRemoteRootRanks*sizeof(PetscInt),cudaMemcpyHostToDevice,PetscDefaultCudaStream));
  PetscCallCUDA(cudaMemcpyAsync(sf->rootsigdisp_d,sf->rootsigdisp,nRemoteRootRanks*sizeof(PetscInt),cudaMemcpyHostToDevice,PetscDefaultCudaStream));
  PetscCallCUDA(cudaMemcpyAsync(sf->ranks_d,sf->ranks+sf->ndranks,nRemoteRootRanks*sizeof(PetscMPIInt),cudaMemcpyHostToDevice,PetscDefaultCudaStream));
  PetscCallCUDA(cudaMemcpyAsync(sf->roffset_d,sf->roffset+sf->ndranks,(nRemoteRootRanks+1)*sizeof(PetscInt),cudaMemcpyHostToDevice,PetscDefaultCudaStream));

  /* Leaf ranks to root ranks: send info about leafsigdisp[] and leafbufdisp[] */
  PetscCall(PetscMalloc2(nRemoteLeafRanks,&bas->leafsigdisp,nRemoteLeafRanks,&bas->leafbufdisp));
  for (i=0; i<nRemoteLeafRanks; i++) PetscCallMPI(MPI_Irecv(&bas->leafsigdisp[i],1,MPIU_INT,bas->iranks[i+bas->ndiranks],tag,comm,&rootreqs[i]));
  for (i=0; i<nRemoteRootRanks; i++) PetscCallMPI(MPI_Send(&i,1,MPIU_INT,sf->ranks[i+sf->ndranks],tag,comm));
  PetscCallMPI(MPI_Waitall(nRemoteLeafRanks,rootreqs,MPI_STATUSES_IGNORE));

  for (i=0; i<nRemoteLeafRanks; i++) PetscCallMPI(MPI_Irecv(&bas->leafbufdisp[i],1,MPIU_INT,bas->iranks[i+bas->ndiranks],tag,comm,&rootreqs[i]));
  for (i=0; i<nRemoteRootRanks; i++) {
    tmp  = sf->roffset[i+sf->ndranks] - sf->roffset[sf->ndranks];
    PetscCallMPI(MPI_Send(&tmp,1,MPIU_INT,sf->ranks[i+sf->ndranks],tag,comm));
  }
  PetscCallMPI(MPI_Waitall(nRemoteLeafRanks,rootreqs,MPI_STATUSES_IGNORE));

  PetscCallCUDA(cudaMalloc((void**)&bas->leafbufdisp_d,nRemoteLeafRanks*sizeof(PetscInt)));
  PetscCallCUDA(cudaMalloc((void**)&bas->leafsigdisp_d,nRemoteLeafRanks*sizeof(PetscInt)));
  PetscCallCUDA(cudaMalloc((void**)&bas->iranks_d,nRemoteLeafRanks*sizeof(PetscMPIInt)));
  PetscCallCUDA(cudaMalloc((void**)&bas->ioffset_d,(nRemoteLeafRanks+1)*sizeof(PetscInt)));

  PetscCallCUDA(cudaMemcpyAsync(bas->leafbufdisp_d,bas->leafbufdisp,nRemoteLeafRanks*sizeof(PetscInt),cudaMemcpyHostToDevice,PetscDefaultCudaStream));
  PetscCallCUDA(cudaMemcpyAsync(bas->leafsigdisp_d,bas->leafsigdisp,nRemoteLeafRanks*sizeof(PetscInt),cudaMemcpyHostToDevice,PetscDefaultCudaStream));
  PetscCallCUDA(cudaMemcpyAsync(bas->iranks_d,bas->iranks+bas->ndiranks,nRemoteLeafRanks*sizeof(PetscMPIInt),cudaMemcpyHostToDevice,PetscDefaultCudaStream));
  PetscCallCUDA(cudaMemcpyAsync(bas->ioffset_d,bas->ioffset+bas->ndiranks,(nRemoteLeafRanks+1)*sizeof(PetscInt),cudaMemcpyHostToDevice,PetscDefaultCudaStream));

  PetscCall(PetscFree2(rootreqs,leafreqs));
  PetscFunctionReturn(0);
}

PetscErrorCode PetscSFLinkNvshmemCheck(PetscSF sf,PetscMemType rootmtype,const void *rootdata,PetscMemType leafmtype,const void *leafdata,PetscBool *use_nvshmem)
{
  MPI_Comm         comm;
  PetscBool        isBasic;
  PetscMPIInt      result = MPI_UNEQUAL;

  PetscFunctionBegin;
  PetscCall(PetscObjectGetComm((PetscObject)sf,&comm));
  /* Check if the sf is eligible for NVSHMEM, if we have not checked yet.
     Note the check result <use_nvshmem> must be the same over comm, since an SFLink must be collectively either NVSHMEM or MPI.
  */
  sf->checked_nvshmem_eligibility = PETSC_TRUE;
  if (sf->use_nvshmem && !sf->checked_nvshmem_eligibility) {
    /* Only use NVSHMEM for SFBASIC on PETSC_COMM_WORLD  */
    PetscCall(PetscObjectTypeCompare((PetscObject)sf,PETSCSFBASIC,&isBasic));
    if (isBasic) PetscCallMPI(MPI_Comm_compare(PETSC_COMM_WORLD,comm,&result));
    if (!isBasic || (result != MPI_IDENT && result != MPI_CONGRUENT)) sf->use_nvshmem = PETSC_FALSE; /* If not eligible, clear the flag so that we don't try again */

    /* Do further check: If on a rank, both rootdata and leafdata are NULL, we might think they are PETSC_MEMTYPE_CUDA (or HOST)
       and then use NVSHMEM. But if root/leafmtypes on other ranks are PETSC_MEMTYPE_HOST (or DEVICE), this would lead to
       inconsistency on the return value <use_nvshmem>. To be safe, we simply disable nvshmem on these rare SFs.
    */
    if (sf->use_nvshmem) {
      PetscInt hasNullRank = (!rootdata && !leafdata) ? 1 : 0;
      PetscCallMPI(MPI_Allreduce(MPI_IN_PLACE,&hasNullRank,1,MPIU_INT,MPI_LOR,comm));
      if (hasNullRank) sf->use_nvshmem = PETSC_FALSE;
    }
    sf->checked_nvshmem_eligibility = PETSC_TRUE; /* If eligible, don't do above check again */
  }

  /* Check if rootmtype and leafmtype collectively are PETSC_MEMTYPE_CUDA */
  if (sf->use_nvshmem) {
    PetscInt oneCuda = (!rootdata || PetscMemTypeCUDA(rootmtype)) && (!leafdata || PetscMemTypeCUDA(leafmtype)) ? 1 : 0; /* Do I use cuda for both root&leafmtype? */
    PetscInt allCuda = oneCuda; /* Assume the same for all ranks. But if not, in opt mode, return value <use_nvshmem> won't be collective! */
   #if defined(PETSC_USE_DEBUG)  /* Check in debug mode. Note MPI_Allreduce is expensive, so only in debug mode */
    PetscCallMPI(MPI_Allreduce(&oneCuda,&allCuda,1,MPIU_INT,MPI_LAND,comm));
    PetscCheckFalse(allCuda != oneCuda,comm,PETSC_ERR_SUP,"root/leaf mtypes are inconsistent among ranks, which may lead to SF nvshmem failure in opt mode. Add -use_nvshmem 0 to disable it.");
   #endif
    if (allCuda) {
      PetscCall(PetscNvshmemInitializeCheck());
      if (!sf->setup_nvshmem) { /* Set up nvshmem related fields on this SF on-demand */
        PetscCall(PetscSFSetUp_Basic_NVSHMEM(sf));
        sf->setup_nvshmem = PETSC_TRUE;
      }
      *use_nvshmem = PETSC_TRUE;
    } else {
      *use_nvshmem = PETSC_FALSE;
    }
  } else {
    *use_nvshmem = PETSC_FALSE;
  }
  PetscFunctionReturn(0);
}

/* Build dependence between <stream> and <remoteCommStream> at the entry of NVSHMEM communication */
static PetscErrorCode PetscSFLinkBuildDependenceBegin(PetscSF sf,PetscSFLink link,PetscSFDirection direction)
{
  cudaError_t    cerr;
  PetscSF_Basic  *bas = (PetscSF_Basic *)sf->data;
  PetscInt       buflen = (direction == PETSCSF_ROOT2LEAF)? bas->rootbuflen[PETSCSF_REMOTE] : sf->leafbuflen[PETSCSF_REMOTE];

  PetscFunctionBegin;
  if (buflen) {
    PetscCallCUDA(cudaEventRecord(link->dataReady,link->stream));
    PetscCallCUDA(cudaStreamWaitEvent(link->remoteCommStream,link->dataReady,0));
  }
  PetscFunctionReturn(0);
}

/* Build dependence between <stream> and <remoteCommStream> at the exit of NVSHMEM communication */
static PetscErrorCode PetscSFLinkBuildDependenceEnd(PetscSF sf,PetscSFLink link,PetscSFDirection direction)
{
  cudaError_t    cerr;
  PetscSF_Basic  *bas = (PetscSF_Basic *)sf->data;
  PetscInt       buflen = (direction == PETSCSF_ROOT2LEAF)? sf->leafbuflen[PETSCSF_REMOTE] : bas->rootbuflen[PETSCSF_REMOTE];

  PetscFunctionBegin;
  /* If unpack to non-null device buffer, build the endRemoteComm dependance */
  if (buflen) {
    PetscCallCUDA(cudaEventRecord(link->endRemoteComm,link->remoteCommStream));
    PetscCallCUDA(cudaStreamWaitEvent(link->stream,link->endRemoteComm,0));
  }
  PetscFunctionReturn(0);
}

/* Send/Put signals to remote ranks

 Input parameters:
  + n        - Number of remote ranks
  . sig      - Signal address in symmetric heap
  . sigdisp  - To i-th rank, use its signal at offset sigdisp[i]
  . ranks    - remote ranks
  - newval   - Set signals to this value
*/
__global__ static void NvshmemSendSignals(PetscInt n,uint64_t *sig,PetscInt *sigdisp,PetscMPIInt *ranks,uint64_t newval)
{
  int i = blockIdx.x*blockDim.x + threadIdx.x;

  /* Each thread puts one remote signal */
  if (i < n) nvshmemx_uint64_signal(sig+sigdisp[i],newval,ranks[i]);
}

/* Wait until local signals equal to the expected value and then set them to a new value

 Input parameters:
  + n        - Number of signals
  . sig      - Local signal address
  . expval   - expected value
  - newval   - Set signals to this new value
*/
__global__ static void NvshmemWaitSignals(PetscInt n,uint64_t *sig,uint64_t expval,uint64_t newval)
{
#if 0
  /* Akhil Langer@NVIDIA said using 1 thread and nvshmem_uint64_wait_until_all is better */
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  if (i < n) {
    nvshmem_signal_wait_until(sig+i,NVSHMEM_CMP_EQ,expval);
    sig[i] = newval;
  }
#else
  nvshmem_uint64_wait_until_all(sig,n,NULL/*no mask*/,NVSHMEM_CMP_EQ,expval);
  for (int i=0; i<n; i++) sig[i] = newval;
#endif
}

/* ===========================================================================================================

   A set of routines to support receiver initiated communication using the get method

    The getting protocol is:

    Sender has a send buf (sbuf) and a signal variable (ssig);  Receiver has a recv buf (rbuf) and a signal variable (rsig);
    All signal variables have an initial value 0.

    Sender:                                 |  Receiver:
  1.  Wait ssig be 0, then set it to 1
  2.  Pack data into stand alone sbuf       |
  3.  Put 1 to receiver's rsig              |   1. Wait rsig to be 1, then set it 0
                                            |   2. Get data from remote sbuf to local rbuf
                                            |   3. Put 1 to sender's ssig
                                            |   4. Unpack data from local rbuf
   ===========================================================================================================*/
/* PrePack operation -- since sender will overwrite the send buffer which the receiver might be getting data from.
   Sender waits for signals (from receivers) indicating receivers have finished getting data
*/
PetscErrorCode PetscSFLinkWaitSignalsOfCompletionOfGettingData_NVSHMEM(PetscSF sf,PetscSFLink link,PetscSFDirection direction)
{
  PetscSF_Basic     *bas = (PetscSF_Basic*)sf->data;
  uint64_t          *sig;
  PetscInt          n;

  PetscFunctionBegin;
  if (direction == PETSCSF_ROOT2LEAF) { /* leaf ranks are getting data */
    sig = link->rootSendSig;            /* leaf ranks set my rootSendsig */
    n   = bas->nRemoteLeafRanks;
  } else { /* LEAF2ROOT */
    sig = link->leafSendSig;
    n   = sf->nRemoteRootRanks;
  }

  if (n) {
    NvshmemWaitSignals<<<1,1,0,link->remoteCommStream>>>(n,sig,0,1); /* wait the signals to be 0, then set them to 1 */
    PetscCallCUDA(cudaGetLastError());
  }
  PetscFunctionReturn(0);
}

/* n thread blocks. Each takes in charge one remote rank */
__global__ static void GetDataFromRemotelyAccessible(PetscInt nsrcranks,PetscMPIInt *srcranks,const char *src,PetscInt *srcdisp,char *dst,PetscInt *dstdisp,PetscInt unitbytes)
{
  int               bid = blockIdx.x;
  PetscMPIInt       pe  = srcranks[bid];

  if (!nvshmem_ptr(src,pe)) {
    PetscInt nelems = (dstdisp[bid+1]-dstdisp[bid])*unitbytes;
    nvshmem_getmem_nbi(dst+(dstdisp[bid]-dstdisp[0])*unitbytes,src+srcdisp[bid]*unitbytes,nelems,pe);
  }
}

/* Start communication -- Get data in the given direction */
PetscErrorCode PetscSFLinkGetDataBegin_NVSHMEM(PetscSF sf,PetscSFLink link,PetscSFDirection direction)
{
  cudaError_t       cerr;
  PetscSF_Basic     *bas = (PetscSF_Basic*)sf->data;

  PetscInt          nsrcranks,ndstranks,nLocallyAccessible = 0;

  char              *src,*dst;
  PetscInt          *srcdisp_h,*dstdisp_h;
  PetscInt          *srcdisp_d,*dstdisp_d;
  PetscMPIInt       *srcranks_h;
  PetscMPIInt       *srcranks_d,*dstranks_d;
  uint64_t          *dstsig;
  PetscInt          *dstsigdisp_d;

  PetscFunctionBegin;
  PetscCall(PetscSFLinkBuildDependenceBegin(sf,link,direction));
  if (direction == PETSCSF_ROOT2LEAF) { /* src is root, dst is leaf; we will move data from src to dst */
    nsrcranks    = sf->nRemoteRootRanks;
    src          = link->rootbuf[PETSCSF_REMOTE][PETSC_MEMTYPE_DEVICE]; /* root buf is the send buf; it is in symmetric heap */

    srcdisp_h    = sf->rootbufdisp;       /* for my i-th remote root rank, I will access its buf at offset rootbufdisp[i] */
    srcdisp_d    = sf->rootbufdisp_d;
    srcranks_h   = sf->ranks+sf->ndranks; /* my (remote) root ranks */
    srcranks_d   = sf->ranks_d;

    ndstranks    = bas->nRemoteLeafRanks;
    dst          = link->leafbuf[PETSCSF_REMOTE][PETSC_MEMTYPE_DEVICE]; /* recv buf is the local leaf buf, also in symmetric heap */

    dstdisp_h    = sf->roffset+sf->ndranks; /* offsets of the local leaf buf. Note dstdisp[0] is not necessarily 0 */
    dstdisp_d    = sf->roffset_d;
    dstranks_d   = bas->iranks_d; /* my (remote) leaf ranks */

    dstsig       = link->leafRecvSig;
    dstsigdisp_d = bas->leafsigdisp_d;
  } else { /* src is leaf, dst is root; we will move data from src to dst */
    nsrcranks    = bas->nRemoteLeafRanks;
    src          = link->leafbuf[PETSCSF_REMOTE][PETSC_MEMTYPE_DEVICE]; /* leaf buf is the send buf */

    srcdisp_h    = bas->leafbufdisp;       /* for my i-th remote root rank, I will access its buf at offset rootbufdisp[i] */
    srcdisp_d    = bas->leafbufdisp_d;
    srcranks_h   = bas->iranks+bas->ndiranks; /* my (remote) root ranks */
    srcranks_d   = bas->iranks_d;

    ndstranks    = sf->nRemoteRootRanks;
    dst          = link->rootbuf[PETSCSF_REMOTE][PETSC_MEMTYPE_DEVICE]; /* the local root buf is the recv buf */

    dstdisp_h    = bas->ioffset+bas->ndiranks; /* offsets of the local root buf. Note dstdisp[0] is not necessarily 0 */
    dstdisp_d    = bas->ioffset_d;
    dstranks_d   = sf->ranks_d; /* my (remote) root ranks */

    dstsig       = link->rootRecvSig;
    dstsigdisp_d = sf->rootsigdisp_d;
  }

  /* After Pack operation -- src tells dst ranks that they are allowed to get data */
  if (ndstranks) {
    NvshmemSendSignals<<<(ndstranks+255)/256,256,0,link->remoteCommStream>>>(ndstranks,dstsig,dstsigdisp_d,dstranks_d,1); /* set signals to 1 */
    PetscCallCUDA(cudaGetLastError());
  }

  /* dst waits for signals (permissions) from src ranks to start getting data */
  if (nsrcranks) {
    NvshmemWaitSignals<<<1,1,0,link->remoteCommStream>>>(nsrcranks,dstsig,1,0); /* wait the signals to be 1, then set them to 0 */
    PetscCallCUDA(cudaGetLastError());
  }

  /* dst gets data from src ranks using non-blocking nvshmem_gets, which are finished in PetscSFLinkGetDataEnd_NVSHMEM() */

  /* Count number of locally accessible src ranks, which should be a small number */
  for (int i=0; i<nsrcranks; i++) {if (nvshmem_ptr(src,srcranks_h[i])) nLocallyAccessible++;}

  /* Get data from remotely accessible PEs */
  if (nLocallyAccessible < nsrcranks) {
    GetDataFromRemotelyAccessible<<<nsrcranks,1,0,link->remoteCommStream>>>(nsrcranks,srcranks_d,src,srcdisp_d,dst,dstdisp_d,link->unitbytes);
    PetscCallCUDA(cudaGetLastError());
  }

  /* Get data from locally accessible PEs */
  if (nLocallyAccessible) {
    for (int i=0; i<nsrcranks; i++) {
      int pe = srcranks_h[i];
      if (nvshmem_ptr(src,pe)) {
        size_t nelems = (dstdisp_h[i+1]-dstdisp_h[i])*link->unitbytes;
        nvshmemx_getmem_nbi_on_stream(dst+(dstdisp_h[i]-dstdisp_h[0])*link->unitbytes,src+srcdisp_h[i]*link->unitbytes,nelems,pe,link->remoteCommStream);
      }
    }
  }
  PetscFunctionReturn(0);
}

/* Finish the communication (can be done before Unpack)
   Receiver tells its senders that they are allowed to reuse their send buffer (since receiver has got data from their send buffer)
*/
PetscErrorCode PetscSFLinkGetDataEnd_NVSHMEM(PetscSF sf,PetscSFLink link,PetscSFDirection direction)
{
  cudaError_t       cerr;
  PetscSF_Basic     *bas = (PetscSF_Basic*)sf->data;
  uint64_t          *srcsig;
  PetscInt          nsrcranks,*srcsigdisp;
  PetscMPIInt       *srcranks;

  PetscFunctionBegin;
  if (direction == PETSCSF_ROOT2LEAF) { /* leaf ranks are getting data */
    nsrcranks   = sf->nRemoteRootRanks;
    srcsig      = link->rootSendSig;     /* I want to set their root signal */
    srcsigdisp  = sf->rootsigdisp_d;     /* offset of each root signal */
    srcranks    = sf->ranks_d;           /* ranks of the n root ranks */
  } else { /* LEAF2ROOT, root ranks are getting data */
    nsrcranks   = bas->nRemoteLeafRanks;
    srcsig      = link->leafSendSig;
    srcsigdisp  = bas->leafsigdisp_d;
    srcranks    = bas->iranks_d;
  }

  if (nsrcranks) {
    nvshmemx_quiet_on_stream(link->remoteCommStream); /* Finish the nonblocking get, so that we can unpack afterwards */
    PetscCallCUDA(cudaGetLastError());
    NvshmemSendSignals<<<(nsrcranks+511)/512,512,0,link->remoteCommStream>>>(nsrcranks,srcsig,srcsigdisp,srcranks,0); /* set signals to 0 */
    PetscCallCUDA(cudaGetLastError());
  }
  PetscCall(PetscSFLinkBuildDependenceEnd(sf,link,direction));
  PetscFunctionReturn(0);
}

/* ===========================================================================================================

   A set of routines to support sender initiated communication using the put-based method (the default)

    The putting protocol is:

    Sender has a send buf (sbuf) and a send signal var (ssig);  Receiver has a stand-alone recv buf (rbuf)
    and a recv signal var (rsig); All signal variables have an initial value 0. rbuf is allocated by SF and
    is in nvshmem space.

    Sender:                                 |  Receiver:
                                            |
  1.  Pack data into sbuf                   |
  2.  Wait ssig be 0, then set it to 1      |
  3.  Put data to remote stand-alone rbuf   |
  4.  Fence // make sure 5 happens after 3  |
  5.  Put 1 to receiver's rsig              |   1. Wait rsig to be 1, then set it 0
                                            |   2. Unpack data from local rbuf
                                            |   3. Put 0 to sender's ssig
   ===========================================================================================================*/

/* n thread blocks. Each takes in charge one remote rank */
__global__ static void WaitAndPutDataToRemotelyAccessible(PetscInt ndstranks,PetscMPIInt *dstranks,char *dst,PetscInt *dstdisp,const char *src,PetscInt *srcdisp,uint64_t *srcsig,PetscInt unitbytes)
{
  int               bid = blockIdx.x;
  PetscMPIInt       pe  = dstranks[bid];

  if (!nvshmem_ptr(dst,pe)) {
    PetscInt nelems = (srcdisp[bid+1]-srcdisp[bid])*unitbytes;
    nvshmem_uint64_wait_until(srcsig+bid,NVSHMEM_CMP_EQ,0); /* Wait until the sig = 0 */
    srcsig[bid] = 1;
    nvshmem_putmem_nbi(dst+dstdisp[bid]*unitbytes,src+(srcdisp[bid]-srcdisp[0])*unitbytes,nelems,pe);
  }
}

/* one-thread kernel, which takes in charge all locally accesible */
__global__ static void WaitSignalsFromLocallyAccessible(PetscInt ndstranks,PetscMPIInt *dstranks,uint64_t *srcsig,const char *dst)
{
  for (int i=0; i<ndstranks; i++) {
    int pe = dstranks[i];
    if (nvshmem_ptr(dst,pe)) {
      nvshmem_uint64_wait_until(srcsig+i,NVSHMEM_CMP_EQ,0); /* Wait until the sig = 0 */
      srcsig[i] = 1;
    }
  }
}

/* Put data in the given direction  */
PetscErrorCode PetscSFLinkPutDataBegin_NVSHMEM(PetscSF sf,PetscSFLink link,PetscSFDirection direction)
{
  cudaError_t       cerr;
  PetscSF_Basic     *bas = (PetscSF_Basic*)sf->data;
  PetscInt          ndstranks,nLocallyAccessible = 0;
  char              *src,*dst;
  PetscInt          *srcdisp_h,*dstdisp_h;
  PetscInt          *srcdisp_d,*dstdisp_d;
  PetscMPIInt       *dstranks_h;
  PetscMPIInt       *dstranks_d;
  uint64_t          *srcsig;

  PetscFunctionBegin;
  PetscCall(PetscSFLinkBuildDependenceBegin(sf,link,direction));
  if (direction == PETSCSF_ROOT2LEAF) { /* put data in rootbuf to leafbuf  */
    ndstranks    = bas->nRemoteLeafRanks; /* number of (remote) leaf ranks */
    src          = link->rootbuf[PETSCSF_REMOTE][PETSC_MEMTYPE_DEVICE]; /* Both src & dst must be symmetric */
    dst          = link->leafbuf[PETSCSF_REMOTE][PETSC_MEMTYPE_DEVICE];

    srcdisp_h    = bas->ioffset+bas->ndiranks;  /* offsets of rootbuf. srcdisp[0] is not necessarily zero */
    srcdisp_d    = bas->ioffset_d;
    srcsig       = link->rootSendSig;

    dstdisp_h    = bas->leafbufdisp;            /* for my i-th remote leaf rank, I will access its leaf buf at offset leafbufdisp[i] */
    dstdisp_d    = bas->leafbufdisp_d;
    dstranks_h   = bas->iranks+bas->ndiranks;   /* remote leaf ranks */
    dstranks_d   = bas->iranks_d;
  } else { /* put data in leafbuf to rootbuf */
    ndstranks    = sf->nRemoteRootRanks;
    src          = link->leafbuf[PETSCSF_REMOTE][PETSC_MEMTYPE_DEVICE];
    dst          = link->rootbuf[PETSCSF_REMOTE][PETSC_MEMTYPE_DEVICE];

    srcdisp_h    = sf->roffset+sf->ndranks; /* offsets of leafbuf */
    srcdisp_d    = sf->roffset_d;
    srcsig       = link->leafSendSig;

    dstdisp_h    = sf->rootbufdisp;         /* for my i-th remote root rank, I will access its root buf at offset rootbufdisp[i] */
    dstdisp_d    = sf->rootbufdisp_d;
    dstranks_h   = sf->ranks+sf->ndranks;   /* remote root ranks */
    dstranks_d   = sf->ranks_d;
  }

  /* Wait for signals and then put data to dst ranks using non-blocking nvshmem_put, which are finished in PetscSFLinkPutDataEnd_NVSHMEM */

  /* Count number of locally accessible neighbors, which should be a small number */
  for (int i=0; i<ndstranks; i++) {if (nvshmem_ptr(dst,dstranks_h[i])) nLocallyAccessible++;}

  /* For remotely accessible PEs, send data to them in one kernel call */
  if (nLocallyAccessible < ndstranks) {
    WaitAndPutDataToRemotelyAccessible<<<ndstranks,1,0,link->remoteCommStream>>>(ndstranks,dstranks_d,dst,dstdisp_d,src,srcdisp_d,srcsig,link->unitbytes);
    PetscCallCUDA(cudaGetLastError());
  }

  /* For locally accessible PEs, use host API, which uses CUDA copy-engines and is much faster than device API */
  if (nLocallyAccessible) {
    WaitSignalsFromLocallyAccessible<<<1,1,0,link->remoteCommStream>>>(ndstranks,dstranks_d,srcsig,dst);
    for (int i=0; i<ndstranks; i++) {
      int pe = dstranks_h[i];
      if (nvshmem_ptr(dst,pe)) { /* If return a non-null pointer, then <pe> is locally accessible */
        size_t nelems = (srcdisp_h[i+1]-srcdisp_h[i])*link->unitbytes;
         /* Initiate the nonblocking communication */
        nvshmemx_putmem_nbi_on_stream(dst+dstdisp_h[i]*link->unitbytes,src+(srcdisp_h[i]-srcdisp_h[0])*link->unitbytes,nelems,pe,link->remoteCommStream);
      }
    }
  }

  if (nLocallyAccessible) {
    nvshmemx_quiet_on_stream(link->remoteCommStream); /* Calling nvshmem_fence/quiet() does not fence the above nvshmemx_putmem_nbi_on_stream! */
  }
  PetscFunctionReturn(0);
}

/* A one-thread kernel. The thread takes in charge all remote PEs */
__global__ static void PutDataEnd(PetscInt nsrcranks,PetscInt ndstranks,PetscMPIInt *dstranks,uint64_t *dstsig,PetscInt *dstsigdisp)
{
  /* TODO: Shall we finished the non-blocking remote puts? */

  /* 1. Send a signal to each dst rank */

  /* According to Akhil@NVIDIA, IB is orderred, so no fence is needed for remote PEs.
     For local PEs, we already called nvshmemx_quiet_on_stream(). Therefore, we are good to send signals to all dst ranks now.
  */
  for (int i=0; i<ndstranks; i++) {nvshmemx_uint64_signal(dstsig+dstsigdisp[i],1,dstranks[i]);} /* set sig to 1 */

  /* 2. Wait for signals from src ranks (if any) */
  if (nsrcranks) {
    nvshmem_uint64_wait_until_all(dstsig,nsrcranks,NULL/*no mask*/,NVSHMEM_CMP_EQ,1); /* wait sigs to be 1, then set them to 0 */
    for (int i=0; i<nsrcranks; i++) dstsig[i] = 0;
  }
}

/* Finish the communication -- A receiver waits until it can access its receive buffer */
PetscErrorCode PetscSFLinkPutDataEnd_NVSHMEM(PetscSF sf,PetscSFLink link,PetscSFDirection direction)
{
  cudaError_t       cerr;
  PetscSF_Basic     *bas = (PetscSF_Basic*)sf->data;
  PetscMPIInt       *dstranks;
  uint64_t          *dstsig;
  PetscInt          nsrcranks,ndstranks,*dstsigdisp;

  PetscFunctionBegin;
  if (direction == PETSCSF_ROOT2LEAF) { /* put root data to leaf */
    nsrcranks    = sf->nRemoteRootRanks;

    ndstranks    = bas->nRemoteLeafRanks;
    dstranks     = bas->iranks_d;       /* leaf ranks */
    dstsig       = link->leafRecvSig;   /* I will set my leaf ranks's RecvSig */
    dstsigdisp   = bas->leafsigdisp_d;  /* for my i-th remote leaf rank, I will access its signal at offset leafsigdisp[i] */
  } else { /* LEAF2ROOT */
    nsrcranks    = bas->nRemoteLeafRanks;

    ndstranks    = sf->nRemoteRootRanks;
    dstranks     = sf->ranks_d;
    dstsig       = link->rootRecvSig;
    dstsigdisp   = sf->rootsigdisp_d;
  }

  if (nsrcranks || ndstranks) {
    PutDataEnd<<<1,1,0,link->remoteCommStream>>>(nsrcranks,ndstranks,dstranks,dstsig,dstsigdisp);
    PetscCallCUDA(cudaGetLastError());
  }
  PetscCall(PetscSFLinkBuildDependenceEnd(sf,link,direction));
  PetscFunctionReturn(0);
}

/* PostUnpack operation -- A receiver tells its senders that they are allowed to put data to here (it implies recv buf is free to take new data) */
PetscErrorCode PetscSFLinkSendSignalsToAllowPuttingData_NVSHMEM(PetscSF sf,PetscSFLink link,PetscSFDirection direction)
{
  PetscSF_Basic     *bas = (PetscSF_Basic*)sf->data;
  uint64_t          *srcsig;
  PetscInt          nsrcranks,*srcsigdisp_d;
  PetscMPIInt       *srcranks_d;

  PetscFunctionBegin;
  if (direction == PETSCSF_ROOT2LEAF) { /* I allow my root ranks to put data to me */
    nsrcranks    = sf->nRemoteRootRanks;
    srcsig       = link->rootSendSig;      /* I want to set their send signals */
    srcsigdisp_d = sf->rootsigdisp_d;      /* offset of each root signal */
    srcranks_d   = sf->ranks_d;            /* ranks of the n root ranks */
  } else { /* LEAF2ROOT */
    nsrcranks    = bas->nRemoteLeafRanks;
    srcsig       = link->leafSendSig;
    srcsigdisp_d = bas->leafsigdisp_d;
    srcranks_d   = bas->iranks_d;
  }

  if (nsrcranks) {
    NvshmemSendSignals<<<(nsrcranks+255)/256,256,0,link->remoteCommStream>>>(nsrcranks,srcsig,srcsigdisp_d,srcranks_d,0); /* Set remote signals to 0 */
    PetscCallCUDA(cudaGetLastError());
  }
  PetscFunctionReturn(0);
}

/* Destructor when the link uses nvshmem for communication */
static PetscErrorCode PetscSFLinkDestroy_NVSHMEM(PetscSF sf,PetscSFLink link)
{
  cudaError_t       cerr;

  PetscFunctionBegin;
  PetscCallCUDA(cudaEventDestroy(link->dataReady));
  PetscCallCUDA(cudaEventDestroy(link->endRemoteComm));
  PetscCallCUDA(cudaStreamDestroy(link->remoteCommStream));

  /* nvshmem does not need buffers on host, which should be NULL */
  PetscCall(PetscNvshmemFree(link->leafbuf_alloc[PETSCSF_REMOTE][PETSC_MEMTYPE_DEVICE]));
  PetscCall(PetscNvshmemFree(link->leafSendSig));
  PetscCall(PetscNvshmemFree(link->leafRecvSig));
  PetscCall(PetscNvshmemFree(link->rootbuf_alloc[PETSCSF_REMOTE][PETSC_MEMTYPE_DEVICE]));
  PetscCall(PetscNvshmemFree(link->rootSendSig));
  PetscCall(PetscNvshmemFree(link->rootRecvSig));
  PetscFunctionReturn(0);
}

PetscErrorCode PetscSFLinkCreate_NVSHMEM(PetscSF sf,MPI_Datatype unit,PetscMemType rootmtype,const void *rootdata,PetscMemType leafmtype,const void *leafdata,MPI_Op op,PetscSFOperation sfop,PetscSFLink *mylink)
{
  cudaError_t       cerr;
  PetscSF_Basic     *bas = (PetscSF_Basic*)sf->data;
  PetscSFLink       *p,link;
  PetscBool         match,rootdirect[2],leafdirect[2];
  int               greatestPriority;

  PetscFunctionBegin;
  /* Check to see if we can directly send/recv root/leafdata with the given sf, sfop and op.
     We only care root/leafdirect[PETSCSF_REMOTE], since we never need intermeidate buffers in local communication with NVSHMEM.
  */
  if (sfop == PETSCSF_BCAST) { /* Move data from rootbuf to leafbuf */
    if (sf->use_nvshmem_get) {
      rootdirect[PETSCSF_REMOTE] = PETSC_FALSE; /* send buffer has to be stand-alone (can't be rootdata) */
      leafdirect[PETSCSF_REMOTE] = (PetscMemTypeNVSHMEM(leafmtype) && sf->leafcontig[PETSCSF_REMOTE] && op == MPI_REPLACE) ? PETSC_TRUE : PETSC_FALSE;
    } else {
      rootdirect[PETSCSF_REMOTE] = (PetscMemTypeNVSHMEM(rootmtype) && bas->rootcontig[PETSCSF_REMOTE]) ? PETSC_TRUE : PETSC_FALSE;
      leafdirect[PETSCSF_REMOTE] = PETSC_FALSE;  /* Our put-protocol always needs a nvshmem alloc'ed recv buffer */
    }
  } else if (sfop == PETSCSF_REDUCE) { /* Move data from leafbuf to rootbuf */
    if (sf->use_nvshmem_get) {
      rootdirect[PETSCSF_REMOTE] = (PetscMemTypeNVSHMEM(rootmtype) && bas->rootcontig[PETSCSF_REMOTE] && op == MPI_REPLACE) ? PETSC_TRUE : PETSC_FALSE;
      leafdirect[PETSCSF_REMOTE] = PETSC_FALSE;
    } else {
      rootdirect[PETSCSF_REMOTE] = PETSC_FALSE;
      leafdirect[PETSCSF_REMOTE] = (PetscMemTypeNVSHMEM(leafmtype) && sf->leafcontig[PETSCSF_REMOTE]) ? PETSC_TRUE : PETSC_FALSE;
    }
  } else { /* PETSCSF_FETCH */
    rootdirect[PETSCSF_REMOTE] = PETSC_FALSE; /* FETCH always need a separate rootbuf */
    leafdirect[PETSCSF_REMOTE] = PETSC_FALSE; /* We also force allocating a separate leafbuf so that leafdata and leafupdate can share mpi requests */
  }

  /* Look for free nvshmem links in cache */
  for (p=&bas->avail; (link=*p); p=&link->next) {
    if (link->use_nvshmem) {
      PetscCall(MPIPetsc_Type_compare(unit,link->unit,&match));
      if (match) {
        *p = link->next; /* Remove from available list */
        goto found;
      }
    }
  }
  PetscCall(PetscNew(&link));
  PetscCall(PetscSFLinkSetUp_Host(sf,link,unit)); /* Compute link->unitbytes, dup link->unit etc. */
  if (sf->backend == PETSCSF_BACKEND_CUDA) PetscCall(PetscSFLinkSetUp_CUDA(sf,link,unit)); /* Setup pack routines, streams etc */
 #if defined(PETSC_HAVE_KOKKOS)
  else if (sf->backend == PETSCSF_BACKEND_KOKKOS) PetscCall(PetscSFLinkSetUp_Kokkos(sf,link,unit));
 #endif

  link->rootdirect[PETSCSF_LOCAL]  = PETSC_TRUE; /* For the local part we directly use root/leafdata */
  link->leafdirect[PETSCSF_LOCAL]  = PETSC_TRUE;

  /* Init signals to zero */
  if (!link->rootSendSig) PetscCall(PetscNvshmemCalloc(bas->nRemoteLeafRanksMax*sizeof(uint64_t),(void**)&link->rootSendSig));
  if (!link->rootRecvSig) PetscCall(PetscNvshmemCalloc(bas->nRemoteLeafRanksMax*sizeof(uint64_t),(void**)&link->rootRecvSig));
  if (!link->leafSendSig) PetscCall(PetscNvshmemCalloc(sf->nRemoteRootRanksMax*sizeof(uint64_t),(void**)&link->leafSendSig));
  if (!link->leafRecvSig) PetscCall(PetscNvshmemCalloc(sf->nRemoteRootRanksMax*sizeof(uint64_t),(void**)&link->leafRecvSig));

  link->use_nvshmem                = PETSC_TRUE;
  link->rootmtype                  = PETSC_MEMTYPE_DEVICE; /* Only need 0/1-based mtype from now on */
  link->leafmtype                  = PETSC_MEMTYPE_DEVICE;
  /* Overwrite some function pointers set by PetscSFLinkSetUp_CUDA */
  link->Destroy                    = PetscSFLinkDestroy_NVSHMEM;
  if (sf->use_nvshmem_get) { /* get-based protocol */
    link->PrePack                  = PetscSFLinkWaitSignalsOfCompletionOfGettingData_NVSHMEM;
    link->StartCommunication       = PetscSFLinkGetDataBegin_NVSHMEM;
    link->FinishCommunication      = PetscSFLinkGetDataEnd_NVSHMEM;
  } else { /* put-based protocol */
    link->StartCommunication       = PetscSFLinkPutDataBegin_NVSHMEM;
    link->FinishCommunication      = PetscSFLinkPutDataEnd_NVSHMEM;
    link->PostUnpack               = PetscSFLinkSendSignalsToAllowPuttingData_NVSHMEM;
  }

  PetscCallCUDA(cudaDeviceGetStreamPriorityRange(NULL,&greatestPriority));
  PetscCallCUDA(cudaStreamCreateWithPriority(&link->remoteCommStream,cudaStreamNonBlocking,greatestPriority));

  PetscCallCUDA(cudaEventCreateWithFlags(&link->dataReady,cudaEventDisableTiming));
  PetscCallCUDA(cudaEventCreateWithFlags(&link->endRemoteComm,cudaEventDisableTiming));

found:
  if (rootdirect[PETSCSF_REMOTE]) {
    link->rootbuf[PETSCSF_REMOTE][PETSC_MEMTYPE_DEVICE] = (char*)rootdata + bas->rootstart[PETSCSF_REMOTE]*link->unitbytes;
  } else {
    if (!link->rootbuf_alloc[PETSCSF_REMOTE][PETSC_MEMTYPE_DEVICE]) {
      PetscCall(PetscNvshmemMalloc(bas->rootbuflen_rmax*link->unitbytes,(void**)&link->rootbuf_alloc[PETSCSF_REMOTE][PETSC_MEMTYPE_DEVICE]));
    }
    link->rootbuf[PETSCSF_REMOTE][PETSC_MEMTYPE_DEVICE] = link->rootbuf_alloc[PETSCSF_REMOTE][PETSC_MEMTYPE_DEVICE];
  }

  if (leafdirect[PETSCSF_REMOTE]) {
    link->leafbuf[PETSCSF_REMOTE][PETSC_MEMTYPE_DEVICE] = (char*)leafdata + sf->leafstart[PETSCSF_REMOTE]*link->unitbytes;
  } else {
    if (!link->leafbuf_alloc[PETSCSF_REMOTE][PETSC_MEMTYPE_DEVICE]) {
      PetscCall(PetscNvshmemMalloc(sf->leafbuflen_rmax*link->unitbytes,(void**)&link->leafbuf_alloc[PETSCSF_REMOTE][PETSC_MEMTYPE_DEVICE]));
    }
    link->leafbuf[PETSCSF_REMOTE][PETSC_MEMTYPE_DEVICE] = link->leafbuf_alloc[PETSCSF_REMOTE][PETSC_MEMTYPE_DEVICE];
  }

  link->rootdirect[PETSCSF_REMOTE] = rootdirect[PETSCSF_REMOTE];
  link->leafdirect[PETSCSF_REMOTE] = leafdirect[PETSCSF_REMOTE];
  link->rootdata                   = rootdata; /* root/leafdata are keys to look up links in PetscSFXxxEnd */
  link->leafdata                   = leafdata;
  link->next                       = bas->inuse;
  bas->inuse                       = link;
  *mylink                          = link;
  PetscFunctionReturn(0);
}

#if defined(PETSC_USE_REAL_SINGLE)
PetscErrorCode PetscNvshmemSum(PetscInt count,float *dst,const float *src)
{
  PetscMPIInt       num; /* Assume nvshmem's int is MPI's int */

  PetscFunctionBegin;
  PetscCall(PetscMPIIntCast(count,&num));
  nvshmemx_float_sum_reduce_on_stream(NVSHMEM_TEAM_WORLD,dst,src,num,PetscDefaultCudaStream);
  PetscFunctionReturn(0);
}

PetscErrorCode PetscNvshmemMax(PetscInt count,float *dst,const float *src)
{
  PetscMPIInt       num;

  PetscFunctionBegin;
  PetscCall(PetscMPIIntCast(count,&num));
  nvshmemx_float_max_reduce_on_stream(NVSHMEM_TEAM_WORLD,dst,src,num,PetscDefaultCudaStream);
  PetscFunctionReturn(0);
}
#elif defined(PETSC_USE_REAL_DOUBLE)
PetscErrorCode PetscNvshmemSum(PetscInt count,double *dst,const double *src)
{
  PetscMPIInt       num;

  PetscFunctionBegin;
  PetscCall(PetscMPIIntCast(count,&num));
  nvshmemx_double_sum_reduce_on_stream(NVSHMEM_TEAM_WORLD,dst,src,num,PetscDefaultCudaStream);
  PetscFunctionReturn(0);
}

PetscErrorCode PetscNvshmemMax(PetscInt count,double *dst,const double *src)
{
  PetscMPIInt       num;

  PetscFunctionBegin;
  PetscCall(PetscMPIIntCast(count,&num));
  nvshmemx_double_max_reduce_on_stream(NVSHMEM_TEAM_WORLD,dst,src,num,PetscDefaultCudaStream);
  PetscFunctionReturn(0);
}
#endif
