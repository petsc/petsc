#include <../src/vec/is/sf/impls/basic/allgatherv/sfallgatherv.h>

PETSC_INTERN PetscErrorCode PetscSFBcastBegin_Gatherv(PetscSF,MPI_Datatype,PetscMemType,const void*,PetscMemType,void*,MPI_Op);

/* PetscSFGetGraph is non-collective. An implementation should not have collective calls */
PETSC_INTERN PetscErrorCode PetscSFGetGraph_Allgatherv(PetscSF sf,PetscInt *nroots,PetscInt *nleaves,const PetscInt **ilocal,const PetscSFNode **iremote)
{
  PetscInt       i,j,k;
  const PetscInt *range;
  PetscMPIInt    size;

  PetscFunctionBegin;
  CHKERRMPI(MPI_Comm_size(PetscObjectComm((PetscObject)sf),&size));
  if (nroots)  *nroots  = sf->nroots;
  if (nleaves) *nleaves = sf->nleaves;
  if (ilocal)  *ilocal  = NULL; /* Contiguous leaves */
  if (iremote) {
    if (!sf->remote && sf->nleaves) { /* The && sf->nleaves makes sfgatherv able to inherit this routine */
      CHKERRQ(PetscLayoutGetRanges(sf->map,&range));
      CHKERRQ(PetscMalloc1(sf->nleaves,&sf->remote));
      sf->remote_alloc = sf->remote;
      for (i=0; i<size; i++) {
        for (j=range[i],k=0; j<range[i+1]; j++,k++) {
          sf->remote[j].rank  = i;
          sf->remote[j].index = k;
        }
      }
    }
    *iremote = sf->remote;
  }
  PetscFunctionReturn(0);
}

PETSC_INTERN PetscErrorCode PetscSFSetUp_Allgatherv(PetscSF sf)
{
  PetscSF_Allgatherv *dat = (PetscSF_Allgatherv*)sf->data;
  PetscMPIInt        size;
  PetscInt           i;
  const PetscInt     *range;

  PetscFunctionBegin;
  CHKERRQ(PetscSFSetUp_Allgather(sf));
  CHKERRMPI(MPI_Comm_size(PetscObjectComm((PetscObject)sf),&size));
  if (sf->nleaves) { /* This if (sf->nleaves) test makes sfgatherv able to inherit this routine */
    CHKERRQ(PetscMalloc1(size,&dat->recvcounts));
    CHKERRQ(PetscMalloc1(size,&dat->displs));
    CHKERRQ(PetscLayoutGetRanges(sf->map,&range));

    for (i=0; i<size; i++) {
      CHKERRQ(PetscMPIIntCast(range[i],&dat->displs[i]));
      CHKERRQ(PetscMPIIntCast(range[i+1]-range[i],&dat->recvcounts[i]));
    }
  }
  PetscFunctionReturn(0);
}

PETSC_INTERN PetscErrorCode PetscSFReset_Allgatherv(PetscSF sf)
{
  PetscSF_Allgatherv     *dat = (PetscSF_Allgatherv*)sf->data;
  PetscSFLink            link = dat->avail,next;

  PetscFunctionBegin;
  CHKERRQ(PetscFree(dat->iranks));
  CHKERRQ(PetscFree(dat->ioffset));
  CHKERRQ(PetscFree(dat->irootloc));
  CHKERRQ(PetscFree(dat->recvcounts));
  CHKERRQ(PetscFree(dat->displs));
  PetscCheckFalse(dat->inuse,PetscObjectComm((PetscObject)sf),PETSC_ERR_ARG_WRONGSTATE,"Outstanding operation has not been completed");
  for (; link; link=next) {next = link->next; CHKERRQ(PetscSFLinkDestroy(sf,link));}
  dat->avail = NULL;
  PetscFunctionReturn(0);
}

PETSC_INTERN PetscErrorCode PetscSFDestroy_Allgatherv(PetscSF sf)
{
  PetscFunctionBegin;
  CHKERRQ(PetscSFReset_Allgatherv(sf));
  CHKERRQ(PetscFree(sf->data));
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscSFBcastBegin_Allgatherv(PetscSF sf,MPI_Datatype unit,PetscMemType rootmtype,const void *rootdata,PetscMemType leafmtype,void *leafdata,MPI_Op op)
{
  PetscSFLink            link;
  PetscMPIInt            sendcount;
  MPI_Comm               comm;
  void                   *rootbuf = NULL,*leafbuf = NULL;
  MPI_Request            *req;
  PetscSF_Allgatherv     *dat = (PetscSF_Allgatherv*)sf->data;

  PetscFunctionBegin;
  CHKERRQ(PetscSFLinkCreate(sf,unit,rootmtype,rootdata,leafmtype,leafdata,op,PETSCSF_BCAST,&link));
  CHKERRQ(PetscSFLinkPackRootData(sf,link,PETSCSF_REMOTE,rootdata));
  CHKERRQ(PetscSFLinkCopyRootBufferInCaseNotUseGpuAwareMPI(sf,link,PETSC_TRUE/* device2host before sending */));
  CHKERRQ(PetscObjectGetComm((PetscObject)sf,&comm));
  CHKERRQ(PetscMPIIntCast(sf->nroots,&sendcount));
  CHKERRQ(PetscSFLinkGetMPIBuffersAndRequests(sf,link,PETSCSF_ROOT2LEAF,&rootbuf,&leafbuf,&req,NULL));
  CHKERRQ(PetscSFLinkSyncStreamBeforeCallMPI(sf,link,PETSCSF_ROOT2LEAF));
  CHKERRMPI(MPIU_Iallgatherv(rootbuf,sendcount,unit,leafbuf,dat->recvcounts,dat->displs,unit,comm,req));
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscSFReduceBegin_Allgatherv(PetscSF sf,MPI_Datatype unit,PetscMemType leafmtype,const void *leafdata,PetscMemType rootmtype,void *rootdata,MPI_Op op)
{
  PetscSFLink            link;
  PetscSF_Allgatherv     *dat = (PetscSF_Allgatherv*)sf->data;
  PetscInt               rstart;
  PetscMPIInt            rank,count,recvcount;
  MPI_Comm               comm;
  void                   *rootbuf = NULL,*leafbuf = NULL;
  MPI_Request            *req;

  PetscFunctionBegin;
  CHKERRQ(PetscSFLinkCreate(sf,unit,rootmtype,rootdata,leafmtype,leafdata,op,PETSCSF_REDUCE,&link));
  if (op == MPI_REPLACE) {
    /* REPLACE is only meaningful when all processes have the same leafdata to reduce. Therefore copying from local leafdata is fine */
    CHKERRQ(PetscLayoutGetRange(sf->map,&rstart,NULL));
    CHKERRQ((*link->Memcpy)(link,rootmtype,rootdata,leafmtype,(const char*)leafdata+(size_t)rstart*link->unitbytes,(size_t)sf->nroots*link->unitbytes));
    if (PetscMemTypeDevice(leafmtype) && PetscMemTypeHost(rootmtype)) CHKERRQ((*link->SyncStream)(link));
  } else {
    /* Reduce leafdata, then scatter to rootdata */
    CHKERRQ(PetscObjectGetComm((PetscObject)sf,&comm));
    CHKERRMPI(MPI_Comm_rank(comm,&rank));
    CHKERRQ(PetscSFLinkPackLeafData(sf,link,PETSCSF_REMOTE,leafdata));
    CHKERRQ(PetscSFLinkCopyLeafBufferInCaseNotUseGpuAwareMPI(sf,link,PETSC_TRUE/* device2host before sending */));
    CHKERRQ(PetscSFLinkGetMPIBuffersAndRequests(sf,link,PETSCSF_LEAF2ROOT,&rootbuf,&leafbuf,&req,NULL));
    CHKERRQ(PetscMPIIntCast(dat->rootbuflen[PETSCSF_REMOTE],&recvcount));
    /* Allocate a separate leaf buffer on rank 0 */
    if (rank == 0 && !link->leafbuf_alloc[PETSCSF_REMOTE][link->leafmtype_mpi]) {
      CHKERRQ(PetscSFMalloc(sf,link->leafmtype_mpi,sf->leafbuflen[PETSCSF_REMOTE]*link->unitbytes,(void**)&link->leafbuf_alloc[PETSCSF_REMOTE][link->leafmtype_mpi]));
    }
    /* In case we already copied leafdata from device to host (i.e., no use_gpu_aware_mpi), we need to adjust leafbuf on rank 0 */
    if (rank == 0 && link->leafbuf_alloc[PETSCSF_REMOTE][link->leafmtype_mpi] == leafbuf) leafbuf = MPI_IN_PLACE;
    CHKERRQ(PetscMPIIntCast(sf->nleaves*link->bs,&count));
    CHKERRQ(PetscSFLinkSyncStreamBeforeCallMPI(sf,link,PETSCSF_LEAF2ROOT));
    CHKERRMPI(MPI_Reduce(leafbuf,link->leafbuf_alloc[PETSCSF_REMOTE][link->leafmtype_mpi],count,link->basicunit,op,0,comm)); /* Must do reduce with MPI builltin datatype basicunit */
    CHKERRMPI(MPIU_Iscatterv(link->leafbuf_alloc[PETSCSF_REMOTE][link->leafmtype_mpi],dat->recvcounts,dat->displs,unit,rootbuf,recvcount,unit,0,comm,req));
  }
  PetscFunctionReturn(0);
}

PETSC_INTERN PetscErrorCode PetscSFReduceEnd_Allgatherv(PetscSF sf,MPI_Datatype unit,const void *leafdata,void *rootdata,MPI_Op op)
{
  PetscSFLink           link;

  PetscFunctionBegin;
  if (op == MPI_REPLACE) {
    /* A rare case happens when op is MPI_REPLACE, using GPUs but no GPU aware MPI. In PetscSFReduceBegin_Allgather(v),
      we did a device to device copy and in effect finished the communication. But in PetscSFLinkFinishCommunication()
      of PetscSFReduceEnd_Basic(), it thinks since there is rootbuf, it calls PetscSFLinkCopyRootBufferInCaseNotUseGpuAwareMPI().
      It does a host to device memory copy on rootbuf, wrongly overwritting the results. So we don't overload
      PetscSFReduceEnd_Basic() in this case, and just reclaim the link.
     */
    CHKERRQ(PetscSFLinkGetInUse(sf,unit,rootdata,leafdata,PETSC_OWN_POINTER,&link));
    CHKERRQ(PetscSFLinkReclaim(sf,&link));
  } else {
    CHKERRQ(PetscSFReduceEnd_Basic(sf,unit,leafdata,rootdata,op));
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscSFBcastToZero_Allgatherv(PetscSF sf,MPI_Datatype unit,PetscMemType rootmtype,const void *rootdata,PetscMemType leafmtype,void *leafdata)
{
  PetscSFLink            link;
  PetscMPIInt            rank;

  PetscFunctionBegin;
  CHKERRQ(PetscSFBcastBegin_Gatherv(sf,unit,rootmtype,rootdata,leafmtype,leafdata,MPI_REPLACE));
  CHKERRQ(PetscSFLinkGetInUse(sf,unit,rootdata,leafdata,PETSC_OWN_POINTER,&link));
  CHKERRQ(PetscSFLinkFinishCommunication(sf,link,PETSCSF_ROOT2LEAF));
  CHKERRMPI(MPI_Comm_rank(PetscObjectComm((PetscObject)sf),&rank));
  if (rank == 0 && PetscMemTypeDevice(leafmtype) && !sf->use_gpu_aware_mpi) {
    CHKERRQ((*link->Memcpy)(link,PETSC_MEMTYPE_DEVICE,leafdata,PETSC_MEMTYPE_HOST,link->leafbuf[PETSC_MEMTYPE_HOST],sf->leafbuflen[PETSCSF_REMOTE]*link->unitbytes));
  }
  CHKERRQ(PetscSFLinkReclaim(sf,&link));
  PetscFunctionReturn(0);
}

/* This routine is very tricky (I believe it is rarely used with this kind of graph so just provide a simple but not-optimal implementation).

   Suppose we have three ranks. Rank 0 has a root with value 1. Rank 0,1,2 has a leaf with value 2,3,4 respectively. The leaves are connected
   to the root on rank 0. Suppose op=MPI_SUM and rank 0,1,2 gets root state in their rank order. By definition of this routine, rank 0 sees 1
   in root, fetches it into its leafupate, then updates root to 1 + 2 = 3; rank 1 sees 3 in root, fetches it into its leafupate, then updates
   root to 3 + 3 = 6; rank 2 sees 6 in root, fetches it into its leafupdate, then updates root to 6 + 4 = 10.  At the end, leafupdate on rank
   0,1,2 is 1,3,6 respectively. root is 10.

   We use a simpler implementation. From the same initial state, we copy leafdata to leafupdate
             rank-0   rank-1    rank-2
        Root     1
        Leaf     2       3         4
     Leafupdate  2       3         4

   Do MPI_Exscan on leafupdate,
             rank-0   rank-1    rank-2
        Root     1
        Leaf     2       3         4
     Leafupdate  2       2         5

   BcastAndOp from root to leafupdate,
             rank-0   rank-1    rank-2
        Root     1
        Leaf     2       3         4
     Leafupdate  3       3         6

   Copy root to leafupdate on rank-0
             rank-0   rank-1    rank-2
        Root     1
        Leaf     2       3         4
     Leafupdate  1       3         6

   Reduce from leaf to root,
             rank-0   rank-1    rank-2
        Root     10
        Leaf     2       3         4
     Leafupdate  1       3         6
*/
PETSC_INTERN PetscErrorCode PetscSFFetchAndOpBegin_Allgatherv(PetscSF sf,MPI_Datatype unit,PetscMemType rootmtype,void *rootdata,PetscMemType leafmtype,const void *leafdata,void *leafupdate,MPI_Op op)
{
  PetscSFLink            link;
  MPI_Comm               comm;
  PetscMPIInt            count;

  PetscFunctionBegin;
  CHKERRQ(PetscObjectGetComm((PetscObject)sf,&comm));
  PetscCheckFalse(PetscMemTypeDevice(rootmtype) || PetscMemTypeDevice(leafmtype),comm,PETSC_ERR_SUP,"Do FetchAndOp on device");
  /* Copy leafdata to leafupdate */
  CHKERRQ(PetscSFLinkCreate(sf,unit,rootmtype,rootdata,leafmtype,leafdata,op,PETSCSF_FETCH,&link));
  CHKERRQ(PetscSFLinkPackLeafData(sf,link,PETSCSF_REMOTE,leafdata)); /* Sync the device */
  CHKERRQ((*link->Memcpy)(link,leafmtype,leafupdate,leafmtype,leafdata,sf->nleaves*link->unitbytes));
  CHKERRQ(PetscSFLinkGetInUse(sf,unit,rootdata,leafdata,PETSC_OWN_POINTER,&link));

  /* Exscan on leafupdate and then BcastAndOp rootdata to leafupdate */
  if (op == MPI_REPLACE) {
    PetscMPIInt size,rank,prev,next;
    CHKERRMPI(MPI_Comm_rank(comm,&rank));
    CHKERRMPI(MPI_Comm_size(comm,&size));
    prev = rank ?            rank-1 : MPI_PROC_NULL;
    next = (rank < size-1) ? rank+1 : MPI_PROC_NULL;
    CHKERRQ(PetscMPIIntCast(sf->nleaves,&count));
    CHKERRMPI(MPI_Sendrecv_replace(leafupdate,count,unit,next,link->tag,prev,link->tag,comm,MPI_STATUSES_IGNORE));
  } else {
    CHKERRQ(PetscMPIIntCast(sf->nleaves*link->bs,&count));
    CHKERRMPI(MPI_Exscan(MPI_IN_PLACE,leafupdate,count,link->basicunit,op,comm));
  }
  CHKERRQ(PetscSFLinkReclaim(sf,&link));
  CHKERRQ(PetscSFBcastBegin(sf,unit,rootdata,leafupdate,op));
  CHKERRQ(PetscSFBcastEnd(sf,unit,rootdata,leafupdate,op));

  /* Bcast roots to rank 0's leafupdate */
  CHKERRQ(PetscSFBcastToZero_Private(sf,unit,rootdata,leafupdate)); /* Using this line makes Allgather SFs able to inherit this routine */

  /* Reduce leafdata to rootdata */
  CHKERRQ(PetscSFReduceBegin(sf,unit,leafdata,rootdata,op));
  PetscFunctionReturn(0);
}

PETSC_INTERN PetscErrorCode PetscSFFetchAndOpEnd_Allgatherv(PetscSF sf,MPI_Datatype unit,void *rootdata,const void *leafdata,void *leafupdate,MPI_Op op)
{
  PetscFunctionBegin;
  CHKERRQ(PetscSFReduceEnd(sf,unit,leafdata,rootdata,op));
  PetscFunctionReturn(0);
}

/* Get root ranks accessing my leaves */
PETSC_INTERN PetscErrorCode PetscSFGetRootRanks_Allgatherv(PetscSF sf,PetscInt *nranks,const PetscMPIInt **ranks,const PetscInt **roffset,const PetscInt **rmine,const PetscInt **rremote)
{
  PetscInt       i,j,k,size;
  const PetscInt *range;

  PetscFunctionBegin;
  /* Lazily construct these large arrays if users really need them for this type of SF. Very likely, they do not */
  if (sf->nranks && !sf->ranks) { /* On rank!=0, sf->nranks=0. The sf->nranks test makes this routine also works for sfgatherv */
    size = sf->nranks;
    CHKERRQ(PetscLayoutGetRanges(sf->map,&range));
    CHKERRQ(PetscMalloc4(size,&sf->ranks,size+1,&sf->roffset,sf->nleaves,&sf->rmine,sf->nleaves,&sf->rremote));
    for (i=0; i<size; i++) sf->ranks[i] = i;
    CHKERRQ(PetscArraycpy(sf->roffset,range,size+1));
    for (i=0; i<sf->nleaves; i++) sf->rmine[i] = i; /*rmine are never NULL even for contiguous leaves */
    for (i=0; i<size; i++) {
      for (j=range[i],k=0; j<range[i+1]; j++,k++) sf->rremote[j] = k;
    }
  }

  if (nranks)  *nranks  = sf->nranks;
  if (ranks)   *ranks   = sf->ranks;
  if (roffset) *roffset = sf->roffset;
  if (rmine)   *rmine   = sf->rmine;
  if (rremote) *rremote = sf->rremote;
  PetscFunctionReturn(0);
}

/* Get leaf ranks accessing my roots */
PETSC_INTERN PetscErrorCode PetscSFGetLeafRanks_Allgatherv(PetscSF sf,PetscInt *niranks,const PetscMPIInt **iranks,const PetscInt **ioffset,const PetscInt **irootloc)
{
  PetscSF_Allgatherv *dat = (PetscSF_Allgatherv*)sf->data;
  MPI_Comm           comm;
  PetscMPIInt        size,rank;
  PetscInt           i,j;

  PetscFunctionBegin;
  /* Lazily construct these large arrays if users really need them for this type of SF. Very likely, they do not */
  CHKERRQ(PetscObjectGetComm((PetscObject)sf,&comm));
  CHKERRMPI(MPI_Comm_size(comm,&size));
  CHKERRMPI(MPI_Comm_rank(comm,&rank));
  if (niranks) *niranks = size;

  /* PetscSF_Basic has distinguished incoming ranks. Here we do not need that. But we must put self as the first and
     sort other ranks. See comments in PetscSFSetUp_Basic about MatGetBrowsOfAoCols_MPIAIJ on why.
   */
  if (iranks) {
    if (!dat->iranks) {
      CHKERRQ(PetscMalloc1(size,&dat->iranks));
      dat->iranks[0] = rank;
      for (i=0,j=1; i<size; i++) {if (i == rank) continue; dat->iranks[j++] = i;}
    }
    *iranks = dat->iranks; /* dat->iranks was init'ed to NULL by PetscNewLog */
  }

  if (ioffset) {
    if (!dat->ioffset) {
      CHKERRQ(PetscMalloc1(size+1,&dat->ioffset));
      for (i=0; i<=size; i++) dat->ioffset[i] = i*sf->nroots;
    }
    *ioffset = dat->ioffset;
  }

  if (irootloc) {
    if (!dat->irootloc) {
      CHKERRQ(PetscMalloc1(sf->nleaves,&dat->irootloc));
      for (i=0; i<size; i++) {
        for (j=0; j<sf->nroots; j++) dat->irootloc[i*sf->nroots+j] = j;
      }
    }
    *irootloc = dat->irootloc;
  }
  PetscFunctionReturn(0);
}

PETSC_INTERN PetscErrorCode PetscSFCreateLocalSF_Allgatherv(PetscSF sf,PetscSF *out)
{
  PetscInt       i,nroots,nleaves,rstart,*ilocal;
  PetscSFNode    *iremote;
  PetscSF        lsf;

  PetscFunctionBegin;
  nleaves = sf->nleaves ? sf->nroots : 0; /* sf->nleaves can be zero with SFGather(v) */
  nroots  = nleaves;
  CHKERRQ(PetscMalloc1(nleaves,&ilocal));
  CHKERRQ(PetscMalloc1(nleaves,&iremote));
  CHKERRQ(PetscLayoutGetRange(sf->map,&rstart,NULL));

  for (i=0; i<nleaves; i++) {
    ilocal[i]        = rstart + i; /* lsf does not change leave indices */
    iremote[i].rank  = 0;          /* rank in PETSC_COMM_SELF */
    iremote[i].index = i;          /* root index */
  }

  CHKERRQ(PetscSFCreate(PETSC_COMM_SELF,&lsf));
  CHKERRQ(PetscSFSetGraph(lsf,nroots,nleaves,ilocal,PETSC_OWN_POINTER,iremote,PETSC_OWN_POINTER));
  CHKERRQ(PetscSFSetUp(lsf));
  *out = lsf;
  PetscFunctionReturn(0);
}

PETSC_INTERN PetscErrorCode PetscSFCreate_Allgatherv(PetscSF sf)
{
  PetscSF_Allgatherv *dat = (PetscSF_Allgatherv*)sf->data;

  PetscFunctionBegin;
  sf->ops->BcastEnd        = PetscSFBcastEnd_Basic;
  sf->ops->ReduceEnd       = PetscSFReduceEnd_Allgatherv;

  sf->ops->SetUp           = PetscSFSetUp_Allgatherv;
  sf->ops->Reset           = PetscSFReset_Allgatherv;
  sf->ops->Destroy         = PetscSFDestroy_Allgatherv;
  sf->ops->GetRootRanks    = PetscSFGetRootRanks_Allgatherv;
  sf->ops->GetLeafRanks    = PetscSFGetLeafRanks_Allgatherv;
  sf->ops->GetGraph        = PetscSFGetGraph_Allgatherv;
  sf->ops->BcastBegin      = PetscSFBcastBegin_Allgatherv;
  sf->ops->ReduceBegin     = PetscSFReduceBegin_Allgatherv;
  sf->ops->FetchAndOpBegin = PetscSFFetchAndOpBegin_Allgatherv;
  sf->ops->FetchAndOpEnd   = PetscSFFetchAndOpEnd_Allgatherv;
  sf->ops->CreateLocalSF   = PetscSFCreateLocalSF_Allgatherv;
  sf->ops->BcastToZero     = PetscSFBcastToZero_Allgatherv;

  CHKERRQ(PetscNewLog(sf,&dat));
  sf->data = (void*)dat;
  PetscFunctionReturn(0);
}
