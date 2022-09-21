#include <../src/vec/is/sf/impls/basic/allgatherv/sfallgatherv.h>

PETSC_INTERN PetscErrorCode PetscSFBcastBegin_Gatherv(PetscSF, MPI_Datatype, PetscMemType, const void *, PetscMemType, void *, MPI_Op);

/* PetscSFGetGraph is non-collective. An implementation should not have collective calls */
PETSC_INTERN PetscErrorCode PetscSFGetGraph_Allgatherv(PetscSF sf, PetscInt *nroots, PetscInt *nleaves, const PetscInt **ilocal, const PetscSFNode **iremote)
{
  PetscInt        i, j, k;
  const PetscInt *range;
  PetscMPIInt     size;

  PetscFunctionBegin;
  PetscCallMPI(MPI_Comm_size(PetscObjectComm((PetscObject)sf), &size));
  if (nroots) *nroots = sf->nroots;
  if (nleaves) *nleaves = sf->nleaves;
  if (ilocal) *ilocal = NULL; /* Contiguous leaves */
  if (iremote) {
    if (!sf->remote && sf->nleaves) { /* The && sf->nleaves makes sfgatherv able to inherit this routine */
      PetscCall(PetscLayoutGetRanges(sf->map, &range));
      PetscCall(PetscMalloc1(sf->nleaves, &sf->remote));
      sf->remote_alloc = sf->remote;
      for (i = 0; i < size; i++) {
        for (j = range[i], k = 0; j < range[i + 1]; j++, k++) {
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
  PetscSF_Allgatherv *dat = (PetscSF_Allgatherv *)sf->data;
  PetscMPIInt         size;
  PetscInt            i;
  const PetscInt     *range;
  MPI_Comm            comm;

  PetscFunctionBegin;
  PetscCall(PetscSFSetUp_Allgather(sf));
  PetscCall(PetscObjectGetComm((PetscObject)sf, &comm));
  PetscCallMPI(MPI_Comm_size(comm, &size));
  if (sf->nleaves) { /* This if (sf->nleaves) test makes sfgatherv able to inherit this routine */
    PetscBool isallgatherv = PETSC_FALSE;

    PetscCall(PetscMalloc1(size, &dat->recvcounts));
    PetscCall(PetscMalloc1(size, &dat->displs));
    PetscCall(PetscLayoutGetRanges(sf->map, &range));

    for (i = 0; i < size; i++) {
      PetscCall(PetscMPIIntCast(range[i], &dat->displs[i]));
      PetscCall(PetscMPIIntCast(range[i + 1] - range[i], &dat->recvcounts[i]));
    }

    /* check if we actually have a one-to-all pattern */
    PetscCall(PetscObjectTypeCompare((PetscObject)sf, PETSCSFALLGATHERV, &isallgatherv));
    if (isallgatherv) {
      PetscMPIInt rank, nRanksWithZeroRoots;

      nRanksWithZeroRoots = (sf->nroots == 0) ? 1 : 0; /* I have no roots */
      PetscCallMPI(MPIU_Allreduce(MPI_IN_PLACE, &nRanksWithZeroRoots, 1, MPI_INT, MPI_SUM, comm));
      if (nRanksWithZeroRoots == size - 1) { /* Only one rank has roots, which indicates a bcast pattern */
        dat->bcast_pattern = PETSC_TRUE;
        PetscCallMPI(MPI_Comm_rank(comm, &rank));
        dat->bcast_root = sf->nroots > 0 ? rank : -1;
        PetscCallMPI(MPIU_Allreduce(MPI_IN_PLACE, &dat->bcast_root, 1, MPI_INT, MPI_MAX, comm));
      }
    }
  }
  PetscFunctionReturn(0);
}

PETSC_INTERN PetscErrorCode PetscSFReset_Allgatherv(PetscSF sf)
{
  PetscSF_Allgatherv *dat  = (PetscSF_Allgatherv *)sf->data;
  PetscSFLink         link = dat->avail, next;

  PetscFunctionBegin;
  PetscCall(PetscFree(dat->iranks));
  PetscCall(PetscFree(dat->ioffset));
  PetscCall(PetscFree(dat->irootloc));
  PetscCall(PetscFree(dat->recvcounts));
  PetscCall(PetscFree(dat->displs));
  PetscCheck(!dat->inuse, PetscObjectComm((PetscObject)sf), PETSC_ERR_ARG_WRONGSTATE, "Outstanding operation has not been completed");
  for (; link; link = next) {
    next = link->next;
    PetscCall(PetscSFLinkDestroy(sf, link));
  }
  dat->avail = NULL;
  PetscFunctionReturn(0);
}

PETSC_INTERN PetscErrorCode PetscSFDestroy_Allgatherv(PetscSF sf)
{
  PetscFunctionBegin;
  PetscCall(PetscSFReset_Allgatherv(sf));
  PetscCall(PetscFree(sf->data));
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscSFBcastBegin_Allgatherv(PetscSF sf, MPI_Datatype unit, PetscMemType rootmtype, const void *rootdata, PetscMemType leafmtype, void *leafdata, MPI_Op op)
{
  PetscSFLink         link;
  PetscMPIInt         sendcount, rank;
  MPI_Comm            comm;
  void               *rootbuf = NULL, *leafbuf = NULL;
  MPI_Request        *req;
  PetscSF_Allgatherv *dat = (PetscSF_Allgatherv *)sf->data;

  PetscFunctionBegin;
  PetscCall(PetscSFLinkCreate(sf, unit, rootmtype, rootdata, leafmtype, leafdata, op, PETSCSF_BCAST, &link));
  PetscCall(PetscSFLinkPackRootData(sf, link, PETSCSF_REMOTE, rootdata));
  PetscCall(PetscSFLinkCopyRootBufferInCaseNotUseGpuAwareMPI(sf, link, PETSC_TRUE /* device2host before sending */));
  PetscCall(PetscObjectGetComm((PetscObject)sf, &comm));
  PetscCallMPI(MPI_Comm_rank(comm, &rank));
  PetscCall(PetscMPIIntCast(sf->nroots, &sendcount));
  PetscCall(PetscSFLinkGetMPIBuffersAndRequests(sf, link, PETSCSF_ROOT2LEAF, &rootbuf, &leafbuf, &req, NULL));

  if (dat->bcast_pattern && rank == dat->bcast_root) PetscCall((*link->Memcpy)(link, link->leafmtype_mpi, leafbuf, link->rootmtype_mpi, rootbuf, (size_t)sendcount * link->unitbytes));
  /* Ready the buffers for MPI */
  PetscCall(PetscSFLinkSyncStreamBeforeCallMPI(sf, link, PETSCSF_ROOT2LEAF));
  if (dat->bcast_pattern) PetscCallMPI(MPIU_Ibcast(leafbuf, sf->nleaves, unit, dat->bcast_root, comm, req));
  else PetscCallMPI(MPIU_Iallgatherv(rootbuf, sendcount, unit, leafbuf, dat->recvcounts, dat->displs, unit, comm, req));
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscSFReduceBegin_Allgatherv(PetscSF sf, MPI_Datatype unit, PetscMemType leafmtype, const void *leafdata, PetscMemType rootmtype, void *rootdata, MPI_Op op)
{
  PetscSFLink         link;
  PetscSF_Allgatherv *dat = (PetscSF_Allgatherv *)sf->data;
  PetscInt            rstart;
  PetscMPIInt         rank, count, recvcount;
  MPI_Comm            comm;
  void               *rootbuf = NULL, *leafbuf = NULL;
  MPI_Request        *req;

  PetscFunctionBegin;
  PetscCall(PetscSFLinkCreate(sf, unit, rootmtype, rootdata, leafmtype, leafdata, op, PETSCSF_REDUCE, &link));
  if (op == MPI_REPLACE) {
    /* REPLACE is only meaningful when all processes have the same leafdata to reduce. Therefore copying from local leafdata is fine */
    PetscCall(PetscLayoutGetRange(sf->map, &rstart, NULL));
    PetscCall((*link->Memcpy)(link, rootmtype, rootdata, leafmtype, (const char *)leafdata + (size_t)rstart * link->unitbytes, (size_t)sf->nroots * link->unitbytes));
    if (PetscMemTypeDevice(leafmtype) && PetscMemTypeHost(rootmtype)) PetscCall((*link->SyncStream)(link));
  } else {
    PetscCall(PetscObjectGetComm((PetscObject)sf, &comm));
    PetscCall(PetscSFLinkPackLeafData(sf, link, PETSCSF_REMOTE, leafdata));
    PetscCall(PetscSFLinkCopyLeafBufferInCaseNotUseGpuAwareMPI(sf, link, PETSC_TRUE /* device2host before sending */));
    PetscCall(PetscSFLinkGetMPIBuffersAndRequests(sf, link, PETSCSF_LEAF2ROOT, &rootbuf, &leafbuf, &req, NULL));
    PetscCall(PetscSFLinkSyncStreamBeforeCallMPI(sf, link, PETSCSF_LEAF2ROOT));
    if (dat->bcast_pattern) {
#if defined(PETSC_HAVE_OMPI_MAJOR_VERSION) /* Workaround: cuda-aware OpenMPI-4.1.3 does not support MPI_Ireduce() with device buffers */
      *req = MPI_REQUEST_NULL;             /* Set NULL so that we can safely MPI_Wait(req) */
      PetscCallMPI(MPI_Reduce(leafbuf, rootbuf, sf->nleaves, unit, op, dat->bcast_root, comm));
#else
      PetscCallMPI(MPIU_Ireduce(leafbuf, rootbuf, sf->nleaves, unit, op, dat->bcast_root, comm, req));
#endif
    } else { /* Reduce leafdata, then scatter to rootdata */
      PetscCallMPI(MPI_Comm_rank(comm, &rank));
      PetscCall(PetscMPIIntCast(dat->rootbuflen[PETSCSF_REMOTE], &recvcount));
      /* Allocate a separate leaf buffer on rank 0 */
      if (rank == 0 && !link->leafbuf_alloc[PETSCSF_REMOTE][link->leafmtype_mpi]) {
        PetscCall(PetscSFMalloc(sf, link->leafmtype_mpi, sf->leafbuflen[PETSCSF_REMOTE] * link->unitbytes, (void **)&link->leafbuf_alloc[PETSCSF_REMOTE][link->leafmtype_mpi]));
      }
      /* In case we already copied leafdata from device to host (i.e., no use_gpu_aware_mpi), we need to adjust leafbuf on rank 0 */
      if (rank == 0 && link->leafbuf_alloc[PETSCSF_REMOTE][link->leafmtype_mpi] == leafbuf) leafbuf = MPI_IN_PLACE;
      PetscCall(PetscMPIIntCast(sf->nleaves * link->bs, &count));
      PetscCallMPI(MPI_Reduce(leafbuf, link->leafbuf_alloc[PETSCSF_REMOTE][link->leafmtype_mpi], count, link->basicunit, op, 0, comm)); /* Must do reduce with MPI builtin datatype basicunit */
      PetscCallMPI(MPIU_Iscatterv(link->leafbuf_alloc[PETSCSF_REMOTE][link->leafmtype_mpi], dat->recvcounts, dat->displs, unit, rootbuf, recvcount, unit, 0, comm, req));
    }
  }
  PetscFunctionReturn(0);
}

PETSC_INTERN PetscErrorCode PetscSFReduceEnd_Allgatherv(PetscSF sf, MPI_Datatype unit, const void *leafdata, void *rootdata, MPI_Op op)
{
  PetscSFLink link;

  PetscFunctionBegin;
  if (op == MPI_REPLACE) {
    /* A rare case happens when op is MPI_REPLACE, using GPUs but no GPU aware MPI. In PetscSFReduceBegin_Allgather(v),
      we did a device to device copy and in effect finished the communication. But in PetscSFLinkFinishCommunication()
      of PetscSFReduceEnd_Basic(), it thinks since there is rootbuf, it calls PetscSFLinkCopyRootBufferInCaseNotUseGpuAwareMPI().
      It does a host to device memory copy on rootbuf, wrongly overwriting the results. So we don't overload
      PetscSFReduceEnd_Basic() in this case, and just reclaim the link.
     */
    PetscCall(PetscSFLinkGetInUse(sf, unit, rootdata, leafdata, PETSC_OWN_POINTER, &link));
    PetscCall(PetscSFLinkReclaim(sf, &link));
  } else {
    PetscCall(PetscSFReduceEnd_Basic(sf, unit, leafdata, rootdata, op));
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscSFBcastToZero_Allgatherv(PetscSF sf, MPI_Datatype unit, PetscMemType rootmtype, const void *rootdata, PetscMemType leafmtype, void *leafdata)
{
  PetscSFLink link;
  PetscMPIInt rank;

  PetscFunctionBegin;
  PetscCall(PetscSFBcastBegin_Gatherv(sf, unit, rootmtype, rootdata, leafmtype, leafdata, MPI_REPLACE));
  PetscCall(PetscSFLinkGetInUse(sf, unit, rootdata, leafdata, PETSC_OWN_POINTER, &link));
  PetscCall(PetscSFLinkFinishCommunication(sf, link, PETSCSF_ROOT2LEAF));
  PetscCallMPI(MPI_Comm_rank(PetscObjectComm((PetscObject)sf), &rank));
  if (rank == 0 && PetscMemTypeDevice(leafmtype) && !sf->use_gpu_aware_mpi) PetscCall((*link->Memcpy)(link, PETSC_MEMTYPE_DEVICE, leafdata, PETSC_MEMTYPE_HOST, link->leafbuf[PETSC_MEMTYPE_HOST], sf->leafbuflen[PETSCSF_REMOTE] * link->unitbytes));
  PetscCall(PetscSFLinkReclaim(sf, &link));
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
PETSC_INTERN PetscErrorCode PetscSFFetchAndOpBegin_Allgatherv(PetscSF sf, MPI_Datatype unit, PetscMemType rootmtype, void *rootdata, PetscMemType leafmtype, const void *leafdata, void *leafupdate, MPI_Op op)
{
  PetscSFLink link;
  MPI_Comm    comm;
  PetscMPIInt count;

  PetscFunctionBegin;
  PetscCall(PetscObjectGetComm((PetscObject)sf, &comm));
  PetscCheck(!PetscMemTypeDevice(rootmtype) && !PetscMemTypeDevice(leafmtype), comm, PETSC_ERR_SUP, "Do FetchAndOp on device");
  /* Copy leafdata to leafupdate */
  PetscCall(PetscSFLinkCreate(sf, unit, rootmtype, rootdata, leafmtype, leafdata, op, PETSCSF_FETCH, &link));
  PetscCall(PetscSFLinkPackLeafData(sf, link, PETSCSF_REMOTE, leafdata)); /* Sync the device */
  PetscCall((*link->Memcpy)(link, leafmtype, leafupdate, leafmtype, leafdata, sf->nleaves * link->unitbytes));
  PetscCall(PetscSFLinkGetInUse(sf, unit, rootdata, leafdata, PETSC_OWN_POINTER, &link));

  /* Exscan on leafupdate and then BcastAndOp rootdata to leafupdate */
  if (op == MPI_REPLACE) {
    PetscMPIInt size, rank, prev, next;
    PetscCallMPI(MPI_Comm_rank(comm, &rank));
    PetscCallMPI(MPI_Comm_size(comm, &size));
    prev = rank ? rank - 1 : MPI_PROC_NULL;
    next = (rank < size - 1) ? rank + 1 : MPI_PROC_NULL;
    PetscCall(PetscMPIIntCast(sf->nleaves, &count));
    PetscCallMPI(MPI_Sendrecv_replace(leafupdate, count, unit, next, link->tag, prev, link->tag, comm, MPI_STATUSES_IGNORE));
  } else {
    PetscCall(PetscMPIIntCast(sf->nleaves * link->bs, &count));
    PetscCallMPI(MPI_Exscan(MPI_IN_PLACE, leafupdate, count, link->basicunit, op, comm));
  }
  PetscCall(PetscSFLinkReclaim(sf, &link));
  PetscCall(PetscSFBcastBegin(sf, unit, rootdata, leafupdate, op));
  PetscCall(PetscSFBcastEnd(sf, unit, rootdata, leafupdate, op));

  /* Bcast roots to rank 0's leafupdate */
  PetscCall(PetscSFBcastToZero_Private(sf, unit, rootdata, leafupdate)); /* Using this line makes Allgather SFs able to inherit this routine */

  /* Reduce leafdata to rootdata */
  PetscCall(PetscSFReduceBegin(sf, unit, leafdata, rootdata, op));
  PetscFunctionReturn(0);
}

PETSC_INTERN PetscErrorCode PetscSFFetchAndOpEnd_Allgatherv(PetscSF sf, MPI_Datatype unit, void *rootdata, const void *leafdata, void *leafupdate, MPI_Op op)
{
  PetscFunctionBegin;
  PetscCall(PetscSFReduceEnd(sf, unit, leafdata, rootdata, op));
  PetscFunctionReturn(0);
}

/* Get root ranks accessing my leaves */
PETSC_INTERN PetscErrorCode PetscSFGetRootRanks_Allgatherv(PetscSF sf, PetscInt *nranks, const PetscMPIInt **ranks, const PetscInt **roffset, const PetscInt **rmine, const PetscInt **rremote)
{
  PetscInt        i, j, k, size;
  const PetscInt *range;

  PetscFunctionBegin;
  /* Lazily construct these large arrays if users really need them for this type of SF. Very likely, they do not */
  if (sf->nranks && !sf->ranks) { /* On rank!=0, sf->nranks=0. The sf->nranks test makes this routine also works for sfgatherv */
    size = sf->nranks;
    PetscCall(PetscLayoutGetRanges(sf->map, &range));
    PetscCall(PetscMalloc4(size, &sf->ranks, size + 1, &sf->roffset, sf->nleaves, &sf->rmine, sf->nleaves, &sf->rremote));
    for (i = 0; i < size; i++) sf->ranks[i] = i;
    PetscCall(PetscArraycpy(sf->roffset, range, size + 1));
    for (i = 0; i < sf->nleaves; i++) sf->rmine[i] = i; /*rmine are never NULL even for contiguous leaves */
    for (i = 0; i < size; i++) {
      for (j = range[i], k = 0; j < range[i + 1]; j++, k++) sf->rremote[j] = k;
    }
  }

  if (nranks) *nranks = sf->nranks;
  if (ranks) *ranks = sf->ranks;
  if (roffset) *roffset = sf->roffset;
  if (rmine) *rmine = sf->rmine;
  if (rremote) *rremote = sf->rremote;
  PetscFunctionReturn(0);
}

/* Get leaf ranks accessing my roots */
PETSC_INTERN PetscErrorCode PetscSFGetLeafRanks_Allgatherv(PetscSF sf, PetscInt *niranks, const PetscMPIInt **iranks, const PetscInt **ioffset, const PetscInt **irootloc)
{
  PetscSF_Allgatherv *dat = (PetscSF_Allgatherv *)sf->data;
  MPI_Comm            comm;
  PetscMPIInt         size, rank;
  PetscInt            i, j;

  PetscFunctionBegin;
  /* Lazily construct these large arrays if users really need them for this type of SF. Very likely, they do not */
  PetscCall(PetscObjectGetComm((PetscObject)sf, &comm));
  PetscCallMPI(MPI_Comm_size(comm, &size));
  PetscCallMPI(MPI_Comm_rank(comm, &rank));
  if (niranks) *niranks = size;

  /* PetscSF_Basic has distinguished incoming ranks. Here we do not need that. But we must put self as the first and
     sort other ranks. See comments in PetscSFSetUp_Basic about MatGetBrowsOfAoCols_MPIAIJ on why.
   */
  if (iranks) {
    if (!dat->iranks) {
      PetscCall(PetscMalloc1(size, &dat->iranks));
      dat->iranks[0] = rank;
      for (i = 0, j = 1; i < size; i++) {
        if (i == rank) continue;
        dat->iranks[j++] = i;
      }
    }
    *iranks = dat->iranks; /* dat->iranks was init'ed to NULL by PetscNew */
  }

  if (ioffset) {
    if (!dat->ioffset) {
      PetscCall(PetscMalloc1(size + 1, &dat->ioffset));
      for (i = 0; i <= size; i++) dat->ioffset[i] = i * sf->nroots;
    }
    *ioffset = dat->ioffset;
  }

  if (irootloc) {
    if (!dat->irootloc) {
      PetscCall(PetscMalloc1(sf->nleaves, &dat->irootloc));
      for (i = 0; i < size; i++) {
        for (j = 0; j < sf->nroots; j++) dat->irootloc[i * sf->nroots + j] = j;
      }
    }
    *irootloc = dat->irootloc;
  }
  PetscFunctionReturn(0);
}

PETSC_INTERN PetscErrorCode PetscSFCreateLocalSF_Allgatherv(PetscSF sf, PetscSF *out)
{
  PetscInt     i, nroots, nleaves, rstart, *ilocal;
  PetscSFNode *iremote;
  PetscSF      lsf;

  PetscFunctionBegin;
  nleaves = sf->nleaves ? sf->nroots : 0; /* sf->nleaves can be zero with SFGather(v) */
  nroots  = nleaves;
  PetscCall(PetscMalloc1(nleaves, &ilocal));
  PetscCall(PetscMalloc1(nleaves, &iremote));
  PetscCall(PetscLayoutGetRange(sf->map, &rstart, NULL));

  for (i = 0; i < nleaves; i++) {
    ilocal[i]        = rstart + i; /* lsf does not change leave indices */
    iremote[i].rank  = 0;          /* rank in PETSC_COMM_SELF */
    iremote[i].index = i;          /* root index */
  }

  PetscCall(PetscSFCreate(PETSC_COMM_SELF, &lsf));
  PetscCall(PetscSFSetGraph(lsf, nroots, nleaves, ilocal, PETSC_OWN_POINTER, iremote, PETSC_OWN_POINTER));
  PetscCall(PetscSFSetUp(lsf));
  *out = lsf;
  PetscFunctionReturn(0);
}

PETSC_INTERN PetscErrorCode PetscSFCreate_Allgatherv(PetscSF sf)
{
  PetscSF_Allgatherv *dat = (PetscSF_Allgatherv *)sf->data;

  PetscFunctionBegin;
  sf->ops->BcastEnd  = PetscSFBcastEnd_Basic;
  sf->ops->ReduceEnd = PetscSFReduceEnd_Allgatherv;

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

  PetscCall(PetscNew(&dat));
  dat->bcast_root = -1;
  sf->data        = (void *)dat;
  PetscFunctionReturn(0);
}
