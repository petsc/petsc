#include <../src/vec/is/sf/impls/basic/sfpack.h>
#include <../src/vec/is/sf/impls/basic/sfbasic.h>

/* Convenience local types */
#if defined(PETSC_HAVE_MPI_LARGE_COUNT) && defined(PETSC_USE_64BIT_INDICES)
typedef MPI_Count PetscSFCount;
typedef MPI_Aint  PetscSFAint;
#else
typedef PetscMPIInt PetscSFCount;
typedef PetscMPIInt PetscSFAint;
#endif

typedef struct {
  SFBASICHEADER;
  MPI_Comm      comms[2];                /* Communicators with distributed topology in both directions */
  PetscBool     initialized[2];          /* Are the two communicators initialized? */
  PetscSFCount *rootcounts, *leafcounts; /* counts for non-distinguished ranks */
  PetscSFAint  *rootdispls, *leafdispls; /* displs for non-distinguished ranks */
  PetscMPIInt  *rootweights, *leafweights;
  PetscInt      rootdegree, leafdegree;
} PetscSF_Neighbor;

/*===================================================================================*/
/*              Internal utility routines                                            */
/*===================================================================================*/

static inline PetscErrorCode PetscLogMPIMessages(PetscInt nsend, PetscSFCount *sendcnts, MPI_Datatype sendtype, PetscInt nrecv, PetscSFCount *recvcnts, MPI_Datatype recvtype)
{
  PetscFunctionBegin;
  if (PetscDefined(USE_LOG)) {
    petsc_isend_ct += (PetscLogDouble)nsend;
    petsc_irecv_ct += (PetscLogDouble)nrecv;

    if (sendtype != MPI_DATATYPE_NULL) {
      PetscMPIInt i, typesize;
      PetscCallMPI(MPI_Type_size(sendtype, &typesize));
      for (i = 0; i < nsend; i++) petsc_isend_len += (PetscLogDouble)(sendcnts[i] * typesize);
    }

    if (recvtype != MPI_DATATYPE_NULL) {
      PetscMPIInt i, typesize;
      PetscCallMPI(MPI_Type_size(recvtype, &typesize));
      for (i = 0; i < nrecv; i++) petsc_irecv_len += (PetscLogDouble)(recvcnts[i] * typesize);
    }
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* Get the communicator with distributed graph topology, which is not cheap to build so we do it on demand (instead of at PetscSFSetUp time) */
static PetscErrorCode PetscSFGetDistComm_Neighbor(PetscSF sf, PetscSFDirection direction, MPI_Comm *distcomm)
{
  PetscSF_Neighbor *dat = (PetscSF_Neighbor *)sf->data;

  PetscFunctionBegin;
  if (!dat->initialized[direction]) {
    PetscInt           nrootranks, ndrootranks, nleafranks, ndleafranks;
    PetscMPIInt        indegree, outdegree;
    const PetscMPIInt *rootranks, *leafranks, *sources, *destinations;
    MPI_Comm           comm, *mycomm = &dat->comms[direction];

    PetscCall(PetscSFGetRootInfo_Basic(sf, &nrootranks, &ndrootranks, &rootranks, NULL, NULL));       /* Which ranks will access my roots (I am a destination) */
    PetscCall(PetscSFGetLeafInfo_Basic(sf, &nleafranks, &ndleafranks, &leafranks, NULL, NULL, NULL)); /* My leaves will access whose roots (I am a source) */
    indegree     = nrootranks - ndrootranks;
    outdegree    = nleafranks - ndleafranks;
    sources      = rootranks + ndrootranks;
    destinations = leafranks + ndleafranks;
    PetscCall(PetscObjectGetComm((PetscObject)sf, &comm));
    if (direction == PETSCSF_LEAF2ROOT) {
      PetscCallMPI(MPI_Dist_graph_create_adjacent(comm, indegree, sources, dat->rootweights, outdegree, destinations, dat->leafweights, MPI_INFO_NULL, 1 /*reorder*/, mycomm));
    } else { /* PETSCSF_ROOT2LEAF, reverse src & dest */
      PetscCallMPI(MPI_Dist_graph_create_adjacent(comm, outdegree, destinations, dat->leafweights, indegree, sources, dat->rootweights, MPI_INFO_NULL, 1 /*reorder*/, mycomm));
    }
    dat->initialized[direction] = PETSC_TRUE;
  }
  *distcomm = dat->comms[direction];
  PetscFunctionReturn(PETSC_SUCCESS);
}

// start MPI_Ineighbor_alltoallv (only used for inter-proccess communication)
static PetscErrorCode PetscSFLinkStartCommunication_Neighbor(PetscSF sf, PetscSFLink link, PetscSFDirection direction)
{
  PetscSF_Neighbor *dat      = (PetscSF_Neighbor *)sf->data;
  MPI_Comm          distcomm = MPI_COMM_NULL;
  void             *rootbuf = NULL, *leafbuf = NULL;
  MPI_Request      *req = NULL;

  PetscFunctionBegin;
  if (direction == PETSCSF_ROOT2LEAF) {
    PetscCall(PetscSFLinkCopyRootBufferInCaseNotUseGpuAwareMPI(sf, link, PETSC_TRUE /* device2host before sending */));
  } else {
    PetscCall(PetscSFLinkCopyLeafBufferInCaseNotUseGpuAwareMPI(sf, link, PETSC_TRUE /* device2host */));
  }

  PetscCall(PetscSFGetDistComm_Neighbor(sf, direction, &distcomm));
  PetscCall(PetscSFLinkGetMPIBuffersAndRequests(sf, link, direction, &rootbuf, &leafbuf, &req, NULL));
  PetscCall(PetscSFLinkSyncStreamBeforeCallMPI(sf, link, direction));

  if (dat->rootdegree || dat->leafdegree) { // OpenMPI-3.0 ran into error with rootdegree = leafdegree = 0, so we skip the call in this case
    if (direction == PETSCSF_ROOT2LEAF) {
      PetscCallMPI(MPIU_Ineighbor_alltoallv(rootbuf, dat->rootcounts, dat->rootdispls, link->unit, leafbuf, dat->leafcounts, dat->leafdispls, link->unit, distcomm, req));
      PetscCall(PetscLogMPIMessages(dat->rootdegree, dat->rootcounts, link->unit, dat->leafdegree, dat->leafcounts, link->unit));
    } else {
      PetscCallMPI(MPIU_Ineighbor_alltoallv(leafbuf, dat->leafcounts, dat->leafdispls, link->unit, rootbuf, dat->rootcounts, dat->rootdispls, link->unit, distcomm, req));
      PetscCall(PetscLogMPIMessages(dat->leafdegree, dat->leafcounts, link->unit, dat->rootdegree, dat->rootcounts, link->unit));
    }
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscSFSetCommunicationOps_Neighbor(PetscSF sf, PetscSFLink link)
{
  PetscFunctionBegin;
  link->StartCommunication = PetscSFLinkStartCommunication_Neighbor;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*===================================================================================*/
/*              Implementations of SF public APIs                                    */
/*===================================================================================*/
static PetscErrorCode PetscSFSetUp_Neighbor(PetscSF sf)
{
  PetscSF_Neighbor *dat = (PetscSF_Neighbor *)sf->data;
  PetscInt          i, j, nrootranks, ndrootranks, nleafranks, ndleafranks;
  const PetscInt   *rootoffset, *leafoffset;
  PetscMPIInt       m, n;

  PetscFunctionBegin;
  /* SFNeighbor inherits from Basic */
  PetscCall(PetscSFSetUp_Basic(sf));
  /* SFNeighbor specific */
  sf->persistent = PETSC_FALSE;
  PetscCall(PetscSFGetRootInfo_Basic(sf, &nrootranks, &ndrootranks, NULL, &rootoffset, NULL));
  PetscCall(PetscSFGetLeafInfo_Basic(sf, &nleafranks, &ndleafranks, NULL, &leafoffset, NULL, NULL));
  dat->rootdegree = m = (PetscMPIInt)(nrootranks - ndrootranks);
  dat->leafdegree = n = (PetscMPIInt)(nleafranks - ndleafranks);
  sf->nleafreqs       = 0;
  dat->nrootreqs      = 1; // collectives only need one MPI_Request. We just put it in rootreqs[]

  /* Only setup MPI displs/counts for non-distinguished ranks. Distinguished ranks use shared memory */
  PetscCall(PetscMalloc6(m, &dat->rootdispls, m, &dat->rootcounts, m, &dat->rootweights, n, &dat->leafdispls, n, &dat->leafcounts, n, &dat->leafweights));

#if defined(PETSC_HAVE_MPI_LARGE_COUNT) && defined(PETSC_USE_64BIT_INDICES)
  for (i = ndrootranks, j = 0; i < nrootranks; i++, j++) {
    dat->rootdispls[j]  = rootoffset[i] - rootoffset[ndrootranks];
    dat->rootcounts[j]  = rootoffset[i + 1] - rootoffset[i];
    dat->rootweights[j] = (PetscMPIInt)((PetscReal)dat->rootcounts[j] / (PetscReal)PETSC_MAX_INT * 2147483647); /* Scale to range of PetscMPIInt */
  }

  for (i = ndleafranks, j = 0; i < nleafranks; i++, j++) {
    dat->leafdispls[j]  = leafoffset[i] - leafoffset[ndleafranks];
    dat->leafcounts[j]  = leafoffset[i + 1] - leafoffset[i];
    dat->leafweights[j] = (PetscMPIInt)((PetscReal)dat->leafcounts[j] / (PetscReal)PETSC_MAX_INT * 2147483647);
  }
#else
  for (i = ndrootranks, j = 0; i < nrootranks; i++, j++) {
    PetscCall(PetscMPIIntCast(rootoffset[i] - rootoffset[ndrootranks], &m));
    dat->rootdispls[j] = m;
    PetscCall(PetscMPIIntCast(rootoffset[i + 1] - rootoffset[i], &n));
    dat->rootcounts[j]  = n;
    dat->rootweights[j] = n;
  }

  for (i = ndleafranks, j = 0; i < nleafranks; i++, j++) {
    PetscCall(PetscMPIIntCast(leafoffset[i] - leafoffset[ndleafranks], &m));
    dat->leafdispls[j] = m;
    PetscCall(PetscMPIIntCast(leafoffset[i + 1] - leafoffset[i], &n));
    dat->leafcounts[j]  = n;
    dat->leafweights[j] = n;
  }
#endif
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscSFReset_Neighbor(PetscSF sf)
{
  PetscInt          i;
  PetscSF_Neighbor *dat = (PetscSF_Neighbor *)sf->data;

  PetscFunctionBegin;
  PetscCheck(!dat->inuse, PetscObjectComm((PetscObject)sf), PETSC_ERR_ARG_WRONGSTATE, "Outstanding operation has not been completed");
  PetscCall(PetscFree6(dat->rootdispls, dat->rootcounts, dat->rootweights, dat->leafdispls, dat->leafcounts, dat->leafweights));
  for (i = 0; i < 2; i++) {
    if (dat->initialized[i]) {
      PetscCallMPI(MPI_Comm_free(&dat->comms[i]));
      dat->initialized[i] = PETSC_FALSE;
    }
  }
  PetscCall(PetscSFReset_Basic(sf)); /* Common part */
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscSFDestroy_Neighbor(PetscSF sf)
{
  PetscFunctionBegin;
  PetscCall(PetscSFReset_Neighbor(sf));
  PetscCall(PetscFree(sf->data));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PETSC_INTERN PetscErrorCode PetscSFCreate_Neighbor(PetscSF sf)
{
  PetscSF_Neighbor *dat;

  PetscFunctionBegin;
  sf->ops->CreateEmbeddedRootSF = PetscSFCreateEmbeddedRootSF_Basic;
  sf->ops->BcastBegin           = PetscSFBcastBegin_Basic;
  sf->ops->BcastEnd             = PetscSFBcastEnd_Basic;
  sf->ops->ReduceBegin          = PetscSFReduceBegin_Basic;
  sf->ops->ReduceEnd            = PetscSFReduceEnd_Basic;
  sf->ops->FetchAndOpBegin      = PetscSFFetchAndOpBegin_Basic;
  sf->ops->FetchAndOpEnd        = PetscSFFetchAndOpEnd_Basic;
  sf->ops->GetLeafRanks         = PetscSFGetLeafRanks_Basic;
  sf->ops->View                 = PetscSFView_Basic;

  sf->ops->SetUp               = PetscSFSetUp_Neighbor;
  sf->ops->Reset               = PetscSFReset_Neighbor;
  sf->ops->Destroy             = PetscSFDestroy_Neighbor;
  sf->ops->SetCommunicationOps = PetscSFSetCommunicationOps_Neighbor;

  PetscCall(PetscNew(&dat));
  sf->data = (void *)dat;
  PetscFunctionReturn(PETSC_SUCCESS);
}
