
#include <../src/vec/is/sf/impls/basic/gatherv/sfgatherv.h>

/* Reuse the type. The difference is some fields (displs, recvcounts) are only significant
   on rank 0 in Gatherv. On other ranks they are harmless NULL.
 */
typedef PetscSF_Allgatherv PetscSF_Gatherv;

PETSC_INTERN PetscErrorCode PetscSFBcastBegin_Gatherv(PetscSF sf, MPI_Datatype unit, PetscMemType rootmtype, const void *rootdata, PetscMemType leafmtype, void *leafdata, MPI_Op op)
{
  PetscSFLink      link;
  PetscMPIInt      sendcount;
  MPI_Comm         comm;
  PetscSF_Gatherv *dat     = (PetscSF_Gatherv *)sf->data;
  void            *rootbuf = NULL, *leafbuf = NULL; /* buffer seen by MPI */
  MPI_Request     *req;

  PetscFunctionBegin;
  PetscCall(PetscSFLinkCreate(sf, unit, rootmtype, rootdata, leafmtype, leafdata, op, PETSCSF_BCAST, &link));
  PetscCall(PetscSFLinkPackRootData(sf, link, PETSCSF_REMOTE, rootdata));
  PetscCall(PetscSFLinkCopyRootBufferInCaseNotUseGpuAwareMPI(sf, link, PETSC_TRUE /* device2host before sending */));
  PetscCall(PetscObjectGetComm((PetscObject)sf, &comm));
  PetscCall(PetscMPIIntCast(sf->nroots, &sendcount));
  PetscCall(PetscSFLinkGetMPIBuffersAndRequests(sf, link, PETSCSF_ROOT2LEAF, &rootbuf, &leafbuf, &req, NULL));
  PetscCall(PetscSFLinkSyncStreamBeforeCallMPI(sf, link, PETSCSF_ROOT2LEAF));
  PetscCallMPI(MPIU_Igatherv(rootbuf, sendcount, unit, leafbuf, dat->recvcounts, dat->displs, unit, 0 /*rank 0*/, comm, req));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscSFReduceBegin_Gatherv(PetscSF sf, MPI_Datatype unit, PetscMemType leafmtype, const void *leafdata, PetscMemType rootmtype, void *rootdata, MPI_Op op)
{
  PetscSFLink      link;
  PetscMPIInt      recvcount;
  MPI_Comm         comm;
  PetscSF_Gatherv *dat     = (PetscSF_Gatherv *)sf->data;
  void            *rootbuf = NULL, *leafbuf = NULL; /* buffer seen by MPI */
  MPI_Request     *req;

  PetscFunctionBegin;
  PetscCall(PetscSFLinkCreate(sf, unit, rootmtype, rootdata, leafmtype, leafdata, op, PETSCSF_REDUCE, &link));
  PetscCall(PetscSFLinkPackLeafData(sf, link, PETSCSF_REMOTE, leafdata));
  PetscCall(PetscSFLinkCopyLeafBufferInCaseNotUseGpuAwareMPI(sf, link, PETSC_TRUE /* device2host before sending */));
  PetscCall(PetscObjectGetComm((PetscObject)sf, &comm));
  PetscCall(PetscMPIIntCast(sf->nroots, &recvcount));
  PetscCall(PetscSFLinkGetMPIBuffersAndRequests(sf, link, PETSCSF_LEAF2ROOT, &rootbuf, &leafbuf, &req, NULL));
  PetscCall(PetscSFLinkSyncStreamBeforeCallMPI(sf, link, PETSCSF_LEAF2ROOT));
  PetscCallMPI(MPIU_Iscatterv(leafbuf, dat->recvcounts, dat->displs, unit, rootbuf, recvcount, unit, 0, comm, req));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PETSC_INTERN PetscErrorCode PetscSFFetchAndOpBegin_Gatherv(PetscSF sf, MPI_Datatype unit, PetscMemType rootmtype, void *rootdata, PetscMemType leafmtype, const void *leafdata, void *leafupdate, MPI_Op op)
{
  PetscFunctionBegin;
  /* In Gatherv, each root only has one leaf. So we just need to bcast rootdata to leafupdate and then reduce leafdata to rootdata */
  PetscCall(PetscSFBcastBegin(sf, unit, rootdata, leafupdate, MPI_REPLACE));
  PetscCall(PetscSFBcastEnd(sf, unit, rootdata, leafupdate, MPI_REPLACE));
  PetscCall(PetscSFReduceBegin(sf, unit, leafdata, rootdata, op));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PETSC_INTERN PetscErrorCode PetscSFCreate_Gatherv(PetscSF sf)
{
  PetscSF_Gatherv *dat = (PetscSF_Gatherv *)sf->data;

  PetscFunctionBegin;
  sf->ops->BcastEnd  = PetscSFBcastEnd_Basic;
  sf->ops->ReduceEnd = PetscSFReduceEnd_Basic;

  /* Inherit from Allgatherv */
  sf->ops->SetUp         = PetscSFSetUp_Allgatherv;
  sf->ops->Reset         = PetscSFReset_Allgatherv;
  sf->ops->Destroy       = PetscSFDestroy_Allgatherv;
  sf->ops->GetGraph      = PetscSFGetGraph_Allgatherv;
  sf->ops->GetLeafRanks  = PetscSFGetLeafRanks_Allgatherv;
  sf->ops->GetRootRanks  = PetscSFGetRootRanks_Allgatherv;
  sf->ops->FetchAndOpEnd = PetscSFFetchAndOpEnd_Allgatherv;
  sf->ops->CreateLocalSF = PetscSFCreateLocalSF_Allgatherv;

  /* Gatherv stuff */
  sf->ops->BcastBegin      = PetscSFBcastBegin_Gatherv;
  sf->ops->ReduceBegin     = PetscSFReduceBegin_Gatherv;
  sf->ops->FetchAndOpBegin = PetscSFFetchAndOpBegin_Gatherv;

  PetscCall(PetscNew(&dat));
  sf->data = (void *)dat;
  PetscFunctionReturn(PETSC_SUCCESS);
}
