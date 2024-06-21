#include <../src/vec/is/sf/impls/basic/gatherv/sfgatherv.h>

/* Reuse the type. The difference is some fields (displs, recvcounts) are only significant
   on rank 0 in Gatherv. On other ranks they are harmless NULL.
 */
typedef PetscSF_Allgatherv PetscSF_Gatherv;

static PetscErrorCode PetscSFLinkStartCommunication_Gatherv(PetscSF sf, PetscSFLink link, PetscSFDirection direction)
{
  MPI_Comm         comm = MPI_COMM_NULL;
  PetscMPIInt      count;
  PetscSF_Gatherv *dat     = (PetscSF_Gatherv *)sf->data;
  void            *rootbuf = NULL, *leafbuf = NULL; /* buffer seen by MPI */
  MPI_Request     *req  = NULL;
  MPI_Datatype     unit = link->unit;

  PetscFunctionBegin;
  if (direction == PETSCSF_ROOT2LEAF) {
    PetscCall(PetscSFLinkCopyRootBufferInCaseNotUseGpuAwareMPI(sf, link, PETSC_TRUE /* device2host before sending */));
  } else {
    PetscCall(PetscSFLinkCopyLeafBufferInCaseNotUseGpuAwareMPI(sf, link, PETSC_TRUE /* device2host */));
  }
  PetscCall(PetscObjectGetComm((PetscObject)sf, &comm));
  PetscCall(PetscMPIIntCast(sf->nroots, &count));
  PetscCall(PetscSFLinkGetMPIBuffersAndRequests(sf, link, direction, &rootbuf, &leafbuf, &req, NULL));
  PetscCall(PetscSFLinkSyncStreamBeforeCallMPI(sf, link));

  if (direction == PETSCSF_ROOT2LEAF) {
    PetscCallMPI(MPIU_Igatherv(rootbuf, count, unit, leafbuf, dat->recvcounts, dat->displs, unit, 0 /*rank 0*/, comm, req));
  } else {
    PetscCallMPI(MPIU_Iscatterv(leafbuf, dat->recvcounts, dat->displs, unit, rootbuf, count, unit, 0, comm, req));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscSFSetCommunicationOps_Gatherv(PetscSF sf, PetscSFLink link)
{
  PetscFunctionBegin;
  link->StartCommunication = PetscSFLinkStartCommunication_Gatherv;
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
  sf->ops->BcastBegin  = PetscSFBcastBegin_Basic;
  sf->ops->BcastEnd    = PetscSFBcastEnd_Basic;
  sf->ops->ReduceBegin = PetscSFReduceBegin_Basic;
  sf->ops->ReduceEnd   = PetscSFReduceEnd_Basic;

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
  sf->ops->FetchAndOpBegin = PetscSFFetchAndOpBegin_Gatherv;

  sf->ops->SetCommunicationOps = PetscSFSetCommunicationOps_Gatherv;

  sf->collective = PETSC_TRUE;

  PetscCall(PetscNew(&dat));
  sf->data = (void *)dat;
  PetscFunctionReturn(PETSC_SUCCESS);
}
