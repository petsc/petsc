#include <../src/vec/is/sf/impls/basic/gatherv/sfgatherv.h>
#include <../src/vec/is/sf/impls/basic/allgather/sfallgather.h>

/* Reuse the type. The difference is some fields (i.e., displs, recvcounts) are not used in Gather, which is not a big deal */
typedef PetscSF_Allgatherv PetscSF_Gather;

static PetscErrorCode PetscSFLinkStartCommunication_Gather(PetscSF sf, PetscSFLink link, PetscSFDirection direction)
{
  MPI_Comm     comm    = MPI_COMM_NULL;
  void        *rootbuf = NULL, *leafbuf = NULL;
  MPI_Request *req = NULL;
  PetscMPIInt  count;
  MPI_Datatype unit = link->unit;

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
    PetscCallMPI(MPIU_Igather(rootbuf == leafbuf ? MPI_IN_PLACE : rootbuf, count, unit, leafbuf, count, unit, 0 /*rank 0*/, comm, req));
  } else {
    PetscCallMPI(MPIU_Iscatter(leafbuf, count, unit, rootbuf == leafbuf ? MPI_IN_PLACE : rootbuf, count, unit, 0 /*rank 0*/, comm, req));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscSFSetCommunicationOps_Gather(PetscSF sf, PetscSFLink link)
{
  PetscFunctionBegin;
  link->StartCommunication = PetscSFLinkStartCommunication_Gather;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PETSC_INTERN PetscErrorCode PetscSFCreate_Gather(PetscSF sf)
{
  PetscSF_Gather *dat = (PetscSF_Gather *)sf->data;

  PetscFunctionBegin;
  sf->ops->BcastBegin  = PetscSFBcastBegin_Basic;
  sf->ops->BcastEnd    = PetscSFBcastEnd_Basic;
  sf->ops->ReduceBegin = PetscSFReduceBegin_Basic;
  sf->ops->ReduceEnd   = PetscSFReduceEnd_Basic;

  /* Inherit from Allgatherv */
  sf->ops->Reset         = PetscSFReset_Allgatherv;
  sf->ops->Destroy       = PetscSFDestroy_Allgatherv;
  sf->ops->GetGraph      = PetscSFGetGraph_Allgatherv;
  sf->ops->GetRootRanks  = PetscSFGetRootRanks_Allgatherv;
  sf->ops->GetLeafRanks  = PetscSFGetLeafRanks_Allgatherv;
  sf->ops->FetchAndOpEnd = PetscSFFetchAndOpEnd_Allgatherv;
  sf->ops->CreateLocalSF = PetscSFCreateLocalSF_Allgatherv;

  /* Inherit from Allgather */
  sf->ops->SetUp = PetscSFSetUp_Allgather;

  /* Inherit from Gatherv */
  sf->ops->FetchAndOpBegin = PetscSFFetchAndOpBegin_Gatherv;

  sf->ops->SetCommunicationOps = PetscSFSetCommunicationOps_Gather;

  sf->collective = PETSC_TRUE;

  PetscCall(PetscNew(&dat));
  sf->data = (void *)dat;
  PetscFunctionReturn(PETSC_SUCCESS);
}
