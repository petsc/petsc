#include <../src/vec/is/sf/impls/basic/gatherv/sfgatherv.h>
#include <../src/vec/is/sf/impls/basic/allgather/sfallgather.h>

/* Reuse the type. The difference is some fields (i.e., displs, recvcounts) are not used in Gather, which is not a big deal */
typedef PetscSF_Allgatherv PetscSF_Gather;

PETSC_INTERN PetscErrorCode PetscSFBcastBegin_Gather(PetscSF sf,MPI_Datatype unit,PetscMemType rootmtype,const void *rootdata,PetscMemType leafmtype,void *leafdata,MPI_Op op)
{
  PetscErrorCode       ierr;
  PetscSFLink          link;
  PetscMPIInt          sendcount;
  MPI_Comm             comm;
  void                 *rootbuf = NULL,*leafbuf = NULL;
  MPI_Request          *req;

  PetscFunctionBegin;
  ierr = PetscSFLinkCreate(sf,unit,rootmtype,rootdata,leafmtype,leafdata,op,PETSCSF_BCAST,&link);CHKERRQ(ierr);
  ierr = PetscSFLinkPackRootData(sf,link,PETSCSF_REMOTE,rootdata);CHKERRQ(ierr);
  ierr = PetscSFLinkCopyRootBufferInCaseNotUseGpuAwareMPI(sf,link,PETSC_TRUE/* device2host before sending */);CHKERRQ(ierr);
  ierr = PetscObjectGetComm((PetscObject)sf,&comm);CHKERRQ(ierr);
  ierr = PetscMPIIntCast(sf->nroots,&sendcount);CHKERRQ(ierr);
  ierr = PetscSFLinkGetMPIBuffersAndRequests(sf,link,PETSCSF_ROOT2LEAF,&rootbuf,&leafbuf,&req,NULL);CHKERRQ(ierr);
  ierr = PetscSFLinkSyncStreamBeforeCallMPI(sf,link,PETSCSF_ROOT2LEAF);CHKERRQ(ierr);
  ierr = MPIU_Igather(rootbuf == leafbuf ? MPI_IN_PLACE : rootbuf,sendcount,unit,leafbuf,sendcount,unit,0/*rank 0*/,comm,req);CHKERRMPI(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscSFReduceBegin_Gather(PetscSF sf,MPI_Datatype unit,PetscMemType leafmtype,const void *leafdata,PetscMemType rootmtype,void *rootdata,MPI_Op op)
{
  PetscErrorCode       ierr;
  PetscSFLink          link;
  PetscMPIInt          recvcount;
  MPI_Comm             comm;
  void                 *rootbuf = NULL,*leafbuf = NULL;
  MPI_Request          *req;

  PetscFunctionBegin;
  ierr = PetscSFLinkCreate(sf,unit,rootmtype,rootdata,leafmtype,leafdata,op,PETSCSF_REDUCE,&link);CHKERRQ(ierr);
  ierr = PetscSFLinkPackLeafData(sf,link,PETSCSF_REMOTE,leafdata);CHKERRQ(ierr);
  ierr = PetscSFLinkCopyLeafBufferInCaseNotUseGpuAwareMPI(sf,link,PETSC_TRUE/* device2host before sending */);CHKERRQ(ierr);
  ierr = PetscObjectGetComm((PetscObject)sf,&comm);CHKERRQ(ierr);
  ierr = PetscMPIIntCast(sf->nroots,&recvcount);CHKERRQ(ierr);
  ierr = PetscSFLinkGetMPIBuffersAndRequests(sf,link,PETSCSF_LEAF2ROOT,&rootbuf,&leafbuf,&req,NULL);CHKERRQ(ierr);
  ierr = PetscSFLinkSyncStreamBeforeCallMPI(sf,link,PETSCSF_LEAF2ROOT);CHKERRQ(ierr);
  ierr = MPIU_Iscatter(leafbuf,recvcount,unit,rootbuf == leafbuf ? MPI_IN_PLACE : rootbuf,recvcount,unit,0/*rank 0*/,comm,req);CHKERRMPI(ierr);
  PetscFunctionReturn(0);
}

PETSC_INTERN PetscErrorCode PetscSFCreate_Gather(PetscSF sf)
{
  PetscErrorCode  ierr;
  PetscSF_Gather  *dat = (PetscSF_Gather*)sf->data;

  PetscFunctionBegin;
  sf->ops->BcastEnd        = PetscSFBcastEnd_Basic;
  sf->ops->ReduceEnd       = PetscSFReduceEnd_Basic;

  /* Inherit from Allgatherv */
  sf->ops->Reset           = PetscSFReset_Allgatherv;
  sf->ops->Destroy         = PetscSFDestroy_Allgatherv;
  sf->ops->GetGraph        = PetscSFGetGraph_Allgatherv;
  sf->ops->GetRootRanks    = PetscSFGetRootRanks_Allgatherv;
  sf->ops->GetLeafRanks    = PetscSFGetLeafRanks_Allgatherv;
  sf->ops->FetchAndOpEnd   = PetscSFFetchAndOpEnd_Allgatherv;
  sf->ops->CreateLocalSF   = PetscSFCreateLocalSF_Allgatherv;

  /* Inherit from Allgather */
  sf->ops->SetUp           = PetscSFSetUp_Allgather;

  /* Inherit from Gatherv */
  sf->ops->FetchAndOpBegin = PetscSFFetchAndOpBegin_Gatherv;

  /* Gather stuff */
  sf->ops->BcastBegin      = PetscSFBcastBegin_Gather;
  sf->ops->ReduceBegin     = PetscSFReduceBegin_Gather;

  ierr     = PetscNewLog(sf,&dat);CHKERRQ(ierr);
  sf->data = (void*)dat;
  PetscFunctionReturn(0);
}

