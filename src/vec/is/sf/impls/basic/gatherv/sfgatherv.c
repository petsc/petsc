
#include <../src/vec/is/sf/impls/basic/gatherv/sfgatherv.h>

#define PetscSFPackGet_Gatherv PetscSFPackGet_Allgatherv

/* Reuse the type. The difference is some fields (displs, recvcounts) are only significant
   on rank 0 in Gatherv. On other ranks they are harmless NULL.
 */
typedef PetscSF_Allgatherv PetscSF_Gatherv;

PETSC_INTERN PetscErrorCode PetscSFBcastAndOpBegin_Gatherv(PetscSF sf,MPI_Datatype unit,PetscMemType rootmtype,const void *rootdata,PetscMemType leafmtype,void *leafdata,MPI_Op op)
{
  PetscErrorCode       ierr;
  PetscSFPack          link;
  PetscMPIInt          rank,sendcount;
  MPI_Comm             comm;
  char                 *recvbuf;
  PetscSF_Gatherv      *dat = (PetscSF_Gatherv*)sf->data;

  PetscFunctionBegin;
  ierr = PetscSFPackGet_Gatherv(sf,unit,rootmtype,rootdata,leafmtype,leafdata,&link);CHKERRQ(ierr);
  ierr = PetscObjectGetComm((PetscObject)sf,&comm);CHKERRQ(ierr);
  ierr = PetscMPIIntCast(sf->nroots,&sendcount);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(comm,&rank);CHKERRQ(ierr);

  if (op == MPIU_REPLACE) {
    recvbuf = (char*)leafdata;
  } else {
    if (!link->leafbuf[leafmtype] && !rank) {ierr = PetscMallocWithMemType(leafmtype,sf->nleaves*link->unitbytes,(void**)&link->leafbuf[leafmtype]);CHKERRQ(ierr);} /* Alloate leafbuf on rank 0 */
    recvbuf = link->leafbuf[leafmtype];
  }

  ierr = MPIU_Igatherv(rootdata,sendcount,unit,recvbuf,dat->recvcounts,dat->displs,unit,0/*rank 0*/,comm,link->rootreqs[PETSCSF_ROOT2LEAF_BCAST][rootmtype]);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscSFReduceBegin_Gatherv(PetscSF sf,MPI_Datatype unit,PetscMemType leafmtype,const void *leafdata,PetscMemType rootmtype,void *rootdata,MPI_Op op)
{
  PetscErrorCode       ierr;
  PetscSFPack          link;
  PetscMPIInt          recvcount;
  MPI_Comm             comm;
  char                 *recvbuf;
  PetscSF_Gatherv      *dat = (PetscSF_Gatherv*)sf->data;

  PetscFunctionBegin;
  ierr = PetscSFPackGet_Gatherv(sf,unit,rootmtype,rootdata,leafmtype,leafdata,&link);CHKERRQ(ierr);
  ierr = PetscObjectGetComm((PetscObject)sf,&comm);CHKERRQ(ierr);

  if (op == MPIU_REPLACE) {
    recvbuf = (char*)rootdata;
  } else {
    if (!link->rootbuf[rootmtype]) {ierr = PetscMallocWithMemType(rootmtype,sf->nroots*link->unitbytes,(void**)&link->rootbuf[rootmtype]);CHKERRQ(ierr);}
    recvbuf = link->rootbuf[rootmtype];
  }

  ierr = PetscMPIIntCast(sf->nroots,&recvcount);CHKERRQ(ierr);
  ierr = MPIU_Iscatterv(leafdata,dat->recvcounts,dat->displs,unit,recvbuf,recvcount,unit,0,comm,link->rootreqs[PETSCSF_LEAF2ROOT_REDUCE][rootmtype]);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PETSC_INTERN PetscErrorCode PetscSFFetchAndOpBegin_Gatherv(PetscSF sf,MPI_Datatype unit,PetscMemType rootmtype,void *rootdata,PetscMemType leafmtype,const void *leafdata,void *leafupdate,MPI_Op op)
{
  PetscErrorCode      ierr;

  PetscFunctionBegin;
  /* In Gatherv, each root only has one leaf. So we just need to bcast rootdata to leafupdate and then reduce leafdata to rootdata */
  ierr = PetscSFBcastAndOpBegin(sf,unit,rootdata,leafupdate,MPIU_REPLACE);CHKERRQ(ierr);
  ierr = PetscSFBcastAndOpEnd(sf,unit,rootdata,leafupdate,MPIU_REPLACE);CHKERRQ(ierr);
  ierr = PetscSFReduceBegin(sf,unit,leafdata,rootdata,op);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PETSC_INTERN PetscErrorCode PetscSFCreate_Gatherv(PetscSF sf)
{
  PetscErrorCode  ierr;
  PetscSF_Gatherv *dat = (PetscSF_Gatherv*)sf->data;

  PetscFunctionBegin;
  /* Inherit from Allgatherv */
  sf->ops->SetUp           = PetscSFSetUp_Allgatherv;
  sf->ops->Reset           = PetscSFReset_Allgatherv;
  sf->ops->Destroy         = PetscSFDestroy_Allgatherv;
  sf->ops->GetGraph        = PetscSFGetGraph_Allgatherv;
  sf->ops->GetLeafRanks    = PetscSFGetLeafRanks_Allgatherv;
  sf->ops->GetRootRanks    = PetscSFGetRootRanks_Allgatherv;
  sf->ops->BcastAndOpEnd   = PetscSFBcastAndOpEnd_Allgatherv;
  sf->ops->ReduceEnd       = PetscSFReduceEnd_Allgatherv;
  sf->ops->FetchAndOpEnd   = PetscSFFetchAndOpEnd_Allgatherv;
  sf->ops->CreateLocalSF   = PetscSFCreateLocalSF_Allgatherv;

  /* Gatherv stuff */
  sf->ops->BcastAndOpBegin = PetscSFBcastAndOpBegin_Gatherv;
  sf->ops->ReduceBegin     = PetscSFReduceBegin_Gatherv;
  sf->ops->FetchAndOpBegin = PetscSFFetchAndOpBegin_Gatherv;

  ierr = PetscNewLog(sf,&dat);CHKERRQ(ierr);
  sf->data = (void*)dat;
  PetscFunctionReturn(0);
}
