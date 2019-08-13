#include <../src/vec/is/sf/impls/basic/gatherv/sfgatherv.h>

typedef PetscSFPack_Allgatherv PetscSFPack_Gather;
#define PetscSFPackGet_Gather PetscSFPackGet_Allgatherv

/* Reuse the type. The difference is some fields (i.e., displs, recvcounts) are not used in Gather, which is not a big deal */
typedef PetscSF_Allgatherv PetscSF_Gather;

PETSC_INTERN PetscErrorCode PetscSFBcastAndOpBegin_Gather(PetscSF sf,MPI_Datatype unit,const void *rootdata,void *leafdata,MPI_Op op)
{
  PetscErrorCode       ierr;
  PetscSFPack_Gatherv  link;
  PetscMPIInt          rank,sendcount;
  MPI_Comm             comm;
  void                 *recvbuf;

  PetscFunctionBegin;
  ierr = PetscSFPackGet_Gatherv(sf,unit,rootdata,leafdata,&link);CHKERRQ(ierr);
  ierr = PetscObjectGetComm((PetscObject)sf,&comm);CHKERRQ(ierr);

  if (op == MPIU_REPLACE) {
    recvbuf = leafdata;
  } else {
    ierr = MPI_Comm_rank(comm,&rank);CHKERRQ(ierr);
    if (!link->leafbuf && !rank) {ierr = PetscMalloc(sf->nleaves*link->unitbytes,&link->leafbuf);CHKERRQ(ierr);}
    recvbuf = link->leafbuf;
  }

  ierr = PetscMPIIntCast(sf->nroots,&sendcount);CHKERRQ(ierr);
  ierr = MPIU_Igather(rootdata,sendcount,unit,recvbuf,sendcount,unit,0/*rank 0*/,comm,&link->request);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscSFReduceBegin_Gather(PetscSF sf,MPI_Datatype unit,const void *leafdata,void *rootdata,MPI_Op op)
{
  PetscErrorCode       ierr;
  PetscSFPack_Gatherv  link;
  PetscMPIInt          recvcount;
  MPI_Comm             comm;
  void                 *recvbuf;

  PetscFunctionBegin;
  ierr = PetscSFPackGet_Gatherv(sf,unit,rootdata,leafdata,&link);CHKERRQ(ierr);
  ierr = PetscObjectGetComm((PetscObject)sf,&comm);CHKERRQ(ierr);

  if (op == MPIU_REPLACE) {
    recvbuf = rootdata;
  } else {
    if (!link->rootbuf) {ierr = PetscMalloc(sf->nroots*link->unitbytes,&link->rootbuf);CHKERRQ(ierr);}
    recvbuf = link->rootbuf;
  }

  ierr = PetscMPIIntCast(sf->nroots,&recvcount);CHKERRQ(ierr);
  ierr = MPIU_Iscatter(leafdata,recvcount,unit,recvbuf,recvcount,unit,0/*rank 0*/,comm,&link->request);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PETSC_INTERN PetscErrorCode PetscSFCreate_Gather(PetscSF sf)
{
  PetscErrorCode  ierr;
  PetscSF_Gather  *dat = (PetscSF_Gather*)sf->data;

  PetscFunctionBegin;
  /* Inherit from Allgatherv */
  sf->ops->Reset           = PetscSFReset_Allgatherv;
  sf->ops->Destroy         = PetscSFDestroy_Allgatherv;
  sf->ops->GetGraph        = PetscSFGetGraph_Allgatherv;
  sf->ops->GetRootRanks    = PetscSFGetRootRanks_Allgatherv;
  sf->ops->GetLeafRanks    = PetscSFGetLeafRanks_Allgatherv;
  sf->ops->BcastAndOpEnd   = PetscSFBcastAndOpEnd_Allgatherv;
  sf->ops->ReduceEnd       = PetscSFReduceEnd_Allgatherv;
  sf->ops->FetchAndOpEnd   = PetscSFFetchAndOpEnd_Allgatherv;
  sf->ops->CreateLocalSF   = PetscSFCreateLocalSF_Allgatherv;

  /* Inherit from Gatherv */
  sf->ops->FetchAndOpBegin = PetscSFFetchAndOpBegin_Gatherv;

  /* Gather stuff */
  sf->ops->BcastAndOpBegin = PetscSFBcastAndOpBegin_Gather;
  sf->ops->ReduceBegin     = PetscSFReduceBegin_Gather;

  ierr     = PetscNewLog(sf,&dat);CHKERRQ(ierr);
  sf->data = (void*)dat;
  PetscFunctionReturn(0);
}

