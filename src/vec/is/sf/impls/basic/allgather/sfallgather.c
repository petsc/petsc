
#include <../src/vec/is/sf/impls/basic/allgatherv/sfallgatherv.h>

typedef PetscSFPack_Allgatherv PetscSFPack_Allgather;
#define PetscSFPackGet_Allgather PetscSFPackGet_Allgatherv

/* Reuse the type. The difference is some fields (i.e., displs, recvcounts) are not used in Allgather on rank != 0, which is not a big deal */
typedef PetscSF_Allgatherv PetscSF_Allgather;

PETSC_INTERN PetscErrorCode PetscSFBcastAndOpBegin_Gather(PetscSF,MPI_Datatype,const void*,void*,MPI_Op);

static PetscErrorCode PetscSFBcastAndOpBegin_Allgather(PetscSF sf,MPI_Datatype unit,const void *rootdata,void *leafdata,MPI_Op op)
{
  PetscErrorCode        ierr;
  PetscSFPack_Allgather link;
  PetscMPIInt           sendcount;
  MPI_Comm              comm;

  PetscFunctionBegin;
  ierr = PetscSFPackGet_Allgather(sf,unit,rootdata,leafdata,&link);CHKERRQ(ierr);
  ierr = PetscObjectGetComm((PetscObject)sf,&comm);CHKERRQ(ierr);
  ierr = PetscMPIIntCast(sf->nroots,&sendcount);CHKERRQ(ierr);

  if (op == MPIU_REPLACE) {
    ierr = MPIU_Iallgather(rootdata,sendcount,unit,leafdata,sendcount,unit,comm,&link->request);CHKERRQ(ierr);
  } else {
    /* Allgather to the leaf buffer and then add leaf buffer to rootdata */
    if (!link->leafbuf) {ierr = PetscMalloc(sf->nleaves*link->unitbytes,&link->leafbuf);CHKERRQ(ierr);}
    ierr = MPIU_Iallgather(rootdata,sendcount,unit,link->leafbuf,sendcount,unit,comm,&link->request);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscSFBcastToZero_Allgather(PetscSF sf,MPI_Datatype unit,const void *rootdata,void *leafdata)
{
  PetscErrorCode         ierr;
  PetscSFPack_Allgather link;

  PetscFunctionBegin;
  ierr = PetscSFBcastAndOpBegin_Gather(sf,unit,rootdata,leafdata,MPIU_REPLACE);CHKERRQ(ierr);
  /* A simplified PetscSFBcastAndOpEnd_Allgatherv */
  ierr = PetscSFPackGetInUse(sf,unit,rootdata,leafdata,PETSC_OWN_POINTER,(PetscSFPack*)&link);CHKERRQ(ierr);
  ierr = MPI_Wait(&link->request,MPI_STATUS_IGNORE);CHKERRQ(ierr);
  ierr = PetscSFPackReclaim(sf,(PetscSFPack*)&link);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscSFReduceBegin_Allgather(PetscSF sf,MPI_Datatype unit,const void *leafdata,void *rootdata,MPI_Op op)
{
  PetscErrorCode        ierr;
  PetscSFPack_Allgather link;
  PetscMPIInt           rank,count,sendcount;
  PetscInt              rstart;
  MPI_Comm              comm;

  PetscFunctionBegin;
  ierr = PetscSFPackGet_Allgather(sf,unit,rootdata,leafdata,&link);CHKERRQ(ierr);
  ierr = PetscObjectGetComm((PetscObject)sf,&comm);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(comm,&rank);CHKERRQ(ierr);

  if (op == MPIU_REPLACE) {
    /* REPLACE is only meaningful when all processes have the same leafdata to reduce. Therefore copy from local leafdata is fine */
    ierr = PetscLayoutGetRange(sf->map,&rstart,NULL);CHKERRQ(ierr);
    ierr = PetscMemcpy(rootdata,(const char*)leafdata+(size_t)rstart*link->unitbytes,(size_t)sf->nroots*link->unitbytes);CHKERRQ(ierr);
  } else {
    /* Reduce all leafdata on rank 0, then scatter the result to root buffer, then reduce root buffer to leafdata */
    if (!rank && !link->leafbuf) {ierr = PetscMalloc(sf->nleaves*link->unitbytes,&link->leafbuf);CHKERRQ(ierr);}
    ierr = PetscMPIIntCast(sf->nleaves*link->bs,&count);CHKERRQ(ierr);
    ierr = PetscMPIIntCast(sf->nroots,&sendcount);CHKERRQ(ierr);
    ierr = MPI_Reduce(leafdata,link->leafbuf,count,link->basicunit,op,0/*rank 0*/,comm);CHKERRQ(ierr); /* Must do reduce with MPI builltin datatype basicunit */
    if (!link->rootbuf) {ierr = PetscMalloc(sf->nroots*link->unitbytes,&link->rootbuf);CHKERRQ(ierr);} /* Allocate root buffer */
    ierr = MPIU_Iscatter(link->leafbuf,sendcount,unit,link->rootbuf,sendcount,unit,0/*rank 0*/,comm,&link->request);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

PETSC_INTERN PetscErrorCode PetscSFCreate_Allgather(PetscSF sf)
{
  PetscErrorCode    ierr;
  PetscSF_Allgather *dat = (PetscSF_Allgather*)sf->data;

  PetscFunctionBegin;

  /* Inherit from Allgatherv */
  sf->ops->Reset           = PetscSFReset_Allgatherv;
  sf->ops->Destroy         = PetscSFDestroy_Allgatherv;
  sf->ops->BcastAndOpEnd   = PetscSFBcastAndOpEnd_Allgatherv;
  sf->ops->ReduceEnd       = PetscSFReduceEnd_Allgatherv;
  sf->ops->FetchAndOpBegin = PetscSFFetchAndOpBegin_Allgatherv;
  sf->ops->FetchAndOpEnd   = PetscSFFetchAndOpEnd_Allgatherv;
  sf->ops->GetRootRanks    = PetscSFGetRootRanks_Allgatherv;
  sf->ops->CreateLocalSF   = PetscSFCreateLocalSF_Allgatherv;
  sf->ops->GetGraph        = PetscSFGetGraph_Allgatherv;
  sf->ops->GetLeafRanks    = PetscSFGetLeafRanks_Allgatherv;

  /* Allgather stuff */
  sf->ops->BcastAndOpBegin = PetscSFBcastAndOpBegin_Allgather;
  sf->ops->ReduceBegin     = PetscSFReduceBegin_Allgather;
  sf->ops->BcastToZero     = PetscSFBcastToZero_Allgather;

  ierr = PetscNewLog(sf,&dat);CHKERRQ(ierr);
  sf->data = (void*)dat;
  PetscFunctionReturn(0);
}
