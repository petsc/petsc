#include <../src/vec/is/sf/impls/basic/allgatherv/sfallgatherv.h>

#define PetscSFPackGet_Allgather PetscSFPackGet_Allgatherv

/* Reuse the type. The difference is some fields (i.e., displs, recvcounts) are not used in Allgather on rank != 0, which is not a big deal */
typedef PetscSF_Allgatherv PetscSF_Allgather;

PETSC_INTERN PetscErrorCode PetscSFBcastAndOpBegin_Gather(PetscSF,MPI_Datatype,PetscMemType,const void*,PetscMemType,void*,MPI_Op);

static PetscErrorCode PetscSFBcastAndOpBegin_Allgather(PetscSF sf,MPI_Datatype unit,PetscMemType rootmtype,const void *rootdata,PetscMemType leafmtype,void *leafdata,MPI_Op op)
{
  PetscErrorCode        ierr;
  PetscSFPack           link;
  PetscMPIInt           sendcount;
  MPI_Comm              comm;

  PetscFunctionBegin;
  ierr = PetscSFPackGet_Allgather(sf,unit,rootmtype,rootdata,leafmtype,leafdata,&link);CHKERRQ(ierr);
  ierr = PetscObjectGetComm((PetscObject)sf,&comm);CHKERRQ(ierr);
  ierr = PetscMPIIntCast(sf->nroots,&sendcount);CHKERRQ(ierr);

  if (op == MPIU_REPLACE) {
    ierr = MPIU_Iallgather(rootdata,sendcount,unit,leafdata,sendcount,unit,comm,link->rootreqs[PETSCSF_ROOT2LEAF_BCAST][rootmtype]);CHKERRQ(ierr);
  } else {
    /* Allgather to the leaf buffer and then add leaf buffer to rootdata */
    if (!link->leafbuf[leafmtype]) {ierr = PetscMallocWithMemType(leafmtype,sf->nleaves*link->unitbytes,(void**)&link->leafbuf[leafmtype]);CHKERRQ(ierr);}
    ierr = MPIU_Iallgather(rootdata,sendcount,unit,link->leafbuf[leafmtype],sendcount,unit,comm,link->rootreqs[PETSCSF_ROOT2LEAF_BCAST][rootmtype]);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscSFBcastToZero_Allgather(PetscSF sf,MPI_Datatype unit,PetscMemType rootmtype,const void *rootdata,PetscMemType leafmtype,void *leafdata)
{
  PetscErrorCode        ierr;
  PetscSFPack           link;

  PetscFunctionBegin;
  ierr = PetscSFBcastAndOpBegin_Gather(sf,unit,rootmtype,rootdata,leafmtype,leafdata,MPIU_REPLACE);CHKERRQ(ierr);
  /* A simplified PetscSFBcastAndOpEnd_Allgatherv */
  ierr = PetscSFPackGetInUse(sf,unit,rootdata,leafdata,PETSC_OWN_POINTER,&link);CHKERRQ(ierr);
  ierr = MPI_Wait(link->rootreqs[PETSCSF_ROOT2LEAF_BCAST][rootmtype],MPI_STATUS_IGNORE);CHKERRQ(ierr);
  ierr = PetscSFPackReclaim(sf,&link);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscSFReduceBegin_Allgather(PetscSF sf,MPI_Datatype unit,PetscMemType leafmtype,const void *leafdata,PetscMemType rootmtype,void *rootdata,MPI_Op op)
{
  PetscErrorCode        ierr;
  PetscSFPack           link;
  PetscMPIInt           rank,count,sendcount;
  PetscInt              rstart;
  MPI_Comm              comm;

  PetscFunctionBegin;
  ierr = PetscSFPackGet_Allgather(sf,unit,rootmtype,rootdata,leafmtype,leafdata,&link);CHKERRQ(ierr);
  ierr = PetscObjectGetComm((PetscObject)sf,&comm);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(comm,&rank);CHKERRQ(ierr);

  if (op == MPIU_REPLACE) {
    /* REPLACE is only meaningful when all processes have the same leafdata to reduce. Therefore copy from local leafdata is fine */
    ierr = PetscLayoutGetRange(sf->map,&rstart,NULL);CHKERRQ(ierr);
    ierr = PetscMemcpyWithMemType(rootmtype,leafmtype,rootdata,(const char*)leafdata+(size_t)rstart*link->unitbytes,(size_t)sf->nroots*link->unitbytes);CHKERRQ(ierr);
  } else {
    /* Reduce all leafdata on rank 0, then scatter the result to root buffer, then reduce root buffer to leafdata */
    if (!rank && !link->leafbuf[leafmtype]) {ierr = PetscMallocWithMemType(leafmtype,sf->nleaves*link->unitbytes,(void**)&link->leafbuf[leafmtype]);CHKERRQ(ierr);}
    ierr = PetscMPIIntCast(sf->nleaves*link->bs,&count);CHKERRQ(ierr);
    ierr = PetscMPIIntCast(sf->nroots,&sendcount);CHKERRQ(ierr);
    ierr = MPI_Reduce(leafdata,link->leafbuf[leafmtype],count,link->basicunit,op,0/*rank 0*/,comm);CHKERRQ(ierr); /* Must do reduce with MPI builltin datatype basicunit */
    if (!link->rootbuf[rootmtype]) {ierr = PetscMallocWithMemType(rootmtype,sf->nroots*link->unitbytes,(void**)&link->rootbuf[rootmtype]);CHKERRQ(ierr);} /* Allocate root buffer */
    ierr = MPIU_Iscatter(link->leafbuf[leafmtype],sendcount,unit,link->rootbuf[rootmtype],sendcount,unit,0/*rank 0*/,comm,link->rootreqs[PETSCSF_LEAF2ROOT_REDUCE][rootmtype]);CHKERRQ(ierr);
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
