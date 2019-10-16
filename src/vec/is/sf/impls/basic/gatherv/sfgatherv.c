
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
  PetscMPIInt          sendcount;
  MPI_Comm             comm;
  PetscSF_Gatherv      *dat = (PetscSF_Gatherv*)sf->data;
  const void           *rootbuf_mpi; /* buffer used by MPI */
  void                 *leafbuf_mpi;
  PetscMemType         rootmtype_mpi,leafmtype_mpi;

  PetscFunctionBegin;
  ierr = PetscSFPackGet_Gatherv(sf,unit,rootmtype,rootdata,leafmtype,leafdata,&link);CHKERRQ(ierr);
  ierr = PetscObjectGetComm((PetscObject)sf,&comm);CHKERRQ(ierr);
  ierr = PetscMPIIntCast(sf->nroots,&sendcount);CHKERRQ(ierr);
  ierr = PetscSFBcastPrepareMPIBuffers_Allgatherv(sf,link,op,&rootmtype_mpi,&rootbuf_mpi,&leafmtype_mpi,&leafbuf_mpi);CHKERRQ(ierr);
  ierr = MPIU_Igatherv(rootbuf_mpi,sendcount,unit,leafbuf_mpi,dat->recvcounts,dat->displs,unit,0/*rank 0*/,comm,link->rootreqs[PETSCSF_ROOT2LEAF_BCAST][rootmtype_mpi]);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*
  Prepare the rootbuf, leafbuf etc used by MPI in PetscSFReduceBegin.

Input Arguments:
+ sf    - the start forest
. link  - the link PetscSFReduceBegin is currently using
- op    - the reduction op

Output Arguments:
+rootmtype_mpi  - memtype of rootbuf_mpi
.rootbuf_mpi    - root buffer used by MPI in the following MPI call
.leafmtype_mpi  - memtype of leafbuf_mpi
-leafbuf_mpi    - leaf buffer used by MPI in the following MPI call
*/
PETSC_INTERN PetscErrorCode PetscSFReducePrepareMPIBuffers_Gatherv(PetscSF sf,PetscSFPack link,MPI_Op op,PetscMemType *rootmtype_mpi,void **rootbuf_mpi,PetscMemType *leafmtype_mpi,const void **leafbuf_mpi)
{
  PetscErrorCode         ierr;
  PetscMPIInt            rank;
  MPI_Comm               comm;

  PetscFunctionBegin;
  ierr = PetscObjectGetComm((PetscObject)sf,&comm);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(comm,&rank);CHKERRQ(ierr);

  if (link->leafmtype == PETSC_MEMTYPE_DEVICE && !use_gpu_aware_mpi) { /* Need to copy leafdata to leafbuf on every rank */
    if (!rank && !link->leafbuf[PETSC_MEMTYPE_HOST]) {ierr = PetscMallocWithMemType(PETSC_MEMTYPE_HOST,link->leafbuflen*link->unitbytes,(void**)&link->leafbuf[PETSC_MEMTYPE_HOST]);CHKERRQ(ierr);}
    ierr = PetscMemcpyWithMemType(PETSC_MEMTYPE_HOST,PETSC_MEMTYPE_DEVICE,link->leafbuf[PETSC_MEMTYPE_HOST],link->lkey,link->leafbuflen*link->unitbytes);CHKERRQ(ierr);
    *leafmtype_mpi = PETSC_MEMTYPE_HOST;
    *leafbuf_mpi   = link->leafbuf[*leafmtype_mpi];
  } else {
    *leafmtype_mpi = link->leafmtype;
    *leafbuf_mpi   = (char*)link->lkey;
  }

  if (link->rootmtype == PETSC_MEMTYPE_DEVICE && !use_gpu_aware_mpi) {  /* If rootdata is on device but no gpu-aware mpi, we need a rootbuf on host to receive reduced data */
    if (!link->rootbuf[PETSC_MEMTYPE_HOST]) {ierr = PetscMallocWithMemType(PETSC_MEMTYPE_HOST,link->rootbuflen*link->unitbytes,(void**)&link->rootbuf[PETSC_MEMTYPE_HOST]);CHKERRQ(ierr);}
    *rootbuf_mpi   = link->rootbuf[PETSC_MEMTYPE_HOST];
    *rootmtype_mpi = PETSC_MEMTYPE_HOST;
  } else if (op == MPIU_REPLACE) { /* Directly use rootdata's memory to receive reduced data. No intermediate buffer needed. */
    *rootbuf_mpi   = (char *)link->rkey;
    *rootmtype_mpi = link->rootmtype;
  } else { /* op is a reduction. Have to allocate a buffer aside rootdata to apply it. The buffer is either on host or device, depending on where rootdata is. */
    if (!link->rootbuf[link->rootmtype]) {ierr = PetscMallocWithMemType(link->rootmtype,link->rootbuflen*link->unitbytes,(void**)&link->rootbuf[link->rootmtype]);CHKERRQ(ierr);}
    *rootbuf_mpi   = link->rootbuf[link->rootmtype];
    *rootmtype_mpi = link->rootmtype;
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscSFReduceBegin_Gatherv(PetscSF sf,MPI_Datatype unit,PetscMemType leafmtype,const void *leafdata,PetscMemType rootmtype,void *rootdata,MPI_Op op)
{
  PetscErrorCode       ierr;
  PetscSFPack          link;
  PetscMPIInt          recvcount;
  MPI_Comm             comm;
  PetscSF_Gatherv      *dat = (PetscSF_Gatherv*)sf->data;
  const void           *leafbuf_mpi;
  void                 *rootbuf_mpi;
  PetscMemType         leafmtype_mpi,rootmtype_mpi;

  PetscFunctionBegin;
  ierr = PetscSFPackGet_Gatherv(sf,unit,rootmtype,rootdata,leafmtype,leafdata,&link);CHKERRQ(ierr);
  ierr = PetscObjectGetComm((PetscObject)sf,&comm);CHKERRQ(ierr);
  ierr = PetscMPIIntCast(sf->nroots,&recvcount);CHKERRQ(ierr);
  ierr = PetscSFReducePrepareMPIBuffers_Gatherv(sf,link,op,&rootmtype_mpi,&rootbuf_mpi,&leafmtype_mpi,&leafbuf_mpi);CHKERRQ(ierr);
  ierr = MPIU_Iscatterv(leafbuf_mpi,dat->recvcounts,dat->displs,unit,rootbuf_mpi,recvcount,unit,0,comm,link->rootreqs[PETSCSF_LEAF2ROOT_REDUCE][rootmtype_mpi]);CHKERRQ(ierr);
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
