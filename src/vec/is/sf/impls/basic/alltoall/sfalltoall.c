#include <../src/vec/is/sf/impls/basic/allgatherv/sfallgatherv.h>
#include <../src/vec/is/sf/impls/basic/allgather/sfallgather.h>
#include <../src/vec/is/sf/impls/basic/gatherv/sfgatherv.h>

/* Reuse the type. The difference is some fields (i.e., displs, recvcounts) are not used, which is not a big deal */
typedef PetscSF_Allgatherv PetscSF_Alltoall;

/*===================================================================================*/
/*              Implementations of SF public APIs                                    */
/*===================================================================================*/
static PetscErrorCode PetscSFGetGraph_Alltoall(PetscSF sf,PetscInt *nroots,PetscInt *nleaves,const PetscInt **ilocal,const PetscSFNode **iremote)
{
  PetscInt       i;

  PetscFunctionBegin;
  if (nroots)  *nroots  = sf->nroots;
  if (nleaves) *nleaves = sf->nleaves;
  if (ilocal)  *ilocal  = NULL; /* Contiguous local indices */
  if (iremote) {
    if (!sf->remote) {
      PetscCall(PetscMalloc1(sf->nleaves,&sf->remote));
      sf->remote_alloc = sf->remote;
      for (i=0; i<sf->nleaves; i++) {
        sf->remote[i].rank  = i;
        sf->remote[i].index = i;
      }
    }
    *iremote = sf->remote;
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscSFBcastBegin_Alltoall(PetscSF sf,MPI_Datatype unit,PetscMemType rootmtype,const void *rootdata,PetscMemType leafmtype,void *leafdata,MPI_Op op)
{
  PetscSFLink          link;
  MPI_Comm             comm;
  void                 *rootbuf = NULL,*leafbuf = NULL; /* buffer used by MPI */
  MPI_Request          *req;

  PetscFunctionBegin;
  PetscCall(PetscSFLinkCreate(sf,unit,rootmtype,rootdata,leafmtype,leafdata,op,PETSCSF_BCAST,&link));
  PetscCall(PetscSFLinkPackRootData(sf,link,PETSCSF_REMOTE,rootdata));
  PetscCall(PetscSFLinkCopyRootBufferInCaseNotUseGpuAwareMPI(sf,link,PETSC_TRUE/* device2host before sending */));
  PetscCall(PetscObjectGetComm((PetscObject)sf,&comm));
  PetscCall(PetscSFLinkGetMPIBuffersAndRequests(sf,link,PETSCSF_ROOT2LEAF,&rootbuf,&leafbuf,&req,NULL));
  PetscCall(PetscSFLinkSyncStreamBeforeCallMPI(sf,link,PETSCSF_ROOT2LEAF));
  PetscCallMPI(MPIU_Ialltoall(rootbuf,1,unit,leafbuf,1,unit,comm,req));
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscSFReduceBegin_Alltoall(PetscSF sf,MPI_Datatype unit,PetscMemType leafmtype,const void *leafdata,PetscMemType rootmtype,void *rootdata,MPI_Op op)
{
  PetscSFLink          link;
  MPI_Comm             comm;
  void                 *rootbuf = NULL,*leafbuf = NULL; /* buffer used by MPI */
  MPI_Request          *req;

  PetscFunctionBegin;
  PetscCall(PetscSFLinkCreate(sf,unit,rootmtype,rootdata,leafmtype,leafdata,op,PETSCSF_REDUCE,&link));
  PetscCall(PetscSFLinkPackLeafData(sf,link,PETSCSF_REMOTE,leafdata));
  PetscCall(PetscSFLinkCopyLeafBufferInCaseNotUseGpuAwareMPI(sf,link,PETSC_TRUE/* device2host before sending */));
  PetscCall(PetscObjectGetComm((PetscObject)sf,&comm));
  PetscCall(PetscSFLinkGetMPIBuffersAndRequests(sf,link,PETSCSF_LEAF2ROOT,&rootbuf,&leafbuf,&req,NULL));
  PetscCall(PetscSFLinkSyncStreamBeforeCallMPI(sf,link,PETSCSF_LEAF2ROOT));
  PetscCallMPI(MPIU_Ialltoall(leafbuf,1,unit,rootbuf,1,unit,comm,req));
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscSFCreateLocalSF_Alltoall(PetscSF sf,PetscSF *out)
{
  PetscInt       nroots = 1,nleaves = 1,*ilocal;
  PetscSFNode    *iremote = NULL;
  PetscSF        lsf;
  PetscMPIInt    rank;

  PetscFunctionBegin;
  nroots  = 1;
  nleaves = 1;
  PetscCallMPI(MPI_Comm_rank(PetscObjectComm((PetscObject)sf),&rank));
  PetscCall(PetscMalloc1(nleaves,&ilocal));
  PetscCall(PetscMalloc1(nleaves,&iremote));
  ilocal[0]        = rank;
  iremote[0].rank  = 0;    /* rank in PETSC_COMM_SELF */
  iremote[0].index = rank; /* LocalSF is an embedded SF. Indices are not remapped */

  PetscCall(PetscSFCreate(PETSC_COMM_SELF,&lsf));
  PetscCall(PetscSFSetGraph(lsf,nroots,nleaves,NULL/*contiguous leaves*/,PETSC_OWN_POINTER,iremote,PETSC_OWN_POINTER));
  PetscCall(PetscSFSetUp(lsf));
  *out = lsf;
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscSFCreateEmbeddedRootSF_Alltoall(PetscSF sf,PetscInt nselected,const PetscInt *selected,PetscSF *newsf)
{
  PetscInt       i,*tmproots,*ilocal,ndranks,ndiranks;
  PetscSFNode    *iremote;
  PetscMPIInt    nroots,*roots,nleaves,*leaves,rank;
  MPI_Comm       comm;
  PetscSF_Basic  *bas;
  PetscSF        esf;

  PetscFunctionBegin;
  PetscCall(PetscObjectGetComm((PetscObject)sf,&comm));
  PetscCallMPI(MPI_Comm_rank(comm,&rank));

  /* Uniq selected[] and store the result in roots[] */
  PetscCall(PetscMalloc1(nselected,&tmproots));
  PetscCall(PetscArraycpy(tmproots,selected,nselected));
  PetscCall(PetscSortRemoveDupsInt(&nselected,tmproots)); /* nselected might be changed */
  PetscCheck(tmproots[0] >= 0 && tmproots[nselected-1] < sf->nroots,comm,PETSC_ERR_ARG_OUTOFRANGE,"Min/Max root indices %" PetscInt_FMT "/%" PetscInt_FMT " are not in [0,%" PetscInt_FMT ")",tmproots[0],tmproots[nselected-1],sf->nroots);
  nroots = nselected;   /* For Alltoall, we know root indices will not overflow MPI_INT */
  PetscCall(PetscMalloc1(nselected,&roots));
  for (i=0; i<nselected; i++) roots[i] = tmproots[i];
  PetscCall(PetscFree(tmproots));

  /* Find out which leaves are still connected to roots in the embedded sf. Expect PetscCommBuildTwoSided is more scalable than MPI_Alltoall */
  PetscCall(PetscCommBuildTwoSided(comm,0/*empty msg*/,MPI_INT/*fake*/,nroots,roots,NULL/*todata*/,&nleaves,&leaves,NULL/*fromdata*/));

  /* Move myself ahead if rank is in leaves[], since I am a distinguished rank */
  ndranks = 0;
  for (i=0; i<nleaves; i++) {
    if (leaves[i] == rank) {leaves[i] = -rank; ndranks = 1; break;}
  }
  PetscCall(PetscSortMPIInt(nleaves,leaves));
  if (nleaves && leaves[0] < 0) leaves[0] = rank;

  /* Build esf and fill its fields manually (without calling PetscSFSetUp) */
  PetscCall(PetscMalloc1(nleaves,&ilocal));
  PetscCall(PetscMalloc1(nleaves,&iremote));
  for (i=0; i<nleaves; i++) { /* 1:1 map from roots to leaves */
    ilocal[i]        = leaves[i];
    iremote[i].rank  = leaves[i];
    iremote[i].index = leaves[i];
  }
  PetscCall(PetscSFCreate(comm,&esf));
  PetscCall(PetscSFSetType(esf,PETSCSFBASIC)); /* This optimized routine can only create a basic sf */
  PetscCall(PetscSFSetGraph(esf,sf->nleaves,nleaves,ilocal,PETSC_OWN_POINTER,iremote,PETSC_OWN_POINTER));

  /* As if we are calling PetscSFSetUpRanks(esf,self's group) */
  PetscCall(PetscMalloc4(nleaves,&esf->ranks,nleaves+1,&esf->roffset,nleaves,&esf->rmine,nleaves,&esf->rremote));
  esf->nranks     = nleaves;
  esf->ndranks    = ndranks;
  esf->roffset[0] = 0;
  for (i=0; i<nleaves; i++) {
    esf->ranks[i]     = leaves[i];
    esf->roffset[i+1] = i+1;
    esf->rmine[i]     = leaves[i];
    esf->rremote[i]   = leaves[i];
  }

  /* Set up esf->data, the incoming communication (i.e., recv info), which is usually done by PetscSFSetUp_Basic */
  bas  = (PetscSF_Basic*)esf->data;
  PetscCall(PetscMalloc2(nroots,&bas->iranks,nroots+1,&bas->ioffset));
  PetscCall(PetscMalloc1(nroots,&bas->irootloc));
  /* Move myself ahead if rank is in roots[], since I am a distinguished irank */
  ndiranks = 0;
  for (i=0; i<nroots; i++) {
    if (roots[i] == rank) {roots[i] = -rank; ndiranks = 1; break;}
  }
  PetscCall(PetscSortMPIInt(nroots,roots));
  if (nroots && roots[0] < 0) roots[0] = rank;

  bas->niranks    = nroots;
  bas->ndiranks   = ndiranks;
  bas->ioffset[0] = 0;
  bas->itotal     = nroots;
  for (i=0; i<nroots; i++) {
    bas->iranks[i]    = roots[i];
    bas->ioffset[i+1] = i+1;
    bas->irootloc[i]  = roots[i];
  }

  /* See PetscSFCreateEmbeddedRootSF_Basic */
  esf->nleafreqs  = esf->nranks - esf->ndranks;
  bas->nrootreqs  = bas->niranks - bas->ndiranks;
  esf->persistent = PETSC_TRUE;
  /* Setup packing related fields */
  PetscCall(PetscSFSetUpPackFields(esf));

  esf->setupcalled = PETSC_TRUE; /* We have done setup ourselves! */
  *newsf = esf;
  PetscFunctionReturn(0);
}

PETSC_INTERN PetscErrorCode PetscSFCreate_Alltoall(PetscSF sf)
{
  PetscSF_Alltoall *dat = (PetscSF_Alltoall*)sf->data;

  PetscFunctionBegin;
  sf->ops->BcastEnd        = PetscSFBcastEnd_Basic;
  sf->ops->ReduceEnd       = PetscSFReduceEnd_Basic;

  /* Inherit from Allgatherv. It is astonishing Alltoall can inherit so much from Allgather(v) */
  sf->ops->Destroy         = PetscSFDestroy_Allgatherv;
  sf->ops->Reset           = PetscSFReset_Allgatherv;
  sf->ops->FetchAndOpEnd   = PetscSFFetchAndOpEnd_Allgatherv;
  sf->ops->GetRootRanks    = PetscSFGetRootRanks_Allgatherv;

  /* Inherit from Allgather. Every process gathers equal-sized data from others, which enables this inheritance. */
  sf->ops->GetLeafRanks    = PetscSFGetLeafRanks_Allgatherv;
  sf->ops->SetUp           = PetscSFSetUp_Allgather;

  /* Inherit from Gatherv. Each root has only one leaf connected, which enables this inheritance */
  sf->ops->FetchAndOpBegin = PetscSFFetchAndOpBegin_Gatherv;

  /* Alltoall stuff */
  sf->ops->GetGraph             = PetscSFGetGraph_Alltoall;
  sf->ops->BcastBegin           = PetscSFBcastBegin_Alltoall;
  sf->ops->ReduceBegin          = PetscSFReduceBegin_Alltoall;
  sf->ops->CreateLocalSF        = PetscSFCreateLocalSF_Alltoall;
  sf->ops->CreateEmbeddedRootSF = PetscSFCreateEmbeddedRootSF_Alltoall;

  PetscCall(PetscNewLog(sf,&dat));
  sf->data = (void*)dat;
  PetscFunctionReturn(0);
}
