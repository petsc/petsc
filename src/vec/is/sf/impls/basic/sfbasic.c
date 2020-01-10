
#include <../src/vec/is/sf/impls/basic/sfbasic.h>

/*===================================================================================*/
/*              Internal routines for PetscSFPack                              */
/*===================================================================================*/

/* Return root and leaf MPI requests for communication in the given direction. If the requests have not been
   initialized (since we use persistent requests), then initialize them.
*/
static PetscErrorCode PetscSFPackGetReqs_Basic(PetscSF sf,PetscSFPack link,PetscSFDirection direction,MPI_Request **rootreqs,MPI_Request **leafreqs)
{
  PetscErrorCode ierr;
  PetscSF_Basic  *bas = (PetscSF_Basic*)sf->data;
  PetscInt       i,j,nrootranks,ndrootranks,nleafranks,ndleafranks;
  const PetscInt *rootoffset,*leafoffset;
  PetscMPIInt    n;
  MPI_Comm       comm = PetscObjectComm((PetscObject)sf);
  MPI_Datatype   unit = link->unit;
  PetscMemType   rootmtype,leafmtype;

  PetscFunctionBegin;
  if (use_gpu_aware_mpi) {
    rootmtype = link->rootmtype;
    leafmtype = link->leafmtype;
  } else {
    rootmtype = PETSC_MEMTYPE_HOST;
    leafmtype = PETSC_MEMTYPE_HOST;
  }

  if (rootreqs && !link->rootreqsinited[direction][rootmtype]) {
    ierr = PetscSFGetRootInfo_Basic(sf,&nrootranks,&ndrootranks,NULL,&rootoffset,NULL);CHKERRQ(ierr);
    if (direction == PETSCSF_LEAF2ROOT_REDUCE) {
      for (i=ndrootranks,j=0; i<nrootranks; i++,j++) {
        MPI_Aint disp = (rootoffset[i] - rootoffset[ndrootranks])*link->unitbytes;
        ierr = PetscMPIIntCast(rootoffset[i+1]-rootoffset[i],&n);CHKERRQ(ierr);
        ierr = MPI_Recv_init(link->rootbuf[rootmtype]+disp,n,unit,bas->iranks[i],link->tag,comm,&link->rootreqs[direction][rootmtype][j]);CHKERRQ(ierr);
      }
    } else if (direction == PETSCSF_ROOT2LEAF_BCAST) {
      for (i=ndrootranks,j=0; i<nrootranks; i++,j++) {
        MPI_Aint disp = (rootoffset[i] - rootoffset[ndrootranks])*link->unitbytes;
        ierr = PetscMPIIntCast(rootoffset[i+1]-rootoffset[i],&n);CHKERRQ(ierr);
        ierr = MPI_Send_init(link->rootbuf[rootmtype]+disp,n,unit,bas->iranks[i],link->tag,comm,&link->rootreqs[direction][rootmtype][j]);CHKERRQ(ierr);
      }
    } else SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Out-of-range PetscSFDirection = %d\n",(int)direction);
    link->rootreqsinited[direction][rootmtype] = PETSC_TRUE;
  }

  if (leafreqs && !link->leafreqsinited[direction][leafmtype]) {
    ierr = PetscSFGetLeafInfo_Basic(sf,&nleafranks,&ndleafranks,NULL,&leafoffset,NULL,NULL);CHKERRQ(ierr);
    if (direction == PETSCSF_LEAF2ROOT_REDUCE) {
      for (i=ndleafranks,j=0; i<nleafranks; i++,j++) {
        MPI_Aint disp = (leafoffset[i] - leafoffset[ndleafranks])*link->unitbytes;
        ierr = PetscMPIIntCast(leafoffset[i+1]-leafoffset[i],&n);CHKERRQ(ierr);
        ierr = MPI_Send_init(link->leafbuf[leafmtype]+disp,n,unit,sf->ranks[i],link->tag,comm,&link->leafreqs[direction][leafmtype][j]);CHKERRQ(ierr);
      }
    } else if (direction == PETSCSF_ROOT2LEAF_BCAST) {
      for (i=ndleafranks,j=0; i<nleafranks; i++,j++) {
        MPI_Aint disp = (leafoffset[i] - leafoffset[ndleafranks])*link->unitbytes;
        ierr = PetscMPIIntCast(leafoffset[i+1]-leafoffset[i],&n);CHKERRQ(ierr);
        ierr = MPI_Recv_init(link->leafbuf[leafmtype]+disp,n,unit,sf->ranks[i],link->tag,comm,&link->leafreqs[direction][leafmtype][j]);CHKERRQ(ierr);
      }
    } else SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Out-of-range PetscSFDirection = %d\n",(int)direction);
    link->leafreqsinited[direction][leafmtype] = PETSC_TRUE;
  }

  if (rootreqs) *rootreqs = link->rootreqs[direction][rootmtype];
  if (leafreqs) *leafreqs = link->leafreqs[direction][leafmtype];
  PetscFunctionReturn(0);
}

/* Common part shared by SFBasic and SFNeighbor based on the fact they all deal with sparse graphs. */
PETSC_INTERN PetscErrorCode PetscSFPackGet_Basic_Common(PetscSF sf,MPI_Datatype unit,PetscMemType rootmtype,const void *rootdata,PetscMemType leafmtype,const void *leafdata,PetscInt nrootreqs,PetscInt nleafreqs,PetscSFPack *mylink)
{
  PetscErrorCode    ierr;
  PetscSF_Basic     *bas = (PetscSF_Basic*)sf->data;
  PetscInt          i,j,nreqs,nrootranks,ndrootranks,nleafranks,ndleafranks;
  const PetscInt    *rootoffset,*leafoffset;
  PetscSFPack       *p,link;
  PetscBool         match;

  PetscFunctionBegin;
  ierr = PetscSFPackSetErrorOnUnsupportedOverlap(sf,unit,rootdata,leafdata);CHKERRQ(ierr);

  /* Look for types in cache */
  for (p=&bas->avail; (link=*p); p=&link->next) {
    ierr = MPIPetsc_Type_compare(unit,link->unit,&match);CHKERRQ(ierr);
    if (match) {
      *p = link->next; /* Remove from available list */
      goto found;
    }
  }

  ierr = PetscSFGetRootInfo_Basic(sf,&nrootranks,&ndrootranks,NULL,&rootoffset,NULL);CHKERRQ(ierr);
  ierr = PetscSFGetLeafInfo_Basic(sf,&nleafranks,&ndleafranks,NULL,&leafoffset,NULL,NULL);CHKERRQ(ierr);
  ierr = PetscNew(&link);CHKERRQ(ierr);
  ierr = PetscSFPackSetUp_Host(sf,link,unit);CHKERRQ(ierr);
  ierr = PetscCommGetNewTag(PetscObjectComm((PetscObject)sf),&link->tag);CHKERRQ(ierr); /* One tag per link */

  /* Allocate root, leaf, self buffers, and MPI requests */
  link->rootbuflen = rootoffset[nrootranks]-rootoffset[ndrootranks];
  link->leafbuflen = leafoffset[nleafranks]-leafoffset[ndleafranks];
  link->selfbuflen = rootoffset[ndrootranks];
  link->nrootreqs  = nrootreqs;
  link->nleafreqs  = nleafreqs;
  nreqs            = (nrootreqs+nleafreqs)*4; /* Quadruple the requests since there are two communication directions and two memory types */
  ierr             = PetscMalloc1(nreqs,&link->reqs);CHKERRQ(ierr);
  for (i=0; i<nreqs; i++) link->reqs[i] = MPI_REQUEST_NULL; /* Initialized to NULL so that we know which need to be freed in Destroy */

  for (i=0; i<2; i++) { /* Two communication directions */
    for (j=0; j<2; j++) { /* Two memory types */
      link->rootreqs[i][j] = link->reqs + nrootreqs*(2*i+j);
      link->leafreqs[i][j] = link->reqs + nrootreqs*4 + nleafreqs*(2*i+j);
    }
  }

found:
  link->rootmtype = rootmtype;
  link->leafmtype = leafmtype;
#if defined(PETSC_HAVE_CUDA)
  ierr = PetscSFPackSetUp_Device(sf,link,unit);CHKERRQ(ierr);
  if (!use_gpu_aware_mpi) {
    /* If not using GPU aware MPI, we always need buffers on host. In case root/leafdata is on device, we copy root/leafdata to/from
       these buffers for MPI. We only need buffers for remote neighbors since self-to-self communication is not done via MPI.
     */
    if (!link->rootbuf[PETSC_MEMTYPE_HOST]) {
      if (rootmtype == PETSC_MEMTYPE_DEVICE && sf->use_pinned_buf) {
        ierr = PetscMallocPinnedMemory(link->rootbuflen*link->unitbytes,(void**)&link->rootbuf[PETSC_MEMTYPE_HOST]);CHKERRQ(ierr);
      } else {
        ierr = PetscMallocWithMemType(PETSC_MEMTYPE_HOST,link->rootbuflen*link->unitbytes,(void**)&link->rootbuf[PETSC_MEMTYPE_HOST]);CHKERRQ(ierr);
      }
    }
    if (!link->leafbuf[PETSC_MEMTYPE_HOST]) {
      if (leafmtype == PETSC_MEMTYPE_DEVICE && sf->use_pinned_buf) {
        ierr = PetscMallocPinnedMemory(link->leafbuflen*link->unitbytes,(void**)&link->leafbuf[PETSC_MEMTYPE_HOST]);CHKERRQ(ierr);
      } else {
        ierr = PetscMallocWithMemType(PETSC_MEMTYPE_HOST,link->leafbuflen*link->unitbytes,(void**)&link->leafbuf[PETSC_MEMTYPE_HOST]);CHKERRQ(ierr);
      }
    }
  }
#endif
  if (!link->rootbuf[rootmtype]) {ierr = PetscMallocWithMemType(rootmtype,link->rootbuflen*link->unitbytes,(void**)&link->rootbuf[rootmtype]);CHKERRQ(ierr);}
  if (!link->leafbuf[leafmtype]) {ierr = PetscMallocWithMemType(leafmtype,link->leafbuflen*link->unitbytes,(void**)&link->leafbuf[leafmtype]);CHKERRQ(ierr);}
  if (!link->selfbuf[rootmtype]) {ierr = PetscMallocWithMemType(rootmtype,link->selfbuflen*link->unitbytes,(void**)&link->selfbuf[rootmtype]);CHKERRQ(ierr);}
  if (rootmtype != leafmtype && !link->selfbuf[leafmtype]) {ierr = PetscMallocWithMemType(leafmtype,link->selfbuflen*link->unitbytes,(void**)&link->selfbuf[leafmtype]);CHKERRQ(ierr);}
  link->rootdata = rootdata;
  link->leafdata = leafdata;
  link->next     = bas->inuse;
  bas->inuse     = link;
  *mylink        = link;
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscSFPackGet_Basic(PetscSF sf,MPI_Datatype unit,PetscMemType rootmtype,const void *rootdata,PetscMemType leafmtype,const void *leafdata,PetscSFDirection direction,PetscSFPack *mylink)
{
  PetscErrorCode    ierr;
  PetscInt          nrootranks,ndrootranks,nleafranks,ndleafranks;

  PetscFunctionBegin;
  ierr = PetscSFGetRootInfo_Basic(sf,&nrootranks,&ndrootranks,NULL,NULL,NULL);CHKERRQ(ierr);
  ierr = PetscSFGetLeafInfo_Basic(sf,&nleafranks,&ndleafranks,NULL,NULL,NULL,NULL);CHKERRQ(ierr);
  ierr = PetscSFPackGet_Basic_Common(sf,unit,rootmtype,rootdata,leafmtype,leafdata,nrootranks-ndrootranks,nleafranks-ndleafranks,mylink);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*===================================================================================*/
/*              SF public interface implementations                                  */
/*===================================================================================*/
PETSC_INTERN PetscErrorCode PetscSFSetUp_Basic(PetscSF sf)
{
  PetscErrorCode ierr;
  PetscSF_Basic  *bas = (PetscSF_Basic*)sf->data;
  PetscInt       *rlengths,*ilengths,i;
  PetscMPIInt    rank,niranks,*iranks,tag;
  MPI_Comm       comm;
  MPI_Group      group;
  MPI_Request    *rootreqs,*leafreqs;

  PetscFunctionBegin;
  ierr = MPI_Comm_group(PETSC_COMM_SELF,&group);CHKERRQ(ierr);
  ierr = PetscSFSetUpRanks(sf,group);CHKERRQ(ierr);
  ierr = MPI_Group_free(&group);CHKERRQ(ierr);
  ierr = PetscObjectGetComm((PetscObject)sf,&comm);CHKERRQ(ierr);
  ierr = PetscObjectGetNewTag((PetscObject)sf,&tag);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(comm,&rank);CHKERRQ(ierr);
  /*
   * Inform roots about how many leaves and from which ranks
   */
  ierr = PetscMalloc1(sf->nranks,&rlengths);CHKERRQ(ierr);
  /* Determine number, sending ranks, and length of incoming */
  for (i=0; i<sf->nranks; i++) {
    rlengths[i] = sf->roffset[i+1] - sf->roffset[i]; /* Number of roots referenced by my leaves; for rank sf->ranks[i] */
  }
  ierr = PetscCommBuildTwoSided(comm,1,MPIU_INT,sf->nranks-sf->ndranks,sf->ranks+sf->ndranks,rlengths+sf->ndranks,&niranks,&iranks,(void**)&ilengths);CHKERRQ(ierr);

  /* Sort iranks. See use of VecScatterGetRemoteOrdered_Private() in MatGetBrowsOfAoCols_MPIAIJ() on why.
     We could sort ranks there at the price of allocating extra working arrays. Presumably, niranks is
     small and the sorting is cheap.
   */
  ierr = PetscSortMPIIntWithIntArray(niranks,iranks,ilengths);CHKERRQ(ierr);

  /* Partition into distinguished and non-distinguished incoming ranks */
  bas->ndiranks = sf->ndranks;
  bas->niranks = bas->ndiranks + niranks;
  ierr = PetscMalloc2(bas->niranks,&bas->iranks,bas->niranks+1,&bas->ioffset);CHKERRQ(ierr);
  bas->ioffset[0] = 0;
  for (i=0; i<bas->ndiranks; i++) {
    bas->iranks[i] = sf->ranks[i];
    bas->ioffset[i+1] = bas->ioffset[i] + rlengths[i];
  }
  if (bas->ndiranks > 1 || (bas->ndiranks == 1 && bas->iranks[0] != rank)) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_PLIB,"Broken setup for shared ranks");
  for ( ; i<bas->niranks; i++) {
    bas->iranks[i] = iranks[i-bas->ndiranks];
    bas->ioffset[i+1] = bas->ioffset[i] + ilengths[i-bas->ndiranks];
  }
  bas->itotal = bas->ioffset[i];
  ierr = PetscFree(rlengths);CHKERRQ(ierr);
  ierr = PetscFree(iranks);CHKERRQ(ierr);
  ierr = PetscFree(ilengths);CHKERRQ(ierr);

  /* Send leaf identities to roots */
  ierr = PetscMalloc1(bas->itotal,&bas->irootloc);CHKERRQ(ierr);
  ierr = PetscMalloc2(bas->niranks-bas->ndiranks,&rootreqs,sf->nranks-sf->ndranks,&leafreqs);CHKERRQ(ierr);
  for (i=bas->ndiranks; i<bas->niranks; i++) {
    ierr = MPI_Irecv(bas->irootloc+bas->ioffset[i],bas->ioffset[i+1]-bas->ioffset[i],MPIU_INT,bas->iranks[i],tag,comm,&rootreqs[i-bas->ndiranks]);CHKERRQ(ierr);
  }
  for (i=0; i<sf->nranks; i++) {
    PetscMPIInt npoints;
    ierr = PetscMPIIntCast(sf->roffset[i+1] - sf->roffset[i],&npoints);CHKERRQ(ierr);
    if (i < sf->ndranks) {
      if (sf->ranks[i] != rank) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_PLIB,"Cannot interpret distinguished leaf rank");
      if (bas->iranks[0] != rank) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_PLIB,"Cannot interpret distinguished root rank");
      if (npoints != bas->ioffset[1]-bas->ioffset[0]) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_PLIB,"Distinguished rank exchange has mismatched lengths");
      ierr = PetscArraycpy(bas->irootloc+bas->ioffset[0],sf->rremote+sf->roffset[i],npoints);CHKERRQ(ierr);
      continue;
    }
    ierr = MPI_Isend(sf->rremote+sf->roffset[i],npoints,MPIU_INT,sf->ranks[i],tag,comm,&leafreqs[i-sf->ndranks]);CHKERRQ(ierr);
  }
  ierr = MPI_Waitall(bas->niranks-bas->ndiranks,rootreqs,MPI_STATUSES_IGNORE);CHKERRQ(ierr);
  ierr = MPI_Waitall(sf->nranks-sf->ndranks,leafreqs,MPI_STATUSES_IGNORE);CHKERRQ(ierr);
  ierr = PetscFree2(rootreqs,leafreqs);CHKERRQ(ierr);

  sf->selfleafdups    = PETSC_TRUE; /* The conservative assumption is there are data race */
  sf->remoteleafdups  = PETSC_TRUE;
  bas->selfrootdups   = PETSC_TRUE;
  bas->remoterootdups = PETSC_TRUE;

  /* Setup packing optimization for roots and leaves */
  ierr = PetscSFPackSetupOptimizations_Basic(sf);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PETSC_INTERN PetscErrorCode PetscSFSetFromOptions_Basic(PetscOptionItems *PetscOptionsObject,PetscSF sf)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscOptionsHead(PetscOptionsObject,"PetscSF Basic options");CHKERRQ(ierr);
  ierr = PetscOptionsBool("-sf_use_pinned_buffer","Use pinned (nonpagable) memory for send/recv buffers on host","PetscSFSetFromOptions",sf->use_pinned_buf,&sf->use_pinned_buf,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsTail();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PETSC_INTERN PetscErrorCode PetscSFReset_Basic(PetscSF sf)
{
  PetscSF_Basic     *bas = (PetscSF_Basic*)sf->data;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  if (bas->inuse) SETERRQ(PetscObjectComm((PetscObject)sf),PETSC_ERR_ARG_WRONGSTATE,"Outstanding operation has not been completed");
  ierr = PetscFree2(bas->iranks,bas->ioffset);CHKERRQ(ierr);
  ierr = PetscFree(bas->irootloc);CHKERRQ(ierr);
#if defined(PETSC_HAVE_CUDA)
  if (bas->irootloc_d) {cudaError_t err = cudaFree(bas->irootloc_d);CHKERRCUDA(err);bas->irootloc_d=NULL;}
#endif
  ierr = PetscSFPackDestroyAvailable(sf,&bas->avail);CHKERRQ(ierr);
  ierr = PetscSFPackDestroyOptimizations_Basic(sf);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PETSC_INTERN PetscErrorCode PetscSFDestroy_Basic(PetscSF sf)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscSFReset(sf);CHKERRQ(ierr);
  ierr = PetscFree(sf->data);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PETSC_INTERN PetscErrorCode PetscSFView_Basic(PetscSF sf,PetscViewer viewer)
{
  PetscErrorCode ierr;
  PetscBool      iascii;

  PetscFunctionBegin;
  ierr = PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&iascii);CHKERRQ(ierr);
  if (iascii) {ierr = PetscViewerASCIIPrintf(viewer,"  sort=%s\n",sf->rankorder ? "rank-order" : "unordered");CHKERRQ(ierr);}
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscSFBcastAndOpBegin_Basic(PetscSF sf,MPI_Datatype unit,PetscMemType rootmtype,const void *rootdata,PetscMemType leafmtype,void *leafdata,MPI_Op op)
{
  PetscErrorCode    ierr;
  PetscSFPack       link;
  const PetscInt    *rootloc = NULL;
  MPI_Request       *rootreqs = NULL,*leafreqs = NULL;

  PetscFunctionBegin;
  ierr = PetscSFPackGet_Basic(sf,unit,rootmtype,rootdata,leafmtype,leafdata,PETSCSF_ROOT2LEAF_BCAST,&link);CHKERRQ(ierr);
  ierr = PetscSFGetRootIndicesWithMemType_Basic(sf,rootmtype,&rootloc);CHKERRQ(ierr);

  ierr = PetscSFPackGetReqs_Basic(sf,link,PETSCSF_ROOT2LEAF_BCAST,&rootreqs,&leafreqs);CHKERRQ(ierr);
  /* Post Irecv. Note distinguished ranks receive data via shared memory (i.e., not via MPI) */
  ierr = MPI_Startall_irecv(link->leafbuflen,unit,link->nleafreqs,leafreqs);CHKERRQ(ierr);

  /* Do Isend */
  ierr = PetscSFPackRootData(sf,link,rootloc,rootdata,PETSC_TRUE);CHKERRQ(ierr);
  ierr = MPI_Startall_isend(link->rootbuflen,unit,link->nrootreqs,rootreqs);CHKERRQ(ierr);

  /* Do self to self communication via memcpy only when rootdata and leafdata are in different memory */
  if (rootmtype != leafmtype) {ierr = PetscMemcpyWithMemType(leafmtype,rootmtype,link->selfbuf[leafmtype],link->selfbuf[rootmtype],link->selfbuflen*link->unitbytes);CHKERRQ(ierr);}
  PetscFunctionReturn(0);
}

PETSC_INTERN PetscErrorCode PetscSFBcastAndOpEnd_Basic(PetscSF sf,MPI_Datatype unit,PetscMemType rootmtype,const void *rootdata,PetscMemType leafmtype,void *leafdata,MPI_Op op)
{
  PetscErrorCode    ierr;
  PetscSFPack       link;
  const PetscInt    *leafloc = NULL;

  PetscFunctionBegin;
  ierr = PetscSFPackGetInUse(sf,unit,rootdata,leafdata,PETSC_OWN_POINTER,&link);CHKERRQ(ierr);
  ierr = PetscSFPackWaitall(link,PETSCSF_ROOT2LEAF_BCAST);CHKERRQ(ierr);
  ierr = PetscSFGetLeafIndicesWithMemType_Basic(sf,leafmtype,&leafloc);CHKERRQ(ierr);
  ierr = PetscSFUnpackAndOpLeafData(sf,link,leafloc,leafdata,op,PETSC_TRUE);CHKERRQ(ierr);
  ierr = PetscSFPackReclaim(sf,&link);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* leaf -> root with reduction */
static PetscErrorCode PetscSFReduceBegin_Basic(PetscSF sf,MPI_Datatype unit,PetscMemType leafmtype,const void *leafdata,PetscMemType rootmtype,void *rootdata,MPI_Op op)
{
  PetscErrorCode    ierr;
  PetscSFPack       link;
  const PetscInt    *leafloc = NULL;
  MPI_Request       *rootreqs = NULL,*leafreqs = NULL; /* dummy values for compiler warnings about uninitialized value */

  PetscFunctionBegin;
  ierr = PetscSFGetLeafIndicesWithMemType_Basic(sf,leafmtype,&leafloc);

  ierr = PetscSFPackGet_Basic(sf,unit,rootmtype,rootdata,leafmtype,leafdata,PETSCSF_LEAF2ROOT_REDUCE,&link);CHKERRQ(ierr);
  ierr = PetscSFPackGetReqs_Basic(sf,link,PETSCSF_LEAF2ROOT_REDUCE,&rootreqs,&leafreqs);CHKERRQ(ierr);
  /* Eagerly post root receives for non-distinguished ranks */
  ierr = MPI_Startall_irecv(link->rootbuflen,unit,link->nrootreqs,rootreqs);CHKERRQ(ierr);

  /* Pack and send leaf data */
  ierr = PetscSFPackLeafData(sf,link,leafloc,leafdata,PETSC_TRUE);CHKERRQ(ierr);
  ierr = MPI_Startall_isend(link->leafbuflen,unit,link->nleafreqs,leafreqs);CHKERRQ(ierr);

  if (rootmtype != leafmtype) {ierr = PetscMemcpyWithMemType(rootmtype,leafmtype,link->selfbuf[rootmtype],link->selfbuf[leafmtype],link->selfbuflen*link->unitbytes);CHKERRQ(ierr);}
  PetscFunctionReturn(0);
}

PETSC_INTERN PetscErrorCode PetscSFReduceEnd_Basic(PetscSF sf,MPI_Datatype unit,PetscMemType leafmtype,const void *leafdata,PetscMemType rootmtype,void *rootdata,MPI_Op op)
{
  PetscErrorCode    ierr;
  PetscSFPack       link;
  const PetscInt    *rootloc = NULL;

  PetscFunctionBegin;
  ierr = PetscSFGetRootIndicesWithMemType_Basic(sf,rootmtype,&rootloc);CHKERRQ(ierr);
  ierr = PetscSFPackGetInUse(sf,unit,rootdata,leafdata,PETSC_OWN_POINTER,&link);CHKERRQ(ierr);
  ierr = PetscSFPackWaitall(link,PETSCSF_LEAF2ROOT_REDUCE);CHKERRQ(ierr);
  ierr = PetscSFUnpackAndOpRootData(sf,link,rootloc,rootdata,op,PETSC_TRUE);CHKERRQ(ierr);
  ierr = PetscSFPackReclaim(sf,&link);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PETSC_INTERN PetscErrorCode PetscSFFetchAndOpBegin_Basic(PetscSF sf,MPI_Datatype unit,PetscMemType rootmtype,void *rootdata,PetscMemType leafmtype,const void *leafdata,void *leafupdate,MPI_Op op)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscSFReduceBegin(sf,unit,leafdata,rootdata,op);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscSFFetchAndOpEnd_Basic(PetscSF sf,MPI_Datatype unit,PetscMemType rootmtype,void *rootdata,PetscMemType leafmtype,const void *leafdata,void *leafupdate,MPI_Op op)
{
  PetscErrorCode    ierr;
  PetscSFPack       link;
  const PetscInt    *rootloc = NULL,*leafloc = NULL;
  MPI_Request       *rootreqs = NULL,*leafreqs = NULL;

  PetscFunctionBegin;
  ierr = PetscSFGetRootIndicesWithMemType_Basic(sf,rootmtype,&rootloc);CHKERRQ(ierr);
  ierr = PetscSFGetLeafIndicesWithMemType_Basic(sf,leafmtype,&leafloc);CHKERRQ(ierr);
  ierr = PetscSFPackGetInUse(sf,unit,rootdata,leafdata,PETSC_OWN_POINTER,&link);CHKERRQ(ierr);
  /* This implementation could be changed to unpack as receives arrive, at the cost of non-determinism */
  ierr = PetscSFPackWaitall(link,PETSCSF_LEAF2ROOT_REDUCE);CHKERRQ(ierr);
  ierr = PetscSFPackGetReqs_Basic(sf,link,PETSCSF_ROOT2LEAF_BCAST,&rootreqs,&leafreqs);CHKERRQ(ierr);

  /* Post leaf receives */
  ierr = MPI_Startall_irecv(link->leafbuflen,unit,link->nleafreqs,leafreqs);CHKERRQ(ierr);

  /* Process local fetch-and-op, post root sends */
  ierr = PetscSFFetchAndOpRootData(sf,link,rootloc,rootdata,op,PETSC_TRUE);CHKERRQ(ierr);
  ierr = MPI_Startall_isend(link->rootbuflen,unit,link->nrootreqs,rootreqs);CHKERRQ(ierr);
  if (rootmtype != leafmtype) {ierr = PetscMemcpyWithMemType(leafmtype,rootmtype,link->selfbuf[leafmtype],link->selfbuf[rootmtype],link->selfbuflen*link->unitbytes);CHKERRQ(ierr);}

  /* Unpack and insert fetched data into leaves */
  ierr = PetscSFPackWaitall(link,PETSCSF_ROOT2LEAF_BCAST);CHKERRQ(ierr);
  ierr = PetscSFUnpackAndOpLeafData(sf,link,leafloc,leafupdate,MPIU_REPLACE,PETSC_TRUE);CHKERRQ(ierr);
  ierr = PetscSFPackReclaim(sf,&link);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PETSC_INTERN PetscErrorCode PetscSFGetLeafRanks_Basic(PetscSF sf,PetscInt *niranks,const PetscMPIInt **iranks,const PetscInt **ioffset,const PetscInt **irootloc)
{
  PetscSF_Basic *bas = (PetscSF_Basic*)sf->data;

  PetscFunctionBegin;
  if (niranks)  *niranks  = bas->niranks;
  if (iranks)   *iranks   = bas->iranks;
  if (ioffset)  *ioffset  = bas->ioffset;
  if (irootloc) *irootloc = bas->irootloc;
  PetscFunctionReturn(0);
}

/* An optimized PetscSFCreateEmbeddedSF. We aggresively make use of the established communication on sf.
   We need one bcast on sf, and no communication anymore to build the embedded sf. Note that selected[]
   was sorted before calling the routine.
 */
PETSC_INTERN PetscErrorCode PetscSFCreateEmbeddedSF_Basic(PetscSF sf,PetscInt nselected,const PetscInt *selected,PetscSF *newsf)
{
  PetscSF           esf;
  PetscInt          esf_nranks,esf_ndranks,*esf_roffset,*esf_rmine,*esf_rremote,count;
  PetscInt          i,j,k,p,q,nroots,*rootdata,*leafdata,connected_leaves,*new_ilocal,nranks,ndranks,niranks,ndiranks,minleaf,maxleaf,maxlocal;
  PetscMPIInt       *esf_ranks;
  const PetscMPIInt *ranks,*iranks;
  const PetscInt    *roffset,*rmine,*rremote,*ioffset,*irootloc,*buffer;
  PetscBool         connected;
  PetscSFPack       link;
  PetscSFNode       *new_iremote;
  PetscSF_Basic     *bas;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  ierr = PetscSFCreate(PetscObjectComm((PetscObject)sf),&esf);CHKERRQ(ierr);
  ierr = PetscSFSetType(esf,PETSCSFBASIC);CHKERRQ(ierr); /* This optimized routine can only create a basic sf */

  /* Find out which leaves are still connected to roots in the embedded sf */
  ierr = PetscSFGetGraph(sf,&nroots,NULL,NULL,NULL);CHKERRQ(ierr);
  ierr = PetscSFGetLeafRange(sf,&minleaf,&maxleaf);CHKERRQ(ierr);
  /* We abused the term leafdata here, whose size is usually the number of leaf data items. Here its size is # of leaves (always >= # of leaf data items) */
  maxlocal = (minleaf > maxleaf)? 0 : maxleaf-minleaf+1; /* maxleaf=-1 and minleaf=0 when nleaves=0 */
  ierr = PetscCalloc2(nroots,&rootdata,maxlocal,&leafdata);CHKERRQ(ierr);
  /* Tag selected roots */
  for (i=0; i<nselected; ++i) rootdata[selected[i]] = 1;

  /* Bcast from roots to leaves to tag connected leaves. We reuse the established bcast communication in
     sf but do not do unpacking (from leaf buffer to leafdata). The raw data in leaf buffer is what we are
     interested in since it tells which leaves are connected to which ranks.
   */
  ierr = PetscSFBcastAndOpBegin_Basic(sf,MPIU_INT,PETSC_MEMTYPE_HOST,rootdata,PETSC_MEMTYPE_HOST,leafdata-minleaf,MPIU_REPLACE);CHKERRQ(ierr); /* Need to give leafdata but we won't use it */
  ierr = PetscSFPackGetInUse(sf,MPIU_INT,rootdata,leafdata-minleaf,PETSC_OWN_POINTER,&link);CHKERRQ(ierr);
  ierr = PetscSFPackWaitall(link,PETSCSF_ROOT2LEAF_BCAST);CHKERRQ(ierr);
  ierr = PetscSFGetLeafInfo_Basic(sf,&nranks,&ndranks,&ranks,&roffset,&rmine,&rremote);CHKERRQ(ierr); /* Get send info */
  esf_nranks = esf_ndranks = connected_leaves = 0;
  for (i=0; i<nranks; i++) {
    connected = PETSC_FALSE; /* Is the current process still connected to this remote root rank? */
    buffer    = i < ndranks? (PetscInt*)link->selfbuf[PETSC_MEMTYPE_HOST] : (PetscInt*)link->leafbuf[PETSC_MEMTYPE_HOST] + (roffset[i] - roffset[ndranks]);
    count     = roffset[i+1] - roffset[i];
    for (j=0; j<count; j++) {if (buffer[j]) {connected_leaves++; connected = PETSC_TRUE;}}
    if (connected) {esf_nranks++; if (i < ndranks) esf_ndranks++;}
  }

  /* Set graph of esf and also set up its outgoing communication (i.e., send info), which is usually done by PetscSFSetUpRanks */
  ierr = PetscMalloc1(connected_leaves,&new_ilocal);CHKERRQ(ierr);
  ierr = PetscMalloc1(connected_leaves,&new_iremote);CHKERRQ(ierr);
  ierr = PetscMalloc4(esf_nranks,&esf_ranks,esf_nranks+1,&esf_roffset,connected_leaves,&esf_rmine,connected_leaves,&esf_rremote);CHKERRQ(ierr);
  p    = 0; /* Counter for connected root ranks */
  q    = 0; /* Counter for connected leaves */
  esf_roffset[0] = 0;
  for (i=0; i<nranks; i++) { /* Scan leaf data again to fill esf arrays */
    buffer    = i < ndranks? (PetscInt*)link->selfbuf[PETSC_MEMTYPE_HOST] : (PetscInt*)link->leafbuf[PETSC_MEMTYPE_HOST] + (roffset[i] - roffset[ndranks]);
    connected = PETSC_FALSE;
    for (j=roffset[i],k=0; j<roffset[i+1]; j++,k++) {
      if (buffer[k]) {
        esf_rmine[q]         = new_ilocal[q] = rmine[j];
        esf_rremote[q]       = rremote[j];
        new_iremote[q].index = rremote[j];
        new_iremote[q].rank  = ranks[i];
        connected            = PETSC_TRUE;
        q++;
      }
    }
    if (connected) {
      esf_ranks[p]     = ranks[i];
      esf_roffset[p+1] = q;
      p++;
    }
  }

  ierr = PetscSFPackReclaim(sf,&link);CHKERRQ(ierr);

  /* SetGraph internally resets the SF, so we only set its fields after the call */
  ierr         = PetscSFSetGraph(esf,nroots,connected_leaves,new_ilocal,PETSC_OWN_POINTER,new_iremote,PETSC_OWN_POINTER);CHKERRQ(ierr);
  esf->nranks  = esf_nranks;
  esf->ndranks = esf_ndranks;
  esf->ranks   = esf_ranks;
  esf->roffset = esf_roffset;
  esf->rmine   = esf_rmine;
  esf->rremote = esf_rremote;

  /* Set up the incoming communication (i.e., recv info) stored in esf->data, which is usually done by PetscSFSetUp_Basic */
  bas  = (PetscSF_Basic*)esf->data;
  ierr = PetscSFGetRootInfo_Basic(sf,&niranks,&ndiranks,&iranks,&ioffset,&irootloc);CHKERRQ(ierr); /* Get recv info */
  /* Embedded sf always has simpler communication than the original one. We might allocate longer arrays than needed here. But we
     expect these arrays are usually short, so we do not care. The benefit is we can fill these arrays by just parsing irootloc once.
   */
  ierr = PetscMalloc2(niranks,&bas->iranks,niranks+1,&bas->ioffset);CHKERRQ(ierr);
  ierr = PetscMalloc1(ioffset[niranks],&bas->irootloc);CHKERRQ(ierr);
  bas->niranks = bas->ndiranks = bas->ioffset[0] = 0;
  p = 0; /* Counter for connected leaf ranks */
  q = 0; /* Counter for connected roots */
  for (i=0; i<niranks; i++) {
    connected = PETSC_FALSE; /* Is the current process still connected to this remote leaf rank? */
    for (j=ioffset[i]; j<ioffset[i+1]; j++) {
      PetscInt loc;
      ierr = PetscFindInt(irootloc[j],nselected,selected,&loc);CHKERRQ(ierr);
      if (loc >= 0) { /* Found in selected this root is connected */
        bas->irootloc[q++] = irootloc[j];
        connected = PETSC_TRUE;
      }
    }
    if (connected) {
      bas->niranks++;
      if (i<ndiranks) bas->ndiranks++; /* Note that order of ranks (including distinguished ranks) is kept */
      bas->iranks[p]    = iranks[i];
      bas->ioffset[p+1] = q;
      p++;
    }
  }
  bas->itotal = q;

  /* Setup packing optimizations */
  ierr = PetscSFPackSetupOptimizations_Basic(esf);CHKERRQ(ierr);
  esf->setupcalled = PETSC_TRUE; /* We have done setup ourselves! */

  ierr = PetscFree2(rootdata,leafdata);CHKERRQ(ierr);
  *newsf = esf;
  PetscFunctionReturn(0);
}

PETSC_INTERN PetscErrorCode PetscSFCreateEmbeddedLeafSF_Basic(PetscSF sf,PetscInt nselected,const PetscInt *selected,PetscSF *newsf)
{
  PetscSF           esf;
  PetscInt          i,j,k,p,q,nroots,*rootdata,*leafdata,*new_ilocal,niranks,ndiranks,minleaf,maxleaf,maxlocal;
  const PetscInt    *ilocal,*ioffset,*irootloc,*buffer;
  const PetscMPIInt *iranks;
  PetscSFPack       link;
  PetscSFNode       *new_iremote;
  const PetscSFNode *iremote;
  PetscSF_Basic     *bas;
  MPI_Group         group;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  ierr = PetscSFCreate(PetscObjectComm((PetscObject)sf),&esf);CHKERRQ(ierr);
  ierr = PetscSFSetType(esf,PETSCSFBASIC);CHKERRQ(ierr); /* This optimized routine can only create a basic sf */

  /* Set the graph of esf, which is easy for CreateEmbeddedLeafSF */
  ierr = PetscSFGetGraph(sf,&nroots,NULL,&ilocal,&iremote);CHKERRQ(ierr);
  ierr = PetscSFGetLeafRange(sf,&minleaf,&maxleaf);CHKERRQ(ierr);
  ierr = PetscMalloc1(nselected,&new_ilocal);CHKERRQ(ierr);
  ierr = PetscMalloc1(nselected,&new_iremote);CHKERRQ(ierr);
  for (i=0; i<nselected; i++) {
    const PetscInt l     = selected[i];
    new_ilocal[i]        = ilocal ? ilocal[l] : l;
    new_iremote[i].rank  = iremote[l].rank;
    new_iremote[i].index = iremote[l].index;
  }

  /* Tag selected leaves before PetscSFSetGraph since new_ilocal might turn into NULL since we use PETSC_OWN_POINTER below */
  maxlocal = (minleaf > maxleaf)? 0 : maxleaf-minleaf+1; /* maxleaf=-1 and minleaf=0 when nleaves=0 */
  ierr = PetscCalloc2(nroots,&rootdata,maxlocal,&leafdata);CHKERRQ(ierr);
  for (i=0; i<nselected; i++) leafdata[new_ilocal[i]-minleaf] = 1; /* -minleaf to adjust indices according to minleaf */

  ierr = PetscSFSetGraph(esf,nroots,nselected,new_ilocal,PETSC_OWN_POINTER,new_iremote,PETSC_OWN_POINTER);CHKERRQ(ierr);

  /* Set up the outgoing communication (i.e., send info). We can not reuse rmine etc in sf since there is no way to
     map rmine[i] (ilocal of leaves) back to selected[j]  (leaf indices).
   */
  ierr = MPI_Comm_group(PETSC_COMM_SELF,&group);CHKERRQ(ierr);
  ierr = PetscSFSetUpRanks(esf,group);CHKERRQ(ierr);
  ierr = MPI_Group_free(&group);CHKERRQ(ierr);

  /* Set up the incoming communication (i.e., recv info) */
  ierr = PetscSFGetRootInfo_Basic(sf,&niranks,&ndiranks,&iranks,&ioffset,&irootloc);CHKERRQ(ierr);
  bas  = (PetscSF_Basic*)esf->data;
  ierr = PetscMalloc2(niranks,&bas->iranks,niranks+1,&bas->ioffset);CHKERRQ(ierr);
  ierr = PetscMalloc1(ioffset[niranks],&bas->irootloc);CHKERRQ(ierr);

  /* Pass info about selected leaves to root buffer */
  ierr = PetscSFReduceBegin_Basic(sf,MPIU_INT,PETSC_MEMTYPE_HOST,leafdata-minleaf,PETSC_MEMTYPE_HOST,rootdata,MPIU_REPLACE);CHKERRQ(ierr); /* -minleaf to re-adjust start address of leafdata */
  ierr = PetscSFPackGetInUse(sf,MPIU_INT,rootdata,leafdata-minleaf,PETSC_OWN_POINTER,&link);CHKERRQ(ierr);
  ierr = PetscSFPackWaitall(link,PETSCSF_LEAF2ROOT_REDUCE);CHKERRQ(ierr);

  bas->niranks = bas->ndiranks = bas->ioffset[0] = 0;
  p = 0; /* Counter for connected leaf ranks */
  q = 0; /* Counter for connected roots */
  for (i=0; i<niranks; i++) {
    PetscBool connected = PETSC_FALSE; /* Is the current process still connected to this remote leaf rank? */
    buffer = i < ndiranks? (PetscInt*)link->selfbuf[PETSC_MEMTYPE_HOST] : (PetscInt*)link->rootbuf[PETSC_MEMTYPE_HOST] + (ioffset[i] - ioffset[ndiranks]);
    for (j=ioffset[i],k=0; j<ioffset[i+1]; j++,k++) {
      if (buffer[k]) {bas->irootloc[q++] = irootloc[j]; connected = PETSC_TRUE;}
    }
    if (connected) {
      bas->niranks++;
      if (i<ndiranks) bas->ndiranks++;
      bas->iranks[p]    = iranks[i];
      bas->ioffset[p+1] = q;
      p++;
    }
  }
  bas->itotal = q;
  ierr = PetscSFPackReclaim(sf,&link);CHKERRQ(ierr);

  /* Setup packing optimizations */
  ierr = PetscSFPackSetupOptimizations_Basic(esf);CHKERRQ(ierr);
  esf->setupcalled = PETSC_TRUE; /* We have done setup ourselves! */

  ierr = PetscFree2(rootdata,leafdata);CHKERRQ(ierr);
  *newsf = esf;
  PetscFunctionReturn(0);
}

PETSC_EXTERN PetscErrorCode PetscSFCreate_Basic(PetscSF sf)
{
  PetscSF_Basic  *dat;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  sf->ops->SetUp                = PetscSFSetUp_Basic;
  sf->ops->SetFromOptions       = PetscSFSetFromOptions_Basic;
  sf->ops->Reset                = PetscSFReset_Basic;
  sf->ops->Destroy              = PetscSFDestroy_Basic;
  sf->ops->View                 = PetscSFView_Basic;
  sf->ops->BcastAndOpBegin      = PetscSFBcastAndOpBegin_Basic;
  sf->ops->BcastAndOpEnd        = PetscSFBcastAndOpEnd_Basic;
  sf->ops->ReduceBegin          = PetscSFReduceBegin_Basic;
  sf->ops->ReduceEnd            = PetscSFReduceEnd_Basic;
  sf->ops->FetchAndOpBegin      = PetscSFFetchAndOpBegin_Basic;
  sf->ops->FetchAndOpEnd        = PetscSFFetchAndOpEnd_Basic;
  sf->ops->GetLeafRanks         = PetscSFGetLeafRanks_Basic;
  sf->ops->CreateEmbeddedSF     = PetscSFCreateEmbeddedSF_Basic;
  sf->ops->CreateEmbeddedLeafSF = PetscSFCreateEmbeddedLeafSF_Basic;

  ierr = PetscNewLog(sf,&dat);CHKERRQ(ierr);
  sf->data = (void*)dat;
  PetscFunctionReturn(0);
}
