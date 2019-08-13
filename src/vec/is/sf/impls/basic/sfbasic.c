
#include <../src/vec/is/sf/impls/basic/sfbasic.h>

/*===================================================================================*/
/*              Internal routines for PetscSFPack_Basic                              */
/*===================================================================================*/

/* Return root and leaf MPI requests for communication in the given direction. If the requests have not been
   initialized (since we use persistent requests), then initialize them.
*/
static PetscErrorCode PetscSFPackGetReqs_Basic(PetscSF sf,PetscSFPack_Basic link,PetscSFDirection direction,MPI_Request **rootreqs,MPI_Request **leafreqs)
{
  PetscErrorCode ierr;
  PetscSF_Basic  *bas = (PetscSF_Basic*)sf->data;
  MPI_Request    *requests = (direction == PETSCSF_LEAF2ROOT_REDUCE)? link->requests : link->requests + link->half;
  PetscInt       i,j,nrootranks,ndrootranks,nleafranks,ndleafranks,disp;
  const PetscInt *rootoffset,*leafoffset;
  PetscMPIInt    n;
  MPI_Comm       comm = PetscObjectComm((PetscObject)sf);
  MPI_Datatype   unit;

  PetscFunctionBegin;
  if (!link->initialized[direction]) {
    ierr = PetscSFGetRootInfo_Basic(sf,&nrootranks,&ndrootranks,NULL,&rootoffset,NULL);CHKERRQ(ierr);
    ierr = PetscSFGetLeafInfo_Basic(sf,&nleafranks,&ndleafranks,NULL,&leafoffset,NULL,NULL);CHKERRQ(ierr);
    unit = link->unit;
    j    = 0; /* request counter */
    if (direction == PETSCSF_LEAF2ROOT_REDUCE) {
      for (i=ndrootranks; i<nrootranks; i++) {
        disp = (rootoffset[i] - rootoffset[ndrootranks])*link->unitbytes;
        ierr = PetscMPIIntCast(rootoffset[i+1]-rootoffset[i],&n);CHKERRQ(ierr);
        ierr = MPI_Recv_init(link->rootbuf+disp,n,unit,bas->iranks[i],link->tag,comm,&requests[j++]);CHKERRQ(ierr);
      }
      for (i=ndleafranks; i<nleafranks; i++) {
        disp = (leafoffset[i] - leafoffset[ndleafranks])*link->unitbytes;
        ierr = PetscMPIIntCast(leafoffset[i+1]-leafoffset[i],&n);CHKERRQ(ierr);
        ierr = MPI_Send_init(link->leafbuf+disp,n,unit,sf->ranks[i],link->tag,comm,&requests[j++]);CHKERRQ(ierr);
      }
    } else if (direction == PETSCSF_ROOT2LEAF_BCAST) {
      for (i=ndrootranks; i<nrootranks; i++) {
        disp = (rootoffset[i] - rootoffset[ndrootranks])*link->unitbytes;
        ierr = PetscMPIIntCast(rootoffset[i+1]-rootoffset[i],&n);CHKERRQ(ierr);
        ierr = MPI_Send_init(link->rootbuf+disp,n,unit,bas->iranks[i],link->tag,comm,&requests[j++]);CHKERRQ(ierr);
      }
      for (i=ndleafranks; i<nleafranks; i++) {
        disp = (leafoffset[i] - leafoffset[ndleafranks])*link->unitbytes;
        ierr = PetscMPIIntCast(leafoffset[i+1]-leafoffset[i],&n);CHKERRQ(ierr);
        ierr = MPI_Recv_init(link->leafbuf+disp,n,unit,sf->ranks[i],link->tag,comm,&requests[j++]);CHKERRQ(ierr);
      }
    } else SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Out-of-range PetscSFDirection = %D\n",direction);

    link->initialized[direction] = PETSC_TRUE;
  }

  if (rootreqs) *rootreqs = requests;
  if (leafreqs) *leafreqs = requests + (bas->niranks - bas->ndiranks);
  PetscFunctionReturn(0);
}

/* Common part shared by SFBasic and SFNeighbor based on the fact they all deal with sparse graphs. */
PETSC_INTERN PetscErrorCode PetscSFPackGet_Basic_Common(PetscSF sf,MPI_Datatype unit,const void *rootdata,const void *leafdata,PetscInt half,PetscSFPack_Basic *mylink)
{
  PetscErrorCode    ierr;
  PetscSF_Basic     *bas = (PetscSF_Basic*)sf->data;
  PetscInt          i,nrootranks,ndrootranks,nleafranks,ndleafranks,rbufsize,lbufsize,sbufsize;
  const PetscInt    *rootoffset,*leafoffset;
  PetscSFPack       *p;
  PetscSFPack_Basic link;
  PetscBool         match;

  PetscFunctionBegin;
  ierr = PetscSFPackSetErrorOnUnsupportedOverlap(sf,unit,rootdata,leafdata);CHKERRQ(ierr);

  /* Look for types in cache */
  for (p=&bas->avail; (link=(PetscSFPack_Basic)*p); p=&link->next) {
    ierr = MPIPetsc_Type_compare(unit,link->unit,&match);CHKERRQ(ierr);
    if (match) {
      *p = link->next; /* Remove from available list */
      goto found;
    }
  }

  ierr = PetscSFGetRootInfo_Basic(sf,&nrootranks,&ndrootranks,NULL,&rootoffset,NULL);CHKERRQ(ierr);
  ierr = PetscSFGetLeafInfo_Basic(sf,&nleafranks,&ndleafranks,NULL,&leafoffset,NULL,NULL);CHKERRQ(ierr);
  ierr = PetscNew(&link);CHKERRQ(ierr);
  ierr = PetscSFPackSetupType((PetscSFPack)link,unit);CHKERRQ(ierr);
  ierr = PetscCommGetNewTag(PetscObjectComm((PetscObject)sf),&link->tag);CHKERRQ(ierr); /* One tag per link */

  /* Allocate root, leaf, self buffers, and MPI requests. We double the requests since there are leaf2root and root2leaf communications. */
  rbufsize   = (rootoffset[nrootranks]-rootoffset[ndrootranks])*link->unitbytes;
  lbufsize   = (leafoffset[nleafranks]-leafoffset[ndleafranks])*link->unitbytes;
  sbufsize   = rootoffset[ndrootranks]*link->unitbytes;
  link->half = half;
  ierr       = PetscMalloc4(rbufsize,&link->rootbuf,lbufsize,&link->leafbuf,sbufsize,&link->selfbuf,link->half*2,&link->requests);CHKERRQ(ierr);
  for (i=0; i<link->half*2; i++) link->requests[i] = MPI_REQUEST_NULL; /* Initialized to NULL so that we know which need to be freed in Destroy */

found:
  link->rkey = rootdata;
  link->lkey = leafdata;
  link->next = bas->inuse;
  bas->inuse = (PetscSFPack)link;

  *mylink    = link;
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscSFPackGet_Basic(PetscSF sf,MPI_Datatype unit,const void *rootdata,const void *leafdata,PetscSFDirection direction,PetscSFPack_Basic *mylink)
{
  PetscErrorCode    ierr;
  PetscInt          nrootranks,ndrootranks,nleafranks,ndleafranks,half;

  PetscFunctionBegin;
  ierr = PetscSFGetRootInfo_Basic(sf,&nrootranks,&ndrootranks,NULL,NULL,NULL);CHKERRQ(ierr);
  ierr = PetscSFGetLeafInfo_Basic(sf,&nleafranks,&ndleafranks,NULL,NULL,NULL,NULL);CHKERRQ(ierr);
  half = (nrootranks - ndrootranks) + (nleafranks - ndleafranks);
  ierr = PetscSFPackGet_Basic_Common(sf,unit,rootdata,leafdata,half,mylink);CHKERRQ(ierr);
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

  /* Setup packing optimization for roots and leaves */
  ierr = PetscSFPackSetupOptimization_Basic(sf);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscSFSetFromOptions_Basic(PetscOptionItems *PetscOptionsObject,PetscSF sf)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscOptionsHead(PetscOptionsObject,"PetscSF Basic options");CHKERRQ(ierr);
  ierr = PetscOptionsTail();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PETSC_INTERN PetscErrorCode PetscSFReset_Basic(PetscSF sf)
{
  PetscSF_Basic     *bas = (PetscSF_Basic*)sf->data;
  PetscErrorCode    ierr;
  PetscSFPack_Basic link,next;
  PetscInt          i;

  PetscFunctionBegin;
  if (bas->inuse) SETERRQ(PetscObjectComm((PetscObject)sf),PETSC_ERR_ARG_WRONGSTATE,"Outstanding operation has not been completed");
  ierr = PetscFree2(bas->iranks,bas->ioffset);CHKERRQ(ierr);
  ierr = PetscFree(bas->irootloc);CHKERRQ(ierr);
  for (link=(PetscSFPack_Basic)bas->avail; link; link=next) {
    next = (PetscSFPack_Basic)link->next;
    if (!link->isbuiltin) {ierr = MPI_Type_free(&link->unit);CHKERRQ(ierr);}
    for (i=0; i<link->half*2; i++) { /* Persistent reqs must be freed. */
      if (link->requests[i] != MPI_REQUEST_NULL) {ierr = MPI_Request_free(&link->requests[i]);CHKERRQ(ierr);}
    }
    ierr = PetscFree4(link->rootbuf,link->leafbuf,link->selfbuf,link->requests);CHKERRQ(ierr);
    ierr = PetscFree(link);CHKERRQ(ierr);
  }
  bas->avail = NULL;
  ierr = PetscSFPackDestroyOptimization_Basic(sf);CHKERRQ(ierr);
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

static PetscErrorCode PetscSFBcastAndOpBegin_Basic(PetscSF sf,MPI_Datatype unit,const void *rootdata,void *leafdata,MPI_Op op)
{
  PetscErrorCode    ierr;
  PetscSFPack_Basic link;
  PetscInt          nrootranks,ndrootranks,nleafranks,ndleafranks,cnt;
  const PetscInt    *rootoffset,*leafoffset,*rootloc,*leafloc;
  const PetscMPIInt *rootranks,*leafranks;
  MPI_Request       *rootreqs,*leafreqs;

  PetscFunctionBegin;
  ierr = PetscSFGetRootInfo_Basic(sf,&nrootranks,&ndrootranks,&rootranks,&rootoffset,NULL);CHKERRQ(ierr);
  ierr = PetscSFGetLeafInfo_Basic(sf,&nleafranks,&ndleafranks,&leafranks,&leafoffset,&leafloc,NULL);CHKERRQ(ierr);
  ierr = PetscSFPackGet_Basic(sf,unit,rootdata,leafdata,PETSCSF_ROOT2LEAF_BCAST,&link);CHKERRQ(ierr);
  ierr = PetscSFGetRootIndicesAtPlace_Basic(sf,PETSC_FALSE,&rootloc);

  ierr = PetscSFPackGetReqs_Basic(sf,link,PETSCSF_ROOT2LEAF_BCAST,&rootreqs,&leafreqs);CHKERRQ(ierr);
  /* Post Irecv. Note distinguished ranks receive data via shared memory (i.e., not via MPI) */
  cnt  = leafoffset[nleafranks]-leafoffset[ndleafranks];
  ierr = MPI_Startall_irecv(cnt,unit,nleafranks-ndleafranks,leafreqs);CHKERRQ(ierr);

  /* Do Isend */
  ierr = PetscSFPackRootData(sf,(PetscSFPack)link,nrootranks,ndrootranks,rootoffset,rootloc,rootdata);CHKERRQ(ierr);
  cnt = rootoffset[nrootranks] - rootoffset[ndrootranks];
  ierr = MPI_Startall_isend(cnt,unit,nrootranks-ndrootranks,rootreqs);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PETSC_INTERN PetscErrorCode PetscSFBcastAndOpEnd_Basic(PetscSF sf,MPI_Datatype unit,const void *rootdata,void *leafdata,MPI_Op op)
{
  PetscErrorCode    ierr;
  PetscSFPack_Basic link;
  PetscInt          nleafranks,ndleafranks;
  const PetscInt    *leafoffset,*leafloc;

  PetscFunctionBegin;
  ierr = PetscSFPackGetInUse(sf,unit,rootdata,leafdata,PETSC_OWN_POINTER,(PetscSFPack*)&link);CHKERRQ(ierr);
  ierr = PetscSFPackWaitall_Basic(link,PETSCSF_ROOT2LEAF_BCAST);CHKERRQ(ierr);
  ierr = PetscSFGetLeafInfo_Basic(sf,&nleafranks,&ndleafranks,NULL,&leafoffset,NULL,NULL);CHKERRQ(ierr);
  ierr = PetscSFGetLeafIndicesAtPlace_Basic(sf,PETSC_FALSE,&leafloc);
  ierr = PetscSFUnpackAndOpLeafData(sf,(PetscSFPack)link,nleafranks,ndleafranks,leafoffset,leafloc,leafdata,op);CHKERRQ(ierr);
  ierr = PetscSFPackReclaim(sf,(PetscSFPack*)&link);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* leaf -> root with reduction */
static PetscErrorCode PetscSFReduceBegin_Basic(PetscSF sf,MPI_Datatype unit,const void *leafdata,void *rootdata,MPI_Op op)
{
  PetscSFPack_Basic link;
  PetscErrorCode    ierr;
  PetscInt          nrootranks,ndrootranks,nleafranks,ndleafranks,cnt;
  const PetscInt    *rootoffset,*leafoffset,*rootloc,*leafloc;
  const PetscMPIInt *rootranks,*leafranks;
  MPI_Request       *rootreqs,*leafreqs;

  PetscFunctionBegin;
  ierr = PetscSFGetRootInfo_Basic(sf,&nrootranks,&ndrootranks,&rootranks,&rootoffset,&rootloc);CHKERRQ(ierr);
  ierr = PetscSFGetLeafInfo_Basic(sf,&nleafranks,&ndleafranks,&leafranks,&leafoffset,NULL,NULL);CHKERRQ(ierr);
  ierr = PetscSFGetLeafIndicesAtPlace_Basic(sf,PETSC_FALSE,&leafloc);

  ierr = PetscSFPackGet_Basic(sf,unit,rootdata,leafdata,PETSCSF_LEAF2ROOT_REDUCE,&link);CHKERRQ(ierr);
  ierr = PetscSFPackGetReqs_Basic(sf,link,PETSCSF_LEAF2ROOT_REDUCE,&rootreqs,&leafreqs);CHKERRQ(ierr);
  /* Eagerly post root receives for non-distinguished ranks */
  cnt  = rootoffset[nrootranks]-rootoffset[ndrootranks];
  ierr = MPI_Startall_irecv(cnt,unit,nrootranks-ndrootranks,rootreqs);CHKERRQ(ierr);

  /* Pack and send leaf data */
  ierr = PetscSFPackLeafData(sf,(PetscSFPack)link,nleafranks,ndleafranks,leafoffset,leafloc,leafdata);CHKERRQ(ierr);
  cnt  = leafoffset[nleafranks] - leafoffset[ndleafranks];
  ierr = MPI_Startall_isend(cnt,unit,nleafranks-ndleafranks,leafreqs);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PETSC_INTERN PetscErrorCode PetscSFReduceEnd_Basic(PetscSF sf,MPI_Datatype unit,const void *leafdata,void *rootdata,MPI_Op op)
{
  PetscErrorCode    ierr;
  PetscSFPack_Basic link;
  PetscInt          nrootranks,ndrootranks;
  const PetscInt    *rootoffset,*rootloc;

  PetscFunctionBegin;
  ierr = PetscSFGetRootInfo_Basic(sf,&nrootranks,&ndrootranks,NULL,&rootoffset,NULL);CHKERRQ(ierr);
  ierr = PetscSFGetRootIndicesAtPlace_Basic(sf,PETSC_FALSE,&rootloc);
  ierr = PetscSFPackGetInUse(sf,unit,rootdata,leafdata,PETSC_OWN_POINTER,(PetscSFPack*)&link);CHKERRQ(ierr);
  ierr = PetscSFPackWaitall_Basic(link,PETSCSF_LEAF2ROOT_REDUCE);CHKERRQ(ierr);
  ierr = PetscSFUnpackAndOpRootData(sf,(PetscSFPack)link,nrootranks,ndrootranks,rootoffset,rootloc,rootdata,op);CHKERRQ(ierr);
  ierr = PetscSFPackReclaim(sf,(PetscSFPack*)&link);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PETSC_INTERN PetscErrorCode PetscSFFetchAndOpBegin_Basic(PetscSF sf,MPI_Datatype unit,void *rootdata,const void *leafdata,void *leafupdate,MPI_Op op)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscSFReduceBegin(sf,unit,leafdata,rootdata,op);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscSFFetchAndOpEnd_Basic(PetscSF sf,MPI_Datatype unit,void *rootdata,const void *leafdata,void *leafupdate,MPI_Op op)
{
  PetscErrorCode    ierr;
  PetscSFPack_Basic link;
  PetscInt          nrootranks,ndrootranks,nleafranks,ndleafranks,cnt;
  const PetscInt    *rootoffset,*leafoffset,*rootloc,*leafloc;
  const PetscMPIInt *rootranks,*leafranks;
  MPI_Request       *rootreqs,*leafreqs;

  PetscFunctionBegin;
  ierr = PetscSFGetRootInfo_Basic(sf,&nrootranks,&ndrootranks,&rootranks,&rootoffset,NULL);CHKERRQ(ierr);
  ierr = PetscSFGetLeafInfo_Basic(sf,&nleafranks,&ndleafranks,&leafranks,&leafoffset,NULL,NULL);CHKERRQ(ierr);
  ierr = PetscSFGetRootIndicesAtPlace_Basic(sf,PETSC_FALSE,&rootloc);CHKERRQ(ierr);
  ierr = PetscSFGetLeafIndicesAtPlace_Basic(sf,PETSC_FALSE,&leafloc);CHKERRQ(ierr);
  ierr = PetscSFPackGetInUse(sf,unit,rootdata,leafdata,PETSC_OWN_POINTER,(PetscSFPack*)&link);CHKERRQ(ierr);
  /* This implementation could be changed to unpack as receives arrive, at the cost of non-determinism */
  ierr = PetscSFPackWaitall_Basic(link,PETSCSF_LEAF2ROOT_REDUCE);CHKERRQ(ierr);
  ierr = PetscSFPackGetReqs_Basic(sf,link,PETSCSF_ROOT2LEAF_BCAST,&rootreqs,&leafreqs);CHKERRQ(ierr);

  /* Post leaf receives */
  cnt  = leafoffset[nleafranks]-leafoffset[ndleafranks];
  ierr = MPI_Startall_irecv(cnt,unit,nleafranks-ndleafranks,leafreqs);CHKERRQ(ierr);

  /* Process local fetch-and-op, post root sends */
  ierr = PetscSFFetchAndOpRootData(sf,(PetscSFPack)link,nrootranks,ndrootranks,rootoffset,rootloc,rootdata,op);CHKERRQ(ierr);
  cnt  = rootoffset[nrootranks] - rootoffset[ndrootranks];
  ierr = MPI_Startall_isend(cnt,unit,nrootranks-ndrootranks,rootreqs);CHKERRQ(ierr);

  /* Unpack and insert fetched data into leaves */
  ierr = PetscSFPackWaitall_Basic(link,PETSCSF_ROOT2LEAF_BCAST);CHKERRQ(ierr);
  ierr = PetscSFUnpackAndOpLeafData(sf,(PetscSFPack)link,nleafranks,ndleafranks,leafoffset,leafloc,leafupdate,MPIU_REPLACE);CHKERRQ(ierr);
  ierr = PetscSFPackReclaim(sf,(PetscSFPack*)&link);CHKERRQ(ierr);
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
  PetscSFPack_Basic link;
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
  ierr = PetscSFBcastAndOpBegin_Basic(sf,MPIU_INT,rootdata,leafdata-minleaf,MPIU_REPLACE);CHKERRQ(ierr); /* Need to give leafdata but we won't use it */
  ierr = PetscSFPackGetInUse(sf,MPIU_INT,rootdata,leafdata-minleaf,PETSC_OWN_POINTER,(PetscSFPack*)&link);CHKERRQ(ierr);
  ierr = PetscSFPackWaitall_Basic(link,PETSCSF_ROOT2LEAF_BCAST);CHKERRQ(ierr);
  ierr = PetscSFGetLeafInfo_Basic(sf,&nranks,&ndranks,&ranks,&roffset,&rmine,&rremote);CHKERRQ(ierr); /* Get send info */
  esf_nranks = esf_ndranks = connected_leaves = 0;
  for (i=0; i<nranks; i++) {
    connected = PETSC_FALSE; /* Is the current process still connected to this remote root rank? */
    buffer    = i < ndranks? (PetscInt*)link->selfbuf : (PetscInt*)link->leafbuf + (roffset[i] - roffset[ndranks]);
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
    buffer    = i < ndranks? (PetscInt*)link->selfbuf : (PetscInt*)link->leafbuf + (roffset[i] - roffset[ndranks]);
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

  ierr = PetscSFPackReclaim(sf,(PetscSFPack*)&link);CHKERRQ(ierr);

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
  ierr = PetscSFPackSetupOptimization_Basic(esf);CHKERRQ(ierr);
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
  PetscSFPack_Basic link;
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
  ierr = PetscSFReduceBegin_Basic(sf,MPIU_INT,leafdata-minleaf,rootdata,MPIU_REPLACE);CHKERRQ(ierr); /* -minleaf to re-adjust start address of leafdata */
  ierr = PetscSFPackGetInUse(sf,MPIU_INT,rootdata,leafdata-minleaf,PETSC_OWN_POINTER,(PetscSFPack*)&link);CHKERRQ(ierr);
  ierr = PetscSFPackWaitall_Basic(link,PETSCSF_LEAF2ROOT_REDUCE);CHKERRQ(ierr);

  bas->niranks = bas->ndiranks = bas->ioffset[0] = 0;
  p = 0; /* Counter for connected leaf ranks */
  q = 0; /* Counter for connected roots */
  for (i=0; i<niranks; i++) {
    buffer    = i < ndiranks? (PetscInt*)link->selfbuf : (PetscInt*)link->rootbuf + (ioffset[i] - ioffset[ndiranks]);
    PetscBool connected = PETSC_FALSE; /* Is the current process still connected to this remote leaf rank? */
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
  ierr = PetscSFPackReclaim(sf,(PetscSFPack*)&link);CHKERRQ(ierr);

  /* Setup packing optimizations */
  ierr = PetscSFPackSetupOptimization_Basic(esf);CHKERRQ(ierr);
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
