#include <private/sfimpl.h>
#include <petscctable.h>

const char *const PetscSFSynchronizationTypes[] = {"FENCE","LOCK","ACTIVE","PetscSFSynchronizationType","PETSCSF_SYNCHRONIZATION_",0};

#undef __FUNCT__
#define __FUNCT__ "PetscSFCreate"
/*@C
   PetscSFCreate - create a bipartite graph communication context

   Not Collective

   Input Arguments:
.  comm - communicator on which the bipartite graph will operate

   Output Arguments:
.  bg - new bipartite graph context

   Level: intermediate

.seealso: PetscSFSetGraph(), PetscSFDestroy()
@*/
PetscErrorCode PetscSFCreate(MPI_Comm comm,PetscSF *bg)
{
  PetscErrorCode ierr;
  PetscSF        b;

  PetscFunctionBegin;
  PetscValidPointer(bg,2);
#ifndef PETSC_USE_DYNAMIC_LIBRARIES
  ierr = PetscSFInitializePackage(PETSC_NULL);CHKERRQ(ierr);
#endif

  ierr = PetscHeaderCreate(b,_p_PetscSF,struct _PetscSFOps,PETSCSF_CLASSID,-1,"PetscSF","Bipartite Graph","PetscSF",comm,PetscSFDestroy,PetscSFView);CHKERRQ(ierr);
  b->nowned    = -1;
  b->nlocal    = -1;
  b->nranks    = -1;
  b->sync      = PETSCSF_SYNCHRONIZATION_FENCE;
  b->rankorder = PETSC_TRUE;
  b->ingroup   = MPI_GROUP_NULL;
  b->outgroup  = MPI_GROUP_NULL;
  *bg = b;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscSFReset"
/*@C
   PetscSFReset - Reset a bipartite graph so that different sizes or neighbors can be used

   Collective

   Input Arguments:
.  bg - bipartite graph

   Level: advanced

.seealso: PetscSFCreate(), PetscSFSetGraph(), PetscSFDestroy()
@*/
PetscErrorCode PetscSFReset(PetscSF bg)
{
  PetscErrorCode ierr;
  PetscSFDataLink link,next;
  PetscSFWinLink  wlink,wnext;
  PetscInt i;

  PetscFunctionBegin;
  bg->mine = PETSC_NULL;
  ierr = PetscFree(bg->mine_alloc);CHKERRQ(ierr);
  bg->remote = PETSC_NULL;
  ierr = PetscFree(bg->remote_alloc);CHKERRQ(ierr);
  ierr = PetscFree4(bg->ranks,bg->roffset,bg->rmine,bg->rremote);CHKERRQ(ierr);
  ierr = PetscFree(bg->degree);CHKERRQ(ierr);
  for (link=bg->link; link; link=next) {
    next = link->next;
    ierr = MPI_Type_free(&link->unit);CHKERRQ(ierr);
    for (i=0; i<bg->nranks; i++) {
      ierr = MPI_Type_free(&link->mine[i]);CHKERRQ(ierr);
      ierr = MPI_Type_free(&link->remote[i]);CHKERRQ(ierr);
    }
    ierr = PetscFree2(link->mine,link->remote);CHKERRQ(ierr);
    ierr = PetscFree(link);CHKERRQ(ierr);
  }
  bg->link = PETSC_NULL;
  for (wlink=bg->wins; wlink; wlink=wnext) {
    wnext = wlink->next;
    if (wlink->inuse) SETERRQ1(((PetscObject)bg)->comm,PETSC_ERR_ARG_WRONGSTATE,"Window still in use with address %p",(void*)wlink->addr);
    ierr = MPI_Win_free(&wlink->win);CHKERRQ(ierr);
    ierr = PetscFree(wlink);CHKERRQ(ierr);
  }
  bg->wins = PETSC_NULL;
  if (bg->ingroup  != MPI_GROUP_NULL) {ierr = MPI_Group_free(&bg->ingroup);CHKERRQ(ierr);}
  if (bg->outgroup != MPI_GROUP_NULL) {ierr = MPI_Group_free(&bg->outgroup);CHKERRQ(ierr);}
  ierr = PetscSFDestroy(&bg->multi);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscSFDestroy"
/*@C
   PetscSFDestroy - destroy bipartite graph

   Collective

   Input Arguments:
.  bg - bipartite graph context

   Level: intermediate

.seealso: PetscSFCreate(), PetscSFReset()
@*/
PetscErrorCode PetscSFDestroy(PetscSF *bg)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (!*bg) PetscFunctionReturn(0);
  PetscValidHeaderSpecific((*bg),PETSCSF_CLASSID,1);
  if (--((PetscObject)(*bg))->refct > 0) {*bg = 0; PetscFunctionReturn(0);}
  ierr = PetscSFReset(*bg);CHKERRQ(ierr);
  ierr = PetscHeaderDestroy(bg);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscSFSetFromOptions"
/*@C
   PetscSFSetFromOptions - set PetscSF options using the options database

   Logically Collective

   Input Arguments:
.  bg - bipartite graph

   Options Database Keys:
.  -bg_synchronization - synchronization type used by PetscSF

   Level: intermediate

.keywords: KSP, set, from, options, database

.seealso: PetscSFSetSynchronizationType()
@*/
PetscErrorCode PetscSFSetFromOptions(PetscSF bg)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(bg,PETSCSF_CLASSID,1);
  ierr = PetscObjectOptionsBegin((PetscObject)bg);CHKERRQ(ierr);
  ierr = PetscOptionsEnum("-bg_synchronization","synchronization type to use for PetscSF communication","PetscSFSetSynchronizationType",PetscSFSynchronizationTypes,(PetscEnum)bg->sync,(PetscEnum*)&bg->sync,PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-bg_rank_order","sort composite points for gathers and scatters in rank order, gathers are non-deterministic otherwise","PetscSFSetRankOrder",bg->rankorder,&bg->rankorder,PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsEnd();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscSFSetSynchronizationType"
/*@C
   PetscSFSetSynchronizationType - set synchrozitaion type for PetscSF communication

   Logically Collective

   Input Arguments:
+  bg - bipartite graph for communication
-  sync - synchronization type

   Options Database Key:
.  -bg_synchronization <sync> - sets the synchronization type

   Level: intermediate

.seealso: PetscSFSetFromOptions()
@*/
PetscErrorCode PetscSFSetSynchronizationType(PetscSF bg,PetscSFSynchronizationType sync)
{

  PetscFunctionBegin;
  PetscValidHeaderSpecific(bg,PETSCSF_CLASSID,1);
  PetscValidLogicalCollectiveEnum(bg,sync,2);
  bg->sync = sync;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscSFSetRankOrder"
/*@C
   PetscSFSetRankOrder - sort multi-points for gathers and scatters by rank order

   Logically Collective

   Input Arguments:
+  bg - bipartite graph
-  flg - PETSC_TRUE to sort, PETSC_FALSE to skip sorting (lower setup cost, but non-deterministic)

   Level: advanced

.seealso: PetscSFGatherBegin(), PetscSFScatterBegin()
@*/
PetscErrorCode PetscSFSetRankOrder(PetscSF bg,PetscBool flg)
{

  PetscFunctionBegin;
  PetscValidHeaderSpecific(bg,PETSCSF_CLASSID,1);
  PetscValidLogicalCollectiveBool(bg,flg,2);
  if (bg->multi) SETERRQ(((PetscObject)bg)->comm,PETSC_ERR_ARG_WRONGSTATE,"Rank ordering must be set before first call to PetscSFGatherBegin() or PetscSFScatterBegin()");
  bg->rankorder = flg;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscSFSetGraph"
/*@C
   PetscSFSetGraph - Set a parallel bipartite graph

   Collective

   Input Arguments:
+  bg - bipartite graph
.  nowned - number of owned points (these are possible targets for remote references)
.  nlocal - number of local nodes referencing remote nodes
.  ilocal - locations of local/ghosted nodes, or PETSC_NULL for contiguous storage
.  localmode - copy mode for ilocal
.  iremote - locations of global nodes
-  remotemode - copy mode for iremote

   Level: intermediate

.seealso: PetscSFCreate(), PetscSFView()
@*/
PetscErrorCode PetscSFSetGraph(PetscSF bg,PetscInt nowned,PetscInt nlocal,const PetscInt *ilocal,PetscCopyMode localmode,const PetscSFNode *iremote,PetscCopyMode remotemode)
{
  PetscErrorCode ierr;
  PetscTable table;
  PetscTablePosition pos;
  PetscMPIInt size;
  PetscInt i,*rcount;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(bg,PETSCSF_CLASSID,1);
  if (nlocal && ilocal) PetscValidIntPointer(ilocal,4);
  if (nlocal) PetscValidPointer(iremote,6);
  ierr = PetscSFReset(bg);CHKERRQ(ierr);
  bg->nowned = nowned;
  bg->nlocal = nlocal;
  if (ilocal) {
    switch (localmode) {
    case PETSC_COPY_VALUES:
      ierr = PetscMalloc(nlocal*sizeof(*bg->mine),&bg->mine_alloc);CHKERRQ(ierr);
      bg->mine = bg->mine_alloc;
      ierr = PetscMemcpy(bg->mine,ilocal,nlocal*sizeof(*bg->mine));CHKERRQ(ierr);
      break;
    case PETSC_OWN_POINTER:
      bg->mine_alloc = (PetscInt*)ilocal;
      bg->mine = bg->mine_alloc;
      break;
    case PETSC_USE_POINTER:
      bg->mine = (PetscInt*)ilocal;
      break;
    default: SETERRQ(((PetscObject)bg)->comm,PETSC_ERR_ARG_OUTOFRANGE,"Unknown localmode");
    }
  }
  switch (remotemode) {
  case PETSC_COPY_VALUES:
    ierr = PetscMalloc(nlocal*sizeof(*bg->remote),&bg->remote_alloc);CHKERRQ(ierr);
    bg->remote = bg->remote_alloc;
    ierr = PetscMemcpy(bg->remote,iremote,nlocal*sizeof(*bg->remote));CHKERRQ(ierr);
    break;
  case PETSC_OWN_POINTER:
    bg->remote_alloc = (PetscSFNode*)iremote;
    bg->remote = bg->remote_alloc;
    break;
  case PETSC_USE_POINTER:
    bg->remote = (PetscSFNode*)iremote;
    break;
  default: SETERRQ(((PetscObject)bg)->comm,PETSC_ERR_ARG_OUTOFRANGE,"Unknown remotemode");
  }

  ierr = MPI_Comm_size(((PetscObject)bg)->comm,&size);CHKERRQ(ierr);
  ierr = PetscTableCreate(10,size,&table);CHKERRQ(ierr);
  for (i=0; i<nlocal; i++) {
    /* Log 1-based rank */
    ierr = PetscTableAdd(table,iremote[i].rank+1,1,ADD_VALUES);CHKERRQ(ierr);
  }
  ierr = PetscTableGetCount(table,&bg->nranks);CHKERRQ(ierr);
  ierr = PetscMalloc4(bg->nranks,PetscInt,&bg->ranks,bg->nranks+1,PetscInt,&bg->roffset,nlocal,PetscMPIInt,&bg->rmine,nlocal,PetscMPIInt,&bg->rremote);CHKERRQ(ierr);
  ierr = PetscMalloc(bg->nranks*sizeof(PetscInt),&rcount);CHKERRQ(ierr);
  ierr = PetscTableGetHeadPosition(table,&pos);CHKERRQ(ierr);
  for (i=0; i<bg->nranks; i++) {
    ierr = PetscTableGetNext(table,&pos,&bg->ranks[i],&rcount[i]);CHKERRQ(ierr);
    bg->ranks[i]--;             /* Convert back to 0-based */
  }
  ierr = PetscTableDestroy(&table);CHKERRQ(ierr);
  ierr = PetscSortIntWithArray(bg->nranks,bg->ranks,rcount);CHKERRQ(ierr);
  bg->roffset[0] = 0;
  for (i=0; i<bg->nranks; i++) {
    bg->roffset[i+1] = bg->roffset[i] + rcount[i];
    rcount[i] = 0;
  }
  for (i=0; i<nlocal; i++) {
    PetscInt lo,hi,irank;
    /* Search for index of iremote[i].rank in bg->ranks */
    lo = 0; hi = bg->nranks;
    while (hi - lo > 1) {
      PetscInt mid = lo + (hi - lo)/2;
      if (iremote[i].rank < bg->ranks[mid]) hi = mid;
      else                                  lo = mid;
    }
    if (hi - lo == 1 && iremote[i].rank == bg->ranks[lo]) irank = lo;
    else SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_PLIB,"Could not find rank %D in array",iremote[i].rank);
    bg->rmine[bg->roffset[irank] + rcount[irank]] = ilocal ? ilocal[i] : i;
    bg->rremote[bg->roffset[irank] + rcount[irank]] = iremote[i].index;
    rcount[irank]++;
  }
  ierr = PetscFree(rcount);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscSFView"
/*@C
   PetscSFView - view a bipartite graph

   Collective

   Input Arguments:
+  bg - bipartite graph
-  viewer - viewer to display graph, for example PETSC_VIEWER_STDOUT_WORLD

   Level: beginner

.seealso: PetscSFCreate(), PetscSFSetGraph()
@*/
PetscErrorCode PetscSFView(PetscSF bg,PetscViewer viewer)
{
  PetscErrorCode ierr;
  PetscBool iascii;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(bg,PETSCSF_CLASSID,1);
  if (!viewer) {ierr = PetscViewerASCIIGetStdout(((PetscObject)bg)->comm,&viewer);CHKERRQ(ierr);}
  PetscValidHeaderSpecific(viewer,PETSC_VIEWER_CLASSID,2);
  PetscCheckSameComm(bg,1,viewer,2);
  ierr = PetscTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&iascii);CHKERRQ(ierr);
  if (iascii) {
    PetscMPIInt rank;
    PetscInt i,j;
    ierr = PetscObjectPrintClassNamePrefixType((PetscObject)bg,viewer,"Bipartite Graph Object");CHKERRQ(ierr);
    ierr = PetscViewerASCIIPushTab(viewer);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"synchronization=%s sort=%s\n",PetscSFSynchronizationTypes[bg->sync],bg->rankorder?"rank-order":"unordered");CHKERRQ(ierr);
    ierr = MPI_Comm_rank(((PetscObject)bg)->comm,&rank);CHKERRQ(ierr);
    ierr = PetscViewerASCIISynchronizedAllow(viewer,PETSC_TRUE);CHKERRQ(ierr);
    ierr = PetscViewerASCIISynchronizedPrintf(viewer,"[%d] Number of outgoing edges=%D, remote ranks=%D\n",rank,bg->nlocal,bg->nranks);CHKERRQ(ierr);
    for (i=0; i<bg->nlocal; i++) {
      ierr = PetscViewerASCIISynchronizedPrintf(viewer,"[%d] %D -> (%D,%D)\n",rank,bg->mine?bg->mine[i]:i,bg->remote[i].rank,bg->remote[i].index);CHKERRQ(ierr);
    }
    ierr = PetscViewerFlush(viewer);CHKERRQ(ierr);
    ierr = PetscViewerASCIISynchronizedPrintf(viewer,"[%d] Outgoing edges by rank\n",rank);CHKERRQ(ierr);
    for (i=0; i<bg->nranks; i++) {
      ierr = PetscViewerASCIISynchronizedPrintf(viewer,"[%d] %D: %D edges\n",rank,bg->ranks[i],bg->roffset[i+1]-bg->roffset[i]);CHKERRQ(ierr);
      for (j=bg->roffset[i]; j<bg->roffset[i+1]; j++) {
        ierr = PetscViewerASCIISynchronizedPrintf(viewer,"[%d]    %D -> %D\n",rank,bg->rmine[j],bg->rremote[j]);CHKERRQ(ierr);
      }
    }
    ierr = PetscViewerFlush(viewer);CHKERRQ(ierr);
    ierr = PetscViewerASCIISynchronizedAllow(viewer,PETSC_FALSE);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MPIU_Type_unwrap"
static PetscErrorCode MPIU_Type_unwrap(MPI_Datatype a,MPI_Datatype *atype)
{
  PetscMPIInt nints,naddrs,ntypes,combiner;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MPI_Type_get_envelope(a,&nints,&naddrs,&ntypes,&combiner);CHKERRQ(ierr);
  if (combiner == MPI_COMBINER_DUP) {
    PetscInt ints[1];
    MPI_Aint addrs[1];
    MPI_Datatype types[1];
    if (nints != 0 || naddrs != 0 || ntypes != 1) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_LIB,"Unexpected returns from MPI_Type_get_envelope()");
    ierr = MPI_Type_get_contents(a,0,0,1,ints,addrs,types);CHKERRQ(ierr);
    *atype = types[0];
  } else {
    *atype = a;
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MPIU_Type_compare"
static PetscErrorCode MPIU_Type_compare(MPI_Datatype a,MPI_Datatype b,PetscBool *match)
{
  PetscErrorCode ierr;
  MPI_Datatype atype,btype;
  PetscMPIInt aintcount,aaddrcount,atypecount,acombiner;
  PetscMPIInt bintcount,baddrcount,btypecount,bcombiner;

  PetscFunctionBegin;
  ierr = MPIU_Type_unwrap(a,&atype);CHKERRQ(ierr);
  ierr = MPIU_Type_unwrap(b,&btype);CHKERRQ(ierr);
  *match = PETSC_FALSE;
  if (atype == btype) {
    *match = PETSC_TRUE;
    PetscFunctionReturn(0);
  }
  ierr = MPI_Type_get_envelope(atype,&aintcount,&aaddrcount,&atypecount,&acombiner);CHKERRQ(ierr);
  ierr = MPI_Type_get_envelope(btype,&bintcount,&baddrcount,&btypecount,&bcombiner);CHKERRQ(ierr);
  if (acombiner == bcombiner && aintcount == bintcount && aaddrcount == baddrcount && atypecount == btypecount) {
    PetscMPIInt  *aints,*bints;
    MPI_Aint     *aaddrs,*baddrs;
    MPI_Datatype *atypes,*btypes;
    PetscBool    same;
    ierr = PetscMalloc6(aintcount,PetscMPIInt,&aints,bintcount,PetscMPIInt,&bints,aaddrcount,MPI_Aint,&aaddrs,baddrcount,MPI_Aint,&baddrs,atypecount,MPI_Datatype,&atypes,btypecount,MPI_Datatype,&btypes);CHKERRQ(ierr);
    ierr = MPI_Type_get_contents(atype,aintcount,aaddrcount,atypecount,aints,aaddrs,atypes);CHKERRQ(ierr);
    ierr = MPI_Type_get_contents(btype,bintcount,baddrcount,btypecount,bints,baddrs,btypes);CHKERRQ(ierr);
    ierr = PetscMemcmp(aints,bints,aintcount*sizeof(aints[0]),&same);CHKERRQ(ierr);
    if (same) {
      ierr = PetscMemcmp(aaddrs,baddrs,aaddrcount*sizeof(aaddrs[0]),&same);CHKERRQ(ierr);
      if (same) {
        /* This comparison should be recursive */
        ierr = PetscMemcmp(atypes,btypes,atypecount*sizeof(atypes[0]),&same);CHKERRQ(ierr);
      }
    }
    if (same) *match = PETSC_TRUE;
    PetscFunctionReturn(0);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscSFGetDataTypes"
/*@C
   PetscSFGetDataTypes - gets composite local and remote data types for each rank

   Not Collective

   Input Arguments:
+  bg - bipartite graph
-  unit - data type for each node

   Output Arguments:
+  localtypes - types describing part of local buffer referencing each remote rank
-  remotetypes - types describing part of remote buffer referenced for each remote rank

   Level: developer

.seealso: PetscSFSetGraph(), PetscSFView()
@*/
PetscErrorCode PetscSFGetDataTypes(PetscSF bg,MPI_Datatype unit,const MPI_Datatype **localtypes,const MPI_Datatype **remotetypes)
{
  PetscErrorCode ierr;
  PetscSFDataLink link;
  PetscInt i,nranks;
  const PetscInt *ranks,*roffset;
  const PetscMPIInt *rmine,*rremote;

  PetscFunctionBegin;
  /* Look for types in cache */
  for (link=bg->link; link; link=link->next) {
    PetscBool match;
    ierr = MPIU_Type_compare(unit,link->unit,&match);CHKERRQ(ierr);
    if (match) {
      *localtypes = link->mine;
      *remotetypes = link->remote;
      PetscFunctionReturn(0);
    }
  }

  /* Create new composite types for each send rank */
  ierr = PetscSFGetRanks(bg,&nranks,&ranks,&roffset,&rmine,&rremote);CHKERRQ(ierr);
  ierr = PetscMalloc(sizeof(*link),&link);CHKERRQ(ierr);
  ierr = MPI_Type_dup(unit,&link->unit);CHKERRQ(ierr);
  ierr = PetscMalloc2(nranks,MPI_Datatype,&link->mine,nranks,MPI_Datatype,&link->remote);CHKERRQ(ierr);
  for (i=0; i<nranks; i++) {
    PetscInt rcount = roffset[i+1] - roffset[i];
    ierr = MPI_Type_create_indexed_block(rcount,1,bg->rmine+bg->roffset[i],link->unit,&link->mine[i]);CHKERRQ(ierr);
    ierr = MPI_Type_create_indexed_block(rcount,1,bg->rremote+bg->roffset[i],link->unit,&link->remote[i]);CHKERRQ(ierr);
    ierr = MPI_Type_commit(&link->mine[i]);CHKERRQ(ierr);
    ierr = MPI_Type_commit(&link->remote[i]);CHKERRQ(ierr);
  }
  link->next = bg->link;
  bg->link = link;

  *localtypes = link->mine;
  *remotetypes = link->remote;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscSFGetRanks"
/*@C
   PetscSFGetRanks - Get ranks and number of vertices referenced by local part of graph

   Not Collective

   Input Arguments:
.  bg - bipartite graph

   Output Arguments:
+  nranks - number of ranks referenced by local part
.  ranks - array of ranks
.  roffset - offset in rmine/rremote for each rank (length nranks+1)
.  rmine - concatenated array holding local indices referencing each remote rank
-  rremote - concatenated array holding remote indices referenced for each remote rank

   Level: developer

.seealso: PetscSFSetGraph(), PetscSFGetDataTypes()
@*/
PetscErrorCode PetscSFGetRanks(PetscSF bg,PetscInt *nranks,const PetscInt **ranks,const PetscInt **roffset,const PetscMPIInt **rmine,const PetscMPIInt **rremote)
{

  PetscFunctionBegin;
  PetscValidHeaderSpecific(bg,PETSCSF_CLASSID,1);
  if (nranks)  *nranks  = bg->nranks;
  if (ranks)   *ranks   = bg->ranks;
  if (roffset) *roffset = bg->roffset;
  if (rmine)   *rmine   = bg->rmine;
  if (rremote) *rremote = bg->rremote;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscSFGetWindow"
/*@C
   PetscSFGetWindow - Get a window for use with a given data type

   Collective on PetscSF

   Input Arguments:
+  bg - bipartite graph
.  unit - data type
.  array - array to be sent
.  epoch - PETSC_TRUE to acquire the window and start an epoch, PETSC_FALSE to just acquire the window
.  fenceassert - assert parameter for call to MPI_Win_fence(), if PETSCSF_SYNCHRONIZATION_FENCE
.  postassert - assert parameter for call to MPI_Win_post(), if PETSCSF_SYNCHRONIZATION_ACTIVE
-  startassert - assert parameter for call to MPI_Win_start(), if PETSCSF_SYNCHRONIZATION_ACTIVE

   Output Arguments:
.  win - window

   Level: developer

   Developer Notes:
   This currently always creates a new window. This is more synchronous than necessary. An alternative is to try to
   reuse an existing window created with the same array. Another alternative is to maintain a cache of windows and reuse
   whichever one is available, by copying the array into it if necessary.

.seealso: PetscSFGetRanks(), PetscSFGetDataTypes()
@*/
PetscErrorCode PetscSFGetWindow(PetscSF bg,MPI_Datatype unit,void *array,PetscBool epoch,PetscMPIInt fenceassert,PetscMPIInt postassert,PetscMPIInt startassert,MPI_Win *win)
{
  PetscErrorCode ierr;
  MPI_Aint lb,lb_true,bytes,bytes_true;
  PetscSFWinLink link;

  PetscFunctionBegin;
  ierr = MPI_Type_get_extent(unit,&lb,&bytes);CHKERRQ(ierr);
  ierr = MPI_Type_get_true_extent(unit,&lb_true,&bytes_true);CHKERRQ(ierr);
  if (lb != 0 || lb_true != 0) SETERRQ(((PetscObject)bg)->comm,PETSC_ERR_SUP,"No support for unit type with nonzero lower bound, write petsc-maint@mcs.anl.gov if you want this feature");
  if (bytes != bytes_true) SETERRQ(((PetscObject)bg)->comm,PETSC_ERR_SUP,"No support for unit type with modified extent, write petsc-maint@mcs.anl.gov if you want this feature");
  ierr = PetscMalloc(sizeof(*link),&link);CHKERRQ(ierr);
  link->bytes = bytes;
  link->addr  = array;
  ierr = MPI_Win_create(array,(MPI_Aint)bytes*bg->nowned,(PetscMPIInt)bytes,MPI_INFO_NULL,((PetscObject)bg)->comm,&link->win);CHKERRQ(ierr);
  link->epoch = epoch;
  link->next = bg->wins;
  link->inuse = PETSC_TRUE;
  bg->wins = link;
  *win = link->win;

  if (epoch) {
    switch (bg->sync) {
    case PETSCSF_SYNCHRONIZATION_FENCE:
      ierr = MPI_Win_fence(fenceassert,*win);CHKERRQ(ierr);
      break;
    case PETSCSF_SYNCHRONIZATION_LOCK: /* Handled outside */
      break;
    case PETSCSF_SYNCHRONIZATION_ACTIVE: {
      MPI_Group ingroup,outgroup;
      ierr = PetscSFGetGroups(bg,&ingroup,&outgroup);CHKERRQ(ierr);
      ierr = MPI_Win_post(ingroup,postassert,*win);CHKERRQ(ierr);
      ierr = MPI_Win_start(outgroup,startassert,*win);CHKERRQ(ierr);
    } break;
    default: SETERRQ(((PetscObject)bg)->comm,PETSC_ERR_PLIB,"Unknown synchronization type");
    }
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscSFFindWindow"
/*@C
   PetscSFFindWindow - Finds a window that is already in use

   Not Collective

   Input Arguments:
+  bg - bipartite graph
.  unit - data type
-  array - array with which the window is associated

   Output Arguments:
.  win - window

   Level: developer

.seealso: PetscSFGetWindow(), PetscSFRestoreWindow()
@*/
PetscErrorCode PetscSFFindWindow(PetscSF bg,MPI_Datatype unit,const void *array,MPI_Win *win)
{
  PetscSFWinLink link;

  PetscFunctionBegin;
  for (link=bg->wins; link; link=link->next) {
    if (array == link->addr) {
      *win = link->win;
      PetscFunctionReturn(0);
    }
  }
  SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_INCOMP,"Requested window not in use");
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscSFRestoreWindow"
/*@C
   PetscSFRestoreWindow - Restores a window obtained with PetscSFGetWindow()

   Collective

   Input Arguments:
+  bg - bipartite graph
.  unit - data type
.  array - array associated with window
.  epoch - close an epoch, must match argument to PetscSFGetWindow()
-  win - window

   Level: developer

.seealso: PetscSFFindWindow()
@*/
PetscErrorCode PetscSFRestoreWindow(PetscSF bg,MPI_Datatype unit,const void *array,PetscBool epoch,PetscMPIInt fenceassert,MPI_Win *win)
{
  PetscErrorCode ierr;
  PetscSFWinLink *p,link;

  PetscFunctionBegin;
  for (p=&bg->wins; *p; p=&(*p)->next) {
    link = *p;
    if (*win == link->win) {
      if (array != link->addr) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_INCOMP,"Matched window, but not array");
      if (epoch != link->epoch) {
        if (epoch) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_INCOMP,"No epoch to end");
        else SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_INCOMP,"Restoring window without ending epoch");
      }
      *p = link->next;
      goto found;
    }
  }
  SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_INCOMP,"Requested window not in use");

  found:
  if (epoch) {
    switch (bg->sync) {
    case PETSCSF_SYNCHRONIZATION_FENCE:
      ierr = MPI_Win_fence(fenceassert,*win);CHKERRQ(ierr);
      break;
    case PETSCSF_SYNCHRONIZATION_LOCK:
      break;                    /* handled outside */
    case PETSCSF_SYNCHRONIZATION_ACTIVE: {
      ierr = MPI_Win_complete(*win);CHKERRQ(ierr);
      ierr = MPI_Win_wait(*win);CHKERRQ(ierr);
    } break;
    default: SETERRQ(((PetscObject)bg)->comm,PETSC_ERR_PLIB,"Unknown synchronization type");
    }
  }

  ierr = MPI_Win_free(&link->win);CHKERRQ(ierr);
  ierr = PetscFree(link);CHKERRQ(ierr);
  *win = MPI_WIN_NULL;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscSFGetGroups"
/*@C
   PetscSFGetGroups - gets incoming and outgoing process groups

   Collective

   Input Argument:
.  bg - bipartite graph

   Output Arguments:
+  incoming - group of origin processes for incoming edges
-  outgoing - group of destination processes for outgoing edges

   Level: developer

.seealso: PetscSFGetWindow(), PetscSFRestoreWindow()
@*/
PetscErrorCode PetscSFGetGroups(PetscSF bg,MPI_Group *incoming,MPI_Group *outgoing)
{
  PetscErrorCode ierr;
  MPI_Group group;

  PetscFunctionBegin;
  if (bg->ingroup == MPI_GROUP_NULL) {
    PetscInt    i,*outranks,*inranks;
    const PetscInt *indegree;
    PetscMPIInt rank;
    PetscSFNode *remote;
    PetscSF     bgcount;

    /* Compute the number of incoming ranks */
    ierr = PetscMalloc(bg->nranks*sizeof(PetscSFNode),&remote);CHKERRQ(ierr);
    for (i=0; i<bg->nranks; i++) {
      remote[i].rank = bg->ranks[i];
      remote[i].index = 0;
    }
    ierr = PetscSFCreate(((PetscObject)bg)->comm,&bgcount);CHKERRQ(ierr);
    ierr = PetscSFSetSynchronizationType(bgcount,PETSCSF_SYNCHRONIZATION_LOCK);CHKERRQ(ierr); /* or FENCE, ACTIVE here would cause recursion */
    ierr = PetscSFSetGraph(bgcount,1,bg->nranks,PETSC_NULL,PETSC_COPY_VALUES,remote,PETSC_OWN_POINTER);CHKERRQ(ierr);
    ierr = PetscSFComputeDegreeBegin(bgcount,&indegree);CHKERRQ(ierr);
    ierr = PetscSFComputeDegreeEnd(bgcount,&indegree);CHKERRQ(ierr);

    /* Enumerate the incoming ranks */
    ierr = PetscMalloc2(indegree[0],PetscInt,&inranks,bg->nranks,PetscInt,&outranks);CHKERRQ(ierr);
    ierr = MPI_Comm_rank(((PetscObject)bg)->comm,&rank);CHKERRQ(ierr);
    for (i=0; i<bg->nranks; i++) outranks[i] = rank;
    ierr = PetscSFGatherBegin(bgcount,MPIU_INT,outranks,inranks);CHKERRQ(ierr);
    ierr = PetscSFGatherEnd(bgcount,MPIU_INT,outranks,inranks);CHKERRQ(ierr);
    ierr = MPI_Comm_group(((PetscObject)bg)->comm,&group);CHKERRQ(ierr);
    ierr = MPI_Group_incl(group,indegree[0],inranks,&bg->ingroup);CHKERRQ(ierr);
    ierr = PetscFree2(inranks,outranks);CHKERRQ(ierr);
    ierr = PetscSFDestroy(&bgcount);CHKERRQ(ierr);
  }
  *incoming = bg->ingroup;

  if (bg->outgroup == MPI_GROUP_NULL) {
    ierr = MPI_Comm_group(((PetscObject)bg)->comm,&group);CHKERRQ(ierr);
    ierr = MPI_Group_incl(group,bg->nranks,bg->ranks,&bg->outgroup);CHKERRQ(ierr);
  }
  *outgoing = bg->outgroup;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscSFGetMultiSF"
/*@C
   PetscSFGetMultiSF - gets the inner SF implemeting gathers and scatters

   Collective

   Input Argument:
.  bg - bipartite graph with possible redundancy

   Output Arguments:
.  multi - bipartite graph with incoming 

   Level: developer

   Notes:

   In most cases, users should use PetscSFGatherBegin() and PetscSFScatterBegin() instead of manipulating multi
   directly. Since multi satisfies the stronger condition that each entry in the global space has exactly one incoming
   edge, it is a candidate for future optimization that might involve its removal.

.seealso: PetscSFSetGraph(), PetscSFGatherBegin(), PetscSFScatterBegin()
@*/
PetscErrorCode PetscSFGetMultiSF(PetscSF bg,PetscSF *multi)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(bg,PETSCSF_CLASSID,1);
  PetscValidPointer(multi,2);
  if (!bg->multi) {
    const PetscInt *indegree;
    PetscInt i,*inoffset,*outones,*outoffset;
    PetscSFNode *remote;
    ierr = PetscSFComputeDegreeBegin(bg,&indegree);CHKERRQ(ierr);
    ierr = PetscSFComputeDegreeEnd(bg,&indegree);CHKERRQ(ierr);
    ierr = PetscMalloc3(bg->nowned+1,PetscInt,&inoffset,bg->nlocal,PetscInt,&outones,bg->nlocal,PetscInt,&outoffset);CHKERRQ(ierr);
    inoffset[0] = 0;
    for (i=0; i<bg->nowned; i++) inoffset[i+1] = inoffset[i] + indegree[i];
    for (i=0; i<bg->nlocal; i++) outones[i] = 1;
    ierr = PetscSFFetchAndOpBegin(bg,MPIU_INT,inoffset,outones,outoffset,MPIU_SUM);CHKERRQ(ierr);
    ierr = PetscSFFetchAndOpEnd(bg,MPIU_INT,inoffset,outones,outoffset,MPIU_SUM);CHKERRQ(ierr);
    for (i=0; i<bg->nowned; i++) inoffset[i] -= indegree[i]; /* Undo the increment */
#if defined(PETSC_USE_DEBUG)                                 /* Check that the expected number of increments occurred */
    for (i=0; i<bg->nowned; i++) {
      if (inoffset[i] + indegree[i] != inoffset[i+1]) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_PLIB,"Incorrect result after PetscSFFetchAndOp");
    }
#endif
    ierr = PetscMalloc(bg->nlocal*sizeof(*remote),&remote);CHKERRQ(ierr);
    for (i=0; i<bg->nlocal; i++) {
      remote[i].rank = bg->remote[i].rank;
      remote[i].index = outoffset[i];
    }
    ierr = PetscSFCreate(((PetscObject)bg)->comm,&bg->multi);CHKERRQ(ierr);
    ierr = PetscSFSetSynchronizationType(bg->multi,bg->sync);CHKERRQ(ierr);
    ierr = PetscSFSetGraph(bg->multi,inoffset[bg->nowned],bg->nlocal,PETSC_NULL,PETSC_COPY_VALUES,remote,PETSC_OWN_POINTER);CHKERRQ(ierr);
    if (bg->rankorder) {        /* Sort the ranks */
      PetscMPIInt rank;
      PetscInt *inranks,*newoffset,*outranks,*newoutoffset,*tmpoffset,maxdegree;
      PetscSFNode *newremote;
      ierr = MPI_Comm_rank(((PetscObject)bg)->comm,&rank);CHKERRQ(ierr);
      for (i=0,maxdegree=0; i<bg->nowned; i++) maxdegree = PetscMax(maxdegree,indegree[i]);
      ierr = PetscMalloc5(bg->multi->nowned,PetscInt,&inranks,bg->multi->nowned,PetscInt,&newoffset,bg->nlocal,PetscInt,&outranks,bg->nlocal,PetscInt,&newoutoffset,maxdegree,PetscInt,&tmpoffset);CHKERRQ(ierr);
      for (i=0; i<bg->nlocal; i++) outranks[i] = rank;
      ierr = PetscSFReduceBegin(bg->multi,MPIU_INT,outranks,inranks,MPIU_SUM);CHKERRQ(ierr);
      ierr = PetscSFReduceEnd(bg->multi,MPIU_INT,outranks,inranks,MPIU_SUM);CHKERRQ(ierr);
      /* Sort the incoming ranks at each vertex, build the inverse map */
      for (i=0; i<bg->nowned; i++) {
        PetscInt j;
        for (j=0; j<indegree[i]; j++) tmpoffset[j] = j;
        ierr = PetscSortIntWithArray(indegree[i],inranks+inoffset[i],tmpoffset);CHKERRQ(ierr);
        for (j=0; j<indegree[i]; j++) newoffset[inoffset[i] + tmpoffset[j]] = inoffset[i] + j;
      }
      ierr = PetscSFBcastBegin(bg->multi,MPIU_INT,newoffset,newoutoffset);CHKERRQ(ierr);
      ierr = PetscSFBcastEnd(bg->multi,MPIU_INT,newoffset,newoutoffset);CHKERRQ(ierr);
      ierr = PetscMalloc(bg->nlocal*sizeof(*newremote),&newremote);CHKERRQ(ierr);
      for (i=0; i<bg->nlocal; i++) {
        newremote[i].rank = bg->remote[i].rank;
        newremote[i].index = newoutoffset[i];
      }
      ierr = PetscSFSetGraph(bg->multi,inoffset[bg->nowned],bg->nlocal,PETSC_NULL,PETSC_COPY_VALUES,newremote,PETSC_OWN_POINTER);CHKERRQ(ierr);
      ierr = PetscFree5(inranks,newoffset,outranks,newoutoffset,tmpoffset);CHKERRQ(ierr);
    }
    ierr = PetscFree3(inoffset,outones,outoffset);CHKERRQ(ierr);
  }
  *multi = bg->multi;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscSFBcastBegin"
/*@C
   PetscSFBcastBegin - begin pointwise broadcast to be concluded with call to PetscSFBcastEnd()

   Collective on PetscSF

   Input Arguments:
+  bg - bipartite graph on which to communicate
.  unit - data type associated with each node
-  owned - buffer to broadcast

   Output Arguments:
.  ghosted - buffer to update with ghosted values

   Level: intermediate

.seealso: PetscSFCreate(), PetscSFSetGraph(), PetscSFView(), PetscSFBcastEnd(), PetscSFReduceBegin()
@*/
PetscErrorCode PetscSFBcastBegin(PetscSF bg,MPI_Datatype unit,const void *owned,void *ghosted)
{
  PetscErrorCode     ierr;
  PetscInt           i,nranks;
  const PetscMPIInt  *ranks;
  const MPI_Datatype *mine,*remote;
  MPI_Win            win;

  PetscFunctionBegin;
  ierr = PetscSFGetRanks(bg,&nranks,&ranks,PETSC_NULL,PETSC_NULL,PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscSFGetDataTypes(bg,unit,&mine,&remote);CHKERRQ(ierr);
  ierr = PetscSFGetWindow(bg,unit,(void*)owned,PETSC_TRUE,MPI_MODE_NOPUT|MPI_MODE_NOPRECEDE,MPI_MODE_NOPUT,0,&win);CHKERRQ(ierr);
  for (i=0; i<nranks; i++) {
    if (bg->sync == PETSCSF_SYNCHRONIZATION_LOCK) {ierr = MPI_Win_lock(MPI_LOCK_SHARED,ranks[i],MPI_MODE_NOCHECK,win);CHKERRQ(ierr);}
    ierr = MPI_Get(ghosted,1,mine[i],ranks[i],0,1,remote[i],win);CHKERRQ(ierr);
    if (bg->sync == PETSCSF_SYNCHRONIZATION_LOCK) {ierr = MPI_Win_unlock(ranks[i],win);CHKERRQ(ierr);}
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscSFBcastEnd"
/*@C
   PetscSFBcastEnd - end a broadcast operation started with PetscSFBcastBegin()

   Collective

   Input Arguments:
+  bg - bipartite graph
.  unit - data type
-  owned - buffer t obroadcast

   Output Arguments:
.  ghosted - buffer to update

   Level: intermediate

.seealso: PetscSFSetGraph(), PetscSFReduceEnd()
@*/
PetscErrorCode PetscSFBcastEnd(PetscSF bg,MPI_Datatype unit,const void *owned,void *ghosted)
{
  PetscErrorCode ierr;
  MPI_Win win;

  PetscFunctionBegin;
  ierr = PetscSFFindWindow(bg,unit,owned,&win);CHKERRQ(ierr);
  ierr = PetscSFRestoreWindow(bg,unit,owned,PETSC_TRUE,MPI_MODE_NOSTORE|MPI_MODE_NOSUCCEED,&win);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscSFReduceBegin"
/*@C
   PetscSFReduceBegin - begin reduce of ghost copies into global, to be completed with call to PetscSFReduceEnd()

   Collective

   Input Arguments:
+  bg - bipartite graph
.  unit - data type
.  ghosted - values to reduce
-  op - reduction operation

   Output Arguments:
.  owned - result of reduction

   Level: intermediate

.seealso: PetscSFBcastBegin()
@*/
PetscErrorCode PetscSFReduceBegin(PetscSF bg,MPI_Datatype unit,const void *ghosted,void *owned,MPI_Op op)
{
  PetscErrorCode     ierr;
  PetscInt           i,nranks;
  const PetscMPIInt  *ranks;
  const MPI_Datatype *mine,*remote;
  MPI_Win            win;

  PetscFunctionBegin;
  ierr = PetscSFGetRanks(bg,&nranks,&ranks,PETSC_NULL,PETSC_NULL,PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscSFGetDataTypes(bg,unit,&mine,&remote);CHKERRQ(ierr);
  ierr = PetscSFGetWindow(bg,unit,owned,PETSC_TRUE,MPI_MODE_NOPRECEDE,0,0,&win);CHKERRQ(ierr);
  for (i=0; i<nranks; i++) {
    if (bg->sync == PETSCSF_SYNCHRONIZATION_LOCK) {ierr = MPI_Win_lock(MPI_LOCK_SHARED,ranks[i],MPI_MODE_NOCHECK,win);CHKERRQ(ierr);}
    ierr = MPI_Accumulate((void*)ghosted,1,mine[i],ranks[i],0,1,remote[i],op,win);CHKERRQ(ierr);
    if (bg->sync == PETSCSF_SYNCHRONIZATION_LOCK) {ierr = MPI_Win_unlock(ranks[i],win);CHKERRQ(ierr);}
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscSFReduceEnd"
/*@C
   PetscSFReduceEnd - end a reduction operation started with PetscSFReduceBegin()

   Collective

   Input Arguments:
+  bg - bipartite graph
.  unit - data type
.  ghosted - values to reduce
-  op - reduction operation

   Output Arguments:
.  owned - result of reduction

   Level: intermediate

.seealso: PetscSFSetGraph(), PetscSFBcastEnd()
@*/
PetscErrorCode PetscSFReduceEnd(PetscSF bg,MPI_Datatype unit,const void *ghosted,void *owned,MPI_Op op)
{
  PetscErrorCode ierr;
  MPI_Win win;

  PetscFunctionBegin;
  ierr = PetscSFFindWindow(bg,unit,owned,&win);CHKERRQ(ierr);
  ierr = MPI_Win_fence(MPI_MODE_NOSUCCEED,win);CHKERRQ(ierr);
  ierr = PetscSFRestoreWindow(bg,unit,owned,PETSC_TRUE,MPI_MODE_NOSUCCEED,&win);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscSFComputeDegreeBegin"
/*@C
   PetscSFComputeDegreeBegin - begin computation of degree for each owned vertex, to be completed with PetscSFComputeDegreeEnd()

   Collective

   Input Arguments:
.  bg - bipartite graph

   Output Arguments:
.  degree - degree of each owned vertex

   Level: advanced

.seealso: PetscSFGatherBegin()
@*/
PetscErrorCode PetscSFComputeDegreeBegin(PetscSF bg,const PetscInt **degree)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(bg,PETSCSF_CLASSID,1);
  PetscValidPointer(degree,2);
  if (!bg->degree) {
    PetscInt i;
    ierr = PetscMalloc(bg->nowned*sizeof(PetscInt),&bg->degree);CHKERRQ(ierr);
    ierr = PetscMalloc(bg->nlocal*sizeof(PetscInt),&bg->degreetmp);CHKERRQ(ierr);
    for (i=0; i<bg->nowned; i++) bg->degree[i] = 0;
    for (i=0; i<bg->nlocal; i++) bg->degreetmp[i] = 1;
    ierr = PetscSFReduceBegin(bg,MPIU_INT,bg->degreetmp,bg->degree,MPIU_SUM);CHKERRQ(ierr);
  }
  *degree = PETSC_NULL;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscSFComputeDegreeEnd"
/*@C
   PetscSFComputeDegreeEnd - complete computation of vertex degree that was started with PetscSFComputeDegreeBegin()

   Collective

   Input Arguments:
.  bg - bipartite graph

   Output Arguments:
.  degree - degree of each owned vertex

   Level: developer

.seealso:
@*/
PetscErrorCode PetscSFComputeDegreeEnd(PetscSF bg,const PetscInt **degree)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (bg->degreetmp) {
    ierr = PetscSFReduceEnd(bg,MPIU_INT,bg->degreetmp,bg->degree,MPIU_SUM);CHKERRQ(ierr);
    ierr = PetscFree(bg->degreetmp);CHKERRQ(ierr);
  }
  *degree = bg->degree;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscSFFetchAndOpBegin"
/*@C
   PetscSFFetchAndOpBegin - begin operation that fetches values from owner and updates atomically by applying operation using local value, to be completed with PetscSFFetchAndOpEnd()

   Collective

   Input Arguments:
+  bg - bipartite graph
.  unit - data type
.  ghosted - ghosted value to use in reduction
-  op - operation to use for reduction

   Output Arguments:
+  owned - owned values to be updated, input state is seen by first process to perform an update
-  result - ghosted array with state in owned at time of update using our ghosted values

   Level: advanced

.seealso: PetscSFComputeDegreeBegin(), PetscSFReduceBegin(), PetscSFSetGraph()
@*/
PetscErrorCode PetscSFFetchAndOpBegin(PetscSF bg,MPI_Datatype unit,void *owned,const void *ghosted,void *result,MPI_Op op)
{
  PetscErrorCode ierr;
  PetscInt           i,nranks;
  const PetscMPIInt  *ranks;
  const MPI_Datatype *mine,*remote;
  MPI_Win            win;

  PetscFunctionBegin;
  ierr = PetscSFGetRanks(bg,&nranks,&ranks,PETSC_NULL,PETSC_NULL,PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscSFGetDataTypes(bg,unit,&mine,&remote);CHKERRQ(ierr);
  ierr = PetscSFGetWindow(bg,unit,owned,PETSC_FALSE,0,0,0,&win);CHKERRQ(ierr);
  for (i=0; i<bg->nranks; i++) {
    ierr = MPI_Win_lock(MPI_LOCK_EXCLUSIVE,bg->ranks[i],0,win);CHKERRQ(ierr);
    ierr = MPI_Get(result,1,mine[i],ranks[i],0,1,remote[i],win);CHKERRQ(ierr);
    ierr = MPI_Accumulate((void*)ghosted,1,mine[i],ranks[i],0,1,remote[i],op,win);CHKERRQ(ierr);
    ierr = MPI_Win_unlock(ranks[i],win);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscSFFetchAndOpEnd"
/*@C
   PetscSFFetchAndOpEnd - end operation started in matching call to PetscSFFetchAndOpBegin() to fetch values from owner and update atomically by applying operation using local value

   Collective

   Input Arguments:
+  bg - bipartite graph
.  unit - data type
.  ghosted - ghosted value to use in reduction
-  op - operation to use for reduction

   Output Arguments:
+  owned - owned values to be updated, input state is seen by first process to perform an update
-  result - ghosted array with state in owned at time of update using our ghosted values

   Level: advanced

.seealso: PetscSFComputeDegreeEnd(), PetscSFReduceEnd(), PetscSFSetGraph()
@*/
PetscErrorCode PetscSFFetchAndOpEnd(PetscSF bg,MPI_Datatype unit,void *owned,const void *ghosted,void *result,MPI_Op op)
{
  PetscErrorCode ierr;
  MPI_Win        win;

  PetscFunctionBegin;
  ierr = PetscSFFindWindow(bg,unit,owned,&win);CHKERRQ(ierr);
  /* Nothing to do currently because MPI_LOCK_EXCLUSIVE is used in PetscSFFetchAndOpBegin(), rendering this implementation synchronous. */
  ierr = PetscSFRestoreWindow(bg,unit,owned,PETSC_FALSE,0,&win);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscSFGatherBegin"
/*@C
   PetscSFGatherBegin - begin pointwise gather operation, to be completed with PetscSFGatherEnd()

   Collective

   Input Arguments:
+  bg - bipartite graph
.  unit - data type
-  ghosted - ghosted data to gather to owning process

   Output Argument:
.  owned - owned values to gather into

   Level: intermediate

.seealso: PetscSFComputeDegreeBegin(), PetscSFScatterBegin()
@*/
PetscErrorCode PetscSFGatherBegin(PetscSF bg,MPI_Datatype unit,const void *ghosted,void *owned)
{
  PetscErrorCode ierr;
  PetscSF        multi;

  PetscFunctionBegin;
  ierr = PetscSFGetMultiSF(bg,&multi);CHKERRQ(ierr);
  ierr = PetscSFReduceBegin(multi,unit,ghosted,owned,MPI_REPLACE);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscSFGatherEnd"
/*@C
   PetscSFGatherEnd - ends pointwise gather operation that was started with PetscSFGatherBegin()

   Collective

   Input Arguments:
+  bg - bipartite graph
.  unit - data type
-  ghosted - ghosted data to gather to owning process

   Output Argument:
.  owned - owned values to gather into

   Level: intermediate

.seealso: PetscSFComputeDegreeEnd(), PetscSFScatterEnd()
@*/
PetscErrorCode PetscSFGatherEnd(PetscSF bg,MPI_Datatype unit,const void *ghosted,void *owned)
{
  PetscErrorCode ierr;
  PetscSF        multi;

  PetscFunctionBegin;
  ierr = PetscSFGetMultiSF(bg,&multi);CHKERRQ(ierr);
  ierr = PetscSFReduceEnd(multi,unit,ghosted,owned,MPI_REPLACE);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscSFScatterBegin"
/*@C
   PetscSFScatterBegin - begin pointwise scatter operation, to be completed with PetscSFScatterEnd()

   Collective

   Input Arguments:
+  bg - bipartite graph
.  unit - data type
-  owned - owned values to scatter

   Output Argument:
.  ghosted - ghosted data to scatter into

   Level: intermediate

.seealso: PetscSFComputeDegreeBegin(), PetscSFScatterBegin()
@*/
PetscErrorCode PetscSFScatterBegin(PetscSF bg,MPI_Datatype unit,const void *owned,void *ghosted)
{
  PetscErrorCode ierr;
  PetscSF        multi;

  PetscFunctionBegin;
  ierr = PetscSFGetMultiSF(bg,&multi);CHKERRQ(ierr);
  ierr = PetscSFBcastBegin(multi,unit,owned,ghosted);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscSFScatterEnd"
/*@C
   PetscSFScatterEnd - ends pointwise scatter operation that was started with PetscSFScatterBegin()

   Collective

   Input Arguments:
+  bg - bipartite graph
.  unit - data type
-  owned - owned values to scatter

   Output Argument:
.  ghosted - ghosted data to scatter into

   Level: intermediate

.seealso: PetscSFComputeDegreeEnd(), PetscSFScatterEnd()
@*/
PetscErrorCode PetscSFScatterEnd(PetscSF bg,MPI_Datatype unit,const void *owned,void *ghosted)
{
  PetscErrorCode ierr;
  PetscSF        multi;

  PetscFunctionBegin;
  ierr = PetscSFGetMultiSF(bg,&multi);CHKERRQ(ierr);
  ierr = PetscSFBcastEnd(multi,unit,owned,ghosted);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
