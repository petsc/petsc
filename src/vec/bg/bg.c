#include <private/bgimpl.h>
#include <petscctable.h>

const char *const PetscBGSynchronizationTypes[] = {"FENCE","LOCK","ACTIVE","PetscBGSynchronizationType","PETSCBG_SYNCHRONIZATION_",0};

#undef __FUNCT__
#define __FUNCT__ "PetscBGCreate"
/*@C
   PetscBGCreate - create a bipartite graph communication context

   Not Collective

   Input Arguments:
.  comm - communicator on which the bipartite graph will operate

   Output Arguments:
.  bg - new bipartite graph context

   Level: intermediate

.seealso: PetscBGSetGraph(), PetscBGDestroy()
@*/
PetscErrorCode PetscBGCreate(MPI_Comm comm,PetscBG *bg)
{
  PetscErrorCode ierr;
  PetscBG        b;

  PetscFunctionBegin;
  PetscValidPointer(bg,2);
#ifndef PETSC_USE_DYNAMIC_LIBRARIES
  ierr = PetscBGInitializePackage(PETSC_NULL);CHKERRQ(ierr);
#endif

  ierr = PetscHeaderCreate(b,_p_PetscBG,struct _PetscBGOps,PETSCBG_CLASSID,-1,"PetscBG","Bipartite Graph","PetscBG",comm,PetscBGDestroy,PetscBGView);CHKERRQ(ierr);
  b->nowned    = -1;
  b->nlocal    = -1;
  b->nranks    = -1;
  b->sync      = PETSCBG_SYNCHRONIZATION_FENCE;
  b->rankorder = PETSC_TRUE;
  b->ingroup   = MPI_GROUP_NULL;
  b->outgroup  = MPI_GROUP_NULL;
  *bg = b;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscBGReset"
/*@C
   PetscBGReset - Reset a bipartite graph so that different sizes or neighbors can be used

   Collective

   Input Arguments:
.  bg - bipartite graph

   Level: advanced

.seealso: PetscBGCreate(), PetscBGSetGraph(), PetscBGDestroy()
@*/
PetscErrorCode PetscBGReset(PetscBG bg)
{
  PetscErrorCode ierr;
  PetscBGDataLink link,next;
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
  if (bg->ingroup  != MPI_GROUP_NULL) {ierr = MPI_Group_free(&bg->ingroup);CHKERRQ(ierr);}
  if (bg->outgroup != MPI_GROUP_NULL) {ierr = MPI_Group_free(&bg->outgroup);CHKERRQ(ierr);}
  ierr = PetscBGDestroy(&bg->multi);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscBGDestroy"
/*@C
   PetscBGDestroy - destroy bipartite graph

   Collective

   Input Arguments:
.  bg - bipartite graph context

   Level: intermediate

.seealso: PetscBGCreate(), PetscBGReset()
@*/
PetscErrorCode PetscBGDestroy(PetscBG *bg)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (!*bg) PetscFunctionReturn(0);
  PetscValidHeaderSpecific((*bg),PETSCBG_CLASSID,1);
  if (--((PetscObject)(*bg))->refct > 0) {*bg = 0; PetscFunctionReturn(0);}
  ierr = PetscBGReset(*bg);CHKERRQ(ierr);
  ierr = PetscHeaderDestroy(bg);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscBGSetFromOptions"
/*@C
   PetscBGSetFromOptions - set PetscBG options using the options database

   Logically Collective

   Input Arguments:
.  bg - bipartite graph

   Options Database Keys:
.  -bg_synchronization - synchronization type used by PetscBG

   Level: intermediate

.keywords: KSP, set, from, options, database

.seealso: PetscBGSetSynchronizationType()
@*/
PetscErrorCode PetscBGSetFromOptions(PetscBG bg)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(bg,PETSCBG_CLASSID,1);
  ierr = PetscObjectOptionsBegin((PetscObject)bg);CHKERRQ(ierr);
  ierr = PetscOptionsEnum("-bg_synchronization","synchronization type to use for PetscBG communication","PetscBGSetSynchronizationType",PetscBGSynchronizationTypes,(PetscEnum)bg->sync,(PetscEnum*)&bg->sync,PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-bg_rank_order","sort composite points for gathers and scatters in rank order, gathers are non-deterministic otherwise","PetscBGSetRankOrder",bg->rankorder,&bg->rankorder,PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsEnd();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscBGSetSynchronizationType"
/*@C
   PetscBGSetSynchronizationType - set synchrozitaion type for PetscBG communication

   Logically Collective

   Input Arguments:
+  bg - bipartite graph for communication
-  sync - synchronization type

   Options Database Key:
.  -bg_synchronization <sync> - sets the synchronization type

   Level: intermediate

.seealso: PetscBGSetFromOptions()
@*/
PetscErrorCode PetscBGSetSynchronizationType(PetscBG bg,PetscBGSynchronizationType sync)
{

  PetscFunctionBegin;
  PetscValidHeaderSpecific(bg,PETSCBG_CLASSID,1);
  PetscValidLogicalCollectiveEnum(bg,sync,2);
  bg->sync = sync;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscBGSetRankOrder"
/*@C
   PetscBGSetRankOrder - sort multi-points for gathers and scatters by rank order

   Logically Collective

   Input Arguments:
+  bg - bipartite graph
-  flg - PETSC_TRUE to sort, PETSC_FALSE to skip sorting (lower setup cost, but non-deterministic)

   Level: advanced

.seealso: PetscBGGatherBegin(), PetscBGScatterBegin()
@*/
PetscErrorCode PetscBGSetRankOrder(PetscBG bg,PetscBool flg)
{

  PetscFunctionBegin;
  PetscValidHeaderSpecific(bg,PETSCBG_CLASSID,1);
  PetscValidLogicalCollectiveBool(bg,flg,2);
  if (bg->multi) SETERRQ(((PetscObject)bg)->comm,PETSC_ERR_ARG_WRONGSTATE,"Rank ordering must be set before first call to PetscBGGatherBegin() or PetscBGScatterBegin()");
  bg->rankorder = flg;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscBGSetGraph"
/*@C
   PetscBGSetGraph - Set a parallel bipartite graph

   Collective

   Input Arguments:
+  bg - bipartite graph
.  nlocal - number of local nodes referencing remote nodes
.  ilocal - locations of local/ghosted nodes, or PETSC_NULL for contiguous storage
.  localmode - copy mode for ilocal
.  iremote - locations of global nodes
-  remotemode - copy mode for iremote

   Level: intermediate

.seealso: PetscBGCreate(), PetscBGView()
@*/
PetscErrorCode PetscBGSetGraph(PetscBG bg,PetscInt nowned,PetscInt nlocal,const PetscInt *ilocal,PetscCopyMode localmode,const PetscBGNode *iremote,PetscCopyMode remotemode)
{
  PetscErrorCode ierr;
  PetscTable table;
  PetscTablePosition pos;
  PetscMPIInt size;
  PetscInt i,*rcount;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(bg,PETSCBG_CLASSID,1);
  if (nlocal && ilocal) PetscValidIntPointer(ilocal,4);
  if (nlocal) PetscValidPointer(iremote,6);
  ierr = PetscBGReset(bg);CHKERRQ(ierr);
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
    bg->remote_alloc = (PetscBGNode*)iremote;
    bg->remote = bg->remote_alloc;
    break;
  case PETSC_USE_POINTER:
    bg->remote = (PetscBGNode*)iremote;
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
#define __FUNCT__ "PetscBGView"
/*@C
   PetscBGView - view a bipartite graph

   Collective

   Input Arguments:
+  bg - bipartite graph
-  viewer - viewer to display graph, for example PETSC_VIEWER_STDOUT_WORLD

   Level: beginner

.seealso: PetscBGCreate(), PetscBGSetGraph()
@*/
PetscErrorCode PetscBGView(PetscBG bg,PetscViewer viewer)
{
  PetscErrorCode ierr;
  PetscBool iascii;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(bg,PETSCBG_CLASSID,1);
  if (!viewer) {ierr = PetscViewerASCIIGetStdout(((PetscObject)bg)->comm,&viewer);CHKERRQ(ierr);}
  PetscValidHeaderSpecific(viewer,PETSC_VIEWER_CLASSID,2);
  PetscCheckSameComm(bg,1,viewer,2);
  ierr = PetscTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&iascii);CHKERRQ(ierr);
  if (iascii) {
    PetscMPIInt rank;
    PetscInt i,j;
    ierr = PetscObjectPrintClassNamePrefixType((PetscObject)bg,viewer,"Bipartite Graph Object");CHKERRQ(ierr);
    ierr = PetscViewerASCIIPushTab(viewer);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"synchronization=%s sort=%s\n",PetscBGSynchronizationTypes[bg->sync],bg->rankorder?"rank-order":"unordered");CHKERRQ(ierr);
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
#define __FUNCT__ "MPIU_Type_get_contiguous_size"
static PetscErrorCode MPIU_Type_get_contiguous_size(MPI_Datatype dtype,size_t *bytes)
{
  static const struct {MPI_Datatype type; size_t size;}
  typemap[] = {
    {MPI_INT,sizeof(PetscMPIInt)},
    {MPI_FLOAT,sizeof(float)},
    {MPI_DOUBLE,sizeof(double)},
    {MPIU_INT,sizeof(PetscInt)},
    {MPIU_2INT,2*sizeof(PetscInt)},
    {MPIU_REAL,sizeof(PetscReal)},
    {MPIU_SCALAR,sizeof(PetscScalar)}
  };
  PetscInt i;

  PetscFunctionBegin;
  for (i=0; i<sizeof(typemap)/sizeof(typemap[0]); i++) {
    if (dtype == typemap[i].type) {
      *bytes = typemap[i].size;
      PetscFunctionReturn(0);
    }
  }
  SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"No support for this data type");
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscBGGetDataTypes"
/*@C
   PetscBGGetDataTypes - gets composite local and remote data types for each rank

   Not Collective

   Input Arguments:
+  bg - bipartite graph
-  unit - data type for each node

   Output Arguments:
+  localtypes - types describing part of local buffer referencing each remote rank
-  remotetypes - types describing part of remote buffer referenced for each remote rank

   Level: developer

.seealso: PetscBGSetGraph(), PetscBGView()
@*/
PetscErrorCode PetscBGGetDataTypes(PetscBG bg,MPI_Datatype unit,const MPI_Datatype **localtypes,const MPI_Datatype **remotetypes)
{
  PetscErrorCode ierr;
  PetscBGDataLink link;
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
  ierr = PetscBGGetRanks(bg,&nranks,&ranks,&roffset,&rmine,&rremote);CHKERRQ(ierr);
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
#define __FUNCT__ "PetscBGGetRanks"
/*@C
   PetscBGGetRanks - Get ranks and number of vertices referenced by local part of graph

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

.seealso: PetscBGSetGraph(), PetscBGGetDataTypes()
@*/
PetscErrorCode PetscBGGetRanks(PetscBG bg,PetscInt *nranks,const PetscInt **ranks,const PetscInt **roffset,const PetscMPIInt **rmine,const PetscMPIInt **rremote)
{

  PetscFunctionBegin;
  PetscValidHeaderSpecific(bg,PETSCBG_CLASSID,1);
  if (nranks)  *nranks  = bg->nranks;
  if (ranks)   *ranks   = bg->ranks;
  if (roffset) *roffset = bg->roffset;
  if (rmine)   *rmine   = bg->rmine;
  if (rremote) *rremote = bg->rremote;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscBGGetWindow"
/*@C
   PetscBGGetWindow - Get a window for use with a given data type

   Collective on PetscBG

   Input Arguments:
+  bg - bipartite graph
.  unit - data type
.  array - array to be sent
.  epoch - PETSC_TRUE to acquire the window and start an epoch, PETSC_FALSE to just acquire the window
.  fenceassert - assert parameter for call to MPI_Win_fence(), if PETSCBG_SYNCHRONIZATION_FENCE
.  postassert - assert parameter for call to MPI_Win_post(), if PETSCBG_SYNCHRONIZATION_ACTIVE
-  startassert - assert parameter for call to MPI_Win_start(), if PETSCBG_SYNCHRONIZATION_ACTIVE

   Output Arguments:
.  win - window

   Level: developer

   Developer Notes:
   This currently always creates a new window. This is more synchronous than necessary. An alternative is to try to
   reuse an existing window created with the same array. Another alternative is to maintain a cache of windows and reuse
   whichever one is available, by copying the array into it if necessary.

.seealso: PetscBGGetRanks(), PetscBGGetDataTypes()
@*/
PetscErrorCode PetscBGGetWindow(PetscBG bg,MPI_Datatype unit,void *array,PetscBool epoch,PetscMPIInt fenceassert,PetscMPIInt postassert,PetscMPIInt startassert,MPI_Win *win)
{
  PetscErrorCode ierr;
  size_t bytes;
  PetscBGWinLink link;

  PetscFunctionBegin;
  ierr = MPIU_Type_get_contiguous_size(unit,&bytes);CHKERRQ(ierr);
  ierr = PetscMalloc(sizeof(*link),&link);CHKERRQ(ierr);
  link->bytes = bytes;
  link->addr  = array;
  ierr = MPI_Win_create(array,(MPI_Aint)bytes*bg->nowned,(PetscMPIInt)bytes,MPI_INFO_NULL,((PetscObject)bg)->comm,&link->win);CHKERRQ(ierr);
  link->epoch = epoch;
  link->next = bg->wins;
  bg->wins = link;
  *win = link->win;

  if (epoch) {
    switch (bg->sync) {
    case PETSCBG_SYNCHRONIZATION_FENCE:
      ierr = MPI_Win_fence(fenceassert,*win);CHKERRQ(ierr);
      break;
    case PETSCBG_SYNCHRONIZATION_LOCK: /* Handled outside */
      break;
    case PETSCBG_SYNCHRONIZATION_ACTIVE: {
      MPI_Group ingroup,outgroup;
      ierr = PetscBGGetGroups(bg,&ingroup,&outgroup);CHKERRQ(ierr);
      ierr = MPI_Win_post(ingroup,postassert,*win);CHKERRQ(ierr);
      ierr = MPI_Win_start(outgroup,startassert,*win);CHKERRQ(ierr);
    } break;
    default: SETERRQ(((PetscObject)bg)->comm,PETSC_ERR_PLIB,"Unknown synchronization type");
    }
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscBGFindWindow"
/*@C
   PetscBGFindWindow - Finds a window that is already in use

   Not Collective

   Input Arguments:
+  bg - bipartite graph
.  unit - data type
-  array - array with which the window is associated

   Output Arguments:
.  win - window

   Level: developer

.seealso: PetscBGGetWindow(), PetscBGRestoreWindow()
@*/
PetscErrorCode PetscBGFindWindow(PetscBG bg,MPI_Datatype unit,const void *array,MPI_Win *win)
{
  PetscBGWinLink link;

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
#define __FUNCT__ "PetscBGRestoreWindow"
/*@C
   PetscBGRestoreWindow - Restores a window obtained with PetscBGGetWindow()

   Collective

   Input Arguments:
+  bg - bipartite graph
.  unit - data type
.  array - array associated with window
.  epoch - close an epoch, must match argument to PetscBGGetWindow()
-  win - window

   Level: developer

.seealso: PetscBGFindWindow()
@*/
PetscErrorCode PetscBGRestoreWindow(PetscBG bg,MPI_Datatype unit,const void *array,PetscBool epoch,PetscMPIInt fenceassert,MPI_Win *win)
{
  PetscErrorCode ierr;
  PetscBGWinLink *p,link;

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
    case PETSCBG_SYNCHRONIZATION_FENCE:
      ierr = MPI_Win_fence(fenceassert,*win);CHKERRQ(ierr);
      break;
    case PETSCBG_SYNCHRONIZATION_LOCK:
      break;                    /* handled outside */
    case PETSCBG_SYNCHRONIZATION_ACTIVE: {
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
#define __FUNCT__ "PetscBGGetGroups"
/*@C
   PetscBGGetGroups - gets incoming and outgoing process groups

   Collective

   Input Argument:
.  bg - bipartite graph

   Output Arguments:
+  incoming - group of origin processes for incoming edges
-  outgoing - group of destination processes for outgoing edges

   Level: developer

.seealso: PetscBGGetWindow(), PetscBGRestoreWindow()
@*/
PetscErrorCode PetscBGGetGroups(PetscBG bg,MPI_Group *incoming,MPI_Group *outgoing)
{
  PetscErrorCode ierr;
  MPI_Group group;

  PetscFunctionBegin;
  if (bg->ingroup == MPI_GROUP_NULL) {
    PetscInt    i,*outranks,*inranks;
    const PetscInt *indegree;
    PetscMPIInt rank;
    PetscBGNode *remote;
    PetscBG     bgcount;

    /* Compute the number of incoming ranks */
    ierr = PetscMalloc(bg->nranks*sizeof(PetscBGNode),&remote);CHKERRQ(ierr);
    for (i=0; i<bg->nranks; i++) {
      remote[i].rank = bg->ranks[i];
      remote[i].index = 0;
    }
    ierr = PetscBGCreate(((PetscObject)bg)->comm,&bgcount);CHKERRQ(ierr);
    ierr = PetscBGSetSynchronizationType(bgcount,PETSCBG_SYNCHRONIZATION_LOCK);CHKERRQ(ierr); /* or FENCE, ACTIVE here would cause recursion */
    ierr = PetscBGSetGraph(bgcount,1,bg->nranks,PETSC_NULL,PETSC_COPY_VALUES,remote,PETSC_OWN_POINTER);CHKERRQ(ierr);
    ierr = PetscBGComputeDegreeBegin(bgcount,&indegree);CHKERRQ(ierr);
    ierr = PetscBGComputeDegreeEnd(bgcount,&indegree);CHKERRQ(ierr);

    /* Enumerate the incoming ranks */
    ierr = PetscMalloc2(indegree[0],PetscInt,&inranks,bg->nranks,PetscInt,&outranks);CHKERRQ(ierr);
    ierr = MPI_Comm_rank(((PetscObject)bg)->comm,&rank);CHKERRQ(ierr);
    for (i=0; i<bg->nranks; i++) outranks[i] = rank;
    ierr = PetscBGGatherBegin(bgcount,MPIU_INT,outranks,inranks);CHKERRQ(ierr);
    ierr = PetscBGGatherEnd(bgcount,MPIU_INT,outranks,inranks);CHKERRQ(ierr);
    ierr = MPI_Comm_group(((PetscObject)bg)->comm,&group);CHKERRQ(ierr);
    ierr = MPI_Group_incl(group,indegree[0],inranks,&bg->ingroup);CHKERRQ(ierr);
    ierr = PetscFree2(inranks,outranks);CHKERRQ(ierr);
    ierr = PetscBGDestroy(&bgcount);CHKERRQ(ierr);
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
#define __FUNCT__ "PetscBGGetMultiBG"
/*@C
   PetscBGGetMultiBG - gets the inner BG implemeting gathers and scatters

   Collective

   Input Argument:
.  bg - bipartite graph with possible redundancy

   Output Arguments:
.  multi - bipartite graph with incoming 

   Level: developer

   Notes:

   In most cases, users should use PetscBGGatherBegin() and PetscBGScatterBegin() instead of manipulating multi
   directly. Since multi satisfies the stronger condition that each entry in the global space has exactly one incoming
   edge, it is a candidate for future optimization that might involve its removal.

.seealso: PetscBGSetGraph(), PetscBGGatherBegin(), PetscBGScatterBegin()
@*/
PetscErrorCode PetscBGGetMultiBG(PetscBG bg,PetscBG *multi)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(bg,PETSCBG_CLASSID,1);
  PetscValidPointer(multi,2);
  if (!bg->multi) {
    const PetscInt *indegree;
    PetscInt i,*inoffset,*outones,*outoffset;
    PetscBGNode *remote;
    ierr = PetscBGComputeDegreeBegin(bg,&indegree);CHKERRQ(ierr);
    ierr = PetscBGComputeDegreeEnd(bg,&indegree);CHKERRQ(ierr);
    ierr = PetscMalloc3(bg->nowned+1,PetscInt,&inoffset,bg->nlocal,PetscInt,&outones,bg->nlocal,PetscInt,&outoffset);CHKERRQ(ierr);
    inoffset[0] = 0;
    for (i=0; i<bg->nowned; i++) inoffset[i+1] = inoffset[i] + indegree[i];
    for (i=0; i<bg->nlocal; i++) outones[i] = 1;
    ierr = PetscBGFetchAndOpBegin(bg,MPIU_INT,inoffset,outones,outoffset,MPIU_SUM);CHKERRQ(ierr);
    ierr = PetscBGFetchAndOpEnd(bg,MPIU_INT,inoffset,outones,outoffset,MPIU_SUM);CHKERRQ(ierr);
    for (i=0; i<bg->nowned; i++) inoffset[i] -= indegree[i]; /* Undo the increment */
#if defined(PETSC_USE_DEBUG)                                 /* Check that the expected number of increments occurred */
    for (i=0; i<bg->nowned; i++) {
      if (inoffset[i] + indegree[i] != inoffset[i+1]) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_PLIB,"Incorrect result after PetscBGFetchAndOp");
    }
#endif
    ierr = PetscMalloc(bg->nlocal*sizeof(*remote),&remote);CHKERRQ(ierr);
    for (i=0; i<bg->nlocal; i++) {
      remote[i].rank = bg->remote[i].rank;
      remote[i].index = outoffset[i];
    }
    ierr = PetscBGCreate(((PetscObject)bg)->comm,&bg->multi);CHKERRQ(ierr);
    ierr = PetscBGSetSynchronizationType(bg->multi,bg->sync);CHKERRQ(ierr);
    ierr = PetscBGSetGraph(bg->multi,inoffset[bg->nowned],bg->nlocal,PETSC_NULL,PETSC_COPY_VALUES,remote,PETSC_OWN_POINTER);CHKERRQ(ierr);
    if (bg->rankorder) {        /* Sort the ranks */
      PetscMPIInt rank;
      PetscInt *inranks,*newoffset,*outranks,*newoutoffset,*tmpoffset,maxdegree;
      PetscBGNode *newremote;
      ierr = MPI_Comm_rank(((PetscObject)bg)->comm,&rank);CHKERRQ(ierr);
      for (i=0,maxdegree=0; i<bg->nowned; i++) maxdegree = PetscMax(maxdegree,indegree[i]);
      ierr = PetscMalloc5(bg->multi->nowned,PetscInt,&inranks,bg->multi->nowned,PetscInt,&newoffset,bg->nlocal,PetscInt,&outranks,bg->nlocal,PetscInt,&newoutoffset,maxdegree,PetscInt,&tmpoffset);CHKERRQ(ierr);
      for (i=0; i<bg->nlocal; i++) outranks[i] = rank;
      ierr = PetscBGReduceBegin(bg->multi,MPIU_INT,outranks,inranks,MPIU_SUM);CHKERRQ(ierr);
      ierr = PetscBGReduceEnd(bg->multi,MPIU_INT,outranks,inranks,MPIU_SUM);CHKERRQ(ierr);
      /* Sort the incoming ranks at each vertex, build the inverse map */
      for (i=0; i<bg->nowned; i++) {
        PetscInt j;
        for (j=0; j<indegree[i]; j++) tmpoffset[j] = j;
        ierr = PetscSortIntWithArray(indegree[i],inranks+inoffset[i],tmpoffset);CHKERRQ(ierr);
        for (j=0; j<indegree[i]; j++) newoffset[inoffset[i] + tmpoffset[j]] = inoffset[i] + j;
      }
      ierr = PetscBGBcastBegin(bg->multi,MPIU_INT,newoffset,newoutoffset);CHKERRQ(ierr);
      ierr = PetscBGBcastEnd(bg->multi,MPIU_INT,newoffset,newoutoffset);CHKERRQ(ierr);
      ierr = PetscMalloc(bg->nlocal*sizeof(*newremote),&newremote);CHKERRQ(ierr);
      for (i=0; i<bg->nlocal; i++) {
        newremote[i].rank = bg->remote[i].rank;
        newremote[i].index = newoutoffset[i];
      }
      ierr = PetscBGSetGraph(bg->multi,inoffset[bg->nowned],bg->nlocal,PETSC_NULL,PETSC_COPY_VALUES,newremote,PETSC_OWN_POINTER);CHKERRQ(ierr);
      ierr = PetscFree5(inranks,newoffset,outranks,newoutoffset,tmpoffset);CHKERRQ(ierr);
    }
    ierr = PetscFree3(inoffset,outones,outoffset);CHKERRQ(ierr);
  }
  *multi = bg->multi;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscBGBcastBegin"
/*@C
   PetscBGBcastBegin - begin pointwise broadcast to be concluded with call to PetscBGBcastEnd()

   Collective on PetscBG

   Input Arguments:
+  bg - bipartite graph on which to communicate
.  unit - data type associated with each node
-  owned - buffer to broadcast

   Output Arguments:
.  ghosted - buffer to update with ghosted values

   Level: intermediate

.seealso: PetscBGCreate(), PetscBGSetGraph(), PetscBGView(), PetscBGBcastEnd(), PetscBGReduceBegin()
@*/
PetscErrorCode PetscBGBcastBegin(PetscBG bg,MPI_Datatype unit,const void *owned,void *ghosted)
{
  PetscErrorCode     ierr;
  PetscInt           i,nranks;
  const PetscMPIInt  *ranks;
  const MPI_Datatype *mine,*remote;
  MPI_Win            win;

  PetscFunctionBegin;
  ierr = PetscBGGetRanks(bg,&nranks,&ranks,PETSC_NULL,PETSC_NULL,PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscBGGetDataTypes(bg,unit,&mine,&remote);CHKERRQ(ierr);
  ierr = PetscBGGetWindow(bg,unit,(void*)owned,PETSC_TRUE,MPI_MODE_NOPUT|MPI_MODE_NOPRECEDE,MPI_MODE_NOPUT,0,&win);CHKERRQ(ierr);
  for (i=0; i<nranks; i++) {
    if (bg->sync == PETSCBG_SYNCHRONIZATION_LOCK) {ierr = MPI_Win_lock(MPI_LOCK_SHARED,ranks[i],MPI_MODE_NOCHECK,win);CHKERRQ(ierr);}
    ierr = MPI_Get(ghosted,1,mine[i],ranks[i],0,1,remote[i],win);CHKERRQ(ierr);
    if (bg->sync == PETSCBG_SYNCHRONIZATION_LOCK) {ierr = MPI_Win_unlock(ranks[i],win);CHKERRQ(ierr);}
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscBGBcastEnd"
/*@C
   PetscBGBcastEnd - end a broadcast operation started with PetscBGBcastBegin()

   Collective

   Input Arguments:
+  bg - bipartite graph
.  unit - data type
-  owned - buffer t obroadcast

   Output Arguments:
.  ghosted - buffer to update

   Level: intermediate

.seealso: PetscBGSetGraph(), PetscBGReduceEnd()
@*/
PetscErrorCode PetscBGBcastEnd(PetscBG bg,MPI_Datatype unit,const void *owned,void *ghosted)
{
  PetscErrorCode ierr;
  MPI_Win win;

  PetscFunctionBegin;
  ierr = PetscBGFindWindow(bg,unit,owned,&win);CHKERRQ(ierr);
  ierr = PetscBGRestoreWindow(bg,unit,owned,PETSC_TRUE,MPI_MODE_NOSTORE|MPI_MODE_NOSUCCEED,&win);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscBGReduceBegin"
/*@C
   PetscBGReduceBegin - begin reduce of ghost copies into global, to be completed with call to PetscBGReduceEnd()

   Collective

   Input Arguments:
+  bg - bipartite graph
.  unit - data type
.  ghosted - values to reduce
-  op - reduction operation

   Output Arguments:
.  owned - result of reduction

   Level: intermediate

.seealso: PetscBGBcastBegin()
@*/
PetscErrorCode PetscBGReduceBegin(PetscBG bg,MPI_Datatype unit,const void *ghosted,void *owned,MPI_Op op)
{
  PetscErrorCode     ierr;
  PetscInt           i,nranks;
  const PetscMPIInt  *ranks;
  const MPI_Datatype *mine,*remote;
  MPI_Win            win;

  PetscFunctionBegin;
  ierr = PetscBGGetRanks(bg,&nranks,&ranks,PETSC_NULL,PETSC_NULL,PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscBGGetDataTypes(bg,unit,&mine,&remote);CHKERRQ(ierr);
  ierr = PetscBGGetWindow(bg,unit,owned,PETSC_TRUE,MPI_MODE_NOPRECEDE,0,0,&win);CHKERRQ(ierr);
  for (i=0; i<nranks; i++) {
    if (bg->sync == PETSCBG_SYNCHRONIZATION_LOCK) {ierr = MPI_Win_lock(MPI_LOCK_SHARED,ranks[i],MPI_MODE_NOCHECK,win);CHKERRQ(ierr);}
    ierr = MPI_Accumulate((void*)ghosted,1,mine[i],ranks[i],0,1,remote[i],op,win);CHKERRQ(ierr);
    if (bg->sync == PETSCBG_SYNCHRONIZATION_LOCK) {ierr = MPI_Win_unlock(ranks[i],win);CHKERRQ(ierr);}
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscBGReduceEnd"
/*@C
   PetscBGReduceEnd - end a reduction operation started with PetscBGReduceBegin()

   Collective

   Input Arguments:
+  bg - bipartite graph
.  unit - data type
.  ghosted - values to reduce
-  op - reduction operation

   Output Arguments:
.  owned - result of reduction

   Level: intermediate

.seealso: PetscBGSetGraph(), PetscBGBcastEnd()
@*/
PetscErrorCode PetscBGReduceEnd(PetscBG bg,MPI_Datatype unit,const void *ghosted,void *owned,MPI_Op op)
{
  PetscErrorCode ierr;
  MPI_Win win;

  PetscFunctionBegin;
  ierr = PetscBGFindWindow(bg,unit,owned,&win);CHKERRQ(ierr);
  ierr = MPI_Win_fence(MPI_MODE_NOSUCCEED,win);CHKERRQ(ierr);
  ierr = PetscBGRestoreWindow(bg,unit,owned,PETSC_TRUE,MPI_MODE_NOSUCCEED,&win);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscBGComputeDegreeBegin"
/*@C
   PetscBGComputeDegreeBegin - begin computation of degree for each owned vertex, to be completed with PetscBGComputeDegreeEnd()

   Collective

   Input Arguments:
.  bg - bipartite graph

   Output Arguments:
.  degree - degree of each owned vertex

   Level: advanced

.seealso: PetscBGGatherBegin()
@*/
PetscErrorCode PetscBGComputeDegreeBegin(PetscBG bg,const PetscInt **degree)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(bg,PETSCBG_CLASSID,1);
  PetscValidPointer(degree,2);
  if (!bg->degree) {
    PetscInt i;
    ierr = PetscMalloc(bg->nowned*sizeof(PetscInt),&bg->degree);CHKERRQ(ierr);
    ierr = PetscMalloc(bg->nlocal*sizeof(PetscInt),&bg->degreetmp);CHKERRQ(ierr);
    for (i=0; i<bg->nowned; i++) bg->degree[i] = 0;
    for (i=0; i<bg->nlocal; i++) bg->degreetmp[i] = 1;
    ierr = PetscBGReduceBegin(bg,MPIU_INT,bg->degreetmp,bg->degree,MPIU_SUM);CHKERRQ(ierr);
  }
  *degree = PETSC_NULL;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscBGComputeDegreeEnd"
/*@C
   PetscBGComputeDegreeEnd - complete computation of vertex degree that was started with PetscBGComputeDegreeBegin()

   Collective

   Input Arguments:
.  bg - bipartite graph

   Output Arguments:
.  degree - degree of each owned vertex

   Level: developer

.seealso:
@*/
PetscErrorCode PetscBGComputeDegreeEnd(PetscBG bg,const PetscInt **degree)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (bg->degreetmp) {
    ierr = PetscBGReduceEnd(bg,MPIU_INT,bg->degreetmp,bg->degree,MPIU_SUM);CHKERRQ(ierr);
    ierr = PetscFree(bg->degreetmp);CHKERRQ(ierr);
  }
  *degree = bg->degree;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscBGFetchAndOpBegin"
/*@C
   PetscBGFetchAndOpBegin - begin operation that fetches values from owner and updates atomically by applying operation using local value, to be completed with PetscBGFetchAndOpEnd()

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

.seealso: PetscBGComputeDegreeBegin(), PetscBGReduceBegin(), PetscBGSetGraph()
@*/
PetscErrorCode PetscBGFetchAndOpBegin(PetscBG bg,MPI_Datatype unit,void *owned,const void *ghosted,void *result,MPI_Op op)
{
  PetscErrorCode ierr;
  PetscInt           i,nranks;
  const PetscMPIInt  *ranks;
  const MPI_Datatype *mine,*remote;
  MPI_Win            win;

  PetscFunctionBegin;
  ierr = PetscBGGetRanks(bg,&nranks,&ranks,PETSC_NULL,PETSC_NULL,PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscBGGetDataTypes(bg,unit,&mine,&remote);CHKERRQ(ierr);
  ierr = PetscBGGetWindow(bg,unit,owned,PETSC_FALSE,0,0,0,&win);CHKERRQ(ierr);
  for (i=0; i<bg->nranks; i++) {
    ierr = MPI_Win_lock(MPI_LOCK_EXCLUSIVE,bg->ranks[i],0,win);CHKERRQ(ierr);
    ierr = MPI_Get(result,1,mine[i],ranks[i],0,1,remote[i],win);CHKERRQ(ierr);
    ierr = MPI_Accumulate((void*)ghosted,1,mine[i],ranks[i],0,1,remote[i],op,win);CHKERRQ(ierr);
    ierr = MPI_Win_unlock(ranks[i],win);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscBGFetchAndOpEnd"
/*@C
   PetscBGFetchAndOpEnd - end operation started in matching call to PetscBGFetchAndOpBegin() to fetch values from owner and update atomically by applying operation using local value

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

.seealso: PetscBGComputeDegreeEnd(), PetscBGReduceEnd(), PetscBGSetGraph()
@*/
PetscErrorCode PetscBGFetchAndOpEnd(PetscBG bg,MPI_Datatype unit,void *owned,const void *ghosted,void *result,MPI_Op op)
{
  PetscErrorCode ierr;
  MPI_Win        win;

  PetscFunctionBegin;
  ierr = PetscBGFindWindow(bg,unit,owned,&win);CHKERRQ(ierr);
  /* Nothing to do currently because MPI_LOCK_EXCLUSIVE is used in PetscBGFetchAndOpBegin(), rendering this implementation synchronous. */
  ierr = PetscBGRestoreWindow(bg,unit,owned,PETSC_FALSE,0,&win);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscBGGatherBegin"
/*@C
   PetscBGGatherBegin - begin pointwise gather operation, to be completed with PetscBGGatherEnd()

   Collective

   Input Arguments:
+  bg - bipartite graph
.  unit - data type
-  ghosted - ghosted data to gather to owning process

   Output Argument:
.  owned - owned values to gather into

   Level: intermediate

.seealso: PetscBGComputeDegreeBegin(), PetscBGScatterBegin()
@*/
PetscErrorCode PetscBGGatherBegin(PetscBG bg,MPI_Datatype unit,const void *ghosted,void *owned)
{
  PetscErrorCode ierr;
  PetscBG        multi;

  PetscFunctionBegin;
  ierr = PetscBGGetMultiBG(bg,&multi);CHKERRQ(ierr);
  ierr = PetscBGReduceBegin(multi,unit,ghosted,owned,MPI_REPLACE);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscBGGatherEnd"
/*@C
   PetscBGGatherEnd - ends pointwise gather operation that was started with PetscBGGatherBegin()

   Collective

   Input Arguments:
+  bg - bipartite graph
.  unit - data type
-  ghosted - ghosted data to gather to owning process

   Output Argument:
.  owned - owned values to gather into

   Level: intermediate

.seealso: PetscBGComputeDegreeEnd(), PetscBGScatterEnd()
@*/
PetscErrorCode PetscBGGatherEnd(PetscBG bg,MPI_Datatype unit,const void *ghosted,void *owned)
{
  PetscErrorCode ierr;
  PetscBG        multi;

  PetscFunctionBegin;
  ierr = PetscBGGetMultiBG(bg,&multi);CHKERRQ(ierr);
  ierr = PetscBGReduceEnd(multi,unit,ghosted,owned,MPI_REPLACE);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscBGScatterBegin"
/*@C
   PetscBGScatterBegin - begin pointwise scatter operation, to be completed with PetscBGScatterEnd()

   Collective

   Input Arguments:
+  bg - bipartite graph
.  unit - data type
-  owned - owned values to scatter

   Output Argument:
.  ghosted - ghosted data to scatter into

   Level: intermediate

.seealso: PetscBGComputeDegreeBegin(), PetscBGScatterBegin()
@*/
PetscErrorCode PetscBGScatterBegin(PetscBG bg,MPI_Datatype unit,const void *owned,void *ghosted)
{
  PetscErrorCode ierr;
  PetscBG        multi;

  PetscFunctionBegin;
  ierr = PetscBGGetMultiBG(bg,&multi);CHKERRQ(ierr);
  ierr = PetscBGBcastBegin(multi,unit,owned,ghosted);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscBGScatterEnd"
/*@C
   PetscBGScatterEnd - ends pointwise scatter operation that was started with PetscBGScatterBegin()

   Collective

   Input Arguments:
+  bg - bipartite graph
.  unit - data type
-  owned - owned values to scatter

   Output Argument:
.  ghosted - ghosted data to scatter into

   Level: intermediate

.seealso: PetscBGComputeDegreeEnd(), PetscBGScatterEnd()
@*/
PetscErrorCode PetscBGScatterEnd(PetscBG bg,MPI_Datatype unit,const void *owned,void *ghosted)
{
  PetscErrorCode ierr;
  PetscBG        multi;

  PetscFunctionBegin;
  ierr = PetscBGGetMultiBG(bg,&multi);CHKERRQ(ierr);
  ierr = PetscBGBcastEnd(multi,unit,owned,ghosted);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
