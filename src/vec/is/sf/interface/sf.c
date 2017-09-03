#include <petsc/private/sfimpl.h> /*I "petscsf.h" I*/
#include <petscctable.h>

/* Logging support */
PetscLogEvent PETSCSF_SetGraph, PETSCSF_BcastBegin, PETSCSF_BcastEnd, PETSCSF_ReduceBegin, PETSCSF_ReduceEnd, PETSCSF_FetchAndOpBegin, PETSCSF_FetchAndOpEnd;

#if defined(PETSC_USE_DEBUG)
#  define PetscSFCheckGraphSet(sf,arg) do {                          \
    if (PetscUnlikely(!(sf)->graphset))                              \
      SETERRQ3(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"Must call PetscSFSetGraph() on argument %D \"%s\" before %s()",(arg),#sf,PETSC_FUNCTION_NAME); \
  } while (0)
#else
#  define PetscSFCheckGraphSet(sf,arg) do {} while (0)
#endif

const char *const PetscSFDuplicateOptions[] = {"CONFONLY","RANKS","GRAPH","PetscSFDuplicateOption","PETSCSF_DUPLICATE_",0};

/*@C
   PetscSFCreate - create a star forest communication context

   Not Collective

   Input Arguments:
.  comm - communicator on which the star forest will operate

   Output Arguments:
.  sf - new star forest context

   Level: intermediate

.seealso: PetscSFSetGraph(), PetscSFDestroy()
@*/
PetscErrorCode PetscSFCreate(MPI_Comm comm,PetscSF *sf)
{
  PetscErrorCode ierr;
  PetscSF        b;

  PetscFunctionBegin;
  PetscValidPointer(sf,2);
  ierr = PetscSFInitializePackage();CHKERRQ(ierr);

  ierr = PetscHeaderCreate(b,PETSCSF_CLASSID,"PetscSF","Star Forest","PetscSF",comm,PetscSFDestroy,PetscSFView);CHKERRQ(ierr);

  b->nroots    = -1;
  b->nleaves   = -1;
  b->nranks    = -1;
  b->rankorder = PETSC_TRUE;
  b->ingroup   = MPI_GROUP_NULL;
  b->outgroup  = MPI_GROUP_NULL;
  b->graphset  = PETSC_FALSE;

  *sf = b;
  PetscFunctionReturn(0);
}

/*@C
   PetscSFReset - Reset a star forest so that different sizes or neighbors can be used

   Collective

   Input Arguments:
.  sf - star forest

   Level: advanced

.seealso: PetscSFCreate(), PetscSFSetGraph(), PetscSFDestroy()
@*/
PetscErrorCode PetscSFReset(PetscSF sf)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(sf,PETSCSF_CLASSID,1);
  if (sf->ops->Reset) {ierr = (*sf->ops->Reset)(sf);CHKERRQ(ierr);}
  sf->mine   = NULL;
  ierr       = PetscFree(sf->mine_alloc);CHKERRQ(ierr);
  sf->remote = NULL;
  ierr       = PetscFree(sf->remote_alloc);CHKERRQ(ierr);
  ierr       = PetscFree4(sf->ranks,sf->roffset,sf->rmine,sf->rremote);CHKERRQ(ierr);
  sf->nranks = -1;
  ierr       = PetscFree(sf->degree);CHKERRQ(ierr);
  if (sf->ingroup  != MPI_GROUP_NULL) {ierr = MPI_Group_free(&sf->ingroup);CHKERRQ(ierr);}
  if (sf->outgroup != MPI_GROUP_NULL) {ierr = MPI_Group_free(&sf->outgroup);CHKERRQ(ierr);}
  ierr         = PetscSFDestroy(&sf->multi);CHKERRQ(ierr);
  sf->graphset = PETSC_FALSE;
  sf->setupcalled = PETSC_FALSE;
  PetscFunctionReturn(0);
}

/*@C
   PetscSFSetType - set the PetscSF communication implementation

   Collective on PetscSF

   Input Parameters:
+  sf - the PetscSF context
-  type - a known method

   Options Database Key:
.  -sf_type <type> - Sets the method; use -help for a list
   of available methods (for instance, window, pt2pt, neighbor)

   Notes:
   See "include/petscsf.h" for available methods (for instance)
+    PETSCSFWINDOW - MPI-2/3 one-sided
-    PETSCSFBASIC - basic implementation using MPI-1 two-sided

  Level: intermediate

.keywords: PetscSF, set, type

.seealso: PetscSFType, PetscSFCreate()
@*/
PetscErrorCode PetscSFSetType(PetscSF sf,PetscSFType type)
{
  PetscErrorCode ierr,(*r)(PetscSF);
  PetscBool      match;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(sf,PETSCSF_CLASSID,1);
  PetscValidCharPointer(type,2);

  ierr = PetscObjectTypeCompare((PetscObject)sf,type,&match);CHKERRQ(ierr);
  if (match) PetscFunctionReturn(0);

  ierr = PetscFunctionListFind(PetscSFList,type,&r);CHKERRQ(ierr);
  if (!r) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_UNKNOWN_TYPE,"Unable to find requested PetscSF type %s",type);
  /* Destroy the previous private PetscSF context */
  if (sf->ops->Destroy) {
    ierr = (*(sf)->ops->Destroy)(sf);CHKERRQ(ierr);
  }
  ierr = PetscMemzero(sf->ops,sizeof(*sf->ops));CHKERRQ(ierr);
  ierr = PetscObjectChangeTypeName((PetscObject)sf,type);CHKERRQ(ierr);
  ierr = (*r)(sf);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@
   PetscSFDestroy - destroy star forest

   Collective

   Input Arguments:
.  sf - address of star forest

   Level: intermediate

.seealso: PetscSFCreate(), PetscSFReset()
@*/
PetscErrorCode PetscSFDestroy(PetscSF *sf)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (!*sf) PetscFunctionReturn(0);
  PetscValidHeaderSpecific((*sf),PETSCSF_CLASSID,1);
  if (--((PetscObject)(*sf))->refct > 0) {*sf = 0; PetscFunctionReturn(0);}
  ierr = PetscSFReset(*sf);CHKERRQ(ierr);
  if ((*sf)->ops->Destroy) {ierr = (*(*sf)->ops->Destroy)(*sf);CHKERRQ(ierr);}
  ierr = PetscHeaderDestroy(sf);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@
   PetscSFSetUp - set up communication structures

   Collective

   Input Arguments:
.  sf - star forest communication object

   Level: beginner

.seealso: PetscSFSetFromOptions(), PetscSFSetType()
@*/
PetscErrorCode PetscSFSetUp(PetscSF sf)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (sf->setupcalled) PetscFunctionReturn(0);
  if (!((PetscObject)sf)->type_name) {ierr = PetscSFSetType(sf,PETSCSFBASIC);CHKERRQ(ierr);}
  if (sf->ops->SetUp) {ierr = (*sf->ops->SetUp)(sf);CHKERRQ(ierr);}
  sf->setupcalled = PETSC_TRUE;
  PetscFunctionReturn(0);
}

/*@C
   PetscSFSetFromOptions - set PetscSF options using the options database

   Logically Collective

   Input Arguments:
.  sf - star forest

   Options Database Keys:
+  -sf_type - implementation type, see PetscSFSetType()
-  -sf_rank_order - sort composite points for gathers and scatters in rank order, gathers are non-deterministic otherwise

   Level: intermediate

.keywords: KSP, set, from, options, database

.seealso: PetscSFWindowSetSyncType()
@*/
PetscErrorCode PetscSFSetFromOptions(PetscSF sf)
{
  PetscSFType    deft;
  char           type[256];
  PetscErrorCode ierr;
  PetscBool      flg;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(sf,PETSCSF_CLASSID,1);
  ierr = PetscObjectOptionsBegin((PetscObject)sf);CHKERRQ(ierr);
  deft = ((PetscObject)sf)->type_name ? ((PetscObject)sf)->type_name : PETSCSFBASIC;
  ierr = PetscOptionsFList("-sf_type","PetscSF implementation type","PetscSFSetType",PetscSFList,deft,type,256,&flg);CHKERRQ(ierr);
  ierr = PetscSFSetType(sf,flg ? type : deft);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-sf_rank_order","sort composite points for gathers and scatters in rank order, gathers are non-deterministic otherwise","PetscSFSetRankOrder",sf->rankorder,&sf->rankorder,NULL);CHKERRQ(ierr);
  if (sf->ops->SetFromOptions) {ierr = (*sf->ops->SetFromOptions)(PetscOptionsObject,sf);CHKERRQ(ierr);}
  ierr = PetscOptionsEnd();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
   PetscSFSetRankOrder - sort multi-points for gathers and scatters by rank order

   Logically Collective

   Input Arguments:
+  sf - star forest
-  flg - PETSC_TRUE to sort, PETSC_FALSE to skip sorting (lower setup cost, but non-deterministic)

   Level: advanced

.seealso: PetscSFGatherBegin(), PetscSFScatterBegin()
@*/
PetscErrorCode PetscSFSetRankOrder(PetscSF sf,PetscBool flg)
{

  PetscFunctionBegin;
  PetscValidHeaderSpecific(sf,PETSCSF_CLASSID,1);
  PetscValidLogicalCollectiveBool(sf,flg,2);
  if (sf->multi) SETERRQ(PetscObjectComm((PetscObject)sf),PETSC_ERR_ARG_WRONGSTATE,"Rank ordering must be set before first call to PetscSFGatherBegin() or PetscSFScatterBegin()");
  sf->rankorder = flg;
  PetscFunctionReturn(0);
}

/*@C
   PetscSFSetGraph - Set a parallel star forest

   Collective

   Input Arguments:
+  sf - star forest
.  nroots - number of root vertices on the current process (these are possible targets for other process to attach leaves)
.  nleaves - number of leaf vertices on the current process, each of these references a root on any process
.  ilocal - locations of leaves in leafdata buffers, pass NULL for contiguous storage
.  localmode - copy mode for ilocal
.  iremote - remote locations of root vertices for each leaf on the current process
-  remotemode - copy mode for iremote

   Level: intermediate

.seealso: PetscSFCreate(), PetscSFView(), PetscSFGetGraph()
@*/
PetscErrorCode PetscSFSetGraph(PetscSF sf,PetscInt nroots,PetscInt nleaves,const PetscInt *ilocal,PetscCopyMode localmode,const PetscSFNode *iremote,PetscCopyMode remotemode)
{
  PetscErrorCode     ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(sf,PETSCSF_CLASSID,1);
  ierr = PetscLogEventBegin(PETSCSF_SetGraph,sf,0,0,0);CHKERRQ(ierr);
  if (nleaves && ilocal) PetscValidIntPointer(ilocal,4);
  if (nleaves) PetscValidPointer(iremote,6);
  if (nroots < 0) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"roots %D, cannot be negative",nroots);
  if (nleaves < 0) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"nleaves %D, cannot be negative",nleaves);
  ierr        = PetscSFReset(sf);CHKERRQ(ierr);
  sf->nroots  = nroots;
  sf->nleaves = nleaves;
  if (ilocal) {
    PetscInt i;
    switch (localmode) {
    case PETSC_COPY_VALUES:
      ierr        = PetscMalloc1(nleaves,&sf->mine_alloc);CHKERRQ(ierr);
      sf->mine    = sf->mine_alloc;
      ierr        = PetscMemcpy(sf->mine,ilocal,nleaves*sizeof(*sf->mine));CHKERRQ(ierr);
      sf->minleaf = PETSC_MAX_INT;
      sf->maxleaf = PETSC_MIN_INT;
      for (i=0; i<nleaves; i++) {
        sf->minleaf = PetscMin(sf->minleaf,ilocal[i]);
        sf->maxleaf = PetscMax(sf->maxleaf,ilocal[i]);
      }
      break;
    case PETSC_OWN_POINTER:
      sf->mine_alloc = (PetscInt*)ilocal;
      sf->mine       = sf->mine_alloc;
      break;
    case PETSC_USE_POINTER:
      sf->mine = (PetscInt*)ilocal;
      break;
    default: SETERRQ(PetscObjectComm((PetscObject)sf),PETSC_ERR_ARG_OUTOFRANGE,"Unknown localmode");
    }
  }
  if (!ilocal || nleaves > 0) {
    sf->minleaf = 0;
    sf->maxleaf = nleaves - 1;
  }
  switch (remotemode) {
  case PETSC_COPY_VALUES:
    ierr       = PetscMalloc1(nleaves,&sf->remote_alloc);CHKERRQ(ierr);
    sf->remote = sf->remote_alloc;
    ierr       = PetscMemcpy(sf->remote,iremote,nleaves*sizeof(*sf->remote));CHKERRQ(ierr);
    break;
  case PETSC_OWN_POINTER:
    sf->remote_alloc = (PetscSFNode*)iremote;
    sf->remote       = sf->remote_alloc;
    break;
  case PETSC_USE_POINTER:
    sf->remote = (PetscSFNode*)iremote;
    break;
  default: SETERRQ(PetscObjectComm((PetscObject)sf),PETSC_ERR_ARG_OUTOFRANGE,"Unknown remotemode");
  }

  sf->graphset = PETSC_TRUE;
  ierr = PetscLogEventEnd(PETSCSF_SetGraph,sf,0,0,0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
   PetscSFCreateInverseSF - given a PetscSF in which all vertices have degree 1, creates the inverse map

   Collective

   Input Arguments:
.  sf - star forest to invert

   Output Arguments:
.  isf - inverse of sf

   Level: advanced

   Notes:
   All roots must have degree 1.

   The local space may be a permutation, but cannot be sparse.

.seealso: PetscSFSetGraph()
@*/
PetscErrorCode PetscSFCreateInverseSF(PetscSF sf,PetscSF *isf)
{
  PetscErrorCode ierr;
  PetscMPIInt    rank;
  PetscInt       i,nroots,nleaves,maxlocal,count,*newilocal;
  const PetscInt *ilocal;
  PetscSFNode    *roots,*leaves;

  PetscFunctionBegin;
  ierr = MPI_Comm_rank(PetscObjectComm((PetscObject)sf),&rank);CHKERRQ(ierr);
  ierr = PetscSFGetGraph(sf,&nroots,&nleaves,&ilocal,NULL);CHKERRQ(ierr);
  for (i=0,maxlocal=0; i<nleaves; i++) maxlocal = PetscMax(maxlocal,(ilocal ? ilocal[i] : i)+1);
  ierr = PetscMalloc2(nroots,&roots,maxlocal,&leaves);CHKERRQ(ierr);
  for (i=0; i<maxlocal; i++) {
    leaves[i].rank  = rank;
    leaves[i].index = i;
  }
  for (i=0; i <nroots; i++) {
    roots[i].rank  = -1;
    roots[i].index = -1;
  }
  ierr = PetscSFReduceBegin(sf,MPIU_2INT,leaves,roots,MPIU_REPLACE);CHKERRQ(ierr);
  ierr = PetscSFReduceEnd(sf,MPIU_2INT,leaves,roots,MPIU_REPLACE);CHKERRQ(ierr);

  /* Check whether our leaves are sparse */
  for (i=0,count=0; i<nroots; i++) if (roots[i].rank >= 0) count++;
  if (count == nroots) newilocal = NULL;
  else {                        /* Index for sparse leaves and compact "roots" array (which is to become our leaves). */
    ierr = PetscMalloc1(count,&newilocal);CHKERRQ(ierr);
    for (i=0,count=0; i<nroots; i++) {
      if (roots[i].rank >= 0) {
        newilocal[count]   = i;
        roots[count].rank  = roots[i].rank;
        roots[count].index = roots[i].index;
        count++;
      }
    }
  }

  ierr = PetscSFDuplicate(sf,PETSCSF_DUPLICATE_CONFONLY,isf);CHKERRQ(ierr);
  ierr = PetscSFSetGraph(*isf,maxlocal,count,newilocal,PETSC_OWN_POINTER,roots,PETSC_COPY_VALUES);CHKERRQ(ierr);
  ierr = PetscFree2(roots,leaves);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@
   PetscSFDuplicate - duplicate a PetscSF, optionally preserving rank connectivity and graph

   Collective

   Input Arguments:
+  sf - communication object to duplicate
-  opt - PETSCSF_DUPLICATE_CONFONLY, PETSCSF_DUPLICATE_RANKS, or PETSCSF_DUPLICATE_GRAPH (see PetscSFDuplicateOption)

   Output Arguments:
.  newsf - new communication object

   Level: beginner

.seealso: PetscSFCreate(), PetscSFSetType(), PetscSFSetGraph()
@*/
PetscErrorCode PetscSFDuplicate(PetscSF sf,PetscSFDuplicateOption opt,PetscSF *newsf)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscSFCreate(PetscObjectComm((PetscObject)sf),newsf);CHKERRQ(ierr);
  ierr = PetscSFSetType(*newsf,((PetscObject)sf)->type_name);CHKERRQ(ierr);
  if (sf->ops->Duplicate) {ierr = (*sf->ops->Duplicate)(sf,opt,*newsf);CHKERRQ(ierr);}
  if (opt == PETSCSF_DUPLICATE_GRAPH) {
    PetscInt          nroots,nleaves;
    const PetscInt    *ilocal;
    const PetscSFNode *iremote;
    ierr = PetscSFGetGraph(sf,&nroots,&nleaves,&ilocal,&iremote);CHKERRQ(ierr);
    ierr = PetscSFSetGraph(*newsf,nroots,nleaves,ilocal,PETSC_COPY_VALUES,iremote,PETSC_COPY_VALUES);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

/*@C
   PetscSFGetGraph - Get the graph specifying a parallel star forest

   Not Collective

   Input Arguments:
.  sf - star forest

   Output Arguments:
+  nroots - number of root vertices on the current process (these are possible targets for other process to attach leaves)
.  nleaves - number of leaf vertices on the current process, each of these references a root on any process
.  ilocal - locations of leaves in leafdata buffers
-  iremote - remote locations of root vertices for each leaf on the current process

   Level: intermediate

.seealso: PetscSFCreate(), PetscSFView(), PetscSFSetGraph()
@*/
PetscErrorCode PetscSFGetGraph(PetscSF sf,PetscInt *nroots,PetscInt *nleaves,const PetscInt **ilocal,const PetscSFNode **iremote)
{

  PetscFunctionBegin;
  PetscValidHeaderSpecific(sf,PETSCSF_CLASSID,1);
  /* We are not currently requiring that the graph is set, thus returning nroots=-1 if it has not been set */
  /* if (!sf->graphset) SETERRQ(PetscObjectComm((PetscObject)sf),PETSC_ERR_ARG_WRONGSTATE,"Graph has not been set, must call PetscSFSetGraph()"); */
  if (nroots) *nroots = sf->nroots;
  if (nleaves) *nleaves = sf->nleaves;
  if (ilocal) *ilocal = sf->mine;
  if (iremote) *iremote = sf->remote;
  PetscFunctionReturn(0);
}

/*@C
   PetscSFGetLeafRange - Get the active leaf ranges

   Not Collective

   Input Arguments:
.  sf - star forest

   Output Arguments:
+  minleaf - minimum active leaf on this process
-  maxleaf - maximum active leaf on this process

   Level: developer

.seealso: PetscSFCreate(), PetscSFView(), PetscSFSetGraph(), PetscSFGetGraph()
@*/
PetscErrorCode PetscSFGetLeafRange(PetscSF sf,PetscInt *minleaf,PetscInt *maxleaf)
{

  PetscFunctionBegin;
  PetscValidHeaderSpecific(sf,PETSCSF_CLASSID,1);
  if (minleaf) *minleaf = sf->minleaf;
  if (maxleaf) *maxleaf = sf->maxleaf;
  PetscFunctionReturn(0);
}

/*@C
   PetscSFView - view a star forest

   Collective

   Input Arguments:
+  sf - star forest
-  viewer - viewer to display graph, for example PETSC_VIEWER_STDOUT_WORLD

   Level: beginner

.seealso: PetscSFCreate(), PetscSFSetGraph()
@*/
PetscErrorCode PetscSFView(PetscSF sf,PetscViewer viewer)
{
  PetscErrorCode    ierr;
  PetscBool         iascii;
  PetscViewerFormat format;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(sf,PETSCSF_CLASSID,1);
  if (!viewer) {ierr = PetscViewerASCIIGetStdout(PetscObjectComm((PetscObject)sf),&viewer);CHKERRQ(ierr);}
  PetscValidHeaderSpecific(viewer,PETSC_VIEWER_CLASSID,2);
  PetscCheckSameComm(sf,1,viewer,2);
  ierr = PetscSFSetUp(sf);CHKERRQ(ierr);
  ierr = PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&iascii);CHKERRQ(ierr);
  if (iascii) {
    PetscMPIInt rank;
    PetscInt    ii,i,j;

    ierr = PetscObjectPrintClassNamePrefixType((PetscObject)sf,viewer);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPushTab(viewer);CHKERRQ(ierr);
    if (sf->ops->View) {ierr = (*sf->ops->View)(sf,viewer);CHKERRQ(ierr);}
    ierr = MPI_Comm_rank(PetscObjectComm((PetscObject)sf),&rank);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPushSynchronized(viewer);CHKERRQ(ierr);
    ierr = PetscViewerASCIISynchronizedPrintf(viewer,"[%d] Number of roots=%D, leaves=%D, remote ranks=%D\n",rank,sf->nroots,sf->nleaves,sf->nranks);CHKERRQ(ierr);
    for (i=0; i<sf->nleaves; i++) {
      ierr = PetscViewerASCIISynchronizedPrintf(viewer,"[%d] %D <- (%D,%D)\n",rank,sf->mine ? sf->mine[i] : i,sf->remote[i].rank,sf->remote[i].index);CHKERRQ(ierr);
    }
    ierr = PetscViewerFlush(viewer);CHKERRQ(ierr);
    ierr = PetscViewerGetFormat(viewer,&format);CHKERRQ(ierr);
    if (format == PETSC_VIEWER_ASCII_INFO_DETAIL) {
      PetscMPIInt *tmpranks,*perm;
      ierr = PetscMalloc2(sf->nranks,&tmpranks,sf->nranks,&perm);CHKERRQ(ierr);
      ierr = PetscMemcpy(tmpranks,sf->ranks,sf->nranks*sizeof(tmpranks[0]));CHKERRQ(ierr);
      for (i=0; i<sf->nranks; i++) perm[i] = i;
      ierr = PetscSortMPIIntWithArray(sf->nranks,tmpranks,perm);CHKERRQ(ierr);
      ierr = PetscViewerASCIISynchronizedPrintf(viewer,"[%d] Roots referenced by my leaves, by rank\n",rank);CHKERRQ(ierr);
      for (ii=0; ii<sf->nranks; ii++) {
        i = perm[ii];
        ierr = PetscViewerASCIISynchronizedPrintf(viewer,"[%d] %d: %D edges\n",rank,sf->ranks[i],sf->roffset[i+1]-sf->roffset[i]);CHKERRQ(ierr);
        for (j=sf->roffset[i]; j<sf->roffset[i+1]; j++) {
          ierr = PetscViewerASCIISynchronizedPrintf(viewer,"[%d]    %D <- %D\n",rank,sf->rmine[j],sf->rremote[j]);CHKERRQ(ierr);
        }
      }
      ierr = PetscFree2(tmpranks,perm);CHKERRQ(ierr);
    }
    ierr = PetscViewerFlush(viewer);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPopSynchronized(viewer);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPopTab(viewer);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

/*@C
   PetscSFGetRanks - Get ranks and number of vertices referenced by leaves on this process

   Not Collective

   Input Arguments:
.  sf - star forest

   Output Arguments:
+  nranks - number of ranks referenced by local part
.  ranks - array of ranks
.  roffset - offset in rmine/rremote for each rank (length nranks+1)
.  rmine - concatenated array holding local indices referencing each remote rank
-  rremote - concatenated array holding remote indices referenced for each remote rank

   Level: developer

.seealso: PetscSFSetGraph()
@*/
PetscErrorCode PetscSFGetRanks(PetscSF sf,PetscInt *nranks,const PetscMPIInt **ranks,const PetscInt **roffset,const PetscInt **rmine,const PetscInt **rremote)
{

  PetscFunctionBegin;
  PetscValidHeaderSpecific(sf,PETSCSF_CLASSID,1);
  if (!sf->setupcalled) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"Must call PetscSFSetUp before obtaining ranks");
  if (nranks)  *nranks  = sf->nranks;
  if (ranks)   *ranks   = sf->ranks;
  if (roffset) *roffset = sf->roffset;
  if (rmine)   *rmine   = sf->rmine;
  if (rremote) *rremote = sf->rremote;
  PetscFunctionReturn(0);
}

static PetscBool InList(PetscMPIInt needle,PetscMPIInt n,const PetscMPIInt *list) {
  PetscInt i;
  for (i=0; i<n; i++) {
    if (needle == list[i]) return PETSC_TRUE;
  }
  return PETSC_FALSE;
}

/*@C
   PetscSFSetUpRanks - Set up data structures associated with ranks; this is for internal use by PetscSF implementations.

   Collective

   Input Arguments:
+  sf - PetscSF to set up; PetscSFSetGraph() must have been called
-  dgroup - MPI_Group of ranks to be distinguished (e.g., for self or shared memory exchange)

   Level: developer

.seealso: PetscSFGetRanks()
@*/
PetscErrorCode PetscSFSetUpRanks(PetscSF sf,MPI_Group dgroup)
{
  PetscErrorCode     ierr;
  PetscTable         table;
  PetscTablePosition pos;
  PetscMPIInt        size,groupsize,*groupranks;
  PetscInt           i,*rcount,*ranks;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(sf,PETSCSF_CLASSID,1);
  if (!sf->graphset) SETERRQ(PetscObjectComm((PetscObject)sf),PETSC_ERR_ARG_WRONGSTATE,"PetscSFSetGraph() has not been called yet");
  ierr = MPI_Comm_size(PetscObjectComm((PetscObject)sf),&size);CHKERRQ(ierr);
  ierr = PetscTableCreate(10,size,&table);CHKERRQ(ierr);
  for (i=0; i<sf->nleaves; i++) {
    /* Log 1-based rank */
    ierr = PetscTableAdd(table,sf->remote[i].rank+1,1,ADD_VALUES);CHKERRQ(ierr);
  }
  ierr = PetscTableGetCount(table,&sf->nranks);CHKERRQ(ierr);
  ierr = PetscMalloc4(sf->nranks,&sf->ranks,sf->nranks+1,&sf->roffset,sf->nleaves,&sf->rmine,sf->nleaves,&sf->rremote);CHKERRQ(ierr);
  ierr = PetscMalloc2(sf->nranks,&rcount,sf->nranks,&ranks);CHKERRQ(ierr);
  ierr = PetscTableGetHeadPosition(table,&pos);CHKERRQ(ierr);
  for (i=0; i<sf->nranks; i++) {
    ierr = PetscTableGetNext(table,&pos,&ranks[i],&rcount[i]);CHKERRQ(ierr);
    ranks[i]--;             /* Convert back to 0-based */
  }
  ierr = PetscTableDestroy(&table);CHKERRQ(ierr);

  /* We expect that dgroup is reliably "small" while nranks could be large */
  {
    MPI_Group group;
    PetscMPIInt *dgroupranks;
    ierr = MPI_Comm_group(PetscObjectComm((PetscObject)sf),&group);CHKERRQ(ierr);
    ierr = MPI_Group_size(dgroup,&groupsize);CHKERRQ(ierr);
    ierr = PetscMalloc1(groupsize,&dgroupranks);CHKERRQ(ierr);
    ierr = PetscMalloc1(groupsize,&groupranks);CHKERRQ(ierr);
    for (i=0; i<groupsize; i++) dgroupranks[i] = i;
    if (groupsize) {ierr = MPI_Group_translate_ranks(dgroup,groupsize,dgroupranks,group,groupranks);CHKERRQ(ierr);}
    ierr = MPI_Group_free(&group);CHKERRQ(ierr);
    ierr = PetscFree(dgroupranks);CHKERRQ(ierr);
  }

  /* Partition ranks[] into distinguished (first sf->ndranks) followed by non-distinguished */
  for (sf->ndranks=0,i=sf->nranks; sf->ndranks<i; ) {
    for (i--; sf->ndranks<i; i--) { /* Scan i backward looking for distinguished rank */
      if (InList(ranks[i],groupsize,groupranks)) break;
    }
    for ( ; sf->ndranks<=i; sf->ndranks++) { /* Scan sf->ndranks forward looking for non-distinguished rank */
      if (!InList(ranks[sf->ndranks],groupsize,groupranks)) break;
    }
    if (sf->ndranks < i) {                         /* Swap ranks[sf->ndranks] with ranks[i] */
      PetscInt    tmprank,tmpcount;
      tmprank = ranks[i];
      tmpcount = rcount[i];
      ranks[i] = ranks[sf->ndranks];
      rcount[i] = rcount[sf->ndranks];
      ranks[sf->ndranks] = tmprank;
      rcount[sf->ndranks] = tmpcount;
      sf->ndranks++;
    }
  }
  ierr = PetscFree(groupranks);CHKERRQ(ierr);
  ierr = PetscSortIntWithArray(sf->ndranks,ranks,rcount);CHKERRQ(ierr);
  ierr = PetscSortIntWithArray(sf->nranks-sf->ndranks,ranks+sf->ndranks,rcount+sf->ndranks);CHKERRQ(ierr);
  sf->roffset[0] = 0;
  for (i=0; i<sf->nranks; i++) {
    ierr = PetscMPIIntCast(ranks[i],sf->ranks+i);CHKERRQ(ierr);
    sf->roffset[i+1] = sf->roffset[i] + rcount[i];
    rcount[i]        = 0;
  }
  for (i=0; i<sf->nleaves; i++) {
    PetscInt irank;
    /* Search for index of iremote[i].rank in sf->ranks */
    ierr = PetscFindMPIInt(sf->remote[i].rank,sf->ndranks,sf->ranks,&irank);CHKERRQ(ierr);
    if (irank < 0) {
      ierr = PetscFindMPIInt(sf->remote[i].rank,sf->nranks-sf->ndranks,sf->ranks+sf->ndranks,&irank);CHKERRQ(ierr);
      if (irank >= 0) irank += sf->ndranks;
    }
    if (irank < 0) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_PLIB,"Could not find rank %D in array",sf->remote[i].rank);
    sf->rmine[sf->roffset[irank] + rcount[irank]]   = sf->mine ? sf->mine[i] : i;
    sf->rremote[sf->roffset[irank] + rcount[irank]] = sf->remote[i].index;
    rcount[irank]++;
  }
  ierr = PetscFree2(rcount,ranks);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
   PetscSFGetGroups - gets incoming and outgoing process groups

   Collective

   Input Argument:
.  sf - star forest

   Output Arguments:
+  incoming - group of origin processes for incoming edges (leaves that reference my roots)
-  outgoing - group of destination processes for outgoing edges (roots that I reference)

   Level: developer

.seealso: PetscSFGetWindow(), PetscSFRestoreWindow()
@*/
PetscErrorCode PetscSFGetGroups(PetscSF sf,MPI_Group *incoming,MPI_Group *outgoing)
{
  PetscErrorCode ierr;
  MPI_Group      group;

  PetscFunctionBegin;
  if (sf->ingroup == MPI_GROUP_NULL) {
    PetscInt       i;
    const PetscInt *indegree;
    PetscMPIInt    rank,*outranks,*inranks;
    PetscSFNode    *remote;
    PetscSF        bgcount;

    /* Compute the number of incoming ranks */
    ierr = PetscMalloc1(sf->nranks,&remote);CHKERRQ(ierr);
    for (i=0; i<sf->nranks; i++) {
      remote[i].rank  = sf->ranks[i];
      remote[i].index = 0;
    }
    ierr = PetscSFDuplicate(sf,PETSCSF_DUPLICATE_CONFONLY,&bgcount);CHKERRQ(ierr);
    ierr = PetscSFSetGraph(bgcount,1,sf->nranks,NULL,PETSC_COPY_VALUES,remote,PETSC_OWN_POINTER);CHKERRQ(ierr);
    ierr = PetscSFComputeDegreeBegin(bgcount,&indegree);CHKERRQ(ierr);
    ierr = PetscSFComputeDegreeEnd(bgcount,&indegree);CHKERRQ(ierr);

    /* Enumerate the incoming ranks */
    ierr = PetscMalloc2(indegree[0],&inranks,sf->nranks,&outranks);CHKERRQ(ierr);
    ierr = MPI_Comm_rank(PetscObjectComm((PetscObject)sf),&rank);CHKERRQ(ierr);
    for (i=0; i<sf->nranks; i++) outranks[i] = rank;
    ierr = PetscSFGatherBegin(bgcount,MPI_INT,outranks,inranks);CHKERRQ(ierr);
    ierr = PetscSFGatherEnd(bgcount,MPI_INT,outranks,inranks);CHKERRQ(ierr);
    ierr = MPI_Comm_group(PetscObjectComm((PetscObject)sf),&group);CHKERRQ(ierr);
    ierr = MPI_Group_incl(group,indegree[0],inranks,&sf->ingroup);CHKERRQ(ierr);
    ierr = MPI_Group_free(&group);CHKERRQ(ierr);
    ierr = PetscFree2(inranks,outranks);CHKERRQ(ierr);
    ierr = PetscSFDestroy(&bgcount);CHKERRQ(ierr);
  }
  *incoming = sf->ingroup;

  if (sf->outgroup == MPI_GROUP_NULL) {
    ierr = MPI_Comm_group(PetscObjectComm((PetscObject)sf),&group);CHKERRQ(ierr);
    ierr = MPI_Group_incl(group,sf->nranks,sf->ranks,&sf->outgroup);CHKERRQ(ierr);
    ierr = MPI_Group_free(&group);CHKERRQ(ierr);
  }
  *outgoing = sf->outgroup;
  PetscFunctionReturn(0);
}

/*@C
   PetscSFGetMultiSF - gets the inner SF implemeting gathers and scatters

   Collective

   Input Argument:
.  sf - star forest that may contain roots with 0 or with more than 1 vertex

   Output Arguments:
.  multi - star forest with split roots, such that each root has degree exactly 1

   Level: developer

   Notes:

   In most cases, users should use PetscSFGatherBegin() and PetscSFScatterBegin() instead of manipulating multi
   directly. Since multi satisfies the stronger condition that each entry in the global space has exactly one incoming
   edge, it is a candidate for future optimization that might involve its removal.

.seealso: PetscSFSetGraph(), PetscSFGatherBegin(), PetscSFScatterBegin()
@*/
PetscErrorCode PetscSFGetMultiSF(PetscSF sf,PetscSF *multi)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(sf,PETSCSF_CLASSID,1);
  PetscValidPointer(multi,2);
  if (sf->nroots < 0) {         /* Graph has not been set yet; why do we need this? */
    ierr   = PetscSFDuplicate(sf,PETSCSF_DUPLICATE_RANKS,&sf->multi);CHKERRQ(ierr);
    *multi = sf->multi;
    PetscFunctionReturn(0);
  }
  if (!sf->multi) {
    const PetscInt *indegree;
    PetscInt       i,*inoffset,*outones,*outoffset,maxlocal;
    PetscSFNode    *remote;
    ierr        = PetscSFComputeDegreeBegin(sf,&indegree);CHKERRQ(ierr);
    ierr        = PetscSFComputeDegreeEnd(sf,&indegree);CHKERRQ(ierr);
    for (i=0,maxlocal=0; i<sf->nleaves; i++) maxlocal = PetscMax(maxlocal,(sf->mine ? sf->mine[i] : i)+1);
    ierr        = PetscMalloc3(sf->nroots+1,&inoffset,maxlocal,&outones,maxlocal,&outoffset);CHKERRQ(ierr);
    inoffset[0] = 0;
    for (i=0; i<sf->nroots; i++) inoffset[i+1] = inoffset[i] + indegree[i];
    for (i=0; i<maxlocal; i++) outones[i] = 1;
    ierr = PetscSFFetchAndOpBegin(sf,MPIU_INT,inoffset,outones,outoffset,MPI_SUM);CHKERRQ(ierr);
    ierr = PetscSFFetchAndOpEnd(sf,MPIU_INT,inoffset,outones,outoffset,MPI_SUM);CHKERRQ(ierr);
    for (i=0; i<sf->nroots; i++) inoffset[i] -= indegree[i]; /* Undo the increment */
#if 0
#if defined(PETSC_USE_DEBUG)                                 /* Check that the expected number of increments occurred */
    for (i=0; i<sf->nroots; i++) {
      if (inoffset[i] + indegree[i] != inoffset[i+1]) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_PLIB,"Incorrect result after PetscSFFetchAndOp");
    }
#endif
#endif
    ierr = PetscMalloc1(sf->nleaves,&remote);CHKERRQ(ierr);
    for (i=0; i<sf->nleaves; i++) {
      remote[i].rank  = sf->remote[i].rank;
      remote[i].index = outoffset[sf->mine ? sf->mine[i] : i];
    }
    ierr = PetscSFDuplicate(sf,PETSCSF_DUPLICATE_RANKS,&sf->multi);CHKERRQ(ierr);
    ierr = PetscSFSetGraph(sf->multi,inoffset[sf->nroots],sf->nleaves,sf->mine,PETSC_COPY_VALUES,remote,PETSC_OWN_POINTER);CHKERRQ(ierr);
    if (sf->rankorder) {        /* Sort the ranks */
      PetscMPIInt rank;
      PetscInt    *inranks,*newoffset,*outranks,*newoutoffset,*tmpoffset,maxdegree;
      PetscSFNode *newremote;
      ierr = MPI_Comm_rank(PetscObjectComm((PetscObject)sf),&rank);CHKERRQ(ierr);
      for (i=0,maxdegree=0; i<sf->nroots; i++) maxdegree = PetscMax(maxdegree,indegree[i]);
      ierr = PetscMalloc5(sf->multi->nroots,&inranks,sf->multi->nroots,&newoffset,maxlocal,&outranks,maxlocal,&newoutoffset,maxdegree,&tmpoffset);CHKERRQ(ierr);
      for (i=0; i<maxlocal; i++) outranks[i] = rank;
      ierr = PetscSFReduceBegin(sf->multi,MPIU_INT,outranks,inranks,MPIU_REPLACE);CHKERRQ(ierr);
      ierr = PetscSFReduceEnd(sf->multi,MPIU_INT,outranks,inranks,MPIU_REPLACE);CHKERRQ(ierr);
      /* Sort the incoming ranks at each vertex, build the inverse map */
      for (i=0; i<sf->nroots; i++) {
        PetscInt j;
        for (j=0; j<indegree[i]; j++) tmpoffset[j] = j;
        ierr = PetscSortIntWithArray(indegree[i],inranks+inoffset[i],tmpoffset);CHKERRQ(ierr);
        for (j=0; j<indegree[i]; j++) newoffset[inoffset[i] + tmpoffset[j]] = inoffset[i] + j;
      }
      ierr = PetscSFBcastBegin(sf->multi,MPIU_INT,newoffset,newoutoffset);CHKERRQ(ierr);
      ierr = PetscSFBcastEnd(sf->multi,MPIU_INT,newoffset,newoutoffset);CHKERRQ(ierr);
      ierr = PetscMalloc1(sf->nleaves,&newremote);CHKERRQ(ierr);
      for (i=0; i<sf->nleaves; i++) {
        newremote[i].rank  = sf->remote[i].rank;
        newremote[i].index = newoutoffset[sf->mine ? sf->mine[i] : i];
      }
      ierr = PetscSFSetGraph(sf->multi,inoffset[sf->nroots],sf->nleaves,sf->mine,PETSC_COPY_VALUES,newremote,PETSC_OWN_POINTER);CHKERRQ(ierr);
      ierr = PetscFree5(inranks,newoffset,outranks,newoutoffset,tmpoffset);CHKERRQ(ierr);
    }
    ierr = PetscFree3(inoffset,outones,outoffset);CHKERRQ(ierr);
  }
  *multi = sf->multi;
  PetscFunctionReturn(0);
}

/*@C
   PetscSFCreateEmbeddedSF - removes edges from all but the selected roots, does not remap indices

   Collective

   Input Arguments:
+  sf - original star forest
.  nroots - number of roots to select on this process
-  selected - selected roots on this process

   Output Arguments:
.  newsf - new star forest

   Level: advanced

   Note:
   To use the new PetscSF, it may be necessary to know the indices of the leaves that are still participating. This can
   be done by calling PetscSFGetGraph().

.seealso: PetscSFSetGraph(), PetscSFGetGraph()
@*/
PetscErrorCode PetscSFCreateEmbeddedSF(PetscSF sf,PetscInt nroots,const PetscInt *selected,PetscSF *newsf)
{
  PetscInt      *rootdata, *leafdata, *ilocal;
  PetscSFNode   *iremote;
  PetscInt       leafsize = 0, nleaves = 0, n, i;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(sf,PETSCSF_CLASSID,1);
  if (nroots) PetscValidPointer(selected,3);
  PetscValidPointer(newsf,4);
  if (sf->mine) for (i = 0; i < sf->nleaves; ++i) {leafsize = PetscMax(leafsize, sf->mine[i]+1);}
  else leafsize = sf->nleaves;
  ierr = PetscCalloc2(sf->nroots,&rootdata,leafsize,&leafdata);CHKERRQ(ierr);
  for (i=0; i<nroots; ++i) rootdata[selected[i]] = 1;
  ierr = PetscSFBcastBegin(sf,MPIU_INT,rootdata,leafdata);CHKERRQ(ierr);
  ierr = PetscSFBcastEnd(sf,MPIU_INT,rootdata,leafdata);CHKERRQ(ierr);

  for (i = 0; i < leafsize; ++i) nleaves += leafdata[i];
  ierr = PetscMalloc1(nleaves,&ilocal);CHKERRQ(ierr);
  ierr = PetscMalloc1(nleaves,&iremote);CHKERRQ(ierr);
  for (i = 0, n = 0; i < sf->nleaves; ++i) {
    const PetscInt lidx = sf->mine ? sf->mine[i] : i;

    if (leafdata[lidx]) {
      ilocal[n]        = lidx;
      iremote[n].rank  = sf->remote[i].rank;
      iremote[n].index = sf->remote[i].index;
      ++n;
    }
  }
  if (n != nleaves) SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_PLIB, "There is a size mismatch in the SF embedding, %d != %d", n, nleaves);
  ierr = PetscSFDuplicate(sf,PETSCSF_DUPLICATE_RANKS,newsf);CHKERRQ(ierr);
  ierr = PetscSFSetGraph(*newsf,sf->nroots,nleaves,ilocal,PETSC_OWN_POINTER,iremote,PETSC_OWN_POINTER);CHKERRQ(ierr);
  ierr = PetscFree2(rootdata,leafdata);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
  PetscSFCreateEmbeddedLeafSF - removes edges from all but the selected leaves, does not remap indices

  Collective

  Input Arguments:
+ sf - original star forest
. nleaves - number of leaves to select on this process
- selected - selected leaves on this process

  Output Arguments:
.  newsf - new star forest

  Level: advanced

.seealso: PetscSFCreateEmbeddedSF(), PetscSFSetGraph(), PetscSFGetGraph()
@*/
PetscErrorCode PetscSFCreateEmbeddedLeafSF(PetscSF sf, PetscInt nleaves, const PetscInt *selected, PetscSF *newsf)
{
  PetscSFNode   *iremote;
  PetscInt      *ilocal;
  PetscInt       i;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(sf, PETSCSF_CLASSID, 1);
  if (nleaves) PetscValidPointer(selected, 3);
  PetscValidPointer(newsf, 4);
  ierr = PetscMalloc1(nleaves, &ilocal);CHKERRQ(ierr);
  ierr = PetscMalloc1(nleaves, &iremote);CHKERRQ(ierr);
  for (i = 0; i < nleaves; ++i) {
    const PetscInt l = selected[i];

    ilocal[i]        = sf->mine ? sf->mine[l] : l;
    iremote[i].rank  = sf->remote[l].rank;
    iremote[i].index = sf->remote[l].index;
  }
  ierr = PetscSFDuplicate(sf, PETSCSF_DUPLICATE_RANKS, newsf);CHKERRQ(ierr);
  ierr = PetscSFSetGraph(*newsf, sf->nroots, nleaves, ilocal, PETSC_OWN_POINTER, iremote, PETSC_OWN_POINTER);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
   PetscSFBcastBegin - begin pointwise broadcast to be concluded with call to PetscSFBcastEnd()

   Collective on PetscSF

   Input Arguments:
+  sf - star forest on which to communicate
.  unit - data type associated with each node
-  rootdata - buffer to broadcast

   Output Arguments:
.  leafdata - buffer to update with values from each leaf's respective root

   Level: intermediate

.seealso: PetscSFCreate(), PetscSFSetGraph(), PetscSFView(), PetscSFBcastEnd(), PetscSFReduceBegin()
@*/
PetscErrorCode PetscSFBcastBegin(PetscSF sf,MPI_Datatype unit,const void *rootdata,void *leafdata)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(sf,PETSCSF_CLASSID,1);
  PetscSFCheckGraphSet(sf,1);
  ierr = PetscLogEventBegin(PETSCSF_BcastBegin,sf,0,0,0);CHKERRQ(ierr);
  ierr = PetscSFSetUp(sf);CHKERRQ(ierr);
  ierr = (*sf->ops->BcastBegin)(sf,unit,rootdata,leafdata);CHKERRQ(ierr);
  ierr = PetscLogEventEnd(PETSCSF_BcastBegin,sf,0,0,0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
   PetscSFBcastEnd - end a broadcast operation started with PetscSFBcastBegin()

   Collective

   Input Arguments:
+  sf - star forest
.  unit - data type
-  rootdata - buffer to broadcast

   Output Arguments:
.  leafdata - buffer to update with values from each leaf's respective root

   Level: intermediate

.seealso: PetscSFSetGraph(), PetscSFReduceEnd()
@*/
PetscErrorCode PetscSFBcastEnd(PetscSF sf,MPI_Datatype unit,const void *rootdata,void *leafdata)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(sf,PETSCSF_CLASSID,1);
  PetscSFCheckGraphSet(sf,1);
  ierr = PetscLogEventBegin(PETSCSF_BcastEnd,sf,0,0,0);CHKERRQ(ierr);
  ierr = PetscSFSetUp(sf);CHKERRQ(ierr);
  ierr = (*sf->ops->BcastEnd)(sf,unit,rootdata,leafdata);CHKERRQ(ierr);
  ierr = PetscLogEventEnd(PETSCSF_BcastEnd,sf,0,0,0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
   PetscSFReduceBegin - begin reduction of leafdata into rootdata, to be completed with call to PetscSFReduceEnd()

   Collective

   Input Arguments:
+  sf - star forest
.  unit - data type
.  leafdata - values to reduce
-  op - reduction operation

   Output Arguments:
.  rootdata - result of reduction of values from all leaves of each root

   Level: intermediate

.seealso: PetscSFBcastBegin()
@*/
PetscErrorCode PetscSFReduceBegin(PetscSF sf,MPI_Datatype unit,const void *leafdata,void *rootdata,MPI_Op op)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(sf,PETSCSF_CLASSID,1);
  PetscSFCheckGraphSet(sf,1);
  ierr = PetscLogEventBegin(PETSCSF_ReduceBegin,sf,0,0,0);CHKERRQ(ierr);
  ierr = PetscSFSetUp(sf);CHKERRQ(ierr);
  ierr = (sf->ops->ReduceBegin)(sf,unit,leafdata,rootdata,op);CHKERRQ(ierr);
  ierr = PetscLogEventEnd(PETSCSF_ReduceBegin,sf,0,0,0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
   PetscSFReduceEnd - end a reduction operation started with PetscSFReduceBegin()

   Collective

   Input Arguments:
+  sf - star forest
.  unit - data type
.  leafdata - values to reduce
-  op - reduction operation

   Output Arguments:
.  rootdata - result of reduction of values from all leaves of each root

   Level: intermediate

.seealso: PetscSFSetGraph(), PetscSFBcastEnd()
@*/
PetscErrorCode PetscSFReduceEnd(PetscSF sf,MPI_Datatype unit,const void *leafdata,void *rootdata,MPI_Op op)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(sf,PETSCSF_CLASSID,1);
  PetscSFCheckGraphSet(sf,1);
  ierr = PetscLogEventBegin(PETSCSF_ReduceEnd,sf,0,0,0);CHKERRQ(ierr);
  ierr = PetscSFSetUp(sf);CHKERRQ(ierr);
  ierr = (*sf->ops->ReduceEnd)(sf,unit,leafdata,rootdata,op);CHKERRQ(ierr);
  ierr = PetscLogEventEnd(PETSCSF_ReduceEnd,sf,0,0,0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
   PetscSFComputeDegreeBegin - begin computation of degree for each root vertex, to be completed with PetscSFComputeDegreeEnd()

   Collective

   Input Arguments:
.  sf - star forest

   Output Arguments:
.  degree - degree of each root vertex

   Level: advanced

.seealso: PetscSFGatherBegin()
@*/
PetscErrorCode PetscSFComputeDegreeBegin(PetscSF sf,const PetscInt **degree)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(sf,PETSCSF_CLASSID,1);
  PetscSFCheckGraphSet(sf,1);
  PetscValidPointer(degree,2);
  if (!sf->degreeknown) {
    PetscInt i,maxlocal;
    if (sf->degree) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONGSTATE, "Calls to PetscSFComputeDegreeBegin() cannot be nested.");
    for (i=0,maxlocal=0; i<sf->nleaves; i++) maxlocal = PetscMax(maxlocal,(sf->mine ? sf->mine[i] : i)+1);
    ierr = PetscMalloc1(sf->nroots,&sf->degree);CHKERRQ(ierr);
    ierr = PetscMalloc1(maxlocal,&sf->degreetmp);CHKERRQ(ierr);
    for (i=0; i<sf->nroots; i++) sf->degree[i] = 0;
    for (i=0; i<maxlocal; i++) sf->degreetmp[i] = 1;
    ierr = PetscSFReduceBegin(sf,MPIU_INT,sf->degreetmp,sf->degree,MPI_SUM);CHKERRQ(ierr);
  }
  *degree = NULL;
  PetscFunctionReturn(0);
}

/*@C
   PetscSFComputeDegreeEnd - complete computation of degree for each root vertex, started with PetscSFComputeDegreeBegin()

   Collective

   Input Arguments:
.  sf - star forest

   Output Arguments:
.  degree - degree of each root vertex

   Level: developer

.seealso:
@*/
PetscErrorCode PetscSFComputeDegreeEnd(PetscSF sf,const PetscInt **degree)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(sf,PETSCSF_CLASSID,1);
  PetscSFCheckGraphSet(sf,1);
  if (!sf->degreeknown) {
    ierr = PetscSFReduceEnd(sf,MPIU_INT,sf->degreetmp,sf->degree,MPI_SUM);CHKERRQ(ierr);
    ierr = PetscFree(sf->degreetmp);CHKERRQ(ierr);

    sf->degreeknown = PETSC_TRUE;
  }
  *degree = sf->degree;
  PetscFunctionReturn(0);
}

/*@C
   PetscSFFetchAndOpBegin - begin operation that fetches values from root and updates atomically by applying operation using my leaf value, to be completed with PetscSFFetchAndOpEnd()

   Collective

   Input Arguments:
+  sf - star forest
.  unit - data type
.  leafdata - leaf values to use in reduction
-  op - operation to use for reduction

   Output Arguments:
+  rootdata - root values to be updated, input state is seen by first process to perform an update
-  leafupdate - state at each leaf's respective root immediately prior to my atomic update

   Level: advanced

   Note:
   The update is only atomic at the granularity provided by the hardware. Different roots referenced by the same process
   might be updated in a different order. Furthermore, if a composite type is used for the unit datatype, atomicity is
   not guaranteed across the whole vertex. Therefore, this function is mostly only used with primitive types such as
   integers.

.seealso: PetscSFComputeDegreeBegin(), PetscSFReduceBegin(), PetscSFSetGraph()
@*/
PetscErrorCode PetscSFFetchAndOpBegin(PetscSF sf,MPI_Datatype unit,void *rootdata,const void *leafdata,void *leafupdate,MPI_Op op)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(sf,PETSCSF_CLASSID,1);
  PetscSFCheckGraphSet(sf,1);
  ierr = PetscLogEventBegin(PETSCSF_FetchAndOpBegin,sf,0,0,0);CHKERRQ(ierr);
  ierr = PetscSFSetUp(sf);CHKERRQ(ierr);
  ierr = (*sf->ops->FetchAndOpBegin)(sf,unit,rootdata,leafdata,leafupdate,op);CHKERRQ(ierr);
  ierr = PetscLogEventEnd(PETSCSF_FetchAndOpBegin,sf,0,0,0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
   PetscSFFetchAndOpEnd - end operation started in matching call to PetscSFFetchAndOpBegin() to fetch values from roots and update atomically by applying operation using my leaf value

   Collective

   Input Arguments:
+  sf - star forest
.  unit - data type
.  leafdata - leaf values to use in reduction
-  op - operation to use for reduction

   Output Arguments:
+  rootdata - root values to be updated, input state is seen by first process to perform an update
-  leafupdate - state at each leaf's respective root immediately prior to my atomic update

   Level: advanced

.seealso: PetscSFComputeDegreeEnd(), PetscSFReduceEnd(), PetscSFSetGraph()
@*/
PetscErrorCode PetscSFFetchAndOpEnd(PetscSF sf,MPI_Datatype unit,void *rootdata,const void *leafdata,void *leafupdate,MPI_Op op)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(sf,PETSCSF_CLASSID,1);
  PetscSFCheckGraphSet(sf,1);
  ierr = PetscLogEventBegin(PETSCSF_FetchAndOpEnd,sf,0,0,0);CHKERRQ(ierr);
  ierr = PetscSFSetUp(sf);CHKERRQ(ierr);
  ierr = (*sf->ops->FetchAndOpEnd)(sf,unit,rootdata,leafdata,leafupdate,op);CHKERRQ(ierr);
  ierr = PetscLogEventEnd(PETSCSF_FetchAndOpEnd,sf,0,0,0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
   PetscSFGatherBegin - begin pointwise gather of all leaves into multi-roots, to be completed with PetscSFGatherEnd()

   Collective

   Input Arguments:
+  sf - star forest
.  unit - data type
-  leafdata - leaf data to gather to roots

   Output Argument:
.  multirootdata - root buffer to gather into, amount of space per root is equal to its degree

   Level: intermediate

.seealso: PetscSFComputeDegreeBegin(), PetscSFScatterBegin()
@*/
PetscErrorCode PetscSFGatherBegin(PetscSF sf,MPI_Datatype unit,const void *leafdata,void *multirootdata)
{
  PetscErrorCode ierr;
  PetscSF        multi;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(sf,PETSCSF_CLASSID,1);
  ierr = PetscSFGetMultiSF(sf,&multi);CHKERRQ(ierr);
  ierr = PetscSFReduceBegin(multi,unit,leafdata,multirootdata,MPIU_REPLACE);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
   PetscSFGatherEnd - ends pointwise gather operation that was started with PetscSFGatherBegin()

   Collective

   Input Arguments:
+  sf - star forest
.  unit - data type
-  leafdata - leaf data to gather to roots

   Output Argument:
.  multirootdata - root buffer to gather into, amount of space per root is equal to its degree

   Level: intermediate

.seealso: PetscSFComputeDegreeEnd(), PetscSFScatterEnd()
@*/
PetscErrorCode PetscSFGatherEnd(PetscSF sf,MPI_Datatype unit,const void *leafdata,void *multirootdata)
{
  PetscErrorCode ierr;
  PetscSF        multi;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(sf,PETSCSF_CLASSID,1);
  PetscSFCheckGraphSet(sf,1);
  ierr = PetscSFSetUp(sf);CHKERRQ(ierr);
  ierr = PetscSFGetMultiSF(sf,&multi);CHKERRQ(ierr);
  ierr = PetscSFReduceEnd(multi,unit,leafdata,multirootdata,MPIU_REPLACE);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
   PetscSFScatterBegin - begin pointwise scatter operation from multi-roots to leaves, to be completed with PetscSFScatterEnd()

   Collective

   Input Arguments:
+  sf - star forest
.  unit - data type
-  multirootdata - root buffer to send to each leaf, one unit of data per leaf

   Output Argument:
.  leafdata - leaf data to be update with personal data from each respective root

   Level: intermediate

.seealso: PetscSFComputeDegreeBegin(), PetscSFScatterBegin()
@*/
PetscErrorCode PetscSFScatterBegin(PetscSF sf,MPI_Datatype unit,const void *multirootdata,void *leafdata)
{
  PetscErrorCode ierr;
  PetscSF        multi;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(sf,PETSCSF_CLASSID,1);
  PetscSFCheckGraphSet(sf,1);
  ierr = PetscSFSetUp(sf);CHKERRQ(ierr);
  ierr = PetscSFGetMultiSF(sf,&multi);CHKERRQ(ierr);
  ierr = PetscSFBcastBegin(multi,unit,multirootdata,leafdata);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
   PetscSFScatterEnd - ends pointwise scatter operation that was started with PetscSFScatterBegin()

   Collective

   Input Arguments:
+  sf - star forest
.  unit - data type
-  multirootdata - root buffer to send to each leaf, one unit of data per leaf

   Output Argument:
.  leafdata - leaf data to be update with personal data from each respective root

   Level: intermediate

.seealso: PetscSFComputeDegreeEnd(), PetscSFScatterEnd()
@*/
PetscErrorCode PetscSFScatterEnd(PetscSF sf,MPI_Datatype unit,const void *multirootdata,void *leafdata)
{
  PetscErrorCode ierr;
  PetscSF        multi;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(sf,PETSCSF_CLASSID,1);
  PetscSFCheckGraphSet(sf,1);
  ierr = PetscSFSetUp(sf);CHKERRQ(ierr);
  ierr = PetscSFGetMultiSF(sf,&multi);CHKERRQ(ierr);
  ierr = PetscSFBcastEnd(multi,unit,multirootdata,leafdata);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@
  PetscSFCompose - Compose a new PetscSF equivalent to action to PetscSFs

  Input Parameters:
+ sfA - The first PetscSF
- sfB - The second PetscSF

  Output Parameters:
. sfBA - equvalent PetscSF for applying A then B

  Level: developer

.seealso: PetscSF, PetscSFGetGraph(), PetscSFSetGraph()
@*/
PetscErrorCode PetscSFCompose(PetscSF sfA, PetscSF sfB, PetscSF *sfBA)
{
  MPI_Comm           comm;
  const PetscSFNode *remotePointsA, *remotePointsB;
  PetscSFNode       *remotePointsBA;
  const PetscInt    *localPointsA, *localPointsB;
  PetscInt           numRootsA, numLeavesA, numRootsB, numLeavesB;
  PetscErrorCode     ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(sfA, PETSCSF_CLASSID, 1);
  PetscValidHeaderSpecific(sfB, PETSCSF_CLASSID, 1);
  ierr = PetscObjectGetComm((PetscObject) sfA, &comm);CHKERRQ(ierr);
  ierr = PetscSFGetGraph(sfA, &numRootsA, &numLeavesA, &localPointsA, &remotePointsA);CHKERRQ(ierr);
  ierr = PetscSFGetGraph(sfB, &numRootsB, &numLeavesB, &localPointsB, &remotePointsB);CHKERRQ(ierr);
  ierr = PetscMalloc1(numLeavesB, &remotePointsBA);CHKERRQ(ierr);
  ierr = PetscSFBcastBegin(sfB, MPIU_2INT, remotePointsA, remotePointsBA);CHKERRQ(ierr);
  ierr = PetscSFBcastEnd(sfB, MPIU_2INT, remotePointsA, remotePointsBA);CHKERRQ(ierr);
  ierr = PetscSFCreate(comm, sfBA);CHKERRQ(ierr);
  ierr = PetscSFSetGraph(*sfBA, numRootsA, numLeavesB, localPointsB, PETSC_COPY_VALUES, remotePointsBA, PETSC_OWN_POINTER);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
