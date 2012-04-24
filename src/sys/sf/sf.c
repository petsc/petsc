#include <petsc-private/sfimpl.h>
#include <petscctable.h>

#if defined(PETSC_USE_DEBUG)
#  define PetscSFCheckGraphSet(sf,arg) do {                          \
    if (PetscUnlikely(!(sf)->graphset))                              \
      SETERRQ3(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"Must call PetscSFSetGraph() on argument %D \"%s\" before %s()",(arg),#sf,PETSC_FUNCTION_NAME); \
  } while (0)
#else
#  define PetscSFCheckGraphSet(sf,arg) do {} while (0)
#endif

const char *const PetscSFSynchronizationTypes[] = {"FENCE","LOCK","ACTIVE","PetscSFSynchronizationType","PETSCSF_SYNCHRONIZATION_",0};

#if !defined(PETSC_HAVE_MPI_WIN_CREATE)
#define MPI_WIN_NULL ((MPI_Win)0)
#define MPI_REPLACE 0
#define MPI_Win_create(base,size,disp_unit,info,comm,win) 1;SETERRQ(comm,PETSC_ERR_SUP_SYS,"Need an MPI-2 implementation")
#define MPI_Win_free(win) 1;SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP_SYS,"Need an MPI-2 implementation")
#define MPI_Win_start(group,assert,win) 1;SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP_SYS,"Need an MPI-2 implementation")
#define MPI_Win_post(group,assert,win) 1;SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP_SYS,"Need an MPI-2 implementation")
#define MPI_Win_complete(win) 1;SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP_SYS,"Need an MPI-2 implementation")
#define MPI_Win_wait(win) 1;SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP_SYS,"Need an MPI-2 implementation")
#define MPI_Win_lock(lock_type,rank,assert,win) 1;SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP_SYS,"Need an MPI-2 implementation")
#define MPI_Win_unlock(rank,win) 1;SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP_SYS,"Need an MPI-2 implementation")
#define MPI_Win_fence(assert,win) 1;SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP_SYS,"Need an MPI-2 implementation")
#define MPI_Get(origin_addr,origin_count,origin_datatype,target_rank,target_displ,target_count,target_datatype,win) 1;SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP_SYS,"Need an MPI-2 implementation")
#define MPI_Accumulate(origin_addr,origin_count,origin_datatype,target_rank,target_displ,target_count,target_datatype,op,win) 1;SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP_SYS,"Need an MPI-2 implementation")
/* Independent of MPI_Win, but also not in MPI-1 */
#define MPI_Type_get_envelope(datatype,num_ints,num_addrs,num_dtypes,combiner) (*(num_ints)=0,*(num_addrs)=0,*(num_dtypes)=0,*(combiner)=0,1);SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP_SYS,"Need an MPI-2 implementation")
#define MPI_Type_get_contents(datatype,num_ints,num_addrs,num_dtypes,ints,addrs,dtypes) (*(ints)=0,*(addrs)=0,*(dtypes)=0,1);SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP_SYS,"Need an MPI-2 implementation")
#define MPI_Type_dup(datatype,newtype) (*(newtype)=0,1);SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP_SYS,"Need an MPI-2 implementation")
#define MPI_Type_create_indexed_block(count,blocklength,displs,oldtype,newtype) (*(newtype)=0,1);SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP_SYS,"Need an MPI-2 implementation")
#define MPI_Type_get_true_extent(type,lb,bytes) (*(lb)=0,*(bytes)=0,1);SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP_SYS,"Need an MPI-2 implementation")
#ifndef PETSC_HAVE_MPIUNI
#define MPI_Type_get_extent(type,lb,bytes) (*(lb)=0,*(bytes)=0,1);SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP_SYS,"Need an MPI-2 implementation")
#endif
#define MPI_COMBINER_DUP   0
#define MPI_MODE_NOPUT     0
#define MPI_MODE_NOPRECEDE 0
#define MPI_MODE_NOSUCCEED 0
#define MPI_MODE_NOSTORE   0
#endif

#undef __FUNCT__
#define __FUNCT__ "PetscSFCreate"
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
#ifndef PETSC_USE_DYNAMIC_LIBRARIES
  ierr = PetscSFInitializePackage(PETSC_NULL);CHKERRQ(ierr);
#endif

  ierr = PetscHeaderCreate(b,_p_PetscSF,struct _PetscSFOps,PETSCSF_CLASSID,-1,"PetscSF","Star Forest","PetscSF",comm,PetscSFDestroy,PetscSFView);CHKERRQ(ierr);
  b->nroots    = -1;
  b->nleaves   = -1;
  b->nranks    = -1;
  b->sync      = PETSCSF_SYNCHRONIZATION_FENCE;
  b->rankorder = PETSC_TRUE;
  b->ingroup   = MPI_GROUP_NULL;
  b->outgroup  = MPI_GROUP_NULL;
  b->graphset  = PETSC_FALSE;
  *sf = b;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscSFReset"
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
  PetscSFDataLink link,next;
  PetscSFWinLink  wlink,wnext;
  PetscInt i;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(sf,PETSCSF_CLASSID,1);
  sf->mine = PETSC_NULL;
  ierr = PetscFree(sf->mine_alloc);CHKERRQ(ierr);
  sf->remote = PETSC_NULL;
  ierr = PetscFree(sf->remote_alloc);CHKERRQ(ierr);
  ierr = PetscFree4(sf->ranks,sf->roffset,sf->rmine,sf->rremote);CHKERRQ(ierr);
  ierr = PetscFree(sf->degree);CHKERRQ(ierr);
  for (link=sf->link; link; link=next) {
    next = link->next;
    ierr = MPI_Type_free(&link->unit);CHKERRQ(ierr);
    for (i=0; i<sf->nranks; i++) {
      ierr = MPI_Type_free(&link->mine[i]);CHKERRQ(ierr);
      ierr = MPI_Type_free(&link->remote[i]);CHKERRQ(ierr);
    }
    ierr = PetscFree2(link->mine,link->remote);CHKERRQ(ierr);
    ierr = PetscFree(link);CHKERRQ(ierr);
  }
  sf->link = PETSC_NULL;
  for (wlink=sf->wins; wlink; wlink=wnext) {
    wnext = wlink->next;
    if (wlink->inuse) SETERRQ1(((PetscObject)sf)->comm,PETSC_ERR_ARG_WRONGSTATE,"Window still in use with address %p",(void*)wlink->addr);
    ierr = MPI_Win_free(&wlink->win);CHKERRQ(ierr);
    ierr = PetscFree(wlink);CHKERRQ(ierr);
  }
  sf->wins = PETSC_NULL;
  if (sf->ingroup  != MPI_GROUP_NULL) {ierr = MPI_Group_free(&sf->ingroup);CHKERRQ(ierr);}
  if (sf->outgroup != MPI_GROUP_NULL) {ierr = MPI_Group_free(&sf->outgroup);CHKERRQ(ierr);}
  ierr = PetscSFDestroy(&sf->multi);CHKERRQ(ierr);
  sf->graphset = PETSC_FALSE;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscSFDestroy"
/*@C
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
  ierr = PetscHeaderDestroy(sf);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscSFSetFromOptions"
/*@C
   PetscSFSetFromOptions - set PetscSF options using the options database

   Logically Collective

   Input Arguments:
.  sf - star forest

   Options Database Keys:
.  -sf_synchronization - synchronization type used by PetscSF

   Level: intermediate

.keywords: KSP, set, from, options, database

.seealso: PetscSFSetSynchronizationType()
@*/
PetscErrorCode PetscSFSetFromOptions(PetscSF sf)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(sf,PETSCSF_CLASSID,1);
  ierr = PetscObjectOptionsBegin((PetscObject)sf);CHKERRQ(ierr);
  ierr = PetscOptionsEnum("-sf_synchronization","synchronization type to use for PetscSF communication","PetscSFSetSynchronizationType",PetscSFSynchronizationTypes,(PetscEnum)sf->sync,(PetscEnum*)&sf->sync,PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-sf_rank_order","sort composite points for gathers and scatters in rank order, gathers are non-deterministic otherwise","PetscSFSetRankOrder",sf->rankorder,&sf->rankorder,PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsEnd();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscSFSetSynchronizationType"
/*@C
   PetscSFSetSynchronizationType - set synchrozitaion type for PetscSF communication

   Logically Collective

   Input Arguments:
+  sf - star forest for communication
-  sync - synchronization type

   Options Database Key:
.  -sf_synchronization <sync> - sets the synchronization type

   Level: intermediate

.seealso: PetscSFSetFromOptions()
@*/
PetscErrorCode PetscSFSetSynchronizationType(PetscSF sf,PetscSFSynchronizationType sync)
{

  PetscFunctionBegin;
  PetscValidHeaderSpecific(sf,PETSCSF_CLASSID,1);
  PetscValidLogicalCollectiveEnum(sf,sync,2);
  sf->sync = sync;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscSFSetRankOrder"
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
  if (sf->multi) SETERRQ(((PetscObject)sf)->comm,PETSC_ERR_ARG_WRONGSTATE,"Rank ordering must be set before first call to PetscSFGatherBegin() or PetscSFScatterBegin()");
  sf->rankorder = flg;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscSFSetGraph"
/*@C
   PetscSFSetGraph - Set a parallel star forest

   Collective

   Input Arguments:
+  sf - star forest
.  nroots - number of root vertices on the current process (these are possible targets for other process to attach leaves)
.  nleaves - number of leaf vertices on the current process, each of these references a root on any process
.  ilocal - locations of leaves in leafdata buffers, pass PETSC_NULL for contiguous storage
.  localmode - copy mode for ilocal
.  iremote - remote locations of root vertices for each leaf on the current process
-  remotemode - copy mode for iremote

   Level: intermediate

.seealso: PetscSFCreate(), PetscSFView(), PetscSFGetGraph()
@*/
PetscErrorCode PetscSFSetGraph(PetscSF sf,PetscInt nroots,PetscInt nleaves,const PetscInt *ilocal,PetscCopyMode localmode,const PetscSFNode *iremote,PetscCopyMode remotemode)
{
  PetscErrorCode ierr;
  PetscTable table;
  PetscTablePosition pos;
  PetscMPIInt size;
  PetscInt i,*rcount,*ranks;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(sf,PETSCSF_CLASSID,1);
  if (nleaves && ilocal) PetscValidIntPointer(ilocal,4);
  if (nleaves) PetscValidPointer(iremote,6);
  if (nroots < 0) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"roots %D, cannot be negative",nroots);
  if (nleaves < 0) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"nleaves %D, cannot be negative",nleaves);
  ierr = PetscSFReset(sf);CHKERRQ(ierr);
  sf->nroots = nroots;
  sf->nleaves = nleaves;
  if (ilocal) {
    switch (localmode) {
    case PETSC_COPY_VALUES:
      ierr = PetscMalloc(nleaves*sizeof(*sf->mine),&sf->mine_alloc);CHKERRQ(ierr);
      sf->mine = sf->mine_alloc;
      ierr = PetscMemcpy(sf->mine,ilocal,nleaves*sizeof(*sf->mine));CHKERRQ(ierr);
      break;
    case PETSC_OWN_POINTER:
      sf->mine_alloc = (PetscInt*)ilocal;
      sf->mine = sf->mine_alloc;
      break;
    case PETSC_USE_POINTER:
      sf->mine = (PetscInt*)ilocal;
      break;
    default: SETERRQ(((PetscObject)sf)->comm,PETSC_ERR_ARG_OUTOFRANGE,"Unknown localmode");
    }
  }
  switch (remotemode) {
  case PETSC_COPY_VALUES:
    ierr = PetscMalloc(nleaves*sizeof(*sf->remote),&sf->remote_alloc);CHKERRQ(ierr);
    sf->remote = sf->remote_alloc;
    ierr = PetscMemcpy(sf->remote,iremote,nleaves*sizeof(*sf->remote));CHKERRQ(ierr);
    break;
  case PETSC_OWN_POINTER:
    sf->remote_alloc = (PetscSFNode*)iremote;
    sf->remote = sf->remote_alloc;
    break;
  case PETSC_USE_POINTER:
    sf->remote = (PetscSFNode*)iremote;
    break;
  default: SETERRQ(((PetscObject)sf)->comm,PETSC_ERR_ARG_OUTOFRANGE,"Unknown remotemode");
  }

  ierr = MPI_Comm_size(((PetscObject)sf)->comm,&size);CHKERRQ(ierr);
  ierr = PetscTableCreate(10,size,&table);CHKERRQ(ierr);
  for (i=0; i<nleaves; i++) {
    /* Log 1-based rank */
    ierr = PetscTableAdd(table,iremote[i].rank+1,1,ADD_VALUES);CHKERRQ(ierr);
  }
  ierr = PetscTableGetCount(table,&sf->nranks);CHKERRQ(ierr);
  ierr = PetscMalloc4(sf->nranks,PetscInt,&sf->ranks,sf->nranks+1,PetscInt,&sf->roffset,nleaves,PetscMPIInt,&sf->rmine,nleaves,PetscMPIInt,&sf->rremote);CHKERRQ(ierr);
  ierr = PetscMalloc2(sf->nranks,PetscInt,&rcount,sf->nranks,PetscInt,&ranks);CHKERRQ(ierr);
  ierr = PetscTableGetHeadPosition(table,&pos);CHKERRQ(ierr);
  for (i=0; i<sf->nranks; i++) {
    ierr = PetscTableGetNext(table,&pos,&ranks[i],&rcount[i]);CHKERRQ(ierr);
    ranks[i]--;             /* Convert back to 0-based */
  }
  ierr = PetscTableDestroy(&table);CHKERRQ(ierr);
  ierr = PetscSortIntWithArray(sf->nranks,ranks,rcount);CHKERRQ(ierr);
  sf->roffset[0] = 0;
  for (i=0; i<sf->nranks; i++) {
    sf->ranks[i] = PetscMPIIntCast(ranks[i]);
    sf->roffset[i+1] = sf->roffset[i] + rcount[i];
    rcount[i] = 0;
  }
  for (i=0; i<nleaves; i++) {
    PetscInt lo,hi,irank;
    /* Search for index of iremote[i].rank in sf->ranks */
    lo = 0; hi = sf->nranks;
    while (hi - lo > 1) {
      PetscInt mid = lo + (hi - lo)/2;
      if (iremote[i].rank < sf->ranks[mid]) hi = mid;
      else                                  lo = mid;
    }
    if (hi - lo == 1 && iremote[i].rank == sf->ranks[lo]) irank = lo;
    else SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_PLIB,"Could not find rank %D in array",iremote[i].rank);
    sf->rmine[sf->roffset[irank] + rcount[irank]] = ilocal ? ilocal[i] : i;
    sf->rremote[sf->roffset[irank] + rcount[irank]] = iremote[i].index;
    rcount[irank]++;
  }
  ierr = PetscFree2(rcount,ranks);CHKERRQ(ierr);
#if !defined(PETSC_USE_64BIT_INDICES)
  if (nroots == PETSC_DETERMINE) {
    /* Jed, if you have a better way to do this, put it in */
    PetscInt *numRankLeaves, *leafOff, *leafIndices, *numRankRoots, *rootOff, *rootIndices, maxRoots = 0;

    /* All to all to determine number of leaf indices from each (you can do this using Scan and asynch messages) */
    ierr = PetscMalloc4(size,PetscInt,&numRankLeaves,size+1,PetscInt,&leafOff,size,PetscInt,&numRankRoots,size+1,PetscInt,&rootOff);CHKERRQ(ierr);
    ierr = PetscMemzero(numRankLeaves, size * sizeof(PetscInt));CHKERRQ(ierr);
    for(i = 0; i < nleaves; ++i) {
      ++numRankLeaves[iremote[i].rank];
    }
    ierr = MPI_Alltoall(numRankLeaves, 1, MPIU_INT, numRankRoots, 1, MPIU_INT, ((PetscObject) sf)->comm);CHKERRQ(ierr);
    /* Could set nroots to this maximum */
    for(i = 0; i < size; ++i) {
      maxRoots += numRankRoots[i];
    }
    /* Gather all indices */
    ierr = PetscMalloc2(nleaves,PetscInt,&leafIndices,maxRoots,PetscInt,&rootIndices);CHKERRQ(ierr);
    leafOff[0] = 0;
    for(i = 0; i < size; ++i) {
      leafOff[i+1] = leafOff[i] + numRankLeaves[i];
    }
    for(i = 0; i < nleaves; ++i) {
      leafIndices[leafOff[iremote[i].rank]++] = iremote[i].index;
    }
    leafOff[0] = 0;
    for(i = 0; i < size; ++i) {
      leafOff[i+1] = leafOff[i] + numRankLeaves[i];
    }
    rootOff[0] = 0;
    for(i = 0; i < size; ++i) {
      rootOff[i+1] = rootOff[i] + numRankRoots[i];
    }
    ierr = MPI_Alltoallv(leafIndices, numRankLeaves, leafOff, MPIU_INT, rootIndices, numRankRoots, rootOff, MPIU_INT, ((PetscObject) sf)->comm);CHKERRQ(ierr);
    /* Sort and reduce */
    ierr = PetscSortRemoveDupsInt(&maxRoots, rootIndices);CHKERRQ(ierr);
    ierr = PetscFree2(leafIndices,rootIndices);CHKERRQ(ierr);
    ierr = PetscFree4(numRankLeaves,leafOff,numRankRoots,rootOff);CHKERRQ(ierr);
    sf->nroots = maxRoots;
  }
#endif

  sf->graphset = PETSC_TRUE;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscSFCreateInverseSF"
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
  PetscMPIInt rank;
  PetscInt i,nroots,nleaves,maxlocal,count,*newilocal;
  const PetscInt *ilocal;
  PetscSFNode *roots,*leaves;

  PetscFunctionBegin;
  ierr = MPI_Comm_rank(((PetscObject)sf)->comm,&rank);CHKERRQ(ierr);
  ierr = PetscSFGetGraph(sf,&nroots,&nleaves,&ilocal,PETSC_NULL);CHKERRQ(ierr);
  for (i=0,maxlocal=0; i<nleaves; i++) maxlocal = PetscMax(maxlocal,(ilocal?ilocal[i]:i)+1);
  ierr = PetscMalloc2(nroots,PetscSFNode,&roots,nleaves,PetscSFNode,&leaves);CHKERRQ(ierr);
  for (i=0; i<nleaves; i++) {
    leaves[i].rank = rank;
    leaves[i].index = i;
  }
  for (i=0;i <nroots; i++) {
    roots[i].rank = -1;
    roots[i].index = -1;
  }
  ierr = PetscSFReduceBegin(sf,MPIU_2INT,leaves,roots,MPI_REPLACE);CHKERRQ(ierr);
  ierr = PetscSFReduceEnd(sf,MPIU_2INT,leaves,roots,MPI_REPLACE);CHKERRQ(ierr);

  /* Check whether our leaves are sparse */
  for (i=0,count=0; i<nroots; i++) if (roots[i].rank >= 0) count++;
  if (count == nroots) newilocal = PETSC_NULL;
  else {                        /* Index for sparse leaves and compact "roots" array (which is to become our leaves). */
    ierr = PetscMalloc(count*sizeof(PetscInt),&newilocal);CHKERRQ(ierr);
    for (i=0,count=0; i<nroots; i++) {
      if (roots[i].rank >= 0) {
        newilocal[count] = i;
        roots[count].rank  = roots[i].rank;
        roots[count].index = roots[i].index;
        count++;
      }
    }
  }

  ierr = PetscSFCreate(((PetscObject)sf)->comm,isf);CHKERRQ(ierr);
  ierr = PetscSFSetSynchronizationType(*isf,sf->sync);CHKERRQ(ierr);
  ierr = PetscSFSetGraph(*isf,maxlocal,count,newilocal,PETSC_OWN_POINTER,roots,PETSC_COPY_VALUES);CHKERRQ(ierr);
  ierr = PetscFree2(roots,leaves);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscSFGetGraph"
/*@C
   PetscSFGetGraph - Get the graph specifying a parallel star forest

   Collective

   Input Arguments:
+  sf - star forest
.  nroots - number of root vertices on the current process (these are possible targets for other process to attach leaves)
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
  if (nroots) *nroots = sf->nroots;
  if (nleaves) *nleaves = sf->nleaves;
  if (ilocal) *ilocal = sf->mine;
  if (iremote) *iremote = sf->remote;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscSFView"
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
  PetscErrorCode ierr;
  PetscBool iascii;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(sf,PETSCSF_CLASSID,1);
  if (!viewer) {ierr = PetscViewerASCIIGetStdout(((PetscObject)sf)->comm,&viewer);CHKERRQ(ierr);}
  PetscValidHeaderSpecific(viewer,PETSC_VIEWER_CLASSID,2);
  PetscCheckSameComm(sf,1,viewer,2);
  ierr = PetscTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&iascii);CHKERRQ(ierr);
  if (iascii) {
    PetscMPIInt rank;
    PetscInt i,j;
    PetscBool verbose;
    ierr = PetscObjectPrintClassNamePrefixType((PetscObject)sf,viewer,"Star Forest Object");CHKERRQ(ierr);
    ierr = PetscViewerASCIIPushTab(viewer);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"synchronization=%s sort=%s\n",PetscSFSynchronizationTypes[sf->sync],sf->rankorder?"rank-order":"unordered");CHKERRQ(ierr);
    ierr = MPI_Comm_rank(((PetscObject)sf)->comm,&rank);CHKERRQ(ierr);
    ierr = PetscViewerASCIISynchronizedAllow(viewer,PETSC_TRUE);CHKERRQ(ierr);
    ierr = PetscViewerASCIISynchronizedPrintf(viewer,"[%d] Number of roots=%D, leaves=%D, remote ranks=%D\n",rank,sf->nroots,sf->nleaves,sf->nranks);CHKERRQ(ierr);
    for (i=0; i<sf->nleaves; i++) {
      ierr = PetscViewerASCIISynchronizedPrintf(viewer,"[%d] %D <- (%D,%D)\n",rank,sf->mine?sf->mine[i]:i,sf->remote[i].rank,sf->remote[i].index);CHKERRQ(ierr);
    }
    ierr = PetscViewerFlush(viewer);CHKERRQ(ierr);
    verbose = PETSC_FALSE;
    ierr = PetscOptionsGetBool(((PetscObject)sf)->prefix,"-sf_view_verbose",&verbose,PETSC_NULL);CHKERRQ(ierr);
    if (verbose) {
      ierr = PetscViewerASCIISynchronizedPrintf(viewer,"[%d] Roots referenced by my leaves, by rank\n",rank);CHKERRQ(ierr);
      for (i=0; i<sf->nranks; i++) {
        ierr = PetscViewerASCIISynchronizedPrintf(viewer,"[%d] %D: %D edges\n",rank,sf->ranks[i],sf->roffset[i+1]-sf->roffset[i]);CHKERRQ(ierr);
        for (j=sf->roffset[i]; j<sf->roffset[i+1]; j++) {
          ierr = PetscViewerASCIISynchronizedPrintf(viewer,"[%d]    %D <- %D\n",rank,sf->rmine[j],sf->rremote[j]);CHKERRQ(ierr);
        }
      }
    }
    ierr = PetscViewerFlush(viewer);CHKERRQ(ierr);
    ierr = PetscViewerASCIISynchronizedAllow(viewer,PETSC_FALSE);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPopTab(viewer);CHKERRQ(ierr);
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
    PetscMPIInt ints[1];
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
  if (acombiner == bcombiner && aintcount == bintcount && aaddrcount == baddrcount && atypecount == btypecount && (aintcount > 0 || aaddrcount > 0 || atypecount > 0)) {
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
+  sf - star forest
-  unit - data type for each node

   Output Arguments:
+  localtypes - types describing part of local leaf buffer referencing each remote rank
-  remotetypes - types describing part of remote root buffer referenced for each remote rank

   Level: developer

.seealso: PetscSFSetGraph(), PetscSFView()
@*/
PetscErrorCode PetscSFGetDataTypes(PetscSF sf,MPI_Datatype unit,const MPI_Datatype **localtypes,const MPI_Datatype **remotetypes)
{
  PetscErrorCode ierr;
  PetscSFDataLink link;
  PetscInt i,nranks;
  const PetscInt *roffset;
  const PetscMPIInt *ranks,*rmine,*rremote;

  PetscFunctionBegin;
  /* Look for types in cache */
  for (link=sf->link; link; link=link->next) {
    PetscBool match;
    ierr = MPIU_Type_compare(unit,link->unit,&match);CHKERRQ(ierr);
    if (match) {
      *localtypes = link->mine;
      *remotetypes = link->remote;
      PetscFunctionReturn(0);
    }
  }

  /* Create new composite types for each send rank */
  ierr = PetscSFGetRanks(sf,&nranks,&ranks,&roffset,&rmine,&rremote);CHKERRQ(ierr);
  ierr = PetscMalloc(sizeof(*link),&link);CHKERRQ(ierr);
  ierr = MPI_Type_dup(unit,&link->unit);CHKERRQ(ierr);
  ierr = PetscMalloc2(nranks,MPI_Datatype,&link->mine,nranks,MPI_Datatype,&link->remote);CHKERRQ(ierr);
  for (i=0; i<nranks; i++) {
    PETSC_UNUSED PetscInt rcount = roffset[i+1] - roffset[i];
    ierr = MPI_Type_create_indexed_block(rcount,1,sf->rmine+sf->roffset[i],link->unit,&link->mine[i]);CHKERRQ(ierr);
    ierr = MPI_Type_create_indexed_block(rcount,1,sf->rremote+sf->roffset[i],link->unit,&link->remote[i]);CHKERRQ(ierr);
    ierr = MPI_Type_commit(&link->mine[i]);CHKERRQ(ierr);
    ierr = MPI_Type_commit(&link->remote[i]);CHKERRQ(ierr);
  }
  link->next = sf->link;
  sf->link = link;

  *localtypes = link->mine;
  *remotetypes = link->remote;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscSFGetRanks"
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

.seealso: PetscSFSetGraph(), PetscSFGetDataTypes()
@*/
PetscErrorCode PetscSFGetRanks(PetscSF sf,PetscInt *nranks,const PetscMPIInt **ranks,const PetscInt **roffset,const PetscMPIInt **rmine,const PetscMPIInt **rremote)
{

  PetscFunctionBegin;
  PetscValidHeaderSpecific(sf,PETSCSF_CLASSID,1);
  if (nranks)  *nranks  = sf->nranks;
  if (ranks)   *ranks   = sf->ranks;
  if (roffset) *roffset = sf->roffset;
  if (rmine)   *rmine   = sf->rmine;
  if (rremote) *rremote = sf->rremote;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscSFGetWindow"
/*@C
   PetscSFGetWindow - Get a window for use with a given data type

   Collective on PetscSF

   Input Arguments:
+  sf - star forest
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
PetscErrorCode PetscSFGetWindow(PetscSF sf,MPI_Datatype unit,void *array,PetscBool epoch,PetscMPIInt fenceassert,PetscMPIInt postassert,PetscMPIInt startassert,MPI_Win *win)
{
  PetscErrorCode ierr;
  MPI_Aint lb,lb_true,bytes,bytes_true;
  PetscSFWinLink link;

  PetscFunctionBegin;
  ierr = MPI_Type_get_extent(unit,&lb,&bytes);CHKERRQ(ierr);
  ierr = MPI_Type_get_true_extent(unit,&lb_true,&bytes_true);CHKERRQ(ierr);
  if (lb != 0 || lb_true != 0) SETERRQ(((PetscObject)sf)->comm,PETSC_ERR_SUP,"No support for unit type with nonzero lower bound, write petsc-maint@mcs.anl.gov if you want this feature");
  if (bytes != bytes_true) SETERRQ(((PetscObject)sf)->comm,PETSC_ERR_SUP,"No support for unit type with modified extent, write petsc-maint@mcs.anl.gov if you want this feature");
  ierr = PetscMalloc(sizeof(*link),&link);CHKERRQ(ierr);
  link->bytes = bytes;
  link->addr  = array;
  ierr = MPI_Win_create(array,(MPI_Aint)bytes*sf->nroots,(PetscMPIInt)bytes,MPI_INFO_NULL,((PetscObject)sf)->comm,&link->win);CHKERRQ(ierr);
  link->epoch = epoch;
  link->next = sf->wins;
  link->inuse = PETSC_TRUE;
  sf->wins = link;
  *win = link->win;

  if (epoch) {
    switch (sf->sync) {
    case PETSCSF_SYNCHRONIZATION_FENCE:
      ierr = MPI_Win_fence(fenceassert,*win);CHKERRQ(ierr);
      break;
    case PETSCSF_SYNCHRONIZATION_LOCK: /* Handled outside */
      break;
    case PETSCSF_SYNCHRONIZATION_ACTIVE: {
      MPI_Group ingroup,outgroup;
      ierr = PetscSFGetGroups(sf,&ingroup,&outgroup);CHKERRQ(ierr);
      ierr = MPI_Win_post(ingroup,postassert,*win);CHKERRQ(ierr);
      ierr = MPI_Win_start(outgroup,startassert,*win);CHKERRQ(ierr);
    } break;
    default: SETERRQ(((PetscObject)sf)->comm,PETSC_ERR_PLIB,"Unknown synchronization type");
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
+  sf - star forest
.  unit - data type
-  array - array with which the window is associated

   Output Arguments:
.  win - window

   Level: developer

.seealso: PetscSFGetWindow(), PetscSFRestoreWindow()
@*/
PetscErrorCode PetscSFFindWindow(PetscSF sf,MPI_Datatype unit,const void *array,MPI_Win *win)
{
  PetscSFWinLink link;

  PetscFunctionBegin;
  for (link=sf->wins; link; link=link->next) {
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
+  sf - star forest
.  unit - data type
.  array - array associated with window
.  epoch - close an epoch, must match argument to PetscSFGetWindow()
-  win - window

   Level: developer

.seealso: PetscSFFindWindow()
@*/
PetscErrorCode PetscSFRestoreWindow(PetscSF sf,MPI_Datatype unit,const void *array,PetscBool epoch,PetscMPIInt fenceassert,MPI_Win *win)
{
  PetscErrorCode ierr;
  PetscSFWinLink *p,link;

  PetscFunctionBegin;
  for (p=&sf->wins; *p; p=&(*p)->next) {
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
    switch (sf->sync) {
    case PETSCSF_SYNCHRONIZATION_FENCE:
      ierr = MPI_Win_fence(fenceassert,*win);CHKERRQ(ierr);
      break;
    case PETSCSF_SYNCHRONIZATION_LOCK:
      break;                    /* handled outside */
    case PETSCSF_SYNCHRONIZATION_ACTIVE: {
      ierr = MPI_Win_complete(*win);CHKERRQ(ierr);
      ierr = MPI_Win_wait(*win);CHKERRQ(ierr);
    } break;
    default: SETERRQ(((PetscObject)sf)->comm,PETSC_ERR_PLIB,"Unknown synchronization type");
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
  MPI_Group group;

  PetscFunctionBegin;
  if (sf->ingroup == MPI_GROUP_NULL) {
    PetscInt    i;
    const PetscInt *indegree;
    PetscMPIInt rank,*outranks,*inranks;
    PetscSFNode *remote;
    PetscSF     bgcount;

    /* Compute the number of incoming ranks */
    ierr = PetscMalloc(sf->nranks*sizeof(PetscSFNode),&remote);CHKERRQ(ierr);
    for (i=0; i<sf->nranks; i++) {
      remote[i].rank = sf->ranks[i];
      remote[i].index = 0;
    }
    ierr = PetscSFCreate(((PetscObject)sf)->comm,&bgcount);CHKERRQ(ierr);
    ierr = PetscSFSetSynchronizationType(bgcount,PETSCSF_SYNCHRONIZATION_LOCK);CHKERRQ(ierr); /* or FENCE, ACTIVE here would cause recursion */
    ierr = PetscSFSetGraph(bgcount,1,sf->nranks,PETSC_NULL,PETSC_COPY_VALUES,remote,PETSC_OWN_POINTER);CHKERRQ(ierr);
    ierr = PetscSFComputeDegreeBegin(bgcount,&indegree);CHKERRQ(ierr);
    ierr = PetscSFComputeDegreeEnd(bgcount,&indegree);CHKERRQ(ierr);

    /* Enumerate the incoming ranks */
    ierr = PetscMalloc2(indegree[0],PetscMPIInt,&inranks,sf->nranks,PetscMPIInt,&outranks);CHKERRQ(ierr);
    ierr = MPI_Comm_rank(((PetscObject)sf)->comm,&rank);CHKERRQ(ierr);
    for (i=0; i<sf->nranks; i++) outranks[i] = rank;
    ierr = PetscSFGatherBegin(bgcount,MPI_INT,outranks,inranks);CHKERRQ(ierr);
    ierr = PetscSFGatherEnd(bgcount,MPI_INT,outranks,inranks);CHKERRQ(ierr);
    ierr = MPI_Comm_group(((PetscObject)sf)->comm,&group);CHKERRQ(ierr);
    ierr = MPI_Group_incl(group,indegree[0],inranks,&sf->ingroup);CHKERRQ(ierr);
    ierr = MPI_Group_free(&group);CHKERRQ(ierr);
    ierr = PetscFree2(inranks,outranks);CHKERRQ(ierr);
    ierr = PetscSFDestroy(&bgcount);CHKERRQ(ierr);
  }
  *incoming = sf->ingroup;

  if (sf->outgroup == MPI_GROUP_NULL) {
    ierr = MPI_Comm_group(((PetscObject)sf)->comm,&group);CHKERRQ(ierr);
    ierr = MPI_Group_incl(group,sf->nranks,sf->ranks,&sf->outgroup);CHKERRQ(ierr);
    ierr = MPI_Group_free(&group);CHKERRQ(ierr);
  }
  *outgoing = sf->outgroup;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscSFGetMultiSF"
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
  if (sf->nroots < 0) {
    ierr = PetscSFCreate(((PetscObject)sf)->comm,&sf->multi);CHKERRQ(ierr);
    *multi = sf->multi;
    PetscFunctionReturn(0);
  }
  if (!sf->multi) {
    const PetscInt *indegree;
    PetscInt i,*inoffset,*outones,*outoffset;
    PetscSFNode *remote;
    ierr = PetscSFComputeDegreeBegin(sf,&indegree);CHKERRQ(ierr);
    ierr = PetscSFComputeDegreeEnd(sf,&indegree);CHKERRQ(ierr);
    ierr = PetscMalloc3(sf->nroots+1,PetscInt,&inoffset,sf->nleaves,PetscInt,&outones,sf->nleaves,PetscInt,&outoffset);CHKERRQ(ierr);
    inoffset[0] = 0;
#if 1
    for (i=0; i<sf->nroots; i++) inoffset[i+1] = PetscMax(i+1, inoffset[i] + indegree[i]);
#else
    for (i=0; i<sf->nroots; i++) inoffset[i+1] = inoffset[i] + indegree[i];
#endif
    for (i=0; i<sf->nleaves; i++) outones[i] = 1;
    ierr = PetscSFFetchAndOpBegin(sf,MPIU_INT,inoffset,outones,outoffset,MPIU_SUM);CHKERRQ(ierr);
    ierr = PetscSFFetchAndOpEnd(sf,MPIU_INT,inoffset,outones,outoffset,MPIU_SUM);CHKERRQ(ierr);
    for (i=0; i<sf->nroots; i++) inoffset[i] -= indegree[i]; /* Undo the increment */
#if 0
#if defined(PETSC_USE_DEBUG)                                 /* Check that the expected number of increments occurred */
    for (i=0; i<sf->nroots; i++) {
      if (inoffset[i] + indegree[i] != inoffset[i+1]) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_PLIB,"Incorrect result after PetscSFFetchAndOp");
    }
#endif
#endif
    ierr = PetscMalloc(sf->nleaves*sizeof(*remote),&remote);CHKERRQ(ierr);
    for (i=0; i<sf->nleaves; i++) {
      remote[i].rank = sf->remote[i].rank;
      remote[i].index = outoffset[i];
    }
    ierr = PetscSFCreate(((PetscObject)sf)->comm,&sf->multi);CHKERRQ(ierr);
    ierr = PetscSFSetSynchronizationType(sf->multi,sf->sync);CHKERRQ(ierr);
    ierr = PetscSFSetGraph(sf->multi,inoffset[sf->nroots],sf->nleaves,PETSC_NULL,PETSC_COPY_VALUES,remote,PETSC_OWN_POINTER);CHKERRQ(ierr);
    if (sf->rankorder) {        /* Sort the ranks */
      PetscMPIInt rank;
      PetscInt *inranks,*newoffset,*outranks,*newoutoffset,*tmpoffset,maxdegree;
      PetscSFNode *newremote;
      ierr = MPI_Comm_rank(((PetscObject)sf)->comm,&rank);CHKERRQ(ierr);
      for (i=0,maxdegree=0; i<sf->nroots; i++) maxdegree = PetscMax(maxdegree,indegree[i]);
      ierr = PetscMalloc5(sf->multi->nroots,PetscInt,&inranks,sf->multi->nroots,PetscInt,&newoffset,sf->nleaves,PetscInt,&outranks,sf->nleaves,PetscInt,&newoutoffset,maxdegree,PetscInt,&tmpoffset);CHKERRQ(ierr);
      for (i=0; i<sf->nleaves; i++) outranks[i] = rank;
      ierr = PetscSFReduceBegin(sf->multi,MPIU_INT,outranks,inranks,MPI_REPLACE);CHKERRQ(ierr);
      ierr = PetscSFReduceEnd(sf->multi,MPIU_INT,outranks,inranks,MPI_REPLACE);CHKERRQ(ierr);
      /* Sort the incoming ranks at each vertex, build the inverse map */
      for (i=0; i<sf->nroots; i++) {
        PetscInt j;
        for (j=0; j<indegree[i]; j++) tmpoffset[j] = j;
        ierr = PetscSortIntWithArray(indegree[i],inranks+inoffset[i],tmpoffset);CHKERRQ(ierr);
        for (j=0; j<indegree[i]; j++) newoffset[inoffset[i] + tmpoffset[j]] = inoffset[i] + j;
      }
      ierr = PetscSFBcastBegin(sf->multi,MPIU_INT,newoffset,newoutoffset);CHKERRQ(ierr);
      ierr = PetscSFBcastEnd(sf->multi,MPIU_INT,newoffset,newoutoffset);CHKERRQ(ierr);
      ierr = PetscMalloc(sf->nleaves*sizeof(*newremote),&newremote);CHKERRQ(ierr);
      for (i=0; i<sf->nleaves; i++) {
        newremote[i].rank = sf->remote[i].rank;
        newremote[i].index = newoutoffset[i];
      }
      ierr = PetscSFSetGraph(sf->multi,inoffset[sf->nroots],sf->nleaves,PETSC_NULL,PETSC_COPY_VALUES,newremote,PETSC_OWN_POINTER);CHKERRQ(ierr);
      ierr = PetscFree5(inranks,newoffset,outranks,newoutoffset,tmpoffset);CHKERRQ(ierr);
    }
    ierr = PetscFree3(inoffset,outones,outoffset);CHKERRQ(ierr);
  }
  *multi = sf->multi;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscSFCreateEmbeddedSF"
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
  PetscErrorCode ierr;
  PetscInt i,nleaves,*ilocal,*rootdata,*leafdata;
  PetscSFNode *iremote;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(sf,PETSCSF_CLASSID,1);
  if (nroots) PetscValidPointer(selected,3);
  PetscValidPointer(newsf,4);
  ierr = PetscMalloc2(sf->nroots,PetscInt,&rootdata,sf->nleaves,PetscInt,&leafdata);CHKERRQ(ierr);
  ierr = PetscMemzero(rootdata,sf->nroots*sizeof(PetscInt));CHKERRQ(ierr);
  ierr = PetscMemzero(leafdata,sf->nleaves*sizeof(PetscInt));CHKERRQ(ierr);
  for (i=0; i<nroots; i++) rootdata[selected[i]] = 1;
  ierr = PetscSFBcastBegin(sf,MPIU_INT,rootdata,leafdata);CHKERRQ(ierr);
  ierr = PetscSFBcastEnd(sf,MPIU_INT,rootdata,leafdata);CHKERRQ(ierr);

  for (i=0,nleaves=0; i<sf->nleaves; i++) nleaves += leafdata[i];
  ierr = PetscMalloc(nleaves*sizeof(PetscInt),&ilocal);CHKERRQ(ierr);
  ierr = PetscMalloc(nleaves*sizeof(PetscSFNode),&iremote);CHKERRQ(ierr);
  for (i=0,nleaves=0; i<sf->nleaves; i++) {
    if (leafdata[i]) {
      ilocal[nleaves]        = sf->mine ? sf->mine[i] : i;
      iremote[nleaves].rank  = sf->remote[i].rank;
      iremote[nleaves].index = sf->remote[i].index;
      nleaves++;
    }
  }
  ierr = PetscSFCreate(((PetscObject)sf)->comm,newsf);CHKERRQ(ierr);
  ierr = PetscSFSetSynchronizationType(*newsf,sf->sync);CHKERRQ(ierr);
  ierr = PetscSFSetGraph(*newsf,sf->nroots,nleaves,ilocal,PETSC_OWN_POINTER,iremote,PETSC_OWN_POINTER);CHKERRQ(ierr);
  ierr = PetscFree2(rootdata,leafdata);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscSFOpTranslate"
/* Built-in MPI_Ops act elementwise inside MPI_Accumulate, but cannot be used with composite types inside collectives (MPI_Allreduce) */
static PetscErrorCode PetscSFOpTranslate(MPI_Op *op)
{

  PetscFunctionBegin;
  if (*op == MPIU_SUM) *op = MPI_SUM;
  else if (*op == MPIU_MAX) *op = MPI_MAX;
  else if (*op == MPIU_MIN) *op = MPI_MIN;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscSFBcastBegin"
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
  PetscErrorCode     ierr;
  PetscInt           i,nranks;
  const PetscMPIInt  *ranks;
  const MPI_Datatype *mine,*remote;
  MPI_Win            win;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(sf,PETSCSF_CLASSID,1);
  PetscSFCheckGraphSet(sf,1);
  ierr = PetscSFGetRanks(sf,&nranks,&ranks,PETSC_NULL,PETSC_NULL,PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscSFGetDataTypes(sf,unit,&mine,&remote);CHKERRQ(ierr);
  ierr = PetscSFGetWindow(sf,unit,(void*)rootdata,PETSC_TRUE,MPI_MODE_NOPUT|MPI_MODE_NOPRECEDE,MPI_MODE_NOPUT,0,&win);CHKERRQ(ierr);
  for (i=0; i<nranks; i++) {
    if (sf->sync == PETSCSF_SYNCHRONIZATION_LOCK) {ierr = MPI_Win_lock(MPI_LOCK_SHARED,ranks[i],MPI_MODE_NOCHECK,win);CHKERRQ(ierr);}
    ierr = MPI_Get(leafdata,1,mine[i],ranks[i],0,1,remote[i],win);CHKERRQ(ierr);
    if (sf->sync == PETSCSF_SYNCHRONIZATION_LOCK) {ierr = MPI_Win_unlock(ranks[i],win);CHKERRQ(ierr);}
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscSFBcastEnd"
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
  MPI_Win win;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(sf,PETSCSF_CLASSID,1);
  PetscSFCheckGraphSet(sf,1);
  ierr = PetscSFFindWindow(sf,unit,rootdata,&win);CHKERRQ(ierr);
  ierr = PetscSFRestoreWindow(sf,unit,rootdata,PETSC_TRUE,MPI_MODE_NOSTORE|MPI_MODE_NOSUCCEED,&win);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscSFReduceBegin"
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
  PetscErrorCode     ierr;
  PetscInt           i,nranks;
  const PetscMPIInt  *ranks;
  const MPI_Datatype *mine,*remote;
  MPI_Win            win;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(sf,PETSCSF_CLASSID,1);
  PetscSFCheckGraphSet(sf,1);
  ierr = PetscSFGetRanks(sf,&nranks,&ranks,PETSC_NULL,PETSC_NULL,PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscSFGetDataTypes(sf,unit,&mine,&remote);CHKERRQ(ierr);
  ierr = PetscSFOpTranslate(&op);CHKERRQ(ierr);
  ierr = PetscSFGetWindow(sf,unit,rootdata,PETSC_TRUE,MPI_MODE_NOPRECEDE,0,0,&win);CHKERRQ(ierr);
  for (i=0; i<nranks; i++) {
    if (sf->sync == PETSCSF_SYNCHRONIZATION_LOCK) {ierr = MPI_Win_lock(MPI_LOCK_SHARED,ranks[i],MPI_MODE_NOCHECK,win);CHKERRQ(ierr);}
    ierr = MPI_Accumulate((void*)leafdata,1,mine[i],ranks[i],0,1,remote[i],op,win);CHKERRQ(ierr);
    if (sf->sync == PETSCSF_SYNCHRONIZATION_LOCK) {ierr = MPI_Win_unlock(ranks[i],win);CHKERRQ(ierr);}
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscSFReduceEnd"
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
  MPI_Win win;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(sf,PETSCSF_CLASSID,1);
  PetscSFCheckGraphSet(sf,1);
  if (!sf->wins) {PetscFunctionReturn(0);}
  ierr = PetscSFFindWindow(sf,unit,rootdata,&win);CHKERRQ(ierr);
  ierr = MPI_Win_fence(MPI_MODE_NOSUCCEED,win);CHKERRQ(ierr);
  ierr = PetscSFRestoreWindow(sf,unit,rootdata,PETSC_TRUE,MPI_MODE_NOSUCCEED,&win);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscSFComputeDegreeBegin"
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
  if (!sf->degree) {
    PetscInt i;
    ierr = PetscMalloc(sf->nroots*sizeof(PetscInt),&sf->degree);CHKERRQ(ierr);
    ierr = PetscMalloc(sf->nleaves*sizeof(PetscInt),&sf->degreetmp);CHKERRQ(ierr);
    for (i=0; i<sf->nroots; i++) sf->degree[i] = 0;
    for (i=0; i<sf->nleaves; i++) sf->degreetmp[i] = 1;
    ierr = PetscSFReduceBegin(sf,MPIU_INT,sf->degreetmp,sf->degree,MPIU_SUM);CHKERRQ(ierr);
  }
  *degree = PETSC_NULL;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscSFComputeDegreeEnd"
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
    ierr = PetscSFReduceEnd(sf,MPIU_INT,sf->degreetmp,sf->degree,MPIU_SUM);CHKERRQ(ierr);
    ierr = PetscFree(sf->degreetmp);CHKERRQ(ierr);
    sf->degreeknown = PETSC_TRUE;
  }
  *degree = sf->degree;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscSFFetchAndOpBegin"
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
  PetscInt           i,nranks;
  const PetscMPIInt  *ranks;
  const MPI_Datatype *mine,*remote;
  MPI_Win            win;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(sf,PETSCSF_CLASSID,1);
  PetscSFCheckGraphSet(sf,1);
  ierr = PetscSFGetRanks(sf,&nranks,&ranks,PETSC_NULL,PETSC_NULL,PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscSFGetDataTypes(sf,unit,&mine,&remote);CHKERRQ(ierr);
  ierr = PetscSFOpTranslate(&op);CHKERRQ(ierr);
  ierr = PetscSFGetWindow(sf,unit,rootdata,PETSC_FALSE,0,0,0,&win);CHKERRQ(ierr);
  for (i=0; i<sf->nranks; i++) {
    ierr = MPI_Win_lock(MPI_LOCK_EXCLUSIVE,sf->ranks[i],0,win);CHKERRQ(ierr);
    ierr = MPI_Get(leafupdate,1,mine[i],ranks[i],0,1,remote[i],win);CHKERRQ(ierr);
    ierr = MPI_Accumulate((void*)leafdata,1,mine[i],ranks[i],0,1,remote[i],op,win);CHKERRQ(ierr);
    ierr = MPI_Win_unlock(ranks[i],win);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscSFFetchAndOpEnd"
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
  MPI_Win        win;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(sf,PETSCSF_CLASSID,1);
  PetscSFCheckGraphSet(sf,1);
  ierr = PetscSFFindWindow(sf,unit,rootdata,&win);CHKERRQ(ierr);
  /* Nothing to do currently because MPI_LOCK_EXCLUSIVE is used in PetscSFFetchAndOpBegin(), rendering this implementation synchronous. */
  ierr = PetscSFRestoreWindow(sf,unit,rootdata,PETSC_FALSE,0,&win);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscSFGatherBegin"
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
  ierr = PetscSFGetMultiSF(sf,&multi);CHKERRQ(ierr);
  ierr = PetscSFReduceBegin(multi,unit,leafdata,multirootdata,MPI_REPLACE);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscSFGatherEnd"
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
  ierr = PetscSFGetMultiSF(sf,&multi);CHKERRQ(ierr);
  ierr = PetscSFReduceEnd(multi,unit,leafdata,multirootdata,MPI_REPLACE);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscSFScatterBegin"
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
  ierr = PetscSFGetMultiSF(sf,&multi);CHKERRQ(ierr);
  ierr = PetscSFBcastBegin(multi,unit,multirootdata,leafdata);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscSFScatterEnd"
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
  ierr = PetscSFGetMultiSF(sf,&multi);CHKERRQ(ierr);
  ierr = PetscSFBcastEnd(multi,unit,multirootdata,leafdata);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
