/*
  This file defines an "generalized" additive Schwarz preconditioner for any Mat implementation.
  In this version each processor may intersect multiple subdomains and any subdomain may
  intersect multiple processors.  Intersections of subdomains with processors are called *local
  subdomains*.

       N    - total number of distinct global subdomains          (set explicitly in PCGASMSetTotalSubdomains() or implicitly PCGASMSetSubdomains() and then calculated in PCSetUp_GASM())
       n    - actual number of local subdomains on this processor (set in PCGASMSetSubdomains() or calculated in PCGASMSetTotalSubdomains())
       nmax - maximum number of local subdomains per processor    (calculated in PCSetUp_GASM())
*/
#include <petsc/private/pcimpl.h>     /*I "petscpc.h" I*/
#include <petscdm.h>

typedef struct {
  PetscInt    N,n,nmax;
  PetscInt    overlap;                  /* overlap requested by user */
  PCGASMType  type;                     /* use reduced interpolation, restriction or both */
  PetscBool   type_set;                 /* if user set this value (so won't change it for symmetric problems) */
  PetscBool   same_subdomain_solvers;   /* flag indicating whether all local solvers are same */
  PetscBool   sort_indices;             /* flag to sort subdomain indices */
  PetscBool   user_subdomains;          /* whether the user set explicit subdomain index sets -- keep them on PCReset() */
  PetscBool   dm_subdomains;            /* whether DM is allowed to define subdomains */
  PetscBool   hierarchicalpartitioning;
  IS          *ois;                     /* index sets that define the outer (conceptually, overlapping) subdomains */
  IS          *iis;                     /* index sets that define the inner (conceptually, nonoverlapping) subdomains */
  KSP         *ksp;                     /* linear solvers for each subdomain */
  Mat         *pmat;                    /* subdomain block matrices */
  Vec         gx,gy;                    /* Merged work vectors */
  Vec         *x,*y;                    /* Split work vectors; storage aliases pieces of storage of the above merged vectors. */
  VecScatter  gorestriction;            /* merged restriction to disjoint union of outer subdomains */
  VecScatter  girestriction;            /* merged restriction to disjoint union of inner subdomains */
  VecScatter  pctoouter;
  IS          permutationIS;
  Mat         permutationP;
  Mat         pcmat;
  Vec         pcx,pcy;
} PC_GASM;

static PetscErrorCode  PCGASMComputeGlobalSubdomainNumbering_Private(PC pc,PetscInt **numbering,PetscInt **permutation)
{
  PC_GASM        *osm = (PC_GASM*)pc->data;
  PetscInt       i;

  PetscFunctionBegin;
  /* Determine the number of globally-distinct subdomains and compute a global numbering for them. */
  PetscCall(PetscMalloc2(osm->n,numbering,osm->n,permutation));
  PetscCall(PetscObjectsListGetGlobalNumbering(PetscObjectComm((PetscObject)pc),osm->n,(PetscObject*)osm->iis,NULL,*numbering));
  for (i = 0; i < osm->n; ++i) (*permutation)[i] = i;
  PetscCall(PetscSortIntWithPermutation(osm->n,*numbering,*permutation));
  PetscFunctionReturn(0);
}

static PetscErrorCode  PCGASMSubdomainView_Private(PC pc, PetscInt i, PetscViewer viewer)
{
  PC_GASM        *osm = (PC_GASM*)pc->data;
  PetscInt       j,nidx;
  const PetscInt *idx;
  PetscViewer    sviewer;
  char           *cidx;

  PetscFunctionBegin;
  PetscCheck(i >= 0 && i < osm->n,PetscObjectComm((PetscObject)viewer), PETSC_ERR_ARG_WRONG, "Invalid subdomain %" PetscInt_FMT ": must nonnegative and less than %" PetscInt_FMT, i, osm->n);
  /* Inner subdomains. */
  PetscCall(ISGetLocalSize(osm->iis[i], &nidx));
  /*
   No more than 15 characters per index plus a space.
   PetscViewerStringSPrintf requires a string of size at least 2, so use (nidx+1) instead of nidx,
   in case nidx == 0. That will take care of the space for the trailing '\0' as well.
   For nidx == 0, the whole string 16 '\0'.
   */
#define len  16*(nidx+1)+1
  PetscCall(PetscMalloc1(len, &cidx));
  PetscCall(PetscViewerStringOpen(PETSC_COMM_SELF, cidx, len, &sviewer));
#undef len
  PetscCall(ISGetIndices(osm->iis[i], &idx));
  for (j = 0; j < nidx; ++j) {
    PetscCall(PetscViewerStringSPrintf(sviewer, "%" PetscInt_FMT " ", idx[j]));
  }
  PetscCall(ISRestoreIndices(osm->iis[i],&idx));
  PetscCall(PetscViewerDestroy(&sviewer));
  PetscCall(PetscViewerASCIIPrintf(viewer, "Inner subdomain:\n"));
  PetscCall(PetscViewerFlush(viewer));
  PetscCall(PetscViewerASCIIPushSynchronized(viewer));
  PetscCall(PetscViewerASCIISynchronizedPrintf(viewer, "%s", cidx));
  PetscCall(PetscViewerFlush(viewer));
  PetscCall(PetscViewerASCIIPopSynchronized(viewer));
  PetscCall(PetscViewerASCIIPrintf(viewer, "\n"));
  PetscCall(PetscViewerFlush(viewer));
  PetscCall(PetscFree(cidx));
  /* Outer subdomains. */
  PetscCall(ISGetLocalSize(osm->ois[i], &nidx));
  /*
   No more than 15 characters per index plus a space.
   PetscViewerStringSPrintf requires a string of size at least 2, so use (nidx+1) instead of nidx,
   in case nidx == 0. That will take care of the space for the trailing '\0' as well.
   For nidx == 0, the whole string 16 '\0'.
   */
#define len  16*(nidx+1)+1
  PetscCall(PetscMalloc1(len, &cidx));
  PetscCall(PetscViewerStringOpen(PETSC_COMM_SELF, cidx, len, &sviewer));
#undef len
  PetscCall(ISGetIndices(osm->ois[i], &idx));
  for (j = 0; j < nidx; ++j) {
    PetscCall(PetscViewerStringSPrintf(sviewer,"%" PetscInt_FMT " ", idx[j]));
  }
  PetscCall(PetscViewerDestroy(&sviewer));
  PetscCall(ISRestoreIndices(osm->ois[i],&idx));
  PetscCall(PetscViewerASCIIPrintf(viewer, "Outer subdomain:\n"));
  PetscCall(PetscViewerFlush(viewer));
  PetscCall(PetscViewerASCIIPushSynchronized(viewer));
  PetscCall(PetscViewerASCIISynchronizedPrintf(viewer, "%s", cidx));
  PetscCall(PetscViewerFlush(viewer));
  PetscCall(PetscViewerASCIIPopSynchronized(viewer));
  PetscCall(PetscViewerASCIIPrintf(viewer, "\n"));
  PetscCall(PetscViewerFlush(viewer));
  PetscCall(PetscFree(cidx));
  PetscFunctionReturn(0);
}

static PetscErrorCode  PCGASMPrintSubdomains(PC pc)
{
  PC_GASM        *osm = (PC_GASM*)pc->data;
  const char     *prefix;
  char           fname[PETSC_MAX_PATH_LEN+1];
  PetscInt       l, d, count;
  PetscBool      found;
  PetscViewer    viewer, sviewer = NULL;
  PetscInt       *numbering,*permutation;/* global numbering of locally-supported subdomains and the permutation from the local ordering */

  PetscFunctionBegin;
  PetscCall(PCGetOptionsPrefix(pc,&prefix));
  PetscCall(PetscOptionsHasName(NULL,prefix,"-pc_gasm_print_subdomains",&found));
  if (!found) PetscFunctionReturn(0);
  PetscCall(PetscOptionsGetString(NULL,prefix,"-pc_gasm_print_subdomains",fname,sizeof(fname),&found));
  if (!found) PetscCall(PetscStrcpy(fname,"stdout"));
  PetscCall(PetscViewerASCIIOpen(PetscObjectComm((PetscObject)pc),fname,&viewer));
  /*
   Make sure the viewer has a name. Otherwise this may cause a deadlock or other weird errors when creating a subcomm viewer:
   the subcomm viewer will attempt to inherit the viewer's name, which, if not set, will be constructed collectively on the comm.
  */
  PetscCall(PetscObjectName((PetscObject)viewer));
  l    = 0;
  PetscCall(PCGASMComputeGlobalSubdomainNumbering_Private(pc,&numbering,&permutation));
  for (count = 0; count < osm->N; ++count) {
    /* Now let subdomains go one at a time in the global numbering order and print their subdomain/solver info. */
    if (l<osm->n) {
      d = permutation[l]; /* d is the local number of the l-th smallest (in the global ordering) among the locally supported subdomains */
      if (numbering[d] == count) {
        PetscCall(PetscViewerGetSubViewer(viewer,((PetscObject)osm->ois[d])->comm, &sviewer));
        PetscCall(PCGASMSubdomainView_Private(pc,d,sviewer));
        PetscCall(PetscViewerRestoreSubViewer(viewer,((PetscObject)osm->ois[d])->comm, &sviewer));
        ++l;
      }
    }
    PetscCallMPI(MPI_Barrier(PetscObjectComm((PetscObject)pc)));
  }
  PetscCall(PetscFree2(numbering,permutation));
  PetscCall(PetscViewerDestroy(&viewer));
  PetscFunctionReturn(0);
}

static PetscErrorCode PCView_GASM(PC pc,PetscViewer viewer)
{
  PC_GASM        *osm = (PC_GASM*)pc->data;
  const char     *prefix;
  PetscMPIInt    rank, size;
  PetscInt       bsz;
  PetscBool      iascii,view_subdomains=PETSC_FALSE;
  PetscViewer    sviewer;
  PetscInt       count, l;
  char           overlap[256]     = "user-defined overlap";
  char           gsubdomains[256] = "unknown total number of subdomains";
  char           msubdomains[256] = "unknown max number of local subdomains";
  PetscInt       *numbering,*permutation;/* global numbering of locally-supported subdomains and the permutation from the local ordering */

  PetscFunctionBegin;
  PetscCallMPI(MPI_Comm_size(PetscObjectComm((PetscObject)pc), &size));
  PetscCallMPI(MPI_Comm_rank(PetscObjectComm((PetscObject)pc), &rank));

  if (osm->overlap >= 0) {
    PetscCall(PetscSNPrintf(overlap,sizeof(overlap),"requested amount of overlap = %" PetscInt_FMT,osm->overlap));
  }
  if (osm->N != PETSC_DETERMINE) {
    PetscCall(PetscSNPrintf(gsubdomains, sizeof(gsubdomains), "total number of subdomains = %" PetscInt_FMT,osm->N));
  }
  if (osm->nmax != PETSC_DETERMINE) {
    PetscCall(PetscSNPrintf(msubdomains,sizeof(msubdomains),"max number of local subdomains = %" PetscInt_FMT,osm->nmax));
  }

  PetscCall(PCGetOptionsPrefix(pc,&prefix));
  PetscCall(PetscOptionsGetBool(NULL,prefix,"-pc_gasm_view_subdomains",&view_subdomains,NULL));

  PetscCall(PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&iascii));
  if (iascii) {
    /*
     Make sure the viewer has a name. Otherwise this may cause a deadlock when creating a subcomm viewer:
     the subcomm viewer will attempt to inherit the viewer's name, which, if not set, will be constructed
     collectively on the comm.
     */
    PetscCall(PetscObjectName((PetscObject)viewer));
    PetscCall(PetscViewerASCIIPrintf(viewer,"  Restriction/interpolation type: %s\n",PCGASMTypes[osm->type]));
    PetscCall(PetscViewerASCIIPrintf(viewer,"  %s\n",overlap));
    PetscCall(PetscViewerASCIIPrintf(viewer,"  %s\n",gsubdomains));
    PetscCall(PetscViewerASCIIPrintf(viewer,"  %s\n",msubdomains));
    PetscCall(PetscViewerASCIIPushSynchronized(viewer));
    PetscCall(PetscViewerASCIISynchronizedPrintf(viewer,"  [%d|%d] number of locally-supported subdomains = %" PetscInt_FMT "\n",rank,size,osm->n));
    PetscCall(PetscViewerFlush(viewer));
    PetscCall(PetscViewerASCIIPopSynchronized(viewer));
    /* Cannot take advantage of osm->same_subdomain_solvers without a global numbering of subdomains. */
    PetscCall(PetscViewerASCIIPrintf(viewer,"  Subdomain solver info is as follows:\n"));
    PetscCall(PetscViewerASCIIPushTab(viewer));
    PetscCall(PetscViewerASCIIPrintf(viewer,"  - - - - - - - - - - - - - - - - - -\n"));
    /* Make sure that everybody waits for the banner to be printed. */
    PetscCallMPI(MPI_Barrier(PetscObjectComm((PetscObject)viewer)));
    /* Now let subdomains go one at a time in the global numbering order and print their subdomain/solver info. */
    PetscCall(PCGASMComputeGlobalSubdomainNumbering_Private(pc,&numbering,&permutation));
    l = 0;
    for (count = 0; count < osm->N; ++count) {
      PetscMPIInt srank, ssize;
      if (l<osm->n) {
        PetscInt d = permutation[l]; /* d is the local number of the l-th smallest (in the global ordering) among the locally supported subdomains */
        if (numbering[d] == count) {
          PetscCallMPI(MPI_Comm_size(((PetscObject)osm->ois[d])->comm, &ssize));
          PetscCallMPI(MPI_Comm_rank(((PetscObject)osm->ois[d])->comm, &srank));
          PetscCall(PetscViewerGetSubViewer(viewer,((PetscObject)osm->ois[d])->comm, &sviewer));
          PetscCall(ISGetLocalSize(osm->ois[d],&bsz));
          PetscCall(PetscViewerASCIISynchronizedPrintf(sviewer,"  [%d|%d] (subcomm [%d|%d]) local subdomain number %" PetscInt_FMT ", local size = %" PetscInt_FMT "\n",rank,size,srank,ssize,d,bsz));
          PetscCall(PetscViewerFlush(sviewer));
          if (view_subdomains) PetscCall(PCGASMSubdomainView_Private(pc,d,sviewer));
          if (!pc->setupcalled) {
            PetscCall(PetscViewerASCIIPrintf(sviewer, "  Solver not set up yet: PCSetUp() not yet called\n"));
          } else {
            PetscCall(KSPView(osm->ksp[d],sviewer));
          }
          PetscCall(PetscViewerASCIIPrintf(sviewer,"  - - - - - - - - - - - - - - - - - -\n"));
          PetscCall(PetscViewerFlush(sviewer));
          PetscCall(PetscViewerRestoreSubViewer(viewer,((PetscObject)osm->ois[d])->comm, &sviewer));
          ++l;
        }
      }
      PetscCallMPI(MPI_Barrier(PetscObjectComm((PetscObject)pc)));
    }
    PetscCall(PetscFree2(numbering,permutation));
    PetscCall(PetscViewerASCIIPopTab(viewer));
    PetscCall(PetscViewerFlush(viewer));
    /* this line is needed to match the extra PetscViewerASCIIPushSynchronized() in PetscViewerGetSubViewer() */
    PetscCall(PetscViewerASCIIPopSynchronized(viewer));
  }
  PetscFunctionReturn(0);
}

PETSC_INTERN PetscErrorCode  PCGASMCreateLocalSubdomains(Mat A, PetscInt nloc, IS *iis[]);

PetscErrorCode PCGASMSetHierarchicalPartitioning(PC pc)
{
   PC_GASM              *osm = (PC_GASM*)pc->data;
   MatPartitioning       part;
   MPI_Comm              comm;
   PetscMPIInt           size;
   PetscInt              nlocalsubdomains,fromrows_localsize;
   IS                    partitioning,fromrows,isn;
   Vec                   outervec;

   PetscFunctionBegin;
   PetscCall(PetscObjectGetComm((PetscObject)pc,&comm));
   PetscCallMPI(MPI_Comm_size(comm,&size));
   /* we do not need a hierarchical partitioning when
    * the total number of subdomains is consistent with
    * the number of MPI tasks.
    * For the following cases, we do not need to use HP
    * */
   if (osm->N==PETSC_DETERMINE || osm->N>=size || osm->N==1) PetscFunctionReturn(0);
   PetscCheck(size%osm->N == 0,PETSC_COMM_WORLD,PETSC_ERR_ARG_INCOMP,"have to specify the total number of subdomains %" PetscInt_FMT " to be a factor of the number of processors %d ",osm->N,size);
   nlocalsubdomains = size/osm->N;
   osm->n           = 1;
   PetscCall(MatPartitioningCreate(comm,&part));
   PetscCall(MatPartitioningSetAdjacency(part,pc->pmat));
   PetscCall(MatPartitioningSetType(part,MATPARTITIONINGHIERARCH));
   PetscCall(MatPartitioningHierarchicalSetNcoarseparts(part,osm->N));
   PetscCall(MatPartitioningHierarchicalSetNfineparts(part,nlocalsubdomains));
   PetscCall(MatPartitioningSetFromOptions(part));
   /* get new processor owner number of each vertex */
   PetscCall(MatPartitioningApply(part,&partitioning));
   PetscCall(ISBuildTwoSided(partitioning,NULL,&fromrows));
   PetscCall(ISPartitioningToNumbering(partitioning,&isn));
   PetscCall(ISDestroy(&isn));
   PetscCall(ISGetLocalSize(fromrows,&fromrows_localsize));
   PetscCall(MatPartitioningDestroy(&part));
   PetscCall(MatCreateVecs(pc->pmat,&outervec,NULL));
   PetscCall(VecCreateMPI(comm,fromrows_localsize,PETSC_DETERMINE,&(osm->pcx)));
   PetscCall(VecDuplicate(osm->pcx,&(osm->pcy)));
   PetscCall(VecScatterCreate(osm->pcx,NULL,outervec,fromrows,&(osm->pctoouter)));
   PetscCall(MatCreateSubMatrix(pc->pmat,fromrows,fromrows,MAT_INITIAL_MATRIX,&(osm->permutationP)));
   PetscCall(PetscObjectReference((PetscObject)fromrows));
   osm->permutationIS = fromrows;
   osm->pcmat =  pc->pmat;
   PetscCall(PetscObjectReference((PetscObject)osm->permutationP));
   pc->pmat = osm->permutationP;
   PetscCall(VecDestroy(&outervec));
   PetscCall(ISDestroy(&fromrows));
   PetscCall(ISDestroy(&partitioning));
   osm->n           = PETSC_DETERMINE;
   PetscFunctionReturn(0);
}

static PetscErrorCode PCSetUp_GASM(PC pc)
{
  PC_GASM        *osm = (PC_GASM*)pc->data;
  PetscInt       i,nInnerIndices,nTotalInnerIndices;
  PetscMPIInt    rank, size;
  MatReuse       scall = MAT_REUSE_MATRIX;
  KSP            ksp;
  PC             subpc;
  const char     *prefix,*pprefix;
  Vec            x,y;
  PetscInt       oni;       /* Number of indices in the i-th local outer subdomain.               */
  const PetscInt *oidxi;    /* Indices from the i-th subdomain local outer subdomain.             */
  PetscInt       on;        /* Number of indices in the disjoint union of local outer subdomains. */
  PetscInt       *oidx;     /* Indices in the disjoint union of local outer subdomains. */
  IS             gois;      /* Disjoint union the global indices of outer subdomains.             */
  IS             goid;      /* Identity IS of the size of the disjoint union of outer subdomains. */
  PetscScalar    *gxarray, *gyarray;
  PetscInt       gostart;   /* Start of locally-owned indices in the vectors -- osm->gx,osm->gy -- over the disjoint union of outer subdomains. */
  PetscInt       num_subdomains    = 0;
  DM             *subdomain_dm     = NULL;
  char           **subdomain_names = NULL;
  PetscInt       *numbering;

  PetscFunctionBegin;
  PetscCallMPI(MPI_Comm_size(PetscObjectComm((PetscObject)pc),&size));
  PetscCallMPI(MPI_Comm_rank(PetscObjectComm((PetscObject)pc),&rank));
  if (!pc->setupcalled) {
        /* use a hierarchical partitioning */
    if (osm->hierarchicalpartitioning) PetscCall(PCGASMSetHierarchicalPartitioning(pc));
    if (osm->n == PETSC_DETERMINE) {
      if (osm->N != PETSC_DETERMINE) {
           /* No local subdomains given, but the desired number of total subdomains is known, so construct them accordingly. */
           PetscCall(PCGASMCreateSubdomains(pc->pmat,osm->N,&osm->n,&osm->iis));
      } else if (osm->dm_subdomains && pc->dm) {
        /* try pc->dm next, if allowed */
        PetscInt  d;
        IS       *inner_subdomain_is, *outer_subdomain_is;
        PetscCall(DMCreateDomainDecomposition(pc->dm, &num_subdomains, &subdomain_names, &inner_subdomain_is, &outer_subdomain_is, &subdomain_dm));
        if (num_subdomains) PetscCall(PCGASMSetSubdomains(pc, num_subdomains, inner_subdomain_is, outer_subdomain_is));
        for (d = 0; d < num_subdomains; ++d) {
          if (inner_subdomain_is) PetscCall(ISDestroy(&inner_subdomain_is[d]));
          if (outer_subdomain_is) PetscCall(ISDestroy(&outer_subdomain_is[d]));
        }
        PetscCall(PetscFree(inner_subdomain_is));
        PetscCall(PetscFree(outer_subdomain_is));
      } else {
        /* still no subdomains; use one per processor */
        osm->nmax = osm->n = 1;
        PetscCallMPI(MPI_Comm_size(PetscObjectComm((PetscObject)pc),&size));
        osm->N    = size;
        PetscCall(PCGASMCreateLocalSubdomains(pc->pmat,osm->n,&osm->iis));
      }
    }
    if (!osm->iis) {
      /*
       osm->n was set in PCGASMSetSubdomains(), but the actual subdomains have not been supplied.
       We create the requisite number of local inner subdomains and then expand them into
       out subdomains, if necessary.
       */
      PetscCall(PCGASMCreateLocalSubdomains(pc->pmat,osm->n,&osm->iis));
    }
    if (!osm->ois) {
      /*
            Initially make outer subdomains the same as inner subdomains. If nonzero additional overlap
            has been requested, copy the inner subdomains over so they can be modified.
      */
      PetscCall(PetscMalloc1(osm->n,&osm->ois));
      for (i=0; i<osm->n; ++i) {
        if (osm->overlap > 0 && osm->N>1) { /* With positive overlap, osm->iis[i] will be modified */
          PetscCall(ISDuplicate(osm->iis[i],(osm->ois)+i));
          PetscCall(ISCopy(osm->iis[i],osm->ois[i]));
        } else {
          PetscCall(PetscObjectReference((PetscObject)((osm->iis)[i])));
          osm->ois[i] = osm->iis[i];
        }
      }
      if (osm->overlap>0 && osm->N>1) {
        /* Extend the "overlapping" regions by a number of steps */
        PetscCall(MatIncreaseOverlapSplit(pc->pmat,osm->n,osm->ois,osm->overlap));
      }
    }

    /* Now the subdomains are defined.  Determine their global and max local numbers, if necessary. */
    if (osm->nmax == PETSC_DETERMINE) {
      PetscMPIInt inwork,outwork;
      /* determine global number of subdomains and the max number of local subdomains */
      inwork     = osm->n;
      PetscCall(MPIU_Allreduce(&inwork,&outwork,1,MPI_INT,MPI_MAX,PetscObjectComm((PetscObject)pc)));
      osm->nmax  = outwork;
    }
    if (osm->N == PETSC_DETERMINE) {
      /* Determine the number of globally-distinct subdomains and compute a global numbering for them. */
      PetscCall(PetscObjectsListGetGlobalNumbering(PetscObjectComm((PetscObject)pc),osm->n,(PetscObject*)osm->ois,&osm->N,NULL));
    }

    if (osm->sort_indices) {
      for (i=0; i<osm->n; i++) {
        PetscCall(ISSort(osm->ois[i]));
        PetscCall(ISSort(osm->iis[i]));
      }
    }
    PetscCall(PCGetOptionsPrefix(pc,&prefix));
    PetscCall(PCGASMPrintSubdomains(pc));

    /*
       Merge the ISs, create merged vectors and restrictions.
     */
    /* Merge outer subdomain ISs and construct a restriction onto the disjoint union of local outer subdomains. */
    on = 0;
    for (i=0; i<osm->n; i++) {
      PetscCall(ISGetLocalSize(osm->ois[i],&oni));
      on  += oni;
    }
    PetscCall(PetscMalloc1(on, &oidx));
    on   = 0;
    /* Merge local indices together */
    for (i=0; i<osm->n; i++) {
      PetscCall(ISGetLocalSize(osm->ois[i],&oni));
      PetscCall(ISGetIndices(osm->ois[i],&oidxi));
      PetscCall(PetscArraycpy(oidx+on,oidxi,oni));
      PetscCall(ISRestoreIndices(osm->ois[i],&oidxi));
      on  += oni;
    }
    PetscCall(ISCreateGeneral(((PetscObject)(pc))->comm,on,oidx,PETSC_OWN_POINTER,&gois));
    nTotalInnerIndices = 0;
    for (i=0; i<osm->n; i++) {
      PetscCall(ISGetLocalSize(osm->iis[i],&nInnerIndices));
      nTotalInnerIndices += nInnerIndices;
    }
    PetscCall(VecCreateMPI(((PetscObject)(pc))->comm,nTotalInnerIndices,PETSC_DETERMINE,&x));
    PetscCall(VecDuplicate(x,&y));

    PetscCall(VecCreateMPI(PetscObjectComm((PetscObject)pc),on,PETSC_DECIDE,&osm->gx));
    PetscCall(VecDuplicate(osm->gx,&osm->gy));
    PetscCall(VecGetOwnershipRange(osm->gx, &gostart, NULL));
    PetscCall(ISCreateStride(PetscObjectComm((PetscObject)pc),on,gostart,1, &goid));
    /* gois might indices not on local */
    PetscCall(VecScatterCreate(x,gois,osm->gx,goid, &(osm->gorestriction)));
    PetscCall(PetscMalloc1(osm->n,&numbering));
    PetscCall(PetscObjectsListGetGlobalNumbering(PetscObjectComm((PetscObject)pc),osm->n,(PetscObject*)osm->ois,NULL,numbering));
    PetscCall(VecDestroy(&x));
    PetscCall(ISDestroy(&gois));

    /* Merge inner subdomain ISs and construct a restriction onto the disjoint union of local inner subdomains. */
    {
      PetscInt        ini;           /* Number of indices the i-th a local inner subdomain. */
      PetscInt        in;            /* Number of indices in the disjoint union of local inner subdomains. */
      PetscInt       *iidx;          /* Global indices in the merged local inner subdomain. */
      PetscInt       *ioidx;         /* Global indices of the disjoint union of inner subdomains within the disjoint union of outer subdomains. */
      IS              giis;          /* IS for the disjoint union of inner subdomains. */
      IS              giois;         /* IS for the disjoint union of inner subdomains within the disjoint union of outer subdomains. */
      PetscScalar    *array;
      const PetscInt *indices;
      PetscInt        k;
      on = 0;
      for (i=0; i<osm->n; i++) {
        PetscCall(ISGetLocalSize(osm->ois[i],&oni));
        on  += oni;
      }
      PetscCall(PetscMalloc1(on, &iidx));
      PetscCall(PetscMalloc1(on, &ioidx));
      PetscCall(VecGetArray(y,&array));
      /* set communicator id to determine where overlap is */
      in   = 0;
      for (i=0; i<osm->n; i++) {
        PetscCall(ISGetLocalSize(osm->iis[i],&ini));
        for (k = 0; k < ini; ++k) {
          array[in+k] = numbering[i];
        }
        in += ini;
      }
      PetscCall(VecRestoreArray(y,&array));
      PetscCall(VecScatterBegin(osm->gorestriction,y,osm->gy,INSERT_VALUES,SCATTER_FORWARD));
      PetscCall(VecScatterEnd(osm->gorestriction,y,osm->gy,INSERT_VALUES,SCATTER_FORWARD));
      PetscCall(VecGetOwnershipRange(osm->gy,&gostart, NULL));
      PetscCall(VecGetArray(osm->gy,&array));
      on  = 0;
      in  = 0;
      for (i=0; i<osm->n; i++) {
        PetscCall(ISGetLocalSize(osm->ois[i],&oni));
        PetscCall(ISGetIndices(osm->ois[i],&indices));
        for (k=0; k<oni; k++) {
          /*  skip overlapping indices to get inner domain */
          if (PetscRealPart(array[on+k]) != numbering[i]) continue;
          iidx[in]    = indices[k];
          ioidx[in++] = gostart+on+k;
        }
        PetscCall(ISRestoreIndices(osm->ois[i], &indices));
        on += oni;
      }
      PetscCall(VecRestoreArray(osm->gy,&array));
      PetscCall(ISCreateGeneral(PetscObjectComm((PetscObject)pc),in,iidx,PETSC_OWN_POINTER,&giis));
      PetscCall(ISCreateGeneral(PetscObjectComm((PetscObject)pc),in,ioidx,PETSC_OWN_POINTER,&giois));
      PetscCall(VecScatterCreate(y,giis,osm->gy,giois,&osm->girestriction));
      PetscCall(VecDestroy(&y));
      PetscCall(ISDestroy(&giis));
      PetscCall(ISDestroy(&giois));
    }
    PetscCall(ISDestroy(&goid));
    PetscCall(PetscFree(numbering));

    /* Create the subdomain work vectors. */
    PetscCall(PetscMalloc1(osm->n,&osm->x));
    PetscCall(PetscMalloc1(osm->n,&osm->y));
    PetscCall(VecGetArray(osm->gx, &gxarray));
    PetscCall(VecGetArray(osm->gy, &gyarray));
    for (i=0, on=0; i<osm->n; ++i, on += oni) {
      PetscInt oNi;
      PetscCall(ISGetLocalSize(osm->ois[i],&oni));
      /* on a sub communicator */
      PetscCall(ISGetSize(osm->ois[i],&oNi));
      PetscCall(VecCreateMPIWithArray(((PetscObject)(osm->ois[i]))->comm,1,oni,oNi,gxarray+on,&osm->x[i]));
      PetscCall(VecCreateMPIWithArray(((PetscObject)(osm->ois[i]))->comm,1,oni,oNi,gyarray+on,&osm->y[i]));
    }
    PetscCall(VecRestoreArray(osm->gx, &gxarray));
    PetscCall(VecRestoreArray(osm->gy, &gyarray));
    /* Create the subdomain solvers */
    PetscCall(PetscMalloc1(osm->n,&osm->ksp));
    for (i=0; i<osm->n; i++) {
      char subprefix[PETSC_MAX_PATH_LEN+1];
      PetscCall(KSPCreate(((PetscObject)(osm->ois[i]))->comm,&ksp));
      PetscCall(KSPSetErrorIfNotConverged(ksp,pc->erroriffailure));
      PetscCall(PetscLogObjectParent((PetscObject)pc,(PetscObject)ksp));
      PetscCall(PetscObjectIncrementTabLevel((PetscObject)ksp,(PetscObject)pc,1));
      PetscCall(KSPSetType(ksp,KSPPREONLY));
      PetscCall(KSPGetPC(ksp,&subpc)); /* Why do we need this here? */
      if (subdomain_dm) {
        PetscCall(KSPSetDM(ksp,subdomain_dm[i]));
        PetscCall(DMDestroy(subdomain_dm+i));
      }
      PetscCall(PCGetOptionsPrefix(pc,&prefix));
      PetscCall(KSPSetOptionsPrefix(ksp,prefix));
      if (subdomain_names && subdomain_names[i]) {
        PetscCall(PetscSNPrintf(subprefix,PETSC_MAX_PATH_LEN,"sub_%s_",subdomain_names[i]));
        PetscCall(KSPAppendOptionsPrefix(ksp,subprefix));
        PetscCall(PetscFree(subdomain_names[i]));
      }
      PetscCall(KSPAppendOptionsPrefix(ksp,"sub_"));
      osm->ksp[i] = ksp;
    }
    PetscCall(PetscFree(subdomain_dm));
    PetscCall(PetscFree(subdomain_names));
    scall = MAT_INITIAL_MATRIX;
  } else { /* if (pc->setupcalled) */
    /*
       Destroy the submatrices from the previous iteration
    */
    if (pc->flag == DIFFERENT_NONZERO_PATTERN) {
      PetscCall(MatDestroyMatrices(osm->n,&osm->pmat));
      scall = MAT_INITIAL_MATRIX;
    }
    if (osm->permutationIS) {
      PetscCall(MatCreateSubMatrix(pc->pmat,osm->permutationIS,osm->permutationIS,scall,&osm->permutationP));
      PetscCall(PetscObjectReference((PetscObject)osm->permutationP));
      osm->pcmat = pc->pmat;
      pc->pmat   = osm->permutationP;
    }
  }

  /*
     Extract the submatrices.
  */
  if (size > 1) {
    PetscCall(MatCreateSubMatricesMPI(pc->pmat,osm->n,osm->ois,osm->ois,scall,&osm->pmat));
  } else {
    PetscCall(MatCreateSubMatrices(pc->pmat,osm->n,osm->ois,osm->ois,scall,&osm->pmat));
  }
  if (scall == MAT_INITIAL_MATRIX) {
    PetscCall(PetscObjectGetOptionsPrefix((PetscObject)pc->pmat,&pprefix));
    for (i=0; i<osm->n; i++) {
      PetscCall(PetscLogObjectParent((PetscObject)pc,(PetscObject)osm->pmat[i]));
      PetscCall(PetscObjectSetOptionsPrefix((PetscObject)osm->pmat[i],pprefix));
    }
  }

  /* Return control to the user so that the submatrices can be modified (e.g., to apply
     different boundary conditions for the submatrices than for the global problem) */
  PetscCall(PCModifySubMatrices(pc,osm->n,osm->ois,osm->ois,osm->pmat,pc->modifysubmatricesP));

  /*
     Loop over submatrices putting them into local ksps
  */
  for (i=0; i<osm->n; i++) {
    PetscCall(KSPSetOperators(osm->ksp[i],osm->pmat[i],osm->pmat[i]));
    PetscCall(KSPGetOptionsPrefix(osm->ksp[i],&prefix));
    PetscCall(MatSetOptionsPrefix(osm->pmat[i],prefix));
    if (!pc->setupcalled) {
      PetscCall(KSPSetFromOptions(osm->ksp[i]));
    }
  }
  if (osm->pcmat) {
    PetscCall(MatDestroy(&pc->pmat));
    pc->pmat   = osm->pcmat;
    osm->pcmat = NULL;
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode PCSetUpOnBlocks_GASM(PC pc)
{
  PC_GASM        *osm = (PC_GASM*)pc->data;
  PetscInt       i;

  PetscFunctionBegin;
  for (i=0; i<osm->n; i++) {
    PetscCall(KSPSetUp(osm->ksp[i]));
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode PCApply_GASM(PC pc,Vec xin,Vec yout)
{
  PC_GASM        *osm = (PC_GASM*)pc->data;
  PetscInt       i;
  Vec            x,y;
  ScatterMode    forward = SCATTER_FORWARD,reverse = SCATTER_REVERSE;

  PetscFunctionBegin;
  if (osm->pctoouter) {
    PetscCall(VecScatterBegin(osm->pctoouter,xin,osm->pcx,INSERT_VALUES,SCATTER_REVERSE));
    PetscCall(VecScatterEnd(osm->pctoouter,xin,osm->pcx,INSERT_VALUES,SCATTER_REVERSE));
    x = osm->pcx;
    y = osm->pcy;
  } else {
    x = xin;
    y = yout;
  }
  /*
     support for limiting the restriction or interpolation only to the inner
     subdomain values (leaving the other values 0).
  */
  if (!(osm->type & PC_GASM_RESTRICT)) {
    /* have to zero the work RHS since scatter may leave some slots empty */
    PetscCall(VecZeroEntries(osm->gx));
    PetscCall(VecScatterBegin(osm->girestriction,x,osm->gx,INSERT_VALUES,forward));
  } else {
    PetscCall(VecScatterBegin(osm->gorestriction,x,osm->gx,INSERT_VALUES,forward));
  }
  PetscCall(VecZeroEntries(osm->gy));
  if (!(osm->type & PC_GASM_RESTRICT)) {
    PetscCall(VecScatterEnd(osm->girestriction,x,osm->gx,INSERT_VALUES,forward));
  } else {
    PetscCall(VecScatterEnd(osm->gorestriction,x,osm->gx,INSERT_VALUES,forward));
  }
  /* do the subdomain solves */
  for (i=0; i<osm->n; ++i) {
    PetscCall(KSPSolve(osm->ksp[i],osm->x[i],osm->y[i]));
    PetscCall(KSPCheckSolve(osm->ksp[i],pc,osm->y[i]));
  }
  /* do we need to zero y? */
  PetscCall(VecZeroEntries(y));
  if (!(osm->type & PC_GASM_INTERPOLATE)) {
    PetscCall(VecScatterBegin(osm->girestriction,osm->gy,y,ADD_VALUES,reverse));
    PetscCall(VecScatterEnd(osm->girestriction,osm->gy,y,ADD_VALUES,reverse));
  } else {
    PetscCall(VecScatterBegin(osm->gorestriction,osm->gy,y,ADD_VALUES,reverse));
    PetscCall(VecScatterEnd(osm->gorestriction,osm->gy,y,ADD_VALUES,reverse));
  }
  if (osm->pctoouter) {
    PetscCall(VecScatterBegin(osm->pctoouter,y,yout,INSERT_VALUES,SCATTER_FORWARD));
    PetscCall(VecScatterEnd(osm->pctoouter,y,yout,INSERT_VALUES,SCATTER_FORWARD));
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode PCMatApply_GASM(PC pc,Mat Xin,Mat Yout)
{
  PC_GASM        *osm = (PC_GASM*)pc->data;
  Mat            X,Y,O=NULL,Z,W;
  Vec            x,y;
  PetscInt       i,m,M,N;
  ScatterMode    forward = SCATTER_FORWARD,reverse = SCATTER_REVERSE;

  PetscFunctionBegin;
  PetscCheck(osm->n == 1,PetscObjectComm((PetscObject)pc),PETSC_ERR_SUP,"Not yet implemented");
  PetscCall(MatGetSize(Xin,NULL,&N));
  if (osm->pctoouter) {
    PetscCall(VecGetLocalSize(osm->pcx,&m));
    PetscCall(VecGetSize(osm->pcx,&M));
    PetscCall(MatCreateDense(PetscObjectComm((PetscObject)osm->ois[0]),m,PETSC_DECIDE,M,N,NULL,&O));
    for (i = 0; i < N; ++i) {
      PetscCall(MatDenseGetColumnVecRead(Xin,i,&x));
      PetscCall(MatDenseGetColumnVecWrite(O,i,&y));
      PetscCall(VecScatterBegin(osm->pctoouter,x,y,INSERT_VALUES,SCATTER_REVERSE));
      PetscCall(VecScatterEnd(osm->pctoouter,x,y,INSERT_VALUES,SCATTER_REVERSE));
      PetscCall(MatDenseRestoreColumnVecWrite(O,i,&y));
      PetscCall(MatDenseRestoreColumnVecRead(Xin,i,&x));
    }
    X = Y = O;
  } else {
    X = Xin;
    Y = Yout;
  }
  /*
     support for limiting the restriction or interpolation only to the inner
     subdomain values (leaving the other values 0).
  */
  PetscCall(VecGetLocalSize(osm->x[0],&m));
  PetscCall(VecGetSize(osm->x[0],&M));
  PetscCall(MatCreateDense(PetscObjectComm((PetscObject)osm->ois[0]),m,PETSC_DECIDE,M,N,NULL,&Z));
  for (i = 0; i < N; ++i) {
    PetscCall(MatDenseGetColumnVecRead(X,i,&x));
    PetscCall(MatDenseGetColumnVecWrite(Z,i,&y));
    if (!(osm->type & PC_GASM_RESTRICT)) {
      /* have to zero the work RHS since scatter may leave some slots empty */
      PetscCall(VecZeroEntries(y));
      PetscCall(VecScatterBegin(osm->girestriction,x,y,INSERT_VALUES,forward));
      PetscCall(VecScatterEnd(osm->girestriction,x,y,INSERT_VALUES,forward));
    } else {
      PetscCall(VecScatterBegin(osm->gorestriction,x,y,INSERT_VALUES,forward));
      PetscCall(VecScatterEnd(osm->gorestriction,x,y,INSERT_VALUES,forward));
    }
    PetscCall(MatDenseRestoreColumnVecWrite(Z,i,&y));
    PetscCall(MatDenseRestoreColumnVecRead(X,i,&x));
  }
  PetscCall(MatCreateDense(PetscObjectComm((PetscObject)osm->ois[0]),m,PETSC_DECIDE,M,N,NULL,&W));
  PetscCall(MatSetOption(Z,MAT_NO_OFF_PROC_ENTRIES,PETSC_TRUE));
  PetscCall(MatAssemblyBegin(Z,MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(Z,MAT_FINAL_ASSEMBLY));
  /* do the subdomain solve */
  PetscCall(KSPMatSolve(osm->ksp[0],Z,W));
  PetscCall(KSPCheckSolve(osm->ksp[0],pc,NULL));
  PetscCall(MatDestroy(&Z));
  /* do we need to zero y? */
  PetscCall(MatZeroEntries(Y));
  for (i = 0; i < N; ++i) {
    PetscCall(MatDenseGetColumnVecWrite(Y,i,&y));
    PetscCall(MatDenseGetColumnVecRead(W,i,&x));
    if (!(osm->type & PC_GASM_INTERPOLATE)) {
      PetscCall(VecScatterBegin(osm->girestriction,x,y,ADD_VALUES,reverse));
      PetscCall(VecScatterEnd(osm->girestriction,x,y,ADD_VALUES,reverse));
    } else {
      PetscCall(VecScatterBegin(osm->gorestriction,x,y,ADD_VALUES,reverse));
      PetscCall(VecScatterEnd(osm->gorestriction,x,y,ADD_VALUES,reverse));
    }
    PetscCall(MatDenseRestoreColumnVecRead(W,i,&x));
    if (osm->pctoouter) {
      PetscCall(MatDenseGetColumnVecWrite(Yout,i,&x));
      PetscCall(VecScatterBegin(osm->pctoouter,y,x,INSERT_VALUES,SCATTER_FORWARD));
      PetscCall(VecScatterEnd(osm->pctoouter,y,x,INSERT_VALUES,SCATTER_FORWARD));
      PetscCall(MatDenseRestoreColumnVecRead(Yout,i,&x));
    }
    PetscCall(MatDenseRestoreColumnVecWrite(Y,i,&y));
  }
  PetscCall(MatDestroy(&W));
  PetscCall(MatDestroy(&O));
  PetscFunctionReturn(0);
}

static PetscErrorCode PCApplyTranspose_GASM(PC pc,Vec xin,Vec yout)
{
  PC_GASM        *osm = (PC_GASM*)pc->data;
  PetscInt       i;
  Vec            x,y;
  ScatterMode    forward = SCATTER_FORWARD,reverse = SCATTER_REVERSE;

  PetscFunctionBegin;
  if (osm->pctoouter) {
   PetscCall(VecScatterBegin(osm->pctoouter,xin,osm->pcx,INSERT_VALUES,SCATTER_REVERSE));
   PetscCall(VecScatterEnd(osm->pctoouter,xin,osm->pcx,INSERT_VALUES,SCATTER_REVERSE));
   x = osm->pcx;
   y = osm->pcy;
  }else{
        x = xin;
        y = yout;
  }
  /*
     Support for limiting the restriction or interpolation to only local
     subdomain values (leaving the other values 0).

     Note: these are reversed from the PCApply_GASM() because we are applying the
     transpose of the three terms
  */
  if (!(osm->type & PC_GASM_INTERPOLATE)) {
    /* have to zero the work RHS since scatter may leave some slots empty */
    PetscCall(VecZeroEntries(osm->gx));
    PetscCall(VecScatterBegin(osm->girestriction,x,osm->gx,INSERT_VALUES,forward));
  } else {
    PetscCall(VecScatterBegin(osm->gorestriction,x,osm->gx,INSERT_VALUES,forward));
  }
  PetscCall(VecZeroEntries(osm->gy));
  if (!(osm->type & PC_GASM_INTERPOLATE)) {
    PetscCall(VecScatterEnd(osm->girestriction,x,osm->gx,INSERT_VALUES,forward));
  } else {
    PetscCall(VecScatterEnd(osm->gorestriction,x,osm->gx,INSERT_VALUES,forward));
  }
  /* do the local solves */
  for (i=0; i<osm->n; ++i) { /* Note that the solves are local, so we can go to osm->n, rather than osm->nmax. */
    PetscCall(KSPSolveTranspose(osm->ksp[i],osm->x[i],osm->y[i]));
    PetscCall(KSPCheckSolve(osm->ksp[i],pc,osm->y[i]));
  }
  PetscCall(VecZeroEntries(y));
  if (!(osm->type & PC_GASM_RESTRICT)) {
    PetscCall(VecScatterBegin(osm->girestriction,osm->gy,y,ADD_VALUES,reverse));
    PetscCall(VecScatterEnd(osm->girestriction,osm->gy,y,ADD_VALUES,reverse));
  } else {
    PetscCall(VecScatterBegin(osm->gorestriction,osm->gy,y,ADD_VALUES,reverse));
    PetscCall(VecScatterEnd(osm->gorestriction,osm->gy,y,ADD_VALUES,reverse));
  }
  if (osm->pctoouter) {
   PetscCall(VecScatterBegin(osm->pctoouter,y,yout,INSERT_VALUES,SCATTER_FORWARD));
   PetscCall(VecScatterEnd(osm->pctoouter,y,yout,INSERT_VALUES,SCATTER_FORWARD));
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode PCReset_GASM(PC pc)
{
  PC_GASM        *osm = (PC_GASM*)pc->data;
  PetscInt       i;

  PetscFunctionBegin;
  if (osm->ksp) {
    for (i=0; i<osm->n; i++) {
      PetscCall(KSPReset(osm->ksp[i]));
    }
  }
  if (osm->pmat) {
    if (osm->n > 0) {
      PetscMPIInt size;
      PetscCallMPI(MPI_Comm_size(PetscObjectComm((PetscObject)pc),&size));
      if (size > 1) {
        /* osm->pmat is created by MatCreateSubMatricesMPI(), cannot use MatDestroySubMatrices() */
        PetscCall(MatDestroyMatrices(osm->n,&osm->pmat));
      } else {
        PetscCall(MatDestroySubMatrices(osm->n,&osm->pmat));
      }
    }
  }
  if (osm->x) {
    for (i=0; i<osm->n; i++) {
      PetscCall(VecDestroy(&osm->x[i]));
      PetscCall(VecDestroy(&osm->y[i]));
    }
  }
  PetscCall(VecDestroy(&osm->gx));
  PetscCall(VecDestroy(&osm->gy));

  PetscCall(VecScatterDestroy(&osm->gorestriction));
  PetscCall(VecScatterDestroy(&osm->girestriction));
  if (!osm->user_subdomains) {
    PetscCall(PCGASMDestroySubdomains(osm->n,&osm->ois,&osm->iis));
    osm->N    = PETSC_DETERMINE;
    osm->nmax = PETSC_DETERMINE;
  }
  if (osm->pctoouter) {
        PetscCall(VecScatterDestroy(&(osm->pctoouter)));
  }
  if (osm->permutationIS) {
        PetscCall(ISDestroy(&(osm->permutationIS)));
  }
  if (osm->pcx) {
        PetscCall(VecDestroy(&(osm->pcx)));
  }
  if (osm->pcy) {
        PetscCall(VecDestroy(&(osm->pcy)));
  }
  if (osm->permutationP) {
    PetscCall(MatDestroy(&(osm->permutationP)));
  }
  if (osm->pcmat) {
        PetscCall(MatDestroy(&osm->pcmat));
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode PCDestroy_GASM(PC pc)
{
  PC_GASM        *osm = (PC_GASM*)pc->data;
  PetscInt       i;

  PetscFunctionBegin;
  PetscCall(PCReset_GASM(pc));
  /* PCReset will not destroy subdomains, if user_subdomains is true. */
  PetscCall(PCGASMDestroySubdomains(osm->n,&osm->ois,&osm->iis));
  if (osm->ksp) {
    for (i=0; i<osm->n; i++) {
      PetscCall(KSPDestroy(&osm->ksp[i]));
    }
    PetscCall(PetscFree(osm->ksp));
  }
  PetscCall(PetscFree(osm->x));
  PetscCall(PetscFree(osm->y));
  PetscCall(PetscObjectComposeFunction((PetscObject)pc,"PCGASMSetSubdomains_C",NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)pc,"PCGASMSetOverlap_C",NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)pc,"PCGASMSetType_C",NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)pc,"PCGASMSetSortIndices_C",NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)pc,"PCGASMGetSubKSP_C",NULL));
  PetscCall(PetscFree(pc->data));
  PetscFunctionReturn(0);
}

static PetscErrorCode PCSetFromOptions_GASM(PetscOptionItems *PetscOptionsObject,PC pc)
{
  PC_GASM        *osm = (PC_GASM*)pc->data;
  PetscInt       blocks,ovl;
  PetscBool      flg;
  PCGASMType     gasmtype;

  PetscFunctionBegin;
  PetscOptionsHeadBegin(PetscOptionsObject,"Generalized additive Schwarz options");
  PetscCall(PetscOptionsBool("-pc_gasm_use_dm_subdomains","If subdomains aren't set, use DMCreateDomainDecomposition() to define subdomains.","PCGASMSetUseDMSubdomains",osm->dm_subdomains,&osm->dm_subdomains,&flg));
  PetscCall(PetscOptionsInt("-pc_gasm_total_subdomains","Total number of subdomains across communicator","PCGASMSetTotalSubdomains",osm->N,&blocks,&flg));
  if (flg) PetscCall(PCGASMSetTotalSubdomains(pc,blocks));
  PetscCall(PetscOptionsInt("-pc_gasm_overlap","Number of overlapping degrees of freedom","PCGASMSetOverlap",osm->overlap,&ovl,&flg));
  if (flg) {
    PetscCall(PCGASMSetOverlap(pc,ovl));
    osm->dm_subdomains = PETSC_FALSE;
  }
  flg  = PETSC_FALSE;
  PetscCall(PetscOptionsEnum("-pc_gasm_type","Type of restriction/extension","PCGASMSetType",PCGASMTypes,(PetscEnum)osm->type,(PetscEnum*)&gasmtype,&flg));
  if (flg) PetscCall(PCGASMSetType(pc,gasmtype));
  PetscCall(PetscOptionsBool("-pc_gasm_use_hierachical_partitioning","use hierarchical partitioning",NULL,osm->hierarchicalpartitioning,&osm->hierarchicalpartitioning,&flg));
  PetscOptionsHeadEnd();
  PetscFunctionReturn(0);
}

/*------------------------------------------------------------------------------------*/

/*@
    PCGASMSetTotalSubdomains - sets the total number of subdomains to use across the
                               communicator.
    Logically collective on pc

    Input Parameters:
+   pc  - the preconditioner
-   N   - total number of subdomains

    Level: beginner

.seealso: `PCGASMSetSubdomains()`, `PCGASMSetOverlap()`
          `PCGASMCreateSubdomains2D()`
@*/
PetscErrorCode  PCGASMSetTotalSubdomains(PC pc,PetscInt N)
{
  PC_GASM        *osm = (PC_GASM*)pc->data;
  PetscMPIInt    size,rank;

  PetscFunctionBegin;
  PetscCheck(N >= 1,PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Total number of subdomains must be 1 or more, got N = %" PetscInt_FMT,N);
  PetscCheck(!pc->setupcalled,PetscObjectComm((PetscObject)pc),PETSC_ERR_ARG_WRONGSTATE,"PCGASMSetTotalSubdomains() should be called before calling PCSetUp().");

  PetscCall(PCGASMDestroySubdomains(osm->n,&osm->iis,&osm->ois));
  osm->ois = osm->iis = NULL;

  PetscCallMPI(MPI_Comm_size(PetscObjectComm((PetscObject)pc),&size));
  PetscCallMPI(MPI_Comm_rank(PetscObjectComm((PetscObject)pc),&rank));
  osm->N    = N;
  osm->n    = PETSC_DETERMINE;
  osm->nmax = PETSC_DETERMINE;
  osm->dm_subdomains = PETSC_FALSE;
  PetscFunctionReturn(0);
}

static PetscErrorCode  PCGASMSetSubdomains_GASM(PC pc,PetscInt n,IS iis[],IS ois[])
{
  PC_GASM         *osm = (PC_GASM*)pc->data;
  PetscInt        i;

  PetscFunctionBegin;
  PetscCheck(n >= 1,PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Each process must have 1 or more subdomains, got n = %" PetscInt_FMT,n);
  PetscCheck(!pc->setupcalled,PetscObjectComm((PetscObject)pc),PETSC_ERR_ARG_WRONGSTATE,"PCGASMSetSubdomains() should be called before calling PCSetUp().");

  PetscCall(PCGASMDestroySubdomains(osm->n,&osm->iis,&osm->ois));
  osm->iis  = osm->ois = NULL;
  osm->n    = n;
  osm->N    = PETSC_DETERMINE;
  osm->nmax = PETSC_DETERMINE;
  if (ois) {
    PetscCall(PetscMalloc1(n,&osm->ois));
    for (i=0; i<n; i++) {
      PetscCall(PetscObjectReference((PetscObject)ois[i]));
      osm->ois[i] = ois[i];
    }
    /*
       Since the user set the outer subdomains, even if nontrivial overlap was requested via PCGASMSetOverlap(),
       it will be ignored.  To avoid confusion later on (e.g., when viewing the PC), the overlap size is set to -1.
    */
    osm->overlap = -1;
    /* inner subdomains must be provided  */
    PetscCheck(iis,PETSC_COMM_SELF,PETSC_ERR_ARG_NULL,"inner indices have to be provided ");
  }/* end if */
  if (iis) {
    PetscCall(PetscMalloc1(n,&osm->iis));
    for (i=0; i<n; i++) {
      PetscCall(PetscObjectReference((PetscObject)iis[i]));
      osm->iis[i] = iis[i];
    }
    if (!ois) {
      osm->ois = NULL;
      /* if user does not provide outer indices, we will create the corresponding outer indices using  osm->overlap =1 in PCSetUp_GASM */
    }
  }
  if (PetscDefined(USE_DEBUG)) {
    PetscInt        j,rstart,rend,*covered,lsize;
    const PetscInt  *indices;
    /* check if the inner indices cover and only cover the local portion of the preconditioning matrix */
    PetscCall(MatGetOwnershipRange(pc->pmat,&rstart,&rend));
    PetscCall(PetscCalloc1(rend-rstart,&covered));
    /* check if the current processor owns indices from others */
    for (i=0; i<n; i++) {
      PetscCall(ISGetIndices(osm->iis[i],&indices));
      PetscCall(ISGetLocalSize(osm->iis[i],&lsize));
      for (j=0; j<lsize; j++) {
        PetscCheck(indices[j] >= rstart && indices[j] < rend,PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"inner subdomains can not own an index %" PetscInt_FMT " from other processors", indices[j]);
        PetscCheck(covered[indices[j]-rstart] != 1,PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"inner subdomains can not have an overlapping index %" PetscInt_FMT " ",indices[j]);
        covered[indices[j]-rstart] = 1;
      }
    PetscCall(ISRestoreIndices(osm->iis[i],&indices));
    }
    /* check if we miss any indices */
    for (i=rstart; i<rend; i++) {
      PetscCheck(covered[i-rstart],PETSC_COMM_SELF,PETSC_ERR_ARG_NULL,"local entity %" PetscInt_FMT " was not covered by inner subdomains",i);
    }
    PetscCall(PetscFree(covered));
  }
  if (iis)  osm->user_subdomains = PETSC_TRUE;
  PetscFunctionReturn(0);
}

static PetscErrorCode  PCGASMSetOverlap_GASM(PC pc,PetscInt ovl)
{
  PC_GASM *osm = (PC_GASM*)pc->data;

  PetscFunctionBegin;
  PetscCheck(ovl >= 0,PetscObjectComm((PetscObject)pc),PETSC_ERR_ARG_OUTOFRANGE,"Negative overlap value requested");
  PetscCheck(!pc->setupcalled || ovl == osm->overlap,PetscObjectComm((PetscObject)pc),PETSC_ERR_ARG_WRONGSTATE,"PCGASMSetOverlap() should be called before PCSetUp().");
  if (!pc->setupcalled) osm->overlap = ovl;
  PetscFunctionReturn(0);
}

static PetscErrorCode  PCGASMSetType_GASM(PC pc,PCGASMType type)
{
  PC_GASM *osm = (PC_GASM*)pc->data;

  PetscFunctionBegin;
  osm->type     = type;
  osm->type_set = PETSC_TRUE;
  PetscFunctionReturn(0);
}

static PetscErrorCode  PCGASMSetSortIndices_GASM(PC pc,PetscBool doSort)
{
  PC_GASM *osm = (PC_GASM*)pc->data;

  PetscFunctionBegin;
  osm->sort_indices = doSort;
  PetscFunctionReturn(0);
}

/*
   FIXME: This routine might need to be modified now that multiple ranks per subdomain are allowed.
        In particular, it would upset the global subdomain number calculation.
*/
static PetscErrorCode  PCGASMGetSubKSP_GASM(PC pc,PetscInt *n,PetscInt *first,KSP **ksp)
{
  PC_GASM        *osm = (PC_GASM*)pc->data;

  PetscFunctionBegin;
  PetscCheck(osm->n >= 1,PetscObjectComm((PetscObject)pc),PETSC_ERR_ORDER,"Need to call PCSetUp() on PC (or KSPSetUp() on the outer KSP object) before calling here");

  if (n) *n = osm->n;
  if (first) {
    PetscCallMPI(MPI_Scan(&osm->n,first,1,MPIU_INT,MPI_SUM,PetscObjectComm((PetscObject)pc)));
    *first -= osm->n;
  }
  if (ksp) {
    /* Assume that local solves are now different; not necessarily
       true, though!  This flag is used only for PCView_GASM() */
    *ksp                        = osm->ksp;
    osm->same_subdomain_solvers = PETSC_FALSE;
  }
  PetscFunctionReturn(0);
} /* PCGASMGetSubKSP_GASM() */

/*@C
    PCGASMSetSubdomains - Sets the subdomains for this processor
    for the additive Schwarz preconditioner.

    Collective on pc

    Input Parameters:
+   pc  - the preconditioner object
.   n   - the number of subdomains for this processor
.   iis - the index sets that define the inner subdomains (or NULL for PETSc to determine subdomains)
-   ois - the index sets that define the outer subdomains (or NULL to use the same as iis, or to construct by expanding iis by the requested overlap)

    Notes:
    The IS indices use the parallel, global numbering of the vector entries.
    Inner subdomains are those where the correction is applied.
    Outer subdomains are those where the residual necessary to obtain the
    corrections is obtained (see PCGASMType for the use of inner/outer subdomains).
    Both inner and outer subdomains can extend over several processors.
    This processor's portion of a subdomain is known as a local subdomain.

    Inner subdomains can not overlap with each other, do not have any entities from remote processors,
    and  have to cover the entire local subdomain owned by the current processor. The index sets on each
    process should be ordered such that the ith local subdomain is connected to the ith remote subdomain
    on another MPI process.

    By default the GASM preconditioner uses 1 (local) subdomain per processor.

    Level: advanced

.seealso: `PCGASMSetOverlap()`, `PCGASMGetSubKSP()`,
          `PCGASMCreateSubdomains2D()`, `PCGASMGetSubdomains()`
@*/
PetscErrorCode  PCGASMSetSubdomains(PC pc,PetscInt n,IS iis[],IS ois[])
{
  PC_GASM *osm = (PC_GASM*)pc->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_CLASSID,1);
  PetscTryMethod(pc,"PCGASMSetSubdomains_C",(PC,PetscInt,IS[],IS[]),(pc,n,iis,ois));
  osm->dm_subdomains = PETSC_FALSE;
  PetscFunctionReturn(0);
}

/*@
    PCGASMSetOverlap - Sets the overlap between a pair of subdomains for the
    additive Schwarz preconditioner.  Either all or no processors in the
    pc communicator must call this routine.

    Logically Collective on pc

    Input Parameters:
+   pc  - the preconditioner context
-   ovl - the amount of overlap between subdomains (ovl >= 0, default value = 0)

    Options Database Key:
.   -pc_gasm_overlap <overlap> - Sets overlap

    Notes:
    By default the GASM preconditioner uses 1 subdomain per processor.  To use
    multiple subdomain per perocessor or "straddling" subdomains that intersect
    multiple processors use PCGASMSetSubdomains() (or option -pc_gasm_total_subdomains <n>).

    The overlap defaults to 0, so if one desires that no additional
    overlap be computed beyond what may have been set with a call to
    PCGASMSetSubdomains(), then ovl must be set to be 0.  In particular, if one does
    not explicitly set the subdomains in application code, then all overlap would be computed
    internally by PETSc, and using an overlap of 0 would result in an GASM
    variant that is equivalent to the block Jacobi preconditioner.

    Note that one can define initial index sets with any overlap via
    PCGASMSetSubdomains(); the routine PCGASMSetOverlap() merely allows
    PETSc to extend that overlap further, if desired.

    Level: intermediate

.seealso: `PCGASMSetSubdomains()`, `PCGASMGetSubKSP()`,
          `PCGASMCreateSubdomains2D()`, `PCGASMGetSubdomains()`
@*/
PetscErrorCode  PCGASMSetOverlap(PC pc,PetscInt ovl)
{
  PC_GASM *osm = (PC_GASM*)pc->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_CLASSID,1);
  PetscValidLogicalCollectiveInt(pc,ovl,2);
  PetscTryMethod(pc,"PCGASMSetOverlap_C",(PC,PetscInt),(pc,ovl));
  osm->dm_subdomains = PETSC_FALSE;
  PetscFunctionReturn(0);
}

/*@
    PCGASMSetType - Sets the type of restriction and interpolation used
    for local problems in the additive Schwarz method.

    Logically Collective on PC

    Input Parameters:
+   pc  - the preconditioner context
-   type - variant of GASM, one of
.vb
      PC_GASM_BASIC       - full interpolation and restriction
      PC_GASM_RESTRICT    - full restriction, local processor interpolation
      PC_GASM_INTERPOLATE - full interpolation, local processor restriction
      PC_GASM_NONE        - local processor restriction and interpolation
.ve

    Options Database Key:
.   -pc_gasm_type [basic,restrict,interpolate,none] - Sets GASM type

    Level: intermediate

.seealso: `PCGASMSetSubdomains()`, `PCGASMGetSubKSP()`,
          `PCGASMCreateSubdomains2D()`
@*/
PetscErrorCode  PCGASMSetType(PC pc,PCGASMType type)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_CLASSID,1);
  PetscValidLogicalCollectiveEnum(pc,type,2);
  PetscTryMethod(pc,"PCGASMSetType_C",(PC,PCGASMType),(pc,type));
  PetscFunctionReturn(0);
}

/*@
    PCGASMSetSortIndices - Determines whether subdomain indices are sorted.

    Logically Collective on PC

    Input Parameters:
+   pc  - the preconditioner context
-   doSort - sort the subdomain indices

    Level: intermediate

.seealso: `PCGASMSetSubdomains()`, `PCGASMGetSubKSP()`,
          `PCGASMCreateSubdomains2D()`
@*/
PetscErrorCode  PCGASMSetSortIndices(PC pc,PetscBool doSort)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_CLASSID,1);
  PetscValidLogicalCollectiveBool(pc,doSort,2);
  PetscTryMethod(pc,"PCGASMSetSortIndices_C",(PC,PetscBool),(pc,doSort));
  PetscFunctionReturn(0);
}

/*@C
   PCGASMGetSubKSP - Gets the local KSP contexts for all blocks on
   this processor.

   Collective on PC iff first_local is requested

   Input Parameter:
.  pc - the preconditioner context

   Output Parameters:
+  n_local - the number of blocks on this processor or NULL
.  first_local - the global number of the first block on this processor or NULL,
                 all processors must request or all must pass NULL
-  ksp - the array of KSP contexts

   Note:
   After PCGASMGetSubKSP() the array of KSPes is not to be freed

   Currently for some matrix implementations only 1 block per processor
   is supported.

   You must call KSPSetUp() before calling PCGASMGetSubKSP().

   Level: advanced

.seealso: `PCGASMSetSubdomains()`, `PCGASMSetOverlap()`,
          `PCGASMCreateSubdomains2D()`,
@*/
PetscErrorCode  PCGASMGetSubKSP(PC pc,PetscInt *n_local,PetscInt *first_local,KSP *ksp[])
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_CLASSID,1);
  PetscUseMethod(pc,"PCGASMGetSubKSP_C",(PC,PetscInt*,PetscInt*,KSP **),(pc,n_local,first_local,ksp));
  PetscFunctionReturn(0);
}

/* -------------------------------------------------------------------------------------*/
/*MC
   PCGASM - Use the (restricted) additive Schwarz method, each block is (approximately) solved with
           its own KSP object.

   Options Database Keys:
+  -pc_gasm_total_subdomains <n>  - Sets total number of local subdomains to be distributed among processors
.  -pc_gasm_view_subdomains       - activates the printing of subdomain indices in PCView(), -ksp_view or -snes_view
.  -pc_gasm_print_subdomains      - activates the printing of subdomain indices in PCSetUp()
.  -pc_gasm_overlap <ovl>         - Sets overlap by which to (automatically) extend local subdomains
-  -pc_gasm_type [basic,restrict,interpolate,none] - Sets GASM type

     IMPORTANT: If you run with, for example, 3 blocks on 1 processor or 3 blocks on 3 processors you
      will get a different convergence rate due to the default option of -pc_gasm_type restrict. Use
      -pc_gasm_type basic to use the standard GASM.

   Notes:
    Blocks can be shared by multiple processes.

     To set options on the solvers for each block append -sub_ to all the KSP, and PC
        options database keys. For example, -sub_pc_type ilu -sub_pc_factor_levels 1 -sub_ksp_type preonly

     To set the options on the solvers separate for each block call PCGASMGetSubKSP()
         and set the options directly on the resulting KSP object (you can access its PC
         with KSPGetPC())

   Level: beginner

    References:
+   * - M Dryja, OB Widlund, An additive variant of the Schwarz alternating method for the case of many subregions
     Courant Institute, New York University Technical report
-   * - Barry Smith, Petter Bjorstad, and William Gropp, Domain Decompositions: Parallel Multilevel Methods for Elliptic Partial Differential Equations,
    Cambridge University Press.

.seealso: `PCCreate()`, `PCSetType()`, `PCType`, `PC`,
          `PCBJACOBI`, `PCGASMGetSubKSP()`, `PCGASMSetSubdomains()`,
          `PCSetModifySubMatrices()`, `PCGASMSetOverlap()`, `PCGASMSetType()`

M*/

PETSC_EXTERN PetscErrorCode PCCreate_GASM(PC pc)
{
  PC_GASM        *osm;

  PetscFunctionBegin;
  PetscCall(PetscNewLog(pc,&osm));

  osm->N                        = PETSC_DETERMINE;
  osm->n                        = PETSC_DECIDE;
  osm->nmax                     = PETSC_DETERMINE;
  osm->overlap                  = 0;
  osm->ksp                      = NULL;
  osm->gorestriction            = NULL;
  osm->girestriction            = NULL;
  osm->pctoouter                = NULL;
  osm->gx                       = NULL;
  osm->gy                       = NULL;
  osm->x                        = NULL;
  osm->y                        = NULL;
  osm->pcx                      = NULL;
  osm->pcy                      = NULL;
  osm->permutationIS            = NULL;
  osm->permutationP             = NULL;
  osm->pcmat                    = NULL;
  osm->ois                      = NULL;
  osm->iis                      = NULL;
  osm->pmat                     = NULL;
  osm->type                     = PC_GASM_RESTRICT;
  osm->same_subdomain_solvers   = PETSC_TRUE;
  osm->sort_indices             = PETSC_TRUE;
  osm->dm_subdomains            = PETSC_FALSE;
  osm->hierarchicalpartitioning = PETSC_FALSE;

  pc->data                 = (void*)osm;
  pc->ops->apply           = PCApply_GASM;
  pc->ops->matapply        = PCMatApply_GASM;
  pc->ops->applytranspose  = PCApplyTranspose_GASM;
  pc->ops->setup           = PCSetUp_GASM;
  pc->ops->reset           = PCReset_GASM;
  pc->ops->destroy         = PCDestroy_GASM;
  pc->ops->setfromoptions  = PCSetFromOptions_GASM;
  pc->ops->setuponblocks   = PCSetUpOnBlocks_GASM;
  pc->ops->view            = PCView_GASM;
  pc->ops->applyrichardson = NULL;

  PetscCall(PetscObjectComposeFunction((PetscObject)pc,"PCGASMSetSubdomains_C",PCGASMSetSubdomains_GASM));
  PetscCall(PetscObjectComposeFunction((PetscObject)pc,"PCGASMSetOverlap_C",PCGASMSetOverlap_GASM));
  PetscCall(PetscObjectComposeFunction((PetscObject)pc,"PCGASMSetType_C",PCGASMSetType_GASM));
  PetscCall(PetscObjectComposeFunction((PetscObject)pc,"PCGASMSetSortIndices_C",PCGASMSetSortIndices_GASM));
  PetscCall(PetscObjectComposeFunction((PetscObject)pc,"PCGASMGetSubKSP_C",PCGASMGetSubKSP_GASM));
  PetscFunctionReturn(0);
}

PetscErrorCode  PCGASMCreateLocalSubdomains(Mat A, PetscInt nloc, IS *iis[])
{
  MatPartitioning mpart;
  const char      *prefix;
  PetscInt        i,j,rstart,rend,bs;
  PetscBool       hasop, isbaij = PETSC_FALSE,foundpart = PETSC_FALSE;
  Mat             Ad     = NULL, adj;
  IS              ispart,isnumb,*is;

  PetscFunctionBegin;
  PetscCheck(nloc >= 1,PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"number of local subdomains must > 0, got nloc = %" PetscInt_FMT,nloc);

  /* Get prefix, row distribution, and block size */
  PetscCall(MatGetOptionsPrefix(A,&prefix));
  PetscCall(MatGetOwnershipRange(A,&rstart,&rend));
  PetscCall(MatGetBlockSize(A,&bs));
  PetscCheck(rstart/bs*bs == rstart && rend/bs*bs == rend,PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"bad row distribution [%" PetscInt_FMT ",%" PetscInt_FMT ") for matrix block size %" PetscInt_FMT,rstart,rend,bs);

  /* Get diagonal block from matrix if possible */
  PetscCall(MatHasOperation(A,MATOP_GET_DIAGONAL_BLOCK,&hasop));
  if (hasop) {
    PetscCall(MatGetDiagonalBlock(A,&Ad));
  }
  if (Ad) {
    PetscCall(PetscObjectBaseTypeCompare((PetscObject)Ad,MATSEQBAIJ,&isbaij));
    if (!isbaij) PetscCall(PetscObjectBaseTypeCompare((PetscObject)Ad,MATSEQSBAIJ,&isbaij));
  }
  if (Ad && nloc > 1) {
    PetscBool  match,done;
    /* Try to setup a good matrix partitioning if available */
    PetscCall(MatPartitioningCreate(PETSC_COMM_SELF,&mpart));
    PetscCall(PetscObjectSetOptionsPrefix((PetscObject)mpart,prefix));
    PetscCall(MatPartitioningSetFromOptions(mpart));
    PetscCall(PetscObjectTypeCompare((PetscObject)mpart,MATPARTITIONINGCURRENT,&match));
    if (!match) {
      PetscCall(PetscObjectTypeCompare((PetscObject)mpart,MATPARTITIONINGSQUARE,&match));
    }
    if (!match) { /* assume a "good" partitioner is available */
      PetscInt       na;
      const PetscInt *ia,*ja;
      PetscCall(MatGetRowIJ(Ad,0,PETSC_TRUE,isbaij,&na,&ia,&ja,&done));
      if (done) {
        /* Build adjacency matrix by hand. Unfortunately a call to
           MatConvert(Ad,MATMPIADJ,MAT_INITIAL_MATRIX,&adj) will
           remove the block-aij structure and we cannot expect
           MatPartitioning to split vertices as we need */
        PetscInt       i,j,len,nnz,cnt,*iia=NULL,*jja=NULL;
        const PetscInt *row;
        nnz = 0;
        for (i=0; i<na; i++) { /* count number of nonzeros */
          len = ia[i+1] - ia[i];
          row = ja + ia[i];
          for (j=0; j<len; j++) {
            if (row[j] == i) { /* don't count diagonal */
              len--; break;
            }
          }
          nnz += len;
        }
        PetscCall(PetscMalloc1(na+1,&iia));
        PetscCall(PetscMalloc1(nnz,&jja));
        nnz    = 0;
        iia[0] = 0;
        for (i=0; i<na; i++) { /* fill adjacency */
          cnt = 0;
          len = ia[i+1] - ia[i];
          row = ja + ia[i];
          for (j=0; j<len; j++) {
            if (row[j] != i) jja[nnz+cnt++] = row[j]; /* if not diagonal */
          }
          nnz += cnt;
          iia[i+1] = nnz;
        }
        /* Partitioning of the adjacency matrix */
        PetscCall(MatCreateMPIAdj(PETSC_COMM_SELF,na,na,iia,jja,NULL,&adj));
        PetscCall(MatPartitioningSetAdjacency(mpart,adj));
        PetscCall(MatPartitioningSetNParts(mpart,nloc));
        PetscCall(MatPartitioningApply(mpart,&ispart));
        PetscCall(ISPartitioningToNumbering(ispart,&isnumb));
        PetscCall(MatDestroy(&adj));
        foundpart = PETSC_TRUE;
      }
      PetscCall(MatRestoreRowIJ(Ad,0,PETSC_TRUE,isbaij,&na,&ia,&ja,&done));
    }
    PetscCall(MatPartitioningDestroy(&mpart));
  }
  PetscCall(PetscMalloc1(nloc,&is));
  if (!foundpart) {

    /* Partitioning by contiguous chunks of rows */

    PetscInt mbs   = (rend-rstart)/bs;
    PetscInt start = rstart;
    for (i=0; i<nloc; i++) {
      PetscInt count = (mbs/nloc + ((mbs % nloc) > i)) * bs;
      PetscCall(ISCreateStride(PETSC_COMM_SELF,count,start,1,&is[i]));
      start += count;
    }

  } else {

    /* Partitioning by adjacency of diagonal block  */

    const PetscInt *numbering;
    PetscInt       *count,nidx,*indices,*newidx,start=0;
    /* Get node count in each partition */
    PetscCall(PetscMalloc1(nloc,&count));
    PetscCall(ISPartitioningCount(ispart,nloc,count));
    if (isbaij && bs > 1) { /* adjust for the block-aij case */
      for (i=0; i<nloc; i++) count[i] *= bs;
    }
    /* Build indices from node numbering */
    PetscCall(ISGetLocalSize(isnumb,&nidx));
    PetscCall(PetscMalloc1(nidx,&indices));
    for (i=0; i<nidx; i++) indices[i] = i; /* needs to be initialized */
    PetscCall(ISGetIndices(isnumb,&numbering));
    PetscCall(PetscSortIntWithPermutation(nidx,numbering,indices));
    PetscCall(ISRestoreIndices(isnumb,&numbering));
    if (isbaij && bs > 1) { /* adjust for the block-aij case */
      PetscCall(PetscMalloc1(nidx*bs,&newidx));
      for (i=0; i<nidx; i++) {
        for (j=0; j<bs; j++) newidx[i*bs+j] = indices[i]*bs + j;
      }
      PetscCall(PetscFree(indices));
      nidx   *= bs;
      indices = newidx;
    }
    /* Shift to get global indices */
    for (i=0; i<nidx; i++) indices[i] += rstart;

    /* Build the index sets for each block */
    for (i=0; i<nloc; i++) {
      PetscCall(ISCreateGeneral(PETSC_COMM_SELF,count[i],&indices[start],PETSC_COPY_VALUES,&is[i]));
      PetscCall(ISSort(is[i]));
      start += count[i];
    }

    PetscCall(PetscFree(count));
    PetscCall(PetscFree(indices));
    PetscCall(ISDestroy(&isnumb));
    PetscCall(ISDestroy(&ispart));
  }
  *iis = is;
  PetscFunctionReturn(0);
}

PETSC_INTERN PetscErrorCode  PCGASMCreateStraddlingSubdomains(Mat A,PetscInt N,PetscInt *n,IS *iis[])
{
  PetscFunctionBegin;
  PetscCall(MatSubdomainsCreateCoalesce(A,N,n,iis));
  PetscFunctionReturn(0);
}

/*@C
   PCGASMCreateSubdomains - Creates n index sets defining n nonoverlapping subdomains for the additive
   Schwarz preconditioner for a any problem based on its matrix.

   Collective

   Input Parameters:
+  A       - The global matrix operator
-  N       - the number of global subdomains requested

   Output Parameters:
+  n   - the number of subdomains created on this processor
-  iis - the array of index sets defining the local inner subdomains (on which the correction is applied)

   Level: advanced

   Note: When N >= A's communicator size, each subdomain is local -- contained within a single processor.
         When N < size, the subdomains are 'straddling' (processor boundaries) and are no longer local.
         The resulting subdomains can be use in PCGASMSetSubdomains(pc,n,iss,NULL).  The overlapping
         outer subdomains will be automatically generated from these according to the requested amount of
         overlap; this is currently supported only with local subdomains.

.seealso: `PCGASMSetSubdomains()`, `PCGASMDestroySubdomains()`
@*/
PetscErrorCode  PCGASMCreateSubdomains(Mat A,PetscInt N,PetscInt *n,IS *iis[])
{
  PetscMPIInt     size;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(A,MAT_CLASSID,1);
  PetscValidPointer(iis,4);

  PetscCheck(N >= 1,PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"Number of subdomains must be > 0, N = %" PetscInt_FMT,N);
  PetscCallMPI(MPI_Comm_size(PetscObjectComm((PetscObject)A),&size));
  if (N >= size) {
    *n = N/size + (N%size);
    PetscCall(PCGASMCreateLocalSubdomains(A,*n,iis));
  } else {
    PetscCall(PCGASMCreateStraddlingSubdomains(A,N,n,iis));
  }
  PetscFunctionReturn(0);
}

/*@C
   PCGASMDestroySubdomains - Destroys the index sets created with
   PCGASMCreateSubdomains() or PCGASMCreateSubdomains2D. Should be
   called after setting subdomains with PCGASMSetSubdomains().

   Collective

   Input Parameters:
+  n   - the number of index sets
.  iis - the array of inner subdomains,
-  ois - the array of outer subdomains, can be NULL

   Level: intermediate

   Notes:
    this is merely a convenience subroutine that walks each list,
   destroys each IS on the list, and then frees the list. At the end the
   list pointers are set to NULL.

.seealso: `PCGASMCreateSubdomains()`, `PCGASMSetSubdomains()`
@*/
PetscErrorCode  PCGASMDestroySubdomains(PetscInt n,IS **iis,IS **ois)
{
  PetscInt       i;

  PetscFunctionBegin;
  if (n <= 0) PetscFunctionReturn(0);
  if (ois) {
    PetscValidPointer(ois,3);
    if (*ois) {
      PetscValidPointer(*ois,3);
      for (i=0; i<n; i++) {
        PetscCall(ISDestroy(&(*ois)[i]));
      }
      PetscCall(PetscFree((*ois)));
    }
  }
  if (iis) {
    PetscValidPointer(iis,2);
    if (*iis) {
      PetscValidPointer(*iis,2);
      for (i=0; i<n; i++) {
        PetscCall(ISDestroy(&(*iis)[i]));
      }
      PetscCall(PetscFree((*iis)));
    }
  }
  PetscFunctionReturn(0);
}

#define PCGASMLocalSubdomainBounds2D(M,N,xleft,ylow,xright,yhigh,first,last,xleft_loc,ylow_loc,xright_loc,yhigh_loc,n) \
  {                                                                                                       \
    PetscInt first_row = first/M, last_row = last/M+1;                                                     \
    /*                                                                                                    \
     Compute ylow_loc and yhigh_loc so that (ylow_loc,xleft) and (yhigh_loc,xright) are the corners       \
     of the bounding box of the intersection of the subdomain with the local ownership range (local       \
     subdomain).                                                                                          \
     Also compute xleft_loc and xright_loc as the lower and upper bounds on the first and last rows       \
     of the intersection.                                                                                 \
    */                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             \
    /* ylow_loc is the grid row containing the first element of the local sumbdomain */                   \
    *ylow_loc = PetscMax(first_row,ylow);                                                                    \
    /* xleft_loc is the offset of first element of the local subdomain within its grid row (might actually be outside the local subdomain) */ \
    *xleft_loc = *ylow_loc==first_row ? PetscMax(first%M,xleft) : xleft;                                                                            \
    /* yhigh_loc is the grid row above the last local subdomain element */                                                                    \
    *yhigh_loc = PetscMin(last_row,yhigh);                                                                                                     \
    /* xright is the offset of the end of the  local subdomain within its grid row (might actually be outside the local subdomain) */         \
    *xright_loc = *yhigh_loc==last_row ? PetscMin(xright,last%M) : xright;                                                                          \
    /* Now compute the size of the local subdomain n. */ \
    *n = 0;                                               \
    if (*ylow_loc < *yhigh_loc) {                           \
      PetscInt width = xright-xleft;                     \
      *n += width*(*yhigh_loc-*ylow_loc-1);                 \
      *n += PetscMin(PetscMax(*xright_loc-xleft,0),width); \
      *n -= PetscMin(PetscMax(*xleft_loc-xleft,0), width); \
    } \
  }

/*@
   PCGASMCreateSubdomains2D - Creates the index sets for the overlapping Schwarz
   preconditioner for a two-dimensional problem on a regular grid.

   Collective

   Input Parameters:
+  pc       - the preconditioner context
.  M        - the global number of grid points in the x direction
.  N        - the global number of grid points in the y direction
.  Mdomains - the global number of subdomains in the x direction
.  Ndomains - the global number of subdomains in the y direction
.  dof      - degrees of freedom per node
-  overlap  - overlap in mesh lines

   Output Parameters:
+  Nsub - the number of local subdomains created
.  iis  - array of index sets defining inner (nonoverlapping) subdomains
-  ois  - array of index sets defining outer (overlapping, if overlap > 0) subdomains

   Level: advanced

.seealso: `PCGASMSetSubdomains()`, `PCGASMGetSubKSP()`, `PCGASMSetOverlap()`
@*/
PetscErrorCode  PCGASMCreateSubdomains2D(PC pc,PetscInt M,PetscInt N,PetscInt Mdomains,PetscInt Ndomains,PetscInt dof,PetscInt overlap,PetscInt *nsub,IS **iis,IS **ois)
{
  PetscMPIInt    size, rank;
  PetscInt       i, j;
  PetscInt       maxheight, maxwidth;
  PetscInt       xstart, xleft, xright, xleft_loc, xright_loc;
  PetscInt       ystart, ylow,  yhigh,  ylow_loc,  yhigh_loc;
  PetscInt       x[2][2], y[2][2], n[2];
  PetscInt       first, last;
  PetscInt       nidx, *idx;
  PetscInt       ii,jj,s,q,d;
  PetscInt       k,kk;
  PetscMPIInt    color;
  MPI_Comm       comm, subcomm;
  IS             **xis = NULL, **is = ois, **is_local = iis;

  PetscFunctionBegin;
  PetscCall(PetscObjectGetComm((PetscObject)pc, &comm));
  PetscCallMPI(MPI_Comm_size(comm, &size));
  PetscCallMPI(MPI_Comm_rank(comm, &rank));
  PetscCall(MatGetOwnershipRange(pc->pmat, &first, &last));
  PetscCheck((first%dof) == 0 && (last%dof) == 0,PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Matrix row partitioning unsuitable for domain decomposition: local row range (%" PetscInt_FMT ",%" PetscInt_FMT ") "
                                      "does not respect the number of degrees of freedom per grid point %" PetscInt_FMT, first, last, dof);

  /* Determine the number of domains with nonzero intersections with the local ownership range. */
  s      = 0;
  ystart = 0;
  for (j=0; j<Ndomains; ++j) {
    maxheight = N/Ndomains + ((N % Ndomains) > j); /* Maximal height of subdomain */
    PetscCheck(maxheight >= 2,PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Too many %" PetscInt_FMT " subdomains in the vertical directon for mesh height %" PetscInt_FMT, Ndomains, N);
    /* Vertical domain limits with an overlap. */
    ylow   = PetscMax(ystart - overlap,0);
    yhigh  = PetscMin(ystart + maxheight + overlap,N);
    xstart = 0;
    for (i=0; i<Mdomains; ++i) {
      maxwidth = M/Mdomains + ((M % Mdomains) > i); /* Maximal width of subdomain */
      PetscCheck(maxwidth >= 2,PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Too many %" PetscInt_FMT " subdomains in the horizontal direction for mesh width %" PetscInt_FMT, Mdomains, M);
      /* Horizontal domain limits with an overlap. */
      xleft  = PetscMax(xstart - overlap,0);
      xright = PetscMin(xstart + maxwidth + overlap,M);
      /*
         Determine whether this subdomain intersects this processor's ownership range of pc->pmat.
      */
      PCGASMLocalSubdomainBounds2D(M,N,xleft,ylow,xright,yhigh,first,last,(&xleft_loc),(&ylow_loc),(&xright_loc),(&yhigh_loc),(&nidx));
      if (nidx) ++s;
      xstart += maxwidth;
    } /* for (i = 0; i < Mdomains; ++i) */
    ystart += maxheight;
  } /* for (j = 0; j < Ndomains; ++j) */

  /* Now we can allocate the necessary number of ISs. */
  *nsub  = s;
  PetscCall(PetscMalloc1(*nsub,is));
  PetscCall(PetscMalloc1(*nsub,is_local));
  s      = 0;
  ystart = 0;
  for (j=0; j<Ndomains; ++j) {
    maxheight = N/Ndomains + ((N % Ndomains) > j); /* Maximal height of subdomain */
    PetscCheck(maxheight >= 2,PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Too many %" PetscInt_FMT " subdomains in the vertical directon for mesh height %" PetscInt_FMT, Ndomains, N);
    /* Vertical domain limits with an overlap. */
    y[0][0] = PetscMax(ystart - overlap,0);
    y[0][1] = PetscMin(ystart + maxheight + overlap,N);
    /* Vertical domain limits without an overlap. */
    y[1][0] = ystart;
    y[1][1] = PetscMin(ystart + maxheight,N);
    xstart  = 0;
    for (i=0; i<Mdomains; ++i) {
      maxwidth = M/Mdomains + ((M % Mdomains) > i); /* Maximal width of subdomain */
      PetscCheck(maxwidth >= 2,PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Too many %" PetscInt_FMT " subdomains in the horizontal direction for mesh width %" PetscInt_FMT, Mdomains, M);
      /* Horizontal domain limits with an overlap. */
      x[0][0] = PetscMax(xstart - overlap,0);
      x[0][1] = PetscMin(xstart + maxwidth + overlap,M);
      /* Horizontal domain limits without an overlap. */
      x[1][0] = xstart;
      x[1][1] = PetscMin(xstart+maxwidth,M);
      /*
         Determine whether this domain intersects this processor's ownership range of pc->pmat.
         Do this twice: first for the domains with overlaps, and once without.
         During the first pass create the subcommunicators, and use them on the second pass as well.
      */
      for (q = 0; q < 2; ++q) {
        PetscBool split = PETSC_FALSE;
        /*
          domain limits, (xleft, xright) and (ylow, yheigh) are adjusted
          according to whether the domain with an overlap or without is considered.
        */
        xleft = x[q][0]; xright = x[q][1];
        ylow  = y[q][0]; yhigh  = y[q][1];
        PCGASMLocalSubdomainBounds2D(M,N,xleft,ylow,xright,yhigh,first,last,(&xleft_loc),(&ylow_loc),(&xright_loc),(&yhigh_loc),(&nidx));
        nidx *= dof;
        n[q]  = nidx;
        /*
         Based on the counted number of indices in the local domain *with an overlap*,
         construct a subcommunicator of all the processors supporting this domain.
         Observe that a domain with an overlap might have nontrivial local support,
         while the domain without an overlap might not.  Hence, the decision to participate
         in the subcommunicator must be based on the domain with an overlap.
         */
        if (q == 0) {
          if (nidx) color = 1;
          else color = MPI_UNDEFINED;
          PetscCallMPI(MPI_Comm_split(comm, color, rank, &subcomm));
          split = PETSC_TRUE;
        }
        /*
         Proceed only if the number of local indices *with an overlap* is nonzero.
         */
        if (n[0]) {
          if (q == 0) xis = is;
          if (q == 1) {
            /*
             The IS for the no-overlap subdomain shares a communicator with the overlapping domain.
             Moreover, if the overlap is zero, the two ISs are identical.
             */
            if (overlap == 0) {
              (*is_local)[s] = (*is)[s];
              PetscCall(PetscObjectReference((PetscObject)(*is)[s]));
              continue;
            } else {
              xis     = is_local;
              subcomm = ((PetscObject)(*is)[s])->comm;
            }
          } /* if (q == 1) */
          idx  = NULL;
          PetscCall(PetscMalloc1(nidx,&idx));
          if (nidx) {
            k = 0;
            for (jj=ylow_loc; jj<yhigh_loc; ++jj) {
              PetscInt x0 = (jj==ylow_loc) ? xleft_loc : xleft;
              PetscInt x1 = (jj==yhigh_loc-1) ? xright_loc : xright;
              kk = dof*(M*jj + x0);
              for (ii=x0; ii<x1; ++ii) {
                for (d = 0; d < dof; ++d) {
                  idx[k++] = kk++;
                }
              }
            }
          }
          PetscCall(ISCreateGeneral(subcomm,nidx,idx,PETSC_OWN_POINTER,(*xis)+s));
          if (split) {
            PetscCallMPI(MPI_Comm_free(&subcomm));
          }
        }/* if (n[0]) */
      }/* for (q = 0; q < 2; ++q) */
      if (n[0]) ++s;
      xstart += maxwidth;
    } /* for (i = 0; i < Mdomains; ++i) */
    ystart += maxheight;
  } /* for (j = 0; j < Ndomains; ++j) */
  PetscFunctionReturn(0);
}

/*@C
    PCGASMGetSubdomains - Gets the subdomains supported on this processor
    for the additive Schwarz preconditioner.

    Not Collective

    Input Parameter:
.   pc - the preconditioner context

    Output Parameters:
+   n   - the number of subdomains for this processor (default value = 1)
.   iis - the index sets that define the inner subdomains (without overlap) supported on this processor (can be NULL)
-   ois - the index sets that define the outer subdomains (with overlap) supported on this processor (can be NULL)

    Notes:
    The user is responsible for destroying the ISs and freeing the returned arrays.
    The IS numbering is in the parallel, global numbering of the vector.

    Level: advanced

.seealso: `PCGASMSetOverlap()`, `PCGASMGetSubKSP()`, `PCGASMCreateSubdomains2D()`,
          `PCGASMSetSubdomains()`, `PCGASMGetSubmatrices()`
@*/
PetscErrorCode  PCGASMGetSubdomains(PC pc,PetscInt *n,IS *iis[],IS *ois[])
{
  PC_GASM        *osm;
  PetscBool      match;
  PetscInt       i;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_CLASSID,1);
  PetscCall(PetscObjectTypeCompare((PetscObject)pc,PCGASM,&match));
  PetscCheck(match,PetscObjectComm((PetscObject)pc), PETSC_ERR_ARG_WRONG, "Incorrect object type: expected %s, got %s instead", PCGASM, ((PetscObject)pc)->type_name);
  osm = (PC_GASM*)pc->data;
  if (n) *n = osm->n;
  if (iis) PetscCall(PetscMalloc1(osm->n, iis));
  if (ois) PetscCall(PetscMalloc1(osm->n, ois));
  if (iis || ois) {
    for (i = 0; i < osm->n; ++i) {
      if (iis) (*iis)[i] = osm->iis[i];
      if (ois) (*ois)[i] = osm->ois[i];
    }
  }
  PetscFunctionReturn(0);
}

/*@C
    PCGASMGetSubmatrices - Gets the local submatrices (for this processor
    only) for the additive Schwarz preconditioner.

    Not Collective

    Input Parameter:
.   pc - the preconditioner context

    Output Parameters:
+   n   - the number of matrices for this processor (default value = 1)
-   mat - the matrices

    Notes:
    matrices returned by this routine have the same communicators as the index sets (IS)
           used to define subdomains in PCGASMSetSubdomains()
    Level: advanced

.seealso: `PCGASMSetOverlap()`, `PCGASMGetSubKSP()`,
          `PCGASMCreateSubdomains2D()`, `PCGASMSetSubdomains()`, `PCGASMGetSubdomains()`
@*/
PetscErrorCode  PCGASMGetSubmatrices(PC pc,PetscInt *n,Mat *mat[])
{
  PC_GASM        *osm;
  PetscBool      match;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_CLASSID,1);
  PetscValidIntPointer(n,2);
  if (mat) PetscValidPointer(mat,3);
  PetscCheck(pc->setupcalled,PetscObjectComm((PetscObject)pc),PETSC_ERR_ARG_WRONGSTATE,"Must call after KSPSetUp() or PCSetUp().");
  PetscCall(PetscObjectTypeCompare((PetscObject)pc,PCGASM,&match));
  PetscCheck(match,PetscObjectComm((PetscObject)pc), PETSC_ERR_ARG_WRONG, "Expected %s, got %s instead", PCGASM, ((PetscObject)pc)->type_name);
  osm = (PC_GASM*)pc->data;
  if (n) *n = osm->n;
  if (mat) *mat = osm->pmat;
  PetscFunctionReturn(0);
}

/*@
    PCGASMSetUseDMSubdomains - Indicates whether to use DMCreateDomainDecomposition() to define the subdomains, whenever possible.
    Logically Collective

    Input Parameters:
+   pc  - the preconditioner
-   flg - boolean indicating whether to use subdomains defined by the DM

    Options Database Key:
.   -pc_gasm_dm_subdomains -pc_gasm_overlap -pc_gasm_total_subdomains

    Level: intermediate

    Notes:
    PCGASMSetSubdomains(), PCGASMSetTotalSubdomains() or PCGASMSetOverlap() take precedence over PCGASMSetUseDMSubdomains(),
    so setting PCGASMSetSubdomains() with nontrivial subdomain ISs or any of PCGASMSetTotalSubdomains() and PCGASMSetOverlap()
    automatically turns the latter off.

.seealso: `PCGASMGetUseDMSubdomains()`, `PCGASMSetSubdomains()`, `PCGASMSetOverlap()`
          `PCGASMCreateSubdomains2D()`
@*/
PetscErrorCode  PCGASMSetUseDMSubdomains(PC pc,PetscBool flg)
{
  PC_GASM        *osm = (PC_GASM*)pc->data;
  PetscBool      match;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_CLASSID,1);
  PetscValidLogicalCollectiveBool(pc,flg,2);
  PetscCheck(!pc->setupcalled,((PetscObject)pc)->comm,PETSC_ERR_ARG_WRONGSTATE,"Not for a setup PC.");
  PetscCall(PetscObjectTypeCompare((PetscObject)pc,PCGASM,&match));
  if (match) {
    if (!osm->user_subdomains && osm->N == PETSC_DETERMINE && osm->overlap < 0) {
      osm->dm_subdomains = flg;
    }
  }
  PetscFunctionReturn(0);
}

/*@
    PCGASMGetUseDMSubdomains - Returns flag indicating whether to use DMCreateDomainDecomposition() to define the subdomains, whenever possible.
    Not Collective

    Input Parameter:
.   pc  - the preconditioner

    Output Parameter:
.   flg - boolean indicating whether to use subdomains defined by the DM

    Level: intermediate

.seealso: `PCGASMSetUseDMSubdomains()`, `PCGASMSetOverlap()`
          `PCGASMCreateSubdomains2D()`
@*/
PetscErrorCode  PCGASMGetUseDMSubdomains(PC pc,PetscBool* flg)
{
  PC_GASM        *osm = (PC_GASM*)pc->data;
  PetscBool      match;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_CLASSID,1);
  PetscValidBoolPointer(flg,2);
  PetscCall(PetscObjectTypeCompare((PetscObject)pc,PCGASM,&match));
  if (match) {
    if (flg) *flg = osm->dm_subdomains;
  }
  PetscFunctionReturn(0);
}
