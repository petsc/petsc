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
  PetscErrorCode ierr;

  PetscFunctionBegin;
  /* Determine the number of globally-distinct subdomains and compute a global numbering for them. */
  ierr = PetscMalloc2(osm->n,numbering,osm->n,permutation);CHKERRQ(ierr);
  ierr = PetscObjectsListGetGlobalNumbering(PetscObjectComm((PetscObject)pc),osm->n,(PetscObject*)osm->iis,NULL,*numbering);CHKERRQ(ierr);
  for (i = 0; i < osm->n; ++i) (*permutation)[i] = i;
  ierr = PetscSortIntWithPermutation(osm->n,*numbering,*permutation);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode  PCGASMSubdomainView_Private(PC pc, PetscInt i, PetscViewer viewer)
{
  PC_GASM        *osm = (PC_GASM*)pc->data;
  PetscInt       j,nidx;
  const PetscInt *idx;
  PetscViewer    sviewer;
  char           *cidx;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (i < 0 || i > osm->n) SETERRQ2(PetscObjectComm((PetscObject)viewer), PETSC_ERR_ARG_WRONG, "Invalid subdomain %D: must nonnegative and less than %D", i, osm->n);
  /* Inner subdomains. */
  ierr = ISGetLocalSize(osm->iis[i], &nidx);CHKERRQ(ierr);
  /*
   No more than 15 characters per index plus a space.
   PetscViewerStringSPrintf requires a string of size at least 2, so use (nidx+1) instead of nidx,
   in case nidx == 0. That will take care of the space for the trailing '\0' as well.
   For nidx == 0, the whole string 16 '\0'.
   */
#define len  16*(nidx+1)+1
  ierr = PetscMalloc1(len, &cidx);CHKERRQ(ierr);
  ierr = PetscViewerStringOpen(PETSC_COMM_SELF, cidx, len, &sviewer);CHKERRQ(ierr);
#undef len
  ierr = ISGetIndices(osm->iis[i], &idx);CHKERRQ(ierr);
  for (j = 0; j < nidx; ++j) {
    ierr = PetscViewerStringSPrintf(sviewer, "%D ", idx[j]);CHKERRQ(ierr);
  }
  ierr = ISRestoreIndices(osm->iis[i],&idx);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&sviewer);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer, "Inner subdomain:\n");CHKERRQ(ierr);
  ierr = PetscViewerFlush(viewer);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPushSynchronized(viewer);CHKERRQ(ierr);
  ierr = PetscViewerASCIISynchronizedPrintf(viewer, "%s", cidx);CHKERRQ(ierr);
  ierr = PetscViewerFlush(viewer);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPopSynchronized(viewer);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer, "\n");CHKERRQ(ierr);
  ierr = PetscViewerFlush(viewer);CHKERRQ(ierr);
  ierr = PetscFree(cidx);CHKERRQ(ierr);
  /* Outer subdomains. */
  ierr = ISGetLocalSize(osm->ois[i], &nidx);CHKERRQ(ierr);
  /*
   No more than 15 characters per index plus a space.
   PetscViewerStringSPrintf requires a string of size at least 2, so use (nidx+1) instead of nidx,
   in case nidx == 0. That will take care of the space for the trailing '\0' as well.
   For nidx == 0, the whole string 16 '\0'.
   */
#define len  16*(nidx+1)+1
  ierr = PetscMalloc1(len, &cidx);CHKERRQ(ierr);
  ierr = PetscViewerStringOpen(PETSC_COMM_SELF, cidx, len, &sviewer);CHKERRQ(ierr);
#undef len
  ierr = ISGetIndices(osm->ois[i], &idx);CHKERRQ(ierr);
  for (j = 0; j < nidx; ++j) {
    ierr = PetscViewerStringSPrintf(sviewer,"%D ", idx[j]);CHKERRQ(ierr);
  }
  ierr = PetscViewerDestroy(&sviewer);CHKERRQ(ierr);
  ierr = ISRestoreIndices(osm->ois[i],&idx);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer, "Outer subdomain:\n");CHKERRQ(ierr);
  ierr = PetscViewerFlush(viewer);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPushSynchronized(viewer);CHKERRQ(ierr);
  ierr = PetscViewerASCIISynchronizedPrintf(viewer, "%s", cidx);CHKERRQ(ierr);
  ierr = PetscViewerFlush(viewer);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPopSynchronized(viewer);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer, "\n");CHKERRQ(ierr);
  ierr = PetscViewerFlush(viewer);CHKERRQ(ierr);
  ierr = PetscFree(cidx);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode  PCGASMPrintSubdomains(PC pc)
{
  PC_GASM        *osm = (PC_GASM*)pc->data;
  const char     *prefix;
  char           fname[PETSC_MAX_PATH_LEN+1];
  PetscInt       l, d, count;
  PetscBool      doprint,found;
  PetscViewer    viewer, sviewer = NULL;
  PetscInt       *numbering,*permutation;/* global numbering of locally-supported subdomains and the permutation from the local ordering */
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PCGetOptionsPrefix(pc,&prefix);CHKERRQ(ierr);
  doprint  = PETSC_FALSE;
  ierr = PetscOptionsGetBool(NULL,prefix,"-pc_gasm_print_subdomains",&doprint,NULL);CHKERRQ(ierr);
  if (!doprint) PetscFunctionReturn(0);
  ierr = PetscOptionsGetString(NULL,prefix,"-pc_gasm_print_subdomains",fname,PETSC_MAX_PATH_LEN,&found);CHKERRQ(ierr);
  if (!found) { ierr = PetscStrcpy(fname,"stdout");CHKERRQ(ierr); };
  ierr = PetscViewerASCIIOpen(PetscObjectComm((PetscObject)pc),fname,&viewer);CHKERRQ(ierr);
  /*
   Make sure the viewer has a name. Otherwise this may cause a deadlock or other weird errors when creating a subcomm viewer:
   the subcomm viewer will attempt to inherit the viewer's name, which, if not set, will be constructed collectively on the comm.
  */
  ierr = PetscObjectName((PetscObject)viewer);CHKERRQ(ierr);
  l    = 0;
  ierr = PCGASMComputeGlobalSubdomainNumbering_Private(pc,&numbering,&permutation);CHKERRQ(ierr);
  for (count = 0; count < osm->N; ++count) {
    /* Now let subdomains go one at a time in the global numbering order and print their subdomain/solver info. */
    if (l<osm->n) {
      d = permutation[l]; /* d is the local number of the l-th smallest (in the global ordering) among the locally supported subdomains */
      if (numbering[d] == count) {
        ierr = PetscViewerGetSubViewer(viewer,((PetscObject)osm->ois[d])->comm, &sviewer);CHKERRQ(ierr);
        ierr = PCGASMSubdomainView_Private(pc,d,sviewer);CHKERRQ(ierr);
        ierr = PetscViewerRestoreSubViewer(viewer,((PetscObject)osm->ois[d])->comm, &sviewer);CHKERRQ(ierr);
        ++l;
      }
    }
    ierr = MPI_Barrier(PetscObjectComm((PetscObject)pc));CHKERRQ(ierr);
  }
  ierr = PetscFree2(numbering,permutation);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


static PetscErrorCode PCView_GASM(PC pc,PetscViewer viewer)
{
  PC_GASM        *osm = (PC_GASM*)pc->data;
  const char     *prefix;
  PetscErrorCode ierr;
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
  ierr = MPI_Comm_size(PetscObjectComm((PetscObject)pc), &size);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(PetscObjectComm((PetscObject)pc), &rank);CHKERRQ(ierr);

  if (osm->overlap >= 0) {
    ierr = PetscSNPrintf(overlap,sizeof(overlap),"requested amount of overlap = %D",osm->overlap);CHKERRQ(ierr);
  }
  if (osm->N != PETSC_DETERMINE) {
    ierr = PetscSNPrintf(gsubdomains, sizeof(gsubdomains), "total number of subdomains = %D",osm->N);CHKERRQ(ierr);
  }
  if (osm->nmax != PETSC_DETERMINE) {
    ierr = PetscSNPrintf(msubdomains,sizeof(msubdomains),"max number of local subdomains = %D",osm->nmax);CHKERRQ(ierr);
  }

  ierr = PCGetOptionsPrefix(pc,&prefix);CHKERRQ(ierr);
  ierr = PetscOptionsGetBool(NULL,prefix,"-pc_gasm_view_subdomains",&view_subdomains,NULL);CHKERRQ(ierr);

  ierr = PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&iascii);CHKERRQ(ierr);
  if (iascii) {
    /*
     Make sure the viewer has a name. Otherwise this may cause a deadlock when creating a subcomm viewer:
     the subcomm viewer will attempt to inherit the viewer's name, which, if not set, will be constructed
     collectively on the comm.
     */
    ierr = PetscObjectName((PetscObject)viewer);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"  Restriction/interpolation type: %s\n",PCGASMTypes[osm->type]);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"  %s\n",overlap);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"  %s\n",gsubdomains);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"  %s\n",msubdomains);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPushSynchronized(viewer);CHKERRQ(ierr);
    ierr = PetscViewerASCIISynchronizedPrintf(viewer,"  [%d|%d] number of locally-supported subdomains = %D\n",rank,size,osm->n);CHKERRQ(ierr);
    ierr = PetscViewerFlush(viewer);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPopSynchronized(viewer);CHKERRQ(ierr);
    /* Cannot take advantage of osm->same_subdomain_solvers without a global numbering of subdomains. */
    ierr = PetscViewerASCIIPrintf(viewer,"  Subdomain solver info is as follows:\n");CHKERRQ(ierr);
    ierr = PetscViewerASCIIPushTab(viewer);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"  - - - - - - - - - - - - - - - - - -\n");CHKERRQ(ierr);
    /* Make sure that everybody waits for the banner to be printed. */
    ierr = MPI_Barrier(PetscObjectComm((PetscObject)viewer));CHKERRQ(ierr);
    /* Now let subdomains go one at a time in the global numbering order and print their subdomain/solver info. */
    ierr = PCGASMComputeGlobalSubdomainNumbering_Private(pc,&numbering,&permutation);CHKERRQ(ierr);
    l = 0;
    for (count = 0; count < osm->N; ++count) {
      PetscMPIInt srank, ssize;
      if (l<osm->n) {
        PetscInt d = permutation[l]; /* d is the local number of the l-th smallest (in the global ordering) among the locally supported subdomains */
        if (numbering[d] == count) {
          ierr = MPI_Comm_size(((PetscObject)osm->ois[d])->comm, &ssize);CHKERRQ(ierr);
          ierr = MPI_Comm_rank(((PetscObject)osm->ois[d])->comm, &srank);CHKERRQ(ierr);
          ierr = PetscViewerGetSubViewer(viewer,((PetscObject)osm->ois[d])->comm, &sviewer);CHKERRQ(ierr);
          ierr = ISGetLocalSize(osm->ois[d],&bsz);CHKERRQ(ierr);
          ierr = PetscViewerASCIISynchronizedPrintf(sviewer,"  [%d|%d] (subcomm [%d|%d]) local subdomain number %D, local size = %D\n",rank,size,srank,ssize,d,bsz);CHKERRQ(ierr);
          ierr = PetscViewerFlush(sviewer);CHKERRQ(ierr);
          if (view_subdomains) {
            ierr = PCGASMSubdomainView_Private(pc,d,sviewer);CHKERRQ(ierr);
          }
          if (!pc->setupcalled) {
            ierr = PetscViewerASCIIPrintf(sviewer, "  Solver not set up yet: PCSetUp() not yet called\n");CHKERRQ(ierr);
          } else {
            ierr = KSPView(osm->ksp[d],sviewer);CHKERRQ(ierr);
          }
          ierr = PetscViewerASCIIPrintf(sviewer,"  - - - - - - - - - - - - - - - - - -\n");CHKERRQ(ierr);
          ierr = PetscViewerFlush(sviewer);CHKERRQ(ierr);
          ierr = PetscViewerRestoreSubViewer(viewer,((PetscObject)osm->ois[d])->comm, &sviewer);CHKERRQ(ierr);
          ++l;
        }
      }
      ierr = MPI_Barrier(PetscObjectComm((PetscObject)pc));CHKERRQ(ierr);
    }
    ierr = PetscFree2(numbering,permutation);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPopTab(viewer);CHKERRQ(ierr);
  }
  ierr = PetscViewerFlush(viewer);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPopSynchronized(viewer);CHKERRQ(ierr);
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
   PetscErrorCode        ierr;

   PetscFunctionBegin;
   ierr = PetscObjectGetComm((PetscObject)pc,&comm);CHKERRQ(ierr);
   ierr = MPI_Comm_size(comm,&size);CHKERRQ(ierr);
   /* we do not need a hierarchical partitioning when
    * the total number of subdomains is consistent with
    * the number of MPI tasks.
    * For the following cases, we do not need to use HP
    * */
   if(osm->N==PETSC_DETERMINE || osm->N>=size || osm->N==1) PetscFunctionReturn(0);
   if(size%osm->N != 0) SETERRQ2(PETSC_COMM_WORLD,PETSC_ERR_ARG_INCOMP,"have to specify the total number of subdomains %D to be a factor of the number of processors %d \n",osm->N,size);
   nlocalsubdomains = size/osm->N;
   osm->n           = 1;
   ierr = MatPartitioningCreate(comm,&part);CHKERRQ(ierr);
   ierr = MatPartitioningSetAdjacency(part,pc->pmat);CHKERRQ(ierr);
   ierr = MatPartitioningSetType(part,MATPARTITIONINGHIERARCH);CHKERRQ(ierr);
   ierr = MatPartitioningHierarchicalSetNcoarseparts(part,osm->N);CHKERRQ(ierr);
   ierr = MatPartitioningHierarchicalSetNfineparts(part,nlocalsubdomains);CHKERRQ(ierr);
   ierr = MatPartitioningSetFromOptions(part);CHKERRQ(ierr);
   /* get new processor owner number of each vertex */
   ierr = MatPartitioningApply(part,&partitioning);CHKERRQ(ierr);
   ierr = ISBuildTwoSided(partitioning,NULL,&fromrows);CHKERRQ(ierr);
   ierr = ISPartitioningToNumbering(partitioning,&isn);CHKERRQ(ierr);
   ierr = ISDestroy(&isn);CHKERRQ(ierr);
   ierr = ISGetLocalSize(fromrows,&fromrows_localsize);CHKERRQ(ierr);
   ierr = MatPartitioningDestroy(&part);CHKERRQ(ierr);
   ierr = MatCreateVecs(pc->pmat,&outervec,NULL);CHKERRQ(ierr);
   ierr = VecCreateMPI(comm,fromrows_localsize,PETSC_DETERMINE,&(osm->pcx));CHKERRQ(ierr);
   ierr = VecDuplicate(osm->pcx,&(osm->pcy));CHKERRQ(ierr);
   ierr = VecScatterCreate(osm->pcx,NULL,outervec,fromrows,&(osm->pctoouter));CHKERRQ(ierr);
   ierr = MatCreateSubMatrix(pc->pmat,fromrows,fromrows,MAT_INITIAL_MATRIX,&(osm->permutationP));CHKERRQ(ierr);
   ierr = PetscObjectReference((PetscObject)fromrows);CHKERRQ(ierr);
   osm->permutationIS = fromrows;
   osm->pcmat =  pc->pmat;
   ierr = PetscObjectReference((PetscObject)osm->permutationP);CHKERRQ(ierr);
   pc->pmat = osm->permutationP;
   ierr = VecDestroy(&outervec);CHKERRQ(ierr);
   ierr = ISDestroy(&fromrows);CHKERRQ(ierr);
   ierr = ISDestroy(&partitioning);CHKERRQ(ierr);
   osm->n           = PETSC_DETERMINE;
   PetscFunctionReturn(0);
}



static PetscErrorCode PCSetUp_GASM(PC pc)
{
  PC_GASM        *osm = (PC_GASM*)pc->data;
  PetscErrorCode ierr;
  PetscBool      symset,flg;
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
  ierr = MPI_Comm_size(PetscObjectComm((PetscObject)pc),&size);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(PetscObjectComm((PetscObject)pc),&rank);CHKERRQ(ierr);
  if (!pc->setupcalled) {
	/* use a hierarchical partitioning */
    if(osm->hierarchicalpartitioning){
      ierr = PCGASMSetHierarchicalPartitioning(pc);CHKERRQ(ierr);
    }
    if (!osm->type_set) {
      ierr = MatIsSymmetricKnown(pc->pmat,&symset,&flg);CHKERRQ(ierr);
      if (symset && flg) osm->type = PC_GASM_BASIC;
    }

    if (osm->n == PETSC_DETERMINE) {
      if (osm->N != PETSC_DETERMINE) {
	   /* No local subdomains given, but the desired number of total subdomains is known, so construct them accordingly. */
	   ierr = PCGASMCreateSubdomains(pc->pmat,osm->N,&osm->n,&osm->iis);CHKERRQ(ierr);
      } else if (osm->dm_subdomains && pc->dm) {
	/* try pc->dm next, if allowed */
	PetscInt  d;
	IS       *inner_subdomain_is, *outer_subdomain_is;
	ierr = DMCreateDomainDecomposition(pc->dm, &num_subdomains, &subdomain_names, &inner_subdomain_is, &outer_subdomain_is, &subdomain_dm);CHKERRQ(ierr);
	if (num_subdomains) {
	  ierr = PCGASMSetSubdomains(pc, num_subdomains, inner_subdomain_is, outer_subdomain_is);CHKERRQ(ierr);
	}
	for (d = 0; d < num_subdomains; ++d) {
	  if (inner_subdomain_is) {ierr = ISDestroy(&inner_subdomain_is[d]);CHKERRQ(ierr);}
	  if (outer_subdomain_is) {ierr = ISDestroy(&outer_subdomain_is[d]);CHKERRQ(ierr);}
	}
	ierr = PetscFree(inner_subdomain_is);CHKERRQ(ierr);
	ierr = PetscFree(outer_subdomain_is);CHKERRQ(ierr);
      } else {
	/* still no subdomains; use one per processor */
	osm->nmax = osm->n = 1;
	ierr      = MPI_Comm_size(PetscObjectComm((PetscObject)pc),&size);CHKERRQ(ierr);
	osm->N    = size;
	ierr = PCGASMCreateLocalSubdomains(pc->pmat,osm->n,&osm->iis);CHKERRQ(ierr);
      }
    }
    if (!osm->iis) {
      /*
       osm->n was set in PCGASMSetSubdomains(), but the actual subdomains have not been supplied.
       We create the requisite number of local inner subdomains and then expand them into
       out subdomains, if necessary.
       */
      ierr = PCGASMCreateLocalSubdomains(pc->pmat,osm->n,&osm->iis);CHKERRQ(ierr);
    }
    if (!osm->ois) {
      /*
	    Initially make outer subdomains the same as inner subdomains. If nonzero additional overlap
	    has been requested, copy the inner subdomains over so they can be modified.
      */
      ierr = PetscMalloc1(osm->n,&osm->ois);CHKERRQ(ierr);
      for (i=0; i<osm->n; ++i) {
	if (osm->overlap > 0 && osm->N>1) { /* With positive overlap, osm->iis[i] will be modified */
	  ierr = ISDuplicate(osm->iis[i],(osm->ois)+i);CHKERRQ(ierr);
	  ierr = ISCopy(osm->iis[i],osm->ois[i]);CHKERRQ(ierr);
	} else {
	  ierr      = PetscObjectReference((PetscObject)((osm->iis)[i]));CHKERRQ(ierr);
	  osm->ois[i] = osm->iis[i];
	}
      }
      if (osm->overlap>0 && osm->N>1) {
	   /* Extend the "overlapping" regions by a number of steps */
	   ierr = MatIncreaseOverlapSplit(pc->pmat,osm->n,osm->ois,osm->overlap);CHKERRQ(ierr);
      }
    }

    /* Now the subdomains are defined.  Determine their global and max local numbers, if necessary. */
    if (osm->nmax == PETSC_DETERMINE) {
      PetscMPIInt inwork,outwork;
      /* determine global number of subdomains and the max number of local subdomains */
      inwork = osm->n;
      ierr       = MPIU_Allreduce(&inwork,&outwork,1,MPI_INT,MPI_MAX,PetscObjectComm((PetscObject)pc));CHKERRQ(ierr);
      osm->nmax  = outwork;
    }
    if (osm->N == PETSC_DETERMINE) {
      /* Determine the number of globally-distinct subdomains and compute a global numbering for them. */
      ierr = PetscObjectsListGetGlobalNumbering(PetscObjectComm((PetscObject)pc),osm->n,(PetscObject*)osm->ois,&osm->N,NULL);CHKERRQ(ierr);
    }


    if (osm->sort_indices) {
      for (i=0; i<osm->n; i++) {
        ierr = ISSort(osm->ois[i]);CHKERRQ(ierr);
        ierr = ISSort(osm->iis[i]);CHKERRQ(ierr);
      }
    }
    ierr = PCGetOptionsPrefix(pc,&prefix);CHKERRQ(ierr);
    ierr = PCGASMPrintSubdomains(pc);CHKERRQ(ierr);

    /*
       Merge the ISs, create merged vectors and restrictions.
     */
    /* Merge outer subdomain ISs and construct a restriction onto the disjoint union of local outer subdomains. */
    on = 0;
    for (i=0; i<osm->n; i++) {
      ierr = ISGetLocalSize(osm->ois[i],&oni);CHKERRQ(ierr);
      on  += oni;
    }
    ierr = PetscMalloc1(on, &oidx);CHKERRQ(ierr);
    on   = 0;
    /* Merge local indices together */
    for (i=0; i<osm->n; i++) {
      ierr = ISGetLocalSize(osm->ois[i],&oni);CHKERRQ(ierr);
      ierr = ISGetIndices(osm->ois[i],&oidxi);CHKERRQ(ierr);
      ierr = PetscArraycpy(oidx+on,oidxi,oni);CHKERRQ(ierr);
      ierr = ISRestoreIndices(osm->ois[i],&oidxi);CHKERRQ(ierr);
      on  += oni;
    }
    ierr = ISCreateGeneral(((PetscObject)(pc))->comm,on,oidx,PETSC_OWN_POINTER,&gois);CHKERRQ(ierr);
    nTotalInnerIndices = 0;
    for(i=0; i<osm->n; i++){
      ierr = ISGetLocalSize(osm->iis[i],&nInnerIndices);CHKERRQ(ierr);
      nTotalInnerIndices += nInnerIndices;
    }
    ierr = VecCreateMPI(((PetscObject)(pc))->comm,nTotalInnerIndices,PETSC_DETERMINE,&x);CHKERRQ(ierr);
    ierr = VecDuplicate(x,&y);CHKERRQ(ierr);

    ierr = VecCreateMPI(PetscObjectComm((PetscObject)pc),on,PETSC_DECIDE,&osm->gx);CHKERRQ(ierr);
    ierr = VecDuplicate(osm->gx,&osm->gy);CHKERRQ(ierr);
    ierr = VecGetOwnershipRange(osm->gx, &gostart, NULL);CHKERRQ(ierr);
    ierr = ISCreateStride(PetscObjectComm((PetscObject)pc),on,gostart,1, &goid);CHKERRQ(ierr);
    /* gois might indices not on local */
    ierr = VecScatterCreate(x,gois,osm->gx,goid, &(osm->gorestriction));CHKERRQ(ierr);
    ierr = PetscMalloc1(osm->n,&numbering);CHKERRQ(ierr);
    ierr = PetscObjectsListGetGlobalNumbering(PetscObjectComm((PetscObject)pc),osm->n,(PetscObject*)osm->ois,NULL,numbering);CHKERRQ(ierr);
    ierr = VecDestroy(&x);CHKERRQ(ierr);
    ierr = ISDestroy(&gois);CHKERRQ(ierr);

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
        ierr = ISGetLocalSize(osm->ois[i],&oni);CHKERRQ(ierr);
        on  += oni;
      }
      ierr = PetscMalloc1(on, &iidx);CHKERRQ(ierr);
      ierr = PetscMalloc1(on, &ioidx);CHKERRQ(ierr);
      ierr = VecGetArray(y,&array);CHKERRQ(ierr);
      /* set communicator id to determine where overlap is */
      in   = 0;
      for (i=0; i<osm->n; i++) {
        ierr   = ISGetLocalSize(osm->iis[i],&ini);CHKERRQ(ierr);
        for (k = 0; k < ini; ++k){
          array[in+k] = numbering[i];
        }
        in += ini;
      }
      ierr = VecRestoreArray(y,&array);CHKERRQ(ierr);
      ierr = VecScatterBegin(osm->gorestriction,y,osm->gy,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
      ierr = VecScatterEnd(osm->gorestriction,y,osm->gy,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
      ierr = VecGetOwnershipRange(osm->gy,&gostart, NULL);CHKERRQ(ierr);
      ierr = VecGetArray(osm->gy,&array);CHKERRQ(ierr);
      on  = 0;
      in  = 0;
      for (i=0; i<osm->n; i++) {
    	ierr = ISGetLocalSize(osm->ois[i],&oni);CHKERRQ(ierr);
    	ierr = ISGetIndices(osm->ois[i],&indices);CHKERRQ(ierr);
    	for (k=0; k<oni; k++) {
          /*  skip overlapping indices to get inner domain */
          if(PetscRealPart(array[on+k]) != numbering[i]) continue;
          iidx[in]    = indices[k];
          ioidx[in++] = gostart+on+k;
    	}
    	ierr   = ISRestoreIndices(osm->ois[i], &indices);CHKERRQ(ierr);
    	on += oni;
      }
      ierr = VecRestoreArray(osm->gy,&array);CHKERRQ(ierr);
      ierr = ISCreateGeneral(PetscObjectComm((PetscObject)pc),in,iidx,PETSC_OWN_POINTER,&giis);CHKERRQ(ierr);
      ierr = ISCreateGeneral(PetscObjectComm((PetscObject)pc),in,ioidx,PETSC_OWN_POINTER,&giois);CHKERRQ(ierr);
      ierr = VecScatterCreate(y,giis,osm->gy,giois,&osm->girestriction);CHKERRQ(ierr);
      ierr = VecDestroy(&y);CHKERRQ(ierr);
      ierr = ISDestroy(&giis);CHKERRQ(ierr);
      ierr = ISDestroy(&giois);CHKERRQ(ierr);
    }
    ierr = ISDestroy(&goid);CHKERRQ(ierr);
    ierr = PetscFree(numbering);CHKERRQ(ierr);

    /* Create the subdomain work vectors. */
    ierr = PetscMalloc1(osm->n,&osm->x);CHKERRQ(ierr);
    ierr = PetscMalloc1(osm->n,&osm->y);CHKERRQ(ierr);
    ierr = VecGetArray(osm->gx, &gxarray);CHKERRQ(ierr);
    ierr = VecGetArray(osm->gy, &gyarray);CHKERRQ(ierr);
    for (i=0, on=0; i<osm->n; ++i, on += oni) {
      PetscInt oNi;
      ierr = ISGetLocalSize(osm->ois[i],&oni);CHKERRQ(ierr);
      /* on a sub communicator */
      ierr = ISGetSize(osm->ois[i],&oNi);CHKERRQ(ierr);
      ierr = VecCreateMPIWithArray(((PetscObject)(osm->ois[i]))->comm,1,oni,oNi,gxarray+on,&osm->x[i]);CHKERRQ(ierr);
      ierr = VecCreateMPIWithArray(((PetscObject)(osm->ois[i]))->comm,1,oni,oNi,gyarray+on,&osm->y[i]);CHKERRQ(ierr);
    }
    ierr = VecRestoreArray(osm->gx, &gxarray);CHKERRQ(ierr);
    ierr = VecRestoreArray(osm->gy, &gyarray);CHKERRQ(ierr);
    /* Create the subdomain solvers */
    ierr = PetscMalloc1(osm->n,&osm->ksp);CHKERRQ(ierr);
    for (i=0; i<osm->n; i++) {
      char subprefix[PETSC_MAX_PATH_LEN+1];
      ierr        = KSPCreate(((PetscObject)(osm->ois[i]))->comm,&ksp);CHKERRQ(ierr);
      ierr        = KSPSetErrorIfNotConverged(ksp,pc->erroriffailure);CHKERRQ(ierr);
      ierr        = PetscLogObjectParent((PetscObject)pc,(PetscObject)ksp);CHKERRQ(ierr);
      ierr        = PetscObjectIncrementTabLevel((PetscObject)ksp,(PetscObject)pc,1);CHKERRQ(ierr);
      ierr        = KSPSetType(ksp,KSPPREONLY);CHKERRQ(ierr);
      ierr        = KSPGetPC(ksp,&subpc);CHKERRQ(ierr); /* Why do we need this here? */
      if (subdomain_dm) {
	    ierr = KSPSetDM(ksp,subdomain_dm[i]);CHKERRQ(ierr);
	    ierr = DMDestroy(subdomain_dm+i);CHKERRQ(ierr);
      }
      ierr        = PCGetOptionsPrefix(pc,&prefix);CHKERRQ(ierr);
      ierr        = KSPSetOptionsPrefix(ksp,prefix);CHKERRQ(ierr);
      if (subdomain_names && subdomain_names[i]) {
	     ierr = PetscSNPrintf(subprefix,PETSC_MAX_PATH_LEN,"sub_%s_",subdomain_names[i]);CHKERRQ(ierr);
	     ierr = KSPAppendOptionsPrefix(ksp,subprefix);CHKERRQ(ierr);
	     ierr = PetscFree(subdomain_names[i]);CHKERRQ(ierr);
      }
      ierr        = KSPAppendOptionsPrefix(ksp,"sub_");CHKERRQ(ierr);
      osm->ksp[i] = ksp;
    }
    ierr = PetscFree(subdomain_dm);CHKERRQ(ierr);
    ierr = PetscFree(subdomain_names);CHKERRQ(ierr);
    scall = MAT_INITIAL_MATRIX;

  } else { /* if (pc->setupcalled) */
    /*
       Destroy the submatrices from the previous iteration
    */
    if (pc->flag == DIFFERENT_NONZERO_PATTERN) {
      ierr  = MatDestroyMatrices(osm->n,&osm->pmat);CHKERRQ(ierr);
      scall = MAT_INITIAL_MATRIX;
    }
    if(osm->permutationIS){
      ierr = MatCreateSubMatrix(pc->pmat,osm->permutationIS,osm->permutationIS,scall,&osm->permutationP);CHKERRQ(ierr);
      ierr = PetscObjectReference((PetscObject)osm->permutationP);CHKERRQ(ierr);
      osm->pcmat = pc->pmat;
      pc->pmat   = osm->permutationP;
    }

  }


  /*
     Extract out the submatrices.
  */
  if (size > 1) {
    ierr = MatCreateSubMatricesMPI(pc->pmat,osm->n,osm->ois,osm->ois,scall,&osm->pmat);CHKERRQ(ierr);
  } else {
    ierr = MatCreateSubMatrices(pc->pmat,osm->n,osm->ois,osm->ois,scall,&osm->pmat);CHKERRQ(ierr);
  }
  if (scall == MAT_INITIAL_MATRIX) {
    ierr = PetscObjectGetOptionsPrefix((PetscObject)pc->pmat,&pprefix);CHKERRQ(ierr);
    for (i=0; i<osm->n; i++) {
      ierr = PetscLogObjectParent((PetscObject)pc,(PetscObject)osm->pmat[i]);CHKERRQ(ierr);
      ierr = PetscObjectSetOptionsPrefix((PetscObject)osm->pmat[i],pprefix);CHKERRQ(ierr);
    }
  }

  /* Return control to the user so that the submatrices can be modified (e.g., to apply
     different boundary conditions for the submatrices than for the global problem) */
  ierr = PCModifySubMatrices(pc,osm->n,osm->ois,osm->ois,osm->pmat,pc->modifysubmatricesP);CHKERRQ(ierr);

  /*
     Loop over submatrices putting them into local ksps
  */
  for (i=0; i<osm->n; i++) {
    ierr = KSPSetOperators(osm->ksp[i],osm->pmat[i],osm->pmat[i]);CHKERRQ(ierr);
    if (!pc->setupcalled) {
      ierr = KSPSetFromOptions(osm->ksp[i]);CHKERRQ(ierr);
    }
  }
  if(osm->pcmat){
    ierr = MatDestroy(&pc->pmat);CHKERRQ(ierr);
    pc->pmat   = osm->pcmat;
    osm->pcmat = 0;
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode PCSetUpOnBlocks_GASM(PC pc)
{
  PC_GASM        *osm = (PC_GASM*)pc->data;
  PetscErrorCode ierr;
  PetscInt       i;

  PetscFunctionBegin;
  for (i=0; i<osm->n; i++) {
    ierr = KSPSetUp(osm->ksp[i]);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode PCApply_GASM(PC pc,Vec xin,Vec yout)
{
  PC_GASM        *osm = (PC_GASM*)pc->data;
  PetscErrorCode ierr;
  PetscInt       i;
  Vec            x,y;
  ScatterMode    forward = SCATTER_FORWARD,reverse = SCATTER_REVERSE;

  PetscFunctionBegin;
  if(osm->pctoouter){
    ierr = VecScatterBegin(osm->pctoouter,xin,osm->pcx,INSERT_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
    ierr = VecScatterEnd(osm->pctoouter,xin,osm->pcx,INSERT_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
    x = osm->pcx;
    y = osm->pcy;
  }else{
	x = xin;
	y = yout;
  }
  /*
     Support for limiting the restriction or interpolation only to the inner
     subdomain values (leaving the other values 0).
  */
  if (!(osm->type & PC_GASM_RESTRICT)) {
    /* have to zero the work RHS since scatter may leave some slots empty */
    ierr = VecZeroEntries(osm->gx);CHKERRQ(ierr);
    ierr = VecScatterBegin(osm->girestriction,x,osm->gx,INSERT_VALUES,forward);CHKERRQ(ierr);
  } else {
    ierr = VecScatterBegin(osm->gorestriction,x,osm->gx,INSERT_VALUES,forward);CHKERRQ(ierr);
  }
  ierr = VecZeroEntries(osm->gy);CHKERRQ(ierr);
  if (!(osm->type & PC_GASM_RESTRICT)) {
    ierr = VecScatterEnd(osm->girestriction,x,osm->gx,INSERT_VALUES,forward);CHKERRQ(ierr);
  } else {
    ierr = VecScatterEnd(osm->gorestriction,x,osm->gx,INSERT_VALUES,forward);CHKERRQ(ierr);
  }
  /* do the subdomain solves */
  for (i=0; i<osm->n; ++i) {
    ierr = KSPSolve(osm->ksp[i],osm->x[i],osm->y[i]);CHKERRQ(ierr);
    ierr = KSPCheckSolve(osm->ksp[i],pc,osm->y[i]);CHKERRQ(ierr);
  }
  /* Do we need to zero y ?? */
  ierr = VecZeroEntries(y);CHKERRQ(ierr);
  if (!(osm->type & PC_GASM_INTERPOLATE)) {
    ierr = VecScatterBegin(osm->girestriction,osm->gy,y,ADD_VALUES,reverse);CHKERRQ(ierr);
    ierr = VecScatterEnd(osm->girestriction,osm->gy,y,ADD_VALUES,reverse);CHKERRQ(ierr);
  } else {
    ierr = VecScatterBegin(osm->gorestriction,osm->gy,y,ADD_VALUES,reverse);CHKERRQ(ierr);
    ierr = VecScatterEnd(osm->gorestriction,osm->gy,y,ADD_VALUES,reverse);CHKERRQ(ierr);
  }
  if(osm->pctoouter){
    ierr = VecScatterBegin(osm->pctoouter,y,yout,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
    ierr = VecScatterEnd(osm->pctoouter,y,yout,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode PCApplyTranspose_GASM(PC pc,Vec xin,Vec yout)
{
  PC_GASM        *osm = (PC_GASM*)pc->data;
  PetscErrorCode ierr;
  PetscInt       i;
  Vec            x,y;
  ScatterMode    forward = SCATTER_FORWARD,reverse = SCATTER_REVERSE;

  PetscFunctionBegin;
  if(osm->pctoouter){
   ierr = VecScatterBegin(osm->pctoouter,xin,osm->pcx,INSERT_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
   ierr = VecScatterEnd(osm->pctoouter,xin,osm->pcx,INSERT_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
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
    ierr = VecZeroEntries(osm->gx);CHKERRQ(ierr);
    ierr = VecScatterBegin(osm->girestriction,x,osm->gx,INSERT_VALUES,forward);CHKERRQ(ierr);
  } else {
    ierr = VecScatterBegin(osm->gorestriction,x,osm->gx,INSERT_VALUES,forward);CHKERRQ(ierr);
  }
  ierr = VecZeroEntries(osm->gy);CHKERRQ(ierr);
  if (!(osm->type & PC_GASM_INTERPOLATE)) {
    ierr = VecScatterEnd(osm->girestriction,x,osm->gx,INSERT_VALUES,forward);CHKERRQ(ierr);
  } else {
    ierr = VecScatterEnd(osm->gorestriction,x,osm->gx,INSERT_VALUES,forward);CHKERRQ(ierr);
  }
  /* do the local solves */
  for (i=0; i<osm->n; ++i) { /* Note that the solves are local, so we can go to osm->n, rather than osm->nmax. */
    ierr = KSPSolveTranspose(osm->ksp[i],osm->x[i],osm->y[i]);CHKERRQ(ierr);
    ierr = KSPCheckSolve(osm->ksp[i],pc,osm->y[i]);CHKERRQ(ierr);
  }
  ierr = VecZeroEntries(y);CHKERRQ(ierr);
  if (!(osm->type & PC_GASM_RESTRICT)) {
    ierr = VecScatterBegin(osm->girestriction,osm->gy,y,ADD_VALUES,reverse);CHKERRQ(ierr);
    ierr = VecScatterEnd(osm->girestriction,osm->gy,y,ADD_VALUES,reverse);CHKERRQ(ierr);
  } else {
    ierr = VecScatterBegin(osm->gorestriction,osm->gy,y,ADD_VALUES,reverse);CHKERRQ(ierr);
    ierr = VecScatterEnd(osm->gorestriction,osm->gy,y,ADD_VALUES,reverse);CHKERRQ(ierr);
  }
  if(osm->pctoouter){
   ierr = VecScatterBegin(osm->pctoouter,y,yout,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
   ierr = VecScatterEnd(osm->pctoouter,y,yout,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode PCReset_GASM(PC pc)
{
  PC_GASM        *osm = (PC_GASM*)pc->data;
  PetscErrorCode ierr;
  PetscInt       i;

  PetscFunctionBegin;
  if (osm->ksp) {
    for (i=0; i<osm->n; i++) {
      ierr = KSPReset(osm->ksp[i]);CHKERRQ(ierr);
    }
  }
  if (osm->pmat) {
    if (osm->n > 0) {
      PetscMPIInt size;
      ierr = MPI_Comm_size(PetscObjectComm((PetscObject)pc),&size);CHKERRQ(ierr);
      if (size > 1) {
        /* osm->pmat is created by MatCreateSubMatricesMPI(), cannot use MatDestroySubMatrices() */
        ierr = MatDestroyMatrices(osm->n,&osm->pmat);CHKERRQ(ierr);
      } else {
        ierr = MatDestroySubMatrices(osm->n,&osm->pmat);CHKERRQ(ierr);
      }
    }
  }
  if (osm->x) {
    for (i=0; i<osm->n; i++) {
      ierr = VecDestroy(&osm->x[i]);CHKERRQ(ierr);
      ierr = VecDestroy(&osm->y[i]);CHKERRQ(ierr);
    }
  }
  ierr = VecDestroy(&osm->gx);CHKERRQ(ierr);
  ierr = VecDestroy(&osm->gy);CHKERRQ(ierr);

  ierr = VecScatterDestroy(&osm->gorestriction);CHKERRQ(ierr);
  ierr = VecScatterDestroy(&osm->girestriction);CHKERRQ(ierr);
  if (!osm->user_subdomains) {
    ierr      = PCGASMDestroySubdomains(osm->n,&osm->ois,&osm->iis);CHKERRQ(ierr);
    osm->N    = PETSC_DETERMINE;
    osm->nmax = PETSC_DETERMINE;
  }
  if(osm->pctoouter){
	ierr = VecScatterDestroy(&(osm->pctoouter));CHKERRQ(ierr);
  }
  if(osm->permutationIS){
	ierr = ISDestroy(&(osm->permutationIS));CHKERRQ(ierr);
  }
  if(osm->pcx){
	ierr = VecDestroy(&(osm->pcx));CHKERRQ(ierr);
  }
  if(osm->pcy){
	ierr = VecDestroy(&(osm->pcy));CHKERRQ(ierr);
  }
  if(osm->permutationP){
    ierr = MatDestroy(&(osm->permutationP));CHKERRQ(ierr);
  }
  if(osm->pcmat){
	ierr = MatDestroy(&osm->pcmat);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode PCDestroy_GASM(PC pc)
{
  PC_GASM        *osm = (PC_GASM*)pc->data;
  PetscErrorCode ierr;
  PetscInt       i;

  PetscFunctionBegin;
  ierr = PCReset_GASM(pc);CHKERRQ(ierr);
  /* PCReset will not destroy subdomains, if user_subdomains is true. */
  ierr = PCGASMDestroySubdomains(osm->n,&osm->ois,&osm->iis);CHKERRQ(ierr);
  if (osm->ksp) {
    for (i=0; i<osm->n; i++) {
      ierr = KSPDestroy(&osm->ksp[i]);CHKERRQ(ierr);
    }
    ierr = PetscFree(osm->ksp);CHKERRQ(ierr);
  }
  ierr = PetscFree(osm->x);CHKERRQ(ierr);
  ierr = PetscFree(osm->y);CHKERRQ(ierr);
  ierr = PetscFree(pc->data);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode PCSetFromOptions_GASM(PetscOptionItems *PetscOptionsObject,PC pc)
{
  PC_GASM        *osm = (PC_GASM*)pc->data;
  PetscErrorCode ierr;
  PetscInt       blocks,ovl;
  PetscBool      symset,flg;
  PCGASMType     gasmtype;

  PetscFunctionBegin;
  /* set the type to symmetric if matrix is symmetric */
  if (!osm->type_set && pc->pmat) {
    ierr = MatIsSymmetricKnown(pc->pmat,&symset,&flg);CHKERRQ(ierr);
    if (symset && flg) osm->type = PC_GASM_BASIC;
  }
  ierr = PetscOptionsHead(PetscOptionsObject,"Generalized additive Schwarz options");CHKERRQ(ierr);
  ierr = PetscOptionsBool("-pc_gasm_use_dm_subdomains","If subdomains aren't set, use DMCreateDomainDecomposition() to define subdomains.","PCGASMSetUseDMSubdomains",osm->dm_subdomains,&osm->dm_subdomains,&flg);CHKERRQ(ierr);
  ierr = PetscOptionsInt("-pc_gasm_total_subdomains","Total number of subdomains across communicator","PCGASMSetTotalSubdomains",osm->N,&blocks,&flg);CHKERRQ(ierr);
  if (flg) {
    ierr = PCGASMSetTotalSubdomains(pc,blocks);CHKERRQ(ierr);
  }
  ierr = PetscOptionsInt("-pc_gasm_overlap","Number of overlapping degrees of freedom","PCGASMSetOverlap",osm->overlap,&ovl,&flg);CHKERRQ(ierr);
  if (flg) {
    ierr = PCGASMSetOverlap(pc,ovl);CHKERRQ(ierr);
    osm->dm_subdomains = PETSC_FALSE;
  }
  flg  = PETSC_FALSE;
  ierr = PetscOptionsEnum("-pc_gasm_type","Type of restriction/extension","PCGASMSetType",PCGASMTypes,(PetscEnum)osm->type,(PetscEnum*)&gasmtype,&flg);CHKERRQ(ierr);
  if (flg) {ierr = PCGASMSetType(pc,gasmtype);CHKERRQ(ierr);}
  ierr = PetscOptionsBool("-pc_gasm_use_hierachical_partitioning","use hierarchical partitioning",NULL,osm->hierarchicalpartitioning,&osm->hierarchicalpartitioning,&flg);CHKERRQ(ierr);
  ierr = PetscOptionsTail();CHKERRQ(ierr);
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

.seealso: PCGASMSetSubdomains(), PCGASMSetOverlap()
          PCGASMCreateSubdomains2D()
@*/
PetscErrorCode  PCGASMSetTotalSubdomains(PC pc,PetscInt N)
{
  PC_GASM        *osm = (PC_GASM*)pc->data;
  PetscMPIInt    size,rank;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (N < 1) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Total number of subdomains must be 1 or more, got N = %D",N);
  if (pc->setupcalled) SETERRQ(PetscObjectComm((PetscObject)pc),PETSC_ERR_ARG_WRONGSTATE,"PCGASMSetTotalSubdomains() should be called before calling PCSetUp().");

  ierr = PCGASMDestroySubdomains(osm->n,&osm->iis,&osm->ois);CHKERRQ(ierr);
  osm->ois = osm->iis = NULL;

  ierr = MPI_Comm_size(PetscObjectComm((PetscObject)pc),&size);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(PetscObjectComm((PetscObject)pc),&rank);CHKERRQ(ierr);
  osm->N    = N;
  osm->n    = PETSC_DETERMINE;
  osm->nmax = PETSC_DETERMINE;
  osm->dm_subdomains = PETSC_FALSE;
  PetscFunctionReturn(0);
}


static PetscErrorCode  PCGASMSetSubdomains_GASM(PC pc,PetscInt n,IS iis[],IS ois[])
{
  PC_GASM         *osm = (PC_GASM*)pc->data;
  PetscErrorCode  ierr;
  PetscInt        i;

  PetscFunctionBegin;
  if (n < 1) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Each process must have 1 or more subdomains, got n = %D",n);
  if (pc->setupcalled) SETERRQ(PetscObjectComm((PetscObject)pc),PETSC_ERR_ARG_WRONGSTATE,"PCGASMSetSubdomains() should be called before calling PCSetUp().");

  ierr = PCGASMDestroySubdomains(osm->n,&osm->iis,&osm->ois);CHKERRQ(ierr);
  osm->iis  = osm->ois = NULL;
  osm->n    = n;
  osm->N    = PETSC_DETERMINE;
  osm->nmax = PETSC_DETERMINE;
  if (ois) {
    ierr = PetscMalloc1(n,&osm->ois);CHKERRQ(ierr);
    for (i=0; i<n; i++) {
      ierr = PetscObjectReference((PetscObject)ois[i]);CHKERRQ(ierr);
      osm->ois[i] = ois[i];
    }
    /*
       Since the user set the outer subdomains, even if nontrivial overlap was requested via PCGASMSetOverlap(),
       it will be ignored.  To avoid confusion later on (e.g., when viewing the PC), the overlap size is set to -1.
    */
    osm->overlap = -1;
    /* inner subdomains must be provided  */
    if (!iis) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_NULL,"inner indices have to be provided \n");
  }/* end if */
  if (iis) {
    ierr = PetscMalloc1(n,&osm->iis);CHKERRQ(ierr);
    for (i=0; i<n; i++) {
      ierr = PetscObjectReference((PetscObject)iis[i]);CHKERRQ(ierr);
      osm->iis[i] = iis[i];
    }
    if (!ois) {
      osm->ois = NULL;
      /* if user does not provide outer indices, we will create the corresponding outer indices using  osm->overlap =1 in PCSetUp_GASM */
    }
  }
#if defined(PETSC_USE_DEBUG)
  {
    PetscInt        j,rstart,rend,*covered,lsize;
    const PetscInt  *indices;
    /* check if the inner indices cover and only cover the local portion of the preconditioning matrix */
    ierr = MatGetOwnershipRange(pc->pmat,&rstart,&rend);CHKERRQ(ierr);
    ierr = PetscCalloc1(rend-rstart,&covered);CHKERRQ(ierr);
    /* check if the current processor owns indices from others */
    for (i=0; i<n; i++) {
      ierr = ISGetIndices(osm->iis[i],&indices);CHKERRQ(ierr);
      ierr = ISGetLocalSize(osm->iis[i],&lsize);CHKERRQ(ierr);
      for (j=0; j<lsize; j++) {
        if (indices[j]<rstart || indices[j]>=rend) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"inner subdomains can not own an index %d from other processors", indices[j]);
        else if (covered[indices[j]-rstart]==1) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"inner subdomains can not have an overlapping index %d ",indices[j]);
        else covered[indices[j]-rstart] = 1;
      }
    ierr = ISRestoreIndices(osm->iis[i],&indices);CHKERRQ(ierr);
    }
    /* check if we miss any indices */
    for (i=rstart; i<rend; i++) {
      if (!covered[i-rstart]) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_NULL,"local entity %d was not covered by inner subdomains",i);
    }
    ierr = PetscFree(covered);CHKERRQ(ierr);
  }
#endif
  if (iis)  osm->user_subdomains = PETSC_TRUE;
  PetscFunctionReturn(0);
}


static PetscErrorCode  PCGASMSetOverlap_GASM(PC pc,PetscInt ovl)
{
  PC_GASM *osm = (PC_GASM*)pc->data;

  PetscFunctionBegin;
  if (ovl < 0) SETERRQ(PetscObjectComm((PetscObject)pc),PETSC_ERR_ARG_OUTOFRANGE,"Negative overlap value requested");
  if (pc->setupcalled && ovl != osm->overlap) SETERRQ(PetscObjectComm((PetscObject)pc),PETSC_ERR_ARG_WRONGSTATE,"PCGASMSetOverlap() should be called before PCSetUp().");
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
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (osm->n < 1) SETERRQ(PetscObjectComm((PetscObject)pc),PETSC_ERR_ORDER,"Need to call PCSetUp() on PC (or KSPSetUp() on the outer KSP object) before calling here");

  if (n) *n = osm->n;
  if (first) {
    ierr    = MPI_Scan(&osm->n,first,1,MPIU_INT,MPI_SUM,PetscObjectComm((PetscObject)pc));CHKERRQ(ierr);
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

.seealso: PCGASMSetNumSubdomains(), PCGASMSetOverlap(), PCGASMGetSubKSP(),
          PCGASMCreateSubdomains2D(), PCGASMGetSubdomains()
@*/
PetscErrorCode  PCGASMSetSubdomains(PC pc,PetscInt n,IS iis[],IS ois[])
{
  PC_GASM *osm = (PC_GASM*)pc->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_CLASSID,1);
  ierr = PetscTryMethod(pc,"PCGASMSetSubdomains_C",(PC,PetscInt,IS[],IS[]),(pc,n,iis,ois));CHKERRQ(ierr);
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

.seealso: PCGASMSetSubdomains(), PCGASMGetSubKSP(),
          PCGASMCreateSubdomains2D(), PCGASMGetSubdomains()
@*/
PetscErrorCode  PCGASMSetOverlap(PC pc,PetscInt ovl)
{
  PetscErrorCode ierr;
  PC_GASM *osm = (PC_GASM*)pc->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_CLASSID,1);
  PetscValidLogicalCollectiveInt(pc,ovl,2);
  ierr = PetscTryMethod(pc,"PCGASMSetOverlap_C",(PC,PetscInt),(pc,ovl));CHKERRQ(ierr);
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

.seealso: PCGASMSetSubdomains(), PCGASMGetSubKSP(),
          PCGASMCreateSubdomains2D()
@*/
PetscErrorCode  PCGASMSetType(PC pc,PCGASMType type)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_CLASSID,1);
  PetscValidLogicalCollectiveEnum(pc,type,2);
  ierr = PetscTryMethod(pc,"PCGASMSetType_C",(PC,PCGASMType),(pc,type));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@
    PCGASMSetSortIndices - Determines whether subdomain indices are sorted.

    Logically Collective on PC

    Input Parameters:
+   pc  - the preconditioner context
-   doSort - sort the subdomain indices

    Level: intermediate

.seealso: PCGASMSetSubdomains(), PCGASMGetSubKSP(),
          PCGASMCreateSubdomains2D()
@*/
PetscErrorCode  PCGASMSetSortIndices(PC pc,PetscBool doSort)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_CLASSID,1);
  PetscValidLogicalCollectiveBool(pc,doSort,2);
  ierr = PetscTryMethod(pc,"PCGASMSetSortIndices_C",(PC,PetscBool),(pc,doSort));CHKERRQ(ierr);
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

.seealso: PCGASMSetSubdomains(), PCGASMSetOverlap(),
          PCGASMCreateSubdomains2D(),
@*/
PetscErrorCode  PCGASMGetSubKSP(PC pc,PetscInt *n_local,PetscInt *first_local,KSP *ksp[])
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_CLASSID,1);
  ierr = PetscUseMethod(pc,"PCGASMGetSubKSP_C",(PC,PetscInt*,PetscInt*,KSP **),(pc,n_local,first_local,ksp));CHKERRQ(ierr);
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
+   1. - M Dryja, OB Widlund, An additive variant of the Schwarz alternating method for the case of many subregions
     Courant Institute, New York University Technical report
-   2. - Barry Smith, Petter Bjorstad, and William Gropp, Domain Decompositions: Parallel Multilevel Methods for Elliptic Partial Differential Equations,
    Cambridge University Press.

.seealso:  PCCreate(), PCSetType(), PCType (for list of available types), PC,
           PCBJACOBI,  PCGASMGetSubKSP(), PCGASMSetSubdomains(),
           PCSetModifySubMatrices(), PCGASMSetOverlap(), PCGASMSetType()

M*/

PETSC_EXTERN PetscErrorCode PCCreate_GASM(PC pc)
{
  PetscErrorCode ierr;
  PC_GASM        *osm;

  PetscFunctionBegin;
  ierr = PetscNewLog(pc,&osm);CHKERRQ(ierr);

  osm->N                        = PETSC_DETERMINE;
  osm->n                        = PETSC_DECIDE;
  osm->nmax                     = PETSC_DETERMINE;
  osm->overlap                  = 0;
  osm->ksp                      = 0;
  osm->gorestriction            = 0;
  osm->girestriction            = 0;
  osm->pctoouter                = 0;
  osm->gx                       = 0;
  osm->gy                       = 0;
  osm->x                        = 0;
  osm->y                        = 0;
  osm->pcx                      = 0;
  osm->pcy                      = 0;
  osm->permutationIS            = 0;
  osm->permutationP             = 0;
  osm->pcmat                    = 0;
  osm->ois                      = 0;
  osm->iis                      = 0;
  osm->pmat                     = 0;
  osm->type                     = PC_GASM_RESTRICT;
  osm->same_subdomain_solvers   = PETSC_TRUE;
  osm->sort_indices             = PETSC_TRUE;
  osm->dm_subdomains            = PETSC_FALSE;
  osm->hierarchicalpartitioning = PETSC_FALSE;

  pc->data                 = (void*)osm;
  pc->ops->apply           = PCApply_GASM;
  pc->ops->applytranspose  = PCApplyTranspose_GASM;
  pc->ops->setup           = PCSetUp_GASM;
  pc->ops->reset           = PCReset_GASM;
  pc->ops->destroy         = PCDestroy_GASM;
  pc->ops->setfromoptions  = PCSetFromOptions_GASM;
  pc->ops->setuponblocks   = PCSetUpOnBlocks_GASM;
  pc->ops->view            = PCView_GASM;
  pc->ops->applyrichardson = 0;

  ierr = PetscObjectComposeFunction((PetscObject)pc,"PCGASMSetSubdomains_C",PCGASMSetSubdomains_GASM);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)pc,"PCGASMSetOverlap_C",PCGASMSetOverlap_GASM);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)pc,"PCGASMSetType_C",PCGASMSetType_GASM);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)pc,"PCGASMSetSortIndices_C",PCGASMSetSortIndices_GASM);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)pc,"PCGASMGetSubKSP_C",PCGASMGetSubKSP_GASM);CHKERRQ(ierr);
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
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  if (nloc < 1) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"number of local subdomains must > 0, got nloc = %D",nloc);

  /* Get prefix, row distribution, and block size */
  ierr = MatGetOptionsPrefix(A,&prefix);CHKERRQ(ierr);
  ierr = MatGetOwnershipRange(A,&rstart,&rend);CHKERRQ(ierr);
  ierr = MatGetBlockSize(A,&bs);CHKERRQ(ierr);
  if (rstart/bs*bs != rstart || rend/bs*bs != rend) SETERRQ3(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"bad row distribution [%D,%D) for matrix block size %D",rstart,rend,bs);

  /* Get diagonal block from matrix if possible */
  ierr = MatHasOperation(A,MATOP_GET_DIAGONAL_BLOCK,&hasop);CHKERRQ(ierr);
  if (hasop) {
    ierr = MatGetDiagonalBlock(A,&Ad);CHKERRQ(ierr);
  }
  if (Ad) {
    ierr = PetscObjectBaseTypeCompare((PetscObject)Ad,MATSEQBAIJ,&isbaij);CHKERRQ(ierr);
    if (!isbaij) {ierr = PetscObjectBaseTypeCompare((PetscObject)Ad,MATSEQSBAIJ,&isbaij);CHKERRQ(ierr);}
  }
  if (Ad && nloc > 1) {
    PetscBool  match,done;
    /* Try to setup a good matrix partitioning if available */
    ierr = MatPartitioningCreate(PETSC_COMM_SELF,&mpart);CHKERRQ(ierr);
    ierr = PetscObjectSetOptionsPrefix((PetscObject)mpart,prefix);CHKERRQ(ierr);
    ierr = MatPartitioningSetFromOptions(mpart);CHKERRQ(ierr);
    ierr = PetscObjectTypeCompare((PetscObject)mpart,MATPARTITIONINGCURRENT,&match);CHKERRQ(ierr);
    if (!match) {
      ierr = PetscObjectTypeCompare((PetscObject)mpart,MATPARTITIONINGSQUARE,&match);CHKERRQ(ierr);
    }
    if (!match) { /* assume a "good" partitioner is available */
      PetscInt       na;
      const PetscInt *ia,*ja;
      ierr = MatGetRowIJ(Ad,0,PETSC_TRUE,isbaij,&na,&ia,&ja,&done);CHKERRQ(ierr);
      if (done) {
        /* Build adjacency matrix by hand. Unfortunately a call to
           MatConvert(Ad,MATMPIADJ,MAT_INITIAL_MATRIX,&adj) will
           remove the block-aij structure and we cannot expect
           MatPartitioning to split vertices as we need */
        PetscInt       i,j,len,nnz,cnt,*iia=0,*jja=0;
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
        ierr   = PetscMalloc1(na+1,&iia);CHKERRQ(ierr);
        ierr   = PetscMalloc1(nnz,&jja);CHKERRQ(ierr);
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
        ierr      = MatCreateMPIAdj(PETSC_COMM_SELF,na,na,iia,jja,NULL,&adj);CHKERRQ(ierr);
        ierr      = MatPartitioningSetAdjacency(mpart,adj);CHKERRQ(ierr);
        ierr      = MatPartitioningSetNParts(mpart,nloc);CHKERRQ(ierr);
        ierr      = MatPartitioningApply(mpart,&ispart);CHKERRQ(ierr);
        ierr      = ISPartitioningToNumbering(ispart,&isnumb);CHKERRQ(ierr);
        ierr      = MatDestroy(&adj);CHKERRQ(ierr);
        foundpart = PETSC_TRUE;
      }
      ierr = MatRestoreRowIJ(Ad,0,PETSC_TRUE,isbaij,&na,&ia,&ja,&done);CHKERRQ(ierr);
    }
    ierr = MatPartitioningDestroy(&mpart);CHKERRQ(ierr);
  }
  ierr = PetscMalloc1(nloc,&is);CHKERRQ(ierr);
  if (!foundpart) {

    /* Partitioning by contiguous chunks of rows */

    PetscInt mbs   = (rend-rstart)/bs;
    PetscInt start = rstart;
    for (i=0; i<nloc; i++) {
      PetscInt count = (mbs/nloc + ((mbs % nloc) > i)) * bs;
      ierr   = ISCreateStride(PETSC_COMM_SELF,count,start,1,&is[i]);CHKERRQ(ierr);
      start += count;
    }

  } else {

    /* Partitioning by adjacency of diagonal block  */

    const PetscInt *numbering;
    PetscInt       *count,nidx,*indices,*newidx,start=0;
    /* Get node count in each partition */
    ierr = PetscMalloc1(nloc,&count);CHKERRQ(ierr);
    ierr = ISPartitioningCount(ispart,nloc,count);CHKERRQ(ierr);
    if (isbaij && bs > 1) { /* adjust for the block-aij case */
      for (i=0; i<nloc; i++) count[i] *= bs;
    }
    /* Build indices from node numbering */
    ierr = ISGetLocalSize(isnumb,&nidx);CHKERRQ(ierr);
    ierr = PetscMalloc1(nidx,&indices);CHKERRQ(ierr);
    for (i=0; i<nidx; i++) indices[i] = i; /* needs to be initialized */
    ierr = ISGetIndices(isnumb,&numbering);CHKERRQ(ierr);
    ierr = PetscSortIntWithPermutation(nidx,numbering,indices);CHKERRQ(ierr);
    ierr = ISRestoreIndices(isnumb,&numbering);CHKERRQ(ierr);
    if (isbaij && bs > 1) { /* adjust for the block-aij case */
      ierr = PetscMalloc1(nidx*bs,&newidx);CHKERRQ(ierr);
      for (i=0; i<nidx; i++) {
        for (j=0; j<bs; j++) newidx[i*bs+j] = indices[i]*bs + j;
      }
      ierr    = PetscFree(indices);CHKERRQ(ierr);
      nidx   *= bs;
      indices = newidx;
    }
    /* Shift to get global indices */
    for (i=0; i<nidx; i++) indices[i] += rstart;

    /* Build the index sets for each block */
    for (i=0; i<nloc; i++) {
      ierr   = ISCreateGeneral(PETSC_COMM_SELF,count[i],&indices[start],PETSC_COPY_VALUES,&is[i]);CHKERRQ(ierr);
      ierr   = ISSort(is[i]);CHKERRQ(ierr);
      start += count[i];
    }

    ierr = PetscFree(count);CHKERRQ(ierr);
    ierr = PetscFree(indices);CHKERRQ(ierr);
    ierr = ISDestroy(&isnumb);CHKERRQ(ierr);
    ierr = ISDestroy(&ispart);CHKERRQ(ierr);
  }
  *iis = is;
  PetscFunctionReturn(0);
}

PETSC_INTERN PetscErrorCode  PCGASMCreateStraddlingSubdomains(Mat A,PetscInt N,PetscInt *n,IS *iis[])
{
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  ierr = MatSubdomainsCreateCoalesce(A,N,n,iis);CHKERRQ(ierr);
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


.seealso: PCGASMSetSubdomains(), PCGASMDestroySubdomains()
@*/
PetscErrorCode  PCGASMCreateSubdomains(Mat A,PetscInt N,PetscInt *n,IS *iis[])
{
  PetscMPIInt     size;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(A,MAT_CLASSID,1);
  PetscValidPointer(iis,4);

  if (N < 1) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"Number of subdomains must be > 0, N = %D",N);
  ierr = MPI_Comm_size(PetscObjectComm((PetscObject)A),&size);CHKERRQ(ierr);
  if (N >= size) {
    *n = N/size + (N%size);
    ierr = PCGASMCreateLocalSubdomains(A,*n,iis);CHKERRQ(ierr);
  } else {
    ierr = PCGASMCreateStraddlingSubdomains(A,N,n,iis);CHKERRQ(ierr);
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

.seealso: PCGASMCreateSubdomains(), PCGASMSetSubdomains()
@*/
PetscErrorCode  PCGASMDestroySubdomains(PetscInt n,IS **iis,IS **ois)
{
  PetscInt       i;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (n <= 0) PetscFunctionReturn(0);
  if (ois) {
    PetscValidPointer(ois,3);
    if (*ois) {
      PetscValidPointer(*ois,3);
      for (i=0; i<n; i++) {
        ierr = ISDestroy(&(*ois)[i]);CHKERRQ(ierr);
      }
      ierr = PetscFree((*ois));CHKERRQ(ierr);
    }
  }
  if (iis) {
    PetscValidPointer(iis,2);
    if (*iis) {
      PetscValidPointer(*iis,2);
      for (i=0; i<n; i++) {
        ierr = ISDestroy(&(*iis)[i]);CHKERRQ(ierr);
      }
      ierr = PetscFree((*iis));CHKERRQ(ierr);
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
+  M, N               - the global number of grid points in the x and y directions
.  Mdomains, Ndomains - the global number of subdomains in the x and y directions
.  dof                - degrees of freedom per node
-  overlap            - overlap in mesh lines

   Output Parameters:
+  Nsub - the number of local subdomains created
.  iis  - array of index sets defining inner (nonoverlapping) subdomains
-  ois  - array of index sets defining outer (overlapping, if overlap > 0) subdomains


   Level: advanced

.seealso: PCGASMSetSubdomains(), PCGASMGetSubKSP(), PCGASMSetOverlap()
@*/
PetscErrorCode  PCGASMCreateSubdomains2D(PC pc,PetscInt M,PetscInt N,PetscInt Mdomains,PetscInt Ndomains,PetscInt dof,PetscInt overlap,PetscInt *nsub,IS **iis,IS **ois)
{
  PetscErrorCode ierr;
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
  IS             **xis = 0, **is = ois, **is_local = iis;

  PetscFunctionBegin;
  ierr = PetscObjectGetComm((PetscObject)pc, &comm);CHKERRQ(ierr);
  ierr = MPI_Comm_size(comm, &size);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(comm, &rank);CHKERRQ(ierr);
  ierr = MatGetOwnershipRange(pc->pmat, &first, &last);CHKERRQ(ierr);
  if (first%dof || last%dof) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Matrix row partitioning unsuitable for domain decomposition: local row range (%D,%D) "
                                      "does not respect the number of degrees of freedom per grid point %D", first, last, dof);

  /* Determine the number of domains with nonzero intersections with the local ownership range. */
  s      = 0;
  ystart = 0;
  for (j=0; j<Ndomains; ++j) {
    maxheight = N/Ndomains + ((N % Ndomains) > j); /* Maximal height of subdomain */
    if (maxheight < 2) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Too many %D subdomains in the vertical directon for mesh height %D", Ndomains, N);
    /* Vertical domain limits with an overlap. */
    ylow   = PetscMax(ystart - overlap,0);
    yhigh  = PetscMin(ystart + maxheight + overlap,N);
    xstart = 0;
    for (i=0; i<Mdomains; ++i) {
      maxwidth = M/Mdomains + ((M % Mdomains) > i); /* Maximal width of subdomain */
      if (maxwidth < 2) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Too many %D subdomains in the horizontal direction for mesh width %D", Mdomains, M);
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
  ierr   = PetscMalloc1(*nsub,is);CHKERRQ(ierr);
  ierr   = PetscMalloc1(*nsub,is_local);CHKERRQ(ierr);
  s      = 0;
  ystart = 0;
  for (j=0; j<Ndomains; ++j) {
    maxheight = N/Ndomains + ((N % Ndomains) > j); /* Maximal height of subdomain */
    if (maxheight < 2) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Too many %D subdomains in the vertical directon for mesh height %D", Ndomains, N);
    /* Vertical domain limits with an overlap. */
    y[0][0] = PetscMax(ystart - overlap,0);
    y[0][1] = PetscMin(ystart + maxheight + overlap,N);
    /* Vertical domain limits without an overlap. */
    y[1][0] = ystart;
    y[1][1] = PetscMin(ystart + maxheight,N);
    xstart  = 0;
    for (i=0; i<Mdomains; ++i) {
      maxwidth = M/Mdomains + ((M % Mdomains) > i); /* Maximal width of subdomain */
      if (maxwidth < 2) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Too many %D subdomains in the horizontal direction for mesh width %D", Mdomains, M);
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
          ierr = MPI_Comm_split(comm, color, rank, &subcomm);CHKERRQ(ierr);
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
              ierr           = PetscObjectReference((PetscObject)(*is)[s]);CHKERRQ(ierr);
              continue;
            } else {
              xis     = is_local;
              subcomm = ((PetscObject)(*is)[s])->comm;
            }
          } /* if (q == 1) */
          idx  = NULL;
          ierr = PetscMalloc1(nidx,&idx);CHKERRQ(ierr);
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
          ierr = ISCreateGeneral(subcomm,nidx,idx,PETSC_OWN_POINTER,(*xis)+s);CHKERRQ(ierr);
          if (split) {
            ierr = MPI_Comm_free(&subcomm);CHKERRQ(ierr);
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

.seealso: PCGASMSetOverlap(), PCGASMGetSubKSP(), PCGASMCreateSubdomains2D(),
          PCGASMSetSubdomains(), PCGASMGetSubmatrices()
@*/
PetscErrorCode  PCGASMGetSubdomains(PC pc,PetscInt *n,IS *iis[],IS *ois[])
{
  PC_GASM        *osm;
  PetscErrorCode ierr;
  PetscBool      match;
  PetscInt       i;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_CLASSID,1);
  ierr = PetscObjectTypeCompare((PetscObject)pc,PCGASM,&match);CHKERRQ(ierr);
  if (!match) SETERRQ2(PetscObjectComm((PetscObject)pc), PETSC_ERR_ARG_WRONG, "Incorrect object type: expected %s, got %s instead", PCGASM, ((PetscObject)pc)->type_name);
  osm = (PC_GASM*)pc->data;
  if (n) *n = osm->n;
  if (iis) {
    ierr = PetscMalloc1(osm->n, iis);CHKERRQ(ierr);
  }
  if (ois) {
    ierr = PetscMalloc1(osm->n, ois);CHKERRQ(ierr);
  }
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

.seealso: PCGASMSetOverlap(), PCGASMGetSubKSP(),
          PCGASMCreateSubdomains2D(), PCGASMSetSubdomains(), PCGASMGetSubdomains()
@*/
PetscErrorCode  PCGASMGetSubmatrices(PC pc,PetscInt *n,Mat *mat[])
{
  PC_GASM        *osm;
  PetscErrorCode ierr;
  PetscBool      match;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_CLASSID,1);
  PetscValidIntPointer(n,2);
  if (mat) PetscValidPointer(mat,3);
  if (!pc->setupcalled) SETERRQ(PetscObjectComm((PetscObject)pc),PETSC_ERR_ARG_WRONGSTATE,"Must call after KSPSetUp() or PCSetUp().");
  ierr = PetscObjectTypeCompare((PetscObject)pc,PCGASM,&match);CHKERRQ(ierr);
  if (!match) SETERRQ2(PetscObjectComm((PetscObject)pc), PETSC_ERR_ARG_WRONG, "Expected %s, got %s instead", PCGASM, ((PetscObject)pc)->type_name);
  osm = (PC_GASM*)pc->data;
  if (n) *n = osm->n;
  if (mat) *mat = osm->pmat;
  PetscFunctionReturn(0);
}

/*@
    PCGASMSetUseDMSubdomains - Indicates whether to use DMCreateDomainDecomposition() to define the subdomains, whenever possible.
    Logically Collective

    Input Parameter:
+   pc  - the preconditioner
-   flg - boolean indicating whether to use subdomains defined by the DM

    Options Database Key:
.   -pc_gasm_dm_subdomains -pc_gasm_overlap -pc_gasm_total_subdomains

    Level: intermediate

    Notes:
    PCGASMSetSubdomains(), PCGASMSetTotalSubdomains() or PCGASMSetOverlap() take precedence over PCGASMSetUseDMSubdomains(),
    so setting PCGASMSetSubdomains() with nontrivial subdomain ISs or any of PCGASMSetTotalSubdomains() and PCGASMSetOverlap()
    automatically turns the latter off.

.seealso: PCGASMGetUseDMSubdomains(), PCGASMSetSubdomains(), PCGASMSetOverlap()
          PCGASMCreateSubdomains2D()
@*/
PetscErrorCode  PCGASMSetUseDMSubdomains(PC pc,PetscBool flg)
{
  PC_GASM        *osm = (PC_GASM*)pc->data;
  PetscErrorCode ierr;
  PetscBool      match;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_CLASSID,1);
  PetscValidLogicalCollectiveBool(pc,flg,2);
  if (pc->setupcalled) SETERRQ(((PetscObject)pc)->comm,PETSC_ERR_ARG_WRONGSTATE,"Not for a setup PC.");
  ierr = PetscObjectTypeCompare((PetscObject)pc,PCGASM,&match);CHKERRQ(ierr);
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

.seealso: PCGASMSetUseDMSubdomains(), PCGASMSetOverlap()
          PCGASMCreateSubdomains2D()
@*/
PetscErrorCode  PCGASMGetUseDMSubdomains(PC pc,PetscBool* flg)
{
  PC_GASM        *osm = (PC_GASM*)pc->data;
  PetscErrorCode ierr;
  PetscBool      match;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_CLASSID,1);
  PetscValidBoolPointer(flg,2);
  ierr = PetscObjectTypeCompare((PetscObject)pc,PCGASM,&match);CHKERRQ(ierr);
  if (match) {
    if (flg) *flg = osm->dm_subdomains;
  }
  PetscFunctionReturn(0);
}
