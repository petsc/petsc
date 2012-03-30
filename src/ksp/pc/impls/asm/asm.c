
/*
  This file defines an additive Schwarz preconditioner for any Mat implementation.

  Note that each processor may have any number of subdomains. But in order to 
  deal easily with the VecScatter(), we treat each processor as if it has the
  same number of subdomains.

       n - total number of true subdomains on all processors
       n_local_true - actual number of subdomains on this processor
       n_local = maximum over all processors of n_local_true
*/
#include <petsc-private/pcimpl.h>     /*I "petscpc.h" I*/

typedef struct {
  PetscInt   n, n_local, n_local_true;
  PetscInt   overlap;             /* overlap requested by user */
  KSP        *ksp;                /* linear solvers for each block */
  VecScatter *restriction;        /* mapping from global to subregion */
  VecScatter *localization;       /* mapping from overlapping to non-overlapping subregion */
  VecScatter *prolongation;       /* mapping from subregion to global */
  Vec        *x,*y,*y_local;      /* work vectors */
  IS         *is;                 /* index set that defines each overlapping subdomain */
  IS         *is_local;           /* index set that defines each non-overlapping subdomain, may be NULL */
  Mat        *mat,*pmat;          /* mat is not currently used */
  PCASMType  type;                /* use reduced interpolation, restriction or both */
  PetscBool  type_set;            /* if user set this value (so won't change it for symmetric problems) */
  PetscBool  same_local_solves;   /* flag indicating whether all local solvers are same */
  PetscBool  sort_indices;        /* flag to sort subdomain indices */
} PC_ASM;

#undef __FUNCT__  
#define __FUNCT__ "PCView_ASM"
static PetscErrorCode PCView_ASM(PC pc,PetscViewer viewer)
{
  PC_ASM         *osm = (PC_ASM*)pc->data;
  PetscErrorCode ierr;
  PetscMPIInt    rank;
  PetscInt       i,bsz;
  PetscBool      iascii,isstring;
  PetscViewer    sviewer;


  PetscFunctionBegin;
  ierr = PetscTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&iascii);CHKERRQ(ierr);
  ierr = PetscTypeCompare((PetscObject)viewer,PETSCVIEWERSTRING,&isstring);CHKERRQ(ierr);
  if (iascii) {
    char overlaps[256] = "user-defined overlap",blocks[256] = "total subdomain blocks not yet set";
    if (osm->overlap >= 0) {ierr = PetscSNPrintf(overlaps,sizeof overlaps,"amount of overlap = %D",osm->overlap);CHKERRQ(ierr);}
    if (osm->n > 0) {ierr = PetscSNPrintf(blocks,sizeof blocks,"total subdomain blocks = %D",osm->n);CHKERRQ(ierr);}
    ierr = PetscViewerASCIIPrintf(viewer,"  Additive Schwarz: %s, %s\n",blocks,overlaps);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"  Additive Schwarz: restriction/interpolation type - %s\n",PCASMTypes[osm->type]);CHKERRQ(ierr);
    ierr = MPI_Comm_rank(((PetscObject)pc)->comm,&rank);CHKERRQ(ierr);
    if (osm->same_local_solves) {
      if (osm->ksp) {
        ierr = PetscViewerASCIIPrintf(viewer,"  Local solve is same for all blocks, in the following KSP and PC objects:\n");CHKERRQ(ierr);
        ierr = PetscViewerGetSingleton(viewer,&sviewer);CHKERRQ(ierr);
        if (!rank) {
          ierr = PetscViewerASCIIPushTab(viewer);CHKERRQ(ierr);
          ierr = KSPView(osm->ksp[0],sviewer);CHKERRQ(ierr);
          ierr = PetscViewerASCIIPopTab(viewer);CHKERRQ(ierr);
        }
        ierr = PetscViewerRestoreSingleton(viewer,&sviewer);CHKERRQ(ierr);
      }
    } else {
      ierr = PetscViewerASCIISynchronizedAllow(viewer,PETSC_TRUE);CHKERRQ(ierr);
      ierr = PetscViewerASCIISynchronizedPrintf(viewer,"  [%d] number of local blocks = %D\n",(int)rank,osm->n_local_true);CHKERRQ(ierr);
      ierr = PetscViewerFlush(viewer);CHKERRQ(ierr);
      ierr = PetscViewerASCIIPrintf(viewer,"  Local solve info for each block is in the following KSP and PC objects:\n");CHKERRQ(ierr);
      ierr = PetscViewerASCIIPushTab(viewer);CHKERRQ(ierr);
      ierr = PetscViewerASCIIPrintf(viewer,"- - - - - - - - - - - - - - - - - -\n");CHKERRQ(ierr);
      for (i=0; i<osm->n_local; i++) {
        ierr = PetscViewerGetSingleton(viewer,&sviewer);CHKERRQ(ierr);
        if (i < osm->n_local_true) {
          ierr = ISGetLocalSize(osm->is[i],&bsz);CHKERRQ(ierr);
          ierr = PetscViewerASCIISynchronizedPrintf(sviewer,"[%d] local block number %D, size = %D\n",(int)rank,i,bsz);CHKERRQ(ierr);
          ierr = KSPView(osm->ksp[i],sviewer);CHKERRQ(ierr);
          ierr = PetscViewerASCIISynchronizedPrintf(sviewer,"- - - - - - - - - - - - - - - - - -\n");CHKERRQ(ierr);
        }
        ierr = PetscViewerRestoreSingleton(viewer,&sviewer);CHKERRQ(ierr);
      }
      ierr = PetscViewerASCIIPopTab(viewer);CHKERRQ(ierr);
      ierr = PetscViewerFlush(viewer);CHKERRQ(ierr);
      ierr = PetscViewerASCIISynchronizedAllow(viewer,PETSC_FALSE);CHKERRQ(ierr);
    }
  } else if (isstring) {
    ierr = PetscViewerStringSPrintf(viewer," blocks=%D, overlap=%D, type=%s",osm->n,osm->overlap,PCASMTypes[osm->type]);CHKERRQ(ierr);
    ierr = PetscViewerGetSingleton(viewer,&sviewer);CHKERRQ(ierr);
    if (osm->ksp) {ierr = KSPView(osm->ksp[0],sviewer);CHKERRQ(ierr);}
    ierr = PetscViewerRestoreSingleton(viewer,&sviewer);CHKERRQ(ierr);
  } else {
    SETERRQ1(((PetscObject)pc)->comm,PETSC_ERR_SUP,"Viewer type %s not supported for PCASM",((PetscObject)viewer)->type_name);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PCASMPrintSubdomains"
static PetscErrorCode PCASMPrintSubdomains(PC pc)
{
  PC_ASM         *osm  = (PC_ASM*)pc->data;
  const char     *prefix;
  char           fname[PETSC_MAX_PATH_LEN+1];
  PetscViewer    viewer;
  PetscInt       i,j,nidx;
  const PetscInt *idx;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PCGetOptionsPrefix(pc,&prefix);CHKERRQ(ierr);
  ierr = PetscOptionsGetString(prefix,"-pc_asm_print_subdomains",fname,PETSC_MAX_PATH_LEN,PETSC_NULL);CHKERRQ(ierr);
  if (fname[0] == 0) { ierr = PetscStrcpy(fname,"stdout");CHKERRQ(ierr); };
  ierr = PetscViewerASCIIOpen(((PetscObject)pc)->comm,fname,&viewer);CHKERRQ(ierr);
  ierr = PetscViewerASCIISynchronizedAllow(viewer,PETSC_TRUE);CHKERRQ(ierr);
  for (i=0;i<osm->n_local_true;i++) {
    ierr = ISGetLocalSize(osm->is[i],&nidx);CHKERRQ(ierr);
    ierr = ISGetIndices(osm->is[i],&idx);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer, "Subdomain with overlap\n"); CHKERRQ(ierr);
    for (j=0; j<nidx; j++) {
      ierr = PetscViewerASCIISynchronizedPrintf(viewer,"%D ",idx[j]);CHKERRQ(ierr);
    }
    ierr = ISRestoreIndices(osm->is[i],&idx);CHKERRQ(ierr);
    ierr = PetscViewerASCIISynchronizedPrintf(viewer,"\n");CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer, "Subdomain without overlap\n"); CHKERRQ(ierr);
    if (osm->is_local) {
      ierr = ISGetLocalSize(osm->is_local[i],&nidx);CHKERRQ(ierr);
      ierr = ISGetIndices(osm->is_local[i],&idx);CHKERRQ(ierr);
      for (j=0; j<nidx; j++) {
        ierr = PetscViewerASCIISynchronizedPrintf(viewer,"%D ",idx[j]);CHKERRQ(ierr);
      }
      ierr = ISRestoreIndices(osm->is_local[i],&idx);CHKERRQ(ierr);
      ierr = PetscViewerASCIISynchronizedPrintf(viewer,"\n");CHKERRQ(ierr);
    }
    else {
      ierr = PetscViewerASCIIPrintf(viewer, "empty\n"); CHKERRQ(ierr);
    }
  }
  ierr = PetscViewerFlush(viewer);CHKERRQ(ierr);
  ierr = PetscViewerASCIISynchronizedAllow(viewer,PETSC_FALSE);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PCSetUp_ASM"
static PetscErrorCode PCSetUp_ASM(PC pc)
{
  PC_ASM         *osm  = (PC_ASM*)pc->data;
  PetscErrorCode ierr;
  PetscBool      symset,flg;
  PetscInt       i,m,m_local,firstRow,lastRow;
  PetscMPIInt    size;
  MatReuse       scall = MAT_REUSE_MATRIX;
  IS             isl;
  KSP            ksp;
  PC             subpc;
  const char     *prefix,*pprefix;
  Vec            vec;
  DM             *domain_dm = PETSC_NULL;

  PetscFunctionBegin;
  if (!pc->setupcalled) {

    if (!osm->type_set) {
      ierr = MatIsSymmetricKnown(pc->pmat,&symset,&flg);CHKERRQ(ierr);
      if (symset && flg) { osm->type = PC_ASM_BASIC; }
    }

    /* Note: if osm->n_local_true has been set, it is at least 1. */
    if (osm->n == PETSC_DECIDE && osm->n_local_true < 1) { 
      /* no subdomains given */
      /* try pc->dm first */
      if(pc->dm) {
        char      ddm_name[1024];
        DM        ddm;
        PetscBool flg;
        PetscInt     num_domains, d;
        char         **domain_names;
        IS           *domain_is;
        /* Allow the user to request a decomposition DM by name */
        ierr = PetscStrncpy(ddm_name, "", 1024); CHKERRQ(ierr);
        ierr = PetscOptionsString("-pc_asm_decomposition_dm", "Name of the DM defining the composition", "PCSetDM", ddm_name, ddm_name,1024,&flg); CHKERRQ(ierr);
        if(flg) {
          ierr = DMCreateDecompositionDM(pc->dm, ddm_name, &ddm); CHKERRQ(ierr);
          if(!ddm) {
            SETERRQ1(((PetscObject)pc)->comm, PETSC_ERR_ARG_WRONGSTATE, "Uknown DM decomposition name %s", ddm_name);
          }
          ierr = PetscInfo(pc,"Using decomposition DM defined using options database\n");CHKERRQ(ierr);
          ierr = PCSetDM(pc,ddm); CHKERRQ(ierr);
        }
        ierr = DMCreateDecomposition(pc->dm, &num_domains, &domain_names, &domain_is, &domain_dm);    CHKERRQ(ierr);
        if(num_domains) {
          ierr = PCASMSetLocalSubdomains(pc, num_domains, domain_is, PETSC_NULL);CHKERRQ(ierr);
        }
        for(d = 0; d < num_domains; ++d) {
          ierr = PetscFree(domain_names[d]); CHKERRQ(ierr);
          ierr = ISDestroy(&domain_is[d]);   CHKERRQ(ierr);
        }
        ierr = PetscFree(domain_names);CHKERRQ(ierr);
        ierr = PetscFree(domain_is);CHKERRQ(ierr);
      }
      if (osm->n == PETSC_DECIDE && osm->n_local_true < 1) {
        /* still no subdomains; use one subdomain per processor */
        osm->n_local = osm->n_local_true = 1;
        ierr = MPI_Comm_size(((PetscObject)pc)->comm,&size);CHKERRQ(ierr);
        osm->n = size;
      }
    } else if (osm->n == PETSC_DECIDE) {
      /* determine global number of subdomains */
      PetscInt inwork[2],outwork[2];
      inwork[0] = inwork[1] = osm->n_local_true;
      ierr = MPI_Allreduce(inwork,outwork,1,MPIU_2INT,PetscMaxSum_Op,((PetscObject)pc)->comm);CHKERRQ(ierr);
      osm->n_local = outwork[0];
      osm->n       = outwork[1];
    }
    if (!osm->is){ /* create the index sets */
      ierr = PCASMCreateSubdomains(pc->pmat,osm->n_local_true,&osm->is);CHKERRQ(ierr);
    }
    if (osm->n_local_true > 1 && !osm->is_local) {
      ierr = PetscMalloc(osm->n_local_true*sizeof(IS),&osm->is_local);CHKERRQ(ierr);
      for (i=0; i<osm->n_local_true; i++) {
        if (osm->overlap > 0) { /* With positive overlap, osm->is[i] will be modified */
          ierr = ISDuplicate(osm->is[i],&osm->is_local[i]);CHKERRQ(ierr);
          ierr = ISCopy(osm->is[i],osm->is_local[i]);CHKERRQ(ierr);
        } else {
          ierr = PetscObjectReference((PetscObject)osm->is[i]);CHKERRQ(ierr);
          osm->is_local[i] = osm->is[i];
        }
      }
    }
    ierr = PCGetOptionsPrefix(pc,&prefix);CHKERRQ(ierr);
    flg  = PETSC_FALSE;
    ierr = PetscOptionsGetBool(prefix,"-pc_asm_print_subdomains",&flg,PETSC_NULL);CHKERRQ(ierr);
    if (flg) { ierr = PCASMPrintSubdomains(pc);CHKERRQ(ierr); }

    if (osm->overlap > 0) {
      /* Extend the "overlapping" regions by a number of steps */
      ierr = MatIncreaseOverlap(pc->pmat,osm->n_local_true,osm->is,osm->overlap);CHKERRQ(ierr);
    }
    if (osm->sort_indices) {
      for (i=0; i<osm->n_local_true; i++) {
        ierr = ISSort(osm->is[i]);CHKERRQ(ierr);
        if (osm->is_local) {
          ierr = ISSort(osm->is_local[i]);CHKERRQ(ierr);
        }
      }
    }

    /* Create the local work vectors and scatter contexts */
    ierr = MatGetVecs(pc->pmat,&vec,0);CHKERRQ(ierr);
    ierr = PetscMalloc(osm->n_local*sizeof(VecScatter),&osm->restriction);CHKERRQ(ierr);
    if (osm->is_local) {ierr = PetscMalloc(osm->n_local*sizeof(VecScatter),&osm->localization);CHKERRQ(ierr);}
    ierr = PetscMalloc(osm->n_local*sizeof(VecScatter),&osm->prolongation);CHKERRQ(ierr);
    ierr = PetscMalloc(osm->n_local*sizeof(Vec),&osm->x);CHKERRQ(ierr);
    ierr = PetscMalloc(osm->n_local*sizeof(Vec),&osm->y);CHKERRQ(ierr);
    ierr = PetscMalloc(osm->n_local*sizeof(Vec),&osm->y_local);CHKERRQ(ierr);
    ierr = VecGetOwnershipRange(vec, &firstRow, &lastRow);CHKERRQ(ierr);
    for (i=0; i<osm->n_local_true; ++i, firstRow += m_local) {
      ierr = ISGetLocalSize(osm->is[i],&m);CHKERRQ(ierr);
      ierr = VecCreateSeq(PETSC_COMM_SELF,m,&osm->x[i]);CHKERRQ(ierr);
      ierr = ISCreateStride(PETSC_COMM_SELF,m,0,1,&isl);CHKERRQ(ierr);
      ierr = VecScatterCreate(vec,osm->is[i],osm->x[i],isl,&osm->restriction[i]);CHKERRQ(ierr);
      ierr = ISDestroy(&isl);CHKERRQ(ierr);
      ierr = VecDuplicate(osm->x[i],&osm->y[i]);CHKERRQ(ierr);
      if (osm->is_local) {
        ISLocalToGlobalMapping ltog;
        IS                     isll;
        const PetscInt         *idx_local;
        PetscInt               *idx,nout;

        ierr = ISLocalToGlobalMappingCreateIS(osm->is[i],&ltog);CHKERRQ(ierr);
        ierr = ISGetLocalSize(osm->is_local[i],&m_local);CHKERRQ(ierr);
        ierr = ISGetIndices(osm->is_local[i], &idx_local);CHKERRQ(ierr);
        ierr = PetscMalloc(m_local*sizeof(PetscInt),&idx);CHKERRQ(ierr);
        ierr = ISGlobalToLocalMappingApply(ltog,IS_GTOLM_DROP,m_local,idx_local,&nout,idx);CHKERRQ(ierr);
        ierr = ISLocalToGlobalMappingDestroy(&ltog);CHKERRQ(ierr);
        if (nout != m_local) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_PLIB,"is_local not a subset of is");
        ierr = ISRestoreIndices(osm->is_local[i], &idx_local);CHKERRQ(ierr);
        ierr = ISCreateGeneral(PETSC_COMM_SELF,m_local,idx,PETSC_OWN_POINTER,&isll);CHKERRQ(ierr);
        ierr = ISCreateStride(PETSC_COMM_SELF,m_local,0,1,&isl);CHKERRQ(ierr);
        ierr = VecCreateSeq(PETSC_COMM_SELF,m_local,&osm->y_local[i]);CHKERRQ(ierr);
        ierr = VecScatterCreate(osm->y[i],isll,osm->y_local[i],isl,&osm->localization[i]);CHKERRQ(ierr);
        ierr = ISDestroy(&isll);CHKERRQ(ierr);

        ierr = VecScatterCreate(vec,osm->is_local[i],osm->y_local[i],isl,&osm->prolongation[i]);CHKERRQ(ierr);
        ierr = ISDestroy(&isl);CHKERRQ(ierr);
      } else {
        ierr = VecGetLocalSize(vec,&m_local);CHKERRQ(ierr);
        osm->y_local[i] = osm->y[i];
        ierr = PetscObjectReference((PetscObject) osm->y[i]);CHKERRQ(ierr);
        osm->prolongation[i] = osm->restriction[i];
        ierr = PetscObjectReference((PetscObject) osm->restriction[i]);CHKERRQ(ierr);
      }
    }
    if (firstRow != lastRow) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_PLIB, "Specified ASM subdomain sizes were invalid: %d != %d", firstRow, lastRow);
    for (i=osm->n_local_true; i<osm->n_local; i++) {
      ierr = VecCreateSeq(PETSC_COMM_SELF,0,&osm->x[i]);CHKERRQ(ierr);
      ierr = VecDuplicate(osm->x[i],&osm->y[i]);CHKERRQ(ierr);
      ierr = VecDuplicate(osm->x[i],&osm->y_local[i]);CHKERRQ(ierr);
      ierr = ISCreateStride(PETSC_COMM_SELF,0,0,1,&isl);CHKERRQ(ierr);
      ierr = VecScatterCreate(vec,isl,osm->x[i],isl,&osm->restriction[i]);CHKERRQ(ierr);
      if (osm->is_local) {
        ierr = VecScatterCreate(osm->y[i],isl,osm->y_local[i],isl,&osm->localization[i]);CHKERRQ(ierr);
        ierr = VecScatterCreate(vec,isl,osm->x[i],isl,&osm->prolongation[i]);CHKERRQ(ierr);
      } else {
        osm->prolongation[i] = osm->restriction[i];
        ierr = PetscObjectReference((PetscObject) osm->restriction[i]);CHKERRQ(ierr);
      }
      ierr = ISDestroy(&isl);CHKERRQ(ierr);
    }
    ierr = VecDestroy(&vec);CHKERRQ(ierr);

    if (!osm->ksp) {
      /* Create the local solvers */
      ierr = PetscMalloc(osm->n_local_true*sizeof(KSP *),&osm->ksp);CHKERRQ(ierr);
      if(domain_dm) {
        ierr = PetscInfo(pc,"Setting up ASM subproblems using the embedded DM\n");CHKERRQ(ierr);
      }
      for (i=0; i<osm->n_local_true; i++) {
        ierr = KSPCreate(PETSC_COMM_SELF,&ksp);CHKERRQ(ierr);
        ierr = PetscLogObjectParent(pc,ksp);CHKERRQ(ierr);
        ierr = PetscObjectIncrementTabLevel((PetscObject)ksp,(PetscObject)pc,1);CHKERRQ(ierr);
        ierr = KSPSetType(ksp,KSPPREONLY);CHKERRQ(ierr);
        ierr = KSPGetPC(ksp,&subpc);CHKERRQ(ierr);
        ierr = PCGetOptionsPrefix(pc,&prefix);CHKERRQ(ierr);
        ierr = KSPSetOptionsPrefix(ksp,prefix);CHKERRQ(ierr);
        ierr = KSPAppendOptionsPrefix(ksp,"sub_");CHKERRQ(ierr);
        if(domain_dm){
          ierr = KSPSetDM(ksp, domain_dm[i]);CHKERRQ(ierr);
          ierr = KSPSetDMActive(ksp, PETSC_FALSE);CHKERRQ(ierr);
          ierr = DMDestroy(&domain_dm[i]); CHKERRQ(ierr);
        }
        osm->ksp[i] = ksp;
      }
      if(domain_dm) {
        ierr = PetscFree(domain_dm);CHKERRQ(ierr);
      }
    }
    scall = MAT_INITIAL_MATRIX;
  } else {
    /* 
       Destroy the blocks from the previous iteration
    */
    if (pc->flag == DIFFERENT_NONZERO_PATTERN) {
      ierr = MatDestroyMatrices(osm->n_local_true,&osm->pmat);CHKERRQ(ierr);
      scall = MAT_INITIAL_MATRIX;
    }
  }

  /* 
     Extract out the submatrices
  */
  ierr = MatGetSubMatrices(pc->pmat,osm->n_local_true,osm->is,osm->is,scall,&osm->pmat);CHKERRQ(ierr);
  if (scall == MAT_INITIAL_MATRIX) {
    ierr = PetscObjectGetOptionsPrefix((PetscObject)pc->pmat,&pprefix);CHKERRQ(ierr);
    for (i=0; i<osm->n_local_true; i++) {
      ierr = PetscLogObjectParent(pc,osm->pmat[i]);CHKERRQ(ierr);
      ierr = PetscObjectSetOptionsPrefix((PetscObject)osm->pmat[i],pprefix);CHKERRQ(ierr);
    }
  }

  /* Return control to the user so that the submatrices can be modified (e.g., to apply
     different boundary conditions for the submatrices than for the global problem) */
  ierr = PCModifySubMatrices(pc,osm->n_local_true,osm->is,osm->is,osm->pmat,pc->modifysubmatricesP);CHKERRQ(ierr);

  /* 
     Loop over subdomains putting them into local ksp
  */
  for (i=0; i<osm->n_local_true; i++) {
    ierr = KSPSetOperators(osm->ksp[i],osm->pmat[i],osm->pmat[i],pc->flag);CHKERRQ(ierr);
    if (!pc->setupcalled) {
      ierr = KSPSetFromOptions(osm->ksp[i]);CHKERRQ(ierr);
    }
  }

  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PCSetUpOnBlocks_ASM"
static PetscErrorCode PCSetUpOnBlocks_ASM(PC pc)
{
  PC_ASM         *osm = (PC_ASM*)pc->data;
  PetscErrorCode ierr;
  PetscInt       i;

  PetscFunctionBegin;
  for (i=0; i<osm->n_local_true; i++) {
    ierr = KSPSetUp(osm->ksp[i]);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PCApply_ASM"
static PetscErrorCode PCApply_ASM(PC pc,Vec x,Vec y)
{
  PC_ASM         *osm = (PC_ASM*)pc->data;
  PetscErrorCode ierr;
  PetscInt       i,n_local = osm->n_local,n_local_true = osm->n_local_true;
  ScatterMode    forward = SCATTER_FORWARD,reverse = SCATTER_REVERSE;

  PetscFunctionBegin;
  /*
     Support for limiting the restriction or interpolation to only local 
     subdomain values (leaving the other values 0). 
  */
  if (!(osm->type & PC_ASM_RESTRICT)) {
    forward = SCATTER_FORWARD_LOCAL;
    /* have to zero the work RHS since scatter may leave some slots empty */
    for (i=0; i<n_local_true; i++) {
      ierr = VecZeroEntries(osm->x[i]);CHKERRQ(ierr);
    }
  }
  if (!(osm->type & PC_ASM_INTERPOLATE)) {
    reverse = SCATTER_REVERSE_LOCAL;
  }

  for (i=0; i<n_local; i++) {
    ierr = VecScatterBegin(osm->restriction[i],x,osm->x[i],INSERT_VALUES,forward);CHKERRQ(ierr);
  }
  ierr = VecZeroEntries(y);CHKERRQ(ierr);
  /* do the local solves */
  for (i=0; i<n_local_true; i++) {
    ierr = VecScatterEnd(osm->restriction[i],x,osm->x[i],INSERT_VALUES,forward);CHKERRQ(ierr);
    ierr = KSPSolve(osm->ksp[i],osm->x[i],osm->y[i]);CHKERRQ(ierr);
    if (osm->localization) {
      ierr = VecScatterBegin(osm->localization[i],osm->y[i],osm->y_local[i],INSERT_VALUES,forward);CHKERRQ(ierr);
      ierr = VecScatterEnd(osm->localization[i],osm->y[i],osm->y_local[i],INSERT_VALUES,forward);CHKERRQ(ierr);
    }
    ierr = VecScatterBegin(osm->prolongation[i],osm->y_local[i],y,ADD_VALUES,reverse);CHKERRQ(ierr);
  }
  /* handle the rest of the scatters that do not have local solves */
  for (i=n_local_true; i<n_local; i++) {
    ierr = VecScatterEnd(osm->restriction[i],x,osm->x[i],INSERT_VALUES,forward);CHKERRQ(ierr);
    ierr = VecScatterBegin(osm->prolongation[i],osm->y_local[i],y,ADD_VALUES,reverse);CHKERRQ(ierr);
  }
  for (i=0; i<n_local; i++) {
    ierr = VecScatterEnd(osm->prolongation[i],osm->y_local[i],y,ADD_VALUES,reverse);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PCApplyTranspose_ASM"
static PetscErrorCode PCApplyTranspose_ASM(PC pc,Vec x,Vec y)
{
  PC_ASM         *osm = (PC_ASM*)pc->data;
  PetscErrorCode ierr;
  PetscInt       i,n_local = osm->n_local,n_local_true = osm->n_local_true;
  ScatterMode    forward = SCATTER_FORWARD,reverse = SCATTER_REVERSE;

  PetscFunctionBegin;
  /*
     Support for limiting the restriction or interpolation to only local 
     subdomain values (leaving the other values 0).

     Note: these are reversed from the PCApply_ASM() because we are applying the 
     transpose of the three terms 
  */
  if (!(osm->type & PC_ASM_INTERPOLATE)) {
    forward = SCATTER_FORWARD_LOCAL;
    /* have to zero the work RHS since scatter may leave some slots empty */
    for (i=0; i<n_local_true; i++) {
      ierr = VecZeroEntries(osm->x[i]);CHKERRQ(ierr);
    }
  }
  if (!(osm->type & PC_ASM_RESTRICT)) {
    reverse = SCATTER_REVERSE_LOCAL;
  }

  for (i=0; i<n_local; i++) {
    ierr = VecScatterBegin(osm->restriction[i],x,osm->x[i],INSERT_VALUES,forward);CHKERRQ(ierr);
  }
  ierr = VecZeroEntries(y);CHKERRQ(ierr);
  /* do the local solves */
  for (i=0; i<n_local_true; i++) {
    ierr = VecScatterEnd(osm->restriction[i],x,osm->x[i],INSERT_VALUES,forward);CHKERRQ(ierr);
    ierr = KSPSolveTranspose(osm->ksp[i],osm->x[i],osm->y[i]);CHKERRQ(ierr); 
    if (osm->localization) {
      ierr = VecScatterBegin(osm->localization[i],osm->y[i],osm->y_local[i],INSERT_VALUES,forward);CHKERRQ(ierr);
      ierr = VecScatterEnd(osm->localization[i],osm->y[i],osm->y_local[i],INSERT_VALUES,forward);CHKERRQ(ierr);
    }
    ierr = VecScatterBegin(osm->prolongation[i],osm->y_local[i],y,ADD_VALUES,reverse);CHKERRQ(ierr);
  }
  /* handle the rest of the scatters that do not have local solves */
  for (i=n_local_true; i<n_local; i++) {
    ierr = VecScatterEnd(osm->restriction[i],x,osm->x[i],INSERT_VALUES,forward);CHKERRQ(ierr);
    ierr = VecScatterBegin(osm->prolongation[i],osm->y_local[i],y,ADD_VALUES,reverse);CHKERRQ(ierr);
  }
  for (i=0; i<n_local; i++) {
    ierr = VecScatterEnd(osm->prolongation[i],osm->y_local[i],y,ADD_VALUES,reverse);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PCReset_ASM"
static PetscErrorCode PCReset_ASM(PC pc)
{
  PC_ASM         *osm = (PC_ASM*)pc->data;
  PetscErrorCode ierr;
  PetscInt       i;

  PetscFunctionBegin;
  if (osm->ksp) {
    for (i=0; i<osm->n_local_true; i++) {
      ierr = KSPReset(osm->ksp[i]);CHKERRQ(ierr);
    }
  }
  if (osm->pmat) {
    if (osm->n_local_true > 0) {
      ierr = MatDestroyMatrices(osm->n_local_true,&osm->pmat);CHKERRQ(ierr);
    }
  }
  if (osm->restriction) {
    for (i=0; i<osm->n_local; i++) {
      ierr = VecScatterDestroy(&osm->restriction[i]);CHKERRQ(ierr);
      if (osm->localization) {ierr = VecScatterDestroy(&osm->localization[i]);CHKERRQ(ierr);}
      ierr = VecScatterDestroy(&osm->prolongation[i]);CHKERRQ(ierr);
      ierr = VecDestroy(&osm->x[i]);CHKERRQ(ierr);
      ierr = VecDestroy(&osm->y[i]);CHKERRQ(ierr);
      ierr = VecDestroy(&osm->y_local[i]);CHKERRQ(ierr);
    }
    ierr = PetscFree(osm->restriction);CHKERRQ(ierr);
    if (osm->localization) {ierr = PetscFree(osm->localization);CHKERRQ(ierr);}
    ierr = PetscFree(osm->prolongation);CHKERRQ(ierr);
    ierr = PetscFree(osm->x);CHKERRQ(ierr);
    ierr = PetscFree(osm->y);CHKERRQ(ierr);
    ierr = PetscFree(osm->y_local);CHKERRQ(ierr);
  }
  if (osm->is) {
    ierr = PCASMDestroySubdomains(osm->n_local_true,osm->is,osm->is_local);CHKERRQ(ierr);
    osm->is       = 0;
    osm->is_local = 0;
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PCDestroy_ASM"
static PetscErrorCode PCDestroy_ASM(PC pc)
{
  PC_ASM         *osm = (PC_ASM*)pc->data;
  PetscErrorCode ierr;
  PetscInt       i;

  PetscFunctionBegin;
  ierr = PCReset_ASM(pc);CHKERRQ(ierr);
  if (osm->ksp) {
    for (i=0; i<osm->n_local_true; i++) {
      ierr = KSPDestroy(&osm->ksp[i]);CHKERRQ(ierr);
    }
    ierr = PetscFree(osm->ksp);CHKERRQ(ierr);
  }
  ierr = PetscFree(pc->data);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PCSetFromOptions_ASM"
static PetscErrorCode PCSetFromOptions_ASM(PC pc)
{
  PC_ASM         *osm = (PC_ASM*)pc->data;
  PetscErrorCode ierr;
  PetscInt       blocks,ovl;
  PetscBool      symset,flg;
  PCASMType      asmtype;

  PetscFunctionBegin;
  /* set the type to symmetric if matrix is symmetric */
  if (!osm->type_set && pc->pmat) {
    ierr = MatIsSymmetricKnown(pc->pmat,&symset,&flg);CHKERRQ(ierr);
    if (symset && flg) { osm->type = PC_ASM_BASIC; }
  }
  ierr = PetscOptionsHead("Additive Schwarz options");CHKERRQ(ierr);
    ierr = PetscOptionsInt("-pc_asm_blocks","Number of subdomains","PCASMSetTotalSubdomains",osm->n,&blocks,&flg);CHKERRQ(ierr);
    if (flg) {ierr = PCASMSetTotalSubdomains(pc,blocks,PETSC_NULL,PETSC_NULL);CHKERRQ(ierr); }
    ierr = PetscOptionsInt("-pc_asm_overlap","Number of grid points overlap","PCASMSetOverlap",osm->overlap,&ovl,&flg);CHKERRQ(ierr);
    if (flg) {ierr = PCASMSetOverlap(pc,ovl);CHKERRQ(ierr); }
    flg  = PETSC_FALSE;
    ierr = PetscOptionsEnum("-pc_asm_type","Type of restriction/extension","PCASMSetType",PCASMTypes,(PetscEnum)osm->type,(PetscEnum*)&asmtype,&flg);CHKERRQ(ierr);
    if (flg) {ierr = PCASMSetType(pc,asmtype);CHKERRQ(ierr); }
  ierr = PetscOptionsTail();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*------------------------------------------------------------------------------------*/

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "PCASMSetLocalSubdomains_ASM"
PetscErrorCode  PCASMSetLocalSubdomains_ASM(PC pc,PetscInt n,IS is[],IS is_local[])
{
  PC_ASM         *osm = (PC_ASM*)pc->data;
  PetscErrorCode ierr;
  PetscInt       i;

  PetscFunctionBegin;
  if (n < 1) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Each process must have 1 or more blocks, n = %D",n);
  if (pc->setupcalled && (n != osm->n_local_true || is)) SETERRQ(((PetscObject)pc)->comm,PETSC_ERR_ARG_WRONGSTATE,"PCASMSetLocalSubdomains() should be called before calling PCSetUp().");

  if (!pc->setupcalled) {
    if (is) {
      for (i=0; i<n; i++) {ierr = PetscObjectReference((PetscObject)is[i]);CHKERRQ(ierr);}
    }
    if (is_local) {
      for (i=0; i<n; i++) {ierr = PetscObjectReference((PetscObject)is_local[i]);CHKERRQ(ierr);}
    }
    if (osm->is) {
      ierr = PCASMDestroySubdomains(osm->n_local_true,osm->is,osm->is_local);CHKERRQ(ierr);
    }
    osm->n_local_true = n;
    osm->is           = 0;
    osm->is_local     = 0;
    if (is) {
      ierr = PetscMalloc(n*sizeof(IS),&osm->is);CHKERRQ(ierr);
      for (i=0; i<n; i++) { osm->is[i] = is[i]; }
      /* Flag indicating that the user has set overlapping subdomains so PCASM should not increase their size. */
      osm->overlap = -1;
    }
    if (is_local) {
      ierr = PetscMalloc(n*sizeof(IS),&osm->is_local);CHKERRQ(ierr);
      for (i=0; i<n; i++) { osm->is_local[i] = is_local[i]; }
    }
  }
  PetscFunctionReturn(0);
}
EXTERN_C_END

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "PCASMSetTotalSubdomains_ASM"
PetscErrorCode  PCASMSetTotalSubdomains_ASM(PC pc,PetscInt N,IS *is,IS *is_local)
{
  PC_ASM         *osm = (PC_ASM*)pc->data;
  PetscErrorCode ierr;
  PetscMPIInt    rank,size;
  PetscInt       n;

  PetscFunctionBegin;
  if (N < 1) SETERRQ1(((PetscObject)pc)->comm,PETSC_ERR_ARG_OUTOFRANGE,"Number of total blocks must be > 0, N = %D",N);
  if (is || is_local) SETERRQ(((PetscObject)pc)->comm,PETSC_ERR_SUP,"Use PCASMSetLocalSubdomains() to set specific index sets\n\they cannot be set globally yet.");

  /*
     Split the subdomains equally among all processors
  */
  ierr = MPI_Comm_rank(((PetscObject)pc)->comm,&rank);CHKERRQ(ierr);
  ierr = MPI_Comm_size(((PetscObject)pc)->comm,&size);CHKERRQ(ierr);
  n = N/size + ((N % size) > rank);
  if (!n) SETERRQ3(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Process %d must have at least one block: total processors %d total blocks %D",(int)rank,(int)size,N);
  if (pc->setupcalled && n != osm->n_local_true) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"PCASMSetTotalSubdomains() should be called before PCSetUp().");
  if (!pc->setupcalled) {
    if (osm->is) {
      ierr = PCASMDestroySubdomains(osm->n_local_true,osm->is,osm->is_local);CHKERRQ(ierr);
    }
    osm->n_local_true = n;
    osm->is           = 0;
    osm->is_local     = 0;
  }
  PetscFunctionReturn(0);
}
EXTERN_C_END

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "PCASMSetOverlap_ASM"
PetscErrorCode  PCASMSetOverlap_ASM(PC pc,PetscInt ovl)
{
  PC_ASM *osm = (PC_ASM*)pc->data;

  PetscFunctionBegin;
  if (ovl < 0) SETERRQ(((PetscObject)pc)->comm,PETSC_ERR_ARG_OUTOFRANGE,"Negative overlap value requested");
  if (pc->setupcalled && ovl != osm->overlap) SETERRQ(((PetscObject)pc)->comm,PETSC_ERR_ARG_WRONGSTATE,"PCASMSetOverlap() should be called before PCSetUp().");
  if (!pc->setupcalled) {
    osm->overlap = ovl;
  }
  PetscFunctionReturn(0);
}
EXTERN_C_END

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "PCASMSetType_ASM"
PetscErrorCode  PCASMSetType_ASM(PC pc,PCASMType type)
{
  PC_ASM *osm = (PC_ASM*)pc->data;

  PetscFunctionBegin;
  osm->type     = type;
  osm->type_set = PETSC_TRUE;
  PetscFunctionReturn(0);
}
EXTERN_C_END

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "PCASMSetSortIndices_ASM"
PetscErrorCode  PCASMSetSortIndices_ASM(PC pc,PetscBool  doSort)
{
  PC_ASM *osm = (PC_ASM*)pc->data;

  PetscFunctionBegin;
  osm->sort_indices = doSort;
  PetscFunctionReturn(0);
}
EXTERN_C_END

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "PCASMGetSubKSP_ASM"
PetscErrorCode  PCASMGetSubKSP_ASM(PC pc,PetscInt *n_local,PetscInt *first_local,KSP **ksp)
{
  PC_ASM         *osm = (PC_ASM*)pc->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (osm->n_local_true < 1) SETERRQ(((PetscObject)pc)->comm,PETSC_ERR_ORDER,"Need to call PCSetUP() on PC (or KSPSetUp() on the outer KSP object) before calling here");

  if (n_local) {
    *n_local = osm->n_local_true;
  }
  if (first_local) {
    ierr = MPI_Scan(&osm->n_local_true,first_local,1,MPIU_INT,MPI_SUM,((PetscObject)pc)->comm);CHKERRQ(ierr);
    *first_local -= osm->n_local_true;
  }
  if (ksp) {
    /* Assume that local solves are now different; not necessarily
       true though!  This flag is used only for PCView_ASM() */
    *ksp                   = osm->ksp;
    osm->same_local_solves = PETSC_FALSE;
  }
  PetscFunctionReturn(0);
}
EXTERN_C_END


#undef __FUNCT__  
#define __FUNCT__ "PCASMSetLocalSubdomains"
/*@C
    PCASMSetLocalSubdomains - Sets the local subdomains (for this processor only) for the additive Schwarz preconditioner. 

    Collective on PC 

    Input Parameters:
+   pc - the preconditioner context
.   n - the number of subdomains for this processor (default value = 1)
.   is - the index set that defines the subdomains for this processor
         (or PETSC_NULL for PETSc to determine subdomains)
-   is_local - the index sets that define the local part of the subdomains for this processor
         (or PETSC_NULL to use the default of 1 subdomain per process)

    Notes:
    The IS numbering is in the parallel, global numbering of the vector for both is and is_local

    By default the ASM preconditioner uses 1 block per processor.  

    Use PCASMSetTotalSubdomains() to set the subdomains for all processors.

    Level: advanced

.keywords: PC, ASM, set, local, subdomains, additive Schwarz

.seealso: PCASMSetTotalSubdomains(), PCASMSetOverlap(), PCASMGetSubKSP(),
          PCASMCreateSubdomains2D(), PCASMGetLocalSubdomains()
@*/
PetscErrorCode  PCASMSetLocalSubdomains(PC pc,PetscInt n,IS is[],IS is_local[])
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_CLASSID,1);
  ierr = PetscTryMethod(pc,"PCASMSetLocalSubdomains_C",(PC,PetscInt,IS[],IS[]),(pc,n,is,is_local));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PCASMSetTotalSubdomains"
/*@C
    PCASMSetTotalSubdomains - Sets the subdomains for all processors for the 
    additive Schwarz preconditioner.  Either all or no processors in the
    PC communicator must call this routine, with the same index sets.

    Collective on PC

    Input Parameters:
+   pc - the preconditioner context
.   N  - the number of subdomains for all processors
.   is - the index sets that define the subdomains for all processors
         (or PETSC_NULL to ask PETSc to compe up with subdomains)
-   is_local - the index sets that define the local part of the subdomains for this processor
         (or PETSC_NULL to use the default of 1 subdomain per process)

    Options Database Key:
    To set the total number of subdomain blocks rather than specify the
    index sets, use the option
.    -pc_asm_blocks <blks> - Sets total blocks

    Notes:
    Currently you cannot use this to set the actual subdomains with the argument is.

    By default the ASM preconditioner uses 1 block per processor.  

    These index sets cannot be destroyed until after completion of the
    linear solves for which the ASM preconditioner is being used.

    Use PCASMSetLocalSubdomains() to set local subdomains.

    The IS numbering is in the parallel, global numbering of the vector for both is and is_local

    Level: advanced

.keywords: PC, ASM, set, total, global, subdomains, additive Schwarz

.seealso: PCASMSetLocalSubdomains(), PCASMSetOverlap(), PCASMGetSubKSP(),
          PCASMCreateSubdomains2D()
@*/
PetscErrorCode  PCASMSetTotalSubdomains(PC pc,PetscInt N,IS is[],IS is_local[])
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_CLASSID,1);
  ierr = PetscTryMethod(pc,"PCASMSetTotalSubdomains_C",(PC,PetscInt,IS[],IS[]),(pc,N,is,is_local));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PCASMSetOverlap"
/*@
    PCASMSetOverlap - Sets the overlap between a pair of subdomains for the
    additive Schwarz preconditioner.  Either all or no processors in the
    PC communicator must call this routine. 

    Logically Collective on PC

    Input Parameters:
+   pc  - the preconditioner context
-   ovl - the amount of overlap between subdomains (ovl >= 0, default value = 1)

    Options Database Key:
.   -pc_asm_overlap <ovl> - Sets overlap

    Notes:
    By default the ASM preconditioner uses 1 block per processor.  To use
    multiple blocks per perocessor, see PCASMSetTotalSubdomains() and
    PCASMSetLocalSubdomains() (and the option -pc_asm_blocks <blks>).

    The overlap defaults to 1, so if one desires that no additional
    overlap be computed beyond what may have been set with a call to
    PCASMSetTotalSubdomains() or PCASMSetLocalSubdomains(), then ovl
    must be set to be 0.  In particular, if one does not explicitly set
    the subdomains an application code, then all overlap would be computed
    internally by PETSc, and using an overlap of 0 would result in an ASM 
    variant that is equivalent to the block Jacobi preconditioner.  

    Note that one can define initial index sets with any overlap via
    PCASMSetTotalSubdomains() or PCASMSetLocalSubdomains(); the routine
    PCASMSetOverlap() merely allows PETSc to extend that overlap further
    if desired.

    Level: intermediate

.keywords: PC, ASM, set, overlap

.seealso: PCASMSetTotalSubdomains(), PCASMSetLocalSubdomains(), PCASMGetSubKSP(),
          PCASMCreateSubdomains2D(), PCASMGetLocalSubdomains()
@*/
PetscErrorCode  PCASMSetOverlap(PC pc,PetscInt ovl)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_CLASSID,1);
  PetscValidLogicalCollectiveInt(pc,ovl,2);
  ierr = PetscTryMethod(pc,"PCASMSetOverlap_C",(PC,PetscInt),(pc,ovl));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PCASMSetType"
/*@
    PCASMSetType - Sets the type of restriction and interpolation used
    for local problems in the additive Schwarz method.

    Logically Collective on PC

    Input Parameters:
+   pc  - the preconditioner context
-   type - variant of ASM, one of
.vb
      PC_ASM_BASIC       - full interpolation and restriction
      PC_ASM_RESTRICT    - full restriction, local processor interpolation
      PC_ASM_INTERPOLATE - full interpolation, local processor restriction
      PC_ASM_NONE        - local processor restriction and interpolation
.ve

    Options Database Key:
.   -pc_asm_type [basic,restrict,interpolate,none] - Sets ASM type

    Level: intermediate

.keywords: PC, ASM, set, type

.seealso: PCASMSetTotalSubdomains(), PCASMSetTotalSubdomains(), PCASMGetSubKSP(),
          PCASMCreateSubdomains2D()
@*/
PetscErrorCode  PCASMSetType(PC pc,PCASMType type)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_CLASSID,1);
  PetscValidLogicalCollectiveEnum(pc,type,2);
  ierr = PetscTryMethod(pc,"PCASMSetType_C",(PC,PCASMType),(pc,type));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PCASMSetSortIndices"
/*@
    PCASMSetSortIndices - Determines whether subdomain indices are sorted.

    Logically Collective on PC

    Input Parameters:
+   pc  - the preconditioner context
-   doSort - sort the subdomain indices

    Level: intermediate

.keywords: PC, ASM, set, type

.seealso: PCASMSetLocalSubdomains(), PCASMSetTotalSubdomains(), PCASMGetSubKSP(),
          PCASMCreateSubdomains2D()
@*/
PetscErrorCode  PCASMSetSortIndices(PC pc,PetscBool  doSort)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_CLASSID,1);
  PetscValidLogicalCollectiveBool(pc,doSort,2);
  ierr = PetscTryMethod(pc,"PCASMSetSortIndices_C",(PC,PetscBool),(pc,doSort));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PCASMGetSubKSP"
/*@C
   PCASMGetSubKSP - Gets the local KSP contexts for all blocks on
   this processor.
   
   Collective on PC iff first_local is requested

   Input Parameter:
.  pc - the preconditioner context

   Output Parameters:
+  n_local - the number of blocks on this processor or PETSC_NULL
.  first_local - the global number of the first block on this processor or PETSC_NULL,
                 all processors must request or all must pass PETSC_NULL
-  ksp - the array of KSP contexts

   Note:  
   After PCASMGetSubKSP() the array of KSPes is not to be freed

   Currently for some matrix implementations only 1 block per processor 
   is supported.
   
   You must call KSPSetUp() before calling PCASMGetSubKSP().

   Level: advanced

.keywords: PC, ASM, additive Schwarz, get, sub, KSP, context

.seealso: PCASMSetTotalSubdomains(), PCASMSetTotalSubdomains(), PCASMSetOverlap(),
          PCASMCreateSubdomains2D(),
@*/
PetscErrorCode  PCASMGetSubKSP(PC pc,PetscInt *n_local,PetscInt *first_local,KSP *ksp[])
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_CLASSID,1);
  ierr = PetscUseMethod(pc,"PCASMGetSubKSP_C",(PC,PetscInt*,PetscInt*,KSP **),(pc,n_local,first_local,ksp));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* -------------------------------------------------------------------------------------*/
/*MC
   PCASM - Use the (restricted) additive Schwarz method, each block is (approximately) solved with 
           its own KSP object.

   Options Database Keys:
+  -pc_asm_truelocal - Activates PCASMSetUseTrueLocal()
.  -pc_asm_blocks <blks> - Sets total blocks
.  -pc_asm_overlap <ovl> - Sets overlap
-  -pc_asm_type [basic,restrict,interpolate,none] - Sets ASM type

     IMPORTANT: If you run with, for example, 3 blocks on 1 processor or 3 blocks on 3 processors you 
      will get a different convergence rate due to the default option of -pc_asm_type restrict. Use
      -pc_asm_type basic to use the standard ASM. 

   Notes: Each processor can have one or more blocks, but a block cannot be shared by more
     than one processor. Defaults to one block per processor.

     To set options on the solvers for each block append -sub_ to all the KSP, and PC
        options database keys. For example, -sub_pc_type ilu -sub_pc_factor_levels 1 -sub_ksp_type preonly
        
     To set the options on the solvers separate for each block call PCASMGetSubKSP()
         and set the options directly on the resulting KSP object (you can access its PC
         with KSPGetPC())


   Level: beginner

   Concepts: additive Schwarz method

    References:
    An additive variant of the Schwarz alternating method for the case of many subregions
    M Dryja, OB Widlund - Courant Institute, New York University Technical report

    Domain Decompositions: Parallel Multilevel Methods for Elliptic Partial Differential Equations, 
    Barry Smith, Petter Bjorstad, and William Gropp, Cambridge University Press, ISBN 0-521-49589-X.

.seealso:  PCCreate(), PCSetType(), PCType (for list of available types), PC,
           PCBJACOBI, PCASMSetUseTrueLocal(), PCASMGetSubKSP(), PCASMSetLocalSubdomains(),
           PCASMSetTotalSubdomains(), PCSetModifySubmatrices(), PCASMSetOverlap(), PCASMSetType()

M*/

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "PCCreate_ASM"
PetscErrorCode  PCCreate_ASM(PC pc)
{
  PetscErrorCode ierr;
  PC_ASM         *osm;

  PetscFunctionBegin;
  ierr = PetscNewLog(pc,PC_ASM,&osm);CHKERRQ(ierr);
  osm->n                 = PETSC_DECIDE;
  osm->n_local           = 0;
  osm->n_local_true      = 0;
  osm->overlap           = 1;
  osm->ksp               = 0;
  osm->restriction       = 0;
  osm->localization      = 0;
  osm->prolongation      = 0;
  osm->x                 = 0;
  osm->y                 = 0;
  osm->y_local           = 0;
  osm->is                = 0;
  osm->is_local          = 0;
  osm->mat               = 0;
  osm->pmat              = 0;
  osm->type              = PC_ASM_RESTRICT;
  osm->same_local_solves = PETSC_TRUE;
  osm->sort_indices      = PETSC_TRUE;

  pc->data                   = (void*)osm;
  pc->ops->apply             = PCApply_ASM;
  pc->ops->applytranspose    = PCApplyTranspose_ASM;
  pc->ops->setup             = PCSetUp_ASM;
  pc->ops->reset             = PCReset_ASM;
  pc->ops->destroy           = PCDestroy_ASM;
  pc->ops->setfromoptions    = PCSetFromOptions_ASM;
  pc->ops->setuponblocks     = PCSetUpOnBlocks_ASM;
  pc->ops->view              = PCView_ASM;
  pc->ops->applyrichardson   = 0;

  ierr = PetscObjectComposeFunctionDynamic((PetscObject)pc,"PCASMSetLocalSubdomains_C","PCASMSetLocalSubdomains_ASM",
                    PCASMSetLocalSubdomains_ASM);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)pc,"PCASMSetTotalSubdomains_C","PCASMSetTotalSubdomains_ASM",
                    PCASMSetTotalSubdomains_ASM);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)pc,"PCASMSetOverlap_C","PCASMSetOverlap_ASM",
                    PCASMSetOverlap_ASM);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)pc,"PCASMSetType_C","PCASMSetType_ASM",
                    PCASMSetType_ASM);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)pc,"PCASMSetSortIndices_C","PCASMSetSortIndices_ASM",
                    PCASMSetSortIndices_ASM);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)pc,"PCASMGetSubKSP_C","PCASMGetSubKSP_ASM",
                    PCASMGetSubKSP_ASM);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
EXTERN_C_END


#undef __FUNCT__
#define __FUNCT__ "PCASMCreateSubdomains"
/*@C
   PCASMCreateSubdomains - Creates the index sets for the overlapping Schwarz
   preconditioner for a any problem on a general grid.

   Collective

   Input Parameters:
+  A - The global matrix operator
-  n - the number of local blocks

   Output Parameters:
.  outis - the array of index sets defining the subdomains

   Level: advanced

   Note: this generates nonoverlapping subdomains; the PCASM will generate the overlap
    from these if you use PCASMSetLocalSubdomains()

    In the Fortran version you must provide the array outis[] already allocated of length n.

.keywords: PC, ASM, additive Schwarz, create, subdomains, unstructured grid

.seealso: PCASMSetLocalSubdomains(), PCASMDestroySubdomains()
@*/
PetscErrorCode  PCASMCreateSubdomains(Mat A, PetscInt n, IS* outis[])
{
  MatPartitioning           mpart;
  const char                *prefix;
  PetscErrorCode            (*f)(Mat,Mat*);
  PetscMPIInt               size;
  PetscInt                  i,j,rstart,rend,bs;
  PetscBool                 isbaij = PETSC_FALSE,foundpart = PETSC_FALSE;
  Mat                       Ad = PETSC_NULL, adj;
  IS                        ispart,isnumb,*is;
  PetscErrorCode            ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(A,MAT_CLASSID,1);
  PetscValidPointer(outis,3);
  if (n < 1) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"number of local blocks must be > 0, n = %D",n);

  /* Get prefix, row distribution, and block size */
  ierr = MatGetOptionsPrefix(A,&prefix);CHKERRQ(ierr);
  ierr = MatGetOwnershipRange(A,&rstart,&rend);CHKERRQ(ierr);
  ierr = MatGetBlockSize(A,&bs);CHKERRQ(ierr);
  if (rstart/bs*bs != rstart || rend/bs*bs != rend) SETERRQ3(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"bad row distribution [%D,%D) for matrix block size %D",rstart,rend,bs);

  /* Get diagonal block from matrix if possible */
  ierr = MPI_Comm_size(((PetscObject)A)->comm,&size);CHKERRQ(ierr);
  ierr = PetscObjectQueryFunction((PetscObject)A,"MatGetDiagonalBlock_C",(void (**)(void))&f);CHKERRQ(ierr);
  if (f) {
    ierr = MatGetDiagonalBlock(A,&Ad);CHKERRQ(ierr);
  } else if (size == 1) {
    Ad = A;
  }
  if (Ad) {
    ierr = PetscTypeCompare((PetscObject)Ad,MATSEQBAIJ,&isbaij);CHKERRQ(ierr);
    if (!isbaij) {ierr = PetscTypeCompare((PetscObject)Ad,MATSEQSBAIJ,&isbaij);CHKERRQ(ierr);}
  }
  if (Ad && n > 1) {
    PetscBool  match,done;
    /* Try to setup a good matrix partitioning if available */
    ierr = MatPartitioningCreate(PETSC_COMM_SELF,&mpart);CHKERRQ(ierr);
    ierr = PetscObjectSetOptionsPrefix((PetscObject)mpart,prefix);CHKERRQ(ierr);
    ierr = MatPartitioningSetFromOptions(mpart);CHKERRQ(ierr);
    ierr = PetscTypeCompare((PetscObject)mpart,MATPARTITIONINGCURRENT,&match);CHKERRQ(ierr);
    if (!match) {
      ierr = PetscTypeCompare((PetscObject)mpart,MATPARTITIONINGSQUARE,&match);CHKERRQ(ierr);
    }
    if (!match) { /* assume a "good" partitioner is available */
      PetscInt na,*ia,*ja;
      ierr = MatGetRowIJ(Ad,0,PETSC_TRUE,isbaij,&na,&ia,&ja,&done);CHKERRQ(ierr);
      if (done) {
	/* Build adjacency matrix by hand. Unfortunately a call to
	   MatConvert(Ad,MATMPIADJ,MAT_INITIAL_MATRIX,&adj) will
	   remove the block-aij structure and we cannot expect
	   MatPartitioning to split vertices as we need */
	PetscInt i,j,*row,len,nnz,cnt,*iia=0,*jja=0;
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
	ierr = PetscMalloc((na+1)*sizeof(PetscInt),&iia);CHKERRQ(ierr);
	ierr = PetscMalloc((nnz)*sizeof(PetscInt),&jja);CHKERRQ(ierr);
	nnz    = 0;
	iia[0] = 0;
	for (i=0; i<na; i++) { /* fill adjacency */
	  cnt = 0;
	  len = ia[i+1] - ia[i];
	  row = ja + ia[i];
	  for (j=0; j<len; j++) {
	    if (row[j] != i) { /* if not diagonal */
	      jja[nnz+cnt++] = row[j];
	    }
	  }
	  nnz += cnt;
	  iia[i+1] = nnz;
	}
	/* Partitioning of the adjacency matrix */
	ierr = MatCreateMPIAdj(PETSC_COMM_SELF,na,na,iia,jja,PETSC_NULL,&adj);CHKERRQ(ierr);
	ierr = MatPartitioningSetAdjacency(mpart,adj);CHKERRQ(ierr);
	ierr = MatPartitioningSetNParts(mpart,n);CHKERRQ(ierr);
	ierr = MatPartitioningApply(mpart,&ispart);CHKERRQ(ierr);
	ierr = ISPartitioningToNumbering(ispart,&isnumb);CHKERRQ(ierr);
	ierr = MatDestroy(&adj);CHKERRQ(ierr);
	foundpart = PETSC_TRUE;
      }
      ierr = MatRestoreRowIJ(Ad,0,PETSC_TRUE,isbaij,&na,&ia,&ja,&done);CHKERRQ(ierr);
    }
    ierr = MatPartitioningDestroy(&mpart);CHKERRQ(ierr);
  }

  ierr = PetscMalloc(n*sizeof(IS),&is);CHKERRQ(ierr);
  *outis = is;

  if (!foundpart) {

    /* Partitioning by contiguous chunks of rows */

    PetscInt mbs   = (rend-rstart)/bs;
    PetscInt start = rstart;
    for (i=0; i<n; i++) {
      PetscInt count = (mbs/n + ((mbs % n) > i)) * bs;
      ierr   = ISCreateStride(PETSC_COMM_SELF,count,start,1,&is[i]);CHKERRQ(ierr);
      start += count;
    }

  } else {

    /* Partitioning by adjacency of diagonal block  */

    const PetscInt *numbering;
    PetscInt       *count,nidx,*indices,*newidx,start=0;
    /* Get node count in each partition */
    ierr = PetscMalloc(n*sizeof(PetscInt),&count);CHKERRQ(ierr);
    ierr = ISPartitioningCount(ispart,n,count);CHKERRQ(ierr);
    if (isbaij && bs > 1) { /* adjust for the block-aij case */
      for (i=0; i<n; i++) count[i] *= bs;
    }
    /* Build indices from node numbering */
    ierr = ISGetLocalSize(isnumb,&nidx);CHKERRQ(ierr);
    ierr = PetscMalloc(nidx*sizeof(PetscInt),&indices);CHKERRQ(ierr);
    for (i=0; i<nidx; i++) indices[i] = i; /* needs to be initialized */
    ierr = ISGetIndices(isnumb,&numbering);CHKERRQ(ierr);
    ierr = PetscSortIntWithPermutation(nidx,numbering,indices);CHKERRQ(ierr);
    ierr = ISRestoreIndices(isnumb,&numbering);CHKERRQ(ierr);
    if (isbaij && bs > 1) { /* adjust for the block-aij case */
      ierr = PetscMalloc(nidx*bs*sizeof(PetscInt),&newidx);CHKERRQ(ierr);
      for (i=0; i<nidx; i++)
        for (j=0; j<bs; j++)
          newidx[i*bs+j] = indices[i]*bs + j;
      ierr = PetscFree(indices);CHKERRQ(ierr);
      nidx   *= bs;
      indices = newidx;
    }
    /* Shift to get global indices */
    for (i=0; i<nidx; i++) indices[i] += rstart;

    /* Build the index sets for each block */
    for (i=0; i<n; i++) {
      ierr   = ISCreateGeneral(PETSC_COMM_SELF,count[i],&indices[start],PETSC_COPY_VALUES,&is[i]);CHKERRQ(ierr);
      ierr   = ISSort(is[i]);CHKERRQ(ierr);
      start += count[i];
    }

    ierr = PetscFree(count);
    ierr = PetscFree(indices);
    ierr = ISDestroy(&isnumb);CHKERRQ(ierr);
    ierr = ISDestroy(&ispart);CHKERRQ(ierr);

  }

  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PCASMDestroySubdomains"
/*@C
   PCASMDestroySubdomains - Destroys the index sets created with
   PCASMCreateSubdomains(). Should be called after setting subdomains
   with PCASMSetLocalSubdomains().

   Collective

   Input Parameters:
+  n - the number of index sets
.  is - the array of index sets
-  is_local - the array of local index sets, can be PETSC_NULL

   Level: advanced

.keywords: PC, ASM, additive Schwarz, create, subdomains, unstructured grid

.seealso: PCASMCreateSubdomains(), PCASMSetLocalSubdomains()
@*/
PetscErrorCode  PCASMDestroySubdomains(PetscInt n, IS is[], IS is_local[])
{
  PetscInt       i;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  if (n <= 0) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"n must be > 0: n = %D",n);
  PetscValidPointer(is,2);
  for (i=0; i<n; i++) { ierr = ISDestroy(&is[i]);CHKERRQ(ierr); }
  ierr = PetscFree(is);CHKERRQ(ierr);
  if (is_local) {
    PetscValidPointer(is_local,3);
    for (i=0; i<n; i++) { ierr = ISDestroy(&is_local[i]);CHKERRQ(ierr); }
    ierr = PetscFree(is_local);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PCASMCreateSubdomains2D"
/*@
   PCASMCreateSubdomains2D - Creates the index sets for the overlapping Schwarz 
   preconditioner for a two-dimensional problem on a regular grid.

   Not Collective

   Input Parameters:
+  m, n - the number of mesh points in the x and y directions
.  M, N - the number of subdomains in the x and y directions
.  dof - degrees of freedom per node
-  overlap - overlap in mesh lines

   Output Parameters:
+  Nsub - the number of subdomains created
.  is - array of index sets defining overlapping (if overlap > 0) subdomains
-  is_local - array of index sets defining non-overlapping subdomains

   Note:
   Presently PCAMSCreateSubdomains2d() is valid only for sequential
   preconditioners.  More general related routines are
   PCASMSetTotalSubdomains() and PCASMSetLocalSubdomains().

   Level: advanced

.keywords: PC, ASM, additive Schwarz, create, subdomains, 2D, regular grid

.seealso: PCASMSetTotalSubdomains(), PCASMSetLocalSubdomains(), PCASMGetSubKSP(),
          PCASMSetOverlap()
@*/
PetscErrorCode  PCASMCreateSubdomains2D(PetscInt m,PetscInt n,PetscInt M,PetscInt N,PetscInt dof,PetscInt overlap,PetscInt *Nsub,IS **is,IS **is_local)
{
  PetscInt       i,j,height,width,ystart,xstart,yleft,yright,xleft,xright,loc_outer;
  PetscErrorCode ierr;
  PetscInt       nidx,*idx,loc,ii,jj,count;

  PetscFunctionBegin;
  if (dof != 1) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP," ");

  *Nsub = N*M;
  ierr = PetscMalloc((*Nsub)*sizeof(IS*),is);CHKERRQ(ierr);
  ierr = PetscMalloc((*Nsub)*sizeof(IS*),is_local);CHKERRQ(ierr);
  ystart = 0;
  loc_outer = 0;
  for (i=0; i<N; i++) {
    height = n/N + ((n % N) > i); /* height of subdomain */
    if (height < 2) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Too many N subdomains for mesh dimension n");
    yleft  = ystart - overlap; if (yleft < 0) yleft = 0;
    yright = ystart + height + overlap; if (yright > n) yright = n;
    xstart = 0;
    for (j=0; j<M; j++) {
      width = m/M + ((m % M) > j); /* width of subdomain */
      if (width < 2) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Too many M subdomains for mesh dimension m");
      xleft  = xstart - overlap; if (xleft < 0) xleft = 0;
      xright = xstart + width + overlap; if (xright > m) xright = m;
      nidx   = (xright - xleft)*(yright - yleft);
      ierr = PetscMalloc(nidx*sizeof(PetscInt),&idx);CHKERRQ(ierr);
      loc    = 0;
      for (ii=yleft; ii<yright; ii++) {
        count = m*ii + xleft;
        for (jj=xleft; jj<xright; jj++) {
          idx[loc++] = count++;
        }
      }
      ierr = ISCreateGeneral(PETSC_COMM_SELF,nidx,idx,PETSC_COPY_VALUES,(*is)+loc_outer);CHKERRQ(ierr);
      if (overlap == 0) {
        ierr = PetscObjectReference((PetscObject)(*is)[loc_outer]);CHKERRQ(ierr);
        (*is_local)[loc_outer] = (*is)[loc_outer];
      } else {
        for (loc=0,ii=ystart; ii<ystart+height; ii++) {
          for (jj=xstart; jj<xstart+width; jj++) {
            idx[loc++] = m*ii + jj;
          }
        }
        ierr = ISCreateGeneral(PETSC_COMM_SELF,loc,idx,PETSC_COPY_VALUES,*is_local+loc_outer);CHKERRQ(ierr);
      }
      ierr = PetscFree(idx);CHKERRQ(ierr);
      xstart += width;
      loc_outer++;
    }
    ystart += height;
  }
  for (i=0; i<*Nsub; i++) { ierr = ISSort((*is)[i]);CHKERRQ(ierr); }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PCASMGetLocalSubdomains"
/*@C
    PCASMGetLocalSubdomains - Gets the local subdomains (for this processor
    only) for the additive Schwarz preconditioner. 

    Not Collective

    Input Parameter:
.   pc - the preconditioner context

    Output Parameters:
+   n - the number of subdomains for this processor (default value = 1)
.   is - the index sets that define the subdomains for this processor
-   is_local - the index sets that define the local part of the subdomains for this processor (can be PETSC_NULL)
         

    Notes:
    The IS numbering is in the parallel, global numbering of the vector.

    Level: advanced

.keywords: PC, ASM, set, local, subdomains, additive Schwarz

.seealso: PCASMSetTotalSubdomains(), PCASMSetOverlap(), PCASMGetSubKSP(),
          PCASMCreateSubdomains2D(), PCASMSetLocalSubdomains(), PCASMGetLocalSubmatrices()
@*/
PetscErrorCode  PCASMGetLocalSubdomains(PC pc,PetscInt *n,IS *is[],IS *is_local[])
{
  PC_ASM         *osm;
  PetscErrorCode ierr;
  PetscBool      match;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_CLASSID,1);
  PetscValidIntPointer(n,2);
  if (is) PetscValidPointer(is,3);
  ierr = PetscTypeCompare((PetscObject)pc,PCASM,&match);CHKERRQ(ierr);
  if (!match) {
    if (n)  *n  = 0;
    if (is) *is = PETSC_NULL;
  } else {
    osm = (PC_ASM*)pc->data;
    if (n)  *n  = osm->n_local_true;
    if (is) *is = osm->is;
    if (is_local) *is_local = osm->is_local;
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PCASMGetLocalSubmatrices"
/*@C
    PCASMGetLocalSubmatrices - Gets the local submatrices (for this processor
    only) for the additive Schwarz preconditioner. 

    Not Collective

    Input Parameter:
.   pc - the preconditioner context

    Output Parameters:
+   n - the number of matrices for this processor (default value = 1)
-   mat - the matrices
         

    Level: advanced

    Notes: Call after PCSetUp() (or KSPSetUp()) but before PCApply() (or KSPApply()) and before PCSetUpOnBlocks())

           Usually one would use PCSetModifySubmatrices() to change the submatrices in building the preconditioner.

.keywords: PC, ASM, set, local, subdomains, additive Schwarz, block Jacobi

.seealso: PCASMSetTotalSubdomains(), PCASMSetOverlap(), PCASMGetSubKSP(),
          PCASMCreateSubdomains2D(), PCASMSetLocalSubdomains(), PCASMGetLocalSubdomains(), PCSetModifySubmatrices()
@*/
PetscErrorCode  PCASMGetLocalSubmatrices(PC pc,PetscInt *n,Mat *mat[])
{
  PC_ASM         *osm;
  PetscErrorCode ierr;
  PetscBool      match;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_CLASSID,1);
  PetscValidIntPointer(n,2);
  if (mat) PetscValidPointer(mat,3);
  if (!pc->setupcalled) SETERRQ(((PetscObject)pc)->comm,PETSC_ERR_ARG_WRONGSTATE,"Must call after KSPSetUP() or PCSetUp().");
  ierr = PetscTypeCompare((PetscObject)pc,PCASM,&match);CHKERRQ(ierr);
  if (!match) {
    if (n)   *n   = 0;
    if (mat) *mat = PETSC_NULL;
  } else {
    osm = (PC_ASM*)pc->data;
    if (n)   *n   = osm->n_local_true;
    if (mat) *mat = osm->pmat;
  }
  PetscFunctionReturn(0);
}
