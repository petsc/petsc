/*
  This file defines an additive Schwarz preconditioner for any Mat implementation.

  Note that each processor may have any number of subdomains. But in order to
  deal easily with the VecScatter(), we treat each processor as if it has the
  same number of subdomains.

       n - total number of true subdomains on all processors
       n_local_true - actual number of subdomains on this processor
       n_local = maximum over all processors of n_local_true
*/

#include <petsc/private/pcasmimpl.h> /*I "petscpc.h" I*/

static PetscErrorCode PCView_ASM(PC pc,PetscViewer viewer)
{
  PC_ASM            *osm = (PC_ASM*)pc->data;
  PetscMPIInt       rank;
  PetscInt          i,bsz;
  PetscBool         iascii,isstring;
  PetscViewer       sviewer;
  PetscViewerFormat format;
  const char        *prefix;

  PetscFunctionBegin;
  PetscCall(PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&iascii));
  PetscCall(PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERSTRING,&isstring));
  if (iascii) {
    char overlaps[256] = "user-defined overlap",blocks[256] = "total subdomain blocks not yet set";
    if (osm->overlap >= 0) PetscCall(PetscSNPrintf(overlaps,sizeof(overlaps),"amount of overlap = %" PetscInt_FMT,osm->overlap));
    if (osm->n > 0) PetscCall(PetscSNPrintf(blocks,sizeof(blocks),"total subdomain blocks = %" PetscInt_FMT,osm->n));
    PetscCall(PetscViewerASCIIPrintf(viewer,"  %s, %s\n",blocks,overlaps));
    PetscCall(PetscViewerASCIIPrintf(viewer,"  restriction/interpolation type - %s\n",PCASMTypes[osm->type]));
    if (osm->dm_subdomains) PetscCall(PetscViewerASCIIPrintf(viewer,"  Additive Schwarz: using DM to define subdomains\n"));
    if (osm->loctype != PC_COMPOSITE_ADDITIVE) PetscCall(PetscViewerASCIIPrintf(viewer,"  Additive Schwarz: local solve composition type - %s\n",PCCompositeTypes[osm->loctype]));
    PetscCallMPI(MPI_Comm_rank(PetscObjectComm((PetscObject)pc),&rank));
    PetscCall(PetscViewerGetFormat(viewer,&format));
    if (format != PETSC_VIEWER_ASCII_INFO_DETAIL) {
      if (osm->ksp) {
        PetscCall(PetscViewerASCIIPrintf(viewer,"  Local solver information for first block is in the following KSP and PC objects on rank 0:\n"));
        PetscCall(PCGetOptionsPrefix(pc,&prefix));
        PetscCall(PetscViewerASCIIPrintf(viewer,"  Use -%sksp_view ::ascii_info_detail to display information for all blocks\n",prefix?prefix:""));
        PetscCall(PetscViewerGetSubViewer(viewer,PETSC_COMM_SELF,&sviewer));
        if (rank == 0) {
          PetscCall(PetscViewerASCIIPushTab(viewer));
          PetscCall(KSPView(osm->ksp[0],sviewer));
          PetscCall(PetscViewerASCIIPopTab(viewer));
        }
        PetscCall(PetscViewerRestoreSubViewer(viewer,PETSC_COMM_SELF,&sviewer));
      }
    } else {
      PetscCall(PetscViewerASCIIPushSynchronized(viewer));
      PetscCall(PetscViewerASCIISynchronizedPrintf(viewer,"  [%d] number of local blocks = %" PetscInt_FMT "\n",(int)rank,osm->n_local_true));
      PetscCall(PetscViewerFlush(viewer));
      PetscCall(PetscViewerASCIIPrintf(viewer,"  Local solver information for each block is in the following KSP and PC objects:\n"));
      PetscCall(PetscViewerASCIIPushTab(viewer));
      PetscCall(PetscViewerASCIIPrintf(viewer,"- - - - - - - - - - - - - - - - - -\n"));
      PetscCall(PetscViewerGetSubViewer(viewer,PETSC_COMM_SELF,&sviewer));
      for (i=0; i<osm->n_local_true; i++) {
        PetscCall(ISGetLocalSize(osm->is[i],&bsz));
        PetscCall(PetscViewerASCIISynchronizedPrintf(sviewer,"[%d] local block number %" PetscInt_FMT ", size = %" PetscInt_FMT "\n",(int)rank,i,bsz));
        PetscCall(KSPView(osm->ksp[i],sviewer));
        PetscCall(PetscViewerASCIISynchronizedPrintf(sviewer,"- - - - - - - - - - - - - - - - - -\n"));
      }
      PetscCall(PetscViewerRestoreSubViewer(viewer,PETSC_COMM_SELF,&sviewer));
      PetscCall(PetscViewerASCIIPopTab(viewer));
      PetscCall(PetscViewerFlush(viewer));
      PetscCall(PetscViewerASCIIPopSynchronized(viewer));
    }
  } else if (isstring) {
    PetscCall(PetscViewerStringSPrintf(viewer," blocks=%" PetscInt_FMT ", overlap=%" PetscInt_FMT ", type=%s",osm->n,osm->overlap,PCASMTypes[osm->type]));
    PetscCall(PetscViewerGetSubViewer(viewer,PETSC_COMM_SELF,&sviewer));
    if (osm->ksp) PetscCall(KSPView(osm->ksp[0],sviewer));
    PetscCall(PetscViewerRestoreSubViewer(viewer,PETSC_COMM_SELF,&sviewer));
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode PCASMPrintSubdomains(PC pc)
{
  PC_ASM         *osm = (PC_ASM*)pc->data;
  const char     *prefix;
  char           fname[PETSC_MAX_PATH_LEN+1];
  PetscViewer    viewer, sviewer;
  char           *s;
  PetscInt       i,j,nidx;
  const PetscInt *idx;
  PetscMPIInt    rank, size;

  PetscFunctionBegin;
  PetscCallMPI(MPI_Comm_size(PetscObjectComm((PetscObject)pc), &size));
  PetscCallMPI(MPI_Comm_rank(PetscObjectComm((PetscObject)pc), &rank));
  PetscCall(PCGetOptionsPrefix(pc,&prefix));
  PetscCall(PetscOptionsGetString(NULL,prefix,"-pc_asm_print_subdomains",fname,sizeof(fname),NULL));
  if (fname[0] == 0) PetscCall(PetscStrcpy(fname,"stdout"));
  PetscCall(PetscViewerASCIIOpen(PetscObjectComm((PetscObject)pc),fname,&viewer));
  for (i=0; i<osm->n_local; i++) {
    if (i < osm->n_local_true) {
      PetscCall(ISGetLocalSize(osm->is[i],&nidx));
      PetscCall(ISGetIndices(osm->is[i],&idx));
      /* Print to a string viewer; no more than 15 characters per index plus 512 char for the header.*/
#define len  16*(nidx+1)+512
      PetscCall(PetscMalloc1(len,&s));
      PetscCall(PetscViewerStringOpen(PETSC_COMM_SELF, s, len, &sviewer));
#undef len
      PetscCall(PetscViewerStringSPrintf(sviewer, "[%d:%d] Subdomain %" PetscInt_FMT " with overlap:\n", rank, size, i));
      for (j=0; j<nidx; j++) {
        PetscCall(PetscViewerStringSPrintf(sviewer,"%" PetscInt_FMT " ",idx[j]));
      }
      PetscCall(ISRestoreIndices(osm->is[i],&idx));
      PetscCall(PetscViewerStringSPrintf(sviewer,"\n"));
      PetscCall(PetscViewerDestroy(&sviewer));
      PetscCall(PetscViewerASCIIPushSynchronized(viewer));
      PetscCall(PetscViewerASCIISynchronizedPrintf(viewer, "%s", s));
      PetscCall(PetscViewerFlush(viewer));
      PetscCall(PetscViewerASCIIPopSynchronized(viewer));
      PetscCall(PetscFree(s));
      if (osm->is_local) {
        /* Print to a string viewer; no more than 15 characters per index plus 512 char for the header.*/
#define len  16*(nidx+1)+512
        PetscCall(PetscMalloc1(len, &s));
        PetscCall(PetscViewerStringOpen(PETSC_COMM_SELF, s, len, &sviewer));
#undef len
        PetscCall(PetscViewerStringSPrintf(sviewer, "[%d:%d] Subdomain %" PetscInt_FMT " without overlap:\n", rank, size, i));
        PetscCall(ISGetLocalSize(osm->is_local[i],&nidx));
        PetscCall(ISGetIndices(osm->is_local[i],&idx));
        for (j=0; j<nidx; j++) {
          PetscCall(PetscViewerStringSPrintf(sviewer,"%" PetscInt_FMT " ",idx[j]));
        }
        PetscCall(ISRestoreIndices(osm->is_local[i],&idx));
        PetscCall(PetscViewerStringSPrintf(sviewer,"\n"));
        PetscCall(PetscViewerDestroy(&sviewer));
        PetscCall(PetscViewerASCIIPushSynchronized(viewer));
        PetscCall(PetscViewerASCIISynchronizedPrintf(viewer, "%s", s));
        PetscCall(PetscViewerFlush(viewer));
        PetscCall(PetscViewerASCIIPopSynchronized(viewer));
        PetscCall(PetscFree(s));
      }
    } else {
      /* Participate in collective viewer calls. */
      PetscCall(PetscViewerASCIIPushSynchronized(viewer));
      PetscCall(PetscViewerFlush(viewer));
      PetscCall(PetscViewerASCIIPopSynchronized(viewer));
      /* Assume either all ranks have is_local or none do. */
      if (osm->is_local) {
        PetscCall(PetscViewerASCIIPushSynchronized(viewer));
        PetscCall(PetscViewerFlush(viewer));
        PetscCall(PetscViewerASCIIPopSynchronized(viewer));
      }
    }
  }
  PetscCall(PetscViewerFlush(viewer));
  PetscCall(PetscViewerDestroy(&viewer));
  PetscFunctionReturn(0);
}

static PetscErrorCode PCSetUp_ASM(PC pc)
{
  PC_ASM         *osm = (PC_ASM*)pc->data;
  PetscBool      flg;
  PetscInt       i,m,m_local;
  MatReuse       scall = MAT_REUSE_MATRIX;
  IS             isl;
  KSP            ksp;
  PC             subpc;
  const char     *prefix,*pprefix;
  Vec            vec;
  DM             *domain_dm = NULL;

  PetscFunctionBegin;
  if (!pc->setupcalled) {
    PetscInt m;

    /* Note: if subdomains have been set either via PCASMSetTotalSubdomains() or via PCASMSetLocalSubdomains(), osm->n_local_true will not be PETSC_DECIDE */
    if (osm->n_local_true == PETSC_DECIDE) {
      /* no subdomains given */
      /* try pc->dm first, if allowed */
      if (osm->dm_subdomains && pc->dm) {
        PetscInt  num_domains, d;
        char      **domain_names;
        IS        *inner_domain_is, *outer_domain_is;
        PetscCall(DMCreateDomainDecomposition(pc->dm, &num_domains, &domain_names, &inner_domain_is, &outer_domain_is, &domain_dm));
        osm->overlap = -1; /* We do not want to increase the overlap of the IS.
                              A future improvement of this code might allow one to use
                              DM-defined subdomains and also increase the overlap,
                              but that is not currently supported */
        if (num_domains) {
          PetscCall(PCASMSetLocalSubdomains(pc, num_domains, outer_domain_is, inner_domain_is));
        }
        for (d = 0; d < num_domains; ++d) {
          if (domain_names)    PetscCall(PetscFree(domain_names[d]));
          if (inner_domain_is) PetscCall(ISDestroy(&inner_domain_is[d]));
          if (outer_domain_is) PetscCall(ISDestroy(&outer_domain_is[d]));
        }
        PetscCall(PetscFree(domain_names));
        PetscCall(PetscFree(inner_domain_is));
        PetscCall(PetscFree(outer_domain_is));
      }
      if (osm->n_local_true == PETSC_DECIDE) {
        /* still no subdomains; use one subdomain per processor */
        osm->n_local_true = 1;
      }
    }
    { /* determine the global and max number of subdomains */
      struct {PetscInt max,sum;} inwork,outwork;
      PetscMPIInt size;

      inwork.max   = osm->n_local_true;
      inwork.sum   = osm->n_local_true;
      PetscCall(MPIU_Allreduce(&inwork,&outwork,1,MPIU_2INT,MPIU_MAXSUM_OP,PetscObjectComm((PetscObject)pc)));
      osm->n_local = outwork.max;
      osm->n       = outwork.sum;

      PetscCallMPI(MPI_Comm_size(PetscObjectComm((PetscObject)pc),&size));
      if (outwork.max == 1 && outwork.sum == size) {
        /* osm->n_local_true = 1 on all processes, set this option may enable use of optimized MatCreateSubMatrices() implementation */
        PetscCall(MatSetOption(pc->pmat,MAT_SUBMAT_SINGLEIS,PETSC_TRUE));
      }
    }
    if (!osm->is) { /* create the index sets */
      PetscCall(PCASMCreateSubdomains(pc->pmat,osm->n_local_true,&osm->is));
    }
    if (osm->n_local_true > 1 && !osm->is_local) {
      PetscCall(PetscMalloc1(osm->n_local_true,&osm->is_local));
      for (i=0; i<osm->n_local_true; i++) {
        if (osm->overlap > 0) { /* With positive overlap, osm->is[i] will be modified */
          PetscCall(ISDuplicate(osm->is[i],&osm->is_local[i]));
          PetscCall(ISCopy(osm->is[i],osm->is_local[i]));
        } else {
          PetscCall(PetscObjectReference((PetscObject)osm->is[i]));
          osm->is_local[i] = osm->is[i];
        }
      }
    }
    PetscCall(PCGetOptionsPrefix(pc,&prefix));
    if (osm->overlap > 0) {
      /* Extend the "overlapping" regions by a number of steps */
      PetscCall(MatIncreaseOverlap(pc->pmat,osm->n_local_true,osm->is,osm->overlap));
    }
    if (osm->sort_indices) {
      for (i=0; i<osm->n_local_true; i++) {
        PetscCall(ISSort(osm->is[i]));
        if (osm->is_local) {
          PetscCall(ISSort(osm->is_local[i]));
        }
      }
    }
    flg = PETSC_FALSE;
    PetscCall(PetscOptionsHasName(NULL,prefix,"-pc_asm_print_subdomains",&flg));
    if (flg) PetscCall(PCASMPrintSubdomains(pc));
    if (!osm->ksp) {
      /* Create the local solvers */
      PetscCall(PetscMalloc1(osm->n_local_true,&osm->ksp));
      if (domain_dm) {
        PetscCall(PetscInfo(pc,"Setting up ASM subproblems using the embedded DM\n"));
      }
      for (i=0; i<osm->n_local_true; i++) {
        PetscCall(KSPCreate(PETSC_COMM_SELF,&ksp));
        PetscCall(KSPSetErrorIfNotConverged(ksp,pc->erroriffailure));
        PetscCall(PetscLogObjectParent((PetscObject)pc,(PetscObject)ksp));
        PetscCall(PetscObjectIncrementTabLevel((PetscObject)ksp,(PetscObject)pc,1));
        PetscCall(KSPSetType(ksp,KSPPREONLY));
        PetscCall(KSPGetPC(ksp,&subpc));
        PetscCall(PCGetOptionsPrefix(pc,&prefix));
        PetscCall(KSPSetOptionsPrefix(ksp,prefix));
        PetscCall(KSPAppendOptionsPrefix(ksp,"sub_"));
        if (domain_dm) {
          PetscCall(KSPSetDM(ksp, domain_dm[i]));
          PetscCall(KSPSetDMActive(ksp, PETSC_FALSE));
          PetscCall(DMDestroy(&domain_dm[i]));
        }
        osm->ksp[i] = ksp;
      }
      if (domain_dm) {
        PetscCall(PetscFree(domain_dm));
      }
    }

    PetscCall(ISConcatenate(PETSC_COMM_SELF, osm->n_local_true, osm->is, &osm->lis));
    PetscCall(ISSortRemoveDups(osm->lis));
    PetscCall(ISGetLocalSize(osm->lis, &m));

    scall = MAT_INITIAL_MATRIX;
  } else {
    /*
       Destroy the blocks from the previous iteration
    */
    if (pc->flag == DIFFERENT_NONZERO_PATTERN) {
      PetscCall(MatDestroyMatrices(osm->n_local_true,&osm->pmat));
      scall = MAT_INITIAL_MATRIX;
    }
  }

  /* Destroy previous submatrices of a different type than pc->pmat since MAT_REUSE_MATRIX won't work in that case */
  if (scall == MAT_REUSE_MATRIX && osm->sub_mat_type) {
    if (osm->n_local_true > 0) {
      PetscCall(MatDestroySubMatrices(osm->n_local_true,&osm->pmat));
    }
    scall = MAT_INITIAL_MATRIX;
  }

  /*
     Extract out the submatrices
  */
  PetscCall(MatCreateSubMatrices(pc->pmat,osm->n_local_true,osm->is,osm->is,scall,&osm->pmat));
  if (scall == MAT_INITIAL_MATRIX) {
    PetscCall(PetscObjectGetOptionsPrefix((PetscObject)pc->pmat,&pprefix));
    for (i=0; i<osm->n_local_true; i++) {
      PetscCall(PetscLogObjectParent((PetscObject)pc,(PetscObject)osm->pmat[i]));
      PetscCall(PetscObjectSetOptionsPrefix((PetscObject)osm->pmat[i],pprefix));
    }
  }

  /* Convert the types of the submatrices (if needbe) */
  if (osm->sub_mat_type) {
    for (i=0; i<osm->n_local_true; i++) {
      PetscCall(MatConvert(osm->pmat[i],osm->sub_mat_type,MAT_INPLACE_MATRIX,&(osm->pmat[i])));
    }
  }

  if (!pc->setupcalled) {
    VecType vtype;

    /* Create the local work vectors (from the local matrices) and scatter contexts */
    PetscCall(MatCreateVecs(pc->pmat,&vec,NULL));

    PetscCheck(!osm->is_local || (osm->type != PC_ASM_INTERPOLATE && osm->type != PC_ASM_NONE),PetscObjectComm((PetscObject)pc),PETSC_ERR_SUP,"Cannot use interpolate or none PCASMType if is_local was provided to PCASMSetLocalSubdomains()");
    if (osm->is_local && osm->type == PC_ASM_RESTRICT && osm->loctype == PC_COMPOSITE_ADDITIVE) {
      PetscCall(PetscMalloc1(osm->n_local_true,&osm->lprolongation));
    }
    PetscCall(PetscMalloc1(osm->n_local_true,&osm->lrestriction));
    PetscCall(PetscMalloc1(osm->n_local_true,&osm->x));
    PetscCall(PetscMalloc1(osm->n_local_true,&osm->y));

    PetscCall(ISGetLocalSize(osm->lis,&m));
    PetscCall(ISCreateStride(PETSC_COMM_SELF,m,0,1,&isl));
    PetscCall(MatGetVecType(osm->pmat[0],&vtype));
    PetscCall(VecCreate(PETSC_COMM_SELF,&osm->lx));
    PetscCall(VecSetSizes(osm->lx,m,m));
    PetscCall(VecSetType(osm->lx,vtype));
    PetscCall(VecDuplicate(osm->lx, &osm->ly));
    PetscCall(VecScatterCreate(vec,osm->lis,osm->lx,isl,&osm->restriction));
    PetscCall(ISDestroy(&isl));

    for (i=0; i<osm->n_local_true; ++i) {
      ISLocalToGlobalMapping ltog;
      IS                     isll;
      const PetscInt         *idx_is;
      PetscInt               *idx_lis,nout;

      PetscCall(ISGetLocalSize(osm->is[i],&m));
      PetscCall(MatCreateVecs(osm->pmat[i],&osm->x[i],NULL));
      PetscCall(VecDuplicate(osm->x[i],&osm->y[i]));

      /* generate a scatter from ly to y[i] picking all the overlapping is[i] entries */
      PetscCall(ISLocalToGlobalMappingCreateIS(osm->lis,&ltog));
      PetscCall(ISGetLocalSize(osm->is[i],&m));
      PetscCall(ISGetIndices(osm->is[i], &idx_is));
      PetscCall(PetscMalloc1(m,&idx_lis));
      PetscCall(ISGlobalToLocalMappingApply(ltog,IS_GTOLM_DROP,m,idx_is,&nout,idx_lis));
      PetscCheck(nout == m,PETSC_COMM_SELF,PETSC_ERR_PLIB,"is not a subset of lis");
      PetscCall(ISRestoreIndices(osm->is[i], &idx_is));
      PetscCall(ISCreateGeneral(PETSC_COMM_SELF,m,idx_lis,PETSC_OWN_POINTER,&isll));
      PetscCall(ISLocalToGlobalMappingDestroy(&ltog));
      PetscCall(ISCreateStride(PETSC_COMM_SELF,m,0,1,&isl));
      PetscCall(VecScatterCreate(osm->ly,isll,osm->y[i],isl,&osm->lrestriction[i]));
      PetscCall(ISDestroy(&isll));
      PetscCall(ISDestroy(&isl));
      if (osm->lprolongation) { /* generate a scatter from y[i] to ly picking only the the non-overlapping is_local[i] entries */
        ISLocalToGlobalMapping ltog;
        IS                     isll,isll_local;
        const PetscInt         *idx_local;
        PetscInt               *idx1, *idx2, nout;

        PetscCall(ISGetLocalSize(osm->is_local[i],&m_local));
        PetscCall(ISGetIndices(osm->is_local[i], &idx_local));

        PetscCall(ISLocalToGlobalMappingCreateIS(osm->is[i],&ltog));
        PetscCall(PetscMalloc1(m_local,&idx1));
        PetscCall(ISGlobalToLocalMappingApply(ltog,IS_GTOLM_DROP,m_local,idx_local,&nout,idx1));
        PetscCall(ISLocalToGlobalMappingDestroy(&ltog));
        PetscCheck(nout == m_local,PETSC_COMM_SELF,PETSC_ERR_PLIB,"is_local not a subset of is");
        PetscCall(ISCreateGeneral(PETSC_COMM_SELF,m_local,idx1,PETSC_OWN_POINTER,&isll));

        PetscCall(ISLocalToGlobalMappingCreateIS(osm->lis,&ltog));
        PetscCall(PetscMalloc1(m_local,&idx2));
        PetscCall(ISGlobalToLocalMappingApply(ltog,IS_GTOLM_DROP,m_local,idx_local,&nout,idx2));
        PetscCall(ISLocalToGlobalMappingDestroy(&ltog));
        PetscCheck(nout == m_local,PETSC_COMM_SELF,PETSC_ERR_PLIB,"is_local not a subset of lis");
        PetscCall(ISCreateGeneral(PETSC_COMM_SELF,m_local,idx2,PETSC_OWN_POINTER,&isll_local));

        PetscCall(ISRestoreIndices(osm->is_local[i], &idx_local));
        PetscCall(VecScatterCreate(osm->y[i],isll,osm->ly,isll_local,&osm->lprolongation[i]));

        PetscCall(ISDestroy(&isll));
        PetscCall(ISDestroy(&isll_local));
      }
    }
    PetscCall(VecDestroy(&vec));
  }

  if (osm->loctype == PC_COMPOSITE_MULTIPLICATIVE) {
    IS      *cis;
    PetscInt c;

    PetscCall(PetscMalloc1(osm->n_local_true, &cis));
    for (c = 0; c < osm->n_local_true; ++c) cis[c] = osm->lis;
    PetscCall(MatCreateSubMatrices(pc->pmat, osm->n_local_true, osm->is, cis, scall, &osm->lmats));
    PetscCall(PetscFree(cis));
  }

  /* Return control to the user so that the submatrices can be modified (e.g., to apply
     different boundary conditions for the submatrices than for the global problem) */
  PetscCall(PCModifySubMatrices(pc,osm->n_local_true,osm->is,osm->is,osm->pmat,pc->modifysubmatricesP));

  /*
     Loop over subdomains putting them into local ksp
  */
  PetscCall(KSPGetOptionsPrefix(osm->ksp[0],&prefix));
  for (i=0; i<osm->n_local_true; i++) {
    PetscCall(KSPSetOperators(osm->ksp[i],osm->pmat[i],osm->pmat[i]));
    PetscCall(MatSetOptionsPrefix(osm->pmat[i],prefix));
    if (!pc->setupcalled) {
      PetscCall(KSPSetFromOptions(osm->ksp[i]));
    }
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode PCSetUpOnBlocks_ASM(PC pc)
{
  PC_ASM             *osm = (PC_ASM*)pc->data;
  PetscInt           i;
  KSPConvergedReason reason;

  PetscFunctionBegin;
  for (i=0; i<osm->n_local_true; i++) {
    PetscCall(KSPSetUp(osm->ksp[i]));
    PetscCall(KSPGetConvergedReason(osm->ksp[i],&reason));
    if (reason == KSP_DIVERGED_PC_FAILED) {
      pc->failedreason = PC_SUBPC_ERROR;
    }
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode PCApply_ASM(PC pc,Vec x,Vec y)
{
  PC_ASM         *osm = (PC_ASM*)pc->data;
  PetscInt       i,n_local_true = osm->n_local_true;
  ScatterMode    forward = SCATTER_FORWARD,reverse = SCATTER_REVERSE;

  PetscFunctionBegin;
  /*
     support for limiting the restriction or interpolation to only local
     subdomain values (leaving the other values 0).
  */
  if (!(osm->type & PC_ASM_RESTRICT)) {
    forward = SCATTER_FORWARD_LOCAL;
    /* have to zero the work RHS since scatter may leave some slots empty */
    PetscCall(VecSet(osm->lx, 0.0));
  }
  if (!(osm->type & PC_ASM_INTERPOLATE)) {
    reverse = SCATTER_REVERSE_LOCAL;
  }

  if (osm->loctype == PC_COMPOSITE_MULTIPLICATIVE || osm->loctype == PC_COMPOSITE_ADDITIVE) {
    /* zero the global and the local solutions */
    PetscCall(VecSet(y, 0.0));
    PetscCall(VecSet(osm->ly, 0.0));

    /* copy the global RHS to local RHS including the ghost nodes */
    PetscCall(VecScatterBegin(osm->restriction, x, osm->lx, INSERT_VALUES, forward));
    PetscCall(VecScatterEnd(osm->restriction, x, osm->lx, INSERT_VALUES, forward));

    /* restrict local RHS to the overlapping 0-block RHS */
    PetscCall(VecScatterBegin(osm->lrestriction[0], osm->lx, osm->x[0], INSERT_VALUES, forward));
    PetscCall(VecScatterEnd(osm->lrestriction[0], osm->lx, osm->x[0], INSERT_VALUES, forward));

    /* do the local solves */
    for (i = 0; i < n_local_true; ++i) {

      /* solve the overlapping i-block */
      PetscCall(PetscLogEventBegin(PC_ApplyOnBlocks, osm->ksp[i], osm->x[i], osm->y[i],0));
      PetscCall(KSPSolve(osm->ksp[i], osm->x[i], osm->y[i]));
      PetscCall(KSPCheckSolve(osm->ksp[i], pc, osm->y[i]));
      PetscCall(PetscLogEventEnd(PC_ApplyOnBlocks, osm->ksp[i], osm->x[i], osm->y[i], 0));

      if (osm->lprolongation) { /* interpolate the non-overlapping i-block solution to the local solution (only for restrictive additive) */
        PetscCall(VecScatterBegin(osm->lprolongation[i], osm->y[i], osm->ly, ADD_VALUES, forward));
        PetscCall(VecScatterEnd(osm->lprolongation[i], osm->y[i], osm->ly, ADD_VALUES, forward));
      } else { /* interpolate the overlapping i-block solution to the local solution */
        PetscCall(VecScatterBegin(osm->lrestriction[i], osm->y[i], osm->ly, ADD_VALUES, reverse));
        PetscCall(VecScatterEnd(osm->lrestriction[i], osm->y[i], osm->ly, ADD_VALUES, reverse));
      }

      if (i < n_local_true-1) {
        /* restrict local RHS to the overlapping (i+1)-block RHS */
        PetscCall(VecScatterBegin(osm->lrestriction[i+1], osm->lx, osm->x[i+1], INSERT_VALUES, forward));
        PetscCall(VecScatterEnd(osm->lrestriction[i+1], osm->lx, osm->x[i+1], INSERT_VALUES, forward));

        if (osm->loctype == PC_COMPOSITE_MULTIPLICATIVE) {
          /* update the overlapping (i+1)-block RHS using the current local solution */
          PetscCall(MatMult(osm->lmats[i+1], osm->ly, osm->y[i+1]));
          PetscCall(VecAXPBY(osm->x[i+1],-1.,1., osm->y[i+1]));
        }
      }
    }
    /* add the local solution to the global solution including the ghost nodes */
    PetscCall(VecScatterBegin(osm->restriction, osm->ly, y, ADD_VALUES, reverse));
    PetscCall(VecScatterEnd(osm->restriction, osm->ly, y, ADD_VALUES, reverse));
  } else SETERRQ(PetscObjectComm((PetscObject)pc), PETSC_ERR_ARG_WRONG, "Invalid local composition type: %s", PCCompositeTypes[osm->loctype]);
  PetscFunctionReturn(0);
}

static PetscErrorCode PCMatApply_ASM(PC pc,Mat X,Mat Y)
{
  PC_ASM         *osm = (PC_ASM*)pc->data;
  Mat            Z,W;
  Vec            x;
  PetscInt       i,m,N;
  ScatterMode    forward = SCATTER_FORWARD,reverse = SCATTER_REVERSE;

  PetscFunctionBegin;
  PetscCheck(osm->n_local_true <= 1,PetscObjectComm((PetscObject)pc),PETSC_ERR_SUP,"Not yet implemented");
  /*
     support for limiting the restriction or interpolation to only local
     subdomain values (leaving the other values 0).
  */
  if (!(osm->type & PC_ASM_RESTRICT)) {
    forward = SCATTER_FORWARD_LOCAL;
    /* have to zero the work RHS since scatter may leave some slots empty */
    PetscCall(VecSet(osm->lx, 0.0));
  }
  if (!(osm->type & PC_ASM_INTERPOLATE)) {
    reverse = SCATTER_REVERSE_LOCAL;
  }
  PetscCall(VecGetLocalSize(osm->x[0], &m));
  PetscCall(MatGetSize(X, NULL, &N));
  PetscCall(MatCreateSeqDense(PETSC_COMM_SELF, m, N, NULL, &Z));
  if (osm->loctype == PC_COMPOSITE_MULTIPLICATIVE || osm->loctype == PC_COMPOSITE_ADDITIVE) {
    /* zero the global and the local solutions */
    PetscCall(MatZeroEntries(Y));
    PetscCall(VecSet(osm->ly, 0.0));

    for (i = 0; i < N; ++i) {
      PetscCall(MatDenseGetColumnVecRead(X, i, &x));
      /* copy the global RHS to local RHS including the ghost nodes */
      PetscCall(VecScatterBegin(osm->restriction, x, osm->lx, INSERT_VALUES, forward));
      PetscCall(VecScatterEnd(osm->restriction, x, osm->lx, INSERT_VALUES, forward));
      PetscCall(MatDenseRestoreColumnVecRead(X, i, &x));

      PetscCall(MatDenseGetColumnVecWrite(Z, i, &x));
      /* restrict local RHS to the overlapping 0-block RHS */
      PetscCall(VecScatterBegin(osm->lrestriction[0], osm->lx, x, INSERT_VALUES, forward));
      PetscCall(VecScatterEnd(osm->lrestriction[0], osm->lx, x, INSERT_VALUES, forward));
      PetscCall(MatDenseRestoreColumnVecWrite(Z, i, &x));
    }
    PetscCall(MatCreateSeqDense(PETSC_COMM_SELF, m, N, NULL, &W));
    /* solve the overlapping 0-block */
    PetscCall(PetscLogEventBegin(PC_ApplyOnBlocks, osm->ksp[0], Z, W, 0));
    PetscCall(KSPMatSolve(osm->ksp[0], Z, W));
    PetscCall(KSPCheckSolve(osm->ksp[0], pc, NULL));
    PetscCall(PetscLogEventEnd(PC_ApplyOnBlocks, osm->ksp[0], Z, W,0));
    PetscCall(MatDestroy(&Z));

    for (i = 0; i < N; ++i) {
      PetscCall(VecSet(osm->ly, 0.0));
      PetscCall(MatDenseGetColumnVecRead(W, i, &x));
      if (osm->lprolongation) { /* interpolate the non-overlapping 0-block solution to the local solution (only for restrictive additive) */
        PetscCall(VecScatterBegin(osm->lprolongation[0], x, osm->ly, ADD_VALUES, forward));
        PetscCall(VecScatterEnd(osm->lprolongation[0], x, osm->ly, ADD_VALUES, forward));
      } else { /* interpolate the overlapping 0-block solution to the local solution */
        PetscCall(VecScatterBegin(osm->lrestriction[0], x, osm->ly, ADD_VALUES, reverse));
        PetscCall(VecScatterEnd(osm->lrestriction[0], x, osm->ly, ADD_VALUES, reverse));
      }
      PetscCall(MatDenseRestoreColumnVecRead(W, i, &x));

      PetscCall(MatDenseGetColumnVecWrite(Y, i, &x));
      /* add the local solution to the global solution including the ghost nodes */
      PetscCall(VecScatterBegin(osm->restriction, osm->ly, x, ADD_VALUES, reverse));
      PetscCall(VecScatterEnd(osm->restriction, osm->ly, x, ADD_VALUES, reverse));
      PetscCall(MatDenseRestoreColumnVecWrite(Y, i, &x));
    }
    PetscCall(MatDestroy(&W));
  } else SETERRQ(PetscObjectComm((PetscObject)pc), PETSC_ERR_ARG_WRONG, "Invalid local composition type: %s", PCCompositeTypes[osm->loctype]);
  PetscFunctionReturn(0);
}

static PetscErrorCode PCApplyTranspose_ASM(PC pc,Vec x,Vec y)
{
  PC_ASM         *osm = (PC_ASM*)pc->data;
  PetscInt       i,n_local_true = osm->n_local_true;
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
    PetscCall(VecSet(osm->lx, 0.0));
  }
  if (!(osm->type & PC_ASM_RESTRICT)) reverse = SCATTER_REVERSE_LOCAL;

  /* zero the global and the local solutions */
  PetscCall(VecSet(y, 0.0));
  PetscCall(VecSet(osm->ly, 0.0));

  /* Copy the global RHS to local RHS including the ghost nodes */
  PetscCall(VecScatterBegin(osm->restriction, x, osm->lx, INSERT_VALUES, forward));
  PetscCall(VecScatterEnd(osm->restriction, x, osm->lx, INSERT_VALUES, forward));

  /* Restrict local RHS to the overlapping 0-block RHS */
  PetscCall(VecScatterBegin(osm->lrestriction[0], osm->lx, osm->x[0], INSERT_VALUES, forward));
  PetscCall(VecScatterEnd(osm->lrestriction[0], osm->lx, osm->x[0], INSERT_VALUES, forward));

  /* do the local solves */
  for (i = 0; i < n_local_true; ++i) {

    /* solve the overlapping i-block */
    PetscCall(PetscLogEventBegin(PC_ApplyOnBlocks,osm->ksp[i],osm->x[i],osm->y[i],0));
    PetscCall(KSPSolveTranspose(osm->ksp[i], osm->x[i], osm->y[i]));
    PetscCall(KSPCheckSolve(osm->ksp[i],pc,osm->y[i]));
    PetscCall(PetscLogEventEnd(PC_ApplyOnBlocks,osm->ksp[i],osm->x[i],osm->y[i],0));

    if (osm->lprolongation) { /* interpolate the non-overlapping i-block solution to the local solution */
      PetscCall(VecScatterBegin(osm->lprolongation[i], osm->y[i], osm->ly, ADD_VALUES, forward));
      PetscCall(VecScatterEnd(osm->lprolongation[i], osm->y[i], osm->ly, ADD_VALUES, forward));
    } else { /* interpolate the overlapping i-block solution to the local solution */
      PetscCall(VecScatterBegin(osm->lrestriction[i], osm->y[i], osm->ly, ADD_VALUES, reverse));
      PetscCall(VecScatterEnd(osm->lrestriction[i], osm->y[i], osm->ly, ADD_VALUES, reverse));
    }

    if (i < n_local_true-1) {
      /* Restrict local RHS to the overlapping (i+1)-block RHS */
      PetscCall(VecScatterBegin(osm->lrestriction[i+1], osm->lx, osm->x[i+1], INSERT_VALUES, forward));
      PetscCall(VecScatterEnd(osm->lrestriction[i+1], osm->lx, osm->x[i+1], INSERT_VALUES, forward));
    }
  }
  /* Add the local solution to the global solution including the ghost nodes */
  PetscCall(VecScatterBegin(osm->restriction, osm->ly, y, ADD_VALUES, reverse));
  PetscCall(VecScatterEnd(osm->restriction, osm->ly, y, ADD_VALUES, reverse));

  PetscFunctionReturn(0);

}

static PetscErrorCode PCReset_ASM(PC pc)
{
  PC_ASM         *osm = (PC_ASM*)pc->data;
  PetscInt       i;

  PetscFunctionBegin;
  if (osm->ksp) {
    for (i=0; i<osm->n_local_true; i++) {
      PetscCall(KSPReset(osm->ksp[i]));
    }
  }
  if (osm->pmat) {
    if (osm->n_local_true > 0) {
      PetscCall(MatDestroySubMatrices(osm->n_local_true,&osm->pmat));
    }
  }
  if (osm->lrestriction) {
    PetscCall(VecScatterDestroy(&osm->restriction));
    for (i=0; i<osm->n_local_true; i++) {
      PetscCall(VecScatterDestroy(&osm->lrestriction[i]));
      if (osm->lprolongation) PetscCall(VecScatterDestroy(&osm->lprolongation[i]));
      PetscCall(VecDestroy(&osm->x[i]));
      PetscCall(VecDestroy(&osm->y[i]));
    }
    PetscCall(PetscFree(osm->lrestriction));
    if (osm->lprolongation) PetscCall(PetscFree(osm->lprolongation));
    PetscCall(PetscFree(osm->x));
    PetscCall(PetscFree(osm->y));

  }
  PetscCall(PCASMDestroySubdomains(osm->n_local_true,osm->is,osm->is_local));
  PetscCall(ISDestroy(&osm->lis));
  PetscCall(VecDestroy(&osm->lx));
  PetscCall(VecDestroy(&osm->ly));
  if (osm->loctype == PC_COMPOSITE_MULTIPLICATIVE) {
    PetscCall(MatDestroyMatrices(osm->n_local_true, &osm->lmats));
  }

  PetscCall(PetscFree(osm->sub_mat_type));

  osm->is       = NULL;
  osm->is_local = NULL;
  PetscFunctionReturn(0);
}

static PetscErrorCode PCDestroy_ASM(PC pc)
{
  PC_ASM         *osm = (PC_ASM*)pc->data;
  PetscInt       i;

  PetscFunctionBegin;
  PetscCall(PCReset_ASM(pc));
  if (osm->ksp) {
    for (i=0; i<osm->n_local_true; i++) {
      PetscCall(KSPDestroy(&osm->ksp[i]));
    }
    PetscCall(PetscFree(osm->ksp));
  }
  PetscCall(PetscFree(pc->data));

  PetscCall(PetscObjectComposeFunction((PetscObject)pc,"PCASMSetLocalSubdomains_C",NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)pc,"PCASMSetTotalSubdomains_C",NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)pc,"PCASMSetOverlap_C",NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)pc,"PCASMSetType_C",NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)pc,"PCASMGetType_C",NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)pc,"PCASMSetLocalType_C",NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)pc,"PCASMGetLocalType_C",NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)pc,"PCASMSetSortIndices_C",NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)pc,"PCASMGetSubKSP_C",NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)pc,"PCASMGetSubMatType_C",NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)pc,"PCASMSetSubMatType_C",NULL));
  PetscFunctionReturn(0);
}

static PetscErrorCode PCSetFromOptions_ASM(PetscOptionItems *PetscOptionsObject,PC pc)
{
  PC_ASM         *osm = (PC_ASM*)pc->data;
  PetscInt       blocks,ovl;
  PetscBool      flg;
  PCASMType      asmtype;
  PCCompositeType loctype;
  char           sub_mat_type[256];

  PetscFunctionBegin;
  PetscOptionsHeadBegin(PetscOptionsObject,"Additive Schwarz options");
  PetscCall(PetscOptionsBool("-pc_asm_dm_subdomains","Use DMCreateDomainDecomposition() to define subdomains","PCASMSetDMSubdomains",osm->dm_subdomains,&osm->dm_subdomains,&flg));
  PetscCall(PetscOptionsInt("-pc_asm_blocks","Number of subdomains","PCASMSetTotalSubdomains",osm->n,&blocks,&flg));
  if (flg) {
    PetscCall(PCASMSetTotalSubdomains(pc,blocks,NULL,NULL));
    osm->dm_subdomains = PETSC_FALSE;
  }
  PetscCall(PetscOptionsInt("-pc_asm_local_blocks","Number of local subdomains","PCASMSetLocalSubdomains",osm->n_local_true,&blocks,&flg));
  if (flg) {
    PetscCall(PCASMSetLocalSubdomains(pc,blocks,NULL,NULL));
    osm->dm_subdomains = PETSC_FALSE;
  }
  PetscCall(PetscOptionsInt("-pc_asm_overlap","Number of grid points overlap","PCASMSetOverlap",osm->overlap,&ovl,&flg));
  if (flg) {
    PetscCall(PCASMSetOverlap(pc,ovl));
    osm->dm_subdomains = PETSC_FALSE;
  }
  flg  = PETSC_FALSE;
  PetscCall(PetscOptionsEnum("-pc_asm_type","Type of restriction/extension","PCASMSetType",PCASMTypes,(PetscEnum)osm->type,(PetscEnum*)&asmtype,&flg));
  if (flg) PetscCall(PCASMSetType(pc,asmtype));
  flg  = PETSC_FALSE;
  PetscCall(PetscOptionsEnum("-pc_asm_local_type","Type of local solver composition","PCASMSetLocalType",PCCompositeTypes,(PetscEnum)osm->loctype,(PetscEnum*)&loctype,&flg));
  if (flg) PetscCall(PCASMSetLocalType(pc,loctype));
  PetscCall(PetscOptionsFList("-pc_asm_sub_mat_type","Subsolve Matrix Type","PCASMSetSubMatType",MatList,NULL,sub_mat_type,256,&flg));
  if (flg) {
    PetscCall(PCASMSetSubMatType(pc,sub_mat_type));
  }
  PetscOptionsHeadEnd();
  PetscFunctionReturn(0);
}

/*------------------------------------------------------------------------------------*/

static PetscErrorCode  PCASMSetLocalSubdomains_ASM(PC pc,PetscInt n,IS is[],IS is_local[])
{
  PC_ASM         *osm = (PC_ASM*)pc->data;
  PetscInt       i;

  PetscFunctionBegin;
  PetscCheck(n >= 1,PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Each process must have 1 or more blocks, n = %" PetscInt_FMT,n);
  PetscCheck(!pc->setupcalled || (n == osm->n_local_true && !is),PetscObjectComm((PetscObject)pc),PETSC_ERR_ARG_WRONGSTATE,"PCASMSetLocalSubdomains() should be called before calling PCSetUp().");

  if (!pc->setupcalled) {
    if (is) {
      for (i=0; i<n; i++) PetscCall(PetscObjectReference((PetscObject)is[i]));
    }
    if (is_local) {
      for (i=0; i<n; i++) PetscCall(PetscObjectReference((PetscObject)is_local[i]));
    }
    PetscCall(PCASMDestroySubdomains(osm->n_local_true,osm->is,osm->is_local));

    osm->n_local_true = n;
    osm->is           = NULL;
    osm->is_local     = NULL;
    if (is) {
      PetscCall(PetscMalloc1(n,&osm->is));
      for (i=0; i<n; i++) osm->is[i] = is[i];
      /* Flag indicating that the user has set overlapping subdomains so PCASM should not increase their size. */
      osm->overlap = -1;
    }
    if (is_local) {
      PetscCall(PetscMalloc1(n,&osm->is_local));
      for (i=0; i<n; i++) osm->is_local[i] = is_local[i];
      if (!is) {
        PetscCall(PetscMalloc1(osm->n_local_true,&osm->is));
        for (i=0; i<osm->n_local_true; i++) {
          if (osm->overlap > 0) { /* With positive overlap, osm->is[i] will be modified */
            PetscCall(ISDuplicate(osm->is_local[i],&osm->is[i]));
            PetscCall(ISCopy(osm->is_local[i],osm->is[i]));
          } else {
            PetscCall(PetscObjectReference((PetscObject)osm->is_local[i]));
            osm->is[i] = osm->is_local[i];
          }
        }
      }
    }
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode  PCASMSetTotalSubdomains_ASM(PC pc,PetscInt N,IS *is,IS *is_local)
{
  PC_ASM         *osm = (PC_ASM*)pc->data;
  PetscMPIInt    rank,size;
  PetscInt       n;

  PetscFunctionBegin;
  PetscCheck(N >= 1,PetscObjectComm((PetscObject)pc),PETSC_ERR_ARG_OUTOFRANGE,"Number of total blocks must be > 0, N = %" PetscInt_FMT,N);
  PetscCheck(!is && !is_local,PetscObjectComm((PetscObject)pc),PETSC_ERR_SUP,"Use PCASMSetLocalSubdomains() to set specific index sets\n\they cannot be set globally yet.");

  /*
     Split the subdomains equally among all processors
  */
  PetscCallMPI(MPI_Comm_rank(PetscObjectComm((PetscObject)pc),&rank));
  PetscCallMPI(MPI_Comm_size(PetscObjectComm((PetscObject)pc),&size));
  n    = N/size + ((N % size) > rank);
  PetscCheck(n,PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Process %d must have at least one block: total processors %d total blocks %" PetscInt_FMT,(int)rank,(int)size,N);
  PetscCheck(!pc->setupcalled || n == osm->n_local_true,PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"PCASMSetTotalSubdomains() should be called before PCSetUp().");
  if (!pc->setupcalled) {
    PetscCall(PCASMDestroySubdomains(osm->n_local_true,osm->is,osm->is_local));

    osm->n_local_true = n;
    osm->is           = NULL;
    osm->is_local     = NULL;
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode  PCASMSetOverlap_ASM(PC pc,PetscInt ovl)
{
  PC_ASM *osm = (PC_ASM*)pc->data;

  PetscFunctionBegin;
  PetscCheck(ovl >= 0,PetscObjectComm((PetscObject)pc),PETSC_ERR_ARG_OUTOFRANGE,"Negative overlap value requested");
  PetscCheck(!pc->setupcalled || ovl == osm->overlap,PetscObjectComm((PetscObject)pc),PETSC_ERR_ARG_WRONGSTATE,"PCASMSetOverlap() should be called before PCSetUp().");
  if (!pc->setupcalled) osm->overlap = ovl;
  PetscFunctionReturn(0);
}

static PetscErrorCode  PCASMSetType_ASM(PC pc,PCASMType type)
{
  PC_ASM *osm = (PC_ASM*)pc->data;

  PetscFunctionBegin;
  osm->type     = type;
  osm->type_set = PETSC_TRUE;
  PetscFunctionReturn(0);
}

static PetscErrorCode  PCASMGetType_ASM(PC pc,PCASMType *type)
{
  PC_ASM *osm = (PC_ASM*)pc->data;

  PetscFunctionBegin;
  *type = osm->type;
  PetscFunctionReturn(0);
}

static PetscErrorCode  PCASMSetLocalType_ASM(PC pc, PCCompositeType type)
{
  PC_ASM *osm = (PC_ASM *) pc->data;

  PetscFunctionBegin;
  PetscCheck(type == PC_COMPOSITE_ADDITIVE || type == PC_COMPOSITE_MULTIPLICATIVE,PetscObjectComm((PetscObject)pc),PETSC_ERR_SUP,"Only supports additive or multiplicative as the local type");
  osm->loctype = type;
  PetscFunctionReturn(0);
}

static PetscErrorCode  PCASMGetLocalType_ASM(PC pc, PCCompositeType *type)
{
  PC_ASM *osm = (PC_ASM *) pc->data;

  PetscFunctionBegin;
  *type = osm->loctype;
  PetscFunctionReturn(0);
}

static PetscErrorCode  PCASMSetSortIndices_ASM(PC pc,PetscBool  doSort)
{
  PC_ASM *osm = (PC_ASM*)pc->data;

  PetscFunctionBegin;
  osm->sort_indices = doSort;
  PetscFunctionReturn(0);
}

static PetscErrorCode  PCASMGetSubKSP_ASM(PC pc,PetscInt *n_local,PetscInt *first_local,KSP **ksp)
{
  PC_ASM         *osm = (PC_ASM*)pc->data;

  PetscFunctionBegin;
  PetscCheck(osm->n_local_true >= 1,PetscObjectComm((PetscObject)pc),PETSC_ERR_ORDER,"Need to call PCSetUp() on PC (or KSPSetUp() on the outer KSP object) before calling here");

  if (n_local) *n_local = osm->n_local_true;
  if (first_local) {
    PetscCallMPI(MPI_Scan(&osm->n_local_true,first_local,1,MPIU_INT,MPI_SUM,PetscObjectComm((PetscObject)pc)));
    *first_local -= osm->n_local_true;
  }
  if (ksp) *ksp   = osm->ksp;
  PetscFunctionReturn(0);
}

static PetscErrorCode  PCASMGetSubMatType_ASM(PC pc,MatType *sub_mat_type)
{
  PC_ASM         *osm = (PC_ASM*)pc->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_CLASSID,1);
  PetscValidPointer(sub_mat_type,2);
  *sub_mat_type = osm->sub_mat_type;
  PetscFunctionReturn(0);
}

static PetscErrorCode PCASMSetSubMatType_ASM(PC pc,MatType sub_mat_type)
{
  PC_ASM            *osm = (PC_ASM*)pc->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_CLASSID,1);
  PetscCall(PetscFree(osm->sub_mat_type));
  PetscCall(PetscStrallocpy(sub_mat_type,(char**)&osm->sub_mat_type));
  PetscFunctionReturn(0);
}

/*@C
    PCASMSetLocalSubdomains - Sets the local subdomains (for this processor only) for the additive Schwarz preconditioner.

    Collective on pc

    Input Parameters:
+   pc - the preconditioner context
.   n - the number of subdomains for this processor (default value = 1)
.   is - the index set that defines the subdomains for this processor
         (or NULL for PETSc to determine subdomains)
-   is_local - the index sets that define the local part of the subdomains for this processor, not used unless PCASMType is PC_ASM_RESTRICT
         (or NULL to not provide these)

    Options Database Key:
    To set the total number of subdomain blocks rather than specify the
    index sets, use the option
.    -pc_asm_local_blocks <blks> - Sets local blocks

    Notes:
    The IS numbering is in the parallel, global numbering of the vector for both is and is_local

    By default the ASM preconditioner uses 1 block per processor.

    Use PCASMSetTotalSubdomains() to set the subdomains for all processors.

    If is_local is provided and PCASMType is PC_ASM_RESTRICT then the solution only over the is_local region is interpolated
    back to form the global solution (this is the standard restricted additive Schwarz method)

    If the is_local is provided and PCASMType is PC_ASM_INTERPOLATE or PC_ASM_NONE then an error is generated since there is
    no code to handle that case.

    Level: advanced

.seealso: `PCASMSetTotalSubdomains()`, `PCASMSetOverlap()`, `PCASMGetSubKSP()`,
          `PCASMCreateSubdomains2D()`, `PCASMGetLocalSubdomains()`, `PCASMType`, `PCASMSetType()`
@*/
PetscErrorCode  PCASMSetLocalSubdomains(PC pc,PetscInt n,IS is[],IS is_local[])
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_CLASSID,1);
  PetscTryMethod(pc,"PCASMSetLocalSubdomains_C",(PC,PetscInt,IS[],IS[]),(pc,n,is,is_local));
  PetscFunctionReturn(0);
}

/*@C
    PCASMSetTotalSubdomains - Sets the subdomains for all processors for the
    additive Schwarz preconditioner.  Either all or no processors in the
    PC communicator must call this routine, with the same index sets.

    Collective on pc

    Input Parameters:
+   pc - the preconditioner context
.   N  - the number of subdomains for all processors
.   is - the index sets that define the subdomains for all processors
         (or NULL to ask PETSc to determine the subdomains)
-   is_local - the index sets that define the local part of the subdomains for this processor
         (or NULL to not provide this information)

    Options Database Key:
    To set the total number of subdomain blocks rather than specify the
    index sets, use the option
.    -pc_asm_blocks <blks> - Sets total blocks

    Notes:
    Currently you cannot use this to set the actual subdomains with the argument is or is_local.

    By default the ASM preconditioner uses 1 block per processor.

    These index sets cannot be destroyed until after completion of the
    linear solves for which the ASM preconditioner is being used.

    Use PCASMSetLocalSubdomains() to set local subdomains.

    The IS numbering is in the parallel, global numbering of the vector for both is and is_local

    Level: advanced

.seealso: `PCASMSetLocalSubdomains()`, `PCASMSetOverlap()`, `PCASMGetSubKSP()`,
          `PCASMCreateSubdomains2D()`
@*/
PetscErrorCode  PCASMSetTotalSubdomains(PC pc,PetscInt N,IS is[],IS is_local[])
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_CLASSID,1);
  PetscTryMethod(pc,"PCASMSetTotalSubdomains_C",(PC,PetscInt,IS[],IS[]),(pc,N,is,is_local));
  PetscFunctionReturn(0);
}

/*@
    PCASMSetOverlap - Sets the overlap between a pair of subdomains for the
    additive Schwarz preconditioner.  Either all or no processors in the
    PC communicator must call this routine.

    Logically Collective on pc

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

    The default algorithm used by PETSc to increase overlap is fast, but not scalable,
    use the option -mat_increase_overlap_scalable when the problem and number of processes is large.

    Note that one can define initial index sets with any overlap via
    PCASMSetLocalSubdomains(); the routine
    PCASMSetOverlap() merely allows PETSc to extend that overlap further
    if desired.

    Level: intermediate

.seealso: `PCASMSetTotalSubdomains()`, `PCASMSetLocalSubdomains()`, `PCASMGetSubKSP()`,
          `PCASMCreateSubdomains2D()`, `PCASMGetLocalSubdomains()`, `MatIncreaseOverlap()`
@*/
PetscErrorCode  PCASMSetOverlap(PC pc,PetscInt ovl)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_CLASSID,1);
  PetscValidLogicalCollectiveInt(pc,ovl,2);
  PetscTryMethod(pc,"PCASMSetOverlap_C",(PC,PetscInt),(pc,ovl));
  PetscFunctionReturn(0);
}

/*@
    PCASMSetType - Sets the type of restriction and interpolation used
    for local problems in the additive Schwarz method.

    Logically Collective on pc

    Input Parameters:
+   pc  - the preconditioner context
-   type - variant of ASM, one of
.vb
      PC_ASM_BASIC       - full interpolation and restriction
      PC_ASM_RESTRICT    - full restriction, local processor interpolation (default)
      PC_ASM_INTERPOLATE - full interpolation, local processor restriction
      PC_ASM_NONE        - local processor restriction and interpolation
.ve

    Options Database Key:
.   -pc_asm_type [basic,restrict,interpolate,none] - Sets ASM type

    Notes:
    if the is_local arguments are passed to PCASMSetLocalSubdomains() then they are used when PC_ASM_RESTRICT has been selected
    to limit the local processor interpolation

    Level: intermediate

.seealso: `PCASMSetTotalSubdomains()`, `PCASMSetTotalSubdomains()`, `PCASMGetSubKSP()`,
          `PCASMCreateSubdomains2D()`, `PCASMType`, `PCASMSetLocalType()`, `PCASMGetLocalType()`
@*/
PetscErrorCode  PCASMSetType(PC pc,PCASMType type)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_CLASSID,1);
  PetscValidLogicalCollectiveEnum(pc,type,2);
  PetscTryMethod(pc,"PCASMSetType_C",(PC,PCASMType),(pc,type));
  PetscFunctionReturn(0);
}

/*@
    PCASMGetType - Gets the type of restriction and interpolation used
    for local problems in the additive Schwarz method.

    Logically Collective on pc

    Input Parameter:
.   pc  - the preconditioner context

    Output Parameter:
.   type - variant of ASM, one of

.vb
      PC_ASM_BASIC       - full interpolation and restriction
      PC_ASM_RESTRICT    - full restriction, local processor interpolation
      PC_ASM_INTERPOLATE - full interpolation, local processor restriction
      PC_ASM_NONE        - local processor restriction and interpolation
.ve

    Options Database Key:
.   -pc_asm_type [basic,restrict,interpolate,none] - Sets ASM type

    Level: intermediate

.seealso: `PCASMSetTotalSubdomains()`, `PCASMSetTotalSubdomains()`, `PCASMGetSubKSP()`,
          `PCASMCreateSubdomains2D()`, `PCASMType`, `PCASMSetType()`, `PCASMSetLocalType()`, `PCASMGetLocalType()`
@*/
PetscErrorCode  PCASMGetType(PC pc,PCASMType *type)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_CLASSID,1);
  PetscUseMethod(pc,"PCASMGetType_C",(PC,PCASMType*),(pc,type));
  PetscFunctionReturn(0);
}

/*@
  PCASMSetLocalType - Sets the type of composition used for local problems in the additive Schwarz method.

  Logically Collective on pc

  Input Parameters:
+ pc  - the preconditioner context
- type - type of composition, one of
.vb
  PC_COMPOSITE_ADDITIVE       - local additive combination
  PC_COMPOSITE_MULTIPLICATIVE - local multiplicative combination
.ve

  Options Database Key:
. -pc_asm_local_type [additive,multiplicative] - Sets local solver composition type

  Level: intermediate

.seealso: `PCASMSetType()`, `PCASMGetType()`, `PCASMGetLocalType()`, `PCASM`, `PCASMType`, `PCASMSetType()`, `PCASMGetType()`, `PCCompositeType`
@*/
PetscErrorCode PCASMSetLocalType(PC pc, PCCompositeType type)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc, PC_CLASSID, 1);
  PetscValidLogicalCollectiveEnum(pc, type, 2);
  PetscTryMethod(pc, "PCASMSetLocalType_C", (PC, PCCompositeType), (pc, type));
  PetscFunctionReturn(0);
}

/*@
  PCASMGetLocalType - Gets the type of composition used for local problems in the additive Schwarz method.

  Logically Collective on pc

  Input Parameter:
. pc  - the preconditioner context

  Output Parameter:
. type - type of composition, one of
.vb
  PC_COMPOSITE_ADDITIVE       - local additive combination
  PC_COMPOSITE_MULTIPLICATIVE - local multiplicative combination
.ve

  Options Database Key:
. -pc_asm_local_type [additive,multiplicative] - Sets local solver composition type

  Level: intermediate

.seealso: `PCASMSetType()`, `PCASMGetType()`, `PCASMSetLocalType()`, `PCASMCreate()`, `PCASMType`, `PCASMSetType()`, `PCASMGetType()`, `PCCompositeType`
@*/
PetscErrorCode PCASMGetLocalType(PC pc, PCCompositeType *type)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc, PC_CLASSID, 1);
  PetscValidPointer(type, 2);
  PetscUseMethod(pc, "PCASMGetLocalType_C", (PC, PCCompositeType *), (pc, type));
  PetscFunctionReturn(0);
}

/*@
    PCASMSetSortIndices - Determines whether subdomain indices are sorted.

    Logically Collective on pc

    Input Parameters:
+   pc  - the preconditioner context
-   doSort - sort the subdomain indices

    Level: intermediate

.seealso: `PCASMSetLocalSubdomains()`, `PCASMSetTotalSubdomains()`, `PCASMGetSubKSP()`,
          `PCASMCreateSubdomains2D()`
@*/
PetscErrorCode  PCASMSetSortIndices(PC pc,PetscBool doSort)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_CLASSID,1);
  PetscValidLogicalCollectiveBool(pc,doSort,2);
  PetscTryMethod(pc,"PCASMSetSortIndices_C",(PC,PetscBool),(pc,doSort));
  PetscFunctionReturn(0);
}

/*@C
   PCASMGetSubKSP - Gets the local KSP contexts for all blocks on
   this processor.

   Collective on pc iff first_local is requested

   Input Parameter:
.  pc - the preconditioner context

   Output Parameters:
+  n_local - the number of blocks on this processor or NULL
.  first_local - the global number of the first block on this processor or NULL,
                 all processors must request or all must pass NULL
-  ksp - the array of KSP contexts

   Note:
   After PCASMGetSubKSP() the array of KSPes is not to be freed.

   You must call KSPSetUp() before calling PCASMGetSubKSP().

   Fortran note:
   The output argument 'ksp' must be an array of sufficient length or PETSC_NULL_KSP. The latter can be used to learn the necessary length.

   Level: advanced

.seealso: `PCASMSetTotalSubdomains()`, `PCASMSetTotalSubdomains()`, `PCASMSetOverlap()`,
          `PCASMCreateSubdomains2D()`,
@*/
PetscErrorCode  PCASMGetSubKSP(PC pc,PetscInt *n_local,PetscInt *first_local,KSP *ksp[])
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_CLASSID,1);
  PetscUseMethod(pc,"PCASMGetSubKSP_C",(PC,PetscInt*,PetscInt*,KSP **),(pc,n_local,first_local,ksp));
  PetscFunctionReturn(0);
}

/* -------------------------------------------------------------------------------------*/
/*MC
   PCASM - Use the (restricted) additive Schwarz method, each block is (approximately) solved with
           its own KSP object.

   Options Database Keys:
+  -pc_asm_blocks <blks> - Sets total blocks
.  -pc_asm_overlap <ovl> - Sets overlap
.  -pc_asm_type [basic,restrict,interpolate,none] - Sets ASM type, default is restrict
-  -pc_asm_local_type [additive, multiplicative] - Sets ASM type, default is additive

     IMPORTANT: If you run with, for example, 3 blocks on 1 processor or 3 blocks on 3 processors you
      will get a different convergence rate due to the default option of -pc_asm_type restrict. Use
      -pc_asm_type basic to use the standard ASM.

   Notes:
    Each processor can have one or more blocks, but a block cannot be shared by more
     than one processor. Use PCGASM for subdomains shared by multiple processes. Defaults to one block per processor.

     To set options on the solvers for each block append -sub_ to all the KSP, and PC
        options database keys. For example, -sub_pc_type ilu -sub_pc_factor_levels 1 -sub_ksp_type preonly

     To set the options on the solvers separate for each block call PCASMGetSubKSP()
         and set the options directly on the resulting KSP object (you can access its PC
         with KSPGetPC())

   Level: beginner

    References:
+   * - M Dryja, OB Widlund, An additive variant of the Schwarz alternating method for the case of many subregions
     Courant Institute, New York University Technical report
-   * - Barry Smith, Petter Bjorstad, and William Gropp, Domain Decompositions: Parallel Multilevel Methods for Elliptic Partial Differential Equations,
    Cambridge University Press.

.seealso: `PCCreate()`, `PCSetType()`, `PCType`, `PC`,
          `PCBJACOBI`, `PCASMGetSubKSP()`, `PCASMSetLocalSubdomains()`, `PCASMType`, `PCASMGetType()`, `PCASMSetLocalType()`, `PCASMGetLocalType()`
          `PCASMSetTotalSubdomains()`, `PCSetModifySubMatrices()`, `PCASMSetOverlap()`, `PCASMSetType()`, `PCCompositeType`

M*/

PETSC_EXTERN PetscErrorCode PCCreate_ASM(PC pc)
{
  PC_ASM         *osm;

  PetscFunctionBegin;
  PetscCall(PetscNewLog(pc,&osm));

  osm->n                 = PETSC_DECIDE;
  osm->n_local           = 0;
  osm->n_local_true      = PETSC_DECIDE;
  osm->overlap           = 1;
  osm->ksp               = NULL;
  osm->restriction       = NULL;
  osm->lprolongation     = NULL;
  osm->lrestriction      = NULL;
  osm->x                 = NULL;
  osm->y                 = NULL;
  osm->is                = NULL;
  osm->is_local          = NULL;
  osm->mat               = NULL;
  osm->pmat              = NULL;
  osm->type              = PC_ASM_RESTRICT;
  osm->loctype           = PC_COMPOSITE_ADDITIVE;
  osm->sort_indices      = PETSC_TRUE;
  osm->dm_subdomains     = PETSC_FALSE;
  osm->sub_mat_type      = NULL;

  pc->data                 = (void*)osm;
  pc->ops->apply           = PCApply_ASM;
  pc->ops->matapply        = PCMatApply_ASM;
  pc->ops->applytranspose  = PCApplyTranspose_ASM;
  pc->ops->setup           = PCSetUp_ASM;
  pc->ops->reset           = PCReset_ASM;
  pc->ops->destroy         = PCDestroy_ASM;
  pc->ops->setfromoptions  = PCSetFromOptions_ASM;
  pc->ops->setuponblocks   = PCSetUpOnBlocks_ASM;
  pc->ops->view            = PCView_ASM;
  pc->ops->applyrichardson = NULL;

  PetscCall(PetscObjectComposeFunction((PetscObject)pc,"PCASMSetLocalSubdomains_C",PCASMSetLocalSubdomains_ASM));
  PetscCall(PetscObjectComposeFunction((PetscObject)pc,"PCASMSetTotalSubdomains_C",PCASMSetTotalSubdomains_ASM));
  PetscCall(PetscObjectComposeFunction((PetscObject)pc,"PCASMSetOverlap_C",PCASMSetOverlap_ASM));
  PetscCall(PetscObjectComposeFunction((PetscObject)pc,"PCASMSetType_C",PCASMSetType_ASM));
  PetscCall(PetscObjectComposeFunction((PetscObject)pc,"PCASMGetType_C",PCASMGetType_ASM));
  PetscCall(PetscObjectComposeFunction((PetscObject)pc,"PCASMSetLocalType_C",PCASMSetLocalType_ASM));
  PetscCall(PetscObjectComposeFunction((PetscObject)pc,"PCASMGetLocalType_C",PCASMGetLocalType_ASM));
  PetscCall(PetscObjectComposeFunction((PetscObject)pc,"PCASMSetSortIndices_C",PCASMSetSortIndices_ASM));
  PetscCall(PetscObjectComposeFunction((PetscObject)pc,"PCASMGetSubKSP_C",PCASMGetSubKSP_ASM));
  PetscCall(PetscObjectComposeFunction((PetscObject)pc,"PCASMGetSubMatType_C",PCASMGetSubMatType_ASM));
  PetscCall(PetscObjectComposeFunction((PetscObject)pc,"PCASMSetSubMatType_C",PCASMSetSubMatType_ASM));
  PetscFunctionReturn(0);
}

/*@C
   PCASMCreateSubdomains - Creates the index sets for the overlapping Schwarz
   preconditioner for any problem on a general grid.

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

.seealso: `PCASMSetLocalSubdomains()`, `PCASMDestroySubdomains()`
@*/
PetscErrorCode  PCASMCreateSubdomains(Mat A, PetscInt n, IS* outis[])
{
  MatPartitioning mpart;
  const char      *prefix;
  PetscInt        i,j,rstart,rend,bs;
  PetscBool       hasop, isbaij = PETSC_FALSE,foundpart = PETSC_FALSE;
  Mat             Ad     = NULL, adj;
  IS              ispart,isnumb,*is;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(A,MAT_CLASSID,1);
  PetscValidPointer(outis,3);
  PetscCheck(n >= 1,PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"number of local blocks must be > 0, n = %" PetscInt_FMT,n);

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
  if (Ad && n > 1) {
    PetscBool match,done;
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
            if (row[j] != i) { /* if not diagonal */
              jja[nnz+cnt++] = row[j];
            }
          }
          nnz     += cnt;
          iia[i+1] = nnz;
        }
        /* Partitioning of the adjacency matrix */
        PetscCall(MatCreateMPIAdj(PETSC_COMM_SELF,na,na,iia,jja,NULL,&adj));
        PetscCall(MatPartitioningSetAdjacency(mpart,adj));
        PetscCall(MatPartitioningSetNParts(mpart,n));
        PetscCall(MatPartitioningApply(mpart,&ispart));
        PetscCall(ISPartitioningToNumbering(ispart,&isnumb));
        PetscCall(MatDestroy(&adj));
        foundpart = PETSC_TRUE;
      }
      PetscCall(MatRestoreRowIJ(Ad,0,PETSC_TRUE,isbaij,&na,&ia,&ja,&done));
    }
    PetscCall(MatPartitioningDestroy(&mpart));
  }

  PetscCall(PetscMalloc1(n,&is));
  *outis = is;

  if (!foundpart) {

    /* Partitioning by contiguous chunks of rows */

    PetscInt mbs   = (rend-rstart)/bs;
    PetscInt start = rstart;
    for (i=0; i<n; i++) {
      PetscInt count = (mbs/n + ((mbs % n) > i)) * bs;
      PetscCall(ISCreateStride(PETSC_COMM_SELF,count,start,1,&is[i]));
      start += count;
    }

  } else {

    /* Partitioning by adjacency of diagonal block  */

    const PetscInt *numbering;
    PetscInt       *count,nidx,*indices,*newidx,start=0;
    /* Get node count in each partition */
    PetscCall(PetscMalloc1(n,&count));
    PetscCall(ISPartitioningCount(ispart,n,count));
    if (isbaij && bs > 1) { /* adjust for the block-aij case */
      for (i=0; i<n; i++) count[i] *= bs;
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
    for (i=0; i<n; i++) {
      PetscCall(ISCreateGeneral(PETSC_COMM_SELF,count[i],&indices[start],PETSC_COPY_VALUES,&is[i]));
      PetscCall(ISSort(is[i]));
      start += count[i];
    }

    PetscCall(PetscFree(count));
    PetscCall(PetscFree(indices));
    PetscCall(ISDestroy(&isnumb));
    PetscCall(ISDestroy(&ispart));

  }
  PetscFunctionReturn(0);
}

/*@C
   PCASMDestroySubdomains - Destroys the index sets created with
   PCASMCreateSubdomains(). Should be called after setting subdomains
   with PCASMSetLocalSubdomains().

   Collective

   Input Parameters:
+  n - the number of index sets
.  is - the array of index sets
-  is_local - the array of local index sets, can be NULL

   Level: advanced

.seealso: `PCASMCreateSubdomains()`, `PCASMSetLocalSubdomains()`
@*/
PetscErrorCode  PCASMDestroySubdomains(PetscInt n, IS is[], IS is_local[])
{
  PetscInt       i;

  PetscFunctionBegin;
  if (n <= 0) PetscFunctionReturn(0);
  if (is) {
    PetscValidPointer(is,2);
    for (i=0; i<n; i++) PetscCall(ISDestroy(&is[i]));
    PetscCall(PetscFree(is));
  }
  if (is_local) {
    PetscValidPointer(is_local,3);
    for (i=0; i<n; i++) PetscCall(ISDestroy(&is_local[i]));
    PetscCall(PetscFree(is_local));
  }
  PetscFunctionReturn(0);
}

/*@
   PCASMCreateSubdomains2D - Creates the index sets for the overlapping Schwarz
   preconditioner for a two-dimensional problem on a regular grid.

   Not Collective

   Input Parameters:
+  m   - the number of mesh points in the x direction
.  n   - the number of mesh points in the y direction
.  M   - the number of subdomains in the x direction
.  N   - the number of subdomains in the y direction
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

.seealso: `PCASMSetTotalSubdomains()`, `PCASMSetLocalSubdomains()`, `PCASMGetSubKSP()`,
          `PCASMSetOverlap()`
@*/
PetscErrorCode  PCASMCreateSubdomains2D(PetscInt m,PetscInt n,PetscInt M,PetscInt N,PetscInt dof,PetscInt overlap,PetscInt *Nsub,IS **is,IS **is_local)
{
  PetscInt       i,j,height,width,ystart,xstart,yleft,yright,xleft,xright,loc_outer;
  PetscInt       nidx,*idx,loc,ii,jj,count;

  PetscFunctionBegin;
  PetscCheck(dof == 1,PETSC_COMM_SELF,PETSC_ERR_SUP,"dof must be 1");

  *Nsub     = N*M;
  PetscCall(PetscMalloc1(*Nsub,is));
  PetscCall(PetscMalloc1(*Nsub,is_local));
  ystart    = 0;
  loc_outer = 0;
  for (i=0; i<N; i++) {
    height = n/N + ((n % N) > i); /* height of subdomain */
    PetscCheck(height >= 2,PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Too many N subdomains for mesh dimension n");
    yleft  = ystart - overlap; if (yleft < 0) yleft = 0;
    yright = ystart + height + overlap; if (yright > n) yright = n;
    xstart = 0;
    for (j=0; j<M; j++) {
      width = m/M + ((m % M) > j); /* width of subdomain */
      PetscCheck(width >= 2,PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Too many M subdomains for mesh dimension m");
      xleft  = xstart - overlap; if (xleft < 0) xleft = 0;
      xright = xstart + width + overlap; if (xright > m) xright = m;
      nidx   = (xright - xleft)*(yright - yleft);
      PetscCall(PetscMalloc1(nidx,&idx));
      loc    = 0;
      for (ii=yleft; ii<yright; ii++) {
        count = m*ii + xleft;
        for (jj=xleft; jj<xright; jj++) idx[loc++] = count++;
      }
      PetscCall(ISCreateGeneral(PETSC_COMM_SELF,nidx,idx,PETSC_COPY_VALUES,(*is)+loc_outer));
      if (overlap == 0) {
        PetscCall(PetscObjectReference((PetscObject)(*is)[loc_outer]));

        (*is_local)[loc_outer] = (*is)[loc_outer];
      } else {
        for (loc=0,ii=ystart; ii<ystart+height; ii++) {
          for (jj=xstart; jj<xstart+width; jj++) {
            idx[loc++] = m*ii + jj;
          }
        }
        PetscCall(ISCreateGeneral(PETSC_COMM_SELF,loc,idx,PETSC_COPY_VALUES,*is_local+loc_outer));
      }
      PetscCall(PetscFree(idx));
      xstart += width;
      loc_outer++;
    }
    ystart += height;
  }
  for (i=0; i<*Nsub; i++) PetscCall(ISSort((*is)[i]));
  PetscFunctionReturn(0);
}

/*@C
    PCASMGetLocalSubdomains - Gets the local subdomains (for this processor
    only) for the additive Schwarz preconditioner.

    Not Collective

    Input Parameter:
.   pc - the preconditioner context

    Output Parameters:
+   n - if requested, the number of subdomains for this processor (default value = 1)
.   is - if requested, the index sets that define the subdomains for this processor
-   is_local - if requested, the index sets that define the local part of the subdomains for this processor (can be NULL)

    Notes:
    The IS numbering is in the parallel, global numbering of the vector.

    Level: advanced

.seealso: `PCASMSetTotalSubdomains()`, `PCASMSetOverlap()`, `PCASMGetSubKSP()`,
          `PCASMCreateSubdomains2D()`, `PCASMSetLocalSubdomains()`, `PCASMGetLocalSubmatrices()`
@*/
PetscErrorCode  PCASMGetLocalSubdomains(PC pc,PetscInt *n,IS *is[],IS *is_local[])
{
  PC_ASM         *osm = (PC_ASM*)pc->data;
  PetscBool      match;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_CLASSID,1);
  if (n) PetscValidIntPointer(n,2);
  if (is) PetscValidPointer(is,3);
  if (is_local) PetscValidPointer(is_local,4);
  PetscCall(PetscObjectTypeCompare((PetscObject)pc,PCASM,&match));
  PetscCheck(match,PetscObjectComm((PetscObject)pc),PETSC_ERR_ARG_WRONG,"PC is not a PCASM");
  if (n) *n = osm->n_local_true;
  if (is) *is = osm->is;
  if (is_local) *is_local = osm->is_local;
  PetscFunctionReturn(0);
}

/*@C
    PCASMGetLocalSubmatrices - Gets the local submatrices (for this processor
    only) for the additive Schwarz preconditioner.

    Not Collective

    Input Parameter:
.   pc - the preconditioner context

    Output Parameters:
+   n - if requested, the number of matrices for this processor (default value = 1)
-   mat - if requested, the matrices

    Level: advanced

    Notes:
    Call after PCSetUp() (or KSPSetUp()) but before PCApply() and before PCSetUpOnBlocks())

           Usually one would use PCSetModifySubMatrices() to change the submatrices in building the preconditioner.

.seealso: `PCASMSetTotalSubdomains()`, `PCASMSetOverlap()`, `PCASMGetSubKSP()`,
          `PCASMCreateSubdomains2D()`, `PCASMSetLocalSubdomains()`, `PCASMGetLocalSubdomains()`, `PCSetModifySubMatrices()`
@*/
PetscErrorCode  PCASMGetLocalSubmatrices(PC pc,PetscInt *n,Mat *mat[])
{
  PC_ASM         *osm;
  PetscBool      match;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_CLASSID,1);
  if (n) PetscValidIntPointer(n,2);
  if (mat) PetscValidPointer(mat,3);
  PetscCheck(pc->setupcalled,PetscObjectComm((PetscObject)pc),PETSC_ERR_ARG_WRONGSTATE,"Must call after KSPSetUp() or PCSetUp().");
  PetscCall(PetscObjectTypeCompare((PetscObject)pc,PCASM,&match));
  if (!match) {
    if (n) *n = 0;
    if (mat) *mat = NULL;
  } else {
    osm = (PC_ASM*)pc->data;
    if (n) *n = osm->n_local_true;
    if (mat) *mat = osm->pmat;
  }
  PetscFunctionReturn(0);
}

/*@
    PCASMSetDMSubdomains - Indicates whether to use DMCreateDomainDecomposition() to define the subdomains, whenever possible.

    Logically Collective

    Input Parameters:
+   pc  - the preconditioner
-   flg - boolean indicating whether to use subdomains defined by the DM

    Options Database Key:
.   -pc_asm_dm_subdomains <bool> - use subdomains defined by the DM

    Level: intermediate

    Notes:
    PCASMSetTotalSubdomains() and PCASMSetOverlap() take precedence over PCASMSetDMSubdomains(),
    so setting either of the first two effectively turns the latter off.

.seealso: `PCASMGetDMSubdomains()`, `PCASMSetTotalSubdomains()`, `PCASMSetOverlap()`
          `PCASMCreateSubdomains2D()`, `PCASMSetLocalSubdomains()`, `PCASMGetLocalSubdomains()`
@*/
PetscErrorCode  PCASMSetDMSubdomains(PC pc,PetscBool flg)
{
  PC_ASM         *osm = (PC_ASM*)pc->data;
  PetscBool      match;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_CLASSID,1);
  PetscValidLogicalCollectiveBool(pc,flg,2);
  PetscCheck(!pc->setupcalled,((PetscObject)pc)->comm,PETSC_ERR_ARG_WRONGSTATE,"Not for a setup PC.");
  PetscCall(PetscObjectTypeCompare((PetscObject)pc,PCASM,&match));
  if (match) {
    osm->dm_subdomains = flg;
  }
  PetscFunctionReturn(0);
}

/*@
    PCASMGetDMSubdomains - Returns flag indicating whether to use DMCreateDomainDecomposition() to define the subdomains, whenever possible.
    Not Collective

    Input Parameter:
.   pc  - the preconditioner

    Output Parameter:
.   flg - boolean indicating whether to use subdomains defined by the DM

    Level: intermediate

.seealso: `PCASMSetDMSubdomains()`, `PCASMSetTotalSubdomains()`, `PCASMSetOverlap()`
          `PCASMCreateSubdomains2D()`, `PCASMSetLocalSubdomains()`, `PCASMGetLocalSubdomains()`
@*/
PetscErrorCode  PCASMGetDMSubdomains(PC pc,PetscBool* flg)
{
  PC_ASM         *osm = (PC_ASM*)pc->data;
  PetscBool      match;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_CLASSID,1);
  PetscValidBoolPointer(flg,2);
  PetscCall(PetscObjectTypeCompare((PetscObject)pc,PCASM,&match));
  if (match) *flg = osm->dm_subdomains;
  else *flg = PETSC_FALSE;
  PetscFunctionReturn(0);
}

/*@
     PCASMGetSubMatType - Gets the matrix type used for ASM subsolves, as a string.

   Not Collective

   Input Parameter:
.  pc - the PC

   Output Parameter:
.  -pc_asm_sub_mat_type - name of matrix type

   Level: advanced

.seealso: `PCASMSetSubMatType()`, `PCASM`, `PCSetType()`, `VecSetType()`, `MatType`, `Mat`
@*/
PetscErrorCode  PCASMGetSubMatType(PC pc,MatType *sub_mat_type)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_CLASSID,1);
  PetscTryMethod(pc,"PCASMGetSubMatType_C",(PC,MatType*),(pc,sub_mat_type));
  PetscFunctionReturn(0);
}

/*@
     PCASMSetSubMatType - Set the type of matrix used for ASM subsolves

   Collective on Mat

   Input Parameters:
+  pc             - the PC object
-  sub_mat_type   - matrix type

   Options Database Key:
.  -pc_asm_sub_mat_type  <sub_mat_type> - Sets the matrix type used for subsolves, for example, seqaijviennacl. If you specify a base name like aijviennacl, the corresponding sequential type is assumed.

   Notes:
   See "${PETSC_DIR}/include/petscmat.h" for available types

  Level: advanced

.seealso: `PCASMGetSubMatType()`, `PCASM`, `PCSetType()`, `VecSetType()`, `MatType`, `Mat`
@*/
PetscErrorCode PCASMSetSubMatType(PC pc,MatType sub_mat_type)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_CLASSID,1);
  PetscTryMethod(pc,"PCASMSetSubMatType_C",(PC,MatType),(pc,sub_mat_type));
  PetscFunctionReturn(0);
}
