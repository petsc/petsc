/*
  This file defines an additive Schwarz preconditioner for any Mat implementation.

  Note that each processor may have any number of subdomains. But in order to
  deal easily with the VecScatter(), we treat each processor as if it has the
  same number of subdomains.

       n - total number of true subdomains on all processors
       n_local_true - actual number of subdomains on this processor
       n_local = maximum over all processors of n_local_true
*/

#include <../src/ksp/pc/impls/asm/asm.h> /*I "petscpc.h" I*/

static PetscErrorCode PCView_ASM(PC pc,PetscViewer viewer)
{
  PC_ASM         *osm = (PC_ASM*)pc->data;
  PetscErrorCode ierr;
  PetscMPIInt    rank;
  PetscInt       i,bsz;
  PetscBool      iascii,isstring;
  PetscViewer    sviewer;

  PetscFunctionBegin;
  ierr = PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&iascii);CHKERRQ(ierr);
  ierr = PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERSTRING,&isstring);CHKERRQ(ierr);
  if (iascii) {
    char overlaps[256] = "user-defined overlap",blocks[256] = "total subdomain blocks not yet set";
    if (osm->overlap >= 0) {ierr = PetscSNPrintf(overlaps,sizeof(overlaps),"amount of overlap = %D",osm->overlap);CHKERRQ(ierr);}
    if (osm->n > 0) {ierr = PetscSNPrintf(blocks,sizeof(blocks),"total subdomain blocks = %D",osm->n);CHKERRQ(ierr);}
    ierr = PetscViewerASCIIPrintf(viewer,"  %s, %s\n",blocks,overlaps);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"  restriction/interpolation type - %s\n",PCASMTypes[osm->type]);CHKERRQ(ierr);
    if (osm->dm_subdomains) {ierr = PetscViewerASCIIPrintf(viewer,"  Additive Schwarz: using DM to define subdomains\n");CHKERRQ(ierr);}
    if (osm->loctype != PC_COMPOSITE_ADDITIVE) {ierr = PetscViewerASCIIPrintf(viewer,"  Additive Schwarz: local solve composition type - %s\n",PCCompositeTypes[osm->loctype]);CHKERRQ(ierr);}
    ierr = MPI_Comm_rank(PetscObjectComm((PetscObject)pc),&rank);CHKERRQ(ierr);
    if (osm->same_local_solves) {
      if (osm->ksp) {
        ierr = PetscViewerASCIIPrintf(viewer,"  Local solve is same for all blocks, in the following KSP and PC objects:\n");CHKERRQ(ierr);
        ierr = PetscViewerGetSubViewer(viewer,PETSC_COMM_SELF,&sviewer);CHKERRQ(ierr);
        if (!rank) {
          ierr = PetscViewerASCIIPushTab(viewer);CHKERRQ(ierr);
          ierr = KSPView(osm->ksp[0],sviewer);CHKERRQ(ierr);
          ierr = PetscViewerASCIIPopTab(viewer);CHKERRQ(ierr);
        }
        ierr = PetscViewerRestoreSubViewer(viewer,PETSC_COMM_SELF,&sviewer);CHKERRQ(ierr);
      }
    } else {
      ierr = PetscViewerASCIIPushSynchronized(viewer);CHKERRQ(ierr);
      ierr = PetscViewerASCIISynchronizedPrintf(viewer,"  [%d] number of local blocks = %D\n",(int)rank,osm->n_local_true);CHKERRQ(ierr);
      ierr = PetscViewerFlush(viewer);CHKERRQ(ierr);
      ierr = PetscViewerASCIIPrintf(viewer,"  Local solve info for each block is in the following KSP and PC objects:\n");CHKERRQ(ierr);
      ierr = PetscViewerASCIIPushTab(viewer);CHKERRQ(ierr);
      ierr = PetscViewerASCIIPrintf(viewer,"- - - - - - - - - - - - - - - - - -\n");CHKERRQ(ierr);
      ierr = PetscViewerGetSubViewer(viewer,PETSC_COMM_SELF,&sviewer);CHKERRQ(ierr);
      for (i=0; i<osm->n_local_true; i++) {
        ierr = ISGetLocalSize(osm->is[i],&bsz);CHKERRQ(ierr);
        ierr = PetscViewerASCIISynchronizedPrintf(sviewer,"[%d] local block number %D, size = %D\n",(int)rank,i,bsz);CHKERRQ(ierr);
        ierr = KSPView(osm->ksp[i],sviewer);CHKERRQ(ierr);
        ierr = PetscViewerASCIISynchronizedPrintf(sviewer,"- - - - - - - - - - - - - - - - - -\n");CHKERRQ(ierr);
      }
      ierr = PetscViewerRestoreSubViewer(viewer,PETSC_COMM_SELF,&sviewer);CHKERRQ(ierr);
      ierr = PetscViewerASCIIPopTab(viewer);CHKERRQ(ierr);
      ierr = PetscViewerFlush(viewer);CHKERRQ(ierr);
      ierr = PetscViewerASCIIPopSynchronized(viewer);CHKERRQ(ierr);
    }
  } else if (isstring) {
    ierr = PetscViewerStringSPrintf(viewer," blocks=%D, overlap=%D, type=%s",osm->n,osm->overlap,PCASMTypes[osm->type]);CHKERRQ(ierr);
    ierr = PetscViewerGetSubViewer(viewer,PETSC_COMM_SELF,&sviewer);CHKERRQ(ierr);
    if (osm->ksp) {ierr = KSPView(osm->ksp[0],sviewer);CHKERRQ(ierr);}
    ierr = PetscViewerRestoreSubViewer(viewer,PETSC_COMM_SELF,&sviewer);CHKERRQ(ierr);
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
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MPI_Comm_size(PetscObjectComm((PetscObject)pc), &size);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(PetscObjectComm((PetscObject)pc), &rank);CHKERRQ(ierr);
  ierr = PCGetOptionsPrefix(pc,&prefix);CHKERRQ(ierr);
  ierr = PetscOptionsGetString(NULL,prefix,"-pc_asm_print_subdomains",fname,PETSC_MAX_PATH_LEN,NULL);CHKERRQ(ierr);
  if (fname[0] == 0) { ierr = PetscStrcpy(fname,"stdout");CHKERRQ(ierr); };
  ierr = PetscViewerASCIIOpen(PetscObjectComm((PetscObject)pc),fname,&viewer);CHKERRQ(ierr);
  for (i=0; i<osm->n_local; i++) {
    if (i < osm->n_local_true) {
      ierr = ISGetLocalSize(osm->is[i],&nidx);CHKERRQ(ierr);
      ierr = ISGetIndices(osm->is[i],&idx);CHKERRQ(ierr);
      /* Print to a string viewer; no more than 15 characters per index plus 512 char for the header.*/
#define len  16*(nidx+1)+512
      ierr = PetscMalloc1(len,&s);CHKERRQ(ierr);
      ierr = PetscViewerStringOpen(PETSC_COMM_SELF, s, len, &sviewer);CHKERRQ(ierr);
#undef len
      ierr = PetscViewerStringSPrintf(sviewer, "[%D:%D] Subdomain %D with overlap:\n", rank, size, i);CHKERRQ(ierr);
      for (j=0; j<nidx; j++) {
        ierr = PetscViewerStringSPrintf(sviewer,"%D ",idx[j]);CHKERRQ(ierr);
      }
      ierr = ISRestoreIndices(osm->is[i],&idx);CHKERRQ(ierr);
      ierr = PetscViewerStringSPrintf(sviewer,"\n");CHKERRQ(ierr);
      ierr = PetscViewerDestroy(&sviewer);CHKERRQ(ierr);
      ierr = PetscViewerASCIIPushSynchronized(viewer);CHKERRQ(ierr);
      ierr = PetscViewerASCIISynchronizedPrintf(viewer, s);CHKERRQ(ierr);
      ierr = PetscViewerFlush(viewer);CHKERRQ(ierr);
      ierr = PetscViewerASCIIPopSynchronized(viewer);CHKERRQ(ierr);
      ierr = PetscFree(s);CHKERRQ(ierr);
      if (osm->is_local) {
        /* Print to a string viewer; no more than 15 characters per index plus 512 char for the header.*/
#define len  16*(nidx+1)+512
        ierr = PetscMalloc1(len, &s);CHKERRQ(ierr);
        ierr = PetscViewerStringOpen(PETSC_COMM_SELF, s, len, &sviewer);CHKERRQ(ierr);
#undef len
        ierr = PetscViewerStringSPrintf(sviewer, "[%D:%D] Subdomain %D without overlap:\n", rank, size, i);CHKERRQ(ierr);
        ierr = ISGetLocalSize(osm->is_local[i],&nidx);CHKERRQ(ierr);
        ierr = ISGetIndices(osm->is_local[i],&idx);CHKERRQ(ierr);
        for (j=0; j<nidx; j++) {
          ierr = PetscViewerStringSPrintf(sviewer,"%D ",idx[j]);CHKERRQ(ierr);
        }
        ierr = ISRestoreIndices(osm->is_local[i],&idx);CHKERRQ(ierr);
        ierr = PetscViewerStringSPrintf(sviewer,"\n");CHKERRQ(ierr);
        ierr = PetscViewerDestroy(&sviewer);CHKERRQ(ierr);
        ierr = PetscViewerASCIIPushSynchronized(viewer);CHKERRQ(ierr);
        ierr = PetscViewerASCIISynchronizedPrintf(viewer, s);CHKERRQ(ierr);
        ierr = PetscViewerFlush(viewer);CHKERRQ(ierr);
        ierr = PetscViewerASCIIPopSynchronized(viewer);CHKERRQ(ierr);
        ierr = PetscFree(s);CHKERRQ(ierr);
      }
    } else {
      /* Participate in collective viewer calls. */
      ierr = PetscViewerASCIIPushSynchronized(viewer);CHKERRQ(ierr);
      ierr = PetscViewerFlush(viewer);CHKERRQ(ierr);
      ierr = PetscViewerASCIIPopSynchronized(viewer);CHKERRQ(ierr);
      /* Assume either all ranks have is_local or none do. */
      if (osm->is_local) {
        ierr = PetscViewerASCIIPushSynchronized(viewer);CHKERRQ(ierr);
        ierr = PetscViewerFlush(viewer);CHKERRQ(ierr);
        ierr = PetscViewerASCIIPopSynchronized(viewer);CHKERRQ(ierr);
      }
    }
  }
  ierr = PetscViewerFlush(viewer);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode PCSetUp_ASM(PC pc)
{
  PC_ASM         *osm = (PC_ASM*)pc->data;
  PetscErrorCode ierr;
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
        ierr = DMCreateDomainDecomposition(pc->dm, &num_domains, &domain_names, &inner_domain_is, &outer_domain_is, &domain_dm);CHKERRQ(ierr);
        osm->overlap = -1; /* We do not want to increase the overlap of the IS.
                              A future improvement of this code might allow one to use
                              DM-defined subdomains and also increase the overlap,
                              but that is not currently supported */
        if (num_domains) {
          ierr = PCASMSetLocalSubdomains(pc, num_domains, outer_domain_is, inner_domain_is);CHKERRQ(ierr);
        }
        for (d = 0; d < num_domains; ++d) {
          if (domain_names)    {ierr = PetscFree(domain_names[d]);CHKERRQ(ierr);}
          if (inner_domain_is) {ierr = ISDestroy(&inner_domain_is[d]);CHKERRQ(ierr);}
          if (outer_domain_is) {ierr = ISDestroy(&outer_domain_is[d]);CHKERRQ(ierr);}
        }
        ierr = PetscFree(domain_names);CHKERRQ(ierr);
        ierr = PetscFree(inner_domain_is);CHKERRQ(ierr);
        ierr = PetscFree(outer_domain_is);CHKERRQ(ierr);
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
      ierr         = MPIU_Allreduce(&inwork,&outwork,1,MPIU_2INT,MPIU_MAXSUM_OP,PetscObjectComm((PetscObject)pc));CHKERRQ(ierr);
      osm->n_local = outwork.max;
      osm->n       = outwork.sum;

      ierr = MPI_Comm_size(PetscObjectComm((PetscObject)pc),&size);CHKERRQ(ierr);
      if (outwork.max == 1 && outwork.sum == size) {
        /* osm->n_local_true = 1 on all processes, set this option may enable use of optimized MatCreateSubMatrices() implementation */
        ierr = MatSetOption(pc->pmat,MAT_SUBMAT_SINGLEIS,PETSC_TRUE);CHKERRQ(ierr);
      }
    }
    if (!osm->is) { /* create the index sets */
      ierr = PCASMCreateSubdomains(pc->pmat,osm->n_local_true,&osm->is);CHKERRQ(ierr);
    }
    if (osm->n_local_true > 1 && !osm->is_local) {
      ierr = PetscMalloc1(osm->n_local_true,&osm->is_local);CHKERRQ(ierr);
      for (i=0; i<osm->n_local_true; i++) {
        if (osm->overlap > 0) { /* With positive overlap, osm->is[i] will be modified */
          ierr = ISDuplicate(osm->is[i],&osm->is_local[i]);CHKERRQ(ierr);
          ierr = ISCopy(osm->is[i],osm->is_local[i]);CHKERRQ(ierr);
        } else {
          ierr             = PetscObjectReference((PetscObject)osm->is[i]);CHKERRQ(ierr);
          osm->is_local[i] = osm->is[i];
        }
      }
    }
    ierr = PCGetOptionsPrefix(pc,&prefix);CHKERRQ(ierr);
    flg  = PETSC_FALSE;
    ierr = PetscOptionsGetBool(NULL,prefix,"-pc_asm_print_subdomains",&flg,NULL);CHKERRQ(ierr);
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

    if (!osm->ksp) {
      /* Create the local solvers */
      ierr = PetscMalloc1(osm->n_local_true,&osm->ksp);CHKERRQ(ierr);
      if (domain_dm) {
        ierr = PetscInfo(pc,"Setting up ASM subproblems using the embedded DM\n");CHKERRQ(ierr);
      }
      for (i=0; i<osm->n_local_true; i++) {
        ierr = KSPCreate(PETSC_COMM_SELF,&ksp);CHKERRQ(ierr);
        ierr = KSPSetErrorIfNotConverged(ksp,pc->erroriffailure);CHKERRQ(ierr);
        ierr = PetscLogObjectParent((PetscObject)pc,(PetscObject)ksp);CHKERRQ(ierr);
        ierr = PetscObjectIncrementTabLevel((PetscObject)ksp,(PetscObject)pc,1);CHKERRQ(ierr);
        ierr = KSPSetType(ksp,KSPPREONLY);CHKERRQ(ierr);
        ierr = KSPGetPC(ksp,&subpc);CHKERRQ(ierr);
        ierr = PCGetOptionsPrefix(pc,&prefix);CHKERRQ(ierr);
        ierr = KSPSetOptionsPrefix(ksp,prefix);CHKERRQ(ierr);
        ierr = KSPAppendOptionsPrefix(ksp,"sub_");CHKERRQ(ierr);
        if (domain_dm) {
          ierr = KSPSetDM(ksp, domain_dm[i]);CHKERRQ(ierr);
          ierr = KSPSetDMActive(ksp, PETSC_FALSE);CHKERRQ(ierr);
          ierr = DMDestroy(&domain_dm[i]);CHKERRQ(ierr);
        }
        osm->ksp[i] = ksp;
      }
      if (domain_dm) {
        ierr = PetscFree(domain_dm);CHKERRQ(ierr);
      }
    }

    ierr = ISConcatenate(PETSC_COMM_SELF, osm->n_local_true, osm->is, &osm->lis);CHKERRQ(ierr);
    ierr = ISSortRemoveDups(osm->lis);CHKERRQ(ierr);
    ierr = ISGetLocalSize(osm->lis, &m);CHKERRQ(ierr);
    ierr = VecCreateSeq(PETSC_COMM_SELF, m, &osm->lx);CHKERRQ(ierr);
    ierr = VecDuplicate(osm->lx, &osm->ly);CHKERRQ(ierr);

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
  ierr = MatCreateSubMatrices(pc->pmat,osm->n_local_true,osm->is,osm->is,scall,&osm->pmat);CHKERRQ(ierr);
  if (scall == MAT_INITIAL_MATRIX) {
    ierr = PetscObjectGetOptionsPrefix((PetscObject)pc->pmat,&pprefix);CHKERRQ(ierr);
    for (i=0; i<osm->n_local_true; i++) {
      ierr = PetscLogObjectParent((PetscObject)pc,(PetscObject)osm->pmat[i]);CHKERRQ(ierr);
      ierr = PetscObjectSetOptionsPrefix((PetscObject)osm->pmat[i],pprefix);CHKERRQ(ierr);
    }
  }

  /* Convert the types of the submatrices (if needbe) */
  if (osm->sub_mat_type) {
    for (i=0; i<osm->n_local_true; i++) {
      ierr = MatConvert(osm->pmat[i],osm->sub_mat_type,MAT_INPLACE_MATRIX,&(osm->pmat[i]));CHKERRQ(ierr);
    }
  }

  if (!pc->setupcalled) {
    /* Create the local work vectors (from the local matrices) and scatter contexts */
    ierr = MatCreateVecs(pc->pmat,&vec,0);CHKERRQ(ierr);

    if (osm->is_local && (osm->type == PC_ASM_INTERPOLATE || osm->type == PC_ASM_NONE )) SETERRQ(PetscObjectComm((PetscObject)pc),PETSC_ERR_SUP,"Cannot use interpolate or none PCASMType if is_local was provided to PCASMSetLocalSubdomains()");
    if (osm->is_local && osm->type == PC_ASM_RESTRICT && osm->loctype == PC_COMPOSITE_ADDITIVE) {
      ierr = PetscMalloc1(osm->n_local_true,&osm->lprolongation);CHKERRQ(ierr);
    }
    ierr = PetscMalloc1(osm->n_local_true,&osm->lrestriction);CHKERRQ(ierr);
    ierr = PetscMalloc1(osm->n_local_true,&osm->x);CHKERRQ(ierr);
    ierr = PetscMalloc1(osm->n_local_true,&osm->y);CHKERRQ(ierr);

    ierr = ISGetLocalSize(osm->lis,&m);CHKERRQ(ierr);
    ierr = ISCreateStride(PETSC_COMM_SELF,m,0,1,&isl);CHKERRQ(ierr);
    ierr = VecScatterCreate(vec,osm->lis,osm->lx,isl,&osm->restriction);CHKERRQ(ierr);
    ierr = ISDestroy(&isl);CHKERRQ(ierr);


    for (i=0; i<osm->n_local_true; ++i) {
      ISLocalToGlobalMapping ltog;
      IS                     isll;
      const PetscInt         *idx_is;
      PetscInt               *idx_lis,nout;

      ierr = ISGetLocalSize(osm->is[i],&m);CHKERRQ(ierr);
      ierr = MatCreateVecs(osm->pmat[i],&osm->x[i],NULL);CHKERRQ(ierr);
      ierr = VecDuplicate(osm->x[i],&osm->y[i]);CHKERRQ(ierr);

      /* generate a scatter from ly to y[i] picking all the overlapping is[i] entries */
      ierr = ISLocalToGlobalMappingCreateIS(osm->lis,&ltog);CHKERRQ(ierr);
      ierr = ISGetLocalSize(osm->is[i],&m);CHKERRQ(ierr);
      ierr = ISGetIndices(osm->is[i], &idx_is);CHKERRQ(ierr);
      ierr = PetscMalloc1(m,&idx_lis);CHKERRQ(ierr);
      ierr = ISGlobalToLocalMappingApply(ltog,IS_GTOLM_DROP,m,idx_is,&nout,idx_lis);CHKERRQ(ierr);
      if (nout != m) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_PLIB,"is not a subset of lis");
      ierr = ISRestoreIndices(osm->is[i], &idx_is);CHKERRQ(ierr);
      ierr = ISCreateGeneral(PETSC_COMM_SELF,m,idx_lis,PETSC_OWN_POINTER,&isll);CHKERRQ(ierr);
      ierr = ISLocalToGlobalMappingDestroy(&ltog);CHKERRQ(ierr);
      ierr = ISCreateStride(PETSC_COMM_SELF,m,0,1,&isl);CHKERRQ(ierr);
      ierr = VecScatterCreate(osm->ly,isll,osm->y[i],isl,&osm->lrestriction[i]);CHKERRQ(ierr);
      ierr = ISDestroy(&isll);CHKERRQ(ierr);
      ierr = ISDestroy(&isl);CHKERRQ(ierr);
      if (osm->lprolongation) { /* generate a scatter from y[i] to ly picking only the the non-overlapping is_local[i] entries */
        ISLocalToGlobalMapping ltog;
        IS                     isll,isll_local;
        const PetscInt         *idx_local;
        PetscInt               *idx1, *idx2, nout;

        ierr = ISGetLocalSize(osm->is_local[i],&m_local);CHKERRQ(ierr);
        ierr = ISGetIndices(osm->is_local[i], &idx_local);CHKERRQ(ierr);

        ierr = ISLocalToGlobalMappingCreateIS(osm->is[i],&ltog);CHKERRQ(ierr);
        ierr = PetscMalloc1(m_local,&idx1);CHKERRQ(ierr);
        ierr = ISGlobalToLocalMappingApply(ltog,IS_GTOLM_DROP,m_local,idx_local,&nout,idx1);CHKERRQ(ierr);
        ierr = ISLocalToGlobalMappingDestroy(&ltog);CHKERRQ(ierr);
        if (nout != m_local) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_PLIB,"is_local not a subset of is");
        ierr = ISCreateGeneral(PETSC_COMM_SELF,m_local,idx1,PETSC_OWN_POINTER,&isll);CHKERRQ(ierr);

        ierr = ISLocalToGlobalMappingCreateIS(osm->lis,&ltog);CHKERRQ(ierr);
        ierr = PetscMalloc1(m_local,&idx2);CHKERRQ(ierr);
        ierr = ISGlobalToLocalMappingApply(ltog,IS_GTOLM_DROP,m_local,idx_local,&nout,idx2);CHKERRQ(ierr);
        ierr = ISLocalToGlobalMappingDestroy(&ltog);CHKERRQ(ierr);
        if (nout != m_local) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_PLIB,"is_local not a subset of lis");
        ierr = ISCreateGeneral(PETSC_COMM_SELF,m_local,idx2,PETSC_OWN_POINTER,&isll_local);CHKERRQ(ierr);

        ierr = ISRestoreIndices(osm->is_local[i], &idx_local);CHKERRQ(ierr);
        ierr = VecScatterCreate(osm->y[i],isll,osm->ly,isll_local,&osm->lprolongation[i]);CHKERRQ(ierr);

        ierr = ISDestroy(&isll);CHKERRQ(ierr);
        ierr = ISDestroy(&isll_local);CHKERRQ(ierr);
      }
    }
    ierr = VecDestroy(&vec);CHKERRQ(ierr);
  }

  if (osm->loctype == PC_COMPOSITE_MULTIPLICATIVE) {
    IS      *cis;
    PetscInt c;

    ierr = PetscMalloc1(osm->n_local_true, &cis);CHKERRQ(ierr);
    for (c = 0; c < osm->n_local_true; ++c) cis[c] = osm->lis;
    ierr = MatCreateSubMatrices(pc->pmat, osm->n_local_true, osm->is, cis, scall, &osm->lmats);CHKERRQ(ierr);
    ierr = PetscFree(cis);CHKERRQ(ierr);
  }

  /* Return control to the user so that the submatrices can be modified (e.g., to apply
     different boundary conditions for the submatrices than for the global problem) */
  ierr = PCModifySubMatrices(pc,osm->n_local_true,osm->is,osm->is,osm->pmat,pc->modifysubmatricesP);CHKERRQ(ierr);

  /*
     Loop over subdomains putting them into local ksp
  */
  for (i=0; i<osm->n_local_true; i++) {
    ierr = KSPSetOperators(osm->ksp[i],osm->pmat[i],osm->pmat[i]);CHKERRQ(ierr);
    if (!pc->setupcalled) {
      ierr = KSPSetFromOptions(osm->ksp[i]);CHKERRQ(ierr);
    }
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode PCSetUpOnBlocks_ASM(PC pc)
{
  PC_ASM             *osm = (PC_ASM*)pc->data;
  PetscErrorCode     ierr;
  PetscInt           i;
  KSPConvergedReason reason;

  PetscFunctionBegin;
  for (i=0; i<osm->n_local_true; i++) {
    ierr = KSPSetUp(osm->ksp[i]);CHKERRQ(ierr);
    ierr = KSPGetConvergedReason(osm->ksp[i],&reason);CHKERRQ(ierr);
    if (reason == KSP_DIVERGED_PC_FAILED) {
      pc->failedreason = PC_SUBPC_ERROR;
    }
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode PCApply_ASM(PC pc,Vec x,Vec y)
{
  PC_ASM         *osm = (PC_ASM*)pc->data;
  PetscErrorCode ierr;
  PetscInt       i,n_local_true = osm->n_local_true;
  ScatterMode    forward = SCATTER_FORWARD,reverse = SCATTER_REVERSE;

  PetscFunctionBegin;
  /*
     Support for limiting the restriction or interpolation to only local
     subdomain values (leaving the other values 0).
  */
  if (!(osm->type & PC_ASM_RESTRICT)) {
    forward = SCATTER_FORWARD_LOCAL;
    /* have to zero the work RHS since scatter may leave some slots empty */
    ierr = VecSet(osm->lx, 0.0);CHKERRQ(ierr);
  }
  if (!(osm->type & PC_ASM_INTERPOLATE)) {
    reverse = SCATTER_REVERSE_LOCAL;
  }

  if (osm->loctype == PC_COMPOSITE_MULTIPLICATIVE || osm->loctype == PC_COMPOSITE_ADDITIVE) {
    /* zero the global and the local solutions */
    ierr = VecSet(y, 0.0);CHKERRQ(ierr);
    ierr = VecSet(osm->ly, 0.0);CHKERRQ(ierr);

    /* Copy the global RHS to local RHS including the ghost nodes */
    ierr = VecScatterBegin(osm->restriction, x, osm->lx, INSERT_VALUES, forward);CHKERRQ(ierr);
    ierr = VecScatterEnd(osm->restriction, x, osm->lx, INSERT_VALUES, forward);CHKERRQ(ierr);

    /* Restrict local RHS to the overlapping 0-block RHS */
    ierr = VecScatterBegin(osm->lrestriction[0], osm->lx, osm->x[0], INSERT_VALUES, forward);CHKERRQ(ierr);
    ierr = VecScatterEnd(osm->lrestriction[0], osm->lx, osm->x[0], INSERT_VALUES, forward);CHKERRQ(ierr);

    /* do the local solves */
    for (i = 0; i < n_local_true; ++i) {

      /* solve the overlapping i-block */
      ierr = PetscLogEventBegin(PC_ApplyOnBlocks,osm->ksp[i],osm->x[i],osm->y[i],0);CHKERRQ(ierr);
      ierr = KSPSolve(osm->ksp[i], osm->x[i], osm->y[i]);CHKERRQ(ierr);
      ierr = KSPCheckSolve(osm->ksp[i],pc,osm->y[i]);CHKERRQ(ierr);
      ierr = PetscLogEventEnd(PC_ApplyOnBlocks,osm->ksp[i],osm->x[i],osm->y[i],0);CHKERRQ(ierr);

      if (osm->lprolongation) { /* interpolate the non-overlapping i-block solution to the local solution (only for restrictive additive) */
        ierr = VecScatterBegin(osm->lprolongation[i], osm->y[i], osm->ly, ADD_VALUES, forward);CHKERRQ(ierr);
        ierr = VecScatterEnd(osm->lprolongation[i], osm->y[i], osm->ly, ADD_VALUES, forward);CHKERRQ(ierr);
      }
      else { /* interpolate the overlapping i-block solution to the local solution */
        ierr = VecScatterBegin(osm->lrestriction[i], osm->y[i], osm->ly, ADD_VALUES, reverse);CHKERRQ(ierr);
        ierr = VecScatterEnd(osm->lrestriction[i], osm->y[i], osm->ly, ADD_VALUES, reverse);CHKERRQ(ierr);
      }

      if (i < n_local_true-1) {
        /* Restrict local RHS to the overlapping (i+1)-block RHS */
        ierr = VecScatterBegin(osm->lrestriction[i+1], osm->lx, osm->x[i+1], INSERT_VALUES, forward);CHKERRQ(ierr);
        ierr = VecScatterEnd(osm->lrestriction[i+1], osm->lx, osm->x[i+1], INSERT_VALUES, forward);CHKERRQ(ierr);

        if ( osm->loctype == PC_COMPOSITE_MULTIPLICATIVE){
          /* update the overlapping (i+1)-block RHS using the current local solution */
          ierr = MatMult(osm->lmats[i+1], osm->ly, osm->y[i+1]);CHKERRQ(ierr);
          ierr = VecAXPBY(osm->x[i+1],-1.,1., osm->y[i+1]); CHKERRQ(ierr);
        }
      }
    }
    /* Add the local solution to the global solution including the ghost nodes */
    ierr = VecScatterBegin(osm->restriction, osm->ly, y, ADD_VALUES, reverse);CHKERRQ(ierr);
    ierr = VecScatterEnd(osm->restriction, osm->ly, y, ADD_VALUES, reverse);CHKERRQ(ierr);
  } else {
    SETERRQ1(PetscObjectComm((PetscObject)pc), PETSC_ERR_ARG_WRONG, "Invalid local composition type: %s", PCCompositeTypes[osm->loctype]);
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode PCApplyTranspose_ASM(PC pc,Vec x,Vec y)
{
  PC_ASM         *osm = (PC_ASM*)pc->data;
  PetscErrorCode ierr;
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
    ierr = VecSet(osm->lx, 0.0);CHKERRQ(ierr);
  }
  if (!(osm->type & PC_ASM_RESTRICT)) reverse = SCATTER_REVERSE_LOCAL;

  /* zero the global and the local solutions */
  ierr = VecSet(y, 0.0);CHKERRQ(ierr);
  ierr = VecSet(osm->ly, 0.0);CHKERRQ(ierr);

  /* Copy the global RHS to local RHS including the ghost nodes */
  ierr = VecScatterBegin(osm->restriction, x, osm->lx, INSERT_VALUES, forward);CHKERRQ(ierr);
  ierr = VecScatterEnd(osm->restriction, x, osm->lx, INSERT_VALUES, forward);CHKERRQ(ierr);

  /* Restrict local RHS to the overlapping 0-block RHS */
  ierr = VecScatterBegin(osm->lrestriction[0], osm->lx, osm->x[0], INSERT_VALUES, forward);CHKERRQ(ierr);
  ierr = VecScatterEnd(osm->lrestriction[0], osm->lx, osm->x[0], INSERT_VALUES, forward);CHKERRQ(ierr);

  /* do the local solves */
  for (i = 0; i < n_local_true; ++i) {

    /* solve the overlapping i-block */
    ierr = PetscLogEventBegin(PC_ApplyOnBlocks,osm->ksp[i],osm->x[i],osm->y[i],0);CHKERRQ(ierr);
    ierr = KSPSolveTranspose(osm->ksp[i], osm->x[i], osm->y[i]);CHKERRQ(ierr);
    ierr = KSPCheckSolve(osm->ksp[i],pc,osm->y[i]);CHKERRQ(ierr);
    ierr = PetscLogEventEnd(PC_ApplyOnBlocks,osm->ksp[i],osm->x[i],osm->y[i],0);CHKERRQ(ierr);

    if (osm->lprolongation) { /* interpolate the non-overlapping i-block solution to the local solution */
      ierr = VecScatterBegin(osm->lprolongation[i], osm->y[i], osm->ly, ADD_VALUES, forward);CHKERRQ(ierr);
      ierr = VecScatterEnd(osm->lprolongation[i], osm->y[i], osm->ly, ADD_VALUES, forward);CHKERRQ(ierr);
    } else { /* interpolate the overlapping i-block solution to the local solution */
      ierr = VecScatterBegin(osm->lrestriction[i], osm->y[i], osm->ly, ADD_VALUES, reverse);CHKERRQ(ierr);
      ierr = VecScatterEnd(osm->lrestriction[i], osm->y[i], osm->ly, ADD_VALUES, reverse);CHKERRQ(ierr);
    }

    if (i < n_local_true-1) {
      /* Restrict local RHS to the overlapping (i+1)-block RHS */
      ierr = VecScatterBegin(osm->lrestriction[i+1], osm->lx, osm->x[i+1], INSERT_VALUES, forward);CHKERRQ(ierr);
      ierr = VecScatterEnd(osm->lrestriction[i+1], osm->lx, osm->x[i+1], INSERT_VALUES, forward);CHKERRQ(ierr);
    }
  }
  /* Add the local solution to the global solution including the ghost nodes */
  ierr = VecScatterBegin(osm->restriction, osm->ly, y, ADD_VALUES, reverse);CHKERRQ(ierr);
  ierr = VecScatterEnd(osm->restriction, osm->ly, y, ADD_VALUES, reverse);CHKERRQ(ierr);

  PetscFunctionReturn(0);

}

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
      ierr = MatDestroySubMatrices(osm->n_local_true,&osm->pmat);CHKERRQ(ierr);
    }
  }
  if (osm->lrestriction) {
    ierr = VecScatterDestroy(&osm->restriction);CHKERRQ(ierr);
    for (i=0; i<osm->n_local_true; i++) {
      ierr = VecScatterDestroy(&osm->lrestriction[i]);CHKERRQ(ierr);
      if (osm->lprolongation) {ierr = VecScatterDestroy(&osm->lprolongation[i]);CHKERRQ(ierr);}
      ierr = VecDestroy(&osm->x[i]);CHKERRQ(ierr);
      ierr = VecDestroy(&osm->y[i]);CHKERRQ(ierr);
    }
    ierr = PetscFree(osm->lrestriction);CHKERRQ(ierr);
    if (osm->lprolongation) {ierr = PetscFree(osm->lprolongation);CHKERRQ(ierr);}
    ierr = PetscFree(osm->x);CHKERRQ(ierr);
    ierr = PetscFree(osm->y);CHKERRQ(ierr);

  }
  ierr = PCASMDestroySubdomains(osm->n_local_true,osm->is,osm->is_local);CHKERRQ(ierr);
  ierr = ISDestroy(&osm->lis);CHKERRQ(ierr);
  ierr = VecDestroy(&osm->lx);CHKERRQ(ierr);
  ierr = VecDestroy(&osm->ly);CHKERRQ(ierr);
  if (osm->loctype == PC_COMPOSITE_MULTIPLICATIVE) {
    ierr = MatDestroyMatrices(osm->n_local_true, &osm->lmats);CHKERRQ(ierr);
  }

  ierr = PetscFree(osm->sub_mat_type);CHKERRQ(ierr);

  osm->is       = 0;
  osm->is_local = 0;
  PetscFunctionReturn(0);
}

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

static PetscErrorCode PCSetFromOptions_ASM(PetscOptionItems *PetscOptionsObject,PC pc)
{
  PC_ASM         *osm = (PC_ASM*)pc->data;
  PetscErrorCode ierr;
  PetscInt       blocks,ovl;
  PetscBool      flg;
  PCASMType      asmtype;
  PCCompositeType loctype;
  char           sub_mat_type[256];

  PetscFunctionBegin;
  ierr = PetscOptionsHead(PetscOptionsObject,"Additive Schwarz options");CHKERRQ(ierr);
  ierr = PetscOptionsBool("-pc_asm_dm_subdomains","Use DMCreateDomainDecomposition() to define subdomains","PCASMSetDMSubdomains",osm->dm_subdomains,&osm->dm_subdomains,&flg);CHKERRQ(ierr);
  ierr = PetscOptionsInt("-pc_asm_blocks","Number of subdomains","PCASMSetTotalSubdomains",osm->n,&blocks,&flg);CHKERRQ(ierr);
  if (flg) {
    ierr = PCASMSetTotalSubdomains(pc,blocks,NULL,NULL);CHKERRQ(ierr);
    osm->dm_subdomains = PETSC_FALSE;
  }
  ierr = PetscOptionsInt("-pc_asm_local_blocks","Number of local subdomains","PCASMSetLocalSubdomains",osm->n_local_true,&blocks,&flg);CHKERRQ(ierr);
  if (flg) {
    ierr = PCASMSetLocalSubdomains(pc,blocks,NULL,NULL);CHKERRQ(ierr);
    osm->dm_subdomains = PETSC_FALSE;
  }
  ierr = PetscOptionsInt("-pc_asm_overlap","Number of grid points overlap","PCASMSetOverlap",osm->overlap,&ovl,&flg);CHKERRQ(ierr);
  if (flg) {
    ierr = PCASMSetOverlap(pc,ovl);CHKERRQ(ierr);
    osm->dm_subdomains = PETSC_FALSE;
  }
  flg  = PETSC_FALSE;
  ierr = PetscOptionsEnum("-pc_asm_type","Type of restriction/extension","PCASMSetType",PCASMTypes,(PetscEnum)osm->type,(PetscEnum*)&asmtype,&flg);CHKERRQ(ierr);
  if (flg) {ierr = PCASMSetType(pc,asmtype);CHKERRQ(ierr); }
  flg  = PETSC_FALSE;
  ierr = PetscOptionsEnum("-pc_asm_local_type","Type of local solver composition","PCASMSetLocalType",PCCompositeTypes,(PetscEnum)osm->loctype,(PetscEnum*)&loctype,&flg);CHKERRQ(ierr);
  if (flg) {ierr = PCASMSetLocalType(pc,loctype);CHKERRQ(ierr); }
  ierr = PetscOptionsFList("-pc_asm_sub_mat_type","Subsolve Matrix Type","PCASMSetSubMatType",MatList,NULL,sub_mat_type,256,&flg);CHKERRQ(ierr);
  if (flg) {
    ierr = PCASMSetSubMatType(pc,sub_mat_type);CHKERRQ(ierr);
  }
  ierr = PetscOptionsTail();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*------------------------------------------------------------------------------------*/

static PetscErrorCode  PCASMSetLocalSubdomains_ASM(PC pc,PetscInt n,IS is[],IS is_local[])
{
  PC_ASM         *osm = (PC_ASM*)pc->data;
  PetscErrorCode ierr;
  PetscInt       i;

  PetscFunctionBegin;
  if (n < 1) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Each process must have 1 or more blocks, n = %D",n);
  if (pc->setupcalled && (n != osm->n_local_true || is)) SETERRQ(PetscObjectComm((PetscObject)pc),PETSC_ERR_ARG_WRONGSTATE,"PCASMSetLocalSubdomains() should be called before calling PCSetUp().");

  if (!pc->setupcalled) {
    if (is) {
      for (i=0; i<n; i++) {ierr = PetscObjectReference((PetscObject)is[i]);CHKERRQ(ierr);}
    }
    if (is_local) {
      for (i=0; i<n; i++) {ierr = PetscObjectReference((PetscObject)is_local[i]);CHKERRQ(ierr);}
    }
    ierr = PCASMDestroySubdomains(osm->n_local_true,osm->is,osm->is_local);CHKERRQ(ierr);

    osm->n_local_true = n;
    osm->is           = 0;
    osm->is_local     = 0;
    if (is) {
      ierr = PetscMalloc1(n,&osm->is);CHKERRQ(ierr);
      for (i=0; i<n; i++) osm->is[i] = is[i];
      /* Flag indicating that the user has set overlapping subdomains so PCASM should not increase their size. */
      osm->overlap = -1;
    }
    if (is_local) {
      ierr = PetscMalloc1(n,&osm->is_local);CHKERRQ(ierr);
      for (i=0; i<n; i++) osm->is_local[i] = is_local[i];
      if (!is) {
        ierr = PetscMalloc1(osm->n_local_true,&osm->is);CHKERRQ(ierr);
        for (i=0; i<osm->n_local_true; i++) {
          if (osm->overlap > 0) { /* With positive overlap, osm->is[i] will be modified */
            ierr = ISDuplicate(osm->is_local[i],&osm->is[i]);CHKERRQ(ierr);
            ierr = ISCopy(osm->is_local[i],osm->is[i]);CHKERRQ(ierr);
          } else {
            ierr = PetscObjectReference((PetscObject)osm->is_local[i]);CHKERRQ(ierr);
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
  PetscErrorCode ierr;
  PetscMPIInt    rank,size;
  PetscInt       n;

  PetscFunctionBegin;
  if (N < 1) SETERRQ1(PetscObjectComm((PetscObject)pc),PETSC_ERR_ARG_OUTOFRANGE,"Number of total blocks must be > 0, N = %D",N);
  if (is || is_local) SETERRQ(PetscObjectComm((PetscObject)pc),PETSC_ERR_SUP,"Use PCASMSetLocalSubdomains() to set specific index sets\n\they cannot be set globally yet.");

  /*
     Split the subdomains equally among all processors
  */
  ierr = MPI_Comm_rank(PetscObjectComm((PetscObject)pc),&rank);CHKERRQ(ierr);
  ierr = MPI_Comm_size(PetscObjectComm((PetscObject)pc),&size);CHKERRQ(ierr);
  n    = N/size + ((N % size) > rank);
  if (!n) SETERRQ3(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Process %d must have at least one block: total processors %d total blocks %D",(int)rank,(int)size,N);
  if (pc->setupcalled && n != osm->n_local_true) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"PCASMSetTotalSubdomains() should be called before PCSetUp().");
  if (!pc->setupcalled) {
    ierr = PCASMDestroySubdomains(osm->n_local_true,osm->is,osm->is_local);CHKERRQ(ierr);

    osm->n_local_true = n;
    osm->is           = 0;
    osm->is_local     = 0;
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode  PCASMSetOverlap_ASM(PC pc,PetscInt ovl)
{
  PC_ASM *osm = (PC_ASM*)pc->data;

  PetscFunctionBegin;
  if (ovl < 0) SETERRQ(PetscObjectComm((PetscObject)pc),PETSC_ERR_ARG_OUTOFRANGE,"Negative overlap value requested");
  if (pc->setupcalled && ovl != osm->overlap) SETERRQ(PetscObjectComm((PetscObject)pc),PETSC_ERR_ARG_WRONGSTATE,"PCASMSetOverlap() should be called before PCSetUp().");
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
  if (type != PC_COMPOSITE_ADDITIVE && type != PC_COMPOSITE_MULTIPLICATIVE) SETERRQ(PetscObjectComm((PetscObject)pc),PETSC_ERR_SUP,"Only supports additive or multiplicative as the local type");
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
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (osm->n_local_true < 1) SETERRQ(PetscObjectComm((PetscObject)pc),PETSC_ERR_ORDER,"Need to call PCSetUp() on PC (or KSPSetUp() on the outer KSP object) before calling here");

  if (n_local) *n_local = osm->n_local_true;
  if (first_local) {
    ierr          = MPI_Scan(&osm->n_local_true,first_local,1,MPIU_INT,MPI_SUM,PetscObjectComm((PetscObject)pc));CHKERRQ(ierr);
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
  PetscErrorCode    ierr;
  PC_ASM            *osm = (PC_ASM*)pc->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_CLASSID,1);
  ierr = PetscFree(osm->sub_mat_type);CHKERRQ(ierr);
  ierr = PetscStrallocpy(sub_mat_type,(char**)&osm->sub_mat_type);CHKERRQ(ierr);
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

.seealso: PCASMSetTotalSubdomains(), PCASMSetOverlap(), PCASMGetSubKSP(),
          PCASMCreateSubdomains2D(), PCASMGetLocalSubdomains(), PCASMType, PCASMSetType()
@*/
PetscErrorCode  PCASMSetLocalSubdomains(PC pc,PetscInt n,IS is[],IS is_local[])
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_CLASSID,1);
  ierr = PetscTryMethod(pc,"PCASMSetLocalSubdomains_C",(PC,PetscInt,IS[],IS[]),(pc,n,is,is_local));CHKERRQ(ierr);
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

.seealso: PCASMSetTotalSubdomains(), PCASMSetLocalSubdomains(), PCASMGetSubKSP(),
          PCASMCreateSubdomains2D(), PCASMGetLocalSubdomains(), MatIncreaseOverlap()
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

/*@
    PCASMSetType - Sets the type of restriction and interpolation used
    for local problems in the additive Schwarz method.

    Logically Collective on pc

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

    Notes:
    if the is_local arguments are passed to PCASMSetLocalSubdomains() then they are used when PC_ASM_RESTRICT has been selected
    to limit the local processor interpolation

    Level: intermediate

.seealso: PCASMSetTotalSubdomains(), PCASMSetTotalSubdomains(), PCASMGetSubKSP(),
          PCASMCreateSubdomains2D(), PCASMType, PCASMSetLocalType(), PCASMGetLocalType()
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

.seealso: PCASMSetTotalSubdomains(), PCASMSetTotalSubdomains(), PCASMGetSubKSP(),
          PCASMCreateSubdomains2D(), PCASMType, PCASMSetType(), PCASMSetLocalType(), PCASMGetLocalType()
@*/
PetscErrorCode  PCASMGetType(PC pc,PCASMType *type)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_CLASSID,1);
  ierr = PetscUseMethod(pc,"PCASMGetType_C",(PC,PCASMType*),(pc,type));CHKERRQ(ierr);
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

.seealso: PCASMSetType(), PCASMGetType(), PCASMGetLocalType(), PCASM, PCASMType, PCASMSetType(), PCASMGetType(), PCCompositeType
@*/
PetscErrorCode PCASMSetLocalType(PC pc, PCCompositeType type)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc, PC_CLASSID, 1);
  PetscValidLogicalCollectiveEnum(pc, type, 2);
  ierr = PetscTryMethod(pc, "PCASMSetLocalType_C", (PC, PCCompositeType), (pc, type));CHKERRQ(ierr);
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

.seealso: PCASMSetType(), PCASMGetType(), PCASMSetLocalType(), PCASMCreate(), PCASMType, PCASMSetType(), PCASMGetType(), PCCompositeType
@*/
PetscErrorCode PCASMGetLocalType(PC pc, PCCompositeType *type)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc, PC_CLASSID, 1);
  PetscValidPointer(type, 2);
  ierr = PetscUseMethod(pc, "PCASMGetLocalType_C", (PC, PCCompositeType *), (pc, type));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@
    PCASMSetSortIndices - Determines whether subdomain indices are sorted.

    Logically Collective on pc

    Input Parameters:
+   pc  - the preconditioner context
-   doSort - sort the subdomain indices

    Level: intermediate

.seealso: PCASMSetLocalSubdomains(), PCASMSetTotalSubdomains(), PCASMGetSubKSP(),
          PCASMCreateSubdomains2D()
@*/
PetscErrorCode  PCASMSetSortIndices(PC pc,PetscBool doSort)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_CLASSID,1);
  PetscValidLogicalCollectiveBool(pc,doSort,2);
  ierr = PetscTryMethod(pc,"PCASMSetSortIndices_C",(PC,PetscBool),(pc,doSort));CHKERRQ(ierr);
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
+   1. - M Dryja, OB Widlund, An additive variant of the Schwarz alternating method for the case of many subregions
     Courant Institute, New York University Technical report
-   2. - Barry Smith, Petter Bjorstad, and William Gropp, Domain Decompositions: Parallel Multilevel Methods for Elliptic Partial Differential Equations,
    Cambridge University Press.

.seealso:  PCCreate(), PCSetType(), PCType (for list of available types), PC,
           PCBJACOBI, PCASMGetSubKSP(), PCASMSetLocalSubdomains(), PCASMType, PCASMGetType(), PCASMSetLocalType(), PCASMGetLocalType()
           PCASMSetTotalSubdomains(), PCSetModifySubMatrices(), PCASMSetOverlap(), PCASMSetType(), PCCompositeType

M*/

PETSC_EXTERN PetscErrorCode PCCreate_ASM(PC pc)
{
  PetscErrorCode ierr;
  PC_ASM         *osm;

  PetscFunctionBegin;
  ierr = PetscNewLog(pc,&osm);CHKERRQ(ierr);

  osm->n                 = PETSC_DECIDE;
  osm->n_local           = 0;
  osm->n_local_true      = PETSC_DECIDE;
  osm->overlap           = 1;
  osm->ksp               = 0;
  osm->restriction       = 0;
  osm->lprolongation     = 0;
  osm->lrestriction      = 0;
  osm->x                 = 0;
  osm->y                 = 0;
  osm->is                = 0;
  osm->is_local          = 0;
  osm->mat               = 0;
  osm->pmat              = 0;
  osm->type              = PC_ASM_RESTRICT;
  osm->loctype           = PC_COMPOSITE_ADDITIVE;
  osm->same_local_solves = PETSC_TRUE;
  osm->sort_indices      = PETSC_TRUE;
  osm->dm_subdomains     = PETSC_FALSE;
  osm->sub_mat_type      = NULL;

  pc->data                 = (void*)osm;
  pc->ops->apply           = PCApply_ASM;
  pc->ops->applytranspose  = PCApplyTranspose_ASM;
  pc->ops->setup           = PCSetUp_ASM;
  pc->ops->reset           = PCReset_ASM;
  pc->ops->destroy         = PCDestroy_ASM;
  pc->ops->setfromoptions  = PCSetFromOptions_ASM;
  pc->ops->setuponblocks   = PCSetUpOnBlocks_ASM;
  pc->ops->view            = PCView_ASM;
  pc->ops->applyrichardson = 0;

  ierr = PetscObjectComposeFunction((PetscObject)pc,"PCASMSetLocalSubdomains_C",PCASMSetLocalSubdomains_ASM);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)pc,"PCASMSetTotalSubdomains_C",PCASMSetTotalSubdomains_ASM);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)pc,"PCASMSetOverlap_C",PCASMSetOverlap_ASM);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)pc,"PCASMSetType_C",PCASMSetType_ASM);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)pc,"PCASMGetType_C",PCASMGetType_ASM);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)pc,"PCASMSetLocalType_C",PCASMSetLocalType_ASM);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)pc,"PCASMGetLocalType_C",PCASMGetLocalType_ASM);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)pc,"PCASMSetSortIndices_C",PCASMSetSortIndices_ASM);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)pc,"PCASMGetSubKSP_C",PCASMGetSubKSP_ASM);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)pc,"PCASMGetSubMatType_C",PCASMGetSubMatType_ASM);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)pc,"PCASMSetSubMatType_C",PCASMSetSubMatType_ASM);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

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

.seealso: PCASMSetLocalSubdomains(), PCASMDestroySubdomains()
@*/
PetscErrorCode  PCASMCreateSubdomains(Mat A, PetscInt n, IS* outis[])
{
  MatPartitioning mpart;
  const char      *prefix;
  PetscInt        i,j,rstart,rend,bs;
  PetscBool       hasop, isbaij = PETSC_FALSE,foundpart = PETSC_FALSE;
  Mat             Ad     = NULL, adj;
  IS              ispart,isnumb,*is;
  PetscErrorCode  ierr;

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
  ierr = MatHasOperation(A,MATOP_GET_DIAGONAL_BLOCK,&hasop);CHKERRQ(ierr);
  if (hasop) {
    ierr = MatGetDiagonalBlock(A,&Ad);CHKERRQ(ierr);
  }
  if (Ad) {
    ierr = PetscObjectBaseTypeCompare((PetscObject)Ad,MATSEQBAIJ,&isbaij);CHKERRQ(ierr);
    if (!isbaij) {ierr = PetscObjectBaseTypeCompare((PetscObject)Ad,MATSEQSBAIJ,&isbaij);CHKERRQ(ierr);}
  }
  if (Ad && n > 1) {
    PetscBool match,done;
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
            if (row[j] != i) { /* if not diagonal */
              jja[nnz+cnt++] = row[j];
            }
          }
          nnz     += cnt;
          iia[i+1] = nnz;
        }
        /* Partitioning of the adjacency matrix */
        ierr      = MatCreateMPIAdj(PETSC_COMM_SELF,na,na,iia,jja,NULL,&adj);CHKERRQ(ierr);
        ierr      = MatPartitioningSetAdjacency(mpart,adj);CHKERRQ(ierr);
        ierr      = MatPartitioningSetNParts(mpart,n);CHKERRQ(ierr);
        ierr      = MatPartitioningApply(mpart,&ispart);CHKERRQ(ierr);
        ierr      = ISPartitioningToNumbering(ispart,&isnumb);CHKERRQ(ierr);
        ierr      = MatDestroy(&adj);CHKERRQ(ierr);
        foundpart = PETSC_TRUE;
      }
      ierr = MatRestoreRowIJ(Ad,0,PETSC_TRUE,isbaij,&na,&ia,&ja,&done);CHKERRQ(ierr);
    }
    ierr = MatPartitioningDestroy(&mpart);CHKERRQ(ierr);
  }

  ierr   = PetscMalloc1(n,&is);CHKERRQ(ierr);
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
    ierr = PetscMalloc1(n,&count);CHKERRQ(ierr);
    ierr = ISPartitioningCount(ispart,n,count);CHKERRQ(ierr);
    if (isbaij && bs > 1) { /* adjust for the block-aij case */
      for (i=0; i<n; i++) count[i] *= bs;
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
    for (i=0; i<n; i++) {
      ierr   = ISCreateGeneral(PETSC_COMM_SELF,count[i],&indices[start],PETSC_COPY_VALUES,&is[i]);CHKERRQ(ierr);
      ierr   = ISSort(is[i]);CHKERRQ(ierr);
      start += count[i];
    }

    ierr = PetscFree(count);CHKERRQ(ierr);
    ierr = PetscFree(indices);CHKERRQ(ierr);
    ierr = ISDestroy(&isnumb);CHKERRQ(ierr);
    ierr = ISDestroy(&ispart);CHKERRQ(ierr);

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

.seealso: PCASMCreateSubdomains(), PCASMSetLocalSubdomains()
@*/
PetscErrorCode  PCASMDestroySubdomains(PetscInt n, IS is[], IS is_local[])
{
  PetscInt       i;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (n <= 0) PetscFunctionReturn(0);
  if (is) {
    PetscValidPointer(is,2);
    for (i=0; i<n; i++) { ierr = ISDestroy(&is[i]);CHKERRQ(ierr); }
    ierr = PetscFree(is);CHKERRQ(ierr);
  }
  if (is_local) {
    PetscValidPointer(is_local,3);
    for (i=0; i<n; i++) { ierr = ISDestroy(&is_local[i]);CHKERRQ(ierr); }
    ierr = PetscFree(is_local);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

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

  *Nsub     = N*M;
  ierr      = PetscMalloc1(*Nsub,is);CHKERRQ(ierr);
  ierr      = PetscMalloc1(*Nsub,is_local);CHKERRQ(ierr);
  ystart    = 0;
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
      ierr   = PetscMalloc1(nidx,&idx);CHKERRQ(ierr);
      loc    = 0;
      for (ii=yleft; ii<yright; ii++) {
        count = m*ii + xleft;
        for (jj=xleft; jj<xright; jj++) idx[loc++] = count++;
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
      ierr    = PetscFree(idx);CHKERRQ(ierr);
      xstart += width;
      loc_outer++;
    }
    ystart += height;
  }
  for (i=0; i<*Nsub; i++) { ierr = ISSort((*is)[i]);CHKERRQ(ierr); }
  PetscFunctionReturn(0);
}

/*@C
    PCASMGetLocalSubdomains - Gets the local subdomains (for this processor
    only) for the additive Schwarz preconditioner.

    Not Collective

    Input Parameter:
.   pc - the preconditioner context

    Output Parameters:
+   n - the number of subdomains for this processor (default value = 1)
.   is - the index sets that define the subdomains for this processor
-   is_local - the index sets that define the local part of the subdomains for this processor (can be NULL)


    Notes:
    The IS numbering is in the parallel, global numbering of the vector.

    Level: advanced

.seealso: PCASMSetTotalSubdomains(), PCASMSetOverlap(), PCASMGetSubKSP(),
          PCASMCreateSubdomains2D(), PCASMSetLocalSubdomains(), PCASMGetLocalSubmatrices()
@*/
PetscErrorCode  PCASMGetLocalSubdomains(PC pc,PetscInt *n,IS *is[],IS *is_local[])
{
  PC_ASM         *osm = (PC_ASM*)pc->data;
  PetscErrorCode ierr;
  PetscBool      match;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_CLASSID,1);
  PetscValidIntPointer(n,2);
  if (is) PetscValidPointer(is,3);
  ierr = PetscObjectTypeCompare((PetscObject)pc,PCASM,&match);CHKERRQ(ierr);
  if (!match) SETERRQ(PetscObjectComm((PetscObject)pc),PETSC_ERR_ARG_WRONG,"PC is not a PCASM");
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
+   n - the number of matrices for this processor (default value = 1)
-   mat - the matrices

    Level: advanced

    Notes:
    Call after PCSetUp() (or KSPSetUp()) but before PCApply() and before PCSetUpOnBlocks())

           Usually one would use PCSetModifySubMatrices() to change the submatrices in building the preconditioner.

.seealso: PCASMSetTotalSubdomains(), PCASMSetOverlap(), PCASMGetSubKSP(),
          PCASMCreateSubdomains2D(), PCASMSetLocalSubdomains(), PCASMGetLocalSubdomains(), PCSetModifySubMatrices()
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
  if (!pc->setupcalled) SETERRQ(PetscObjectComm((PetscObject)pc),PETSC_ERR_ARG_WRONGSTATE,"Must call after KSPSetUp() or PCSetUp().");
  ierr = PetscObjectTypeCompare((PetscObject)pc,PCASM,&match);CHKERRQ(ierr);
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

    Input Parameter:
+   pc  - the preconditioner
-   flg - boolean indicating whether to use subdomains defined by the DM

    Options Database Key:
.   -pc_asm_dm_subdomains

    Level: intermediate

    Notes:
    PCASMSetTotalSubdomains() and PCASMSetOverlap() take precedence over PCASMSetDMSubdomains(),
    so setting either of the first two effectively turns the latter off.

.seealso: PCASMGetDMSubdomains(), PCASMSetTotalSubdomains(), PCASMSetOverlap()
          PCASMCreateSubdomains2D(), PCASMSetLocalSubdomains(), PCASMGetLocalSubdomains()
@*/
PetscErrorCode  PCASMSetDMSubdomains(PC pc,PetscBool flg)
{
  PC_ASM         *osm = (PC_ASM*)pc->data;
  PetscErrorCode ierr;
  PetscBool      match;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_CLASSID,1);
  PetscValidLogicalCollectiveBool(pc,flg,2);
  if (pc->setupcalled) SETERRQ(((PetscObject)pc)->comm,PETSC_ERR_ARG_WRONGSTATE,"Not for a setup PC.");
  ierr = PetscObjectTypeCompare((PetscObject)pc,PCASM,&match);CHKERRQ(ierr);
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

.seealso: PCASMSetDMSubdomains(), PCASMSetTotalSubdomains(), PCASMSetOverlap()
          PCASMCreateSubdomains2D(), PCASMSetLocalSubdomains(), PCASMGetLocalSubdomains()
@*/
PetscErrorCode  PCASMGetDMSubdomains(PC pc,PetscBool* flg)
{
  PC_ASM         *osm = (PC_ASM*)pc->data;
  PetscErrorCode ierr;
  PetscBool      match;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_CLASSID,1);
  PetscValidBoolPointer(flg,2);
  ierr = PetscObjectTypeCompare((PetscObject)pc,PCASM,&match);CHKERRQ(ierr);
  if (match) {
    if (flg) *flg = osm->dm_subdomains;
  }
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

.seealso: PCASMSetSubMatType(), PCASM, PCSetType(), VecSetType(), MatType, Mat
@*/
PetscErrorCode  PCASMGetSubMatType(PC pc,MatType *sub_mat_type){
  PetscErrorCode ierr;

  ierr = PetscTryMethod(pc,"PCASMGetSubMatType_C",(PC,MatType*),(pc,sub_mat_type));CHKERRQ(ierr);
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

.seealso: PCASMGetSubMatType(), PCASM, PCSetType(), VecSetType(), MatType, Mat
@*/
PetscErrorCode PCASMSetSubMatType(PC pc,MatType sub_mat_type)
{
  PetscErrorCode ierr;

  ierr = PetscTryMethod(pc,"PCASMSetSubMatType_C",(PC,MatType),(pc,sub_mat_type));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
