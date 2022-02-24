#include <petsc/private/snesimpl.h> /*I "petscsnes.h" I*/
#include <petscdmcomposite.h>

typedef struct _BlockDesc *BlockDesc;
struct _BlockDesc {
  char       *name;     /* Block name */
  PetscInt   nfields;   /* If block is defined on a DA, the number of DA fields */
  PetscInt   *fields;   /* If block is defined on a DA, the list of DA fields */
  IS         is;        /* Index sets defining the block */
  VecScatter sctx;      /* Scatter mapping global Vec to blockVec */
  SNES       snes;      /* Solver for this block */
  Vec        x;
  BlockDesc  next, previous;
};

typedef struct {
  PetscBool       issetup;       /* Flag is true after the all ISs and operators have been defined */
  PetscBool       defined;       /* Flag is true after the blocks have been defined, to prevent more blocks from being added */
  PetscBool       defaultblocks; /* Flag is true for a system with a set of 'k' scalar fields with the same layout (and bs = k) */
  PetscInt        numBlocks;     /* Number of blocks (can be fields, domains, etc.) */
  PetscInt        bs;            /* Block size for IS, Vec and Mat structures */
  PCCompositeType type;          /* Solver combination method (additive, multiplicative, etc.) */
  BlockDesc       blocks;        /* Linked list of block descriptors */
} SNES_Multiblock;

PetscErrorCode SNESReset_Multiblock(SNES snes)
{
  SNES_Multiblock *mb    = (SNES_Multiblock*) snes->data;
  BlockDesc       blocks = mb->blocks, next;

  PetscFunctionBegin;
  while (blocks) {
    CHKERRQ(SNESReset(blocks->snes));
#if 0
    CHKERRQ(VecDestroy(&blocks->x));
#endif
    CHKERRQ(VecScatterDestroy(&blocks->sctx));
    CHKERRQ(ISDestroy(&blocks->is));
    next   = blocks->next;
    blocks = next;
  }
  PetscFunctionReturn(0);
}

/*
  SNESDestroy_Multiblock - Destroys the private SNES_Multiblock context that was created with SNESCreate_Multiblock().

  Input Parameter:
. snes - the SNES context

  Application Interface Routine: SNESDestroy()
*/
PetscErrorCode SNESDestroy_Multiblock(SNES snes)
{
  SNES_Multiblock *mb    = (SNES_Multiblock*) snes->data;
  BlockDesc       blocks = mb->blocks, next;

  PetscFunctionBegin;
  CHKERRQ(SNESReset_Multiblock(snes));
  while (blocks) {
    next   = blocks->next;
    CHKERRQ(SNESDestroy(&blocks->snes));
    CHKERRQ(PetscFree(blocks->name));
    CHKERRQ(PetscFree(blocks->fields));
    CHKERRQ(PetscFree(blocks));
    blocks = next;
  }
  CHKERRQ(PetscFree(snes->data));
  PetscFunctionReturn(0);
}

/* Precondition: blocksize is set to a meaningful value */
static PetscErrorCode SNESMultiblockSetFieldsRuntime_Private(SNES snes)
{
  SNES_Multiblock *mb = (SNES_Multiblock*) snes->data;
  PetscInt        *ifields;
  PetscInt        i, nfields;
  PetscBool       flg = PETSC_TRUE;
  char            optionname[128], name[8];

  PetscFunctionBegin;
  CHKERRQ(PetscMalloc1(mb->bs, &ifields));
  for (i = 0;; ++i) {
    CHKERRQ(PetscSNPrintf(name, sizeof(name), "%D", i));
    CHKERRQ(PetscSNPrintf(optionname, sizeof(optionname), "-snes_multiblock_%D_fields", i));
    nfields = mb->bs;
    CHKERRQ(PetscOptionsGetIntArray(NULL,((PetscObject) snes)->prefix, optionname, ifields, &nfields, &flg));
    if (!flg) break;
    PetscCheckFalse(!nfields,PETSC_COMM_SELF, PETSC_ERR_USER, "Cannot list zero fields");
    CHKERRQ(SNESMultiblockSetFields(snes, name, nfields, ifields));
  }
  if (i > 0) {
    /* Makes command-line setting of blocks take precedence over setting them in code.
       Otherwise subsequent calls to SNESMultiblockSetIS() or SNESMultiblockSetFields() would
       create new blocks, which would probably not be what the user wanted. */
    mb->defined = PETSC_TRUE;
  }
  CHKERRQ(PetscFree(ifields));
  PetscFunctionReturn(0);
}

static PetscErrorCode SNESMultiblockSetDefaults(SNES snes)
{
  SNES_Multiblock *mb    = (SNES_Multiblock*) snes->data;
  BlockDesc       blocks = mb->blocks;
  PetscInt        i;

  PetscFunctionBegin;
  if (!blocks) {
    if (snes->dm) {
      PetscBool dmcomposite;

      CHKERRQ(PetscObjectTypeCompare((PetscObject) snes->dm, DMCOMPOSITE, &dmcomposite));
      if (dmcomposite) {
        PetscInt nDM;
        IS       *fields;

        CHKERRQ(PetscInfo(snes,"Setting up physics based multiblock solver using the embedded DM\n"));
        CHKERRQ(DMCompositeGetNumberDM(snes->dm, &nDM));
        CHKERRQ(DMCompositeGetGlobalISs(snes->dm, &fields));
        for (i = 0; i < nDM; ++i) {
          char name[8];

          CHKERRQ(PetscSNPrintf(name, sizeof(name), "%D", i));
          CHKERRQ(SNESMultiblockSetIS(snes, name, fields[i]));
          CHKERRQ(ISDestroy(&fields[i]));
        }
        CHKERRQ(PetscFree(fields));
      }
    } else {
      PetscBool flg    = PETSC_FALSE;
      PetscBool stokes = PETSC_FALSE;

      if (mb->bs <= 0) {
        if (snes->jacobian_pre) {
          CHKERRQ(MatGetBlockSize(snes->jacobian_pre, &mb->bs));
        } else mb->bs = 1;
      }

      CHKERRQ(PetscOptionsGetBool(NULL,((PetscObject) snes)->prefix, "-snes_multiblock_default", &flg, NULL));
      CHKERRQ(PetscOptionsGetBool(NULL,((PetscObject) snes)->prefix, "-snes_multiblock_detect_saddle_point", &stokes, NULL));
      if (stokes) {
        IS       zerodiags, rest;
        PetscInt nmin, nmax;

        CHKERRQ(MatGetOwnershipRange(snes->jacobian_pre, &nmin, &nmax));
        CHKERRQ(MatFindZeroDiagonals(snes->jacobian_pre, &zerodiags));
        CHKERRQ(ISComplement(zerodiags, nmin, nmax, &rest));
        CHKERRQ(SNESMultiblockSetIS(snes, "0", rest));
        CHKERRQ(SNESMultiblockSetIS(snes, "1", zerodiags));
        CHKERRQ(ISDestroy(&zerodiags));
        CHKERRQ(ISDestroy(&rest));
      } else {
        if (!flg) {
          /* Allow user to set fields from command line, if bs was known at the time of SNESSetFromOptions_Multiblock()
           then it is set there. This is not ideal because we should only have options set in XXSetFromOptions(). */
          CHKERRQ(SNESMultiblockSetFieldsRuntime_Private(snes));
          if (mb->defined) CHKERRQ(PetscInfo(snes, "Blocks defined using the options database\n"));
        }
        if (flg || !mb->defined) {
          CHKERRQ(PetscInfo(snes, "Using default splitting of fields\n"));
          for (i = 0; i < mb->bs; ++i) {
            char name[8];

            CHKERRQ(PetscSNPrintf(name, sizeof(name), "%D", i));
            CHKERRQ(SNESMultiblockSetFields(snes, name, 1, &i));
          }
          mb->defaultblocks = PETSC_TRUE;
        }
      }
    }
  } else if (mb->numBlocks == 1) {
    if (blocks->is) {
      IS       is2;
      PetscInt nmin, nmax;

      CHKERRQ(MatGetOwnershipRange(snes->jacobian_pre, &nmin, &nmax));
      CHKERRQ(ISComplement(blocks->is, nmin, nmax, &is2));
      CHKERRQ(SNESMultiblockSetIS(snes, "1", is2));
      CHKERRQ(ISDestroy(&is2));
    } else SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONGSTATE, "Must provide at least two sets of fields to SNES multiblock");
  }
  PetscCheckFalse(mb->numBlocks < 2,PETSC_COMM_SELF, PETSC_ERR_PLIB, "Unhandled case, must have at least two blocks");
  PetscFunctionReturn(0);
}

/*
   SNESSetUp_Multiblock - Sets up the internal data structures for the later use
   of the SNESMULTIBLOCK nonlinear solver.

   Input Parameters:
+  snes - the SNES context
-  x - the solution vector

   Application Interface Routine: SNESSetUp()
*/
PetscErrorCode SNESSetUp_Multiblock(SNES snes)
{
  SNES_Multiblock *mb = (SNES_Multiblock*) snes->data;
  BlockDesc       blocks;
  PetscInt        i, numBlocks;

  PetscFunctionBegin;
  CHKERRQ(SNESMultiblockSetDefaults(snes));
  numBlocks = mb->numBlocks;
  blocks    = mb->blocks;

  /* Create ISs */
  if (!mb->issetup) {
    PetscInt  ccsize, rstart, rend, nslots, bs;
    PetscBool sorted;

    mb->issetup = PETSC_TRUE;
    bs          = mb->bs;
    CHKERRQ(MatGetOwnershipRange(snes->jacobian_pre, &rstart, &rend));
    CHKERRQ(MatGetLocalSize(snes->jacobian_pre, NULL, &ccsize));
    nslots      = (rend - rstart)/bs;
    for (i = 0; i < numBlocks; ++i) {
      if (mb->defaultblocks) {
        CHKERRQ(ISCreateStride(PetscObjectComm((PetscObject)snes), nslots, rstart+i, numBlocks, &blocks->is));
      } else if (!blocks->is) {
        if (blocks->nfields > 1) {
          PetscInt *ii, j, k, nfields = blocks->nfields, *fields = blocks->fields;

          CHKERRQ(PetscMalloc1(nfields*nslots, &ii));
          for (j = 0; j < nslots; ++j) {
            for (k = 0; k < nfields; ++k) {
              ii[nfields*j + k] = rstart + bs*j + fields[k];
            }
          }
          CHKERRQ(ISCreateGeneral(PetscObjectComm((PetscObject)snes), nslots*nfields, ii, PETSC_OWN_POINTER, &blocks->is));
        } else {
          CHKERRQ(ISCreateStride(PetscObjectComm((PetscObject)snes), nslots, rstart+blocks->fields[0], bs, &blocks->is));
        }
      }
      CHKERRQ(ISSorted(blocks->is, &sorted));
      PetscCheckFalse(!sorted,PETSC_COMM_SELF, PETSC_ERR_USER, "Fields must be sorted when creating split");
      blocks = blocks->next;
    }
  }

#if 0
  /* Create matrices */
  ilink = jac->head;
  if (!jac->pmat) {
    CHKERRQ(PetscMalloc1(nsplit,&jac->pmat));
    for (i=0; i<nsplit; i++) {
      CHKERRQ(MatCreateSubMatrix(pc->pmat,ilink->is,ilink->is,MAT_INITIAL_MATRIX,&jac->pmat[i]));
      ilink = ilink->next;
    }
  } else {
    for (i=0; i<nsplit; i++) {
      CHKERRQ(MatCreateSubMatrix(pc->pmat,ilink->is,ilink->is,MAT_REUSE_MATRIX,&jac->pmat[i]));
      ilink = ilink->next;
    }
  }
  if (jac->realdiagonal) {
    ilink = jac->head;
    if (!jac->mat) {
      CHKERRQ(PetscMalloc1(nsplit,&jac->mat));
      for (i=0; i<nsplit; i++) {
        CHKERRQ(MatCreateSubMatrix(pc->mat,ilink->is,ilink->is,MAT_INITIAL_MATRIX,&jac->mat[i]));
        ilink = ilink->next;
      }
    } else {
      for (i=0; i<nsplit; i++) {
        if (jac->mat[i]) CHKERRQ(MatCreateSubMatrix(pc->mat,ilink->is,ilink->is,MAT_REUSE_MATRIX,&jac->mat[i]));
        ilink = ilink->next;
      }
    }
  } else jac->mat = jac->pmat;
#endif

#if 0
  if (jac->type != PC_COMPOSITE_ADDITIVE  && jac->type != PC_COMPOSITE_SCHUR) {
    /* extract the rows of the matrix associated with each field: used for efficient computation of residual inside algorithm */
    ilink = jac->head;
    if (!jac->Afield) {
      CHKERRQ(PetscMalloc1(nsplit,&jac->Afield));
      for (i=0; i<nsplit; i++) {
        CHKERRQ(MatCreateSubMatrix(pc->mat,ilink->is,NULL,MAT_INITIAL_MATRIX,&jac->Afield[i]));
        ilink = ilink->next;
      }
    } else {
      for (i=0; i<nsplit; i++) {
        CHKERRQ(MatCreateSubMatrix(pc->mat,ilink->is,NULL,MAT_REUSE_MATRIX,&jac->Afield[i]));
        ilink = ilink->next;
      }
    }
  }
#endif

  if (mb->type == PC_COMPOSITE_SCHUR) {
#if 0
    IS       ccis;
    PetscInt rstart,rend;
    PetscCheckFalse(nsplit != 2,PetscObjectComm((PetscObject)pc),PETSC_ERR_ARG_INCOMP,"To use Schur complement preconditioner you must have exactly 2 fields");

    /* When extracting off-diagonal submatrices, we take complements from this range */
    CHKERRQ(MatGetOwnershipRangeColumn(pc->mat,&rstart,&rend));

    /* need to handle case when one is resetting up the preconditioner */
    if (jac->schur) {
      ilink = jac->head;
      CHKERRQ(ISComplement(ilink->is,rstart,rend,&ccis));
      CHKERRQ(MatCreateSubMatrix(pc->mat,ilink->is,ccis,MAT_REUSE_MATRIX,&jac->B));
      CHKERRQ(ISDestroy(&ccis));
      ilink = ilink->next;
      CHKERRQ(ISComplement(ilink->is,rstart,rend,&ccis));
      CHKERRQ(MatCreateSubMatrix(pc->mat,ilink->is,ccis,MAT_REUSE_MATRIX,&jac->C));
      CHKERRQ(ISDestroy(&ccis));
      CHKERRQ(MatSchurComplementUpdateSubMatrices(jac->schur,jac->mat[0],jac->pmat[0],jac->B,jac->C,jac->pmat[1]));
      CHKERRQ(KSPSetOperators(jac->kspschur,jac->schur,FieldSplitSchurPre(jac),pc->flag));

    } else {
      KSP  ksp;
      char schurprefix[256];

      /* extract the A01 and A10 matrices */
      ilink = jac->head;
      CHKERRQ(ISComplement(ilink->is,rstart,rend,&ccis));
      CHKERRQ(MatCreateSubMatrix(pc->mat,ilink->is,ccis,MAT_INITIAL_MATRIX,&jac->B));
      CHKERRQ(ISDestroy(&ccis));
      ilink = ilink->next;
      CHKERRQ(ISComplement(ilink->is,rstart,rend,&ccis));
      CHKERRQ(MatCreateSubMatrix(pc->mat,ilink->is,ccis,MAT_INITIAL_MATRIX,&jac->C));
      CHKERRQ(ISDestroy(&ccis));
      /* Use mat[0] (diagonal block of the real matrix) preconditioned by pmat[0] */
      CHKERRQ(MatCreateSchurComplement(jac->mat[0],jac->pmat[0],jac->B,jac->C,jac->mat[1],&jac->schur));
      /* set tabbing and options prefix of KSP inside the MatSchur */
      CHKERRQ(MatSchurComplementGetKSP(jac->schur,&ksp));
      CHKERRQ(PetscObjectIncrementTabLevel((PetscObject)ksp,(PetscObject)pc,2));
      CHKERRQ(PetscSNPrintf(schurprefix,sizeof(schurprefix),"%sfieldsplit_%s_",((PetscObject)pc)->prefix ? ((PetscObject)pc)->prefix : "",jac->head->splitname));
      CHKERRQ(KSPSetOptionsPrefix(ksp,schurprefix));
      CHKERRQ(MatSetFromOptions(jac->schur));

      CHKERRQ(KSPCreate(PetscObjectComm((PetscObject)pc),&jac->kspschur));
      CHKERRQ(PetscLogObjectParent((PetscObject)pc,(PetscObject)jac->kspschur));
      CHKERRQ(PetscObjectIncrementTabLevel((PetscObject)jac->kspschur,(PetscObject)pc,1));
      CHKERRQ(KSPSetOperators(jac->kspschur,jac->schur,FieldSplitSchurPre(jac)));
      if (jac->schurpre == PC_FIELDSPLIT_SCHUR_PRE_SELF) {
        PC pc;
        CHKERRQ(KSPGetPC(jac->kspschur,&pc));
        CHKERRQ(PCSetType(pc,PCNONE));
        /* Note: This is bad if there exist preconditioners for MATSCHURCOMPLEMENT */
      }
      CHKERRQ(PetscSNPrintf(schurprefix,sizeof(schurprefix),"%sfieldsplit_%s_",((PetscObject)pc)->prefix ? ((PetscObject)pc)->prefix : "",ilink->splitname));
      CHKERRQ(KSPSetOptionsPrefix(jac->kspschur,schurprefix));
      /* really want setfromoptions called in PCSetFromOptions_FieldSplit(), but it is not ready yet */
      CHKERRQ(KSPSetFromOptions(jac->kspschur));

      CHKERRQ(PetscMalloc2(2,&jac->x,2,&jac->y));
      CHKERRQ(MatCreateVecs(jac->pmat[0],&jac->x[0],&jac->y[0]));
      CHKERRQ(MatCreateVecs(jac->pmat[1],&jac->x[1],&jac->y[1]));
      ilink    = jac->head;
      ilink->x = jac->x[0]; ilink->y = jac->y[0];
      ilink    = ilink->next;
      ilink->x = jac->x[1]; ilink->y = jac->y[1];
    }
#endif
  } else {
    /* Set up the individual SNESs */
    blocks = mb->blocks;
    i      = 0;
    while (blocks) {
      /*TODO: Set these correctly */
      /*CHKERRQ(SNESSetFunction(blocks->snes, blocks->x, func));*/
      /*CHKERRQ(SNESSetJacobian(blocks->snes, blocks->x, jac));*/
      CHKERRQ(VecDuplicate(blocks->snes->vec_sol, &blocks->x));
      /* really want setfromoptions called in SNESSetFromOptions_Multiblock(), but it is not ready yet */
      CHKERRQ(SNESSetFromOptions(blocks->snes));
      CHKERRQ(SNESSetUp(blocks->snes));
      blocks = blocks->next;
      i++;
    }
  }

  /* Compute scatter contexts needed by multiplicative versions and non-default splits */
  if (!mb->blocks->sctx) {
    Vec xtmp;

    blocks = mb->blocks;
    CHKERRQ(MatCreateVecs(snes->jacobian_pre, &xtmp, NULL));
    while (blocks) {
      CHKERRQ(VecScatterCreate(xtmp, blocks->is, blocks->x, NULL, &blocks->sctx));
      blocks = blocks->next;
    }
    CHKERRQ(VecDestroy(&xtmp));
  }
  PetscFunctionReturn(0);
}

/*
  SNESSetFromOptions_Multiblock - Sets various parameters for the SNESMULTIBLOCK method.

  Input Parameter:
. snes - the SNES context

  Application Interface Routine: SNESSetFromOptions()
*/
static PetscErrorCode SNESSetFromOptions_Multiblock(PetscOptionItems *PetscOptionsObject,SNES snes)
{
  SNES_Multiblock *mb = (SNES_Multiblock*) snes->data;
  PCCompositeType ctype;
  PetscInt        bs;
  PetscBool       flg;

  PetscFunctionBegin;
  CHKERRQ(PetscOptionsHead(PetscOptionsObject,"SNES Multiblock options"));
  CHKERRQ(PetscOptionsInt("-snes_multiblock_block_size", "Blocksize that defines number of fields", "PCFieldSplitSetBlockSize", mb->bs, &bs, &flg));
  if (flg) CHKERRQ(SNESMultiblockSetBlockSize(snes, bs));
  CHKERRQ(PetscOptionsEnum("-snes_multiblock_type", "Type of composition", "PCFieldSplitSetType", PCCompositeTypes, (PetscEnum) mb->type, (PetscEnum*) &ctype, &flg));
  if (flg) {
    CHKERRQ(SNESMultiblockSetType(snes,ctype));
  }
  /* Only setup fields once */
  if ((mb->bs > 0) && (mb->numBlocks == 0)) {
    /* only allow user to set fields from command line if bs is already known, otherwise user can set them in SNESMultiblockSetDefaults() */
    CHKERRQ(SNESMultiblockSetFieldsRuntime_Private(snes));
    if (mb->defined) CHKERRQ(PetscInfo(snes, "Blocks defined using the options database\n"));
  }
  CHKERRQ(PetscOptionsTail());
  PetscFunctionReturn(0);
}

/*
  SNESView_Multiblock - Prints info from the SNESMULTIBLOCK data structure.

  Input Parameters:
+ SNES - the SNES context
- viewer - visualization context

  Application Interface Routine: SNESView()
*/
static PetscErrorCode SNESView_Multiblock(SNES snes, PetscViewer viewer)
{
  SNES_Multiblock *mb    = (SNES_Multiblock*) snes->data;
  BlockDesc       blocks = mb->blocks;
  PetscBool       iascii;

  PetscFunctionBegin;
  CHKERRQ(PetscObjectTypeCompare((PetscObject) viewer, PETSCVIEWERASCII, &iascii));
  if (iascii) {
    CHKERRQ(PetscViewerASCIIPrintf(viewer,"  Multiblock with %s composition: total blocks = %D, blocksize = %D\n", PCCompositeTypes[mb->type], mb->numBlocks, mb->bs));
    CHKERRQ(PetscViewerASCIIPrintf(viewer,"  Solver info for each split is in the following SNES objects:\n"));
    CHKERRQ(PetscViewerASCIIPushTab(viewer));
    while (blocks) {
      if (blocks->fields) {
        PetscInt j;

        CHKERRQ(PetscViewerASCIIPrintf(viewer, "  Block %s Fields ", blocks->name));
        CHKERRQ(PetscViewerASCIIUseTabs(viewer, PETSC_FALSE));
        for (j = 0; j < blocks->nfields; ++j) {
          if (j > 0) {
            CHKERRQ(PetscViewerASCIIPrintf(viewer, ","));
          }
          CHKERRQ(PetscViewerASCIIPrintf(viewer, " %D", blocks->fields[j]));
        }
        CHKERRQ(PetscViewerASCIIPrintf(viewer, "\n"));
        CHKERRQ(PetscViewerASCIIUseTabs(viewer, PETSC_TRUE));
      } else {
        CHKERRQ(PetscViewerASCIIPrintf(viewer, "  Block %s Defined by IS\n", blocks->name));
      }
      CHKERRQ(SNESView(blocks->snes, viewer));
      blocks = blocks->next;
    }
    CHKERRQ(PetscViewerASCIIPopTab(viewer));
  }
  PetscFunctionReturn(0);
}

/*
  SNESSolve_Multiblock - Solves a nonlinear system with the Multiblock method.

  Input Parameters:
. snes - the SNES context

  Output Parameter:
. outits - number of iterations until termination

  Application Interface Routine: SNESSolve()
*/
PetscErrorCode SNESSolve_Multiblock(SNES snes)
{
  SNES_Multiblock *mb = (SNES_Multiblock*) snes->data;
  Vec             X, Y, F;
  PetscReal       fnorm;
  PetscInt        maxits, i;

  PetscFunctionBegin;
  PetscCheckFalse(snes->xl || snes->xu || snes->ops->computevariablebounds,PetscObjectComm((PetscObject)snes),PETSC_ERR_ARG_WRONGSTATE, "SNES solver %s does not support bounds", ((PetscObject)snes)->type_name);

  snes->reason = SNES_CONVERGED_ITERATING;

  maxits = snes->max_its;        /* maximum number of iterations */
  X      = snes->vec_sol;        /* X^n */
  Y      = snes->vec_sol_update; /* \tilde X */
  F      = snes->vec_func;       /* residual vector */

  CHKERRQ(VecSetBlockSize(X, mb->bs));
  CHKERRQ(VecSetBlockSize(Y, mb->bs));
  CHKERRQ(VecSetBlockSize(F, mb->bs));
  CHKERRQ(PetscObjectSAWsTakeAccess((PetscObject)snes));
  snes->iter = 0;
  snes->norm = 0.;
  CHKERRQ(PetscObjectSAWsGrantAccess((PetscObject)snes));

  if (!snes->vec_func_init_set) {
    CHKERRQ(SNESComputeFunction(snes, X, F));
  } else snes->vec_func_init_set = PETSC_FALSE;

  CHKERRQ(VecNorm(F, NORM_2, &fnorm)); /* fnorm <- ||F||  */
  SNESCheckFunctionNorm(snes,fnorm);
  CHKERRQ(PetscObjectSAWsTakeAccess((PetscObject)snes));
  snes->norm = fnorm;
  CHKERRQ(PetscObjectSAWsGrantAccess((PetscObject)snes));
  CHKERRQ(SNESLogConvergenceHistory(snes,fnorm,0));
  CHKERRQ(SNESMonitor(snes,0,fnorm));

  /* test convergence */
  CHKERRQ((*snes->ops->converged)(snes,0,0.0,0.0,fnorm,&snes->reason,snes->cnvP));
  if (snes->reason) PetscFunctionReturn(0);

  for (i = 0; i < maxits; i++) {
    /* Call general purpose update function */
    if (snes->ops->update) {
      CHKERRQ((*snes->ops->update)(snes, snes->iter));
    }
    /* Compute X^{new} from subsolves */
    if (mb->type == PC_COMPOSITE_ADDITIVE) {
      BlockDesc blocks = mb->blocks;

      if (mb->defaultblocks) {
        /*TODO: Make an array of Vecs for this */
        /*CHKERRQ(VecStrideGatherAll(X, mb->x, INSERT_VALUES));*/
        while (blocks) {
          CHKERRQ(SNESSolve(blocks->snes, NULL, blocks->x));
          blocks = blocks->next;
        }
        /*CHKERRQ(VecStrideScatterAll(mb->x, X, INSERT_VALUES));*/
      } else {
        while (blocks) {
          CHKERRQ(VecScatterBegin(blocks->sctx, X, blocks->x, INSERT_VALUES, SCATTER_FORWARD));
          CHKERRQ(VecScatterEnd(blocks->sctx, X, blocks->x, INSERT_VALUES, SCATTER_FORWARD));
          CHKERRQ(SNESSolve(blocks->snes, NULL, blocks->x));
          CHKERRQ(VecScatterBegin(blocks->sctx, blocks->x, X, INSERT_VALUES, SCATTER_REVERSE));
          CHKERRQ(VecScatterEnd(blocks->sctx, blocks->x, X, INSERT_VALUES, SCATTER_REVERSE));
          blocks = blocks->next;
        }
      }
    } else SETERRQ(PetscObjectComm((PetscObject)snes), PETSC_ERR_SUP, "Unsupported or unknown composition", (int) mb->type);
    /* Compute F(X^{new}) */
    CHKERRQ(SNESComputeFunction(snes, X, F));
    CHKERRQ(VecNorm(F, NORM_2, &fnorm));
    SNESCheckFunctionNorm(snes,fnorm);

    if (snes->nfuncs >= snes->max_funcs && snes->max_funcs >=0) {
      snes->reason = SNES_DIVERGED_FUNCTION_COUNT;
      break;
    }

    /* Monitor convergence */
    CHKERRQ(PetscObjectSAWsTakeAccess((PetscObject)snes));
    snes->iter = i+1;
    snes->norm = fnorm;
    CHKERRQ(PetscObjectSAWsGrantAccess((PetscObject)snes));
    CHKERRQ(SNESLogConvergenceHistory(snes,snes->norm,0));
    CHKERRQ(SNESMonitor(snes,snes->iter,snes->norm));
    /* Test for convergence */
    CHKERRQ((*snes->ops->converged)(snes,snes->iter,0.0,0.0,fnorm,&snes->reason,snes->cnvP));
    if (snes->reason) break;
  }
  if (i == maxits) {
    CHKERRQ(PetscInfo(snes, "Maximum number of iterations has been reached: %D\n", maxits));
    if (!snes->reason) snes->reason = SNES_DIVERGED_MAX_IT;
  }
  PetscFunctionReturn(0);
}

PetscErrorCode SNESMultiblockSetFields_Default(SNES snes, const char name[], PetscInt n, const PetscInt fields[])
{
  SNES_Multiblock *mb = (SNES_Multiblock*) snes->data;
  BlockDesc       newblock, next = mb->blocks;
  char            prefix[128];
  PetscInt        i;

  PetscFunctionBegin;
  if (mb->defined) {
    CHKERRQ(PetscInfo(snes, "Ignoring new block \"%s\" because the blocks have already been defined\n", name));
    PetscFunctionReturn(0);
  }
  for (i = 0; i < n; ++i) {
    PetscCheckFalse(fields[i] >= mb->bs,PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Field %D requested but only %D exist", fields[i], mb->bs);
    PetscCheckFalse(fields[i] < 0,PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Negative field %D requested", fields[i]);
  }
  CHKERRQ(PetscNew(&newblock));
  if (name) {
    CHKERRQ(PetscStrallocpy(name, &newblock->name));
  } else {
    PetscInt len = floor(log10(mb->numBlocks))+1;

    CHKERRQ(PetscMalloc1(len+1, &newblock->name));
    CHKERRQ(PetscSNPrintf(newblock->name, len, "%s", mb->numBlocks));
  }
  newblock->nfields = n;

  CHKERRQ(PetscMalloc1(n, &newblock->fields));
  CHKERRQ(PetscArraycpy(newblock->fields, fields, n));

  newblock->next = NULL;

  CHKERRQ(SNESCreate(PetscObjectComm((PetscObject)snes), &newblock->snes));
  CHKERRQ(PetscObjectIncrementTabLevel((PetscObject) newblock->snes, (PetscObject) snes, 1));
  CHKERRQ(SNESSetType(newblock->snes, SNESNRICHARDSON));
  CHKERRQ(PetscLogObjectParent((PetscObject) snes, (PetscObject) newblock->snes));
  CHKERRQ(PetscSNPrintf(prefix, sizeof(prefix), "%smultiblock_%s_", ((PetscObject) snes)->prefix ? ((PetscObject) snes)->prefix : "", newblock->name));
  CHKERRQ(SNESSetOptionsPrefix(newblock->snes, prefix));

  if (!next) {
    mb->blocks         = newblock;
    newblock->previous = NULL;
  } else {
    while (next->next) {
      next = next->next;
    }
    next->next         = newblock;
    newblock->previous = next;
  }
  mb->numBlocks++;
  PetscFunctionReturn(0);
}

PetscErrorCode SNESMultiblockSetIS_Default(SNES snes, const char name[], IS is)
{
  SNES_Multiblock *mb = (SNES_Multiblock*) snes->data;
  BlockDesc       newblock, next = mb->blocks;
  char            prefix[128];

  PetscFunctionBegin;
  if (mb->defined) {
    CHKERRQ(PetscInfo(snes, "Ignoring new block \"%s\" because the blocks have already been defined\n", name));
    PetscFunctionReturn(0);
  }
  CHKERRQ(PetscNew(&newblock));
  if (name) {
    CHKERRQ(PetscStrallocpy(name, &newblock->name));
  } else {
    PetscInt len = floor(log10(mb->numBlocks))+1;

    CHKERRQ(PetscMalloc1(len+1, &newblock->name));
    CHKERRQ(PetscSNPrintf(newblock->name, len, "%s", mb->numBlocks));
  }
  newblock->is = is;

  CHKERRQ(PetscObjectReference((PetscObject) is));

  newblock->next = NULL;

  CHKERRQ(SNESCreate(PetscObjectComm((PetscObject)snes), &newblock->snes));
  CHKERRQ(PetscObjectIncrementTabLevel((PetscObject) newblock->snes, (PetscObject) snes, 1));
  CHKERRQ(SNESSetType(newblock->snes, SNESNRICHARDSON));
  CHKERRQ(PetscLogObjectParent((PetscObject) snes, (PetscObject) newblock->snes));
  CHKERRQ(PetscSNPrintf(prefix, sizeof(prefix), "%smultiblock_%s_", ((PetscObject) snes)->prefix ? ((PetscObject) snes)->prefix : "", newblock->name));
  CHKERRQ(SNESSetOptionsPrefix(newblock->snes, prefix));

  if (!next) {
    mb->blocks         = newblock;
    newblock->previous = NULL;
  } else {
    while (next->next) {
      next = next->next;
    }
    next->next         = newblock;
    newblock->previous = next;
  }
  mb->numBlocks++;
  PetscFunctionReturn(0);
}

PetscErrorCode  SNESMultiblockSetBlockSize_Default(SNES snes, PetscInt bs)
{
  SNES_Multiblock *mb = (SNES_Multiblock*) snes->data;

  PetscFunctionBegin;
  PetscCheckFalse(bs < 1,PetscObjectComm((PetscObject)snes), PETSC_ERR_ARG_OUTOFRANGE, "Blocksize must be positive, you gave %D", bs);
  PetscCheckFalse(mb->bs > 0 && mb->bs != bs,PetscObjectComm((PetscObject)snes), PETSC_ERR_ARG_WRONGSTATE, "Cannot change blocksize from %D to %D after it has been set", mb->bs, bs);
  mb->bs = bs;
  PetscFunctionReturn(0);
}

PetscErrorCode SNESMultiblockGetSubSNES_Default(SNES snes, PetscInt *n, SNES **subsnes)
{
  SNES_Multiblock *mb    = (SNES_Multiblock*) snes->data;
  BlockDesc       blocks = mb->blocks;
  PetscInt        cnt    = 0;

  PetscFunctionBegin;
  CHKERRQ(PetscMalloc1(mb->numBlocks, subsnes));
  while (blocks) {
    (*subsnes)[cnt++] = blocks->snes;
    blocks            = blocks->next;
  }
  PetscCheckFalse(cnt != mb->numBlocks,PETSC_COMM_SELF, PETSC_ERR_PLIB, "Corrupt SNESMULTIBLOCK object: number of blocks in linked list %D does not match number in object %D", cnt, mb->numBlocks);

  if (n) *n = mb->numBlocks;
  PetscFunctionReturn(0);
}

PetscErrorCode  SNESMultiblockSetType_Default(SNES snes, PCCompositeType type)
{
  SNES_Multiblock *mb = (SNES_Multiblock*) snes->data;

  PetscFunctionBegin;
  mb->type = type;
  if (type == PC_COMPOSITE_SCHUR) {
#if 1
    SETERRQ(PetscObjectComm((PetscObject)snes), PETSC_ERR_SUP, "The Schur composite type is not yet supported");
#else
    snes->ops->solve = SNESSolve_Multiblock_Schur;
    snes->ops->view  = SNESView_Multiblock_Schur;

    CHKERRQ(PetscObjectComposeFunction((PetscObject) snes, "SNESMultiblockGetSubSNES_C", SNESMultiblockGetSubSNES_Schur));
    CHKERRQ(PetscObjectComposeFunction((PetscObject) snes, "SNESMultiblockSchurPrecondition_C", SNESMultiblockSchurPrecondition_Default));
#endif
  } else {
    snes->ops->solve = SNESSolve_Multiblock;
    snes->ops->view  = SNESView_Multiblock;

    CHKERRQ(PetscObjectComposeFunction((PetscObject) snes, "SNESMultiblockGetSubSNES_C", SNESMultiblockGetSubSNES_Default));
    CHKERRQ(PetscObjectComposeFunction((PetscObject) snes, "SNESMultiblockSchurPrecondition_C", 0));
  }
  PetscFunctionReturn(0);
}

/*@
  SNESMultiblockSetFields - Sets the fields for one particular block in the solver

  Logically Collective on SNES

  Input Parameters:
+ snes   - the solver
. name   - name of this block, if NULL the number of the block is used
. n      - the number of fields in this block
- fields - the fields in this block

  Level: intermediate

  Notes:
    Use SNESMultiblockSetIS() to set a completely general set of row indices as a block.

  The SNESMultiblockSetFields() is for defining blocks as a group of strided indices, or fields.
  For example, if the vector block size is three then one can define a block as field 0, or
  1 or 2, or field 0,1 or 0,2 or 1,2 which means
    0xx3xx6xx9xx12 ... x1xx4xx7xx ... xx2xx5xx8xx.. 01x34x67x... 0x1x3x5x7.. x12x45x78x....
  where the numbered entries indicate what is in the block.

  This function is called once per block (it creates a new block each time). Solve options
  for this block will be available under the prefix -multiblock_BLOCKNAME_.

.seealso: SNESMultiblockGetSubSNES(), SNESMULTIBLOCK, SNESMultiblockSetBlockSize(), SNESMultiblockSetIS()
@*/
PetscErrorCode SNESMultiblockSetFields(SNES snes, const char name[], PetscInt n, const PetscInt *fields)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(snes, SNES_CLASSID, 1);
  PetscValidCharPointer(name, 2);
  PetscCheckFalse(n < 1,PetscObjectComm((PetscObject)snes), PETSC_ERR_ARG_OUTOFRANGE, "Provided number of fields %D in split \"%s\" not positive", n, name);
  PetscValidIntPointer(fields, 4);
  CHKERRQ(PetscTryMethod(snes, "SNESMultiblockSetFields_C", (SNES, const char[], PetscInt, const PetscInt*), (snes, name, n, fields)));
  PetscFunctionReturn(0);
}

/*@
  SNESMultiblockSetIS - Sets the global row indices for the block

  Logically Collective on SNES

  Input Parameters:
+ snes - the solver context
. name - name of this block, if NULL the number of the block is used
- is   - the index set that defines the global row indices in this block

  Notes:
  Use SNESMultiblockSetFields(), for blocks defined by strides.

  This function is called once per block (it creates a new block each time). Solve options
  for this block will be available under the prefix -multiblock_BLOCKNAME_.

  Level: intermediate

.seealso: SNESMultiblockGetSubSNES(), SNESMULTIBLOCK, SNESMultiblockSetBlockSize()
@*/
PetscErrorCode SNESMultiblockSetIS(SNES snes, const char name[], IS is)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(snes, SNES_CLASSID, 1);
  PetscValidCharPointer(name, 2);
  PetscValidHeaderSpecific(is, IS_CLASSID, 3);
  CHKERRQ(PetscTryMethod(snes, "SNESMultiblockSetIS_C", (SNES, const char[], IS), (snes, name, is)));
  PetscFunctionReturn(0);
}

/*@
  SNESMultiblockSetType - Sets the type of block combination.

  Collective on SNES

  Input Parameters:
+ snes - the solver context
- type - PC_COMPOSITE_ADDITIVE, PC_COMPOSITE_MULTIPLICATIVE (default), PC_COMPOSITE_SYMMETRIC_MULTIPLICATIVE

  Options Database Key:
. -snes_multiblock_type <type: one of multiplicative, additive, symmetric_multiplicative> - Sets block combination type

  Level: Developer

.seealso: PCCompositeSetType()
@*/
PetscErrorCode SNESMultiblockSetType(SNES snes, PCCompositeType type)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(snes, SNES_CLASSID, 1);
  CHKERRQ(PetscTryMethod(snes, "SNESMultiblockSetType_C", (SNES, PCCompositeType), (snes, type)));
  PetscFunctionReturn(0);
}

/*@
  SNESMultiblockSetBlockSize - Sets the block size for structured mesh block division. If not set the matrix block size is used.

  Logically Collective on SNES

  Input Parameters:
+ snes - the solver context
- bs   - the block size

  Level: intermediate

.seealso: SNESMultiblockGetSubSNES(), SNESMULTIBLOCK, SNESMultiblockSetFields()
@*/
PetscErrorCode SNESMultiblockSetBlockSize(SNES snes, PetscInt bs)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(snes, SNES_CLASSID, 1);
  PetscValidLogicalCollectiveInt(snes, bs, 2);
  CHKERRQ(PetscTryMethod(snes, "SNESMultiblockSetBlockSize_C", (SNES, PetscInt), (snes,bs)));
  PetscFunctionReturn(0);
}

/*@C
  SNESMultiblockGetSubSNES - Gets the SNES contexts for all blocks

  Collective on SNES

  Input Parameter:
. snes - the solver context

  Output Parameters:
+ n       - the number of blocks
- subsnes - the array of SNES contexts

  Note:
  After SNESMultiblockGetSubSNES() the array of SNESs MUST be freed by the user
  (not each SNES, just the array that contains them).

  You must call SNESSetUp() before calling SNESMultiblockGetSubSNES().

  Level: advanced

.seealso: SNESMULTIBLOCK
@*/
PetscErrorCode SNESMultiblockGetSubSNES(SNES snes, PetscInt *n, SNES *subsnes[])
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(snes, SNES_CLASSID, 1);
  if (n) PetscValidIntPointer(n, 2);
  CHKERRQ(PetscUseMethod(snes, "SNESMultiblockGetSubSNES_C", (SNES, PetscInt*, SNES **), (snes, n, subsnes)));
  PetscFunctionReturn(0);
}

/*MC
  SNESMULTIBLOCK - Multiblock nonlinear solver that can use overlapping or nonoverlapping blocks, organized
  additively (Jacobi) or multiplicatively (Gauss-Seidel).

  Level: beginner

.seealso:  SNESCreate(), SNES, SNESSetType(), SNESNEWTONLS, SNESNEWTONTR, SNESNRICHARDSON
M*/
PETSC_EXTERN PetscErrorCode SNESCreate_Multiblock(SNES snes)
{
  SNES_Multiblock *mb;

  PetscFunctionBegin;
  snes->ops->destroy        = SNESDestroy_Multiblock;
  snes->ops->setup          = SNESSetUp_Multiblock;
  snes->ops->setfromoptions = SNESSetFromOptions_Multiblock;
  snes->ops->view           = SNESView_Multiblock;
  snes->ops->solve          = SNESSolve_Multiblock;
  snes->ops->reset          = SNESReset_Multiblock;

  snes->usesksp = PETSC_FALSE;
  snes->usesnpc = PETSC_FALSE;

  snes->alwayscomputesfinalresidual = PETSC_TRUE;

  CHKERRQ(PetscNewLog(snes,&mb));
  snes->data    = (void*) mb;
  mb->defined   = PETSC_FALSE;
  mb->numBlocks = 0;
  mb->bs        = -1;
  mb->type      = PC_COMPOSITE_MULTIPLICATIVE;

  /* We attach functions so that they can be called on another PC without crashing the program */
  CHKERRQ(PetscObjectComposeFunction((PetscObject) snes, "SNESMultiblockSetFields_C",    SNESMultiblockSetFields_Default));
  CHKERRQ(PetscObjectComposeFunction((PetscObject) snes, "SNESMultiblockSetIS_C",        SNESMultiblockSetIS_Default));
  CHKERRQ(PetscObjectComposeFunction((PetscObject) snes, "SNESMultiblockSetType_C",      SNESMultiblockSetType_Default));
  CHKERRQ(PetscObjectComposeFunction((PetscObject) snes, "SNESMultiblockSetBlockSize_C", SNESMultiblockSetBlockSize_Default));
  CHKERRQ(PetscObjectComposeFunction((PetscObject) snes, "SNESMultiblockGetSubSNES_C",   SNESMultiblockGetSubSNES_Default));
  PetscFunctionReturn(0);
}
