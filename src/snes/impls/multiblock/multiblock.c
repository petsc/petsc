#include <petsc-private/snesimpl.h> /*I "petscsnes.h" I*/
#include <petscdmcomposite.h>

typedef struct _BlockDesc *BlockDesc;
struct _BlockDesc {
  char        *name;    /* Block name */
  PetscInt     nfields; /* If block is defined on a DA, the number of DA fields */
  PetscInt    *fields;  /* If block is defined on a DA, the list of DA fields */
  IS           is;      /* Index sets defining the block */
  VecScatter   sctx;    /* Scatter mapping global Vec to blockVec */
  SNES         snes;    /* Solver for this block */
  Vec          x;
  BlockDesc    next, previous;
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

#undef __FUNCT__
#define __FUNCT__ "SNESReset_Multiblock"
PetscErrorCode SNESReset_Multiblock(SNES snes)
{
  SNES_Multiblock *mb     = (SNES_Multiblock *) snes->data;
  BlockDesc        blocks = mb->blocks, next;
  PetscErrorCode   ierr;

  PetscFunctionBegin;
  while (blocks) {
    ierr = SNESReset(blocks->snes);CHKERRQ(ierr);
#if 0
    ierr = VecDestroy(&blocks->x);CHKERRQ(ierr);
#endif
    ierr = VecScatterDestroy(&blocks->sctx);CHKERRQ(ierr);
    ierr = ISDestroy(&blocks->is);CHKERRQ(ierr);
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
#undef __FUNCT__
#define __FUNCT__ "SNESDestroy_Multiblock"
PetscErrorCode SNESDestroy_Multiblock(SNES snes)
{
  SNES_Multiblock *mb     = (SNES_Multiblock *) snes->data;
  BlockDesc        blocks = mb->blocks, next;
  PetscErrorCode   ierr;

  PetscFunctionBegin;
  ierr = SNESReset_Multiblock(snes);CHKERRQ(ierr);
  while (blocks) {
    next = blocks->next;
    ierr = SNESDestroy(&blocks->snes);CHKERRQ(ierr);
    ierr = PetscFree(blocks->name);CHKERRQ(ierr);
    ierr = PetscFree(blocks->fields);CHKERRQ(ierr);
    ierr = PetscFree(blocks);CHKERRQ(ierr);
    blocks = next;
  }
  ierr = PetscFree(snes->data);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SNESMultiblockSetFieldsRuntime_Private"
/* Precondition: blocksize is set to a meaningful value */
static PetscErrorCode SNESMultiblockSetFieldsRuntime_Private(SNES snes)
{
  SNES_Multiblock *mb = (SNES_Multiblock *) snes->data;
  PetscInt        *ifields;
  PetscInt         i, nfields;
  PetscBool        flg = PETSC_TRUE;
  char             optionname[128], name[8];
  PetscErrorCode   ierr;

  PetscFunctionBegin;
  ierr = PetscMalloc(mb->bs * sizeof(PetscInt), &ifields);CHKERRQ(ierr);
  for(i = 0; ; ++i) {
    ierr = PetscSNPrintf(name, sizeof name, "%D", i);CHKERRQ(ierr);
    ierr = PetscSNPrintf(optionname, sizeof optionname, "-snes_multiblock_%D_fields", i);CHKERRQ(ierr);
    nfields = mb->bs;
    ierr    = PetscOptionsGetIntArray(((PetscObject) snes)->prefix, optionname, ifields, &nfields, &flg);CHKERRQ(ierr);
    if (!flg) break;
    if (!nfields) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_USER, "Cannot list zero fields");
    ierr = SNESMultiblockSetFields(snes, name, nfields, ifields);CHKERRQ(ierr);
  }
  if (i > 0) {
    /* Makes command-line setting of blocks take precedence over setting them in code.
       Otherwise subsequent calls to SNESMultiblockSetIS() or SNESMultiblockSetFields() would
       create new blocks, which would probably not be what the user wanted. */
    mb->defined = PETSC_TRUE;
  }
  ierr = PetscFree(ifields);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SNESMultiblockSetDefaults"
static PetscErrorCode SNESMultiblockSetDefaults(SNES snes)
{
  SNES_Multiblock *mb = (SNES_Multiblock *) snes->data;
  BlockDesc        blocks = mb->blocks;
  PetscInt         i;
  PetscErrorCode   ierr;

  PetscFunctionBegin;
  if (!blocks) {
    if (snes->dm) {
      PetscBool dmcomposite;

      ierr = PetscTypeCompare((PetscObject) snes->dm, DMCOMPOSITE, &dmcomposite);CHKERRQ(ierr);
      if (dmcomposite) {
        PetscInt nDM;
        IS      *fields;

        ierr = PetscInfo(snes,"Setting up physics based multiblock solver using the embedded DM\n");CHKERRQ(ierr);
        ierr = DMCompositeGetNumberDM(snes->dm, &nDM);CHKERRQ(ierr);
        ierr = DMCompositeGetGlobalISs(snes->dm, &fields);CHKERRQ(ierr);
        for(i = 0; i < nDM; ++i) {
          char name[8];

          ierr = PetscSNPrintf(name, sizeof name, "%D", i);CHKERRQ(ierr);
          ierr = SNESMultiblockSetIS(snes, name, fields[i]);CHKERRQ(ierr);
          ierr = ISDestroy(&fields[i]);CHKERRQ(ierr);
        }
        ierr = PetscFree(fields);CHKERRQ(ierr);
      }
    } else {
      PetscBool flg    = PETSC_FALSE;
      PetscBool stokes = PETSC_FALSE;

      if (mb->bs <= 0) {
        if (snes->jacobian_pre) {
          ierr = MatGetBlockSize(snes->jacobian_pre, &mb->bs);CHKERRQ(ierr);
        } else {
          mb->bs = 1;
        }
      }

      ierr = PetscOptionsGetBool(((PetscObject) snes)->prefix, "-snes_multiblock_default", &flg, PETSC_NULL);CHKERRQ(ierr);
      ierr = PetscOptionsGetBool(((PetscObject) snes)->prefix, "-snes_multiblock_detect_saddle_point", &stokes, PETSC_NULL);CHKERRQ(ierr);
      if (stokes) {
        IS       zerodiags, rest;
        PetscInt nmin, nmax;

        ierr = MatGetOwnershipRange(snes->jacobian_pre, &nmin, &nmax);CHKERRQ(ierr);
        ierr = MatFindZeroDiagonals(snes->jacobian_pre, &zerodiags);CHKERRQ(ierr);
        ierr = ISComplement(zerodiags, nmin, nmax, &rest);CHKERRQ(ierr);
        ierr = SNESMultiblockSetIS(snes, "0", rest);CHKERRQ(ierr);
        ierr = SNESMultiblockSetIS(snes, "1", zerodiags);CHKERRQ(ierr);
        ierr = ISDestroy(&zerodiags);CHKERRQ(ierr);
        ierr = ISDestroy(&rest);CHKERRQ(ierr);
      } else {
        if (!flg) {
          /* Allow user to set fields from command line, if bs was known at the time of SNESSetFromOptions_Multiblock()
           then it is set there. This is not ideal because we should only have options set in XXSetFromOptions(). */
          ierr = SNESMultiblockSetFieldsRuntime_Private(snes);CHKERRQ(ierr);
          if (mb->defined) {ierr = PetscInfo(snes, "Blocks defined using the options database\n");CHKERRQ(ierr);}
        }
        if (flg || !mb->defined) {
          ierr = PetscInfo(snes, "Using default splitting of fields\n");CHKERRQ(ierr);
          for(i = 0; i < mb->bs; ++i) {
            char name[8];

            ierr = PetscSNPrintf(name, sizeof name, "%D", i);CHKERRQ(ierr);
            ierr = SNESMultiblockSetFields(snes, name, 1, &i);CHKERRQ(ierr);
          }
          mb->defaultblocks = PETSC_TRUE;
        }
      }
    }
  } else if (mb->numBlocks == 1) {
    if (blocks->is) {
      IS       is2;
      PetscInt nmin, nmax;

      ierr = MatGetOwnershipRange(snes->jacobian_pre, &nmin, &nmax);CHKERRQ(ierr);
      ierr = ISComplement(blocks->is, nmin, nmax, &is2);CHKERRQ(ierr);
      ierr = SNESMultiblockSetIS(snes, "1", is2);CHKERRQ(ierr);
      ierr = ISDestroy(&is2);CHKERRQ(ierr);
    } else {
      SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONGSTATE, "Must provide at least two sets of fields to SNES multiblock");
    }
  }
  if (mb->numBlocks < 2) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Unhandled case, must have at least two blocks");
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
#undef __FUNCT__
#define __FUNCT__ "SNESSetUp_Multiblock"
PetscErrorCode SNESSetUp_Multiblock(SNES snes)
{
  SNES_Multiblock *mb = (SNES_Multiblock *) snes->data;
  BlockDesc        blocks;
  PetscInt         i, numBlocks;
  PetscErrorCode   ierr;

  PetscFunctionBegin;
  /* ierr = SNESDefaultGetWork(snes, 1);CHKERRQ(ierr); */
  ierr = SNESMultiblockSetDefaults(snes);CHKERRQ(ierr);
  numBlocks = mb->numBlocks;
  blocks    = mb->blocks;

  /* Create ISs */
  if (!mb->issetup) {
    PetscInt  ccsize, rstart, rend, nslots, bs;
    PetscBool sorted;

    mb->issetup = PETSC_TRUE;
    bs     = mb->bs;
    ierr   = MatGetOwnershipRange(snes->jacobian_pre, &rstart, &rend);CHKERRQ(ierr);
    ierr   = MatGetLocalSize(snes->jacobian_pre, PETSC_NULL, &ccsize);CHKERRQ(ierr);
    nslots = (rend - rstart)/bs;
    for(i = 0; i < numBlocks; ++i) {
      if (mb->defaultblocks) {
        ierr = ISCreateStride(((PetscObject) snes)->comm, nslots, rstart+i, numBlocks, &blocks->is);CHKERRQ(ierr);
      } else if (!blocks->is) {
        if (blocks->nfields > 1) {
          PetscInt *ii, j, k, nfields = blocks->nfields, *fields = blocks->fields;

          ierr = PetscMalloc(nfields*nslots*sizeof(PetscInt), &ii);CHKERRQ(ierr);
          for(j = 0; j < nslots; ++j) {
            for(k = 0; k < nfields; ++k) {
              ii[nfields*j + k] = rstart + bs*j + fields[k];
            }
          }
          ierr = ISCreateGeneral(((PetscObject) snes)->comm, nslots*nfields, ii, PETSC_OWN_POINTER, &blocks->is);CHKERRQ(ierr);
        } else {
          ierr = ISCreateStride(((PetscObject) snes)->comm, nslots, rstart+blocks->fields[0], bs, &blocks->is);CHKERRQ(ierr);
        }
      }
      ierr = ISSorted(blocks->is, &sorted);CHKERRQ(ierr);
      if (!sorted) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_USER, "Fields must be sorted when creating split");
      blocks = blocks->next;
    }
  }

#if 0
  /* Create matrices */
  ilink  = jac->head;
  if (!jac->pmat) {
    ierr = PetscMalloc(nsplit*sizeof(Mat),&jac->pmat);CHKERRQ(ierr);
    for (i=0; i<nsplit; i++) {
      ierr = MatGetSubMatrix(pc->pmat,ilink->is,ilink->is,MAT_INITIAL_MATRIX,&jac->pmat[i]);CHKERRQ(ierr);
      ilink = ilink->next;
    }
  } else {
    for (i=0; i<nsplit; i++) {
      ierr = MatGetSubMatrix(pc->pmat,ilink->is,ilink->is,MAT_REUSE_MATRIX,&jac->pmat[i]);CHKERRQ(ierr);
      ilink = ilink->next;
    }
  }
  if (jac->realdiagonal) {
    ilink = jac->head;
    if (!jac->mat) {
      ierr = PetscMalloc(nsplit*sizeof(Mat),&jac->mat);CHKERRQ(ierr);
      for (i=0; i<nsplit; i++) {
        ierr = MatGetSubMatrix(pc->mat,ilink->is,ilink->is,MAT_INITIAL_MATRIX,&jac->mat[i]);CHKERRQ(ierr);
        ilink = ilink->next;
      }
    } else {
      for (i=0; i<nsplit; i++) {
        if (jac->mat[i]) {ierr = MatGetSubMatrix(pc->mat,ilink->is,ilink->is,MAT_REUSE_MATRIX,&jac->mat[i]);CHKERRQ(ierr);}
        ilink = ilink->next;
      }
    }
  } else {
    jac->mat = jac->pmat;
  }
#endif

#if 0
  if (jac->type != PC_COMPOSITE_ADDITIVE  && jac->type != PC_COMPOSITE_SCHUR) {
    /* extract the rows of the matrix associated with each field: used for efficient computation of residual inside algorithm */
    ilink  = jac->head;
    if (!jac->Afield) {
      ierr = PetscMalloc(nsplit*sizeof(Mat),&jac->Afield);CHKERRQ(ierr);
      for (i=0; i<nsplit; i++) {
        ierr = MatGetSubMatrix(pc->mat,ilink->is,PETSC_NULL,MAT_INITIAL_MATRIX,&jac->Afield[i]);CHKERRQ(ierr);
        ilink = ilink->next;
      }
    } else {
      for (i=0; i<nsplit; i++) {
        ierr = MatGetSubMatrix(pc->mat,ilink->is,PETSC_NULL,MAT_REUSE_MATRIX,&jac->Afield[i]);CHKERRQ(ierr);
        ilink = ilink->next;
      }
    }
  }
#endif

  if (mb->type == PC_COMPOSITE_SCHUR) {
#if 0
    IS       ccis;
    PetscInt rstart,rend;
    if (nsplit != 2) SETERRQ(((PetscObject)pc)->comm,PETSC_ERR_ARG_INCOMP,"To use Schur complement preconditioner you must have exactly 2 fields");

    /* When extracting off-diagonal submatrices, we take complements from this range */
    ierr  = MatGetOwnershipRangeColumn(pc->mat,&rstart,&rend);CHKERRQ(ierr);

    /* need to handle case when one is resetting up the preconditioner */
    if (jac->schur) {
      ilink = jac->head;
      ierr  = ISComplement(ilink->is,rstart,rend,&ccis);CHKERRQ(ierr);
      ierr  = MatGetSubMatrix(pc->mat,ilink->is,ccis,MAT_REUSE_MATRIX,&jac->B);CHKERRQ(ierr);
      ierr  = ISDestroy(&ccis);CHKERRQ(ierr);
      ilink = ilink->next;
      ierr  = ISComplement(ilink->is,rstart,rend,&ccis);CHKERRQ(ierr);
      ierr  = MatGetSubMatrix(pc->mat,ilink->is,ccis,MAT_REUSE_MATRIX,&jac->C);CHKERRQ(ierr);
      ierr  = ISDestroy(&ccis);CHKERRQ(ierr);
      ierr  = MatSchurComplementUpdate(jac->schur,jac->mat[0],jac->pmat[0],jac->B,jac->C,jac->pmat[1],pc->flag);CHKERRQ(ierr);
      ierr  = KSPSetOperators(jac->kspschur,jac->schur,FieldSplitSchurPre(jac),pc->flag);CHKERRQ(ierr);

     } else {
      KSP ksp;
      char schurprefix[256];

      /* extract the A01 and A10 matrices */
      ilink = jac->head;
      ierr  = ISComplement(ilink->is,rstart,rend,&ccis);CHKERRQ(ierr);
      ierr  = MatGetSubMatrix(pc->mat,ilink->is,ccis,MAT_INITIAL_MATRIX,&jac->B);CHKERRQ(ierr);
      ierr  = ISDestroy(&ccis);CHKERRQ(ierr);
      ilink = ilink->next;
      ierr  = ISComplement(ilink->is,rstart,rend,&ccis);CHKERRQ(ierr);
      ierr  = MatGetSubMatrix(pc->mat,ilink->is,ccis,MAT_INITIAL_MATRIX,&jac->C);CHKERRQ(ierr);
      ierr  = ISDestroy(&ccis);CHKERRQ(ierr);
      /* Use mat[0] (diagonal block of the real matrix) preconditioned by pmat[0] */
      ierr  = MatCreateSchurComplement(jac->mat[0],jac->pmat[0],jac->B,jac->C,jac->mat[1],&jac->schur);CHKERRQ(ierr);
      /* set tabbing and options prefix of KSP inside the MatSchur */
      ierr  = MatSchurComplementGetKSP(jac->schur,&ksp);CHKERRQ(ierr);
      ierr  = PetscObjectIncrementTabLevel((PetscObject)ksp,(PetscObject)pc,2);CHKERRQ(ierr);
      ierr  = PetscSNPrintf(schurprefix,sizeof schurprefix,"%sfieldsplit_%s_",((PetscObject)pc)->prefix?((PetscObject)pc)->prefix:"",jac->head->splitname);CHKERRQ(ierr);
      ierr  = KSPSetOptionsPrefix(ksp,schurprefix);CHKERRQ(ierr);
      ierr  = MatSetFromOptions(jac->schur);CHKERRQ(ierr);

      ierr  = KSPCreate(((PetscObject)pc)->comm,&jac->kspschur);CHKERRQ(ierr);
      ierr  = PetscLogObjectParent((PetscObject)pc,(PetscObject)jac->kspschur);CHKERRQ(ierr);
      ierr  = PetscObjectIncrementTabLevel((PetscObject)jac->kspschur,(PetscObject)pc,1);CHKERRQ(ierr);
      ierr  = KSPSetOperators(jac->kspschur,jac->schur,FieldSplitSchurPre(jac),DIFFERENT_NONZERO_PATTERN);CHKERRQ(ierr);
      if (jac->schurpre == PC_FIELDSPLIT_SCHUR_PRE_SELF) {
        PC pc;
        ierr = KSPGetPC(jac->kspschur,&pc);CHKERRQ(ierr);
        ierr = PCSetType(pc,PCNONE);CHKERRQ(ierr);
        /* Note: This is bad if there exist preconditioners for MATSCHURCOMPLEMENT */
      }
      ierr = PetscSNPrintf(schurprefix,sizeof schurprefix,"%sfieldsplit_%s_",((PetscObject)pc)->prefix?((PetscObject)pc)->prefix:"",ilink->splitname);CHKERRQ(ierr);
      ierr  = KSPSetOptionsPrefix(jac->kspschur,schurprefix);CHKERRQ(ierr);
      /* really want setfromoptions called in PCSetFromOptions_FieldSplit(), but it is not ready yet */
      ierr = KSPSetFromOptions(jac->kspschur);CHKERRQ(ierr);

      ierr = PetscMalloc2(2,Vec,&jac->x,2,Vec,&jac->y);CHKERRQ(ierr);
      ierr = MatGetVecs(jac->pmat[0],&jac->x[0],&jac->y[0]);CHKERRQ(ierr);
      ierr = MatGetVecs(jac->pmat[1],&jac->x[1],&jac->y[1]);CHKERRQ(ierr);
      ilink = jac->head;
      ilink->x = jac->x[0]; ilink->y = jac->y[0];
      ilink = ilink->next;
      ilink->x = jac->x[1]; ilink->y = jac->y[1];
    }
#endif
  } else {
    /* Set up the individual SNESs */
    blocks = mb->blocks;
    i      = 0;
    while (blocks) {
      /*TODO: Set these correctly */
      /*ierr = SNESSetFunction(blocks->snes, blocks->x, func);CHKERRQ(ierr);*/
      /*ierr = SNESSetJacobian(blocks->snes, blocks->x, jac);CHKERRQ(ierr);*/
      ierr = VecDuplicate(blocks->snes->vec_sol, &blocks->x);CHKERRQ(ierr);
      /* really want setfromoptions called in SNESSetFromOptions_Multiblock(), but it is not ready yet */
      ierr = SNESSetFromOptions(blocks->snes);CHKERRQ(ierr);
      ierr = SNESSetUp(blocks->snes);CHKERRQ(ierr);
      blocks = blocks->next;
      i++;
    }
  }

  /* Compute scatter contexts needed by multiplicative versions and non-default splits */
  if (!mb->blocks->sctx) {
    Vec xtmp;

    blocks = mb->blocks;
    ierr = MatGetVecs(snes->jacobian_pre, &xtmp, PETSC_NULL);CHKERRQ(ierr);
    while(blocks) {
      ierr = VecScatterCreate(xtmp, blocks->is, blocks->x, PETSC_NULL, &blocks->sctx);CHKERRQ(ierr);
      blocks = blocks->next;
    }
    ierr = VecDestroy(&xtmp);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

/*
  SNESSetFromOptions_Multiblock - Sets various parameters for the SNESMULTIBLOCK method.

  Input Parameter:
. snes - the SNES context

  Application Interface Routine: SNESSetFromOptions()
*/
#undef __FUNCT__
#define __FUNCT__ "SNESSetFromOptions_Multiblock"
static PetscErrorCode SNESSetFromOptions_Multiblock(SNES snes)
{
  SNES_Multiblock *mb = (SNES_Multiblock *) snes->data;
  PCCompositeType  ctype;
  PetscInt         bs;
  PetscBool        flg;
  PetscErrorCode   ierr;

  PetscFunctionBegin;
  ierr = PetscOptionsHead("SNES Multiblock options");CHKERRQ(ierr);
  ierr = PetscOptionsInt("-snes_multiblock_block_size", "Blocksize that defines number of fields", "PCFieldSplitSetBlockSize", mb->bs, &bs, &flg);CHKERRQ(ierr);
  if (flg) {ierr = SNESMultiblockSetBlockSize(snes, bs);CHKERRQ(ierr);}
  ierr = PetscOptionsEnum("-snes_multiblock_type", "Type of composition", "PCFieldSplitSetType", PCCompositeTypes, (PetscEnum) mb->type, (PetscEnum *) &ctype, &flg);CHKERRQ(ierr);
  if (flg) {
    ierr = SNESMultiblockSetType(snes,ctype);CHKERRQ(ierr);
  }
  /* Only setup fields once */
  if ((mb->bs > 0) && (mb->numBlocks == 0)) {
    /* only allow user to set fields from command line if bs is already known, otherwise user can set them in SNESMultiblockSetDefaults() */
    ierr = SNESMultiblockSetFieldsRuntime_Private(snes);CHKERRQ(ierr);
    if (mb->defined) {ierr = PetscInfo(snes, "Blocks defined using the options database\n");CHKERRQ(ierr);}
  }
  ierr = PetscOptionsTail();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*
  SNESView_Multiblock - Prints info from the SNESMULTIBLOCK data structure.

  Input Parameters:
+ SNES - the SNES context
- viewer - visualization context

  Application Interface Routine: SNESView()
*/
#undef __FUNCT__
#define __FUNCT__ "SNESView_Multiblock"
static PetscErrorCode SNESView_Multiblock(SNES snes, PetscViewer viewer)
{
  SNES_Multiblock *mb = (SNES_Multiblock *) snes->data;
  BlockDesc        blocks = mb->blocks;
  PetscBool        iascii;
  PetscErrorCode   ierr;

  PetscFunctionBegin;
  ierr = PetscTypeCompare((PetscObject) viewer, PETSCVIEWERASCII, &iascii);CHKERRQ(ierr);
  if (iascii) {
    ierr = PetscViewerASCIIPrintf(viewer,"  Multiblock with %s composition: total blocks = %D, blocksize = %D\n", PCCompositeTypes[mb->type], mb->numBlocks, mb->bs);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"  Solver info for each split is in the following SNES objects:\n");CHKERRQ(ierr);
    ierr = PetscViewerASCIIPushTab(viewer);CHKERRQ(ierr);
    while (blocks) {
      if (blocks->fields) {
        PetscInt j;

	ierr = PetscViewerASCIIPrintf(viewer, "Block %s Fields ", blocks->name);CHKERRQ(ierr);
	ierr = PetscViewerASCIIUseTabs(viewer, PETSC_FALSE);CHKERRQ(ierr);
	for(j = 0; j < blocks->nfields; ++j) {
	  if (j > 0) {
	    ierr = PetscViewerASCIIPrintf(viewer, ",");CHKERRQ(ierr);
	  }
	  ierr = PetscViewerASCIIPrintf(viewer, " %D", blocks->fields[j]);CHKERRQ(ierr);
	}
	ierr = PetscViewerASCIIPrintf(viewer, "\n");CHKERRQ(ierr);
        ierr = PetscViewerASCIIUseTabs(viewer, PETSC_TRUE);CHKERRQ(ierr);
      } else {
	ierr = PetscViewerASCIIPrintf(viewer, "Block %s Defined by IS\n", blocks->name);CHKERRQ(ierr);
      }
      ierr = SNESView(blocks->snes, viewer);CHKERRQ(ierr);
      blocks = blocks->next;
    }
    ierr = PetscViewerASCIIPopTab(viewer);CHKERRQ(ierr);
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
#undef __FUNCT__
#define __FUNCT__ "SNESSolve_Multiblock"
PetscErrorCode SNESSolve_Multiblock(SNES snes)
{
  SNES_Multiblock *mb = (SNES_Multiblock *) snes->data;
  Vec              X, Y, F;
  PetscReal        fnorm;
  PetscInt         maxits, i;
  PetscErrorCode   ierr;

  PetscFunctionBegin;
  snes->reason = SNES_CONVERGED_ITERATING;

  maxits = snes->max_its;	 /* maximum number of iterations */
  X      = snes->vec_sol;	 /* X^n */
  Y      = snes->vec_sol_update; /* \tilde X */
  F      = snes->vec_func;       /* residual vector */

  ierr = VecSetBlockSize(X, mb->bs);CHKERRQ(ierr);
  ierr = VecSetBlockSize(Y, mb->bs);CHKERRQ(ierr);
  ierr = VecSetBlockSize(F, mb->bs);CHKERRQ(ierr);
  ierr = PetscObjectTakeAccess(snes);CHKERRQ(ierr);
  snes->iter = 0;
  snes->norm = 0.;
  ierr = PetscObjectGrantAccess(snes);CHKERRQ(ierr);

  if (!snes->vec_func_init_set){
    ierr = SNESComputeFunction(snes, X, F);CHKERRQ(ierr);
    if (snes->domainerror) {
      snes->reason = SNES_DIVERGED_FUNCTION_DOMAIN;
      PetscFunctionReturn(0);
    }
  } else {
    snes->vec_func_init_set = PETSC_FALSE;
  }
  if (!snes->norm_init_set) {
    ierr = VecNorm(F, NORM_2, &fnorm);CHKERRQ(ierr); /* fnorm <- ||F||  */
    if (PetscIsInfOrNanReal(fnorm)) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_FP, "Infinite or not-a-number generated in norm");
  } else {
    fnorm = snes->norm_init;
    snes->norm_init_set = PETSC_FALSE;
  }
  ierr = PetscObjectTakeAccess(snes);CHKERRQ(ierr);
  snes->norm = fnorm;
  ierr = PetscObjectGrantAccess(snes);CHKERRQ(ierr);
  SNESLogConvHistory(snes,fnorm,0);
  ierr = SNESMonitor(snes,0,fnorm);CHKERRQ(ierr);

  /* set parameter for default relative tolerance convergence test */
  snes->ttol = fnorm*snes->rtol;
  /* test convergence */
  ierr = (*snes->ops->converged)(snes,0,0.0,0.0,fnorm,&snes->reason,snes->cnvP);CHKERRQ(ierr);
  if (snes->reason) PetscFunctionReturn(0);

  for(i = 0; i < maxits; i++) {
    /* Call general purpose update function */
    if (snes->ops->update) {
      ierr = (*snes->ops->update)(snes, snes->iter);CHKERRQ(ierr);
    }
    /* Compute X^{new} from subsolves */
    if (mb->type == PC_COMPOSITE_ADDITIVE) {
      BlockDesc blocks = mb->blocks;

      if (mb->defaultblocks) {
        /*TODO: Make an array of Vecs for this */
        /*ierr = VecStrideGatherAll(X, mb->x, INSERT_VALUES);CHKERRQ(ierr);*/
        while (blocks) {
          ierr = SNESSolve(blocks->snes, PETSC_NULL, blocks->x);CHKERRQ(ierr);
          blocks = blocks->next;
        }
        /*ierr = VecStrideScatterAll(mb->x, X, INSERT_VALUES);CHKERRQ(ierr);*/
      } else {
        while (blocks) {
          ierr = VecScatterBegin(blocks->sctx, X, blocks->x, INSERT_VALUES, SCATTER_FORWARD);CHKERRQ(ierr);
          ierr = VecScatterEnd(blocks->sctx, X, blocks->x, INSERT_VALUES, SCATTER_FORWARD);CHKERRQ(ierr);
          ierr = SNESSolve(blocks->snes, PETSC_NULL, blocks->x);CHKERRQ(ierr);
          ierr = VecScatterBegin(blocks->sctx, blocks->x, X, INSERT_VALUES, SCATTER_REVERSE);CHKERRQ(ierr);
          ierr = VecScatterEnd(blocks->sctx, blocks->x, X, INSERT_VALUES, SCATTER_REVERSE);CHKERRQ(ierr);
          blocks = blocks->next;
        }
      }
    } else {
      SETERRQ1(((PetscObject) snes)->comm, PETSC_ERR_SUP, "Unsupported or unknown composition", (int) mb->type);
    }
    CHKMEMQ;
    /* Compute F(X^{new}) */
    ierr = SNESComputeFunction(snes, X, F);CHKERRQ(ierr);
    ierr = VecNorm(F, NORM_2, &fnorm);CHKERRQ(ierr);
    if (PetscIsInfOrNanReal(fnorm)) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_FP,"Infinite or not-a-number generated norm");

    if (snes->nfuncs >= snes->max_funcs) {
      snes->reason = SNES_DIVERGED_FUNCTION_COUNT;
      break;
    }
    if (snes->domainerror) {
      snes->reason = SNES_DIVERGED_FUNCTION_DOMAIN;
      PetscFunctionReturn(0);
    }
    /* Monitor convergence */
    ierr = PetscObjectTakeAccess(snes);CHKERRQ(ierr);
    snes->iter = i+1;
    snes->norm = fnorm;
    ierr = PetscObjectGrantAccess(snes);CHKERRQ(ierr);
    SNESLogConvHistory(snes,snes->norm,0);
    ierr = SNESMonitor(snes,snes->iter,snes->norm);CHKERRQ(ierr);
    /* Test for convergence */
    ierr = (*snes->ops->converged)(snes,snes->iter,0.0,0.0,fnorm,&snes->reason,snes->cnvP);CHKERRQ(ierr);
    if (snes->reason) break;
  }
  if (i == maxits) {
    ierr = PetscInfo1(snes, "Maximum number of iterations has been reached: %D\n", maxits);CHKERRQ(ierr);
    if (!snes->reason) snes->reason = SNES_DIVERGED_MAX_IT;
  }
  PetscFunctionReturn(0);
}

EXTERN_C_BEGIN
#undef __FUNCT__
#define __FUNCT__ "SNESMultiblockSetFields_Default"
PetscErrorCode SNESMultiblockSetFields_Default(SNES snes, const char name[], PetscInt n, const PetscInt fields[])
{
  SNES_Multiblock *mb = (SNES_Multiblock *) snes->data;
  BlockDesc        newblock, next = mb->blocks;
  char             prefix[128];
  PetscInt         i;
  PetscErrorCode   ierr;

  PetscFunctionBegin;
  if (mb->defined) {
    ierr = PetscInfo1(snes, "Ignoring new block \"%s\" because the blocks have already been defined\n", name);CHKERRQ(ierr);
    PetscFunctionReturn(0);
  }
  for(i = 0; i < n; ++i) {
    if (fields[i] >= mb->bs) SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Field %D requested but only %D exist", fields[i], mb->bs);
    if (fields[i] < 0)       SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Negative field %D requested", fields[i]);
  }
  ierr = PetscNew(struct _BlockDesc, &newblock);CHKERRQ(ierr);
  if (name) {
    ierr = PetscStrallocpy(name, &newblock->name);CHKERRQ(ierr);
  } else {
    PetscInt len = floor(log10(mb->numBlocks))+1;

    ierr = PetscMalloc((len+1)*sizeof(char), &newblock->name);CHKERRQ(ierr);
    ierr = PetscSNPrintf(newblock->name, len, "%s", mb->numBlocks);CHKERRQ(ierr);
  }
  newblock->nfields = n;
  ierr = PetscMalloc(n*sizeof(PetscInt), &newblock->fields);CHKERRQ(ierr);
  ierr = PetscMemcpy(newblock->fields, fields, n*sizeof(PetscInt));CHKERRQ(ierr);
  newblock->next    = PETSC_NULL;
  ierr = SNESCreate(((PetscObject) snes)->comm, &newblock->snes);CHKERRQ(ierr);
  ierr = PetscObjectIncrementTabLevel((PetscObject) newblock->snes, (PetscObject) snes, 1);CHKERRQ(ierr);
  ierr = SNESSetType(newblock->snes, SNESNRICHARDSON);CHKERRQ(ierr);
  ierr = PetscLogObjectParent((PetscObject) snes, (PetscObject) newblock->snes);CHKERRQ(ierr);
  ierr = PetscSNPrintf(prefix, sizeof prefix, "%smultiblock_%s_", ((PetscObject) snes)->prefix ? ((PetscObject) snes)->prefix : "", newblock->name);CHKERRQ(ierr);
  ierr = SNESSetOptionsPrefix(newblock->snes, prefix);CHKERRQ(ierr);

  if (!next) {
    mb->blocks         = newblock;
    newblock->previous = PETSC_NULL;
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
EXTERN_C_END

EXTERN_C_BEGIN
#undef __FUNCT__
#define __FUNCT__ "SNESMultiblockSetIS_Default"
PetscErrorCode SNESMultiblockSetIS_Default(SNES snes, const char name[], IS is)
{
  SNES_Multiblock *mb = (SNES_Multiblock *) snes->data;
  BlockDesc        newblock, next = mb->blocks;
  char             prefix[128];
  PetscErrorCode   ierr;

  PetscFunctionBegin;
  if (mb->defined) {
    ierr = PetscInfo1(snes, "Ignoring new block \"%s\" because the blocks have already been defined\n", name);CHKERRQ(ierr);
    PetscFunctionReturn(0);
  }
  ierr = PetscNew(struct _BlockDesc, &newblock);CHKERRQ(ierr);
  if (name) {
    ierr = PetscStrallocpy(name, &newblock->name);CHKERRQ(ierr);
  } else {
    PetscInt len = floor(log10(mb->numBlocks))+1;

    ierr = PetscMalloc((len+1)*sizeof(char), &newblock->name);CHKERRQ(ierr);
    ierr = PetscSNPrintf(newblock->name, len, "%s", mb->numBlocks);CHKERRQ(ierr);
  }
  newblock->is   = is;
  ierr = PetscObjectReference((PetscObject) is);CHKERRQ(ierr);
  newblock->next = PETSC_NULL;
  ierr = SNESCreate(((PetscObject) snes)->comm, &newblock->snes);CHKERRQ(ierr);
  ierr = PetscObjectIncrementTabLevel((PetscObject) newblock->snes, (PetscObject) snes, 1);CHKERRQ(ierr);
  ierr = SNESSetType(newblock->snes, SNESNRICHARDSON);CHKERRQ(ierr);
  ierr = PetscLogObjectParent((PetscObject) snes, (PetscObject) newblock->snes);CHKERRQ(ierr);
  ierr = PetscSNPrintf(prefix, sizeof prefix, "%smultiblock_%s_", ((PetscObject) snes)->prefix ? ((PetscObject) snes)->prefix : "", newblock->name);CHKERRQ(ierr);
  ierr = SNESSetOptionsPrefix(newblock->snes, prefix);CHKERRQ(ierr);

  if (!next) {
    mb->blocks         = newblock;
    newblock->previous = PETSC_NULL;
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
EXTERN_C_END

EXTERN_C_BEGIN
#undef __FUNCT__
#define __FUNCT__ "SNESMultiblockSetBlockSize_Default"
PetscErrorCode  SNESMultiblockSetBlockSize_Default(SNES snes, PetscInt bs)
{
  SNES_Multiblock *mb = (SNES_Multiblock *) snes->data;

  PetscFunctionBegin;
  if (bs < 1) SETERRQ1(((PetscObject) snes)->comm, PETSC_ERR_ARG_OUTOFRANGE, "Blocksize must be positive, you gave %D", bs);
  if (mb->bs > 0 && mb->bs != bs) SETERRQ2(((PetscObject) snes)->comm, PETSC_ERR_ARG_WRONGSTATE, "Cannot change blocksize from %D to %D after it has been set", mb->bs, bs);
  mb->bs = bs;
  PetscFunctionReturn(0);
}
EXTERN_C_END

EXTERN_C_BEGIN
#undef __FUNCT__
#define __FUNCT__ "SNESMultiblockGetSubSNES_Default"
PetscErrorCode SNESMultiblockGetSubSNES_Default(SNES snes, PetscInt *n, SNES **subsnes)
{
  SNES_Multiblock *mb     = (SNES_Multiblock *) snes->data;
  BlockDesc        blocks = mb->blocks;
  PetscInt         cnt    = 0;
  PetscErrorCode   ierr;

  PetscFunctionBegin;
  ierr = PetscMalloc(mb->numBlocks * sizeof(SNES), subsnes);CHKERRQ(ierr);
  while (blocks) {
    (*subsnes)[cnt++] = blocks->snes;
    blocks = blocks->next;
  }
  if (cnt != mb->numBlocks) {
    SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Corrupt SNESMULTIBLOCK object: number of blocks in linked list %D does not match number in object %D", cnt, mb->numBlocks);
  }
  if (n) {*n = mb->numBlocks;}
  PetscFunctionReturn(0);
}
EXTERN_C_END

EXTERN_C_BEGIN
#undef __FUNCT__
#define __FUNCT__ "SNESMultiblockSetType_Default"
PetscErrorCode  SNESMultiblockSetType_Default(SNES snes, PCCompositeType type)
{
  SNES_Multiblock *mb = (SNES_Multiblock *) snes->data;
  PetscErrorCode   ierr;

  PetscFunctionBegin;
  mb->type = type;
  if (type == PC_COMPOSITE_SCHUR) {
#if 1
    SETERRQ(((PetscObject) snes)->comm, PETSC_ERR_SUP, "The Schur composite type is not yet supported");
#else
    snes->ops->solve = SNESSolve_Multiblock_Schur;
    snes->ops->view  = SNESView_Multiblock_Schur;
    ierr = PetscObjectComposeFunctionDynamic((PetscObject) snes, "SNESMultiblockGetSubSNES_C", "SNESMultiblockGetSubSNES_Schur", SNESMultiblockGetSubSNES_Schur);CHKERRQ(ierr);
    ierr = PetscObjectComposeFunctionDynamic((PetscObject) snes, "SNESMultiblockSchurPrecondition_C", "SNESMultiblockSchurPrecondition_Default", SNESMultiblockSchurPrecondition_Default);CHKERRQ(ierr);
#endif
  } else {
    snes->ops->solve = SNESSolve_Multiblock;
    snes->ops->view  = SNESView_Multiblock;
    ierr = PetscObjectComposeFunctionDynamic((PetscObject) snes, "SNESMultiblockGetSubSNES_C", "SNESMultiblockGetSubSNES_Default", SNESMultiblockGetSubSNES_Default);CHKERRQ(ierr);
    ierr = PetscObjectComposeFunctionDynamic((PetscObject) snes, "SNESMultiblockSchurPrecondition_C", "", 0);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}
EXTERN_C_END

#undef __FUNCT__
#define __FUNCT__ "SNESMultiblockSetFields"
/*@
  SNESMultiblockSetFields - Sets the fields for one particular block in the solver

  Logically Collective on SNES

  Input Parameters:
+ snes   - the solver
. name   - name of this block, if PETSC_NULL the number of the block is used
. n      - the number of fields in this block
- fields - the fields in this block

  Level: intermediate

  Notes: Use SNESMultiblockSetIS() to set a completely general set of row indices as a block.

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
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(snes, SNES_CLASSID, 1);
  PetscValidCharPointer(name, 2);
  if (n < 1) SETERRQ2(((PetscObject) snes)->comm, PETSC_ERR_ARG_OUTOFRANGE, "Provided number of fields %D in split \"%s\" not positive", n, name);
  PetscValidIntPointer(fields, 3);
  ierr = PetscTryMethod(snes, "SNESMultiblockSetFields_C", (SNES, const char[], PetscInt, const PetscInt *), (snes, name, n, fields));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SNESMultiblockSetIS"
/*@
  SNESMultiblockSetIS - Sets the global row indices for the block

  Logically Collective on SNES

  Input Parameters:
+ snes - the solver context
. name - name of this block, if PETSC_NULL the number of the block is used
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
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(snes, SNES_CLASSID, 1);
  PetscValidCharPointer(name, 2);
  PetscValidHeaderSpecific(is, IS_CLASSID, 3);
  ierr = PetscTryMethod(snes, "SNESMultiblockSetIS_C", (SNES, const char[], IS), (snes, name, is));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SNESMultiblockSetType"
/*@
  SNESMultiblockSetType - Sets the type of block combination.

  Collective on SNES

  Input Parameters:
+ snes - the solver context
- type - PC_COMPOSITE_ADDITIVE, PC_COMPOSITE_MULTIPLICATIVE (default), PC_COMPOSITE_SYMMETRIC_MULTIPLICATIVE

  Options Database Key:
. -snes_multiblock_type <type: one of multiplicative, additive, symmetric_multiplicative> - Sets block combination type

  Level: Developer

.keywords: SNES, set, type, composite preconditioner, additive, multiplicative
.seealso: PCCompositeSetType()
@*/
PetscErrorCode SNESMultiblockSetType(SNES snes, PCCompositeType type)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(snes, SNES_CLASSID, 1);
  ierr = PetscTryMethod(snes, "SNESMultiblockSetType_C", (SNES, PCCompositeType), (snes, type));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SNESMultiblockSetBlockSize"
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
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(snes, SNES_CLASSID, 1);
  PetscValidLogicalCollectiveInt(snes, bs, 2);
  ierr = PetscTryMethod(snes, "SNESMultiblockSetBlockSize_C", (SNES, PetscInt), (snes,bs));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SNESMultiblockGetSubSNES"
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
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(snes, SNES_CLASSID, 1);
  if (n) PetscValidIntPointer(n, 2);
  ierr = PetscUseMethod(snes, "SNESMultiblockGetSubSNES_C", (SNES, PetscInt*, SNES **), (snes, n, subsnes));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*MC
  SNESMULTIBLOCK - Multiblock nonlinear solver that can use overlapping or nonoverlapping blocks, organized
  additively (Jacobi) or multiplicatively (Gauss-Seidel).

  Level: beginner

.seealso:  SNESCreate(), SNES, SNESSetType(), SNESLS, SNESTR, SNESNRICHARDSON
M*/
EXTERN_C_BEGIN
#undef __FUNCT__
#define __FUNCT__ "SNESCreate_Multiblock"
PetscErrorCode  SNESCreate_Multiblock(SNES snes)
{
  SNES_Multiblock *mb;
  PetscErrorCode   ierr;

  PetscFunctionBegin;
  snes->ops->destroy	    = SNESDestroy_Multiblock;
  snes->ops->setup	    = SNESSetUp_Multiblock;
  snes->ops->setfromoptions = SNESSetFromOptions_Multiblock;
  snes->ops->view           = SNESView_Multiblock;
  snes->ops->solve	    = SNESSolve_Multiblock;
  snes->ops->reset          = SNESReset_Multiblock;

  snes->usesksp             = PETSC_FALSE;

  ierr = PetscNewLog(snes, SNES_Multiblock, &mb);CHKERRQ(ierr);
  snes->data = (void*) mb;
  mb->defined   = PETSC_FALSE;
  mb->numBlocks = 0;
  mb->bs        = -1;
  mb->type      = PC_COMPOSITE_MULTIPLICATIVE;

  /* We attach functions so that they can be called on another PC without crashing the program */
  ierr = PetscObjectComposeFunctionDynamic((PetscObject) snes, "SNESMultiblockSetFields_C",    "SNESMultiblockSetFields_Default",    SNESMultiblockSetFields_Default);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject) snes, "SNESMultiblockSetIS_C",        "SNESMultiblockSetIS_Default",        SNESMultiblockSetIS_Default);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject) snes, "SNESMultiblockSetType_C",      "SNESMultiblockSetType_Default",      SNESMultiblockSetType_Default);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject) snes, "SNESMultiblockSetBlockSize_C", "SNESMultiblockSetBlockSize_Default", SNESMultiblockSetBlockSize_Default);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject) snes, "SNESMultiblockGetSubSNES_C",   "SNESMultiblockGetSubSNES_Default",   SNESMultiblockGetSubSNES_Default);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
EXTERN_C_END
