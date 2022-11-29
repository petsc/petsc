#include <../src/ksp/pc/impls/gamg/gamg.h> /*I "petscpc.h" I*/
#include <petscsf.h>

PetscFunctionList PCGAMGClassicalProlongatorList    = NULL;
PetscBool         PCGAMGClassicalPackageInitialized = PETSC_FALSE;

typedef struct {
  PetscReal interp_threshold; /* interpolation threshold */
  char      prolongtype[256];
  PetscInt  nsmooths; /* number of jacobi smoothings on the prolongator */
} PC_GAMG_Classical;

/*@C
   PCGAMGClassicalSetType - Sets the type of classical interpolation to use with `PCGAMG`

   Collective on pc

   Input Parameters:
.  pc - the preconditioner context

   Options Database Key:
.  -pc_gamg_classical_type <direct,standard> - set type of classical AMG prolongation

   Level: intermediate

.seealso: `PCGAMG`
@*/
PetscErrorCode PCGAMGClassicalSetType(PC pc, PCGAMGClassicalType type)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc, PC_CLASSID, 1);
  PetscTryMethod(pc, "PCGAMGClassicalSetType_C", (PC, PCGAMGClassicalType), (pc, type));
  PetscFunctionReturn(0);
}

/*@C
   PCGAMGClassicalGetType - Gets the type of classical interpolation to use with `PCGAMG`

   Collective on pc

   Input Parameter:
.  pc - the preconditioner context

   Output Parameter:
.  type - the type used

   Level: intermediate

.seealso: `PCGAMG`
@*/
PetscErrorCode PCGAMGClassicalGetType(PC pc, PCGAMGClassicalType *type)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc, PC_CLASSID, 1);
  PetscUseMethod(pc, "PCGAMGClassicalGetType_C", (PC, PCGAMGClassicalType *), (pc, type));
  PetscFunctionReturn(0);
}

static PetscErrorCode PCGAMGClassicalSetType_GAMG(PC pc, PCGAMGClassicalType type)
{
  PC_MG             *mg      = (PC_MG *)pc->data;
  PC_GAMG           *pc_gamg = (PC_GAMG *)mg->innerctx;
  PC_GAMG_Classical *cls     = (PC_GAMG_Classical *)pc_gamg->subctx;

  PetscFunctionBegin;
  PetscCall(PetscStrcpy(cls->prolongtype, type));
  PetscFunctionReturn(0);
}

static PetscErrorCode PCGAMGClassicalGetType_GAMG(PC pc, PCGAMGClassicalType *type)
{
  PC_MG             *mg      = (PC_MG *)pc->data;
  PC_GAMG           *pc_gamg = (PC_GAMG *)mg->innerctx;
  PC_GAMG_Classical *cls     = (PC_GAMG_Classical *)pc_gamg->subctx;

  PetscFunctionBegin;
  *type = cls->prolongtype;
  PetscFunctionReturn(0);
}

PetscErrorCode PCGAMGCreateGraph_Classical(PC pc, Mat A, Mat *G)
{
  PetscInt           s, f, n, idx, lidx, gidx;
  PetscInt           r, c, ncols;
  const PetscInt    *rcol;
  const PetscScalar *rval;
  PetscInt          *gcol;
  PetscScalar       *gval;
  PetscReal          rmax;
  PetscInt           cmax = 0;
  PC_MG             *mg   = (PC_MG *)pc->data;
  PC_GAMG           *gamg = (PC_GAMG *)mg->innerctx;
  PetscInt          *gsparse, *lsparse;
  PetscScalar       *Amax;
  MatType            mtype;

  PetscFunctionBegin;
  PetscCall(MatGetOwnershipRange(A, &s, &f));
  n = f - s;
  PetscCall(PetscMalloc3(n, &lsparse, n, &gsparse, n, &Amax));

  for (r = 0; r < n; r++) {
    lsparse[r] = 0;
    gsparse[r] = 0;
  }

  for (r = s; r < f; r++) {
    /* determine the maximum off-diagonal in each row */
    rmax = 0.;
    PetscCall(MatGetRow(A, r, &ncols, &rcol, &rval));
    for (c = 0; c < ncols; c++) {
      if (PetscRealPart(-rval[c]) > rmax && rcol[c] != r) rmax = PetscRealPart(-rval[c]);
    }
    Amax[r - s] = rmax;
    if (ncols > cmax) cmax = ncols;
    lidx = 0;
    gidx = 0;
    /* create the local and global sparsity patterns */
    for (c = 0; c < ncols; c++) {
      if (PetscRealPart(-rval[c]) > gamg->threshold[0] * PetscRealPart(Amax[r - s]) || rcol[c] == r) {
        if (rcol[c] < f && rcol[c] >= s) {
          lidx++;
        } else {
          gidx++;
        }
      }
    }
    PetscCall(MatRestoreRow(A, r, &ncols, &rcol, &rval));
    lsparse[r - s] = lidx;
    gsparse[r - s] = gidx;
  }
  PetscCall(PetscMalloc2(cmax, &gval, cmax, &gcol));

  PetscCall(MatCreate(PetscObjectComm((PetscObject)A), G));
  PetscCall(MatGetType(A, &mtype));
  PetscCall(MatSetType(*G, mtype));
  PetscCall(MatSetSizes(*G, n, n, PETSC_DETERMINE, PETSC_DETERMINE));
  PetscCall(MatMPIAIJSetPreallocation(*G, 0, lsparse, 0, gsparse));
  PetscCall(MatSeqAIJSetPreallocation(*G, 0, lsparse));
  for (r = s; r < f; r++) {
    PetscCall(MatGetRow(A, r, &ncols, &rcol, &rval));
    idx = 0;
    for (c = 0; c < ncols; c++) {
      /* classical strength of connection */
      if (PetscRealPart(-rval[c]) > gamg->threshold[0] * PetscRealPart(Amax[r - s]) || rcol[c] == r) {
        gcol[idx] = rcol[c];
        gval[idx] = rval[c];
        idx++;
      }
    }
    PetscCall(MatSetValues(*G, 1, &r, idx, gcol, gval, INSERT_VALUES));
    PetscCall(MatRestoreRow(A, r, &ncols, &rcol, &rval));
  }
  PetscCall(MatAssemblyBegin(*G, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(*G, MAT_FINAL_ASSEMBLY));

  PetscCall(PetscFree2(gval, gcol));
  PetscCall(PetscFree3(lsparse, gsparse, Amax));
  PetscFunctionReturn(0);
}

PetscErrorCode PCGAMGCoarsen_Classical(PC pc, Mat *G, PetscCoarsenData **agg_lists)
{
  MatCoarsen crs;
  MPI_Comm   fcomm = ((PetscObject)pc)->comm;

  PetscFunctionBegin;
  PetscCheck(G, fcomm, PETSC_ERR_ARG_WRONGSTATE, "Must set Graph in PC in PCGAMG before coarsening");

  PetscCall(MatCoarsenCreate(fcomm, &crs));
  PetscCall(MatCoarsenSetFromOptions(crs));
  PetscCall(MatCoarsenSetAdjacency(crs, *G));
  PetscCall(MatCoarsenSetStrictAggs(crs, PETSC_TRUE));
  PetscCall(MatCoarsenApply(crs));
  PetscCall(MatCoarsenGetData(crs, agg_lists));
  PetscCall(MatCoarsenDestroy(&crs));
  PetscFunctionReturn(0);
}

PetscErrorCode PCGAMGProlongator_Classical_Direct(PC pc, Mat A, Mat G, PetscCoarsenData *agg_lists, Mat *P)
{
  PC_MG             *mg   = (PC_MG *)pc->data;
  PC_GAMG           *gamg = (PC_GAMG *)mg->innerctx;
  PetscBool          iscoarse, isMPIAIJ, isSEQAIJ;
  PetscInt           fn, cn, fs, fe, cs, ce, i, j, ncols, col, row_f, row_c, cmax = 0, idx, noff;
  PetscInt          *lcid, *gcid, *lsparse, *gsparse, *colmap, *pcols;
  const PetscInt    *rcol;
  PetscReal         *Amax_pos, *Amax_neg;
  PetscScalar        g_pos, g_neg, a_pos, a_neg, diag, invdiag, alpha, beta, pij;
  PetscScalar       *pvals;
  const PetscScalar *rval;
  Mat                lA, gA = NULL;
  MatType            mtype;
  Vec                C, lvec;
  PetscLayout        clayout;
  PetscSF            sf;
  Mat_MPIAIJ        *mpiaij;

  PetscFunctionBegin;
  PetscCall(MatGetOwnershipRange(A, &fs, &fe));
  fn = fe - fs;
  PetscCall(PetscObjectTypeCompare((PetscObject)A, MATMPIAIJ, &isMPIAIJ));
  PetscCall(PetscObjectTypeCompare((PetscObject)A, MATSEQAIJ, &isSEQAIJ));
  PetscCheck(isMPIAIJ || isSEQAIJ, PetscObjectComm((PetscObject)A), PETSC_ERR_ARG_WRONG, "Classical AMG requires MPIAIJ matrix");
  if (isMPIAIJ) {
    mpiaij = (Mat_MPIAIJ *)A->data;
    lA     = mpiaij->A;
    gA     = mpiaij->B;
    lvec   = mpiaij->lvec;
    PetscCall(VecGetSize(lvec, &noff));
    colmap = mpiaij->garray;
    PetscCall(MatGetLayouts(A, NULL, &clayout));
    PetscCall(PetscSFCreate(PetscObjectComm((PetscObject)A), &sf));
    PetscCall(PetscSFSetGraphLayout(sf, clayout, noff, NULL, PETSC_COPY_VALUES, colmap));
    PetscCall(PetscMalloc1(noff, &gcid));
  } else {
    lA = A;
  }
  PetscCall(PetscMalloc5(fn, &lsparse, fn, &gsparse, fn, &lcid, fn, &Amax_pos, fn, &Amax_neg));

  /* count the number of coarse unknowns */
  cn = 0;
  for (i = 0; i < fn; i++) {
    /* filter out singletons */
    PetscCall(PetscCDEmptyAt(agg_lists, i, &iscoarse));
    lcid[i] = -1;
    if (!iscoarse) cn++;
  }

  /* create the coarse vector */
  PetscCall(VecCreateMPI(PetscObjectComm((PetscObject)A), cn, PETSC_DECIDE, &C));
  PetscCall(VecGetOwnershipRange(C, &cs, &ce));

  cn = 0;
  for (i = 0; i < fn; i++) {
    PetscCall(PetscCDEmptyAt(agg_lists, i, &iscoarse));
    if (!iscoarse) {
      lcid[i] = cs + cn;
      cn++;
    } else {
      lcid[i] = -1;
    }
  }

  if (gA) {
    PetscCall(PetscSFBcastBegin(sf, MPIU_INT, lcid, gcid, MPI_REPLACE));
    PetscCall(PetscSFBcastEnd(sf, MPIU_INT, lcid, gcid, MPI_REPLACE));
  }

  /* determine the largest off-diagonal entries in each row */
  for (i = fs; i < fe; i++) {
    Amax_pos[i - fs] = 0.;
    Amax_neg[i - fs] = 0.;
    PetscCall(MatGetRow(A, i, &ncols, &rcol, &rval));
    for (j = 0; j < ncols; j++) {
      if ((PetscRealPart(-rval[j]) > Amax_neg[i - fs]) && i != rcol[j]) Amax_neg[i - fs] = PetscAbsScalar(rval[j]);
      if ((PetscRealPart(rval[j]) > Amax_pos[i - fs]) && i != rcol[j]) Amax_pos[i - fs] = PetscAbsScalar(rval[j]);
    }
    if (ncols > cmax) cmax = ncols;
    PetscCall(MatRestoreRow(A, i, &ncols, &rcol, &rval));
  }
  PetscCall(PetscMalloc2(cmax, &pcols, cmax, &pvals));
  PetscCall(VecDestroy(&C));

  /* count the on and off processor sparsity patterns for the prolongator */
  for (i = 0; i < fn; i++) {
    /* on */
    lsparse[i] = 0;
    gsparse[i] = 0;
    if (lcid[i] >= 0) {
      lsparse[i] = 1;
      gsparse[i] = 0;
    } else {
      PetscCall(MatGetRow(lA, i, &ncols, &rcol, &rval));
      for (j = 0; j < ncols; j++) {
        col = rcol[j];
        if (lcid[col] >= 0 && (PetscRealPart(rval[j]) > gamg->threshold[0] * Amax_pos[i] || PetscRealPart(-rval[j]) > gamg->threshold[0] * Amax_neg[i])) lsparse[i] += 1;
      }
      PetscCall(MatRestoreRow(lA, i, &ncols, &rcol, &rval));
      /* off */
      if (gA) {
        PetscCall(MatGetRow(gA, i, &ncols, &rcol, &rval));
        for (j = 0; j < ncols; j++) {
          col = rcol[j];
          if (gcid[col] >= 0 && (PetscRealPart(rval[j]) > gamg->threshold[0] * Amax_pos[i] || PetscRealPart(-rval[j]) > gamg->threshold[0] * Amax_neg[i])) gsparse[i] += 1;
        }
        PetscCall(MatRestoreRow(gA, i, &ncols, &rcol, &rval));
      }
    }
  }

  /* preallocate and create the prolongator */
  PetscCall(MatCreate(PetscObjectComm((PetscObject)A), P));
  PetscCall(MatGetType(G, &mtype));
  PetscCall(MatSetType(*P, mtype));
  PetscCall(MatSetSizes(*P, fn, cn, PETSC_DETERMINE, PETSC_DETERMINE));
  PetscCall(MatMPIAIJSetPreallocation(*P, 0, lsparse, 0, gsparse));
  PetscCall(MatSeqAIJSetPreallocation(*P, 0, lsparse));

  /* loop over local fine nodes -- get the diagonal, the sum of positive and negative strong and weak weights, and set up the row */
  for (i = 0; i < fn; i++) {
    /* determine on or off */
    row_f = i + fs;
    row_c = lcid[i];
    if (row_c >= 0) {
      pij = 1.;
      PetscCall(MatSetValues(*P, 1, &row_f, 1, &row_c, &pij, INSERT_VALUES));
    } else {
      g_pos = 0.;
      g_neg = 0.;
      a_pos = 0.;
      a_neg = 0.;
      diag  = 0.;

      /* local connections */
      PetscCall(MatGetRow(lA, i, &ncols, &rcol, &rval));
      for (j = 0; j < ncols; j++) {
        col = rcol[j];
        if (lcid[col] >= 0 && (PetscRealPart(rval[j]) > gamg->threshold[0] * Amax_pos[i] || PetscRealPart(-rval[j]) > gamg->threshold[0] * Amax_neg[i])) {
          if (PetscRealPart(rval[j]) > 0.) {
            g_pos += rval[j];
          } else {
            g_neg += rval[j];
          }
        }
        if (col != i) {
          if (PetscRealPart(rval[j]) > 0.) {
            a_pos += rval[j];
          } else {
            a_neg += rval[j];
          }
        } else {
          diag = rval[j];
        }
      }
      PetscCall(MatRestoreRow(lA, i, &ncols, &rcol, &rval));

      /* ghosted connections */
      if (gA) {
        PetscCall(MatGetRow(gA, i, &ncols, &rcol, &rval));
        for (j = 0; j < ncols; j++) {
          col = rcol[j];
          if (gcid[col] >= 0 && (PetscRealPart(rval[j]) > gamg->threshold[0] * Amax_pos[i] || PetscRealPart(-rval[j]) > gamg->threshold[0] * Amax_neg[i])) {
            if (PetscRealPart(rval[j]) > 0.) {
              g_pos += rval[j];
            } else {
              g_neg += rval[j];
            }
          }
          if (PetscRealPart(rval[j]) > 0.) {
            a_pos += rval[j];
          } else {
            a_neg += rval[j];
          }
        }
        PetscCall(MatRestoreRow(gA, i, &ncols, &rcol, &rval));
      }

      if (g_neg == 0.) {
        alpha = 0.;
      } else {
        alpha = -a_neg / g_neg;
      }

      if (g_pos == 0.) {
        diag += a_pos;
        beta = 0.;
      } else {
        beta = -a_pos / g_pos;
      }
      if (diag == 0.) {
        invdiag = 0.;
      } else invdiag = 1. / diag;
      /* on */
      PetscCall(MatGetRow(lA, i, &ncols, &rcol, &rval));
      idx = 0;
      for (j = 0; j < ncols; j++) {
        col = rcol[j];
        if (lcid[col] >= 0 && (PetscRealPart(rval[j]) > gamg->threshold[0] * Amax_pos[i] || PetscRealPart(-rval[j]) > gamg->threshold[0] * Amax_neg[i])) {
          row_f = i + fs;
          row_c = lcid[col];
          /* set the values for on-processor ones */
          if (PetscRealPart(rval[j]) < 0.) {
            pij = rval[j] * alpha * invdiag;
          } else {
            pij = rval[j] * beta * invdiag;
          }
          if (PetscAbsScalar(pij) != 0.) {
            pvals[idx] = pij;
            pcols[idx] = row_c;
            idx++;
          }
        }
      }
      PetscCall(MatRestoreRow(lA, i, &ncols, &rcol, &rval));
      /* off */
      if (gA) {
        PetscCall(MatGetRow(gA, i, &ncols, &rcol, &rval));
        for (j = 0; j < ncols; j++) {
          col = rcol[j];
          if (gcid[col] >= 0 && (PetscRealPart(rval[j]) > gamg->threshold[0] * Amax_pos[i] || PetscRealPart(-rval[j]) > gamg->threshold[0] * Amax_neg[i])) {
            row_f = i + fs;
            row_c = gcid[col];
            /* set the values for on-processor ones */
            if (PetscRealPart(rval[j]) < 0.) {
              pij = rval[j] * alpha * invdiag;
            } else {
              pij = rval[j] * beta * invdiag;
            }
            if (PetscAbsScalar(pij) != 0.) {
              pvals[idx] = pij;
              pcols[idx] = row_c;
              idx++;
            }
          }
        }
        PetscCall(MatRestoreRow(gA, i, &ncols, &rcol, &rval));
      }
      PetscCall(MatSetValues(*P, 1, &row_f, idx, pcols, pvals, INSERT_VALUES));
    }
  }

  PetscCall(MatAssemblyBegin(*P, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(*P, MAT_FINAL_ASSEMBLY));

  PetscCall(PetscFree5(lsparse, gsparse, lcid, Amax_pos, Amax_neg));

  PetscCall(PetscFree2(pcols, pvals));
  if (gA) {
    PetscCall(PetscSFDestroy(&sf));
    PetscCall(PetscFree(gcid));
  }
  PetscFunctionReturn(0);
}

PetscErrorCode PCGAMGTruncateProlongator_Private(PC pc, Mat *P)
{
  PetscInt           j, i, ps, pf, pn, pcs, pcf, pcn, idx, cmax;
  const PetscScalar *pval;
  const PetscInt    *pcol;
  PetscScalar       *pnval;
  PetscInt          *pncol;
  PetscInt           ncols;
  Mat                Pnew;
  PetscInt          *lsparse, *gsparse;
  PetscReal          pmax_pos, pmax_neg, ptot_pos, ptot_neg, pthresh_pos, pthresh_neg;
  PC_MG             *mg      = (PC_MG *)pc->data;
  PC_GAMG           *pc_gamg = (PC_GAMG *)mg->innerctx;
  PC_GAMG_Classical *cls     = (PC_GAMG_Classical *)pc_gamg->subctx;
  MatType            mtype;

  PetscFunctionBegin;
  /* trim and rescale with reallocation */
  PetscCall(MatGetOwnershipRange(*P, &ps, &pf));
  PetscCall(MatGetOwnershipRangeColumn(*P, &pcs, &pcf));
  pn  = pf - ps;
  pcn = pcf - pcs;
  PetscCall(PetscMalloc2(pn, &lsparse, pn, &gsparse));
  /* allocate */
  cmax = 0;
  for (i = ps; i < pf; i++) {
    lsparse[i - ps] = 0;
    gsparse[i - ps] = 0;
    PetscCall(MatGetRow(*P, i, &ncols, &pcol, &pval));
    if (ncols > cmax) cmax = ncols;
    pmax_pos = 0.;
    pmax_neg = 0.;
    for (j = 0; j < ncols; j++) {
      if (PetscRealPart(pval[j]) > pmax_pos) {
        pmax_pos = PetscRealPart(pval[j]);
      } else if (PetscRealPart(pval[j]) < pmax_neg) {
        pmax_neg = PetscRealPart(pval[j]);
      }
    }
    for (j = 0; j < ncols; j++) {
      if (PetscRealPart(pval[j]) >= pmax_pos * cls->interp_threshold || PetscRealPart(pval[j]) <= pmax_neg * cls->interp_threshold) {
        if (pcol[j] >= pcs && pcol[j] < pcf) {
          lsparse[i - ps]++;
        } else {
          gsparse[i - ps]++;
        }
      }
    }
    PetscCall(MatRestoreRow(*P, i, &ncols, &pcol, &pval));
  }

  PetscCall(PetscMalloc2(cmax, &pnval, cmax, &pncol));

  PetscCall(MatGetType(*P, &mtype));
  PetscCall(MatCreate(PetscObjectComm((PetscObject)*P), &Pnew));
  PetscCall(MatSetType(Pnew, mtype));
  PetscCall(MatSetSizes(Pnew, pn, pcn, PETSC_DETERMINE, PETSC_DETERMINE));
  PetscCall(MatSeqAIJSetPreallocation(Pnew, 0, lsparse));
  PetscCall(MatMPIAIJSetPreallocation(Pnew, 0, lsparse, 0, gsparse));

  for (i = ps; i < pf; i++) {
    PetscCall(MatGetRow(*P, i, &ncols, &pcol, &pval));
    pmax_pos = 0.;
    pmax_neg = 0.;
    for (j = 0; j < ncols; j++) {
      if (PetscRealPart(pval[j]) > pmax_pos) {
        pmax_pos = PetscRealPart(pval[j]);
      } else if (PetscRealPart(pval[j]) < pmax_neg) {
        pmax_neg = PetscRealPart(pval[j]);
      }
    }
    pthresh_pos = 0.;
    pthresh_neg = 0.;
    ptot_pos    = 0.;
    ptot_neg    = 0.;
    for (j = 0; j < ncols; j++) {
      if (PetscRealPart(pval[j]) >= cls->interp_threshold * pmax_pos) {
        pthresh_pos += PetscRealPart(pval[j]);
      } else if (PetscRealPart(pval[j]) <= cls->interp_threshold * pmax_neg) {
        pthresh_neg += PetscRealPart(pval[j]);
      }
      if (PetscRealPart(pval[j]) > 0.) {
        ptot_pos += PetscRealPart(pval[j]);
      } else {
        ptot_neg += PetscRealPart(pval[j]);
      }
    }
    if (PetscAbsReal(pthresh_pos) > 0.) ptot_pos /= pthresh_pos;
    if (PetscAbsReal(pthresh_neg) > 0.) ptot_neg /= pthresh_neg;
    idx = 0;
    for (j = 0; j < ncols; j++) {
      if (PetscRealPart(pval[j]) >= pmax_pos * cls->interp_threshold) {
        pnval[idx] = ptot_pos * pval[j];
        pncol[idx] = pcol[j];
        idx++;
      } else if (PetscRealPart(pval[j]) <= pmax_neg * cls->interp_threshold) {
        pnval[idx] = ptot_neg * pval[j];
        pncol[idx] = pcol[j];
        idx++;
      }
    }
    PetscCall(MatRestoreRow(*P, i, &ncols, &pcol, &pval));
    PetscCall(MatSetValues(Pnew, 1, &i, idx, pncol, pnval, INSERT_VALUES));
  }

  PetscCall(MatAssemblyBegin(Pnew, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(Pnew, MAT_FINAL_ASSEMBLY));
  PetscCall(MatDestroy(P));

  *P = Pnew;
  PetscCall(PetscFree2(lsparse, gsparse));
  PetscCall(PetscFree2(pnval, pncol));
  PetscFunctionReturn(0);
}

PetscErrorCode PCGAMGProlongator_Classical_Standard(PC pc, Mat A, Mat G, PetscCoarsenData *agg_lists, Mat *P)
{
  Mat                lA, *lAs;
  MatType            mtype;
  Vec                cv;
  PetscInt          *gcid, *lcid, *lsparse, *gsparse, *picol;
  PetscInt           fs, fe, cs, ce, nl, i, j, k, li, lni, ci, ncols, maxcols, fn, cn, cid;
  PetscMPIInt        size;
  const PetscInt    *lidx, *icol, *gidx;
  PetscBool          iscoarse;
  PetscScalar        vi, pentry, pjentry;
  PetscScalar       *pcontrib, *pvcol;
  const PetscScalar *vcol;
  PetscReal          diag, jdiag, jwttotal;
  PetscInt           pncols;
  PetscSF            sf;
  PetscLayout        clayout;
  IS                 lis;

  PetscFunctionBegin;
  PetscCallMPI(MPI_Comm_size(PetscObjectComm((PetscObject)A), &size));
  PetscCall(MatGetOwnershipRange(A, &fs, &fe));
  fn = fe - fs;
  PetscCall(ISCreateStride(PETSC_COMM_SELF, fe - fs, fs, 1, &lis));
  if (size > 1) {
    PetscCall(MatGetLayouts(A, NULL, &clayout));
    /* increase the overlap by two to get neighbors of neighbors */
    PetscCall(MatIncreaseOverlap(A, 1, &lis, 2));
    PetscCall(ISSort(lis));
    /* get the local part of A */
    PetscCall(MatCreateSubMatrices(A, 1, &lis, &lis, MAT_INITIAL_MATRIX, &lAs));
    lA = lAs[0];
    /* build an SF out of it */
    PetscCall(ISGetLocalSize(lis, &nl));
    PetscCall(PetscSFCreate(PetscObjectComm((PetscObject)A), &sf));
    PetscCall(ISGetIndices(lis, &lidx));
    PetscCall(PetscSFSetGraphLayout(sf, clayout, nl, NULL, PETSC_COPY_VALUES, lidx));
    PetscCall(ISRestoreIndices(lis, &lidx));
  } else {
    lA = A;
    nl = fn;
  }
  /* create a communication structure for the overlapped portion and transmit coarse indices */
  PetscCall(PetscMalloc3(fn, &lsparse, fn, &gsparse, nl, &pcontrib));
  /* create coarse vector */
  cn = 0;
  for (i = 0; i < fn; i++) {
    PetscCall(PetscCDEmptyAt(agg_lists, i, &iscoarse));
    if (!iscoarse) cn++;
  }
  PetscCall(PetscMalloc1(fn, &gcid));
  PetscCall(VecCreateMPI(PetscObjectComm((PetscObject)A), cn, PETSC_DECIDE, &cv));
  PetscCall(VecGetOwnershipRange(cv, &cs, &ce));
  cn = 0;
  for (i = 0; i < fn; i++) {
    PetscCall(PetscCDEmptyAt(agg_lists, i, &iscoarse));
    if (!iscoarse) {
      gcid[i] = cs + cn;
      cn++;
    } else {
      gcid[i] = -1;
    }
  }
  if (size > 1) {
    PetscCall(PetscMalloc1(nl, &lcid));
    PetscCall(PetscSFBcastBegin(sf, MPIU_INT, gcid, lcid, MPI_REPLACE));
    PetscCall(PetscSFBcastEnd(sf, MPIU_INT, gcid, lcid, MPI_REPLACE));
  } else {
    lcid = gcid;
  }
  /* count to preallocate the prolongator */
  PetscCall(ISGetIndices(lis, &gidx));
  maxcols = 0;
  /* count the number of unique contributing coarse cells for each fine */
  for (i = 0; i < nl; i++) {
    pcontrib[i] = 0.;
    PetscCall(MatGetRow(lA, i, &ncols, &icol, NULL));
    if (gidx[i] >= fs && gidx[i] < fe) {
      li          = gidx[i] - fs;
      lsparse[li] = 0;
      gsparse[li] = 0;
      cid         = lcid[i];
      if (cid >= 0) {
        lsparse[li] = 1;
      } else {
        for (j = 0; j < ncols; j++) {
          if (lcid[icol[j]] >= 0) {
            pcontrib[icol[j]] = 1.;
          } else {
            ci = icol[j];
            PetscCall(MatRestoreRow(lA, i, &ncols, &icol, NULL));
            PetscCall(MatGetRow(lA, ci, &ncols, &icol, NULL));
            for (k = 0; k < ncols; k++) {
              if (lcid[icol[k]] >= 0) pcontrib[icol[k]] = 1.;
            }
            PetscCall(MatRestoreRow(lA, ci, &ncols, &icol, NULL));
            PetscCall(MatGetRow(lA, i, &ncols, &icol, NULL));
          }
        }
        for (j = 0; j < ncols; j++) {
          if (lcid[icol[j]] >= 0 && pcontrib[icol[j]] != 0.) {
            lni = lcid[icol[j]];
            if (lni >= cs && lni < ce) {
              lsparse[li]++;
            } else {
              gsparse[li]++;
            }
            pcontrib[icol[j]] = 0.;
          } else {
            ci = icol[j];
            PetscCall(MatRestoreRow(lA, i, &ncols, &icol, NULL));
            PetscCall(MatGetRow(lA, ci, &ncols, &icol, NULL));
            for (k = 0; k < ncols; k++) {
              if (lcid[icol[k]] >= 0 && pcontrib[icol[k]] != 0.) {
                lni = lcid[icol[k]];
                if (lni >= cs && lni < ce) {
                  lsparse[li]++;
                } else {
                  gsparse[li]++;
                }
                pcontrib[icol[k]] = 0.;
              }
            }
            PetscCall(MatRestoreRow(lA, ci, &ncols, &icol, NULL));
            PetscCall(MatGetRow(lA, i, &ncols, &icol, NULL));
          }
        }
      }
      if (lsparse[li] + gsparse[li] > maxcols) maxcols = lsparse[li] + gsparse[li];
    }
    PetscCall(MatRestoreRow(lA, i, &ncols, &icol, &vcol));
  }
  PetscCall(PetscMalloc2(maxcols, &picol, maxcols, &pvcol));
  PetscCall(MatCreate(PetscObjectComm((PetscObject)A), P));
  PetscCall(MatGetType(A, &mtype));
  PetscCall(MatSetType(*P, mtype));
  PetscCall(MatSetSizes(*P, fn, cn, PETSC_DETERMINE, PETSC_DETERMINE));
  PetscCall(MatMPIAIJSetPreallocation(*P, 0, lsparse, 0, gsparse));
  PetscCall(MatSeqAIJSetPreallocation(*P, 0, lsparse));
  for (i = 0; i < nl; i++) {
    diag = 0.;
    if (gidx[i] >= fs && gidx[i] < fe) {
      pncols = 0;
      cid    = lcid[i];
      if (cid >= 0) {
        pncols   = 1;
        picol[0] = cid;
        pvcol[0] = 1.;
      } else {
        PetscCall(MatGetRow(lA, i, &ncols, &icol, &vcol));
        for (j = 0; j < ncols; j++) {
          pentry = vcol[j];
          if (lcid[icol[j]] >= 0) {
            /* coarse neighbor */
            pcontrib[icol[j]] += pentry;
          } else if (icol[j] != i) {
            /* the neighbor is a strongly connected fine node */
            ci = icol[j];
            vi = vcol[j];
            PetscCall(MatRestoreRow(lA, i, &ncols, &icol, &vcol));
            PetscCall(MatGetRow(lA, ci, &ncols, &icol, &vcol));
            jwttotal = 0.;
            jdiag    = 0.;
            for (k = 0; k < ncols; k++) {
              if (ci == icol[k]) jdiag = PetscRealPart(vcol[k]);
            }
            for (k = 0; k < ncols; k++) {
              if (lcid[icol[k]] >= 0 && jdiag * PetscRealPart(vcol[k]) < 0.) {
                pjentry = vcol[k];
                jwttotal += PetscRealPart(pjentry);
              }
            }
            if (jwttotal != 0.) {
              jwttotal = PetscRealPart(vi) / jwttotal;
              for (k = 0; k < ncols; k++) {
                if (lcid[icol[k]] >= 0 && jdiag * PetscRealPart(vcol[k]) < 0.) {
                  pjentry = vcol[k] * jwttotal;
                  pcontrib[icol[k]] += pjentry;
                }
              }
            } else {
              diag += PetscRealPart(vi);
            }
            PetscCall(MatRestoreRow(lA, ci, &ncols, &icol, &vcol));
            PetscCall(MatGetRow(lA, i, &ncols, &icol, &vcol));
          } else {
            diag += PetscRealPart(vcol[j]);
          }
        }
        if (diag != 0.) {
          diag = 1. / diag;
          for (j = 0; j < ncols; j++) {
            if (lcid[icol[j]] >= 0 && pcontrib[icol[j]] != 0.) {
              /* the neighbor is a coarse node */
              if (PetscAbsScalar(pcontrib[icol[j]]) > 0.0) {
                lni           = lcid[icol[j]];
                pvcol[pncols] = -pcontrib[icol[j]] * diag;
                picol[pncols] = lni;
                pncols++;
              }
              pcontrib[icol[j]] = 0.;
            } else {
              /* the neighbor is a strongly connected fine node */
              ci = icol[j];
              PetscCall(MatRestoreRow(lA, i, &ncols, &icol, &vcol));
              PetscCall(MatGetRow(lA, ci, &ncols, &icol, &vcol));
              for (k = 0; k < ncols; k++) {
                if (lcid[icol[k]] >= 0 && pcontrib[icol[k]] != 0.) {
                  if (PetscAbsScalar(pcontrib[icol[k]]) > 0.0) {
                    lni           = lcid[icol[k]];
                    pvcol[pncols] = -pcontrib[icol[k]] * diag;
                    picol[pncols] = lni;
                    pncols++;
                  }
                  pcontrib[icol[k]] = 0.;
                }
              }
              PetscCall(MatRestoreRow(lA, ci, &ncols, &icol, &vcol));
              PetscCall(MatGetRow(lA, i, &ncols, &icol, &vcol));
            }
            pcontrib[icol[j]] = 0.;
          }
          PetscCall(MatRestoreRow(lA, i, &ncols, &icol, &vcol));
        }
      }
      ci = gidx[i];
      if (pncols > 0) PetscCall(MatSetValues(*P, 1, &ci, pncols, picol, pvcol, INSERT_VALUES));
    }
  }
  PetscCall(ISRestoreIndices(lis, &gidx));
  PetscCall(PetscFree2(picol, pvcol));
  PetscCall(PetscFree3(lsparse, gsparse, pcontrib));
  PetscCall(ISDestroy(&lis));
  PetscCall(PetscFree(gcid));
  if (size > 1) {
    PetscCall(PetscFree(lcid));
    PetscCall(MatDestroyMatrices(1, &lAs));
    PetscCall(PetscSFDestroy(&sf));
  }
  PetscCall(VecDestroy(&cv));
  PetscCall(MatAssemblyBegin(*P, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(*P, MAT_FINAL_ASSEMBLY));
  PetscFunctionReturn(0);
}

PetscErrorCode PCGAMGOptProlongator_Classical_Jacobi(PC pc, Mat A, Mat *P)
{
  PetscInt           f, s, n, cf, cs, i, idx;
  PetscInt          *coarserows;
  PetscInt           ncols;
  const PetscInt    *pcols;
  const PetscScalar *pvals;
  Mat                Pnew;
  Vec                diag;
  PC_MG             *mg      = (PC_MG *)pc->data;
  PC_GAMG           *pc_gamg = (PC_GAMG *)mg->innerctx;
  PC_GAMG_Classical *cls     = (PC_GAMG_Classical *)pc_gamg->subctx;

  PetscFunctionBegin;
  if (cls->nsmooths == 0) {
    PetscCall(PCGAMGTruncateProlongator_Private(pc, P));
    PetscFunctionReturn(0);
  }
  PetscCall(MatGetOwnershipRange(*P, &s, &f));
  n = f - s;
  PetscCall(MatGetOwnershipRangeColumn(*P, &cs, &cf));
  PetscCall(PetscMalloc1(n, &coarserows));
  /* identify the rows corresponding to coarse unknowns */
  idx = 0;
  for (i = s; i < f; i++) {
    PetscCall(MatGetRow(*P, i, &ncols, &pcols, &pvals));
    /* assume, for now, that it's a coarse unknown if it has a single unit entry */
    if (ncols == 1) {
      if (pvals[0] == 1.) {
        coarserows[idx] = i;
        idx++;
      }
    }
    PetscCall(MatRestoreRow(*P, i, &ncols, &pcols, &pvals));
  }
  PetscCall(MatCreateVecs(A, &diag, NULL));
  PetscCall(MatGetDiagonal(A, diag));
  PetscCall(VecReciprocal(diag));
  for (i = 0; i < cls->nsmooths; i++) {
    PetscCall(MatMatMult(A, *P, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &Pnew));
    PetscCall(MatZeroRows(Pnew, idx, coarserows, 0., NULL, NULL));
    PetscCall(MatDiagonalScale(Pnew, diag, NULL));
    PetscCall(MatAYPX(Pnew, -1.0, *P, DIFFERENT_NONZERO_PATTERN));
    PetscCall(MatDestroy(P));
    *P   = Pnew;
    Pnew = NULL;
  }
  PetscCall(VecDestroy(&diag));
  PetscCall(PetscFree(coarserows));
  PetscCall(PCGAMGTruncateProlongator_Private(pc, P));
  PetscFunctionReturn(0);
}

static PetscErrorCode PCGAMGProlongator_Classical(PC pc, Mat A, Mat G, PetscCoarsenData *agg_lists, Mat *P)
{
  PetscErrorCode (*f)(PC, Mat, Mat, PetscCoarsenData *, Mat *);
  PC_MG             *mg      = (PC_MG *)pc->data;
  PC_GAMG           *pc_gamg = (PC_GAMG *)mg->innerctx;
  PC_GAMG_Classical *cls     = (PC_GAMG_Classical *)pc_gamg->subctx;

  PetscFunctionBegin;
  PetscCall(PetscFunctionListFind(PCGAMGClassicalProlongatorList, cls->prolongtype, &f));
  PetscCheck(f, PetscObjectComm((PetscObject)pc), PETSC_ERR_ARG_WRONGSTATE, "Cannot find PCGAMG Classical prolongator type");
  PetscCall((*f)(pc, A, G, agg_lists, P));
  PetscFunctionReturn(0);
}

static PetscErrorCode PCGAMGDestroy_Classical(PC pc)
{
  PC_MG   *mg      = (PC_MG *)pc->data;
  PC_GAMG *pc_gamg = (PC_GAMG *)mg->innerctx;

  PetscFunctionBegin;
  PetscCall(PetscFree(pc_gamg->subctx));
  PetscCall(PetscObjectComposeFunction((PetscObject)pc, "PCGAMGClassicalSetType_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)pc, "PCGAMGClassicalGetType_C", NULL));
  PetscFunctionReturn(0);
}

PetscErrorCode PCGAMGSetFromOptions_Classical(PC pc, PetscOptionItems *PetscOptionsObject)
{
  PC_MG             *mg      = (PC_MG *)pc->data;
  PC_GAMG           *pc_gamg = (PC_GAMG *)mg->innerctx;
  PC_GAMG_Classical *cls     = (PC_GAMG_Classical *)pc_gamg->subctx;
  char               tname[256];
  PetscBool          flg;

  PetscFunctionBegin;
  PetscOptionsHeadBegin(PetscOptionsObject, "GAMG-Classical options");
  PetscCall(PetscOptionsFList("-pc_gamg_classical_type", "Type of Classical AMG prolongation", "PCGAMGClassicalSetType", PCGAMGClassicalProlongatorList, cls->prolongtype, tname, sizeof(tname), &flg));
  if (flg) PetscCall(PCGAMGClassicalSetType(pc, tname));
  PetscCall(PetscOptionsReal("-pc_gamg_classical_interp_threshold", "Threshold for classical interpolator entries", "", cls->interp_threshold, &cls->interp_threshold, NULL));
  PetscCall(PetscOptionsInt("-pc_gamg_classical_nsmooths", "Threshold for classical interpolator entries", "", cls->nsmooths, &cls->nsmooths, NULL));
  PetscOptionsHeadEnd();
  PetscFunctionReturn(0);
}

static PetscErrorCode PCGAMGSetData_Classical(PC pc, Mat A)
{
  PC_MG   *mg      = (PC_MG *)pc->data;
  PC_GAMG *pc_gamg = (PC_GAMG *)mg->innerctx;

  PetscFunctionBegin;
  /* no data for classical AMG */
  pc_gamg->data           = NULL;
  pc_gamg->data_cell_cols = 0;
  pc_gamg->data_cell_rows = 0;
  pc_gamg->data_sz        = 0;
  PetscFunctionReturn(0);
}

PetscErrorCode PCGAMGClassicalFinalizePackage(void)
{
  PetscFunctionBegin;
  PCGAMGClassicalPackageInitialized = PETSC_FALSE;
  PetscCall(PetscFunctionListDestroy(&PCGAMGClassicalProlongatorList));
  PetscFunctionReturn(0);
}

PetscErrorCode PCGAMGClassicalInitializePackage(void)
{
  PetscFunctionBegin;
  if (PCGAMGClassicalPackageInitialized) PetscFunctionReturn(0);
  PetscCall(PetscFunctionListAdd(&PCGAMGClassicalProlongatorList, PCGAMGCLASSICALDIRECT, PCGAMGProlongator_Classical_Direct));
  PetscCall(PetscFunctionListAdd(&PCGAMGClassicalProlongatorList, PCGAMGCLASSICALSTANDARD, PCGAMGProlongator_Classical_Standard));
  PetscCall(PetscRegisterFinalize(PCGAMGClassicalFinalizePackage));
  PetscFunctionReturn(0);
}

/*
   PCCreateGAMG_Classical

*/
PetscErrorCode PCCreateGAMG_Classical(PC pc)
{
  PC_MG             *mg      = (PC_MG *)pc->data;
  PC_GAMG           *pc_gamg = (PC_GAMG *)mg->innerctx;
  PC_GAMG_Classical *pc_gamg_classical;

  PetscFunctionBegin;
  PetscCall(PCGAMGClassicalInitializePackage());
  if (pc_gamg->subctx) {
    /* call base class */
    PetscCall(PCDestroy_GAMG(pc));
  }

  /* create sub context for SA */
  PetscCall(PetscNew(&pc_gamg_classical));
  pc_gamg->subctx         = pc_gamg_classical;
  pc->ops->setfromoptions = PCGAMGSetFromOptions_Classical;
  /* reset does not do anything; setup not virtual */

  /* set internal function pointers */
  pc_gamg->ops->destroy        = PCGAMGDestroy_Classical;
  pc_gamg->ops->creategraph    = PCGAMGCreateGraph_Classical;
  pc_gamg->ops->coarsen        = PCGAMGCoarsen_Classical;
  pc_gamg->ops->prolongator    = PCGAMGProlongator_Classical;
  pc_gamg->ops->optprolongator = PCGAMGOptProlongator_Classical_Jacobi;
  pc_gamg->ops->setfromoptions = PCGAMGSetFromOptions_Classical;

  pc_gamg->ops->createdefaultdata     = PCGAMGSetData_Classical;
  pc_gamg_classical->interp_threshold = 0.2;
  pc_gamg_classical->nsmooths         = 0;
  PetscCall(PetscObjectComposeFunction((PetscObject)pc, "PCGAMGClassicalSetType_C", PCGAMGClassicalSetType_GAMG));
  PetscCall(PetscObjectComposeFunction((PetscObject)pc, "PCGAMGClassicalGetType_C", PCGAMGClassicalGetType_GAMG));
  PetscCall(PCGAMGClassicalSetType(pc, PCGAMGCLASSICALSTANDARD));
  PetscFunctionReturn(0);
}
