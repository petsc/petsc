#include <petsc/private/matimpl.h> /*I "petscmat.h"  I*/
#include <../src/mat/impls/aij/seq/aij.h>
#include <../src/mat/impls/aij/mpi/mpiaij.h>
#include <petscsf.h>

typedef struct {
  PetscBool symmetric;
} MC_Greedy;

static PetscErrorCode MatColoringDestroy_Greedy(MatColoring mc)
{
  PetscFunctionBegin;
  PetscCall(PetscFree(mc->data));
  PetscFunctionReturn(0);
}

static PetscErrorCode GreedyColoringLocalDistanceOne_Private(MatColoring mc, PetscReal *wts, PetscInt *lperm, ISColoringValue *colors)
{
  PetscInt        i, j, k, s, e, n, no, nd, nd_global, n_global, idx, ncols, maxcolors, masksize, ccol, *mask;
  Mat             m   = mc->mat;
  Mat_MPIAIJ     *aij = (Mat_MPIAIJ *)m->data;
  Mat             md = NULL, mo = NULL;
  const PetscInt *md_i, *mo_i, *md_j, *mo_j;
  PetscBool       isMPIAIJ, isSEQAIJ;
  ISColoringValue pcol;
  const PetscInt *cidx;
  PetscInt       *lcolors, *ocolors;
  PetscReal      *owts = NULL;
  PetscSF         sf;
  PetscLayout     layout;

  PetscFunctionBegin;
  PetscCall(MatGetSize(m, &n_global, NULL));
  PetscCall(MatGetOwnershipRange(m, &s, &e));
  n         = e - s;
  masksize  = 20;
  nd_global = 0;
  /* get the matrix communication structures */
  PetscCall(PetscObjectTypeCompare((PetscObject)m, MATMPIAIJ, &isMPIAIJ));
  PetscCall(PetscObjectBaseTypeCompare((PetscObject)m, MATSEQAIJ, &isSEQAIJ));
  if (isMPIAIJ) {
    /* get the CSR data for on and off diagonal portions of m */
    Mat_SeqAIJ *dseq;
    Mat_SeqAIJ *oseq;
    md   = aij->A;
    dseq = (Mat_SeqAIJ *)md->data;
    mo   = aij->B;
    oseq = (Mat_SeqAIJ *)mo->data;
    md_i = dseq->i;
    md_j = dseq->j;
    mo_i = oseq->i;
    mo_j = oseq->j;
  } else if (isSEQAIJ) {
    /* get the CSR data for m */
    Mat_SeqAIJ *dseq;
    /* no off-processor nodes */
    md   = m;
    dseq = (Mat_SeqAIJ *)md->data;
    mo   = NULL;
    no   = 0;
    md_i = dseq->i;
    md_j = dseq->j;
    mo_i = NULL;
    mo_j = NULL;
  } else SETERRQ(PetscObjectComm((PetscObject)mc), PETSC_ERR_ARG_WRONG, "Matrix must be AIJ for greedy coloring");
  PetscCall(MatColoringGetMaxColors(mc, &maxcolors));
  if (mo) {
    PetscCall(VecGetSize(aij->lvec, &no));
    PetscCall(PetscMalloc2(no, &ocolors, no, &owts));
    for (i = 0; i < no; i++) ocolors[i] = maxcolors;
  }

  PetscCall(PetscMalloc1(masksize, &mask));
  PetscCall(PetscMalloc1(n, &lcolors));
  for (i = 0; i < n; i++) lcolors[i] = maxcolors;
  for (i = 0; i < masksize; i++) mask[i] = -1;
  if (mo) {
    /* transfer neighbor weights */
    PetscCall(PetscSFCreate(PetscObjectComm((PetscObject)m), &sf));
    PetscCall(MatGetLayouts(m, &layout, NULL));
    PetscCall(PetscSFSetGraphLayout(sf, layout, no, NULL, PETSC_COPY_VALUES, aij->garray));
    PetscCall(PetscSFBcastBegin(sf, MPIU_REAL, wts, owts, MPI_REPLACE));
    PetscCall(PetscSFBcastEnd(sf, MPIU_REAL, wts, owts, MPI_REPLACE));
  }
  while (nd_global < n_global) {
    nd = n;
    /* assign lowest possible color to each local vertex */
    PetscCall(PetscLogEventBegin(MATCOLORING_Local, mc, 0, 0, 0));
    for (i = 0; i < n; i++) {
      idx = lperm[i];
      if (lcolors[idx] == maxcolors) {
        ncols = md_i[idx + 1] - md_i[idx];
        cidx  = &(md_j[md_i[idx]]);
        for (j = 0; j < ncols; j++) {
          if (lcolors[cidx[j]] != maxcolors) {
            ccol = lcolors[cidx[j]];
            if (ccol >= masksize) {
              PetscInt *newmask;
              PetscCall(PetscMalloc1(masksize * 2, &newmask));
              for (k = 0; k < 2 * masksize; k++) newmask[k] = -1;
              for (k = 0; k < masksize; k++) newmask[k] = mask[k];
              PetscCall(PetscFree(mask));
              mask = newmask;
              masksize *= 2;
            }
            mask[ccol] = idx;
          }
        }
        if (mo) {
          ncols = mo_i[idx + 1] - mo_i[idx];
          cidx  = &(mo_j[mo_i[idx]]);
          for (j = 0; j < ncols; j++) {
            if (ocolors[cidx[j]] != maxcolors) {
              ccol = ocolors[cidx[j]];
              if (ccol >= masksize) {
                PetscInt *newmask;
                PetscCall(PetscMalloc1(masksize * 2, &newmask));
                for (k = 0; k < 2 * masksize; k++) newmask[k] = -1;
                for (k = 0; k < masksize; k++) newmask[k] = mask[k];
                PetscCall(PetscFree(mask));
                mask = newmask;
                masksize *= 2;
              }
              mask[ccol] = idx;
            }
          }
        }
        for (j = 0; j < masksize; j++) {
          if (mask[j] != idx) break;
        }
        pcol = j;
        if (pcol > maxcolors) pcol = maxcolors;
        lcolors[idx] = pcol;
      }
    }
    PetscCall(PetscLogEventEnd(MATCOLORING_Local, mc, 0, 0, 0));
    if (mo) {
      /* transfer neighbor colors */
      PetscCall(PetscLogEventBegin(MATCOLORING_Comm, mc, 0, 0, 0));
      PetscCall(PetscSFBcastBegin(sf, MPIU_INT, lcolors, ocolors, MPI_REPLACE));
      PetscCall(PetscSFBcastEnd(sf, MPIU_INT, lcolors, ocolors, MPI_REPLACE));
      /* check for conflicts -- this is merely checking if any adjacent off-processor rows have the same color and marking the ones that are lower weight locally for changing */
      for (i = 0; i < n; i++) {
        ncols = mo_i[i + 1] - mo_i[i];
        cidx  = &(mo_j[mo_i[i]]);
        for (j = 0; j < ncols; j++) {
          /* in the case of conflicts, the highest weight one stays and the others go */
          if ((ocolors[cidx[j]] == lcolors[i]) && (owts[cidx[j]] > wts[i]) && lcolors[i] < maxcolors) {
            lcolors[i] = maxcolors;
            nd--;
          }
        }
      }
      nd_global = 0;
    }
    PetscCall(MPIU_Allreduce(&nd, &nd_global, 1, MPIU_INT, MPI_SUM, PetscObjectComm((PetscObject)mc)));
  }
  for (i = 0; i < n; i++) colors[i] = (ISColoringValue)lcolors[i];
  PetscCall(PetscFree(mask));
  PetscCall(PetscFree(lcolors));
  if (mo) {
    PetscCall(PetscFree2(ocolors, owts));
    PetscCall(PetscSFDestroy(&sf));
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode GreedyColoringLocalDistanceTwo_Private(MatColoring mc, PetscReal *wts, PetscInt *lperm, ISColoringValue *colors)
{
  MC_Greedy       *gr = (MC_Greedy *)mc->data;
  PetscInt         i, j, k, l, s, e, n, nd, nd_global, n_global, idx, ncols, maxcolors, mcol, mcol_global, nd1cols, *mask, masksize, *d1cols, *bad, *badnext, nbad, badsize, ccol, no, cbad;
  Mat              m   = mc->mat, mt;
  Mat_MPIAIJ      *aij = (Mat_MPIAIJ *)m->data;
  Mat              md = NULL, mo = NULL;
  const PetscInt  *md_i, *mo_i, *md_j, *mo_j;
  const PetscInt  *rmd_i, *rmo_i, *rmd_j, *rmo_j;
  PetscBool        isMPIAIJ, isSEQAIJ;
  PetscInt         pcol, *dcolors, *ocolors;
  ISColoringValue *badidx;
  const PetscInt  *cidx;
  PetscReal       *owts, *colorweights;
  PetscInt        *oconf, *conf;
  PetscSF          sf;
  PetscLayout      layout;

  PetscFunctionBegin;
  PetscCall(MatGetSize(m, &n_global, NULL));
  PetscCall(MatGetOwnershipRange(m, &s, &e));
  n         = e - s;
  nd_global = 0;
  /* get the matrix communication structures */
  PetscCall(PetscObjectBaseTypeCompare((PetscObject)m, MATMPIAIJ, &isMPIAIJ));
  PetscCall(PetscObjectBaseTypeCompare((PetscObject)m, MATSEQAIJ, &isSEQAIJ));
  if (isMPIAIJ) {
    Mat_SeqAIJ *dseq;
    Mat_SeqAIJ *oseq;
    md    = aij->A;
    dseq  = (Mat_SeqAIJ *)md->data;
    mo    = aij->B;
    oseq  = (Mat_SeqAIJ *)mo->data;
    md_i  = dseq->i;
    md_j  = dseq->j;
    mo_i  = oseq->i;
    mo_j  = oseq->j;
    rmd_i = dseq->i;
    rmd_j = dseq->j;
    rmo_i = oseq->i;
    rmo_j = oseq->j;
  } else if (isSEQAIJ) {
    Mat_SeqAIJ *dseq;
    /* no off-processor nodes */
    md    = m;
    dseq  = (Mat_SeqAIJ *)md->data;
    md_i  = dseq->i;
    md_j  = dseq->j;
    mo_i  = NULL;
    mo_j  = NULL;
    rmd_i = dseq->i;
    rmd_j = dseq->j;
    rmo_i = NULL;
    rmo_j = NULL;
  } else SETERRQ(PetscObjectComm((PetscObject)mc), PETSC_ERR_ARG_WRONG, "Matrix must be AIJ for greedy coloring");
  if (!gr->symmetric) {
    PetscCall(MatTranspose(m, MAT_INITIAL_MATRIX, &mt));
    if (isSEQAIJ) {
      Mat_SeqAIJ *dseq = (Mat_SeqAIJ *)mt->data;
      rmd_i            = dseq->i;
      rmd_j            = dseq->j;
      rmo_i            = NULL;
      rmo_j            = NULL;
    } else SETERRQ(PetscObjectComm((PetscObject)mc), PETSC_ERR_SUP, "Nonsymmetric greedy coloring only works in serial");
  }
  /* create the vectors and communication structures if necessary */
  no = 0;
  if (mo) {
    PetscCall(VecGetLocalSize(aij->lvec, &no));
    PetscCall(PetscSFCreate(PetscObjectComm((PetscObject)m), &sf));
    PetscCall(MatGetLayouts(m, &layout, NULL));
    PetscCall(PetscSFSetGraphLayout(sf, layout, no, NULL, PETSC_COPY_VALUES, aij->garray));
  }
  PetscCall(MatColoringGetMaxColors(mc, &maxcolors));
  masksize = n;
  nbad     = 0;
  badsize  = n;
  PetscCall(PetscMalloc1(masksize, &mask));
  PetscCall(PetscMalloc4(n, &d1cols, n, &dcolors, n, &conf, n, &bad));
  PetscCall(PetscMalloc2(badsize, &badidx, badsize, &badnext));
  for (i = 0; i < masksize; i++) mask[i] = -1;
  for (i = 0; i < n; i++) {
    dcolors[i] = maxcolors;
    bad[i]     = -1;
  }
  for (i = 0; i < badsize; i++) badnext[i] = -1;
  if (mo) {
    PetscCall(PetscMalloc3(no, &owts, no, &oconf, no, &ocolors));
    PetscCall(PetscSFBcastBegin(sf, MPIU_REAL, wts, owts, MPI_REPLACE));
    PetscCall(PetscSFBcastEnd(sf, MPIU_REAL, wts, owts, MPI_REPLACE));
    for (i = 0; i < no; i++) ocolors[i] = maxcolors;
  } else { /* Appease overzealous -Wmaybe-initialized */
    owts    = NULL;
    oconf   = NULL;
    ocolors = NULL;
  }
  mcol = 0;
  while (nd_global < n_global) {
    nd = n;
    /* assign lowest possible color to each local vertex */
    mcol_global = 0;
    PetscCall(PetscLogEventBegin(MATCOLORING_Local, mc, 0, 0, 0));
    for (i = 0; i < n; i++) {
      idx = lperm[i];
      if (dcolors[idx] == maxcolors) {
        /* entries in bad */
        cbad = bad[idx];
        while (cbad >= 0) {
          ccol = badidx[cbad];
          if (ccol >= masksize) {
            PetscInt *newmask;
            PetscCall(PetscMalloc1(masksize * 2, &newmask));
            for (k = 0; k < 2 * masksize; k++) newmask[k] = -1;
            for (k = 0; k < masksize; k++) newmask[k] = mask[k];
            PetscCall(PetscFree(mask));
            mask = newmask;
            masksize *= 2;
          }
          mask[ccol] = idx;
          cbad       = badnext[cbad];
        }
        /* diagonal distance-one rows */
        nd1cols = 0;
        ncols   = rmd_i[idx + 1] - rmd_i[idx];
        cidx    = &(rmd_j[rmd_i[idx]]);
        for (j = 0; j < ncols; j++) {
          d1cols[nd1cols] = cidx[j];
          nd1cols++;
          ccol = dcolors[cidx[j]];
          if (ccol != maxcolors) {
            if (ccol >= masksize) {
              PetscInt *newmask;
              PetscCall(PetscMalloc1(masksize * 2, &newmask));
              for (k = 0; k < 2 * masksize; k++) newmask[k] = -1;
              for (k = 0; k < masksize; k++) newmask[k] = mask[k];
              PetscCall(PetscFree(mask));
              mask = newmask;
              masksize *= 2;
            }
            mask[ccol] = idx;
          }
        }
        /* off-diagonal distance-one rows */
        if (mo) {
          ncols = rmo_i[idx + 1] - rmo_i[idx];
          cidx  = &(rmo_j[rmo_i[idx]]);
          for (j = 0; j < ncols; j++) {
            ccol = ocolors[cidx[j]];
            if (ccol != maxcolors) {
              if (ccol >= masksize) {
                PetscInt *newmask;
                PetscCall(PetscMalloc1(masksize * 2, &newmask));
                for (k = 0; k < 2 * masksize; k++) newmask[k] = -1;
                for (k = 0; k < masksize; k++) newmask[k] = mask[k];
                PetscCall(PetscFree(mask));
                mask = newmask;
                masksize *= 2;
              }
              mask[ccol] = idx;
            }
          }
        }
        /* diagonal distance-two rows */
        for (j = 0; j < nd1cols; j++) {
          ncols = md_i[d1cols[j] + 1] - md_i[d1cols[j]];
          cidx  = &(md_j[md_i[d1cols[j]]]);
          for (l = 0; l < ncols; l++) {
            ccol = dcolors[cidx[l]];
            if (ccol != maxcolors) {
              if (ccol >= masksize) {
                PetscInt *newmask;
                PetscCall(PetscMalloc1(masksize * 2, &newmask));
                for (k = 0; k < 2 * masksize; k++) newmask[k] = -1;
                for (k = 0; k < masksize; k++) newmask[k] = mask[k];
                PetscCall(PetscFree(mask));
                mask = newmask;
                masksize *= 2;
              }
              mask[ccol] = idx;
            }
          }
        }
        /* off-diagonal distance-two rows */
        if (mo) {
          for (j = 0; j < nd1cols; j++) {
            ncols = mo_i[d1cols[j] + 1] - mo_i[d1cols[j]];
            cidx  = &(mo_j[mo_i[d1cols[j]]]);
            for (l = 0; l < ncols; l++) {
              ccol = ocolors[cidx[l]];
              if (ccol != maxcolors) {
                if (ccol >= masksize) {
                  PetscInt *newmask;
                  PetscCall(PetscMalloc1(masksize * 2, &newmask));
                  for (k = 0; k < 2 * masksize; k++) newmask[k] = -1;
                  for (k = 0; k < masksize; k++) newmask[k] = mask[k];
                  PetscCall(PetscFree(mask));
                  mask = newmask;
                  masksize *= 2;
                }
                mask[ccol] = idx;
              }
            }
          }
        }
        /* assign this one the lowest color possible by seeing if there's a gap in the sequence of sorted neighbor colors */
        for (j = 0; j < masksize; j++) {
          if (mask[j] != idx) break;
        }
        pcol = j;
        if (pcol > maxcolors) pcol = maxcolors;
        dcolors[idx] = pcol;
        if (pcol > mcol) mcol = pcol;
      }
    }
    PetscCall(PetscLogEventEnd(MATCOLORING_Local, mc, 0, 0, 0));
    if (mo) {
      /* transfer neighbor colors */
      PetscCall(PetscSFBcastBegin(sf, MPIU_INT, dcolors, ocolors, MPI_REPLACE));
      PetscCall(PetscSFBcastEnd(sf, MPIU_INT, dcolors, ocolors, MPI_REPLACE));
      /* find the maximum color assigned locally and allocate a mask */
      PetscCall(MPIU_Allreduce(&mcol, &mcol_global, 1, MPIU_INT, MPI_MAX, PetscObjectComm((PetscObject)mc)));
      PetscCall(PetscMalloc1(mcol_global + 1, &colorweights));
      /* check for conflicts */
      for (i = 0; i < n; i++) conf[i] = PETSC_FALSE;
      for (i = 0; i < no; i++) oconf[i] = PETSC_FALSE;
      for (i = 0; i < n; i++) {
        ncols = mo_i[i + 1] - mo_i[i];
        cidx  = &(mo_j[mo_i[i]]);
        if (ncols > 0) {
          /* fill in the mask */
          for (j = 0; j < mcol_global + 1; j++) colorweights[j] = 0;
          colorweights[dcolors[i]] = wts[i];
          /* fill in the off-diagonal part of the mask */
          for (j = 0; j < ncols; j++) {
            ccol = ocolors[cidx[j]];
            if (ccol < maxcolors) {
              if (colorweights[ccol] < owts[cidx[j]]) colorweights[ccol] = owts[cidx[j]];
            }
          }
          /* fill in the on-diagonal part of the mask */
          ncols = md_i[i + 1] - md_i[i];
          cidx  = &(md_j[md_i[i]]);
          for (j = 0; j < ncols; j++) {
            ccol = dcolors[cidx[j]];
            if (ccol < maxcolors) {
              if (colorweights[ccol] < wts[cidx[j]]) colorweights[ccol] = wts[cidx[j]];
            }
          }
          /* go back through and set up on and off-diagonal conflict vectors */
          ncols = md_i[i + 1] - md_i[i];
          cidx  = &(md_j[md_i[i]]);
          for (j = 0; j < ncols; j++) {
            ccol = dcolors[cidx[j]];
            if (ccol < maxcolors) {
              if (colorweights[ccol] > wts[cidx[j]]) conf[cidx[j]] = PETSC_TRUE;
            }
          }
          ncols = mo_i[i + 1] - mo_i[i];
          cidx  = &(mo_j[mo_i[i]]);
          for (j = 0; j < ncols; j++) {
            ccol = ocolors[cidx[j]];
            if (ccol < maxcolors) {
              if (colorweights[ccol] > owts[cidx[j]]) oconf[cidx[j]] = PETSC_TRUE;
            }
          }
        }
      }
      nd_global = 0;
      PetscCall(PetscFree(colorweights));
      PetscCall(PetscLogEventBegin(MATCOLORING_Comm, mc, 0, 0, 0));
      PetscCall(PetscSFReduceBegin(sf, MPIU_INT, oconf, conf, MPI_SUM));
      PetscCall(PetscSFReduceEnd(sf, MPIU_INT, oconf, conf, MPI_SUM));
      PetscCall(PetscLogEventEnd(MATCOLORING_Comm, mc, 0, 0, 0));
      /* go through and unset local colors that have conflicts */
      for (i = 0; i < n; i++) {
        if (conf[i] > 0) {
          /* push this color onto the bad stack */
          badidx[nbad]  = dcolors[i];
          badnext[nbad] = bad[i];
          bad[i]        = nbad;
          nbad++;
          if (nbad >= badsize) {
            PetscInt        *newbadnext;
            ISColoringValue *newbadidx;
            PetscCall(PetscMalloc2(badsize * 2, &newbadidx, badsize * 2, &newbadnext));
            for (k = 0; k < 2 * badsize; k++) newbadnext[k] = -1;
            for (k = 0; k < badsize; k++) {
              newbadidx[k]  = badidx[k];
              newbadnext[k] = badnext[k];
            }
            PetscCall(PetscFree2(badidx, badnext));
            badidx  = newbadidx;
            badnext = newbadnext;
            badsize *= 2;
          }
          dcolors[i] = maxcolors;
          nd--;
        }
      }
    }
    PetscCall(MPIU_Allreduce(&nd, &nd_global, 1, MPIU_INT, MPI_SUM, PetscObjectComm((PetscObject)mc)));
  }
  if (mo) {
    PetscCall(PetscSFDestroy(&sf));
    PetscCall(PetscFree3(owts, oconf, ocolors));
  }
  for (i = 0; i < n; i++) colors[i] = dcolors[i];
  PetscCall(PetscFree(mask));
  PetscCall(PetscFree4(d1cols, dcolors, conf, bad));
  PetscCall(PetscFree2(badidx, badnext));
  if (!gr->symmetric) PetscCall(MatDestroy(&mt));
  PetscFunctionReturn(0);
}

static PetscErrorCode MatColoringApply_Greedy(MatColoring mc, ISColoring *iscoloring)
{
  PetscInt         finalcolor, finalcolor_global;
  ISColoringValue *colors;
  PetscInt         ncolstotal, ncols;
  PetscReal       *wts;
  PetscInt         i, *lperm;

  PetscFunctionBegin;
  PetscCall(MatGetSize(mc->mat, NULL, &ncolstotal));
  PetscCall(MatGetLocalSize(mc->mat, NULL, &ncols));
  if (!mc->user_weights) {
    PetscCall(MatColoringCreateWeights(mc, &wts, &lperm));
  } else {
    wts   = mc->user_weights;
    lperm = mc->user_lperm;
  }
  PetscCall(PetscMalloc1(ncols, &colors));
  if (mc->dist == 1) {
    PetscCall(GreedyColoringLocalDistanceOne_Private(mc, wts, lperm, colors));
  } else if (mc->dist == 2) {
    PetscCall(GreedyColoringLocalDistanceTwo_Private(mc, wts, lperm, colors));
  } else SETERRQ(PetscObjectComm((PetscObject)mc), PETSC_ERR_ARG_OUTOFRANGE, "Only distance 1 and distance 2 supported by MatColoringGreedy");
  finalcolor = 0;
  for (i = 0; i < ncols; i++) {
    if (colors[i] > finalcolor) finalcolor = colors[i];
  }
  finalcolor_global = 0;
  PetscCall(MPIU_Allreduce(&finalcolor, &finalcolor_global, 1, MPIU_INT, MPI_MAX, PetscObjectComm((PetscObject)mc)));
  PetscCall(PetscLogEventBegin(MATCOLORING_ISCreate, mc, 0, 0, 0));
  PetscCall(ISColoringCreate(PetscObjectComm((PetscObject)mc), finalcolor_global + 1, ncols, colors, PETSC_OWN_POINTER, iscoloring));
  PetscCall(PetscLogEventEnd(MATCOLORING_ISCreate, mc, 0, 0, 0));
  if (!mc->user_weights) {
    PetscCall(PetscFree(wts));
    PetscCall(PetscFree(lperm));
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode MatColoringSetFromOptions_Greedy(MatColoring mc, PetscOptionItems *PetscOptionsObject)
{
  MC_Greedy *gr = (MC_Greedy *)mc->data;

  PetscFunctionBegin;
  PetscOptionsHeadBegin(PetscOptionsObject, "Greedy options");
  PetscCall(PetscOptionsBool("-mat_coloring_greedy_symmetric", "Flag for assuming a symmetric matrix", "", gr->symmetric, &gr->symmetric, NULL));
  PetscOptionsHeadEnd();
  PetscFunctionReturn(0);
}

/*MC
  MATCOLORINGGREEDY - Greedy-with-conflict correction based matrix coloring for distance 1 and 2.

   Level: beginner

   Notes:

   These algorithms proceed in two phases -- local coloring and conflict resolution.  The local coloring
   tentatively colors all vertices at the distance required given what's known of the global coloring.  Then,
   the updated colors are transferred to different processors at distance one.  In the distance one case, each
   vertex with nonlocal neighbors is then checked to see if it conforms, with the vertex being
   marked for recoloring if its lower weight than its same colored neighbor.  In the distance two case,
   each boundary vertex's immediate star is checked for validity of the coloring.  Lower-weight conflict
   vertices are marked, and then the conflicts are gathered back on owning processors.  In both cases
   this is done until each column has received a valid color.

   Supports both distance one and distance two colorings.

   References:
.  * - Bozdag et al. "A Parallel Distance 2 Graph Coloring Algorithm for Distributed Memory Computers"
   HPCC'05 Proceedings of the First international conference on High Performance Computing and Communications

.seealso: `MatColoringType`, `MatColoringCreate()`, `MatColoring`, `MatColoringSetType()`, `MatColoringType`
M*/
PETSC_EXTERN PetscErrorCode MatColoringCreate_Greedy(MatColoring mc)
{
  MC_Greedy *gr;

  PetscFunctionBegin;
  PetscCall(PetscNew(&gr));
  mc->data                = gr;
  mc->ops->apply          = MatColoringApply_Greedy;
  mc->ops->view           = NULL;
  mc->ops->destroy        = MatColoringDestroy_Greedy;
  mc->ops->setfromoptions = MatColoringSetFromOptions_Greedy;

  gr->symmetric = PETSC_TRUE;
  PetscFunctionReturn(0);
}
