#include <petsc/private/matimpl.h> /*I "petscmat.h" I*/
#include <../src/mat/impls/aij/seq/aij.h>
#include <../src/mat/impls/aij/mpi/mpiaij.h>
#include <petscsf.h>

#define MIS_NOT_DONE       -2
#define MIS_DELETED        -1
#define MIS_REMOVED        -3
#define MIS_IS_SELECTED(s) (s != MIS_DELETED && s != MIS_NOT_DONE && s != MIS_REMOVED)

/*
   MatCoarsenApply_MIS_private - parallel maximal independent set (MIS) with data locality info. MatAIJ specific!!!

   Input Parameter:
   . perm - serial permutation of rows of local to process in MIS
   . Gmat - global matrix of graph (data not defined)
   . strict_aggs - flag for whether to keep strict (non overlapping) aggregates in 'llist';

   Output Parameter:
   . a_selected - IS of selected vertices, includes 'ghost' nodes at end with natural local indices
   . a_locals_llist - array of list of nodes rooted at selected nodes
*/
static PetscErrorCode MatCoarsenApply_MIS_private(IS perm, Mat Gmat, PetscBool strict_aggs, PetscCoarsenData **a_locals_llist)
{
  Mat_SeqAIJ       *matA, *matB = NULL;
  Mat_MPIAIJ       *mpimat = NULL;
  MPI_Comm          comm;
  PetscInt          num_fine_ghosts, kk, n, ix, j, *idx, *ii, Iend, my0, nremoved, gid, lid, cpid, lidj, sgid, t1, t2, slid, nDone, nselected = 0, state, statej;
  PetscInt         *cpcol_gid, *cpcol_state, *lid_cprowID, *lid_gid, *cpcol_sel_gid, *icpcol_gid, *lid_state, *lid_parent_gid = NULL, nrm_tot = 0;
  PetscBool        *lid_removed;
  PetscBool         isMPI, isAIJ, isOK;
  const PetscInt   *perm_ix;
  const PetscInt    nloc = Gmat->rmap->n;
  PetscCoarsenData *agg_lists;
  PetscLayout       layout;
  PetscSF           sf;
  IS                info_is;

  PetscFunctionBegin;
  PetscCall(PetscObjectGetComm((PetscObject)Gmat, &comm));
  PetscCall(ISCreate(comm, &info_is));
  PetscCall(PetscInfo(info_is, "mis: nloc = %d\n", (int)nloc));
  /* get submatrices */
  PetscCall(PetscObjectBaseTypeCompare((PetscObject)Gmat, MATMPIAIJ, &isMPI));
  if (isMPI) {
    mpimat = (Mat_MPIAIJ *)Gmat->data;
    matA   = (Mat_SeqAIJ *)mpimat->A->data;
    matB   = (Mat_SeqAIJ *)mpimat->B->data;
    /* force compressed storage of B */
    PetscCall(MatCheckCompressedRow(mpimat->B, matB->nonzerorowcnt, &matB->compressedrow, matB->i, Gmat->rmap->n, -1.0));
  } else {
    matA = (Mat_SeqAIJ *)Gmat->data;
    PetscCall(PetscObjectBaseTypeCompare((PetscObject)Gmat, MATSEQAIJ, &isAIJ));
    PetscCheck(isAIJ, comm, PETSC_ERR_PLIB, "Require AIJ matrix.");
  }
  PetscCall(MatGetOwnershipRange(Gmat, &my0, &Iend));
  PetscCall(PetscMalloc4(nloc, &lid_gid, nloc, &lid_cprowID, nloc, &lid_removed, nloc, &lid_state));
  if (strict_aggs) PetscCall(PetscMalloc1(nloc, &lid_parent_gid));
  if (isMPI) {
    for (kk = 0, gid = my0; kk < nloc; kk++, gid++) lid_gid[kk] = gid;
    PetscCall(VecGetLocalSize(mpimat->lvec, &num_fine_ghosts));
    PetscCall(PetscMalloc2(num_fine_ghosts, &cpcol_gid, num_fine_ghosts, &cpcol_state));
    PetscCall(PetscSFCreate(PetscObjectComm((PetscObject)Gmat), &sf));
    PetscCall(MatGetLayouts(Gmat, &layout, NULL));
    PetscCall(PetscSFSetGraphLayout(sf, layout, num_fine_ghosts, NULL, PETSC_COPY_VALUES, mpimat->garray));
    PetscCall(PetscSFBcastBegin(sf, MPIU_INT, lid_gid, cpcol_gid, MPI_REPLACE));
    PetscCall(PetscSFBcastEnd(sf, MPIU_INT, lid_gid, cpcol_gid, MPI_REPLACE));
    for (kk = 0; kk < num_fine_ghosts; kk++) cpcol_state[kk] = MIS_NOT_DONE;
  } else num_fine_ghosts = 0;

  /* has ghost nodes for !strict and uses local indexing (yuck) */
  PetscCall(PetscCDCreate(strict_aggs ? nloc : num_fine_ghosts + nloc, &agg_lists));
  if (a_locals_llist) *a_locals_llist = agg_lists;

  /* need an inverse map - locals */
  for (kk = 0; kk < nloc; kk++) {
    lid_cprowID[kk] = -1;
    lid_removed[kk] = PETSC_FALSE;
    if (strict_aggs) lid_parent_gid[kk] = -1.0;
    lid_state[kk] = MIS_NOT_DONE;
  }
  /* set index into cmpressed row 'lid_cprowID' */
  if (matB) {
    for (ix = 0; ix < matB->compressedrow.nrows; ix++) {
      lid = matB->compressedrow.rindex[ix];
      if (lid >= 0) lid_cprowID[lid] = ix;
    }
  }
  /* MIS */
  nremoved = nDone = 0;

  PetscCall(ISGetIndices(perm, &perm_ix));
  while (nDone < nloc || PETSC_TRUE) { /* asynchronous not implemented */
    /* check all vertices */
    for (kk = 0; kk < nloc; kk++) {
      lid   = perm_ix[kk];
      state = lid_state[lid];
      if (lid_removed[lid]) continue;
      if (state == MIS_NOT_DONE) {
        /* parallel test, delete if selected ghost */
        isOK = PETSC_TRUE;
        if ((ix = lid_cprowID[lid]) != -1) { /* if I have any ghost neighbors */
          ii  = matB->compressedrow.i;
          n   = ii[ix + 1] - ii[ix];
          idx = matB->j + ii[ix];
          for (j = 0; j < n; j++) {
            cpid   = idx[j]; /* compressed row ID in B mat */
            gid    = cpcol_gid[cpid];
            statej = cpcol_state[cpid];
            PetscCheck(!MIS_IS_SELECTED(statej), PETSC_COMM_SELF, PETSC_ERR_SUP, "selected ghost: %" PetscInt_FMT, gid);
            if (statej == MIS_NOT_DONE && gid >= Iend) { /* should be (pe>rank), use gid as pe proxy */
              isOK = PETSC_FALSE;                        /* can not delete */
              break;
            }
          }
        } /* parallel test */
        if (isOK) { /* select or remove this vertex */
          nDone++;
          /* check for singleton */
          ii = matA->i;
          n  = ii[lid + 1] - ii[lid];
          if (n < 2) {
            /* if I have any ghost adj then not a sing */
            ix = lid_cprowID[lid];
            if (ix == -1 || !(matB->compressedrow.i[ix + 1] - matB->compressedrow.i[ix])) {
              nremoved++;
              nrm_tot++;
              lid_removed[lid] = PETSC_TRUE;
              continue;
              // lid_state[lidj] = MIS_REMOVED; /* add singleton to MIS (can cause low rank with elasticity on fine grid) */
            }
          }
          /* SELECTED state encoded with global index */
          lid_state[lid] = lid + my0;
          nselected++;
          if (strict_aggs) {
            PetscCall(PetscCDAppendID(agg_lists, lid, lid + my0));
          } else {
            PetscCall(PetscCDAppendID(agg_lists, lid, lid));
          }
          /* delete local adj */
          idx = matA->j + ii[lid];
          for (j = 0; j < n; j++) {
            lidj   = idx[j];
            statej = lid_state[lidj];
            if (statej == MIS_NOT_DONE) {
              nDone++;
              if (strict_aggs) {
                PetscCall(PetscCDAppendID(agg_lists, lid, lidj + my0));
              } else {
                PetscCall(PetscCDAppendID(agg_lists, lid, lidj));
              }
              lid_state[lidj] = MIS_DELETED; /* delete this */
            }
          }
          /* delete ghost adj of lid - deleted ghost done later for strict_aggs */
          if (!strict_aggs) {
            if ((ix = lid_cprowID[lid]) != -1) { /* if I have any ghost neighbors */
              ii  = matB->compressedrow.i;
              n   = ii[ix + 1] - ii[ix];
              idx = matB->j + ii[ix];
              for (j = 0; j < n; j++) {
                cpid   = idx[j]; /* compressed row ID in B mat */
                statej = cpcol_state[cpid];
                if (statej == MIS_NOT_DONE) PetscCall(PetscCDAppendID(agg_lists, lid, nloc + cpid));
              }
            }
          }
        } /* selected */
      } /* not done vertex */
    } /* vertex loop */

    /* update ghost states and count todos */
    if (isMPI) {
      /* scatter states, check for done */
      PetscCall(PetscSFBcastBegin(sf, MPIU_INT, lid_state, cpcol_state, MPI_REPLACE));
      PetscCall(PetscSFBcastEnd(sf, MPIU_INT, lid_state, cpcol_state, MPI_REPLACE));
      ii = matB->compressedrow.i;
      for (ix = 0; ix < matB->compressedrow.nrows; ix++) {
        lid   = matB->compressedrow.rindex[ix]; /* local boundary node */
        state = lid_state[lid];
        if (state == MIS_NOT_DONE) {
          /* look at ghosts */
          n   = ii[ix + 1] - ii[ix];
          idx = matB->j + ii[ix];
          for (j = 0; j < n; j++) {
            cpid   = idx[j]; /* compressed row ID in B mat */
            statej = cpcol_state[cpid];
            if (MIS_IS_SELECTED(statej)) { /* lid is now deleted, do it */
              nDone++;
              lid_state[lid] = MIS_DELETED; /* delete this */
              if (!strict_aggs) {
                lidj = nloc + cpid;
                PetscCall(PetscCDAppendID(agg_lists, lidj, lid));
              } else {
                sgid                = cpcol_gid[cpid];
                lid_parent_gid[lid] = sgid; /* keep track of proc that I belong to */
              }
              break;
            }
          }
        }
      }
      /* all done? */
      t1 = nloc - nDone;
      PetscCallMPI(MPIU_Allreduce(&t1, &t2, 1, MPIU_INT, MPI_SUM, comm)); /* synchronous version */
      if (!t2) break;
    } else break; /* all done */
  } /* outer parallel MIS loop */
  PetscCall(ISRestoreIndices(perm, &perm_ix));
  PetscCall(PetscInfo(info_is, "\t removed %" PetscInt_FMT " of %" PetscInt_FMT " vertices.  %" PetscInt_FMT " selected.\n", nremoved, nloc, nselected));

  /* tell adj who my lid_parent_gid vertices belong to - fill in agg_lists selected ghost lists */
  if (strict_aggs && matB) {
    /* need to copy this to free buffer -- should do this globally */
    PetscCall(PetscMalloc2(num_fine_ghosts, &cpcol_sel_gid, num_fine_ghosts, &icpcol_gid));
    for (cpid = 0; cpid < num_fine_ghosts; cpid++) icpcol_gid[cpid] = cpcol_gid[cpid];

    /* get proc of deleted ghost */
    PetscCall(PetscSFBcastBegin(sf, MPIU_INT, lid_parent_gid, cpcol_sel_gid, MPI_REPLACE));
    PetscCall(PetscSFBcastEnd(sf, MPIU_INT, lid_parent_gid, cpcol_sel_gid, MPI_REPLACE));
    for (cpid = 0; cpid < num_fine_ghosts; cpid++) {
      sgid = cpcol_sel_gid[cpid];
      gid  = icpcol_gid[cpid];
      if (sgid >= my0 && sgid < Iend) { /* I own this deleted */
        slid = sgid - my0;
        PetscCall(PetscCDAppendID(agg_lists, slid, gid));
      }
    }
    PetscCall(PetscFree2(cpcol_sel_gid, icpcol_gid));
  }
  if (isMPI) {
    PetscCall(PetscSFDestroy(&sf));
    PetscCall(PetscFree2(cpcol_gid, cpcol_state));
  }
  PetscCall(PetscFree4(lid_gid, lid_cprowID, lid_removed, lid_state));
  if (strict_aggs) {
    // check sizes -- all vertices must get in graph
    PetscInt aa[2] = {0, nrm_tot}, bb[2], MM;

    PetscCall(PetscFree(lid_parent_gid));
    PetscCall(MatGetSize(Gmat, &MM, NULL));
    // check sizes -- all vertices must get in graph
    PetscCall(PetscCDCount(agg_lists, &aa[0]));
    PetscCallMPI(MPIU_Allreduce(aa, bb, 2, MPIU_INT, MPI_SUM, comm));
    if (MM != bb[0]) PetscCall(PetscInfo(info_is, "Warning: N = %" PetscInt_FMT ", sum of aggregates %" PetscInt_FMT ", %" PetscInt_FMT " removed total\n", MM, bb[0], bb[1]));
    PetscCheck(MM >= bb[0], comm, PETSC_ERR_PLIB, "Sum of aggs is too large");
  }
  PetscCall(ISDestroy(&info_is));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
   MIS coarsen, simple greedy.
*/
static PetscErrorCode MatCoarsenApply_MIS(MatCoarsen coarse)
{
  Mat mat = coarse->graph;

  PetscFunctionBegin;
  if (!coarse->perm) {
    IS       perm;
    PetscInt n, m;
    MPI_Comm comm;

    PetscCall(PetscObjectGetComm((PetscObject)mat, &comm));
    PetscCall(MatGetLocalSize(mat, &m, &n));
    PetscCall(ISCreateStride(comm, m, 0, 1, &perm));
    PetscCall(MatCoarsenApply_MIS_private(perm, mat, coarse->strict_aggs, &coarse->agg_lists));
    PetscCall(ISDestroy(&perm));
  } else {
    PetscCall(MatCoarsenApply_MIS_private(coarse->perm, mat, coarse->strict_aggs, &coarse->agg_lists));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatCoarsenView_MIS(MatCoarsen coarse, PetscViewer viewer)
{
  PetscMPIInt       rank;
  PetscBool         isascii;
  PetscViewerFormat format;

  PetscFunctionBegin;
  PetscCallMPI(MPI_Comm_rank(PetscObjectComm((PetscObject)coarse), &rank));
  PetscCall(PetscObjectTypeCompare((PetscObject)viewer, PETSCVIEWERASCII, &isascii));
  PetscCall(PetscViewerGetFormat(viewer, &format));
  if (isascii && format == PETSC_VIEWER_ASCII_INFO_DETAIL) {
    if (coarse->agg_lists) {
      PetscCall(PetscViewerASCIIPushSynchronized(viewer));
      PetscCall(PetscViewerASCIISynchronizedPrintf(viewer, "  [%d] MIS aggregator\n", rank));
      if (!rank) {
        PetscCDIntNd *pos, *pos2;
        for (PetscInt kk = 0; kk < coarse->agg_lists->size; kk++) {
          PetscCall(PetscCDGetHeadPos(coarse->agg_lists, kk, &pos));
          if ((pos2 = pos)) PetscCall(PetscViewerASCIISynchronizedPrintf(viewer, "selected %" PetscInt_FMT ": ", kk));
          while (pos) {
            PetscInt gid1;
            PetscCall(PetscCDIntNdGetID(pos, &gid1));
            PetscCall(PetscCDGetNextPos(coarse->agg_lists, kk, &pos));
            PetscCall(PetscViewerASCIISynchronizedPrintf(viewer, " %" PetscInt_FMT " ", gid1));
          }
          if (pos2) PetscCall(PetscViewerASCIISynchronizedPrintf(viewer, "\n"));
        }
      }
      PetscCall(PetscViewerFlush(viewer));
      PetscCall(PetscViewerASCIIPopSynchronized(viewer));
    } else {
      PetscCall(PetscViewerASCIIPrintf(viewer, "  MIS aggregator lists are not available\n"));
    }
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*MC
   MATCOARSENMIS - Creates a coarsening object that uses a maximal independent set (MIS) algorithm

   Collective

   Input Parameter:
.  coarse - the coarsen context

   Level: beginner

.seealso: `MatCoarsen`, `MatCoarsenApply()`, `MatCoarsenGetData()`, `MatCoarsenSetType()`, `MatCoarsenType`
M*/
PETSC_EXTERN PetscErrorCode MatCoarsenCreate_MIS(MatCoarsen coarse)
{
  PetscFunctionBegin;
  coarse->ops->apply = MatCoarsenApply_MIS;
  coarse->ops->view  = MatCoarsenView_MIS;
  PetscFunctionReturn(PETSC_SUCCESS);
}
