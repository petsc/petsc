/*
 GAMG geometric-algebric multigrid PC - Mark Adams 2011
 */

#include <../src/ksp/pc/impls/gamg/gamg.h> /*I "petscpc.h" I*/
#include <petscblaslapack.h>
#include <petscdm.h>
#include <petsc/private/kspimpl.h>

typedef struct {
  PetscInt   nsmooths;                     // number of smoothing steps to construct prolongation
  PetscInt   aggressive_coarsening_levels; // number of aggressive coarsening levels (square or MISk)
  PetscInt   aggressive_mis_k;             // the k in MIS-k
  PetscBool  use_aggressive_square_graph;
  PetscBool  use_minimum_degree_ordering;
  PetscBool  use_low_mem_filter;
  PetscBool  graph_symmetrize;
  MatCoarsen crs;
} PC_GAMG_AGG;

/*@
  PCGAMGSetNSmooths - Set number of smoothing steps (1 is typical) used to construct the prolongation operator

  Logically Collective

  Input Parameters:
+ pc - the preconditioner context
- n  - the number of smooths

  Options Database Key:
. -pc_gamg_agg_nsmooths <nsmooth, default=1> - the flag

  Level: intermediate

  Note:
  This is a different concept from the number smoothing steps used during the linear solution process which
  can be set with `-mg_levels_ksp_max_it`

  Developer Note:
  This should be named `PCGAMGAGGSetNSmooths()`.

.seealso: [the Users Manual section on PCGAMG](sec_amg), [the Users Manual section on PCMG](sec_mg), [](ch_ksp), `PCMG`, `PCGAMG`
@*/
PetscErrorCode PCGAMGSetNSmooths(PC pc, PetscInt n)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc, PC_CLASSID, 1);
  PetscValidLogicalCollectiveInt(pc, n, 2);
  PetscTryMethod(pc, "PCGAMGSetNSmooths_C", (PC, PetscInt), (pc, n));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PCGAMGSetNSmooths_AGG(PC pc, PetscInt n)
{
  PC_MG       *mg          = (PC_MG *)pc->data;
  PC_GAMG     *pc_gamg     = (PC_GAMG *)mg->innerctx;
  PC_GAMG_AGG *pc_gamg_agg = (PC_GAMG_AGG *)pc_gamg->subctx;

  PetscFunctionBegin;
  pc_gamg_agg->nsmooths = n;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PCGAMGSetAggressiveLevels -  Use aggressive coarsening on first n levels

  Logically Collective

  Input Parameters:
+ pc - the preconditioner context
- n  - 0, 1 or more

  Options Database Key:
. -pc_gamg_aggressive_coarsening <n,default = 1> - the flag

  Level: intermediate

  Note:
  By default, aggressive coarsening squares the matrix (computes $ A^T A$) before coarsening. Calling `PCGAMGSetAggressiveSquareGraph()` with a value of `PETSC_FALSE` changes the aggressive coarsening strategy to use MIS-k, see `PCGAMGMISkSetAggressive()`.

.seealso: [the Users Manual section on PCGAMG](sec_amg), [the Users Manual section on PCMG](sec_mg), [](ch_ksp), `PCGAMG`, `PCGAMGSetThreshold()`, `PCGAMGMISkSetAggressive()`, `PCGAMGSetAggressiveSquareGraph()`, `PCGAMGMISkSetMinDegreeOrdering()`, `PCGAMGSetLowMemoryFilter()`
@*/
PetscErrorCode PCGAMGSetAggressiveLevels(PC pc, PetscInt n)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc, PC_CLASSID, 1);
  PetscValidLogicalCollectiveInt(pc, n, 2);
  PetscTryMethod(pc, "PCGAMGSetAggressiveLevels_C", (PC, PetscInt), (pc, n));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PCGAMGMISkSetAggressive - Number (k) distance in MIS coarsening (>2 is 'aggressive')

  Logically Collective

  Input Parameters:
+ pc - the preconditioner context
- n  - 1 or more (default = 2)

  Options Database Key:
. -pc_gamg_aggressive_mis_k <n,default=2> - the flag

  Level: intermediate

.seealso: [the Users Manual section on PCGAMG](sec_amg), [the Users Manual section on PCMG](sec_mg), [](ch_ksp), `PCGAMG`, `PCGAMGSetThreshold()`, `PCGAMGSetAggressiveLevels()`, `PCGAMGSetAggressiveSquareGraph()`, `PCGAMGMISkSetMinDegreeOrdering()`, `PCGAMGSetLowMemoryFilter()`
@*/
PetscErrorCode PCGAMGMISkSetAggressive(PC pc, PetscInt n)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc, PC_CLASSID, 1);
  PetscValidLogicalCollectiveInt(pc, n, 2);
  PetscTryMethod(pc, "PCGAMGMISkSetAggressive_C", (PC, PetscInt), (pc, n));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PCGAMGSetAggressiveSquareGraph - Use graph square, $A^T A$, for aggressive coarsening. Coarsening is slower than the alternative (MIS-2), which is faster and uses less memory

  Logically Collective

  Input Parameters:
+ pc - the preconditioner context
- b  - default true

  Options Database Key:
. -pc_gamg_aggressive_square_graph <bool,default=true> - the flag

  Level: intermediate

  Notes:
  If `b` is `PETSC_FALSE` then MIS-k is used for aggressive coarsening, see `PCGAMGMISkSetAggressive()`

  Squaring the matrix to perform the aggressive coarsening is slower and requires more memory than using MIS-k, but may result in a better preconditioner
  that converges faster.

.seealso: [the Users Manual section on PCGAMG](sec_amg), [the Users Manual section on PCMG](sec_mg), [](ch_ksp), `PCGAMG`, `PCGAMGSetThreshold()`, `PCGAMGSetAggressiveLevels()`, `PCGAMGMISkSetAggressive()`, `PCGAMGMISkSetMinDegreeOrdering()`, `PCGAMGSetLowMemoryFilter()`
@*/
PetscErrorCode PCGAMGSetAggressiveSquareGraph(PC pc, PetscBool b)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc, PC_CLASSID, 1);
  PetscValidLogicalCollectiveBool(pc, b, 2);
  PetscTryMethod(pc, "PCGAMGSetAggressiveSquareGraph_C", (PC, PetscBool), (pc, b));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PCGAMGMISkSetMinDegreeOrdering - Use minimum degree ordering in greedy MIS algorithm

  Logically Collective

  Input Parameters:
+ pc - the preconditioner context
- b  - default false

  Options Database Key:
. -pc_gamg_mis_k_minimum_degree_ordering <bool,default=false> - the flag

  Level: intermediate

.seealso: [the Users Manual section on PCGAMG](sec_amg), [the Users Manual section on PCMG](sec_mg), [](ch_ksp), `PCGAMG`, `PCGAMGSetThreshold()`, `PCGAMGSetAggressiveLevels()`, `PCGAMGMISkSetAggressive()`, `PCGAMGSetAggressiveSquareGraph()`, `PCGAMGSetLowMemoryFilter()`
@*/
PetscErrorCode PCGAMGMISkSetMinDegreeOrdering(PC pc, PetscBool b)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc, PC_CLASSID, 1);
  PetscValidLogicalCollectiveBool(pc, b, 2);
  PetscTryMethod(pc, "PCGAMGMISkSetMinDegreeOrdering_C", (PC, PetscBool), (pc, b));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PCGAMGSetLowMemoryFilter - Use low memory graph/matrix filter

  Logically Collective

  Input Parameters:
+ pc - the preconditioner context
- b  - default false

  Options Database Key:
. -pc_gamg_low_memory_threshold_filter <bool,default=false> - the flag

  Level: intermediate

.seealso: [the Users Manual section on PCGAMG](sec_amg), [the Users Manual section on PCMG](sec_mg), `PCGAMG`, `PCGAMGSetThreshold()`, `PCGAMGSetAggressiveLevels()`,
  `PCGAMGMISkSetAggressive()`, `PCGAMGSetAggressiveSquareGraph()`, `PCGAMGMISkSetMinDegreeOrdering()`
@*/
PetscErrorCode PCGAMGSetLowMemoryFilter(PC pc, PetscBool b)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc, PC_CLASSID, 1);
  PetscValidLogicalCollectiveBool(pc, b, 2);
  PetscTryMethod(pc, "PCGAMGSetLowMemoryFilter_C", (PC, PetscBool), (pc, b));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PCGAMGSetGraphSymmetrize - Symmetrize graph used for coarsening. Defaults to true, but if matrix has symmetric attribute, then not needed since the graph is already known to be symmetric

  Logically Collective

  Input Parameters:
+ pc - the preconditioner context
- b  - default true

  Options Database Key:
. -pc_gamg_graph_symmetrize <bool,default=true> - the flag

  Level: intermediate

.seealso: [the Users Manual section on PCGAMG](sec_amg), [the Users Manual section on PCMG](sec_mg), `PCGAMG`, `PCGAMGSetThreshold()`, `PCGAMGSetAggressiveLevels()`, `MatCreateGraph()`,
  `PCGAMGMISkSetAggressive()`, `PCGAMGSetAggressiveSquareGraph()`, `PCGAMGMISkSetMinDegreeOrdering()`
@*/
PetscErrorCode PCGAMGSetGraphSymmetrize(PC pc, PetscBool b)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc, PC_CLASSID, 1);
  PetscValidLogicalCollectiveBool(pc, b, 2);
  PetscTryMethod(pc, "PCGAMGSetGraphSymmetrize_C", (PC, PetscBool), (pc, b));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PCGAMGSetAggressiveLevels_AGG(PC pc, PetscInt n)
{
  PC_MG       *mg          = (PC_MG *)pc->data;
  PC_GAMG     *pc_gamg     = (PC_GAMG *)mg->innerctx;
  PC_GAMG_AGG *pc_gamg_agg = (PC_GAMG_AGG *)pc_gamg->subctx;

  PetscFunctionBegin;
  pc_gamg_agg->aggressive_coarsening_levels = n;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PCGAMGMISkSetAggressive_AGG(PC pc, PetscInt n)
{
  PC_MG       *mg          = (PC_MG *)pc->data;
  PC_GAMG     *pc_gamg     = (PC_GAMG *)mg->innerctx;
  PC_GAMG_AGG *pc_gamg_agg = (PC_GAMG_AGG *)pc_gamg->subctx;

  PetscFunctionBegin;
  pc_gamg_agg->aggressive_mis_k = n;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PCGAMGSetAggressiveSquareGraph_AGG(PC pc, PetscBool b)
{
  PC_MG       *mg          = (PC_MG *)pc->data;
  PC_GAMG     *pc_gamg     = (PC_GAMG *)mg->innerctx;
  PC_GAMG_AGG *pc_gamg_agg = (PC_GAMG_AGG *)pc_gamg->subctx;

  PetscFunctionBegin;
  pc_gamg_agg->use_aggressive_square_graph = b;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PCGAMGSetLowMemoryFilter_AGG(PC pc, PetscBool b)
{
  PC_MG       *mg          = (PC_MG *)pc->data;
  PC_GAMG     *pc_gamg     = (PC_GAMG *)mg->innerctx;
  PC_GAMG_AGG *pc_gamg_agg = (PC_GAMG_AGG *)pc_gamg->subctx;

  PetscFunctionBegin;
  pc_gamg_agg->use_low_mem_filter = b;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PCGAMGSetGraphSymmetrize_AGG(PC pc, PetscBool b)
{
  PC_MG       *mg          = (PC_MG *)pc->data;
  PC_GAMG     *pc_gamg     = (PC_GAMG *)mg->innerctx;
  PC_GAMG_AGG *pc_gamg_agg = (PC_GAMG_AGG *)pc_gamg->subctx;

  PetscFunctionBegin;
  pc_gamg_agg->graph_symmetrize = b;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PCGAMGMISkSetMinDegreeOrdering_AGG(PC pc, PetscBool b)
{
  PC_MG       *mg          = (PC_MG *)pc->data;
  PC_GAMG     *pc_gamg     = (PC_GAMG *)mg->innerctx;
  PC_GAMG_AGG *pc_gamg_agg = (PC_GAMG_AGG *)pc_gamg->subctx;

  PetscFunctionBegin;
  pc_gamg_agg->use_minimum_degree_ordering = b;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PCSetFromOptions_GAMG_AGG(PC pc, PetscOptionItems PetscOptionsObject)
{
  PC_MG       *mg          = (PC_MG *)pc->data;
  PC_GAMG     *pc_gamg     = (PC_GAMG *)mg->innerctx;
  PC_GAMG_AGG *pc_gamg_agg = (PC_GAMG_AGG *)pc_gamg->subctx;
  PetscBool    n_aggressive_flg, old_sq_provided = PETSC_FALSE, new_sq_provided = PETSC_FALSE, new_sqr_graph = pc_gamg_agg->use_aggressive_square_graph;
  PetscInt     nsq_graph_old = 0;

  PetscFunctionBegin;
  PetscOptionsHeadBegin(PetscOptionsObject, "GAMG-AGG options");
  PetscCall(PetscOptionsInt("-pc_gamg_agg_nsmooths", "number of smoothing steps to construct prolongation, usually 1", "PCGAMGSetNSmooths", pc_gamg_agg->nsmooths, &pc_gamg_agg->nsmooths, NULL));
  // aggressive coarsening logic with deprecated -pc_gamg_square_graph
  PetscCall(PetscOptionsInt("-pc_gamg_aggressive_coarsening", "Number of aggressive coarsening (MIS-2) levels from finest", "PCGAMGSetAggressiveLevels", pc_gamg_agg->aggressive_coarsening_levels, &pc_gamg_agg->aggressive_coarsening_levels, &n_aggressive_flg));
  if (!n_aggressive_flg)
    PetscCall(PetscOptionsInt("-pc_gamg_square_graph", "Number of aggressive coarsening (MIS-2) levels from finest (deprecated alias for -pc_gamg_aggressive_coarsening)", "PCGAMGSetAggressiveLevels", nsq_graph_old, &nsq_graph_old, &old_sq_provided));
  PetscCall(PetscOptionsBool("-pc_gamg_aggressive_square_graph", "Use square graph $ (A^T A)$ for aggressive coarsening, if false, MIS-k (k=2) is used, see PCGAMGMISkSetAggressive()", "PCGAMGSetAggressiveSquareGraph", new_sqr_graph, &pc_gamg_agg->use_aggressive_square_graph, &new_sq_provided));
  if (!new_sq_provided && old_sq_provided) {
    pc_gamg_agg->aggressive_coarsening_levels = nsq_graph_old; // could be zero
    pc_gamg_agg->use_aggressive_square_graph  = PETSC_TRUE;
  }
  if (new_sq_provided && old_sq_provided)
    PetscCall(PetscInfo(pc, "Warning: both -pc_gamg_square_graph and -pc_gamg_aggressive_coarsening are used. -pc_gamg_square_graph is deprecated, Number of aggressive levels is %" PetscInt_FMT "\n", pc_gamg_agg->aggressive_coarsening_levels));
  PetscCall(PetscOptionsBool("-pc_gamg_mis_k_minimum_degree_ordering", "Use minimum degree ordering for greedy MIS", "PCGAMGMISkSetMinDegreeOrdering", pc_gamg_agg->use_minimum_degree_ordering, &pc_gamg_agg->use_minimum_degree_ordering, NULL));
  PetscCall(PetscOptionsBool("-pc_gamg_low_memory_threshold_filter", "Use the (built-in) low memory graph/matrix filter", "PCGAMGSetLowMemoryFilter", pc_gamg_agg->use_low_mem_filter, &pc_gamg_agg->use_low_mem_filter, NULL));
  PetscCall(PetscOptionsInt("-pc_gamg_aggressive_mis_k", "Number of levels of multigrid to use.", "PCGAMGMISkSetAggressive", pc_gamg_agg->aggressive_mis_k, &pc_gamg_agg->aggressive_mis_k, NULL));
  PetscCall(PetscOptionsBool("-pc_gamg_graph_symmetrize", "Symmetrize graph for coarsening", "PCGAMGSetGraphSymmetrize", pc_gamg_agg->graph_symmetrize, &pc_gamg_agg->graph_symmetrize, NULL));
  PetscOptionsHeadEnd();
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PCDestroy_GAMG_AGG(PC pc)
{
  PC_MG       *mg          = (PC_MG *)pc->data;
  PC_GAMG     *pc_gamg     = (PC_GAMG *)mg->innerctx;
  PC_GAMG_AGG *pc_gamg_agg = (PC_GAMG_AGG *)pc_gamg->subctx;

  PetscFunctionBegin;
  PetscCall(MatCoarsenDestroy(&pc_gamg_agg->crs));
  PetscCall(PetscFree(pc_gamg->subctx));
  PetscCall(PetscObjectComposeFunction((PetscObject)pc, "PCGAMGSetNSmooths_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)pc, "PCGAMGSetAggressiveLevels_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)pc, "PCGAMGMISkSetAggressive_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)pc, "PCGAMGMISkSetMinDegreeOrdering_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)pc, "PCGAMGSetLowMemoryFilter_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)pc, "PCGAMGSetAggressiveSquareGraph_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)pc, "PCGAMGSetGraphSymmetrize_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)pc, "PCSetCoordinates_C", NULL));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
   PCSetCoordinates_AGG

   Collective

   Input Parameter:
   . pc - the preconditioner context
   . ndm - dimension of data (used for dof/vertex for Stokes)
   . a_nloc - number of vertices local
   . coords - [a_nloc][ndm] - interleaved coordinate data: {x_0, y_0, z_0, x_1, y_1, ...}
*/

static PetscErrorCode PCSetCoordinates_AGG(PC pc, PetscInt ndm, PetscInt a_nloc, PetscReal *coords)
{
  PC_MG   *mg      = (PC_MG *)pc->data;
  PC_GAMG *pc_gamg = (PC_GAMG *)mg->innerctx;
  PetscInt arrsz, kk, ii, jj, nloc, ndatarows, ndf;
  Mat      mat = pc->pmat;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc, PC_CLASSID, 1);
  PetscValidHeaderSpecific(mat, MAT_CLASSID, 1);
  nloc = a_nloc;

  /* SA: null space vectors */
  PetscCall(MatGetBlockSize(mat, &ndf));               /* this does not work for Stokes */
  if (coords && ndf == 1) pc_gamg->data_cell_cols = 1; /* scalar w/ coords and SA (not needed) */
  else if (coords) {
    PetscCheck(ndm <= ndf, PETSC_COMM_SELF, PETSC_ERR_PLIB, "degrees of motion %" PetscInt_FMT " > block size %" PetscInt_FMT, ndm, ndf);
    pc_gamg->data_cell_cols = (ndm == 2 ? 3 : 6); /* displacement elasticity */
    if (ndm != ndf) PetscCheck(pc_gamg->data_cell_cols == ndf, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Don't know how to create null space for ndm=%" PetscInt_FMT ", ndf=%" PetscInt_FMT ".  Use MatSetNearNullSpace().", ndm, ndf);
  } else pc_gamg->data_cell_cols = ndf; /* no data, force SA with constant null space vectors */
  pc_gamg->data_cell_rows = ndatarows = ndf;
  PetscCheck(pc_gamg->data_cell_cols > 0, PETSC_COMM_SELF, PETSC_ERR_PLIB, "pc_gamg->data_cell_cols %" PetscInt_FMT " <= 0", pc_gamg->data_cell_cols);
  arrsz = nloc * pc_gamg->data_cell_rows * pc_gamg->data_cell_cols;

  if (!pc_gamg->data || (pc_gamg->data_sz != arrsz)) {
    PetscCall(PetscFree(pc_gamg->data));
    PetscCall(PetscMalloc1(arrsz + 1, &pc_gamg->data));
  }
  /* copy data in - column-oriented */
  for (kk = 0; kk < nloc; kk++) {
    const PetscInt M    = nloc * pc_gamg->data_cell_rows; /* stride into data */
    PetscReal     *data = &pc_gamg->data[kk * ndatarows]; /* start of cell */

    if (pc_gamg->data_cell_cols == 1) *data = 1.0;
    else {
      /* translational modes */
      for (ii = 0; ii < ndatarows; ii++) {
        for (jj = 0; jj < ndatarows; jj++) {
          if (ii == jj) data[ii * M + jj] = 1.0;
          else data[ii * M + jj] = 0.0;
        }
      }

      /* rotational modes */
      if (coords) {
        if (ndm == 2) {
          data += 2 * M;
          data[0] = -coords[2 * kk + 1];
          data[1] = coords[2 * kk];
        } else {
          data += 3 * M;
          data[0]         = 0.0;
          data[M + 0]     = coords[3 * kk + 2];
          data[2 * M + 0] = -coords[3 * kk + 1];
          data[1]         = -coords[3 * kk + 2];
          data[M + 1]     = 0.0;
          data[2 * M + 1] = coords[3 * kk];
          data[2]         = coords[3 * kk + 1];
          data[M + 2]     = -coords[3 * kk];
          data[2 * M + 2] = 0.0;
        }
      }
    }
  }
  pc_gamg->data_sz = arrsz;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
   PCSetData_AGG - called if data is not set with PCSetCoordinates.
      Looks in Mat for near null space.
      Does not work for Stokes

  Input Parameter:
   . pc -
   . a_A - matrix to get (near) null space out of.
*/
static PetscErrorCode PCSetData_AGG(PC pc, Mat a_A)
{
  PC_MG       *mg      = (PC_MG *)pc->data;
  PC_GAMG     *pc_gamg = (PC_GAMG *)mg->innerctx;
  MatNullSpace mnull;

  PetscFunctionBegin;
  PetscCall(MatGetNearNullSpace(a_A, &mnull));
  if (!mnull) {
    DM dm;

    PetscCall(PCGetDM(pc, &dm));
    if (!dm) PetscCall(MatGetDM(a_A, &dm));
    if (dm) {
      PetscObject deformation;
      PetscInt    Nf;

      PetscCall(DMGetNumFields(dm, &Nf));
      if (Nf) {
        PetscCall(DMGetField(dm, 0, NULL, &deformation));
        if (deformation) {
          PetscCall(PetscObjectQuery(deformation, "nearnullspace", (PetscObject *)&mnull));
          if (!mnull) PetscCall(PetscObjectQuery(deformation, "nullspace", (PetscObject *)&mnull));
        }
      }
    }
  }

  if (!mnull) {
    PetscInt bs, NN, MM;

    PetscCall(MatGetBlockSize(a_A, &bs));
    PetscCall(MatGetLocalSize(a_A, &MM, &NN));
    PetscCheck(MM % bs == 0, PETSC_COMM_SELF, PETSC_ERR_PLIB, "MM %" PetscInt_FMT " must be divisible by bs %" PetscInt_FMT, MM, bs);
    PetscCall(PCSetCoordinates_AGG(pc, bs, MM / bs, NULL));
  } else {
    PetscReal         *nullvec;
    PetscBool          has_const;
    PetscInt           i, j, mlocal, nvec, bs;
    const Vec         *vecs;
    const PetscScalar *v;

    PetscCall(MatGetLocalSize(a_A, &mlocal, NULL));
    PetscCall(MatNullSpaceGetVecs(mnull, &has_const, &nvec, &vecs));
    for (i = 0; i < nvec; i++) {
      PetscCall(VecGetLocalSize(vecs[i], &j));
      PetscCheck(j == mlocal, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Attached null space vector size %" PetscInt_FMT " != matrix size %" PetscInt_FMT, j, mlocal);
    }
    pc_gamg->data_sz = (nvec + !!has_const) * mlocal;
    PetscCall(PetscMalloc1((nvec + !!has_const) * mlocal, &nullvec));
    if (has_const)
      for (i = 0; i < mlocal; i++) nullvec[i] = 1.0;
    for (i = 0; i < nvec; i++) {
      PetscCall(VecGetArrayRead(vecs[i], &v));
      for (j = 0; j < mlocal; j++) nullvec[(i + !!has_const) * mlocal + j] = PetscRealPart(v[j]);
      PetscCall(VecRestoreArrayRead(vecs[i], &v));
    }
    pc_gamg->data           = nullvec;
    pc_gamg->data_cell_cols = (nvec + !!has_const);
    PetscCall(MatGetBlockSize(a_A, &bs));
    pc_gamg->data_cell_rows = bs;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
  formProl0 - collect null space data for each aggregate, do QR, put R in coarse grid data and Q in P_0

  Input Parameter:
   . agg_llists - list of arrays with aggregates -- list from selected vertices of aggregate unselected vertices
   . bs - row block size
   . nSAvec - column bs of new P
   . my0crs - global index of start of locals
   . data_stride - bs*(nloc nodes + ghost nodes) [data_stride][nSAvec]
   . data_in[data_stride*nSAvec] - local data on fine grid
   . flid_fgid[data_stride/bs] - make local to global IDs, includes ghosts in 'locals_llist'

  Output Parameter:
   . a_data_out - in with fine grid data (w/ghosts), out with coarse grid data
   . a_Prol - prolongation operator
*/
static PetscErrorCode formProl0(PetscCoarsenData *agg_llists, PetscInt bs, PetscInt nSAvec, PetscInt my0crs, PetscInt data_stride, PetscReal data_in[], const PetscInt flid_fgid[], PetscReal **a_data_out, Mat a_Prol)
{
  PetscInt      Istart, my0, Iend, nloc, clid, flid = 0, aggID, kk, jj, ii, mm, nSelected, minsz, nghosts, out_data_stride;
  MPI_Comm      comm;
  PetscReal    *out_data;
  PetscCDIntNd *pos;
  PetscHMapI    fgid_flid;

  PetscFunctionBegin;
  PetscCall(PetscObjectGetComm((PetscObject)a_Prol, &comm));
  PetscCall(MatGetOwnershipRange(a_Prol, &Istart, &Iend));
  nloc = (Iend - Istart) / bs;
  my0  = Istart / bs;
  PetscCheck((Iend - Istart) % bs == 0, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Iend %" PetscInt_FMT " - Istart %" PetscInt_FMT " must be divisible by bs %" PetscInt_FMT, Iend, Istart, bs);
  Iend /= bs;
  nghosts = data_stride / bs - nloc;

  PetscCall(PetscHMapICreateWithSize(2 * nghosts + 1, &fgid_flid));

  for (kk = 0; kk < nghosts; kk++) PetscCall(PetscHMapISet(fgid_flid, flid_fgid[nloc + kk], nloc + kk));

  /* count selected -- same as number of cols of P */
  for (nSelected = mm = 0; mm < nloc; mm++) {
    PetscBool ise;

    PetscCall(PetscCDIsEmptyAt(agg_llists, mm, &ise));
    if (!ise) nSelected++;
  }
  PetscCall(MatGetOwnershipRangeColumn(a_Prol, &ii, &jj));
  PetscCheck((ii / nSAvec) == my0crs, PETSC_COMM_SELF, PETSC_ERR_PLIB, "ii %" PetscInt_FMT " /nSAvec %" PetscInt_FMT "  != my0crs %" PetscInt_FMT, ii, nSAvec, my0crs);
  PetscCheck(nSelected == (jj - ii) / nSAvec, PETSC_COMM_SELF, PETSC_ERR_PLIB, "nSelected %" PetscInt_FMT " != (jj %" PetscInt_FMT " - ii %" PetscInt_FMT ")/nSAvec %" PetscInt_FMT, nSelected, jj, ii, nSAvec);

  /* aloc space for coarse point data (output) */
  out_data_stride = nSelected * nSAvec;

  PetscCall(PetscMalloc1(out_data_stride * nSAvec, &out_data));
  for (ii = 0; ii < out_data_stride * nSAvec; ii++) out_data[ii] = PETSC_MAX_REAL;
  *a_data_out = out_data; /* output - stride nSelected*nSAvec */

  /* find points and set prolongation */
  minsz = 100;
  for (mm = clid = 0; mm < nloc; mm++) {
    PetscCall(PetscCDCountAt(agg_llists, mm, &jj));
    if (jj > 0) {
      const PetscInt lid = mm, cgid = my0crs + clid;
      PetscInt       cids[100]; /* max bs */
      PetscBLASInt   asz, M, N, INFO;
      PetscBLASInt   Mdata, LDA, LWORK;
      PetscScalar   *qqc, *qqr, *TAU, *WORK;
      PetscInt      *fids;
      PetscReal     *data;

      PetscCall(PetscBLASIntCast(jj, &asz));
      PetscCall(PetscBLASIntCast(asz * bs, &M));
      PetscCall(PetscBLASIntCast(nSAvec, &N));
      PetscCall(PetscBLASIntCast(M + ((N - M > 0) ? N - M : 0), &Mdata));
      PetscCall(PetscBLASIntCast(Mdata, &LDA));
      PetscCall(PetscBLASIntCast(N * bs, &LWORK));
      /* count agg */
      if (asz < minsz) minsz = asz;

      /* get block */
      PetscCall(PetscMalloc5(Mdata * N, &qqc, M * N, &qqr, N, &TAU, LWORK, &WORK, M, &fids));

      aggID = 0;
      PetscCall(PetscCDGetHeadPos(agg_llists, lid, &pos));
      while (pos) {
        PetscInt gid1;

        PetscCall(PetscCDIntNdGetID(pos, &gid1));
        PetscCall(PetscCDGetNextPos(agg_llists, lid, &pos));

        if (gid1 >= my0 && gid1 < Iend) flid = gid1 - my0;
        else {
          PetscCall(PetscHMapIGet(fgid_flid, gid1, &flid));
          PetscCheck(flid >= 0, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Cannot find gid1 in table");
        }
        /* copy in B_i matrix - column-oriented */
        data = &data_in[flid * bs];
        for (ii = 0; ii < bs; ii++) {
          for (jj = 0; jj < N; jj++) {
            PetscReal d = data[jj * data_stride + ii];

            qqc[jj * Mdata + aggID * bs + ii] = d;
          }
        }
        /* set fine IDs */
        for (kk = 0; kk < bs; kk++) fids[aggID * bs + kk] = flid_fgid[flid] * bs + kk;
        aggID++;
      }

      /* pad with zeros */
      for (ii = asz * bs; ii < Mdata; ii++) {
        for (jj = 0; jj < N; jj++, kk++) qqc[jj * Mdata + ii] = .0;
      }

      /* QR */
      PetscCall(PetscFPTrapPush(PETSC_FP_TRAP_OFF));
      PetscCallBLAS("LAPACKgeqrf", LAPACKgeqrf_(&Mdata, &N, qqc, &LDA, TAU, WORK, &LWORK, &INFO));
      PetscCall(PetscFPTrapPop());
      PetscCheck(INFO == 0, PETSC_COMM_SELF, PETSC_ERR_PLIB, "xGEQRF error");
      /* get R - column-oriented - output B_{i+1} */
      {
        PetscReal *data = &out_data[clid * nSAvec];

        for (jj = 0; jj < nSAvec; jj++) {
          for (ii = 0; ii < nSAvec; ii++) {
            PetscCheck(data[jj * out_data_stride + ii] == PETSC_MAX_REAL, PETSC_COMM_SELF, PETSC_ERR_PLIB, "data[jj*out_data_stride + ii] != %e", (double)PETSC_MAX_REAL);
            if (ii <= jj) data[jj * out_data_stride + ii] = PetscRealPart(qqc[jj * Mdata + ii]);
            else data[jj * out_data_stride + ii] = 0.;
          }
        }
      }

      /* get Q - row-oriented */
      PetscCallBLAS("LAPACKorgqr", LAPACKorgqr_(&Mdata, &N, &N, qqc, &LDA, TAU, WORK, &LWORK, &INFO));
      PetscCheck(INFO == 0, PETSC_COMM_SELF, PETSC_ERR_PLIB, "xORGQR error arg %" PetscBLASInt_FMT, -INFO);

      for (ii = 0; ii < M; ii++) {
        for (jj = 0; jj < N; jj++) qqr[N * ii + jj] = qqc[jj * Mdata + ii];
      }

      /* add diagonal block of P0 */
      for (kk = 0; kk < N; kk++) cids[kk] = N * cgid + kk; /* global col IDs in P0 */
      PetscCall(MatSetValues(a_Prol, M, fids, N, cids, qqr, INSERT_VALUES));
      PetscCall(PetscFree5(qqc, qqr, TAU, WORK, fids));
      clid++;
    } /* coarse agg */
  } /* for all fine nodes */
  PetscCall(MatAssemblyBegin(a_Prol, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(a_Prol, MAT_FINAL_ASSEMBLY));
  PetscCall(PetscHMapIDestroy(&fgid_flid));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PCView_GAMG_AGG(PC pc, PetscViewer viewer)
{
  PC_MG       *mg          = (PC_MG *)pc->data;
  PC_GAMG     *pc_gamg     = (PC_GAMG *)mg->innerctx;
  PC_GAMG_AGG *pc_gamg_agg = (PC_GAMG_AGG *)pc_gamg->subctx;

  PetscFunctionBegin;
  PetscCall(PetscViewerASCIIPrintf(viewer, "      AGG specific options\n"));
  PetscCall(PetscViewerASCIIPrintf(viewer, "        Number of levels of aggressive coarsening %" PetscInt_FMT "\n", pc_gamg_agg->aggressive_coarsening_levels));
  if (pc_gamg_agg->aggressive_coarsening_levels > 0) {
    PetscCall(PetscViewerASCIIPrintf(viewer, "        %s aggressive coarsening\n", !pc_gamg_agg->use_aggressive_square_graph ? "MIS-k" : "Square graph"));
    if (!pc_gamg_agg->use_aggressive_square_graph) PetscCall(PetscViewerASCIIPrintf(viewer, "        MIS-%" PetscInt_FMT " coarsening on aggressive levels\n", pc_gamg_agg->aggressive_mis_k));
  }
  PetscCall(PetscViewerASCIIPushTab(viewer));
  PetscCall(PetscViewerASCIIPushTab(viewer));
  PetscCall(PetscViewerASCIIPushTab(viewer));
  PetscCall(PetscViewerASCIIPushTab(viewer));
  if (pc_gamg_agg->crs) PetscCall(MatCoarsenView(pc_gamg_agg->crs, viewer));
  else PetscCall(PetscViewerASCIIPrintf(viewer, "Coarsening algorithm not yet selected\n"));
  PetscCall(PetscViewerASCIIPopTab(viewer));
  PetscCall(PetscViewerASCIIPopTab(viewer));
  PetscCall(PetscViewerASCIIPopTab(viewer));
  PetscCall(PetscViewerASCIIPopTab(viewer));
  PetscCall(PetscViewerASCIIPrintf(viewer, "        Number smoothing steps to construct prolongation %" PetscInt_FMT "\n", pc_gamg_agg->nsmooths));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PCGAMGCreateGraph_AGG(PC pc, Mat Amat, Mat *a_Gmat)
{
  PC_MG          *mg          = (PC_MG *)pc->data;
  PC_GAMG        *pc_gamg     = (PC_GAMG *)mg->innerctx;
  PC_GAMG_AGG    *pc_gamg_agg = (PC_GAMG_AGG *)pc_gamg->subctx;
  const PetscReal vfilter     = pc_gamg->threshold[pc_gamg->current_level];
  PetscBool       ishem, ismis;
  const char     *prefix;
  MatInfo         info0, info1;
  PetscInt        bs;

  PetscFunctionBegin;
  PetscCall(PetscLogEventBegin(petsc_gamg_setup_events[GAMG_COARSEN], 0, 0, 0, 0));
  /* Note: depending on the algorithm that will be used for computing the coarse grid points this should pass PETSC_TRUE or PETSC_FALSE as the first argument */
  /* MATCOARSENHEM requires numerical weights for edges so ensure they are computed */
  PetscCall(MatCoarsenDestroy(&pc_gamg_agg->crs));
  PetscCall(MatCoarsenCreate(PetscObjectComm((PetscObject)pc), &pc_gamg_agg->crs));
  PetscCall(PetscObjectGetOptionsPrefix((PetscObject)pc, &prefix));
  PetscCall(PetscObjectSetOptionsPrefix((PetscObject)pc_gamg_agg->crs, prefix));
  PetscCall(PetscObjectAppendOptionsPrefix((PetscObject)pc_gamg_agg->crs, "pc_gamg_"));
  PetscCall(MatCoarsenSetFromOptions(pc_gamg_agg->crs));
  PetscCall(MatGetBlockSize(Amat, &bs));
  // check for valid indices wrt bs
  for (int ii = 0; ii < pc_gamg_agg->crs->strength_index_size; ii++) {
    PetscCheck(pc_gamg_agg->crs->strength_index[ii] >= 0 && pc_gamg_agg->crs->strength_index[ii] < bs, PetscObjectComm((PetscObject)pc), PETSC_ERR_ARG_WRONG, "Indices (%" PetscInt_FMT ") must be non-negative and < block size (%" PetscInt_FMT "), NB, can not use -mat_coarsen_strength_index with -mat_coarsen_strength_index",
               pc_gamg_agg->crs->strength_index[ii], bs);
  }
  PetscCall(PetscObjectTypeCompare((PetscObject)pc_gamg_agg->crs, MATCOARSENHEM, &ishem));
  if (ishem) {
    if (pc_gamg_agg->aggressive_coarsening_levels) PetscCall(PetscInfo(pc, "HEM and aggressive coarsening ignored: HEM using %" PetscInt_FMT " iterations\n", pc_gamg_agg->crs->max_it));
    pc_gamg_agg->aggressive_coarsening_levels = 0;                                         // aggressive and HEM does not make sense
    PetscCall(MatCoarsenSetMaximumIterations(pc_gamg_agg->crs, pc_gamg_agg->crs->max_it)); // for code coverage
    PetscCall(MatCoarsenSetThreshold(pc_gamg_agg->crs, vfilter));                          // for code coverage
  } else {
    PetscCall(PetscObjectTypeCompare((PetscObject)pc_gamg_agg->crs, MATCOARSENMIS, &ismis));
    if (ismis && pc_gamg_agg->aggressive_coarsening_levels && !pc_gamg_agg->use_aggressive_square_graph) {
      PetscCall(PetscInfo(pc, "MIS and aggressive coarsening and no square graph: force square graph\n"));
      pc_gamg_agg->use_aggressive_square_graph = PETSC_TRUE;
    }
  }
  PetscCall(PetscLogEventEnd(petsc_gamg_setup_events[GAMG_COARSEN], 0, 0, 0, 0));
  PetscCall(PetscLogEventBegin(petsc_gamg_setup_events[GAMG_GRAPH], 0, 0, 0, 0));
  PetscCall(MatGetInfo(Amat, MAT_LOCAL, &info0)); /* global reduction */

  if (ishem || pc_gamg_agg->use_low_mem_filter) {
    PetscCall(MatCreateGraph(Amat, pc_gamg_agg->graph_symmetrize, (vfilter >= 0 || ishem) ? PETSC_TRUE : PETSC_FALSE, vfilter, pc_gamg_agg->crs->strength_index_size, pc_gamg_agg->crs->strength_index, a_Gmat));
  } else {
    // make scalar graph, symmetrize if not known to be symmetric, scale, but do not filter (expensive)
    PetscCall(MatCreateGraph(Amat, pc_gamg_agg->graph_symmetrize, PETSC_TRUE, -1, pc_gamg_agg->crs->strength_index_size, pc_gamg_agg->crs->strength_index, a_Gmat));
    if (vfilter >= 0) {
      PetscInt           Istart, Iend, ncols, nnz0, nnz1, NN, MM, nloc;
      Mat                tGmat, Gmat = *a_Gmat;
      MPI_Comm           comm;
      const PetscScalar *vals;
      const PetscInt    *idx;
      PetscInt          *d_nnz, *o_nnz, kk, *garray = NULL, *AJ, maxcols = 0;
      MatScalar         *AA; // this is checked in graph
      PetscBool          isseqaij;
      Mat                a, b, c;
      MatType            jtype;

      PetscCall(PetscObjectGetComm((PetscObject)Gmat, &comm));
      PetscCall(PetscObjectBaseTypeCompare((PetscObject)Gmat, MATSEQAIJ, &isseqaij));
      PetscCall(MatGetType(Gmat, &jtype));
      PetscCall(MatCreate(comm, &tGmat));
      PetscCall(MatSetType(tGmat, jtype));

      /* TODO GPU: this can be called when filter = 0 -> Probably provide MatAIJThresholdCompress that compresses the entries below a threshold?
        Also, if the matrix is symmetric, can we skip this
        operation? It can be very expensive on large matrices. */

      // global sizes
      PetscCall(MatGetSize(Gmat, &MM, &NN));
      PetscCall(MatGetOwnershipRange(Gmat, &Istart, &Iend));
      nloc = Iend - Istart;
      PetscCall(PetscMalloc2(nloc, &d_nnz, nloc, &o_nnz));
      if (isseqaij) {
        a = Gmat;
        b = NULL;
      } else {
        Mat_MPIAIJ *d = (Mat_MPIAIJ *)Gmat->data;

        a      = d->A;
        b      = d->B;
        garray = d->garray;
      }
      /* Determine upper bound on non-zeros needed in new filtered matrix */
      for (PetscInt row = 0; row < nloc; row++) {
        PetscCall(MatGetRow(a, row, &ncols, NULL, NULL));
        d_nnz[row] = ncols;
        if (ncols > maxcols) maxcols = ncols;
        PetscCall(MatRestoreRow(a, row, &ncols, NULL, NULL));
      }
      if (b) {
        for (PetscInt row = 0; row < nloc; row++) {
          PetscCall(MatGetRow(b, row, &ncols, NULL, NULL));
          o_nnz[row] = ncols;
          if (ncols > maxcols) maxcols = ncols;
          PetscCall(MatRestoreRow(b, row, &ncols, NULL, NULL));
        }
      }
      PetscCall(MatSetSizes(tGmat, nloc, nloc, MM, MM));
      PetscCall(MatSetBlockSizes(tGmat, 1, 1));
      PetscCall(MatSeqAIJSetPreallocation(tGmat, 0, d_nnz));
      PetscCall(MatMPIAIJSetPreallocation(tGmat, 0, d_nnz, 0, o_nnz));
      PetscCall(MatSetOption(tGmat, MAT_NO_OFF_PROC_ENTRIES, PETSC_TRUE));
      PetscCall(PetscFree2(d_nnz, o_nnz));
      PetscCall(PetscMalloc2(maxcols, &AA, maxcols, &AJ));
      nnz0 = nnz1 = 0;
      for (c = a, kk = 0; c && kk < 2; c = b, kk++) {
        for (PetscInt row = 0, grow = Istart, ncol_row, jj; row < nloc; row++, grow++) {
          PetscCall(MatGetRow(c, row, &ncols, &idx, &vals));
          for (ncol_row = jj = 0; jj < ncols; jj++, nnz0++) {
            PetscScalar sv = PetscAbs(PetscRealPart(vals[jj]));
            if (PetscRealPart(sv) > vfilter) {
              PetscInt cid = idx[jj] + Istart; //diag

              nnz1++;
              if (c != a) cid = garray[idx[jj]];
              AA[ncol_row] = vals[jj];
              AJ[ncol_row] = cid;
              ncol_row++;
            }
          }
          PetscCall(MatRestoreRow(c, row, &ncols, &idx, &vals));
          PetscCall(MatSetValues(tGmat, 1, &grow, ncol_row, AJ, AA, INSERT_VALUES));
        }
      }
      PetscCall(PetscFree2(AA, AJ));
      PetscCall(MatAssemblyBegin(tGmat, MAT_FINAL_ASSEMBLY));
      PetscCall(MatAssemblyEnd(tGmat, MAT_FINAL_ASSEMBLY));
      PetscCall(MatPropagateSymmetryOptions(Gmat, tGmat)); /* Normal Mat options are not relevant ? */
      PetscCall(PetscInfo(pc, "\t %g%% nnz after filtering, with threshold %g, %g nnz ave. (N=%" PetscInt_FMT ", max row size %" PetscInt_FMT "\n", (!nnz0) ? 1. : 100. * (double)nnz1 / (double)nnz0, (double)vfilter, (!nloc) ? 1. : (double)nnz0 / (double)nloc, MM, maxcols));
      PetscCall(MatViewFromOptions(tGmat, NULL, "-mat_filter_graph_view"));
      PetscCall(MatDestroy(&Gmat));
      *a_Gmat = tGmat;
    }
  }

  PetscCall(MatGetInfo(*a_Gmat, MAT_LOCAL, &info1)); /* global reduction */
  if (info0.nz_used > 0) PetscCall(PetscInfo(pc, "Filtering left %g %% edges in graph (%e %e)\n", 100.0 * info1.nz_used * (double)(bs * bs) / info0.nz_used, info0.nz_used, info1.nz_used));
  PetscCall(PetscLogEventEnd(petsc_gamg_setup_events[GAMG_GRAPH], 0, 0, 0, 0));
  PetscFunctionReturn(PETSC_SUCCESS);
}

typedef PetscInt    NState;
static const NState NOT_DONE = -2;
static const NState DELETED  = -1;
static const NState REMOVED  = -3;
#define IS_SELECTED(s) (s != DELETED && s != NOT_DONE && s != REMOVED)

/*
   fixAggregatesWithSquare - greedy grab of with G1 (unsquared graph) -- AIJ specific -- change to fixAggregatesWithSquare -- TODD
     - AGG-MG specific: clears singletons out of 'selected_2'

   Input Parameter:
   . Gmat_2 - global matrix of squared graph (data not defined)
   . Gmat_1 - base graph to grab with base graph
   Input/Output Parameter:
   . aggs_2 - linked list of aggs with gids)
*/
static PetscErrorCode fixAggregatesWithSquare(PC pc, Mat Gmat_2, Mat Gmat_1, PetscCoarsenData *aggs_2)
{
  PetscBool      isMPI;
  Mat_SeqAIJ    *matA_1, *matB_1 = NULL;
  MPI_Comm       comm;
  PetscInt       lid, *ii, *idx, ix, Iend, my0, kk, n, j;
  Mat_MPIAIJ    *mpimat_2 = NULL, *mpimat_1 = NULL;
  const PetscInt nloc = Gmat_2->rmap->n;
  PetscScalar   *cpcol_1_state, *cpcol_2_state, *cpcol_2_par_orig, *lid_parent_gid;
  PetscInt      *lid_cprowID_1 = NULL;
  NState        *lid_state;
  Vec            ghost_par_orig2;
  PetscMPIInt    rank;

  PetscFunctionBegin;
  PetscCall(PetscObjectGetComm((PetscObject)Gmat_2, &comm));
  PetscCallMPI(MPI_Comm_rank(comm, &rank));
  PetscCall(MatGetOwnershipRange(Gmat_1, &my0, &Iend));

  /* get submatrices */
  PetscCall(PetscStrbeginswith(((PetscObject)Gmat_1)->type_name, MATMPIAIJ, &isMPI));
  PetscCall(PetscInfo(pc, "isMPI = %s\n", isMPI ? "yes" : "no"));
  PetscCall(PetscMalloc3(nloc, &lid_state, nloc, &lid_parent_gid, nloc, &lid_cprowID_1));
  for (lid = 0; lid < nloc; lid++) lid_cprowID_1[lid] = -1;
  if (isMPI) {
    /* grab matrix objects */
    mpimat_2 = (Mat_MPIAIJ *)Gmat_2->data;
    mpimat_1 = (Mat_MPIAIJ *)Gmat_1->data;
    matA_1   = (Mat_SeqAIJ *)mpimat_1->A->data;
    matB_1   = (Mat_SeqAIJ *)mpimat_1->B->data;

    /* force compressed row storage for B matrix in AuxMat */
    PetscCall(MatCheckCompressedRow(mpimat_1->B, matB_1->nonzerorowcnt, &matB_1->compressedrow, matB_1->i, Gmat_1->rmap->n, -1.0));
    for (ix = 0; ix < matB_1->compressedrow.nrows; ix++) {
      PetscInt lid = matB_1->compressedrow.rindex[ix];

      PetscCheck(lid <= nloc && lid >= -1, PETSC_COMM_SELF, PETSC_ERR_USER, "lid %" PetscInt_FMT " out of range. nloc = %" PetscInt_FMT, lid, nloc);
      if (lid != -1) lid_cprowID_1[lid] = ix;
    }
  } else {
    PetscBool isAIJ;

    PetscCall(PetscStrbeginswith(((PetscObject)Gmat_1)->type_name, MATSEQAIJ, &isAIJ));
    PetscCheck(isAIJ, PETSC_COMM_SELF, PETSC_ERR_USER, "Require AIJ matrix.");
    matA_1 = (Mat_SeqAIJ *)Gmat_1->data;
  }
  if (nloc > 0) PetscCheck(!matB_1 || matB_1->compressedrow.use, PETSC_COMM_SELF, PETSC_ERR_PLIB, "matB_1 && !matB_1->compressedrow.use: PETSc bug???");
  /* get state of locals and selected gid for deleted */
  for (lid = 0; lid < nloc; lid++) {
    lid_parent_gid[lid] = -1.0;
    lid_state[lid]      = DELETED;
  }

  /* set lid_state */
  for (lid = 0; lid < nloc; lid++) {
    PetscCDIntNd *pos;

    PetscCall(PetscCDGetHeadPos(aggs_2, lid, &pos));
    if (pos) {
      PetscInt gid1;

      PetscCall(PetscCDIntNdGetID(pos, &gid1));
      PetscCheck(gid1 == lid + my0, PETSC_COMM_SELF, PETSC_ERR_PLIB, "gid1 %" PetscInt_FMT " != lid %" PetscInt_FMT " + my0 %" PetscInt_FMT, gid1, lid, my0);
      lid_state[lid] = gid1;
    }
  }

  /* map local to selected local, DELETED means a ghost owns it */
  for (lid = 0; lid < nloc; lid++) {
    NState state = lid_state[lid];

    if (IS_SELECTED(state)) {
      PetscCDIntNd *pos;

      PetscCall(PetscCDGetHeadPos(aggs_2, lid, &pos));
      while (pos) {
        PetscInt gid1;

        PetscCall(PetscCDIntNdGetID(pos, &gid1));
        PetscCall(PetscCDGetNextPos(aggs_2, lid, &pos));
        if (gid1 >= my0 && gid1 < Iend) lid_parent_gid[gid1 - my0] = (PetscScalar)(lid + my0);
      }
    }
  }
  /* get 'cpcol_1/2_state' & cpcol_2_par_orig - uses mpimat_1/2->lvec for temp space */
  if (isMPI) {
    Vec tempVec;

    /* get 'cpcol_1_state' */
    PetscCall(MatCreateVecs(Gmat_1, &tempVec, NULL));
    for (kk = 0, j = my0; kk < nloc; kk++, j++) {
      PetscScalar v = (PetscScalar)lid_state[kk];

      PetscCall(VecSetValues(tempVec, 1, &j, &v, INSERT_VALUES));
    }
    PetscCall(VecAssemblyBegin(tempVec));
    PetscCall(VecAssemblyEnd(tempVec));
    PetscCall(VecScatterBegin(mpimat_1->Mvctx, tempVec, mpimat_1->lvec, INSERT_VALUES, SCATTER_FORWARD));
    PetscCall(VecScatterEnd(mpimat_1->Mvctx, tempVec, mpimat_1->lvec, INSERT_VALUES, SCATTER_FORWARD));
    PetscCall(VecGetArray(mpimat_1->lvec, &cpcol_1_state));
    /* get 'cpcol_2_state' */
    PetscCall(VecScatterBegin(mpimat_2->Mvctx, tempVec, mpimat_2->lvec, INSERT_VALUES, SCATTER_FORWARD));
    PetscCall(VecScatterEnd(mpimat_2->Mvctx, tempVec, mpimat_2->lvec, INSERT_VALUES, SCATTER_FORWARD));
    PetscCall(VecGetArray(mpimat_2->lvec, &cpcol_2_state));
    /* get 'cpcol_2_par_orig' */
    for (kk = 0, j = my0; kk < nloc; kk++, j++) {
      PetscScalar v = lid_parent_gid[kk];

      PetscCall(VecSetValues(tempVec, 1, &j, &v, INSERT_VALUES));
    }
    PetscCall(VecAssemblyBegin(tempVec));
    PetscCall(VecAssemblyEnd(tempVec));
    PetscCall(VecDuplicate(mpimat_2->lvec, &ghost_par_orig2));
    PetscCall(VecScatterBegin(mpimat_2->Mvctx, tempVec, ghost_par_orig2, INSERT_VALUES, SCATTER_FORWARD));
    PetscCall(VecScatterEnd(mpimat_2->Mvctx, tempVec, ghost_par_orig2, INSERT_VALUES, SCATTER_FORWARD));
    PetscCall(VecGetArray(ghost_par_orig2, &cpcol_2_par_orig));

    PetscCall(VecDestroy(&tempVec));
  } /* ismpi */
  for (lid = 0; lid < nloc; lid++) {
    NState state = lid_state[lid];

    if (IS_SELECTED(state)) {
      /* steal locals */
      ii  = matA_1->i;
      n   = ii[lid + 1] - ii[lid];
      idx = matA_1->j + ii[lid];
      for (j = 0; j < n; j++) {
        PetscInt lidj   = idx[j], sgid;
        NState   statej = lid_state[lidj];

        if (statej == DELETED && (sgid = (PetscInt)PetscRealPart(lid_parent_gid[lidj])) != lid + my0) { /* steal local */
          lid_parent_gid[lidj] = (PetscScalar)(lid + my0);                                              /* send this if sgid is not local */
          if (sgid >= my0 && sgid < Iend) {                                                             /* I'm stealing this local from a local sgid */
            PetscInt      hav = 0, slid = sgid - my0, gidj = lidj + my0;
            PetscCDIntNd *pos, *last = NULL;

            /* looking for local from local so id_llist_2 works */
            PetscCall(PetscCDGetHeadPos(aggs_2, slid, &pos));
            while (pos) {
              PetscInt gid;

              PetscCall(PetscCDIntNdGetID(pos, &gid));
              if (gid == gidj) {
                PetscCheck(last, PETSC_COMM_SELF, PETSC_ERR_PLIB, "last cannot be null");
                PetscCall(PetscCDRemoveNextNode(aggs_2, slid, last));
                PetscCall(PetscCDAppendNode(aggs_2, lid, pos));
                hav = 1;
                break;
              } else last = pos;
              PetscCall(PetscCDGetNextPos(aggs_2, slid, &pos));
            }
            if (hav != 1) {
              PetscCheck(hav, PETSC_COMM_SELF, PETSC_ERR_PLIB, "failed to find adj in 'selected' lists - structurally unsymmetric matrix");
              SETERRQ(PETSC_COMM_SELF, PETSC_ERR_PLIB, "found node %" PetscInt_FMT " times???", hav);
            }
          } else { /* I'm stealing this local, owned by a ghost */
            PetscCheck(sgid == -1, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Mat has an un-symmetric graph. Use '-%spc_gamg_sym_graph true' to symmetrize the graph or '-%spc_gamg_threshold -1' if the matrix is structurally symmetric.",
                       ((PetscObject)pc)->prefix ? ((PetscObject)pc)->prefix : "", ((PetscObject)pc)->prefix ? ((PetscObject)pc)->prefix : "");
            PetscCall(PetscCDAppendID(aggs_2, lid, lidj + my0));
          }
        }
      } /* local neighbors */
    } else if (state == DELETED /* && lid_cprowID_1 */) {
      PetscInt sgidold = (PetscInt)PetscRealPart(lid_parent_gid[lid]);

      /* see if I have a selected ghost neighbor that will steal me */
      if ((ix = lid_cprowID_1[lid]) != -1) {
        ii  = matB_1->compressedrow.i;
        n   = ii[ix + 1] - ii[ix];
        idx = matB_1->j + ii[ix];
        for (j = 0; j < n; j++) {
          PetscInt cpid   = idx[j];
          NState   statej = (NState)PetscRealPart(cpcol_1_state[cpid]);

          if (IS_SELECTED(statej) && sgidold != statej) { /* ghost will steal this, remove from my list */
            lid_parent_gid[lid] = (PetscScalar)statej;    /* send who selected */
            if (sgidold >= my0 && sgidold < Iend) {       /* this was mine */
              PetscInt      hav = 0, oldslidj = sgidold - my0;
              PetscCDIntNd *pos, *last        = NULL;

              /* remove from 'oldslidj' list */
              PetscCall(PetscCDGetHeadPos(aggs_2, oldslidj, &pos));
              while (pos) {
                PetscInt gid;

                PetscCall(PetscCDIntNdGetID(pos, &gid));
                if (lid + my0 == gid) {
                  /* id_llist_2[lastid] = id_llist_2[flid];   /\* remove lid from oldslidj list *\/ */
                  PetscCheck(last, PETSC_COMM_SELF, PETSC_ERR_PLIB, "last cannot be null");
                  PetscCall(PetscCDRemoveNextNode(aggs_2, oldslidj, last));
                  /* ghost (PetscScalar)statej will add this later */
                  hav = 1;
                  break;
                } else last = pos;
                PetscCall(PetscCDGetNextPos(aggs_2, oldslidj, &pos));
              }
              if (hav != 1) {
                PetscCheck(hav, PETSC_COMM_SELF, PETSC_ERR_PLIB, "failed to find (hav=%" PetscInt_FMT ") adj in 'selected' lists - structurally unsymmetric matrix", hav);
                SETERRQ(PETSC_COMM_SELF, PETSC_ERR_PLIB, "found node %" PetscInt_FMT " times???", hav);
              }
            } else {
              /* TODO: ghosts remove this later */
            }
          }
        }
      }
    } /* selected/deleted */
  } /* node loop */

  if (isMPI) {
    PetscScalar *cpcol_2_parent, *cpcol_2_gid;
    Vec          tempVec, ghostgids2, ghostparents2;
    PetscInt     cpid, nghost_2;
    PetscHMapI   gid_cpid;

    PetscCall(VecGetSize(mpimat_2->lvec, &nghost_2));
    PetscCall(MatCreateVecs(Gmat_2, &tempVec, NULL));

    /* get 'cpcol_2_parent' */
    for (kk = 0, j = my0; kk < nloc; kk++, j++) PetscCall(VecSetValues(tempVec, 1, &j, &lid_parent_gid[kk], INSERT_VALUES));
    PetscCall(VecAssemblyBegin(tempVec));
    PetscCall(VecAssemblyEnd(tempVec));
    PetscCall(VecDuplicate(mpimat_2->lvec, &ghostparents2));
    PetscCall(VecScatterBegin(mpimat_2->Mvctx, tempVec, ghostparents2, INSERT_VALUES, SCATTER_FORWARD));
    PetscCall(VecScatterEnd(mpimat_2->Mvctx, tempVec, ghostparents2, INSERT_VALUES, SCATTER_FORWARD));
    PetscCall(VecGetArray(ghostparents2, &cpcol_2_parent));

    /* get 'cpcol_2_gid' */
    for (kk = 0, j = my0; kk < nloc; kk++, j++) {
      PetscScalar v = (PetscScalar)j;

      PetscCall(VecSetValues(tempVec, 1, &j, &v, INSERT_VALUES));
    }
    PetscCall(VecAssemblyBegin(tempVec));
    PetscCall(VecAssemblyEnd(tempVec));
    PetscCall(VecDuplicate(mpimat_2->lvec, &ghostgids2));
    PetscCall(VecScatterBegin(mpimat_2->Mvctx, tempVec, ghostgids2, INSERT_VALUES, SCATTER_FORWARD));
    PetscCall(VecScatterEnd(mpimat_2->Mvctx, tempVec, ghostgids2, INSERT_VALUES, SCATTER_FORWARD));
    PetscCall(VecGetArray(ghostgids2, &cpcol_2_gid));
    PetscCall(VecDestroy(&tempVec));

    /* look for deleted ghosts and add to table */
    PetscCall(PetscHMapICreateWithSize(2 * nghost_2 + 1, &gid_cpid));
    for (cpid = 0; cpid < nghost_2; cpid++) {
      NState state = (NState)PetscRealPart(cpcol_2_state[cpid]);

      if (state == DELETED) {
        PetscInt sgid_new = (PetscInt)PetscRealPart(cpcol_2_parent[cpid]);
        PetscInt sgid_old = (PetscInt)PetscRealPart(cpcol_2_par_orig[cpid]);

        if (sgid_old == -1 && sgid_new != -1) {
          PetscInt gid = (PetscInt)PetscRealPart(cpcol_2_gid[cpid]);

          PetscCall(PetscHMapISet(gid_cpid, gid, cpid));
        }
      }
    }

    /* look for deleted ghosts and see if they moved - remove it */
    for (lid = 0; lid < nloc; lid++) {
      NState state = lid_state[lid];

      if (IS_SELECTED(state)) {
        PetscCDIntNd *pos, *last = NULL;

        /* look for deleted ghosts and see if they moved */
        PetscCall(PetscCDGetHeadPos(aggs_2, lid, &pos));
        while (pos) {
          PetscInt gid;

          PetscCall(PetscCDIntNdGetID(pos, &gid));
          if (gid < my0 || gid >= Iend) {
            PetscCall(PetscHMapIGet(gid_cpid, gid, &cpid));
            if (cpid != -1) {
              /* a moved ghost - */
              /* id_llist_2[lastid] = id_llist_2[flid];    /\* remove 'flid' from list *\/ */
              PetscCall(PetscCDRemoveNextNode(aggs_2, lid, last));
            } else last = pos;
          } else last = pos;

          PetscCall(PetscCDGetNextPos(aggs_2, lid, &pos));
        } /* loop over list of deleted */
      } /* selected */
    }
    PetscCall(PetscHMapIDestroy(&gid_cpid));

    /* look at ghosts, see if they changed - and it */
    for (cpid = 0; cpid < nghost_2; cpid++) {
      PetscInt sgid_new = (PetscInt)PetscRealPart(cpcol_2_parent[cpid]);

      if (sgid_new >= my0 && sgid_new < Iend) { /* this is mine */
        PetscInt      gid      = (PetscInt)PetscRealPart(cpcol_2_gid[cpid]);
        PetscInt      slid_new = sgid_new - my0, hav = 0;
        PetscCDIntNd *pos;

        /* search for this gid to see if I have it */
        PetscCall(PetscCDGetHeadPos(aggs_2, slid_new, &pos));
        while (pos) {
          PetscInt gidj;

          PetscCall(PetscCDIntNdGetID(pos, &gidj));
          PetscCall(PetscCDGetNextPos(aggs_2, slid_new, &pos));

          if (gidj == gid) {
            hav = 1;
            break;
          }
        }
        if (hav != 1) {
          /* insert 'flidj' into head of llist */
          PetscCall(PetscCDAppendID(aggs_2, slid_new, gid));
        }
      }
    }
    PetscCall(VecRestoreArray(mpimat_1->lvec, &cpcol_1_state));
    PetscCall(VecRestoreArray(mpimat_2->lvec, &cpcol_2_state));
    PetscCall(VecRestoreArray(ghostparents2, &cpcol_2_parent));
    PetscCall(VecRestoreArray(ghostgids2, &cpcol_2_gid));
    PetscCall(VecDestroy(&ghostgids2));
    PetscCall(VecDestroy(&ghostparents2));
    PetscCall(VecDestroy(&ghost_par_orig2));
  }
  PetscCall(PetscFree3(lid_state, lid_parent_gid, lid_cprowID_1));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
   PCGAMGCoarsen_AGG - supports squaring the graph (deprecated) and new graph for
     communication of QR data used with HEM and MISk coarsening

  Input Parameter:
   . a_pc - this

  Input/Output Parameter:
   . a_Gmat1 - graph to coarsen (in), graph off processor edges for QR gather scatter (out)

  Output Parameter:
   . agg_lists - list of aggregates

*/
static PetscErrorCode PCGAMGCoarsen_AGG(PC a_pc, Mat *a_Gmat1, PetscCoarsenData **agg_lists)
{
  PC_MG       *mg          = (PC_MG *)a_pc->data;
  PC_GAMG     *pc_gamg     = (PC_GAMG *)mg->innerctx;
  PC_GAMG_AGG *pc_gamg_agg = (PC_GAMG_AGG *)pc_gamg->subctx;
  Mat          Gmat2, Gmat1 = *a_Gmat1; /* aggressive graph */
  IS           perm;
  PetscInt     Istart, Iend, Ii, nloc, bs, nn;
  PetscInt    *permute, *degree;
  PetscBool   *bIndexSet;
  PetscReal    hashfact;
  PetscInt     iSwapIndex;
  PetscRandom  random;
  MPI_Comm     comm;

  PetscFunctionBegin;
  PetscCall(PetscObjectGetComm((PetscObject)Gmat1, &comm));
  PetscCall(PetscLogEventBegin(petsc_gamg_setup_events[GAMG_COARSEN], 0, 0, 0, 0));
  PetscCall(MatGetLocalSize(Gmat1, &nn, NULL));
  PetscCall(MatGetBlockSize(Gmat1, &bs));
  PetscCheck(bs == 1, PETSC_COMM_SELF, PETSC_ERR_PLIB, "bs %" PetscInt_FMT " must be 1", bs);
  nloc = nn / bs;
  /* get MIS aggs - randomize */
  PetscCall(PetscMalloc2(nloc, &permute, nloc, &degree));
  PetscCall(PetscCalloc1(nloc, &bIndexSet));
  for (Ii = 0; Ii < nloc; Ii++) permute[Ii] = Ii;
  PetscCall(PetscRandomCreate(PETSC_COMM_SELF, &random));
  PetscCall(MatGetOwnershipRange(Gmat1, &Istart, &Iend));
  for (Ii = 0; Ii < nloc; Ii++) {
    PetscInt nc;

    PetscCall(MatGetRow(Gmat1, Istart + Ii, &nc, NULL, NULL));
    degree[Ii] = nc;
    PetscCall(MatRestoreRow(Gmat1, Istart + Ii, &nc, NULL, NULL));
  }
  for (Ii = 0; Ii < nloc; Ii++) {
    PetscCall(PetscRandomGetValueReal(random, &hashfact));
    iSwapIndex = (PetscInt)(hashfact * nloc) % nloc;
    if (!bIndexSet[iSwapIndex] && iSwapIndex != Ii) {
      PetscInt iTemp = permute[iSwapIndex];

      permute[iSwapIndex]   = permute[Ii];
      permute[Ii]           = iTemp;
      iTemp                 = degree[iSwapIndex];
      degree[iSwapIndex]    = degree[Ii];
      degree[Ii]            = iTemp;
      bIndexSet[iSwapIndex] = PETSC_TRUE;
    }
  }
  // apply minimum degree ordering -- NEW
  if (pc_gamg_agg->use_minimum_degree_ordering) PetscCall(PetscSortIntWithArray(nloc, degree, permute));
  PetscCall(PetscFree(bIndexSet));
  PetscCall(PetscRandomDestroy(&random));
  PetscCall(ISCreateGeneral(PETSC_COMM_SELF, nloc, permute, PETSC_USE_POINTER, &perm));
  PetscCall(PetscLogEventBegin(petsc_gamg_setup_events[GAMG_MIS], 0, 0, 0, 0));
  // square graph
  if (pc_gamg->current_level < pc_gamg_agg->aggressive_coarsening_levels && pc_gamg_agg->use_aggressive_square_graph) PetscCall(PCGAMGSquareGraph_GAMG(a_pc, Gmat1, &Gmat2));
  else Gmat2 = Gmat1;
  // switch to old MIS-1 for square graph
  if (pc_gamg->current_level < pc_gamg_agg->aggressive_coarsening_levels) {
    if (!pc_gamg_agg->use_aggressive_square_graph) PetscCall(MatCoarsenMISKSetDistance(pc_gamg_agg->crs, pc_gamg_agg->aggressive_mis_k)); // hardwire to MIS-2
    else PetscCall(MatCoarsenSetType(pc_gamg_agg->crs, MATCOARSENMIS));                                                                   // old MIS -- side effect
  } else if (pc_gamg_agg->use_aggressive_square_graph && pc_gamg_agg->aggressive_coarsening_levels > 0) {                                 // we reset the MIS
    const char *prefix;

    PetscCall(PetscObjectGetOptionsPrefix((PetscObject)a_pc, &prefix));
    PetscCall(PetscObjectSetOptionsPrefix((PetscObject)pc_gamg_agg->crs, prefix));
    PetscCall(MatCoarsenSetFromOptions(pc_gamg_agg->crs)); // get the default back on non-aggressive levels when square graph switched to old MIS
  }
  PetscCall(MatCoarsenSetAdjacency(pc_gamg_agg->crs, Gmat2));
  PetscCall(MatCoarsenSetStrictAggs(pc_gamg_agg->crs, PETSC_TRUE));
  PetscCall(MatCoarsenSetGreedyOrdering(pc_gamg_agg->crs, perm));
  PetscCall(MatCoarsenApply(pc_gamg_agg->crs));
  PetscCall(MatCoarsenGetData(pc_gamg_agg->crs, agg_lists)); /* output */

  PetscCall(ISDestroy(&perm));
  PetscCall(PetscFree2(permute, degree));
  PetscCall(PetscLogEventEnd(petsc_gamg_setup_events[GAMG_MIS], 0, 0, 0, 0));

  if (Gmat2 != Gmat1) { // square graph, we need ghosts for selected
    PetscCoarsenData *llist = *agg_lists;

    PetscCall(fixAggregatesWithSquare(a_pc, Gmat2, Gmat1, *agg_lists));
    PetscCall(MatDestroy(&Gmat1));
    *a_Gmat1 = Gmat2;                          /* output */
    PetscCall(PetscCDSetMat(llist, *a_Gmat1)); /* Need a graph with ghosts here */
  }
  PetscCall(PetscLogEventEnd(petsc_gamg_setup_events[GAMG_COARSEN], 0, 0, 0, 0));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
 PCGAMGConstructProlongator_AGG

 Input Parameter:
 . pc - this
 . Amat - matrix on this fine level
 . Graph - used to get ghost data for nodes in
 . agg_lists - list of aggregates
 Output Parameter:
 . a_P_out - prolongation operator to the next level
 */
static PetscErrorCode PCGAMGConstructProlongator_AGG(PC pc, Mat Amat, PetscCoarsenData *agg_lists, Mat *a_P_out)
{
  PC_MG         *mg      = (PC_MG *)pc->data;
  PC_GAMG       *pc_gamg = (PC_GAMG *)mg->innerctx;
  const PetscInt col_bs  = pc_gamg->data_cell_cols;
  PetscInt       Istart, Iend, nloc, ii, jj, kk, my0, nLocalSelected, bs;
  Mat            Gmat, Prol;
  PetscMPIInt    size;
  MPI_Comm       comm;
  PetscReal     *data_w_ghost;
  PetscInt       myCrs0, nbnodes = 0, *flid_fgid;
  MatType        mtype;

  PetscFunctionBegin;
  PetscCall(PetscObjectGetComm((PetscObject)Amat, &comm));
  PetscCheck(col_bs >= 1, comm, PETSC_ERR_PLIB, "Column bs cannot be less than 1");
  PetscCall(PetscLogEventBegin(petsc_gamg_setup_events[GAMG_PROL], 0, 0, 0, 0));
  PetscCallMPI(MPI_Comm_size(comm, &size));
  PetscCall(MatGetOwnershipRange(Amat, &Istart, &Iend));
  PetscCall(MatGetBlockSize(Amat, &bs));
  nloc = (Iend - Istart) / bs;
  my0  = Istart / bs;
  PetscCheck((Iend - Istart) % bs == 0, PETSC_COMM_SELF, PETSC_ERR_PLIB, "(Iend %" PetscInt_FMT " - Istart %" PetscInt_FMT ") not divisible by bs %" PetscInt_FMT, Iend, Istart, bs);
  PetscCall(PetscCDGetMat(agg_lists, &Gmat)); // get auxiliary matrix for ghost edges for size > 1

  /* get 'nLocalSelected' */
  for (ii = 0, nLocalSelected = 0; ii < nloc; ii++) {
    PetscBool ise;

    /* filter out singletons 0 or 1? */
    PetscCall(PetscCDIsEmptyAt(agg_lists, ii, &ise));
    if (!ise) nLocalSelected++;
  }

  /* create prolongator, create P matrix */
  PetscCall(MatGetType(Amat, &mtype));
  PetscCall(MatCreate(comm, &Prol));
  PetscCall(MatSetSizes(Prol, nloc * bs, nLocalSelected * col_bs, PETSC_DETERMINE, PETSC_DETERMINE));
  PetscCall(MatSetBlockSizes(Prol, bs, col_bs)); // should this be before MatSetSizes?
  PetscCall(MatSetType(Prol, mtype));
#if PetscDefined(HAVE_DEVICE)
  PetscBool flg;
  PetscCall(MatBoundToCPU(Amat, &flg));
  PetscCall(MatBindToCPU(Prol, flg));
  if (flg) PetscCall(MatSetBindingPropagates(Prol, PETSC_TRUE));
#endif
  PetscCall(MatSeqAIJSetPreallocation(Prol, col_bs, NULL));
  PetscCall(MatMPIAIJSetPreallocation(Prol, col_bs, NULL, col_bs, NULL));

  /* can get all points "removed" */
  PetscCall(MatGetSize(Prol, &kk, &ii));
  if (!ii) {
    PetscCall(PetscInfo(pc, "%s: No selected points on coarse grid\n", ((PetscObject)pc)->prefix));
    PetscCall(MatDestroy(&Prol));
    *a_P_out = NULL; /* out */
    PetscCall(PetscLogEventEnd(petsc_gamg_setup_events[GAMG_PROL], 0, 0, 0, 0));
    PetscFunctionReturn(PETSC_SUCCESS);
  }
  PetscCall(PetscInfo(pc, "%s: New grid %" PetscInt_FMT " nodes\n", ((PetscObject)pc)->prefix, ii / col_bs));
  PetscCall(MatGetOwnershipRangeColumn(Prol, &myCrs0, &kk));

  PetscCheck((kk - myCrs0) % col_bs == 0, PETSC_COMM_SELF, PETSC_ERR_PLIB, "(kk %" PetscInt_FMT " -myCrs0 %" PetscInt_FMT ") not divisible by col_bs %" PetscInt_FMT, kk, myCrs0, col_bs);
  myCrs0 = myCrs0 / col_bs;
  PetscCheck((kk / col_bs - myCrs0) == nLocalSelected, PETSC_COMM_SELF, PETSC_ERR_PLIB, "(kk %" PetscInt_FMT "/col_bs %" PetscInt_FMT " - myCrs0 %" PetscInt_FMT ") != nLocalSelected %" PetscInt_FMT ")", kk, col_bs, myCrs0, nLocalSelected);

  /* create global vector of data in 'data_w_ghost' */
  PetscCall(PetscLogEventBegin(petsc_gamg_setup_events[GAMG_PROLA], 0, 0, 0, 0));
  if (size > 1) { /* get ghost null space data */
    PetscReal *tmp_gdata, *tmp_ldata, *tp2;

    PetscCall(PetscMalloc1(nloc, &tmp_ldata));
    for (jj = 0; jj < col_bs; jj++) {
      for (kk = 0; kk < bs; kk++) {
        PetscInt         ii, stride;
        const PetscReal *tp = PetscSafePointerPlusOffset(pc_gamg->data, jj * bs * nloc + kk);

        for (ii = 0; ii < nloc; ii++, tp += bs) tmp_ldata[ii] = *tp;

        PetscCall(PCGAMGGetDataWithGhosts(Gmat, 1, tmp_ldata, &stride, &tmp_gdata));

        if (!jj && !kk) { /* now I know how many total nodes - allocate TODO: move below and do in one 'col_bs' call */
          PetscCall(PetscMalloc1(stride * bs * col_bs, &data_w_ghost));
          nbnodes = bs * stride;
        }
        tp2 = PetscSafePointerPlusOffset(data_w_ghost, jj * bs * stride + kk);
        for (ii = 0; ii < stride; ii++, tp2 += bs) *tp2 = tmp_gdata[ii];
        PetscCall(PetscFree(tmp_gdata));
      }
    }
    PetscCall(PetscFree(tmp_ldata));
  } else {
    nbnodes      = bs * nloc;
    data_w_ghost = pc_gamg->data;
  }

  /* get 'flid_fgid' TODO - move up to get 'stride' and do get null space data above in one step (jj loop) */
  if (size > 1) {
    PetscReal *fid_glid_loc, *fiddata;
    PetscInt   stride;

    PetscCall(PetscMalloc1(nloc, &fid_glid_loc));
    for (kk = 0; kk < nloc; kk++) fid_glid_loc[kk] = (PetscReal)(my0 + kk);
    PetscCall(PCGAMGGetDataWithGhosts(Gmat, 1, fid_glid_loc, &stride, &fiddata));
    PetscCall(PetscMalloc1(stride, &flid_fgid)); /* copy real data to in */
    for (kk = 0; kk < stride; kk++) flid_fgid[kk] = (PetscInt)fiddata[kk];
    PetscCall(PetscFree(fiddata));

    PetscCheck(stride == nbnodes / bs, PETSC_COMM_SELF, PETSC_ERR_PLIB, "stride %" PetscInt_FMT " != nbnodes %" PetscInt_FMT "/bs %" PetscInt_FMT, stride, nbnodes, bs);
    PetscCall(PetscFree(fid_glid_loc));
  } else {
    PetscCall(PetscMalloc1(nloc, &flid_fgid));
    for (kk = 0; kk < nloc; kk++) flid_fgid[kk] = my0 + kk;
  }
  PetscCall(PetscLogEventEnd(petsc_gamg_setup_events[GAMG_PROLA], 0, 0, 0, 0));
  /* get P0 */
  PetscCall(PetscLogEventBegin(petsc_gamg_setup_events[GAMG_PROLB], 0, 0, 0, 0));
  {
    PetscReal *data_out = NULL;

    PetscCall(formProl0(agg_lists, bs, col_bs, myCrs0, nbnodes, data_w_ghost, flid_fgid, &data_out, Prol));
    PetscCall(PetscFree(pc_gamg->data));

    pc_gamg->data           = data_out;
    pc_gamg->data_cell_rows = col_bs;
    pc_gamg->data_sz        = col_bs * col_bs * nLocalSelected;
  }
  PetscCall(PetscLogEventEnd(petsc_gamg_setup_events[GAMG_PROLB], 0, 0, 0, 0));
  if (size > 1) PetscCall(PetscFree(data_w_ghost));
  PetscCall(PetscFree(flid_fgid));

  *a_P_out = Prol; /* out */
  PetscCall(MatViewFromOptions(Prol, NULL, "-pc_gamg_agg_view_initial_prolongation"));

  PetscCall(PetscLogEventEnd(petsc_gamg_setup_events[GAMG_PROL], 0, 0, 0, 0));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
   PCGAMGOptimizeProlongator_AGG - given the initial prolongator optimizes it by smoothed aggregation pc_gamg_agg->nsmooths times

  Input Parameter:
   . pc - this
   . Amat - matrix on this fine level
 In/Output Parameter:
   . a_P - prolongation operator to the next level
*/
static PetscErrorCode PCGAMGOptimizeProlongator_AGG(PC pc, Mat Amat, Mat *a_P)
{
  PC_MG       *mg          = (PC_MG *)pc->data;
  PC_GAMG     *pc_gamg     = (PC_GAMG *)mg->innerctx;
  PC_GAMG_AGG *pc_gamg_agg = (PC_GAMG_AGG *)pc_gamg->subctx;
  PetscInt     jj;
  Mat          Prol = *a_P;
  MPI_Comm     comm;
  KSP          eksp;
  Vec          bb, xx;
  PC           epc;
  PetscReal    alpha, emax, emin;

  PetscFunctionBegin;
  PetscCall(PetscObjectGetComm((PetscObject)Amat, &comm));
  PetscCall(PetscLogEventBegin(petsc_gamg_setup_events[GAMG_OPT], 0, 0, 0, 0));

  /* compute maximum singular value of operator to be used in smoother */
  if (0 < pc_gamg_agg->nsmooths) {
    /* get eigen estimates */
    if (pc_gamg->emax > 0) {
      emin = pc_gamg->emin;
      emax = pc_gamg->emax;
    } else {
      const char *prefix;

      PetscCall(MatCreateVecs(Amat, &bb, NULL));
      PetscCall(MatCreateVecs(Amat, &xx, NULL));
      PetscCall(KSPSetNoisy_Private(Amat, bb));

      PetscCall(KSPCreate(comm, &eksp));
      PetscCall(KSPSetNestLevel(eksp, pc->kspnestlevel));
      PetscCall(PCGetOptionsPrefix(pc, &prefix));
      PetscCall(KSPSetOptionsPrefix(eksp, prefix));
      PetscCall(KSPAppendOptionsPrefix(eksp, "pc_gamg_esteig_"));
      {
        PetscBool isset, sflg;

        PetscCall(MatIsSPDKnown(Amat, &isset, &sflg));
        if (isset && sflg) PetscCall(KSPSetType(eksp, KSPCG));
      }
      PetscCall(KSPSetErrorIfNotConverged(eksp, pc->erroriffailure));
      PetscCall(KSPSetNormType(eksp, KSP_NORM_NONE));

      PetscCall(KSPSetInitialGuessNonzero(eksp, PETSC_FALSE));
      PetscCall(KSPSetOperators(eksp, Amat, Amat));

      PetscCall(KSPGetPC(eksp, &epc));
      PetscCall(PCSetType(epc, PCJACOBI)); /* smoother in smoothed agg. */

      PetscCall(KSPSetTolerances(eksp, PETSC_CURRENT, PETSC_CURRENT, PETSC_CURRENT, 10)); // 10 is safer, but 5 is often fine, can override with -pc_gamg_esteig_ksp_max_it -mg_levels_ksp_chebyshev_esteig 0,0.25,0,1.2

      PetscCall(KSPSetFromOptions(eksp));
      PetscCall(KSPSetComputeSingularValues(eksp, PETSC_TRUE));
      PetscCall(KSPSolve(eksp, bb, xx));
      PetscCall(KSPCheckSolve(eksp, pc, xx));

      PetscCall(KSPComputeExtremeSingularValues(eksp, &emax, &emin));
      PetscCall(PetscInfo(pc, "%s: Smooth P0: max eigen=%e min=%e PC=%s\n", ((PetscObject)pc)->prefix, (double)emax, (double)emin, PCJACOBI));
      PetscCall(VecDestroy(&xx));
      PetscCall(VecDestroy(&bb));
      PetscCall(KSPDestroy(&eksp));
    }
    if (pc_gamg->use_sa_esteig) {
      mg->min_eigen_DinvA[pc_gamg->current_level] = emin;
      mg->max_eigen_DinvA[pc_gamg->current_level] = emax;
      PetscCall(PetscInfo(pc, "%s: Smooth P0: level %" PetscInt_FMT ", cache spectra %g %g\n", ((PetscObject)pc)->prefix, pc_gamg->current_level, (double)emin, (double)emax));
    } else {
      mg->min_eigen_DinvA[pc_gamg->current_level] = 0;
      mg->max_eigen_DinvA[pc_gamg->current_level] = 0;
    }
  } else {
    mg->min_eigen_DinvA[pc_gamg->current_level] = 0;
    mg->max_eigen_DinvA[pc_gamg->current_level] = 0;
  }

  /* smooth P0 */
  if (pc_gamg_agg->nsmooths > 0) {
    Vec diag;

    /* TODO: Set a PCFailedReason and exit the building of the AMG preconditioner */
    PetscCheck(emax != 0.0, PetscObjectComm((PetscObject)pc), PETSC_ERR_PLIB, "Computed maximum singular value as zero");

    PetscCall(MatCreateVecs(Amat, &diag, NULL));
    PetscCall(MatGetDiagonal(Amat, diag)); /* effectively PCJACOBI */
    PetscCall(VecReciprocal(diag));

    for (jj = 0; jj < pc_gamg_agg->nsmooths; jj++) {
      Mat tMat;

      PetscCall(PetscLogEventBegin(petsc_gamg_setup_events[GAMG_OPTSM], 0, 0, 0, 0));
      /*
        Smooth aggregation on the prolongator

        P_{i} := (I - 1.4/emax D^{-1}A) P_i\{i-1}
      */
      PetscCall(PetscLogEventBegin(petsc_gamg_setup_matmat_events[pc_gamg->current_level][2], 0, 0, 0, 0));
      PetscCall(MatMatMult(Amat, Prol, MAT_INITIAL_MATRIX, PETSC_CURRENT, &tMat));
      PetscCall(PetscLogEventEnd(petsc_gamg_setup_matmat_events[pc_gamg->current_level][2], 0, 0, 0, 0));
      PetscCall(MatProductClear(tMat));
      PetscCall(MatDiagonalScale(tMat, diag, NULL));

      /* TODO: Document the 1.4 and don't hardwire it in this routine */
      alpha = -1.4 / emax;
      PetscCall(MatAYPX(tMat, alpha, Prol, SUBSET_NONZERO_PATTERN));
      PetscCall(MatDestroy(&Prol));
      Prol = tMat;
      PetscCall(PetscLogEventEnd(petsc_gamg_setup_events[GAMG_OPTSM], 0, 0, 0, 0));
    }
    PetscCall(VecDestroy(&diag));
  }
  PetscCall(PetscLogEventEnd(petsc_gamg_setup_events[GAMG_OPT], 0, 0, 0, 0));
  PetscCall(MatViewFromOptions(Prol, NULL, "-pc_gamg_agg_view_prolongation"));
  *a_P = Prol;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*MC
  PCGAMGAGG - Smooth aggregation, {cite}`vanek1996algebraic`, {cite}`vanek2001convergence`, variant of PETSc's algebraic multigrid (`PCGAMG`) preconditioner

  Options Database Keys:
+ -pc_gamg_agg_nsmooths <nsmooth, default=1> - number of smoothing steps to use with smooth aggregation to construct prolongation
. -pc_gamg_aggressive_coarsening <n,default=1> - number of aggressive coarsening (MIS-2) levels from finest.
. -pc_gamg_aggressive_square_graph <bool,default=true> - Use square graph (A'A), alternative is MIS-k (k=2), for aggressive coarsening
. -pc_gamg_mis_k_minimum_degree_ordering <bool,default=false> - Use minimum degree ordering in greedy MIS algorithm
. -pc_gamg_pc_gamg_asm_hem_aggs <n,default=0> - Number of HEM aggregation steps for ASM smoother
- -pc_gamg_aggressive_mis_k <n,default=2> - Number (k) distance in MIS coarsening (>2 is 'aggressive')

  Level: intermediate

  Notes:
  To obtain good performance for `PCGAMG` for vector valued problems you must
  call `MatSetBlockSize()` to indicate the number of degrees of freedom per grid point.
  Call `MatSetNearNullSpace()` (or `PCSetCoordinates()` if solving the equations of elasticity) to indicate the near null space of the operator

  The many options for `PCMG` and `PCGAMG` such as controlling the smoothers on each level etc. also work for `PCGAMGAGG`

.seealso: `PCGAMG`, [the Users Manual section on PCGAMG](sec_amg), [the Users Manual section on PCMG](sec_mg), [](ch_ksp), `PCCreate()`, `PCSetType()`,
          `MatSetBlockSize()`, `PCMGType`, `PCSetCoordinates()`, `MatSetNearNullSpace()`, `PCGAMGSetType()`,
          `PCGAMGAGG`, `PCGAMGGEO`, `PCGAMGCLASSICAL`, `PCGAMGSetProcEqLim()`, `PCGAMGSetCoarseEqLim()`, `PCGAMGSetRepartition()`, `PCGAMGRegister()`,
          `PCGAMGSetReuseInterpolation()`, `PCGAMGASMSetUseAggs()`, `PCGAMGSetParallelCoarseGridSolve()`, `PCGAMGSetNlevels()`, `PCGAMGSetThreshold()`,
          `PCGAMGGetType()`, `PCGAMGSetUseSAEstEig()`
M*/
PetscErrorCode PCCreateGAMG_AGG(PC pc)
{
  PC_MG       *mg      = (PC_MG *)pc->data;
  PC_GAMG     *pc_gamg = (PC_GAMG *)mg->innerctx;
  PC_GAMG_AGG *pc_gamg_agg;

  PetscFunctionBegin;
  /* create sub context for SA */
  PetscCall(PetscNew(&pc_gamg_agg));
  pc_gamg->subctx = pc_gamg_agg;

  pc_gamg->ops->setfromoptions = PCSetFromOptions_GAMG_AGG;
  pc_gamg->ops->destroy        = PCDestroy_GAMG_AGG;
  /* reset does not do anything; setup not virtual */

  /* set internal function pointers */
  pc_gamg->ops->creategraph       = PCGAMGCreateGraph_AGG;
  pc_gamg->ops->coarsen           = PCGAMGCoarsen_AGG;
  pc_gamg->ops->prolongator       = PCGAMGConstructProlongator_AGG;
  pc_gamg->ops->optprolongator    = PCGAMGOptimizeProlongator_AGG;
  pc_gamg->ops->createdefaultdata = PCSetData_AGG;
  pc_gamg->ops->view              = PCView_GAMG_AGG;

  pc_gamg_agg->nsmooths                     = 1;
  pc_gamg_agg->aggressive_coarsening_levels = 1;
  pc_gamg_agg->use_aggressive_square_graph  = PETSC_TRUE;
  pc_gamg_agg->use_minimum_degree_ordering  = PETSC_FALSE;
  pc_gamg_agg->use_low_mem_filter           = PETSC_FALSE;
  pc_gamg_agg->aggressive_mis_k             = 2;
  pc_gamg_agg->graph_symmetrize             = PETSC_TRUE;

  PetscCall(PetscObjectComposeFunction((PetscObject)pc, "PCGAMGSetNSmooths_C", PCGAMGSetNSmooths_AGG));
  PetscCall(PetscObjectComposeFunction((PetscObject)pc, "PCGAMGSetAggressiveLevels_C", PCGAMGSetAggressiveLevels_AGG));
  PetscCall(PetscObjectComposeFunction((PetscObject)pc, "PCGAMGSetAggressiveSquareGraph_C", PCGAMGSetAggressiveSquareGraph_AGG));
  PetscCall(PetscObjectComposeFunction((PetscObject)pc, "PCGAMGMISkSetMinDegreeOrdering_C", PCGAMGMISkSetMinDegreeOrdering_AGG));
  PetscCall(PetscObjectComposeFunction((PetscObject)pc, "PCGAMGSetLowMemoryFilter_C", PCGAMGSetLowMemoryFilter_AGG));
  PetscCall(PetscObjectComposeFunction((PetscObject)pc, "PCGAMGMISkSetAggressive_C", PCGAMGMISkSetAggressive_AGG));
  PetscCall(PetscObjectComposeFunction((PetscObject)pc, "PCGAMGSetGraphSymmetrize_C", PCGAMGSetGraphSymmetrize_AGG));
  PetscCall(PetscObjectComposeFunction((PetscObject)pc, "PCSetCoordinates_C", PCSetCoordinates_AGG));
  PetscFunctionReturn(PETSC_SUCCESS);
}
