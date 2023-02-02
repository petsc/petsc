#include <petscdm.h>
#include <petsc/private/hashmapi.h>
#include <petsc/private/matimpl.h>
#include <petsc/private/pcmgimpl.h>
#include <petsc/private/pcimpl.h> /*I "petscpc.h" I*/

typedef struct {
  PC        innerpc;       /* A MG inner PC (Hypre or PCGAMG) to setup interpolations and coarse operators  */
  char     *innerpctype;   /* PCGAMG or PCHYPRE */
  PetscBool reuseinterp;   /* A flag indicates if or not to reuse the interpolations */
  PetscBool subcoarsening; /* If or not to use a subspace-based coarsening algorithm */
  PetscBool usematmaij;    /* If or not to use MatMAIJ for saving memory */
  PetscInt  component;     /* Which subspace is used for the subspace-based coarsening algorithm? */
} PC_HMG;

PetscErrorCode PCSetFromOptions_HMG(PC, PetscOptionItems *);
PetscErrorCode PCReset_MG(PC);

static PetscErrorCode PCHMGExtractSubMatrix_Private(Mat pmat, Mat *submat, MatReuse reuse, PetscInt component, PetscInt blocksize)
{
  IS       isrow;
  PetscInt rstart, rend;
  MPI_Comm comm;

  PetscFunctionBegin;
  PetscCall(PetscObjectGetComm((PetscObject)pmat, &comm));
  PetscCheck(component < blocksize, comm, PETSC_ERR_ARG_INCOMP, "Component %" PetscInt_FMT " should be less than block size %" PetscInt_FMT " ", component, blocksize);
  PetscCall(MatGetOwnershipRange(pmat, &rstart, &rend));
  PetscCheck((rend - rstart) % blocksize == 0, comm, PETSC_ERR_ARG_INCOMP, "Block size %" PetscInt_FMT " is inconsistent for [%" PetscInt_FMT ", %" PetscInt_FMT ") ", blocksize, rstart, rend);
  PetscCall(ISCreateStride(comm, (rend - rstart) / blocksize, rstart + component, blocksize, &isrow));
  PetscCall(MatCreateSubMatrix(pmat, isrow, isrow, reuse, submat));
  PetscCall(ISDestroy(&isrow));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PCHMGExpandInterpolation_Private(Mat subinterp, Mat *interp, PetscInt blocksize)
{
  PetscInt           subrstart, subrend, subrowsize, subcolsize, subcstart, subcend, rowsize, colsize;
  PetscInt           subrow, row, nz, *d_nnz, *o_nnz, i, j, dnz, onz, max_nz, *indices;
  const PetscInt    *idx;
  const PetscScalar *values;
  MPI_Comm           comm;

  PetscFunctionBegin;
  PetscCall(PetscObjectGetComm((PetscObject)subinterp, &comm));
  PetscCall(MatGetOwnershipRange(subinterp, &subrstart, &subrend));
  subrowsize = subrend - subrstart;
  rowsize    = subrowsize * blocksize;
  PetscCall(PetscCalloc2(rowsize, &d_nnz, rowsize, &o_nnz));
  PetscCall(MatGetOwnershipRangeColumn(subinterp, &subcstart, &subcend));
  subcolsize = subcend - subcstart;
  colsize    = subcolsize * blocksize;
  max_nz     = 0;
  for (subrow = subrstart; subrow < subrend; subrow++) {
    PetscCall(MatGetRow(subinterp, subrow, &nz, &idx, NULL));
    if (max_nz < nz) max_nz = nz;
    dnz = 0;
    onz = 0;
    for (i = 0; i < nz; i++) {
      if (idx[i] >= subcstart && idx[i] < subcend) dnz++;
      else onz++;
    }
    for (i = 0; i < blocksize; i++) {
      d_nnz[(subrow - subrstart) * blocksize + i] = dnz;
      o_nnz[(subrow - subrstart) * blocksize + i] = onz;
    }
    PetscCall(MatRestoreRow(subinterp, subrow, &nz, &idx, NULL));
  }
  PetscCall(MatCreateAIJ(comm, rowsize, colsize, PETSC_DETERMINE, PETSC_DETERMINE, 0, d_nnz, 0, o_nnz, interp));
  PetscCall(MatSetOption(*interp, MAT_IGNORE_OFF_PROC_ENTRIES, PETSC_TRUE));
  PetscCall(MatSetOption(*interp, MAT_IGNORE_ZERO_ENTRIES, PETSC_TRUE));
  PetscCall(MatSetOption(*interp, MAT_NEW_NONZERO_ALLOCATION_ERR, PETSC_TRUE));
  PetscCall(MatSetFromOptions(*interp));

  PetscCall(MatSetUp(*interp));
  PetscCall(PetscFree2(d_nnz, o_nnz));
  PetscCall(PetscMalloc1(max_nz, &indices));
  for (subrow = subrstart; subrow < subrend; subrow++) {
    PetscCall(MatGetRow(subinterp, subrow, &nz, &idx, &values));
    for (i = 0; i < blocksize; i++) {
      row = subrow * blocksize + i;
      for (j = 0; j < nz; j++) indices[j] = idx[j] * blocksize + i;
      PetscCall(MatSetValues(*interp, 1, &row, nz, indices, values, INSERT_VALUES));
    }
    PetscCall(MatRestoreRow(subinterp, subrow, &nz, &idx, &values));
  }
  PetscCall(PetscFree(indices));
  PetscCall(MatAssemblyBegin(*interp, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(*interp, MAT_FINAL_ASSEMBLY));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode PCSetUp_HMG(PC pc)
{
  Mat              PA, submat;
  PC_MG           *mg  = (PC_MG *)pc->data;
  PC_HMG          *hmg = (PC_HMG *)mg->innerctx;
  MPI_Comm         comm;
  PetscInt         level;
  PetscInt         num_levels;
  Mat             *operators, *interpolations;
  PetscInt         blocksize;
  const char      *prefix;
  PCMGGalerkinType galerkin;

  PetscFunctionBegin;
  PetscCall(PetscObjectGetComm((PetscObject)pc, &comm));
  if (pc->setupcalled) {
    if (hmg->reuseinterp) {
      /* If we did not use Galerkin in the last call or we have a different sparsity pattern now,
      * we have to build from scratch
      * */
      PetscCall(PCMGGetGalerkin(pc, &galerkin));
      if (galerkin == PC_MG_GALERKIN_NONE || pc->flag != SAME_NONZERO_PATTERN) pc->setupcalled = PETSC_FALSE;
      PetscCall(PCMGSetGalerkin(pc, PC_MG_GALERKIN_PMAT));
      PetscCall(PCSetUp_MG(pc));
      PetscFunctionReturn(PETSC_SUCCESS);
    } else {
      PetscCall(PCReset_MG(pc));
      pc->setupcalled = PETSC_FALSE;
    }
  }

  /* Create an inner PC (GAMG or HYPRE) */
  if (!hmg->innerpc) {
    PetscCall(PCCreate(comm, &hmg->innerpc));
    /* If users do not set an inner pc type, we need to set a default value */
    if (!hmg->innerpctype) {
      /* If hypre is available, use hypre, otherwise, use gamg */
#if PETSC_HAVE_HYPRE
      PetscCall(PetscStrallocpy(PCHYPRE, &(hmg->innerpctype)));
#else
      PetscCall(PetscStrallocpy(PCGAMG, &(hmg->innerpctype)));
#endif
    }
    PetscCall(PCSetType(hmg->innerpc, hmg->innerpctype));
  }
  PetscCall(PCGetOperators(pc, NULL, &PA));
  /* Users need to correctly set a block size of matrix in order to use subspace coarsening */
  PetscCall(MatGetBlockSize(PA, &blocksize));
  if (blocksize <= 1) hmg->subcoarsening = PETSC_FALSE;
  /* Extract a submatrix for constructing subinterpolations */
  if (hmg->subcoarsening) {
    PetscCall(PCHMGExtractSubMatrix_Private(PA, &submat, MAT_INITIAL_MATRIX, hmg->component, blocksize));
    PA = submat;
  }
  PetscCall(PCSetOperators(hmg->innerpc, PA, PA));
  if (hmg->subcoarsening) PetscCall(MatDestroy(&PA));
  /* Setup inner PC correctly. During this step, matrix will be coarsened */
  PetscCall(PCSetUseAmat(hmg->innerpc, PETSC_FALSE));
  PetscCall(PetscObjectGetOptionsPrefix((PetscObject)pc, &prefix));
  PetscCall(PetscObjectSetOptionsPrefix((PetscObject)hmg->innerpc, prefix));
  PetscCall(PetscObjectAppendOptionsPrefix((PetscObject)hmg->innerpc, "hmg_inner_"));
  PetscCall(PCSetFromOptions(hmg->innerpc));
  PetscCall(PCSetUp(hmg->innerpc));

  /* Obtain interpolations IN PLACE. For BoomerAMG, (I,J,data) is reused to avoid memory overhead */
  PetscCall(PCGetInterpolations(hmg->innerpc, &num_levels, &interpolations));
  /* We can reuse the coarse operators when we do the full space coarsening */
  if (!hmg->subcoarsening) PetscCall(PCGetCoarseOperators(hmg->innerpc, &num_levels, &operators));

  PetscCall(PCDestroy(&hmg->innerpc));
  hmg->innerpc = NULL;
  PetscCall(PCMGSetLevels_MG(pc, num_levels, NULL));
  /* Set coarse matrices and interpolations to PCMG */
  for (level = num_levels - 1; level > 0; level--) {
    Mat P = NULL, pmat = NULL;
    Vec b, x, r;
    if (hmg->subcoarsening) {
      if (hmg->usematmaij) {
        PetscCall(MatCreateMAIJ(interpolations[level - 1], blocksize, &P));
        PetscCall(MatDestroy(&interpolations[level - 1]));
      } else {
        /* Grow interpolation. In the future, we should use MAIJ */
        PetscCall(PCHMGExpandInterpolation_Private(interpolations[level - 1], &P, blocksize));
        PetscCall(MatDestroy(&interpolations[level - 1]));
      }
    } else {
      P = interpolations[level - 1];
    }
    PetscCall(MatCreateVecs(P, &b, &r));
    PetscCall(PCMGSetInterpolation(pc, level, P));
    PetscCall(PCMGSetRestriction(pc, level, P));
    PetscCall(MatDestroy(&P));
    /* We reuse the matrices when we do not do subspace coarsening */
    if ((level - 1) >= 0 && !hmg->subcoarsening) {
      pmat = operators[level - 1];
      PetscCall(PCMGSetOperators(pc, level - 1, pmat, pmat));
      PetscCall(MatDestroy(&pmat));
    }
    PetscCall(PCMGSetRhs(pc, level - 1, b));

    PetscCall(PCMGSetR(pc, level, r));
    PetscCall(VecDestroy(&r));

    PetscCall(VecDuplicate(b, &x));
    PetscCall(PCMGSetX(pc, level - 1, x));
    PetscCall(VecDestroy(&x));
    PetscCall(VecDestroy(&b));
  }
  PetscCall(PetscFree(interpolations));
  if (!hmg->subcoarsening) PetscCall(PetscFree(operators));
  /* Turn Galerkin off when we already have coarse operators */
  PetscCall(PCMGSetGalerkin(pc, hmg->subcoarsening ? PC_MG_GALERKIN_PMAT : PC_MG_GALERKIN_NONE));
  PetscCall(PCSetDM(pc, NULL));
  PetscCall(PCSetUseAmat(pc, PETSC_FALSE));
  PetscObjectOptionsBegin((PetscObject)pc);
  PetscCall(PCSetFromOptions_MG(pc, PetscOptionsObject)); /* should be called in PCSetFromOptions_HMG(), but cannot be called prior to PCMGSetLevels() */
  PetscOptionsEnd();
  PetscCall(PCSetUp_MG(pc));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode PCDestroy_HMG(PC pc)
{
  PC_MG  *mg  = (PC_MG *)pc->data;
  PC_HMG *hmg = (PC_HMG *)mg->innerctx;

  PetscFunctionBegin;
  PetscCall(PCDestroy(&hmg->innerpc));
  PetscCall(PetscFree(hmg->innerpctype));
  PetscCall(PetscFree(hmg));
  PetscCall(PCDestroy_MG(pc));

  PetscCall(PetscObjectComposeFunction((PetscObject)pc, "PCHMGSetReuseInterpolation_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)pc, "PCHMGSetUseSubspaceCoarsening_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)pc, "PCHMGSetInnerPCType_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)pc, "PCHMGSetCoarseningComponent_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)pc, "PCHMGUseMatMAIJ_C", NULL));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode PCView_HMG(PC pc, PetscViewer viewer)
{
  PC_MG    *mg  = (PC_MG *)pc->data;
  PC_HMG   *hmg = (PC_HMG *)mg->innerctx;
  PetscBool iascii;

  PetscFunctionBegin;
  PetscCall(PetscObjectTypeCompare((PetscObject)viewer, PETSCVIEWERASCII, &iascii));
  if (iascii) {
    PetscCall(PetscViewerASCIIPrintf(viewer, " Reuse interpolation: %s\n", hmg->reuseinterp ? "true" : "false"));
    PetscCall(PetscViewerASCIIPrintf(viewer, " Use subspace coarsening: %s\n", hmg->subcoarsening ? "true" : "false"));
    PetscCall(PetscViewerASCIIPrintf(viewer, " Coarsening component: %" PetscInt_FMT " \n", hmg->component));
    PetscCall(PetscViewerASCIIPrintf(viewer, " Use MatMAIJ: %s \n", hmg->usematmaij ? "true" : "false"));
    PetscCall(PetscViewerASCIIPrintf(viewer, " Inner PC type: %s \n", hmg->innerpctype));
  }
  PetscCall(PCView_MG(pc, viewer));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode PCSetFromOptions_HMG(PC pc, PetscOptionItems *PetscOptionsObject)
{
  PC_MG  *mg  = (PC_MG *)pc->data;
  PC_HMG *hmg = (PC_HMG *)mg->innerctx;

  PetscFunctionBegin;
  PetscOptionsHeadBegin(PetscOptionsObject, "HMG");
  PetscCall(PetscOptionsBool("-pc_hmg_reuse_interpolation", "Reuse the interpolation operators when possible (cheaper, weaker when matrix entries change a lot)", "PCHMGSetReuseInterpolation", hmg->reuseinterp, &hmg->reuseinterp, NULL));
  PetscCall(PetscOptionsBool("-pc_hmg_use_subspace_coarsening", "Use the subspace coarsening to compute the interpolations", "PCHMGSetUseSubspaceCoarsening", hmg->subcoarsening, &hmg->subcoarsening, NULL));
  PetscCall(PetscOptionsBool("-pc_hmg_use_matmaij", "Use MatMAIJ store interpolation for saving memory", "PCHMGSetInnerPCType", hmg->usematmaij, &hmg->usematmaij, NULL));
  PetscCall(PetscOptionsInt("-pc_hmg_coarsening_component", "Which component is chosen for the subspace-based coarsening algorithm", "PCHMGSetCoarseningComponent", hmg->component, &hmg->component, NULL));
  PetscOptionsHeadEnd();
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PCHMGSetReuseInterpolation_HMG(PC pc, PetscBool reuse)
{
  PC_MG  *mg  = (PC_MG *)pc->data;
  PC_HMG *hmg = (PC_HMG *)mg->innerctx;

  PetscFunctionBegin;
  hmg->reuseinterp = reuse;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
   PCHMGSetReuseInterpolation - Reuse the interpolation matrices in `PCHMG` after changing the matrices numerical values

   Logically Collective

   Input Parameters:
+  pc - the `PCHMG` context
-  reuse - `PETSC_TRUE` indicates that `PCHMG` will reuse the interpolations

   Options Database Key:
.  -pc_hmg_reuse_interpolation <true | false> - Whether or not to reuse the interpolations. If true, it potentially save the compute time.

   Level: beginner

.seealso: `PCHMG`, `PCGAMG`, `PCHMGSetUseSubspaceCoarsening()`, `PCHMGSetCoarseningComponent()`, `PCHMGSetInnerPCType()`
@*/
PetscErrorCode PCHMGSetReuseInterpolation(PC pc, PetscBool reuse)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc, PC_CLASSID, 1);
  PetscUseMethod(pc, "PCHMGSetReuseInterpolation_C", (PC, PetscBool), (pc, reuse));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PCHMGSetUseSubspaceCoarsening_HMG(PC pc, PetscBool subspace)
{
  PC_MG  *mg  = (PC_MG *)pc->data;
  PC_HMG *hmg = (PC_HMG *)mg->innerctx;

  PetscFunctionBegin;
  hmg->subcoarsening = subspace;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
   PCHMGSetUseSubspaceCoarsening - Use subspace coarsening in `PCHMG`

   Logically Collective

   Input Parameters:
+  pc - the `PCHMG` context
-  reuse - `PETSC_TRUE` indicates that `PCHMG` will use the subspace coarsening

   Options Database Key:
.  -pc_hmg_use_subspace_coarsening  <true | false> - Whether or not to use subspace coarsening (that is, coarsen a submatrix).

   Level: beginner

.seealso: `PCHMG`, `PCHMGSetReuseInterpolation()`, `PCHMGSetCoarseningComponent()`, `PCHMGSetInnerPCType()`
@*/
PetscErrorCode PCHMGSetUseSubspaceCoarsening(PC pc, PetscBool subspace)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc, PC_CLASSID, 1);
  PetscUseMethod(pc, "PCHMGSetUseSubspaceCoarsening_C", (PC, PetscBool), (pc, subspace));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PCHMGSetInnerPCType_HMG(PC pc, PCType type)
{
  PC_MG  *mg  = (PC_MG *)pc->data;
  PC_HMG *hmg = (PC_HMG *)mg->innerctx;

  PetscFunctionBegin;
  PetscCall(PetscStrallocpy(type, &(hmg->innerpctype)));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
   PCHMGSetInnerPCType - Set an inner `PC` type

   Logically Collective

   Input Parameters:
+  pc - the `PCHMG` context
-  type - `PCHYPRE` or `PCGAMG` coarsening algorithm

   Options Database Key:
.  -hmg_inner_pc_type <hypre, gamg> - What method is used to coarsen matrix

   Level: beginner

.seealso: `PCHMG`, `PCType`, `PCHMGSetReuseInterpolation()`, `PCHMGSetUseSubspaceCoarsening()`, `PCHMGSetCoarseningComponent()`
@*/
PetscErrorCode PCHMGSetInnerPCType(PC pc, PCType type)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc, PC_CLASSID, 1);
  PetscUseMethod(pc, "PCHMGSetInnerPCType_C", (PC, PCType), (pc, type));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PCHMGSetCoarseningComponent_HMG(PC pc, PetscInt component)
{
  PC_MG  *mg  = (PC_MG *)pc->data;
  PC_HMG *hmg = (PC_HMG *)mg->innerctx;

  PetscFunctionBegin;
  hmg->component = component;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
   PCHMGSetCoarseningComponent - Set which component of the PDE is used for the subspace-based coarsening algorithm

   Logically Collective

   Input Parameters:
+  pc - the `PCHMG` context
-  component - which component `PC` will coarsen

   Options Database Key:
.  -pc_hmg_coarsening_component <i> - Which component is chosen for the subspace-based coarsening algorithm

   Level: beginner

.seealso: `PCHMG`, `PCType`, `PCGAMG`, `PCHMGSetReuseInterpolation()`, `PCHMGSetUseSubspaceCoarsening()`, `PCHMGSetInnerPCType()`
@*/
PetscErrorCode PCHMGSetCoarseningComponent(PC pc, PetscInt component)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc, PC_CLASSID, 1);
  PetscUseMethod(pc, "PCHMGSetCoarseningComponent_C", (PC, PetscInt), (pc, component));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PCHMGUseMatMAIJ_HMG(PC pc, PetscBool usematmaij)
{
  PC_MG  *mg  = (PC_MG *)pc->data;
  PC_HMG *hmg = (PC_HMG *)mg->innerctx;

  PetscFunctionBegin;
  hmg->usematmaij = usematmaij;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
   PCHMGUseMatMAIJ - Set a flag that indicates if or not to use `MATMAIJ` for the interpolation matrices for saving memory

   Logically Collective

   Input Parameters:
+  pc - the `PCHMG` context
-  usematmaij - `PETSC_TRUE` (default) to use `MATMAIJ` for interpolations.

   Options Database Key:
.  -pc_hmg_use_matmaij - <true | false >

   Level: beginner

.seealso: `PCHMG`, `PCType`, `PCGAMG`
@*/
PetscErrorCode PCHMGUseMatMAIJ(PC pc, PetscBool usematmaij)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc, PC_CLASSID, 1);
  PetscUseMethod(pc, "PCHMGUseMatMAIJ_C", (PC, PetscBool), (pc, usematmaij));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*MC
   PCHMG - For multiple component PDE problems constructs a hierarchy of restriction operators to coarse grid problems using the submatrix of
   a single component with either `PCHYPRE` or `PCGAMG`. The same restriction operators are used for each of the components of the PDE with `PCMG`
   resulting in a much more efficient to build and apply preconditioner than using `PCGAMG` on the entire system.

   Options Database Keys:
+  -pc_hmg_reuse_interpolation <true | false> - Whether or not to reuse the interpolations for new matrix values. It can potentially save compute time.
.  -pc_hmg_use_subspace_coarsening  <true | false> - Whether or not to use subspace coarsening (that is, coarsen a submatrix).
.  -hmg_inner_pc_type <hypre, gamg, ...> - What method is used to coarsen matrix
-  -pc_hmg_use_matmaij <true | false> - Whether or not to use `MATMAIJ` for multicomponent problems for saving memory

   Level: intermediate

   Note:
   `MatSetBlockSize()` must be called on the linear system matrix to set the number of components of the PDE.

    References:
.   * - Fande Kong, Yaqi Wang, Derek R Gaston, Cody J Permann, Andrew E Slaughter, Alexander D Lindsay, Richard C Martineau, A highly parallel multilevel
    Newton-Krylov-Schwarz method with subspace-based coarsening and partition-based balancing for the multigroup neutron transport equations on
    3D unstructured meshes, arXiv preprint arXiv:1903.03659, 2019

.seealso: `PCCreate()`, `PCSetType()`, `PCType`, `PC`, `PCMG`, `PCHYPRE`, `PCHMG`, `PCGetCoarseOperators()`, `PCGetInterpolations()`,
          `PCHMGSetReuseInterpolation()`, `PCHMGSetUseSubspaceCoarsening()`, `PCHMGSetInnerPCType()`
M*/
PETSC_EXTERN PetscErrorCode PCCreate_HMG(PC pc)
{
  PC_HMG *hmg;
  PC_MG  *mg;

  PetscFunctionBegin;
  /* if type was previously mg; must manually destroy it because call to PCSetType(pc,PCMG) will not destroy it */
  PetscTryTypeMethod(pc, destroy);
  pc->data = NULL;
  PetscCall(PetscFree(((PetscObject)pc)->type_name));

  PetscCall(PCSetType(pc, PCMG));
  PetscCall(PetscObjectChangeTypeName((PetscObject)pc, PCHMG));
  PetscCall(PetscNew(&hmg));

  mg                 = (PC_MG *)pc->data;
  mg->innerctx       = hmg;
  hmg->reuseinterp   = PETSC_FALSE;
  hmg->subcoarsening = PETSC_FALSE;
  hmg->usematmaij    = PETSC_TRUE;
  hmg->component     = 0;
  hmg->innerpc       = NULL;

  pc->ops->setfromoptions = PCSetFromOptions_HMG;
  pc->ops->view           = PCView_HMG;
  pc->ops->destroy        = PCDestroy_HMG;
  pc->ops->setup          = PCSetUp_HMG;

  PetscCall(PetscObjectComposeFunction((PetscObject)pc, "PCHMGSetReuseInterpolation_C", PCHMGSetReuseInterpolation_HMG));
  PetscCall(PetscObjectComposeFunction((PetscObject)pc, "PCHMGSetUseSubspaceCoarsening_C", PCHMGSetUseSubspaceCoarsening_HMG));
  PetscCall(PetscObjectComposeFunction((PetscObject)pc, "PCHMGSetInnerPCType_C", PCHMGSetInnerPCType_HMG));
  PetscCall(PetscObjectComposeFunction((PetscObject)pc, "PCHMGSetCoarseningComponent_C", PCHMGSetCoarseningComponent_HMG));
  PetscCall(PetscObjectComposeFunction((PetscObject)pc, "PCHMGUseMatMAIJ_C", PCHMGUseMatMAIJ_HMG));
  PetscFunctionReturn(PETSC_SUCCESS);
}
