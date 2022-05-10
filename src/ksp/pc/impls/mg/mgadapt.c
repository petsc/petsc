#include <petsc/private/pcmgimpl.h>       /*I "petscksp.h" I*/
#include <petscdm.h>

static PetscErrorCode xfunc(PetscInt dim, PetscReal time, const PetscReal coords[], PetscInt Nc, PetscScalar *u, void *ctx)
{
  PetscInt k = *((PetscInt *) ctx), c;

  for (c = 0; c < Nc; ++c) u[c] = PetscPowRealInt(coords[0], k);
  return 0;
}
static PetscErrorCode yfunc(PetscInt dim, PetscReal time, const PetscReal coords[], PetscInt Nc, PetscScalar *u, void *ctx)
{
  PetscInt k = *((PetscInt *) ctx), c;

  for (c = 0; c < Nc; ++c) u[c] = PetscPowRealInt(coords[1], k);
  return 0;
}
static PetscErrorCode zfunc(PetscInt dim, PetscReal time, const PetscReal coords[], PetscInt Nc, PetscScalar *u, void *ctx)
{
  PetscInt k = *((PetscInt *) ctx), c;

  for (c = 0; c < Nc; ++c) u[c] = PetscPowRealInt(coords[2], k);
  return 0;
}
static PetscErrorCode xsin(PetscInt dim, PetscReal time, const PetscReal coords[], PetscInt Nc, PetscScalar *u, void *ctx)
{
  PetscInt k = *((PetscInt *) ctx), c;

  for (c = 0; c < Nc; ++c) u[c] = PetscSinReal(PETSC_PI*(k+1)*coords[0]);
  return 0;
}
static PetscErrorCode ysin(PetscInt dim, PetscReal time, const PetscReal coords[], PetscInt Nc, PetscScalar *u, void *ctx)
{
  PetscInt k = *((PetscInt *) ctx), c;

  for (c = 0; c < Nc; ++c) u[c] = PetscSinReal(PETSC_PI*(k+1)*coords[1]);
  return 0;
}
static PetscErrorCode zsin(PetscInt dim, PetscReal time, const PetscReal coords[], PetscInt Nc, PetscScalar *u, void *ctx)
{
  PetscInt k = *((PetscInt *) ctx), c;

  for (c = 0; c < Nc; ++c) u[c] = PetscSinReal(PETSC_PI*(k+1)*coords[2]);
  return 0;
}

PetscErrorCode DMSetBasisFunction_Internal(PetscInt Nf, PetscBool usePoly, PetscInt dir, PetscErrorCode (**funcs)(PetscInt, PetscReal, const PetscReal[], PetscInt, PetscScalar *, void *))
{
  PetscInt f;

  PetscFunctionBeginUser;
  for (f = 0; f < Nf; ++f) {
    if (usePoly) {
      switch (dir) {
      case 0: funcs[f] = xfunc;break;
      case 1: funcs[f] = yfunc;break;
      case 2: funcs[f] = zfunc;break;
      default: SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "No function for direction %" PetscInt_FMT, dir);
      }
    } else {
      switch (dir) {
      case 0: funcs[f] = xsin;break;
      case 1: funcs[f] = ysin;break;
      case 2: funcs[f] = zsin;break;
      default: SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "No function for direction %" PetscInt_FMT, dir);
      }
    }
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode PCMGCreateCoarseSpaceDefault_Private(PC pc, PetscInt level, PCMGCoarseSpaceType cstype, DM dm, KSP ksp, PetscInt Nc, Mat initialGuess, Mat *coarseSpace)
{
  PetscBool         poly = cstype == PCMG_ADAPT_POLYNOMIAL ? PETSC_TRUE : PETSC_FALSE;
  PetscErrorCode (**funcs)(PetscInt,PetscReal,const PetscReal[],PetscInt,PetscScalar*,void*);
  void            **ctxs;
  PetscInt          dim, d, Nf, f, k, m, M;
  Vec               tmp;

  PetscFunctionBegin;
  Nc = Nc < 0 ? 6 : Nc;
  PetscCall(DMGetCoordinateDim(dm, &dim));
  PetscCall(DMGetNumFields(dm, &Nf));
  PetscCheck(Nc % dim == 0,PetscObjectComm((PetscObject) pc), PETSC_ERR_ARG_WRONG, "The number of coarse vectors %" PetscInt_FMT " must be divisible by the dimension %" PetscInt_FMT, Nc, dim);
  PetscCall(PetscMalloc2(Nf, &funcs, Nf, &ctxs));
  PetscCall(DMGetGlobalVector(dm, &tmp));
  PetscCall(VecGetSize(tmp, &M));
  PetscCall(VecGetLocalSize(tmp, &m));
  PetscCall(MatCreateDense(PetscObjectComm((PetscObject) pc), m, PETSC_DECIDE, M, Nc, NULL, coarseSpace));
  PetscCall(DMRestoreGlobalVector(dm, &tmp));
  for (k = 0; k < Nc/dim; ++k) {
    for (f = 0; f < Nf; ++f) ctxs[f] = &k;
    for (d = 0; d < dim; ++d) {
      PetscCall(MatDenseGetColumnVecWrite(*coarseSpace,k*dim+d,&tmp));
      PetscCall(DMSetBasisFunction_Internal(Nf, poly, d, funcs));
      PetscCall(DMProjectFunction(dm, 0.0, funcs, ctxs, INSERT_ALL_VALUES, tmp));
      PetscCall(MatDenseRestoreColumnVecWrite(*coarseSpace,k*dim+d,&tmp));
    }
  }
  PetscCall(PetscFree2(funcs, ctxs));
  PetscFunctionReturn(0);
}

static PetscErrorCode PCMGCreateCoarseSpace_Polynomial(PC pc, PetscInt level, DM dm, KSP ksp, PetscInt Nc, Mat initialGuess, Mat *coarseSpace)
{
  PetscFunctionBegin;
  PetscCall(PCMGCreateCoarseSpaceDefault_Private(pc, level, PCMG_ADAPT_POLYNOMIAL, dm, ksp, Nc, initialGuess, coarseSpace));
  PetscFunctionReturn(0);
}

PetscErrorCode PCMGCreateCoarseSpace_Harmonic(PC pc, PetscInt level, DM dm, KSP ksp, PetscInt Nc, Mat initialGuess, Mat *coarseSpace)
{
  PetscFunctionBegin;
  PetscCall(PCMGCreateCoarseSpaceDefault_Private(pc, level, PCMG_ADAPT_HARMONIC, dm, ksp, Nc, initialGuess, coarseSpace));
  PetscFunctionReturn(0);
}

/*
  PCMGComputeCoarseSpace_Internal - Compute vectors on level l that must be accurately interpolated.

  Input Parameters:
+ pc     - The PCMG
. l      - The level
. Nc     - The number of vectors requested
- cspace - The initial guess for the space, or NULL

  Output Parameter:
. space  - The space which must be accurately interpolated.

  Level: developer

  Note: This space is normally used to adapt the interpolator. If Nc is negative, an adaptive choice can be made.

.seealso: `PCMGAdaptInterpolator_Private()`
*/
PetscErrorCode PCMGComputeCoarseSpace_Internal(PC pc, PetscInt l, PCMGCoarseSpaceType cstype, PetscInt Nc, Mat cspace, Mat *space)
{
  PetscErrorCode (*coarseConstructor)(PC, PetscInt, DM, KSP, PetscInt, Mat, Mat*) = NULL;
  DM             dm;
  KSP            smooth;

  PetscFunctionBegin;
  *space = NULL;
  switch (cstype) {
  case PCMG_ADAPT_POLYNOMIAL:
    coarseConstructor = &PCMGCreateCoarseSpace_Polynomial;
    break;
  case PCMG_ADAPT_HARMONIC:
    coarseConstructor = &PCMGCreateCoarseSpace_Harmonic;
    break;
  case PCMG_ADAPT_EIGENVECTOR:
    Nc = Nc < 0 ? 6 : Nc;
    if (l > 0) PetscCall(PCMGGetCoarseSpaceConstructor("BAMG_MEV", &coarseConstructor));
    else       PetscCall(PCMGGetCoarseSpaceConstructor("BAMG_EV", &coarseConstructor));
    break;
  case PCMG_ADAPT_GENERALIZED_EIGENVECTOR:
    Nc = Nc < 0 ? 6 : Nc;
    if (l > 0) PetscCall(PCMGGetCoarseSpaceConstructor("BAMG_MGEV", &coarseConstructor));
    else       PetscCall(PCMGGetCoarseSpaceConstructor("BAMG_GEV", &coarseConstructor));
    break;
  case PCMG_ADAPT_GDSW:
    coarseConstructor = &PCMGGDSWCreateCoarseSpace_Private;
    break;
  case PCMG_ADAPT_NONE:
    break;
  default: SETERRQ(PetscObjectComm((PetscObject)pc), PETSC_ERR_SUP, "Cannot handle coarse space type %d", cstype);
  }
  if (coarseConstructor) {
    PetscCall(PCMGGetSmoother(pc, l, &smooth));
    PetscCall(KSPGetDM(smooth, &dm));
    PetscCall((*coarseConstructor)(pc, l, dm, smooth, Nc, cspace, space));
  }
  PetscFunctionReturn(0);
}

/*
  PCMGAdaptInterpolator_Internal - Adapt interpolator from level l-1 to level l

  Input Parameters:
+ pc      - The PCMG
. l       - The level l
. csmooth - The (coarse) smoother for level l-1
. fsmooth - The (fine) smoother for level l
. cspace  - The (coarse) vectors in the subspace for level l-1
- fspace  - The (fine) vectors in the subspace for level l

  Level: developer

  Note: This routine resets the interpolation and restriction for level l.

.seealso: `PCMGComputeCoarseSpace_Private()`
*/
PetscErrorCode PCMGAdaptInterpolator_Internal(PC pc, PetscInt l, KSP csmooth, KSP fsmooth, Mat cspace, Mat fspace)
{
  PC_MG         *mg = (PC_MG *) pc->data;
  DM             dm, cdm;
  Mat            Interp, InterpAdapt;

  PetscFunctionBegin;
  /* There is no interpolator for the coarse level */
  if (!l) PetscFunctionReturn(0);
  PetscCall(KSPGetDM(csmooth, &cdm));
  PetscCall(KSPGetDM(fsmooth, &dm));
  PetscCall(PCMGGetInterpolation(pc, l, &Interp));
  if (Interp == fspace && !cspace) PetscFunctionReturn(0);
  PetscCall(DMAdaptInterpolator(cdm, dm, Interp, fsmooth, fspace, cspace, &InterpAdapt, pc));
  if (mg->mespMonitor) PetscCall(DMCheckInterpolator(dm, InterpAdapt, cspace, fspace, 0.5/* PETSC_SMALL */));
  PetscCall(PCMGSetInterpolation(pc, l, InterpAdapt));
  PetscCall(PCMGSetRestriction(pc, l, InterpAdapt)); /* MATT: Remove????? */
  PetscCall(MatDestroy(&InterpAdapt));
  PetscFunctionReturn(0);
}

/*
  PCMGRecomputeLevelOperators_Internal - Recomputes Galerkin coarse operator when interpolation is adapted

  Input Parameters:
+ pc - The PCMG
- l  - The level l

  Level: developer

  Note: This routine recomputes the Galerkin triple product for the operator on level l.
*/
PetscErrorCode PCMGRecomputeLevelOperators_Internal(PC pc, PetscInt l)
{
  Mat              fA, fB;                   /* The system and preconditioning operators on level l+1 */
  Mat              A,  B;                    /* The system and preconditioning operators on level l */
  Mat              Interp, Restrc;           /* The interpolation operator from level l to l+1, and restriction operator from level l+1 to l */
  KSP              smooth, fsmooth;          /* The smoothers on levels l and l+1 */
  PCMGGalerkinType galerkin;                 /* The Galerkin projection flag */
  MatReuse         reuse = MAT_REUSE_MATRIX; /* The matrices are always assumed to be present already */
  PetscBool        doA   = PETSC_FALSE;      /* Updates the system operator */
  PetscBool        doB   = PETSC_FALSE;      /* Updates the preconditioning operator (A == B, then update B) */
  PetscInt         n;                        /* The number of multigrid levels */

  PetscFunctionBegin;
  PetscCall(PCMGGetGalerkin(pc, &galerkin));
  if (galerkin >= PC_MG_GALERKIN_NONE) PetscFunctionReturn(0);
  PetscCall(PCMGGetLevels(pc, &n));
  /* Do not recompute operator for the finest grid */
  if (l == n-1) PetscFunctionReturn(0);
  PetscCall(PCMGGetSmoother(pc, l,   &smooth));
  PetscCall(KSPGetOperators(smooth, &A, &B));
  PetscCall(PCMGGetSmoother(pc, l+1, &fsmooth));
  PetscCall(KSPGetOperators(fsmooth, &fA, &fB));
  PetscCall(PCMGGetInterpolation(pc, l+1, &Interp));
  PetscCall(PCMGGetRestriction(pc, l+1, &Restrc));
  if ((galerkin == PC_MG_GALERKIN_PMAT) ||  (galerkin == PC_MG_GALERKIN_BOTH))                doB = PETSC_TRUE;
  if ((galerkin == PC_MG_GALERKIN_MAT)  || ((galerkin == PC_MG_GALERKIN_BOTH) && (fA != fB))) doA = PETSC_TRUE;
  if (doA) PetscCall(MatGalerkin(Restrc, fA, Interp, reuse, 1.0, &A));
  if (doB) PetscCall(MatGalerkin(Restrc, fB, Interp, reuse, 1.0, &B));
  PetscFunctionReturn(0);
}
