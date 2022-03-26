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
      default: SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "No function for direction %D", dir);
      }
    } else {
      switch (dir) {
      case 0: funcs[f] = xsin;break;
      case 1: funcs[f] = ysin;break;
      case 2: funcs[f] = zsin;break;
      default: SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "No function for direction %D", dir);
      }
    }
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode PCMGCreateCoarseSpaceDefault_Private(PC pc, PetscInt level, PCMGCoarseSpaceType cstype, DM dm, KSP ksp, PetscInt Nc, const Vec initialGuess[], Vec **coarseSpace)
{
  PetscBool         poly = cstype == PCMG_POLYNOMIAL ? PETSC_TRUE : PETSC_FALSE;
  PetscErrorCode (**funcs)(PetscInt,PetscReal,const PetscReal[],PetscInt,PetscScalar*,void*);
  void            **ctxs;
  PetscInt          dim, d, Nf, f, k;

  PetscFunctionBegin;
  PetscCall(DMGetCoordinateDim(dm, &dim));
  PetscCall(DMGetNumFields(dm, &Nf));
  PetscCheckFalse(Nc % dim,PetscObjectComm((PetscObject) pc), PETSC_ERR_ARG_WRONG, "The number of coarse vectors %D must be divisible by the dimension %D", Nc, dim);
  PetscCall(PetscMalloc2(Nf, &funcs, Nf, &ctxs));
  if (!*coarseSpace) PetscCall(PetscCalloc1(Nc, coarseSpace));
  for (k = 0; k < Nc/dim; ++k) {
    for (f = 0; f < Nf; ++f) {ctxs[f] = &k;}
    for (d = 0; d < dim; ++d) {
      if (!(*coarseSpace)[k*dim+d]) PetscCall(DMCreateGlobalVector(dm, &(*coarseSpace)[k*dim+d]));
      PetscCall(DMSetBasisFunction_Internal(Nf, poly, d, funcs));
      PetscCall(DMProjectFunction(dm, 0.0, funcs, ctxs, INSERT_ALL_VALUES, (*coarseSpace)[k*dim+d]));
    }
  }
  PetscCall(PetscFree2(funcs, ctxs));
  PetscFunctionReturn(0);
}

static PetscErrorCode PCMGCreateCoarseSpace_Polynomial(PC pc, PetscInt level, DM dm, KSP ksp, PetscInt Nc, const Vec initialGuess[], Vec **coarseSpace)
{
  PetscFunctionBegin;
  PetscCall(PCMGCreateCoarseSpaceDefault_Private(pc, level, PCMG_POLYNOMIAL, dm, ksp, Nc, initialGuess, coarseSpace));
  PetscFunctionReturn(0);
}

PetscErrorCode PCMGCreateCoarseSpace_Harmonic(PC pc, PetscInt level, DM dm, KSP ksp, PetscInt Nc, const Vec initialGuess[], Vec **coarseSpace)
{
  PetscFunctionBegin;
  PetscCall(PCMGCreateCoarseSpaceDefault_Private(pc, level, PCMG_HARMONIC, dm, ksp, Nc, initialGuess, coarseSpace));
  PetscFunctionReturn(0);
}

/*
  PCMGComputeCoarseSpace_Internal - Compute vectors on level l that must be accurately interpolated.

  Input Parameters:
+ pc     - The PCMG
. l      - The level
. Nc     - The size of the space (number of vectors)
- cspace - The space from level l-1, or NULL

  Output Parameter:
. space  - The space which must be accurately interpolated.

  Level: developer

  Note: This space is normally used to adapt the interpolator.

.seealso: PCMGAdaptInterpolator_Private()
*/
PetscErrorCode PCMGComputeCoarseSpace_Internal(PC pc, PetscInt l, PCMGCoarseSpaceType cstype, PetscInt Nc, const Vec cspace[], Vec *space[])
{
  PetscErrorCode (*coarseConstructor)(PC, PetscInt, DM, KSP, PetscInt, const Vec[], Vec*[]);
  DM             dm;
  KSP            smooth;

  PetscFunctionBegin;
  switch (cstype) {
  case PCMG_POLYNOMIAL: coarseConstructor = &PCMGCreateCoarseSpace_Polynomial;break;
  case PCMG_HARMONIC:   coarseConstructor = &PCMGCreateCoarseSpace_Harmonic;break;
  case PCMG_EIGENVECTOR:
    if (l > 0) PetscCall(PCMGGetCoarseSpaceConstructor("BAMG_MEV", &coarseConstructor));
    else       PetscCall(PCMGGetCoarseSpaceConstructor("BAMG_EV", &coarseConstructor));
    break;
  case PCMG_GENERALIZED_EIGENVECTOR:
    if (l > 0) PetscCall(PCMGGetCoarseSpaceConstructor("BAMG_MGEV", &coarseConstructor));
    else       PetscCall(PCMGGetCoarseSpaceConstructor("BAMG_GEV", &coarseConstructor));
    break;
  default: SETERRQ(PetscObjectComm((PetscObject)pc), PETSC_ERR_ARG_OUTOFRANGE, "Cannot handle coarse space type %D", cstype);
  }
  PetscCall(PCMGGetSmoother(pc, l, &smooth));
  PetscCall(KSPGetDM(smooth, &dm));
  PetscCall((*coarseConstructor)(pc, l, dm, smooth, Nc, cspace, space));
  PetscFunctionReturn(0);
}

/*
  PCMGAdaptInterpolator_Internal - Adapt interpolator from level l-1 to level 1

  Input Parameters:
+ pc      - The PCMG
. l       - The level l
. csmooth - The (coarse) smoother for level l-1
. fsmooth - The (fine) smoother for level l
. Nc      - The size of the subspace used for adaptation
. cspace  - The (coarse) vectors in the subspace for level l-1
- fspace  - The (fine) vectors in the subspace for level l

  Level: developer

  Note: This routine resets the interpolation and restriction for level l.

.seealso: PCMGComputeCoarseSpace_Private()
*/
PetscErrorCode PCMGAdaptInterpolator_Internal(PC pc, PetscInt l, KSP csmooth, KSP fsmooth, PetscInt Nc, Vec cspace[], Vec fspace[])
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

  PetscCall(DMAdaptInterpolator(cdm, dm, Interp, fsmooth, Nc, fspace, cspace, &InterpAdapt, pc));
  if (mg->mespMonitor) PetscCall(DMCheckInterpolator(dm, InterpAdapt, Nc, cspace, fspace, 0.5/* PETSC_SMALL */));
  PetscCall(PCMGSetInterpolation(pc, l, InterpAdapt));
  PetscCall(PCMGSetRestriction(pc, l, InterpAdapt));
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
