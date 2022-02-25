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
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  ierr = DMGetCoordinateDim(dm, &dim);CHKERRQ(ierr);
  ierr = DMGetNumFields(dm, &Nf);CHKERRQ(ierr);
  PetscCheckFalse(Nc % dim,PetscObjectComm((PetscObject) pc), PETSC_ERR_ARG_WRONG, "The number of coarse vectors %D must be divisible by the dimension %D", Nc, dim);
  ierr = PetscMalloc2(Nf, &funcs, Nf, &ctxs);CHKERRQ(ierr);
  if (!*coarseSpace) {ierr = PetscCalloc1(Nc, coarseSpace);CHKERRQ(ierr);}
  for (k = 0; k < Nc/dim; ++k) {
    for (f = 0; f < Nf; ++f) {ctxs[f] = &k;}
    for (d = 0; d < dim; ++d) {
      if (!(*coarseSpace)[k*dim+d]) {ierr = DMCreateGlobalVector(dm, &(*coarseSpace)[k*dim+d]);CHKERRQ(ierr);}
      ierr = DMSetBasisFunction_Internal(Nf, poly, d, funcs);CHKERRQ(ierr);
      ierr = DMProjectFunction(dm, 0.0, funcs, ctxs, INSERT_ALL_VALUES, (*coarseSpace)[k*dim+d]);CHKERRQ(ierr);
    }
  }
  ierr = PetscFree2(funcs, ctxs);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode PCMGCreateCoarseSpace_Polynomial(PC pc, PetscInt level, DM dm, KSP ksp, PetscInt Nc, const Vec initialGuess[], Vec **coarseSpace)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PCMGCreateCoarseSpaceDefault_Private(pc, level, PCMG_POLYNOMIAL, dm, ksp, Nc, initialGuess, coarseSpace);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode PCMGCreateCoarseSpace_Harmonic(PC pc, PetscInt level, DM dm, KSP ksp, PetscInt Nc, const Vec initialGuess[], Vec **coarseSpace)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PCMGCreateCoarseSpaceDefault_Private(pc, level, PCMG_HARMONIC, dm, ksp, Nc, initialGuess, coarseSpace);CHKERRQ(ierr);
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
  PetscErrorCode ierr;

  PetscFunctionBegin;
  switch (cstype) {
  case PCMG_POLYNOMIAL: coarseConstructor = &PCMGCreateCoarseSpace_Polynomial;break;
  case PCMG_HARMONIC:   coarseConstructor = &PCMGCreateCoarseSpace_Harmonic;break;
  case PCMG_EIGENVECTOR:
    if (l > 0) {ierr = PCMGGetCoarseSpaceConstructor("BAMG_MEV", &coarseConstructor);CHKERRQ(ierr);}
    else       {ierr = PCMGGetCoarseSpaceConstructor("BAMG_EV", &coarseConstructor);CHKERRQ(ierr);}
    break;
  case PCMG_GENERALIZED_EIGENVECTOR:
    if (l > 0) {ierr = PCMGGetCoarseSpaceConstructor("BAMG_MGEV", &coarseConstructor);CHKERRQ(ierr);}
    else       {ierr = PCMGGetCoarseSpaceConstructor("BAMG_GEV", &coarseConstructor);CHKERRQ(ierr);}
    break;
  default: SETERRQ(PetscObjectComm((PetscObject)pc), PETSC_ERR_ARG_OUTOFRANGE, "Cannot handle coarse space type %D", cstype);
  }
  ierr = PCMGGetSmoother(pc, l, &smooth);CHKERRQ(ierr);
  ierr = KSPGetDM(smooth, &dm);CHKERRQ(ierr);
  ierr = (*coarseConstructor)(pc, l, dm, smooth, Nc, cspace, space);CHKERRQ(ierr);
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
  PetscErrorCode ierr;

  PetscFunctionBegin;
  /* There is no interpolator for the coarse level */
  if (!l) PetscFunctionReturn(0);
  ierr = KSPGetDM(csmooth, &cdm);CHKERRQ(ierr);
  ierr = KSPGetDM(fsmooth, &dm);CHKERRQ(ierr);
  ierr = PCMGGetInterpolation(pc, l, &Interp);CHKERRQ(ierr);

  ierr = DMAdaptInterpolator(cdm, dm, Interp, fsmooth, Nc, fspace, cspace, &InterpAdapt, pc);CHKERRQ(ierr);
  if (mg->mespMonitor) {ierr = DMCheckInterpolator(dm, InterpAdapt, Nc, cspace, fspace, 0.5/* PETSC_SMALL */);CHKERRQ(ierr);}
  ierr = PCMGSetInterpolation(pc, l, InterpAdapt);CHKERRQ(ierr);
  ierr = PCMGSetRestriction(pc, l, InterpAdapt);CHKERRQ(ierr);
  ierr = MatDestroy(&InterpAdapt);CHKERRQ(ierr);
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
  PetscErrorCode   ierr;

  PetscFunctionBegin;
  ierr = PCMGGetGalerkin(pc, &galerkin);CHKERRQ(ierr);
  if (galerkin >= PC_MG_GALERKIN_NONE) PetscFunctionReturn(0);
  ierr = PCMGGetLevels(pc, &n);CHKERRQ(ierr);
  /* Do not recompute operator for the finest grid */
  if (l == n-1) PetscFunctionReturn(0);
  ierr = PCMGGetSmoother(pc, l,   &smooth);CHKERRQ(ierr);
  ierr = KSPGetOperators(smooth, &A, &B);CHKERRQ(ierr);
  ierr = PCMGGetSmoother(pc, l+1, &fsmooth);CHKERRQ(ierr);
  ierr = KSPGetOperators(fsmooth, &fA, &fB);CHKERRQ(ierr);
  ierr = PCMGGetInterpolation(pc, l+1, &Interp);CHKERRQ(ierr);
  ierr = PCMGGetRestriction(pc, l+1, &Restrc);CHKERRQ(ierr);
  if ((galerkin == PC_MG_GALERKIN_PMAT) ||  (galerkin == PC_MG_GALERKIN_BOTH))                doB = PETSC_TRUE;
  if ((galerkin == PC_MG_GALERKIN_MAT)  || ((galerkin == PC_MG_GALERKIN_BOTH) && (fA != fB))) doA = PETSC_TRUE;
  if (doA) {ierr = MatGalerkin(Restrc, fA, Interp, reuse, 1.0, &A);CHKERRQ(ierr);}
  if (doB) {ierr = MatGalerkin(Restrc, fB, Interp, reuse, 1.0, &B);CHKERRQ(ierr);}
  PetscFunctionReturn(0);
}
