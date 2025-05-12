#include <../src/snes/impls/al/alimpl.h> /*I "petscsnes.h" I*/

/*
     This file implements a truncated Newton method with arc length continuation,
     for solving a system of nonlinear equations, using the KSP, Vec,
     and Mat interfaces for linear solvers, vectors, and matrices,
     respectively.
*/

const char NewtonALExactCitation[]   = "@article{Ritto-CorreaCamotim2008,\n"
                                       "  title={On the arc-length and other quadratic control methods: Established, less known and new implementation procedures},\n"
                                       "  volume={86},\n"
                                       "  ISSN={0045-7949},\n"
                                       "  DOI={10.1016/j.compstruc.2007.08.003},\n"
                                       "  number={11},\n"
                                       "  journal={Computers & Structures},\n"
                                       "  author={Ritto-Corr{\\^{e}}a, Manuel and Camotim, Dinar},\n"
                                       "  year={2008},\n"
                                       "  month=jun,\n"
                                       "  pages={1353-1368},\n"
                                       "}\n";
PetscBool  NewtonALExactCitationSet  = PETSC_FALSE;
const char NewtonALNormalCitation[]  = "@article{LeonPaulinoPereiraMenezesLages_2011,\n"
                                       "  title={A Unified Library of Nonlinear Solution Schemes},\n"
                                       "  volume={64},\n"
                                       "  ISSN={0003-6900, 2379-0407},\n"
                                       "  DOI={10.1115/1.4006992},\n"
                                       "  number={4},\n"
                                       "  journal={Applied Mechanics Reviews},\n"
                                       "  author={Leon, Sofie E. and Paulino, Glaucio H. and Pereira, Anderson and Menezes, Ivan F. M. and Lages, Eduardo N.},\n"
                                       "  year={2011},\n"
                                       "  month=jul,\n"
                                       "  pages={040803},\n"
                                       "  language={en}\n"
                                       "}\n";
PetscBool  NewtonALNormalCitationSet = PETSC_FALSE;

const char *const SNESNewtonALCorrectionTypes[] = {"EXACT", "NORMAL", "SNESNewtonALCorrectionType", "SNES_NEWTONAL_CORRECTION_", NULL};

static PetscErrorCode SNESNewtonALCheckArcLength(SNES snes, Vec XStep, PetscReal lambdaStep, PetscReal stepSize)
{
  PetscReal      arcLength, arcLengthError;
  SNES_NEWTONAL *al = (SNES_NEWTONAL *)snes->data;

  PetscFunctionBegin;
  PetscCall(VecNorm(XStep, NORM_2, &arcLength));
  arcLength      = PetscSqrtReal(PetscSqr(arcLength) + al->psisq * lambdaStep * lambdaStep);
  arcLengthError = PetscAbsReal(arcLength - stepSize);

  if (arcLengthError > PETSC_SQRT_MACHINE_EPSILON) PetscCall(PetscInfo(snes, "Arc length differs from specified step size: computed=%18.16e, expected=%18.16e, error=%18.16e \n", (double)arcLength, (double)stepSize, (double)arcLengthError));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* stable implementation of roots of a*x^2 + b*x + c = 0 */
static inline void PetscQuadraticRoots(PetscReal a, PetscReal b, PetscReal c, PetscReal *xm, PetscReal *xp)
{
  PetscReal temp = -0.5 * (b + PetscCopysignReal(1.0, b) * PetscSqrtReal(b * b - 4 * a * c));
  PetscReal x1   = temp / a;
  PetscReal x2   = c / temp;
  *xm            = PetscMin(x1, x2);
  *xp            = PetscMax(x1, x2);
}

static PetscErrorCode SNESNewtonALSetCorrectionType_NEWTONAL(SNES snes, SNESNewtonALCorrectionType ctype)
{
  SNES_NEWTONAL *al = (SNES_NEWTONAL *)snes->data;

  PetscFunctionBegin;
  al->correction_type = ctype;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  SNESNewtonALSetCorrectionType - Set the type of correction to use in the arc-length continuation method.

  Logically Collective

  Input Parameters:
+ snes  - the nonlinear solver object
- ctype - the type of correction to use

  Options Database Key:
. -snes_newtonal_correction_type <type> - Set the type of correction to use; use -help for a list of available types

  Level: intermediate

.seealso: [](ch_snes), `SNES`, `SNESNEWTONAL`, `SNESNewtonALCorrectionType`
@*/
PetscErrorCode SNESNewtonALSetCorrectionType(SNES snes, SNESNewtonALCorrectionType ctype)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(snes, SNES_CLASSID, 1);
  PetscValidLogicalCollectiveEnum(snes, ctype, 2);
  PetscTryMethod(snes, "SNESNewtonALSetCorrectionType_C", (SNES, SNESNewtonALCorrectionType), (snes, ctype));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode SNESNewtonALSetFunction_NEWTONAL(SNES snes, SNESFunctionFn *func, void *ctx)
{
  SNES_NEWTONAL *al = (SNES_NEWTONAL *)snes->data;

  PetscFunctionBegin;
  al->computealfunction = func;
  al->alctx             = ctx;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  SNESNewtonALSetFunction - Sets a user function that is called at each function evaluation to
  compute the tangent load vector for the arc-length continuation method.

  Logically Collective

  Input Parameters:
+ snes - the nonlinear solver object
. func - [optional] tangent load function evaluation routine, see `SNESFunctionFn` for the calling sequence. `U` is the current solution vector, `Q` is the output tangent load vector
- ctx  - [optional] user-defined context for private data for the function evaluation routine (may be `NULL`)

  Level: intermediate

  Notes:
  If the current value of the load parameter is needed in `func`, it can be obtained with `SNESNewtonALGetLoadParameter()`.

  The tangent load vector is the partial derivative of external load with respect to the load parameter.
  In the case of proportional loading, the tangent load vector is the full external load vector at the end of the load step.

.seealso: [](ch_snes), `SNES`, `SNESNEWTONAL`, `SNESNewtonALGetFunction()`, `SNESNewtonALGetLoadParameter()`
@*/
PetscErrorCode SNESNewtonALSetFunction(SNES snes, SNESFunctionFn *func, void *ctx)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(snes, SNES_CLASSID, 1);
  PetscTryMethod(snes, "SNESNewtonALSetFunction_C", (SNES, SNESFunctionFn *, void *), (snes, func, ctx));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode SNESNewtonALGetFunction_NEWTONAL(SNES snes, SNESFunctionFn **func, void **ctx)
{
  SNES_NEWTONAL *al = (SNES_NEWTONAL *)snes->data;

  PetscFunctionBegin;
  if (func) *func = al->computealfunction;
  if (ctx) *ctx = al->alctx;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  SNESNewtonALGetFunction - Get the user function and context set with `SNESNewtonALSetFunction`

  Logically Collective

  Input Parameters:
+ snes - the nonlinear solver object
. func - [optional] tangent load function evaluation routine, see `SNESNewtonALSetFunction()` for the call sequence
- ctx  - [optional] user-defined context for private data for the function evaluation routine (may be `NULL`)

  Level: intermediate

.seealso: [](ch_snes), `SNES`, `SNESNEWTONAL`, `SNESNewtonALSetFunction()`
@*/
PetscErrorCode SNESNewtonALGetFunction(SNES snes, SNESFunctionFn **func, void **ctx)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(snes, SNES_CLASSID, 1);
  PetscUseMethod(snes, "SNESNewtonALGetFunction_C", (SNES, SNESFunctionFn **, void **), (snes, func, ctx));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode SNESNewtonALGetLoadParameter_NEWTONAL(SNES snes, PetscReal *lambda)
{
  SNES_NEWTONAL *al;

  PetscFunctionBeginHot;
  al      = (SNES_NEWTONAL *)snes->data;
  *lambda = al->lambda;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  SNESNewtonALGetLoadParameter - Get the value of the load parameter `lambda` for the arc-length continuation method.

  Logically Collective

  Input Parameter:
. snes - the nonlinear solver object

  Output Parameter:
. lambda - the arc-length parameter

  Level: intermediate

  Notes:
  This function should be used in the functions provided to `SNESSetFunction()` and `SNESNewtonALSetFunction()`
  to compute the residual and tangent load vectors for a given value of `lambda` (0 <= lambda <= 1).

  Usually, `lambda` is used to scale the external force vector in the residual function, i.e. proportional loading,
  in which case the tangent load vector is the full external force vector.

.seealso: [](ch_snes), `SNES`, `SNESNEWTONAL`, `SNESNewtonALSetFunction()`
@*/
PetscErrorCode SNESNewtonALGetLoadParameter(SNES snes, PetscReal *lambda)
{
  PetscFunctionBeginHot;
  PetscValidHeaderSpecific(snes, SNES_CLASSID, 1);
  PetscAssertPointer(lambda, 2);
  PetscUseMethod(snes, "SNESNewtonALGetLoadParameter_C", (SNES, PetscReal *), (snes, lambda));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode SNESNewtonALComputeFunction_NEWTONAL(SNES snes, Vec X, Vec Q)
{
  void           *ctx               = NULL;
  SNESFunctionFn *computealfunction = NULL;
  SNES_NEWTONAL  *al;

  PetscFunctionBegin;
  al = (SNES_NEWTONAL *)snes->data;
  PetscCall(SNESNewtonALGetFunction(snes, &computealfunction, &ctx));

  PetscCall(VecZeroEntries(Q));
  PetscCheck(computealfunction || (snes->vec_rhs && al->scale_rhs), PETSC_COMM_SELF, PETSC_ERR_ARG_WRONGSTATE, "No tangent load function or rhs vector has been set");
  if (computealfunction) {
    PetscCall(VecLockReadPush(X));
    PetscCallBack("SNES callback NewtonAL tangent load function", (*computealfunction)(snes, X, Q, ctx));
    PetscCall(VecLockReadPop(X));
  }
  if (al->scale_rhs && snes->vec_rhs) {
    /* Save original RHS vector values, then scale `snes->vec_rhs` by load parameter */
    if (!al->vec_rhs_orig) PetscCall(VecDuplicate(snes->vec_rhs, &al->vec_rhs_orig));
    if (!al->copied_rhs) {
      PetscCall(VecCopy(snes->vec_rhs, al->vec_rhs_orig));
      al->copied_rhs = PETSC_TRUE;
    }
    PetscCall(VecAXPBY(snes->vec_rhs, al->lambda, 0.0, al->vec_rhs_orig));
    PetscCall(VecAXPY(Q, 1, al->vec_rhs_orig));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  SNESNewtonALComputeFunction - Calls the function that has been set with `SNESNewtonALSetFunction()`.

  Collective

  Input Parameters:
+ snes - the `SNES` context
- X    - input vector

  Output Parameter:
. Q - tangent load vector, as set by `SNESNewtonALSetFunction()`

  Level: developer

  Note:
  `SNESNewtonALComputeFunction()` is typically used within nonlinear solvers
  implementations, so users would not generally call this routine themselves.

.seealso: [](ch_snes), `SNES`, `SNESNewtonALSetFunction()`, `SNESNewtonALGetFunction()`
@*/
PetscErrorCode SNESNewtonALComputeFunction(SNES snes, Vec X, Vec Q)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(snes, SNES_CLASSID, 1);
  PetscValidHeaderSpecific(X, VEC_CLASSID, 2);
  PetscValidHeaderSpecific(Q, VEC_CLASSID, 3);
  PetscCheckSameComm(snes, 1, X, 2);
  PetscCheckSameComm(snes, 1, Q, 3);
  PetscCall(VecValidValues_Internal(X, 2, PETSC_TRUE));
  PetscCall(PetscLogEventBegin(SNES_NewtonALEval, snes, X, Q, 0));
  PetscTryMethod(snes, "SNESNewtonALComputeFunction_C", (SNES, Vec, Vec), (snes, X, Q));
  PetscCall(PetscLogEventEnd(SNES_NewtonALEval, snes, X, Q, 0));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
  SNESSolve_NEWTONAL - Solves a nonlinear system with Newton's method with arc length continuation.

  Input Parameter:
. snes - the `SNES` context

  Application Interface Routine: SNESSolve()
*/
static PetscErrorCode SNESSolve_NEWTONAL(SNES snes)
{
  SNES_NEWTONAL *data = (SNES_NEWTONAL *)snes->data;
  PetscInt       maxits, maxincs, lits;
  PetscReal      fnorm, xnorm, ynorm, stepSize;
  Vec            DeltaX, deltaX, X, R, Q, deltaX_Q, deltaX_R;

  PetscFunctionBegin;
  PetscCheck(!snes->xl && !snes->xu && !snes->ops->computevariablebounds, PetscObjectComm((PetscObject)snes), PETSC_ERR_ARG_WRONGSTATE, "SNES solver %s does not support bounds", ((PetscObject)snes)->type_name);

  /* Register citations */
  PetscCall(PetscCitationsRegister(SNESCitation, &SNEScite));
  if (data->correction_type == SNES_NEWTONAL_CORRECTION_EXACT) {
    PetscCall(PetscCitationsRegister(NewtonALExactCitation, &NewtonALExactCitationSet));
  } else if (data->correction_type == SNES_NEWTONAL_CORRECTION_NORMAL) {
    PetscCall(PetscCitationsRegister(NewtonALNormalCitation, &NewtonALNormalCitationSet));
  }

  data->copied_rhs             = PETSC_FALSE;
  data->lambda_update          = 0.0;
  data->lambda                 = 0.0;
  snes->numFailures            = 0;
  snes->numLinearSolveFailures = 0;
  snes->reason                 = SNES_CONVERGED_ITERATING;
  snes->iter                   = 0;

  maxits   = snes->max_its;                /* maximum number of iterations */
  maxincs  = data->max_continuation_steps; /* maximum number of increments */
  X        = snes->vec_sol;                /* solution vector */
  R        = snes->vec_func;               /* residual vector */
  Q        = snes->work[0];                /* tangent load vector */
  deltaX_Q = snes->work[1];                /* variation of X with respect to lambda */
  deltaX_R = snes->work[2];                /* linearized error correction */
  DeltaX   = snes->work[3];                /* step from equilibrium */
  deltaX   = snes->vec_sol_update;         /* full newton step */
  stepSize = data->step_size;              /* initial step size */

  PetscCall(VecZeroEntries(DeltaX));

  /* set snes->max_its for convergence test */
  snes->max_its = maxits * maxincs;

  /* main incremental-iterative loop */
  for (PetscInt i = 0; i < maxincs || maxincs < 0; i++) {
    PetscReal deltaLambda;

    PetscCall(PetscObjectSAWsTakeAccess((PetscObject)snes));
    snes->norm = 0.0;
    PetscCall(PetscObjectSAWsGrantAccess((PetscObject)snes));
    PetscCall(SNESNewtonALComputeFunction(snes, X, Q));
    PetscCall(SNESComputeFunction(snes, X, R));
    PetscCall(VecAXPY(R, 1, Q));           /* R <- R + Q */
    PetscCall(VecNorm(R, NORM_2, &fnorm)); /* fnorm <- ||R|| */
    SNESCheckFunctionNorm(snes, fnorm);

    /* Monitor convergence */
    PetscCall(SNESConverged(snes, snes->iter, 0.0, 0.0, fnorm));
    PetscCall(SNESMonitor(snes, snes->iter, fnorm));
    if (i == 0 && snes->reason) break;
    for (PetscInt j = 0; j < maxits; j++) {
      PetscReal normsqX_Q, deltaS = 1;

      /* Call general purpose update function */
      PetscTryTypeMethod(snes, update, snes->iter);

      PetscCall(SNESComputeJacobian(snes, X, snes->jacobian, snes->jacobian_pre));
      SNESCheckJacobianDomainerror(snes);
      PetscCall(KSPSetOperators(snes->ksp, snes->jacobian, snes->jacobian_pre));
      /* Solve J deltaX_Q = Q, where J is Jacobian matrix */
      PetscCall(KSPSolve(snes->ksp, Q, deltaX_Q));
      SNESCheckKSPSolve(snes);
      PetscCall(KSPGetIterationNumber(snes->ksp, &lits));
      PetscCall(PetscInfo(snes, "iter=%" PetscInt_FMT ", tangent load linear solve iterations=%" PetscInt_FMT "\n", snes->iter, lits));
      /* Compute load parameter variation */
      PetscCall(VecNorm(deltaX_Q, NORM_2, &normsqX_Q));
      normsqX_Q *= normsqX_Q;
      /* On first iter, use predictor. This is the same regardless of corrector scheme. */
      if (j == 0) {
        PetscReal sign = 1.0;
        if (i > 0) {
          PetscCall(VecDotRealPart(DeltaX, deltaX_Q, &sign));
          sign += data->psisq * data->lambda_update;
          sign = sign >= 0 ? 1.0 : -1.0;
        }
        data->lambda_update = 0.0;
        PetscCall(VecZeroEntries(DeltaX));
        deltaLambda = sign * stepSize / PetscSqrtReal(normsqX_Q + data->psisq);
      } else {
        /* Solve J deltaX_R = -R */
        PetscCall(KSPSolve(snes->ksp, R, deltaX_R));
        SNESCheckKSPSolve(snes);
        PetscCall(KSPGetIterationNumber(snes->ksp, &lits));
        PetscCall(PetscInfo(snes, "iter=%" PetscInt_FMT ", residual linear solve iterations=%" PetscInt_FMT "\n", snes->iter, lits));
        PetscCall(VecScale(deltaX_R, -1));

        if (data->correction_type == SNES_NEWTONAL_CORRECTION_NORMAL) {
          /*
            Take a step orthogonal to the current incremental update DeltaX.
            Note, this approach is cheaper than the exact correction, but may exhibit convergence
            issues due to the iterative trial points not being on the quadratic constraint surface.
            On the bright side, we always have a real and unique solution for deltaLambda.
          */
          PetscScalar coefs[2];
          Vec         rhs[] = {deltaX_R, deltaX_Q};

          PetscCall(VecMDot(DeltaX, 2, rhs, coefs));
          deltaLambda = -PetscRealPart(coefs[0]) / (PetscRealPart(coefs[1]) + data->psisq * data->lambda_update);
        } else {
          /*
            Solve
              a*deltaLambda^2 + b*deltaLambda + c = 0  (*)
            where
              a = a0
              b = b0 + b1*deltaS
              c = c0 + c1*deltaS + c2*deltaS^2
            and deltaS is either 1, or the largest value in (0, 1) that satisfies
              b^2 - 4*a*c = as*deltaS^2 + bs*deltaS + cs >= 0
            where
              as = b1^2 - 4*a0*c2
              bs = 2*b1*b0 - 4*a0*c1
              cs = b0^2 - 4*a0*c0
            These "partial corrections" prevent (*) from having complex roots.
          */
          PetscReal   psisqLambdaUpdate, discriminant;
          PetscReal   as, bs, cs;
          PetscReal   a0, b0, b1, c0, c1, c2;
          PetscScalar coefs1[3]; /* coefs[0] = deltaX_Q*DeltaX, coefs[1] = deltaX_R*DeltaX, coefs[2] = DeltaX*DeltaX */
          PetscScalar coefs2[2]; /* coefs[0] = deltaX_Q*deltaX_R, coefs[1] = deltaX_R*deltaX_R */
          const Vec   rhs1[3] = {deltaX_Q, deltaX_R, DeltaX};
          const Vec   rhs2[2] = {deltaX_Q, deltaX_R};

          psisqLambdaUpdate = data->psisq * data->lambda_update;
          PetscCall(VecMDotBegin(DeltaX, 3, rhs1, coefs1));
          PetscCall(VecMDotBegin(deltaX_R, 2, rhs2, coefs2));
          PetscCall(VecMDotEnd(DeltaX, 3, rhs1, coefs1));
          PetscCall(VecMDotEnd(deltaX_R, 2, rhs2, coefs2));

          a0 = normsqX_Q + data->psisq;
          b0 = 2 * (PetscRealPart(coefs1[0]) + psisqLambdaUpdate);
          b1 = 2 * PetscRealPart(coefs2[0]);
          c0 = PetscRealPart(coefs1[2]) + psisqLambdaUpdate * data->lambda_update - stepSize * stepSize;
          c1 = 2 * PetscRealPart(coefs1[1]);
          c2 = PetscRealPart(coefs2[1]);

          as = b1 * b1 - 4 * a0 * c2;
          bs = 2 * (b1 * b0 - 2 * a0 * c1);
          cs = b0 * b0 - 4 * a0 * c0;

          discriminant = cs + bs * deltaS + as * deltaS * deltaS;

          if (discriminant < 0) {
            /* Take deltaS < 1 with the unique root -b/(2*a) */
            PetscReal x1;

            /* Compute deltaS to be the largest root of (as * x^2 + bs * x + cs = 0) */
            PetscQuadraticRoots(as, bs, cs, &x1, &deltaS);
            PetscCall(PetscInfo(snes, "iter=%" PetscInt_FMT ", discriminant=%18.16e < 0, shrinking residual update size to deltaS = %18.16e\n", snes->iter, (double)discriminant, (double)deltaS));
            deltaLambda = -0.5 * (b0 + b1 * deltaS) / a0;
          } else {
            /* Use deltaS = 1, pick root that is closest to the last point to prevent doubling back */
            PetscReal dlambda1, dlambda2;

            PetscQuadraticRoots(a0, b0 + b1, c0 + c1 + c2, &dlambda1, &dlambda2);
            deltaLambda = (b0 * dlambda1) > (b0 * dlambda2) ? dlambda1 : dlambda2;
          }
        }
      }
      PetscCall(PetscObjectSAWsTakeAccess((PetscObject)snes));
      data->lambda = data->lambda + deltaLambda;
      if (data->lambda > data->lambda_max) {
        /* Ensure that lambda = lambda_max exactly at the end of incremental process. This ensures the final solution matches the problem we want to solve. */
        deltaLambda  = deltaLambda - (data->lambda - data->lambda_max);
        data->lambda = data->lambda_max;
      }
      if (data->lambda < data->lambda_min) {
        // LCOV_EXCL_START
        /* Ensure that lambda >= lambda_min. This prevents some potential oscillatory behavior. */
        deltaLambda  = deltaLambda - (data->lambda - data->lambda_min);
        data->lambda = data->lambda_min;
        // LCOV_EXCL_STOP
      }
      data->lambda_update = data->lambda_update + deltaLambda;
      PetscCall(PetscObjectSAWsGrantAccess((PetscObject)snes));
      PetscCall(PetscInfo(snes, "iter=%" PetscInt_FMT ", lambda=%18.16e, lambda_update=%18.16e\n", snes->iter, (double)data->lambda, (double)data->lambda_update));
      if (j == 0) {
        /* deltaX = deltaLambda*deltaX_Q */
        PetscCall(VecCopy(deltaX_Q, deltaX));
        PetscCall(VecScale(deltaX, deltaLambda));
      } else {
        /* deltaX = deltaS*deltaX_R + deltaLambda*deltaX_Q */
        PetscCall(VecAXPBYPCZ(deltaX, deltaS, deltaLambda, 0, deltaX_R, deltaX_Q));
      }
      PetscCall(VecAXPY(DeltaX, 1, deltaX));
      PetscCall(VecAXPY(X, 1, deltaX));
      /* Q = -dF/dlambda(X, lambda)*/
      PetscCall(SNESNewtonALComputeFunction(snes, X, Q));
      /* R = F(X, lambda) */
      PetscCall(SNESComputeFunction(snes, X, R));
      PetscCall(VecNormBegin(R, NORM_2, &fnorm));
      PetscCall(VecNormBegin(X, NORM_2, &xnorm));
      PetscCall(VecNormBegin(deltaX, NORM_2, &ynorm));
      PetscCall(VecNormEnd(R, NORM_2, &fnorm));
      PetscCall(VecNormEnd(X, NORM_2, &xnorm));
      PetscCall(VecNormEnd(deltaX, NORM_2, &ynorm));

      if (PetscLogPrintInfo) PetscCall(SNESNewtonALCheckArcLength(snes, DeltaX, data->lambda_update, stepSize));

      /* Monitor convergence */
      PetscCall(PetscObjectSAWsTakeAccess((PetscObject)snes));
      snes->iter++;
      snes->norm  = fnorm;
      snes->ynorm = ynorm;
      snes->xnorm = xnorm;
      PetscCall(PetscObjectSAWsGrantAccess((PetscObject)snes));
      PetscCall(SNESLogConvergenceHistory(snes, snes->norm, lits));
      PetscCall(SNESConverged(snes, snes->iter, xnorm, ynorm, fnorm));
      PetscCall(SNESMonitor(snes, snes->iter, snes->norm));
      if (!snes->reason && j == maxits - 1) snes->reason = SNES_DIVERGED_MAX_IT;
      if (snes->reason) break;
    }
    if (snes->reason < 0) break;
    if (data->lambda >= data->lambda_max) {
      break;
    } else if (maxincs > 0 && i == maxincs - 1) {
      snes->reason = SNES_DIVERGED_MAX_IT;
      break;
    } else {
      snes->reason = SNES_CONVERGED_ITERATING;
    }
  }
  /* Reset RHS vector, if changed */
  if (data->copied_rhs) {
    PetscCall(VecCopy(data->vec_rhs_orig, snes->vec_rhs));
    data->copied_rhs = PETSC_FALSE;
  }
  snes->max_its = maxits; /* reset snes->max_its */
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
   SNESSetUp_NEWTONAL - Sets up the internal data structures for the later use
   of the SNESNEWTONAL nonlinear solver.

   Input Parameter:
.  snes - the SNES context
.  x - the solution vector

   Application Interface Routine: SNESSetUp()
 */
static PetscErrorCode SNESSetUp_NEWTONAL(SNES snes)
{
  PetscFunctionBegin;
  PetscCall(SNESSetWorkVecs(snes, 4));
  PetscCall(SNESSetUpMatrices(snes));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
   SNESSetFromOptions_NEWTONAL - Sets various parameters for the SNESNEWTONAL method.

   Input Parameter:
.  snes - the SNES context

   Application Interface Routine: SNESSetFromOptions()
*/
static PetscErrorCode SNESSetFromOptions_NEWTONAL(SNES snes, PetscOptionItems PetscOptionsObject)
{
  SNES_NEWTONAL             *data            = (SNES_NEWTONAL *)snes->data;
  SNESNewtonALCorrectionType correction_type = data->correction_type;

  PetscFunctionBegin;
  PetscOptionsHeadBegin(PetscOptionsObject, "SNES Newton Arc Length options");
  PetscCall(PetscOptionsReal("-snes_newtonal_step_size", "Initial arc length increment step size", "SNESNewtonAL", data->step_size, &data->step_size, NULL));
  PetscCall(PetscOptionsInt("-snes_newtonal_max_continuation_steps", "Maximum number of increment steps", "SNESNewtonAL", data->max_continuation_steps, &data->max_continuation_steps, NULL));
  PetscCall(PetscOptionsReal("-snes_newtonal_psisq", "Regularization parameter for arc length continuation, 0 for cylindrical", "SNESNewtonAL", data->psisq, &data->psisq, NULL));
  PetscCall(PetscOptionsReal("-snes_newtonal_lambda_min", "Minimum value of the load parameter lambda", "SNESNewtonAL", data->lambda_min, &data->lambda_min, NULL));
  PetscCall(PetscOptionsReal("-snes_newtonal_lambda_max", "Maximum value of the load parameter lambda", "SNESNewtonAL", data->lambda_max, &data->lambda_max, NULL));
  PetscCall(PetscOptionsBool("-snes_newtonal_scale_rhs", "Scale the constant vector passed to `SNESSolve` by the load parameter lambda", "SNESNewtonAL", data->scale_rhs, &data->scale_rhs, NULL));
  PetscCall(PetscOptionsEnum("-snes_newtonal_correction_type", "Type of correction to use in the arc-length continuation method", "SNESNewtonALCorrectionType", SNESNewtonALCorrectionTypes, (PetscEnum)correction_type, (PetscEnum *)&correction_type, NULL));
  PetscCall(SNESNewtonALSetCorrectionType(snes, correction_type));
  PetscOptionsHeadEnd();
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode SNESReset_NEWTONAL(SNES snes)
{
  SNES_NEWTONAL *al = (SNES_NEWTONAL *)snes->data;

  PetscFunctionBegin;
  PetscCall(VecDestroy(&al->vec_rhs_orig));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
   SNESDestroy_NEWTONAL - Destroys the private SNES_NEWTONAL context that was created
   with SNESCreate_NEWTONAL().

   Input Parameter:
.  snes - the SNES context

   Application Interface Routine: SNESDestroy()
 */
static PetscErrorCode SNESDestroy_NEWTONAL(SNES snes)
{
  PetscFunctionBegin;
  PetscCall(PetscObjectComposeFunction((PetscObject)snes, "SNESNewtonALSetCorrectionType_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)snes, "SNESNewtonALGetLoadParameter_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)snes, "SNESNewtonALSetFunction_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)snes, "SNESNewtonALGetFunction_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)snes, "SNESNewtonALComputeFunction_C", NULL));
  PetscCall(PetscFree(snes->data));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*MC
  SNESNEWTONAL - Newton based nonlinear solver that uses a arc-length continuation method to solve the nonlinear system.

  Options Database Keys:
+   -snes_newtonal_step_size <1.0>              - Initial arc length increment step size
.   -snes_newtonal_max_continuation_steps <100> - Maximum number of continuation steps, or negative for no limit (not recommended)
.   -snes_newtonal_psisq <1.0>                  - Regularization parameter for arc length continuation, 0 for cylindrical. Larger values generally lead to more steps.
.   -snes_newtonal_lambda_min <0.0>             - Minimum value of the load parameter lambda
.   -snes_newtonal_lambda_max <1.0>             - Maximum value of the load parameter lambda
.   -snes_newtonal_scale_rhs <true>             - Scale the constant vector passed to `SNESSolve` by the load parameter lambda
-   -snes_newtonal_correction_type <exact>      - Type of correction to use in the arc-length continuation method, `exact` or `normal`

  Level: intermediate

  Note:
  The exact correction scheme with partial updates is detailed in {cite}`Ritto-CorreaCamotim2008` and the implementation of the
  normal correction scheme is based on {cite}`LeonPaulinoPereiraMenezesLages_2011`.

.seealso: [](ch_snes), `SNESCreate()`, `SNES`, `SNESSetType()`, `SNESNEWTONAL`, `SNESNewtonALSetFunction()`, `SNESNewtonALGetFunction()`, `SNESNewtonALGetLoadParameter()`
M*/
PETSC_EXTERN PetscErrorCode SNESCreate_NEWTONAL(SNES snes)
{
  SNES_NEWTONAL *arclengthParameters;

  PetscFunctionBegin;
  snes->ops->setup          = SNESSetUp_NEWTONAL;
  snes->ops->solve          = SNESSolve_NEWTONAL;
  snes->ops->destroy        = SNESDestroy_NEWTONAL;
  snes->ops->setfromoptions = SNESSetFromOptions_NEWTONAL;
  snes->ops->reset          = SNESReset_NEWTONAL;

  snes->usesksp = PETSC_TRUE;
  snes->usesnpc = PETSC_FALSE;

  PetscCall(SNESParametersInitialize(snes));
  PetscObjectParameterSetDefault(snes, max_funcs, 30000);
  PetscObjectParameterSetDefault(snes, max_its, 50);

  snes->alwayscomputesfinalresidual = PETSC_TRUE;

  PetscCall(PetscNew(&arclengthParameters));
  arclengthParameters->lambda                 = 0.0;
  arclengthParameters->lambda_update          = 0.0;
  arclengthParameters->step_size              = 1.0;
  arclengthParameters->max_continuation_steps = 100;
  arclengthParameters->psisq                  = 1.0;
  arclengthParameters->lambda_min             = 0.0;
  arclengthParameters->lambda_max             = 1.0;
  arclengthParameters->scale_rhs              = PETSC_TRUE;
  arclengthParameters->correction_type        = SNES_NEWTONAL_CORRECTION_EXACT;
  snes->data                                  = (void *)arclengthParameters;

  PetscCall(PetscObjectComposeFunction((PetscObject)snes, "SNESNewtonALSetCorrectionType_C", SNESNewtonALSetCorrectionType_NEWTONAL));
  PetscCall(PetscObjectComposeFunction((PetscObject)snes, "SNESNewtonALGetLoadParameter_C", SNESNewtonALGetLoadParameter_NEWTONAL));
  PetscCall(PetscObjectComposeFunction((PetscObject)snes, "SNESNewtonALSetFunction_C", SNESNewtonALSetFunction_NEWTONAL));
  PetscCall(PetscObjectComposeFunction((PetscObject)snes, "SNESNewtonALGetFunction_C", SNESNewtonALGetFunction_NEWTONAL));
  PetscCall(PetscObjectComposeFunction((PetscObject)snes, "SNESNewtonALComputeFunction_C", SNESNewtonALComputeFunction_NEWTONAL));
  PetscFunctionReturn(PETSC_SUCCESS);
}
