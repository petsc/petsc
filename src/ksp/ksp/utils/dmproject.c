
#include <petsc/private/dmimpl.h>
#include <petscdm.h>     /*I "petscdm.h" I*/
#include <petscdmplex.h> /*I "petscdmplex.h" I*/
#include <petscksp.h>    /*I "petscksp.h" I*/
#include <petscblaslapack.h>

typedef struct _projectConstraintsCtx
{
  DM  dm;
  Vec mask;
}
projectConstraintsCtx;

PetscErrorCode MatMult_GlobalToLocalNormal(Mat CtC, Vec x, Vec y)
{
  DM                    dm;
  Vec                   local, mask;
  projectConstraintsCtx *ctx;

  PetscFunctionBegin;
  PetscCall(MatShellGetContext(CtC,&ctx));
  dm   = ctx->dm;
  mask = ctx->mask;
  PetscCall(DMGetLocalVector(dm,&local));
  PetscCall(DMGlobalToLocalBegin(dm,x,INSERT_VALUES,local));
  PetscCall(DMGlobalToLocalEnd(dm,x,INSERT_VALUES,local));
  if (mask) PetscCall(VecPointwiseMult(local,mask,local));
  PetscCall(VecSet(y,0.));
  PetscCall(DMLocalToGlobalBegin(dm,local,ADD_VALUES,y));
  PetscCall(DMLocalToGlobalEnd(dm,local,ADD_VALUES,y));
  PetscCall(DMRestoreLocalVector(dm,&local));
  PetscFunctionReturn(0);
}

static PetscErrorCode DMGlobalToLocalSolve_project1 (PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nf, PetscScalar u[], void *ctx)
{
  PetscInt f;

  PetscFunctionBegin;
  for (f = 0; f < Nf; f++) {
    u[f] = 1.;
  }
  PetscFunctionReturn(0);
}

/*@
  DMGlobalToLocalSolve - Solve for the global vector that is mapped to a given local vector by DMGlobalToLocalBegin()/DMGlobalToLocalEnd() with mode
  = INSERT_VALUES.  It is assumed that the sum of all the local vector sizes is greater than or equal to the global vector size, so the solution is
  a least-squares solution.  It is also assumed that DMLocalToGlobalBegin()/DMLocalToGlobalEnd() with mode = ADD_VALUES is the adjoint of the
  global-to-local map, so that the least-squares solution may be found by the normal equations.

  collective

  Input Parameters:
+ dm - The DM object
. x - The local vector
- y - The global vector: the input value of globalVec is used as an initial guess

  Output Parameters:
. y - The least-squares solution

  Level: advanced

  Note: If the DM is of type DMPLEX, then y is the solution of L' * D * L * y = L' * D * x, where D is a diagonal mask that is 1 for every point in
  the union of the closures of the local cells and 0 otherwise.  This difference is only relevant if there are anchor points that are not in the
  closure of any local cell (see DMPlexGetAnchors()/DMPlexSetAnchors()).

.seealso: `DMGlobalToLocalBegin()`, `DMGlobalToLocalEnd()`, `DMLocalToGlobalBegin()`, `DMLocalToGlobalEnd()`, `DMPlexGetAnchors()`, `DMPlexSetAnchors()`
@*/
PetscErrorCode DMGlobalToLocalSolve(DM dm, Vec x, Vec y)
{
  Mat                   CtC;
  PetscInt              n, N, cStart, cEnd, c;
  PetscBool             isPlex;
  KSP                   ksp;
  PC                    pc;
  Vec                   global, mask=NULL;
  projectConstraintsCtx ctx;

  PetscFunctionBegin;
  PetscCall(PetscObjectTypeCompare((PetscObject)dm,DMPLEX,&isPlex));
  if (isPlex) {
    /* mark points in the closure */
    PetscCall(DMCreateLocalVector(dm,&mask));
    PetscCall(VecSet(mask,0.0));
    PetscCall(DMPlexGetSimplexOrBoxCells(dm,0,&cStart,&cEnd));
    if (cEnd > cStart) {
      PetscScalar *ones;
      PetscInt numValues, i;

      PetscCall(DMPlexVecGetClosure(dm,NULL,mask,cStart,&numValues,NULL));
      PetscCall(PetscMalloc1(numValues,&ones));
      for (i = 0; i < numValues; i++) {
        ones[i] = 1.;
      }
      for (c = cStart; c < cEnd; c++) {
        PetscCall(DMPlexVecSetClosure(dm,NULL,mask,c,ones,INSERT_VALUES));
      }
      PetscCall(PetscFree(ones));
    }
  }
  else {
    PetscBool hasMask;

    PetscCall(DMHasNamedLocalVector(dm, "_DMGlobalToLocalSolve_mask", &hasMask));
    if (!hasMask) {
      PetscErrorCode (**func) (PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nf, PetscScalar *u, void *ctx);
      void            **ctx;
      PetscInt          Nf, f;

      PetscCall(DMGetNumFields(dm, &Nf));
      PetscCall(PetscMalloc2(Nf, &func, Nf, &ctx));
      for (f = 0; f < Nf; ++f) {
        func[f] = DMGlobalToLocalSolve_project1;
        ctx[f]  = NULL;
      }
      PetscCall(DMGetNamedLocalVector(dm, "_DMGlobalToLocalSolve_mask", &mask));
      PetscCall(DMProjectFunctionLocal(dm,0.0,func,ctx,INSERT_ALL_VALUES,mask));
      PetscCall(DMRestoreNamedLocalVector(dm, "_DMGlobalToLocalSolve_mask", &mask));
      PetscCall(PetscFree2(func, ctx));
    }
    PetscCall(DMGetNamedLocalVector(dm, "_DMGlobalToLocalSolve_mask", &mask));
  }
  ctx.dm   = dm;
  ctx.mask = mask;
  PetscCall(VecGetSize(y,&N));
  PetscCall(VecGetLocalSize(y,&n));
  PetscCall(MatCreate(PetscObjectComm((PetscObject)dm),&CtC));
  PetscCall(MatSetSizes(CtC,n,n,N,N));
  PetscCall(MatSetType(CtC,MATSHELL));
  PetscCall(MatSetUp(CtC));
  PetscCall(MatShellSetContext(CtC,&ctx));
  PetscCall(MatShellSetOperation(CtC,MATOP_MULT,(void(*)(void))MatMult_GlobalToLocalNormal));
  PetscCall(KSPCreate(PetscObjectComm((PetscObject)dm),&ksp));
  PetscCall(KSPSetOperators(ksp,CtC,CtC));
  PetscCall(KSPSetType(ksp,KSPCG));
  PetscCall(KSPGetPC(ksp,&pc));
  PetscCall(PCSetType(pc,PCNONE));
  PetscCall(KSPSetInitialGuessNonzero(ksp,PETSC_TRUE));
  PetscCall(KSPSetUp(ksp));
  PetscCall(DMGetGlobalVector(dm,&global));
  PetscCall(VecSet(global,0.));
  if (mask) PetscCall(VecPointwiseMult(x,mask,x));
  PetscCall(DMLocalToGlobalBegin(dm,x,ADD_VALUES,global));
  PetscCall(DMLocalToGlobalEnd(dm,x,ADD_VALUES,global));
  PetscCall(KSPSolve(ksp,global,y));
  PetscCall(DMRestoreGlobalVector(dm,&global));
  /* clean up */
  PetscCall(KSPDestroy(&ksp));
  PetscCall(MatDestroy(&CtC));
  if (isPlex) {
    PetscCall(VecDestroy(&mask));
  }
  else {
    PetscCall(DMRestoreNamedLocalVector(dm, "_DMGlobalToLocalSolve_mask", &mask));
  }

  PetscFunctionReturn(0);
}

/*@C
  DMProjectField - This projects the given function of the input fields into the function space provided, putting the coefficients in a global vector.

  Collective on DM

  Input Parameters:
+ dm      - The DM
. time    - The time
. U       - The input field vector
. funcs   - The functions to evaluate, one per field
- mode    - The insertion mode for values

  Output Parameter:
. X       - The output vector

   Calling sequence of func:
$    func(PetscInt dim, PetscInt Nf, PetscInt NfAux,
$         const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
$         const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
$         PetscReal t, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar f[]);

+  dim          - The spatial dimension
.  Nf           - The number of input fields
.  NfAux        - The number of input auxiliary fields
.  uOff         - The offset of each field in u[]
.  uOff_x       - The offset of each field in u_x[]
.  u            - The field values at this point in space
.  u_t          - The field time derivative at this point in space (or NULL)
.  u_x          - The field derivatives at this point in space
.  aOff         - The offset of each auxiliary field in u[]
.  aOff_x       - The offset of each auxiliary field in u_x[]
.  a            - The auxiliary field values at this point in space
.  a_t          - The auxiliary field time derivative at this point in space (or NULL)
.  a_x          - The auxiliary field derivatives at this point in space
.  t            - The current time
.  x            - The coordinates of this point
.  numConstants - The number of constants
.  constants    - The value of each constant
-  f            - The value of the function at this point in space

  Note: There are three different DMs that potentially interact in this function. The output DM, dm, specifies the layout of the values calculates by funcs.
  The input DM, attached to U, may be different. For example, you can input the solution over the full domain, but output over a piece of the boundary, or
  a subdomain. You can also output a different number of fields than the input, with different discretizations. Last the auxiliary DM, attached to the
  auxiliary field vector, which is attached to dm, can also be different. It can have a different topology, number of fields, and discretizations.

  Level: intermediate

.seealso: `DMProjectFieldLocal()`, `DMProjectFieldLabelLocal()`, `DMProjectFunction()`, `DMComputeL2Diff()`
@*/
PetscErrorCode DMProjectField(DM dm, PetscReal time, Vec U,
                              void (**funcs)(PetscInt, PetscInt, PetscInt,
                                             const PetscInt[], const PetscInt[], const PetscScalar[], const PetscScalar[], const PetscScalar[],
                                             const PetscInt[], const PetscInt[], const PetscScalar[], const PetscScalar[], const PetscScalar[],
                                             PetscReal, const PetscReal[], PetscInt, const PetscScalar[], PetscScalar[]),
                              InsertMode mode, Vec X)
{
  Vec            localX, localU;
  DM             dmIn;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscCall(DMGetLocalVector(dm, &localX));
  /* We currently check whether locU == locX to see if we need to apply BC */
  if (U != X) {
    PetscCall(VecGetDM(U, &dmIn));
    PetscCall(DMGetLocalVector(dmIn, &localU));
  } else {
    dmIn   = dm;
    localU = localX;
  }
  PetscCall(DMGlobalToLocalBegin(dmIn, U, INSERT_VALUES, localU));
  PetscCall(DMGlobalToLocalEnd(dmIn, U, INSERT_VALUES, localU));
  PetscCall(DMProjectFieldLocal(dm, time, localU, funcs, mode, localX));
  PetscCall(DMLocalToGlobalBegin(dm, localX, mode, X));
  PetscCall(DMLocalToGlobalEnd(dm, localX, mode, X));
  if (mode == INSERT_VALUES || mode == INSERT_ALL_VALUES || mode == INSERT_BC_VALUES) {
    Mat cMat;

    PetscCall(DMGetDefaultConstraints(dm, NULL, &cMat, NULL));
    if (cMat) {
      PetscCall(DMGlobalToLocalSolve(dm, localX, X));
    }
  }
  PetscCall(DMRestoreLocalVector(dm, &localX));
  if (U != X) PetscCall(DMRestoreLocalVector(dmIn, &localU));
  PetscFunctionReturn(0);
}

/********************* Adaptive Interpolation **************************/

/* See the discussion of Adaptive Interpolation in manual/high_level_mg.rst */
PetscErrorCode DMAdaptInterpolator(DM dmc, DM dmf, Mat In, KSP smoother, PetscInt Nc, Vec vf[], Vec vc[], Mat *InAdapt, void *user)
{
  Mat            globalA;
  Vec            tmp, tmp2;
  PetscScalar   *A, *b, *x, *workscalar;
  PetscReal     *w, *sing, *workreal, rcond = PETSC_SMALL;
  PetscBLASInt   M, N, one = 1, irank, lwrk, info;
  PetscInt       debug = 0, rStart, rEnd, r, maxcols = 0, k;
  PetscBool      allocVc = PETSC_FALSE;

  PetscFunctionBegin;
  PetscCall(PetscLogEventBegin(DM_AdaptInterpolator,dmc,dmf,0,0));
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-dm_interpolator_adapt_debug", &debug, NULL));
  PetscCall(MatDuplicate(In, MAT_SHARE_NONZERO_PATTERN, InAdapt));
  PetscCall(MatGetOwnershipRange(In, &rStart, &rEnd));
  #if 0
  PetscCall(MatGetMaxRowLen(In, &maxcols));
  #else
  for (r = rStart; r < rEnd; ++r) {
    PetscInt           ncols;
    const PetscInt    *cols;
    const PetscScalar *vals;

    PetscCall(MatGetRow(In, r, &ncols, &cols, &vals));
    maxcols = PetscMax(maxcols, ncols);
    PetscCall(MatRestoreRow(In, r, &ncols, &cols, &vals));
  }
  #endif
  if (Nc < maxcols) PetscPrintf(PETSC_COMM_SELF, "The number of input vectors %" PetscInt_FMT " < %" PetscInt_FMT " the maximum number of column entries\n", Nc, maxcols);
  for (k = 0; k < Nc; ++k) {
    char        name[PETSC_MAX_PATH_LEN];
    const char *prefix;

    PetscCall(PetscObjectGetOptionsPrefix((PetscObject) smoother, &prefix));
    PetscCall(PetscSNPrintf(name, PETSC_MAX_PATH_LEN, "%sCoarse Vector %" PetscInt_FMT, prefix ? prefix : NULL, k));
    PetscCall(PetscObjectSetName((PetscObject) vc[k], name));
    PetscCall(VecViewFromOptions(vc[k], NULL, "-dm_adapt_interp_view_coarse"));
    PetscCall(PetscSNPrintf(name, PETSC_MAX_PATH_LEN, "%sFine Vector %" PetscInt_FMT, prefix ? prefix : NULL, k));
    PetscCall(PetscObjectSetName((PetscObject) vf[k], name));
    PetscCall(VecViewFromOptions(vf[k], NULL, "-dm_adapt_interp_view_fine"));
  }
  PetscCall(PetscBLASIntCast(3*PetscMin(Nc, maxcols) + PetscMax(2*PetscMin(Nc, maxcols), PetscMax(Nc, maxcols)), &lwrk));
  PetscCall(PetscMalloc7(Nc*maxcols, &A, PetscMax(Nc, maxcols), &b, Nc, &w, maxcols, &x, maxcols, &sing, lwrk, &workscalar, 5*PetscMin(Nc, maxcols), &workreal));
  /* w_k = \frac{\HC{v_k} B_l v_k}{\HC{v_k} A_l v_k} or the inverse Rayleigh quotient, which we calculate using \frac{\HC{v_k} v_k}{\HC{v_k} B^{-1}_l A_l v_k} */
  PetscCall(KSPGetOperators(smoother, &globalA, NULL));
  PetscCall(DMGetGlobalVector(dmf, &tmp));
  PetscCall(DMGetGlobalVector(dmf, &tmp2));
  for (k = 0; k < Nc; ++k) {
    PetscScalar vnorm, vAnorm;
    PetscBool   canMult = PETSC_FALSE;
    const char *type;

    w[k] = 1.0;
    PetscCall(PetscObjectGetType((PetscObject) globalA, &type));
    if (type) PetscCall(MatAssembled(globalA, &canMult));
    if (type && canMult) {
      PetscCall(VecDot(vf[k], vf[k], &vnorm));
      PetscCall(MatMult(globalA, vf[k], tmp));
#if 0
      PetscCall(KSPSolve(smoother, tmp, tmp2));
      PetscCall(VecDot(vf[k], tmp2, &vAnorm));
#else
      PetscCall(VecDot(vf[k], tmp, &vAnorm));
#endif
      w[k] = PetscRealPart(vnorm) / PetscRealPart(vAnorm);
    } else SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "System matrix is not assembled.");
  }
  PetscCall(DMRestoreGlobalVector(dmf, &tmp));
  PetscCall(DMRestoreGlobalVector(dmf, &tmp2));
  if (!vc) {
    allocVc = PETSC_TRUE;
    PetscCall(PetscMalloc1(Nc, &vc));
    for (k = 0; k < Nc; ++k) {
      PetscCall(DMGetGlobalVector(dmc, &vc[k]));
      PetscCall(MatMultTranspose(In, vf[k], vc[k]));
    }
  }
  /* Solve a LS system for each fine row */
  for (r = rStart; r < rEnd; ++r) {
    PetscInt           ncols, c;
    const PetscInt    *cols;
    const PetscScalar *vals, *a;

    PetscCall(MatGetRow(In, r, &ncols, &cols, &vals));
    for (k = 0; k < Nc; ++k) {
      /* Need to fit lowest mode exactly */
      const PetscReal wk = ((ncols == 1) && (k > 0)) ? 0.0 : PetscSqrtReal(w[k]);

      /* b_k = \sqrt{w_k} f^{F,k}_r */
      PetscCall(VecGetArrayRead(vf[k], &a));
      b[k] = wk * a[r-rStart];
      PetscCall(VecRestoreArrayRead(vf[k], &a));
      /* A_{kc} = \sqrt{w_k} f^{C,k}_c */
      /* TODO Must pull out VecScatter from In, scatter in vc[k] values up front, and access them indirectly just as in MatMult() */
      PetscCall(VecGetArrayRead(vc[k], &a));
      for (c = 0; c < ncols; ++c) {
        /* This is element (k, c) of A */
        A[c*Nc+k] = wk * a[cols[c]-rStart];
      }
      PetscCall(VecRestoreArrayRead(vc[k], &a));
    }
    PetscCall(PetscBLASIntCast(Nc,    &M));
    PetscCall(PetscBLASIntCast(ncols, &N));
    if (debug) {
#if defined(PETSC_USE_COMPLEX)
      PetscScalar *tmp;
      PetscInt     j;

      PetscCall(DMGetWorkArray(dmc, Nc, MPIU_SCALAR, (void *) &tmp));
      for (j = 0; j < Nc; ++j) tmp[j] = w[j];
      PetscCall(DMPrintCellMatrix(r, "Interpolator Row LS weights", Nc, 1, tmp));
      PetscCall(DMPrintCellMatrix(r, "Interpolator Row LS matrix", Nc, ncols, A));
      for (j = 0; j < Nc; ++j) tmp[j] = b[j];
      PetscCall(DMPrintCellMatrix(r, "Interpolator Row LS rhs", Nc, 1, tmp));
      PetscCall(DMRestoreWorkArray(dmc, Nc, MPIU_SCALAR, (void *) &tmp));
#else
      PetscCall(DMPrintCellMatrix(r, "Interpolator Row LS weights", Nc, 1, w));
      PetscCall(DMPrintCellMatrix(r, "Interpolator Row LS matrix", Nc, ncols, A));
      PetscCall(DMPrintCellMatrix(r, "Interpolator Row LS rhs", Nc, 1, b));
#endif
    }
#if defined(PETSC_USE_COMPLEX)
    /* ZGELSS( M, N, NRHS, A, LDA, B, LDB, S, RCOND, RANK, WORK, LWORK, RWORK, INFO) */
    PetscStackCallBLAS("LAPACKgelss",LAPACKgelss_(&M, &N, &one, A, &M, b, M > N ? &M : &N, sing, &rcond, &irank, workscalar, &lwrk, workreal, &info));
#else
    /* DGELSS( M, N, NRHS, A, LDA, B, LDB, S, RCOND, RANK, WORK, LWORK, INFO) */
    PetscStackCallBLAS("LAPACKgelss",LAPACKgelss_(&M, &N, &one, A, &M, b, M > N ? &M : &N, sing, &rcond, &irank, workscalar, &lwrk, &info));
#endif
    PetscCheck(info >= 0,PETSC_COMM_SELF, PETSC_ERR_LIB, "Bad argument to GELSS");
    PetscCheck(info <= 0,PETSC_COMM_SELF, PETSC_ERR_LIB, "SVD failed to converge");
    if (debug) {
      PetscCall(PetscPrintf(PETSC_COMM_SELF, "rank %" PetscBLASInt_FMT " rcond %g\n", irank, (double) rcond));
#if defined(PETSC_USE_COMPLEX)
      {
        PetscScalar *tmp;
        PetscInt     j;

        PetscCall(DMGetWorkArray(dmc, Nc, MPIU_SCALAR, (void *) &tmp));
        for (j = 0; j < PetscMin(Nc, ncols); ++j) tmp[j] = sing[j];
        PetscCall(DMPrintCellMatrix(r, "Interpolator Row LS singular values", PetscMin(Nc, ncols), 1, tmp));
        PetscCall(DMRestoreWorkArray(dmc, Nc, MPIU_SCALAR, (void *) &tmp));
      }
#else
      PetscCall(DMPrintCellMatrix(r, "Interpolator Row LS singular values", PetscMin(Nc, ncols), 1, sing));
#endif
      PetscCall(DMPrintCellMatrix(r, "Interpolator Row LS old P", ncols, 1, vals));
      PetscCall(DMPrintCellMatrix(r, "Interpolator Row LS sol", ncols, 1, b));
    }
    PetscCall(MatSetValues(*InAdapt, 1, &r, ncols, cols, b, INSERT_VALUES));
    PetscCall(MatRestoreRow(In, r, &ncols, &cols, &vals));
  }
  PetscCall(PetscFree7(A, b, w, x, sing, workscalar, workreal));
  if (allocVc) {
    for (k = 0; k < Nc; ++k) PetscCall(DMRestoreGlobalVector(dmc, &vc[k]));
    PetscCall(PetscFree(vc));
  }
  PetscCall(MatAssemblyBegin(*InAdapt, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(*InAdapt, MAT_FINAL_ASSEMBLY));
  PetscCall(PetscLogEventEnd(DM_AdaptInterpolator,dmc,dmf,0,0));
  PetscFunctionReturn(0);
}

PetscErrorCode DMCheckInterpolator(DM dmf, Mat In, PetscInt Nc, Vec vc[], Vec vf[], PetscReal tol)
{
  Vec            tmp;
  PetscReal      norminf, norm2, maxnorminf = 0.0, maxnorm2 = 0.0;
  PetscInt       k;

  PetscFunctionBegin;
  PetscCall(DMGetGlobalVector(dmf, &tmp));
  PetscCall(MatViewFromOptions(In, NULL, "-dm_interpolator_adapt_error"));
  for (k = 0; k < Nc; ++k) {
    PetscCall(MatMult(In, vc[k], tmp));
    PetscCall(VecAXPY(tmp, -1.0, vf[k]));
    PetscCall(VecViewFromOptions(vc[k], NULL, "-dm_interpolator_adapt_error"));
    PetscCall(VecViewFromOptions(vf[k], NULL, "-dm_interpolator_adapt_error"));
    PetscCall(VecViewFromOptions(tmp, NULL, "-dm_interpolator_adapt_error"));
    PetscCall(VecNorm(tmp, NORM_INFINITY, &norminf));
    PetscCall(VecNorm(tmp, NORM_2, &norm2));
    maxnorminf = PetscMax(maxnorminf, norminf);
    maxnorm2   = PetscMax(maxnorm2,   norm2);
    PetscCall(PetscPrintf(PetscObjectComm((PetscObject) dmf), "Coarse vec %" PetscInt_FMT " ||vf - P vc||_\\infty %g, ||vf - P vc||_2 %g\n", k, (double)norminf, (double)norm2));
  }
  PetscCall(DMRestoreGlobalVector(dmf, &tmp));
  PetscCheck(maxnorm2 <= tol,PetscObjectComm((PetscObject) dmf), PETSC_ERR_ARG_WRONG, "max_k ||vf_k - P vc_k||_2 %g > tol %g", (double)maxnorm2, (double)tol);
  PetscFunctionReturn(0);
}
