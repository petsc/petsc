
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
  CHKERRQ(MatShellGetContext(CtC,&ctx));
  dm   = ctx->dm;
  mask = ctx->mask;
  CHKERRQ(DMGetLocalVector(dm,&local));
  CHKERRQ(DMGlobalToLocalBegin(dm,x,INSERT_VALUES,local));
  CHKERRQ(DMGlobalToLocalEnd(dm,x,INSERT_VALUES,local));
  if (mask) CHKERRQ(VecPointwiseMult(local,mask,local));
  CHKERRQ(VecSet(y,0.));
  CHKERRQ(DMLocalToGlobalBegin(dm,local,ADD_VALUES,y));
  CHKERRQ(DMLocalToGlobalEnd(dm,local,ADD_VALUES,y));
  CHKERRQ(DMRestoreLocalVector(dm,&local));
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

.seealso: DMGlobalToLocalBegin(), DMGlobalToLocalEnd(), DMLocalToGlobalBegin(), DMLocalToGlobalEnd(), DMPlexGetAnchors(), DMPlexSetAnchors()
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
  CHKERRQ(PetscObjectTypeCompare((PetscObject)dm,DMPLEX,&isPlex));
  if (isPlex) {
    /* mark points in the closure */
    CHKERRQ(DMCreateLocalVector(dm,&mask));
    CHKERRQ(VecSet(mask,0.0));
    CHKERRQ(DMPlexGetSimplexOrBoxCells(dm,0,&cStart,&cEnd));
    if (cEnd > cStart) {
      PetscScalar *ones;
      PetscInt numValues, i;

      CHKERRQ(DMPlexVecGetClosure(dm,NULL,mask,cStart,&numValues,NULL));
      CHKERRQ(PetscMalloc1(numValues,&ones));
      for (i = 0; i < numValues; i++) {
        ones[i] = 1.;
      }
      for (c = cStart; c < cEnd; c++) {
        CHKERRQ(DMPlexVecSetClosure(dm,NULL,mask,c,ones,INSERT_VALUES));
      }
      CHKERRQ(PetscFree(ones));
    }
  }
  else {
    PetscBool hasMask;

    CHKERRQ(DMHasNamedLocalVector(dm, "_DMGlobalToLocalSolve_mask", &hasMask));
    if (!hasMask) {
      PetscErrorCode (**func) (PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nf, PetscScalar *u, void *ctx);
      void            **ctx;
      PetscInt          Nf, f;

      CHKERRQ(DMGetNumFields(dm, &Nf));
      CHKERRQ(PetscMalloc2(Nf, &func, Nf, &ctx));
      for (f = 0; f < Nf; ++f) {
        func[f] = DMGlobalToLocalSolve_project1;
        ctx[f]  = NULL;
      }
      CHKERRQ(DMGetNamedLocalVector(dm, "_DMGlobalToLocalSolve_mask", &mask));
      CHKERRQ(DMProjectFunctionLocal(dm,0.0,func,ctx,INSERT_ALL_VALUES,mask));
      CHKERRQ(DMRestoreNamedLocalVector(dm, "_DMGlobalToLocalSolve_mask", &mask));
      CHKERRQ(PetscFree2(func, ctx));
    }
    CHKERRQ(DMGetNamedLocalVector(dm, "_DMGlobalToLocalSolve_mask", &mask));
  }
  ctx.dm   = dm;
  ctx.mask = mask;
  CHKERRQ(VecGetSize(y,&N));
  CHKERRQ(VecGetLocalSize(y,&n));
  CHKERRQ(MatCreate(PetscObjectComm((PetscObject)dm),&CtC));
  CHKERRQ(MatSetSizes(CtC,n,n,N,N));
  CHKERRQ(MatSetType(CtC,MATSHELL));
  CHKERRQ(MatSetUp(CtC));
  CHKERRQ(MatShellSetContext(CtC,&ctx));
  CHKERRQ(MatShellSetOperation(CtC,MATOP_MULT,(void(*)(void))MatMult_GlobalToLocalNormal));
  CHKERRQ(KSPCreate(PetscObjectComm((PetscObject)dm),&ksp));
  CHKERRQ(KSPSetOperators(ksp,CtC,CtC));
  CHKERRQ(KSPSetType(ksp,KSPCG));
  CHKERRQ(KSPGetPC(ksp,&pc));
  CHKERRQ(PCSetType(pc,PCNONE));
  CHKERRQ(KSPSetInitialGuessNonzero(ksp,PETSC_TRUE));
  CHKERRQ(KSPSetUp(ksp));
  CHKERRQ(DMGetGlobalVector(dm,&global));
  CHKERRQ(VecSet(global,0.));
  if (mask) CHKERRQ(VecPointwiseMult(x,mask,x));
  CHKERRQ(DMLocalToGlobalBegin(dm,x,ADD_VALUES,global));
  CHKERRQ(DMLocalToGlobalEnd(dm,x,ADD_VALUES,global));
  CHKERRQ(KSPSolve(ksp,global,y));
  CHKERRQ(DMRestoreGlobalVector(dm,&global));
  /* clean up */
  CHKERRQ(KSPDestroy(&ksp));
  CHKERRQ(MatDestroy(&CtC));
  if (isPlex) {
    CHKERRQ(VecDestroy(&mask));
  }
  else {
    CHKERRQ(DMRestoreNamedLocalVector(dm, "_DMGlobalToLocalSolve_mask", &mask));
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

.seealso: DMProjectFieldLocal(), DMProjectFieldLabelLocal(), DMProjectFunction(), DMComputeL2Diff()
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
  CHKERRQ(DMGetLocalVector(dm, &localX));
  /* We currently check whether locU == locX to see if we need to apply BC */
  if (U != X) {
    CHKERRQ(VecGetDM(U, &dmIn));
    CHKERRQ(DMGetLocalVector(dmIn, &localU));
  } else {
    dmIn   = dm;
    localU = localX;
  }
  CHKERRQ(DMGlobalToLocalBegin(dmIn, U, INSERT_VALUES, localU));
  CHKERRQ(DMGlobalToLocalEnd(dmIn, U, INSERT_VALUES, localU));
  CHKERRQ(DMProjectFieldLocal(dm, time, localU, funcs, mode, localX));
  CHKERRQ(DMLocalToGlobalBegin(dm, localX, mode, X));
  CHKERRQ(DMLocalToGlobalEnd(dm, localX, mode, X));
  if (mode == INSERT_VALUES || mode == INSERT_ALL_VALUES || mode == INSERT_BC_VALUES) {
    Mat cMat;

    CHKERRQ(DMGetDefaultConstraints(dm, NULL, &cMat, NULL));
    if (cMat) {
      CHKERRQ(DMGlobalToLocalSolve(dm, localX, X));
    }
  }
  CHKERRQ(DMRestoreLocalVector(dm, &localX));
  if (U != X) CHKERRQ(DMRestoreLocalVector(dmIn, &localU));
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
  CHKERRQ(PetscLogEventBegin(DM_AdaptInterpolator,dmc,dmf,0,0));
  CHKERRQ(PetscOptionsGetInt(NULL, NULL, "-dm_interpolator_adapt_debug", &debug, NULL));
  CHKERRQ(MatDuplicate(In, MAT_SHARE_NONZERO_PATTERN, InAdapt));
  CHKERRQ(MatGetOwnershipRange(In, &rStart, &rEnd));
  #if 0
  CHKERRQ(MatGetMaxRowLen(In, &maxcols));
  #else
  for (r = rStart; r < rEnd; ++r) {
    PetscInt           ncols;
    const PetscInt    *cols;
    const PetscScalar *vals;

    CHKERRQ(MatGetRow(In, r, &ncols, &cols, &vals));
    maxcols = PetscMax(maxcols, ncols);
    CHKERRQ(MatRestoreRow(In, r, &ncols, &cols, &vals));
  }
  #endif
  if (Nc < maxcols) PetscPrintf(PETSC_COMM_SELF, "The number of input vectors %D < %D the maximum number of column entries\n", Nc, maxcols);
  for (k = 0; k < Nc; ++k) {
    char        name[PETSC_MAX_PATH_LEN];
    const char *prefix;

    CHKERRQ(PetscObjectGetOptionsPrefix((PetscObject) smoother, &prefix));
    CHKERRQ(PetscSNPrintf(name, PETSC_MAX_PATH_LEN, "%sCoarse Vector %D", prefix ? prefix : NULL, k));
    CHKERRQ(PetscObjectSetName((PetscObject) vc[k], name));
    CHKERRQ(VecViewFromOptions(vc[k], NULL, "-dm_adapt_interp_view_coarse"));
    CHKERRQ(PetscSNPrintf(name, PETSC_MAX_PATH_LEN, "%sFine Vector %D", prefix ? prefix : NULL, k));
    CHKERRQ(PetscObjectSetName((PetscObject) vf[k], name));
    CHKERRQ(VecViewFromOptions(vf[k], NULL, "-dm_adapt_interp_view_fine"));
  }
  CHKERRQ(PetscBLASIntCast(3*PetscMin(Nc, maxcols) + PetscMax(2*PetscMin(Nc, maxcols), PetscMax(Nc, maxcols)), &lwrk));
  CHKERRQ(PetscMalloc7(Nc*maxcols, &A, PetscMax(Nc, maxcols), &b, Nc, &w, maxcols, &x, maxcols, &sing, lwrk, &workscalar, 5*PetscMin(Nc, maxcols), &workreal));
  /* w_k = \frac{\HC{v_k} B_l v_k}{\HC{v_k} A_l v_k} or the inverse Rayleigh quotient, which we calculate using \frac{\HC{v_k} v_k}{\HC{v_k} B^{-1}_l A_l v_k} */
  CHKERRQ(KSPGetOperators(smoother, &globalA, NULL));
  CHKERRQ(DMGetGlobalVector(dmf, &tmp));
  CHKERRQ(DMGetGlobalVector(dmf, &tmp2));
  for (k = 0; k < Nc; ++k) {
    PetscScalar vnorm, vAnorm;
    PetscBool   canMult = PETSC_FALSE;
    const char *type;

    w[k] = 1.0;
    CHKERRQ(PetscObjectGetType((PetscObject) globalA, &type));
    if (type) CHKERRQ(MatAssembled(globalA, &canMult));
    if (type && canMult) {
      CHKERRQ(VecDot(vf[k], vf[k], &vnorm));
      CHKERRQ(MatMult(globalA, vf[k], tmp));
#if 0
      CHKERRQ(KSPSolve(smoother, tmp, tmp2));
      CHKERRQ(VecDot(vf[k], tmp2, &vAnorm));
#else
      CHKERRQ(VecDot(vf[k], tmp, &vAnorm));
#endif
      w[k] = PetscRealPart(vnorm) / PetscRealPart(vAnorm);
    } else SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "System matrix is not assembled.");
  }
  CHKERRQ(DMRestoreGlobalVector(dmf, &tmp));
  CHKERRQ(DMRestoreGlobalVector(dmf, &tmp2));
  if (!vc) {
    allocVc = PETSC_TRUE;
    CHKERRQ(PetscMalloc1(Nc, &vc));
    for (k = 0; k < Nc; ++k) {
      CHKERRQ(DMGetGlobalVector(dmc, &vc[k]));
      CHKERRQ(MatMultTranspose(In, vf[k], vc[k]));
    }
  }
  /* Solve a LS system for each fine row */
  for (r = rStart; r < rEnd; ++r) {
    PetscInt           ncols, c;
    const PetscInt    *cols;
    const PetscScalar *vals, *a;

    CHKERRQ(MatGetRow(In, r, &ncols, &cols, &vals));
    for (k = 0; k < Nc; ++k) {
      /* Need to fit lowest mode exactly */
      const PetscReal wk = ((ncols == 1) && (k > 0)) ? 0.0 : PetscSqrtReal(w[k]);

      /* b_k = \sqrt{w_k} f^{F,k}_r */
      CHKERRQ(VecGetArrayRead(vf[k], &a));
      b[k] = wk * a[r-rStart];
      CHKERRQ(VecRestoreArrayRead(vf[k], &a));
      /* A_{kc} = \sqrt{w_k} f^{C,k}_c */
      /* TODO Must pull out VecScatter from In, scatter in vc[k] values up front, and access them indirectly just as in MatMult() */
      CHKERRQ(VecGetArrayRead(vc[k], &a));
      for (c = 0; c < ncols; ++c) {
        /* This is element (k, c) of A */
        A[c*Nc+k] = wk * a[cols[c]-rStart];
      }
      CHKERRQ(VecRestoreArrayRead(vc[k], &a));
    }
    CHKERRQ(PetscBLASIntCast(Nc,    &M));
    CHKERRQ(PetscBLASIntCast(ncols, &N));
    if (debug) {
#if defined(PETSC_USE_COMPLEX)
      PetscScalar *tmp;
      PetscInt     j;

      CHKERRQ(DMGetWorkArray(dmc, Nc, MPIU_SCALAR, (void *) &tmp));
      for (j = 0; j < Nc; ++j) tmp[j] = w[j];
      CHKERRQ(DMPrintCellMatrix(r, "Interpolator Row LS weights", Nc, 1, tmp));
      CHKERRQ(DMPrintCellMatrix(r, "Interpolator Row LS matrix", Nc, ncols, A));
      for (j = 0; j < Nc; ++j) tmp[j] = b[j];
      CHKERRQ(DMPrintCellMatrix(r, "Interpolator Row LS rhs", Nc, 1, tmp));
      CHKERRQ(DMRestoreWorkArray(dmc, Nc, MPIU_SCALAR, (void *) &tmp));
#else
      CHKERRQ(DMPrintCellMatrix(r, "Interpolator Row LS weights", Nc, 1, w));
      CHKERRQ(DMPrintCellMatrix(r, "Interpolator Row LS matrix", Nc, ncols, A));
      CHKERRQ(DMPrintCellMatrix(r, "Interpolator Row LS rhs", Nc, 1, b));
#endif
    }
#if defined(PETSC_USE_COMPLEX)
    /* ZGELSS( M, N, NRHS, A, LDA, B, LDB, S, RCOND, RANK, WORK, LWORK, RWORK, INFO) */
    PetscStackCallBLAS("LAPACKgelss",LAPACKgelss_(&M, &N, &one, A, &M, b, M > N ? &M : &N, sing, &rcond, &irank, workscalar, &lwrk, workreal, &info));
#else
    /* DGELSS( M, N, NRHS, A, LDA, B, LDB, S, RCOND, RANK, WORK, LWORK, INFO) */
    PetscStackCallBLAS("LAPACKgelss",LAPACKgelss_(&M, &N, &one, A, &M, b, M > N ? &M : &N, sing, &rcond, &irank, workscalar, &lwrk, &info));
#endif
    PetscCheckFalse(info < 0,PETSC_COMM_SELF, PETSC_ERR_LIB, "Bad argument to GELSS");
    PetscCheckFalse(info > 0,PETSC_COMM_SELF, PETSC_ERR_LIB, "SVD failed to converge");
    if (debug) {
      CHKERRQ(PetscPrintf(PETSC_COMM_SELF, "rank %d rcond %g\n", irank, (double) rcond));
#if defined(PETSC_USE_COMPLEX)
      {
        PetscScalar *tmp;
        PetscInt     j;

        CHKERRQ(DMGetWorkArray(dmc, Nc, MPIU_SCALAR, (void *) &tmp));
        for (j = 0; j < PetscMin(Nc, ncols); ++j) tmp[j] = sing[j];
        CHKERRQ(DMPrintCellMatrix(r, "Interpolator Row LS singular values", PetscMin(Nc, ncols), 1, tmp));
        CHKERRQ(DMRestoreWorkArray(dmc, Nc, MPIU_SCALAR, (void *) &tmp));
      }
#else
      CHKERRQ(DMPrintCellMatrix(r, "Interpolator Row LS singular values", PetscMin(Nc, ncols), 1, sing));
#endif
      CHKERRQ(DMPrintCellMatrix(r, "Interpolator Row LS old P", ncols, 1, vals));
      CHKERRQ(DMPrintCellMatrix(r, "Interpolator Row LS sol", ncols, 1, b));
    }
    CHKERRQ(MatSetValues(*InAdapt, 1, &r, ncols, cols, b, INSERT_VALUES));
    CHKERRQ(MatRestoreRow(In, r, &ncols, &cols, &vals));
  }
  CHKERRQ(PetscFree7(A, b, w, x, sing, workscalar, workreal));
  if (allocVc) {
    for (k = 0; k < Nc; ++k) CHKERRQ(DMRestoreGlobalVector(dmc, &vc[k]));
    CHKERRQ(PetscFree(vc));
  }
  CHKERRQ(MatAssemblyBegin(*InAdapt, MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatAssemblyEnd(*InAdapt, MAT_FINAL_ASSEMBLY));
  CHKERRQ(PetscLogEventEnd(DM_AdaptInterpolator,dmc,dmf,0,0));
  PetscFunctionReturn(0);
}

PetscErrorCode DMCheckInterpolator(DM dmf, Mat In, PetscInt Nc, Vec vc[], Vec vf[], PetscReal tol)
{
  Vec            tmp;
  PetscReal      norminf, norm2, maxnorminf = 0.0, maxnorm2 = 0.0;
  PetscInt       k;

  PetscFunctionBegin;
  CHKERRQ(DMGetGlobalVector(dmf, &tmp));
  CHKERRQ(MatViewFromOptions(In, NULL, "-dm_interpolator_adapt_error"));
  for (k = 0; k < Nc; ++k) {
    CHKERRQ(MatMult(In, vc[k], tmp));
    CHKERRQ(VecAXPY(tmp, -1.0, vf[k]));
    CHKERRQ(VecViewFromOptions(vc[k], NULL, "-dm_interpolator_adapt_error"));
    CHKERRQ(VecViewFromOptions(vf[k], NULL, "-dm_interpolator_adapt_error"));
    CHKERRQ(VecViewFromOptions(tmp, NULL, "-dm_interpolator_adapt_error"));
    CHKERRQ(VecNorm(tmp, NORM_INFINITY, &norminf));
    CHKERRQ(VecNorm(tmp, NORM_2, &norm2));
    maxnorminf = PetscMax(maxnorminf, norminf);
    maxnorm2   = PetscMax(maxnorm2,   norm2);
    CHKERRQ(PetscPrintf(PetscObjectComm((PetscObject) dmf), "Coarse vec %D ||vf - P vc||_\\infty %g, ||vf - P vc||_2 %g\n", k, norminf, norm2));
  }
  CHKERRQ(DMRestoreGlobalVector(dmf, &tmp));
  PetscCheckFalse(maxnorm2 > tol,PetscObjectComm((PetscObject) dmf), PETSC_ERR_ARG_WRONG, "max_k ||vf_k - P vc_k||_2 %g > tol %g", maxnorm2, tol);
  PetscFunctionReturn(0);
}
