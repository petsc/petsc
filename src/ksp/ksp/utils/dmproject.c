
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
  PetscErrorCode        ierr;

  PetscFunctionBegin;
  ierr = MatShellGetContext(CtC,&ctx);CHKERRQ(ierr);
  dm   = ctx->dm;
  mask = ctx->mask;
  ierr = DMGetLocalVector(dm,&local);CHKERRQ(ierr);
  ierr = DMGlobalToLocalBegin(dm,x,INSERT_VALUES,local);CHKERRQ(ierr);
  ierr = DMGlobalToLocalEnd(dm,x,INSERT_VALUES,local);CHKERRQ(ierr);
  if (mask) {ierr = VecPointwiseMult(local,mask,local);CHKERRQ(ierr);}
  ierr = VecSet(y,0.);CHKERRQ(ierr);
  ierr = DMLocalToGlobalBegin(dm,local,ADD_VALUES,y);CHKERRQ(ierr);
  ierr = DMLocalToGlobalEnd(dm,local,ADD_VALUES,y);CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(dm,&local);CHKERRQ(ierr);
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
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscObjectTypeCompare((PetscObject)dm,DMPLEX,&isPlex);CHKERRQ(ierr);
  if (isPlex) {
    /* mark points in the closure */
    ierr = DMCreateLocalVector(dm,&mask);CHKERRQ(ierr);
    ierr = VecSet(mask,0.0);CHKERRQ(ierr);
    ierr = DMPlexGetSimplexOrBoxCells(dm,0,&cStart,&cEnd);CHKERRQ(ierr);
    if (cEnd > cStart) {
      PetscScalar *ones;
      PetscInt numValues, i;

      ierr = DMPlexVecGetClosure(dm,NULL,mask,cStart,&numValues,NULL);CHKERRQ(ierr);
      ierr = PetscMalloc1(numValues,&ones);CHKERRQ(ierr);
      for (i = 0; i < numValues; i++) {
        ones[i] = 1.;
      }
      for (c = cStart; c < cEnd; c++) {
        ierr = DMPlexVecSetClosure(dm,NULL,mask,c,ones,INSERT_VALUES);CHKERRQ(ierr);
      }
      ierr = PetscFree(ones);CHKERRQ(ierr);
    }
  }
  else {
    PetscBool hasMask;

    ierr = DMHasNamedLocalVector(dm, "_DMGlobalToLocalSolve_mask", &hasMask);CHKERRQ(ierr);
    if (!hasMask) {
      PetscErrorCode (**func) (PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nf, PetscScalar *u, void *ctx);
      void            **ctx;
      PetscInt          Nf, f;

      ierr = DMGetNumFields(dm, &Nf);CHKERRQ(ierr);
      ierr = PetscMalloc2(Nf, &func, Nf, &ctx);CHKERRQ(ierr);
      for (f = 0; f < Nf; ++f) {
        func[f] = DMGlobalToLocalSolve_project1;
        ctx[f]  = NULL;
      }
      ierr = DMGetNamedLocalVector(dm, "_DMGlobalToLocalSolve_mask", &mask);CHKERRQ(ierr);
      ierr = DMProjectFunctionLocal(dm,0.0,func,ctx,INSERT_ALL_VALUES,mask);CHKERRQ(ierr);
      ierr = DMRestoreNamedLocalVector(dm, "_DMGlobalToLocalSolve_mask", &mask);CHKERRQ(ierr);
      ierr = PetscFree2(func, ctx);CHKERRQ(ierr);
    }
    ierr = DMGetNamedLocalVector(dm, "_DMGlobalToLocalSolve_mask", &mask);CHKERRQ(ierr);
  }
  ctx.dm   = dm;
  ctx.mask = mask;
  ierr = VecGetSize(y,&N);CHKERRQ(ierr);
  ierr = VecGetLocalSize(y,&n);CHKERRQ(ierr);
  ierr = MatCreate(PetscObjectComm((PetscObject)dm),&CtC);CHKERRQ(ierr);
  ierr = MatSetSizes(CtC,n,n,N,N);CHKERRQ(ierr);
  ierr = MatSetType(CtC,MATSHELL);CHKERRQ(ierr);
  ierr = MatSetUp(CtC);CHKERRQ(ierr);
  ierr = MatShellSetContext(CtC,&ctx);CHKERRQ(ierr);
  ierr = MatShellSetOperation(CtC,MATOP_MULT,(void(*)(void))MatMult_GlobalToLocalNormal);CHKERRQ(ierr);
  ierr = KSPCreate(PetscObjectComm((PetscObject)dm),&ksp);CHKERRQ(ierr);
  ierr = KSPSetOperators(ksp,CtC,CtC);CHKERRQ(ierr);
  ierr = KSPSetType(ksp,KSPCG);CHKERRQ(ierr);
  ierr = KSPGetPC(ksp,&pc);CHKERRQ(ierr);
  ierr = PCSetType(pc,PCNONE);CHKERRQ(ierr);
  ierr = KSPSetInitialGuessNonzero(ksp,PETSC_TRUE);CHKERRQ(ierr);
  ierr = KSPSetUp(ksp);CHKERRQ(ierr);
  ierr = DMGetGlobalVector(dm,&global);CHKERRQ(ierr);
  ierr = VecSet(global,0.);CHKERRQ(ierr);
  if (mask) {ierr = VecPointwiseMult(x,mask,x);CHKERRQ(ierr);}
  ierr = DMLocalToGlobalBegin(dm,x,ADD_VALUES,global);CHKERRQ(ierr);
  ierr = DMLocalToGlobalEnd(dm,x,ADD_VALUES,global);CHKERRQ(ierr);
  ierr = KSPSolve(ksp,global,y);CHKERRQ(ierr);
  ierr = DMRestoreGlobalVector(dm,&global);CHKERRQ(ierr);
  /* clean up */
  ierr = KSPDestroy(&ksp);CHKERRQ(ierr);
  ierr = MatDestroy(&CtC);CHKERRQ(ierr);
  if (isPlex) {
    ierr = VecDestroy(&mask);CHKERRQ(ierr);
  }
  else {
    ierr = DMRestoreNamedLocalVector(dm, "_DMGlobalToLocalSolve_mask", &mask);CHKERRQ(ierr);
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
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  ierr = DMGetLocalVector(dm, &localX);CHKERRQ(ierr);
  /* We currently check whether locU == locX to see if we need to apply BC */
  if (U != X) {
    ierr = VecGetDM(U, &dmIn);CHKERRQ(ierr);
    ierr = DMGetLocalVector(dmIn, &localU);CHKERRQ(ierr);
  } else {
    dmIn   = dm;
    localU = localX;
  }
  ierr = DMGlobalToLocalBegin(dmIn, U, INSERT_VALUES, localU);CHKERRQ(ierr);
  ierr = DMGlobalToLocalEnd(dmIn, U, INSERT_VALUES, localU);CHKERRQ(ierr);
  ierr = DMProjectFieldLocal(dm, time, localU, funcs, mode, localX);CHKERRQ(ierr);
  ierr = DMLocalToGlobalBegin(dm, localX, mode, X);CHKERRQ(ierr);
  ierr = DMLocalToGlobalEnd(dm, localX, mode, X);CHKERRQ(ierr);
  if (mode == INSERT_VALUES || mode == INSERT_ALL_VALUES || mode == INSERT_BC_VALUES) {
    Mat cMat;

    ierr = DMGetDefaultConstraints(dm, NULL, &cMat);CHKERRQ(ierr);
    if (cMat) {
      ierr = DMGlobalToLocalSolve(dm, localX, X);CHKERRQ(ierr);
    }
  }
  ierr = DMRestoreLocalVector(dm, &localX);CHKERRQ(ierr);
  if (U != X) {ierr = DMRestoreLocalVector(dmIn, &localU);CHKERRQ(ierr);}
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
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscLogEventBegin(DM_AdaptInterpolator,dmc,dmf,0,0);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(NULL, NULL, "-dm_interpolator_adapt_debug", &debug, NULL);CHKERRQ(ierr);
  ierr = MatDuplicate(In, MAT_SHARE_NONZERO_PATTERN, InAdapt);CHKERRQ(ierr);
  ierr = MatGetOwnershipRange(In, &rStart, &rEnd);CHKERRQ(ierr);
  #if 0
  ierr = MatGetMaxRowLen(In, &maxcols);CHKERRQ(ierr);
  #else
  for (r = rStart; r < rEnd; ++r) {
    PetscInt           ncols;
    const PetscInt    *cols;
    const PetscScalar *vals;

    ierr = MatGetRow(In, r, &ncols, &cols, &vals);CHKERRQ(ierr);
    maxcols = PetscMax(maxcols, ncols);
    ierr = MatRestoreRow(In, r, &ncols, &cols, &vals);CHKERRQ(ierr);
  }
  #endif
  if (Nc < maxcols) PetscPrintf(PETSC_COMM_SELF, "The number of input vectors %D < %D the maximum number of column entries\n", Nc, maxcols);
  for (k = 0; k < Nc; ++k) {
    char        name[PETSC_MAX_PATH_LEN];
    const char *prefix;

    ierr = PetscObjectGetOptionsPrefix((PetscObject) smoother, &prefix);CHKERRQ(ierr);
    ierr = PetscSNPrintf(name, PETSC_MAX_PATH_LEN, "%sCoarse Vector %D", prefix ? prefix : NULL, k);CHKERRQ(ierr);
    ierr = PetscObjectSetName((PetscObject) vc[k], name);CHKERRQ(ierr);
    ierr = VecViewFromOptions(vc[k], NULL, "-dm_adapt_interp_view_coarse");CHKERRQ(ierr);
    ierr = PetscSNPrintf(name, PETSC_MAX_PATH_LEN, "%sFine Vector %D", prefix ? prefix : NULL, k);CHKERRQ(ierr);
    ierr = PetscObjectSetName((PetscObject) vf[k], name);CHKERRQ(ierr);
    ierr = VecViewFromOptions(vf[k], NULL, "-dm_adapt_interp_view_fine");CHKERRQ(ierr);
  }
  ierr = PetscBLASIntCast(3*PetscMin(Nc, maxcols) + PetscMax(2*PetscMin(Nc, maxcols), PetscMax(Nc, maxcols)), &lwrk);CHKERRQ(ierr);
  ierr = PetscMalloc7(Nc*maxcols, &A, PetscMax(Nc, maxcols), &b, Nc, &w, maxcols, &x, maxcols, &sing, lwrk, &workscalar, 5*PetscMin(Nc, maxcols), &workreal);CHKERRQ(ierr);
  /* w_k = \frac{\HC{v_k} B_l v_k}{\HC{v_k} A_l v_k} or the inverse Rayleigh quotient, which we calculate using \frac{\HC{v_k} v_k}{\HC{v_k} B^{-1}_l A_l v_k} */
  ierr = KSPGetOperators(smoother, &globalA, NULL);CHKERRQ(ierr);
  ierr = DMGetGlobalVector(dmf, &tmp);CHKERRQ(ierr);
  ierr = DMGetGlobalVector(dmf, &tmp2);CHKERRQ(ierr);
  for (k = 0; k < Nc; ++k) {
    PetscScalar vnorm, vAnorm;
    PetscBool   canMult = PETSC_FALSE;
    const char *type;

    w[k] = 1.0;
    ierr = PetscObjectGetType((PetscObject) globalA, &type);CHKERRQ(ierr);
    if (type) {ierr = MatAssembled(globalA, &canMult);CHKERRQ(ierr);}
    if (type && canMult) {
      ierr = VecDot(vf[k], vf[k], &vnorm);CHKERRQ(ierr);
      ierr = MatMult(globalA, vf[k], tmp);CHKERRQ(ierr);
#if 0
      ierr = KSPSolve(smoother, tmp, tmp2);CHKERRQ(ierr);
      ierr = VecDot(vf[k], tmp2, &vAnorm);CHKERRQ(ierr);
#else
      ierr = VecDot(vf[k], tmp, &vAnorm);CHKERRQ(ierr);
#endif
      w[k] = PetscRealPart(vnorm) / PetscRealPart(vAnorm);
    } else SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "System matrix is not assembled.");
  }
  ierr = DMRestoreGlobalVector(dmf, &tmp);CHKERRQ(ierr);
  ierr = DMRestoreGlobalVector(dmf, &tmp2);CHKERRQ(ierr);
  if (!vc) {
    allocVc = PETSC_TRUE;
    ierr = PetscMalloc1(Nc, &vc);CHKERRQ(ierr);
    for (k = 0; k < Nc; ++k) {
      ierr = DMGetGlobalVector(dmc, &vc[k]);CHKERRQ(ierr);
      ierr = MatMultTranspose(In, vf[k], vc[k]);CHKERRQ(ierr);
    }
  }
  /* Solve a LS system for each fine row */
  for (r = rStart; r < rEnd; ++r) {
    PetscInt           ncols, c;
    const PetscInt    *cols;
    const PetscScalar *vals, *a;

    ierr = MatGetRow(In, r, &ncols, &cols, &vals);CHKERRQ(ierr);
    for (k = 0; k < Nc; ++k) {
      /* Need to fit lowest mode exactly */
      const PetscReal wk = ((ncols == 1) && (k > 0)) ? 0.0 : PetscSqrtReal(w[k]);

      /* b_k = \sqrt{w_k} f^{F,k}_r */
      ierr = VecGetArrayRead(vf[k], &a);CHKERRQ(ierr);
      b[k] = wk * a[r-rStart];
      ierr = VecRestoreArrayRead(vf[k], &a);CHKERRQ(ierr);
      /* A_{kc} = \sqrt{w_k} f^{C,k}_c */
      /* TODO Must pull out VecScatter from In, scatter in vc[k] values up front, and access them indirectly just as in MatMult() */
      ierr = VecGetArrayRead(vc[k], &a);CHKERRQ(ierr);
      for (c = 0; c < ncols; ++c) {
        /* This is element (k, c) of A */
        A[c*Nc+k] = wk * a[cols[c]-rStart];
      }
      ierr = VecRestoreArrayRead(vc[k], &a);CHKERRQ(ierr);
    }
    ierr = PetscBLASIntCast(Nc,    &M);CHKERRQ(ierr);
    ierr = PetscBLASIntCast(ncols, &N);CHKERRQ(ierr);
    if (debug) {
#if defined(PETSC_USE_COMPLEX)
      PetscScalar *tmp;
      PetscInt     j;

      ierr = DMGetWorkArray(dmc, Nc, MPIU_SCALAR, (void *) &tmp);CHKERRQ(ierr);
      for (j = 0; j < Nc; ++j) tmp[j] = w[j];
      ierr = DMPrintCellMatrix(r, "Interpolator Row LS weights", Nc, 1, tmp);CHKERRQ(ierr);
      ierr = DMPrintCellMatrix(r, "Interpolator Row LS matrix", Nc, ncols, A);CHKERRQ(ierr);
      for (j = 0; j < Nc; ++j) tmp[j] = b[j];
      ierr = DMPrintCellMatrix(r, "Interpolator Row LS rhs", Nc, 1, tmp);CHKERRQ(ierr);
      ierr = DMRestoreWorkArray(dmc, Nc, MPIU_SCALAR, (void *) &tmp);CHKERRQ(ierr);
#else
      ierr = DMPrintCellMatrix(r, "Interpolator Row LS weights", Nc, 1, w);CHKERRQ(ierr);
      ierr = DMPrintCellMatrix(r, "Interpolator Row LS matrix", Nc, ncols, A);CHKERRQ(ierr);
      ierr = DMPrintCellMatrix(r, "Interpolator Row LS rhs", Nc, 1, b);CHKERRQ(ierr);
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
      ierr = PetscPrintf(PETSC_COMM_SELF, "rank %d rcond %g\n", irank, (double) rcond);CHKERRQ(ierr);
#if defined(PETSC_USE_COMPLEX)
      {
        PetscScalar *tmp;
        PetscInt     j;

        ierr = DMGetWorkArray(dmc, Nc, MPIU_SCALAR, (void *) &tmp);CHKERRQ(ierr);
        for (j = 0; j < PetscMin(Nc, ncols); ++j) tmp[j] = sing[j];
        ierr = DMPrintCellMatrix(r, "Interpolator Row LS singular values", PetscMin(Nc, ncols), 1, tmp);CHKERRQ(ierr);
        ierr = DMRestoreWorkArray(dmc, Nc, MPIU_SCALAR, (void *) &tmp);CHKERRQ(ierr);
      }
#else
      ierr = DMPrintCellMatrix(r, "Interpolator Row LS singular values", PetscMin(Nc, ncols), 1, sing);CHKERRQ(ierr);
#endif
      ierr = DMPrintCellMatrix(r, "Interpolator Row LS old P", ncols, 1, vals);CHKERRQ(ierr);
      ierr = DMPrintCellMatrix(r, "Interpolator Row LS sol", ncols, 1, b);CHKERRQ(ierr);
    }
    ierr = MatSetValues(*InAdapt, 1, &r, ncols, cols, b, INSERT_VALUES);CHKERRQ(ierr);
    ierr = MatRestoreRow(In, r, &ncols, &cols, &vals);CHKERRQ(ierr);
  }
  ierr = PetscFree7(A, b, w, x, sing, workscalar, workreal);CHKERRQ(ierr);
  if (allocVc) {
    for (k = 0; k < Nc; ++k) {ierr = DMRestoreGlobalVector(dmc, &vc[k]);CHKERRQ(ierr);}
    ierr = PetscFree(vc);CHKERRQ(ierr);
  }
  ierr = MatAssemblyBegin(*InAdapt, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(*InAdapt, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = PetscLogEventEnd(DM_AdaptInterpolator,dmc,dmf,0,0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode DMCheckInterpolator(DM dmf, Mat In, PetscInt Nc, Vec vc[], Vec vf[], PetscReal tol)
{
  Vec            tmp;
  PetscReal      norminf, norm2, maxnorminf = 0.0, maxnorm2 = 0.0;
  PetscInt       k;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = DMGetGlobalVector(dmf, &tmp);CHKERRQ(ierr);
  ierr = MatViewFromOptions(In, NULL, "-dm_interpolator_adapt_error");CHKERRQ(ierr);
  for (k = 0; k < Nc; ++k) {
    ierr = MatMult(In, vc[k], tmp);CHKERRQ(ierr);
    ierr = VecAXPY(tmp, -1.0, vf[k]);CHKERRQ(ierr);
    ierr = VecViewFromOptions(vc[k], NULL, "-dm_interpolator_adapt_error");CHKERRQ(ierr);
    ierr = VecViewFromOptions(vf[k], NULL, "-dm_interpolator_adapt_error");CHKERRQ(ierr);
    ierr = VecViewFromOptions(tmp, NULL, "-dm_interpolator_adapt_error");CHKERRQ(ierr);
    ierr = VecNorm(tmp, NORM_INFINITY, &norminf);CHKERRQ(ierr);
    ierr = VecNorm(tmp, NORM_2, &norm2);CHKERRQ(ierr);
    maxnorminf = PetscMax(maxnorminf, norminf);
    maxnorm2   = PetscMax(maxnorm2,   norm2);
    ierr = PetscPrintf(PetscObjectComm((PetscObject) dmf), "Coarse vec %D ||vf - P vc||_\\infty %g, ||vf - P vc||_2 %g\n", k, norminf, norm2);CHKERRQ(ierr);
  }
  ierr = DMRestoreGlobalVector(dmf, &tmp);CHKERRQ(ierr);
  PetscCheckFalse(maxnorm2 > tol,PetscObjectComm((PetscObject) dmf), PETSC_ERR_ARG_WRONG, "max_k ||vf_k - P vc_k||_2 %g > tol %g", maxnorm2, tol);
  PetscFunctionReturn(0);
}
