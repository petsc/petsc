/*
      Interface KSP routines that the user calls.
*/

#include <petsc/private/kspimpl.h>   /*I "petscksp.h" I*/
#include <petsc/private/matimpl.h>   /*I "petscmat.h" I*/
#include <petscdm.h>

/* number of nested levels of KSPSetUp/Solve(). This is used to determine if KSP_DIVERGED_ITS should be fatal. */
static PetscInt level = 0;

static inline PetscErrorCode ObjectView(PetscObject obj, PetscViewer viewer, PetscViewerFormat format)
{

  PetscCall(PetscViewerPushFormat(viewer, format));
  PetscCall(PetscObjectView(obj, viewer));
  PetscCall(PetscViewerPopFormat(viewer));
  return(0);
}

/*@
   KSPComputeExtremeSingularValues - Computes the extreme singular values
   for the preconditioned operator. Called after or during KSPSolve().

   Not Collective

   Input Parameter:
.  ksp - iterative context obtained from KSPCreate()

   Output Parameters:
.  emin, emax - extreme singular values

   Options Database Keys:
.  -ksp_view_singularvalues - compute extreme singular values and print when KSPSolve() completes.

   Notes:
   One must call KSPSetComputeSingularValues() before calling KSPSetUp()
   (or use the option -ksp_view_eigenvalues) in order for this routine to work correctly.

   Many users may just want to use the monitoring routine
   KSPMonitorSingularValue() (which can be set with option -ksp_monitor_singular_value)
   to print the extreme singular values at each iteration of the linear solve.

   Estimates of the smallest singular value may be very inaccurate, especially if the Krylov method has not converged.
   The largest singular value is usually accurate to within a few percent if the method has converged, but is still not
   intended for eigenanalysis.

   Disable restarts if using KSPGMRES, otherwise this estimate will only be using those iterations after the last
   restart. See KSPGMRESSetRestart() for more details.

   Level: advanced

.seealso: KSPSetComputeSingularValues(), KSPMonitorSingularValue(), KSPComputeEigenvalues(), KSP
@*/
PetscErrorCode  KSPComputeExtremeSingularValues(KSP ksp,PetscReal *emax,PetscReal *emin)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ksp,KSP_CLASSID,1);
  PetscValidRealPointer(emax,2);
  PetscValidRealPointer(emin,3);
  PetscCheck(ksp->calc_sings,PetscObjectComm((PetscObject)ksp),PETSC_ERR_ARG_WRONGSTATE,"Singular values not requested before KSPSetUp()");

  if (ksp->ops->computeextremesingularvalues) {
    PetscCall((*ksp->ops->computeextremesingularvalues)(ksp,emax,emin));
  } else {
    *emin = -1.0;
    *emax = -1.0;
  }
  PetscFunctionReturn(0);
}

/*@
   KSPComputeEigenvalues - Computes the extreme eigenvalues for the
   preconditioned operator. Called after or during KSPSolve().

   Not Collective

   Input Parameters:
+  ksp - iterative context obtained from KSPCreate()
-  n - size of arrays r and c. The number of eigenvalues computed (neig) will, in
       general, be less than this.

   Output Parameters:
+  r - real part of computed eigenvalues, provided by user with a dimension of at least n
.  c - complex part of computed eigenvalues, provided by user with a dimension of at least n
-  neig - actual number of eigenvalues computed (will be less than or equal to n)

   Options Database Keys:
.  -ksp_view_eigenvalues - Prints eigenvalues to stdout

   Notes:
   The number of eigenvalues estimated depends on the size of the Krylov space
   generated during the KSPSolve() ; for example, with
   CG it corresponds to the number of CG iterations, for GMRES it is the number
   of GMRES iterations SINCE the last restart. Any extra space in r[] and c[]
   will be ignored.

   KSPComputeEigenvalues() does not usually provide accurate estimates; it is
   intended only for assistance in understanding the convergence of iterative
   methods, not for eigenanalysis. For accurate computation of eigenvalues we recommend using
   the excellent package SLEPc.

   One must call KSPSetComputeEigenvalues() before calling KSPSetUp()
   in order for this routine to work correctly.

   Many users may just want to use the monitoring routine
   KSPMonitorSingularValue() (which can be set with option -ksp_monitor_singular_value)
   to print the singular values at each iteration of the linear solve.

   Level: advanced

.seealso: KSPSetComputeSingularValues(), KSPMonitorSingularValue(), KSPComputeExtremeSingularValues(), KSP
@*/
PetscErrorCode  KSPComputeEigenvalues(KSP ksp,PetscInt n,PetscReal r[],PetscReal c[],PetscInt *neig)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ksp,KSP_CLASSID,1);
  if (n) PetscValidRealPointer(r,3);
  if (n) PetscValidRealPointer(c,4);
  PetscCheck(n>=0,PetscObjectComm((PetscObject)ksp),PETSC_ERR_ARG_OUTOFRANGE,"Requested < 0 Eigenvalues");
  PetscValidIntPointer(neig,5);
  PetscCheck(ksp->calc_sings,PetscObjectComm((PetscObject)ksp),PETSC_ERR_ARG_WRONGSTATE,"Eigenvalues not requested before KSPSetUp()");

  if (n && ksp->ops->computeeigenvalues) {
    PetscCall((*ksp->ops->computeeigenvalues)(ksp,n,r,c,neig));
  } else {
    *neig = 0;
  }
  PetscFunctionReturn(0);
}

/*@
   KSPComputeRitz - Computes the Ritz or harmonic Ritz pairs associated to the
   smallest or largest in modulus, for the preconditioned operator.
   Called after KSPSolve().

   Not Collective

   Input Parameters:
+  ksp   - iterative context obtained from KSPCreate()
.  ritz  - PETSC_TRUE or PETSC_FALSE for Ritz pairs or harmonic Ritz pairs, respectively
-  small - PETSC_TRUE or PETSC_FALSE for smallest or largest (harmonic) Ritz values, respectively

   Output Parameters:
+  nrit  - On input number of (harmonic) Ritz pairs to compute; on output, actual number of computed (harmonic) Ritz pairs
.  S     - an array of the Ritz vectors
.  tetar - real part of the Ritz values
-  tetai - imaginary part of the Ritz values

   Notes:
   -For GMRES, the (harmonic) Ritz pairs are computed from the Hessenberg matrix obtained during
   the last complete cycle, or obtained at the end of the solution if the method is stopped before
   a restart. Then, the number of actual (harmonic) Ritz pairs computed is less or equal to the restart
   parameter for GMRES if a complete cycle has been performed or less or equal to the number of GMRES
   iterations.
   -Moreover, for real matrices, the (harmonic) Ritz pairs are possibly complex-valued. In such a case,
   the routine selects the complex (harmonic) Ritz value and its conjugate, and two successive entries of S
   are equal to the real and the imaginary parts of the associated vectors.
   -the (harmonic) Ritz pairs are given in order of increasing (harmonic) Ritz values in modulus
   -this is currently not implemented when PETSc is built with complex numbers

   One must call KSPSetComputeRitz() before calling KSPSetUp()
   in order for this routine to work correctly.

   Level: advanced

.seealso: KSPSetComputeRitz(), KSP
@*/
PetscErrorCode  KSPComputeRitz(KSP ksp,PetscBool ritz,PetscBool small,PetscInt *nrit,Vec S[],PetscReal tetar[],PetscReal tetai[])
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ksp,KSP_CLASSID,1);
  PetscCheck(ksp->calc_ritz,PetscObjectComm((PetscObject)ksp),PETSC_ERR_ARG_WRONGSTATE,"Ritz pairs not requested before KSPSetUp()");
  if (ksp->ops->computeritz) PetscCall((*ksp->ops->computeritz)(ksp,ritz,small,nrit,S,tetar,tetai));
  PetscFunctionReturn(0);
}
/*@
   KSPSetUpOnBlocks - Sets up the preconditioner for each block in
   the block Jacobi, block Gauss-Seidel, and overlapping Schwarz
   methods.

   Collective on ksp

   Input Parameter:
.  ksp - the KSP context

   Notes:
   KSPSetUpOnBlocks() is a routine that the user can optinally call for
   more precise profiling (via -log_view) of the setup phase for these
   block preconditioners.  If the user does not call KSPSetUpOnBlocks(),
   it will automatically be called from within KSPSolve().

   Calling KSPSetUpOnBlocks() is the same as calling PCSetUpOnBlocks()
   on the PC context within the KSP context.

   Level: advanced

.seealso: PCSetUpOnBlocks(), KSPSetUp(), PCSetUp(), KSP
@*/
PetscErrorCode  KSPSetUpOnBlocks(KSP ksp)
{
  PC             pc;
  PCFailedReason pcreason;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ksp,KSP_CLASSID,1);
  level++;
  PetscCall(KSPGetPC(ksp,&pc));
  PetscCall(PCSetUpOnBlocks(pc));
  PetscCall(PCGetFailedReasonRank(pc,&pcreason));
  level--;
  /*
     This is tricky since only a subset of MPI ranks may set this; each KSPSolve_*() is responsible for checking
     this flag and initializing an appropriate vector with VecSetInf() so that the first norm computation can
     produce a result at KSPCheckNorm() thus communicating the known problem to all MPI ranks so they may
     terminate the Krylov solve. For many KSP implementations this is handled within KSPInitialResidual()
  */
  if (pcreason) {
    ksp->reason = KSP_DIVERGED_PC_FAILED;
  }
  PetscFunctionReturn(0);
}

/*@
   KSPSetReusePreconditioner - reuse the current preconditioner, do not construct a new one even if the operator changes

   Collective on ksp

   Input Parameters:
+  ksp   - iterative context obtained from KSPCreate()
-  flag - PETSC_TRUE to reuse the current preconditioner

   Level: intermediate

.seealso: KSPCreate(), KSPSolve(), KSPDestroy(), PCSetReusePreconditioner(), KSP
@*/
PetscErrorCode  KSPSetReusePreconditioner(KSP ksp,PetscBool flag)
{
  PC             pc;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ksp,KSP_CLASSID,1);
  PetscCall(KSPGetPC(ksp,&pc));
  PetscCall(PCSetReusePreconditioner(pc,flag));
  PetscFunctionReturn(0);
}

/*@
   KSPGetReusePreconditioner - Determines if the KSP reuses the current preconditioner even if the operator in the preconditioner has changed.

   Collective on ksp

   Input Parameters:
.  ksp   - iterative context obtained from KSPCreate()

   Output Parameters:
.  flag - the boolean flag

   Level: intermediate

.seealso: KSPCreate(), KSPSolve(), KSPDestroy(), KSPSetReusePreconditioner(), KSP
@*/
PetscErrorCode  KSPGetReusePreconditioner(KSP ksp,PetscBool *flag)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ksp,KSP_CLASSID,1);
  PetscValidBoolPointer(flag,2);
  *flag = PETSC_FALSE;
  if (ksp->pc) {
    PetscCall(PCGetReusePreconditioner(ksp->pc,flag));
  }
  PetscFunctionReturn(0);
}

/*@
   KSPSetSkipPCSetFromOptions - prevents KSPSetFromOptions() from call PCSetFromOptions(). This is used if the same PC is shared by more than one KSP so its options are not resetable for each KSP

   Collective on ksp

   Input Parameters:
+  ksp   - iterative context obtained from KSPCreate()
-  flag - PETSC_TRUE to skip calling the PCSetFromOptions()

   Level: intermediate

.seealso: KSPCreate(), KSPSolve(), KSPDestroy(), PCSetReusePreconditioner(), KSP
@*/
PetscErrorCode  KSPSetSkipPCSetFromOptions(KSP ksp,PetscBool flag)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ksp,KSP_CLASSID,1);
  ksp->skippcsetfromoptions = flag;
  PetscFunctionReturn(0);
}

/*@
   KSPSetUp - Sets up the internal data structures for the
   later use of an iterative solver.

   Collective on ksp

   Input Parameter:
.  ksp   - iterative context obtained from KSPCreate()

   Level: developer

.seealso: KSPCreate(), KSPSolve(), KSPDestroy(), KSP
@*/
PetscErrorCode KSPSetUp(KSP ksp)
{
  Mat            A,B;
  Mat            mat,pmat;
  MatNullSpace   nullsp;
  PCFailedReason pcreason;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ksp,KSP_CLASSID,1);
  level++;

  /* reset the convergence flag from the previous solves */
  ksp->reason = KSP_CONVERGED_ITERATING;

  if (!((PetscObject)ksp)->type_name) {
    PetscCall(KSPSetType(ksp,KSPGMRES));
  }
  PetscCall(KSPSetUpNorms_Private(ksp,PETSC_TRUE,&ksp->normtype,&ksp->pc_side));

  if (ksp->dmActive && !ksp->setupstage) {
    /* first time in so build matrix and vector data structures using DM */
    if (!ksp->vec_rhs) PetscCall(DMCreateGlobalVector(ksp->dm,&ksp->vec_rhs));
    if (!ksp->vec_sol) PetscCall(DMCreateGlobalVector(ksp->dm,&ksp->vec_sol));
    PetscCall(DMCreateMatrix(ksp->dm,&A));
    PetscCall(KSPSetOperators(ksp,A,A));
    PetscCall(PetscObjectDereference((PetscObject)A));
  }

  if (ksp->dmActive) {
    DMKSP kdm;
    PetscCall(DMGetDMKSP(ksp->dm,&kdm));

    if (kdm->ops->computeinitialguess && ksp->setupstage != KSP_SETUP_NEWRHS) {
      /* only computes initial guess the first time through */
      PetscCall((*kdm->ops->computeinitialguess)(ksp,ksp->vec_sol,kdm->initialguessctx));
      PetscCall(KSPSetInitialGuessNonzero(ksp,PETSC_TRUE));
    }
    if (kdm->ops->computerhs) {
      PetscCall((*kdm->ops->computerhs)(ksp,ksp->vec_rhs,kdm->rhsctx));
    }

    if (ksp->setupstage != KSP_SETUP_NEWRHS) {
      if (kdm->ops->computeoperators) {
        PetscCall(KSPGetOperators(ksp,&A,&B));
        PetscCall((*kdm->ops->computeoperators)(ksp,A,B,kdm->operatorsctx));
      } else SETERRQ(PetscObjectComm((PetscObject)ksp),PETSC_ERR_ARG_WRONGSTATE,"You called KSPSetDM() but did not use DMKSPSetComputeOperators() or KSPSetDMActive(ksp,PETSC_FALSE);");
    }
  }

  if (ksp->setupstage == KSP_SETUP_NEWRHS) {
    level--;
    PetscFunctionReturn(0);
  }
  PetscCall(PetscLogEventBegin(KSP_SetUp,ksp,ksp->vec_rhs,ksp->vec_sol,0));

  switch (ksp->setupstage) {
  case KSP_SETUP_NEW:
    PetscCall((*ksp->ops->setup)(ksp));
    break;
  case KSP_SETUP_NEWMATRIX: {   /* This should be replaced with a more general mechanism */
    if (ksp->setupnewmatrix) {
      PetscCall((*ksp->ops->setup)(ksp));
    }
  } break;
  default: break;
  }

  if (!ksp->pc) PetscCall(KSPGetPC(ksp,&ksp->pc));
  PetscCall(PCGetOperators(ksp->pc,&mat,&pmat));
  /* scale the matrix if requested */
  if (ksp->dscale) {
    PetscScalar *xx;
    PetscInt    i,n;
    PetscBool   zeroflag = PETSC_FALSE;
    if (!ksp->pc) PetscCall(KSPGetPC(ksp,&ksp->pc));
    if (!ksp->diagonal) { /* allocate vector to hold diagonal */
      PetscCall(MatCreateVecs(pmat,&ksp->diagonal,NULL));
    }
    PetscCall(MatGetDiagonal(pmat,ksp->diagonal));
    PetscCall(VecGetLocalSize(ksp->diagonal,&n));
    PetscCall(VecGetArray(ksp->diagonal,&xx));
    for (i=0; i<n; i++) {
      if (xx[i] != 0.0) xx[i] = 1.0/PetscSqrtReal(PetscAbsScalar(xx[i]));
      else {
        xx[i]    = 1.0;
        zeroflag = PETSC_TRUE;
      }
    }
    PetscCall(VecRestoreArray(ksp->diagonal,&xx));
    if (zeroflag) {
      PetscCall(PetscInfo(ksp,"Zero detected in diagonal of matrix, using 1 at those locations\n"));
    }
    PetscCall(MatDiagonalScale(pmat,ksp->diagonal,ksp->diagonal));
    if (mat != pmat) PetscCall(MatDiagonalScale(mat,ksp->diagonal,ksp->diagonal));
    ksp->dscalefix2 = PETSC_FALSE;
  }
  PetscCall(PetscLogEventEnd(KSP_SetUp,ksp,ksp->vec_rhs,ksp->vec_sol,0));
  PetscCall(PCSetErrorIfFailure(ksp->pc,ksp->errorifnotconverged));
  PetscCall(PCSetUp(ksp->pc));
  PetscCall(PCGetFailedReasonRank(ksp->pc,&pcreason));
  /* TODO: this code was wrong and is still wrong, there is no way to propagate the failure to all processes; their is no code to handle a ksp->reason on only some ranks */
  if (pcreason) {
    ksp->reason = KSP_DIVERGED_PC_FAILED;
  }

  PetscCall(MatGetNullSpace(mat,&nullsp));
  if (nullsp) {
    PetscBool test = PETSC_FALSE;
    PetscCall(PetscOptionsGetBool(((PetscObject)ksp)->options,((PetscObject)ksp)->prefix,"-ksp_test_null_space",&test,NULL));
    if (test) {
      PetscCall(MatNullSpaceTest(nullsp,mat,NULL));
    }
  }
  ksp->setupstage = KSP_SETUP_NEWRHS;
  level--;
  PetscFunctionReturn(0);
}

/*@C
   KSPConvergedReasonView - Displays the reason a KSP solve converged or diverged to a viewer

   Collective on ksp

   Parameter:
+  ksp - iterative context obtained from KSPCreate()
-  viewer - the viewer to display the reason

   Options Database Keys:
+  -ksp_converged_reason - print reason for converged or diverged, also prints number of iterations
-  -ksp_converged_reason ::failed - only print reason and number of iterations when diverged

   Notes:
     To change the format of the output call PetscViewerPushFormat(viewer,format) before this call. Use PETSC_VIEWER_DEFAULT for the default,
     use PETSC_VIEWER_FAILED to only display a reason if it fails.

   Level: beginner

.seealso: KSPCreate(), KSPSetUp(), KSPDestroy(), KSPSetTolerances(), KSPConvergedDefault(),
          KSPSolveTranspose(), KSPGetIterationNumber(), KSP, KSPGetConvergedReason(), PetscViewerPushFormat(), PetscViewerPopFormat()
@*/
PetscErrorCode KSPConvergedReasonView(KSP ksp, PetscViewer viewer)
{
  PetscBool         isAscii;
  PetscViewerFormat format;

  PetscFunctionBegin;
  if (!viewer) viewer = PETSC_VIEWER_STDOUT_(PetscObjectComm((PetscObject)ksp));
  PetscCall(PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&isAscii));
  if (isAscii) {
    PetscCall(PetscViewerGetFormat(viewer, &format));
    PetscCall(PetscViewerASCIIAddTab(viewer,((PetscObject)ksp)->tablevel));
    if (ksp->reason > 0 && format != PETSC_VIEWER_FAILED) {
      if (((PetscObject) ksp)->prefix) {
        PetscCall(PetscViewerASCIIPrintf(viewer,"Linear %s solve converged due to %s iterations %D\n",((PetscObject) ksp)->prefix,KSPConvergedReasons[ksp->reason],ksp->its));
      } else {
        PetscCall(PetscViewerASCIIPrintf(viewer,"Linear solve converged due to %s iterations %D\n",KSPConvergedReasons[ksp->reason],ksp->its));
      }
    } else if (ksp->reason <= 0) {
      if (((PetscObject) ksp)->prefix) {
        PetscCall(PetscViewerASCIIPrintf(viewer,"Linear %s solve did not converge due to %s iterations %D\n",((PetscObject) ksp)->prefix,KSPConvergedReasons[ksp->reason],ksp->its));
      } else {
        PetscCall(PetscViewerASCIIPrintf(viewer,"Linear solve did not converge due to %s iterations %D\n",KSPConvergedReasons[ksp->reason],ksp->its));
      }
      if (ksp->reason == KSP_DIVERGED_PC_FAILED) {
        PCFailedReason reason;
        PetscCall(PCGetFailedReason(ksp->pc,&reason));
        PetscCall(PetscViewerASCIIPrintf(viewer,"               PC failed due to %s \n",PCFailedReasons[reason]));
      }
    }
    PetscCall(PetscViewerASCIISubtractTab(viewer,((PetscObject)ksp)->tablevel));
  }
  PetscFunctionReturn(0);
}

/*@C
   KSPConvergedReasonViewSet - Sets an ADDITIONAL function that is to be used at the
    end of the linear solver to display the convergence reason of the linear solver.

   Logically Collective on KSP

   Input Parameters:
+  ksp - the KSP context
.  f - the ksp converged reason view function
.  vctx - [optional] user-defined context for private data for the
          ksp converged reason view routine (use NULL if no context is desired)
-  reasonviewdestroy - [optional] routine that frees reasonview context
          (may be NULL)

   Options Database Keys:
+    -ksp_converged_reason        - sets a default KSPConvergedReasonView()
-    -ksp_converged_reason_view_cancel - cancels all converged reason viewers that have
                            been hardwired into a code by
                            calls to KSPConvergedReasonViewSet(), but
                            does not cancel those set via
                            the options database.

   Notes:
   Several different converged reason view routines may be set by calling
   KSPConvergedReasonViewSet() multiple times; all will be called in the
   order in which they were set.

   Level: intermediate

.seealso: KSPConvergedReasonView(), KSPConvergedReasonViewCancel()
@*/
PetscErrorCode  KSPConvergedReasonViewSet(KSP ksp,PetscErrorCode (*f)(KSP,void*),void *vctx,PetscErrorCode (*reasonviewdestroy)(void**))
{
  PetscInt       i;
  PetscBool      identical;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ksp,KSP_CLASSID,1);
  for (i=0; i<ksp->numberreasonviews;i++) {
    PetscCall(PetscMonitorCompare((PetscErrorCode (*)(void))f,vctx,reasonviewdestroy,(PetscErrorCode (*)(void))ksp->reasonview[i],ksp->reasonviewcontext[i],ksp->reasonviewdestroy[i],&identical));
    if (identical) PetscFunctionReturn(0);
  }
  PetscCheck(ksp->numberreasonviews < MAXKSPREASONVIEWS,PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Too many KSP reasonview set");
  ksp->reasonview[ksp->numberreasonviews]          = f;
  ksp->reasonviewdestroy[ksp->numberreasonviews]   = reasonviewdestroy;
  ksp->reasonviewcontext[ksp->numberreasonviews++] = (void*)vctx;
  PetscFunctionReturn(0);
}

/*@
   KSPConvergedReasonViewCancel - Clears all the reasonview functions for a KSP object.

   Collective on KSP

   Input Parameter:
.  ksp - iterative context obtained from KSPCreate()

   Level: intermediate

.seealso: KSPCreate(), KSPDestroy(), KSPReset()
@*/
PetscErrorCode  KSPConvergedReasonViewCancel(KSP ksp)
{
  PetscInt       i;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ksp,KSP_CLASSID,1);
  for (i=0; i<ksp->numberreasonviews; i++) {
    if (ksp->reasonviewdestroy[i]) {
      PetscCall((*ksp->reasonviewdestroy[i])(&ksp->reasonviewcontext[i]));
    }
  }
  ksp->numberreasonviews = 0;
  PetscFunctionReturn(0);
}

/*@
  KSPConvergedReasonViewFromOptions - Processes command line options to determine if/how a KSPReason is to be viewed.

  Collective on ksp

  Input Parameters:
. ksp   - the KSP object

  Level: intermediate

.seealso: KSPConvergedReasonView()
@*/
PetscErrorCode KSPConvergedReasonViewFromOptions(KSP ksp)
{
  PetscViewer       viewer;
  PetscBool         flg;
  PetscViewerFormat format;
  PetscInt          i;

  PetscFunctionBegin;

  /* Call all user-provided reason review routines */
  for (i=0; i<ksp->numberreasonviews; i++) {
    PetscCall((*ksp->reasonview[i])(ksp,ksp->reasonviewcontext[i]));
  }

  /* Call the default PETSc routine */
  PetscCall(PetscOptionsGetViewer(PetscObjectComm((PetscObject)ksp),((PetscObject)ksp)->options,((PetscObject)ksp)->prefix,"-ksp_converged_reason",&viewer,&format,&flg));
  if (flg) {
    PetscCall(PetscViewerPushFormat(viewer,format));
    PetscCall(KSPConvergedReasonView(ksp, viewer));
    PetscCall(PetscViewerPopFormat(viewer));
    PetscCall(PetscViewerDestroy(&viewer));
  }
  PetscFunctionReturn(0);
}

/*@C
  KSPConvergedRateView - Displays the reason a KSP solve converged or diverged to a viewer

  Collective on ksp

  Input Parameters:
+  ksp    - iterative context obtained from KSPCreate()
-  viewer - the viewer to display the reason

  Options Database Keys:
. -ksp_converged_rate - print reason for convergence or divergence and the convergence rate (or 0.0 for divergence)

  Notes:
  To change the format of the output, call PetscViewerPushFormat(viewer,format) before this call.

  Suppose that the residual is reduced linearly, $r_k = c^k r_0$, which means $log r_k = log r_0 + k log c$. After linear regression,
  the slope is $\log c$. The coefficient of determination is given by $1 - \frac{\sum_i (y_i - f(x_i))^2}{\sum_i (y_i - \bar y)}$,
  see also https://en.wikipedia.org/wiki/Coefficient_of_determination

  Level: intermediate

.seealso: KSPConvergedReasonView(), KSPGetConvergedRate(), KSPSetTolerances(), KSPConvergedDefault()
@*/
PetscErrorCode KSPConvergedRateView(KSP ksp, PetscViewer viewer)
{
  PetscViewerFormat format;
  PetscBool         isAscii;
  PetscReal         rrate, rRsq, erate = 0.0, eRsq = 0.0;
  PetscInt          its;
  const char       *prefix, *reason = KSPConvergedReasons[ksp->reason];

  PetscFunctionBegin;
  PetscCall(KSPGetOptionsPrefix(ksp, &prefix));
  PetscCall(KSPGetIterationNumber(ksp, &its));
  PetscCall(KSPComputeConvergenceRate(ksp, &rrate, &rRsq, &erate, &eRsq));
  if (!viewer) viewer = PETSC_VIEWER_STDOUT_(PetscObjectComm((PetscObject)ksp));
  PetscCall(PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&isAscii));
  if (isAscii) {
    PetscCall(PetscViewerGetFormat(viewer, &format));
    PetscCall(PetscViewerASCIIAddTab(viewer,((PetscObject)ksp)->tablevel));
    if (ksp->reason > 0) {
      if (prefix) PetscCall(PetscViewerASCIIPrintf(viewer, "Linear %s solve converged due to %s iterations %D", prefix, reason, its));
      else        PetscCall(PetscViewerASCIIPrintf(viewer, "Linear solve converged due to %s iterations %D", reason, its));
      PetscCall(PetscViewerASCIIUseTabs(viewer, PETSC_FALSE));
      if (rRsq >= 0.0) PetscCall(PetscViewerASCIIPrintf(viewer, " res rate %g R^2 %g", rrate, rRsq));
      if (eRsq >= 0.0) PetscCall(PetscViewerASCIIPrintf(viewer, " error rate %g R^2 %g", erate, eRsq));
      PetscCall(PetscViewerASCIIPrintf(viewer, "\n"));
      PetscCall(PetscViewerASCIIUseTabs(viewer, PETSC_TRUE));
    } else if (ksp->reason <= 0) {
      if (prefix) PetscCall(PetscViewerASCIIPrintf(viewer, "Linear %s solve did not converge due to %s iterations %D", prefix, reason, its));
      else        PetscCall(PetscViewerASCIIPrintf(viewer, "Linear solve did not converge due to %s iterations %D", reason, its));
      PetscCall(PetscViewerASCIIUseTabs(viewer, PETSC_FALSE));
      if (rRsq >= 0.0) PetscCall(PetscViewerASCIIPrintf(viewer, " res rate %g R^2 %g", rrate, rRsq));
      if (eRsq >= 0.0) PetscCall(PetscViewerASCIIPrintf(viewer, " error rate %g R^2 %g", erate, eRsq));
      PetscCall(PetscViewerASCIIPrintf(viewer, "\n"));
      PetscCall(PetscViewerASCIIUseTabs(viewer, PETSC_TRUE));
      if (ksp->reason == KSP_DIVERGED_PC_FAILED) {
        PCFailedReason reason;
        PetscCall(PCGetFailedReason(ksp->pc,&reason));
        PetscCall(PetscViewerASCIIPrintf(viewer,"               PC failed due to %s \n",PCFailedReasons[reason]));
      }
    }
    PetscCall(PetscViewerASCIISubtractTab(viewer,((PetscObject)ksp)->tablevel));
  }
  PetscFunctionReturn(0);
}

#include <petscdraw.h>

static PetscErrorCode KSPViewEigenvalues_Internal(KSP ksp, PetscBool isExplicit, PetscViewer viewer, PetscViewerFormat format)
{
  PetscReal     *r, *c;
  PetscInt       n, i, neig;
  PetscBool      isascii, isdraw;
  PetscMPIInt    rank;

  PetscFunctionBegin;
  PetscCallMPI(MPI_Comm_rank(PetscObjectComm((PetscObject) ksp), &rank));
  PetscCall(PetscObjectTypeCompare((PetscObject) viewer, PETSCVIEWERASCII, &isascii));
  PetscCall(PetscObjectTypeCompare((PetscObject) viewer, PETSCVIEWERDRAW,  &isdraw));
  if (isExplicit) {
    PetscCall(VecGetSize(ksp->vec_sol,&n));
    PetscCall(PetscMalloc2(n, &r, n, &c));
    PetscCall(KSPComputeEigenvaluesExplicitly(ksp, n, r, c));
    neig = n;
  } else {
    PetscInt nits;

    PetscCall(KSPGetIterationNumber(ksp, &nits));
    n    = nits+2;
    if (!nits) {PetscCall(PetscViewerASCIIPrintf(viewer, "Zero iterations in solver, cannot approximate any eigenvalues\n"));PetscFunctionReturn(0);}
    PetscCall(PetscMalloc2(n, &r, n, &c));
    PetscCall(KSPComputeEigenvalues(ksp, n, r, c, &neig));
  }
  if (isascii) {
    PetscCall(PetscViewerASCIIPrintf(viewer, "%s computed eigenvalues\n", isExplicit ? "Explicitly" : "Iteratively"));
    for (i = 0; i < neig; ++i) {
      if (c[i] >= 0.0) PetscCall(PetscViewerASCIIPrintf(viewer, "%g + %gi\n", (double) r[i],  (double) c[i]));
      else             PetscCall(PetscViewerASCIIPrintf(viewer, "%g - %gi\n", (double) r[i], -(double) c[i]));
    }
  } else if (isdraw && rank == 0) {
    PetscDraw   draw;
    PetscDrawSP drawsp;

    if (format == PETSC_VIEWER_DRAW_CONTOUR) {
      PetscCall(KSPPlotEigenContours_Private(ksp,neig,r,c));
    } else {
      if (!ksp->eigviewer) PetscCall(PetscViewerDrawOpen(PETSC_COMM_SELF,NULL,isExplicit ? "Explicitly Computed Eigenvalues" : "Iteratively Computed Eigenvalues",PETSC_DECIDE,PETSC_DECIDE,400,400,&ksp->eigviewer));
      PetscCall(PetscViewerDrawGetDraw(ksp->eigviewer,0,&draw));
      PetscCall(PetscDrawSPCreate(draw,1,&drawsp));
      PetscCall(PetscDrawSPReset(drawsp));
      for (i = 0; i < neig; ++i) PetscCall(PetscDrawSPAddPoint(drawsp,r+i,c+i));
      PetscCall(PetscDrawSPDraw(drawsp,PETSC_TRUE));
      PetscCall(PetscDrawSPSave(drawsp));
      PetscCall(PetscDrawSPDestroy(&drawsp));
    }
  }
  PetscCall(PetscFree2(r, c));
  PetscFunctionReturn(0);
}

static PetscErrorCode KSPViewSingularvalues_Internal(KSP ksp, PetscViewer viewer, PetscViewerFormat format)
{
  PetscReal      smax, smin;
  PetscInt       nits;
  PetscBool      isascii;

  PetscFunctionBegin;
  PetscCall(PetscObjectTypeCompare((PetscObject) viewer, PETSCVIEWERASCII, &isascii));
  PetscCall(KSPGetIterationNumber(ksp, &nits));
  if (!nits) {PetscCall(PetscViewerASCIIPrintf(viewer, "Zero iterations in solver, cannot approximate any singular values\n"));PetscFunctionReturn(0);}
  PetscCall(KSPComputeExtremeSingularValues(ksp, &smax, &smin));
  if (isascii) PetscCall(PetscViewerASCIIPrintf(viewer, "Iteratively computed extreme singular values: max %g min %g max/min %g\n",(double)smax,(double)smin,(double)(smax/smin)));
  PetscFunctionReturn(0);
}

static PetscErrorCode KSPViewFinalResidual_Internal(KSP ksp, PetscViewer viewer, PetscViewerFormat format)
{
  PetscBool      isascii;

  PetscFunctionBegin;
  PetscCall(PetscObjectTypeCompare((PetscObject) viewer, PETSCVIEWERASCII, &isascii));
  PetscCheck(!ksp->dscale || ksp->dscalefix,PetscObjectComm((PetscObject) ksp), PETSC_ERR_ARG_WRONGSTATE, "Cannot compute final scale with -ksp_diagonal_scale except also with -ksp_diagonal_scale_fix");
  if (isascii) {
    Mat       A;
    Vec       t;
    PetscReal norm;

    PetscCall(PCGetOperators(ksp->pc, &A, NULL));
    PetscCall(VecDuplicate(ksp->vec_rhs, &t));
    PetscCall(KSP_MatMult(ksp, A, ksp->vec_sol, t));
    PetscCall(VecAYPX(t, -1.0, ksp->vec_rhs));
    PetscCall(VecNorm(t, NORM_2, &norm));
    PetscCall(VecDestroy(&t));
    PetscCall(PetscViewerASCIIPrintf(viewer, "KSP final norm of residual %g\n", (double) norm));
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode KSPMonitorPauseFinal_Internal(KSP ksp)
{
  PetscInt       i;

  PetscFunctionBegin;
  if (!ksp->pauseFinal) PetscFunctionReturn(0);
  for (i = 0; i < ksp->numbermonitors; ++i) {
    PetscViewerAndFormat *vf = (PetscViewerAndFormat *) ksp->monitorcontext[i];
    PetscDraw             draw;
    PetscReal             lpause;

    if (!vf) continue;
    if (vf->lg) {
      if (!PetscCheckPointer(vf->lg, PETSC_OBJECT)) continue;
      if (((PetscObject) vf->lg)->classid != PETSC_DRAWLG_CLASSID) continue;
      PetscCall(PetscDrawLGGetDraw(vf->lg, &draw));
      PetscCall(PetscDrawGetPause(draw, &lpause));
      PetscCall(PetscDrawSetPause(draw, -1.0));
      PetscCall(PetscDrawPause(draw));
      PetscCall(PetscDrawSetPause(draw, lpause));
    } else {
      PetscBool isdraw;

      if (!PetscCheckPointer(vf->viewer, PETSC_OBJECT)) continue;
      if (((PetscObject) vf->viewer)->classid != PETSC_VIEWER_CLASSID) continue;
      PetscCall(PetscObjectTypeCompare((PetscObject) vf->viewer, PETSCVIEWERDRAW, &isdraw));
      if (!isdraw) continue;
      PetscCall(PetscViewerDrawGetDraw(vf->viewer, 0, &draw));
      PetscCall(PetscDrawGetPause(draw, &lpause));
      PetscCall(PetscDrawSetPause(draw, -1.0));
      PetscCall(PetscDrawPause(draw));
      PetscCall(PetscDrawSetPause(draw, lpause));
    }
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode KSPSolve_Private(KSP ksp,Vec b,Vec x)
{
  PetscBool       flg = PETSC_FALSE,inXisinB = PETSC_FALSE,guess_zero;
  Mat             mat,pmat;
  MPI_Comm        comm;
  MatNullSpace    nullsp;
  Vec             btmp,vec_rhs = NULL;

  PetscFunctionBegin;
  level++;
  comm = PetscObjectComm((PetscObject)ksp);
  if (x && x == b) {
    PetscCheck(ksp->guess_zero,comm,PETSC_ERR_ARG_INCOMP,"Cannot use x == b with nonzero initial guess");
    PetscCall(VecDuplicate(b,&x));
    inXisinB = PETSC_TRUE;
  }
  if (b) {
    PetscCall(PetscObjectReference((PetscObject)b));
    PetscCall(VecDestroy(&ksp->vec_rhs));
    ksp->vec_rhs = b;
  }
  if (x) {
    PetscCall(PetscObjectReference((PetscObject)x));
    PetscCall(VecDestroy(&ksp->vec_sol));
    ksp->vec_sol = x;
  }

  if (ksp->viewPre) PetscCall(ObjectView((PetscObject) ksp, ksp->viewerPre, ksp->formatPre));

  if (ksp->presolve) PetscCall((*ksp->presolve)(ksp,ksp->vec_rhs,ksp->vec_sol,ksp->prectx));

  /* reset the residual history list if requested */
  if (ksp->res_hist_reset) ksp->res_hist_len = 0;
  if (ksp->err_hist_reset) ksp->err_hist_len = 0;

  if (ksp->guess) {
    PetscObjectState ostate,state;

    PetscCall(KSPGuessSetUp(ksp->guess));
    PetscCall(PetscObjectStateGet((PetscObject)ksp->vec_sol,&ostate));
    PetscCall(KSPGuessFormGuess(ksp->guess,ksp->vec_rhs,ksp->vec_sol));
    PetscCall(PetscObjectStateGet((PetscObject)ksp->vec_sol,&state));
    if (state != ostate) {
      ksp->guess_zero = PETSC_FALSE;
    } else {
      PetscCall(PetscInfo(ksp,"Using zero initial guess since the KSPGuess object did not change the vector\n"));
      ksp->guess_zero = PETSC_TRUE;
    }
  }

  /* KSPSetUp() scales the matrix if needed */
  PetscCall(KSPSetUp(ksp));
  PetscCall(KSPSetUpOnBlocks(ksp));

  PetscCall(VecSetErrorIfLocked(ksp->vec_sol,3));

  PetscCall(PetscLogEventBegin(KSP_Solve,ksp,ksp->vec_rhs,ksp->vec_sol,0));
  PetscCall(PCGetOperators(ksp->pc,&mat,&pmat));
  /* diagonal scale RHS if called for */
  if (ksp->dscale) {
    PetscCall(VecPointwiseMult(ksp->vec_rhs,ksp->vec_rhs,ksp->diagonal));
    /* second time in, but matrix was scaled back to original */
    if (ksp->dscalefix && ksp->dscalefix2) {
      Mat mat,pmat;

      PetscCall(PCGetOperators(ksp->pc,&mat,&pmat));
      PetscCall(MatDiagonalScale(pmat,ksp->diagonal,ksp->diagonal));
      if (mat != pmat) PetscCall(MatDiagonalScale(mat,ksp->diagonal,ksp->diagonal));
    }

    /* scale initial guess */
    if (!ksp->guess_zero) {
      if (!ksp->truediagonal) {
        PetscCall(VecDuplicate(ksp->diagonal,&ksp->truediagonal));
        PetscCall(VecCopy(ksp->diagonal,ksp->truediagonal));
        PetscCall(VecReciprocal(ksp->truediagonal));
      }
      PetscCall(VecPointwiseMult(ksp->vec_sol,ksp->vec_sol,ksp->truediagonal));
    }
  }
  PetscCall(PCPreSolve(ksp->pc,ksp));

  if (ksp->guess_zero) PetscCall(VecSet(ksp->vec_sol,0.0));
  if (ksp->guess_knoll) { /* The Knoll trick is independent on the KSPGuess specified */
    PetscCall(PCApply(ksp->pc,ksp->vec_rhs,ksp->vec_sol));
    PetscCall(KSP_RemoveNullSpace(ksp,ksp->vec_sol));
    ksp->guess_zero = PETSC_FALSE;
  }

  /* can we mark the initial guess as zero for this solve? */
  guess_zero = ksp->guess_zero;
  if (!ksp->guess_zero) {
    PetscReal norm;

    PetscCall(VecNormAvailable(ksp->vec_sol,NORM_2,&flg,&norm));
    if (flg && !norm) ksp->guess_zero = PETSC_TRUE;
  }
  if (ksp->transpose_solve) {
    PetscCall(MatGetNullSpace(pmat,&nullsp));
  } else {
    PetscCall(MatGetTransposeNullSpace(pmat,&nullsp));
  }
  if (nullsp) {
    PetscCall(VecDuplicate(ksp->vec_rhs,&btmp));
    PetscCall(VecCopy(ksp->vec_rhs,btmp));
    PetscCall(MatNullSpaceRemove(nullsp,btmp));
    vec_rhs      = ksp->vec_rhs;
    ksp->vec_rhs = btmp;
  }
  PetscCall(VecLockReadPush(ksp->vec_rhs));
  PetscCall((*ksp->ops->solve)(ksp));
  PetscCall(KSPMonitorPauseFinal_Internal(ksp));

  PetscCall(VecLockReadPop(ksp->vec_rhs));
  if (nullsp) {
    ksp->vec_rhs = vec_rhs;
    PetscCall(VecDestroy(&btmp));
  }

  ksp->guess_zero = guess_zero;

  PetscCheck(ksp->reason,comm,PETSC_ERR_PLIB,"Internal error, solver returned without setting converged reason");
  ksp->totalits += ksp->its;

  PetscCall(KSPConvergedReasonViewFromOptions(ksp));

  if (ksp->viewRate) {
    PetscCall(PetscViewerPushFormat(ksp->viewerRate,ksp->formatRate));
    PetscCall(KSPConvergedRateView(ksp, ksp->viewerRate));
    PetscCall(PetscViewerPopFormat(ksp->viewerRate));
  }
  PetscCall(PCPostSolve(ksp->pc,ksp));

  /* diagonal scale solution if called for */
  if (ksp->dscale) {
    PetscCall(VecPointwiseMult(ksp->vec_sol,ksp->vec_sol,ksp->diagonal));
    /* unscale right hand side and matrix */
    if (ksp->dscalefix) {
      Mat mat,pmat;

      PetscCall(VecReciprocal(ksp->diagonal));
      PetscCall(VecPointwiseMult(ksp->vec_rhs,ksp->vec_rhs,ksp->diagonal));
      PetscCall(PCGetOperators(ksp->pc,&mat,&pmat));
      PetscCall(MatDiagonalScale(pmat,ksp->diagonal,ksp->diagonal));
      if (mat != pmat) PetscCall(MatDiagonalScale(mat,ksp->diagonal,ksp->diagonal));
      PetscCall(VecReciprocal(ksp->diagonal));
      ksp->dscalefix2 = PETSC_TRUE;
    }
  }
  PetscCall(PetscLogEventEnd(KSP_Solve,ksp,ksp->vec_rhs,ksp->vec_sol,0));
  if (ksp->guess) {
    PetscCall(KSPGuessUpdate(ksp->guess,ksp->vec_rhs,ksp->vec_sol));
  }
  if (ksp->postsolve) {
    PetscCall((*ksp->postsolve)(ksp,ksp->vec_rhs,ksp->vec_sol,ksp->postctx));
  }

  PetscCall(PCGetOperators(ksp->pc,&mat,&pmat));
  if (ksp->viewEV)       PetscCall(KSPViewEigenvalues_Internal(ksp, PETSC_FALSE, ksp->viewerEV,    ksp->formatEV));
  if (ksp->viewEVExp)    PetscCall(KSPViewEigenvalues_Internal(ksp, PETSC_TRUE,  ksp->viewerEVExp, ksp->formatEVExp));
  if (ksp->viewSV)       PetscCall(KSPViewSingularvalues_Internal(ksp, ksp->viewerSV, ksp->formatSV));
  if (ksp->viewFinalRes) PetscCall(KSPViewFinalResidual_Internal(ksp, ksp->viewerFinalRes, ksp->formatFinalRes));
  if (ksp->viewMat)      PetscCall(ObjectView((PetscObject) mat,           ksp->viewerMat,    ksp->formatMat));
  if (ksp->viewPMat)     PetscCall(ObjectView((PetscObject) pmat,          ksp->viewerPMat,   ksp->formatPMat));
  if (ksp->viewRhs)      PetscCall(ObjectView((PetscObject) ksp->vec_rhs,  ksp->viewerRhs,    ksp->formatRhs));
  if (ksp->viewSol)      PetscCall(ObjectView((PetscObject) ksp->vec_sol,  ksp->viewerSol,    ksp->formatSol));
  if (ksp->view)         PetscCall(ObjectView((PetscObject) ksp,           ksp->viewer,       ksp->format));
  if (ksp->viewDScale)   PetscCall(ObjectView((PetscObject) ksp->diagonal, ksp->viewerDScale, ksp->formatDScale));
  if (ksp->viewMatExp)   {
    Mat A, B;

    PetscCall(PCGetOperators(ksp->pc, &A, NULL));
    if (ksp->transpose_solve) {
      Mat AT;

      PetscCall(MatCreateTranspose(A, &AT));
      PetscCall(MatComputeOperator(AT, MATAIJ, &B));
      PetscCall(MatDestroy(&AT));
    } else {
      PetscCall(MatComputeOperator(A, MATAIJ, &B));
    }
    PetscCall(ObjectView((PetscObject) B, ksp->viewerMatExp, ksp->formatMatExp));
    PetscCall(MatDestroy(&B));
  }
  if (ksp->viewPOpExp)   {
    Mat B;

    PetscCall(KSPComputeOperator(ksp, MATAIJ, &B));
    PetscCall(ObjectView((PetscObject) B, ksp->viewerPOpExp, ksp->formatPOpExp));
    PetscCall(MatDestroy(&B));
  }

  if (inXisinB) {
    PetscCall(VecCopy(x,b));
    PetscCall(VecDestroy(&x));
  }
  PetscCall(PetscObjectSAWsBlock((PetscObject)ksp));
  if (ksp->errorifnotconverged && ksp->reason < 0 && ((level == 1) || (ksp->reason != KSP_DIVERGED_ITS))) {
    if (ksp->reason == KSP_DIVERGED_PC_FAILED) {
      PCFailedReason reason;
      PetscCall(PCGetFailedReason(ksp->pc,&reason));
      SETERRQ(comm,PETSC_ERR_NOT_CONVERGED,"KSPSolve has not converged, reason %s PC failed due to %s",KSPConvergedReasons[ksp->reason],PCFailedReasons[reason]);
    } else SETERRQ(comm,PETSC_ERR_NOT_CONVERGED,"KSPSolve has not converged, reason %s",KSPConvergedReasons[ksp->reason]);
  }
  level--;
  PetscFunctionReturn(0);
}

/*@
   KSPSolve - Solves linear system.

   Collective on ksp

   Parameters:
+  ksp - iterative context obtained from KSPCreate()
.  b - the right hand side vector
-  x - the solution (this may be the same vector as b, then b will be overwritten with answer)

   Options Database Keys:
+  -ksp_view_eigenvalues - compute preconditioned operators eigenvalues
.  -ksp_view_eigenvalues_explicit - compute the eigenvalues by forming the dense operator and using LAPACK
.  -ksp_view_mat binary - save matrix to the default binary viewer
.  -ksp_view_pmat binary - save matrix used to build preconditioner to the default binary viewer
.  -ksp_view_rhs binary - save right hand side vector to the default binary viewer
.  -ksp_view_solution binary - save computed solution vector to the default binary viewer
           (can be read later with src/ksp/tutorials/ex10.c for testing solvers)
.  -ksp_view_mat_explicit - for matrix-free operators, computes the matrix entries and views them
.  -ksp_view_preconditioned_operator_explicit - computes the product of the preconditioner and matrix as an explicit matrix and views it
.  -ksp_converged_reason - print reason for converged or diverged, also prints number of iterations
.  -ksp_view_final_residual - print 2-norm of true linear system residual at the end of the solution process
.  -ksp_error_if_not_converged - stop the program as soon as an error is detected in a KSPSolve()
-  -ksp_view - print the ksp data structure at the end of the system solution

   Notes:

   If one uses KSPSetDM() then x or b need not be passed. Use KSPGetSolution() to access the solution in this case.

   The operator is specified with KSPSetOperators().

   KSPSolve() will normally return without generating an error regardless of whether the linear system was solved or if constructing the preconditioner failed.
   Call KSPGetConvergedReason() to determine if the solver converged or failed and why. The option -ksp_error_if_not_converged or function KSPSetErrorIfNotConverged()
   will cause KSPSolve() to error as soon as an error occurs in the linear solver.  In inner KSPSolves() KSP_DIVERGED_ITS is not treated as an error because when using nested solvers
   it may be fine that inner solvers in the preconditioner do not converge during the solution process.

   The number of iterations can be obtained from KSPGetIterationNumber().

   If you provide a matrix that has a MatSetNullSpace() and MatSetTransposeNullSpace() this will use that information to solve singular systems
   in the least squares sense with a norm minimizing solution.
$
$                   A x = b   where b = b_p + b_t where b_t is not in the range of A (and hence by the fundamental theorem of linear algebra is in the nullspace(A') see MatSetNullSpace()
$
$    KSP first removes b_t producing the linear system  A x = b_p (which has multiple solutions) and solves this to find the ||x|| minimizing solution (and hence
$    it finds the solution x orthogonal to the nullspace(A). The algorithm is simply in each iteration of the Krylov method we remove the nullspace(A) from the search
$    direction thus the solution which is a linear combination of the search directions has no component in the nullspace(A).
$
$    We recommend always using GMRES for such singular systems.
$    If nullspace(A) = nullspace(A') (note symmetric matrices always satisfy this property) then both left and right preconditioning will work
$    If nullspace(A) != nullspace(A') then left preconditioning will work but right preconditioning may not work (or it may).

   Developer Note: The reason we cannot always solve  nullspace(A) != nullspace(A') systems with right preconditioning is because we need to remove at each iteration
       the nullspace(AB) from the search direction. While we know the nullspace(A) the nullspace(AB) equals B^-1 times the nullspace(A) but except for trivial preconditioners
       such as diagonal scaling we cannot apply the inverse of the preconditioner to a vector and thus cannot compute the nullspace(AB).

   If using a direct method (e.g., via the KSP solver
   KSPPREONLY and a preconditioner such as PCLU/PCILU),
   then its=1.  See KSPSetTolerances() and KSPConvergedDefault()
   for more details.

   Understanding Convergence:
   The routines KSPMonitorSet(), KSPComputeEigenvalues(), and
   KSPComputeEigenvaluesExplicitly() provide information on additional
   options to monitor convergence and print eigenvalue information.

   Level: beginner

.seealso: KSPCreate(), KSPSetUp(), KSPDestroy(), KSPSetTolerances(), KSPConvergedDefault(),
          KSPSolveTranspose(), KSPGetIterationNumber(), MatNullSpaceCreate(), MatSetNullSpace(), MatSetTransposeNullSpace(), KSP,
          KSPConvergedReasonView(), KSPCheckSolve(), KSPSetErrorIfNotConverged()
@*/
PetscErrorCode KSPSolve(KSP ksp,Vec b,Vec x)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ksp,KSP_CLASSID,1);
  if (b) PetscValidHeaderSpecific(b,VEC_CLASSID,2);
  if (x) PetscValidHeaderSpecific(x,VEC_CLASSID,3);
  ksp->transpose_solve = PETSC_FALSE;
  PetscCall(KSPSolve_Private(ksp,b,x));
  PetscFunctionReturn(0);
}

/*@
   KSPSolveTranspose - Solves the transpose of a linear system.

   Collective on ksp

   Input Parameters:
+  ksp - iterative context obtained from KSPCreate()
.  b - right hand side vector
-  x - solution vector

   Notes:
    For complex numbers this solve the non-Hermitian transpose system.

   Developer Notes:
    We need to implement a KSPSolveHermitianTranspose()

   Level: developer

.seealso: KSPCreate(), KSPSetUp(), KSPDestroy(), KSPSetTolerances(), KSPConvergedDefault(),
          KSPSolve(), KSP
@*/
PetscErrorCode KSPSolveTranspose(KSP ksp,Vec b,Vec x)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ksp,KSP_CLASSID,1);
  if (b) PetscValidHeaderSpecific(b,VEC_CLASSID,2);
  if (x) PetscValidHeaderSpecific(x,VEC_CLASSID,3);
  if (ksp->transpose.use_explicittranspose) {
    Mat J,Jpre;
    PetscCall(KSPGetOperators(ksp,&J,&Jpre));
    if (!ksp->transpose.reuse_transpose) {
      PetscCall(MatTranspose(J,MAT_INITIAL_MATRIX,&ksp->transpose.AT));
      if (J != Jpre) {
        PetscCall(MatTranspose(Jpre,MAT_INITIAL_MATRIX,&ksp->transpose.BT));
      }
      ksp->transpose.reuse_transpose = PETSC_TRUE;
    } else {
      PetscCall(MatTranspose(J,MAT_REUSE_MATRIX,&ksp->transpose.AT));
      if (J != Jpre) {
        PetscCall(MatTranspose(Jpre,MAT_REUSE_MATRIX,&ksp->transpose.BT));
      }
    }
    if (J == Jpre && ksp->transpose.BT != ksp->transpose.AT) {
      PetscCall(PetscObjectReference((PetscObject)ksp->transpose.AT));
      ksp->transpose.BT = ksp->transpose.AT;
    }
    PetscCall(KSPSetOperators(ksp,ksp->transpose.AT,ksp->transpose.BT));
  } else {
    ksp->transpose_solve = PETSC_TRUE;
  }
  PetscCall(KSPSolve_Private(ksp,b,x));
  PetscFunctionReturn(0);
}

static PetscErrorCode KSPViewFinalMatResidual_Internal(KSP ksp, Mat B, Mat X, PetscViewer viewer, PetscViewerFormat format, PetscInt shift)
{
  Mat            A, R;
  PetscReal      *norms;
  PetscInt       i, N;
  PetscBool      flg;

  PetscFunctionBegin;
  PetscCall(PetscObjectTypeCompare((PetscObject)viewer, PETSCVIEWERASCII, &flg));
  if (flg) {
    PetscCall(PCGetOperators(ksp->pc, &A, NULL));
    PetscCall(MatMatMult(A, X, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &R));
    PetscCall(MatAYPX(R, -1.0, B, SAME_NONZERO_PATTERN));
    PetscCall(MatGetSize(R, NULL, &N));
    PetscCall(PetscMalloc1(N, &norms));
    PetscCall(MatGetColumnNorms(R, NORM_2, norms));
    PetscCall(MatDestroy(&R));
    for (i = 0; i < N; ++i) {
      PetscCall(PetscViewerASCIIPrintf(viewer, "%s #%D %g\n", i == 0 ? "KSP final norm of residual" : "                          ", shift + i, (double)norms[i]));
    }
    PetscCall(PetscFree(norms));
  }
  PetscFunctionReturn(0);
}

/*@
     KSPMatSolve - Solves a linear system with multiple right-hand sides stored as a MATDENSE. Unlike KSPSolve(), B and X must be different matrices.

   Input Parameters:
+     ksp - iterative context
-     B - block of right-hand sides

   Output Parameter:
.     X - block of solutions

   Notes:
     This is a stripped-down version of KSPSolve(), which only handles -ksp_view, -ksp_converged_reason, and -ksp_view_final_residual.

   Level: intermediate

.seealso:  KSPSolve(), MatMatSolve(), MATDENSE, KSPHPDDM, PCBJACOBI, PCASM
@*/
PetscErrorCode KSPMatSolve(KSP ksp, Mat B, Mat X)
{
  Mat            A, P, vB, vX;
  Vec            cb, cx;
  PetscInt       n1, N1, n2, N2, Bbn = PETSC_DECIDE;
  PetscBool      match;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ksp, KSP_CLASSID, 1);
  PetscValidHeaderSpecific(B, MAT_CLASSID, 2);
  PetscValidHeaderSpecific(X, MAT_CLASSID, 3);
  PetscCheckSameComm(ksp, 1, B, 2);
  PetscCheckSameComm(ksp, 1, X, 3);
  PetscCheck(B->assembled,PETSC_COMM_SELF, PETSC_ERR_ARG_WRONGSTATE, "Not for unassembled matrix");
  MatCheckPreallocated(X, 3);
  if (!X->assembled) {
    PetscCall(MatSetOption(X, MAT_NO_OFF_PROC_ENTRIES, PETSC_TRUE));
    PetscCall(MatAssemblyBegin(X, MAT_FINAL_ASSEMBLY));
    PetscCall(MatAssemblyEnd(X, MAT_FINAL_ASSEMBLY));
  }
  PetscCheck(B != X,PetscObjectComm((PetscObject)ksp), PETSC_ERR_ARG_IDN, "B and X must be different matrices");
  PetscCall(KSPGetOperators(ksp, &A, &P));
  PetscCall(MatGetLocalSize(B, NULL, &n2));
  PetscCall(MatGetLocalSize(X, NULL, &n1));
  PetscCall(MatGetSize(B, NULL, &N2));
  PetscCall(MatGetSize(X, NULL, &N1));
  PetscCheckFalse(n1 != n2 || N1 != N2,PETSC_COMM_SELF, PETSC_ERR_ARG_SIZ, "Incompatible number of columns between block of right-hand sides (n,N) = (%D,%D) and block of solutions (n,N) = (%D,%D)", n2, N2, n1, N1);
  PetscCall(PetscObjectBaseTypeCompareAny((PetscObject)B, &match, MATSEQDENSE, MATMPIDENSE, ""));
  PetscCheck(match,PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Provided block of right-hand sides not stored in a dense Mat");
  PetscCall(PetscObjectBaseTypeCompareAny((PetscObject)X, &match, MATSEQDENSE, MATMPIDENSE, ""));
  PetscCheck(match,PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Provided block of solutions not stored in a dense Mat");
  PetscCall(KSPSetUp(ksp));
  PetscCall(KSPSetUpOnBlocks(ksp));
  if (ksp->ops->matsolve) {
    if (ksp->guess_zero) {
      PetscCall(MatZeroEntries(X));
    }
    PetscCall(PetscLogEventBegin(KSP_MatSolve, ksp, B, X, 0));
    PetscCall(KSPGetMatSolveBatchSize(ksp, &Bbn));
    /* by default, do a single solve with all columns */
    if (Bbn == PETSC_DECIDE) Bbn = N2;
    else PetscCheck(Bbn >= 1,PetscObjectComm((PetscObject)ksp), PETSC_ERR_ARG_OUTOFRANGE, "KSPMatSolve() batch size %D must be positive", Bbn);
    PetscCall(PetscInfo(ksp, "KSP type %s solving using batches of width at most %D\n", ((PetscObject)ksp)->type_name, Bbn));
    /* if -ksp_matsolve_batch_size is greater than the actual number of columns, do a single solve with all columns */
    if (Bbn >= N2) {
      PetscCall((*ksp->ops->matsolve)(ksp, B, X));
      if (ksp->viewFinalRes) {
        PetscCall(KSPViewFinalMatResidual_Internal(ksp, B, X, ksp->viewerFinalRes, ksp->formatFinalRes, 0));
      }

      PetscCall(KSPConvergedReasonViewFromOptions(ksp));

      if (ksp->viewRate) {
        PetscCall(PetscViewerPushFormat(ksp->viewerRate,PETSC_VIEWER_DEFAULT));
        PetscCall(KSPConvergedRateView(ksp, ksp->viewerRate));
        PetscCall(PetscViewerPopFormat(ksp->viewerRate));
      }
    } else {
      for (n2 = 0; n2 < N2; n2 += Bbn) {
        PetscCall(MatDenseGetSubMatrix(B, n2, PetscMin(n2+Bbn, N2), &vB));
        PetscCall(MatDenseGetSubMatrix(X, n2, PetscMin(n2+Bbn, N2), &vX));
        PetscCall((*ksp->ops->matsolve)(ksp, vB, vX));
        if (ksp->viewFinalRes) {
          PetscCall(KSPViewFinalMatResidual_Internal(ksp, vB, vX, ksp->viewerFinalRes, ksp->formatFinalRes, n2));
        }

        PetscCall(KSPConvergedReasonViewFromOptions(ksp));

        if (ksp->viewRate) {
          PetscCall(PetscViewerPushFormat(ksp->viewerRate,PETSC_VIEWER_DEFAULT));
          PetscCall(KSPConvergedRateView(ksp, ksp->viewerRate));
          PetscCall(PetscViewerPopFormat(ksp->viewerRate));
        }
        PetscCall(MatDenseRestoreSubMatrix(B, &vB));
        PetscCall(MatDenseRestoreSubMatrix(X, &vX));
      }
    }
    if (ksp->viewMat)  PetscCall(ObjectView((PetscObject) A, ksp->viewerMat, ksp->formatMat));
    if (ksp->viewPMat) PetscCall(ObjectView((PetscObject) P, ksp->viewerPMat,ksp->formatPMat));
    if (ksp->viewRhs)  PetscCall(ObjectView((PetscObject) B, ksp->viewerRhs, ksp->formatRhs));
    if (ksp->viewSol)  PetscCall(ObjectView((PetscObject) X, ksp->viewerSol, ksp->formatSol));
    if (ksp->view) {
      PetscCall(KSPView(ksp, ksp->viewer));
    }
    PetscCall(PetscLogEventEnd(KSP_MatSolve, ksp, B, X, 0));
  } else {
    PetscCall(PetscInfo(ksp, "KSP type %s solving column by column\n", ((PetscObject)ksp)->type_name));
    for (n2 = 0; n2 < N2; ++n2) {
      PetscCall(MatDenseGetColumnVecRead(B, n2, &cb));
      PetscCall(MatDenseGetColumnVecWrite(X, n2, &cx));
      PetscCall(KSPSolve(ksp, cb, cx));
      PetscCall(MatDenseRestoreColumnVecWrite(X, n2, &cx));
      PetscCall(MatDenseRestoreColumnVecRead(B, n2, &cb));
    }
  }
  PetscFunctionReturn(0);
}

/*@
     KSPSetMatSolveBatchSize - Sets the maximum number of columns treated simultaneously in KSPMatSolve().

    Logically collective

   Input Parameters:
+     ksp - iterative context
-     bs - batch size

   Level: advanced

.seealso:  KSPMatSolve(), KSPGetMatSolveBatchSize(), -mat_mumps_icntl_27, -matmatmult_Bbn
@*/
PetscErrorCode KSPSetMatSolveBatchSize(KSP ksp, PetscInt bs)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ksp, KSP_CLASSID, 1);
  PetscValidLogicalCollectiveInt(ksp, bs, 2);
  ksp->nmax = bs;
  PetscFunctionReturn(0);
}

/*@
     KSPGetMatSolveBatchSize - Gets the maximum number of columns treated simultaneously in KSPMatSolve().

   Input Parameter:
.     ksp - iterative context

   Output Parameter:
.     bs - batch size

   Level: advanced

.seealso:  KSPMatSolve(), KSPSetMatSolveBatchSize(), -mat_mumps_icntl_27, -matmatmult_Bbn
@*/
PetscErrorCode KSPGetMatSolveBatchSize(KSP ksp, PetscInt *bs)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ksp, KSP_CLASSID, 1);
  PetscValidIntPointer(bs, 2);
  *bs = ksp->nmax;
  PetscFunctionReturn(0);
}

/*@
   KSPResetViewers - Resets all the viewers set from the options database during KSPSetFromOptions()

   Collective on ksp

   Input Parameter:
.  ksp - iterative context obtained from KSPCreate()

   Level: beginner

.seealso: KSPCreate(), KSPSetUp(), KSPSolve(), KSPSetFromOptions(), KSP
@*/
PetscErrorCode  KSPResetViewers(KSP ksp)
{
  PetscFunctionBegin;
  if (ksp) PetscValidHeaderSpecific(ksp,KSP_CLASSID,1);
  if (!ksp) PetscFunctionReturn(0);
  PetscCall(PetscViewerDestroy(&ksp->viewer));
  PetscCall(PetscViewerDestroy(&ksp->viewerPre));
  PetscCall(PetscViewerDestroy(&ksp->viewerRate));
  PetscCall(PetscViewerDestroy(&ksp->viewerMat));
  PetscCall(PetscViewerDestroy(&ksp->viewerPMat));
  PetscCall(PetscViewerDestroy(&ksp->viewerRhs));
  PetscCall(PetscViewerDestroy(&ksp->viewerSol));
  PetscCall(PetscViewerDestroy(&ksp->viewerMatExp));
  PetscCall(PetscViewerDestroy(&ksp->viewerEV));
  PetscCall(PetscViewerDestroy(&ksp->viewerSV));
  PetscCall(PetscViewerDestroy(&ksp->viewerEVExp));
  PetscCall(PetscViewerDestroy(&ksp->viewerFinalRes));
  PetscCall(PetscViewerDestroy(&ksp->viewerPOpExp));
  PetscCall(PetscViewerDestroy(&ksp->viewerDScale));
  ksp->view         = PETSC_FALSE;
  ksp->viewPre      = PETSC_FALSE;
  ksp->viewMat      = PETSC_FALSE;
  ksp->viewPMat     = PETSC_FALSE;
  ksp->viewRhs      = PETSC_FALSE;
  ksp->viewSol      = PETSC_FALSE;
  ksp->viewMatExp   = PETSC_FALSE;
  ksp->viewEV       = PETSC_FALSE;
  ksp->viewSV       = PETSC_FALSE;
  ksp->viewEVExp    = PETSC_FALSE;
  ksp->viewFinalRes = PETSC_FALSE;
  ksp->viewPOpExp   = PETSC_FALSE;
  ksp->viewDScale   = PETSC_FALSE;
  PetscFunctionReturn(0);
}

/*@
   KSPReset - Resets a KSP context to the kspsetupcalled = 0 state and removes any allocated Vecs and Mats

   Collective on ksp

   Input Parameter:
.  ksp - iterative context obtained from KSPCreate()

   Level: beginner

.seealso: KSPCreate(), KSPSetUp(), KSPSolve(), KSP
@*/
PetscErrorCode  KSPReset(KSP ksp)
{
  PetscFunctionBegin;
  if (ksp) PetscValidHeaderSpecific(ksp,KSP_CLASSID,1);
  if (!ksp) PetscFunctionReturn(0);
  if (ksp->ops->reset) {
    PetscCall((*ksp->ops->reset)(ksp));
  }
  if (ksp->pc) PetscCall(PCReset(ksp->pc));
  if (ksp->guess) {
    KSPGuess guess = ksp->guess;
    if (guess->ops->reset) PetscCall((*guess->ops->reset)(guess));
  }
  PetscCall(VecDestroyVecs(ksp->nwork,&ksp->work));
  PetscCall(VecDestroy(&ksp->vec_rhs));
  PetscCall(VecDestroy(&ksp->vec_sol));
  PetscCall(VecDestroy(&ksp->diagonal));
  PetscCall(VecDestroy(&ksp->truediagonal));

  PetscCall(KSPResetViewers(ksp));

  ksp->setupstage = KSP_SETUP_NEW;
  ksp->nmax = PETSC_DECIDE;
  PetscFunctionReturn(0);
}

/*@C
   KSPDestroy - Destroys KSP context.

   Collective on ksp

   Input Parameter:
.  ksp - iterative context obtained from KSPCreate()

   Level: beginner

.seealso: KSPCreate(), KSPSetUp(), KSPSolve(), KSP
@*/
PetscErrorCode  KSPDestroy(KSP *ksp)
{
  PC             pc;

  PetscFunctionBegin;
  if (!*ksp) PetscFunctionReturn(0);
  PetscValidHeaderSpecific((*ksp),KSP_CLASSID,1);
  if (--((PetscObject)(*ksp))->refct > 0) {*ksp = NULL; PetscFunctionReturn(0);}

  PetscCall(PetscObjectSAWsViewOff((PetscObject)*ksp));

  /*
   Avoid a cascading call to PCReset(ksp->pc) from the following call:
   PCReset() shouldn't be called from KSPDestroy() as it is unprotected by pc's
   refcount (and may be shared, e.g., by other ksps).
   */
  pc         = (*ksp)->pc;
  (*ksp)->pc = NULL;
  PetscCall(KSPReset((*ksp)));
  (*ksp)->pc = pc;
  if ((*ksp)->ops->destroy) PetscCall((*(*ksp)->ops->destroy)(*ksp));

  if ((*ksp)->transpose.use_explicittranspose) {
    PetscCall(MatDestroy(&(*ksp)->transpose.AT));
    PetscCall(MatDestroy(&(*ksp)->transpose.BT));
    (*ksp)->transpose.reuse_transpose = PETSC_FALSE;
  }

  PetscCall(KSPGuessDestroy(&(*ksp)->guess));
  PetscCall(DMDestroy(&(*ksp)->dm));
  PetscCall(PCDestroy(&(*ksp)->pc));
  PetscCall(PetscFree((*ksp)->res_hist_alloc));
  PetscCall(PetscFree((*ksp)->err_hist_alloc));
  if ((*ksp)->convergeddestroy) {
    PetscCall((*(*ksp)->convergeddestroy)((*ksp)->cnvP));
  }
  PetscCall(KSPMonitorCancel((*ksp)));
  PetscCall(KSPConvergedReasonViewCancel((*ksp)));
  PetscCall(PetscViewerDestroy(&(*ksp)->eigviewer));
  PetscCall(PetscHeaderDestroy(ksp));
  PetscFunctionReturn(0);
}

/*@
    KSPSetPCSide - Sets the preconditioning side.

    Logically Collective on ksp

    Input Parameter:
.   ksp - iterative context obtained from KSPCreate()

    Output Parameter:
.   side - the preconditioning side, where side is one of
.vb
      PC_LEFT - left preconditioning (default)
      PC_RIGHT - right preconditioning
      PC_SYMMETRIC - symmetric preconditioning
.ve

    Options Database Keys:
.   -ksp_pc_side <right,left,symmetric> - KSP preconditioner side

    Notes:
    Left preconditioning is used by default for most Krylov methods except KSPFGMRES which only supports right preconditioning.

    For methods changing the side of the preconditioner changes the norm type that is used, see KSPSetNormType().

    Symmetric preconditioning is currently available only for the KSPQCG method. Note, however, that
    symmetric preconditioning can be emulated by using either right or left
    preconditioning and a pre or post processing step.

    Setting the PC side often affects the default norm type.  See KSPSetNormType() for details.

    Level: intermediate

.seealso: KSPGetPCSide(), KSPSetNormType(), KSPGetNormType(), KSP
@*/
PetscErrorCode  KSPSetPCSide(KSP ksp,PCSide side)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ksp,KSP_CLASSID,1);
  PetscValidLogicalCollectiveEnum(ksp,side,2);
  ksp->pc_side = ksp->pc_side_set = side;
  PetscFunctionReturn(0);
}

/*@
    KSPGetPCSide - Gets the preconditioning side.

    Not Collective

    Input Parameter:
.   ksp - iterative context obtained from KSPCreate()

    Output Parameter:
.   side - the preconditioning side, where side is one of
.vb
      PC_LEFT - left preconditioning (default)
      PC_RIGHT - right preconditioning
      PC_SYMMETRIC - symmetric preconditioning
.ve

    Level: intermediate

.seealso: KSPSetPCSide(), KSP
@*/
PetscErrorCode  KSPGetPCSide(KSP ksp,PCSide *side)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ksp,KSP_CLASSID,1);
  PetscValidPointer(side,2);
  PetscCall(KSPSetUpNorms_Private(ksp,PETSC_TRUE,&ksp->normtype,&ksp->pc_side));
  *side = ksp->pc_side;
  PetscFunctionReturn(0);
}

/*@
   KSPGetTolerances - Gets the relative, absolute, divergence, and maximum
   iteration tolerances used by the default KSP convergence tests.

   Not Collective

   Input Parameter:
.  ksp - the Krylov subspace context

   Output Parameters:
+  rtol - the relative convergence tolerance
.  abstol - the absolute convergence tolerance
.  dtol - the divergence tolerance
-  maxits - maximum number of iterations

   Notes:
   The user can specify NULL for any parameter that is not needed.

   Level: intermediate

           maximum, iterations

.seealso: KSPSetTolerances(), KSP
@*/
PetscErrorCode  KSPGetTolerances(KSP ksp,PetscReal *rtol,PetscReal *abstol,PetscReal *dtol,PetscInt *maxits)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ksp,KSP_CLASSID,1);
  if (abstol) *abstol = ksp->abstol;
  if (rtol) *rtol = ksp->rtol;
  if (dtol) *dtol = ksp->divtol;
  if (maxits) *maxits = ksp->max_it;
  PetscFunctionReturn(0);
}

/*@
   KSPSetTolerances - Sets the relative, absolute, divergence, and maximum
   iteration tolerances used by the default KSP convergence testers.

   Logically Collective on ksp

   Input Parameters:
+  ksp - the Krylov subspace context
.  rtol - the relative convergence tolerance, relative decrease in the (possibly preconditioned) residual norm
.  abstol - the absolute convergence tolerance   absolute size of the (possibly preconditioned) residual norm
.  dtol - the divergence tolerance,   amount (possibly preconditioned) residual norm can increase before KSPConvergedDefault() concludes that the method is diverging
-  maxits - maximum number of iterations to use

   Options Database Keys:
+  -ksp_atol <abstol> - Sets abstol
.  -ksp_rtol <rtol> - Sets rtol
.  -ksp_divtol <dtol> - Sets dtol
-  -ksp_max_it <maxits> - Sets maxits

   Notes:
   Use PETSC_DEFAULT to retain the default value of any of the tolerances.

   See KSPConvergedDefault() for details how these parameters are used in the default convergence test.  See also KSPSetConvergenceTest()
   for setting user-defined stopping criteria.

   Level: intermediate

           convergence, maximum, iterations

.seealso: KSPGetTolerances(), KSPConvergedDefault(), KSPSetConvergenceTest(), KSP
@*/
PetscErrorCode  KSPSetTolerances(KSP ksp,PetscReal rtol,PetscReal abstol,PetscReal dtol,PetscInt maxits)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ksp,KSP_CLASSID,1);
  PetscValidLogicalCollectiveReal(ksp,rtol,2);
  PetscValidLogicalCollectiveReal(ksp,abstol,3);
  PetscValidLogicalCollectiveReal(ksp,dtol,4);
  PetscValidLogicalCollectiveInt(ksp,maxits,5);

  if (rtol != PETSC_DEFAULT) {
    PetscCheckFalse(rtol < 0.0 || 1.0 <= rtol,PetscObjectComm((PetscObject)ksp),PETSC_ERR_ARG_OUTOFRANGE,"Relative tolerance %g must be non-negative and less than 1.0",(double)rtol);
    ksp->rtol = rtol;
  }
  if (abstol != PETSC_DEFAULT) {
    PetscCheck(abstol >= 0.0,PetscObjectComm((PetscObject)ksp),PETSC_ERR_ARG_OUTOFRANGE,"Absolute tolerance %g must be non-negative",(double)abstol);
    ksp->abstol = abstol;
  }
  if (dtol != PETSC_DEFAULT) {
    PetscCheck(dtol >= 0.0,PetscObjectComm((PetscObject)ksp),PETSC_ERR_ARG_OUTOFRANGE,"Divergence tolerance %g must be larger than 1.0",(double)dtol);
    ksp->divtol = dtol;
  }
  if (maxits != PETSC_DEFAULT) {
    PetscCheck(maxits >= 0,PetscObjectComm((PetscObject)ksp),PETSC_ERR_ARG_OUTOFRANGE,"Maximum number of iterations %D must be non-negative",maxits);
    ksp->max_it = maxits;
  }
  PetscFunctionReturn(0);
}

/*@
   KSPSetInitialGuessNonzero - Tells the iterative solver that the
   initial guess is nonzero; otherwise KSP assumes the initial guess
   is to be zero (and thus zeros it out before solving).

   Logically Collective on ksp

   Input Parameters:
+  ksp - iterative context obtained from KSPCreate()
-  flg - PETSC_TRUE indicates the guess is non-zero, PETSC_FALSE indicates the guess is zero

   Options database keys:
.  -ksp_initial_guess_nonzero <true,false> - use nonzero initial guess

   Level: beginner

   Notes:
    If this is not called the X vector is zeroed in the call to KSPSolve().

.seealso: KSPGetInitialGuessNonzero(), KSPSetGuessType(), KSPGuessType, KSP
@*/
PetscErrorCode  KSPSetInitialGuessNonzero(KSP ksp,PetscBool flg)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ksp,KSP_CLASSID,1);
  PetscValidLogicalCollectiveBool(ksp,flg,2);
  ksp->guess_zero = (PetscBool) !(int)flg;
  PetscFunctionReturn(0);
}

/*@
   KSPGetInitialGuessNonzero - Determines whether the KSP solver is using
   a zero initial guess.

   Not Collective

   Input Parameter:
.  ksp - iterative context obtained from KSPCreate()

   Output Parameter:
.  flag - PETSC_TRUE if guess is nonzero, else PETSC_FALSE

   Level: intermediate

.seealso: KSPSetInitialGuessNonzero(), KSP
@*/
PetscErrorCode  KSPGetInitialGuessNonzero(KSP ksp,PetscBool  *flag)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ksp,KSP_CLASSID,1);
  PetscValidBoolPointer(flag,2);
  if (ksp->guess_zero) *flag = PETSC_FALSE;
  else *flag = PETSC_TRUE;
  PetscFunctionReturn(0);
}

/*@
   KSPSetErrorIfNotConverged - Causes KSPSolve() to generate an error if the solver has not converged.

   Logically Collective on ksp

   Input Parameters:
+  ksp - iterative context obtained from KSPCreate()
-  flg - PETSC_TRUE indicates you want the error generated

   Options database keys:
.  -ksp_error_if_not_converged <true,false> - generate an error and stop the program

   Level: intermediate

   Notes:
    Normally PETSc continues if a linear solver fails to converge, you can call KSPGetConvergedReason() after a KSPSolve()
    to determine if it has converged.

   A KSP_DIVERGED_ITS will not generate an error in a KSPSolve() inside a nested linear solver

.seealso: KSPGetErrorIfNotConverged(), KSP
@*/
PetscErrorCode  KSPSetErrorIfNotConverged(KSP ksp,PetscBool flg)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ksp,KSP_CLASSID,1);
  PetscValidLogicalCollectiveBool(ksp,flg,2);
  ksp->errorifnotconverged = flg;
  PetscFunctionReturn(0);
}

/*@
   KSPGetErrorIfNotConverged - Will KSPSolve() generate an error if the solver does not converge?

   Not Collective

   Input Parameter:
.  ksp - iterative context obtained from KSPCreate()

   Output Parameter:
.  flag - PETSC_TRUE if it will generate an error, else PETSC_FALSE

   Level: intermediate

.seealso: KSPSetErrorIfNotConverged(), KSP
@*/
PetscErrorCode  KSPGetErrorIfNotConverged(KSP ksp,PetscBool  *flag)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ksp,KSP_CLASSID,1);
  PetscValidBoolPointer(flag,2);
  *flag = ksp->errorifnotconverged;
  PetscFunctionReturn(0);
}

/*@
   KSPSetInitialGuessKnoll - Tells the iterative solver to use PCApply(pc,b,..) to compute the initial guess (The Knoll trick)

   Logically Collective on ksp

   Input Parameters:
+  ksp - iterative context obtained from KSPCreate()
-  flg - PETSC_TRUE or PETSC_FALSE

   Level: advanced

   Developer Note: the Knoll trick is not currently implemented using the KSPGuess class

.seealso: KSPGetInitialGuessKnoll(), KSPSetInitialGuessNonzero(), KSPGetInitialGuessNonzero(), KSP
@*/
PetscErrorCode  KSPSetInitialGuessKnoll(KSP ksp,PetscBool flg)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ksp,KSP_CLASSID,1);
  PetscValidLogicalCollectiveBool(ksp,flg,2);
  ksp->guess_knoll = flg;
  PetscFunctionReturn(0);
}

/*@
   KSPGetInitialGuessKnoll - Determines whether the KSP solver is using the Knoll trick (using PCApply(pc,b,...) to compute
     the initial guess

   Not Collective

   Input Parameter:
.  ksp - iterative context obtained from KSPCreate()

   Output Parameter:
.  flag - PETSC_TRUE if using Knoll trick, else PETSC_FALSE

   Level: advanced

.seealso: KSPSetInitialGuessKnoll(), KSPSetInitialGuessNonzero(), KSPGetInitialGuessNonzero(), KSP
@*/
PetscErrorCode  KSPGetInitialGuessKnoll(KSP ksp,PetscBool  *flag)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ksp,KSP_CLASSID,1);
  PetscValidBoolPointer(flag,2);
  *flag = ksp->guess_knoll;
  PetscFunctionReturn(0);
}

/*@
   KSPGetComputeSingularValues - Gets the flag indicating whether the extreme singular
   values will be calculated via a Lanczos or Arnoldi process as the linear
   system is solved.

   Not Collective

   Input Parameter:
.  ksp - iterative context obtained from KSPCreate()

   Output Parameter:
.  flg - PETSC_TRUE or PETSC_FALSE

   Options Database Key:
.  -ksp_monitor_singular_value - Activates KSPSetComputeSingularValues()

   Notes:
   Currently this option is not valid for all iterative methods.

   Many users may just want to use the monitoring routine
   KSPMonitorSingularValue() (which can be set with option -ksp_monitor_singular_value)
   to print the singular values at each iteration of the linear solve.

   Level: advanced

.seealso: KSPComputeExtremeSingularValues(), KSPMonitorSingularValue(), KSP
@*/
PetscErrorCode  KSPGetComputeSingularValues(KSP ksp,PetscBool  *flg)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ksp,KSP_CLASSID,1);
  PetscValidBoolPointer(flg,2);
  *flg = ksp->calc_sings;
  PetscFunctionReturn(0);
}

/*@
   KSPSetComputeSingularValues - Sets a flag so that the extreme singular
   values will be calculated via a Lanczos or Arnoldi process as the linear
   system is solved.

   Logically Collective on ksp

   Input Parameters:
+  ksp - iterative context obtained from KSPCreate()
-  flg - PETSC_TRUE or PETSC_FALSE

   Options Database Key:
.  -ksp_monitor_singular_value - Activates KSPSetComputeSingularValues()

   Notes:
   Currently this option is not valid for all iterative methods.

   Many users may just want to use the monitoring routine
   KSPMonitorSingularValue() (which can be set with option -ksp_monitor_singular_value)
   to print the singular values at each iteration of the linear solve.

   Level: advanced

.seealso: KSPComputeExtremeSingularValues(), KSPMonitorSingularValue(), KSP
@*/
PetscErrorCode  KSPSetComputeSingularValues(KSP ksp,PetscBool flg)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ksp,KSP_CLASSID,1);
  PetscValidLogicalCollectiveBool(ksp,flg,2);
  ksp->calc_sings = flg;
  PetscFunctionReturn(0);
}

/*@
   KSPGetComputeEigenvalues - Gets the flag indicating that the extreme eigenvalues
   values will be calculated via a Lanczos or Arnoldi process as the linear
   system is solved.

   Not Collective

   Input Parameter:
.  ksp - iterative context obtained from KSPCreate()

   Output Parameter:
.  flg - PETSC_TRUE or PETSC_FALSE

   Notes:
   Currently this option is not valid for all iterative methods.

   Level: advanced

.seealso: KSPComputeEigenvalues(), KSPComputeEigenvaluesExplicitly(), KSP
@*/
PetscErrorCode  KSPGetComputeEigenvalues(KSP ksp,PetscBool  *flg)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ksp,KSP_CLASSID,1);
  PetscValidBoolPointer(flg,2);
  *flg = ksp->calc_sings;
  PetscFunctionReturn(0);
}

/*@
   KSPSetComputeEigenvalues - Sets a flag so that the extreme eigenvalues
   values will be calculated via a Lanczos or Arnoldi process as the linear
   system is solved.

   Logically Collective on ksp

   Input Parameters:
+  ksp - iterative context obtained from KSPCreate()
-  flg - PETSC_TRUE or PETSC_FALSE

   Notes:
   Currently this option is not valid for all iterative methods.

   Level: advanced

.seealso: KSPComputeEigenvalues(), KSPComputeEigenvaluesExplicitly(), KSP
@*/
PetscErrorCode  KSPSetComputeEigenvalues(KSP ksp,PetscBool flg)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ksp,KSP_CLASSID,1);
  PetscValidLogicalCollectiveBool(ksp,flg,2);
  ksp->calc_sings = flg;
  PetscFunctionReturn(0);
}

/*@
   KSPSetComputeRitz - Sets a flag so that the Ritz or harmonic Ritz pairs
   will be calculated via a Lanczos or Arnoldi process as the linear
   system is solved.

   Logically Collective on ksp

   Input Parameters:
+  ksp - iterative context obtained from KSPCreate()
-  flg - PETSC_TRUE or PETSC_FALSE

   Notes:
   Currently this option is only valid for the GMRES method.

   Level: advanced

.seealso: KSPComputeRitz(), KSP
@*/
PetscErrorCode  KSPSetComputeRitz(KSP ksp, PetscBool flg)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ksp,KSP_CLASSID,1);
  PetscValidLogicalCollectiveBool(ksp,flg,2);
  ksp->calc_ritz = flg;
  PetscFunctionReturn(0);
}

/*@
   KSPGetRhs - Gets the right-hand-side vector for the linear system to
   be solved.

   Not Collective

   Input Parameter:
.  ksp - iterative context obtained from KSPCreate()

   Output Parameter:
.  r - right-hand-side vector

   Level: developer

.seealso: KSPGetSolution(), KSPSolve(), KSP
@*/
PetscErrorCode  KSPGetRhs(KSP ksp,Vec *r)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ksp,KSP_CLASSID,1);
  PetscValidPointer(r,2);
  *r = ksp->vec_rhs;
  PetscFunctionReturn(0);
}

/*@
   KSPGetSolution - Gets the location of the solution for the
   linear system to be solved.  Note that this may not be where the solution
   is stored during the iterative process; see KSPBuildSolution().

   Not Collective

   Input Parameters:
.  ksp - iterative context obtained from KSPCreate()

   Output Parameters:
.  v - solution vector

   Level: developer

.seealso: KSPGetRhs(),  KSPBuildSolution(), KSPSolve(), KSP
@*/
PetscErrorCode  KSPGetSolution(KSP ksp,Vec *v)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ksp,KSP_CLASSID,1);
  PetscValidPointer(v,2);
  *v = ksp->vec_sol;
  PetscFunctionReturn(0);
}

/*@
   KSPSetPC - Sets the preconditioner to be used to calculate the
   application of the preconditioner on a vector.

   Collective on ksp

   Input Parameters:
+  ksp - iterative context obtained from KSPCreate()
-  pc   - the preconditioner object (can be NULL)

   Notes:
   Use KSPGetPC() to retrieve the preconditioner context.

   Level: developer

.seealso: KSPGetPC(), KSP
@*/
PetscErrorCode  KSPSetPC(KSP ksp,PC pc)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ksp,KSP_CLASSID,1);
  if (pc) {
    PetscValidHeaderSpecific(pc,PC_CLASSID,2);
    PetscCheckSameComm(ksp,1,pc,2);
  }
  PetscCall(PetscObjectReference((PetscObject)pc));
  PetscCall(PCDestroy(&ksp->pc));
  ksp->pc = pc;
  PetscCall(PetscLogObjectParent((PetscObject)ksp,(PetscObject)ksp->pc));
  PetscFunctionReturn(0);
}

/*@
   KSPGetPC - Returns a pointer to the preconditioner context
   set with KSPSetPC().

   Not Collective

   Input Parameters:
.  ksp - iterative context obtained from KSPCreate()

   Output Parameter:
.  pc - preconditioner context

   Level: developer

.seealso: KSPSetPC(), KSP
@*/
PetscErrorCode  KSPGetPC(KSP ksp,PC *pc)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ksp,KSP_CLASSID,1);
  PetscValidPointer(pc,2);
  if (!ksp->pc) {
    PetscCall(PCCreate(PetscObjectComm((PetscObject)ksp),&ksp->pc));
    PetscCall(PetscObjectIncrementTabLevel((PetscObject)ksp->pc,(PetscObject)ksp,0));
    PetscCall(PetscLogObjectParent((PetscObject)ksp,(PetscObject)ksp->pc));
    PetscCall(PetscObjectSetOptions((PetscObject)ksp->pc,((PetscObject)ksp)->options));
  }
  *pc = ksp->pc;
  PetscFunctionReturn(0);
}

/*@
   KSPMonitor - runs the user provided monitor routines, if they exist

   Collective on ksp

   Input Parameters:
+  ksp - iterative context obtained from KSPCreate()
.  it - iteration number
-  rnorm - relative norm of the residual

   Notes:
   This routine is called by the KSP implementations.
   It does not typically need to be called by the user.

   Level: developer

.seealso: KSPMonitorSet()
@*/
PetscErrorCode KSPMonitor(KSP ksp,PetscInt it,PetscReal rnorm)
{
  PetscInt       i, n = ksp->numbermonitors;

  PetscFunctionBegin;
  for (i=0; i<n; i++) {
    PetscCall((*ksp->monitor[i])(ksp,it,rnorm,ksp->monitorcontext[i]));
  }
  PetscFunctionReturn(0);
}

/*@C
   KSPMonitorSet - Sets an ADDITIONAL function to be called at every iteration to monitor
   the residual/error etc.

   Logically Collective on ksp

   Input Parameters:
+  ksp - iterative context obtained from KSPCreate()
.  monitor - pointer to function (if this is NULL, it turns off monitoring
.  mctx    - [optional] context for private data for the
             monitor routine (use NULL if no context is desired)
-  monitordestroy - [optional] routine that frees monitor context
          (may be NULL)

   Calling Sequence of monitor:
$     monitor (KSP ksp, PetscInt it, PetscReal rnorm, void *mctx)

+  ksp - iterative context obtained from KSPCreate()
.  it - iteration number
.  rnorm - (estimated) 2-norm of (preconditioned) residual
-  mctx  - optional monitoring context, as set by KSPMonitorSet()

   Options Database Keys:
+    -ksp_monitor               - sets KSPMonitorResidual()
.    -ksp_monitor draw          - sets KSPMonitorResidualDraw() and plots residual
.    -ksp_monitor draw::draw_lg - sets KSPMonitorResidualDrawLG() and plots residual
.    -ksp_monitor_pause_final   - Pauses any graphics when the solve finishes (only works for internal monitors)
.    -ksp_monitor_true_residual - sets KSPMonitorTrueResidual()
.    -ksp_monitor_true_residual draw::draw_lg - sets KSPMonitorTrueResidualDrawLG() and plots residual
.    -ksp_monitor_max           - sets KSPMonitorTrueResidualMax()
.    -ksp_monitor_singular_value - sets KSPMonitorSingularValue()
-    -ksp_monitor_cancel - cancels all monitors that have
                          been hardwired into a code by
                          calls to KSPMonitorSet(), but
                          does not cancel those set via
                          the options database.

   Notes:
   The default is to do nothing.  To print the residual, or preconditioned
   residual if KSPSetNormType(ksp,KSP_NORM_PRECONDITIONED) was called, use
   KSPMonitorResidual() as the monitoring routine, with a ASCII viewer as the
   context.

   Several different monitoring routines may be set by calling
   KSPMonitorSet() multiple times; all will be called in the
   order in which they were set.

   Fortran Notes:
    Only a single monitor function can be set for each KSP object

   Level: beginner

.seealso: KSPMonitorResidual(), KSPMonitorCancel(), KSP
@*/
PetscErrorCode  KSPMonitorSet(KSP ksp,PetscErrorCode (*monitor)(KSP,PetscInt,PetscReal,void*),void *mctx,PetscErrorCode (*monitordestroy)(void**))
{
  PetscInt       i;
  PetscBool      identical;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ksp,KSP_CLASSID,1);
  for (i=0; i<ksp->numbermonitors;i++) {
    PetscCall(PetscMonitorCompare((PetscErrorCode (*)(void))monitor,mctx,monitordestroy,(PetscErrorCode (*)(void))ksp->monitor[i],ksp->monitorcontext[i],ksp->monitordestroy[i],&identical));
    if (identical) PetscFunctionReturn(0);
  }
  PetscCheck(ksp->numbermonitors < MAXKSPMONITORS,PetscObjectComm((PetscObject)ksp),PETSC_ERR_ARG_OUTOFRANGE,"Too many KSP monitors set");
  ksp->monitor[ksp->numbermonitors]          = monitor;
  ksp->monitordestroy[ksp->numbermonitors]   = monitordestroy;
  ksp->monitorcontext[ksp->numbermonitors++] = (void*)mctx;
  PetscFunctionReturn(0);
}

/*@
   KSPMonitorCancel - Clears all monitors for a KSP object.

   Logically Collective on ksp

   Input Parameters:
.  ksp - iterative context obtained from KSPCreate()

   Options Database Key:
.  -ksp_monitor_cancel - Cancels all monitors that have
    been hardwired into a code by calls to KSPMonitorSet(),
    but does not cancel those set via the options database.

   Level: intermediate

.seealso: KSPMonitorResidual(), KSPMonitorSet(), KSP
@*/
PetscErrorCode  KSPMonitorCancel(KSP ksp)
{
  PetscInt       i;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ksp,KSP_CLASSID,1);
  for (i=0; i<ksp->numbermonitors; i++) {
    if (ksp->monitordestroy[i]) {
      PetscCall((*ksp->monitordestroy[i])(&ksp->monitorcontext[i]));
    }
  }
  ksp->numbermonitors = 0;
  PetscFunctionReturn(0);
}

/*@C
   KSPGetMonitorContext - Gets the monitoring context, as set by
   KSPMonitorSet() for the FIRST monitor only.

   Not Collective

   Input Parameter:
.  ksp - iterative context obtained from KSPCreate()

   Output Parameter:
.  ctx - monitoring context

   Level: intermediate

.seealso: KSPMonitorResidual(), KSP
@*/
PetscErrorCode  KSPGetMonitorContext(KSP ksp,void *ctx)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ksp,KSP_CLASSID,1);
  *(void**)ctx = ksp->monitorcontext[0];
  PetscFunctionReturn(0);
}

/*@
   KSPSetResidualHistory - Sets the array used to hold the residual history.
   If set, this array will contain the residual norms computed at each
   iteration of the solver.

   Not Collective

   Input Parameters:
+  ksp - iterative context obtained from KSPCreate()
.  a   - array to hold history
.  na  - size of a
-  reset - PETSC_TRUE indicates the history counter is reset to zero
           for each new linear solve

   Level: advanced

   Notes:
   If provided, he array is NOT freed by PETSc so the user needs to keep track of it and destroy once the KSP object is destroyed.
   If 'a' is NULL then space is allocated for the history. If 'na' PETSC_DECIDE or PETSC_DEFAULT then a
   default array of length 10000 is allocated.

.seealso: KSPGetResidualHistory(), KSP

@*/
PetscErrorCode KSPSetResidualHistory(KSP ksp,PetscReal a[],PetscInt na,PetscBool reset)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ksp,KSP_CLASSID,1);

  PetscCall(PetscFree(ksp->res_hist_alloc));
  if (na != PETSC_DECIDE && na != PETSC_DEFAULT && a) {
    ksp->res_hist     = a;
    ksp->res_hist_max = (size_t) na;
  } else {
    if (na != PETSC_DECIDE && na != PETSC_DEFAULT) ksp->res_hist_max = (size_t) na;
    else                                           ksp->res_hist_max = 10000; /* like default ksp->max_it */
    PetscCall(PetscCalloc1(ksp->res_hist_max,&ksp->res_hist_alloc));

    ksp->res_hist = ksp->res_hist_alloc;
  }
  ksp->res_hist_len   = 0;
  ksp->res_hist_reset = reset;
  PetscFunctionReturn(0);
}

/*@C
   KSPGetResidualHistory - Gets the array used to hold the residual history
   and the number of residuals it contains.

   Not Collective

   Input Parameter:
.  ksp - iterative context obtained from KSPCreate()

   Output Parameters:
+  a   - pointer to array to hold history (or NULL)
-  na  - number of used entries in a (or NULL)

   Level: advanced

   Notes:
     This array is borrowed and should not be freed by the caller.
     Can only be called after a KSPSetResidualHistory() otherwise a and na are set to zero

     The Fortran version of this routine has a calling sequence
$   call KSPGetResidualHistory(KSP ksp, integer na, integer ierr)
    note that you have passed a Fortran array into KSPSetResidualHistory() and you need
    to access the residual values from this Fortran array you provided. Only the na (number of
    residual norms currently held) is set.

.seealso: KSPSetResidualHistory(), KSP

@*/
PetscErrorCode KSPGetResidualHistory(KSP ksp, const PetscReal *a[],PetscInt *na)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ksp,KSP_CLASSID,1);
  if (a) *a = ksp->res_hist;
  if (na) *na = (PetscInt) ksp->res_hist_len;
  PetscFunctionReturn(0);
}

/*@
  KSPSetErrorHistory - Sets the array used to hold the error history. If set, this array will contain the error norms computed at each iteration of the solver.

  Not Collective

  Input Parameters:
+ ksp   - iterative context obtained from KSPCreate()
. a     - array to hold history
. na    - size of a
- reset - PETSC_TRUE indicates the history counter is reset to zero for each new linear solve

  Level: advanced

  Notes:
  If provided, the array is NOT freed by PETSc so the user needs to keep track of it and destroy once the KSP object is destroyed.
  If 'a' is NULL then space is allocated for the history. If 'na' PETSC_DECIDE or PETSC_DEFAULT then a default array of length 10000 is allocated.

.seealso: KSPGetErrorHistory(), KSPSetResidualHistory(), KSP
@*/
PetscErrorCode KSPSetErrorHistory(KSP ksp, PetscReal a[], PetscInt na, PetscBool reset)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ksp,KSP_CLASSID,1);

  PetscCall(PetscFree(ksp->err_hist_alloc));
  if (na != PETSC_DECIDE && na != PETSC_DEFAULT && a) {
    ksp->err_hist     = a;
    ksp->err_hist_max = (size_t) na;
  } else {
    if (na != PETSC_DECIDE && na != PETSC_DEFAULT) ksp->err_hist_max = (size_t) na;
    else                                           ksp->err_hist_max = 10000; /* like default ksp->max_it */
    PetscCall(PetscCalloc1(ksp->err_hist_max, &ksp->err_hist_alloc));

    ksp->err_hist = ksp->err_hist_alloc;
  }
  ksp->err_hist_len   = 0;
  ksp->err_hist_reset = reset;
  PetscFunctionReturn(0);
}

/*@C
  KSPGetErrorHistory - Gets the array used to hold the error history and the number of residuals it contains.

  Not Collective

  Input Parameter:
. ksp - iterative context obtained from KSPCreate()

  Output Parameters:
+ a  - pointer to array to hold history (or NULL)
- na - number of used entries in a (or NULL)

  Level: advanced

  Notes:
  This array is borrowed and should not be freed by the caller.
  Can only be called after a KSPSetErrorHistory() otherwise a and na are set to zero
  The Fortran version of this routine has a calling sequence
$   call KSPGetErrorHistory(KSP ksp, integer na, integer ierr)
  note that you have passed a Fortran array into KSPSetErrorHistory() and you need
  to access the residual values from this Fortran array you provided. Only the na (number of
  residual norms currently held) is set.

.seealso: KSPSetErrorHistory(), KSPGetResidualHistory(), KSP
@*/
PetscErrorCode KSPGetErrorHistory(KSP ksp, const PetscReal *a[], PetscInt *na)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ksp,KSP_CLASSID,1);
  if (a)  *a  = ksp->err_hist;
  if (na) *na = (PetscInt) ksp->err_hist_len;
  PetscFunctionReturn(0);
}

/*
  KSPComputeConvergenceRate - Compute the convergence rate for the iteration

  Not collective

  Input Parameter:
. ksp - The KSP

  Output Parameters:
+ cr   - The residual contraction rate
. rRsq - The coefficient of determination, R^2, indicating the linearity of the data
. ce   - The error contraction rate
- eRsq - The coefficient of determination, R^2, indicating the linearity of the data

  Note:
  Suppose that the residual is reduced linearly, $r_k = c^k r_0$, which means $log r_k = log r_0 + k log c$. After linear regression,
  the slope is $\log c$. The coefficient of determination is given by $1 - \frac{\sum_i (y_i - f(x_i))^2}{\sum_i (y_i - \bar y)}$,
  see also https://en.wikipedia.org/wiki/Coefficient_of_determination

  Level: advanced

.seealso: KSPConvergedRateView()
*/
PetscErrorCode KSPComputeConvergenceRate(KSP ksp, PetscReal *cr, PetscReal *rRsq, PetscReal *ce, PetscReal *eRsq)
{
  PetscReal      const *hist;
  PetscReal      *x, *y, slope, intercept, mean = 0.0, var = 0.0, res = 0.0;
  PetscInt       n, k;

  PetscFunctionBegin;
  if (cr || rRsq) {
    PetscCall(KSPGetResidualHistory(ksp, &hist, &n));
    if (!n) {
      if (cr)   *cr   =  0.0;
      if (rRsq) *rRsq = -1.0;
    } else {
      PetscCall(PetscMalloc2(n, &x, n, &y));
      for (k = 0; k < n; ++k) {
        x[k] = k;
        y[k] = PetscLogReal(hist[k]);
        mean += y[k];
      }
      mean /= n;
      PetscCall(PetscLinearRegression(n, x, y, &slope, &intercept));
      for (k = 0; k < n; ++k) {
        res += PetscSqr(y[k] - (slope*x[k] + intercept));
        var += PetscSqr(y[k] - mean);
      }
      PetscCall(PetscFree2(x, y));
      if (cr)   *cr   = PetscExpReal(slope);
      if (rRsq) *rRsq = var < PETSC_MACHINE_EPSILON ? 0.0 : 1.0 - (res / var);
    }
  }
  if (ce || eRsq) {
    PetscCall(KSPGetErrorHistory(ksp, &hist, &n));
    if (!n) {
      if (ce)   *ce   =  0.0;
      if (eRsq) *eRsq = -1.0;
    } else {
      PetscCall(PetscMalloc2(n, &x, n, &y));
      for (k = 0; k < n; ++k) {
        x[k] = k;
        y[k] = PetscLogReal(hist[k]);
        mean += y[k];
      }
      mean /= n;
      PetscCall(PetscLinearRegression(n, x, y, &slope, &intercept));
      for (k = 0; k < n; ++k) {
        res += PetscSqr(y[k] - (slope*x[k] + intercept));
        var += PetscSqr(y[k] - mean);
      }
      PetscCall(PetscFree2(x, y));
      if (ce)   *ce   = PetscExpReal(slope);
      if (eRsq) *eRsq = var < PETSC_MACHINE_EPSILON ? 0.0 : 1.0 - (res / var);
    }
  }
  PetscFunctionReturn(0);
}

/*@C
   KSPSetConvergenceTest - Sets the function to be used to determine
   convergence.

   Logically Collective on ksp

   Input Parameters:
+  ksp - iterative context obtained from KSPCreate()
.  converge - pointer to the function
.  cctx    - context for private data for the convergence routine (may be null)
-  destroy - a routine for destroying the context (may be null)

   Calling sequence of converge:
$     converge (KSP ksp, PetscInt it, PetscReal rnorm, KSPConvergedReason *reason,void *mctx)

+  ksp - iterative context obtained from KSPCreate()
.  it - iteration number
.  rnorm - (estimated) 2-norm of (preconditioned) residual
.  reason - the reason why it has converged or diverged
-  cctx  - optional convergence context, as set by KSPSetConvergenceTest()

   Notes:
   Must be called after the KSP type has been set so put this after
   a call to KSPSetType(), or KSPSetFromOptions().

   The default convergence test, KSPConvergedDefault(), aborts if the
   residual grows to more than 10000 times the initial residual.

   The default is a combination of relative and absolute tolerances.
   The residual value that is tested may be an approximation; routines
   that need exact values should compute them.

   In the default PETSc convergence test, the precise values of reason
   are macros such as KSP_CONVERGED_RTOL, which are defined in petscksp.h.

   Level: advanced

.seealso: KSPConvergedDefault(), KSPGetConvergenceContext(), KSPSetTolerances(), KSP, KSPGetConvergenceTest(), KSPGetAndClearConvergenceTest()
@*/
PetscErrorCode  KSPSetConvergenceTest(KSP ksp,PetscErrorCode (*converge)(KSP,PetscInt,PetscReal,KSPConvergedReason*,void*),void *cctx,PetscErrorCode (*destroy)(void*))
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ksp,KSP_CLASSID,1);
  if (ksp->convergeddestroy) {
    PetscCall((*ksp->convergeddestroy)(ksp->cnvP));
  }
  ksp->converged        = converge;
  ksp->convergeddestroy = destroy;
  ksp->cnvP             = (void*)cctx;
  PetscFunctionReturn(0);
}

/*@C
   KSPGetConvergenceTest - Gets the function to be used to determine
   convergence.

   Logically Collective on ksp

   Input Parameter:
.   ksp - iterative context obtained from KSPCreate()

   Output Parameters:
+  converge - pointer to convergence test function
.  cctx    - context for private data for the convergence routine (may be null)
-  destroy - a routine for destroying the context (may be null)

   Calling sequence of converge:
$     converge (KSP ksp, PetscInt it, PetscReal rnorm, KSPConvergedReason *reason,void *mctx)

+  ksp - iterative context obtained from KSPCreate()
.  it - iteration number
.  rnorm - (estimated) 2-norm of (preconditioned) residual
.  reason - the reason why it has converged or diverged
-  cctx  - optional convergence context, as set by KSPSetConvergenceTest()

   Level: advanced

.seealso: KSPConvergedDefault(), KSPGetConvergenceContext(), KSPSetTolerances(), KSP, KSPSetConvergenceTest(), KSPGetAndClearConvergenceTest()
@*/
PetscErrorCode  KSPGetConvergenceTest(KSP ksp,PetscErrorCode (**converge)(KSP,PetscInt,PetscReal,KSPConvergedReason*,void*),void **cctx,PetscErrorCode (**destroy)(void*))
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ksp,KSP_CLASSID,1);
  if (converge) *converge = ksp->converged;
  if (destroy)  *destroy  = ksp->convergeddestroy;
  if (cctx)     *cctx     = ksp->cnvP;
  PetscFunctionReturn(0);
}

/*@C
   KSPGetAndClearConvergenceTest - Gets the function to be used to determine convergence. Removes the current test without calling destroy on the test context

   Logically Collective on ksp

   Input Parameter:
.   ksp - iterative context obtained from KSPCreate()

   Output Parameters:
+  converge - pointer to convergence test function
.  cctx    - context for private data for the convergence routine
-  destroy - a routine for destroying the context

   Calling sequence of converge:
$     converge (KSP ksp, PetscInt it, PetscReal rnorm, KSPConvergedReason *reason,void *mctx)

+  ksp - iterative context obtained from KSPCreate()
.  it - iteration number
.  rnorm - (estimated) 2-norm of (preconditioned) residual
.  reason - the reason why it has converged or diverged
-  cctx  - optional convergence context, as set by KSPSetConvergenceTest()

   Level: advanced

   Notes: This is intended to be used to allow transferring the convergence test (and its context) to another testing object (for example another KSP) and then calling
          KSPSetConvergenceTest() on this original KSP. If you just called KSPGetConvergenceTest() followed by KSPSetConvergenceTest() the original context information
          would be destroyed and hence the transferred context would be invalid and trigger a crash on use

.seealso: KSPConvergedDefault(), KSPGetConvergenceContext(), KSPSetTolerances(), KSP, KSPSetConvergenceTest(), KSPGetConvergenceTest()
@*/
PetscErrorCode  KSPGetAndClearConvergenceTest(KSP ksp,PetscErrorCode (**converge)(KSP,PetscInt,PetscReal,KSPConvergedReason*,void*),void **cctx,PetscErrorCode (**destroy)(void*))
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ksp,KSP_CLASSID,1);
  *converge             = ksp->converged;
  *destroy              = ksp->convergeddestroy;
  *cctx                 = ksp->cnvP;
  ksp->converged        = NULL;
  ksp->cnvP             = NULL;
  ksp->convergeddestroy = NULL;
  PetscFunctionReturn(0);
}

/*@C
   KSPGetConvergenceContext - Gets the convergence context set with
   KSPSetConvergenceTest().

   Not Collective

   Input Parameter:
.  ksp - iterative context obtained from KSPCreate()

   Output Parameter:
.  ctx - monitoring context

   Level: advanced

.seealso: KSPConvergedDefault(), KSPSetConvergenceTest(), KSP
@*/
PetscErrorCode  KSPGetConvergenceContext(KSP ksp,void *ctx)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ksp,KSP_CLASSID,1);
  *(void**)ctx = ksp->cnvP;
  PetscFunctionReturn(0);
}

/*@C
   KSPBuildSolution - Builds the approximate solution in a vector provided.
   This routine is NOT commonly needed (see KSPSolve()).

   Collective on ksp

   Input Parameter:
.  ctx - iterative context obtained from KSPCreate()

   Output Parameter:
   Provide exactly one of
+  v - location to stash solution.
-  V - the solution is returned in this location. This vector is created
       internally. This vector should NOT be destroyed by the user with
       VecDestroy().

   Notes:
   This routine can be used in one of two ways
.vb
      KSPBuildSolution(ksp,NULL,&V);
   or
      KSPBuildSolution(ksp,v,NULL); or KSPBuildSolution(ksp,v,&v);
.ve
   In the first case an internal vector is allocated to store the solution
   (the user cannot destroy this vector). In the second case the solution
   is generated in the vector that the user provides. Note that for certain
   methods, such as KSPCG, the second case requires a copy of the solution,
   while in the first case the call is essentially free since it simply
   returns the vector where the solution already is stored. For some methods
   like GMRES this is a reasonably expensive operation and should only be
   used in truly needed.

   Level: advanced

.seealso: KSPGetSolution(), KSPBuildResidual(), KSP
@*/
PetscErrorCode  KSPBuildSolution(KSP ksp,Vec v,Vec *V)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ksp,KSP_CLASSID,1);
  PetscCheck(V || v,PetscObjectComm((PetscObject)ksp),PETSC_ERR_ARG_WRONG,"Must provide either v or V");
  if (!V) V = &v;
  PetscCall((*ksp->ops->buildsolution)(ksp,v,V));
  PetscFunctionReturn(0);
}

/*@C
   KSPBuildResidual - Builds the residual in a vector provided.

   Collective on ksp

   Input Parameter:
.  ksp - iterative context obtained from KSPCreate()

   Output Parameters:
+  v - optional location to stash residual.  If v is not provided,
       then a location is generated.
.  t - work vector.  If not provided then one is generated.
-  V - the residual

   Notes:
   Regardless of whether or not v is provided, the residual is
   returned in V.

   Level: advanced

.seealso: KSPBuildSolution()
@*/
PetscErrorCode  KSPBuildResidual(KSP ksp,Vec t,Vec v,Vec *V)
{
  PetscBool      flag = PETSC_FALSE;
  Vec            w    = v,tt = t;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ksp,KSP_CLASSID,1);
  if (!w) {
    PetscCall(VecDuplicate(ksp->vec_rhs,&w));
    PetscCall(PetscLogObjectParent((PetscObject)ksp,(PetscObject)w));
  }
  if (!tt) {
    PetscCall(VecDuplicate(ksp->vec_sol,&tt)); flag = PETSC_TRUE;
    PetscCall(PetscLogObjectParent((PetscObject)ksp,(PetscObject)tt));
  }
  PetscCall((*ksp->ops->buildresidual)(ksp,tt,w,V));
  if (flag) PetscCall(VecDestroy(&tt));
  PetscFunctionReturn(0);
}

/*@
   KSPSetDiagonalScale - Tells KSP to symmetrically diagonally scale the system
     before solving. This actually CHANGES the matrix (and right hand side).

   Logically Collective on ksp

   Input Parameters:
+  ksp - the KSP context
-  scale - PETSC_TRUE or PETSC_FALSE

   Options Database Key:
+   -ksp_diagonal_scale -
-   -ksp_diagonal_scale_fix - scale the matrix back AFTER the solve

    Notes:
    Scales the matrix by  D^(-1/2)  A  D^(-1/2)  [D^(1/2) x ] = D^(-1/2) b
       where D_{ii} is 1/abs(A_{ii}) unless A_{ii} is zero and then it is 1.

    BE CAREFUL with this routine: it actually scales the matrix and right
    hand side that define the system. After the system is solved the matrix
    and right hand side remain scaled unless you use KSPSetDiagonalScaleFix()

    This should NOT be used within the SNES solves if you are using a line
    search.

    If you use this with the PCType Eisenstat preconditioner than you can
    use the PCEisenstatSetNoDiagonalScaling() option, or -pc_eisenstat_no_diagonal_scaling
    to save some unneeded, redundant flops.

   Level: intermediate

.seealso: KSPGetDiagonalScale(), KSPSetDiagonalScaleFix(), KSP
@*/
PetscErrorCode  KSPSetDiagonalScale(KSP ksp,PetscBool scale)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ksp,KSP_CLASSID,1);
  PetscValidLogicalCollectiveBool(ksp,scale,2);
  ksp->dscale = scale;
  PetscFunctionReturn(0);
}

/*@
   KSPGetDiagonalScale - Checks if KSP solver scales the matrix and
                          right hand side

   Not Collective

   Input Parameter:
.  ksp - the KSP context

   Output Parameter:
.  scale - PETSC_TRUE or PETSC_FALSE

   Notes:
    BE CAREFUL with this routine: it actually scales the matrix and right
    hand side that define the system. After the system is solved the matrix
    and right hand side remain scaled  unless you use KSPSetDiagonalScaleFix()

   Level: intermediate

.seealso: KSPSetDiagonalScale(), KSPSetDiagonalScaleFix(), KSP
@*/
PetscErrorCode  KSPGetDiagonalScale(KSP ksp,PetscBool  *scale)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ksp,KSP_CLASSID,1);
  PetscValidBoolPointer(scale,2);
  *scale = ksp->dscale;
  PetscFunctionReturn(0);
}

/*@
   KSPSetDiagonalScaleFix - Tells KSP to diagonally scale the system
     back after solving.

   Logically Collective on ksp

   Input Parameters:
+  ksp - the KSP context
-  fix - PETSC_TRUE to scale back after the system solve, PETSC_FALSE to not
         rescale (default)

   Notes:
     Must be called after KSPSetDiagonalScale()

     Using this will slow things down, because it rescales the matrix before and
     after each linear solve. This is intended mainly for testing to allow one
     to easily get back the original system to make sure the solution computed is
     accurate enough.

   Level: intermediate

.seealso: KSPGetDiagonalScale(), KSPSetDiagonalScale(), KSPGetDiagonalScaleFix(), KSP
@*/
PetscErrorCode  KSPSetDiagonalScaleFix(KSP ksp,PetscBool fix)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ksp,KSP_CLASSID,1);
  PetscValidLogicalCollectiveBool(ksp,fix,2);
  ksp->dscalefix = fix;
  PetscFunctionReturn(0);
}

/*@
   KSPGetDiagonalScaleFix - Determines if KSP diagonally scales the system
     back after solving.

   Not Collective

   Input Parameter:
.  ksp - the KSP context

   Output Parameter:
.  fix - PETSC_TRUE to scale back after the system solve, PETSC_FALSE to not
         rescale (default)

   Notes:
     Must be called after KSPSetDiagonalScale()

     If PETSC_TRUE will slow things down, because it rescales the matrix before and
     after each linear solve. This is intended mainly for testing to allow one
     to easily get back the original system to make sure the solution computed is
     accurate enough.

   Level: intermediate

.seealso: KSPGetDiagonalScale(), KSPSetDiagonalScale(), KSPSetDiagonalScaleFix(), KSP
@*/
PetscErrorCode  KSPGetDiagonalScaleFix(KSP ksp,PetscBool  *fix)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ksp,KSP_CLASSID,1);
  PetscValidBoolPointer(fix,2);
  *fix = ksp->dscalefix;
  PetscFunctionReturn(0);
}

/*@C
   KSPSetComputeOperators - set routine to compute the linear operators

   Logically Collective

   Input Parameters:
+  ksp - the KSP context
.  func - function to compute the operators
-  ctx - optional context

   Calling sequence of func:
$  func(KSP ksp,Mat A,Mat B,void *ctx)

+  ksp - the KSP context
.  A - the linear operator
.  B - preconditioning matrix
-  ctx - optional user-provided context

   Notes:
    The user provided func() will be called automatically at the very next call to KSPSolve(). It will not be called at future KSPSolve() calls
          unless either KSPSetComputeOperators() or KSPSetOperators() is called before that KSPSolve() is called.

          To reuse the same preconditioner for the next KSPSolve() and not compute a new one based on the most recently computed matrix call KSPSetReusePreconditioner()

   Level: beginner

.seealso: KSPSetOperators(), KSPSetComputeRHS(), DMKSPSetComputeOperators(), KSPSetComputeInitialGuess()
@*/
PetscErrorCode KSPSetComputeOperators(KSP ksp,PetscErrorCode (*func)(KSP,Mat,Mat,void*),void *ctx)
{
  DM             dm;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ksp,KSP_CLASSID,1);
  PetscCall(KSPGetDM(ksp,&dm));
  PetscCall(DMKSPSetComputeOperators(dm,func,ctx));
  if (ksp->setupstage == KSP_SETUP_NEWRHS) ksp->setupstage = KSP_SETUP_NEWMATRIX;
  PetscFunctionReturn(0);
}

/*@C
   KSPSetComputeRHS - set routine to compute the right hand side of the linear system

   Logically Collective

   Input Parameters:
+  ksp - the KSP context
.  func - function to compute the right hand side
-  ctx - optional context

   Calling sequence of func:
$  func(KSP ksp,Vec b,void *ctx)

+  ksp - the KSP context
.  b - right hand side of linear system
-  ctx - optional user-provided context

   Notes:
    The routine you provide will be called EACH you call KSPSolve() to prepare the new right hand side for that solve

   Level: beginner

.seealso: KSPSolve(), DMKSPSetComputeRHS(), KSPSetComputeOperators()
@*/
PetscErrorCode KSPSetComputeRHS(KSP ksp,PetscErrorCode (*func)(KSP,Vec,void*),void *ctx)
{
  DM             dm;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ksp,KSP_CLASSID,1);
  PetscCall(KSPGetDM(ksp,&dm));
  PetscCall(DMKSPSetComputeRHS(dm,func,ctx));
  PetscFunctionReturn(0);
}

/*@C
   KSPSetComputeInitialGuess - set routine to compute the initial guess of the linear system

   Logically Collective

   Input Parameters:
+  ksp - the KSP context
.  func - function to compute the initial guess
-  ctx - optional context

   Calling sequence of func:
$  func(KSP ksp,Vec x,void *ctx)

+  ksp - the KSP context
.  x - solution vector
-  ctx - optional user-provided context

   Notes: This should only be used in conjunction with KSPSetComputeRHS(), KSPSetComputeOperators(), otherwise
   call KSPSetInitialGuessNonzero() and set the initial guess values in the solution vector passed to KSPSolve().

   Level: beginner

.seealso: KSPSolve(), KSPSetComputeRHS(), KSPSetComputeOperators(), DMKSPSetComputeInitialGuess()
@*/
PetscErrorCode KSPSetComputeInitialGuess(KSP ksp,PetscErrorCode (*func)(KSP,Vec,void*),void *ctx)
{
  DM             dm;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ksp,KSP_CLASSID,1);
  PetscCall(KSPGetDM(ksp,&dm));
  PetscCall(DMKSPSetComputeInitialGuess(dm,func,ctx));
  PetscFunctionReturn(0);
}

/*@
   KSPSetUseExplicitTranspose - Determines if transpose the system explicitly
   in KSPSolveTranspose.

   Logically Collective on ksp

   Input Parameter:
.  ksp - the KSP context

   Output Parameter:
.  flg - PETSC_TRUE to transpose the system in KSPSolveTranspose, PETSC_FALSE to not
         transpose (default)

   Level: advanced

.seealso: KSPSolveTranspose(), KSP
@*/
PetscErrorCode KSPSetUseExplicitTranspose(KSP ksp,PetscBool flg)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ksp,KSP_CLASSID,1);
  PetscValidLogicalCollectiveBool(ksp,flg,2);
  ksp->transpose.use_explicittranspose = flg;
  PetscFunctionReturn(0);
}
