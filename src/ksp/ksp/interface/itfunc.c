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
  PetscErrorCode ierr;

  ierr = PetscViewerPushFormat(viewer, format);CHKERRQ(ierr);
  ierr = PetscObjectView(obj, viewer);CHKERRQ(ierr);
  ierr = PetscViewerPopFormat(viewer);CHKERRQ(ierr);
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
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ksp,KSP_CLASSID,1);
  PetscValidRealPointer(emax,2);
  PetscValidRealPointer(emin,3);
  PetscAssertFalse(!ksp->calc_sings,PetscObjectComm((PetscObject)ksp),PETSC_ERR_ARG_WRONGSTATE,"Singular values not requested before KSPSetUp()");

  if (ksp->ops->computeextremesingularvalues) {
    ierr = (*ksp->ops->computeextremesingularvalues)(ksp,emax,emin);CHKERRQ(ierr);
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
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ksp,KSP_CLASSID,1);
  if (n) PetscValidRealPointer(r,3);
  if (n) PetscValidRealPointer(c,4);
  PetscAssertFalse(n<0,PetscObjectComm((PetscObject)ksp),PETSC_ERR_ARG_OUTOFRANGE,"Requested < 0 Eigenvalues");
  PetscValidIntPointer(neig,5);
  PetscAssertFalse(!ksp->calc_sings,PetscObjectComm((PetscObject)ksp),PETSC_ERR_ARG_WRONGSTATE,"Eigenvalues not requested before KSPSetUp()");

  if (n && ksp->ops->computeeigenvalues) {
    ierr = (*ksp->ops->computeeigenvalues)(ksp,n,r,c,neig);CHKERRQ(ierr);
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
.  ritz  - PETSC_TRUE or PETSC_FALSE for ritz pairs or harmonic Ritz pairs, respectively
-  small - PETSC_TRUE or PETSC_FALSE for smallest or largest (harmonic) Ritz values, respectively

   Input/Output Parameter:
.  nrit  - number of (harmonic) Ritz pairs to compute; on output,
           actual number of computed (harmonic) Ritz pairs

   Output Parameters:
+  S     - multidimensional vector with Ritz vectors
.  tetar - real part of the Ritz values
-  tetai - imaginary part of the Ritz values

   Notes:
   -For GMRES, the (harmonic) Ritz pairs are computed from the Hessenberg matrix obtained during
   the last complete cycle, or obtained at the end of the solution if the method is stopped before
   a restart. Then, the number of actual (harmonic) Ritz pairs computed is less or equal to the restart
   parameter for GMRES if a complete cycle has been performed or less or equal to the number of GMRES
   iterations.
   -Moreover, for real matrices, the (harmonic) Ritz pairs are possibly complex-valued. In such a case,
   the routine selects the complex (harmonic) Ritz value and its conjugate, and two successive columns of S
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
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ksp,KSP_CLASSID,1);
  PetscAssertFalse(!ksp->calc_ritz,PetscObjectComm((PetscObject)ksp),PETSC_ERR_ARG_WRONGSTATE,"Ritz pairs not requested before KSPSetUp()");
  if (ksp->ops->computeritz) {ierr = (*ksp->ops->computeritz)(ksp,ritz,small,nrit,S,tetar,tetai);CHKERRQ(ierr);}
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
  PetscErrorCode ierr;
  PCFailedReason pcreason;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ksp,KSP_CLASSID,1);
  level++;
  ierr = KSPGetPC(ksp,&pc);CHKERRQ(ierr);
  ierr = PCSetUpOnBlocks(pc);CHKERRQ(ierr);
  ierr = PCGetFailedReasonRank(pc,&pcreason);CHKERRQ(ierr);
  level--;
  /* TODO: this code was wrong and is still wrong, there is no way to propagate the failure to all processes; there is no code to handle a ksp->reason on only some ranks */
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
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ksp,KSP_CLASSID,1);
  ierr = KSPGetPC(ksp,&pc);CHKERRQ(ierr);
  ierr = PCSetReusePreconditioner(pc,flag);CHKERRQ(ierr);
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
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ksp,KSP_CLASSID,1);
  PetscValidPointer(flag,2);
  *flag = PETSC_FALSE;
  if (ksp->pc) {
    ierr = PCGetReusePreconditioner(ksp->pc,flag);CHKERRQ(ierr);
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
  PetscErrorCode ierr;
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
    ierr = KSPSetType(ksp,KSPGMRES);CHKERRQ(ierr);
  }
  ierr = KSPSetUpNorms_Private(ksp,PETSC_TRUE,&ksp->normtype,&ksp->pc_side);CHKERRQ(ierr);

  if (ksp->dmActive && !ksp->setupstage) {
    /* first time in so build matrix and vector data structures using DM */
    if (!ksp->vec_rhs) {ierr = DMCreateGlobalVector(ksp->dm,&ksp->vec_rhs);CHKERRQ(ierr);}
    if (!ksp->vec_sol) {ierr = DMCreateGlobalVector(ksp->dm,&ksp->vec_sol);CHKERRQ(ierr);}
    ierr = DMCreateMatrix(ksp->dm,&A);CHKERRQ(ierr);
    ierr = KSPSetOperators(ksp,A,A);CHKERRQ(ierr);
    ierr = PetscObjectDereference((PetscObject)A);CHKERRQ(ierr);
  }

  if (ksp->dmActive) {
    DMKSP kdm;
    ierr = DMGetDMKSP(ksp->dm,&kdm);CHKERRQ(ierr);

    if (kdm->ops->computeinitialguess && ksp->setupstage != KSP_SETUP_NEWRHS) {
      /* only computes initial guess the first time through */
      ierr = (*kdm->ops->computeinitialguess)(ksp,ksp->vec_sol,kdm->initialguessctx);CHKERRQ(ierr);
      ierr = KSPSetInitialGuessNonzero(ksp,PETSC_TRUE);CHKERRQ(ierr);
    }
    if (kdm->ops->computerhs) {
      ierr = (*kdm->ops->computerhs)(ksp,ksp->vec_rhs,kdm->rhsctx);CHKERRQ(ierr);
    }

    if (ksp->setupstage != KSP_SETUP_NEWRHS) {
      if (kdm->ops->computeoperators) {
        ierr = KSPGetOperators(ksp,&A,&B);CHKERRQ(ierr);
        ierr = (*kdm->ops->computeoperators)(ksp,A,B,kdm->operatorsctx);CHKERRQ(ierr);
      } else SETERRQ(PetscObjectComm((PetscObject)ksp),PETSC_ERR_ARG_WRONGSTATE,"You called KSPSetDM() but did not use DMKSPSetComputeOperators() or KSPSetDMActive(ksp,PETSC_FALSE);");
    }
  }

  if (ksp->setupstage == KSP_SETUP_NEWRHS) {
    level--;
    PetscFunctionReturn(0);
  }
  ierr = PetscLogEventBegin(KSP_SetUp,ksp,ksp->vec_rhs,ksp->vec_sol,0);CHKERRQ(ierr);

  switch (ksp->setupstage) {
  case KSP_SETUP_NEW:
    ierr = (*ksp->ops->setup)(ksp);CHKERRQ(ierr);
    break;
  case KSP_SETUP_NEWMATRIX: {   /* This should be replaced with a more general mechanism */
    if (ksp->setupnewmatrix) {
      ierr = (*ksp->ops->setup)(ksp);CHKERRQ(ierr);
    }
  } break;
  default: break;
  }

  if (!ksp->pc) {ierr = KSPGetPC(ksp,&ksp->pc);CHKERRQ(ierr);}
  ierr = PCGetOperators(ksp->pc,&mat,&pmat);CHKERRQ(ierr);
  /* scale the matrix if requested */
  if (ksp->dscale) {
    PetscScalar *xx;
    PetscInt    i,n;
    PetscBool   zeroflag = PETSC_FALSE;
    if (!ksp->pc) {ierr = KSPGetPC(ksp,&ksp->pc);CHKERRQ(ierr);}
    if (!ksp->diagonal) { /* allocate vector to hold diagonal */
      ierr = MatCreateVecs(pmat,&ksp->diagonal,NULL);CHKERRQ(ierr);
    }
    ierr = MatGetDiagonal(pmat,ksp->diagonal);CHKERRQ(ierr);
    ierr = VecGetLocalSize(ksp->diagonal,&n);CHKERRQ(ierr);
    ierr = VecGetArray(ksp->diagonal,&xx);CHKERRQ(ierr);
    for (i=0; i<n; i++) {
      if (xx[i] != 0.0) xx[i] = 1.0/PetscSqrtReal(PetscAbsScalar(xx[i]));
      else {
        xx[i]    = 1.0;
        zeroflag = PETSC_TRUE;
      }
    }
    ierr = VecRestoreArray(ksp->diagonal,&xx);CHKERRQ(ierr);
    if (zeroflag) {
      ierr = PetscInfo(ksp,"Zero detected in diagonal of matrix, using 1 at those locations\n");CHKERRQ(ierr);
    }
    ierr = MatDiagonalScale(pmat,ksp->diagonal,ksp->diagonal);CHKERRQ(ierr);
    if (mat != pmat) {ierr = MatDiagonalScale(mat,ksp->diagonal,ksp->diagonal);CHKERRQ(ierr);}
    ksp->dscalefix2 = PETSC_FALSE;
  }
  ierr = PetscLogEventEnd(KSP_SetUp,ksp,ksp->vec_rhs,ksp->vec_sol,0);CHKERRQ(ierr);
  ierr = PCSetErrorIfFailure(ksp->pc,ksp->errorifnotconverged);CHKERRQ(ierr);
  ierr = PCSetUp(ksp->pc);CHKERRQ(ierr);
  ierr = PCGetFailedReasonRank(ksp->pc,&pcreason);CHKERRQ(ierr);
  /* TODO: this code was wrong and is still wrong, there is no way to propagate the failure to all processes; their is no code to handle a ksp->reason on only some ranks */
  if (pcreason) {
    ksp->reason = KSP_DIVERGED_PC_FAILED;
  }

  ierr = MatGetNullSpace(mat,&nullsp);CHKERRQ(ierr);
  if (nullsp) {
    PetscBool test = PETSC_FALSE;
    ierr = PetscOptionsGetBool(((PetscObject)ksp)->options,((PetscObject)ksp)->prefix,"-ksp_test_null_space",&test,NULL);CHKERRQ(ierr);
    if (test) {
      ierr = MatNullSpaceTest(nullsp,mat,NULL);CHKERRQ(ierr);
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
  PetscErrorCode    ierr;
  PetscBool         isAscii;
  PetscViewerFormat format;

  PetscFunctionBegin;
  if (!viewer) viewer = PETSC_VIEWER_STDOUT_(PetscObjectComm((PetscObject)ksp));
  ierr = PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&isAscii);CHKERRQ(ierr);
  if (isAscii) {
    ierr = PetscViewerGetFormat(viewer, &format);CHKERRQ(ierr);
    ierr = PetscViewerASCIIAddTab(viewer,((PetscObject)ksp)->tablevel);CHKERRQ(ierr);
    if (ksp->reason > 0 && format != PETSC_VIEWER_FAILED) {
      if (((PetscObject) ksp)->prefix) {
        ierr = PetscViewerASCIIPrintf(viewer,"Linear %s solve converged due to %s iterations %D\n",((PetscObject) ksp)->prefix,KSPConvergedReasons[ksp->reason],ksp->its);CHKERRQ(ierr);
      } else {
        ierr = PetscViewerASCIIPrintf(viewer,"Linear solve converged due to %s iterations %D\n",KSPConvergedReasons[ksp->reason],ksp->its);CHKERRQ(ierr);
      }
    } else if (ksp->reason <= 0) {
      if (((PetscObject) ksp)->prefix) {
        ierr = PetscViewerASCIIPrintf(viewer,"Linear %s solve did not converge due to %s iterations %D\n",((PetscObject) ksp)->prefix,KSPConvergedReasons[ksp->reason],ksp->its);CHKERRQ(ierr);
      } else {
        ierr = PetscViewerASCIIPrintf(viewer,"Linear solve did not converge due to %s iterations %D\n",KSPConvergedReasons[ksp->reason],ksp->its);CHKERRQ(ierr);
      }
      if (ksp->reason == KSP_DIVERGED_PC_FAILED) {
        PCFailedReason reason;
        ierr = PCGetFailedReason(ksp->pc,&reason);CHKERRQ(ierr);
        ierr = PetscViewerASCIIPrintf(viewer,"               PC failed due to %s \n",PCFailedReasons[reason]);CHKERRQ(ierr);
      }
    }
    ierr = PetscViewerASCIISubtractTab(viewer,((PetscObject)ksp)->tablevel);CHKERRQ(ierr);
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
  PetscErrorCode ierr;
  PetscBool      identical;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ksp,KSP_CLASSID,1);
  for (i=0; i<ksp->numberreasonviews;i++) {
    ierr = PetscMonitorCompare((PetscErrorCode (*)(void))f,vctx,reasonviewdestroy,(PetscErrorCode (*)(void))ksp->reasonview[i],ksp->reasonviewcontext[i],ksp->reasonviewdestroy[i],&identical);CHKERRQ(ierr);
    if (identical) PetscFunctionReturn(0);
  }
  PetscAssertFalse(ksp->numberreasonviews >= MAXKSPREASONVIEWS,PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Too many KSP reasonview set");
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
  PetscErrorCode ierr;
  PetscInt       i;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ksp,KSP_CLASSID,1);
  for (i=0; i<ksp->numberreasonviews; i++) {
    if (ksp->reasonviewdestroy[i]) {
      ierr = (*ksp->reasonviewdestroy[i])(&ksp->reasonviewcontext[i]);CHKERRQ(ierr);
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
  PetscErrorCode    ierr;
  PetscInt          i;

  PetscFunctionBegin;

  /* Call all user-provided reason review routines */
  for (i=0; i<ksp->numberreasonviews; i++) {
    ierr = (*ksp->reasonview[i])(ksp,ksp->reasonviewcontext[i]);CHKERRQ(ierr);
  }

  /* Call the default PETSc routine */
  ierr   = PetscOptionsGetViewer(PetscObjectComm((PetscObject)ksp),((PetscObject)ksp)->options,((PetscObject)ksp)->prefix,"-ksp_converged_reason",&viewer,&format,&flg);CHKERRQ(ierr);
  if (flg) {
    ierr = PetscViewerPushFormat(viewer,format);CHKERRQ(ierr);
    ierr = KSPConvergedReasonView(ksp, viewer);CHKERRQ(ierr);
    ierr = PetscViewerPopFormat(viewer);CHKERRQ(ierr);
    ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);
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
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  ierr = KSPGetOptionsPrefix(ksp, &prefix);CHKERRQ(ierr);
  ierr = KSPGetIterationNumber(ksp, &its);CHKERRQ(ierr);
  ierr = KSPComputeConvergenceRate(ksp, &rrate, &rRsq, &erate, &eRsq);CHKERRQ(ierr);
  if (!viewer) viewer = PETSC_VIEWER_STDOUT_(PetscObjectComm((PetscObject)ksp));
  ierr = PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&isAscii);CHKERRQ(ierr);
  if (isAscii) {
    ierr = PetscViewerGetFormat(viewer, &format);CHKERRQ(ierr);
    ierr = PetscViewerASCIIAddTab(viewer,((PetscObject)ksp)->tablevel);CHKERRQ(ierr);
    if (ksp->reason > 0) {
      if (prefix) {ierr = PetscViewerASCIIPrintf(viewer, "Linear %s solve converged due to %s iterations %D", prefix, reason, its);CHKERRQ(ierr);}
      else        {ierr = PetscViewerASCIIPrintf(viewer, "Linear solve converged due to %s iterations %D", reason, its);CHKERRQ(ierr);}
      ierr = PetscViewerASCIIUseTabs(viewer, PETSC_FALSE);CHKERRQ(ierr);
      if (rRsq >= 0.0) {ierr = PetscViewerASCIIPrintf(viewer, " res rate %g R^2 %g", rrate, rRsq);CHKERRQ(ierr);}
      if (eRsq >= 0.0) {ierr = PetscViewerASCIIPrintf(viewer, " error rate %g R^2 %g", erate, eRsq);CHKERRQ(ierr);}
      ierr = PetscViewerASCIIPrintf(viewer, "\n");CHKERRQ(ierr);
      ierr = PetscViewerASCIIUseTabs(viewer, PETSC_TRUE);CHKERRQ(ierr);
    } else if (ksp->reason <= 0) {
      if (prefix) {ierr = PetscViewerASCIIPrintf(viewer, "Linear %s solve did not converge due to %s iterations %D", prefix, reason, its);CHKERRQ(ierr);}
      else        {ierr = PetscViewerASCIIPrintf(viewer, "Linear solve did not converge due to %s iterations %D", reason, its);CHKERRQ(ierr);}
      ierr = PetscViewerASCIIUseTabs(viewer, PETSC_FALSE);CHKERRQ(ierr);
      if (rRsq >= 0.0) {ierr = PetscViewerASCIIPrintf(viewer, " res rate %g R^2 %g", rrate, rRsq);CHKERRQ(ierr);}
      if (eRsq >= 0.0) {ierr = PetscViewerASCIIPrintf(viewer, " error rate %g R^2 %g", erate, eRsq);CHKERRQ(ierr);}
      ierr = PetscViewerASCIIPrintf(viewer, "\n");CHKERRQ(ierr);
      ierr = PetscViewerASCIIUseTabs(viewer, PETSC_TRUE);CHKERRQ(ierr);
      if (ksp->reason == KSP_DIVERGED_PC_FAILED) {
        PCFailedReason reason;
        ierr = PCGetFailedReason(ksp->pc,&reason);CHKERRQ(ierr);
        ierr = PetscViewerASCIIPrintf(viewer,"               PC failed due to %s \n",PCFailedReasons[reason]);CHKERRQ(ierr);
      }
    }
    ierr = PetscViewerASCIISubtractTab(viewer,((PetscObject)ksp)->tablevel);CHKERRQ(ierr);
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
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MPI_Comm_rank(PetscObjectComm((PetscObject) ksp), &rank);CHKERRMPI(ierr);
  ierr = PetscObjectTypeCompare((PetscObject) viewer, PETSCVIEWERASCII, &isascii);CHKERRQ(ierr);
  ierr = PetscObjectTypeCompare((PetscObject) viewer, PETSCVIEWERDRAW,  &isdraw);CHKERRQ(ierr);
  if (isExplicit) {
    ierr = VecGetSize(ksp->vec_sol,&n);CHKERRQ(ierr);
    ierr = PetscMalloc2(n, &r, n, &c);CHKERRQ(ierr);
    ierr = KSPComputeEigenvaluesExplicitly(ksp, n, r, c);CHKERRQ(ierr);
    neig = n;
  } else {
    PetscInt nits;

    ierr = KSPGetIterationNumber(ksp, &nits);CHKERRQ(ierr);
    n    = nits+2;
    if (!nits) {ierr = PetscViewerASCIIPrintf(viewer, "Zero iterations in solver, cannot approximate any eigenvalues\n");CHKERRQ(ierr);PetscFunctionReturn(0);}
    ierr = PetscMalloc2(n, &r, n, &c);CHKERRQ(ierr);
    ierr = KSPComputeEigenvalues(ksp, n, r, c, &neig);CHKERRQ(ierr);
  }
  if (isascii) {
    ierr = PetscViewerASCIIPrintf(viewer, "%s computed eigenvalues\n", isExplicit ? "Explicitly" : "Iteratively");CHKERRQ(ierr);
    for (i = 0; i < neig; ++i) {
      if (c[i] >= 0.0) {ierr = PetscViewerASCIIPrintf(viewer, "%g + %gi\n", (double) r[i],  (double) c[i]);CHKERRQ(ierr);}
      else             {ierr = PetscViewerASCIIPrintf(viewer, "%g - %gi\n", (double) r[i], -(double) c[i]);CHKERRQ(ierr);}
    }
  } else if (isdraw && rank == 0) {
    PetscDraw   draw;
    PetscDrawSP drawsp;

    if (format == PETSC_VIEWER_DRAW_CONTOUR) {
      ierr = KSPPlotEigenContours_Private(ksp,neig,r,c);CHKERRQ(ierr);
    } else {
      if (!ksp->eigviewer) {ierr = PetscViewerDrawOpen(PETSC_COMM_SELF,NULL,isExplicit ? "Explicitly Computed Eigenvalues" : "Iteratively Computed Eigenvalues",PETSC_DECIDE,PETSC_DECIDE,400,400,&ksp->eigviewer);CHKERRQ(ierr);}
      ierr = PetscViewerDrawGetDraw(ksp->eigviewer,0,&draw);CHKERRQ(ierr);
      ierr = PetscDrawSPCreate(draw,1,&drawsp);CHKERRQ(ierr);
      ierr = PetscDrawSPReset(drawsp);CHKERRQ(ierr);
      for (i = 0; i < neig; ++i) {ierr = PetscDrawSPAddPoint(drawsp,r+i,c+i);CHKERRQ(ierr);}
      ierr = PetscDrawSPDraw(drawsp,PETSC_TRUE);CHKERRQ(ierr);
      ierr = PetscDrawSPSave(drawsp);CHKERRQ(ierr);
      ierr = PetscDrawSPDestroy(&drawsp);CHKERRQ(ierr);
    }
  }
  ierr = PetscFree2(r, c);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode KSPViewSingularvalues_Internal(KSP ksp, PetscViewer viewer, PetscViewerFormat format)
{
  PetscReal      smax, smin;
  PetscInt       nits;
  PetscBool      isascii;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscObjectTypeCompare((PetscObject) viewer, PETSCVIEWERASCII, &isascii);CHKERRQ(ierr);
  ierr = KSPGetIterationNumber(ksp, &nits);CHKERRQ(ierr);
  if (!nits) {ierr = PetscViewerASCIIPrintf(viewer, "Zero iterations in solver, cannot approximate any singular values\n");CHKERRQ(ierr);PetscFunctionReturn(0);}
  ierr = KSPComputeExtremeSingularValues(ksp, &smax, &smin);CHKERRQ(ierr);
  if (isascii) {ierr = PetscViewerASCIIPrintf(viewer, "Iteratively computed extreme singular values: max %g min %g max/min %g\n",(double)smax,(double)smin,(double)(smax/smin));CHKERRQ(ierr);}
  PetscFunctionReturn(0);
}

static PetscErrorCode KSPViewFinalResidual_Internal(KSP ksp, PetscViewer viewer, PetscViewerFormat format)
{
  PetscBool      isascii;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscObjectTypeCompare((PetscObject) viewer, PETSCVIEWERASCII, &isascii);CHKERRQ(ierr);
  PetscAssertFalse(ksp->dscale && !ksp->dscalefix,PetscObjectComm((PetscObject) ksp), PETSC_ERR_ARG_WRONGSTATE, "Cannot compute final scale with -ksp_diagonal_scale except also with -ksp_diagonal_scale_fix");
  if (isascii) {
    Mat       A;
    Vec       t;
    PetscReal norm;

    ierr = PCGetOperators(ksp->pc, &A, NULL);CHKERRQ(ierr);
    ierr = VecDuplicate(ksp->vec_rhs, &t);CHKERRQ(ierr);
    ierr = KSP_MatMult(ksp, A, ksp->vec_sol, t);CHKERRQ(ierr);
    ierr = VecAYPX(t, -1.0, ksp->vec_rhs);CHKERRQ(ierr);
    ierr = VecNorm(t, NORM_2, &norm);CHKERRQ(ierr);
    ierr = VecDestroy(&t);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer, "KSP final norm of residual %g\n", (double) norm);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode KSPMonitorPauseFinal_Internal(KSP ksp)
{
  PetscInt       i;
  PetscErrorCode ierr;

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
      ierr = PetscDrawLGGetDraw(vf->lg, &draw);CHKERRQ(ierr);
      ierr = PetscDrawGetPause(draw, &lpause);CHKERRQ(ierr);
      ierr = PetscDrawSetPause(draw, -1.0);CHKERRQ(ierr);
      ierr = PetscDrawPause(draw);CHKERRQ(ierr);
      ierr = PetscDrawSetPause(draw, lpause);CHKERRQ(ierr);
    } else {
      PetscBool isdraw;

      if (!PetscCheckPointer(vf->viewer, PETSC_OBJECT)) continue;
      if (((PetscObject) vf->viewer)->classid != PETSC_VIEWER_CLASSID) continue;
      ierr = PetscObjectTypeCompare((PetscObject) vf->viewer, PETSCVIEWERDRAW, &isdraw);CHKERRQ(ierr);
      if (!isdraw) continue;
      ierr = PetscViewerDrawGetDraw(vf->viewer, 0, &draw);CHKERRQ(ierr);
      ierr = PetscDrawGetPause(draw, &lpause);CHKERRQ(ierr);
      ierr = PetscDrawSetPause(draw, -1.0);CHKERRQ(ierr);
      ierr = PetscDrawPause(draw);CHKERRQ(ierr);
      ierr = PetscDrawSetPause(draw, lpause);CHKERRQ(ierr);
    }
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode KSPSolve_Private(KSP ksp,Vec b,Vec x)
{
  PetscErrorCode  ierr;
  PetscBool       flg = PETSC_FALSE,inXisinB = PETSC_FALSE,guess_zero;
  Mat             mat,pmat;
  MPI_Comm        comm;
  MatNullSpace    nullsp;
  Vec             btmp,vec_rhs = NULL;

  PetscFunctionBegin;
  level++;
  comm = PetscObjectComm((PetscObject)ksp);
  if (x && x == b) {
    PetscAssertFalse(!ksp->guess_zero,comm,PETSC_ERR_ARG_INCOMP,"Cannot use x == b with nonzero initial guess");
    ierr     = VecDuplicate(b,&x);CHKERRQ(ierr);
    inXisinB = PETSC_TRUE;
  }
  if (b) {
    ierr         = PetscObjectReference((PetscObject)b);CHKERRQ(ierr);
    ierr         = VecDestroy(&ksp->vec_rhs);CHKERRQ(ierr);
    ksp->vec_rhs = b;
  }
  if (x) {
    ierr         = PetscObjectReference((PetscObject)x);CHKERRQ(ierr);
    ierr         = VecDestroy(&ksp->vec_sol);CHKERRQ(ierr);
    ksp->vec_sol = x;
  }

  if (ksp->viewPre) {ierr = ObjectView((PetscObject) ksp, ksp->viewerPre, ksp->formatPre);CHKERRQ(ierr);}

  if (ksp->presolve) {ierr = (*ksp->presolve)(ksp,ksp->vec_rhs,ksp->vec_sol,ksp->prectx);CHKERRQ(ierr);}

  /* reset the residual history list if requested */
  if (ksp->res_hist_reset) ksp->res_hist_len = 0;
  if (ksp->err_hist_reset) ksp->err_hist_len = 0;

  if (ksp->guess) {
    PetscObjectState ostate,state;

    ierr = KSPGuessSetUp(ksp->guess);CHKERRQ(ierr);
    ierr = PetscObjectStateGet((PetscObject)ksp->vec_sol,&ostate);CHKERRQ(ierr);
    ierr = KSPGuessFormGuess(ksp->guess,ksp->vec_rhs,ksp->vec_sol);CHKERRQ(ierr);
    ierr = PetscObjectStateGet((PetscObject)ksp->vec_sol,&state);CHKERRQ(ierr);
    if (state != ostate) {
      ksp->guess_zero = PETSC_FALSE;
    } else {
      ierr = PetscInfo(ksp,"Using zero initial guess since the KSPGuess object did not change the vector\n");CHKERRQ(ierr);
      ksp->guess_zero = PETSC_TRUE;
    }
  }

  /* KSPSetUp() scales the matrix if needed */
  ierr = KSPSetUp(ksp);CHKERRQ(ierr);
  ierr = KSPSetUpOnBlocks(ksp);CHKERRQ(ierr);

  ierr = VecSetErrorIfLocked(ksp->vec_sol,3);CHKERRQ(ierr);

  ierr = PetscLogEventBegin(KSP_Solve,ksp,ksp->vec_rhs,ksp->vec_sol,0);CHKERRQ(ierr);
  ierr = PCGetOperators(ksp->pc,&mat,&pmat);CHKERRQ(ierr);
  /* diagonal scale RHS if called for */
  if (ksp->dscale) {
    ierr = VecPointwiseMult(ksp->vec_rhs,ksp->vec_rhs,ksp->diagonal);CHKERRQ(ierr);
    /* second time in, but matrix was scaled back to original */
    if (ksp->dscalefix && ksp->dscalefix2) {
      Mat mat,pmat;

      ierr = PCGetOperators(ksp->pc,&mat,&pmat);CHKERRQ(ierr);
      ierr = MatDiagonalScale(pmat,ksp->diagonal,ksp->diagonal);CHKERRQ(ierr);
      if (mat != pmat) {ierr = MatDiagonalScale(mat,ksp->diagonal,ksp->diagonal);CHKERRQ(ierr);}
    }

    /* scale initial guess */
    if (!ksp->guess_zero) {
      if (!ksp->truediagonal) {
        ierr = VecDuplicate(ksp->diagonal,&ksp->truediagonal);CHKERRQ(ierr);
        ierr = VecCopy(ksp->diagonal,ksp->truediagonal);CHKERRQ(ierr);
        ierr = VecReciprocal(ksp->truediagonal);CHKERRQ(ierr);
      }
      ierr = VecPointwiseMult(ksp->vec_sol,ksp->vec_sol,ksp->truediagonal);CHKERRQ(ierr);
    }
  }
  ierr = PCPreSolve(ksp->pc,ksp);CHKERRQ(ierr);

  if (ksp->guess_zero) { ierr = VecSet(ksp->vec_sol,0.0);CHKERRQ(ierr);}
  if (ksp->guess_knoll) { /* The Knoll trick is independent on the KSPGuess specified */
    ierr            = PCApply(ksp->pc,ksp->vec_rhs,ksp->vec_sol);CHKERRQ(ierr);
    ierr            = KSP_RemoveNullSpace(ksp,ksp->vec_sol);CHKERRQ(ierr);
    ksp->guess_zero = PETSC_FALSE;
  }

  /* can we mark the initial guess as zero for this solve? */
  guess_zero = ksp->guess_zero;
  if (!ksp->guess_zero) {
    PetscReal norm;

    ierr = VecNormAvailable(ksp->vec_sol,NORM_2,&flg,&norm);CHKERRQ(ierr);
    if (flg && !norm) ksp->guess_zero = PETSC_TRUE;
  }
  if (ksp->transpose_solve) {
    ierr = MatGetNullSpace(pmat,&nullsp);CHKERRQ(ierr);
  } else {
    ierr = MatGetTransposeNullSpace(pmat,&nullsp);CHKERRQ(ierr);
  }
  if (nullsp) {
    ierr = VecDuplicate(ksp->vec_rhs,&btmp);CHKERRQ(ierr);
    ierr = VecCopy(ksp->vec_rhs,btmp);CHKERRQ(ierr);
    ierr = MatNullSpaceRemove(nullsp,btmp);CHKERRQ(ierr);
    vec_rhs      = ksp->vec_rhs;
    ksp->vec_rhs = btmp;
  }
  ierr = VecLockReadPush(ksp->vec_rhs);CHKERRQ(ierr);
  if (ksp->reason == KSP_DIVERGED_PC_FAILED) {
    ierr = VecSetInf(ksp->vec_sol);CHKERRQ(ierr);
  }
  ierr = (*ksp->ops->solve)(ksp);CHKERRQ(ierr);
  ierr  = KSPMonitorPauseFinal_Internal(ksp);CHKERRQ(ierr);

  ierr = VecLockReadPop(ksp->vec_rhs);CHKERRQ(ierr);
  if (nullsp) {
    ksp->vec_rhs = vec_rhs;
    ierr = VecDestroy(&btmp);CHKERRQ(ierr);
  }

  ksp->guess_zero = guess_zero;

  PetscAssertFalse(!ksp->reason,comm,PETSC_ERR_PLIB,"Internal error, solver returned without setting converged reason");
  ksp->totalits += ksp->its;

  ierr = KSPConvergedReasonViewFromOptions(ksp);CHKERRQ(ierr);

  if (ksp->viewRate) {
    ierr = PetscViewerPushFormat(ksp->viewerRate,ksp->formatRate);CHKERRQ(ierr);
    ierr = KSPConvergedRateView(ksp, ksp->viewerRate);CHKERRQ(ierr);
    ierr = PetscViewerPopFormat(ksp->viewerRate);CHKERRQ(ierr);
  }
  ierr = PCPostSolve(ksp->pc,ksp);CHKERRQ(ierr);

  /* diagonal scale solution if called for */
  if (ksp->dscale) {
    ierr = VecPointwiseMult(ksp->vec_sol,ksp->vec_sol,ksp->diagonal);CHKERRQ(ierr);
    /* unscale right hand side and matrix */
    if (ksp->dscalefix) {
      Mat mat,pmat;

      ierr = VecReciprocal(ksp->diagonal);CHKERRQ(ierr);
      ierr = VecPointwiseMult(ksp->vec_rhs,ksp->vec_rhs,ksp->diagonal);CHKERRQ(ierr);
      ierr = PCGetOperators(ksp->pc,&mat,&pmat);CHKERRQ(ierr);
      ierr = MatDiagonalScale(pmat,ksp->diagonal,ksp->diagonal);CHKERRQ(ierr);
      if (mat != pmat) {ierr = MatDiagonalScale(mat,ksp->diagonal,ksp->diagonal);CHKERRQ(ierr);}
      ierr            = VecReciprocal(ksp->diagonal);CHKERRQ(ierr);
      ksp->dscalefix2 = PETSC_TRUE;
    }
  }
  ierr = PetscLogEventEnd(KSP_Solve,ksp,ksp->vec_rhs,ksp->vec_sol,0);CHKERRQ(ierr);
  if (ksp->guess) {
    ierr = KSPGuessUpdate(ksp->guess,ksp->vec_rhs,ksp->vec_sol);CHKERRQ(ierr);
  }
  if (ksp->postsolve) {
    ierr = (*ksp->postsolve)(ksp,ksp->vec_rhs,ksp->vec_sol,ksp->postctx);CHKERRQ(ierr);
  }

  ierr = PCGetOperators(ksp->pc,&mat,&pmat);CHKERRQ(ierr);
  if (ksp->viewEV)       {ierr = KSPViewEigenvalues_Internal(ksp, PETSC_FALSE, ksp->viewerEV,    ksp->formatEV);CHKERRQ(ierr);}
  if (ksp->viewEVExp)    {ierr = KSPViewEigenvalues_Internal(ksp, PETSC_TRUE,  ksp->viewerEVExp, ksp->formatEVExp);CHKERRQ(ierr);}
  if (ksp->viewSV)       {ierr = KSPViewSingularvalues_Internal(ksp, ksp->viewerSV, ksp->formatSV);CHKERRQ(ierr);}
  if (ksp->viewFinalRes) {ierr = KSPViewFinalResidual_Internal(ksp, ksp->viewerFinalRes, ksp->formatFinalRes);CHKERRQ(ierr);}
  if (ksp->viewMat)      {ierr = ObjectView((PetscObject) mat,           ksp->viewerMat,    ksp->formatMat);CHKERRQ(ierr);}
  if (ksp->viewPMat)     {ierr = ObjectView((PetscObject) pmat,          ksp->viewerPMat,   ksp->formatPMat);CHKERRQ(ierr);}
  if (ksp->viewRhs)      {ierr = ObjectView((PetscObject) ksp->vec_rhs,  ksp->viewerRhs,    ksp->formatRhs);CHKERRQ(ierr);}
  if (ksp->viewSol)      {ierr = ObjectView((PetscObject) ksp->vec_sol,  ksp->viewerSol,    ksp->formatSol);CHKERRQ(ierr);}
  if (ksp->view)         {ierr = ObjectView((PetscObject) ksp,           ksp->viewer,       ksp->format);CHKERRQ(ierr);}
  if (ksp->viewDScale)   {ierr = ObjectView((PetscObject) ksp->diagonal, ksp->viewerDScale, ksp->formatDScale);CHKERRQ(ierr);}
  if (ksp->viewMatExp)   {
    Mat A, B;

    ierr = PCGetOperators(ksp->pc, &A, NULL);CHKERRQ(ierr);
    if (ksp->transpose_solve) {
      Mat AT;

      ierr = MatCreateTranspose(A, &AT);CHKERRQ(ierr);
      ierr = MatComputeOperator(AT, MATAIJ, &B);CHKERRQ(ierr);
      ierr = MatDestroy(&AT);CHKERRQ(ierr);
    } else {
      ierr = MatComputeOperator(A, MATAIJ, &B);CHKERRQ(ierr);
    }
    ierr = ObjectView((PetscObject) B, ksp->viewerMatExp, ksp->formatMatExp);CHKERRQ(ierr);
    ierr = MatDestroy(&B);CHKERRQ(ierr);
  }
  if (ksp->viewPOpExp)   {
    Mat B;

    ierr = KSPComputeOperator(ksp, MATAIJ, &B);CHKERRQ(ierr);
    ierr = ObjectView((PetscObject) B, ksp->viewerPOpExp, ksp->formatPOpExp);CHKERRQ(ierr);
    ierr = MatDestroy(&B);CHKERRQ(ierr);
  }

  if (inXisinB) {
    ierr = VecCopy(x,b);CHKERRQ(ierr);
    ierr = VecDestroy(&x);CHKERRQ(ierr);
  }
  ierr = PetscObjectSAWsBlock((PetscObject)ksp);CHKERRQ(ierr);
  if (ksp->errorifnotconverged && ksp->reason < 0 && ((level == 1) || (ksp->reason != KSP_DIVERGED_ITS))) {
    if (ksp->reason == KSP_DIVERGED_PC_FAILED) {
      PCFailedReason reason;
      ierr = PCGetFailedReason(ksp->pc,&reason);CHKERRQ(ierr);
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
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ksp,KSP_CLASSID,1);
  if (b) PetscValidHeaderSpecific(b,VEC_CLASSID,2);
  if (x) PetscValidHeaderSpecific(x,VEC_CLASSID,3);
  ksp->transpose_solve = PETSC_FALSE;
  ierr = KSPSolve_Private(ksp,b,x);CHKERRQ(ierr);
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
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ksp,KSP_CLASSID,1);
  if (b) PetscValidHeaderSpecific(b,VEC_CLASSID,2);
  if (x) PetscValidHeaderSpecific(x,VEC_CLASSID,3);
  if (ksp->transpose.use_explicittranspose) {
    Mat J,Jpre;
    ierr = KSPGetOperators(ksp,&J,&Jpre);CHKERRQ(ierr);
    if (!ksp->transpose.reuse_transpose) {
      ierr = MatTranspose(J,MAT_INITIAL_MATRIX,&ksp->transpose.AT);CHKERRQ(ierr);
      if (J != Jpre) {
        ierr = MatTranspose(Jpre,MAT_INITIAL_MATRIX,&ksp->transpose.BT);CHKERRQ(ierr);
      }
      ksp->transpose.reuse_transpose = PETSC_TRUE;
    } else {
      ierr = MatTranspose(J,MAT_REUSE_MATRIX,&ksp->transpose.AT);CHKERRQ(ierr);
      if (J != Jpre) {
        ierr = MatTranspose(Jpre,MAT_REUSE_MATRIX,&ksp->transpose.BT);CHKERRQ(ierr);
      }
    }
    if (J == Jpre && ksp->transpose.BT != ksp->transpose.AT) {
      ierr = PetscObjectReference((PetscObject)ksp->transpose.AT);CHKERRQ(ierr);
      ksp->transpose.BT = ksp->transpose.AT;
    }
    ierr = KSPSetOperators(ksp,ksp->transpose.AT,ksp->transpose.BT);CHKERRQ(ierr);
  } else {
    ksp->transpose_solve = PETSC_TRUE;
  }
  ierr = KSPSolve_Private(ksp,b,x);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode KSPViewFinalMatResidual_Internal(KSP ksp, Mat B, Mat X, PetscViewer viewer, PetscViewerFormat format, PetscInt shift)
{
  Mat            A, R;
  PetscReal      *norms;
  PetscInt       i, N;
  PetscBool      flg;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscObjectTypeCompare((PetscObject)viewer, PETSCVIEWERASCII, &flg);CHKERRQ(ierr);
  if (flg) {
    ierr = PCGetOperators(ksp->pc, &A, NULL);CHKERRQ(ierr);
    ierr = MatMatMult(A, X, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &R);CHKERRQ(ierr);
    ierr = MatAYPX(R, -1.0, B, SAME_NONZERO_PATTERN);CHKERRQ(ierr);
    ierr = MatGetSize(R, NULL, &N);CHKERRQ(ierr);
    ierr = PetscMalloc1(N, &norms);CHKERRQ(ierr);
    ierr = MatGetColumnNorms(R, NORM_2, norms);CHKERRQ(ierr);
    ierr = MatDestroy(&R);CHKERRQ(ierr);
    for (i = 0; i < N; ++i) {
      ierr = PetscViewerASCIIPrintf(viewer, "%s #%D %g\n", i == 0 ? "KSP final norm of residual" : "                          ", shift + i, (double)norms[i]);CHKERRQ(ierr);
    }
    ierr = PetscFree(norms);CHKERRQ(ierr);
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
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ksp, KSP_CLASSID, 1);
  PetscValidHeaderSpecific(B, MAT_CLASSID, 2);
  PetscValidHeaderSpecific(X, MAT_CLASSID, 3);
  PetscCheckSameComm(ksp, 1, B, 2);
  PetscCheckSameComm(ksp, 1, X, 3);
  PetscAssertFalse(!B->assembled,PETSC_COMM_SELF, PETSC_ERR_ARG_WRONGSTATE, "Not for unassembled matrix");
  MatCheckPreallocated(X, 3);
  if (!X->assembled) {
    ierr = MatSetOption(X, MAT_NO_OFF_PROC_ENTRIES, PETSC_TRUE);CHKERRQ(ierr);
    ierr = MatAssemblyBegin(X, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatAssemblyEnd(X, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  }
  PetscAssertFalse(B == X,PetscObjectComm((PetscObject)ksp), PETSC_ERR_ARG_IDN, "B and X must be different matrices");
  ierr = KSPGetOperators(ksp, &A, &P);CHKERRQ(ierr);
  ierr = MatGetLocalSize(B, NULL, &n2);CHKERRQ(ierr);
  ierr = MatGetLocalSize(X, NULL, &n1);CHKERRQ(ierr);
  ierr = MatGetSize(B, NULL, &N2);CHKERRQ(ierr);
  ierr = MatGetSize(X, NULL, &N1);CHKERRQ(ierr);
  PetscAssertFalse(n1 != n2 || N1 != N2,PETSC_COMM_SELF, PETSC_ERR_ARG_SIZ, "Incompatible number of columns between block of right-hand sides (n,N) = (%D,%D) and block of solutions (n,N) = (%D,%D)", n2, N2, n1, N1);
  ierr = PetscObjectBaseTypeCompareAny((PetscObject)B, &match, MATSEQDENSE, MATMPIDENSE, "");CHKERRQ(ierr);
  PetscAssertFalse(!match,PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Provided block of right-hand sides not stored in a dense Mat");
  ierr = PetscObjectBaseTypeCompareAny((PetscObject)X, &match, MATSEQDENSE, MATMPIDENSE, "");CHKERRQ(ierr);
  PetscAssertFalse(!match,PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Provided block of solutions not stored in a dense Mat");
  ierr = KSPSetUp(ksp);CHKERRQ(ierr);
  ierr = KSPSetUpOnBlocks(ksp);CHKERRQ(ierr);
  if (ksp->ops->matsolve) {
    if (ksp->guess_zero) {
      ierr = MatZeroEntries(X);CHKERRQ(ierr);
    }
    ierr = PetscLogEventBegin(KSP_MatSolve, ksp, B, X, 0);CHKERRQ(ierr);
    ierr = KSPGetMatSolveBatchSize(ksp, &Bbn);CHKERRQ(ierr);
    /* by default, do a single solve with all columns */
    if (Bbn == PETSC_DECIDE) Bbn = N2;
    else PetscAssertFalse(Bbn < 1,PetscObjectComm((PetscObject)ksp), PETSC_ERR_ARG_OUTOFRANGE, "KSPMatSolve() batch size %D must be positive", Bbn);
    ierr = PetscInfo(ksp, "KSP type %s solving using batches of width at most %D\n", ((PetscObject)ksp)->type_name, Bbn);CHKERRQ(ierr);
    /* if -ksp_matsolve_batch_size is greater than the actual number of columns, do a single solve with all columns */
    if (Bbn >= N2) {
      ierr = (*ksp->ops->matsolve)(ksp, B, X);CHKERRQ(ierr);
      if (ksp->viewFinalRes) {
        ierr = KSPViewFinalMatResidual_Internal(ksp, B, X, ksp->viewerFinalRes, ksp->formatFinalRes, 0);CHKERRQ(ierr);
      }

      ierr = KSPConvergedReasonViewFromOptions(ksp);CHKERRQ(ierr);

      if (ksp->viewRate) {
        ierr = PetscViewerPushFormat(ksp->viewerRate,PETSC_VIEWER_DEFAULT);CHKERRQ(ierr);
        ierr = KSPConvergedRateView(ksp, ksp->viewerRate);CHKERRQ(ierr);
        ierr = PetscViewerPopFormat(ksp->viewerRate);CHKERRQ(ierr);
      }
    } else {
      for (n2 = 0; n2 < N2; n2 += Bbn) {
        ierr = MatDenseGetSubMatrix(B, n2, PetscMin(n2+Bbn, N2), &vB);CHKERRQ(ierr);
        ierr = MatDenseGetSubMatrix(X, n2, PetscMin(n2+Bbn, N2), &vX);CHKERRQ(ierr);
        ierr = (*ksp->ops->matsolve)(ksp, vB, vX);CHKERRQ(ierr);
        if (ksp->viewFinalRes) {
          ierr = KSPViewFinalMatResidual_Internal(ksp, vB, vX, ksp->viewerFinalRes, ksp->formatFinalRes, n2);CHKERRQ(ierr);
        }

        ierr = KSPConvergedReasonViewFromOptions(ksp);CHKERRQ(ierr);

        if (ksp->viewRate) {
          ierr = PetscViewerPushFormat(ksp->viewerRate,PETSC_VIEWER_DEFAULT);CHKERRQ(ierr);
          ierr = KSPConvergedRateView(ksp, ksp->viewerRate);CHKERRQ(ierr);
          ierr = PetscViewerPopFormat(ksp->viewerRate);CHKERRQ(ierr);
        }
        ierr = MatDenseRestoreSubMatrix(B, &vB);CHKERRQ(ierr);
        ierr = MatDenseRestoreSubMatrix(X, &vX);CHKERRQ(ierr);
      }
    }
    if (ksp->viewMat)  {ierr = ObjectView((PetscObject) A, ksp->viewerMat, ksp->formatMat);CHKERRQ(ierr);}
    if (ksp->viewPMat) {ierr = ObjectView((PetscObject) P, ksp->viewerPMat,ksp->formatPMat);CHKERRQ(ierr);}
    if (ksp->viewRhs)  {ierr = ObjectView((PetscObject) B, ksp->viewerRhs, ksp->formatRhs);CHKERRQ(ierr);}
    if (ksp->viewSol)  {ierr = ObjectView((PetscObject) X, ksp->viewerSol, ksp->formatSol);CHKERRQ(ierr);}
    if (ksp->view) {
      ierr = KSPView(ksp, ksp->viewer);CHKERRQ(ierr);
    }
    ierr = PetscLogEventEnd(KSP_MatSolve, ksp, B, X, 0);CHKERRQ(ierr);
  } else {
    ierr = PetscInfo(ksp, "KSP type %s solving column by column\n", ((PetscObject)ksp)->type_name);CHKERRQ(ierr);
    for (n2 = 0; n2 < N2; ++n2) {
      ierr = MatDenseGetColumnVecRead(B, n2, &cb);CHKERRQ(ierr);
      ierr = MatDenseGetColumnVecWrite(X, n2, &cx);CHKERRQ(ierr);
      ierr = KSPSolve(ksp, cb, cx);CHKERRQ(ierr);
      ierr = MatDenseRestoreColumnVecWrite(X, n2, &cx);CHKERRQ(ierr);
      ierr = MatDenseRestoreColumnVecRead(B, n2, &cb);CHKERRQ(ierr);
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
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (ksp) PetscValidHeaderSpecific(ksp,KSP_CLASSID,1);
  if (!ksp) PetscFunctionReturn(0);
  ierr = PetscViewerDestroy(&ksp->viewer);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&ksp->viewerPre);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&ksp->viewerRate);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&ksp->viewerMat);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&ksp->viewerPMat);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&ksp->viewerRhs);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&ksp->viewerSol);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&ksp->viewerMatExp);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&ksp->viewerEV);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&ksp->viewerSV);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&ksp->viewerEVExp);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&ksp->viewerFinalRes);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&ksp->viewerPOpExp);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&ksp->viewerDScale);CHKERRQ(ierr);
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
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (ksp) PetscValidHeaderSpecific(ksp,KSP_CLASSID,1);
  if (!ksp) PetscFunctionReturn(0);
  if (ksp->ops->reset) {
    ierr = (*ksp->ops->reset)(ksp);CHKERRQ(ierr);
  }
  if (ksp->pc) {ierr = PCReset(ksp->pc);CHKERRQ(ierr);}
  if (ksp->guess) {
    KSPGuess guess = ksp->guess;
    if (guess->ops->reset) { ierr = (*guess->ops->reset)(guess);CHKERRQ(ierr); }
  }
  ierr = VecDestroyVecs(ksp->nwork,&ksp->work);CHKERRQ(ierr);
  ierr = VecDestroy(&ksp->vec_rhs);CHKERRQ(ierr);
  ierr = VecDestroy(&ksp->vec_sol);CHKERRQ(ierr);
  ierr = VecDestroy(&ksp->diagonal);CHKERRQ(ierr);
  ierr = VecDestroy(&ksp->truediagonal);CHKERRQ(ierr);

  ierr = KSPResetViewers(ksp);CHKERRQ(ierr);

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
  PetscErrorCode ierr;
  PC             pc;

  PetscFunctionBegin;
  if (!*ksp) PetscFunctionReturn(0);
  PetscValidHeaderSpecific((*ksp),KSP_CLASSID,1);
  if (--((PetscObject)(*ksp))->refct > 0) {*ksp = NULL; PetscFunctionReturn(0);}

  ierr = PetscObjectSAWsViewOff((PetscObject)*ksp);CHKERRQ(ierr);

  /*
   Avoid a cascading call to PCReset(ksp->pc) from the following call:
   PCReset() shouldn't be called from KSPDestroy() as it is unprotected by pc's
   refcount (and may be shared, e.g., by other ksps).
   */
  pc         = (*ksp)->pc;
  (*ksp)->pc = NULL;
  ierr       = KSPReset((*ksp));CHKERRQ(ierr);
  (*ksp)->pc = pc;
  if ((*ksp)->ops->destroy) {ierr = (*(*ksp)->ops->destroy)(*ksp);CHKERRQ(ierr);}

  if ((*ksp)->transpose.use_explicittranspose) {
    ierr = MatDestroy(&(*ksp)->transpose.AT);CHKERRQ(ierr);
    ierr = MatDestroy(&(*ksp)->transpose.BT);CHKERRQ(ierr);
    (*ksp)->transpose.reuse_transpose = PETSC_FALSE;
  }

  ierr = KSPGuessDestroy(&(*ksp)->guess);CHKERRQ(ierr);
  ierr = DMDestroy(&(*ksp)->dm);CHKERRQ(ierr);
  ierr = PCDestroy(&(*ksp)->pc);CHKERRQ(ierr);
  ierr = PetscFree((*ksp)->res_hist_alloc);CHKERRQ(ierr);
  ierr = PetscFree((*ksp)->err_hist_alloc);CHKERRQ(ierr);
  if ((*ksp)->convergeddestroy) {
    ierr = (*(*ksp)->convergeddestroy)((*ksp)->cnvP);CHKERRQ(ierr);
  }
  ierr = KSPMonitorCancel((*ksp));CHKERRQ(ierr);
  ierr = KSPConvergedReasonViewCancel((*ksp));CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&(*ksp)->eigviewer);CHKERRQ(ierr);
  ierr = PetscHeaderDestroy(ksp);CHKERRQ(ierr);
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
.   -ksp_pc_side <right,left,symmetric>

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
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ksp,KSP_CLASSID,1);
  PetscValidPointer(side,2);
  ierr  = KSPSetUpNorms_Private(ksp,PETSC_TRUE,&ksp->normtype,&ksp->pc_side);CHKERRQ(ierr);
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
    PetscAssertFalse(rtol < 0.0 || 1.0 <= rtol,PetscObjectComm((PetscObject)ksp),PETSC_ERR_ARG_OUTOFRANGE,"Relative tolerance %g must be non-negative and less than 1.0",(double)rtol);
    ksp->rtol = rtol;
  }
  if (abstol != PETSC_DEFAULT) {
    PetscAssertFalse(abstol < 0.0,PetscObjectComm((PetscObject)ksp),PETSC_ERR_ARG_OUTOFRANGE,"Absolute tolerance %g must be non-negative",(double)abstol);
    ksp->abstol = abstol;
  }
  if (dtol != PETSC_DEFAULT) {
    PetscAssertFalse(dtol < 0.0,PetscObjectComm((PetscObject)ksp),PETSC_ERR_ARG_OUTOFRANGE,"Divergence tolerance %g must be larger than 1.0",(double)dtol);
    ksp->divtol = dtol;
  }
  if (maxits != PETSC_DEFAULT) {
    PetscAssertFalse(maxits < 0,PetscObjectComm((PetscObject)ksp),PETSC_ERR_ARG_OUTOFRANGE,"Maximum number of iterations %D must be non-negative",maxits);
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
.  -ksp_initial_guess_nonzero : use nonzero initial guess; this takes an optional truth value (0/1/no/yes/true/false)

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
.  -ksp_error_if_not_converged : this takes an optional truth value (0/1/no/yes/true/false)

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
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ksp,KSP_CLASSID,1);
  if (pc) {
    PetscValidHeaderSpecific(pc,PC_CLASSID,2);
    PetscCheckSameComm(ksp,1,pc,2);
  }
  ierr    = PetscObjectReference((PetscObject)pc);CHKERRQ(ierr);
  ierr    = PCDestroy(&ksp->pc);CHKERRQ(ierr);
  ksp->pc = pc;
  ierr    = PetscLogObjectParent((PetscObject)ksp,(PetscObject)ksp->pc);CHKERRQ(ierr);
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
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ksp,KSP_CLASSID,1);
  PetscValidPointer(pc,2);
  if (!ksp->pc) {
    ierr = PCCreate(PetscObjectComm((PetscObject)ksp),&ksp->pc);CHKERRQ(ierr);
    ierr = PetscObjectIncrementTabLevel((PetscObject)ksp->pc,(PetscObject)ksp,0);CHKERRQ(ierr);
    ierr = PetscLogObjectParent((PetscObject)ksp,(PetscObject)ksp->pc);CHKERRQ(ierr);
    ierr = PetscObjectSetOptions((PetscObject)ksp->pc,((PetscObject)ksp)->options);CHKERRQ(ierr);
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
  PetscErrorCode ierr;

  PetscFunctionBegin;
  for (i=0; i<n; i++) {
    ierr = (*ksp->monitor[i])(ksp,it,rnorm,ksp->monitorcontext[i]);CHKERRQ(ierr);
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
  PetscErrorCode ierr;
  PetscBool      identical;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ksp,KSP_CLASSID,1);
  for (i=0; i<ksp->numbermonitors;i++) {
    ierr = PetscMonitorCompare((PetscErrorCode (*)(void))monitor,mctx,monitordestroy,(PetscErrorCode (*)(void))ksp->monitor[i],ksp->monitorcontext[i],ksp->monitordestroy[i],&identical);CHKERRQ(ierr);
    if (identical) PetscFunctionReturn(0);
  }
  PetscAssertFalse(ksp->numbermonitors >= MAXKSPMONITORS,PetscObjectComm((PetscObject)ksp),PETSC_ERR_ARG_OUTOFRANGE,"Too many KSP monitors set");
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
  PetscErrorCode ierr;
  PetscInt       i;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ksp,KSP_CLASSID,1);
  for (i=0; i<ksp->numbermonitors; i++) {
    if (ksp->monitordestroy[i]) {
      ierr = (*ksp->monitordestroy[i])(&ksp->monitorcontext[i]);CHKERRQ(ierr);
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
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ksp,KSP_CLASSID,1);

  ierr = PetscFree(ksp->res_hist_alloc);CHKERRQ(ierr);
  if (na != PETSC_DECIDE && na != PETSC_DEFAULT && a) {
    ksp->res_hist     = a;
    ksp->res_hist_max = na;
  } else {
    if (na != PETSC_DECIDE && na != PETSC_DEFAULT) ksp->res_hist_max = na;
    else                                           ksp->res_hist_max = 10000; /* like default ksp->max_it */
    ierr = PetscCalloc1(ksp->res_hist_max,&ksp->res_hist_alloc);CHKERRQ(ierr);

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
  if (na) *na = ksp->res_hist_len;
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
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ksp,KSP_CLASSID,1);

  ierr = PetscFree(ksp->err_hist_alloc);CHKERRQ(ierr);
  if (na != PETSC_DECIDE && na != PETSC_DEFAULT && a) {
    ksp->err_hist     = a;
    ksp->err_hist_max = na;
  } else {
    if (na != PETSC_DECIDE && na != PETSC_DEFAULT) ksp->err_hist_max = na;
    else                                           ksp->err_hist_max = 10000; /* like default ksp->max_it */
    ierr = PetscCalloc1(ksp->err_hist_max, &ksp->err_hist_alloc);CHKERRQ(ierr);

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
  if (na) *na = ksp->err_hist_len;
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
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (cr || rRsq) {
    ierr = KSPGetResidualHistory(ksp, &hist, &n);CHKERRQ(ierr);
    if (!n) {
      if (cr)   *cr   =  0.0;
      if (rRsq) *rRsq = -1.0;
    } else {
      ierr = PetscMalloc2(n, &x, n, &y);CHKERRQ(ierr);
      for (k = 0; k < n; ++k) {
        x[k] = k;
        y[k] = PetscLogReal(hist[k]);
        mean += y[k];
      }
      mean /= n;
      ierr = PetscLinearRegression(n, x, y, &slope, &intercept);CHKERRQ(ierr);
      for (k = 0; k < n; ++k) {
        res += PetscSqr(y[k] - (slope*x[k] + intercept));
        var += PetscSqr(y[k] - mean);
      }
      ierr = PetscFree2(x, y);CHKERRQ(ierr);
      if (cr)   *cr   = PetscExpReal(slope);
      if (rRsq) *rRsq = var < PETSC_MACHINE_EPSILON ? 0.0 : 1.0 - (res / var);
    }
  }
  if (ce || eRsq) {
    ierr = KSPGetErrorHistory(ksp, &hist, &n);CHKERRQ(ierr);
    if (!n) {
      if (ce)   *ce   =  0.0;
      if (eRsq) *eRsq = -1.0;
    } else {
      ierr = PetscMalloc2(n, &x, n, &y);CHKERRQ(ierr);
      for (k = 0; k < n; ++k) {
        x[k] = k;
        y[k] = PetscLogReal(hist[k]);
        mean += y[k];
      }
      mean /= n;
      ierr = PetscLinearRegression(n, x, y, &slope, &intercept);CHKERRQ(ierr);
      for (k = 0; k < n; ++k) {
        res += PetscSqr(y[k] - (slope*x[k] + intercept));
        var += PetscSqr(y[k] - mean);
      }
      ierr = PetscFree2(x, y);CHKERRQ(ierr);
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
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ksp,KSP_CLASSID,1);
  if (ksp->convergeddestroy) {
    ierr = (*ksp->convergeddestroy)(ksp->cnvP);CHKERRQ(ierr);
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
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ksp,KSP_CLASSID,1);
  PetscAssertFalse(!V && !v,PetscObjectComm((PetscObject)ksp),PETSC_ERR_ARG_WRONG,"Must provide either v or V");
  if (!V) V = &v;
  ierr = (*ksp->ops->buildsolution)(ksp,v,V);CHKERRQ(ierr);
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
  PetscErrorCode ierr;
  PetscBool      flag = PETSC_FALSE;
  Vec            w    = v,tt = t;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ksp,KSP_CLASSID,1);
  if (!w) {
    ierr = VecDuplicate(ksp->vec_rhs,&w);CHKERRQ(ierr);
    ierr = PetscLogObjectParent((PetscObject)ksp,(PetscObject)w);CHKERRQ(ierr);
  }
  if (!tt) {
    ierr = VecDuplicate(ksp->vec_sol,&tt);CHKERRQ(ierr); flag = PETSC_TRUE;
    ierr = PetscLogObjectParent((PetscObject)ksp,(PetscObject)tt);CHKERRQ(ierr);
  }
  ierr = (*ksp->ops->buildresidual)(ksp,tt,w,V);CHKERRQ(ierr);
  if (flag) {ierr = VecDestroy(&tt);CHKERRQ(ierr);}
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
  PetscValidPointer(scale,2);
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
  PetscValidPointer(fix,2);
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
  PetscErrorCode ierr;
  DM             dm;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ksp,KSP_CLASSID,1);
  ierr = KSPGetDM(ksp,&dm);CHKERRQ(ierr);
  ierr = DMKSPSetComputeOperators(dm,func,ctx);CHKERRQ(ierr);
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
  PetscErrorCode ierr;
  DM             dm;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ksp,KSP_CLASSID,1);
  ierr = KSPGetDM(ksp,&dm);CHKERRQ(ierr);
  ierr = DMKSPSetComputeRHS(dm,func,ctx);CHKERRQ(ierr);
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
  PetscErrorCode ierr;
  DM             dm;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ksp,KSP_CLASSID,1);
  ierr = KSPGetDM(ksp,&dm);CHKERRQ(ierr);
  ierr = DMKSPSetComputeInitialGuess(dm,func,ctx);CHKERRQ(ierr);
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
