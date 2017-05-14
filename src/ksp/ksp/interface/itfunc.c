
/*
      Interface KSP routines that the user calls.
*/

#include <petsc/private/kspimpl.h>   /*I "petscksp.h" I*/
#include <petscdm.h>

/*@
   KSPComputeExtremeSingularValues - Computes the extreme singular values
   for the preconditioned operator. Called after or during KSPSolve().

   Not Collective

   Input Parameter:
.  ksp - iterative context obtained from KSPCreate()

   Output Parameters:
.  emin, emax - extreme singular values

   Options Database Keys:
.  -ksp_compute_singularvalues - compute extreme singular values and print when KSPSolve completes.

   Notes:
   One must call KSPSetComputeSingularValues() before calling KSPSetUp()
   (or use the option -ksp_compute_eigenvalues) in order for this routine to work correctly.

   Many users may just want to use the monitoring routine
   KSPMonitorSingularValue() (which can be set with option -ksp_monitor_singular_value)
   to print the extreme singular values at each iteration of the linear solve.

   Estimates of the smallest singular value may be very inaccurate, especially if the Krylov method has not converged.
   The largest singular value is usually accurate to within a few percent if the method has converged, but is still not
   intended for eigenanalysis.

   Disable restarts if using KSPGMRES, otherwise this estimate will only be using those iterations after the last
   restart. See KSPGMRESSetRestart() for more details.

   Level: advanced

.keywords: KSP, compute, extreme, singular, values

.seealso: KSPSetComputeSingularValues(), KSPMonitorSingularValue(), KSPComputeEigenvalues()
@*/
PetscErrorCode  KSPComputeExtremeSingularValues(KSP ksp,PetscReal *emax,PetscReal *emin)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ksp,KSP_CLASSID,1);
  PetscValidScalarPointer(emax,2);
  PetscValidScalarPointer(emin,3);
  if (!ksp->calc_sings) SETERRQ(PetscObjectComm((PetscObject)ksp),4,"Singular values not requested before KSPSetUp()");

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

   Input Parameter:
+  ksp - iterative context obtained from KSPCreate()
-  n - size of arrays r and c. The number of eigenvalues computed (neig) will, in
       general, be less than this.

   Output Parameters:
+  r - real part of computed eigenvalues, provided by user with a dimension of at least n
.  c - complex part of computed eigenvalues, provided by user with a dimension of at least n
-  neig - actual number of eigenvalues computed (will be less than or equal to n)

   Options Database Keys:
+  -ksp_compute_eigenvalues - Prints eigenvalues to stdout
-  -ksp_plot_eigenvalues - Plots eigenvalues in an x-window display

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

.keywords: KSP, compute, extreme, singular, values

.seealso: KSPSetComputeSingularValues(), KSPMonitorSingularValue(), KSPComputeExtremeSingularValues()
@*/
PetscErrorCode  KSPComputeEigenvalues(KSP ksp,PetscInt n,PetscReal r[],PetscReal c[],PetscInt *neig)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ksp,KSP_CLASSID,1);
  if (n) PetscValidScalarPointer(r,3);
  if (n) PetscValidScalarPointer(c,4);
  if (n<0) SETERRQ(PetscObjectComm((PetscObject)ksp),PETSC_ERR_ARG_OUTOFRANGE,"Requested < 0 Eigenvalues");
  PetscValidIntPointer(neig,5);
  if (!ksp->calc_sings) SETERRQ(PetscObjectComm((PetscObject)ksp),4,"Eigenvalues not requested before KSPSetUp()");

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

   Input Parameter:
+  ksp   - iterative context obtained from KSPCreate()
.  ritz  - PETSC_TRUE or PETSC_FALSE for ritz pairs or harmonic Ritz pairs, respectively
.  small - PETSC_TRUE or PETSC_FALSE for smallest or largest (harmonic) Ritz values, respectively
.  nrit  - number of (harmonic) Ritz pairs to compute

   Output Parameters:
+  nrit  - actual number of computed (harmonic) Ritz pairs 
.  S     - multidimensional vector with Ritz vectors
.  tetar - real part of the Ritz values        
.  tetai - imaginary part of the Ritz values

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

.keywords: KSP, compute, ritz, values

.seealso: KSPSetComputeRitz()
@*/
PetscErrorCode  KSPComputeRitz(KSP ksp,PetscBool ritz,PetscBool small,PetscInt *nrit,Vec S[],PetscReal tetar[],PetscReal tetai[])
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ksp,KSP_CLASSID,1);
  if (!ksp->calc_ritz) SETERRQ(PetscObjectComm((PetscObject)ksp),4,"Ritz pairs not requested before KSPSetUp()");
  if (ksp->ops->computeritz) {ierr = (*ksp->ops->computeritz)(ksp,ritz,small,nrit,S,tetar,tetai);CHKERRQ(ierr);}
  PetscFunctionReturn(0);
}
/*@
   KSPSetUpOnBlocks - Sets up the preconditioner for each block in
   the block Jacobi, block Gauss-Seidel, and overlapping Schwarz
   methods.

   Collective on KSP

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

.keywords: KSP, setup, blocks

.seealso: PCSetUpOnBlocks(), KSPSetUp(), PCSetUp()
@*/
PetscErrorCode  KSPSetUpOnBlocks(KSP ksp)
{
  PetscErrorCode ierr;
  PCFailedReason pcreason;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ksp,KSP_CLASSID,1);
  if (!ksp->pc) {ierr = KSPGetPC(ksp,&ksp->pc);CHKERRQ(ierr);}
  ierr = PCSetUpOnBlocks(ksp->pc);CHKERRQ(ierr);
  ierr = PCGetSetUpFailedReason(ksp->pc,&pcreason);CHKERRQ(ierr); 
  if (pcreason) {
    ksp->reason = KSP_DIVERGED_PCSETUP_FAILED;
  }
  PetscFunctionReturn(0);
}

/*@
   KSPSetReusePreconditioner - reuse the current preconditioner, do not construct a new one even if the operator changes

   Collective on KSP

   Input Parameters:
+  ksp   - iterative context obtained from KSPCreate()
-  flag - PETSC_TRUE to reuse the current preconditioner

   Level: intermediate

.keywords: KSP, setup

.seealso: KSPCreate(), KSPSolve(), KSPDestroy(), PCSetReusePreconditioner()
@*/
PetscErrorCode  KSPSetReusePreconditioner(KSP ksp,PetscBool flag)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ksp,KSP_CLASSID,1);
  ierr = PCSetReusePreconditioner(ksp->pc,flag);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@
   KSPSetSkipPCSetFromOptions - prevents KSPSetFromOptions() from call PCSetFromOptions(). This is used if the same PC is shared by more than one KSP so its options are not resetable for each KSP

   Collective on KSP

   Input Parameters:
+  ksp   - iterative context obtained from KSPCreate()
-  flag - PETSC_TRUE to skip calling the PCSetFromOptions()

   Level: intermediate

.keywords: KSP, setup

.seealso: KSPCreate(), KSPSolve(), KSPDestroy(), PCSetReusePreconditioner()
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

   Collective on KSP

   Input Parameter:
.  ksp   - iterative context obtained from KSPCreate()

   Level: developer

.keywords: KSP, setup

.seealso: KSPCreate(), KSPSolve(), KSPDestroy()
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

  if (ksp->setupstage == KSP_SETUP_NEWRHS) PetscFunctionReturn(0);
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

  ierr = PCGetOperators(ksp->pc,&mat,&pmat);CHKERRQ(ierr);
  /* scale the matrix if requested */
  if (ksp->dscale) {
    PetscScalar *xx;
    PetscInt    i,n;
    PetscBool   zeroflag = PETSC_FALSE;
    if (!ksp->pc) {ierr = KSPGetPC(ksp,&ksp->pc);CHKERRQ(ierr);}
    if (!ksp->diagonal) { /* allocate vector to hold diagonal */
      ierr = MatCreateVecs(pmat,&ksp->diagonal,0);CHKERRQ(ierr);
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
  if (!ksp->pc) {ierr = KSPGetPC(ksp,&ksp->pc);CHKERRQ(ierr);}
  ierr = PCSetErrorIfFailure(ksp->pc,ksp->errorifnotconverged);CHKERRQ(ierr);
  ierr = PCSetUp(ksp->pc);CHKERRQ(ierr);
  ierr = PCGetSetUpFailedReason(ksp->pc,&pcreason);CHKERRQ(ierr); 
  if (pcreason) {
    ksp->reason = KSP_DIVERGED_PCSETUP_FAILED;
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
  PetscFunctionReturn(0);
}

/*@
   KSPReasonView - Displays the reason a KSP solve converged or diverged to a viewer

   Collective on KSP

   Parameter:
+  ksp - iterative context obtained from KSPCreate()
-  viewer - the viewer to display the reason


   Options Database Keys:
.  -ksp_converged_reason - print reason for converged or diverged, also prints number of iterations

   Level: beginner

.keywords: KSP, solve, linear system

.seealso: KSPCreate(), KSPSetUp(), KSPDestroy(), KSPSetTolerances(), KSPConvergedDefault(),
          KSPSolveTranspose(), KSPGetIterationNumber()
@*/
PetscErrorCode KSPReasonView(KSP ksp,PetscViewer viewer)
{
  PetscErrorCode ierr;
  PetscBool      isAscii;

  PetscFunctionBegin;
  ierr = PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&isAscii);CHKERRQ(ierr);
  if (isAscii) {
    ierr = PetscViewerASCIIAddTab(viewer,((PetscObject)ksp)->tablevel);CHKERRQ(ierr);
    if (ksp->reason > 0) {
      if (((PetscObject) ksp)->prefix) {
        ierr = PetscViewerASCIIPrintf(viewer,"Linear %s solve converged due to %s iterations %D\n",((PetscObject) ksp)->prefix,KSPConvergedReasons[ksp->reason],ksp->its);CHKERRQ(ierr);
      } else {
        ierr = PetscViewerASCIIPrintf(viewer,"Linear solve converged due to %s iterations %D\n",KSPConvergedReasons[ksp->reason],ksp->its);CHKERRQ(ierr);
      }
    } else {
      if (((PetscObject) ksp)->prefix) {
        ierr = PetscViewerASCIIPrintf(viewer,"Linear %s solve did not converge due to %s iterations %D\n",((PetscObject) ksp)->prefix,KSPConvergedReasons[ksp->reason],ksp->its);CHKERRQ(ierr);
      } else {
        ierr = PetscViewerASCIIPrintf(viewer,"Linear solve did not converge due to %s iterations %D\n",KSPConvergedReasons[ksp->reason],ksp->its);CHKERRQ(ierr);
      }
      if (ksp->reason == KSP_DIVERGED_PCSETUP_FAILED) {
        PCFailedReason reason;
        ierr = PCGetSetUpFailedReason(ksp->pc,&reason);CHKERRQ(ierr);
        ierr = PetscViewerASCIIPrintf(viewer,"               PCSETUP_FAILED due to %s \n",PCFailedReasons[reason]);CHKERRQ(ierr);
      }
    }
    ierr = PetscViewerASCIISubtractTab(viewer,((PetscObject)ksp)->tablevel);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#if defined(PETSC_HAVE_THREADSAFETY)
#define KSPReasonViewFromOptions KSPReasonViewFromOptionsUnsafe
#else
#endif
/*@C
  KSPReasonViewFromOptions - Processes command line options to determine if/how a KSPReason is to be viewed.

  Collective on KSP

  Input Parameters:
. ksp   - the KSP object

  Level: intermediate

@*/
PetscErrorCode KSPReasonViewFromOptions(KSP ksp)
{
  PetscErrorCode    ierr;
  PetscViewer       viewer;
  PetscBool         flg;
  PetscViewerFormat format;

  PetscFunctionBegin;
  ierr   = PetscOptionsGetViewer(PetscObjectComm((PetscObject)ksp),((PetscObject)ksp)->prefix,"-ksp_converged_reason",&viewer,&format,&flg);CHKERRQ(ierr);
  if (flg) {
    ierr = PetscViewerPushFormat(viewer,format);CHKERRQ(ierr);
    ierr = KSPReasonView(ksp,viewer);CHKERRQ(ierr);
    ierr = PetscViewerPopFormat(viewer);CHKERRQ(ierr);
    ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#include <petscdraw.h>
/*@C
   KSPSolve - Solves linear system.

   Collective on KSP

   Parameter:
+  ksp - iterative context obtained from KSPCreate()
.  b - the right hand side vector
-  x - the solution  (this may be the same vector as b, then b will be overwritten with answer)

   Options Database Keys:
+  -ksp_compute_eigenvalues - compute preconditioned operators eigenvalues
.  -ksp_plot_eigenvalues - plot the computed eigenvalues in an X-window
.  -ksp_plot_eigencontours - plot the computed eigenvalues in an X-window with contours
.  -ksp_compute_eigenvalues_explicitly - compute the eigenvalues by forming the dense operator and using LAPACK
.  -ksp_plot_eigenvalues_explicitly - plot the explicitly computing eigenvalues
.  -ksp_view_mat binary - save matrix to the default binary viewer
.  -ksp_view_pmat binary - save matrix used to build preconditioner to the default binary viewer
.  -ksp_view_rhs binary - save right hand side vector to the default binary viewer
.  -ksp_view_solution binary - save computed solution vector to the default binary viewer
           (can be read later with src/ksp/examples/tutorials/ex10.c for testing solvers)
.  -ksp_view_mat_explicit - for matrix-free operators, computes the matrix entries and views them
.  -ksp_view_preconditioned_operator_explicit - computes the product of the preconditioner and matrix as an explicit matrix and views it
.  -ksp_converged_reason - print reason for converged or diverged, also prints number of iterations
.  -ksp_final_residual - print 2-norm of true linear system residual at the end of the solution process
-  -ksp_view - print the ksp data structure at the end of the system solution

   Notes:

   If one uses KSPSetDM() then x or b need not be passed. Use KSPGetSolution() to access the solution in this case.

   The operator is specified with KSPSetOperators().

   Call KSPGetConvergedReason() to determine if the solver converged or failed and
   why. The number of iterations can be obtained from KSPGetIterationNumber().

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

.keywords: KSP, solve, linear system

.seealso: KSPCreate(), KSPSetUp(), KSPDestroy(), KSPSetTolerances(), KSPConvergedDefault(),
          KSPSolveTranspose(), KSPGetIterationNumber(), MatNullSpaceCreate(), MatSetNullSpace(), MatSetTransposeNullSpace()
@*/
PetscErrorCode KSPSolve(KSP ksp,Vec b,Vec x)
{
  PetscErrorCode    ierr;
  PetscBool         flag1,flag2,flag3,flg = PETSC_FALSE,inXisinB=PETSC_FALSE,guess_zero;
  Mat               mat,pmat;
  MPI_Comm          comm;
  MatNullSpace      nullsp;
  Vec               btmp,vec_rhs=0;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ksp,KSP_CLASSID,1);
  if (b) PetscValidHeaderSpecific(b,VEC_CLASSID,2);
  if (x) PetscValidHeaderSpecific(x,VEC_CLASSID,3);
  comm = PetscObjectComm((PetscObject)ksp);
  if (x && x == b) {
    if (!ksp->guess_zero) SETERRQ(comm,PETSC_ERR_ARG_INCOMP,"Cannot use x == b with nonzero initial guess");
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
  ierr = KSPViewFromOptions(ksp,NULL,"-ksp_view_pre");CHKERRQ(ierr);

  if (ksp->presolve) {
    ierr = (*ksp->presolve)(ksp,ksp->vec_rhs,ksp->vec_sol,ksp->prectx);CHKERRQ(ierr);
  }
  ierr = PetscLogEventBegin(KSP_Solve,ksp,ksp->vec_rhs,ksp->vec_sol,0);CHKERRQ(ierr);

  /* reset the residual history list if requested */
  if (ksp->res_hist_reset) ksp->res_hist_len = 0;
  ksp->transpose_solve = PETSC_FALSE;

  if (ksp->guess) {
    PetscObjectState ostate,state;

    ierr = KSPGuessSetUp(ksp->guess);CHKERRQ(ierr);
    ierr = PetscObjectStateGet((PetscObject)ksp->vec_sol,&ostate);CHKERRQ(ierr);
    ierr = KSPGuessFormGuess(ksp->guess,ksp->vec_rhs,ksp->vec_sol);CHKERRQ(ierr);
    ierr = PetscObjectStateGet((PetscObject)ksp->vec_sol,&state);CHKERRQ(ierr);
    if (state != ostate) {
      ksp->guess_zero = PETSC_FALSE;
    } else {
      PetscInfo(ksp,"Using zero initial guess since the KSPGuess object did not change the vector\n");
      ksp->guess_zero = PETSC_TRUE;
    }
  }

  /* KSPSetUp() scales the matrix if needed */
  ierr = KSPSetUp(ksp);CHKERRQ(ierr);
  ierr = KSPSetUpOnBlocks(ksp);CHKERRQ(ierr);

  VecLocked(ksp->vec_sol,3);

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
  ierr = MatGetTransposeNullSpace(pmat,&nullsp);CHKERRQ(ierr);
  if (nullsp) {
    ierr = VecDuplicate(ksp->vec_rhs,&btmp);CHKERRQ(ierr);
    ierr = VecCopy(ksp->vec_rhs,btmp);CHKERRQ(ierr);
    ierr = MatNullSpaceRemove(nullsp,btmp);CHKERRQ(ierr);
    vec_rhs      = ksp->vec_rhs;
    ksp->vec_rhs = btmp;
  }
  ierr = VecLockPush(ksp->vec_rhs);CHKERRQ(ierr);
  if (ksp->reason == KSP_DIVERGED_PCSETUP_FAILED) {
    ierr = VecSetInf(ksp->vec_sol);CHKERRQ(ierr);
  } 
  ierr = (*ksp->ops->solve)(ksp);CHKERRQ(ierr);
 
  ierr = VecLockPop(ksp->vec_rhs);CHKERRQ(ierr);
  if (nullsp) {
    ksp->vec_rhs = vec_rhs;
    ierr = VecDestroy(&btmp);CHKERRQ(ierr);
  }

  ksp->guess_zero = guess_zero;


  if (!ksp->reason) SETERRQ(comm,PETSC_ERR_PLIB,"Internal error, solver returned without setting converged reason");
  ksp->totalits += ksp->its;

  ierr = KSPReasonViewFromOptions(ksp);CHKERRQ(ierr);
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
  if (ksp->postsolve) {
    ierr = (*ksp->postsolve)(ksp,ksp->vec_rhs,ksp->vec_sol,ksp->postctx);CHKERRQ(ierr);
  }
  if (ksp->guess) {
    ierr = KSPGuessUpdate(ksp->guess,ksp->vec_rhs,ksp->vec_sol);CHKERRQ(ierr);
  }

  ierr = PCGetOperators(ksp->pc,&mat,&pmat);CHKERRQ(ierr);
  ierr = MatViewFromOptions(mat,(PetscObject)ksp,"-ksp_view_mat");CHKERRQ(ierr);
  ierr = MatViewFromOptions(pmat,(PetscObject)ksp,"-ksp_view_pmat");CHKERRQ(ierr);
  ierr = VecViewFromOptions(ksp->vec_rhs,(PetscObject)ksp,"-ksp_view_rhs");CHKERRQ(ierr);

  ierr = PetscOptionsGetViewer(PetscObjectComm((PetscObject)ksp),((PetscObject)ksp)->prefix,"-ksp_compute_eigenvalues",NULL,NULL,&flag1);CHKERRQ(ierr);
  ierr = PetscOptionsGetViewer(PetscObjectComm((PetscObject)ksp),((PetscObject)ksp)->prefix,"-ksp_plot_eigenvalues",NULL,NULL,&flag2);CHKERRQ(ierr);
  ierr = PetscOptionsGetViewer(PetscObjectComm((PetscObject)ksp),((PetscObject)ksp)->prefix,"-ksp_plot_eigencontours",NULL,NULL,&flag3);CHKERRQ(ierr);
  if (flag1 || flag2 || flag3) {
    PetscInt    nits,n,i,neig;
    PetscReal   *r,*c;

    ierr = KSPGetIterationNumber(ksp,&nits);CHKERRQ(ierr);
    n    = nits+2;

    if (!nits) {
      ierr = PetscPrintf(comm,"Zero iterations in solver, cannot approximate any eigenvalues\n");CHKERRQ(ierr);
    } else {
      PetscMPIInt rank;
      ierr = MPI_Comm_rank(comm,&rank);CHKERRQ(ierr);
      ierr = PetscMalloc2(n,&r,n,&c);CHKERRQ(ierr);
      ierr = KSPComputeEigenvalues(ksp,n,r,c,&neig);CHKERRQ(ierr);
      if (flag1) {
        ierr = PetscPrintf(comm,"Iteratively computed eigenvalues\n");CHKERRQ(ierr);
        for (i=0; i<neig; i++) {
          if (c[i] >= 0.0) {
            ierr = PetscPrintf(comm,"%g + %gi\n",(double)r[i],(double)c[i]);CHKERRQ(ierr);
          } else {
            ierr = PetscPrintf(comm,"%g - %gi\n",(double)r[i],-(double)c[i]);CHKERRQ(ierr);
          }
        }
      }
      if (flag2 && !rank) {
        PetscDraw   draw;
        PetscDrawSP drawsp;

        if (!ksp->eigviewer) {
          ierr = PetscViewerDrawOpen(PETSC_COMM_SELF,0,"Iteratively Computed Eigenvalues",PETSC_DECIDE,PETSC_DECIDE,400,400,&ksp->eigviewer);CHKERRQ(ierr);
        }
        ierr = PetscViewerDrawGetDraw(ksp->eigviewer,0,&draw);CHKERRQ(ierr);
        ierr = PetscDrawSPCreate(draw,1,&drawsp);CHKERRQ(ierr);
        for (i=0; i<neig; i++) {
          ierr = PetscDrawSPAddPoint(drawsp,r+i,c+i);CHKERRQ(ierr);
        }
        ierr = PetscDrawSPDraw(drawsp,PETSC_TRUE);CHKERRQ(ierr);
        ierr = PetscDrawSPSave(drawsp);CHKERRQ(ierr);
        ierr = PetscDrawSPDestroy(&drawsp);CHKERRQ(ierr);
      }
      if (flag3 && !rank) {
        ierr = KSPPlotEigenContours_Private(ksp,neig,r,c);CHKERRQ(ierr);
      }
      ierr = PetscFree2(r,c);CHKERRQ(ierr);
    }
  }

  ierr = PetscOptionsGetViewer(PetscObjectComm((PetscObject)ksp),((PetscObject)ksp)->prefix,"-ksp_compute_singularvalues",NULL,NULL,&flag1);CHKERRQ(ierr);
  if (flag1) {
    PetscInt nits;

    ierr = KSPGetIterationNumber(ksp,&nits);CHKERRQ(ierr);
    if (!nits) {
      ierr = PetscPrintf(comm,"Zero iterations in solver, cannot approximate any singular values\n");CHKERRQ(ierr);
    } else {
      PetscReal emax,emin;

      ierr = KSPComputeExtremeSingularValues(ksp,&emax,&emin);CHKERRQ(ierr);
      ierr = PetscPrintf(comm,"Iteratively computed extreme singular values: max %g min %g max/min %g\n",(double)emax,(double)emin,(double)(emax/emin));CHKERRQ(ierr);
    }
  }

  ierr = PetscOptionsGetViewer(PetscObjectComm((PetscObject)ksp),((PetscObject)ksp)->prefix,"-ksp_compute_eigenvalues_explicitly",NULL,NULL,&flag1);CHKERRQ(ierr);
  ierr = PetscOptionsGetViewer(PetscObjectComm((PetscObject)ksp),((PetscObject)ksp)->prefix,"-ksp_plot_eigenvalues_explicitly",NULL,NULL,&flag2);CHKERRQ(ierr);
  if (flag1 || flag2) {
    PetscInt    n,i;
    PetscReal   *r,*c;
    PetscMPIInt rank;
    ierr = MPI_Comm_rank(comm,&rank);CHKERRQ(ierr);
    ierr = VecGetSize(ksp->vec_sol,&n);CHKERRQ(ierr);
    ierr = PetscMalloc2(n,&r,n,&c);CHKERRQ(ierr);
    ierr = KSPComputeEigenvaluesExplicitly(ksp,n,r,c);CHKERRQ(ierr);
    if (flag1) {
      ierr = PetscPrintf(comm,"Explicitly computed eigenvalues\n");CHKERRQ(ierr);
      for (i=0; i<n; i++) {
        if (c[i] >= 0.0) {
          ierr = PetscPrintf(comm,"%g + %gi\n",(double)r[i],(double)c[i]);CHKERRQ(ierr);
        } else {
          ierr = PetscPrintf(comm,"%g - %gi\n",(double)r[i],-(double)c[i]);CHKERRQ(ierr);
        }
      }
    }
    if (flag2 && !rank) {
      PetscDraw   draw;
      PetscDrawSP drawsp;

      if (!ksp->eigviewer) {
        ierr = PetscViewerDrawOpen(PETSC_COMM_SELF,0,"Explicitly Computed Eigenvalues",PETSC_DECIDE,PETSC_DECIDE,400,400,&ksp->eigviewer);CHKERRQ(ierr);
      }
      ierr = PetscViewerDrawGetDraw(ksp->eigviewer,0,&draw);CHKERRQ(ierr);
      ierr = PetscDrawSPCreate(draw,1,&drawsp);CHKERRQ(ierr);
      ierr = PetscDrawSPReset(drawsp);CHKERRQ(ierr);
      for (i=0; i<n; i++) {
        ierr = PetscDrawSPAddPoint(drawsp,r+i,c+i);CHKERRQ(ierr);
      }
      ierr = PetscDrawSPDraw(drawsp,PETSC_TRUE);CHKERRQ(ierr);
      ierr = PetscDrawSPSave(drawsp);CHKERRQ(ierr);
      ierr = PetscDrawSPDestroy(&drawsp);CHKERRQ(ierr);
    }
    ierr = PetscFree2(r,c);CHKERRQ(ierr);
  }

  ierr = PetscOptionsGetViewer(PetscObjectComm((PetscObject)ksp),((PetscObject)ksp)->prefix,"-ksp_view_mat_explicit",NULL,NULL,&flag2);CHKERRQ(ierr);
  if (flag2) {
    Mat A,B;
    ierr = PCGetOperators(ksp->pc,&A,NULL);CHKERRQ(ierr);
    ierr = MatComputeExplicitOperator(A,&B);CHKERRQ(ierr);
    ierr = MatViewFromOptions(B,(PetscObject)ksp,"-ksp_view_mat_explicit");CHKERRQ(ierr);
    ierr = MatDestroy(&B);CHKERRQ(ierr);
  }
  ierr = PetscOptionsGetViewer(PetscObjectComm((PetscObject)ksp),((PetscObject)ksp)->prefix,"-ksp_view_preconditioned_operator_explicit",NULL,NULL,&flag2);CHKERRQ(ierr);
  if (flag2) {
    Mat B;
    ierr = KSPComputeExplicitOperator(ksp,&B);CHKERRQ(ierr);
    ierr = MatViewFromOptions(B,(PetscObject)ksp,"-ksp_view_preconditioned_operator_explicit");CHKERRQ(ierr);
    ierr = MatDestroy(&B);CHKERRQ(ierr);
  }
  ierr = KSPViewFromOptions(ksp,NULL,"-ksp_view");CHKERRQ(ierr);

  ierr = PetscOptionsGetViewer(PetscObjectComm((PetscObject)ksp),((PetscObject)ksp)->prefix,"-ksp_final_residual",NULL,NULL,&flg);CHKERRQ(ierr);
  if (flg) {
    Mat       A;
    Vec       t;
    PetscReal norm;
    if (ksp->dscale && !ksp->dscalefix) SETERRQ(comm,PETSC_ERR_ARG_WRONGSTATE,"Cannot compute final scale with -ksp_diagonal_scale except also with -ksp_diagonal_scale_fix");
    ierr = PCGetOperators(ksp->pc,&A,NULL);CHKERRQ(ierr);
    ierr = VecDuplicate(ksp->vec_rhs,&t);CHKERRQ(ierr);
    ierr = KSP_MatMult(ksp,A,ksp->vec_sol,t);CHKERRQ(ierr);
    ierr = VecAYPX(t, -1.0, ksp->vec_rhs);CHKERRQ(ierr);
    ierr = VecNorm(t,NORM_2,&norm);CHKERRQ(ierr);
    ierr = VecDestroy(&t);CHKERRQ(ierr);
    ierr = PetscPrintf(comm,"KSP final norm of residual %g\n",(double)norm);CHKERRQ(ierr);
  }
  ierr = VecViewFromOptions(ksp->vec_sol,(PetscObject)ksp,"-ksp_view_solution");CHKERRQ(ierr);

  if (inXisinB) {
    ierr = VecCopy(x,b);CHKERRQ(ierr);
    ierr = VecDestroy(&x);CHKERRQ(ierr);
  }
  ierr = PetscObjectSAWsBlock((PetscObject)ksp);CHKERRQ(ierr);
  if (ksp->errorifnotconverged && ksp->reason < 0) SETERRQ(comm,PETSC_ERR_NOT_CONVERGED,"KSPSolve has not converged");
  PetscFunctionReturn(0);
}

/*@
   KSPSolveTranspose - Solves the transpose of a linear system.

   Collective on KSP

   Input Parameter:
+  ksp - iterative context obtained from KSPCreate()
.  b - right hand side vector
-  x - solution vector

   Notes: For complex numbers this solve the non-Hermitian transpose system.

   This currently does NOT correctly use the null space of the operator and its transpose for solving singular systems.

   Developer Notes: We need to implement a KSPSolveHermitianTranspose()

   Level: developer

.keywords: KSP, solve, linear system

.seealso: KSPCreate(), KSPSetUp(), KSPDestroy(), KSPSetTolerances(), KSPConvergedDefault(),
          KSPSolve()
@*/

PetscErrorCode  KSPSolveTranspose(KSP ksp,Vec b,Vec x)
{
  PetscErrorCode ierr;
  PetscBool      inXisinB=PETSC_FALSE;
  Vec            vec_rhs = 0,btmp;
  Mat            mat,pmat;
  MatNullSpace   nullsp;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ksp,KSP_CLASSID,1);
  PetscValidHeaderSpecific(b,VEC_CLASSID,2);
  PetscValidHeaderSpecific(x,VEC_CLASSID,3);
  if (x == b) {
    ierr     = VecDuplicate(b,&x);CHKERRQ(ierr);
    inXisinB = PETSC_TRUE;
  }
  ierr = PetscObjectReference((PetscObject)b);CHKERRQ(ierr);
  ierr = PetscObjectReference((PetscObject)x);CHKERRQ(ierr);
  ierr = VecDestroy(&ksp->vec_rhs);CHKERRQ(ierr);
  ierr = VecDestroy(&ksp->vec_sol);CHKERRQ(ierr);

  ksp->vec_rhs         = b;
  ksp->vec_sol         = x;
  ksp->transpose_solve = PETSC_TRUE;

  ierr = KSPSetUp(ksp);CHKERRQ(ierr);
  ierr = KSPSetUpOnBlocks(ksp);CHKERRQ(ierr);
  if (ksp->guess_zero) { ierr = VecSet(ksp->vec_sol,0.0);CHKERRQ(ierr);}

  ierr = PCGetOperators(ksp->pc,&mat,&pmat);CHKERRQ(ierr);
  ierr = MatGetNullSpace(pmat,&nullsp);CHKERRQ(ierr);
  if (nullsp) {
    ierr = VecDuplicate(ksp->vec_rhs,&btmp);CHKERRQ(ierr);
    ierr = VecCopy(ksp->vec_rhs,btmp);CHKERRQ(ierr);
    ierr = MatNullSpaceRemove(nullsp,btmp);CHKERRQ(ierr);
    vec_rhs      = ksp->vec_rhs;
    ksp->vec_rhs = btmp;
  }

  ierr = (*ksp->ops->solve)(ksp);CHKERRQ(ierr);
  if (nullsp) {
    ksp->vec_rhs = vec_rhs;
    ierr = VecDestroy(&btmp);CHKERRQ(ierr);
  }
  if (!ksp->reason) SETERRQ(PetscObjectComm((PetscObject)ksp),PETSC_ERR_PLIB,"Internal error, solver returned without setting converged reason");
  ierr = KSPReasonViewFromOptions(ksp);CHKERRQ(ierr);
  if (inXisinB) {
    ierr = VecCopy(x,b);CHKERRQ(ierr);
    ierr = VecDestroy(&x);CHKERRQ(ierr);
  }
  if (ksp->errorifnotconverged && ksp->reason < 0) SETERRQ(PetscObjectComm((PetscObject)ksp),PETSC_ERR_NOT_CONVERGED,"KSPSolve has not converged");
  PetscFunctionReturn(0);
}

/*@
   KSPReset - Resets a KSP context to the kspsetupcalled = 0 state and removes any allocated Vecs and Mats

   Collective on KSP

   Input Parameter:
.  ksp - iterative context obtained from KSPCreate()

   Level: beginner

.keywords: KSP, destroy

.seealso: KSPCreate(), KSPSetUp(), KSPSolve()
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

  ksp->setupstage = KSP_SETUP_NEW;
  PetscFunctionReturn(0);
}

/*@
   KSPDestroy - Destroys KSP context.

   Collective on KSP

   Input Parameter:
.  ksp - iterative context obtained from KSPCreate()

   Level: beginner

.keywords: KSP, destroy

.seealso: KSPCreate(), KSPSetUp(), KSPSolve()
@*/
PetscErrorCode  KSPDestroy(KSP *ksp)
{
  PetscErrorCode ierr;
  PC             pc;

  PetscFunctionBegin;
  if (!*ksp) PetscFunctionReturn(0);
  PetscValidHeaderSpecific((*ksp),KSP_CLASSID,1);
  if (--((PetscObject)(*ksp))->refct > 0) {*ksp = 0; PetscFunctionReturn(0);}

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

  ierr = DMDestroy(&(*ksp)->dm);CHKERRQ(ierr);
  ierr = PCDestroy(&(*ksp)->pc);CHKERRQ(ierr);
  ierr = PetscFree((*ksp)->res_hist_alloc);CHKERRQ(ierr);
  if ((*ksp)->convergeddestroy) {
    ierr = (*(*ksp)->convergeddestroy)((*ksp)->cnvP);CHKERRQ(ierr);
  }
  ierr = KSPMonitorCancel((*ksp));CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&(*ksp)->eigviewer);CHKERRQ(ierr);
  ierr = PetscHeaderDestroy(ksp);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@
    KSPSetPCSide - Sets the preconditioning side.

    Logically Collective on KSP

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

.keywords: KSP, set, right, left, symmetric, side, preconditioner, flag

.seealso: KSPGetPCSide(), KSPSetNormType(), KSPGetNormType()
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

.keywords: KSP, get, right, left, symmetric, side, preconditioner, flag

.seealso: KSPSetPCSide()
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

.keywords: KSP, get, tolerance, absolute, relative, divergence, convergence,
           maximum, iterations

.seealso: KSPSetTolerances()
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

   Logically Collective on KSP

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

.keywords: KSP, set, tolerance, absolute, relative, divergence,
           convergence, maximum, iterations

.seealso: KSPGetTolerances(), KSPConvergedDefault(), KSPSetConvergenceTest()
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
    if (rtol < 0.0 || 1.0 <= rtol) SETERRQ1(PetscObjectComm((PetscObject)ksp),PETSC_ERR_ARG_OUTOFRANGE,"Relative tolerance %g must be non-negative and less than 1.0",(double)rtol);
    ksp->rtol = rtol;
  }
  if (abstol != PETSC_DEFAULT) {
    if (abstol < 0.0) SETERRQ1(PetscObjectComm((PetscObject)ksp),PETSC_ERR_ARG_OUTOFRANGE,"Absolute tolerance %g must be non-negative",(double)abstol);
    ksp->abstol = abstol;
  }
  if (dtol != PETSC_DEFAULT) {
    if (dtol < 0.0) SETERRQ1(PetscObjectComm((PetscObject)ksp),PETSC_ERR_ARG_OUTOFRANGE,"Divergence tolerance %g must be larger than 1.0",(double)dtol);
    ksp->divtol = dtol;
  }
  if (maxits != PETSC_DEFAULT) {
    if (maxits < 0) SETERRQ1(PetscObjectComm((PetscObject)ksp),PETSC_ERR_ARG_OUTOFRANGE,"Maximum number of iterations %D must be non-negative",maxits);
    ksp->max_it = maxits;
  }
  PetscFunctionReturn(0);
}

/*@
   KSPSetInitialGuessNonzero - Tells the iterative solver that the
   initial guess is nonzero; otherwise KSP assumes the initial guess
   is to be zero (and thus zeros it out before solving).

   Logically Collective on KSP

   Input Parameters:
+  ksp - iterative context obtained from KSPCreate()
-  flg - PETSC_TRUE indicates the guess is non-zero, PETSC_FALSE indicates the guess is zero

   Options database keys:
.  -ksp_initial_guess_nonzero : use nonzero initial guess; this takes an optional truth value (0/1/no/yes/true/false)

   Level: beginner

   Notes:
    If this is not called the X vector is zeroed in the call to KSPSolve().

.keywords: KSP, set, initial guess, nonzero

.seealso: KSPGetInitialGuessNonzero(), KSPSetGuessType(), KSPGuessType
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

.keywords: KSP, set, initial guess, nonzero

.seealso: KSPSetInitialGuessNonzero()
@*/
PetscErrorCode  KSPGetInitialGuessNonzero(KSP ksp,PetscBool  *flag)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ksp,KSP_CLASSID,1);
  PetscValidPointer(flag,2);
  if (ksp->guess_zero) *flag = PETSC_FALSE;
  else *flag = PETSC_TRUE;
  PetscFunctionReturn(0);
}

/*@
   KSPSetErrorIfNotConverged - Causes KSPSolve() to generate an error if the solver has not converged.

   Logically Collective on KSP

   Input Parameters:
+  ksp - iterative context obtained from KSPCreate()
-  flg - PETSC_TRUE indicates you want the error generated

   Options database keys:
.  -ksp_error_if_not_converged : this takes an optional truth value (0/1/no/yes/true/false)

   Level: intermediate

   Notes:
    Normally PETSc continues if a linear solver fails to converge, you can call KSPGetConvergedReason() after a KSPSolve()
    to determine if it has converged.

.keywords: KSP

.seealso: KSPGetErrorIfNotConverged()
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

.keywords: KSP

.seealso: KSPSetErrorIfNotConverged()
@*/
PetscErrorCode  KSPGetErrorIfNotConverged(KSP ksp,PetscBool  *flag)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ksp,KSP_CLASSID,1);
  PetscValidPointer(flag,2);
  *flag = ksp->errorifnotconverged;
  PetscFunctionReturn(0);
}

/*@
   KSPSetInitialGuessKnoll - Tells the iterative solver to use PCApply(pc,b,..) to compute the initial guess (The Knoll trick)

   Logically Collective on KSP

   Input Parameters:
+  ksp - iterative context obtained from KSPCreate()
-  flg - PETSC_TRUE or PETSC_FALSE

   Level: advanced

   Developer Note: the Knoll trick is not currently implemented using the KSPGuess class

.keywords: KSP, set, initial guess, nonzero

.seealso: KSPGetInitialGuessKnoll(), KSPSetInitialGuessNonzero(), KSPGetInitialGuessNonzero()
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

.keywords: KSP, set, initial guess, nonzero

.seealso: KSPSetInitialGuessKnoll(), KSPSetInitialGuessNonzero(), KSPGetInitialGuessNonzero()
@*/
PetscErrorCode  KSPGetInitialGuessKnoll(KSP ksp,PetscBool  *flag)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ksp,KSP_CLASSID,1);
  PetscValidPointer(flag,2);
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

.keywords: KSP, set, compute, singular values

.seealso: KSPComputeExtremeSingularValues(), KSPMonitorSingularValue()
@*/
PetscErrorCode  KSPGetComputeSingularValues(KSP ksp,PetscBool  *flg)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ksp,KSP_CLASSID,1);
  PetscValidPointer(flg,2);
  *flg = ksp->calc_sings;
  PetscFunctionReturn(0);
}

/*@
   KSPSetComputeSingularValues - Sets a flag so that the extreme singular
   values will be calculated via a Lanczos or Arnoldi process as the linear
   system is solved.

   Logically Collective on KSP

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

.keywords: KSP, set, compute, singular values

.seealso: KSPComputeExtremeSingularValues(), KSPMonitorSingularValue()
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

.keywords: KSP, set, compute, eigenvalues

.seealso: KSPComputeEigenvalues(), KSPComputeEigenvaluesExplicitly()
@*/
PetscErrorCode  KSPGetComputeEigenvalues(KSP ksp,PetscBool  *flg)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ksp,KSP_CLASSID,1);
  PetscValidPointer(flg,2);
  *flg = ksp->calc_sings;
  PetscFunctionReturn(0);
}

/*@
   KSPSetComputeEigenvalues - Sets a flag so that the extreme eigenvalues
   values will be calculated via a Lanczos or Arnoldi process as the linear
   system is solved.

   Logically Collective on KSP

   Input Parameters:
+  ksp - iterative context obtained from KSPCreate()
-  flg - PETSC_TRUE or PETSC_FALSE

   Notes:
   Currently this option is not valid for all iterative methods.

   Level: advanced

.keywords: KSP, set, compute, eigenvalues

.seealso: KSPComputeEigenvalues(), KSPComputeEigenvaluesExplicitly()
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

   Logically Collective on KSP

   Input Parameters:
+  ksp - iterative context obtained from KSPCreate()
-  flg - PETSC_TRUE or PETSC_FALSE

   Notes:
   Currently this option is only valid for the GMRES method.

   Level: advanced

.keywords: KSP, set, compute, ritz

.seealso: KSPComputeRitz()
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

.keywords: KSP, get, right-hand-side, rhs

.seealso: KSPGetSolution(), KSPSolve()
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

.keywords: KSP, get, solution

.seealso: KSPGetRhs(),  KSPBuildSolution(), KSPSolve()
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

   Collective on KSP

   Input Parameters:
+  ksp - iterative context obtained from KSPCreate()
-  pc   - the preconditioner object

   Notes:
   Use KSPGetPC() to retrieve the preconditioner context (for example,
   to free it at the end of the computations).

   Level: developer

.keywords: KSP, set, precondition, Binv

.seealso: KSPGetPC()
@*/
PetscErrorCode  KSPSetPC(KSP ksp,PC pc)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ksp,KSP_CLASSID,1);
  PetscValidHeaderSpecific(pc,PC_CLASSID,2);
  PetscCheckSameComm(ksp,1,pc,2);
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

.keywords: KSP, get, preconditioner, Binv

.seealso: KSPSetPC()
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
  }
  *pc = ksp->pc;
  PetscFunctionReturn(0);
}

/*@
   KSPMonitor - runs the user provided monitor routines, if they exist

   Collective on KSP

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

/*

    Checks if two monitors are identical; if they are then it destroys the new one
*/
PetscErrorCode PetscMonitorCompare(PetscErrorCode (*nmon)(void),void *nmctx,PetscErrorCode (*nmdestroy)(void**),PetscErrorCode (*mon)(void),void *mctx,PetscErrorCode (*mdestroy)(void**),PetscBool *identical)
{
  *identical = PETSC_FALSE;
  if (nmon == mon && nmdestroy == mdestroy) {
    if (nmctx == mctx) *identical = PETSC_TRUE;
    else if (nmdestroy == (PetscErrorCode (*)(void**)) PetscViewerAndFormatDestroy) {
      PetscViewerAndFormat *old = (PetscViewerAndFormat*)mctx, *newo = (PetscViewerAndFormat*)nmctx;
      if (old->viewer == newo->viewer && old->format == newo->format) *identical = PETSC_TRUE;
    }
    if (*identical) {
      if (mdestroy) {
        PetscErrorCode ierr;
        ierr = (*mdestroy)(&nmctx);CHKERRQ(ierr);
      }
    }
  }
  PetscFunctionReturn(0);
}

/*@C
   KSPMonitorSet - Sets an ADDITIONAL function to be called at every iteration to monitor
   the residual/error etc.

   Logically Collective on KSP

   Input Parameters:
+  ksp - iterative context obtained from KSPCreate()
.  monitor - pointer to function (if this is NULL, it turns off monitoring
.  mctx    - [optional] context for private data for the
             monitor routine (use NULL if no context is desired)
-  monitordestroy - [optional] routine that frees monitor context
          (may be NULL)

   Calling Sequence of monitor:
$     monitor (KSP ksp, int it, PetscReal rnorm, void *mctx)

+  ksp - iterative context obtained from KSPCreate()
.  it - iteration number
.  rnorm - (estimated) 2-norm of (preconditioned) residual
-  mctx  - optional monitoring context, as set by KSPMonitorSet()

   Options Database Keys:
+    -ksp_monitor        - sets KSPMonitorDefault()
.    -ksp_monitor_true_residual    - sets KSPMonitorTrueResidualNorm()
.    -ksp_monitor_max    - sets KSPMonitorTrueResidualMaxNorm()
.    -ksp_monitor_lg_residualnorm    - sets line graph monitor,
                           uses KSPMonitorLGResidualNormCreate()
.    -ksp_monitor_lg_true_residualnorm   - sets line graph monitor,
                           uses KSPMonitorLGResidualNormCreate()
.    -ksp_monitor_singular_value    - sets KSPMonitorSingularValue()
-    -ksp_monitor_cancel - cancels all monitors that have
                          been hardwired into a code by
                          calls to KSPMonitorSet(), but
                          does not cancel those set via
                          the options database.

   Notes:
   The default is to do nothing.  To print the residual, or preconditioned
   residual if KSPSetNormType(ksp,KSP_NORM_PRECONDITIONED) was called, use
   KSPMonitorDefault() as the monitoring routine, with a ASCII viewer as the
   context.

   Several different monitoring routines may be set by calling
   KSPMonitorSet() multiple times; all will be called in the
   order in which they were set.

   Fortran notes: Only a single monitor function can be set for each KSP object

   Level: beginner

.keywords: KSP, set, monitor

.seealso: KSPMonitorDefault(), KSPMonitorLGResidualNormCreate(), KSPMonitorCancel()
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
  if (ksp->numbermonitors >= MAXKSPMONITORS) SETERRQ(PetscObjectComm((PetscObject)ksp),PETSC_ERR_ARG_OUTOFRANGE,"Too many KSP monitors set");
  ksp->monitor[ksp->numbermonitors]          = monitor;
  ksp->monitordestroy[ksp->numbermonitors]   = monitordestroy;
  ksp->monitorcontext[ksp->numbermonitors++] = (void*)mctx;
  PetscFunctionReturn(0);
}

/*@
   KSPMonitorCancel - Clears all monitors for a KSP object.

   Logically Collective on KSP

   Input Parameters:
.  ksp - iterative context obtained from KSPCreate()

   Options Database Key:
.  -ksp_monitor_cancel - Cancels all monitors that have
    been hardwired into a code by calls to KSPMonitorSet(),
    but does not cancel those set via the options database.

   Level: intermediate

.keywords: KSP, set, monitor

.seealso: KSPMonitorDefault(), KSPMonitorLGResidualNormCreate(), KSPMonitorSet()
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

.keywords: KSP, get, monitor, context

.seealso: KSPMonitorDefault(), KSPMonitorLGResidualNormCreate()
@*/
PetscErrorCode  KSPGetMonitorContext(KSP ksp,void **ctx)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ksp,KSP_CLASSID,1);
  *ctx =      (ksp->monitorcontext[0]);
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

   Notes: The array is NOT freed by PETSc so the user needs to keep track of
           it and destroy once the KSP object is destroyed.

   If 'a' is NULL then space is allocated for the history. If 'na' PETSC_DECIDE or PETSC_DEFAULT then a
   default array of length 10000 is allocated.

.keywords: KSP, set, residual, history, norm

.seealso: KSPGetResidualHistory()

@*/
PetscErrorCode  KSPSetResidualHistory(KSP ksp,PetscReal a[],PetscInt na,PetscBool reset)
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
     Can only be called after a KSPSetResidualHistory() otherwise a and na are set to zero

     The Fortran version of this routine has a calling sequence
$   call KSPGetResidualHistory(KSP ksp, integer na, integer ierr)
    note that you have passed a Fortran array into KSPSetResidualHistory() and you need
    to access the residual values from this Fortran array you provided. Only the na (number of
    residual norms currently held) is set.

.keywords: KSP, get, residual, history, norm

.seealso: KSPGetResidualHistory()

@*/
PetscErrorCode  KSPGetResidualHistory(KSP ksp,PetscReal *a[],PetscInt *na)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ksp,KSP_CLASSID,1);
  if (a) *a = ksp->res_hist;
  if (na) *na = ksp->res_hist_len;
  PetscFunctionReturn(0);
}

/*@C
   KSPSetConvergenceTest - Sets the function to be used to determine
   convergence.

   Logically Collective on KSP

   Input Parameters:
+  ksp - iterative context obtained from KSPCreate()
.  converge - pointer to int function
.  cctx    - context for private data for the convergence routine (may be null)
-  destroy - a routine for destroying the context (may be null)

   Calling sequence of converge:
$     converge (KSP ksp, int it, PetscReal rnorm, KSPConvergedReason *reason,void *mctx)

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

.keywords: KSP, set, convergence, test, context

.seealso: KSPConvergedDefault(), KSPGetConvergenceContext(), KSPSetTolerances()
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
   KSPGetConvergenceContext - Gets the convergence context set with
   KSPSetConvergenceTest().

   Not Collective

   Input Parameter:
.  ksp - iterative context obtained from KSPCreate()

   Output Parameter:
.  ctx - monitoring context

   Level: advanced

.keywords: KSP, get, convergence, test, context

.seealso: KSPConvergedDefault(), KSPSetConvergenceTest()
@*/
PetscErrorCode  KSPGetConvergenceContext(KSP ksp,void **ctx)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ksp,KSP_CLASSID,1);
  *ctx = ksp->cnvP;
  PetscFunctionReturn(0);
}

/*@C
   KSPBuildSolution - Builds the approximate solution in a vector provided.
   This routine is NOT commonly needed (see KSPSolve()).

   Collective on KSP

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

.keywords: KSP, build, solution

.seealso: KSPGetSolution(), KSPBuildResidual()
@*/
PetscErrorCode  KSPBuildSolution(KSP ksp,Vec v,Vec *V)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ksp,KSP_CLASSID,1);
  if (!V && !v) SETERRQ(PetscObjectComm((PetscObject)ksp),PETSC_ERR_ARG_WRONG,"Must provide either v or V");
  if (!V) V = &v;
  ierr = (*ksp->ops->buildsolution)(ksp,v,V);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
   KSPBuildResidual - Builds the residual in a vector provided.

   Collective on KSP

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

.keywords: KSP, build, residual

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

   Logically Collective on KSP

   Input Parameter:
+  ksp - the KSP context
-  scale - PETSC_TRUE or PETSC_FALSE

   Options Database Key:
+   -ksp_diagonal_scale -
-   -ksp_diagonal_scale_fix - scale the matrix back AFTER the solve


    Notes: Scales the matrix by  D^(-1/2)  A  D^(-1/2)  [D^(1/2) x ] = D^(-1/2) b
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

.keywords: KSP, set, options, prefix, database

.seealso: KSPGetDiagonalScale(), KSPSetDiagonalScaleFix()
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

.keywords: KSP, set, options, prefix, database

.seealso: KSPSetDiagonalScale(), KSPSetDiagonalScaleFix()
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

   Logically Collective on KSP

   Input Parameter:
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

.keywords: KSP, set, options, prefix, database

.seealso: KSPGetDiagonalScale(), KSPSetDiagonalScale(), KSPGetDiagonalScaleFix()
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

.keywords: KSP, set, options, prefix, database

.seealso: KSPGetDiagonalScale(), KSPSetDiagonalScale(), KSPSetDiagonalScaleFix()
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

   Input Arguments:
+  ksp - the KSP context
.  func - function to compute the operators
-  ctx - optional context

   Calling sequence of func:
$  func(KSP ksp,Mat A,Mat B,void *ctx)

+  ksp - the KSP context
.  A - the linear operator
.  B - preconditioning matrix
-  ctx - optional user-provided context

   Notes: The user provided func() will be called automatically at the very next call to KSPSolve(). It will not be called at future KSPSolve() calls
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

   Input Arguments:
+  ksp - the KSP context
.  func - function to compute the right hand side
-  ctx - optional context

   Calling sequence of func:
$  func(KSP ksp,Vec b,void *ctx)

+  ksp - the KSP context
.  b - right hand side of linear system
-  ctx - optional user-provided context

   Notes: The routine you provide will be called EACH you call KSPSolve() to prepare the new right hand side for that solve

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

   Input Arguments:
+  ksp - the KSP context
.  func - function to compute the initial guess
-  ctx - optional context

   Calling sequence of func:
$  func(KSP ksp,Vec x,void *ctx)

+  ksp - the KSP context
.  x - solution vector
-  ctx - optional user-provided context

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
