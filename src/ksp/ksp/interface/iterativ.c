/*
   This file contains some simple default routines.
   These routines should be SHORT, since they will be included in every
   executable image that uses the iterative routines (note that, through
   the registry system, we provide a way to load only the truely necessary
   files)
 */
#include <petsc-private/kspimpl.h>   /*I "petscksp.h" I*/
#include <petscdmshell.h>

#undef __FUNCT__
#define __FUNCT__ "KSPGetResidualNorm"
/*@
   KSPGetResidualNorm - Gets the last (approximate preconditioned)
   residual norm that has been computed.

   Not Collective

   Input Parameters:
.  ksp - the iterative context

   Output Parameters:
.  rnorm - residual norm

   Level: intermediate

.keywords: KSP, get, residual norm

.seealso: KSPBuildResidual()
@*/
PetscErrorCode  KSPGetResidualNorm(KSP ksp,PetscReal *rnorm)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ksp,KSP_CLASSID,1);
  PetscValidRealPointer(rnorm,2);
  *rnorm = ksp->rnorm;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "KSPGetIterationNumber"
/*@
   KSPGetIterationNumber - Gets the current iteration number; if the
         KSPSolve() is complete, returns the number of iterations
         used.

   Not Collective

   Input Parameters:
.  ksp - the iterative context

   Output Parameters:
.  its - number of iterations

   Level: intermediate

   Notes:
      During the ith iteration this returns i-1
.keywords: KSP, get, residual norm

.seealso: KSPBuildResidual(), KSPGetResidualNorm()
@*/
PetscErrorCode  KSPGetIterationNumber(KSP ksp,PetscInt *its)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ksp,KSP_CLASSID,1);
  PetscValidIntPointer(its,2);
  *its = ksp->its;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "KSPMonitorSingularValue"
/*@C
    KSPMonitorSingularValue - Prints the two norm of the true residual and
    estimation of the extreme singular values of the preconditioned problem
    at each iteration.

    Logically Collective on KSP

    Input Parameters:
+   ksp - the iterative context
.   n  - the iteration
-   rnorm - the two norm of the residual

    Options Database Key:
.   -ksp_monitor_singular_value - Activates KSPMonitorSingularValue()

    Notes:
    The CG solver uses the Lanczos technique for eigenvalue computation,
    while GMRES uses the Arnoldi technique; other iterative methods do
    not currently compute singular values.

    Level: intermediate

.keywords: KSP, CG, default, monitor, extreme, singular values, Lanczos, Arnoldi

.seealso: KSPComputeExtremeSingularValues()
@*/
PetscErrorCode  KSPMonitorSingularValue(KSP ksp,PetscInt n,PetscReal rnorm,void *dummy)
{
  PetscReal       emin,emax,c;
  PetscErrorCode  ierr;
  PetscViewer     viewer = dummy ? (PetscViewer) dummy : PETSC_VIEWER_STDOUT_(((PetscObject)ksp)->comm);

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ksp,KSP_CLASSID,1);
  ierr = PetscViewerASCIIAddTab(viewer,((PetscObject)ksp)->tablevel);CHKERRQ(ierr);
  if (!ksp->calc_sings) {
    ierr = PetscViewerASCIIPrintf(viewer,"%3D KSP Residual norm %14.12e \n",n,(double)rnorm);CHKERRQ(ierr);
  } else {
    ierr = KSPComputeExtremeSingularValues(ksp,&emax,&emin);CHKERRQ(ierr);
    c = emax/emin;
    ierr = PetscViewerASCIIPrintf(viewer,"%3D KSP Residual norm %14.12e %% max %14.12e min %14.12e max/min %14.12e\n",n,(double)rnorm,(double)emax,(double)emin,(double)c);CHKERRQ(ierr);
  }
  ierr = PetscViewerASCIISubtractTab(viewer,((PetscObject)ksp)->tablevel);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "KSPMonitorSolution"
/*@C
   KSPMonitorSolution - Monitors progress of the KSP solvers by calling
   VecView() for the approximate solution at each iteration.

   Collective on KSP

   Input Parameters:
+  ksp - the KSP context
.  its - iteration number
.  fgnorm - 2-norm of residual (or gradient)
-  dummy - either a viewer or PETSC_NULL

   Level: intermediate

   Notes:
    For some Krylov methods such as GMRES constructing the solution at
  each iteration is expensive, hence using this will slow the code.

.keywords: KSP, nonlinear, vector, monitor, view

.seealso: KSPMonitorSet(), KSPMonitorDefault(), VecView()
@*/
PetscErrorCode  KSPMonitorSolution(KSP ksp,PetscInt its,PetscReal fgnorm,void *dummy)
{
  PetscErrorCode ierr;
  Vec            x;
  PetscViewer    viewer = (PetscViewer) dummy;

  PetscFunctionBegin;
  ierr = KSPBuildSolution(ksp,PETSC_NULL,&x);CHKERRQ(ierr);
  if (!viewer) {
    MPI_Comm comm;
    ierr   = PetscObjectGetComm((PetscObject)ksp,&comm);CHKERRQ(ierr);
    viewer = PETSC_VIEWER_DRAW_(comm);
  }
  ierr = VecView(x,viewer);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "KSPMonitorDefault"
/*@C
   KSPMonitorDefault - Print the residual norm at each iteration of an
   iterative solver.

   Collective on KSP

   Input Parameters:
+  ksp   - iterative context
.  n     - iteration number
.  rnorm - 2-norm (preconditioned) residual value (may be estimated).
-  dummy - unused monitor context

   Level: intermediate

.keywords: KSP, default, monitor, residual

.seealso: KSPMonitorSet(), KSPMonitorTrueResidualNorm(), KSPMonitorLGResidualNormCreate()
@*/
PetscErrorCode  KSPMonitorDefault(KSP ksp,PetscInt n,PetscReal rnorm,void *dummy)
{
  PetscErrorCode ierr;
  PetscViewer    viewer = dummy ? (PetscViewer) dummy : PETSC_VIEWER_STDOUT_(((PetscObject)ksp)->comm);

  PetscFunctionBegin;
  ierr = PetscViewerASCIIAddTab(viewer,((PetscObject)ksp)->tablevel);CHKERRQ(ierr);
  if (n == 0 && ((PetscObject)ksp)->prefix) {
    ierr = PetscViewerASCIIPrintf(viewer,"  Residual norms for %s solve.\n",((PetscObject)ksp)->prefix);CHKERRQ(ierr);
  }
  ierr = PetscViewerASCIIPrintf(viewer,"%3D KSP Residual norm %14.12e \n",n,(double)rnorm);CHKERRQ(ierr);
  ierr = PetscViewerASCIISubtractTab(viewer,((PetscObject)ksp)->tablevel);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "KSPMonitorTrueResidualNorm"
/*@C
   KSPMonitorTrueResidualNorm - Prints the true residual norm as well as the preconditioned
   residual norm at each iteration of an iterative solver.

   Collective on KSP

   Input Parameters:
+  ksp   - iterative context
.  n     - iteration number
.  rnorm - 2-norm (preconditioned) residual value (may be estimated).
-  dummy - unused monitor context

   Options Database Key:
.  -ksp_monitor_true_residual - Activates KSPMonitorTrueResidualNorm()

   Notes:
   When using right preconditioning, these values are equivalent.

   Level: intermediate

.keywords: KSP, default, monitor, residual

.seealso: KSPMonitorSet(), KSPMonitorDefault(), KSPMonitorLGResidualNormCreate(),KSPMonitorTrueResidualMaxNorm()
@*/
PetscErrorCode  KSPMonitorTrueResidualNorm(KSP ksp,PetscInt n,PetscReal rnorm,void *dummy)
{
  PetscErrorCode  ierr;
  Vec             resid,work;
  PetscReal       scnorm,bnorm;
  PetscViewer     viewer = dummy ? (PetscViewer) dummy : PETSC_VIEWER_STDOUT_(((PetscObject)ksp)->comm);
  char            normtype[256];

  PetscFunctionBegin;
  ierr = PetscViewerASCIIAddTab(viewer,((PetscObject)ksp)->tablevel);CHKERRQ(ierr);
  if (n == 0 && ((PetscObject)ksp)->prefix) {
    ierr = PetscViewerASCIIPrintf(viewer,"  Residual norms for %s solve.\n",((PetscObject)ksp)->prefix);CHKERRQ(ierr);
  }
  ierr = VecDuplicate(ksp->vec_rhs,&work);CHKERRQ(ierr);
  ierr = KSPBuildResidual(ksp,0,work,&resid);CHKERRQ(ierr);

  /*
     Unscale the residual but only if both matrices are the same matrix, since only then would
    they be scaled.
  */
  ierr = VecCopy(resid,work);CHKERRQ(ierr);
  ierr = VecNorm(work,NORM_2,&scnorm);CHKERRQ(ierr);
  ierr = VecDestroy(&work);CHKERRQ(ierr);
  ierr = VecNorm(ksp->vec_rhs,NORM_2,&bnorm);CHKERRQ(ierr);
  ierr = PetscStrncpy(normtype,KSPNormTypes[ksp->normtype],sizeof(normtype));CHKERRQ(ierr);
  ierr = PetscStrtolower(normtype);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"%3D KSP %s resid norm %14.12e true resid norm %14.12e ||r(i)||/||b|| %14.12e\n",n,normtype,(double)rnorm,(double)scnorm,(double)(scnorm/bnorm));CHKERRQ(ierr);
  ierr = PetscViewerASCIISubtractTab(viewer,((PetscObject)ksp)->tablevel);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "KSPMonitorTrueResidualMaxNorm"
/*@C
   KSPMonitorTrueResidualMaxNorm - Prints the true residual max norm as well as the preconditioned
   residual norm at each iteration of an iterative solver.

   Collective on KSP

   Input Parameters:
+  ksp   - iterative context
.  n     - iteration number
.  rnorm - norm (preconditioned) residual value (may be estimated).
-  dummy - unused monitor context

   Options Database Key:
.  -ksp_monitor_max - Activates KSPMonitorTrueResidualMaxNorm()

   Notes:
   This could be implemented (better) with a flag in ksp.

   Level: intermediate

.keywords: KSP, default, monitor, residual

.seealso: KSPMonitorSet(), KSPMonitorDefault(), KSPMonitorLGResidualNormCreate(),KSPMonitorTrueResidualNorm()
@*/
PetscErrorCode  KSPMonitorTrueResidualMaxNorm(KSP ksp,PetscInt n,PetscReal rnorm,void *dummy)
{
  PetscErrorCode  ierr;
  Vec             resid,work;
  PetscReal       scnorm,bnorm;
  PetscViewer     viewer = dummy ? (PetscViewer) dummy : PETSC_VIEWER_STDOUT_(((PetscObject)ksp)->comm);
  char            normtype[256];

  PetscFunctionBegin;
  ierr = PetscViewerASCIIAddTab(viewer,((PetscObject)ksp)->tablevel);CHKERRQ(ierr);
  if (n == 0 && ((PetscObject)ksp)->prefix) {
    ierr = PetscViewerASCIIPrintf(viewer,"  Residual norms (max) for %s solve.\n",((PetscObject)ksp)->prefix);CHKERRQ(ierr);
  }
  ierr = VecDuplicate(ksp->vec_rhs,&work);CHKERRQ(ierr);
  ierr = KSPBuildResidual(ksp,0,work,&resid);CHKERRQ(ierr);

  /*
     Unscale the residual but only if both matrices are the same matrix, since only then would
    they be scaled.
  */
  ierr = VecCopy(resid,work);CHKERRQ(ierr);
  ierr = VecNorm(work,NORM_INFINITY,&scnorm);CHKERRQ(ierr);
  ierr = VecDestroy(&work);CHKERRQ(ierr);
  ierr = VecNorm(ksp->vec_rhs,NORM_INFINITY,&bnorm);CHKERRQ(ierr);
  ierr = PetscStrncpy(normtype,KSPNormTypes[ksp->normtype],sizeof(normtype));CHKERRQ(ierr);
  ierr = PetscStrtolower(normtype);CHKERRQ(ierr);
  /* ierr = PetscViewerASCIIPrintf(viewer,"%3D KSP %s resid norm %14.12e true resid norm %14.12e ||r(i)||_inf/||b||_inf %14.12e\n",n,normtype,(double)rnorm,(double)scnorm,(double)(scnorm/bnorm));CHKERRQ(ierr); */
  ierr = PetscViewerASCIIPrintf(viewer,"%3D KSP true resid max norm %14.12e ||r(i)||/||b|| %14.12e\n",n,(double)scnorm,(double)(scnorm/bnorm));CHKERRQ(ierr);
  ierr = PetscViewerASCIISubtractTab(viewer,((PetscObject)ksp)->tablevel);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "KSPMonitorRange_Private"
PetscErrorCode  KSPMonitorRange_Private(KSP ksp,PetscInt it,PetscReal *per)
{
  PetscErrorCode          ierr;
  Vec                     resid,work;
  PetscReal               rmax,pwork;
  PetscInt                i,n,N;
  PetscScalar             *r;

  PetscFunctionBegin;
  ierr = VecDuplicate(ksp->vec_rhs,&work);CHKERRQ(ierr);
  ierr = KSPBuildResidual(ksp,0,work,&resid);CHKERRQ(ierr);

  /*
     Unscale the residual if the matrix is, but only if both matrices are the same matrix, since only then would
    they be scaled.
  */
  ierr = VecCopy(resid,work);CHKERRQ(ierr);
  ierr = VecNorm(work,NORM_INFINITY,&rmax);CHKERRQ(ierr);
  ierr = VecGetLocalSize(work,&n);CHKERRQ(ierr);
  ierr = VecGetSize(work,&N);CHKERRQ(ierr);
  ierr = VecGetArray(work,&r);CHKERRQ(ierr);
  pwork = 0.0;
  for (i=0; i<n; i++) {
    pwork += (PetscAbsScalar(r[i]) > .20*rmax);
  }
  ierr = MPI_Allreduce(&pwork,per,1,MPIU_REAL,MPIU_SUM,((PetscObject)ksp)->comm);CHKERRQ(ierr);
  ierr = VecRestoreArray(work,&r);CHKERRQ(ierr);
  ierr = VecDestroy(&work);CHKERRQ(ierr);
  *per  = *per/N;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "KSPMonitorRange"
/*@C
   KSPMonitorRange - Prints the percentage of residual elements that are more then 10 percent of the maximum value.

   Collective on KSP

   Input Parameters:
+  ksp   - iterative context
.  it    - iteration number
.  rnorm - 2-norm (preconditioned) residual value (may be estimated).
-  dummy - unused monitor context

   Options Database Key:
.  -ksp_monitor_range - Activates KSPMonitorRange()

   Level: intermediate

.keywords: KSP, default, monitor, residual

.seealso: KSPMonitorSet(), KSPMonitorDefault(), KSPMonitorLGResidualNormCreate()
@*/
PetscErrorCode  KSPMonitorRange(KSP ksp,PetscInt it,PetscReal rnorm,void *dummy)
{
  PetscErrorCode   ierr;
  PetscReal        perc,rel;
  PetscViewer      viewer = dummy ? (PetscViewer) dummy : PETSC_VIEWER_STDOUT_(((PetscObject)ksp)->comm);
  /* should be in a MonitorRangeContext */
  static PetscReal prev;

  PetscFunctionBegin;
  ierr = PetscViewerASCIIAddTab(viewer,((PetscObject)ksp)->tablevel);CHKERRQ(ierr);
  if (!it) prev = rnorm;
  if (it == 0 && ((PetscObject)ksp)->prefix) {
    ierr = PetscViewerASCIIPrintf(viewer,"  Residual norms for %s solve.\n",((PetscObject)ksp)->prefix);CHKERRQ(ierr);
  }
  ierr = KSPMonitorRange_Private(ksp,it,&perc);CHKERRQ(ierr);

  rel  = (prev - rnorm)/prev;
  prev = rnorm;
  ierr = PetscViewerASCIIPrintf(viewer,"%3D KSP preconditioned resid norm %14.12e Percent values above 20 percent of maximum %5.2f relative decrease %5.2e ratio %5.2e \n",it,(double)rnorm,(double)100.0*perc,(double)rel,(double)rel/perc);CHKERRQ(ierr);
  ierr = PetscViewerASCIISubtractTab(viewer,((PetscObject)ksp)->tablevel);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "KSPMonitorDynamicTolerance"
/*
 A hack to using dynamic tolerance in preconditioner
 */
PetscErrorCode KSPMonitorDynamicTolerance(KSP ksp,PetscInt its,PetscReal fnorm,void *dummy)
{
  PetscErrorCode ierr;
  PC             pc;
  PetscReal      outer_rtol, outer_abstol, outer_dtol, inner_rtol;
  PetscInt       outer_maxits,nksp,first,i;
  KSPDynTolCtx   *scale = (KSPDynTolCtx*)dummy;
  KSP            kspinner = NULL, *subksp = NULL;

  PetscFunctionBegin;
  ierr = KSPGetPC(ksp, &pc);CHKERRQ(ierr);

  /* compute inner_rtol */
  if (scale->bnrm < 0.0) {
    Vec b;
    ierr = KSPGetRhs(ksp, &b);CHKERRQ(ierr);
    ierr = VecNorm(b, NORM_2, &(scale->bnrm));CHKERRQ(ierr);
  }
  ierr = KSPGetTolerances(ksp, &outer_rtol, &outer_abstol, &outer_dtol, &outer_maxits);CHKERRQ(ierr);
  inner_rtol = PetscMin( scale->coef * scale->bnrm * outer_rtol / fnorm, 0.999 );

  /* force printing. will remove later */
  ierr = PetscPrintf(PETSC_COMM_WORLD, "        Inner rtol = %G\n", inner_rtol);CHKERRQ(ierr);

  /* if pc is ksp */
  ierr = PCKSPGetKSP(pc, &kspinner);CHKERRQ(ierr);
  if (kspinner) {
    ierr = KSPSetTolerances(kspinner, inner_rtol, outer_abstol, outer_dtol, outer_maxits);CHKERRQ(ierr);
    PetscFunctionReturn(0);
  }

  /* if pc is bjacobi */
  ierr = PCBJacobiGetSubKSP(pc, &nksp, &first, &subksp);CHKERRQ(ierr);
  if (subksp) {
    for (i=0; i<nksp; i++) {
      ierr = KSPSetTolerances(subksp[i], inner_rtol, outer_abstol, outer_dtol, outer_maxits);CHKERRQ(ierr);
    }
    PetscFunctionReturn(0);
  }

  /* todo: dynamic tolerance may apply to other types of pc too */
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "KSPMonitorDynamicToleranceDestroy"
/*
 A hack to using dynamic tolerance in preconditioner
 */
PetscErrorCode KSPMonitorDynamicToleranceDestroy(void **dummy) {
  PetscErrorCode ierr;
  PetscFunctionBegin;
  ierr = PetscFree(*dummy);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "KSPMonitorDefaultShort"
/*
  Default (short) KSP Monitor, same as KSPMonitorDefault() except
  it prints fewer digits of the residual as the residual gets smaller.
  This is because the later digits are meaningless and are often
  different on different machines; by using this routine different
  machines will usually generate the same output.
*/
PetscErrorCode  KSPMonitorDefaultShort(KSP ksp,PetscInt its,PetscReal fnorm,void *dummy)
{
  PetscErrorCode ierr;
  PetscViewer    viewer = dummy ? (PetscViewer) dummy : PETSC_VIEWER_STDOUT_(((PetscObject)ksp)->comm);

  PetscFunctionBegin;
  ierr = PetscViewerASCIIAddTab(viewer,((PetscObject)ksp)->tablevel);CHKERRQ(ierr);
  if (its == 0 && ((PetscObject)ksp)->prefix) {
    ierr = PetscViewerASCIIPrintf(viewer,"  Residual norms for %s solve.\n",((PetscObject)ksp)->prefix);CHKERRQ(ierr);
  }

  if (fnorm > 1.e-9) {
    ierr = PetscViewerASCIIPrintf(viewer,"%3D KSP Residual norm %G \n",its,fnorm);CHKERRQ(ierr);
  } else if (fnorm > 1.e-11){
    ierr = PetscViewerASCIIPrintf(viewer,"%3D KSP Residual norm %5.3e \n",its,(double)fnorm);CHKERRQ(ierr);
  } else {
    ierr = PetscViewerASCIIPrintf(viewer,"%3D KSP Residual norm < 1.e-11\n",its);CHKERRQ(ierr);
  }
  ierr = PetscViewerASCIISubtractTab(viewer,((PetscObject)ksp)->tablevel);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "KSPSkipConverged"
/*@C
   KSPSkipConverged - Convergence test that do not return as converged
   until the maximum number of iterations is reached.

   Collective on KSP

   Input Parameters:
+  ksp   - iterative context
.  n     - iteration number
.  rnorm - 2-norm residual value (may be estimated)
-  dummy - unused convergence context

   Returns:
.  reason - KSP_CONVERGED_ITERATING, KSP_CONVERGED_ITS

   Notes:
   This should be used as the convergence test with the option
   KSPSetNormType(ksp,KSP_NORM_NONE), since norms of the residual are
   not computed. Convergence is then declared after the maximum number
   of iterations have been reached. Useful when one is using CG or
   BiCGStab as a smoother.

   Level: advanced

.keywords: KSP, default, convergence, residual

.seealso: KSPSetConvergenceTest(), KSPSetTolerances(), KSPSetNormType()
@*/
PetscErrorCode  KSPSkipConverged(KSP ksp,PetscInt n,PetscReal rnorm,KSPConvergedReason *reason,void *dummy)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ksp,KSP_CLASSID,1);
  PetscValidPointer(reason,4);
  *reason = KSP_CONVERGED_ITERATING;
  if (n >= ksp->max_it) *reason = KSP_CONVERGED_ITS;
  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "KSPDefaultConvergedCreate"
/*@C
   KSPDefaultConvergedCreate - Creates and initializes the space used by the KSPDefaultConverged() function context

   Collective on KSP

   Output Parameter:
.  ctx - convergence context

   Level: intermediate

.keywords: KSP, default, convergence, residual

.seealso: KSPDefaultConverged(), KSPDefaultConvergedDestroy(), KSPSetConvergenceTest(), KSPSetTolerances(),
          KSPSkipConverged(), KSPConvergedReason, KSPGetConvergedReason(), KSPDefaultConvergedSetUIRNorm(), KSPDefaultConvergedSetUMIRNorm()
@*/
PetscErrorCode  KSPDefaultConvergedCreate(void **ctx)
{
  PetscErrorCode         ierr;
  KSPDefaultConvergedCtx *cctx;

  PetscFunctionBegin;
  ierr = PetscNew(KSPDefaultConvergedCtx,&cctx);CHKERRQ(ierr);
  *ctx = cctx;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "KSPDefaultConvergedSetUIRNorm"
/*@
   KSPDefaultConvergedSetUIRNorm - makes the default convergence test use || B*(b - A*(initial guess))||
      instead of || B*b ||. In the case of right preconditioner or if KSPSetNormType(ksp,KSP_NORM_UNPRECONDIITONED)
      is used there is no B in the above formula. UIRNorm is short for Use Initial Residual Norm.

   Collective on KSP

   Input Parameters:
.  ksp   - iterative context

   Options Database:
.   -ksp_converged_use_initial_residual_norm

   Notes:
   Use KSPSetTolerances() to alter the defaults for rtol, abstol, dtol.

   The precise values of reason are macros such as KSP_CONVERGED_RTOL, which
   are defined in petscksp.h.

   If the convergence test is not KSPDefaultConverged() then this is ignored.

   If right preconditioning is being used then B does not appear in the above formula.


   Level: intermediate

.keywords: KSP, default, convergence, residual

.seealso: KSPSetConvergenceTest(), KSPSetTolerances(), KSPSkipConverged(), KSPConvergedReason, KSPGetConvergedReason(), KSPDefaultConvergedSetUMIRNorm()
@*/
PetscErrorCode  KSPDefaultConvergedSetUIRNorm(KSP ksp)
{
  KSPDefaultConvergedCtx *ctx = (KSPDefaultConvergedCtx*) ksp->cnvP;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ksp,KSP_CLASSID,1);
  if (ksp->converged != KSPDefaultConverged) PetscFunctionReturn(0);
  if (ctx->mininitialrtol) SETERRQ(((PetscObject)ksp)->comm,PETSC_ERR_ARG_WRONGSTATE,"Cannot use KSPDefaultConvergedSetUIRNorm() and KSPDefaultConvergedSetUMIRNorm() together");
  ctx->initialrtol = PETSC_TRUE;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "KSPDefaultConvergedSetUMIRNorm"
/*@
   KSPDefaultConvergedSetUMIRNorm - makes the default convergence test use min(|| B*(b - A*(initial guess))||,|| B*b ||)
      In the case of right preconditioner or if KSPSetNormType(ksp,KSP_NORM_UNPRECONDIITONED)
      is used there is no B in the above formula. UMIRNorm is short for Use Minimum Initial Residual Norm.

   Collective on KSP

   Input Parameters:
.  ksp   - iterative context

   Options Database:
.   -ksp_converged_use_min_initial_residual_norm

   Use KSPSetTolerances() to alter the defaults for rtol, abstol, dtol.

   The precise values of reason are macros such as KSP_CONVERGED_RTOL, which
   are defined in petscksp.h.

   Level: intermediate

.keywords: KSP, default, convergence, residual

.seealso: KSPSetConvergenceTest(), KSPSetTolerances(), KSPSkipConverged(), KSPConvergedReason, KSPGetConvergedReason(), KSPDefaultConvergedSetUIRNorm()
@*/
PetscErrorCode  KSPDefaultConvergedSetUMIRNorm(KSP ksp)
{
  KSPDefaultConvergedCtx *ctx = (KSPDefaultConvergedCtx*) ksp->cnvP;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ksp,KSP_CLASSID,1);
  if (ksp->converged != KSPDefaultConverged) PetscFunctionReturn(0);
  if (ctx->initialrtol) SETERRQ(((PetscObject)ksp)->comm,PETSC_ERR_ARG_WRONGSTATE,"Cannot use KSPDefaultConvergedSetUIRNorm() and KSPDefaultConvergedSetUMIRNorm() together");
  ctx->mininitialrtol = PETSC_TRUE;
   PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "KSPDefaultConverged"
/*@C
   KSPDefaultConverged - Determines convergence of
   the iterative solvers (default code).

   Collective on KSP

   Input Parameters:
+  ksp   - iterative context
.  n     - iteration number
.  rnorm - 2-norm residual value (may be estimated)
-  ctx - convergence context which must be created by KSPDefaultConvergedCreate()

   reason is set to:
+   positive - if the iteration has converged;
.   negative - if residual norm exceeds divergence threshold;
-   0 - otherwise.

   Notes:
   KSPDefaultConverged() reaches convergence when
$      rnorm < MAX (rtol * rnorm_0, abstol);
   Divergence is detected if
$      rnorm > dtol * rnorm_0,

   where
+     rtol = relative tolerance,
.     abstol = absolute tolerance.
.     dtol = divergence tolerance,
-     rnorm_0 is the two norm of the right hand side. When initial guess is non-zero you
          can call KSPDefaultConvergedSetUIRNorm() to use the norm of (b - A*(initial guess))
          as the starting point for relative norm convergence testing.

   Use KSPSetTolerances() to alter the defaults for rtol, abstol, dtol.

   The precise values of reason are macros such as KSP_CONVERGED_RTOL, which
   are defined in petscksp.h.

   Level: intermediate

.keywords: KSP, default, convergence, residual

.seealso: KSPSetConvergenceTest(), KSPSetTolerances(), KSPSkipConverged(), KSPConvergedReason, KSPGetConvergedReason(),
          KSPDefaultConvergedSetUIRNorm(), KSPDefaultConvergedSetUMIRNorm(), KSPDefaultConvergedCreate(), KSPDefaultConvergedDestroy()
@*/
PetscErrorCode  KSPDefaultConverged(KSP ksp,PetscInt n,PetscReal rnorm,KSPConvergedReason *reason,void *ctx)
{
  PetscErrorCode         ierr;
  KSPDefaultConvergedCtx *cctx = (KSPDefaultConvergedCtx*) ctx;
  KSPNormType            normtype;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ksp,KSP_CLASSID,1);
  PetscValidPointer(reason,4);
  *reason = KSP_CONVERGED_ITERATING;

  ierr = KSPGetNormType(ksp,&normtype);CHKERRQ(ierr);
  if (normtype == KSP_NORM_NONE) SETERRQ(((PetscObject)ksp)->comm,PETSC_ERR_ARG_WRONGSTATE,"Use KSPSkipConverged() with KSPNormType of KSP_NORM_NONE");

  if (!cctx) SETERRQ(((PetscObject)ksp)->comm,PETSC_ERR_ARG_NULL,"Convergence context must have been created with KSPDefaultConvergedCreate()");
  if (!n) {
    /* if user gives initial guess need to compute norm of b */
    if (!ksp->guess_zero && !cctx->initialrtol) {
      PetscReal      snorm;
      if (ksp->normtype == KSP_NORM_UNPRECONDITIONED || ksp->pc_side == PC_RIGHT) {
        ierr = PetscInfo(ksp,"user has provided nonzero initial guess, computing 2-norm of RHS\n");CHKERRQ(ierr);
        ierr = VecNorm(ksp->vec_rhs,NORM_2,&snorm);CHKERRQ(ierr);        /*     <- b'*b */
      } else {
        Vec z;
        /* Should avoid allocating the z vector each time but cannot stash it in cctx because if KSPReset() is called the vector size might change */
        ierr = VecDuplicate(ksp->vec_rhs,&z);CHKERRQ(ierr);
        ierr = KSP_PCApply(ksp,ksp->vec_rhs,z);CHKERRQ(ierr);
        if (ksp->normtype == KSP_NORM_PRECONDITIONED) {
          ierr = PetscInfo(ksp,"user has provided nonzero initial guess, computing 2-norm of preconditioned RHS\n");CHKERRQ(ierr);
          ierr = VecNorm(z,NORM_2,&snorm);CHKERRQ(ierr);                 /*    dp <- b'*B'*B*b */
        } else if (ksp->normtype == KSP_NORM_NATURAL) {
          PetscScalar norm;
           ierr = PetscInfo(ksp,"user has provided nonzero initial guess, computing natural norm of RHS\n");CHKERRQ(ierr);
          ierr  = VecDot(ksp->vec_rhs,z,&norm);
          snorm = PetscSqrtReal(PetscAbsScalar(norm));                            /*    dp <- b'*B*b */
        }
        ierr = VecDestroy(&z);CHKERRQ(ierr);
      }
      /* handle special case of zero RHS and nonzero guess */
      if (!snorm) {
        ierr = PetscInfo(ksp,"Special case, user has provided nonzero initial guess and zero RHS\n");CHKERRQ(ierr);
        snorm = rnorm;
      }
      if (cctx->mininitialrtol) {
        ksp->rnorm0 = PetscMin(snorm,rnorm);
      } else {
        ksp->rnorm0 = snorm;
      }
    } else {
      ksp->rnorm0 = rnorm;
    }
    ksp->ttol   = PetscMax(ksp->rtol*ksp->rnorm0,ksp->abstol);
  }

  if (n <= ksp->chknorm) PetscFunctionReturn(0);

  if (PetscIsInfOrNanScalar(rnorm)) {
    ierr = PetscInfo(ksp,"Linear solver has created a not a number (NaN) as the residual norm, declaring divergence \n");CHKERRQ(ierr);
    *reason = KSP_DIVERGED_NAN;
  } else if (rnorm <= ksp->ttol) {
    if (rnorm < ksp->abstol) {
      ierr = PetscInfo3(ksp,"Linear solver has converged. Residual norm %14.12e is less than absolute tolerance %14.12e at iteration %D\n",(double)rnorm,(double)ksp->abstol,n);CHKERRQ(ierr);
      *reason = KSP_CONVERGED_ATOL;
    } else {
      if (cctx->initialrtol) {
        ierr = PetscInfo4(ksp,"Linear solver has converged. Residual norm %14.12e is less than relative tolerance %14.12e times initial residual norm %14.12e at iteration %D\n",(double)rnorm,(double)ksp->rtol,(double)ksp->rnorm0,n);CHKERRQ(ierr);
      } else {
        ierr = PetscInfo4(ksp,"Linear solver has converged. Residual norm %14.12e is less than relative tolerance %14.12e times initial right hand side norm %14.12e at iteration %D\n",(double)rnorm,(double)ksp->rtol,(double)ksp->rnorm0,n);CHKERRQ(ierr);
      }
      *reason = KSP_CONVERGED_RTOL;
    }
  } else if (rnorm >= ksp->divtol*ksp->rnorm0) {
    ierr = PetscInfo3(ksp,"Linear solver is diverging. Initial right hand size norm %14.12e, current residual norm %14.12e at iteration %D\n",(double)ksp->rnorm0,(double)rnorm,n);CHKERRQ(ierr);
    *reason = KSP_DIVERGED_DTOL;
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "KSPDefaultConvergedDestroy"
/*@C
   KSPDefaultConvergedDestroy - Frees the space used by the KSPDefaultConverged() function context

   Collective on KSP

   Input Parameters:
.  ctx - convergence context

   Level: intermediate

.keywords: KSP, default, convergence, residual

.seealso: KSPDefaultConverged(), KSPDefaultConvergedCreate(), KSPSetConvergenceTest(), KSPSetTolerances(), KSPSkipConverged(),
          KSPConvergedReason, KSPGetConvergedReason(), KSPDefaultConvergedSetUIRNorm(), KSPDefaultConvergedSetUMIRNorm()
@*/
PetscErrorCode  KSPDefaultConvergedDestroy(void *ctx)
{
  PetscErrorCode         ierr;
  KSPDefaultConvergedCtx *cctx = (KSPDefaultConvergedCtx*) ctx;

  PetscFunctionBegin;
  ierr = VecDestroy(&cctx->work);CHKERRQ(ierr);
  ierr = PetscFree(ctx);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "KSPDefaultBuildSolution"
/*
   KSPDefaultBuildSolution - Default code to create/move the solution.

   Input Parameters:
+  ksp - iterative context
-  v   - pointer to the user's vector

   Output Parameter:
.  V - pointer to a vector containing the solution

   Level: advanced

.keywords:  KSP, build, solution, default

.seealso: KSPGetSolution(), KSPDefaultBuildResidual()
*/
PetscErrorCode KSPDefaultBuildSolution(KSP ksp,Vec v,Vec *V)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  if (ksp->pc_side == PC_RIGHT) {
    if (ksp->pc) {
      if (v) {ierr = KSP_PCApply(ksp,ksp->vec_sol,v);CHKERRQ(ierr); *V = v;}
      else SETERRQ(((PetscObject)ksp)->comm,PETSC_ERR_SUP,"Not working with right preconditioner");
    } else {
      if (v) {ierr = VecCopy(ksp->vec_sol,v);CHKERRQ(ierr); *V = v;}
      else { *V = ksp->vec_sol;}
    }
  } else if (ksp->pc_side == PC_SYMMETRIC) {
    if (ksp->pc) {
      if (ksp->transpose_solve) SETERRQ(((PetscObject)ksp)->comm,PETSC_ERR_SUP,"Not working with symmetric preconditioner and transpose solve");
      if (v) {ierr = PCApplySymmetricRight(ksp->pc,ksp->vec_sol,v);CHKERRQ(ierr); *V = v;}
      else SETERRQ(((PetscObject)ksp)->comm,PETSC_ERR_SUP,"Not working with symmetric preconditioner");
    } else  {
      if (v) {ierr = VecCopy(ksp->vec_sol,v);CHKERRQ(ierr); *V = v;}
      else { *V = ksp->vec_sol;}
    }
  } else {
    if (v) {ierr = VecCopy(ksp->vec_sol,v);CHKERRQ(ierr); *V = v;}
    else { *V = ksp->vec_sol; }
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "KSPDefaultBuildResidual"
/*
   KSPDefaultBuildResidual - Default code to compute the residual.

   Input Parameters:
.  ksp - iterative context
.  t   - pointer to temporary vector
.  v   - pointer to user vector

   Output Parameter:
.  V - pointer to a vector containing the residual

   Level: advanced

.keywords:  KSP, build, residual, default

.seealso: KSPDefaultBuildSolution()
*/
PetscErrorCode KSPDefaultBuildResidual(KSP ksp,Vec t,Vec v,Vec *V)
{
  PetscErrorCode ierr;
  MatStructure   pflag;
  Mat            Amat,Pmat;

  PetscFunctionBegin;
  if (!ksp->pc) {ierr = KSPGetPC(ksp,&ksp->pc);CHKERRQ(ierr);}
  ierr = PCGetOperators(ksp->pc,&Amat,&Pmat,&pflag);CHKERRQ(ierr);
  ierr = KSPBuildSolution(ksp,t,PETSC_NULL);CHKERRQ(ierr);
  ierr = KSP_MatMult(ksp,Amat,t,v);CHKERRQ(ierr);
  ierr = VecAYPX(v,-1.0,ksp->vec_rhs);CHKERRQ(ierr);
  *V = v;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "KSPGetVecs"
/*@C
  KSPGetVecs - Gets a number of work vectors.

  Input Parameters:
+ ksp  - iterative context
. rightn  - number of right work vectors
- leftn   - number of left work vectors to allocate

  Output Parameter:
+  right - the array of vectors created
-  left - the array of left vectors

   Note: The right vector has as many elements as the matrix has columns. The left
     vector has as many elements as the matrix has rows.

   Level: advanced

.seealso:   MatGetVecs()

@*/
PetscErrorCode KSPGetVecs(KSP ksp,PetscInt rightn, Vec **right,PetscInt leftn,Vec **left)
{
  PetscErrorCode ierr;
  Vec            vecr,vecl;

  PetscFunctionBegin;
  if (rightn) {
    if (!right) SETERRQ(((PetscObject)ksp)->comm,PETSC_ERR_ARG_INCOMP,"You asked for right vectors but did not pass a pointer to hold them");
    if (ksp->vec_sol) vecr = ksp->vec_sol;
    else {
      if (ksp->dm) {
	ierr = DMGetGlobalVector(ksp->dm,&vecr);CHKERRQ(ierr);
      } else {
	Mat mat;
	if (!ksp->pc) {ierr = KSPGetPC(ksp,&ksp->pc);CHKERRQ(ierr);}
	ierr = PCGetOperators(ksp->pc,&mat,PETSC_NULL,PETSC_NULL);CHKERRQ(ierr);
	ierr = MatGetVecs(mat,&vecr,PETSC_NULL);CHKERRQ(ierr);
      }
    }
    ierr = VecDuplicateVecs(vecr,rightn,right);CHKERRQ(ierr);
    if (!ksp->vec_sol) {
      if (ksp->dm) {
        ierr = DMRestoreGlobalVector(ksp->dm,&vecr);CHKERRQ(ierr);
      } else {
	ierr = VecDestroy(&vecr);CHKERRQ(ierr);
      }
    }
  }
  if (leftn) {
    if (!left) SETERRQ(((PetscObject)ksp)->comm,PETSC_ERR_ARG_INCOMP,"You asked for left vectors but did not pass a pointer to hold them");
    if (ksp->vec_rhs) vecl = ksp->vec_rhs;
    else {
      if (ksp->dm) {
	ierr = DMGetGlobalVector(ksp->dm,&vecl);CHKERRQ(ierr);
      } else {
	Mat mat;
	if (!ksp->pc) {ierr = KSPGetPC(ksp,&ksp->pc);CHKERRQ(ierr);}
	ierr = PCGetOperators(ksp->pc,&mat,PETSC_NULL,PETSC_NULL);CHKERRQ(ierr);
	ierr = MatGetVecs(mat,PETSC_NULL,&vecl);CHKERRQ(ierr);
      }
    }
    ierr = VecDuplicateVecs(vecl,leftn,left);CHKERRQ(ierr);
    if (!ksp->vec_rhs) {
      if (ksp->dm) {
        ierr = DMRestoreGlobalVector(ksp->dm,&vecl);CHKERRQ(ierr);
      } else {
	ierr = VecDestroy(&vecl);CHKERRQ(ierr);
      }
    }
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "KSPDefaultGetWork"
/*
  KSPDefaultGetWork - Gets a number of work vectors.

  Input Parameters:
. ksp  - iterative context
. nw   - number of work vectors to allocate

  Notes:
  Call this only if no work vectors have been allocated
 */
PetscErrorCode KSPDefaultGetWork(KSP ksp,PetscInt nw)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = VecDestroyVecs(ksp->nwork,&ksp->work);CHKERRQ(ierr);
  ksp->nwork = nw;
  ierr = KSPGetVecs(ksp,nw,&ksp->work,0,PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscLogObjectParents(ksp,nw,ksp->work);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "KSPDefaultDestroy"
/*
  KSPDefaultDestroy - Destroys a iterative context variable for methods with
  no separate context.  Preferred calling sequence KSPDestroy().

  Input Parameter:
. ksp - the iterative context
*/
PetscErrorCode KSPDefaultDestroy(KSP ksp)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ksp,KSP_CLASSID,1);
  ierr = PetscFree(ksp->data);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "KSPGetConvergedReason"
/*@
   KSPGetConvergedReason - Gets the reason the KSP iteration was stopped.

   Not Collective

   Input Parameter:
.  ksp - the KSP context

   Output Parameter:
.  reason - negative value indicates diverged, positive value converged, see KSPConvergedReason

   Possible values for reason:
+  KSP_CONVERGED_RTOL (residual 2-norm decreased by a factor of rtol, from 2-norm of right hand side)
.  KSP_CONVERGED_ATOL (residual 2-norm less than abstol)
.  KSP_CONVERGED_ITS (used by the preonly preconditioner that always uses ONE iteration, or when the KSPSkipConverged() convergence
           test routine is set.
.  KSP_CONVERGED_CG_NEG_CURVE
.  KSP_CONVERGED_CG_CONSTRAINED
.  KSP_CONVERGED_STEP_LENGTH
.  KSP_DIVERGED_ITS  (required more than its to reach convergence)
.  KSP_DIVERGED_DTOL (residual norm increased by a factor of divtol)
.  KSP_DIVERGED_NAN (residual norm became Not-a-number likely due to 0/0)
.  KSP_DIVERGED_BREAKDOWN (generic breakdown in method)
-  KSP_DIVERGED_BREAKDOWN_BICG (Initial residual is orthogonal to preconditioned initial
                                residual. Try a different preconditioner, or a different initial Level.)

   See also manual page for each reason.

   guess: beginner

   Notes: Can only be called after the call the KSPSolve() is complete.

   Level: intermediate

.keywords: KSP, nonlinear, set, convergence, test

.seealso: KSPSetConvergenceTest(), KSPDefaultConverged(), KSPSetTolerances(), KSPConvergedReason
@*/
PetscErrorCode  KSPGetConvergedReason(KSP ksp,KSPConvergedReason *reason)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ksp,KSP_CLASSID,1);
  PetscValidPointer(reason,2);
  *reason = ksp->reason;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "KSPSetDM"
/*@
   KSPSetDM - Sets the DM that may be used by some preconditioners

   Logically Collective on KSP

   Input Parameters:
+  ksp - the preconditioner context
-  dm - the dm

   Level: intermediate


.seealso: KSPGetDM(), KSPSetDM(), KSPGetDM()
@*/
PetscErrorCode  KSPSetDM(KSP ksp,DM dm)
{
  PetscErrorCode ierr;
  PC             pc;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ksp,KSP_CLASSID,1);
  if (dm) {ierr = PetscObjectReference((PetscObject)dm);CHKERRQ(ierr);}
  if (ksp->dm) {                /* Move the SNESDM context over to the new DM unless the new DM already has one */
    PetscContainer oldcontainer,container;
    KSPDM          kdm;
    ierr = PetscObjectQuery((PetscObject)ksp->dm,"KSPDM",(PetscObject*)&oldcontainer);CHKERRQ(ierr);
    ierr = PetscObjectQuery((PetscObject)dm,"KSPDM",(PetscObject*)&container);CHKERRQ(ierr);
    if (oldcontainer && ksp->dmAuto && !container) {
      ierr = DMKSPCopyContext(ksp->dm,dm);CHKERRQ(ierr);
      ierr = DMKSPGetContext(ksp->dm,&kdm);CHKERRQ(ierr);
      if (kdm->originaldm == ksp->dm) { /* Grant write privileges to the replacement DM */
        kdm->originaldm = dm;
      }
    }
    ierr = DMDestroy(&ksp->dm);CHKERRQ(ierr);
  }
  ksp->dm = dm;
  ksp->dmAuto = PETSC_FALSE;
  ierr = KSPGetPC(ksp,&pc);CHKERRQ(ierr);
  ierr = PCSetDM(pc,dm);CHKERRQ(ierr);
  ksp->dmActive = PETSC_TRUE;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "KSPSetDMActive"
/*@
   KSPSetDMActive - Indicates the DM should be used to generate the linear system matrix and right hand side

   Logically Collective on KSP

   Input Parameters:
+  ksp - the preconditioner context
-  flg - use the DM

   Level: intermediate

   Notes:
   By default KSPSetDM() sets the DM as active, call KSPSetDMActive(dm,PETSC_FALSE); after KSPSetDM(dm) to not have the KSP object use the DM to generate the matrices

.seealso: KSPGetDM(), KSPSetDM(), KSPGetDM()
@*/
PetscErrorCode  KSPSetDMActive(KSP ksp,PetscBool  flg)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ksp,KSP_CLASSID,1);
  PetscValidLogicalCollectiveBool(ksp,flg,2);
  ksp->dmActive = flg;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "KSPGetDM"
/*@
   KSPGetDM - Gets the DM that may be used by some preconditioners

   Not Collective

   Input Parameter:
. ksp - the preconditioner context

   Output Parameter:
.  dm - the dm

   Level: intermediate


.seealso: KSPSetDM(), KSPSetDM(), KSPGetDM()
@*/
PetscErrorCode  KSPGetDM(KSP ksp,DM *dm)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ksp,KSP_CLASSID,1);
  if (!ksp->dm) {
    ierr = DMShellCreate(((PetscObject)ksp)->comm,&ksp->dm);CHKERRQ(ierr);
    ksp->dmAuto = PETSC_TRUE;
  }
  *dm = ksp->dm;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "KSPSetApplicationContext"
/*@
   KSPSetApplicationContext - Sets the optional user-defined context for the linear solver.

   Logically Collective on KSP

   Input Parameters:
+  ksp - the KSP context
-  usrP - optional user context

   Level: intermediate

.keywords: KSP, set, application, context

.seealso: KSPGetApplicationContext()
@*/
PetscErrorCode  KSPSetApplicationContext(KSP ksp,void *usrP)
{
  PetscErrorCode ierr;
  PC             pc;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ksp,KSP_CLASSID,1);
  ksp->user = usrP;
  ierr      = KSPGetPC(ksp,&pc);CHKERRQ(ierr);
  ierr      = PCSetApplicationContext(pc,usrP);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "KSPGetApplicationContext"
/*@
   KSPGetApplicationContext - Gets the user-defined context for the linear solver.

   Not Collective

   Input Parameter:
.  ksp - KSP context

   Output Parameter:
.  usrP - user context

   Level: intermediate

.keywords: KSP, get, application, context

.seealso: KSPSetApplicationContext()
@*/
PetscErrorCode  KSPGetApplicationContext(KSP ksp,void *usrP)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ksp,KSP_CLASSID,1);
  *(void**)usrP = ksp->user;
  PetscFunctionReturn(0);
}
