
#include <petsc/private/snesimpl.h>       /*I   "petsc/private/snesimpl.h"   I*/
#include <petscdm.h>
#include <petscsection.h>
#include <petscblaslapack.h>

/*@C
   SNESMonitorSolution - Monitors progress of the SNES solvers by calling
   VecView() for the approximate solution at each iteration.

   Collective on SNES

   Input Parameters:
+  snes - the SNES context
.  its - iteration number
.  fgnorm - 2-norm of residual
-  dummy -  a viewer

   Options Database Keys:
.  -snes_monitor_solution [ascii binary draw][:filename][:viewer format] - plots solution at each iteration

   Level: intermediate

.seealso: SNESMonitorSet(), SNESMonitorDefault(), VecView()
@*/
PetscErrorCode  SNESMonitorSolution(SNES snes,PetscInt its,PetscReal fgnorm,PetscViewerAndFormat *vf)
{
  PetscErrorCode ierr;
  Vec            x;
  PetscViewer    viewer = vf->viewer;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(viewer,PETSC_VIEWER_CLASSID,4);
  ierr = SNESGetSolution(snes,&x);CHKERRQ(ierr);
  ierr = PetscViewerPushFormat(viewer,vf->format);CHKERRQ(ierr);
  ierr = VecView(x,viewer);CHKERRQ(ierr);
  ierr = PetscViewerPopFormat(viewer);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
   SNESMonitorResidual - Monitors progress of the SNES solvers by calling
   VecView() for the residual at each iteration.

   Collective on SNES

   Input Parameters:
+  snes - the SNES context
.  its - iteration number
.  fgnorm - 2-norm of residual
-  dummy -  a viewer

   Options Database Keys:
.  -snes_monitor_residual [ascii binary draw][:filename][:viewer format] - plots residual (not its norm) at each iteration

   Level: intermediate

.seealso: SNESMonitorSet(), SNESMonitorDefault(), VecView()
@*/
PetscErrorCode  SNESMonitorResidual(SNES snes,PetscInt its,PetscReal fgnorm,PetscViewerAndFormat *vf)
{
  PetscErrorCode ierr;
  Vec            x;
  PetscViewer    viewer = vf->viewer;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(viewer,PETSC_VIEWER_CLASSID,4);
  ierr = SNESGetFunction(snes,&x,NULL,NULL);CHKERRQ(ierr);
  ierr = PetscViewerPushFormat(viewer,vf->format);CHKERRQ(ierr);
  ierr = VecView(x,viewer);CHKERRQ(ierr);
  ierr = PetscViewerPopFormat(viewer);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
   SNESMonitorSolutionUpdate - Monitors progress of the SNES solvers by calling
   VecView() for the UPDATE to the solution at each iteration.

   Collective on SNES

   Input Parameters:
+  snes - the SNES context
.  its - iteration number
.  fgnorm - 2-norm of residual
-  dummy - a viewer

   Options Database Keys:
.  -snes_monitor_solution_update [ascii binary draw][:filename][:viewer format] - plots update to solution at each iteration

   Level: intermediate

.seealso: SNESMonitorSet(), SNESMonitorDefault(), VecView()
@*/
PetscErrorCode  SNESMonitorSolutionUpdate(SNES snes,PetscInt its,PetscReal fgnorm,PetscViewerAndFormat *vf)
{
  PetscErrorCode ierr;
  Vec            x;
  PetscViewer    viewer = vf->viewer;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(viewer,PETSC_VIEWER_CLASSID,4);
  ierr = SNESGetSolutionUpdate(snes,&x);CHKERRQ(ierr);
  ierr = PetscViewerPushFormat(viewer,vf->format);CHKERRQ(ierr);
  ierr = VecView(x,viewer);CHKERRQ(ierr);
  ierr = PetscViewerPopFormat(viewer);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#include <petscdraw.h>

/*@C
  KSPMonitorSNESResidual - Prints the SNES residual norm, as well as the linear residual norm, at each iteration of an iterative solver.

  Collective on ksp

  Input Parameters:
+ ksp   - iterative context
. n     - iteration number
. rnorm - 2-norm (preconditioned) residual value (may be estimated).
- vf    - The viewer context

  Options Database Key:
. -snes_monitor_ksp - Activates KSPMonitorSNESResidual()

  Level: intermediate

.seealso: KSPMonitorSet(), KSPMonitorResidual(),KSPMonitorTrueResidualMaxNorm()
@*/
PetscErrorCode KSPMonitorSNESResidual(KSP ksp, PetscInt n, PetscReal rnorm, PetscViewerAndFormat *vf)
{
  PetscViewer       viewer = vf->viewer;
  PetscViewerFormat format = vf->format;
  SNES              snes   = (SNES) vf->data;
  Vec               snes_solution, work1, work2;
  PetscReal         snorm;
  PetscInt          tablevel;
  const char       *prefix;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(viewer, PETSC_VIEWER_CLASSID, 4);
  ierr = SNESGetSolution(snes, &snes_solution);CHKERRQ(ierr);
  ierr = VecDuplicate(snes_solution, &work1);CHKERRQ(ierr);
  ierr = VecDuplicate(snes_solution, &work2);CHKERRQ(ierr);
  ierr = KSPBuildSolution(ksp, work1, NULL);CHKERRQ(ierr);
  ierr = VecAYPX(work1, -1.0, snes_solution);CHKERRQ(ierr);
  ierr = SNESComputeFunction(snes, work1, work2);CHKERRQ(ierr);
  ierr = VecNorm(work2, NORM_2, &snorm);CHKERRQ(ierr);
  ierr = VecDestroy(&work1);CHKERRQ(ierr);
  ierr = VecDestroy(&work2);CHKERRQ(ierr);

  ierr = PetscObjectGetTabLevel((PetscObject) ksp, &tablevel);CHKERRQ(ierr);
  ierr = PetscObjectGetOptionsPrefix((PetscObject) ksp, &prefix);CHKERRQ(ierr);
  ierr = PetscViewerPushFormat(viewer, format);CHKERRQ(ierr);
  ierr = PetscViewerASCIIAddTab(viewer, tablevel);CHKERRQ(ierr);
  if (n == 0 && prefix) {ierr = PetscViewerASCIIPrintf(viewer, "  Residual norms for %s solve.\n", prefix);CHKERRQ(ierr);}
  ierr = PetscViewerASCIIPrintf(viewer, "%3D SNES Residual norm %5.3e KSP Residual norm %5.3e \n", n, (double) snorm, (double) rnorm);CHKERRQ(ierr);
  ierr = PetscViewerASCIISubtractTab(viewer, tablevel);CHKERRQ(ierr);
  ierr = PetscViewerPopFormat(viewer);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
  KSPMonitorSNESResidualDrawLG - Plots the linear SNES residual norm at each iteration of an iterative solver.

  Collective on ksp

  Input Parameters:
+ ksp   - iterative context
. n     - iteration number
. rnorm - 2-norm (preconditioned) residual value (may be estimated).
- vf    - The viewer context

  Options Database Key:
. -snes_monitor_ksp draw::draw_lg - Activates KSPMonitorSNESResidualDrawLG()

  Level: intermediate

.seealso: KSPMonitorSet(), KSPMonitorTrueResidual()
@*/
PetscErrorCode KSPMonitorSNESResidualDrawLG(KSP ksp, PetscInt n, PetscReal rnorm, PetscViewerAndFormat *vf)
{
  PetscViewer        viewer = vf->viewer;
  PetscViewerFormat  format = vf->format;
  PetscDrawLG        lg     = vf->lg;
  SNES               snes   = (SNES) vf->data;
  Vec                snes_solution, work1, work2;
  PetscReal          snorm;
  KSPConvergedReason reason;
  PetscReal          x[2], y[2];
  PetscErrorCode     ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(viewer, PETSC_VIEWER_CLASSID, 4);
  PetscValidHeaderSpecific(lg, PETSC_DRAWLG_CLASSID, 4);
  ierr = SNESGetSolution(snes, &snes_solution);CHKERRQ(ierr);
  ierr = VecDuplicate(snes_solution, &work1);CHKERRQ(ierr);
  ierr = VecDuplicate(snes_solution, &work2);CHKERRQ(ierr);
  ierr = KSPBuildSolution(ksp, work1, NULL);CHKERRQ(ierr);
  ierr = VecAYPX(work1, -1.0, snes_solution);CHKERRQ(ierr);
  ierr = SNESComputeFunction(snes, work1, work2);CHKERRQ(ierr);
  ierr = VecNorm(work2, NORM_2, &snorm);CHKERRQ(ierr);
  ierr = VecDestroy(&work1);CHKERRQ(ierr);
  ierr = VecDestroy(&work2);CHKERRQ(ierr);

  ierr = PetscViewerPushFormat(viewer, format);CHKERRQ(ierr);
  if (!n) {ierr = PetscDrawLGReset(lg);CHKERRQ(ierr);}
  x[0] = (PetscReal) n;
  if (rnorm > 0.0) y[0] = PetscLog10Real(rnorm);
  else y[0] = -15.0;
  x[1] = (PetscReal) n;
  if (snorm > 0.0) y[1] = PetscLog10Real(snorm);
  else y[1] = -15.0;
  ierr = PetscDrawLGAddPoint(lg, x, y);CHKERRQ(ierr);
  ierr = KSPGetConvergedReason(ksp, &reason);CHKERRQ(ierr);
  if (n <= 20 || !(n % 5) || reason) {
    ierr = PetscDrawLGDraw(lg);CHKERRQ(ierr);
    ierr = PetscDrawLGSave(lg);CHKERRQ(ierr);
  }
  ierr = PetscViewerPopFormat(viewer);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
  KSPMonitorSNESResidualDrawLGCreate - Creates the plotter for the linear SNES residual.

  Collective on ksp

  Input Parameters:
+ viewer - The PetscViewer
. format - The viewer format
- ctx    - An optional user context

  Output Parameter:
. vf    - The viewer context

  Level: intermediate

.seealso: KSPMonitorSet(), KSPMonitorTrueResidual()
@*/
PetscErrorCode KSPMonitorSNESResidualDrawLGCreate(PetscViewer viewer, PetscViewerFormat format, void *ctx, PetscViewerAndFormat **vf)
{
  const char    *names[] = {"linear", "nonlinear"};
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscViewerAndFormatCreate(viewer, format, vf);CHKERRQ(ierr);
  (*vf)->data = ctx;
  ierr = KSPMonitorLGCreate(PetscObjectComm((PetscObject) viewer), NULL, NULL, "Log Residual Norm", 2, names, PETSC_DECIDE, PETSC_DECIDE, 400, 300, &(*vf)->lg);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode SNESMonitorDefaultSetUp(SNES snes, PetscViewerAndFormat *vf)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (vf->format == PETSC_VIEWER_DRAW_LG) {
    ierr = KSPMonitorLGCreate(PetscObjectComm((PetscObject) vf->viewer), NULL, NULL, "Log Residual Norm", 1, NULL, PETSC_DECIDE, PETSC_DECIDE, 400, 300, &vf->lg);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

/*@C
   SNESMonitorDefault - Monitors progress of the SNES solvers (default).

   Collective on SNES

   Input Parameters:
+  snes - the SNES context
.  its - iteration number
.  fgnorm - 2-norm of residual
-  vf - viewer and format structure

   Notes:
   This routine prints the residual norm at each iteration.

   Level: intermediate

.seealso: SNESMonitorSet(), SNESMonitorSolution()
@*/
PetscErrorCode  SNESMonitorDefault(SNES snes,PetscInt its,PetscReal fgnorm,PetscViewerAndFormat *vf)
{
  PetscViewer       viewer = vf->viewer;
  PetscViewerFormat format = vf->format;
  PetscBool         isascii, isdraw;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(viewer,PETSC_VIEWER_CLASSID,4);
  ierr = PetscObjectTypeCompare((PetscObject) viewer, PETSCVIEWERASCII, &isascii);CHKERRQ(ierr);
  ierr = PetscObjectTypeCompare((PetscObject) viewer, PETSCVIEWERDRAW,  &isdraw);CHKERRQ(ierr);
  ierr = PetscViewerPushFormat(viewer,format);CHKERRQ(ierr);
  if (isascii) {
    ierr = PetscViewerASCIIAddTab(viewer,((PetscObject)snes)->tablevel);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"%3D SNES Function norm %14.12e \n",its,(double)fgnorm);CHKERRQ(ierr);
    ierr = PetscViewerASCIISubtractTab(viewer,((PetscObject)snes)->tablevel);CHKERRQ(ierr);
  } else if (isdraw) {
    if (format == PETSC_VIEWER_DRAW_LG) {
      PetscDrawLG lg = (PetscDrawLG) vf->lg;
      PetscReal   x, y;

      PetscValidHeaderSpecific(lg,PETSC_DRAWLG_CLASSID,4);
      if (!its) {ierr = PetscDrawLGReset(lg);CHKERRQ(ierr);}
      x = (PetscReal) its;
      if (fgnorm > 0.0) y = PetscLog10Real(fgnorm);
      else y = -15.0;
      ierr = PetscDrawLGAddPoint(lg,&x,&y);CHKERRQ(ierr);
      if (its <= 20 || !(its % 5) || snes->reason) {
        ierr = PetscDrawLGDraw(lg);CHKERRQ(ierr);
        ierr = PetscDrawLGSave(lg);CHKERRQ(ierr);
      }
    }
  }
  ierr = PetscViewerPopFormat(viewer);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
   SNESMonitorScaling - Monitors the largest value in each row of the Jacobian.

   Collective on SNES

   Input Parameters:
+  snes - the SNES context
.  its - iteration number
.  fgnorm - 2-norm of residual
-  vf - viewer and format structure

   Notes:
   This routine prints the largest value in each row of the Jacobian

   Level: intermediate

.seealso: SNESMonitorSet(), SNESMonitorSolution()
@*/
PetscErrorCode  SNESMonitorScaling(SNES snes,PetscInt its,PetscReal fgnorm,PetscViewerAndFormat *vf)
{
  PetscErrorCode ierr;
  PetscViewer    viewer = vf->viewer;
  KSP            ksp;
  Mat            J;
  Vec            v;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(viewer,PETSC_VIEWER_CLASSID,4);
  ierr = SNESGetKSP(snes,&ksp);CHKERRQ(ierr);
  ierr = KSPGetOperators(ksp,&J,NULL);CHKERRQ(ierr);
  ierr = MatCreateVecs(J,&v,NULL);CHKERRQ(ierr);
  ierr = MatGetRowMaxAbs(J,v,NULL);CHKERRQ(ierr);
  ierr = PetscViewerPushFormat(viewer,vf->format);CHKERRQ(ierr);
  ierr = PetscViewerASCIIAddTab(viewer,((PetscObject)snes)->tablevel);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"%3D SNES Jacobian maximum row entries \n");CHKERRQ(ierr);
  ierr = VecView(v,viewer);CHKERRQ(ierr);
  ierr = PetscViewerASCIISubtractTab(viewer,((PetscObject)snes)->tablevel);CHKERRQ(ierr);
  ierr = PetscViewerPopFormat(viewer);CHKERRQ(ierr);
  ierr = VecDestroy(&v);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode SNESMonitorJacUpdateSpectrum(SNES snes,PetscInt it,PetscReal fnorm,PetscViewerAndFormat *vf)
{
  Vec            X;
  Mat            J,dJ,dJdense;
  PetscErrorCode ierr;
  PetscErrorCode (*func)(SNES,Vec,Mat,Mat,void*);
  PetscInt       n;
  PetscBLASInt   nb = 0,lwork;
  PetscReal      *eigr,*eigi;
  PetscScalar    *work;
  PetscScalar    *a;

  PetscFunctionBegin;
  if (it == 0) PetscFunctionReturn(0);
  /* create the difference between the current update and the current jacobian */
  ierr = SNESGetSolution(snes,&X);CHKERRQ(ierr);
  ierr = SNESGetJacobian(snes,NULL,&J,&func,NULL);CHKERRQ(ierr);
  ierr = MatDuplicate(J,MAT_COPY_VALUES,&dJ);CHKERRQ(ierr);
  ierr = SNESComputeJacobian(snes,X,dJ,dJ);CHKERRQ(ierr);
  ierr = MatAXPY(dJ,-1.0,J,SAME_NONZERO_PATTERN);CHKERRQ(ierr);

  /* compute the spectrum directly */
  ierr  = MatConvert(dJ,MATSEQDENSE,MAT_INITIAL_MATRIX,&dJdense);CHKERRQ(ierr);
  ierr  = MatGetSize(dJ,&n,NULL);CHKERRQ(ierr);
  ierr  = PetscBLASIntCast(n,&nb);CHKERRQ(ierr);
  lwork = 3*nb;
  ierr  = PetscMalloc1(n,&eigr);CHKERRQ(ierr);
  ierr  = PetscMalloc1(n,&eigi);CHKERRQ(ierr);
  ierr  = PetscMalloc1(lwork,&work);CHKERRQ(ierr);
  ierr  = MatDenseGetArray(dJdense,&a);CHKERRQ(ierr);
#if !defined(PETSC_USE_COMPLEX)
  {
    PetscBLASInt lierr;
    PetscInt     i;
    ierr = PetscFPTrapPush(PETSC_FP_TRAP_OFF);CHKERRQ(ierr);
    PetscStackCallBLAS("LAPACKgeev",LAPACKgeev_("N","N",&nb,a,&nb,eigr,eigi,NULL,&nb,NULL,&nb,work,&lwork,&lierr));
    if (lierr) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_LIB,"geev() error %d",lierr);
    ierr = PetscFPTrapPop();CHKERRQ(ierr);
    ierr = PetscPrintf(PetscObjectComm((PetscObject)snes),"Eigenvalues of J_%d - J_%d:\n",it,it-1);CHKERRQ(ierr);
    for (i=0;i<n;i++) {
      ierr = PetscPrintf(PetscObjectComm((PetscObject)snes),"%5d: %20.5g + %20.5gi\n",i,(double)eigr[i],(double)eigi[i]);CHKERRQ(ierr);
    }
  }
  ierr = MatDenseRestoreArray(dJdense,&a);CHKERRQ(ierr);
  ierr = MatDestroy(&dJ);CHKERRQ(ierr);
  ierr = MatDestroy(&dJdense);CHKERRQ(ierr);
  ierr = PetscFree(eigr);CHKERRQ(ierr);
  ierr = PetscFree(eigi);CHKERRQ(ierr);
  ierr = PetscFree(work);CHKERRQ(ierr);
  PetscFunctionReturn(0);
#else
  SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"Not coded for complex");
#endif
}

PETSC_INTERN PetscErrorCode  SNESMonitorRange_Private(SNES,PetscInt,PetscReal*);

PetscErrorCode  SNESMonitorRange_Private(SNES snes,PetscInt it,PetscReal *per)
{
  PetscErrorCode ierr;
  Vec            resid;
  PetscReal      rmax,pwork;
  PetscInt       i,n,N;
  PetscScalar    *r;

  PetscFunctionBegin;
  ierr  = SNESGetFunction(snes,&resid,NULL,NULL);CHKERRQ(ierr);
  ierr  = VecNorm(resid,NORM_INFINITY,&rmax);CHKERRQ(ierr);
  ierr  = VecGetLocalSize(resid,&n);CHKERRQ(ierr);
  ierr  = VecGetSize(resid,&N);CHKERRQ(ierr);
  ierr  = VecGetArray(resid,&r);CHKERRQ(ierr);
  pwork = 0.0;
  for (i=0; i<n; i++) {
    pwork += (PetscAbsScalar(r[i]) > .20*rmax);
  }
  ierr = MPIU_Allreduce(&pwork,per,1,MPIU_REAL,MPIU_SUM,PetscObjectComm((PetscObject)snes));CHKERRMPI(ierr);
  ierr = VecRestoreArray(resid,&r);CHKERRQ(ierr);
  *per = *per/N;
  PetscFunctionReturn(0);
}

/*@C
   SNESMonitorRange - Prints the percentage of residual elements that are more then 10 percent of the maximum value.

   Collective on SNES

   Input Parameters:
+  snes   - iterative context
.  it    - iteration number
.  rnorm - 2-norm (preconditioned) residual value (may be estimated).
-  dummy - unused monitor context

   Options Database Key:
.  -snes_monitor_range - Activates SNESMonitorRange()

   Level: intermediate

.seealso: SNESMonitorSet(), SNESMonitorDefault(), SNESMonitorLGCreate()
@*/
PetscErrorCode  SNESMonitorRange(SNES snes,PetscInt it,PetscReal rnorm,PetscViewerAndFormat *vf)
{
  PetscErrorCode ierr;
  PetscReal      perc,rel;
  PetscViewer    viewer = vf->viewer;
  /* should be in a MonitorRangeContext */
  static PetscReal prev;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(viewer,PETSC_VIEWER_CLASSID,4);
  if (!it) prev = rnorm;
  ierr = SNESMonitorRange_Private(snes,it,&perc);CHKERRQ(ierr);

  rel  = (prev - rnorm)/prev;
  prev = rnorm;
  ierr = PetscViewerPushFormat(viewer,vf->format);CHKERRQ(ierr);
  ierr = PetscViewerASCIIAddTab(viewer,((PetscObject)snes)->tablevel);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"%3D SNES preconditioned resid norm %14.12e Percent values above 20 percent of maximum %5.2f relative decrease %5.2e ratio %5.2e \n",it,(double)rnorm,(double)(100.0*perc),(double)rel,(double)(rel/perc));CHKERRQ(ierr);
  ierr = PetscViewerASCIISubtractTab(viewer,((PetscObject)snes)->tablevel);CHKERRQ(ierr);
  ierr = PetscViewerPopFormat(viewer);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
   SNESMonitorRatio - Monitors progress of the SNES solvers by printing the ratio
   of residual norm at each iteration to the previous.

   Collective on SNES

   Input Parameters:
+  snes - the SNES context
.  its - iteration number
.  fgnorm - 2-norm of residual (or gradient)
-  dummy -  context of monitor

   Level: intermediate

   Notes:
    Insure that SNESMonitorRatio() is called when you set this monitor
.seealso: SNESMonitorSet(), SNESMonitorSolution(), SNESMonitorRatio()
@*/
PetscErrorCode  SNESMonitorRatio(SNES snes,PetscInt its,PetscReal fgnorm,PetscViewerAndFormat *vf)
{
  PetscErrorCode          ierr;
  PetscInt                len;
  PetscReal               *history;
  PetscViewer             viewer = vf->viewer;

  PetscFunctionBegin;
  ierr = SNESGetConvergenceHistory(snes,&history,NULL,&len);CHKERRQ(ierr);
  ierr = PetscViewerPushFormat(viewer,vf->format);CHKERRQ(ierr);
  ierr = PetscViewerASCIIAddTab(viewer,((PetscObject)snes)->tablevel);CHKERRQ(ierr);
  if (!its || !history || its > len) {
    ierr = PetscViewerASCIIPrintf(viewer,"%3D SNES Function norm %14.12e \n",its,(double)fgnorm);CHKERRQ(ierr);
  } else {
    PetscReal ratio = fgnorm/history[its-1];
    ierr = PetscViewerASCIIPrintf(viewer,"%3D SNES Function norm %14.12e %14.12e \n",its,(double)fgnorm,(double)ratio);CHKERRQ(ierr);
  }
  ierr = PetscViewerASCIISubtractTab(viewer,((PetscObject)snes)->tablevel);CHKERRQ(ierr);
  ierr = PetscViewerPopFormat(viewer);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
   SNESMonitorRatioSetUp - Insures the SNES object is saving its history since this monitor needs access to it

   Collective on SNES

   Input Parameters:
+   snes - the SNES context
-   viewer - the PetscViewer object (ignored)

   Level: intermediate

.seealso: SNESMonitorSet(), SNESMonitorSolution(), SNESMonitorDefault(), SNESMonitorRatio()
@*/
PetscErrorCode  SNESMonitorRatioSetUp(SNES snes,PetscViewerAndFormat *vf)
{
  PetscErrorCode          ierr;
  PetscReal               *history;

  PetscFunctionBegin;
  ierr = SNESGetConvergenceHistory(snes,&history,NULL,NULL);CHKERRQ(ierr);
  if (!history) {
    ierr = SNESSetConvergenceHistory(snes,NULL,NULL,100,PETSC_TRUE);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

/* ---------------------------------------------------------------- */
/*
     Default (short) SNES Monitor, same as SNESMonitorDefault() except
  it prints fewer digits of the residual as the residual gets smaller.
  This is because the later digits are meaningless and are often
  different on different machines; by using this routine different
  machines will usually generate the same output.

  Deprecated: Intentionally has no manual page
*/
PetscErrorCode  SNESMonitorDefaultShort(SNES snes,PetscInt its,PetscReal fgnorm,PetscViewerAndFormat *vf)
{
  PetscErrorCode ierr;
  PetscViewer    viewer = vf->viewer;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(viewer,PETSC_VIEWER_CLASSID,4);
  ierr = PetscViewerPushFormat(viewer,vf->format);CHKERRQ(ierr);
  ierr = PetscViewerASCIIAddTab(viewer,((PetscObject)snes)->tablevel);CHKERRQ(ierr);
  if (fgnorm > 1.e-9) {
    ierr = PetscViewerASCIIPrintf(viewer,"%3D SNES Function norm %g \n",its,(double)fgnorm);CHKERRQ(ierr);
  } else if (fgnorm > 1.e-11) {
    ierr = PetscViewerASCIIPrintf(viewer,"%3D SNES Function norm %5.3e \n",its,(double)fgnorm);CHKERRQ(ierr);
  } else {
    ierr = PetscViewerASCIIPrintf(viewer,"%3D SNES Function norm < 1.e-11\n",its);CHKERRQ(ierr);
  }
  ierr = PetscViewerASCIISubtractTab(viewer,((PetscObject)snes)->tablevel);CHKERRQ(ierr);
  ierr = PetscViewerPopFormat(viewer);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
  SNESMonitorDefaultField - Monitors progress of the SNES solvers, separated into fields.

  Collective on SNES

  Input Parameters:
+ snes   - the SNES context
. its    - iteration number
. fgnorm - 2-norm of residual
- ctx    - the PetscViewer

  Notes:
  This routine uses the DM attached to the residual vector

  Level: intermediate

.seealso: SNESMonitorSet(), SNESMonitorSolution(), SNESMonitorDefault()
@*/
PetscErrorCode SNESMonitorDefaultField(SNES snes, PetscInt its, PetscReal fgnorm, PetscViewerAndFormat *vf)
{
  PetscViewer    viewer = vf->viewer;
  Vec            r;
  DM             dm;
  PetscReal      res[256];
  PetscInt       tablevel;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(viewer,PETSC_VIEWER_CLASSID,4);
  ierr = SNESGetFunction(snes, &r, NULL, NULL);CHKERRQ(ierr);
  ierr = VecGetDM(r, &dm);CHKERRQ(ierr);
  if (!dm) {ierr = SNESMonitorDefault(snes, its, fgnorm, vf);CHKERRQ(ierr);}
  else {
    PetscSection s, gs;
    PetscInt     Nf, f;

    ierr = DMGetLocalSection(dm, &s);CHKERRQ(ierr);
    ierr = DMGetGlobalSection(dm, &gs);CHKERRQ(ierr);
    if (!s || !gs) {ierr = SNESMonitorDefault(snes, its, fgnorm, vf);CHKERRQ(ierr);}
    ierr = PetscSectionGetNumFields(s, &Nf);CHKERRQ(ierr);
    if (Nf > 256) SETERRQ(PetscObjectComm((PetscObject) snes), PETSC_ERR_SUP, "Do not support %d fields > 256", Nf);
    ierr = PetscSectionVecNorm(s, gs, r, NORM_2, res);CHKERRQ(ierr);
    ierr = PetscObjectGetTabLevel((PetscObject) snes, &tablevel);CHKERRQ(ierr);
    ierr = PetscViewerPushFormat(viewer,vf->format);CHKERRQ(ierr);
    ierr = PetscViewerASCIIAddTab(viewer, tablevel);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer, "%3D SNES Function norm %14.12e [", its, (double) fgnorm);CHKERRQ(ierr);
    for (f = 0; f < Nf; ++f) {
      if (f) {ierr = PetscViewerASCIIPrintf(viewer, ", ");CHKERRQ(ierr);}
      ierr = PetscViewerASCIIPrintf(viewer, "%14.12e", res[f]);CHKERRQ(ierr);
    }
    ierr = PetscViewerASCIIPrintf(viewer, "] \n");CHKERRQ(ierr);
    ierr = PetscViewerASCIISubtractTab(viewer, tablevel);CHKERRQ(ierr);
    ierr = PetscViewerPopFormat(viewer);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}
/* ---------------------------------------------------------------- */
/*@C
   SNESConvergedDefault - Default onvergence test of the solvers for
   systems of nonlinear equations.

   Collective on SNES

   Input Parameters:
+  snes - the SNES context
.  it - the iteration (0 indicates before any Newton steps)
.  xnorm - 2-norm of current iterate
.  snorm - 2-norm of current step
.  fnorm - 2-norm of function at current iterate
-  dummy - unused context

   Output Parameter:
.   reason  - one of
$  SNES_CONVERGED_FNORM_ABS       - (fnorm < abstol),
$  SNES_CONVERGED_SNORM_RELATIVE  - (snorm < stol*xnorm),
$  SNES_CONVERGED_FNORM_RELATIVE  - (fnorm < rtol*fnorm0),
$  SNES_DIVERGED_FUNCTION_COUNT   - (nfct > maxf),
$  SNES_DIVERGED_FNORM_NAN        - (fnorm == NaN),
$  SNES_CONVERGED_ITERATING       - (otherwise),
$  SNES_DIVERGED_DTOL             - (fnorm > divtol*snes->fnorm0)

   where
+    maxf - maximum number of function evaluations,  set with SNESSetTolerances()
.    nfct - number of function evaluations,
.    abstol - absolute function norm tolerance, set with SNESSetTolerances()
.    rtol - relative function norm tolerance, set with SNESSetTolerances()
.    divtol - divergence tolerance, set with SNESSetDivergenceTolerance()
-    fnorm0 - 2-norm of the function at the initial solution (initial guess; zeroth iteration)

  Options Database Keys:
+  -snes_stol - convergence tolerance in terms of the norm  of the change in the solution between steps
.  -snes_atol <abstol> - absolute tolerance of residual norm
.  -snes_rtol <rtol> - relative decrease in tolerance norm from the initial 2-norm of the solution
.  -snes_divergence_tolerance <divtol> - if the residual goes above divtol*rnorm0, exit with divergence
.  -snes_max_funcs <max_funcs> - maximum number of function evaluations
.  -snes_max_fail <max_fail> - maximum number of line search failures allowed before stopping, default is none
-  -snes_max_linear_solve_fail - number of linear solver failures before SNESSolve() stops

   Level: intermediate

.seealso: SNESSetConvergenceTest(), SNESConvergedSkip(), SNESSetTolerances(), SNESSetDivergenceTolerance()
@*/
PetscErrorCode  SNESConvergedDefault(SNES snes,PetscInt it,PetscReal xnorm,PetscReal snorm,PetscReal fnorm,SNESConvergedReason *reason,void *dummy)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(snes,SNES_CLASSID,1);
  PetscValidPointer(reason,6);

  *reason = SNES_CONVERGED_ITERATING;

  if (!it) {
    /* set parameter for default relative tolerance convergence test */
    snes->ttol   = fnorm*snes->rtol;
    snes->rnorm0 = fnorm;
  }
  if (PetscIsInfOrNanReal(fnorm)) {
    ierr    = PetscInfo(snes,"Failed to converged, function norm is NaN\n");CHKERRQ(ierr);
    *reason = SNES_DIVERGED_FNORM_NAN;
  } else if (fnorm < snes->abstol && (it || !snes->forceiteration)) {
    ierr    = PetscInfo2(snes,"Converged due to function norm %14.12e < %14.12e\n",(double)fnorm,(double)snes->abstol);CHKERRQ(ierr);
    *reason = SNES_CONVERGED_FNORM_ABS;
  } else if (snes->nfuncs >= snes->max_funcs && snes->max_funcs >= 0) {
    ierr    = PetscInfo2(snes,"Exceeded maximum number of function evaluations: %D > %D\n",snes->nfuncs,snes->max_funcs);CHKERRQ(ierr);
    *reason = SNES_DIVERGED_FUNCTION_COUNT;
  }

  if (it && !*reason) {
    if (fnorm <= snes->ttol) {
      ierr    = PetscInfo2(snes,"Converged due to function norm %14.12e < %14.12e (relative tolerance)\n",(double)fnorm,(double)snes->ttol);CHKERRQ(ierr);
      *reason = SNES_CONVERGED_FNORM_RELATIVE;
    } else if (snorm < snes->stol*xnorm) {
      ierr    = PetscInfo3(snes,"Converged due to small update length: %14.12e < %14.12e * %14.12e\n",(double)snorm,(double)snes->stol,(double)xnorm);CHKERRQ(ierr);
      *reason = SNES_CONVERGED_SNORM_RELATIVE;
    } else if (snes->divtol > 0 && (fnorm > snes->divtol*snes->rnorm0)) {
      ierr    = PetscInfo3(snes,"Diverged due to increase in function norm: %14.12e > %14.12e * %14.12e\n",(double)fnorm,(double)snes->divtol,(double)snes->rnorm0);CHKERRQ(ierr);
      *reason = SNES_DIVERGED_DTOL;
    }

  }
  PetscFunctionReturn(0);
}

/*@C
   SNESConvergedSkip - Convergence test for SNES that NEVER returns as
   converged, UNLESS the maximum number of iteration have been reached.

   Logically Collective on SNES

   Input Parameters:
+  snes - the SNES context
.  it - the iteration (0 indicates before any Newton steps)
.  xnorm - 2-norm of current iterate
.  snorm - 2-norm of current step
.  fnorm - 2-norm of function at current iterate
-  dummy - unused context

   Output Parameter:
.   reason  - SNES_CONVERGED_ITERATING, SNES_CONVERGED_ITS, or SNES_DIVERGED_FNORM_NAN

   Notes:
   Convergence is then declared after a fixed number of iterations have been used.

   Level: advanced

.seealso: SNESConvergedDefault(), SNESSetConvergenceTest()
@*/
PetscErrorCode  SNESConvergedSkip(SNES snes,PetscInt it,PetscReal xnorm,PetscReal snorm,PetscReal fnorm,SNESConvergedReason *reason,void *dummy)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(snes,SNES_CLASSID,1);
  PetscValidPointer(reason,6);

  *reason = SNES_CONVERGED_ITERATING;

  if (fnorm != fnorm) {
    ierr    = PetscInfo(snes,"Failed to converged, function norm is NaN\n");CHKERRQ(ierr);
    *reason = SNES_DIVERGED_FNORM_NAN;
  } else if (it == snes->max_its) {
    *reason = SNES_CONVERGED_ITS;
  }
  PetscFunctionReturn(0);
}

/*@C
  SNESSetWorkVecs - Gets a number of work vectors.

  Input Parameters:
+ snes  - the SNES context
- nw - number of work vectors to allocate

  Level: developer

@*/
PetscErrorCode SNESSetWorkVecs(SNES snes,PetscInt nw)
{
  DM             dm;
  Vec            v;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (snes->work) {ierr = VecDestroyVecs(snes->nwork,&snes->work);CHKERRQ(ierr);}
  snes->nwork = nw;

  ierr = SNESGetDM(snes, &dm);CHKERRQ(ierr);
  ierr = DMGetGlobalVector(dm, &v);CHKERRQ(ierr);
  ierr = VecDuplicateVecs(v,snes->nwork,&snes->work);CHKERRQ(ierr);
  ierr = DMRestoreGlobalVector(dm, &v);CHKERRQ(ierr);
  ierr = PetscLogObjectParents(snes,nw,snes->work);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
