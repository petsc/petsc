
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
  Vec            x;
  PetscViewer    viewer = vf->viewer;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(viewer,PETSC_VIEWER_CLASSID,4);
  CHKERRQ(SNESGetSolution(snes,&x));
  CHKERRQ(PetscViewerPushFormat(viewer,vf->format));
  CHKERRQ(VecView(x,viewer));
  CHKERRQ(PetscViewerPopFormat(viewer));
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
  Vec            x;
  PetscViewer    viewer = vf->viewer;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(viewer,PETSC_VIEWER_CLASSID,4);
  CHKERRQ(SNESGetFunction(snes,&x,NULL,NULL));
  CHKERRQ(PetscViewerPushFormat(viewer,vf->format));
  CHKERRQ(VecView(x,viewer));
  CHKERRQ(PetscViewerPopFormat(viewer));
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
  Vec            x;
  PetscViewer    viewer = vf->viewer;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(viewer,PETSC_VIEWER_CLASSID,4);
  CHKERRQ(SNESGetSolutionUpdate(snes,&x));
  CHKERRQ(PetscViewerPushFormat(viewer,vf->format));
  CHKERRQ(VecView(x,viewer));
  CHKERRQ(PetscViewerPopFormat(viewer));
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

  PetscFunctionBegin;
  PetscValidHeaderSpecific(viewer, PETSC_VIEWER_CLASSID, 4);
  CHKERRQ(SNESGetSolution(snes, &snes_solution));
  CHKERRQ(VecDuplicate(snes_solution, &work1));
  CHKERRQ(VecDuplicate(snes_solution, &work2));
  CHKERRQ(KSPBuildSolution(ksp, work1, NULL));
  CHKERRQ(VecAYPX(work1, -1.0, snes_solution));
  CHKERRQ(SNESComputeFunction(snes, work1, work2));
  CHKERRQ(VecNorm(work2, NORM_2, &snorm));
  CHKERRQ(VecDestroy(&work1));
  CHKERRQ(VecDestroy(&work2));

  CHKERRQ(PetscObjectGetTabLevel((PetscObject) ksp, &tablevel));
  CHKERRQ(PetscObjectGetOptionsPrefix((PetscObject) ksp, &prefix));
  CHKERRQ(PetscViewerPushFormat(viewer, format));
  CHKERRQ(PetscViewerASCIIAddTab(viewer, tablevel));
  if (n == 0 && prefix) CHKERRQ(PetscViewerASCIIPrintf(viewer, "  Residual norms for %s solve.\n", prefix));
  CHKERRQ(PetscViewerASCIIPrintf(viewer, "%3D SNES Residual norm %5.3e KSP Residual norm %5.3e \n", n, (double) snorm, (double) rnorm));
  CHKERRQ(PetscViewerASCIISubtractTab(viewer, tablevel));
  CHKERRQ(PetscViewerPopFormat(viewer));
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

  PetscFunctionBegin;
  PetscValidHeaderSpecific(viewer, PETSC_VIEWER_CLASSID, 4);
  PetscValidHeaderSpecific(lg, PETSC_DRAWLG_CLASSID, 4);
  CHKERRQ(SNESGetSolution(snes, &snes_solution));
  CHKERRQ(VecDuplicate(snes_solution, &work1));
  CHKERRQ(VecDuplicate(snes_solution, &work2));
  CHKERRQ(KSPBuildSolution(ksp, work1, NULL));
  CHKERRQ(VecAYPX(work1, -1.0, snes_solution));
  CHKERRQ(SNESComputeFunction(snes, work1, work2));
  CHKERRQ(VecNorm(work2, NORM_2, &snorm));
  CHKERRQ(VecDestroy(&work1));
  CHKERRQ(VecDestroy(&work2));

  CHKERRQ(PetscViewerPushFormat(viewer, format));
  if (!n) CHKERRQ(PetscDrawLGReset(lg));
  x[0] = (PetscReal) n;
  if (rnorm > 0.0) y[0] = PetscLog10Real(rnorm);
  else y[0] = -15.0;
  x[1] = (PetscReal) n;
  if (snorm > 0.0) y[1] = PetscLog10Real(snorm);
  else y[1] = -15.0;
  CHKERRQ(PetscDrawLGAddPoint(lg, x, y));
  CHKERRQ(KSPGetConvergedReason(ksp, &reason));
  if (n <= 20 || !(n % 5) || reason) {
    CHKERRQ(PetscDrawLGDraw(lg));
    CHKERRQ(PetscDrawLGSave(lg));
  }
  CHKERRQ(PetscViewerPopFormat(viewer));
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

  PetscFunctionBegin;
  CHKERRQ(PetscViewerAndFormatCreate(viewer, format, vf));
  (*vf)->data = ctx;
  CHKERRQ(KSPMonitorLGCreate(PetscObjectComm((PetscObject) viewer), NULL, NULL, "Log Residual Norm", 2, names, PETSC_DECIDE, PETSC_DECIDE, 400, 300, &(*vf)->lg));
  PetscFunctionReturn(0);
}

PetscErrorCode SNESMonitorDefaultSetUp(SNES snes, PetscViewerAndFormat *vf)
{
  PetscFunctionBegin;
  if (vf->format == PETSC_VIEWER_DRAW_LG) {
    CHKERRQ(KSPMonitorLGCreate(PetscObjectComm((PetscObject) vf->viewer), NULL, NULL, "Log Residual Norm", 1, NULL, PETSC_DECIDE, PETSC_DECIDE, 400, 300, &vf->lg));
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

  PetscFunctionBegin;
  PetscValidHeaderSpecific(viewer,PETSC_VIEWER_CLASSID,4);
  CHKERRQ(PetscObjectTypeCompare((PetscObject) viewer, PETSCVIEWERASCII, &isascii));
  CHKERRQ(PetscObjectTypeCompare((PetscObject) viewer, PETSCVIEWERDRAW,  &isdraw));
  CHKERRQ(PetscViewerPushFormat(viewer,format));
  if (isascii) {
    CHKERRQ(PetscViewerASCIIAddTab(viewer,((PetscObject)snes)->tablevel));
    CHKERRQ(PetscViewerASCIIPrintf(viewer,"%3D SNES Function norm %14.12e \n",its,(double)fgnorm));
    CHKERRQ(PetscViewerASCIISubtractTab(viewer,((PetscObject)snes)->tablevel));
  } else if (isdraw) {
    if (format == PETSC_VIEWER_DRAW_LG) {
      PetscDrawLG lg = (PetscDrawLG) vf->lg;
      PetscReal   x, y;

      PetscValidHeaderSpecific(lg,PETSC_DRAWLG_CLASSID,4);
      if (!its) CHKERRQ(PetscDrawLGReset(lg));
      x = (PetscReal) its;
      if (fgnorm > 0.0) y = PetscLog10Real(fgnorm);
      else y = -15.0;
      CHKERRQ(PetscDrawLGAddPoint(lg,&x,&y));
      if (its <= 20 || !(its % 5) || snes->reason) {
        CHKERRQ(PetscDrawLGDraw(lg));
        CHKERRQ(PetscDrawLGSave(lg));
      }
    }
  }
  CHKERRQ(PetscViewerPopFormat(viewer));
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
  PetscViewer    viewer = vf->viewer;
  KSP            ksp;
  Mat            J;
  Vec            v;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(viewer,PETSC_VIEWER_CLASSID,4);
  CHKERRQ(SNESGetKSP(snes,&ksp));
  CHKERRQ(KSPGetOperators(ksp,&J,NULL));
  CHKERRQ(MatCreateVecs(J,&v,NULL));
  CHKERRQ(MatGetRowMaxAbs(J,v,NULL));
  CHKERRQ(PetscViewerPushFormat(viewer,vf->format));
  CHKERRQ(PetscViewerASCIIAddTab(viewer,((PetscObject)snes)->tablevel));
  CHKERRQ(PetscViewerASCIIPrintf(viewer,"%3D SNES Jacobian maximum row entries \n"));
  CHKERRQ(VecView(v,viewer));
  CHKERRQ(PetscViewerASCIISubtractTab(viewer,((PetscObject)snes)->tablevel));
  CHKERRQ(PetscViewerPopFormat(viewer));
  CHKERRQ(VecDestroy(&v));
  PetscFunctionReturn(0);
}

PetscErrorCode SNESMonitorJacUpdateSpectrum(SNES snes,PetscInt it,PetscReal fnorm,PetscViewerAndFormat *vf)
{
  Vec            X;
  Mat            J,dJ,dJdense;
  PetscErrorCode (*func)(SNES,Vec,Mat,Mat,void*);
  PetscInt       n;
  PetscBLASInt   nb = 0,lwork;
  PetscReal      *eigr,*eigi;
  PetscScalar    *work;
  PetscScalar    *a;

  PetscFunctionBegin;
  if (it == 0) PetscFunctionReturn(0);
  /* create the difference between the current update and the current jacobian */
  CHKERRQ(SNESGetSolution(snes,&X));
  CHKERRQ(SNESGetJacobian(snes,NULL,&J,&func,NULL));
  CHKERRQ(MatDuplicate(J,MAT_COPY_VALUES,&dJ));
  CHKERRQ(SNESComputeJacobian(snes,X,dJ,dJ));
  CHKERRQ(MatAXPY(dJ,-1.0,J,SAME_NONZERO_PATTERN));

  /* compute the spectrum directly */
  CHKERRQ(MatConvert(dJ,MATSEQDENSE,MAT_INITIAL_MATRIX,&dJdense));
  CHKERRQ(MatGetSize(dJ,&n,NULL));
  CHKERRQ(PetscBLASIntCast(n,&nb));
  lwork = 3*nb;
  CHKERRQ(PetscMalloc1(n,&eigr));
  CHKERRQ(PetscMalloc1(n,&eigi));
  CHKERRQ(PetscMalloc1(lwork,&work));
  CHKERRQ(MatDenseGetArray(dJdense,&a));
#if !defined(PETSC_USE_COMPLEX)
  {
    PetscBLASInt lierr;
    PetscInt     i;
    CHKERRQ(PetscFPTrapPush(PETSC_FP_TRAP_OFF));
    PetscStackCallBLAS("LAPACKgeev",LAPACKgeev_("N","N",&nb,a,&nb,eigr,eigi,NULL,&nb,NULL,&nb,work,&lwork,&lierr));
    PetscCheck(!lierr,PETSC_COMM_SELF,PETSC_ERR_LIB,"geev() error %d",lierr);
    CHKERRQ(PetscFPTrapPop());
    CHKERRQ(PetscPrintf(PetscObjectComm((PetscObject)snes),"Eigenvalues of J_%d - J_%d:\n",it,it-1));
    for (i=0;i<n;i++) {
      CHKERRQ(PetscPrintf(PetscObjectComm((PetscObject)snes),"%5d: %20.5g + %20.5gi\n",i,(double)eigr[i],(double)eigi[i]));
    }
  }
  CHKERRQ(MatDenseRestoreArray(dJdense,&a));
  CHKERRQ(MatDestroy(&dJ));
  CHKERRQ(MatDestroy(&dJdense));
  CHKERRQ(PetscFree(eigr));
  CHKERRQ(PetscFree(eigi));
  CHKERRQ(PetscFree(work));
  PetscFunctionReturn(0);
#else
  SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"Not coded for complex");
#endif
}

PETSC_INTERN PetscErrorCode  SNESMonitorRange_Private(SNES,PetscInt,PetscReal*);

PetscErrorCode  SNESMonitorRange_Private(SNES snes,PetscInt it,PetscReal *per)
{
  Vec            resid;
  PetscReal      rmax,pwork;
  PetscInt       i,n,N;
  PetscScalar    *r;

  PetscFunctionBegin;
  CHKERRQ(SNESGetFunction(snes,&resid,NULL,NULL));
  CHKERRQ(VecNorm(resid,NORM_INFINITY,&rmax));
  CHKERRQ(VecGetLocalSize(resid,&n));
  CHKERRQ(VecGetSize(resid,&N));
  CHKERRQ(VecGetArray(resid,&r));
  pwork = 0.0;
  for (i=0; i<n; i++) {
    pwork += (PetscAbsScalar(r[i]) > .20*rmax);
  }
  CHKERRMPI(MPIU_Allreduce(&pwork,per,1,MPIU_REAL,MPIU_SUM,PetscObjectComm((PetscObject)snes)));
  CHKERRQ(VecRestoreArray(resid,&r));
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
  PetscReal      perc,rel;
  PetscViewer    viewer = vf->viewer;
  /* should be in a MonitorRangeContext */
  static PetscReal prev;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(viewer,PETSC_VIEWER_CLASSID,4);
  if (!it) prev = rnorm;
  CHKERRQ(SNESMonitorRange_Private(snes,it,&perc));

  rel  = (prev - rnorm)/prev;
  prev = rnorm;
  CHKERRQ(PetscViewerPushFormat(viewer,vf->format));
  CHKERRQ(PetscViewerASCIIAddTab(viewer,((PetscObject)snes)->tablevel));
  CHKERRQ(PetscViewerASCIIPrintf(viewer,"%3D SNES preconditioned resid norm %14.12e Percent values above 20 percent of maximum %5.2f relative decrease %5.2e ratio %5.2e \n",it,(double)rnorm,(double)(100.0*perc),(double)rel,(double)(rel/perc)));
  CHKERRQ(PetscViewerASCIISubtractTab(viewer,((PetscObject)snes)->tablevel));
  CHKERRQ(PetscViewerPopFormat(viewer));
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
  PetscInt                len;
  PetscReal               *history;
  PetscViewer             viewer = vf->viewer;

  PetscFunctionBegin;
  CHKERRQ(SNESGetConvergenceHistory(snes,&history,NULL,&len));
  CHKERRQ(PetscViewerPushFormat(viewer,vf->format));
  CHKERRQ(PetscViewerASCIIAddTab(viewer,((PetscObject)snes)->tablevel));
  if (!its || !history || its > len) {
    CHKERRQ(PetscViewerASCIIPrintf(viewer,"%3D SNES Function norm %14.12e \n",its,(double)fgnorm));
  } else {
    PetscReal ratio = fgnorm/history[its-1];
    CHKERRQ(PetscViewerASCIIPrintf(viewer,"%3D SNES Function norm %14.12e %14.12e \n",its,(double)fgnorm,(double)ratio));
  }
  CHKERRQ(PetscViewerASCIISubtractTab(viewer,((PetscObject)snes)->tablevel));
  CHKERRQ(PetscViewerPopFormat(viewer));
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
  PetscReal               *history;

  PetscFunctionBegin;
  CHKERRQ(SNESGetConvergenceHistory(snes,&history,NULL,NULL));
  if (!history) {
    CHKERRQ(SNESSetConvergenceHistory(snes,NULL,NULL,100,PETSC_TRUE));
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
  PetscViewer    viewer = vf->viewer;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(viewer,PETSC_VIEWER_CLASSID,4);
  CHKERRQ(PetscViewerPushFormat(viewer,vf->format));
  CHKERRQ(PetscViewerASCIIAddTab(viewer,((PetscObject)snes)->tablevel));
  if (fgnorm > 1.e-9) {
    CHKERRQ(PetscViewerASCIIPrintf(viewer,"%3D SNES Function norm %g \n",its,(double)fgnorm));
  } else if (fgnorm > 1.e-11) {
    CHKERRQ(PetscViewerASCIIPrintf(viewer,"%3D SNES Function norm %5.3e \n",its,(double)fgnorm));
  } else {
    CHKERRQ(PetscViewerASCIIPrintf(viewer,"%3D SNES Function norm < 1.e-11\n",its));
  }
  CHKERRQ(PetscViewerASCIISubtractTab(viewer,((PetscObject)snes)->tablevel));
  CHKERRQ(PetscViewerPopFormat(viewer));
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

  PetscFunctionBegin;
  PetscValidHeaderSpecific(viewer,PETSC_VIEWER_CLASSID,4);
  CHKERRQ(SNESGetFunction(snes, &r, NULL, NULL));
  CHKERRQ(VecGetDM(r, &dm));
  if (!dm) CHKERRQ(SNESMonitorDefault(snes, its, fgnorm, vf));
  else {
    PetscSection s, gs;
    PetscInt     Nf, f;

    CHKERRQ(DMGetLocalSection(dm, &s));
    CHKERRQ(DMGetGlobalSection(dm, &gs));
    if (!s || !gs) CHKERRQ(SNESMonitorDefault(snes, its, fgnorm, vf));
    CHKERRQ(PetscSectionGetNumFields(s, &Nf));
    PetscCheckFalse(Nf > 256,PetscObjectComm((PetscObject) snes), PETSC_ERR_SUP, "Do not support %d fields > 256", Nf);
    CHKERRQ(PetscSectionVecNorm(s, gs, r, NORM_2, res));
    CHKERRQ(PetscObjectGetTabLevel((PetscObject) snes, &tablevel));
    CHKERRQ(PetscViewerPushFormat(viewer,vf->format));
    CHKERRQ(PetscViewerASCIIAddTab(viewer, tablevel));
    CHKERRQ(PetscViewerASCIIPrintf(viewer, "%3D SNES Function norm %14.12e [", its, (double) fgnorm));
    for (f = 0; f < Nf; ++f) {
      if (f) CHKERRQ(PetscViewerASCIIPrintf(viewer, ", "));
      CHKERRQ(PetscViewerASCIIPrintf(viewer, "%14.12e", res[f]));
    }
    CHKERRQ(PetscViewerASCIIPrintf(viewer, "] \n"));
    CHKERRQ(PetscViewerASCIISubtractTab(viewer, tablevel));
    CHKERRQ(PetscViewerPopFormat(viewer));
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
    CHKERRQ(PetscInfo(snes,"Failed to converged, function norm is NaN\n"));
    *reason = SNES_DIVERGED_FNORM_NAN;
  } else if (fnorm < snes->abstol && (it || !snes->forceiteration)) {
    CHKERRQ(PetscInfo(snes,"Converged due to function norm %14.12e < %14.12e\n",(double)fnorm,(double)snes->abstol));
    *reason = SNES_CONVERGED_FNORM_ABS;
  } else if (snes->nfuncs >= snes->max_funcs && snes->max_funcs >= 0) {
    CHKERRQ(PetscInfo(snes,"Exceeded maximum number of function evaluations: %D > %D\n",snes->nfuncs,snes->max_funcs));
    *reason = SNES_DIVERGED_FUNCTION_COUNT;
  }

  if (it && !*reason) {
    if (fnorm <= snes->ttol) {
      CHKERRQ(PetscInfo(snes,"Converged due to function norm %14.12e < %14.12e (relative tolerance)\n",(double)fnorm,(double)snes->ttol));
      *reason = SNES_CONVERGED_FNORM_RELATIVE;
    } else if (snorm < snes->stol*xnorm) {
      CHKERRQ(PetscInfo(snes,"Converged due to small update length: %14.12e < %14.12e * %14.12e\n",(double)snorm,(double)snes->stol,(double)xnorm));
      *reason = SNES_CONVERGED_SNORM_RELATIVE;
    } else if (snes->divtol > 0 && (fnorm > snes->divtol*snes->rnorm0)) {
      CHKERRQ(PetscInfo(snes,"Diverged due to increase in function norm: %14.12e > %14.12e * %14.12e\n",(double)fnorm,(double)snes->divtol,(double)snes->rnorm0));
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
  PetscFunctionBegin;
  PetscValidHeaderSpecific(snes,SNES_CLASSID,1);
  PetscValidPointer(reason,6);

  *reason = SNES_CONVERGED_ITERATING;

  if (fnorm != fnorm) {
    CHKERRQ(PetscInfo(snes,"Failed to converged, function norm is NaN\n"));
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

  PetscFunctionBegin;
  if (snes->work) CHKERRQ(VecDestroyVecs(snes->nwork,&snes->work));
  snes->nwork = nw;

  CHKERRQ(SNESGetDM(snes, &dm));
  CHKERRQ(DMGetGlobalVector(dm, &v));
  CHKERRQ(VecDuplicateVecs(v,snes->nwork,&snes->work));
  CHKERRQ(DMRestoreGlobalVector(dm, &v));
  CHKERRQ(PetscLogObjectParents(snes,nw,snes->work));
  PetscFunctionReturn(0);
}
