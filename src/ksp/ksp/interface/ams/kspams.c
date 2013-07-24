#include <petsc-private/kspimpl.h>  /*I "petscksp.h" I*/
#include <petscviewersaws.h>

typedef struct {
  PetscViewer viewer;
  PetscInt    neigs;
  PetscReal   *eigi;
  PetscReal   *eigr;
  SAWS_Directory amem;
} KSPMonitor_SAWs;

#undef __FUNCT__
#define __FUNCT__ "KSPMonitorSAWsCreate"
/*@C
   KSPMonitorSAWsCreate - create an SAWs monitor context

   Collective

   Input Arguments:
.  ksp - KSP to monitor

   Output Arguments:
.  ctx - context for monitor

   Level: developer

.seealso: KSPMonitorSAWs(), KSPMonitorSAWsDestroy()
@*/
PetscErrorCode KSPMonitorSAWsCreate(KSP ksp,void **ctx)
{
  PetscErrorCode  ierr;
  KSPMonitor_SAWs *mon;

  PetscFunctionBegin;
  ierr      = PetscNewLog(ksp,KSPMonitor_SAWs,&mon);CHKERRQ(ierr);
  mon->viewer = PETSC_VIEWER_SAWS_(PetscObjectComm((PetscObject)ksp));
  if (!mon->viewer) SETERRQ(PetscObjectComm((PetscObject)ksp),PETSC_ERR_PLIB,"Cannot create SAWs default viewer");CHKERRQ(ierr);
  mon->amem = NULL;
  *ctx = (void*)mon;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "KSPMonitorSAWsDestroy"
/*@C
   KSPMonitorSAWsDestroy - destroy a monitor context created with KSPMonitorSAWsCreate()

   Collective

   Input Arguments:
.  ctx - monitor context

   Level: developer

.seealso: KSPMonitorSAWsCreate()
@*/
PetscErrorCode KSPMonitorSAWsDestroy(void **ctx)
{
  KSPMonitor_SAWs *mon = (KSPMonitor_SAWs*)*ctx;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  if (mon->amem) {
    PetscStackCallSAWs(SAWS_Directory_Destroy,(&mon->amem));
  }
  /* ierr      = PetscViewerDestroy(&mon->viewer);CHKERRQ(ierr);*/
  ierr      = PetscFree(mon->eigr);CHKERRQ(ierr);
  mon->eigi = NULL;
  ierr      = PetscFree(*ctx);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "KSPMonitorSAWs"
/*@C
   KSPMonitorSAWs - monitor solution using AMS

   Logically Collective on KSP

   Input Parameters:
+  ksp   - iterative context
.  n     - iteration number
.  rnorm - 2-norm (preconditioned) residual value (may be estimated).
-  ctx -  PetscViewer of type AMS

   Level: advanced

.keywords: KSP, CG, monitor, AMS, singular values

.seealso: KSPMonitorSingularValue(), KSPComputeExtremeSingularValues(), PetscViewerSAWsOpen()
@*/
PetscErrorCode KSPMonitorSAWs(KSP ksp,PetscInt n,PetscReal rnorm,void *ctx)
{
  PetscErrorCode  ierr;
  KSPMonitor_SAWs *mon   = (KSPMonitor_SAWs*)ctx;
  PetscViewer     viewer = mon->viewer;
  PetscReal       emax,emin;;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ksp,KSP_CLASSID,1);
  PetscValidHeaderSpecific(viewer,PETSC_VIEWER_CLASSID,4);
  ierr = KSPComputeExtremeSingularValues(ksp,&emax,&emin);CHKERRQ(ierr);

  /* UnPublish  */
  if (mon->amem) PetscStackCallSAWs(SAWS_Directory_Destroy,(&mon->amem));

  ierr      = PetscFree(mon->eigr);CHKERRQ(ierr);
  ierr      = PetscMalloc(2*n*sizeof(PetscReal),&mon->eigr);CHKERRQ(ierr);
  mon->eigi = mon->eigr + n;
  if (n) {ierr = KSPComputeEigenvalues(ksp,n,mon->eigr,mon->eigi,&mon->neigs);CHKERRQ(ierr);}

  PetscStackCallSAWs(SAWS_Directory_Create,("ksp_monitor_saws",&mon->amem));
  PetscStackCallSAWs(SAWS_New_Variable,(mon->amem,"rnorm",&ksp->rnorm,1,SAWS_READ,SAWS_DOUBLE));
  PetscStackCallSAWs(SAWS_New_Variable,(mon->amem,"neigs",&mon->neigs,1,SAWS_READ,SAWS_INT));
  if (mon->neigs > 0) {
    PetscStackCallSAWs(SAWS_New_Variable,(mon->amem,"eigr",&mon->eigr,mon->neigs,SAWS_READ,SAWS_DOUBLE));
    PetscStackCallSAWs(SAWS_New_Variable,(mon->amem,"eigi",&mon->eigr,mon->neigs,SAWS_READ,SAWS_DOUBLE));
  }
  ierr = PetscObjectSAWsBlock((PetscObject)ksp);CHKERRQ(ierr);
  ierr = PetscInfo2(ksp,"KSP extreme singular values min=%G max=%G\n",emin,emax);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
