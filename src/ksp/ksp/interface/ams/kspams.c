#include <petsc-private/kspimpl.h>  /*I "petscksp.h" I*/
#include <petscviewerams.h>

typedef struct {
  PetscViewer viewer;
  PetscInt    neigs;
  PetscReal   *eigi;
  PetscReal   *eigr;
  AMS_Memory amem;
} KSPMonitor_AMS;

#undef __FUNCT__
#define __FUNCT__ "KSPMonitorAMSCreate"
/*@C
   KSPMonitorAMSCreate - create an AMS monitor context

   Collective

   Input Arguments:
.  ksp - KSP to monitor

   Output Arguments:
.  ctx - context for monitor

   Level: developer

.seealso: KSPMonitorAMS(), KSPMonitorAMSDestroy()
@*/
PetscErrorCode KSPMonitorAMSCreate(KSP ksp,void **ctx)
{
  PetscErrorCode ierr;
  KSPMonitor_AMS *mon;

  PetscFunctionBegin;
  ierr      = PetscNewLog(ksp,KSPMonitor_AMS,&mon);CHKERRQ(ierr);
  mon->viewer = PETSC_VIEWER_AMS_(PetscObjectComm((PetscObject)ksp));
  if (!mon->viewer) SETERRQ(PetscObjectComm((PetscObject)ksp),PETSC_ERR_PLIB,"Cannot create AMS default viewer");CHKERRQ(ierr);
  mon->amem = NULL;
  *ctx = (void*)mon;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "KSPMonitorAMSDestroy"
/*@C
   KSPMonitorAMSDestroy - destroy a monitor context created with KSPMonitorAMSCreate()

   Collective

   Input Arguments:
.  ctx - monitor context

   Level: developer

.seealso: KSPMonitorAMSCreate()
@*/
PetscErrorCode KSPMonitorAMSDestroy(void **ctx)
{
  KSPMonitor_AMS *mon = (KSPMonitor_AMS*)*ctx;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (mon->amem) {
    PetscStackCallAMS(AMS_Memory_Destroy,(mon->amem));
    mon->amem = NULL;
  }
  /* ierr      = PetscViewerDestroy(&mon->viewer);CHKERRQ(ierr);*/
  ierr      = PetscFree(mon->eigr);CHKERRQ(ierr);
  mon->eigi = NULL;
  ierr      = PetscFree(*ctx);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "KSPMonitorAMS"
/*@C
   KSPMonitorAMS - monitor solution using AMS

   Logically Collective on KSP

   Input Parameters:
+  ksp   - iterative context
.  n     - iteration number
.  rnorm - 2-norm (preconditioned) residual value (may be estimated).
-  ctx -  PetscViewer of type AMS

   Level: advanced

.keywords: KSP, CG, monitor, AMS, singular values

.seealso: KSPMonitorSingularValue(), KSPComputeExtremeSingularValues(), PetscViewerAMSOpen()
@*/
PetscErrorCode KSPMonitorAMS(KSP ksp,PetscInt n,PetscReal rnorm,void *ctx)
{
  PetscErrorCode ierr;
  KSPMonitor_AMS *mon   = (KSPMonitor_AMS*)ctx;
  PetscViewer    viewer = mon->viewer;
  PetscReal      emax,emin;;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ksp,KSP_CLASSID,1);
  PetscValidHeaderSpecific(viewer,PETSC_VIEWER_CLASSID,4);
  ierr = KSPComputeExtremeSingularValues(ksp,&emax,&emin);CHKERRQ(ierr);

  /* UnPublish  */
  if (mon->amem) PetscStackCallAMS(AMS_Memory_Destroy,(mon->amem));
  mon->amem = NULL;

  ierr      = PetscFree(mon->eigr);CHKERRQ(ierr);
  ierr      = PetscMalloc(2*n*sizeof(PetscReal),&mon->eigr);CHKERRQ(ierr);
  mon->eigi = mon->eigr + n;
  if (n) {ierr = KSPComputeEigenvalues(ksp,n,mon->eigr,mon->eigi,&mon->neigs);CHKERRQ(ierr);}

  PetscStackCallAMS(AMS_Memory_Create,("ksp_monitor_ams",&mon->amem));
  PetscStackCallAMS(AMS_New_Field,(mon->amem,"rnorm",&ksp->rnorm,1,AMS_READ,AMS_DOUBLE));
  PetscStackCallAMS(AMS_New_Field,(mon->amem,"neigs",&mon->neigs,1,AMS_READ,AMS_INT));
  if (mon->neigs > 0) {
    PetscStackCallAMS(AMS_New_Field,(mon->amem,"eigr",&mon->eigr,mon->neigs,AMS_READ,AMS_DOUBLE));
    PetscStackCallAMS(AMS_New_Field,(mon->amem,"eigi",&mon->eigr,mon->neigs,AMS_READ,AMS_DOUBLE));
  }
  ierr = PetscObjectAMSBlock((PetscObject)ksp);CHKERRQ(ierr);
  ierr = PetscInfo2(ksp,"KSP extreme singular values min=%G max=%G\n",emin,emax);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
