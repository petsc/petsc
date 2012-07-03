#include <petsc-private/kspimpl.h>  /*I "petscksp.h" I*/

typedef struct {
  PetscViewer viewer;
  PetscInt    neigs;
  PetscReal   *eigi;
  PetscReal   *eigr;
#ifdef PETSC_HAVE_AMS
  AMS_Memory  amem;
#endif
} KSPMonitor_AMS;

#undef __FUNCT__
#define __FUNCT__ "KSPMonitorAMSCreate"
/*@C
   KSPMonitorAMSCreate - create an AMS monitor context

   Collective

   Input Arguments:
+  ksp - KSP to monitor
-  amscommname - name of AMS communicator to use

   Output Arguments:
.  ctx - context for monitor

   Level: developer

.seealso: KSPMonitorAMS(), KSPMonitorAMSDestroy()
@*/
PetscErrorCode KSPMonitorAMSCreate(KSP ksp,const char *amscommname,void **ctx)
{
  PetscErrorCode ierr;
  KSPMonitor_AMS *mon;

  PetscFunctionBegin;
  ierr = PetscNewLog(ksp,KSPMonitor_AMS,&mon);CHKERRQ(ierr);
#ifdef PETSC_HAVE_AMS
  ierr = PetscViewerAMSOpen(((PetscObject)ksp)->comm,amscommname,&mon->viewer);CHKERRQ(ierr);
  mon->amem = -1;
#endif
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
#ifdef PETSC_HAVE_AMS
  if (mon->amem != -1) {
    ierr = AMS_Memory_destroy(mon->amem);CHKERRQ(ierr);
    mon->amem = -1;
  }
#endif
  ierr = PetscViewerDestroy(&mon->viewer);CHKERRQ(ierr);
  ierr = PetscFree(mon->eigr);CHKERRQ(ierr);
  mon->eigi = PETSC_NULL;
  ierr = PetscFree(*ctx);CHKERRQ(ierr);
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
#if defined(PETSC_HAVE_AMS)
  PetscErrorCode ierr;
  KSPMonitor_AMS *mon = (KSPMonitor_AMS*)ctx;
  PetscViewer viewer = mon->viewer;
  PetscReal emax,emin;;
  AMS_Comm acomm;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ksp,KSP_CLASSID,1);
  PetscValidHeaderSpecific(viewer,PETSC_VIEWER_CLASSID,4);
  ierr = KSPComputeExtremeSingularValues(ksp,&emax,&emin);CHKERRQ(ierr);

  /* UnPublish  */
  if (mon->amem != -1) {ierr = AMS_Memory_destroy(mon->amem);CHKERRQ(ierr);}
  mon->amem = -1;

  ierr = PetscFree(mon->eigr);CHKERRQ(ierr);
  ierr = PetscMalloc(2*n*sizeof(PetscReal),&mon->eigr);CHKERRQ(ierr);
  mon->eigi = mon->eigr + n;
  if (n) {ierr = KSPComputeEigenvalues(ksp,n,mon->eigr,mon->eigi,&mon->neigs);CHKERRQ(ierr);}

  ierr = PetscViewerAMSGetAMSComm(viewer,&acomm);CHKERRQ(ierr);
  ierr = AMS_Memory_create(acomm,"ksp_monitor_ams",&mon->amem);CHKERRQ(ierr);
  ierr = AMS_Memory_take_access(mon->amem);CHKERRQ(ierr);

  ierr = AMS_Memory_add_field(mon->amem,"rnorm",&ksp->rnorm,1,AMS_DOUBLE,AMS_READ,AMS_COMMON,AMS_REDUCT_UNDEF);CHKERRQ(ierr);
  ierr = AMS_Memory_add_field(mon->amem,"neigs",&mon->neigs,1,AMS_INT,AMS_READ,AMS_COMMON,AMS_REDUCT_UNDEF);CHKERRQ(ierr);
  if (mon->neigs > 0) {
    ierr = AMS_Memory_add_field(mon->amem,"eigr",&mon->eigr,mon->neigs,AMS_DOUBLE,AMS_READ,AMS_COMMON,AMS_REDUCT_UNDEF);CHKERRQ(ierr);
    ierr = AMS_Memory_add_field(mon->amem,"eigi",&mon->eigr,mon->neigs,AMS_DOUBLE,AMS_READ,AMS_COMMON,AMS_REDUCT_UNDEF);CHKERRQ(ierr);
  }
  ierr = AMS_Memory_publish(mon->amem);CHKERRQ(ierr);
  ierr = AMS_Memory_grant_access(mon->amem);CHKERRQ(ierr);

  ierr = PetscInfo2(ksp,"KSP extreme singular values min=%G max=%G\n",emin,emax);CHKERRQ(ierr);
  PetscFunctionReturn(0);
#else
  PetscFunctionBegin;
  SETERRQ(((PetscObject)ksp)->comm,PETSC_ERR_SUP,"Missing package AMS");
  PetscFunctionReturn(0);
#endif
}
