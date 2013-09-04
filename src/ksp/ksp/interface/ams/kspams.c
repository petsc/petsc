#include <petsc-private/kspimpl.h>  /*I "petscksp.h" I*/
#include <petscviewersaws.h>

typedef struct {
  PetscViewer    viewer;
  PetscInt       neigs;
  PetscReal      *eigi;
  PetscReal      *eigr;
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
  ierr      = PetscFree2(mon->eigr,mon->eigi);CHKERRQ(ierr);
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
  PetscReal       emax,emin;
  PetscMPIInt     rank;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ksp,KSP_CLASSID,1);
  PetscValidHeaderSpecific(viewer,PETSC_VIEWER_CLASSID,4);
  ierr = KSPComputeExtremeSingularValues(ksp,&emax,&emin);CHKERRQ(ierr);

  ierr      = PetscFree(mon->eigr);CHKERRQ(ierr);
  ierr      = PetscMalloc2(n,PetscReal,&mon->eigr,n,PetscReal,&mon->eigi);CHKERRQ(ierr);
  if (n) {ierr = KSPComputeEigenvalues(ksp,n,mon->eigr,mon->eigi,&mon->neigs);CHKERRQ(ierr);}

  ierr = MPI_Comm_rank(PETSC_COMM_WORLD,&rank);CHKERRQ(ierr);
  if (!rank) {
    PetscStackCallSAWs(SAWs_Register,("/PETSc/ksp_monitor_saws/rnorm",&ksp->rnorm,1,SAWs_READ,SAWs_DOUBLE));
    PetscStackCallSAWs(SAWs_Register,("/PETSc/ksp_monitor_saws/neigs",&mon->neigs,1,SAWs_READ,SAWs_INT));
    if (mon->neigs > 0) {
      PetscStackCallSAWs(SAWs_Register,("/PETSc/ksp_monitor_saws/eigr",&mon->eigr,mon->neigs,SAWs_READ,SAWs_DOUBLE));
      PetscStackCallSAWs(SAWs_Register,("/PETSc/ksp_monitor_saws/eigi",&mon->eigr,mon->neigs,SAWs_READ,SAWs_DOUBLE));
    }
    ierr = PetscObjectSAWsBlock((PetscObject)ksp);CHKERRQ(ierr);
  }
  ierr = PetscInfo2(ksp,"KSP extreme singular values min=%G max=%G\n",emin,emax);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
