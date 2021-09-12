#include <petsc/private/snesimpl.h>  /*I "petscsnes.h" I*/
#include <petscviewersaws.h>

typedef struct {
  PetscViewer    viewer;
} SNESMonitor_SAWs;

/*@C
   SNESMonitorSAWsCreate - create an SAWs monitor context

   Collective

   Input Parameter:
.  snes - SNES to monitor

   Output Parameter:
.  ctx - context for monitor

   Level: developer

.seealso: SNESMonitorSAWs(), SNESMonitorSAWsDestroy()
@*/
PetscErrorCode SNESMonitorSAWsCreate(SNES snes,void **ctx)
{
  PetscErrorCode  ierr;
  SNESMonitor_SAWs *mon;

  PetscFunctionBegin;
  ierr      = PetscNewLog(snes,&mon);CHKERRQ(ierr);
  mon->viewer = PETSC_VIEWER_SAWS_(PetscObjectComm((PetscObject)snes));
  if (!mon->viewer) SETERRQ(PetscObjectComm((PetscObject)snes),PETSC_ERR_PLIB,"Cannot create SAWs default viewer");
  *ctx = (void*)mon;
  PetscFunctionReturn(0);
}

/*@C
   SNESMonitorSAWsDestroy - destroy a monitor context created with SNESMonitorSAWsCreate()

   Collective

   Input Parameter:
.  ctx - monitor context

   Level: developer

.seealso: SNESMonitorSAWsCreate()
@*/
PetscErrorCode SNESMonitorSAWsDestroy(void **ctx)
{
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  ierr = PetscFree(*ctx);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
   SNESMonitorSAWs - monitor solution using SAWs

   Logically Collective on SNES

   Input Parameters:
+  snes   - iterative context
.  n     - iteration number
.  rnorm - 2-norm (preconditioned) residual value (may be estimated).
-  ctx -  PetscViewer of type SAWs

   Level: advanced

.seealso: PetscViewerSAWsOpen()
@*/
PetscErrorCode SNESMonitorSAWs(SNES snes,PetscInt n,PetscReal rnorm,void *ctx)
{
  PetscErrorCode   ierr;
  PetscMPIInt      rank;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(snes,SNES_CLASSID,1);

  ierr = MPI_Comm_rank(PETSC_COMM_WORLD,&rank);CHKERRMPI(ierr);
  if (!rank) {
    PetscStackCallSAWs(SAWs_Register,("/PETSc/snes_monitor_saws/its",&snes->iter,1,SAWs_READ,SAWs_INT));
    PetscStackCallSAWs(SAWs_Register,("/PETSc/snes_monitor_saws/rnorm",&snes->norm,1,SAWs_READ,SAWs_DOUBLE));
    ierr = PetscSAWsBlock();CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}
