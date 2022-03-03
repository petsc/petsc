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
  SNESMonitor_SAWs *mon;

  PetscFunctionBegin;
  CHKERRQ(PetscNewLog(snes,&mon));
  mon->viewer = PETSC_VIEWER_SAWS_(PetscObjectComm((PetscObject)snes));
  PetscCheck(mon->viewer,PetscObjectComm((PetscObject)snes),PETSC_ERR_PLIB,"Cannot create SAWs default viewer");
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
  PetscFunctionBegin;
  CHKERRQ(PetscFree(*ctx));
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
  PetscMPIInt      rank;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(snes,SNES_CLASSID,1);

  CHKERRMPI(MPI_Comm_rank(PETSC_COMM_WORLD,&rank));
  if (rank == 0) {
    PetscStackCallSAWs(SAWs_Register,("/PETSc/snes_monitor_saws/its",&snes->iter,1,SAWs_READ,SAWs_INT));
    PetscStackCallSAWs(SAWs_Register,("/PETSc/snes_monitor_saws/rnorm",&snes->norm,1,SAWs_READ,SAWs_DOUBLE));
    CHKERRQ(PetscSAWsBlock());
  }
  PetscFunctionReturn(0);
}
