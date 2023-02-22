#include <petsc/private/snesimpl.h> /*I "petscsnes.h" I*/
#include <petscviewersaws.h>

typedef struct {
  PetscViewer viewer;
} SNESMonitor_SAWs;

/*@C
   SNESMonitorSAWsCreate - create an SAWs monitor context for `SNES`

   Collective

   Input Parameter:
.  snes - `SNES` to monitor

   Output Parameter:
.  ctx - context for monitor

   Level: developer

.seealso: `SNESSetMonitor()`, `SNES`, `SNESMonitorSAWs()`, `SNESMonitorSAWsDestroy()`
@*/
PetscErrorCode SNESMonitorSAWsCreate(SNES snes, void **ctx)
{
  SNESMonitor_SAWs *mon;

  PetscFunctionBegin;
  PetscCall(PetscNew(&mon));
  mon->viewer = PETSC_VIEWER_SAWS_(PetscObjectComm((PetscObject)snes));
  PetscCheck(mon->viewer, PetscObjectComm((PetscObject)snes), PETSC_ERR_PLIB, "Cannot create SAWs default viewer");
  *ctx = (void *)mon;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
   SNESMonitorSAWsDestroy - destroy a monitor context created with `SNESMonitorSAWsCreate()`

   Collective

   Input Parameter:
.  ctx - monitor context

   Level: developer

.seealso: `SNESMonitorSAWsCreate()`
@*/
PetscErrorCode SNESMonitorSAWsDestroy(void **ctx)
{
  PetscFunctionBegin;
  PetscCall(PetscFree(*ctx));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
   SNESMonitorSAWs - monitor solution process of `SNES` using SAWs

   Collective

   Input Parameters:
+  snes   - iterative context
.  n     - iteration number
.  rnorm - 2-norm (preconditioned) residual value (may be estimated).
-  ctx -  `PetscViewer` of type `PETSCVIEWERSAWS`

   Level: advanced

.seealso: `PetscViewerSAWsOpen()`, `SNESMonitorSAWsDestroy()`, `SNESMonitorSAWsCreate()`
@*/
PetscErrorCode SNESMonitorSAWs(SNES snes, PetscInt n, PetscReal rnorm, void *ctx)
{
  PetscMPIInt rank;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(snes, SNES_CLASSID, 1);

  PetscCallMPI(MPI_Comm_rank(PETSC_COMM_WORLD, &rank));
  if (rank == 0) {
    PetscCallSAWs(SAWs_Register, ("/PETSc/snes_monitor_saws/its", &snes->iter, 1, SAWs_READ, SAWs_INT));
    PetscCallSAWs(SAWs_Register, ("/PETSc/snes_monitor_saws/rnorm", &snes->norm, 1, SAWs_READ, SAWs_DOUBLE));
    PetscCall(PetscSAWsBlock());
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}
