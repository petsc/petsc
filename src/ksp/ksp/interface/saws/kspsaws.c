#include <petsc/private/kspimpl.h> /*I "petscksp.h" I*/
#include <petscviewersaws.h>

typedef struct {
  PetscViewer viewer;
  PetscInt    neigs;
  PetscReal  *eigi;
  PetscReal  *eigr;
} KSPMonitor_SAWs;

/*@C
   KSPMonitorSAWsCreate - create an SAWs monitor context for `KSP`

   Collective

   Input Parameter:
.  ksp - `KSP` to monitor

   Output Parameter:
.  ctx - context for monitor

   Level: developer

.seealso: [](chapter_ksp), `KSP`, `KSPMonitorSet()`, `KSPMonitorSAWs()`, `KSPMonitorSAWsDestroy()`
@*/
PetscErrorCode KSPMonitorSAWsCreate(KSP ksp, void **ctx)
{
  KSPMonitor_SAWs *mon;

  PetscFunctionBegin;
  PetscCall(PetscNew(&mon));
  mon->viewer = PETSC_VIEWER_SAWS_(PetscObjectComm((PetscObject)ksp));
  PetscCheck(mon->viewer, PetscObjectComm((PetscObject)ksp), PETSC_ERR_PLIB, "Cannot create SAWs default viewer");
  *ctx = (void *)mon;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
   KSPMonitorSAWsDestroy - destroy a monitor context created with `KSPMonitorSAWsCreate()`

   Collective

   Input Parameter:
.  ctx - monitor context

   Level: developer

.seealso: [](chapter_ksp), `KSP`, `KSPMonitorSet()`, `KSPMonitorSAWsCreate()`
@*/
PetscErrorCode KSPMonitorSAWsDestroy(void **ctx)
{
  KSPMonitor_SAWs *mon = (KSPMonitor_SAWs *)*ctx;

  PetscFunctionBegin;
  PetscCall(PetscFree2(mon->eigr, mon->eigi));
  PetscCall(PetscFree(*ctx));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
   KSPMonitorSAWs - monitor `KSP` solution using SAWs

   Logically Collective

   Input Parameters:
+  ksp   - iterative context
.  n     - iteration number
.  rnorm - 2-norm (preconditioned) residual value (may be estimated).
-  ctx -  created with `KSPMonitorSAWsCreate()`

   Level: advanced

   Note:
   Create the ctx with `KSPMonitorSAWsCreate()` then call `KSPMonitorSet()` with the context, this function, and `KSPMonitorSAWsDestroy()`

.seealso: [](chapter_ksp), `KSP`, `KSPMonitorSet()`, `KSPMonitorSAWsCreate()`, `KSPMonitorSAWsDestroy()`, `KSPMonitorSingularValue()`, `KSPComputeExtremeSingularValues()`, `PetscViewerSAWsOpen()`
@*/
PetscErrorCode KSPMonitorSAWs(KSP ksp, PetscInt n, PetscReal rnorm, void *ctx)
{
  KSPMonitor_SAWs *mon = (KSPMonitor_SAWs *)ctx;
  PetscReal        emax, emin;
  PetscMPIInt      rank;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ksp, KSP_CLASSID, 1);
  PetscCall(KSPComputeExtremeSingularValues(ksp, &emax, &emin));

  PetscCall(PetscFree2(mon->eigr, mon->eigi));
  PetscCall(PetscMalloc2(n, &mon->eigr, n, &mon->eigi));
  if (n) {
    PetscCall(KSPComputeEigenvalues(ksp, n, mon->eigr, mon->eigi, &mon->neigs));

    PetscCallMPI(MPI_Comm_rank(PETSC_COMM_WORLD, &rank));
    if (rank == 0) {
      SAWs_Delete("/PETSc/ksp_monitor_saws/eigr");
      SAWs_Delete("/PETSc/ksp_monitor_saws/eigi");

      PetscCallSAWs(SAWs_Register, ("/PETSc/ksp_monitor_saws/rnorm", &ksp->rnorm, 1, SAWs_READ, SAWs_DOUBLE));
      PetscCallSAWs(SAWs_Register, ("/PETSc/ksp_monitor_saws/neigs", &mon->neigs, 1, SAWs_READ, SAWs_INT));
      if (mon->neigs > 0) {
        PetscCallSAWs(SAWs_Register, ("/PETSc/ksp_monitor_saws/eigr", mon->eigr, mon->neigs, SAWs_READ, SAWs_DOUBLE));
        PetscCallSAWs(SAWs_Register, ("/PETSc/ksp_monitor_saws/eigi", mon->eigi, mon->neigs, SAWs_READ, SAWs_DOUBLE));
      }
      PetscCall(PetscInfo(ksp, "KSP extreme singular values min=%g max=%g\n", (double)emin, (double)emax));
      PetscCall(PetscSAWsBlock());
    }
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}
