#include <petscsys.h>
#include <petscbt.h>
#include <petscviewer.h>

PetscErrorCode PetscBTView(PetscCount m, const PetscBT bt, PetscViewer viewer)
{
  PetscFunctionBegin;
  if (!viewer) PetscCall(PetscViewerASCIIGetStdout(PETSC_COMM_SELF, &viewer));
  PetscCall(PetscViewerASCIIPushSynchronized(viewer));
  for (PetscCount i = 0; i < m; ++i) PetscCall(PetscViewerASCIISynchronizedPrintf(viewer, "%" PetscCount_FMT " %hhu\n", i, PetscBTLookup(bt, i)));
  PetscCall(PetscViewerFlush(viewer));
  PetscCall(PetscViewerASCIIPopSynchronized(viewer));
  PetscFunctionReturn(PETSC_SUCCESS);
}
