#include <petscsys.h>
#include <petscbt.h>
#include <petscviewer.h>

/*@C
  PetscBTView - View the contents of a `PetscBT` (bit array) on a `PetscViewer`, one line per bit

  Collective on `viewer`; No Fortran Support

  Input Parameters:
+ m      - the number of bits in the array to print
. bt     - the `PetscBT`
- viewer - the `PetscViewer` to print to, or `NULL` to use `PETSC_VIEWER_STDOUT_SELF`

  Level: developer

.seealso: `PetscBT`, `PetscBTCreate()`, `PetscBTLookup()`, `PetscViewer`
@*/
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
