#include <petscsys.h>
#include <petsc/private/viewerimpl.h>

struct _n_PetscViewers {
  MPI_Comm     comm;
  PetscViewer *viewer;
  int          n;
};

/*@C
  PetscViewersDestroy - Destroys a set of `PetscViewer`s created with `PetscViewersCreate()`.

  Collective

  Input Parameter:
. v - the `PetscViewers` to be destroyed.

  Level: intermediate

.seealso: [](sec_viewers), `PetscViewer`, `PetscViewerDestroy()`, `PetscViewers`, `PetscViewerSocketOpen()`, `PetscViewerASCIIOpen()`, `PetscViewerCreate()`, `PetscViewerDrawOpen()`, `PetscViewersCreate()`
@*/
PetscErrorCode PetscViewersDestroy(PetscViewers *v)
{
  int i;

  PetscFunctionBegin;
  if (!*v) PetscFunctionReturn(PETSC_SUCCESS);
  for (i = 0; i < (*v)->n; i++) PetscCall(PetscViewerDestroy(&(*v)->viewer[i]));
  PetscCall(PetscFree((*v)->viewer));
  PetscCall(PetscFree(*v));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  PetscViewersCreate - Creates a container to hold a set of `PetscViewer`'s. The container is essentially a sparse, growable in length array of `PetscViewer`s

  Collective

  Input Parameter:
. comm - the MPI communicator

  Output Parameter:
. v - the collection of `PetscViewers`

  Level: intermediate

.seealso: [](sec_viewers), `PetscViewer`, `PetscViewers`, `PetscViewerCreate()`, `PetscViewersDestroy()`
@*/
PetscErrorCode PetscViewersCreate(MPI_Comm comm, PetscViewers *v)
{
  PetscFunctionBegin;
  PetscAssertPointer(v, 2);
  PetscCall(PetscNew(v));
  (*v)->n    = 64;
  (*v)->comm = comm;

  PetscCall(PetscCalloc1(64, &(*v)->viewer));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  PetscViewersGetViewer - Gets a `PetscViewer` from a `PetscViewers` collection

  Collective if the viewer has not previously be obtained.

  Input Parameters:
+ viewers - object created with `PetscViewersCreate()`
- n       - number of `PetscViewer` you want

  Output Parameter:
. viewer - the `PetscViewer`

  Level: intermediate

.seealso: [](sec_viewers), `PetscViewer`, `PetscViewers`, `PetscViewersCreate()`, `PetscViewersDestroy()`
@*/
PetscErrorCode PetscViewersGetViewer(PetscViewers viewers, PetscInt n, PetscViewer *viewer)
{
  PetscFunctionBegin;
  PetscAssertPointer(viewers, 1);
  PetscAssertPointer(viewer, 3);
  PetscCheck(n >= 0, PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Cannot access using a negative index - %" PetscInt_FMT, n);
  if (n >= viewers->n) {
    PetscViewer *v;

    PetscCall(PetscCalloc1(n + 64, &v));
    PetscCall(PetscArraycpy(v, viewers->viewer, viewers->n));
    PetscCall(PetscFree(viewers->viewer));

    viewers->viewer = v;
  }
  if (!viewers->viewer[n]) PetscCall(PetscViewerCreate(viewers->comm, &viewers->viewer[n]));
  *viewer = viewers->viewer[n];
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  PetscMonitorCompare - Checks if two monitors are identical; if they are then it destroys the new one

  Not Collective

  Input Parameters:
+ nmon      - The new monitor
. nmctx     - The new monitor context, or `NULL`
. nmdestroy - The new monitor context destroy function, or `NULL`, see `PetscCtxDestroyFn` for its calling sequence
. mon       - The old monitor
. mctx      - The old monitor context, or `NULL`
- mdestroy  - The old monitor context destroy function, or `NULL`, see `PetscCtxDestroyFn` for its calling sequence

  Output Parameter:
. identical - `PETSC_TRUE` if the monitors are the same

  Level: developer

.seealso: [](sec_viewers), `DMMonitorSetFromOptions()`, `KSPMonitorSetFromOptions()`, `SNESMonitorSetFromOptions()`, `PetscCtxDestroyFn`
@*/
PetscErrorCode PetscMonitorCompare(PetscErrorCode (*nmon)(void), void *nmctx, PetscCtxDestroyFn *nmdestroy, PetscErrorCode (*mon)(void), void *mctx, PetscCtxDestroyFn *mdestroy, PetscBool *identical)
{
  PetscFunctionBegin;
  PetscAssertPointer(identical, 7);
  *identical = PETSC_FALSE;
  if (nmon == mon && nmdestroy == mdestroy) {
    if (nmctx == mctx) *identical = PETSC_TRUE;
    else if (nmdestroy == (PetscCtxDestroyFn *)PetscViewerAndFormatDestroy) {
      PetscViewerAndFormat *old = (PetscViewerAndFormat *)mctx, *newo = (PetscViewerAndFormat *)nmctx;
      if (old->viewer == newo->viewer && old->format == newo->format) *identical = PETSC_TRUE;
    }
    if (*identical) {
      if (mdestroy) PetscCall((*mdestroy)(&nmctx));
    }
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}
