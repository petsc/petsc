
#include <petscsys.h>
#include <petsc/private/viewerimpl.h>

struct _n_PetscViewers {
  MPI_Comm    comm;
  PetscViewer *viewer;
  int         n;
};

/*@C
   PetscViewersDestroy - Destroys a set of PetscViewers created with PetscViewersCreate().

   Collective on PetscViewers

   Input Parameters:
.  v - the PetscViewers to be destroyed.

   Level: intermediate

.seealso: PetscViewerSocketOpen(), PetscViewerASCIIOpen(), PetscViewerCreate(), PetscViewerDrawOpen(), PetscViewersCreate()

@*/
PetscErrorCode  PetscViewersDestroy(PetscViewers *v)
{
  int            i;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (!*v) PetscFunctionReturn(0);
  for (i=0; i<(*v)->n; i++) {
    ierr = PetscViewerDestroy(&(*v)->viewer[i]);CHKERRQ(ierr);
  }
  ierr = PetscFree((*v)->viewer);CHKERRQ(ierr);
  ierr = PetscFree(*v);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
   PetscViewersCreate - Creates a container to hold a set of PetscViewers.

   Collective

   Input Parameter:
.   comm - the MPI communicator

   Output Parameter:
.  v - the collection of PetscViewers

   Level: intermediate

.seealso: PetscViewerCreate(), PetscViewersDestroy()

@*/
PetscErrorCode  PetscViewersCreate(MPI_Comm comm,PetscViewers *v)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidPointer(v,2);
  ierr       = PetscNew(v);CHKERRQ(ierr);
  (*v)->n    = 64;
  (*v)->comm = comm;

  ierr = PetscCalloc1(64,&(*v)->viewer);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
   PetscViewersGetViewer - Gets a PetscViewer from a PetscViewer collection

   Not Collective, but PetscViewer will be collective object on PetscViewers

   Input Parameters:
+   viewers - object created with PetscViewersCreate()
-   n - number of PetscViewer you want

   Output Parameter:
.  viewer - the PetscViewer

   Level: intermediate

.seealso: PetscViewersCreate(), PetscViewersDestroy()

@*/
PetscErrorCode  PetscViewersGetViewer(PetscViewers viewers,PetscInt n,PetscViewer *viewer)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidPointer(viewers,1);
  PetscValidPointer(viewer,3);
  if (PetscUnlikely(n < 0)) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Cannot access using a negative index - %" PetscInt_FMT "\n",n);
  if (n >= viewers->n) {
    PetscViewer *v;
    int         newn = n + 64; /* add 64 new ones at a time */

    ierr = PetscCalloc1(newn,&v);CHKERRQ(ierr);
    ierr = PetscArraycpy(v,viewers->viewer,viewers->n);CHKERRQ(ierr);
    ierr = PetscFree(viewers->viewer);CHKERRQ(ierr);

    viewers->viewer = v;
  }
  if (!viewers->viewer[n]) {
    ierr = PetscViewerCreate(viewers->comm,&viewers->viewer[n]);CHKERRQ(ierr);
  }
  *viewer = viewers->viewer[n];
  PetscFunctionReturn(0);
}

/*
  PetscMonitorCompare - Checks if two monitors are identical; if they are then it destroys the new one

  Not collective

  Input Parameters:
+ nmon      - The new monitor
. nmctx     - The new monitor context, or NULL
. nmdestroy - The new monitor destroy function, or NULL
. mon       - The old monitor
. mctx      - The old monitor context, or NULL
- mdestroy  - The old monitor destroy function, or NULL

  Output Parameter:
. identical - PETSC_TRUE if the monitors are the same

  Level: developer

.seealsp: DMMonitorSetFromOptions(), KSPMonitorSetFromOptions(), SNESMonitorSetFromOptions()
*/
PetscErrorCode PetscMonitorCompare(PetscErrorCode (*nmon)(void), void *nmctx, PetscErrorCode (*nmdestroy)(void **), PetscErrorCode (*mon)(void), void *mctx, PetscErrorCode (*mdestroy)(void **), PetscBool *identical)
{
  PetscFunctionBegin;
  PetscValidBoolPointer(identical,7);
  *identical = PETSC_FALSE;
  if (nmon == mon && nmdestroy == mdestroy) {
    if (nmctx == mctx) *identical = PETSC_TRUE;
    else if (nmdestroy == (PetscErrorCode (*)(void**)) PetscViewerAndFormatDestroy) {
      PetscViewerAndFormat *old = (PetscViewerAndFormat*)mctx, *newo = (PetscViewerAndFormat*)nmctx;
      if (old->viewer == newo->viewer && old->format == newo->format) *identical = PETSC_TRUE;
    }
    if (*identical) {
      if (mdestroy) {
        PetscErrorCode ierr;
        ierr = (*mdestroy)(&nmctx);CHKERRQ(ierr);
      }
    }
  }
  PetscFunctionReturn(0);
}
