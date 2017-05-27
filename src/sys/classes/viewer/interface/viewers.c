
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

   Collective on MPI_Comm

   Input Parameter:
.   comm - the MPI communicator

   Output Parameter:
.  v - the collection of PetscViewers

   Level: intermediate

   Concepts: PetscViewer^array of

.seealso: PetscViewerCreate(), PetscViewersDestroy()

@*/
PetscErrorCode  PetscViewersCreate(MPI_Comm comm,PetscViewers *v)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr       = PetscNew(v);CHKERRQ(ierr);
  (*v)->n    = 64;
  (*v)->comm = comm;

  ierr = PetscCalloc1(64,&(*v)->viewer);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
   PetscViewersGetViewer - Gets a PetscViewer from a PetscViewer collection

   Not Collective, but PetscViewer will be collective object on PetscViewers

   Input Parameter:
+   viewers - object created with PetscViewersCreate()
-   n - number of PetscViewer you want

   Output Parameter:
.  viewer - the PetscViewer

   Level: intermediate

   Concepts: PetscViewer^array of

.seealso: PetscViewersCreate(), PetscViewersDestroy()

@*/
PetscErrorCode  PetscViewersGetViewer(PetscViewers viewers,PetscInt n,PetscViewer *viewer)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (n < 0) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Cannot access using a negative index - %d\n",n);
  if (n >= viewers->n) {
    PetscViewer *v;
    int         newn = n + 64; /* add 64 new ones at a time */

    ierr = PetscCalloc1(newn,&v);CHKERRQ(ierr);
    ierr = PetscMemcpy(v,viewers->viewer,viewers->n*sizeof(PetscViewer));CHKERRQ(ierr);
    ierr = PetscFree(viewers->viewer);CHKERRQ(ierr);

    viewers->viewer = v;
  }
  if (!viewers->viewer[n]) {
    ierr = PetscViewerCreate(viewers->comm,&viewers->viewer[n]);CHKERRQ(ierr);
  }
  *viewer = viewers->viewer[n];
  PetscFunctionReturn(0);
}






