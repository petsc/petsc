#define PETSC_DLL

#include "petscsys.h"

struct _n_PetscViewers {
   MPI_Comm    comm;
   PetscViewer *viewer;
   int         n;
} ;

#undef __FUNCT__  
#define __FUNCT__ "PetscViewersDestroy" 
/*@C
   PetscViewersDestroy - Destroys a set of PetscViewers created with PetscViewersCreate().

   Collective on PetscViewers

   Input Parameters:
.  v - the PetscViewers to be destroyed.

   Level: intermediate

.seealso: PetscViewerSocketOpen(), PetscViewerASCIIOpen(), PetscViewerCreate(), PetscViewerDrawOpen(), PetscViewersCreate()

@*/
PetscErrorCode PETSC_DLLEXPORT PetscViewersDestroy(PetscViewers v)
{
  int         i;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  for (i=0; i<v->n; i++) {
    if (v->viewer[i]) {ierr = PetscViewerDestroy(v->viewer[i]);CHKERRQ(ierr);}
  }
  ierr = PetscFree(v->viewer);CHKERRQ(ierr);
  ierr = PetscFree(v);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PetscViewersCreate" 
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
PetscErrorCode PETSC_DLLEXPORT PetscViewersCreate(MPI_Comm comm,PetscViewers *v)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr         = PetscNew(struct _n_PetscViewers,v);CHKERRQ(ierr);
  (*v)->n      = 64;
  (*v)->comm   = comm;
  ierr = PetscMalloc(64*sizeof(PetscViewer),&(*v)->viewer);CHKERRQ(ierr);
  ierr = PetscMemzero((*v)->viewer,64*sizeof(PetscViewer));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PetscViewersGetViewer" 
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
PetscErrorCode PETSC_DLLEXPORT PetscViewersGetViewer(PetscViewers viewers,PetscInt n,PetscViewer *viewer)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (n < 0) SETERRQ1(PETSC_ERR_ARG_OUTOFRANGE,"Cannot access using a negative index - %d\n",n);
  if (n >= viewers->n) {
    PetscViewer *v;
    int    newn = n + 64; /* add 64 new ones at a time */
     
    ierr = PetscMalloc(newn*sizeof(PetscViewer),&v);CHKERRQ(ierr);
    ierr = PetscMemzero(v,newn*sizeof(PetscViewer));CHKERRQ(ierr);
    ierr = PetscMemcpy(v,viewers->viewer,viewers->n*sizeof(PetscViewer));CHKERRQ(ierr);
    ierr = PetscFree(viewers->viewer);CHKERRQ(ierr);
    viewers->viewer = v;
  }
  if (!viewers->viewer[n]) {
    ierr = PetscViewerCreate(((PetscObject)viewers)->comm,&viewers->viewer[n]);CHKERRQ(ierr);
  }
  *viewer = viewers->viewer[n];
  PetscFunctionReturn(0);
}






