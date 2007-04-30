#include "src/sys/f90/f90impl.h"
#include "zpetsc.h"
#include "petscbag.h"

#if defined(PETSC_HAVE_FORTRAN_CAPS)
#define petscbagcreate_ PETSCBAGCREATE
#define petscbagdestroy_ PETSCBAGDESTROY
#define petscbagview_ PETSCBAGVIEW
#define petscbaggetdata_ PETSCBAGGETDATA
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
#define petscbagcreate_ petscbagcreate
#define petscbagdestroy_ petscbagdestroy
#define petscbagview_ petscbagview
#define petscbaggetdata_ petscbaggetdata
#endif

EXTERN_C_BEGIN


void PETSC_STDCALL petscbagcreate_(MPI_Comm *comm,PetscInt *size,PetscBag *bag,PetscErrorCode *ierr)
{
  *ierr = PetscBagCreate((MPI_Comm)PetscToPointerComm(*comm),*size,bag);
}

void PETSC_STDCALL petscbagdestroy_(PetscBag *bag,PetscErrorCode *ierr)
{
  *ierr = PetscBagDestroy(*bag);
}

void PETSC_STDCALL petscbagview_(PetscBag *bag,PetscViewer *viewer,PetscErrorCode *ierr)
{
  PetscViewer v;
  PetscPatchDefaultViewers_Fortran(viewer,v);
  *ierr = PetscBagView(*bag,v);
}

void PETSC_STDCALL petscbaggetdata_(PetscBag *bag,void **data,PetscErrorCode *ierr)
{
  *ierr = PetscBagGetData(*bag,data);
}

EXTERN_C_END
