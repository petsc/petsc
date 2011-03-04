#include "private/fortranimpl.h"
#include "petscis.h"

#if defined(PETSC_HAVE_FORTRAN_CAPS)
#define isview_                ISVIEW
#define isgetindices_          ISGETINDICES
#define isrestoreindices_      ISRESTOREINDICES
#define isdestroy_             ISDESTROY
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
#define isview_                isview
#define isgetindices_          isgetindices
#define isrestoreindices_      isrestoreindices
#define isdestroy_             isdestroy
#endif


EXTERN_C_BEGIN

void PETSC_STDCALL  isdestroy_(IS *is, int *__ierr )
{
  *__ierr = ISDestroy_(*is);
}

void PETSC_STDCALL isview_(IS *is,PetscViewer *vin,PetscErrorCode *ierr)
{
  PetscViewer v;
  PetscPatchDefaultViewers_Fortran(vin,v);
  *ierr = ISView(*is,v);
}

void PETSC_STDCALL isgetindices_(IS *x,PetscInt *fa,size_t *ia,PetscErrorCode *ierr)
{
  const PetscInt *lx;

  *ierr = ISGetIndices(*x,&lx); if (*ierr) return;
  *ia   = PetscIntAddressToFortran(fa,(PetscInt*)lx);
}

void PETSC_STDCALL isrestoreindices_(IS *x,PetscInt *fa,size_t *ia,PetscErrorCode *ierr)
{
  const PetscInt *lx = PetscIntAddressFromFortran(fa,*ia);
  *ierr = ISRestoreIndices(*x,&lx);
}


EXTERN_C_END
