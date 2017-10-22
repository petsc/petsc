#include <petsc/private/f90impl.h>
#include <petsc/private/sfimpl.h>

#if defined(PETSC_HAVE_FORTRAN_CAPS)
#define sfview_           PETSCSFVIEW
#define petscsfbcastbegin_ PETSCSFBCASTBEGIN
#define petscsfbcastend_   PETSCSFBCASTEND
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
#define sfview_            petscsfview
#define petscsfbcastbegin_ petscsfbcastbegin
#define petscsfbcastend_   petscsfbcastend
#endif

PETSC_EXTERN void PETSC_STDCALL petscsfview_(PetscSF *sf, PetscViewer *vin, PetscErrorCode *ierr)
{
  PetscViewer v;

  PetscPatchDefaultViewers_Fortran(vin, v);
  *ierr = PetscSFView(*sf, v);
}

PETSC_EXTERN void PETSC_STDCALL petscsfbcastbegin_(PetscSF *sf, MPI_Fint *unit,F90Array1d *rptr, F90Array1d *lptr, int *ierr PETSC_F90_2PTR_PROTO(rptrd) PETSC_F90_2PTR_PROTO(lptrd))
{
  MPI_Datatype dtype;
  const void   *rootdata;
  void         *leafdata;

  dtype = MPI_Type_f2c(*unit);
  *ierr = F90Array1dAccess(rptr, dtype, (void**) &rootdata PETSC_F90_2PTR_PARAM(rptrd));if (*ierr) return;
  *ierr = F90Array1dAccess(lptr, dtype, (void**) &leafdata PETSC_F90_2PTR_PARAM(lptrd));if (*ierr) return;
  *ierr = PetscSFBcastBegin(*sf, dtype, rootdata, leafdata);
}

PETSC_EXTERN void PETSC_STDCALL petscsfbcastend_(PetscSF *sf, MPI_Fint *unit,F90Array1d *rptr, F90Array1d *lptr, int *ierr PETSC_F90_2PTR_PROTO(rptrd) PETSC_F90_2PTR_PROTO(lptrd)) 
{
  MPI_Datatype dtype;
  const void   *rootdata;
  void         *leafdata;

  dtype = MPI_Type_f2c(*unit);
  *ierr = F90Array1dAccess(rptr, dtype, (void**) &rootdata PETSC_F90_2PTR_PARAM(rptrd));if (*ierr) return;
  *ierr = F90Array1dAccess(lptr, dtype, (void**) &leafdata PETSC_F90_2PTR_PARAM(lptrd));if (*ierr) return;
  *ierr = PetscSFBcastEnd(*sf, dtype, rootdata, leafdata);
}
