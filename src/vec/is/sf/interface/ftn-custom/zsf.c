#include <petsc/private/f90impl.h>
#include <petsc/private/sfimpl.h>

#if defined(PETSC_HAVE_FORTRAN_CAPS)
#define petscsfview_       PETSCSFVIEW
#define petscsfgetgraph_   PETSCSFGETGRAPH
#define petscsfbcastbegin_ PETSCSFBCASTBEGIN
#define petscsfbcastend_   PETSCSFBCASTEND
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
#define petscsfgetgraph_   petscsfgetgraph
#define petscsfview_       petscsfview
#define petscsfbcastbegin_ petscsfbcastbegin
#define petscsfbcastend_   petscsfbcastend
#endif

PETSC_EXTERN void PETSC_STDCALL petscsfview_(PetscSF *sf, PetscViewer *vin, PetscErrorCode *ierr)
{
  PetscViewer v;

  PetscPatchDefaultViewers_Fortran(vin, v);
  *ierr = PetscSFView(*sf, v);
}

PETSC_EXTERN void PETSC_STDCALL  petscsfgetgraph_(PetscSF *sf,PetscInt *nroots,PetscInt *nleaves, F90Array1d  *ailocal, F90Array1d  *airemote, int *ierr PETSC_F90_2PTR_PROTO(pilocal) PETSC_F90_2PTR_PROTO(piremote))
{
  const PetscInt    *ilocal;
  const PetscSFNode *iremote;

  *ierr = PetscSFGetGraph(*sf,nroots,nleaves,&ilocal,&iremote);if (*ierr) return;
  *ierr = F90Array1dCreate((void*)ilocal,MPIU_INT,1,*nleaves, ailocal PETSC_F90_2PTR_PARAM(pilocal));
  /* this is iffy, since airemote is actually an array of type(PetscSFNode) not an array of PetscInt; works with gfortran, needs testing with all other compilers */
  *ierr = F90Array1dCreate((void*)iremote,MPIU_INT,1,2*(*nleaves), airemote PETSC_F90_2PTR_PARAM(pilocal));
}

PETSC_EXTERN void PETSC_STDCALL petscsfbcastbegin_(PetscSF *sf, MPI_Fint *unit,F90Array1d *rptr, F90Array1d *lptr, int *ierr PETSC_F90_2PTR_PROTO(rptrd) PETSC_F90_2PTR_PROTO(lptrd))
{
  MPI_Datatype dtype;
  const void   *rootdata;
  void         *leafdata;


  *ierr = PetscMPIFortranDatatypeToC(*unit,&dtype);if (*ierr) return;
  *ierr = F90Array1dAccess(rptr, dtype, (void**) &rootdata PETSC_F90_2PTR_PARAM(rptrd));if (*ierr) return;
  *ierr = F90Array1dAccess(lptr, dtype, (void**) &leafdata PETSC_F90_2PTR_PARAM(lptrd));if (*ierr) return;
  *ierr = PetscSFBcastBegin(*sf, dtype, rootdata, leafdata);
}

PETSC_EXTERN void PETSC_STDCALL petscsfbcastend_(PetscSF *sf, MPI_Fint *unit,F90Array1d *rptr, F90Array1d *lptr, int *ierr PETSC_F90_2PTR_PROTO(rptrd) PETSC_F90_2PTR_PROTO(lptrd))
{
  MPI_Datatype dtype;
  const void   *rootdata;
  void         *leafdata;

  *ierr = PetscMPIFortranDatatypeToC(*unit,&dtype);if (*ierr) return;
  *ierr = F90Array1dAccess(rptr, dtype, (void**) &rootdata PETSC_F90_2PTR_PARAM(rptrd));if (*ierr) return;
  *ierr = F90Array1dAccess(lptr, dtype, (void**) &leafdata PETSC_F90_2PTR_PARAM(lptrd));if (*ierr) return;
  *ierr = PetscSFBcastEnd(*sf, dtype, rootdata, leafdata);
}
