#include <petsc/private/f90impl.h>
#include <petscsf.h>
#include <petscsection.h>

#if defined(PETSC_HAVE_FORTRAN_CAPS)
#define petscsfdistributesectionf90_          PETSCSFDISTRIBUTESECTIONF90
#define petscsfcreatesectionsff90_            PETSCSFCREATESECTIONSFF90
#define petscsfcreateremoteoffsetsf90_        PETSCSFCREATEREMOTEOFFSETSF90
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
#define petscsfdistributesectionf90_          petscsfdistributesectionf90
#define petscsfcreatesectionsff90_            petscsfcreatesectionsff90
#define petscsfcreateremoteoffsetsf90_        petscsfcreateremoteoffsetsf90
#endif

PETSC_EXTERN void  petscsfdistributesectionf90_(PetscSF *sf,PetscSection *rootSection,F90Array1d *ptr,PetscSection *leafSection, int *__ierr PETSC_F90_2PTR_PROTO(ptrd))
{
  PetscInt *fa;

  *__ierr = F90Array1dAccess(ptr,MPIU_INT,(void**)&fa PETSC_F90_2PTR_PARAM(ptrd));if (*__ierr) return;
  *__ierr = PetscSFDistributeSection(*sf,*rootSection,&fa,*leafSection);
}

PETSC_EXTERN void  petscsfcreatesectionsff90_(PetscSF *pointSF,PetscSection *rootSection,F90Array1d *ptr,PetscSection *leafSection, PetscSF *sf, int *__ierr PETSC_F90_2PTR_PROTO(ptrd))
{
  PetscInt *fa;

  *__ierr = F90Array1dAccess(ptr,MPIU_INT,(void**)&fa PETSC_F90_2PTR_PARAM(ptrd));if (*__ierr) return;
  *__ierr = PetscSFCreateSectionSF(*pointSF,*rootSection,fa,*leafSection,sf);
}

PETSC_EXTERN void  petscsfcreateremoteoffsetsf90_(PetscSF *pointSF,PetscSection *rootSection,PetscSection *leafSection, F90Array1d *ptr, int *__ierr PETSC_F90_2PTR_PROTO(ptrd))
{
  PetscInt *fa;
  PetscInt lpStart,lpEnd;

  *__ierr = PetscSFCreateRemoteOffsets(*pointSF,*rootSection,*leafSection,&fa);
  *__ierr = PetscSectionGetChart(*leafSection, &lpStart, &lpEnd);   if (*__ierr) return;
  *__ierr = F90Array1dCreate((void*)fa,MPIU_INT,1,lpEnd-lpStart,ptr PETSC_F90_2PTR_PARAM(ptrd));
}
