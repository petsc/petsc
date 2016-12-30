
#include <petscis.h>
#include <petsc/private/f90impl.h>

#if defined(PETSC_HAVE_FORTRAN_CAPS)
#define islocaltoglobalmappinggetindicesf90_           ISLOCALTOGLOBALMAPPINGGETINDICESF90
#define islocaltoglobalmappingrestoreindicesf90_       ISLOCALTOGLOBALMAPPINGRESTOREINDICESF90
#define islocaltoglobalmappinggetblockindicesf90_      ISLOCALTOGLOBALMAPPINGGETBLOCKINDICESF90
#define islocaltoglobalmappingrestoreblockindicesf90_  ISLOCALTOGLOBALMAPPINGRESTOREBLOCKINDICESF90
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
#define islocaltoglobalmappinggetindicesf90_           islocaltoglobalmappinggetindicesf90
#define islocaltoglobalmappingrestoreindicesf90        islocaltoglobalmappingrestoreindicesf90
#define islocaltoglobalmappinggetindicesf90_           islocaltoglobalmappinggetindicesf90
#define islocaltoglobalmappingrestorebindicesf90       islocaltoglobalmappingrestoreindicesf90
#define islocaltoglobalmappinggetblockindicesf90_      islocaltoglobalmappinggetblockindicesf90
#define islocaltoglobalmappingrestoreblockindicesf90   islocaltoglobalmappingrestoreblockindicesf90
#endif

PETSC_EXTERN void PETSC_STDCALL islocaltoglobalmappinggetindicesf90_(ISLocalToGlobalMapping *da,F90Array1d *indices,int *ierr PETSC_F90_2PTR_PROTO(ptrd))
{
  const PetscInt *idx;
  PetscInt       n;
  *ierr = ISLocalToGlobalMappingGetIndices(*da,&idx); if (*ierr) return;
  *ierr = ISLocalToGlobalMappingGetSize(*da,&n); if (*ierr) return;
  *ierr = F90Array1dCreate((void*)idx,PETSC_INT,1,n,indices PETSC_F90_2PTR_PARAM(ptrd));
}

PETSC_EXTERN void PETSC_STDCALL islocaltoglobalmappingrestoreindicesf90_(ISLocalToGlobalMapping *da,F90Array1d *ptr,int *ierr PETSC_F90_2PTR_PROTO(ptrd))
{
  const PetscInt *fa;

  *ierr = F90Array1dAccess(ptr,PETSC_INT,(void**)&fa PETSC_F90_2PTR_PARAM(ptrd));if (*ierr) return;
  *ierr = F90Array1dDestroy(ptr,PETSC_INT PETSC_F90_2PTR_PARAM(ptrd));if (*ierr) return;
  *ierr = ISLocalToGlobalMappingRestoreIndices(*da,&fa); if (*ierr) return;
}

PETSC_EXTERN void PETSC_STDCALL islocaltoglobalmappinggetblockindicesf90_(ISLocalToGlobalMapping *da,F90Array1d *indices,int *ierr PETSC_F90_2PTR_PROTO(ptrd))
{
  const PetscInt *idx;
  PetscInt       n;
  *ierr = ISLocalToGlobalMappingGetBlockIndices(*da,&idx); if (*ierr) return;
  *ierr = ISLocalToGlobalMappingGetSize(*da,&n); if (*ierr) return;
  *ierr = F90Array1dCreate((void*)idx,PETSC_INT,1,n,indices PETSC_F90_2PTR_PARAM(ptrd));
}

PETSC_EXTERN void PETSC_STDCALL islocaltoglobalmappingrestoreblockindicesf90_(ISLocalToGlobalMapping *da,F90Array1d *ptr,int *ierr PETSC_F90_2PTR_PROTO(ptrd))
{
  const PetscInt *fa;

  *ierr = F90Array1dAccess(ptr,PETSC_INT,(void**)&fa PETSC_F90_2PTR_PARAM(ptrd));if (*ierr) return;
  *ierr = F90Array1dDestroy(ptr,PETSC_INT PETSC_F90_2PTR_PARAM(ptrd));if (*ierr) return;
  *ierr = ISLocalToGlobalMappingRestoreBlockIndices(*da,&fa); if (*ierr) return;
}



