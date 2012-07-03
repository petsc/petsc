
#if !defined(__PETSCF90_H)
#define __PETSCF90_H
#include <petsc-private/fortranimpl.h>

/* PGI compilers pass in f90 pointers as 2 arguments */
#if defined(PETSC_HAVE_F90_2PTR_ARG)
#define PETSC_F90_2PTR_PROTO_NOVAR ,void*
#define PETSC_F90_2PTR_PROTO(ptr) ,void* ptr
#define PETSC_F90_2PTR_PARAM(ptr) , ptr
#else
#define PETSC_F90_2PTR_PROTO_NOVAR
#define PETSC_F90_2PTR_PROTO(ptr)
#define PETSC_F90_2PTR_PARAM(ptr)
#endif

#if defined (PETSC_USING_F90)

#define F90Array1d void
#define F90Array2d void
#define F90Array3d void
#define F90Array4d void

PETSC_EXTERN PetscErrorCode F90Array1dCreate(void*,PetscDataType,PetscInt,PetscInt,F90Array1d* PETSC_F90_2PTR_PROTO_NOVAR);
PETSC_EXTERN PetscErrorCode F90Array1dAccess(F90Array1d*,PetscDataType,void** PETSC_F90_2PTR_PROTO_NOVAR);
PETSC_EXTERN PetscErrorCode F90Array1dDestroy(F90Array1d*,PetscDataType PETSC_F90_2PTR_PROTO_NOVAR);
PETSC_EXTERN PetscErrorCode F90Array1dGetNextRecord(F90Array1d*,void** PETSC_F90_2PTR_PROTO_NOVAR);

PETSC_EXTERN PetscErrorCode F90Array2dCreate(void*,PetscDataType,PetscInt,PetscInt,PetscInt,PetscInt,F90Array2d* PETSC_F90_2PTR_PROTO_NOVAR);
PETSC_EXTERN PetscErrorCode F90Array2dAccess(F90Array2d*,PetscDataType,void** PETSC_F90_2PTR_PROTO_NOVAR);
PETSC_EXTERN PetscErrorCode F90Array2dDestroy(F90Array2d*,PetscDataType PETSC_F90_2PTR_PROTO_NOVAR);
PETSC_EXTERN PetscErrorCode F90Array2dGetNextRecord(F90Array2d*,void** PETSC_F90_2PTR_PROTO_NOVAR);

PETSC_EXTERN PetscErrorCode F90Array3dCreate(void*,PetscDataType,PetscInt,PetscInt,PetscInt,PetscInt,PetscInt,PetscInt,F90Array3d* PETSC_F90_2PTR_PROTO_NOVAR);
PETSC_EXTERN PetscErrorCode F90Array3dAccess(F90Array3d*,PetscDataType,void** PETSC_F90_2PTR_PROTO_NOVAR);
PETSC_EXTERN PetscErrorCode F90Array3dDestroy(F90Array3d*,PetscDataType PETSC_F90_2PTR_PROTO_NOVAR);
PETSC_EXTERN PetscErrorCode F90Array3dGetNextRecord(F90Array3d*,void** PETSC_F90_2PTR_PROTO_NOVAR);

PETSC_EXTERN PetscErrorCode F90Array4dCreate(void*,PetscDataType,PetscInt,PetscInt,PetscInt,PetscInt,PetscInt,PetscInt,PetscInt,PetscInt,F90Array4d* PETSC_F90_2PTR_PROTO_NOVAR);
PETSC_EXTERN PetscErrorCode F90Array4dAccess(F90Array4d*,PetscDataType,void** PETSC_F90_2PTR_PROTO_NOVAR);
PETSC_EXTERN PetscErrorCode F90Array4dDestroy(F90Array4d*,PetscDataType PETSC_F90_2PTR_PROTO_NOVAR);
PETSC_EXTERN PetscErrorCode F90Array4dGetNextRecord(F90Array4d*,void** PETSC_F90_2PTR_PROTO_NOVAR);

/* 
PETSC_EXTERN PetscErrorCode  F90Array1dGetInfo(F90Array1d*,PetscDataType*,int*,int*);
PETSC_EXTERN PetscErrorCode  F90Array2dGetInfo(F90Array2d*,PetscDataType*,int*,int*,int*,int*);
*/

/*
  F90Array1dCreate - Given a C pointer to a one dimensional
  array and its length; this fills in the appropriate Fortran 90
  pointer data structure.

  Input Parameters:
+   array - regular C pointer (address)
.   type  - DataType of the array
.   start - starting index of the array
-   len   - length of array (in items)

  Output Parameters:
.   ptr - Fortran 90 pointer
*/ 

#endif /* PETSC_USING_F90 */
#endif
