/* $Id: petscf90.h,v 1.1 2000/08/23 20:48:19 balay Exp balay $ */

#if !defined(__PETSCF90_H)
#define __PETSCF90_H

#include "petsc.h"

typedef struct _p_F90Array1d *F90Array1d;
typedef struct _p_F90Array2d *F90Array2d;
typedef struct _p_F90Array3d *F90Array3d;
typedef struct _p_F90Array4d *F90Array4d;

#if !defined(PETSC_HAVE_IRIXF90) && !defined(PETSC_HAVE_XLF90) && !defined(PETSC_HAVE_T3EF90) && \
!defined(PETSC_HAVE_SOLARISF90) && !defined(PETSC_HAVE_SOLARISF90_OLD) && !defined(PETSC_HAVE_NAGF90) && \
!defined (PETSC_HAVE_WIN32F90) && !defined (PETSC_HAVE_DECF90) && !defined (PETSC_HAVE_HPUXF90)
#define PETSC_HAVE_NOF90
#endif

EXTERN int F90Array1dCreate(void*,PetscDataType,int,int,F90Array1d);
EXTERN int F90Array1dGetInfo(F90Array1d,PetscDataType*,int*,int*);
EXTERN int F90Array1dAccess(F90Array1d,void**);
EXTERN int F90Array1dDestroy(F90Array1d);
EXTERN int F90Array1dGetNextRecord(F90Array1d,void**);

EXTERN int F90Array2dCreate(void*,PetscDataType,int,int,int,int,F90Array2d);
EXTERN int F90Array2dGetInfo(F90Array2d,PetscDataType*,int*,int*,int*,int*);
EXTERN int F90Array2dAccess(F90Array2d,void**);
EXTERN int F90Array2dDestroy(F90Array2d);
EXTERN int F90Array2dGetNextRecord(F90Array2d,void**);

#endif


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


/*
    F90Array1dAccess - Gets the address for the data 
       stored in a Fortran pointer array.

  Input Parameters:
.   ptr - Fortran 90 pointer

  Output Parameters:
.   array - regular C pointer (address)

*/ 


/*
    F90Array1dDestroy - Deletes a Fortran pointer.

  Input Parameters:
.   ptr - Fortran 90 pointer
*/ 


/*
    F90Array2dAccess - Gets the address for the data 
       stored in a 2d Fortran pointer array.

  Input Parameters:
.   ptr - Fortran 90 pointer

  Output Parameters:
.   array - regular C pointer (address)

*/ 


/*
    PetscF90Destroy2dArrayScalar - Deletes a Fortran pointer.

  Input Parameters:
.   ptr - Fortran 90 pointer
*/ 

