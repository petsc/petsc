/* $Id: zf90.h,v 1.16 2000/07/21 01:12:21 balay Exp $ */

#if !defined(__PETSCF90_H)
#define __PETSCF90_H

#include "petsc.h"

typedef struct _p_F90Array1d *F90Array1d;
typedef struct _p_F90Array2d *F90Array2d;
typedef struct _p_F90Array3d *F90Array3d;
typedef struct _p_F90Array4d *F90Array4d;

EXTERN int F90Array1dCreate(void*,PetsDatatype,int,F90Array1d);
EXTERN int F90Array1dAccess(F90Array1d,void**);
EXTERN int F90Array1dDestroy(F90Array1d);

EXTERN int F90Array2dCreate(void*,PetscDatatype,int,int,F90Array2d);
EXTERN int F90Array2dAccess(F90Array2d,void**);
EXTERN int F90Array2dDestroy(F90Array2d);

#endif
#endif
