/*$Id: f90_common.c,v 1.3 2000/09/07 03:14:24 balay Exp balay $*/

#include "petscf90.h"
#if defined (PETSC_HAVE_F90_C)
#include PETSC_HAVE_F90_C

/* Check if PETSC_HAVE_F90_H is also specified */
#if !defined(PETSC_HAVE_F90_H)
#error "Both PETSC_HAVE_F90_H and PETSC_HAVE_F90_C flags have to be speficied"
#endif

/*-------------------------------------------------------------*/
#undef __FUNC__  
#define __FUNC__ /*<a name=""></a>*/"F90Array1dAccess"
int F90Array1dAccess(F90Array1d *ptr,void **array)
{
  PetscFunctionBegin;
  PetscValidPointer(array);
  PetscValidPointer(ptr);
  *array = ptr->addr;
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ /*<a name=""></a>*/"F90Array1dDestroy"
int F90Array1dDestroy(F90Array1d *ptr)
{
  PetscFunctionBegin;
  PetscValidPointer(ptr);
  ptr->addr = (void *)0;
  PetscFunctionReturn(0);
}
#undef __FUNC__  
#define __FUNC__ /*<a name=""></a>*/"F90Array1dGetNextRecord"
int F90Array1dGetNextRecord(F90Array1d *ptr,void **next)
{
  PetscFunctionBegin;
  PetscValidPointer(ptr);
  *next = (void*)(ptr + 1);
  PetscFunctionReturn(0);
}

/*-------------------------------------------------------------*/

#undef __FUNC__  
#define __FUNC__ /*<a name=""></a>*/"F90Array2dAccess"
int F90Array2dAccess(F90Array2d *ptr,void **array)
{
  PetscFunctionBegin;
  PetscValidPointer(array);
  PetscValidPointer(ptr);
  *array = ptr->addr;
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ /*<a name=""></a>*/"F90Array2dDestroy"
int F90Array2dDestroy(F90Array2d *ptr)
{
  PetscFunctionBegin;
  PetscValidPointer(ptr);
  ptr->addr = (void *)0;
  PetscFunctionReturn(0);
}
#undef __FUNC__  
#define __FUNC__ /*<a name=""></a>*/"F90Array1dGetNextRecord"
int F90Array2dGetNextRecord(F90Array2d *ptr,void **next)
{
  PetscFunctionBegin;
  PetscValidPointer(ptr);
  *next = (void*)(ptr + 1);
  PetscFunctionReturn(0);
}
/*-------------------------------------------------------------*/

#else

/*
     Dummy function so that compilers won't complain about 
  empty files.
*/
int F90_Dummy(int dummy)
{
  return 0;
}

#endif
