/*$Id: f90_hpux.c,v 1.4 2000/01/11 21:03:54 bsmith Exp balay $*/

#include "petscf90.h"
#include "src/sys/src/f90/f90_hpux.h"

#if defined(PETSC_HAVE_HPUXF90)


#undef __FUNC__  
#define __FUNC__ /*<a name=""></a>*/"F90Array1dCreate"
int F90Array1dCreate(void *array,PetscDataType type,int start,int len,F90Array1d ptr)
{
  int size,ierr;

  PetscFunctionBegin;
  PetscValidPointer(array);
  PetscValidPointer(ptr);  
  ierr               = PetscDataTypeGetSize(type,&size);CHKERRQ(ierr);
  ptr->addr          = array;
  ptr->cookie        = F90_COOKIE;
  ptr->sd            = size;
  ptr->ndim          = F90_1D_ID;
  ptr->dim[0].extent = len;
  ptr->dim[0].mult   = size;
  ptr->dim[0].lower  = start;
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ /*<a name=""></a>*/"F90Array2dCreate"
int F90Array2dCreate(void *array,PetscDataType type,int start1,int len1,int start2,int len2,F90Array2d ptr)
{
  int size,ierr;

  PetscFunctionBegin;
  PetscValidPointer(array);
  PetscValidPointer(ptr);
  ierr               = PetscDataTypeGetSize(type,&size);CHKERRQ(ierr);
  ptr->addr          = array;
  ptr->cookie        = F90_COOKIE;
  ptr->sd            = size;
  ptr->ndim          = F90_2D_ID;
  ptr->dim[0].extent = start1;
  ptr->dim[0].mult   = size;
  ptr->dim[0].lower  = start;
  ptr->dim[1].extent = len2;
  ptr->dim[1].mult   = len1*size;
  ptr->dim[1].lower  = start2;
  PetscFunctionReturn(0);
}

#include "src/sys/src/f90/f90_common.c"

#else
/*
     Dummy function so that compilers won't complain about 
  empty files.
*/
int F90_hpux_Dummy(int dummy)
{
  return 0;
}

#endif
