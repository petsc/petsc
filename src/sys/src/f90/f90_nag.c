/*$Id: f90_nag.c,v 1.12 2000/01/11 21:03:54 bsmith Exp balay $*/

#include "petscf90.h"

#if defined(PETSC_HAVE_NAGF90)

#include "/usr/local/lib/f90/f90.h"
#define _p_F90Array1d Dope1
#define _p_F90Array2d Dope2
#define _p_F90Array3d Dope3


#undef __FUNC__  
#define __FUNC__ /*<a name=""></a>*/"F90Array1dCreate"
int F90Array1dCreate(void *array,PetscDataType type,int start,int len,F90Array1d ptr)
{
  int size,ierr;

  PetscFunctionBegin;
  PetscValidPointer(array);
  PetscValidPointer(ptr);  
  ierr               = PetscDataTypeGetSize(type,&size);CHKERRQ(ierr);
  ptr->addr          = (Pointer)array;
  ptr->offset        = -size;
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
  ptr->addr          = (Pointer)array;
  ptr->offset        = -(1+len1)*size;
  ptr->dim[0].extent = len1;
  ptr->dim[0].mult   = size;
  ptr->dim[0].lower  = start1;
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
int F90_Nag_Dummy(int dummy)
{
  return 0;
}

#endif



