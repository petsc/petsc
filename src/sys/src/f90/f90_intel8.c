/*$Id: f90_hpux.c,v 1.9 2001/03/23 23:20:56 balay Exp $*/

/*-------------------------------------------------------------*/

#undef __FUNCT__  
#define __FUNCT__ "F90Array1dCreate"
int F90Array1dCreate(void *array,PetscDataType type,int start,int len,F90Array1d *ptr)
{
  int size,ierr;

  PetscFunctionBegin;
  PetscValidPointer(array);
  PetscValidPointer(ptr);  
  ierr               = PetscDataTypeGetSize(type,&size);CHKERRQ(ierr);
  ptr->addr          = array;
  ptr->sd            = size;
  ptr->ndim          = 1;
  ptr->a             = F90_COOKIE7;
  ptr->b             = F90_COOKIE0;
  ptr->dim[0].extent = len;
  ptr->dim[0].mult   = size;
  ptr->dim[0].lower  = start;
  ptr->sum_d         = -(ptr->dim[0].lower*ptr->dim[0].mult);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "F90Array2dCreate"
int F90Array2dCreate(void *array,PetscDataType type,int start1,int len1,int start2,int len2,F90Array2d *ptr)
{
  int size,ierr;

  PetscFunctionBegin;
  PetscValidPointer(array);
  PetscValidPointer(ptr);
  ierr               = PetscDataTypeGetSize(type,&size);CHKERRQ(ierr);
  ptr->addr          = array;
  ptr->sd            = size;
  ptr->ndim          = 2;
  ptr->a             = F90_COOKIE7;
  ptr->b             = F90_COOKIE0;
  ptr->dim[0].extent = len1;
  ptr->dim[0].mult   = size;
  ptr->dim[0].lower  = start1;
  ptr->dim[1].extent = len2;
  ptr->dim[1].mult   = len1*size;
  ptr->dim[1].lower  = start2;
  ptr->sum_d         = -(ptr->dim[0].lower*ptr->dim[0].mult+ptr->dim[1].lower*ptr->dim[1].mult);
  PetscFunctionReturn(0);
}
/*-------------------------------------------------------------*/
