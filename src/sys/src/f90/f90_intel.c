/*$Id: f90_hpux.c,v 1.9 2001/03/23 23:20:56 balay Exp $*/

/*-------------------------------------------------------------*/

#undef __FUNCT__  
#define __FUNCT__ "F90Array1dCreate"
int F90Array1dCreate(void *array,PetscDataType type,int start,int len,F90Array1d *ptr)
{
  int size,ierr;

  PetscFunctionBegin;
  PetscValidPointer(array,1);
  PetscValidPointer(ptr,5);  
  ierr               = PetscDataTypeGetSize(type,&size);CHKERRQ(ierr);
  ptr->addr          = array;
  ptr->size          = len;
  ptr->sd            = size;
  ptr->cookie        = size;
  ptr->ndim          = 1;
  ptr->dim[0].lower  = start;
  ptr->dim[0].upper  = len+start;
  ptr->dim[0].mult   = 1;
  ptr->addr_d        = (void*)((long)array -(ptr->dim[0].lower*ptr->dim[0].mult));
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "F90Array2dCreate"
int F90Array2dCreate(void *array,PetscDataType type,int start1,int len1,int start2,int len2,F90Array2d *ptr)
{
  int size,ierr;

  PetscFunctionBegin;
  PetscValidPointer(array,1);
  PetscValidPointer(ptr,7);
  ierr               = PetscDataTypeGetSize(type,&size);CHKERRQ(ierr);
  ptr->addr          = array;
  ptr->size          = len1*len2;
  ptr->sd            = size;
  ptr->cookie        = size;
  ptr->ndim          = 2;
  ptr->dim[0].lower  = start1;
  ptr->dim[0].upper  = len1+start1;
  ptr->dim[0].mult   = 1;
  ptr->dim[1].lower  = start2;
  ptr->dim[1].upper  = len2+start2;
  ptr->dim[1].mult   = len1;
  ptr->addr_d        = (void*)((long)array -(ptr->dim[0].lower*ptr->dim[0].mult+ptr->dim[1].lower*ptr->dim[1].mult));
  PetscFunctionReturn(0);
}
/*-------------------------------------------------------------*/
