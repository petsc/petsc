
/*-------------------------------------------------------------*/

#undef __FUNCT__  
#define __FUNCT__ "F90Array1dCreate"
PetscErrorCode F90Array1dCreate(void *array,PetscDataType type,PetscInt start,PetscInt len,F90Array1d *ptr)
{
  PetscErrorCode ierr;
  PetscInt size;

  PetscFunctionBegin;
  PetscValidPointer(array,1);
  PetscValidPointer(ptr,5);  
  ierr           = PetscDataTypeGetSize(type,&size);CHKERRQ(ierr);
  ptr->addr      = array;
  ptr->extent[0] = len;
  ptr->mult[0]   = size;
  ptr->lower[0]  = start;
  ptr->addr_d    = (void*)((long)array - (ptr->lower[0]*ptr->mult[0]));
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "F90Array2dCreate"
PetscErrorCode F90Array2dCreate(void *array,PetscDataType type,PetscInt start1,PetscInt len1,PetscInt start2,PetscInt len2,F90Array2d *ptr)
{
  PetscErrorCode ierr;
  PetscInt size;

  PetscFunctionBegin;
  PetscValidPointer(array,1);
  PetscValidPointer(ptr,7);
  ierr           = PetscDataTypeGetSize(type,&size);CHKERRQ(ierr);
  ptr->addr      = array;
  ptr->extent[0] = len1;
  ptr->mult[0]   = size;
  ptr->lower[0]  = start1;
  ptr->extent[1] = len2;
  ptr->mult[1]   = len1*size;
  ptr->lower[1]  = start2;
  ptr->addr_d    = (void*)((long)array -(ptr->lower[0]*ptr->mult[0]+ptr->lower[1]*ptr->mult[1]));
  PetscFunctionReturn(0);
}
/*-------------------------------------------------------------*/
