
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
  ierr               = PetscDataTypeGetSize(type,&size);CHKERRQ(ierr);
  ptr->addr          = (Pointer)array;
  ptr->offset        = -size;
  ptr->dim[0].extent = len;
  ptr->dim[0].mult   = size;
  ptr->dim[0].lower  = start;

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

/*-------------------------------------------------------------*/
