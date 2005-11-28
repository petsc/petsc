#define PETSC_DLL

/*-------------------------------------------------------------*/

#undef __FUNCT__  
#define __FUNCT__ "F90Array1dCreate"
PetscErrorCode PETSC_DLLEXPORT F90Array1dCreate(void *array,PetscDataType type,PetscInt start,PetscInt len,F90Array1d *ptr)
{
  PetscErrorCode ierr;
  PetscInt size;

  PetscFunctionBegin;
  PetscValidPointer(array,1);
  PetscValidPointer(ptr,5);  
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
PetscErrorCode PETSC_DLLEXPORT F90Array2dCreate(void *array,PetscDataType type,PetscInt start1,PetscInt len1,PetscInt start2,PetscInt len2,F90Array2d *ptr)
{
  PetscErrorCode ierr;
  PetscInt size;

  PetscFunctionBegin;
  PetscValidPointer(array,1);
  PetscValidPointer(ptr,7);
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

#undef __FUNCT__  
#define __FUNCT__ "F90Array3dCreate"
PetscErrorCode PETSC_DLLEXPORT F90Array3dCreate(void *array,PetscDataType type,PetscInt start1,PetscInt len1,PetscInt start2,PetscInt len2,PetscInt start3,PetscInt len3,F90Array3d *ptr)
{
  PetscErrorCode ierr;
  PetscInt size;

  PetscFunctionBegin;
  PetscValidPointer(array,1);
  PetscValidPointer(ptr,9);
  ierr               = PetscDataTypeGetSize(type,&size);CHKERRQ(ierr);
  ptr->addr          = array;
  ptr->sd            = size;
  ptr->ndim          = 3;
  ptr->a             = F90_COOKIE7;
  ptr->b             = F90_COOKIE0;
  ptr->dim[0].extent = len1;
  ptr->dim[0].mult   = size;
  ptr->dim[0].lower  = start1;
  ptr->dim[1].extent = len2;
  ptr->dim[1].mult   = len1*size;
  ptr->dim[1].lower  = start2;
  ptr->dim[2].extent = len3;
  ptr->dim[2].mult   = len2*len1*size;
  ptr->dim[2].lower  = start3;
  ptr->sum_d         = -(ptr->dim[0].lower*ptr->dim[0].mult+
                         ptr->dim[1].lower*ptr->dim[1].mult+
                         ptr->dim[2].lower*ptr->dim[2].mult);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "F90Array4dCreate"
PetscErrorCode PETSC_DLLEXPORT F90Array4dCreate(void *array,PetscDataType type,PetscInt start1,PetscInt len1,PetscInt start2,PetscInt len2,PetscInt start3,PetscInt len3,PetscInt start4,PetscInt len4,F90Array4d *ptr)
{
  PetscErrorCode ierr;
  PetscInt size;

  PetscFunctionBegin;
  PetscValidPointer(array,1);
  PetscValidPointer(ptr,11);
  ierr               = PetscDataTypeGetSize(type,&size);CHKERRQ(ierr);
  ptr->addr          = array;
  ptr->sd            = size;
  ptr->ndim          = 4;
  ptr->a             = F90_COOKIE7;
  ptr->b             = F90_COOKIE0;
  ptr->dim[0].extent = len1;
  ptr->dim[0].mult   = size;
  ptr->dim[0].lower  = start1;
  ptr->dim[1].extent = len2;
  ptr->dim[1].mult   = len1*size;
  ptr->dim[1].lower  = start2;
  ptr->dim[2].extent = len3;
  ptr->dim[2].mult   = len2*len1*size;
  ptr->dim[2].lower  = start3;
  ptr->dim[3].extent = len4;
  ptr->dim[3].mult   = len3*len2*len1*size;
  ptr->dim[3].lower  = start4;
  ptr->sum_d         = -(ptr->dim[0].lower*ptr->dim[0].mult+
                         ptr->dim[1].lower*ptr->dim[1].mult+
                         ptr->dim[2].lower*ptr->dim[2].mult+
                         ptr->dim[3].lower*ptr->dim[3].mult);
  PetscFunctionReturn(0);
}
/*-------------------------------------------------------------*/
