/*-------------------------------------------------------------*/

#undef __FUNCT__
#define __FUNCT__ "F90Array1dCreate"
PetscErrorCode F90Array1dCreate(void *array,PetscDataType type,PetscInt start,PetscInt len,F90Array1d *ptr)
{
  PetscInt size;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidPointer(array,1);
  PetscValidPointer(ptr,5);
  ierr               = PetscDataTypeGetSize(type,&size);CHKERRQ(ierr);
  ptr->addr          = array;
  ptr->dim[0].lower  = start;
  ptr->dim[0].upper  = start+len-1;
  ptr->dim[0].mult   = size;
  ptr->dim[0].extent = len;
  ptr->dimn          = len;
  ptr->dimb          = 0;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "F90Array2dCreate"
PetscErrorCode F90Array2dCreate(void *array,PetscDataType type,PetscInt start1,PetscInt len1,PetscInt start2,PetscInt len2,F90Array2d *ptr)
{

  PetscInt size;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidPointer(array,1);
  PetscValidPointer(ptr,7);
  ierr               = PetscDataTypeGetSize(type,&size);CHKERRQ(ierr);
  ptr->addr          = array;
  ptr->dim[0].lower  = start1;
  ptr->dim[0].upper  = start1+len1-1;
  ptr->dim[0].mult   = size;
  ptr->dim[0].extent = len1;
  ptr->dim[1].lower  = start2;
  ptr->dim[1].upper  = start2+len2-1;
  ptr->dim[1].mult   = len1*size;
  ptr->dim[1].extent = len2;
  ptr->dimn          = len2*len1;
  ptr->dimb          = ptr->dimn*size;

  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "F90Array3dCreate"
PetscErrorCode F90Array3dCreate(void *array,PetscDataType type,PetscInt start1,PetscInt len1,PetscInt start2,PetscInt len2,PetscInt start3,PetscInt len3,F90Array3d *ptr)
{

  PetscInt size;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidPointer(array,1);
  PetscValidPointer(ptr,9);
  ierr               = PetscDataTypeGetSize(type,&size);CHKERRQ(ierr);
  ptr->addr          = array;
  ptr->dim[0].lower  = start1;
  ptr->dim[0].upper  = start1+len1-1;
  ptr->dim[0].mult   = size;
  ptr->dim[0].extent = len1;
  ptr->dim[1].lower  = start2;
  ptr->dim[1].upper  = start2+len2-1;
  ptr->dim[1].mult   = len1*size;
  ptr->dim[1].extent = len2;
  ptr->dim[2].lower  = start3;
  ptr->dim[2].upper  = start3+len3-1;
  ptr->dim[2].mult   = len2*len1*size;
  ptr->dim[2].extent = len3;
  ptr->dimn          = len3*len2*len1;
  ptr->dimb          = ptr->dimn*size;

  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "F90Array4dCreate"
PetscErrorCode F90Array4dCreate(void *array,PetscDataType type,PetscInt start1,PetscInt len1,PetscInt start2,PetscInt len2,PetscInt start3,PetscInt len3,PetscInt start4,PetscInt len4,F90Array4d *ptr)
{

  PetscInt size;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidPointer(array,1);
  PetscValidPointer(ptr,11);
  ierr               = PetscDataTypeGetSize(type,&size);CHKERRQ(ierr);
  ptr->addr          = array;
  ptr->dim[0].lower  = start1;
  ptr->dim[0].upper  = start1+len1-1;
  ptr->dim[0].mult   = size;
  ptr->dim[0].extent = len1;
  ptr->dim[1].lower  = start2;
  ptr->dim[1].upper  = start2+len2-1;
  ptr->dim[1].mult   = len1*size;
  ptr->dim[1].extent = len2;
  ptr->dim[2].lower  = start3;
  ptr->dim[2].upper  = start3+len3-1;
  ptr->dim[2].mult   = len2*len1*size;
  ptr->dim[2].extent = len3;
  ptr->dim[3].lower  = start4;
  ptr->dim[3].upper  = start4+len4-1;
  ptr->dim[3].mult   = len3*len2*len1*size;
  ptr->dim[3].extent = len4;
  ptr->dimn          = len4*len3*len2*len1;
  ptr->dimb          = ptr->dimn*size;

  PetscFunctionReturn(0);
}

/*-------------------------------------------------------------*/
