/*$Id: f90_alpha.c,v 1.10 2001/03/23 23:20:56 balay Exp $*/

/*-------------------------------------------------------------*/

#undef __FUNCT__  
#define __FUNCT__ "F90GetID"
int F90GetID(PetscDataType type,int *id)
{
  PetscFunctionBegin;
  if (type == PETSC_INT) {
    *id = F90_INT_ID;
  } else if (type == PETSC_DOUBLE) {
    *id = F90_DOUBLE_ID;
#if defined(PETSC_USE_COMPLEX)
  } else if (type == PETSC_COMPLEX) {
    *id = F90_COMPLEX_ID;
#endif
  } else if (type == PETSC_LONG) {
    *id = F90_LONG_ID;   
  } else if (type == PETSC_CHAR) {
    *id = F90_CHAR_ID;
  } else {
    SETERRQ(PETSC_ERR_ARG_OUTOFRANGE,"Unknown PETSc datatype");
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "F90Array1dCreate"
int F90Array1dCreate(void *array,PetscDataType type,int start,int len,F90Array1d *ptr)
{
  int size,size_int,ierr,id;

  PetscFunctionBegin;
  PetscValidPointer(array,1);
  PetscValidPointer(ptr,5);  
  ierr               = PetscDataTypeGetSize(type,&size);CHKERRQ(ierr);
  ierr               = F90GetID(type,&id);
  ptr->addr          = array;
  ptr->id            = (char)id;
  ptr->a             = A_VAL;
  ptr->b             = B_VAL;
  ptr->sd            = size;
  ptr->ndim          = 1;
  ptr->dim[0].upper  = len+start;
  ptr->dim[0].mult   = size;
  ptr->dim[0].lower  = start;
  ptr->addr_d         =  (void*)((long)array - (ptr->dim[0].lower*ptr->dim[0].mult));

  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "F90Array2dCreate"
int F90Array2dCreate(void *array,PetscDataType type,int start1,int len1,int start2,int len2,F90Array2d *ptr)
{

  int size,size_int,ierr,id;

  PetscFunctionBegin;
  PetscValidPointer(array,1);
  PetscValidPointer(ptr,7);  
  ierr               = PetscDataTypeGetSize(type,&size);CHKERRQ(ierr);
  ierr               = F90GetID(type,&id);
  ptr->addr          = array;
  ptr->id            = (char)id;
  ptr->a             = A_VAL;
  ptr->b             = B_VAL;
  ptr->sd            = size;
  ptr->ndim          = 2;
  ptr->dim[1].upper  = len1+start1;
  ptr->dim[1].mult   = size;
  ptr->dim[1].lower  = start1;
  ptr->dim[0].upper  = len2+start2;
  ptr->dim[0].mult   = len1*size;
  ptr->dim[0].lower  = start2;
  ptr->addr_d        = (void*)((long)array -(ptr->dim[0].lower*ptr->dim[0].mult+ptr->dim[1].lower*ptr->dim[1].mult));
  PetscFunctionReturn(0);
}

/*-------------------------------------------------------------*/

