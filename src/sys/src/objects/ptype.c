/*$Id: inherit.c,v 1.62 2000/08/17 04:50:45 bsmith Exp balay $*/
/*
     Provides utility routines for manipulating any type of PETSc object.
*/
#include "petsc.h"  /*I   "petsc.h"    I*/

#undef __FUNC__  
#define __FUNC__ /*<a name=""></a>*/"PetscDataTypeToMPIDataType"
int PetscDataTypeToMPIDataType(PetscDataType ptype,MPI_Datatype* mtype)
{
  PetscFunctionBegin;
  if (ptype == PETSC_INT) {
    *mtype = MPI_INT;
  } else if (ptype == PETSC_DOUBLE) {
    *mtype = MPI_DOUBLE;
  } else if (ptype == PETSC_SCALAR) {
    *mtype = MPIU_SCALAR;
#if defined(PETSC_USE_COMPLEX)
  } else if (ptype == PETSC_COMPLEX) {
    *mtype = MPIU_COMPLEX;
#endif
  } else if (ptype == PETSC_CHAR) {
    *mtype = MPI_CHAR;
  } else if (ptype == PETSC_LOGICAL) {
    *mtype = MPI_BYTE;
  } else {
    SETERRQ(PETSC_ERR_ARG_OUTOFRANGE,1,"Unknown PETSc datatype");
  }
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ /*<a name=""></a>*/"PetscDataTypeGetSize"
int PetscDataTypeGetSize(PetscDataType ptype,int *size)
{
  PetscFunctionBegin;
  if (ptype == PETSC_INT) {
    *size = PETSC_INT_SIZE;
  } else if (ptype == PETSC_DOUBLE) {
    *size = PETSC_DOUBLE_SIZE;
  } else if (ptype == PETSC_SCALAR) {
    *size = PETSC_SCALAR_SIZE;
#if defined(PETSC_USE_COMPLEX)
  } else if (ptype == PETSC_COMPLEX) {
    *size = PETSC_COMPLEX_SIZE;
#endif
  } else if (ptype == PETSC_CHAR) {
    *size = PETSC_CHAR_SIZE;
  } else if (ptype == PETSC_LOGICAL) {
    *size = PETSC_LOGICAL_SIZE;
  } else {
    SETERRQ(PETSC_ERR_ARG_OUTOFRANGE,1,"Unknown PETSc datatype");
  }
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ /*<a name=""></a>*/"PetscDataTypeGetName"
int PetscDataTypeGetName(PetscDataType ptype,char *name[])
{
  PetscFunctionBegin;
  if (ptype == PETSC_INT) {
    *name = "int";
  } else if (ptype == PETSC_DOUBLE) {
    *name = "double";
  } else if (ptype == PETSC_SCALAR) {
    *name = "Scalar";
#if defined(PETSC_USE_COMPLEX)
  } else if (ptype == PETSC_COMPLEX) {
    *name = "complex";
#endif
  } else if (ptype == PETSC_CHAR) {
    *name = "char";
  } else if (ptype == PETSC_LOGICAL) {
    *name = "logical";
  } else {
    SETERRQ(PETSC_ERR_ARG_OUTOFRANGE,1,"Unknown PETSc datatype");
  }
  PetscFunctionReturn(0);
}
