#define PETSC_DLL
/*
     Provides utility routines for manipulating any type of PETSc object.
*/
#include "petsc.h"  /*I   "petsc.h"    I*/

#undef __FUNCT__  
#define __FUNCT__ "PetscDataTypeToMPIDataType"
/*@
     PetscDataTypeToMPIDataType - Converts the PETSc name of a datatype to its MPI name.

   Not collective

    Input Parameter:
.     ptype - the PETSc datatype name (for example PETSC_DOUBLE)

    Output Parameter:
.     mtype - the MPI datatype (for example MPI_DOUBLE, ...)

    Level: advanced
   
.seealso: PetscDataType, PetscDataTypeGetName()
@*/
PetscErrorCode PETSC_DLLEXPORT PetscDataTypeToMPIDataType(PetscDataType ptype,MPI_Datatype* mtype)
{
  PetscFunctionBegin;
  if (ptype == PETSC_INT) {
    *mtype = MPIU_INT;
  } else if (ptype == PETSC_DOUBLE) {
    *mtype = MPI_DOUBLE;
#if defined(PETSC_USE_COMPLEX)
  } else if (ptype == PETSC_COMPLEX) {
    *mtype = MPIU_COMPLEX;
#endif
  } else if (ptype == PETSC_LONG) {
    *mtype = MPI_LONG;
  } else if (ptype == PETSC_SHORT) {
    *mtype = MPI_SHORT;
  } else if (ptype == PETSC_FLOAT) {
    *mtype = MPI_FLOAT;
  } else if (ptype == PETSC_CHAR) {
    *mtype = MPI_CHAR;
  } else if (ptype == PETSC_LOGICAL) {
    *mtype = MPI_BYTE;
  } else {
    SETERRQ(PETSC_ERR_ARG_OUTOFRANGE,"Unknown PETSc datatype");
  }
  PetscFunctionReturn(0);
}

typedef enum {PETSC_INT_SIZE = sizeof(PetscInt),PETSC_DOUBLE_SIZE = sizeof(double),
              PETSC_COMPLEX_SIZE = sizeof(PetscScalar),PETSC_LONG_SIZE=sizeof(long),
              PETSC_SHORT_SIZE = sizeof(short),PETSC_FLOAT_SIZE = sizeof(float),
              PETSC_CHAR_SIZE = sizeof(char),PETSC_LOGICAL_SIZE = 1} PetscDataTypeSize;
#if defined(PETSC_USE_COMPLEX)
#define PETSC_SCALAR_SIZE PETSC_COMPLEX_SIZE
#else
#define PETSC_SCALAR_SIZE PETSC_DOUBLE_SIZE
#endif
#if defined(PETSC_USE_SINGLE)
#define PETSC_REAL_SIZE PETSC_FLOAT_SIZE
#else
#define PETSC_REAL_SIZE PETSC_DOUBLE_SIZE
#endif
#define PETSC_FORTRANADDR_SIZE PETSC_LONG_SIZE


#undef __FUNCT__  
#define __FUNCT__ "PetscDataTypeGetSize"
/*@
     PetscDataTypeGetSize - Gets the size (in bytes) of a PETSc datatype

   Not collective

    Input Parameter:
.     ptype - the PETSc datatype name (for example PETSC_DOUBLE)

    Output Parameter:
.     size - the size in bytes (for example the size of PETSC_DOUBLE is 8)

    Level: advanced
   
.seealso: PetscDataType, PetscDataTypeGetName(), PetscDataTypeToMPIDataType()
@*/
PetscErrorCode PETSC_DLLEXPORT PetscDataTypeGetSize(PetscDataType ptype,PetscInt *size)
{
  PetscFunctionBegin;
  if (ptype == PETSC_INT) {
    *size = PETSC_INT_SIZE;
  } else if (ptype == PETSC_DOUBLE) {
    *size = PETSC_DOUBLE_SIZE;
#if defined(PETSC_USE_COMPLEX)
  } else if (ptype == PETSC_COMPLEX) {
    *size = PETSC_COMPLEX_SIZE;
#endif
  } else if (ptype == PETSC_LONG) {
    *size = PETSC_LONG_SIZE;
  } else if (ptype == PETSC_SHORT) {
    *size = PETSC_SHORT_SIZE;
  } else if (ptype == PETSC_FLOAT) {
    *size = PETSC_FLOAT_SIZE;
  } else if (ptype == PETSC_CHAR) {
    *size = PETSC_CHAR_SIZE;
  } else if (ptype == PETSC_LOGICAL) {
    *size = PETSC_LOGICAL_SIZE;
  } else {
    SETERRQ(PETSC_ERR_ARG_OUTOFRANGE,"Unknown PETSc datatype");
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PetscDataTypeGetName"
/*@C
     PetscDataTypeGetName - Gets the string representation of a PETSc datatype

   Not collective

    Input Parameter:
.     ptype - the PETSc datatype name (for example PETSC_DOUBLE)

    Output Parameter:
.     name - the name as a string (for example "double")

    Level: advanced
   
.seealso: PetscDataType, PetscDataTypeGetSize(), PetscDataTypeToMPIDataType()
@*/
PetscErrorCode PETSC_DLLEXPORT PetscDataTypeGetName(PetscDataType ptype,const char *name[])
{
  PetscFunctionBegin;
  if (ptype == PETSC_INT) {
    *name = "int";
  } else if (ptype == PETSC_DOUBLE) {
    *name = "double";
#if defined(PETSC_USE_COMPLEX)
  } else if (ptype == PETSC_COMPLEX) {
    *name = "complex";
#endif
  } else if (ptype == PETSC_LONG) {
    *name = "long";
  } else if (ptype == PETSC_SHORT) {
    *name = "short";
  } else if (ptype == PETSC_FLOAT) {
    *name = "float";
  } else if (ptype == PETSC_CHAR) {
    *name = "char";
  } else if (ptype == PETSC_LOGICAL) {
    *name = "logical";
  } else {
    SETERRQ(PETSC_ERR_ARG_OUTOFRANGE,"Unknown PETSc datatype");
  }
  PetscFunctionReturn(0);
}
