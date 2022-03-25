
/*
     Provides utility routines for manipulating any type of PETSc object.
*/
#include <petscsys.h>  /*I   "petscsys.h"    I*/

const char *const PetscDataTypes[] = {"UNKNOWN",
                                      "DOUBLE","COMPLEX","LONG","SHORT","FLOAT",
                                      "CHAR","LOGICAL","ENUM","BOOL","LONGDOUBLE",
                                      "OBJECT","FUNCTION","STRING","FP16","STRUCT",
                                      "INT","INT64",
                                      "PetscDataType","PETSC_",NULL};

/*@C
     PetscDataTypeToMPIDataType - Converts the PETSc name of a datatype to its MPI name.

   Not collective

    Input Parameter:
.     ptype - the PETSc datatype name (for example PETSC_DOUBLE)

    Output Parameter:
.     mtype - the MPI datatype (for example MPI_DOUBLE, ...)

    Level: advanced

.seealso: PetscDataType, PetscMPIDataTypeToPetscDataType()
@*/
PetscErrorCode  PetscDataTypeToMPIDataType(PetscDataType ptype,MPI_Datatype *mtype)
{
  PetscFunctionBegin;
  if (ptype == PETSC_INT)              *mtype = MPIU_INT;
  else if (ptype == PETSC_DOUBLE)      *mtype = MPI_DOUBLE;
#if defined(PETSC_HAVE_COMPLEX)
#if defined(PETSC_USE_REAL_SINGLE)
  else if (ptype == PETSC_COMPLEX)     *mtype = MPI_C_COMPLEX;
#elif defined(PETSC_USE_REAL___FLOAT128)
  else if (ptype == PETSC_COMPLEX)     *mtype = MPIU___COMPLEX128;
#else
  else if (ptype == PETSC_COMPLEX)     *mtype = MPI_C_DOUBLE_COMPLEX;
#endif
#endif
  else if (ptype == PETSC_LONG)        *mtype = MPI_LONG;
  else if (ptype == PETSC_SHORT)       *mtype = MPI_SHORT;
  else if (ptype == PETSC_ENUM)        *mtype = MPI_INT;
  else if (ptype == PETSC_BOOL)        *mtype = MPI_INT;
  else if (ptype == PETSC_INT64)       *mtype = MPIU_INT64;
  else if (ptype == PETSC_FLOAT)       *mtype = MPI_FLOAT;
  else if (ptype == PETSC_CHAR)        *mtype = MPI_CHAR;
  else if (ptype == PETSC_BIT_LOGICAL) *mtype = MPI_BYTE;
#if defined(PETSC_USE_REAL___FLOAT128)
  else if (ptype == PETSC___FLOAT128)  *mtype = MPIU___FLOAT128;
#elif defined(PETSC_USE_REAL___FP16)
  else if (ptype == PETSC___FP16)      *mtype = MPIU___FP16;
#endif
  else SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Unknown PETSc datatype");
  PetscFunctionReturn(0);
}

/*@C
     PetscMPIDataTypeToPetscDataType Finds the PETSc name of a datatype from its MPI name

   Not collective

    Input Parameter:
.     mtype - the MPI datatype (for example MPI_DOUBLE, ...)

    Output Parameter:
.     ptype - the PETSc datatype name (for example PETSC_DOUBLE)

    Level: advanced

.seealso: PetscDataType, PetscMPIDataTypeToPetscDataType()
@*/
PetscErrorCode  PetscMPIDataTypeToPetscDataType(MPI_Datatype mtype,PetscDataType *ptype)
{
  PetscFunctionBegin;
  if (mtype == MPIU_INT)             *ptype = PETSC_INT;
#if defined(PETSC_USE_64BIT_INDICES)
  else if (mtype == MPI_INT)         *ptype = PETSC_ENUM;
#endif
  else if (mtype == MPIU_INT64)      *ptype = PETSC_INT64;
  else if (mtype == MPI_DOUBLE)      *ptype = PETSC_DOUBLE;
#if defined(PETSC_HAVE_COMPLEX)
#if defined(PETSC_USE_REAL_SINGLE)
  else if (mtype == MPI_C_COMPLEX)   *ptype = PETSC_COMPLEX;
#elif defined(PETSC_USE_REAL___FLOAT128)
  else if (mtype == MPIU___COMPLEX128) *ptype = PETSC_COMPLEX;
#else
  else if (mtype == MPI_C_DOUBLE_COMPLEX) *ptype = PETSC_COMPLEX;
#endif
#endif
  else if (mtype == MPI_LONG)        *ptype = PETSC_LONG;
  else if (mtype == MPI_SHORT)       *ptype = PETSC_SHORT;
  else if (mtype == MPI_FLOAT)       *ptype = PETSC_FLOAT;
  else if (mtype == MPI_CHAR)        *ptype = PETSC_CHAR;
#if defined(PETSC_USE_REAL___FLOAT128)
  else if (mtype == MPIU___FLOAT128) *ptype = PETSC___FLOAT128;
#elif defined(PETSC_USE_REAL___FP16)
  else if (mtype == MPIU___FP16) *ptype = PETSC___FP16;
#endif
  else SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"Unhandled MPI datatype");
  PetscFunctionReturn(0);
}

typedef enum {PETSC_INT_SIZE         = sizeof(PetscInt),
              PETSC_DOUBLE_SIZE      = sizeof(double),
#if defined(PETSC_HAVE_COMPLEX)
              PETSC_COMPLEX_SIZE     = sizeof(PetscComplex),
#else
              PETSC_COMPLEX_SIZE     = 2*sizeof(PetscReal),
#endif
              PETSC_LONG_SIZE        = sizeof(long),
              PETSC_SHORT_SIZE       = sizeof(short),
              PETSC_FLOAT_SIZE       = sizeof(float),
              PETSC_CHAR_SIZE        = sizeof(char),
              PETSC_ENUM_SIZE        = sizeof(PetscEnum),
              PETSC_BOOL_SIZE        = sizeof(PetscBool),
              PETSC_INT64_SIZE       = sizeof(PetscInt64),
              PETSC_BIT_LOGICAL_SIZE = sizeof(char)
#if defined(PETSC_USE_REAL___FLOAT128)
              ,PETSC___FLOAT128_SIZE  = sizeof(__float128)
#elif defined(PETSC_USE_REAL___FP16)
              ,PETSC___FP16_SIZE      = sizeof(__fp16)
#endif
             } PetscDataTypeSize;

/*@C
     PetscDataTypeGetSize - Gets the size (in bytes) of a PETSc datatype

   Not collective

    Input Parameter:
.     ptype - the PETSc datatype name (for example PETSC_DOUBLE)

    Output Parameter:
.     size - the size in bytes (for example the size of PETSC_DOUBLE is 8)

    Level: advanced

.seealso: PetscDataType, PetscDataTypeToMPIDataType()
@*/
PetscErrorCode  PetscDataTypeGetSize(PetscDataType ptype,size_t *size)
{
  PetscFunctionBegin;
  if ((int) ptype < 0)                 *size = -(int)ptype;
  else if (ptype == PETSC_INT)         *size = PETSC_INT_SIZE;
  else if (ptype == PETSC_DOUBLE)      *size = PETSC_DOUBLE_SIZE;
  else if (ptype == PETSC_COMPLEX)     *size = PETSC_COMPLEX_SIZE;
  else if (ptype == PETSC_LONG)        *size = PETSC_LONG_SIZE;
  else if (ptype == PETSC_SHORT)       *size = PETSC_SHORT_SIZE;
  else if (ptype == PETSC_FLOAT)       *size = PETSC_FLOAT_SIZE;
  else if (ptype == PETSC_CHAR)        *size = PETSC_CHAR_SIZE;
  else if (ptype == PETSC_ENUM)        *size = PETSC_ENUM_SIZE;
  else if (ptype == PETSC_BOOL)        *size = PETSC_BOOL_SIZE;
  else if (ptype == PETSC_INT64)       *size = PETSC_INT64_SIZE;
  else if (ptype == PETSC_BIT_LOGICAL) *size = PETSC_BIT_LOGICAL_SIZE;
#if defined(PETSC_USE_REAL___FLOAT128)
  else if (ptype == PETSC___FLOAT128)  *size = PETSC___FLOAT128_SIZE;
#elif defined(PETSC_USE_REAL___FP16)
  else if (ptype == PETSC___FP16)      *size = PETSC___FP16_SIZE;
#endif
  else SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Unknown PETSc datatype");
  PetscFunctionReturn(0);
}

/*@C
     PetscDataTypeFromString - Gets the enum value of a PETSc datatype represented as a string

   Not collective

    Input Parameter:
.     name - the PETSc datatype name (for example, DOUBLE or real or Scalar)

    Output Parameters:
+    ptype - the enum value
-    found - the string matches one of the data types

    Level: advanced

.seealso: PetscDataType, PetscDataTypeToMPIDataType(), PetscDataTypeGetSize()
@*/
PetscErrorCode  PetscDataTypeFromString(const char*name, PetscDataType *ptype,PetscBool *found)
{
  PetscFunctionBegin;
  PetscCall(PetscEnumFind(PetscDataTypes,name,(PetscEnum*)ptype,found));
  if (!*found) {
    char formatted[16];

    PetscCall(PetscStrncpy(formatted,name,16));
    PetscCall(PetscStrtolower(formatted));
    PetscCall(PetscStrcmp(formatted,"scalar",found));
    if (*found) {
      *ptype = PETSC_SCALAR;
    } else {
      PetscCall(PetscStrcmp(formatted,"real",found));
      if (*found) {
        *ptype = PETSC_REAL;
      }
    }
  }
  PetscFunctionReturn(0);
}
