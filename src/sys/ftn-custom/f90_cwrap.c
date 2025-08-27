#include <petsc/private/ftnimpl.h>

/*@C
   PetscMPIFortranDatatypeToC - Converts a `MPI_Fint` that contains a Fortran `MPI_Datatype` to its C `MPI_Datatype` equivalent

   Not Collective, No Fortran Support

   Input Parameter:
.  unit - The Fortran `MPI_Datatype`

   Output Parameter:
.  dtype - the corresponding C `MPI_Datatype`

   Level: developer

   Developer Note:
   The MPI documentation in multiple places says that one can never us
   Fortran `MPI_Datatype`s in C (or vice-versa) but this is problematic since users could never
   call C routines from Fortran that have `MPI_Datatype` arguments. Jed states that the Fortran
   `MPI_Datatype`s will always be available in C if the MPI was built to support Fortran. This function
   relies on this.

.seealso: `MPI_Fint`, `MPI_Datatype`
@*/
PetscErrorCode PetscMPIFortranDatatypeToC(MPI_Fint unit, MPI_Datatype *dtype)
{
  MPI_Datatype ftype;

  PetscFunctionBegin;
  ftype = MPI_Type_f2c(unit);
  if (ftype == MPI_INTEGER || ftype == MPI_INT) *dtype = MPI_INT;
  else if (ftype == MPI_INTEGER8 || ftype == MPIU_INT64) *dtype = MPIU_INT64;
  else if (ftype == MPI_DOUBLE_PRECISION || ftype == MPI_DOUBLE) *dtype = MPI_DOUBLE;
  else if (ftype == MPI_FLOAT) *dtype = MPI_FLOAT;
#if defined(PETSC_HAVE_COMPLEX)
  else if (ftype == MPI_COMPLEX16 || ftype == MPI_C_DOUBLE_COMPLEX) *dtype = MPI_C_DOUBLE_COMPLEX;
#endif
#if defined(PETSC_HAVE_REAL___FLOAT128)
  else if (ftype == MPIU___FLOAT128) *dtype = MPIU___FLOAT128;
#endif
  else SETERRQ(PETSC_COMM_SELF, PETSC_ERR_SUP, "Unknown Fortran MPI_Datatype");
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*************************************************************************/

#if defined(PETSC_HAVE_FORTRAN_CAPS)
  #define f90array1dcreatescalar_       F90ARRAY1DCREATESCALAR
  #define f90array1daccessscalar_       F90ARRAY1DACCESSSCALAR
  #define f90array1ddestroyscalar_      F90ARRAY1DDESTROYSCALAR
  #define f90array1dcreatereal_         F90ARRAY1DCREATEREAL
  #define f90array1daccessreal_         F90ARRAY1DACCESSREAL
  #define f90array1ddestroyreal_        F90ARRAY1DDESTROYREAL
  #define f90array1dcreateint_          F90ARRAY1DCREATEINT
  #define f90array1daccessint_          F90ARRAY1DACCESSINT
  #define f90array1ddestroyint_         F90ARRAY1DDESTROYINT
  #define f90array1dcreatempiint_       F90ARRAY1DCREATEMPIINT
  #define f90array1daccessmpiint_       F90ARRAY1DACCESSMPIINT
  #define f90array1ddestroympiint_      F90ARRAY1DDESTROYMPIINT
  #define f90array1dcreatefortranaddr_  F90ARRAY1DCREATEFORTRANADDR
  #define f90array1daccessfortranaddr_  F90ARRAY1DACCESSFORTRANADDR
  #define f90array1ddestroyfortranaddr_ F90ARRAY1DDESTROYFORTRANADDR
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
  #define f90array1dcreatescalar_       f90array1dcreatescalar
  #define f90array1daccessscalar_       f90array1daccessscalar
  #define f90array1ddestroyscalar_      f90array1ddestroyscalar
  #define f90array1dcreatereal_         f90array1dcreatereal
  #define f90array1daccessreal_         f90array1daccessreal
  #define f90array1ddestroyreal_        f90array1ddestroyreal
  #define f90array1dcreateint_          f90array1dcreateint
  #define f90array1daccessint_          f90array1daccessint
  #define f90array1ddestroyint_         f90array1ddestroyint
  #define f90array1dcreatempiint_       f90array1dcreatempiint
  #define f90array1daccessmpiint_       f90array1daccessmpiint
  #define f90array1ddestroympiint_      f90array1ddestroympiint
  #define f90array1dcreatefortranaddr_  f90array1dcreatefortranaddr
  #define f90array1daccessfortranaddr_  f90array1daccessfortranaddr
  #define f90array1ddestroyfortranaddr_ f90array1ddestroyfortranaddr
#endif

PETSC_EXTERN void f90array1dcreatescalar_(void *, PetscInt *, PetscInt *, F90Array1d *PETSC_F90_2PTR_PROTO_NOVAR);
PETSC_EXTERN void f90array1daccessscalar_(F90Array1d *, void **PETSC_F90_2PTR_PROTO_NOVAR);
PETSC_EXTERN void f90array1ddestroyscalar_(F90Array1d *ptr PETSC_F90_2PTR_PROTO_NOVAR);
PETSC_EXTERN void f90array1dcreatereal_(void *, PetscInt *, PetscInt *, F90Array1d *PETSC_F90_2PTR_PROTO_NOVAR);
PETSC_EXTERN void f90array1daccessreal_(F90Array1d *, void **PETSC_F90_2PTR_PROTO_NOVAR);
PETSC_EXTERN void f90array1ddestroyreal_(F90Array1d *ptr PETSC_F90_2PTR_PROTO_NOVAR);
PETSC_EXTERN void f90array1dcreateint_(void *, PetscInt *, PetscInt *, F90Array1d *PETSC_F90_2PTR_PROTO_NOVAR);
PETSC_EXTERN void f90array1daccessint_(F90Array1d *, void **PETSC_F90_2PTR_PROTO_NOVAR);
PETSC_EXTERN void f90array1ddestroyint_(F90Array1d *ptr PETSC_F90_2PTR_PROTO_NOVAR);
PETSC_EXTERN void f90array1dcreatempiint_(void *, PetscInt *, PetscInt *, F90Array1d *PETSC_F90_2PTR_PROTO_NOVAR);
PETSC_EXTERN void f90array1daccessmpiint_(F90Array1d *, void **PETSC_F90_2PTR_PROTO_NOVAR);
PETSC_EXTERN void f90array1ddestroympiint_(F90Array1d *ptr PETSC_F90_2PTR_PROTO_NOVAR);
PETSC_EXTERN void f90array1dcreatefortranaddr_(void *, PetscInt *, PetscInt *, F90Array1d *PETSC_F90_2PTR_PROTO_NOVAR);
PETSC_EXTERN void f90array1daccessfortranaddr_(F90Array1d *, void **PETSC_F90_2PTR_PROTO_NOVAR);
PETSC_EXTERN void f90array1ddestroyfortranaddr_(F90Array1d *ptr PETSC_F90_2PTR_PROTO_NOVAR);

/*@C
   F90Array1dCreate - given a `F90Array1d` passed from Fortran associate with it a C array, its starting index and length

   Not Collective, No Fortran Support

   Input Parameters:
+  array - the C address pointer
.  type  - the MPI datatype of the array
.  start - the first index of the array
.  len   - the length of the array
.  ptr   - the `F90Array1d` passed from Fortran
-  ptrd   - an extra pointer passed by some Fortran compilers

   Level: developer

   Developer Notes:
   This is used in PETSc Fortran stubs that are used to pass C arrays to Fortran, for example `VecGetArray()`

   This doesn't actually create the `F90Array1d()`, it just associates a C pointer with it.

   There are equivalent routines for 2, 3, and 4 dimensional Fortran arrays.

.seealso: `F90Array1d`, `F90Array1dAccess()`, `F90Array1dDestroy()`, `F90Array2dCreate()`, `F90Array2dAccess()`, `F90Array2dDestroy()`
@*/
PetscErrorCode F90Array1dCreate(void *array, MPI_Datatype type, PetscInt start, PetscInt len, F90Array1d *ptr PETSC_F90_2PTR_PROTO(ptrd))
{
  PetscFunctionBegin;
  if (type == MPIU_SCALAR) {
    if (!len) array = PETSC_NULL_SCALAR_Fortran;
    f90array1dcreatescalar_(array, &start, &len, ptr PETSC_F90_2PTR_PARAM(ptrd));
  } else if (type == MPIU_REAL) {
    if (!len) array = PETSC_NULL_REAL_Fortran;
    f90array1dcreatereal_(array, &start, &len, ptr PETSC_F90_2PTR_PARAM(ptrd));
  } else if (type == MPIU_INT) {
    if (!len) array = PETSC_NULL_INTEGER_Fortran;
    f90array1dcreateint_(array, &start, &len, ptr PETSC_F90_2PTR_PARAM(ptrd));
  } else if (type == MPI_INT) {
    /* PETSC_NULL_MPIINT_Fortran is not needed since there is no PETSc APIs allowing NULL in place of 'PetscMPIInt *' arguments.
       At this line, we only need to assign 'array' a valid address when len is 0, thus PETSC_NULL_INTEGER_Fortran is enough.
    */
    if (!len) array = PETSC_NULL_INTEGER_Fortran;
    f90array1dcreatempiint_(array, &start, &len, ptr PETSC_F90_2PTR_PARAM(ptrd));
  } else if (type == MPIU_FORTRANADDR) {
    f90array1dcreatefortranaddr_(array, &start, &len, ptr PETSC_F90_2PTR_PARAM(ptrd));
  } else SETERRQ(PETSC_COMM_SELF, PETSC_ERR_SUP, "Unsupported MPI_Datatype");
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
   F90Array1dAccess - given a `F90Array1d` passed from Fortran, accesses from it the associate C array that was provided with `F90Array1dCreate()`

   Not Collective, No Fortran Support

   Input Parameters:
+  ptr   - the `F90Array1d` passed from Fortran
.  type  - the MPI datatype of the array
-  ptrd   - an extra pointer passed by some Fortran compilers

   Output Parameter:
.  array - the C address pointer

   Level: developer

   Developer Note:
   This is used in PETSc Fortran stubs that access C arrays inside Fortran pointer arrays to Fortran. It is usually used in `XXXRestore()`` Fortran stubs.

   There are equivalent routines for 2, 3, and 4 dimensional Fortran arrays.

.seealso: `F90Array1d`, `F90Array1dCreate()`, `F90Array1dDestroy()`, `F90Array2dCreate()`, `F90Array2dAccess()`, `F90Array2dDestroy()`
@*/
PetscErrorCode F90Array1dAccess(F90Array1d *ptr, MPI_Datatype type, void **array PETSC_F90_2PTR_PROTO(ptrd))
{
  PetscFunctionBegin;
  if (type == MPIU_SCALAR) {
    f90array1daccessscalar_(ptr, array PETSC_F90_2PTR_PARAM(ptrd));
    if (*array == PETSC_NULL_SCALAR_Fortran) *array = NULL;
  } else if (type == MPIU_REAL) {
    f90array1daccessreal_(ptr, array PETSC_F90_2PTR_PARAM(ptrd));
    if (*array == PETSC_NULL_REAL_Fortran) *array = NULL;
  } else if (type == MPIU_INT) {
    f90array1daccessint_(ptr, array PETSC_F90_2PTR_PARAM(ptrd));
    if (*array == PETSC_NULL_INTEGER_Fortran) *array = NULL;
  } else if (type == MPI_INT) {
    f90array1daccessmpiint_(ptr, array PETSC_F90_2PTR_PARAM(ptrd));
    if (*array == PETSC_NULL_INTEGER_Fortran) *array = NULL;
  } else if (type == MPIU_FORTRANADDR) {
    f90array1daccessfortranaddr_(ptr, array PETSC_F90_2PTR_PARAM(ptrd));
  } else SETERRQ(PETSC_COMM_SELF, PETSC_ERR_SUP, "Unsupported MPI_Datatype");
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
   F90Array1dDestroy - given a `F90Array1d` passed from Fortran removes the C array associate with it with `F90Array1dCreate()`

   Not Collective, No Fortran Support

   Input Parameters:
+  ptr   - the `F90Array1d` passed from Fortran
.  type  - the MPI datatype of the array
-  ptrd   - an extra pointer passed by some Fortran compilers

   Level: developer

   Developer Notes:
   This is used in PETSc Fortran stubs that are used to end access to C arrays from Fortran, for example `VecRestoreArray()`

   This doesn't actually destroy the `F90Array1d()`, it just removes the associated C pointer from it.

   There are equivalent routines for 2, 3, and 4 dimensional Fortran arrays.

.seealso: `F90Array1d`, `F90Array1dAccess()`, `F90Array1dCreate()`, `F90Array2dCreate()`, `F90Array2dAccess()`, `F90Array2dDestroy()`
@*/
PetscErrorCode F90Array1dDestroy(F90Array1d *ptr, MPI_Datatype type PETSC_F90_2PTR_PROTO(ptrd))
{
  PetscFunctionBegin;
  if (type == MPIU_SCALAR) {
    f90array1ddestroyscalar_(ptr PETSC_F90_2PTR_PARAM(ptrd));
  } else if (type == MPIU_REAL) {
    f90array1ddestroyreal_(ptr PETSC_F90_2PTR_PARAM(ptrd));
  } else if (type == MPIU_INT) {
    f90array1ddestroyint_(ptr PETSC_F90_2PTR_PARAM(ptrd));
  } else if (type == MPI_INT) {
    f90array1ddestroympiint_(ptr PETSC_F90_2PTR_PARAM(ptrd));
  } else if (type == MPIU_FORTRANADDR) {
    f90array1ddestroyfortranaddr_(ptr PETSC_F90_2PTR_PARAM(ptrd));
  } else SETERRQ(PETSC_COMM_SELF, PETSC_ERR_SUP, "Unsupported MPI_Datatype");
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*MC
   F90Array1d - a PETSc C representation of a Fortran `XXX, pointer :: array(:)` object

   Not Collective, No Fortran Support

   Level: developer

   Developer Notes:
   This is used in PETSc Fortran stubs that are used to control access to C arrays from Fortran, for example `VecGetArray()`

   PETSc does not require any information about the format of this object, all operations on the object are performed by calling Fortran routines.

   There are equivalent objects for 2, 3, and 4 dimensional Fortran arrays.

.seealso: `F90Array1dAccess()`, `F90Array1dCreate()`, , `F90Array1dDestroy()`, `F90Array2dCreate()`, `F90Array2dAccess()`, `F90Array2dDestroy()`
M*/

#if defined(PETSC_HAVE_FORTRAN_CAPS)
  #define f90array2dcreatescalar_       F90ARRAY2DCREATESCALAR
  #define f90array2daccessscalar_       F90ARRAY2DACCESSSCALAR
  #define f90array2ddestroyscalar_      F90ARRAY2DDESTROYSCALAR
  #define f90array2dcreatereal_         F90ARRAY2DCREATEREAL
  #define f90array2daccessreal_         F90ARRAY2DACCESSREAL
  #define f90array2ddestroyreal_        F90ARRAY2DDESTROYREAL
  #define f90array2dcreateint_          F90ARRAY2DCREATEINT
  #define f90array2daccessint_          F90ARRAY2DACCESSINT
  #define f90array2ddestroyint_         F90ARRAY2DDESTROYINT
  #define f90array2dcreatefortranaddr_  F90ARRAY2DCREATEFORTRANADDR
  #define f90array2daccessfortranaddr_  F90ARRAY2DACCESSFORTRANADDR
  #define f90array2ddestroyfortranaddr_ F90ARRAY2DDESTROYFORTRANADDR
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
  #define f90array2dcreatescalar_       f90array2dcreatescalar
  #define f90array2daccessscalar_       f90array2daccessscalar
  #define f90array2ddestroyscalar_      f90array2ddestroyscalar
  #define f90array2dcreatereal_         f90array2dcreatereal
  #define f90array2daccessreal_         f90array2daccessreal
  #define f90array2ddestroyreal_        f90array2ddestroyreal
  #define f90array2dcreateint_          f90array2dcreateint
  #define f90array2daccessint_          f90array2daccessint
  #define f90array2ddestroyint_         f90array2ddestroyint
  #define f90array2dcreatefortranaddr_  f90array2dcreatefortranaddr
  #define f90array2daccessfortranaddr_  f90array2daccessfortranaddr
  #define f90array2ddestroyfortranaddr_ f90array2ddestroyfortranaddr
#endif

PETSC_EXTERN void f90array2dcreatescalar_(void *, PetscInt *, PetscInt *, PetscInt *, PetscInt *, F90Array2d *PETSC_F90_2PTR_PROTO_NOVAR);
PETSC_EXTERN void f90array2daccessscalar_(F90Array2d *, void **PETSC_F90_2PTR_PROTO_NOVAR);
PETSC_EXTERN void f90array2ddestroyscalar_(F90Array2d *ptr PETSC_F90_2PTR_PROTO_NOVAR);
PETSC_EXTERN void f90array2dcreatereal_(void *, PetscInt *, PetscInt *, PetscInt *, PetscInt *, F90Array2d *PETSC_F90_2PTR_PROTO_NOVAR);
PETSC_EXTERN void f90array2daccessreal_(F90Array2d *, void **PETSC_F90_2PTR_PROTO_NOVAR);
PETSC_EXTERN void f90array2ddestroyreal_(F90Array2d *ptr PETSC_F90_2PTR_PROTO_NOVAR);
PETSC_EXTERN void f90array2dcreateint_(void *, PetscInt *, PetscInt *, PetscInt *, PetscInt *, F90Array2d *PETSC_F90_2PTR_PROTO_NOVAR);
PETSC_EXTERN void f90array2daccessint_(F90Array2d *, void **PETSC_F90_2PTR_PROTO_NOVAR);
PETSC_EXTERN void f90array2ddestroyint_(F90Array2d *ptr PETSC_F90_2PTR_PROTO_NOVAR);
PETSC_EXTERN void f90array2dcreatefortranaddr_(void *, PetscInt *, PetscInt *, PetscInt *, PetscInt *, F90Array2d *PETSC_F90_2PTR_PROTO_NOVAR);
PETSC_EXTERN void f90array2daccessfortranaddr_(F90Array2d *, void **PETSC_F90_2PTR_PROTO_NOVAR);
PETSC_EXTERN void f90array2ddestroyfortranaddr_(F90Array2d *ptr PETSC_F90_2PTR_PROTO_NOVAR);

PetscErrorCode F90Array2dCreate(void *array, MPI_Datatype type, PetscInt start1, PetscInt len1, PetscInt start2, PetscInt len2, F90Array2d *ptr PETSC_F90_2PTR_PROTO(ptrd))
{
  PetscFunctionBegin;
  if (type == MPIU_SCALAR) {
    f90array2dcreatescalar_(array, &start1, &len1, &start2, &len2, ptr PETSC_F90_2PTR_PARAM(ptrd));
  } else if (type == MPIU_REAL) {
    f90array2dcreatereal_(array, &start1, &len1, &start2, &len2, ptr PETSC_F90_2PTR_PARAM(ptrd));
  } else if (type == MPIU_INT) {
    f90array2dcreateint_(array, &start1, &len1, &start2, &len2, ptr PETSC_F90_2PTR_PARAM(ptrd));
  } else if (type == MPIU_FORTRANADDR) {
    f90array2dcreatefortranaddr_(array, &start1, &len1, &start2, &len2, ptr PETSC_F90_2PTR_PARAM(ptrd));
  } else SETERRQ(PETSC_COMM_SELF, PETSC_ERR_SUP, "Unsupported MPI_Datatype");
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode F90Array2dAccess(F90Array2d *ptr, MPI_Datatype type, void **array PETSC_F90_2PTR_PROTO(ptrd))
{
  PetscFunctionBegin;
  if (type == MPIU_SCALAR) {
    f90array2daccessscalar_(ptr, array PETSC_F90_2PTR_PARAM(ptrd));
  } else if (type == MPIU_REAL) {
    f90array2daccessreal_(ptr, array PETSC_F90_2PTR_PARAM(ptrd));
  } else if (type == MPIU_INT) {
    f90array2daccessint_(ptr, array PETSC_F90_2PTR_PARAM(ptrd));
  } else if (type == MPIU_FORTRANADDR) {
    f90array2daccessfortranaddr_(ptr, array PETSC_F90_2PTR_PARAM(ptrd));
  } else SETERRQ(PETSC_COMM_SELF, PETSC_ERR_SUP, "Unsupported MPI_Datatype");
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode F90Array2dDestroy(F90Array2d *ptr, MPI_Datatype type PETSC_F90_2PTR_PROTO(ptrd))
{
  PetscFunctionBegin;
  if (type == MPIU_SCALAR) {
    f90array2ddestroyscalar_(ptr PETSC_F90_2PTR_PARAM(ptrd));
  } else if (type == MPIU_REAL) {
    f90array2ddestroyreal_(ptr PETSC_F90_2PTR_PARAM(ptrd));
  } else if (type == MPIU_INT) {
    f90array2ddestroyint_(ptr PETSC_F90_2PTR_PARAM(ptrd));
  } else if (type == MPIU_FORTRANADDR) {
    f90array2ddestroyfortranaddr_(ptr PETSC_F90_2PTR_PARAM(ptrd));
  } else SETERRQ(PETSC_COMM_SELF, PETSC_ERR_SUP, "Unsupported MPI_Datatype");
  PetscFunctionReturn(PETSC_SUCCESS);
}

#if defined(PETSC_HAVE_FORTRAN_CAPS)
  #define f90array3dcreatescalar_       F90ARRAY3DCREATESCALAR
  #define f90array3daccessscalar_       F90ARRAY3DACCESSSCALAR
  #define f90array3ddestroyscalar_      F90ARRAY3DDESTROYSCALAR
  #define f90array3dcreatereal_         F90ARRAY3DCREATEREAL
  #define f90array3daccessreal_         F90ARRAY3DACCESSREAL
  #define f90array3ddestroyreal_        F90ARRAY3DDESTROYREAL
  #define f90array3dcreateint_          F90ARRAY3DCREATEINT
  #define f90array3daccessint_          F90ARRAY3DACCESSINT
  #define f90array3ddestroyint_         F90ARRAY3DDESTROYINT
  #define f90array3dcreatefortranaddr_  F90ARRAY3DCREATEFORTRANADDR
  #define f90array3daccessfortranaddr_  F90ARRAY3DACCESSFORTRANADDR
  #define f90array3ddestroyfortranaddr_ F90ARRAY3DDESTROYFORTRANADDR
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
  #define f90array3dcreatescalar_       f90array3dcreatescalar
  #define f90array3daccessscalar_       f90array3daccessscalar
  #define f90array3ddestroyscalar_      f90array3ddestroyscalar
  #define f90array3dcreatereal_         f90array3dcreatereal
  #define f90array3daccessreal_         f90array3daccessreal
  #define f90array3ddestroyreal_        f90array3ddestroyreal
  #define f90array3dcreateint_          f90array3dcreateint
  #define f90array3daccessint_          f90array3daccessint
  #define f90array3ddestroyint_         f90array3ddestroyint
  #define f90array3dcreatefortranaddr_  f90array3dcreatefortranaddr
  #define f90array3daccessfortranaddr_  f90array3daccessfortranaddr
  #define f90array3ddestroyfortranaddr_ f90array3ddestroyfortranaddr
#endif

PETSC_EXTERN void f90array3dcreatescalar_(void *, PetscInt *, PetscInt *, PetscInt *, PetscInt *, PetscInt *, PetscInt *, F90Array3d *PETSC_F90_2PTR_PROTO_NOVAR);
PETSC_EXTERN void f90array3daccessscalar_(F90Array3d *, void **PETSC_F90_2PTR_PROTO_NOVAR);
PETSC_EXTERN void f90array3ddestroyscalar_(F90Array3d *ptr PETSC_F90_2PTR_PROTO_NOVAR);
PETSC_EXTERN void f90array3dcreatereal_(void *, PetscInt *, PetscInt *, PetscInt *, PetscInt *, PetscInt *, PetscInt *, F90Array3d *PETSC_F90_2PTR_PROTO_NOVAR);
PETSC_EXTERN void f90array3daccessreal_(F90Array3d *, void **PETSC_F90_2PTR_PROTO_NOVAR);
PETSC_EXTERN void f90array3ddestroyreal_(F90Array3d *ptr PETSC_F90_2PTR_PROTO_NOVAR);
PETSC_EXTERN void f90array3dcreateint_(void *, PetscInt *, PetscInt *, PetscInt *, PetscInt *, PetscInt *, PetscInt *, F90Array3d *PETSC_F90_2PTR_PROTO_NOVAR);
PETSC_EXTERN void f90array3daccessint_(F90Array3d *, void **PETSC_F90_2PTR_PROTO_NOVAR);
PETSC_EXTERN void f90array3ddestroyint_(F90Array3d *ptr PETSC_F90_2PTR_PROTO_NOVAR);
PETSC_EXTERN void f90array3dcreatefortranaddr_(void *, PetscInt *, PetscInt *, PetscInt *, PetscInt *, PetscInt *, PetscInt *, F90Array3d *PETSC_F90_2PTR_PROTO_NOVAR);
PETSC_EXTERN void f90array3daccessfortranaddr_(F90Array3d *, void **PETSC_F90_2PTR_PROTO_NOVAR);
PETSC_EXTERN void f90array3ddestroyfortranaddr_(F90Array3d *ptr PETSC_F90_2PTR_PROTO_NOVAR);

PetscErrorCode F90Array3dCreate(void *array, MPI_Datatype type, PetscInt start1, PetscInt len1, PetscInt start2, PetscInt len2, PetscInt start3, PetscInt len3, F90Array3d *ptr PETSC_F90_2PTR_PROTO(ptrd))
{
  PetscFunctionBegin;
  if (type == MPIU_SCALAR) {
    f90array3dcreatescalar_(array, &start1, &len1, &start2, &len2, &start3, &len3, ptr PETSC_F90_2PTR_PARAM(ptrd));
  } else if (type == MPIU_REAL) {
    f90array3dcreatereal_(array, &start1, &len1, &start2, &len2, &start3, &len3, ptr PETSC_F90_2PTR_PARAM(ptrd));
  } else if (type == MPIU_INT) {
    f90array3dcreateint_(array, &start1, &len1, &start2, &len2, &start3, &len3, ptr PETSC_F90_2PTR_PARAM(ptrd));
  } else if (type == MPIU_FORTRANADDR) {
    f90array3dcreatefortranaddr_(array, &start1, &len1, &start2, &len2, &start3, &len3, ptr PETSC_F90_2PTR_PARAM(ptrd));
  } else SETERRQ(PETSC_COMM_SELF, PETSC_ERR_SUP, "Unsupported MPI_Datatype");
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode F90Array3dAccess(F90Array3d *ptr, MPI_Datatype type, void **array PETSC_F90_2PTR_PROTO(ptrd))
{
  PetscFunctionBegin;
  if (type == MPIU_SCALAR) {
    f90array3daccessscalar_(ptr, array PETSC_F90_2PTR_PARAM(ptrd));
  } else if (type == MPIU_REAL) {
    f90array3daccessreal_(ptr, array PETSC_F90_2PTR_PARAM(ptrd));
  } else if (type == MPIU_INT) {
    f90array3daccessint_(ptr, array PETSC_F90_2PTR_PARAM(ptrd));
  } else if (type == MPIU_FORTRANADDR) {
    f90array3daccessfortranaddr_(ptr, array PETSC_F90_2PTR_PARAM(ptrd));
  } else SETERRQ(PETSC_COMM_SELF, PETSC_ERR_SUP, "Unsupported MPI_Datatype");
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode F90Array3dDestroy(F90Array3d *ptr, MPI_Datatype type PETSC_F90_2PTR_PROTO(ptrd))
{
  PetscFunctionBegin;
  if (type == MPIU_SCALAR) {
    f90array3ddestroyscalar_(ptr PETSC_F90_2PTR_PARAM(ptrd));
  } else if (type == MPIU_REAL) {
    f90array3ddestroyreal_(ptr PETSC_F90_2PTR_PARAM(ptrd));
  } else if (type == MPIU_INT) {
    f90array3ddestroyint_(ptr PETSC_F90_2PTR_PARAM(ptrd));
  } else if (type == MPIU_FORTRANADDR) {
    f90array3ddestroyfortranaddr_(ptr PETSC_F90_2PTR_PARAM(ptrd));
  } else SETERRQ(PETSC_COMM_SELF, PETSC_ERR_SUP, "Unsupported MPI_Datatype");
  PetscFunctionReturn(PETSC_SUCCESS);
}

#if defined(PETSC_HAVE_FORTRAN_CAPS)
  #define f90array4dcreatescalar_       F90ARRAY4DCREATESCALAR
  #define f90array4daccessscalar_       F90ARRAY4DACCESSSCALAR
  #define f90array4ddestroyscalar_      F90ARRAY4DDESTROYSCALAR
  #define f90array4dcreatereal_         F90ARRAY4DCREATEREAL
  #define f90array4daccessreal_         F90ARRAY4DACCESSREAL
  #define f90array4ddestroyreal_        F90ARRAY4DDESTROYREAL
  #define f90array4dcreateint_          F90ARRAY4DCREATEINT
  #define f90array4daccessint_          F90ARRAY4DACCESSINT
  #define f90array4ddestroyint_         F90ARRAY4DDESTROYINT
  #define f90array4dcreatefortranaddr_  F90ARRAY4DCREATEFORTRANADDR
  #define f90array4daccessfortranaddr_  F90ARRAY4DACCESSFORTRANADDR
  #define f90array4ddestroyfortranaddr_ F90ARRAY4DDESTROYFORTRANADDR
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
  #define f90array4dcreatescalar_       f90array4dcreatescalar
  #define f90array4daccessscalar_       f90array4daccessscalar
  #define f90array4ddestroyscalar_      f90array4ddestroyscalar
  #define f90array4dcreatereal_         f90array4dcreatereal
  #define f90array4daccessreal_         f90array4daccessreal
  #define f90array4ddestroyreal_        f90array4ddestroyreal
  #define f90array4dcreateint_          f90array4dcreateint
  #define f90array4daccessint_          f90array4daccessint
  #define f90array4ddestroyint_         f90array4ddestroyint
  #define f90array4dcreatefortranaddr_  f90array4dcreatefortranaddr
  #define f90array4daccessfortranaddr_  f90array4daccessfortranaddr
  #define f90array4ddestroyfortranaddr_ f90array4ddestroyfortranaddr
#endif

PETSC_EXTERN void f90array4dcreatescalar_(void *, PetscInt *, PetscInt *, PetscInt *, PetscInt *, PetscInt *, PetscInt *, PetscInt *, PetscInt *, F90Array4d *PETSC_F90_2PTR_PROTO_NOVAR);
PETSC_EXTERN void f90array4daccessscalar_(F90Array4d *, void **PETSC_F90_2PTR_PROTO_NOVAR);
PETSC_EXTERN void f90array4ddestroyscalar_(F90Array4d *ptr PETSC_F90_2PTR_PROTO_NOVAR);
PETSC_EXTERN void f90array4dcreatereal_(void *, PetscInt *, PetscInt *, PetscInt *, PetscInt *, PetscInt *, PetscInt *, PetscInt *, PetscInt *, F90Array4d *PETSC_F90_2PTR_PROTO_NOVAR);
PETSC_EXTERN void f90array4daccessreal_(F90Array4d *, void **PETSC_F90_2PTR_PROTO_NOVAR);
PETSC_EXTERN void f90array4ddestroyreal_(F90Array4d *ptr PETSC_F90_2PTR_PROTO_NOVAR);
PETSC_EXTERN void f90array4dcreateint_(void *, PetscInt *, PetscInt *, PetscInt *, PetscInt *, PetscInt *, PetscInt *, PetscInt *, PetscInt *, F90Array4d *PETSC_F90_2PTR_PROTO_NOVAR);
PETSC_EXTERN void f90array4daccessint_(F90Array4d *, void **PETSC_F90_2PTR_PROTO_NOVAR);
PETSC_EXTERN void f90array4ddestroyint_(F90Array4d *ptr PETSC_F90_2PTR_PROTO_NOVAR);
PETSC_EXTERN void f90array4dcreatefortranaddr_(void *, PetscInt *, PetscInt *, PetscInt *, PetscInt *, PetscInt *, PetscInt *, PetscInt *, PetscInt *, F90Array4d *PETSC_F90_2PTR_PROTO_NOVAR);
PETSC_EXTERN void f90array4daccessfortranaddr_(F90Array4d *, void **PETSC_F90_2PTR_PROTO_NOVAR);
PETSC_EXTERN void f90array4ddestroyfortranaddr_(F90Array4d *ptr PETSC_F90_2PTR_PROTO_NOVAR);

PetscErrorCode F90Array4dCreate(void *array, MPI_Datatype type, PetscInt start1, PetscInt len1, PetscInt start2, PetscInt len2, PetscInt start3, PetscInt len3, PetscInt start4, PetscInt len4, F90Array4d *ptr PETSC_F90_2PTR_PROTO(ptrd))
{
  PetscFunctionBegin;
  PetscCheck(type == MPIU_SCALAR, PETSC_COMM_SELF, PETSC_ERR_SUP, "Unsupported MPI_Datatype");
  f90array4dcreatescalar_(array, &start1, &len1, &start2, &len2, &start3, &len3, &start4, &len4, ptr PETSC_F90_2PTR_PARAM(ptrd));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode F90Array4dAccess(F90Array4d *ptr, MPI_Datatype type, void **array PETSC_F90_2PTR_PROTO(ptrd))
{
  PetscFunctionBegin;
  if (type == MPIU_SCALAR) {
    f90array4daccessscalar_(ptr, array PETSC_F90_2PTR_PARAM(ptrd));
  } else if (type == MPIU_REAL) {
    f90array4daccessreal_(ptr, array PETSC_F90_2PTR_PARAM(ptrd));
  } else if (type == MPIU_INT) {
    f90array4daccessint_(ptr, array PETSC_F90_2PTR_PARAM(ptrd));
  } else if (type == MPIU_FORTRANADDR) {
    f90array4daccessfortranaddr_(ptr, array PETSC_F90_2PTR_PARAM(ptrd));
  } else SETERRQ(PETSC_COMM_SELF, PETSC_ERR_SUP, "Unsupported MPI_Datatype");
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode F90Array4dDestroy(F90Array4d *ptr, MPI_Datatype type PETSC_F90_2PTR_PROTO(ptrd))
{
  PetscFunctionBegin;
  PetscCheck(type == MPIU_SCALAR, PETSC_COMM_SELF, PETSC_ERR_SUP, "Unsupported MPI_Datatype");
  f90array4ddestroyscalar_(ptr PETSC_F90_2PTR_PARAM(ptrd));
  PetscFunctionReturn(PETSC_SUCCESS);
}

#if defined(PETSC_HAVE_FORTRAN_CAPS)
  #define f90array1dgetaddrscalar_      F90ARRAY1DGETADDRSCALAR
  #define f90array1dgetaddrreal_        F90ARRAY1DGETADDRREAL
  #define f90array1dgetaddrint_         F90ARRAY1DGETADDRINT
  #define f90array1dgetaddrmpiint_      F90ARRAY1DGETADDRMPIINT
  #define f90array1dgetaddrfortranaddr_ F90ARRAY1DGETADDRFORTRANADDR
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
  #define f90array1dgetaddrscalar_      f90array1dgetaddrscalar
  #define f90array1dgetaddrreal_        f90array1dgetaddrreal
  #define f90array1dgetaddrint_         f90array1dgetaddrint
  #define f90array1dgetaddrmpiint_      f90array1dgetaddrmpiint
  #define f90array1dgetaddrfortranaddr_ f90array1dgetaddrfortranaddr
#endif

PETSC_EXTERN void f90array1dgetaddrscalar_(void *array, PetscFortranAddr *address)
{
  *address = (PetscFortranAddr)array;
}
PETSC_EXTERN void f90array1dgetaddrreal_(void *array, PetscFortranAddr *address)
{
  *address = (PetscFortranAddr)array;
}
PETSC_EXTERN void f90array1dgetaddrint_(void *array, PetscFortranAddr *address)
{
  *address = (PetscFortranAddr)array;
}
PETSC_EXTERN void f90array1dgetaddrmpiint_(void *array, PetscFortranAddr *address)
{
  *address = (PetscFortranAddr)array;
}
PETSC_EXTERN void f90array1dgetaddrfortranaddr_(void *array, PetscFortranAddr *address)
{
  *address = (PetscFortranAddr)array;
}

#if defined(PETSC_HAVE_FORTRAN_CAPS)
  #define f90array2dgetaddrscalar_      F90ARRAY2DGETADDRSCALAR
  #define f90array2dgetaddrreal_        F90ARRAY2DGETADDRREAL
  #define f90array2dgetaddrint_         F90ARRAY2DGETADDRINT
  #define f90array2dgetaddrfortranaddr_ F90ARRAY2DGETADDRFORTRANADDR
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
  #define f90array2dgetaddrscalar_      f90array2dgetaddrscalar
  #define f90array2dgetaddrreal_        f90array2dgetaddrreal
  #define f90array2dgetaddrint_         f90array2dgetaddrint
  #define f90array2dgetaddrfortranaddr_ f90array2dgetaddrfortranaddr
#endif

PETSC_EXTERN void f90array2dgetaddrscalar_(void *array, PetscFortranAddr *address)
{
  *address = (PetscFortranAddr)array;
}
PETSC_EXTERN void f90array2dgetaddrreal_(void *array, PetscFortranAddr *address)
{
  *address = (PetscFortranAddr)array;
}
PETSC_EXTERN void f90array2dgetaddrint_(void *array, PetscFortranAddr *address)
{
  *address = (PetscFortranAddr)array;
}
PETSC_EXTERN void f90array2dgetaddrfortranaddr_(void *array, PetscFortranAddr *address)
{
  *address = (PetscFortranAddr)array;
}

#if defined(PETSC_HAVE_FORTRAN_CAPS)
  #define f90array3dgetaddrscalar_      F90ARRAY3DGETADDRSCALAR
  #define f90array3dgetaddrreal_        F90ARRAY3DGETADDRREAL
  #define f90array3dgetaddrint_         F90ARRAY3DGETADDRINT
  #define f90array3dgetaddrfortranaddr_ F90ARRAY3DGETADDRFORTRANADDR
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
  #define f90array3dgetaddrscalar_      f90array3dgetaddrscalar
  #define f90array3dgetaddrreal_        f90array3dgetaddrreal
  #define f90array3dgetaddrint_         f90array3dgetaddrint
  #define f90array3dgetaddrfortranaddr_ f90array3dgetaddrfortranaddr
#endif

PETSC_EXTERN void f90array3dgetaddrscalar_(void *array, PetscFortranAddr *address)
{
  *address = (PetscFortranAddr)array;
}
PETSC_EXTERN void f90array3dgetaddrreal_(void *array, PetscFortranAddr *address)
{
  *address = (PetscFortranAddr)array;
}
PETSC_EXTERN void f90array3dgetaddrint_(void *array, PetscFortranAddr *address)
{
  *address = (PetscFortranAddr)array;
}
PETSC_EXTERN void f90array3dgetaddrfortranaddr_(void *array, PetscFortranAddr *address)
{
  *address = (PetscFortranAddr)array;
}

#if defined(PETSC_HAVE_FORTRAN_CAPS)
  #define f90array4dgetaddrscalar_      F90ARRAY4DGETADDRSCALAR
  #define f90array4dgetaddrreal_        F90ARRAY4DGETADDRREAL
  #define f90array4dgetaddrint_         F90ARRAY4DGETADDRINT
  #define f90array4dgetaddrfortranaddr_ F90ARRAY4DGETADDRFORTRANADDR
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
  #define f90array4dgetaddrscalar_      f90array4dgetaddrscalar
  #define f90array4dgetaddrreal_        f90array4dgetaddrreal
  #define f90array4dgetaddrint_         f90array4dgetaddrint
  #define f90array4dgetaddrfortranaddr_ f90array4dgetaddrfortranaddr
#endif

PETSC_EXTERN void f90array4dgetaddrscalar_(void *array, PetscFortranAddr *address)
{
  *address = (PetscFortranAddr)array;
}
PETSC_EXTERN void f90array4dgetaddrreal_(void *array, PetscFortranAddr *address)
{
  *address = (PetscFortranAddr)array;
}
PETSC_EXTERN void f90array4dgetaddrint_(void *array, PetscFortranAddr *address)
{
  *address = (PetscFortranAddr)array;
}
PETSC_EXTERN void f90array4dgetaddrfortranaddr_(void *array, PetscFortranAddr *address)
{
  *address = (PetscFortranAddr)array;
}
