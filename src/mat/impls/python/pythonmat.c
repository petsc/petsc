#include <petsc/private/matimpl.h>          /*I "petscmat.h" I*/

/*@C
   MatPythonSetType - Initialize a Mat object implemented in Python.

   Collective on Mat

   Input Parameters:
+  mat - the matrix (Mat) object.
-  pyname - full dotted Python name [package].module[.{class|function}]

   Options Database Key:
.  -mat_python_type <pyname> - python class

   Level: intermediate

.seealso: MatCreate(), MatSetType(), MATPYTHON, PetscPythonInitialize()
@*/
PetscErrorCode  MatPythonSetType(Mat mat,const char pyname[])
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(mat,MAT_CLASSID,1);
  PetscValidCharPointer(pyname,2);
  PetscTryMethod(mat,"MatPythonSetType_C",(Mat, const char[]),(mat,pyname));
  PetscFunctionReturn(0);
}

/*@C
   MatPythonCreate - Create a Mat object implemented in Python.

   Collective on Mat

   Input Parameters:
+  comm - MPI communicator
.  m - number of local rows (or PETSC_DECIDE to have calculated if M is given)
.  n - number of local columns (or PETSC_DECIDE to have calculated if N is given)
.  M - number of global rows (or PETSC_DECIDE to have calculated if m is given)
.  N - number of global columns (or PETSC_DECIDE to have calculated if n is given)
-  pyname - full dotted Python name [package].module[.{class|function}]

   Output Parameter:
.  A - the matrix

   Level: intermediate

.seealso: MATPYTHON, MatPythonSetType(), PetscPythonInitialize()

@*/
PetscErrorCode  MatPythonCreate(MPI_Comm comm,PetscInt m,PetscInt n,PetscInt M,PetscInt N,const char pyname[],Mat *A)
{
  PetscFunctionBegin;
  PetscValidCharPointer(pyname,6);
  PetscValidPointer(A,6);
  PetscCall(MatCreate(comm,A));
  PetscCall(MatSetSizes(*A,m,n,M,N));
  PetscCall(MatSetType(*A,MATPYTHON));
  PetscCall(MatPythonSetType(*A,pyname));
  PetscFunctionReturn(0);
}
