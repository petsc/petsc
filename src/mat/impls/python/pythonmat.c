#include "private/matimpl.h"          /*I "petscmat.h" I*/

#undef __FUNCT__
#define __FUNCT__ "MatPythonSetType"
/*@C
   MatPythonSetType - Initalize a Mat object implemented in Python.

   Collective on Mat

   Input Parameter:
+  mat - the matrix (Mat) object.
-  pyname - full dotted Python name [package].module[.{class|function}]

   Options Database Key:
.  -mat_python_type <pyname>

   Level: intermediate

.keywords: Mat, Python

.seealso: MATPYTHON, MatCreatePython(), PetscPythonInitialize()
@*/
PetscErrorCode PETSCMAT_DLLEXPORT MatPythonSetType(Mat mat,const char pyname[])
{
  PetscErrorCode (*f)(Mat, const char[]) = 0;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(mat,MAT_COOKIE,1);
  PetscValidCharPointer(pyname,2);
  ierr = PetscObjectQueryFunction((PetscObject)mat,"MatPythonSetType_C",
				  (PetscVoidFunction*)&f);CHKERRQ(ierr);
  if (f) {ierr = (*f)(mat,pyname);CHKERRQ(ierr);}
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

.keywords: Mat, Python

.seealso: MATPYTHON, MatPythonSetType(), PetscPythonInitialize()

@*/
#undef __FUNCT__
#define __FUNCT__ "MatPythonCreate"
PetscErrorCode PETSCMAT_DLLEXPORT MatPythonCreate(MPI_Comm comm,PetscInt m,PetscInt n,PetscInt M,PetscInt N,const char pyname[],Mat *A)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidCharPointer(pyname,6);
  PetscValidPointer(A,6);
  ierr = MatCreate(comm,A);CHKERRQ(ierr);
  ierr = MatSetSizes(*A,m,n,M,N);CHKERRQ(ierr);
  ierr = MatSetType(*A,MATPYTHON);CHKERRQ(ierr);
  ierr = MatPythonSetType(*A,pyname);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
