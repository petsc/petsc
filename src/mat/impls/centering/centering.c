
#include <petsc/private/matimpl.h>          /*I "petscmat.h" I*/

PetscErrorCode MatMult_Centering(Mat A,Vec xx,Vec yy)
{
  PetscScalar       *y;
  const PetscScalar *x;
  PetscScalar       sum,mean;
  PetscInt          i,m=A->rmap->n,size;

  PetscFunctionBegin;
  PetscCall(VecSum(xx,&sum));
  PetscCall(VecGetSize(xx,&size));
  mean = sum / (PetscScalar)size;
  PetscCall(VecGetArrayRead(xx,&x));
  PetscCall(VecGetArray(yy,&y));
  for (i=0; i<m; i++) {
    y[i] = x[i] - mean;
  }
  PetscCall(VecRestoreArrayRead(xx,&x));
  PetscCall(VecRestoreArray(yy,&y));
  PetscFunctionReturn(0);
}

/*@
   MatCreateCentering - Creates a new matrix object that implements the (symmetric and idempotent) centering matrix, I - (1/N) * ones*ones'

   Collective on Mat

   Input Parameters:
+  comm - MPI communicator
.  n - number of local rows (or PETSC_DECIDE to have calculated if N is given)
           This value should be the same as the local size used in creating the
           y vector for the matrix-vector product y = Ax.
-  N - number of global rows (or PETSC_DETERMINE to have calculated if n is given)

   Output Parameter:
.  C - the matrix

   Notes:
   The entries of the matrix C are not explicitly stored. Instead, the new matrix
   object is a shell that simply performs the action of the centering matrix, i.e.,
   multiplying C*x subtracts the mean of the vector x from each of its elements.
   This is useful for preserving sparsity when mean-centering the columns of a
   matrix is required. For instance, to perform principal components analysis with
   a matrix A, the composite matrix C*A can be passed to a partial SVD solver.

   Level: intermediate

.seealso: MatCreateLRC(), MatCreateComposite()
@*/
PetscErrorCode MatCreateCentering(MPI_Comm comm,PetscInt n,PetscInt N,Mat *C)
{
  PetscMPIInt    size;

  PetscFunctionBegin;
  PetscCall(MatCreate(comm,C));
  PetscCall(MatSetSizes(*C,n,n,N,N));
  PetscCallMPI(MPI_Comm_size(comm,&size));
  PetscCall(PetscObjectChangeTypeName((PetscObject)*C,MATCENTERING));

  (*C)->ops->mult         = MatMult_Centering;
  (*C)->assembled         = PETSC_TRUE;
  (*C)->preallocated      = PETSC_TRUE;
  (*C)->symmetric         = PETSC_TRUE;
  (*C)->symmetric_eternal = PETSC_TRUE;
  PetscCall(MatSetUp(*C));
  PetscFunctionReturn(0);
}
