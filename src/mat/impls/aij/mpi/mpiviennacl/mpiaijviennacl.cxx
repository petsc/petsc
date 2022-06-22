#define PETSC_SKIP_IMMINTRIN_H_CUDAWORKAROUND 1

#include <petscconf.h>
#include <../src/mat/impls/aij/mpi/mpiaij.h>   /*I "petscmat.h" I*/
#include <../src/mat/impls/aij/seq/seqviennacl/viennaclmatimpl.h>

PetscErrorCode  MatMPIAIJSetPreallocation_MPIAIJViennaCL(Mat B,PetscInt d_nz,const PetscInt d_nnz[],PetscInt o_nz,const PetscInt o_nnz[])
{
  Mat_MPIAIJ     *b = (Mat_MPIAIJ*)B->data;

  PetscFunctionBegin;
  PetscCall(PetscLayoutSetUp(B->rmap));
  PetscCall(PetscLayoutSetUp(B->cmap));
  if (!B->preallocated) {
    /* Explicitly create the two MATSEQAIJVIENNACL matrices. */
    PetscCall(MatCreate(PETSC_COMM_SELF,&b->A));
    PetscCall(MatSetSizes(b->A,B->rmap->n,B->cmap->n,B->rmap->n,B->cmap->n));
    PetscCall(MatSetType(b->A,MATSEQAIJVIENNACL));
    PetscCall(PetscLogObjectParent((PetscObject)B,(PetscObject)b->A));
    PetscCall(MatCreate(PETSC_COMM_SELF,&b->B));
    PetscCall(MatSetSizes(b->B,B->rmap->n,B->cmap->N,B->rmap->n,B->cmap->N));
    PetscCall(MatSetType(b->B,MATSEQAIJVIENNACL));
    PetscCall(PetscLogObjectParent((PetscObject)B,(PetscObject)b->B));
  }
  PetscCall(MatSeqAIJSetPreallocation(b->A,d_nz,d_nnz));
  PetscCall(MatSeqAIJSetPreallocation(b->B,o_nz,o_nnz));
  B->preallocated = PETSC_TRUE;
  PetscFunctionReturn(0);
}

PetscErrorCode MatAssemblyEnd_MPIAIJViennaCL(Mat A,MatAssemblyType mode)
{
  Mat_MPIAIJ     *b = (Mat_MPIAIJ*)A->data;
  PetscBool      v;

  PetscFunctionBegin;
  PetscCall(MatAssemblyEnd_MPIAIJ(A,mode));
  PetscCall(PetscObjectTypeCompare((PetscObject)b->lvec,VECSEQVIENNACL,&v));
  if (!v) {
    PetscInt m;
    PetscCall(VecGetSize(b->lvec,&m));
    PetscCall(VecDestroy(&b->lvec));
    PetscCall(VecCreateSeqViennaCL(PETSC_COMM_SELF,m,&b->lvec));
  }
  PetscFunctionReturn(0);
}

PetscErrorCode MatDestroy_MPIAIJViennaCL(Mat A)
{
  PetscFunctionBegin;
  PetscCall(MatDestroy_MPIAIJ(A));
  PetscFunctionReturn(0);
}

PETSC_EXTERN PetscErrorCode MatCreate_MPIAIJViennaCL(Mat A)
{
  PetscFunctionBegin;
  PetscCall(MatCreate_MPIAIJ(A));
  A->boundtocpu = PETSC_FALSE;
  PetscCall(PetscFree(A->defaultvectype));
  PetscCall(PetscStrallocpy(VECVIENNACL,&A->defaultvectype));
  PetscCall(PetscObjectComposeFunction((PetscObject)A,"MatMPIAIJSetPreallocation_C",MatMPIAIJSetPreallocation_MPIAIJViennaCL));
  A->ops->assemblyend = MatAssemblyEnd_MPIAIJViennaCL;
  PetscCall(PetscObjectChangeTypeName((PetscObject)A,MATMPIAIJVIENNACL));
  PetscFunctionReturn(0);
}

/*@C
   MatCreateAIJViennaCL - Creates a sparse matrix in AIJ (compressed row) format
   (the default parallel PETSc format).  This matrix will ultimately be pushed down
   to GPUs and use the ViennaCL library for calculations. For good matrix
   assembly performance the user should preallocate the matrix storage by setting
   the parameter nz (or the array nnz).  By setting these parameters accurately,
   performance during matrix assembly can be increased substantially.

   Collective

   Input Parameters:
+  comm - MPI communicator, set to PETSC_COMM_SELF
.  m - number of rows
.  n - number of columns
.  nz - number of nonzeros per row (same for all rows)
-  nnz - array containing the number of nonzeros in the various rows
         (possibly different for each row) or NULL

   Output Parameter:
.  A - the matrix

   It is recommended that one use the MatCreate(), MatSetType() and/or MatSetFromOptions(),
   MatXXXXSetPreallocation() paradigm instead of this routine directly.
   [MatXXXXSetPreallocation() is, for example, MatSeqAIJSetPreallocation]

   Notes:
   If nnz is given then nz is ignored

   The AIJ format (also called the Yale sparse matrix format or
   compressed row storage), is fully compatible with standard Fortran 77
   storage.  That is, the stored row and column indices can begin at
   either one (as in Fortran) or zero.  See the users' manual for details.

   Specify the preallocated storage with either nz or nnz (not both).
   Set nz=PETSC_DEFAULT and nnz=NULL for PETSc to control dynamic memory
   allocation.  For large problems you MUST preallocate memory or you
   will get TERRIBLE performance, see the users' manual chapter on matrices.

   Level: intermediate

.seealso: `MatCreate()`, `MatCreateAIJ()`, `MatCreateAIJCUSPARSE()`, `MatSetValues()`, `MatSeqAIJSetColumnIndices()`, `MatCreateSeqAIJWithArrays()`, `MatCreateAIJ()`, `MATMPIAIJVIENNACL`, `MATAIJVIENNACL`
@*/
PetscErrorCode  MatCreateAIJViennaCL(MPI_Comm comm,PetscInt m,PetscInt n,PetscInt M,PetscInt N,PetscInt d_nz,const PetscInt d_nnz[],PetscInt o_nz,const PetscInt o_nnz[],Mat *A)
{
  PetscMPIInt    size;

  PetscFunctionBegin;
  PetscCall(MatCreate(comm,A));
  PetscCall(MatSetSizes(*A,m,n,M,N));
  PetscCallMPI(MPI_Comm_size(comm,&size));
  if (size > 1) {
    PetscCall(MatSetType(*A,MATMPIAIJVIENNACL));
    PetscCall(MatMPIAIJSetPreallocation(*A,d_nz,d_nnz,o_nz,o_nnz));
  } else {
    PetscCall(MatSetType(*A,MATSEQAIJVIENNACL));
    PetscCall(MatSeqAIJSetPreallocation(*A,d_nz,d_nnz));
  }
  PetscFunctionReturn(0);
}

/*MC
   MATAIJVIENNACL - MATMPIAIJVIENNACL= "aijviennacl" = "mpiaijviennacl" - A matrix type to be used for sparse matrices.

   A matrix type (CSR format) whose data resides on GPUs.
   All matrix calculations are performed using the ViennaCL library.

   This matrix type is identical to MATSEQAIJVIENNACL when constructed with a single process communicator,
   and MATMPIAIJVIENNACL otherwise.  As a result, for single process communicators,
   MatSeqAIJSetPreallocation is supported, and similarly MatMPIAIJSetPreallocation is supported
   for communicators controlling multiple processes.  It is recommended that you call both of
   the above preallocation routines for simplicity.

   Options Database Keys:
.  -mat_type mpiaijviennacl - sets the matrix type to "mpiaijviennacl" during a call to MatSetFromOptions()

  Level: beginner

 .seealso: `MatCreateAIJViennaCL()`, `MATSEQAIJVIENNACL`, `MatCreateSeqAIJVIENNACL()`
M*/
