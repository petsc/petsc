#include <../src/mat/impls/baij/mpi/mpibaij.h>

PETSC_INTERN PetscErrorCode MatConvert_SeqBAIJ_SeqBAIJMKL(Mat,MatType,MatReuse,Mat*);

static PetscErrorCode  MatMPIBAIJSetPreallocation_MPIBAIJMKL(Mat B,PetscInt bs,PetscInt d_nz,const PetscInt *d_nnz,PetscInt o_nz,const PetscInt *o_nnz)
{
  Mat_MPIBAIJ     *b = (Mat_MPIBAIJ*)B->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatMPIBAIJSetPreallocation_MPIBAIJ(B,bs,d_nz,d_nnz,o_nz,o_nnz);CHKERRQ(ierr);
  ierr = MatConvert_SeqBAIJ_SeqBAIJMKL(b->A,MATSEQBAIJMKL,MAT_INPLACE_MATRIX,&b->A);CHKERRQ(ierr);
  ierr = MatConvert_SeqBAIJ_SeqBAIJMKL(b->B,MATSEQBAIJMKL,MAT_INPLACE_MATRIX,&b->B);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode MatConvert_MPIBAIJ_MPIBAIJMKL(Mat A,MatType type,MatReuse reuse,Mat *newmat)
{
  PetscErrorCode ierr;
  Mat            B = *newmat;

  PetscFunctionBegin;
  if (reuse == MAT_INITIAL_MATRIX) {
    ierr = MatDuplicate(A,MAT_COPY_VALUES,&B);CHKERRQ(ierr);
  }

  ierr = PetscObjectChangeTypeName((PetscObject) B, MATMPIBAIJMKL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)B,"MatMPIBAIJSetPreallocation_C",MatMPIBAIJSetPreallocation_MPIBAIJMKL);CHKERRQ(ierr);
  *newmat = B;
  PetscFunctionReturn(0);
}

/*@C
   MatCreateBAIJMKL - Creates a sparse parallel matrix in block AIJ format
   (block compressed row).
   This type inherits from BAIJ and is largely identical, but uses sparse BLAS
   routines from Intel MKL whenever possible.
   MatMult, MatMultAdd, MatMultTranspose, and MatMultTransposeAdd
   operations are currently supported.
   If the installed version of MKL supports the "SpMV2" sparse
   inspector-executor routines, then those are used by default.
   Default PETSc kernels are used otherwise.
   For good matrix assembly performance the user should preallocate the matrix
   storage by setting the parameters d_nz (or d_nnz) and o_nz (or o_nnz).
   By setting these parameters accurately, performance can be increased by more
   than a factor of 50.

   Collective

   Input Parameters:
+  comm - MPI communicator
.  bs   - size of block, the blocks are ALWAYS square. One can use MatSetBlockSizes() to set a different row and column blocksize but the row
          blocksize always defines the size of the blocks. The column blocksize sets the blocksize of the vectors obtained with MatCreateVecs()
.  m - number of local rows (or PETSC_DECIDE to have calculated if M is given)
           This value should be the same as the local size used in creating the
           y vector for the matrix-vector product y = Ax.
.  n - number of local columns (or PETSC_DECIDE to have calculated if N is given)
           This value should be the same as the local size used in creating the
           x vector for the matrix-vector product y = Ax.
.  M - number of global rows (or PETSC_DETERMINE to have calculated if m is given)
.  N - number of global columns (or PETSC_DETERMINE to have calculated if n is given)
.  d_nz  - number of nonzero blocks per block row in diagonal portion of local
           submatrix  (same for all local rows)
.  d_nnz - array containing the number of nonzero blocks in the various block rows
           of the in diagonal portion of the local (possibly different for each block
           row) or NULL.  If you plan to factor the matrix you must leave room for the diagonal entry
           and set it even if it is zero.
.  o_nz  - number of nonzero blocks per block row in the off-diagonal portion of local
           submatrix (same for all local rows).
-  o_nnz - array containing the number of nonzero blocks in the various block rows of the
           off-diagonal portion of the local submatrix (possibly different for
           each block row) or NULL.

   Output Parameter:
.  A - the matrix

   Options Database Keys:
+   -mat_block_size - size of the blocks to use
-   -mat_use_hash_table <fact> - set hash table factor

   It is recommended that one use the MatCreate(), MatSetType() and/or MatSetFromOptions(),
   MatXXXXSetPreallocation() paradigm instead of this routine directly.
   [MatXXXXSetPreallocation() is, for example, MatSeqAIJSetPreallocation]

   Notes:
   If the *_nnz parameter is given then the *_nz parameter is ignored

   A nonzero block is any block that as 1 or more nonzeros in it

   The user MUST specify either the local or global matrix dimensions
   (possibly both).

   If PETSC_DECIDE or  PETSC_DETERMINE is used for a particular argument on one processor
   than it must be used on all processors that share the object for that argument.

   Storage Information:
   For a square global matrix we define each processor's diagonal portion
   to be its local rows and the corresponding columns (a square submatrix);
   each processor's off-diagonal portion encompasses the remainder of the
   local matrix (a rectangular submatrix).

   The user can specify preallocated storage for the diagonal part of
   the local submatrix with either d_nz or d_nnz (not both).  Set
   d_nz=PETSC_DEFAULT and d_nnz=NULL for PETSc to control dynamic
   memory allocation.  Likewise, specify preallocated storage for the
   off-diagonal part of the local submatrix with o_nz or o_nnz (not both).

   Consider a processor that owns rows 3, 4 and 5 of a parallel matrix. In
   the figure below we depict these three local rows and all columns (0-11).

.vb
           0 1 2 3 4 5 6 7 8 9 10 11
          --------------------------
   row 3  |o o o d d d o o o o  o  o
   row 4  |o o o d d d o o o o  o  o
   row 5  |o o o d d d o o o o  o  o
          --------------------------
.ve

   Thus, any entries in the d locations are stored in the d (diagonal)
   submatrix, and any entries in the o locations are stored in the
   o (off-diagonal) submatrix.  Note that the d and the o submatrices are
   stored simply in the MATSEQBAIJMKL format for compressed row storage.

   Now d_nz should indicate the number of block nonzeros per row in the d matrix,
   and o_nz should indicate the number of block nonzeros per row in the o matrix.
   In general, for PDE problems in which most nonzeros are near the diagonal,
   one expects d_nz >> o_nz.   For large problems you MUST preallocate memory
   or you will get TERRIBLE performance; see the users' manual chapter on
   matrices.

   Level: intermediate

.seealso: MatCreate(), MatCreateSeqBAIJMKL(), MatSetValues(), MatCreateBAIJMKL(), MatMPIBAIJSetPreallocation(), MatMPIBAIJSetPreallocationCSR()
@*/

PetscErrorCode  MatCreateBAIJMKL(MPI_Comm comm,PetscInt bs,PetscInt m,PetscInt n,PetscInt M,PetscInt N,PetscInt d_nz,const PetscInt d_nnz[],PetscInt o_nz,const PetscInt o_nnz[],Mat *A)
{
  PetscErrorCode ierr;
  PetscMPIInt    size;

  PetscFunctionBegin;
  ierr = MatCreate(comm,A);CHKERRQ(ierr);
  ierr = MatSetSizes(*A,m,n,M,N);CHKERRQ(ierr);
  ierr = MPI_Comm_size(comm,&size);CHKERRMPI(ierr);
  if (size > 1) {
    ierr = MatSetType(*A,MATMPIBAIJMKL);CHKERRQ(ierr);
    ierr = MatMPIBAIJSetPreallocation(*A,bs,d_nz,d_nnz,o_nz,o_nnz);CHKERRQ(ierr);
  } else {
    ierr = MatSetType(*A,MATSEQBAIJMKL);CHKERRQ(ierr);
    ierr = MatSeqBAIJSetPreallocation(*A,bs,d_nz,d_nnz);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

PETSC_EXTERN PetscErrorCode MatCreate_MPIBAIJMKL(Mat A)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatSetType(A,MATMPIBAIJ);CHKERRQ(ierr);
  ierr = MatConvert_MPIBAIJ_MPIBAIJMKL(A,MATMPIBAIJMKL,MAT_INPLACE_MATRIX,&A);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*MC
   MATBAIJMKL - MATBAIJMKL = "BAIJMKL" - A matrix type to be used for sparse matrices.

   This matrix type is identical to MATSEQBAIJMKL when constructed with a single process communicator,
   and MATMPIBAIJMKL otherwise.  As a result, for single process communicators,
  MatSeqBAIJSetPreallocation() is supported, and similarly MatMPIBAIJSetPreallocation() is supported
  for communicators controlling multiple processes.  It is recommended that you call both of
  the above preallocation routines for simplicity.

   Options Database Keys:
. -mat_type baijmkl - sets the matrix type to "BAIJMKL" during a call to MatSetFromOptions()

  Level: beginner

.seealso: MatCreateBAIJMKL(), MATSEQBAIJMKL, MATMPIBAIJMKL
M*/
