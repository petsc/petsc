#define PETSCMAT_DLL


/*
    Defines the basic matrix operations for the AIJ (compressed row)
  matrix storage format.
*/

#include "petscconf.h"
PETSC_CUDA_EXTERN_C_BEGIN
#include "../src/mat/impls/aij/seq/aij.h"          /*I "petscmat.h" I*/
#include "petscbt.h"
#include "../src/vec/vec/impls/dvecimpl.h"
#include "private/vecimpl.h"
PETSC_CUDA_EXTERN_C_END
#include "../src/vec/vec/impls/seq/seqcuda/cudavecimpl.h"

#include <cusp/csr_matrix.h>
#include <cusp/multiply.h>

#ifndef CUSPARRAY
#define CUSPARRAY cusp::array1d<PetscScalar,cusp::device_memory>
#endif
#define CUSPMATRIX cusp::csr_matrix<PetscInt,PetscScalar,cusp::device_memory>

EXTERN PetscErrorCode VecCUDACopyToGPU_Public(Vec);
EXTERN PetscErrorCode VecCUDAAllocateCheck_Public(Vec);

#undef __FUNCT__  
#define __FUNCT__ "MatMult_SeqAIJCUDA"
PetscErrorCode MatMult_SeqAIJCUDA(Mat A,Vec xx,Vec yy)
{
  Mat_SeqAIJ        *a = (Mat_SeqAIJ*)A->data;
  PetscScalar       *y;
  PetscScalar *x;
  const MatScalar   *aa;
  PetscErrorCode    ierr;
  PetscInt          m=A->rmap->n;
  const PetscInt    *aj,*ii,*ridx=PETSC_NULL;
  PetscInt          n,i,nonzerorow=0;
  PetscScalar       sum;
  PetscTruth        usecprow=a->compressedrow.use;

#if defined(PETSC_HAVE_PRAGMA_DISJOINT)
#pragma disjoint(*x,*y,*aa)
#endif

  PetscFunctionBegin;
  aj  = a->j;
  aa  = a->a;
  ii  = a->i;
  if (usecprow){ /* use compressed row format */
    ierr = VecGetArray(xx,&x);CHKERRQ(ierr);
    ierr = VecGetArray(yy,&y);CHKERRQ(ierr);
    m    = a->compressedrow.nrows;
    ii   = a->compressedrow.i;
    ridx = a->compressedrow.rindex;
    for (i=0; i<m; i++){
      n   = ii[i+1] - ii[i]; 
      aj  = a->j + ii[i];
      aa  = a->a + ii[i];
      sum = 0.0;
      nonzerorow += (n>0);
      PetscSparseDensePlusDot(sum,x,aa,aj,n); 
      /* for (j=0; j<n; j++) sum += (*aa++)*x[*aj++]; */
      y[*ridx++] = sum;
    }
    ierr = VecRestoreArray(xx,&x);CHKERRQ(ierr);
    ierr = VecRestoreArray(yy,&y);CHKERRQ(ierr);
  } else { /* do not use compressed row format */
  ierr = VecCUDACopyToGPU(xx);CHKERRQ(ierr);
  ierr = VecCUDAAllocateCheck(yy);CHKERRQ(ierr);
  cusp::multiply(*(CUSPMATRIX *)(A->spptr),*(CUSPARRAY *)(xx->spptr),*(CUSPARRAY *)(yy->spptr));
  yy->valid_GPU_array = PETSC_CUDA_GPU;
  }
  ierr = PetscLogFlops(2.0*a->nz - nonzerorow);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


#undef __FUNCT__  
#define __FUNCT__ "MatAssemblyEnd_SeqAIJCUDA"
PetscErrorCode MatAssemblyEnd_SeqAIJCUDA(Mat A,MatAssemblyType mode)
{
  Mat_SeqAIJ     *a = (Mat_SeqAIJ*)A->data;
  PetscErrorCode ierr;
  PetscInt       m = A->rmap->n;

  PetscFunctionBegin;
  ierr = MatAssemblyEnd_SeqAIJ(A,mode);CHKERRQ(ierr);
  A->spptr = new CUSPMATRIX;
  ((CUSPMATRIX *)(A->spptr))->resize(m,A->cmap->n,a->nz);
  ((CUSPMATRIX *)(A->spptr))->row_offsets.assign(a->i,a->i+m+1);
  ((CUSPMATRIX *)(A->spptr))->column_indices.assign(a->j,a->j+a->nz);
  ((CUSPMATRIX *)(A->spptr))->values.assign(a->a,a->a+a->nz);

  /* This shouldn't be necessary
  A->ops->mult = MatMult_SeqAIJCUDA;
  */
  PetscFunctionReturn(0);
}




/* --------------------------------------------------------------------------------*/
#undef __FUNCT__  
#define __FUNCT__ "MatCreateSeqAIJCUDA"
/*@C
   MatCreateSeqAIJCUDA - Creates a sparse matrix in AIJ (compressed row) format
   (the default parallel PETSc format).  For good matrix assembly performance
   the user should preallocate the matrix storage by setting the parameter nz
   (or the array nnz).  By setting these parameters accurately, performance
   during matrix assembly can be increased by more than a factor of 50.

   Collective on MPI_Comm

   Input Parameters:
+  comm - MPI communicator, set to PETSC_COMM_SELF
.  m - number of rows
.  n - number of columns
.  nz - number of nonzeros per row (same for all rows)
-  nnz - array containing the number of nonzeros in the various rows 
         (possibly different for each row) or PETSC_NULL

   Output Parameter:
.  A - the matrix 

   It is recommended that one use the MatCreate(), MatSetType() and/or MatSetFromOptions(),
   MatXXXXSetPreallocation() paradgm instead of this routine directly.
   [MatXXXXSetPreallocation() is, for example, MatSeqAIJSetPreallocation]

   Notes:
   If nnz is given then nz is ignored

   The AIJ format (also called the Yale sparse matrix format or
   compressed row storage), is fully compatible with standard Fortran 77
   storage.  That is, the stored row and column indices can begin at
   either one (as in Fortran) or zero.  See the users' manual for details.

   Specify the preallocated storage with either nz or nnz (not both).
   Set nz=PETSC_DEFAULT and nnz=PETSC_NULL for PETSc to control dynamic memory 
   allocation.  For large problems you MUST preallocate memory or you 
   will get TERRIBLE performance, see the users' manual chapter on matrices.

   By default, this format uses inodes (identical nodes) when possible, to 
   improve numerical efficiency of matrix-vector products and solves. We 
   search for consecutive rows with the same nonzero structure, thereby
   reusing matrix information to achieve increased efficiency.

   Level: intermediate

.seealso: MatCreate(), MatCreateMPIAIJ(), MatSetValues(), MatSeqAIJSetColumnIndices(), MatCreateSeqAIJWithArrays(), MatCreateMPIAIJ()

@*/
PetscErrorCode PETSCMAT_DLLEXPORT MatCreateSeqAIJCUDA(MPI_Comm comm,PetscInt m,PetscInt n,PetscInt nz,const PetscInt nnz[],Mat *A)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatCreate(comm,A);CHKERRQ(ierr);
  ierr = MatSetSizes(*A,m,n,m,n);CHKERRQ(ierr);
  ierr = MatSetType(*A,MATSEQAIJCUDA);CHKERRQ(ierr);
  ierr = MatSeqAIJSetPreallocation_SeqAIJ(*A,nz,(PetscInt*)nnz);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatDestroy_SeqAIJCUDA"
PetscErrorCode MatDestroy_SeqAIJCUDA(Mat A)
{
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  delete (CUSPMATRIX *)(A->spptr);
  /*this next line is because MatDestroy tries to PetscFree spptr if it is not zero, and PetscFree only works if the memory was allocated with PetscNew or PetscMalloc, which don't call the constructor */
  A->spptr = 0;
  ierr = MatDestroy_SeqAIJ(A);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "MatCreate_SeqAIJCUDA"
PetscErrorCode PETSCMAT_DLLEXPORT MatCreate_SeqAIJCUDA(Mat B)
{
  PetscErrorCode ierr;
  Mat_SeqAIJ     *aij;

  PetscFunctionBegin;
  ierr           = MatCreate_SeqAIJ(B);CHKERRQ(ierr);
  aij            = (Mat_SeqAIJ*)B->data;
  aij->inode.use = PETSC_FALSE;
#if !defined(PETSC_USE_FORTRAN_KERNEL_MULTAIJ)
  B->ops->mult   = MatMult_SeqAIJCUDA;
#endif
  B->ops->assemblyend = MatAssemblyEnd_SeqAIJCUDA;
  B->ops->destroy     = MatDestroy_SeqAIJCUDA;
  ierr = PetscObjectChangeTypeName((PetscObject)B,MATSEQAIJCUDA);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
EXTERN_C_END
