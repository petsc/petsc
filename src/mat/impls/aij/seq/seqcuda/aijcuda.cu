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
#undef VecType
#include "../src/mat/impls/aij/seq/seqcuda/cudamatimpl.h"

#undef __FUNCT__
#define __FUNCT__ "MatCUDACopyToGPU"
PetscErrorCode MatCUDACopyToGPU(Mat A)
{
  Mat_SeqAIJCUDA *cudastruct  = (Mat_SeqAIJCUDA*)A->spptr;
  Mat_SeqAIJ      *a          = (Mat_SeqAIJ*)A->data;
  PetscInt        m           = A->rmap->n,*ii,*ridx;

  PetscFunctionBegin;
  if (A->valid_GPU_matrix == PETSC_CUDA_UNALLOCATED){
    try {
      cudastruct->mat = new CUSPMATRIX;
      if (a->compressedrow.use) {
        m    = a->compressedrow.nrows;
        ii   = a->compressedrow.i;
        ridx = a->compressedrow.rindex;
        cudastruct->mat->resize(m,A->cmap->n,a->nz);
        cudastruct->mat->row_offsets.assign(ii,ii+m+1);
        cudastruct->mat->column_indices.assign(a->j,a->j+a->nz);
        cudastruct->mat->values.assign(a->a,a->a+a->nz);
        cudastruct->indices = new CUSPINTARRAYGPU;
        cudastruct->indices->assign(ridx,ridx+m);
      } else {
        cudastruct->mat->resize(m,A->cmap->n,a->nz);
        cudastruct->mat->row_offsets.assign(a->i,a->i+m+1);
        cudastruct->mat->column_indices.assign(a->j,a->j+a->nz);
        cudastruct->mat->values.assign(a->a,a->a+a->nz);
      }
      cudastruct->tempvec = new CUSPARRAY;
      cudastruct->tempvec->resize(m);
    } catch(char* ex) {
      SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"CUDA error: %s", ex);
    }
  } else if (A->valid_GPU_matrix == PETSC_CUDA_CPU) {
  /*
       It may be possible to reuse nonzero structure with new matrix values but 
     for simplicity and insured correctness we delete and build a new matrix on
     the GPU. Likely a very small performance hit.
  */
    if (cudastruct->mat){
      try {
        delete (cudastruct->mat);
        if (cudastruct->tempvec) {
          delete (cudastruct->tempvec);
        }
        if (cudastruct->indices) {
          delete (cudastruct->indices);
        }
      } catch(char* ex) {
        SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"CUDA error: %s", ex);
      } 
    }
    try {
      cudastruct->mat = new CUSPMATRIX;
      if (a->compressedrow.use) {
        m    = a->compressedrow.nrows;
        ii   = a->compressedrow.i;
        ridx = a->compressedrow.rindex;
        cudastruct->mat->resize(m,A->cmap->n,a->nz);
        cudastruct->mat->row_offsets.assign(ii,ii+m+1);
        cudastruct->mat->column_indices.assign(a->j,a->j+a->nz);
        cudastruct->mat->values.assign(a->a,a->a+a->nz);
        cudastruct->indices = new CUSPINTARRAYGPU;
        cudastruct->indices->assign(ridx,ridx+m);
      } else {
        cudastruct->mat->resize(m,A->cmap->n,a->nz);
        cudastruct->mat->row_offsets.assign(a->i,a->i+m+1);
        cudastruct->mat->column_indices.assign(a->j,a->j+a->nz);
        cudastruct->mat->values.assign(a->a,a->a+a->nz);
      }
      cudastruct->tempvec = new CUSPARRAY;
      cudastruct->tempvec->resize(m);
    } catch(char* ex) {
      SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"CUDA error: %s", ex);
    } 
  }
  A->valid_GPU_matrix = PETSC_CUDA_BOTH;
  PetscFunctionReturn(0);
}
#undef __FUNCT__  
#define __FUNCT__ "MatMult_SeqAIJCUDA"
PetscErrorCode MatMult_SeqAIJCUDA(Mat A,Vec xx,Vec yy)
{
  Mat_SeqAIJ     *a = (Mat_SeqAIJ*)A->data;
  PetscErrorCode ierr;
  PetscInt       nonzerorow=0;
  PetscBool      usecprow    = a->compressedrow.use;
  Mat_SeqAIJCUDA *cudastruct = (Mat_SeqAIJCUDA *)A->spptr;

  PetscFunctionBegin;
  ierr = MatCUDACopyToGPU(A);CHKERRQ(ierr);
  ierr = VecCUDACopyToGPU(xx);CHKERRQ(ierr);
  ierr = VecCUDAAllocateCheck(yy);CHKERRQ(ierr);
  if (usecprow){ /* use compressed row format */
    try {
      cusp::multiply(*cudastruct->mat,*((Vec_CUDA *)xx->spptr)->GPUarray,*cudastruct->tempvec);
      ierr = VecSet_SeqCUDA(yy,0.0);CHKERRQ(ierr);
      thrust::copy(cudastruct->tempvec->begin(),cudastruct->tempvec->end(),thrust::make_permutation_iterator(((Vec_CUDA *)yy->spptr)->GPUarray->begin(),cudastruct->indices->begin()));
    } catch (char* ex) {
      SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"CUDA error: %s", ex);
    }
  } else { /* do not use compressed row format */
    try {
      cusp::multiply(*cudastruct->mat,*((Vec_CUDA *)xx->spptr)->GPUarray,*((Vec_CUDA *)yy->spptr)->GPUarray);
    } catch(char* ex) {
      SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"CUDA error: %s", ex);
    } 
  }
  yy->valid_GPU_array = PETSC_CUDA_GPU;
  ierr = WaitForGPU();CHKERRCUDA(ierr);
  ierr = PetscLogFlops(2.0*a->nz - nonzerorow);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

struct VecCUDAPlusEquals
{
  template <typename Tuple>
  __host__ __device__
  void operator()(Tuple t)
  {
    thrust::get<1>(t) = thrust::get<1>(t) + thrust::get<0>(t);
  }
};

#undef __FUNCT__  
#define __FUNCT__ "MatMultAdd_SeqAIJCUDA"
PetscErrorCode MatMultAdd_SeqAIJCUDA(Mat A,Vec xx,Vec yy,Vec zz)
{
  Mat_SeqAIJ     *a = (Mat_SeqAIJ*)A->data;
  PetscErrorCode ierr;
  PetscBool      usecprow=a->compressedrow.use;
  Mat_SeqAIJCUDA *cudastruct = (Mat_SeqAIJCUDA *)A->spptr;

  PetscFunctionBegin;
  ierr = MatCUDACopyToGPU(A);CHKERRQ(ierr);
  ierr = VecCUDACopyToGPU(xx);CHKERRQ(ierr);
  ierr = VecCUDACopyToGPU(yy);CHKERRQ(ierr);
  ierr = VecCUDAAllocateCheck(zz);CHKERRQ(ierr);
  if (usecprow) {
    try {
      ierr = VecCopy_SeqCUDA(yy,zz);CHKERRQ(ierr);
      if (a->compressedrow.nrows) {
        cusp::multiply(*cudastruct->mat,*((Vec_CUDA *)xx->spptr)->GPUarray, *cudastruct->tempvec);
        thrust::for_each(
           thrust::make_zip_iterator(
                 thrust::make_tuple(
                                    cudastruct->tempvec->begin(),
                                    thrust::make_permutation_iterator(((Vec_CUDA *)zz->spptr)->GPUarray->begin(), cudastruct->indices->begin()))),
           thrust::make_zip_iterator(
                 thrust::make_tuple(
                                    cudastruct->tempvec->begin(),
                                    thrust::make_permutation_iterator(((Vec_CUDA *)zz->spptr)->GPUarray->begin(),cudastruct->indices->begin()))) + cudastruct->tempvec->size(),
           VecCUDAPlusEquals());
      }
    } catch(char* ex) {
      SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"CUDA error: %s", ex);
    }
  } else {
    try {
      ierr = VecCopy_SeqCUDA(yy,zz);CHKERRQ(ierr);
      cusp::multiply(*cudastruct->mat,*((Vec_CUDA *)xx->spptr)->GPUarray,*cudastruct->tempvec);
      thrust::for_each(
         thrust::make_zip_iterator(
                 thrust::make_tuple(
                                    cudastruct->tempvec->begin(),
                                    ((Vec_CUDA *)zz->spptr)->GPUarray->begin())),
         thrust::make_zip_iterator(
                 thrust::make_tuple(
                                    cudastruct->tempvec->end(),
                                   ((Vec_CUDA *)zz->spptr)->GPUarray->end())),
         VecCUDAPlusEquals());
    } catch(char* ex) {
      SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"CUDA error: %s", ex);
    } 
  }
  ierr = PetscLogFlops(2.0*a->nz);CHKERRQ(ierr);
  ierr = WaitForGPU();CHKERRCUDA(ierr);
  zz->valid_GPU_array = PETSC_CUDA_GPU;
  PetscFunctionReturn(0);
}


#undef __FUNCT__  
#define __FUNCT__ "MatAssemblyEnd_SeqAIJCUDA"
PetscErrorCode MatAssemblyEnd_SeqAIJCUDA(Mat A,MatAssemblyType mode)
{
  PetscErrorCode  ierr;
  
  PetscFunctionBegin;
  ierr = MatAssemblyEnd_SeqAIJ(A,mode);CHKERRQ(ierr);
  if (mode == MAT_FLUSH_ASSEMBLY) PetscFunctionReturn(0);
  if (A->valid_GPU_matrix != PETSC_CUDA_UNALLOCATED){
    A->valid_GPU_matrix = PETSC_CUDA_CPU;
  }
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
  PetscErrorCode ierr;
  Mat_SeqAIJCUDA *cudacontainer = (Mat_SeqAIJCUDA*)A->spptr;

  PetscFunctionBegin;
  try {
    if (A->valid_GPU_matrix != PETSC_CUDA_UNALLOCATED){
      delete (CUSPMATRIX *)(cudacontainer->mat);
    }
    delete cudacontainer;
    A->valid_GPU_matrix = PETSC_CUDA_UNALLOCATED;
  } catch(char* ex) {
      SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"CUDA error: %s", ex);
  } 
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
  ierr            = MatCreate_SeqAIJ(B);CHKERRQ(ierr);
  aij             = (Mat_SeqAIJ*)B->data;
  aij->inode.use  = PETSC_FALSE;
  B->ops->mult    = MatMult_SeqAIJCUDA;
  B->ops->multadd = MatMultAdd_SeqAIJCUDA;
  B->spptr        = new Mat_SeqAIJCUDA;
  ((Mat_SeqAIJCUDA *)B->spptr)->mat = 0;
  ((Mat_SeqAIJCUDA *)B->spptr)->tempvec = 0;
  ((Mat_SeqAIJCUDA *)B->spptr)->indices = 0;
  
  B->ops->assemblyend = MatAssemblyEnd_SeqAIJCUDA;
  B->ops->destroy     = MatDestroy_SeqAIJCUDA;
  ierr = PetscObjectChangeTypeName((PetscObject)B,MATSEQAIJCUDA);CHKERRQ(ierr);
  B->valid_GPU_matrix = PETSC_CUDA_UNALLOCATED;
  PetscFunctionReturn(0);
}
EXTERN_C_END
