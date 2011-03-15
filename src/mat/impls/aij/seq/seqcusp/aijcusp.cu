
/*
    Defines the basic matrix operations for the AIJ (compressed row)
  matrix storage format.
*/

#include <petscconf.h>
PETSC_CUDA_EXTERN_C_BEGIN
#include <../src/mat/impls/aij/seq/aij.h>          /*I "petscmat.h" I*/
#include <petscbt.h>
#include <../src/vec/vec/impls/dvecimpl.h>
#include <private/vecimpl.h>
PETSC_CUDA_EXTERN_C_END
#undef VecType
#include <../src/mat/impls/aij/seq/seqcusp/cuspmatimpl.h>

#undef __FUNCT__
#define __FUNCT__ "MatCUSPCopyToGPU"
PetscErrorCode MatCUSPCopyToGPU(Mat A)
{
  Mat_SeqAIJCUSP *cuspstruct  = (Mat_SeqAIJCUSP*)A->spptr;
  Mat_SeqAIJ      *a          = (Mat_SeqAIJ*)A->data;
  PetscInt        m           = A->rmap->n,*ii,*ridx;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  if (A->valid_GPU_matrix == PETSC_CUSP_UNALLOCATED || A->valid_GPU_matrix == PETSC_CUSP_CPU){
    ierr = PetscLogEventBegin(MAT_CUSPCopyToGPU,A,0,0,0);CHKERRQ(ierr);
    if (A->valid_GPU_matrix == PETSC_CUSP_UNALLOCATED){
      try {
        cuspstruct->mat = new CUSPMATRIX;
        if (a->compressedrow.use) {
          m    = a->compressedrow.nrows;
          ii   = a->compressedrow.i;
          ridx = a->compressedrow.rindex;
          cuspstruct->mat->resize(m,A->cmap->n,a->nz);
          cuspstruct->mat->row_offsets.assign(ii,ii+m+1);
          cuspstruct->mat->column_indices.assign(a->j,a->j+a->nz);
          cuspstruct->mat->values.assign(a->a,a->a+a->nz);
          cuspstruct->indices = new CUSPINTARRAYGPU;
          cuspstruct->indices->assign(ridx,ridx+m);
        } else {
          cuspstruct->mat->resize(m,A->cmap->n,a->nz);
          cuspstruct->mat->row_offsets.assign(a->i,a->i+m+1);
          cuspstruct->mat->column_indices.assign(a->j,a->j+a->nz);
          cuspstruct->mat->values.assign(a->a,a->a+a->nz);
        }
        cuspstruct->tempvec = new CUSPARRAY;
        cuspstruct->tempvec->resize(m);
      } catch(char* ex) {
        SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"CUSP error: %s", ex);
      }
    } else if (A->valid_GPU_matrix == PETSC_CUSP_CPU) {
      /*
       It may be possible to reuse nonzero structure with new matrix values but
       for simplicity and insured correctness we delete and build a new matrix on
       the GPU. Likely a very small performance hit.
       */
      if (cuspstruct->mat){
        try {
          delete (cuspstruct->mat);
          if (cuspstruct->tempvec) {
            delete (cuspstruct->tempvec);
          }
          if (cuspstruct->indices) {
            delete (cuspstruct->indices);
          }
        } catch(char* ex) {
          SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"CUSP error: %s", ex);
        }
      }
      try {
        cuspstruct->mat = new CUSPMATRIX;
        if (a->compressedrow.use) {
          m    = a->compressedrow.nrows;
          ii   = a->compressedrow.i;
          ridx = a->compressedrow.rindex;
          cuspstruct->mat->resize(m,A->cmap->n,a->nz);
          cuspstruct->mat->row_offsets.assign(ii,ii+m+1);
          cuspstruct->mat->column_indices.assign(a->j,a->j+a->nz);
          cuspstruct->mat->values.assign(a->a,a->a+a->nz);
          cuspstruct->indices = new CUSPINTARRAYGPU;
          cuspstruct->indices->assign(ridx,ridx+m);
        } else {
          cuspstruct->mat->resize(m,A->cmap->n,a->nz);
          cuspstruct->mat->row_offsets.assign(a->i,a->i+m+1);
          cuspstruct->mat->column_indices.assign(a->j,a->j+a->nz);
          cuspstruct->mat->values.assign(a->a,a->a+a->nz);
        }
        cuspstruct->tempvec = new CUSPARRAY;
        cuspstruct->tempvec->resize(m);
      } catch(char* ex) {
        SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"CUSP error: %s", ex);
      }
    }
    A->valid_GPU_matrix = PETSC_CUSP_BOTH;
    ierr = PetscLogEventEnd(MAT_CUSPCopyToGPU,A,0,0,0);CHKERRQ(ierr);
  }
    PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatCUSPCopyFromGPU"
PetscErrorCode MatCUSPCopyFromGPU(Mat A, CUSPMATRIX *Agpu)
{
  Mat_SeqAIJCUSP *cuspstruct = (Mat_SeqAIJCUSP *) A->spptr;
  Mat_SeqAIJ     *a          = (Mat_SeqAIJ *) A->data;
  PetscInt        m          = A->rmap->n;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  if (A->valid_GPU_matrix == PETSC_CUSP_UNALLOCATED) {
    if (A->valid_GPU_matrix == PETSC_CUSP_UNALLOCATED) {
      try {
        cuspstruct->mat = Agpu;
        if (a->compressedrow.use) {
          //PetscInt *ii, *ridx;
          SETERRQ(PETSC_COMM_WORLD, PETSC_ERR_ARG_WRONG, "Cannot handle row compression for GPU matrices");
        } else {
          PetscInt i;

          if (m+1 != (PetscInt) cuspstruct->mat->row_offsets.size()) {SETERRQ2(PETSC_COMM_WORLD, PETSC_ERR_ARG_SIZ, "GPU matrix has %d rows, should be %d", cuspstruct->mat->row_offsets.size()-1, m);}
          a->nz    = cuspstruct->mat->values.size();
          a->maxnz = a->nz; /* Since we allocate exactly the right amount */
          A->preallocated = PETSC_TRUE;
          // Copy ai, aj, aa
          if (a->singlemalloc) {
            if (a->a) {ierr = PetscFree3(a->a,a->j,a->i);CHKERRQ(ierr);}
          } else {
            if (a->i) {ierr = PetscFree(a->i);CHKERRQ(ierr);}
            if (a->j) {ierr = PetscFree(a->j);CHKERRQ(ierr);}
            if (a->a) {ierr = PetscFree(a->a);CHKERRQ(ierr);}
          }
          ierr = PetscMalloc3(a->nz,PetscScalar,&a->a,a->nz,PetscInt,&a->j,m+1,PetscInt,&a->i);CHKERRQ(ierr);
          ierr = PetscLogObjectMemory(A, a->nz*(sizeof(PetscScalar)+sizeof(PetscInt))+(m+1)*sizeof(PetscInt));CHKERRQ(ierr);
          a->singlemalloc = PETSC_TRUE;
          thrust::copy(cuspstruct->mat->row_offsets.begin(), cuspstruct->mat->row_offsets.end(), a->i);
          thrust::copy(cuspstruct->mat->column_indices.begin(), cuspstruct->mat->column_indices.end(), a->j);
          thrust::copy(cuspstruct->mat->values.begin(), cuspstruct->mat->values.end(), a->a);
          // Setup row lengths
          if (a->imax) {ierr = PetscFree2(a->imax,a->ilen);CHKERRQ(ierr);}
          ierr = PetscMalloc2(m,PetscInt,&a->imax,m,PetscInt,&a->ilen);CHKERRQ(ierr);
          ierr = PetscLogObjectMemory(A, 2*m*sizeof(PetscInt));CHKERRQ(ierr);
          for(i = 0; i < m; ++i) {
            a->imax[i] = a->ilen[i] = a->i[i+1] - a->i[i];
          }
          // a->diag?
        }
        cuspstruct->tempvec = new CUSPARRAY;
        cuspstruct->tempvec->resize(m);
      } catch(char *ex) {
        SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_LIB, "CUSP error: %s", ex);
      }
    }
    // This assembly prevents resetting the flag to PETSC_CUSP_CPU and recopying
    ierr = MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    A->valid_GPU_matrix = PETSC_CUSP_BOTH;
  } else {
    SETERRQ(PETSC_COMM_WORLD, PETSC_ERR_ARG_WRONG, "Only valid for unallocated GPU matrices");
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatGetVecs_SeqAIJCUSP"
PetscErrorCode MatGetVecs_SeqAIJCUSP(Mat mat, Vec *right, Vec *left)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;

  if (right) {
    ierr = VecCreate(((PetscObject)mat)->comm,right);CHKERRQ(ierr);
    ierr = VecSetSizes(*right,mat->cmap->n,PETSC_DETERMINE);CHKERRQ(ierr);
    ierr = VecSetBlockSize(*right,mat->rmap->bs);CHKERRQ(ierr);
    ierr = VecSetType(*right,VECSEQCUSP);CHKERRQ(ierr);
  }
  if (left) {
    ierr = VecCreate(((PetscObject)mat)->comm,left);CHKERRQ(ierr);
    ierr = VecSetSizes(*left,mat->rmap->n,PETSC_DETERMINE);CHKERRQ(ierr);
    ierr = VecSetBlockSize(*left,mat->rmap->bs);CHKERRQ(ierr);
    ierr = VecSetType(*left,VECSEQCUSP);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatMult_SeqAIJCUSP"
PetscErrorCode MatMult_SeqAIJCUSP(Mat A,Vec xx,Vec yy)
{
  Mat_SeqAIJ     *a = (Mat_SeqAIJ*)A->data;
  PetscErrorCode ierr;
  PetscInt       nonzerorow=0;
  PetscBool      usecprow    = a->compressedrow.use;
  Mat_SeqAIJCUSP *cuspstruct = (Mat_SeqAIJCUSP *)A->spptr;
  CUSPARRAY      *xarray,*yarray;

  PetscFunctionBegin;
  ierr = MatCUSPCopyToGPU(A);CHKERRQ(ierr);
  ierr = VecCUSPGetArrayRead(xx,&xarray);CHKERRQ(ierr);
  ierr = VecCUSPGetArrayWrite(yy,&yarray);CHKERRQ(ierr);
  if (usecprow){ /* use compressed row format */
    try {
      cusp::multiply(*cuspstruct->mat,*xarray,*cuspstruct->tempvec);
      ierr = VecSet_SeqCUSP(yy,0.0);CHKERRQ(ierr);
      thrust::copy(cuspstruct->tempvec->begin(),cuspstruct->tempvec->end(),thrust::make_permutation_iterator(yarray->begin(),cuspstruct->indices->begin()));
    } catch (char* ex) {
      SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"CUSP error: %s", ex);
    }
  } else { /* do not use compressed row format */
    try {
      cusp::multiply(*cuspstruct->mat,*xarray,*yarray);
    } catch(char* ex) {
      SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"CUSP error: %s", ex);
    }
  }
  ierr = VecCUSPRestoreArrayRead(xx,&xarray);CHKERRQ(ierr);
  ierr = VecCUSPRestoreArrayWrite(yy,&yarray);CHKERRQ(ierr);
  ierr = WaitForGPU();CHKERRCUSP(ierr);
  ierr = PetscLogFlops(2.0*a->nz - nonzerorow);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

struct VecCUSPPlusEquals
{
  template <typename Tuple>
  __host__ __device__
  void operator()(Tuple t)
  {
    thrust::get<1>(t) = thrust::get<1>(t) + thrust::get<0>(t);
  }
};

#undef __FUNCT__  
#define __FUNCT__ "MatMultAdd_SeqAIJCUSP"
PetscErrorCode MatMultAdd_SeqAIJCUSP(Mat A,Vec xx,Vec yy,Vec zz)
{
  Mat_SeqAIJ     *a = (Mat_SeqAIJ*)A->data;
  PetscErrorCode ierr;
  PetscBool      usecprow=a->compressedrow.use;
  Mat_SeqAIJCUSP *cuspstruct = (Mat_SeqAIJCUSP *)A->spptr;
  CUSPARRAY      *xarray,*yarray,*zarray;

  PetscFunctionBegin;
  ierr = MatCUSPCopyToGPU(A);CHKERRQ(ierr);
  if (usecprow) {
    try {
      ierr = VecCopy_SeqCUSP(yy,zz);CHKERRQ(ierr);
      ierr = VecCUSPGetArrayRead(xx,&xarray);CHKERRQ(ierr);
      ierr = VecCUSPGetArrayRead(yy,&yarray);CHKERRQ(ierr);
      ierr = VecCUSPGetArrayWrite(zz,&zarray);CHKERRQ(ierr);
      if (a->compressedrow.nrows) {
        cusp::multiply(*cuspstruct->mat,*xarray, *cuspstruct->tempvec);
        thrust::for_each(
           thrust::make_zip_iterator(
                 thrust::make_tuple(
                                    cuspstruct->tempvec->begin(),
                                    thrust::make_permutation_iterator(zarray->begin(), cuspstruct->indices->begin()))),
           thrust::make_zip_iterator(
                 thrust::make_tuple(
                                    cuspstruct->tempvec->begin(),
                                    thrust::make_permutation_iterator(zarray->begin(),cuspstruct->indices->begin()))) + cuspstruct->tempvec->size(),
           VecCUSPPlusEquals());
      }
    } catch(char* ex) {
      SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"CUSP error: %s", ex);
    }
    ierr = VecCUSPRestoreArrayRead(xx,&xarray);CHKERRQ(ierr);
    ierr = VecCUSPRestoreArrayRead(yy,&yarray);CHKERRQ(ierr);
    ierr = VecCUSPRestoreArrayWrite(zz,&zarray);CHKERRQ(ierr);
  } else {
    try {
      ierr = VecCopy_SeqCUSP(yy,zz);CHKERRQ(ierr);
      ierr = VecCUSPGetArrayRead(xx,&xarray);CHKERRQ(ierr);
      ierr = VecCUSPGetArrayRead(yy,&yarray);CHKERRQ(ierr);
      ierr = VecCUSPGetArrayWrite(zz,&zarray);CHKERRQ(ierr);
      cusp::multiply(*cuspstruct->mat,*xarray,*cuspstruct->tempvec);
      thrust::for_each(
         thrust::make_zip_iterator(
                 thrust::make_tuple(
                                    cuspstruct->tempvec->begin(),
                                    zarray->begin())),
         thrust::make_zip_iterator(
                 thrust::make_tuple(
                                    cuspstruct->tempvec->end(),
                                   zarray->end())),
         VecCUSPPlusEquals());
    } catch(char* ex) {
      SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"CUSP error: %s", ex);
    }
    ierr = VecCUSPRestoreArrayRead(xx,&xarray);CHKERRQ(ierr);
    ierr = VecCUSPRestoreArrayRead(yy,&yarray);CHKERRQ(ierr);
    ierr = VecCUSPRestoreArrayWrite(zz,&zarray);CHKERRQ(ierr);
  }
  ierr = PetscLogFlops(2.0*a->nz);CHKERRQ(ierr);
  ierr = WaitForGPU();CHKERRCUSP(ierr);
  PetscFunctionReturn(0);
}


#undef __FUNCT__  
#define __FUNCT__ "MatAssemblyEnd_SeqAIJCUSP"
PetscErrorCode MatAssemblyEnd_SeqAIJCUSP(Mat A,MatAssemblyType mode)
{
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  ierr = MatAssemblyEnd_SeqAIJ(A,mode);CHKERRQ(ierr);
  if (mode == MAT_FLUSH_ASSEMBLY) PetscFunctionReturn(0);
  if (A->valid_GPU_matrix != PETSC_CUSP_UNALLOCATED){
    A->valid_GPU_matrix = PETSC_CUSP_CPU;
  }
  PetscFunctionReturn(0);
}




/* --------------------------------------------------------------------------------*/
#undef __FUNCT__  
#define __FUNCT__ "MatCreateSeqAIJCUSP"
/*@C
   MatCreateSeqAIJCUSP - Creates a sparse matrix in AIJ (compressed row) format
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
PetscErrorCode  MatCreateSeqAIJCUSP(MPI_Comm comm,PetscInt m,PetscInt n,PetscInt nz,const PetscInt nnz[],Mat *A)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatCreate(comm,A);CHKERRQ(ierr);
  ierr = MatSetSizes(*A,m,n,m,n);CHKERRQ(ierr);
  ierr = MatSetType(*A,MATSEQAIJCUSP);CHKERRQ(ierr);
  ierr = MatSeqAIJSetPreallocation_SeqAIJ(*A,nz,(PetscInt*)nnz);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatDestroy_SeqAIJCUSP"
PetscErrorCode MatDestroy_SeqAIJCUSP(Mat A)
{
  PetscErrorCode ierr;
  Mat_SeqAIJCUSP *cuspcontainer = (Mat_SeqAIJCUSP*)A->spptr;

  PetscFunctionBegin;
  try {
    if (A->valid_GPU_matrix != PETSC_CUSP_UNALLOCATED){
      delete (CUSPMATRIX *)(cuspcontainer->mat);
    }
    delete cuspcontainer;
    A->valid_GPU_matrix = PETSC_CUSP_UNALLOCATED;
  } catch(char* ex) {
      SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"CUSP error: %s", ex);
  }
  /*this next line is because MatDestroy tries to PetscFree spptr if it is not zero, and PetscFree only works if the memory was allocated with PetscNew or PetscMalloc, which don't call the constructor */
  A->spptr = 0;
  ierr = MatDestroy_SeqAIJ(A);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatLoad_SeqAIJCUSP"
PetscErrorCode MatLoad_SeqAIJCUSP(Mat newMat, PetscViewer viewer)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;

  ierr = MatLoad_SeqAIJ(newMat,viewer);CHKERRQ(ierr);
  newMat->valid_GPU_matrix = PETSC_CUSP_GPU; /* MatLoad allocates the CPU and then copies to GPU so GPU is valid but both are allocated */
  PetscFunctionReturn(0);
}

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "MatCreate_SeqAIJCUSP"
PetscErrorCode  MatCreate_SeqAIJCUSP(Mat B)
{
  PetscErrorCode ierr;
  Mat_SeqAIJ     *aij;

  PetscFunctionBegin;
  ierr            = MatCreate_SeqAIJ(B);CHKERRQ(ierr);
  aij             = (Mat_SeqAIJ*)B->data;
  aij->inode.use  = PETSC_FALSE;
  B->ops->mult    = MatMult_SeqAIJCUSP;
  B->ops->multadd = MatMultAdd_SeqAIJCUSP;
  B->spptr        = new Mat_SeqAIJCUSP;
  ((Mat_SeqAIJCUSP *)B->spptr)->mat = 0;
  ((Mat_SeqAIJCUSP *)B->spptr)->tempvec = 0;
  ((Mat_SeqAIJCUSP *)B->spptr)->indices = 0;

  B->ops->assemblyend = MatAssemblyEnd_SeqAIJCUSP;
  B->ops->destroy     = MatDestroy_SeqAIJCUSP;
  B->ops->getvecs     = MatGetVecs_SeqAIJCUSP;
  ierr = PetscObjectChangeTypeName((PetscObject)B,MATSEQAIJCUSP);CHKERRQ(ierr);
  B->valid_GPU_matrix = PETSC_CUSP_UNALLOCATED;
  PetscFunctionReturn(0);
}
EXTERN_C_END
