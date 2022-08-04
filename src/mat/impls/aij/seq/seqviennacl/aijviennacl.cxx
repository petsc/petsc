
/*
    Defines the basic matrix operations for the AIJ (compressed row)
  matrix storage format.
*/

#include <petscconf.h>
#define PETSC_SKIP_IMMINTRIN_H_CUDAWORKAROUND 1
#include <../src/mat/impls/aij/seq/aij.h>          /*I "petscmat.h" I*/
#include <petscbt.h>
#include <../src/vec/vec/impls/dvecimpl.h>
#include <petsc/private/vecimpl.h>

#include <../src/mat/impls/aij/seq/seqviennacl/viennaclmatimpl.h>

#include <algorithm>
#include <vector>
#include <string>

#include "viennacl/linalg/prod.hpp"

PETSC_INTERN PetscErrorCode MatConvert_SeqAIJ_SeqAIJViennaCL(Mat A, MatType type, MatReuse reuse, Mat *newmat);
PETSC_INTERN PetscErrorCode MatGetFactor_seqaij_petsc(Mat,MatFactorType,Mat*);
PETSC_INTERN PetscErrorCode MatProductSetFromOptions_SeqAIJ_SeqDense(Mat);

PetscErrorCode MatViennaCLCopyToGPU(Mat A)
{
  Mat_SeqAIJViennaCL *viennaclstruct = (Mat_SeqAIJViennaCL*)A->spptr;
  Mat_SeqAIJ         *a              = (Mat_SeqAIJ*)A->data;

  PetscFunctionBegin;
  if (A->rmap->n > 0 && A->cmap->n > 0 && a->nz) { //some OpenCL SDKs have issues with buffers of size 0
    if (A->offloadmask == PETSC_OFFLOAD_UNALLOCATED || A->offloadmask == PETSC_OFFLOAD_CPU) {
      PetscCall(PetscLogEventBegin(MAT_ViennaCLCopyToGPU,A,0,0,0));

      try {
        if (a->compressedrow.use) {
          if (!viennaclstruct->compressed_mat) viennaclstruct->compressed_mat = new ViennaCLCompressedAIJMatrix();

          // Since PetscInt is different from cl_uint, we have to convert:
          viennacl::backend::mem_handle dummy;

          viennacl::backend::typesafe_host_array<unsigned int> row_buffer; row_buffer.raw_resize(dummy, a->compressedrow.nrows+1);
          for (PetscInt i=0; i<=a->compressedrow.nrows; ++i)
            row_buffer.set(i, (a->compressedrow.i)[i]);

          viennacl::backend::typesafe_host_array<unsigned int> row_indices; row_indices.raw_resize(dummy, a->compressedrow.nrows);
          for (PetscInt i=0; i<a->compressedrow.nrows; ++i)
            row_indices.set(i, (a->compressedrow.rindex)[i]);

          viennacl::backend::typesafe_host_array<unsigned int> col_buffer; col_buffer.raw_resize(dummy, a->nz);
          for (PetscInt i=0; i<a->nz; ++i)
            col_buffer.set(i, (a->j)[i]);

          viennaclstruct->compressed_mat->set(row_buffer.get(), row_indices.get(), col_buffer.get(), a->a, A->rmap->n, A->cmap->n, a->compressedrow.nrows, a->nz);
          PetscCall(PetscLogCpuToGpu(((2*a->compressedrow.nrows)+1+a->nz)*sizeof(PetscInt) + (a->nz)*sizeof(PetscScalar)));
        } else {
          if (!viennaclstruct->mat) viennaclstruct->mat = new ViennaCLAIJMatrix();

          // Since PetscInt is in general different from cl_uint, we have to convert:
          viennacl::backend::mem_handle dummy;

          viennacl::backend::typesafe_host_array<unsigned int> row_buffer; row_buffer.raw_resize(dummy, A->rmap->n+1);
          for (PetscInt i=0; i<=A->rmap->n; ++i)
            row_buffer.set(i, (a->i)[i]);

          viennacl::backend::typesafe_host_array<unsigned int> col_buffer; col_buffer.raw_resize(dummy, a->nz);
          for (PetscInt i=0; i<a->nz; ++i)
            col_buffer.set(i, (a->j)[i]);

          viennaclstruct->mat->set(row_buffer.get(), col_buffer.get(), a->a, A->rmap->n, A->cmap->n, a->nz);
          PetscCall(PetscLogCpuToGpu(((A->rmap->n+1)+a->nz)*sizeof(PetscInt)+(a->nz)*sizeof(PetscScalar)));
        }
        ViennaCLWaitForGPU();
      } catch(std::exception const & ex) {
        SETERRQ(PETSC_COMM_SELF,PETSC_ERR_LIB,"ViennaCL error: %s", ex.what());
      }

      // Create temporary vector for v += A*x:
      if (viennaclstruct->tempvec) {
        if (viennaclstruct->tempvec->size() != static_cast<std::size_t>(A->rmap->n)) {
          delete (ViennaCLVector*)viennaclstruct->tempvec;
          viennaclstruct->tempvec = new ViennaCLVector(A->rmap->n);
        } else {
          viennaclstruct->tempvec->clear();
        }
      } else {
        viennaclstruct->tempvec = new ViennaCLVector(A->rmap->n);
      }

      A->offloadmask = PETSC_OFFLOAD_BOTH;

      PetscCall(PetscLogEventEnd(MAT_ViennaCLCopyToGPU,A,0,0,0));
    }
  }
  PetscFunctionReturn(0);
}

PetscErrorCode MatViennaCLCopyFromGPU(Mat A, const ViennaCLAIJMatrix *Agpu)
{
  Mat_SeqAIJViennaCL *viennaclstruct = (Mat_SeqAIJViennaCL*)A->spptr;
  Mat_SeqAIJ         *a = (Mat_SeqAIJ*)A->data;
  PetscInt           m  = A->rmap->n;

  PetscFunctionBegin;
  if (A->offloadmask == PETSC_OFFLOAD_BOTH) PetscFunctionReturn(0);
  if (A->offloadmask == PETSC_OFFLOAD_UNALLOCATED && Agpu) {
    try {
      PetscCheck(!a->compressedrow.use,PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "ViennaCL: Cannot handle row compression for GPU matrices");
      PetscCheck((PetscInt)Agpu->size1() == m,PETSC_COMM_SELF, PETSC_ERR_ARG_SIZ, "GPU matrix has %lu rows, should be %" PetscInt_FMT, Agpu->size1(), m);
      a->nz           = Agpu->nnz();
      a->maxnz        = a->nz; /* Since we allocate exactly the right amount */
      A->preallocated = PETSC_TRUE;
      if (a->singlemalloc) {
        if (a->a) PetscCall(PetscFree3(a->a,a->j,a->i));
      } else {
        if (a->i) PetscCall(PetscFree(a->i));
        if (a->j) PetscCall(PetscFree(a->j));
        if (a->a) PetscCall(PetscFree(a->a));
      }
      PetscCall(PetscMalloc3(a->nz,&a->a,a->nz,&a->j,m+1,&a->i));
      PetscCall(PetscLogObjectMemory((PetscObject)A, a->nz*(sizeof(PetscScalar)+sizeof(PetscInt))+(m+1)*sizeof(PetscInt)));

      a->singlemalloc = PETSC_TRUE;

      /* Setup row lengths */
      PetscCall(PetscFree(a->imax));
      PetscCall(PetscFree(a->ilen));
      PetscCall(PetscMalloc1(m,&a->imax));
      PetscCall(PetscMalloc1(m,&a->ilen));
      PetscCall(PetscLogObjectMemory((PetscObject)A, 2*m*sizeof(PetscInt)));

      /* Copy data back from GPU */
      viennacl::backend::typesafe_host_array<unsigned int> row_buffer; row_buffer.raw_resize(Agpu->handle1(), Agpu->size1() + 1);

      // copy row array
      viennacl::backend::memory_read(Agpu->handle1(), 0, row_buffer.raw_size(), row_buffer.get());
      (a->i)[0] = row_buffer[0];
      for (PetscInt i = 0; i < (PetscInt)Agpu->size1(); ++i) {
        (a->i)[i+1] = row_buffer[i+1];
        a->imax[i]  = a->ilen[i] = a->i[i+1] - a->i[i];  //Set imax[] and ilen[] arrays at the same time as i[] for better cache reuse
      }

      // copy column indices
      viennacl::backend::typesafe_host_array<unsigned int> col_buffer; col_buffer.raw_resize(Agpu->handle2(), Agpu->nnz());
      viennacl::backend::memory_read(Agpu->handle2(), 0, col_buffer.raw_size(), col_buffer.get());
      for (PetscInt i=0; i < (PetscInt)Agpu->nnz(); ++i)
        (a->j)[i] = col_buffer[i];

      // copy nonzero entries directly to destination (no conversion required)
      viennacl::backend::memory_read(Agpu->handle(), 0, sizeof(PetscScalar)*Agpu->nnz(), a->a);

      PetscCall(PetscLogGpuToCpu(row_buffer.raw_size()+col_buffer.raw_size()+(Agpu->nnz()*sizeof(PetscScalar))));
      ViennaCLWaitForGPU();
      /* TODO: Once a->diag is moved out of MatAssemblyEnd(), invalidate it here. */
    } catch(std::exception const & ex) {
      SETERRQ(PETSC_COMM_SELF, PETSC_ERR_LIB, "ViennaCL error: %s", ex.what());
    }
  } else if (A->offloadmask == PETSC_OFFLOAD_UNALLOCATED) {
    PetscFunctionReturn(0);
  } else {
    if (!Agpu && A->offloadmask != PETSC_OFFLOAD_GPU) PetscFunctionReturn(0);

    PetscCheck(!a->compressedrow.use,PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "ViennaCL: Cannot handle row compression for GPU matrices");
    if (!Agpu) {
      viennacl::backend::memory_read(viennaclstruct->mat->handle(), 0, sizeof(PetscScalar)*viennaclstruct->mat->nnz(), a->a);
    } else {
      viennacl::backend::memory_read(Agpu->handle(), 0, sizeof(PetscScalar)*Agpu->nnz(), a->a);
    }
  }
  A->offloadmask = PETSC_OFFLOAD_BOTH;
  /* This assembly prevents resetting the flag to PETSC_OFFLOAD_CPU and recopying */
  PetscCall(MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY));
  PetscFunctionReturn(0);
}

PetscErrorCode MatMult_SeqAIJViennaCL(Mat A,Vec xx,Vec yy)
{
  Mat_SeqAIJ           *a = (Mat_SeqAIJ*)A->data;
  Mat_SeqAIJViennaCL   *viennaclstruct = (Mat_SeqAIJViennaCL*)A->spptr;
  const ViennaCLVector *xgpu=NULL;
  ViennaCLVector       *ygpu=NULL;

  PetscFunctionBegin;
  /* The line below is necessary due to the operations that modify the matrix on the CPU (axpy, scale, etc) */
  PetscCall(MatViennaCLCopyToGPU(A));
  if (A->rmap->n > 0 && A->cmap->n > 0 && a->nz) {
    PetscCall(VecViennaCLGetArrayRead(xx,&xgpu));
    PetscCall(VecViennaCLGetArrayWrite(yy,&ygpu));
    PetscCall(PetscLogGpuTimeBegin());
    try {
      if (a->compressedrow.use) {
        *ygpu = viennacl::linalg::prod(*viennaclstruct->compressed_mat, *xgpu);
      } else {
        *ygpu = viennacl::linalg::prod(*viennaclstruct->mat,*xgpu);
      }
      ViennaCLWaitForGPU();
    } catch (std::exception const & ex) {
      SETERRQ(PETSC_COMM_SELF,PETSC_ERR_LIB,"ViennaCL error: %s", ex.what());
    }
    PetscCall(PetscLogGpuTimeEnd());
    PetscCall(VecViennaCLRestoreArrayRead(xx,&xgpu));
    PetscCall(VecViennaCLRestoreArrayWrite(yy,&ygpu));
    PetscCall(PetscLogGpuFlops(2.0*a->nz - a->nonzerorowcnt));
  } else {
    PetscCall(VecSet_SeqViennaCL(yy,0));
  }
  PetscFunctionReturn(0);
}

PetscErrorCode MatMultAdd_SeqAIJViennaCL(Mat A,Vec xx,Vec yy,Vec zz)
{
  Mat_SeqAIJ           *a = (Mat_SeqAIJ*)A->data;
  Mat_SeqAIJViennaCL   *viennaclstruct = (Mat_SeqAIJViennaCL*)A->spptr;
  const ViennaCLVector *xgpu=NULL,*ygpu=NULL;
  ViennaCLVector       *zgpu=NULL;

  PetscFunctionBegin;
  /* The line below is necessary due to the operations that modify the matrix on the CPU (axpy, scale, etc) */
  PetscCall(MatViennaCLCopyToGPU(A));
  if (A->rmap->n > 0 && A->cmap->n > 0 && a->nz) {
    try {
      PetscCall(VecViennaCLGetArrayRead(xx,&xgpu));
      PetscCall(VecViennaCLGetArrayRead(yy,&ygpu));
      PetscCall(VecViennaCLGetArrayWrite(zz,&zgpu));
      PetscCall(PetscLogGpuTimeBegin());
      if (a->compressedrow.use) *viennaclstruct->tempvec = viennacl::linalg::prod(*viennaclstruct->compressed_mat, *xgpu);
      else *viennaclstruct->tempvec = viennacl::linalg::prod(*viennaclstruct->mat, *xgpu);
      if (zz != yy) *zgpu = *ygpu + *viennaclstruct->tempvec;
      else *zgpu += *viennaclstruct->tempvec;
      ViennaCLWaitForGPU();
      PetscCall(PetscLogGpuTimeEnd());
      PetscCall(VecViennaCLRestoreArrayRead(xx,&xgpu));
      PetscCall(VecViennaCLRestoreArrayRead(yy,&ygpu));
      PetscCall(VecViennaCLRestoreArrayWrite(zz,&zgpu));

    } catch(std::exception const & ex) {
      SETERRQ(PETSC_COMM_SELF,PETSC_ERR_LIB,"ViennaCL error: %s", ex.what());
    }
    PetscCall(PetscLogGpuFlops(2.0*a->nz));
  } else {
    PetscCall(VecCopy_SeqViennaCL(yy,zz));
  }
  PetscFunctionReturn(0);
}

PetscErrorCode MatAssemblyEnd_SeqAIJViennaCL(Mat A,MatAssemblyType mode)
{
  PetscFunctionBegin;
  PetscCall(MatAssemblyEnd_SeqAIJ(A,mode));
  if (mode == MAT_FLUSH_ASSEMBLY) PetscFunctionReturn(0);
  if (!A->boundtocpu) PetscCall(MatViennaCLCopyToGPU(A));
  PetscFunctionReturn(0);
}

/* --------------------------------------------------------------------------------*/
/*@C
   MatCreateSeqAIJViennaCL - Creates a sparse matrix in AIJ (compressed row) format
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

.seealso: `MatCreate()`, `MatCreateAIJ()`, `MatCreateAIJCUSPARSE()`, `MatSetValues()`, `MatSeqAIJSetColumnIndices()`, `MatCreateSeqAIJWithArrays()`, `MatCreateAIJ()`

@*/
PetscErrorCode  MatCreateSeqAIJViennaCL(MPI_Comm comm,PetscInt m,PetscInt n,PetscInt nz,const PetscInt nnz[],Mat *A)
{
  PetscFunctionBegin;
  PetscCall(MatCreate(comm,A));
  PetscCall(MatSetSizes(*A,m,n,m,n));
  PetscCall(MatSetType(*A,MATSEQAIJVIENNACL));
  PetscCall(MatSeqAIJSetPreallocation_SeqAIJ(*A,nz,(PetscInt*)nnz));
  PetscFunctionReturn(0);
}

PetscErrorCode MatDestroy_SeqAIJViennaCL(Mat A)
{
  Mat_SeqAIJViennaCL *viennaclcontainer = (Mat_SeqAIJViennaCL*)A->spptr;

  PetscFunctionBegin;
  try {
    if (viennaclcontainer) {
      delete viennaclcontainer->tempvec;
      delete viennaclcontainer->mat;
      delete viennaclcontainer->compressed_mat;
      delete viennaclcontainer;
    }
    A->offloadmask = PETSC_OFFLOAD_UNALLOCATED;
  } catch(std::exception const & ex) {
    SETERRQ(PETSC_COMM_SELF,PETSC_ERR_LIB,"ViennaCL error: %s", ex.what());
  }

  PetscCall(PetscObjectComposeFunction((PetscObject)A,"MatConvert_seqaij_seqaijviennacl_C",NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)A,"MatProductSetFromOptions_seqaijviennacl_seqdense_C",NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)A,"MatProductSetFromOptions_seqaijviennacl_seqaij_C",NULL));

  /* this next line is because MatDestroy tries to PetscFree spptr if it is not zero, and PetscFree only works if the memory was allocated with PetscNew or PetscMalloc, which don't call the constructor */
  A->spptr = 0;
  PetscCall(MatDestroy_SeqAIJ(A));
  PetscFunctionReturn(0);
}

PETSC_EXTERN PetscErrorCode MatCreate_SeqAIJViennaCL(Mat B)
{
  PetscFunctionBegin;
  PetscCall(MatCreate_SeqAIJ(B));
  PetscCall(MatConvert_SeqAIJ_SeqAIJViennaCL(B,MATSEQAIJVIENNACL,MAT_INPLACE_MATRIX,&B));
  PetscFunctionReturn(0);
}

static PetscErrorCode MatBindToCPU_SeqAIJViennaCL(Mat,PetscBool);
static PetscErrorCode MatDuplicate_SeqAIJViennaCL(Mat A,MatDuplicateOption cpvalues,Mat *B)
{
  Mat C;

  PetscFunctionBegin;
  PetscCall(MatDuplicate_SeqAIJ(A,cpvalues,B));
  C = *B;

  PetscCall(MatBindToCPU_SeqAIJViennaCL(A,PETSC_FALSE));
  C->ops->bindtocpu = MatBindToCPU_SeqAIJViennaCL;

  C->spptr = new Mat_SeqAIJViennaCL();
  ((Mat_SeqAIJViennaCL*)C->spptr)->tempvec        = NULL;
  ((Mat_SeqAIJViennaCL*)C->spptr)->mat            = NULL;
  ((Mat_SeqAIJViennaCL*)C->spptr)->compressed_mat = NULL;

  PetscCall(PetscObjectChangeTypeName((PetscObject)C,MATSEQAIJVIENNACL));

  C->offloadmask = PETSC_OFFLOAD_UNALLOCATED;

  /* If the source matrix is already assembled, copy the destination matrix to the GPU */
  if (C->assembled) PetscCall(MatViennaCLCopyToGPU(C));
  PetscFunctionReturn(0);
}

static PetscErrorCode MatSeqAIJGetArray_SeqAIJViennaCL(Mat A,PetscScalar *array[])
{
  PetscFunctionBegin;
  PetscCall(MatViennaCLCopyFromGPU(A,(const ViennaCLAIJMatrix *)NULL));
  *array = ((Mat_SeqAIJ*)A->data)->a;
  PetscFunctionReturn(0);
}

static PetscErrorCode MatSeqAIJRestoreArray_SeqAIJViennaCL(Mat A,PetscScalar *array[])
{
  PetscFunctionBegin;
  A->offloadmask = PETSC_OFFLOAD_CPU;
  *array         = NULL;
  PetscFunctionReturn(0);
}

static PetscErrorCode MatSeqAIJGetArrayRead_SeqAIJViennaCL(Mat A,const PetscScalar *array[])
{
  PetscFunctionBegin;
  PetscCall(MatViennaCLCopyFromGPU(A,(const ViennaCLAIJMatrix *)NULL));
  *array = ((Mat_SeqAIJ*)A->data)->a;
  PetscFunctionReturn(0);
}

static PetscErrorCode MatSeqAIJRestoreArrayRead_SeqAIJViennaCL(Mat A,const PetscScalar *array[])
{
  PetscFunctionBegin;
  *array = NULL;
  /* No A->offloadmask = PETSC_OFFLOAD_CPU since if A->offloadmask was PETSC_OFFLOAD_BOTH, it is still BOTH */
  PetscFunctionReturn(0);
}

static PetscErrorCode MatSeqAIJGetArrayWrite_SeqAIJViennaCL(Mat A,PetscScalar *array[])
{
  PetscFunctionBegin;
  *array = ((Mat_SeqAIJ*)A->data)->a;
  PetscFunctionReturn(0);
}

static PetscErrorCode MatSeqAIJRestoreArrayWrite_SeqAIJViennaCL(Mat A,PetscScalar *array[])
{
  PetscFunctionBegin;
  A->offloadmask = PETSC_OFFLOAD_CPU;
  *array         = NULL;
  PetscFunctionReturn(0);
}

static PetscErrorCode MatBindToCPU_SeqAIJViennaCL(Mat A,PetscBool flg)
{
  Mat_SeqAIJ *a = (Mat_SeqAIJ*)A->data;

  PetscFunctionBegin;
  A->boundtocpu  = flg;
  if (flg && a->inode.size) {
    a->inode.use = PETSC_TRUE;
  } else {
    a->inode.use = PETSC_FALSE;
  }
  if (flg) {
    /* make sure we have an up-to-date copy on the CPU */
    PetscCall(MatViennaCLCopyFromGPU(A,(const ViennaCLAIJMatrix *)NULL));
    A->ops->mult        = MatMult_SeqAIJ;
    A->ops->multadd     = MatMultAdd_SeqAIJ;
    A->ops->assemblyend = MatAssemblyEnd_SeqAIJ;
    A->ops->duplicate   = MatDuplicate_SeqAIJ;
    PetscCall(PetscMemzero(a->ops,sizeof(Mat_SeqAIJOps)));
  } else {
    A->ops->mult        = MatMult_SeqAIJViennaCL;
    A->ops->multadd     = MatMultAdd_SeqAIJViennaCL;
    A->ops->assemblyend = MatAssemblyEnd_SeqAIJViennaCL;
    A->ops->destroy     = MatDestroy_SeqAIJViennaCL;
    A->ops->duplicate   = MatDuplicate_SeqAIJViennaCL;

    a->ops->getarray           = MatSeqAIJGetArray_SeqAIJViennaCL;
    a->ops->restorearray       = MatSeqAIJRestoreArray_SeqAIJViennaCL;
    a->ops->getarrayread       = MatSeqAIJGetArrayRead_SeqAIJViennaCL;
    a->ops->restorearrayread   = MatSeqAIJRestoreArrayRead_SeqAIJViennaCL;
    a->ops->getarraywrite      = MatSeqAIJGetArrayWrite_SeqAIJViennaCL;
    a->ops->restorearraywrite  = MatSeqAIJRestoreArrayWrite_SeqAIJViennaCL;
  }
  PetscFunctionReturn(0);
}

PETSC_INTERN PetscErrorCode MatConvert_SeqAIJ_SeqAIJViennaCL(Mat A,MatType type,MatReuse reuse,Mat *newmat)
{
  Mat B;

  PetscFunctionBegin;
  PetscCheck(reuse != MAT_REUSE_MATRIX,PetscObjectComm((PetscObject)A),PETSC_ERR_SUP,"MAT_REUSE_MATRIX is not supported. Consider using MAT_INPLACE_MATRIX instead");

  if (reuse == MAT_INITIAL_MATRIX) PetscCall(MatDuplicate(A,MAT_COPY_VALUES,newmat));

  B = *newmat;

  B->spptr = new Mat_SeqAIJViennaCL();

  ((Mat_SeqAIJViennaCL*)B->spptr)->tempvec        = NULL;
  ((Mat_SeqAIJViennaCL*)B->spptr)->mat            = NULL;
  ((Mat_SeqAIJViennaCL*)B->spptr)->compressed_mat = NULL;

  PetscCall(MatBindToCPU_SeqAIJViennaCL(A,PETSC_FALSE));
  A->ops->bindtocpu = MatBindToCPU_SeqAIJViennaCL;

  PetscCall(PetscObjectChangeTypeName((PetscObject)B,MATSEQAIJVIENNACL));
  PetscCall(PetscFree(B->defaultvectype));
  PetscCall(PetscStrallocpy(VECVIENNACL,&B->defaultvectype));

  PetscCall(PetscObjectComposeFunction((PetscObject)A,"MatConvert_seqaij_seqaijviennacl_C",MatConvert_SeqAIJ_SeqAIJViennaCL));
  PetscCall(PetscObjectComposeFunction((PetscObject)A,"MatProductSetFromOptions_seqaijviennacl_seqdense_C",MatProductSetFromOptions_SeqAIJ_SeqDense));
  PetscCall(PetscObjectComposeFunction((PetscObject)A,"MatProductSetFromOptions_seqaijviennacl_seqaij_C",MatProductSetFromOptions_SeqAIJ));

  B->offloadmask = PETSC_OFFLOAD_UNALLOCATED;

  /* If the source matrix is already assembled, copy the destination matrix to the GPU */
  if (B->assembled) PetscCall(MatViennaCLCopyToGPU(B));
  PetscFunctionReturn(0);
}

/*MC
   MATSEQAIJVIENNACL - MATAIJVIENNACL = "aijviennacl" = "seqaijviennacl" - A matrix type to be used for sparse matrices.

   A matrix type type whose data resides on GPUs. These matrices are in CSR format by
   default. All matrix calculations are performed using the ViennaCL library.

   Options Database Keys:
+  -mat_type aijviennacl - sets the matrix type to "seqaijviennacl" during a call to MatSetFromOptions()
.  -mat_viennacl_storage_format csr - sets the storage format of matrices for MatMult during a call to MatSetFromOptions().
-  -mat_viennacl_mult_storage_format csr - sets the storage format of matrices for MatMult during a call to MatSetFromOptions().

  Level: beginner

.seealso: `MatCreateSeqAIJViennaCL()`, `MATAIJVIENNACL`, `MatCreateAIJViennaCL()`
M*/

PETSC_EXTERN PetscErrorCode MatSolverTypeRegister_ViennaCL(void)
{
  PetscFunctionBegin;
  PetscCall(MatSolverTypeRegister(MATSOLVERPETSC, MATSEQAIJVIENNACL, MAT_FACTOR_LU,MatGetFactor_seqaij_petsc));
  PetscCall(MatSolverTypeRegister(MATSOLVERPETSC, MATSEQAIJVIENNACL, MAT_FACTOR_CHOLESKY,MatGetFactor_seqaij_petsc));
  PetscCall(MatSolverTypeRegister(MATSOLVERPETSC, MATSEQAIJVIENNACL, MAT_FACTOR_ILU,MatGetFactor_seqaij_petsc));
  PetscCall(MatSolverTypeRegister(MATSOLVERPETSC, MATSEQAIJVIENNACL, MAT_FACTOR_ICC,MatGetFactor_seqaij_petsc));
  PetscFunctionReturn(0);
}
