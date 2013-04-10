

/*
    Defines the basic matrix operations for the AIJ (compressed row)
  matrix storage format.
*/

#include "petscconf.h"
PETSC_CUDA_EXTERN_C_BEGIN
#include "../src/mat/impls/aij/seq/aij.h"          /*I "petscmat.h" I*/
#include "petscbt.h"
#include "../src/vec/vec/impls/dvecimpl.h"
#include "petsc-private/vecimpl.h"
PETSC_CUDA_EXTERN_C_END
#undef VecType
#include "../src/mat/impls/aij/seq/seqcusp/cuspmatimpl.h"

#if defined(PETSC_HAVE_TXPETSCGPU)
const char *const MatCUSPStorageFormats[] = {"CSR","DIA","ELL","MatCUSPStorageFormat","MAT_CUSP_",0};

/* this is such a hack ... but I haven't written another way to pass this variable
   from one GPU_Matrix_Ifc class to another. This is necessary for the parallel
   SpMV. Essentially, I need to use the same stream variable in two different
   data structures. I do this by creating a single instance of that stream
   and reuse it.*/
cudaStream_t theCUSPBodyStream=0;
#endif

#include <algorithm>
#include <vector>
#include <string>
#include <thrust/sort.h>
#include <thrust/fill.h>

#undef __FUNCT__
#define __FUNCT__ "MatCUSPSetFormat_SeqAIJCUSP"
PetscErrorCode MatCUSPSetFormat_SeqAIJCUSP(Mat A,MatCUSPFormatOperation op,MatCUSPStorageFormat format)
{
#if defined PETSC_HAVE_TXPETSCGPU
  Mat_SeqAIJCUSP *cuspMat = (Mat_SeqAIJCUSP*)A->spptr;
#endif

  PetscFunctionBegin;
#if defined PETSC_HAVE_TXPETSCGPU
  switch (op) {
  case MAT_CUSP_MULT:
    cuspMat->format = format;
    break;
  case MAT_CUSP_ALL:
    cuspMat->format = format;
    break;
  default:
    SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_SUP,"unsupported operation %d for MatCUSPFormatOperation. Only MAT_CUSP_MULT and MAT_CUSP_ALL are currently supported.",op);
  }
#endif
  PetscFunctionReturn(0);
}

/*@
   MatCUSPSetFormat - Sets the storage format of CUSP matrices for a particular
   operation. Only the MatMult operation can use different GPU storage formats
   for AIJCUSP matrices. This requires the txpetscgpu package. Use --download-txpetscgpu
   to build/install PETSc to use these capabilities. If txpetscgpu is not enabled,
   this function simply passes through leaving the matrix in the original CSR format
   for usage on the GPU.

   Not Collective

   Input Parameters:
+  A - Matrix of type SEQAIJCUSP
.  op - MatCUSPFormatOperation. SEQAIJCUSP matrices support MAT_CUSP_MULT and MAT_CUSP_ALL. MPIAIJCUSP matrices support MAT_CUSP_MULT_DIAG, MAT_CUSP_MULT_OFFDIAG, and MAT_CUSP_ALL.
-  format - MatCUSPStorageFormat (one of MAT_CUSP_CSR, MAT_CUSP_DIA, MAT_CUSP_ELL)

   Output Parameter:

   Level: intermediate

.seealso: MatCUSPStorageFormat, MatCUSPFormatOperation
@*/
#undef __FUNCT__
#define __FUNCT__ "MatCUSPSetFormat"
PetscErrorCode MatCUSPSetFormat(Mat A,MatCUSPFormatOperation op,MatCUSPStorageFormat format)
{
#if defined PETSC_HAVE_TXPETSCGPU
  PetscErrorCode ierr;
#endif

  PetscFunctionBegin;
  PetscValidHeaderSpecific(A, MAT_CLASSID,1);
#if defined PETSC_HAVE_TXPETSCGPU
  ierr = PetscTryMethod(A, "MatCUSPSetFormat_C",(Mat,MatCUSPFormatOperation,MatCUSPStorageFormat),(A,op,format));CHKERRQ(ierr);
#endif
  PetscFunctionReturn(0);
}

#if defined PETSC_HAVE_TXPETSCGPU
#undef __FUNCT__
#define __FUNCT__ "MatSetFromOptions_SeqAIJCUSP"
PetscErrorCode MatSetFromOptions_SeqAIJCUSP(Mat A)
{
  PetscErrorCode       ierr;
  MatCUSPStorageFormat format;
  PetscBool            flg;

  PetscFunctionBegin;
  ierr = PetscOptionsHead("SeqAIJCUSP options");CHKERRQ(ierr);
  ierr = PetscObjectOptionsBegin((PetscObject)A);
  ierr = PetscOptionsEnum("-mat_cusp_mult_storage_format","sets storage format of (seq)aijcusp gpu matrices for SpMV",
                          "MatCUSPSetFormat",MatCUSPStorageFormats,(PetscEnum)MAT_CUSP_CSR,(PetscEnum*)&format,&flg);CHKERRQ(ierr);
  if (flg) {
    ierr = MatCUSPSetFormat(A,MAT_CUSP_MULT,format);CHKERRQ(ierr);
  }
  ierr = PetscOptionsEnum("-mat_cusp_storage_format","sets storage format of (seq)aijcusp gpu matrices for SpMV",
                          "MatCUSPSetFormat",MatCUSPStorageFormats,(PetscEnum)MAT_CUSP_CSR,(PetscEnum*)&format,&flg);CHKERRQ(ierr);
  if (flg) {
    ierr = MatCUSPSetFormat(A,MAT_CUSP_ALL,format);CHKERRQ(ierr);
  }
  ierr = PetscOptionsEnd();CHKERRQ(ierr);
  PetscFunctionReturn(0);

}
#endif

#undef __FUNCT__
#define __FUNCT__ "MatCUSPCopyToGPU"
PetscErrorCode MatCUSPCopyToGPU(Mat A)
{

  Mat_SeqAIJCUSP *cuspstruct = (Mat_SeqAIJCUSP*)A->spptr;
  Mat_SeqAIJ     *a          = (Mat_SeqAIJ*)A->data;
  PetscInt       m           = A->rmap->n,*ii,*ridx;
  PetscErrorCode ierr;


  PetscFunctionBegin;
  if (A->valid_GPU_matrix == PETSC_CUSP_UNALLOCATED || A->valid_GPU_matrix == PETSC_CUSP_CPU) {
    ierr = PetscLogEventBegin(MAT_CUSPCopyToGPU,A,0,0,0);CHKERRQ(ierr);
    /*
      It may be possible to reuse nonzero structure with new matrix values but
      for simplicity and insured correctness we delete and build a new matrix on
      the GPU. Likely a very small performance hit.
    */
    if (cuspstruct->mat) {
      try {
        delete cuspstruct->mat;
        if (cuspstruct->tempvec) delete cuspstruct->tempvec;

      } catch(char *ex) {
        SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"CUSP error: %s", ex);
      }
    }
    try {
      cuspstruct->nonzerorow=0;
      for (int j = 0; j<m; j++) cuspstruct->nonzerorow += ((a->i[j+1]-a->i[j])>0);

#if defined(PETSC_HAVE_TXPETSCGPU)
      if (a->compressedrow.use) {
        m    = a->compressedrow.nrows;
        ii   = a->compressedrow.i;
        ridx = a->compressedrow.rindex;
      } else {
        /* Forcing compressed row on the GPU ... only relevant for CSR storage */
        int k=0;
        ierr = PetscMalloc((cuspstruct->nonzerorow+1)*sizeof(PetscInt), &ii);CHKERRQ(ierr);
        ierr = PetscMalloc((cuspstruct->nonzerorow)*sizeof(PetscInt), &ridx);CHKERRQ(ierr);
        ii[0]=0;
        for (int j = 0; j<m; j++) {
          if ((a->i[j+1]-a->i[j])>0) {
            ii[k]  = a->i[j];
            ridx[k]= j;
            k++;
          }
        }
        ii[cuspstruct->nonzerorow] = a->nz;

        m = cuspstruct->nonzerorow;
      }

      /* Build our matrix ... first determine the GPU storage type */
      cuspstruct->mat = GPU_Matrix_Factory::getNew(MatCUSPStorageFormats[cuspstruct->format]);

      /* Create the streams and events (if desired). */
      PetscMPIInt size;
      ierr = MPI_Comm_size(PETSC_COMM_WORLD,&size);CHKERRQ(ierr);
      ierr = cuspstruct->mat->buildStreamsAndEvents(size, &theCUSPBodyStream);CHKERRCUSP(ierr);

      /* lastly, build the matrix */
      ierr = cuspstruct->mat->setMatrix(m, A->cmap->n, a->nz, ii, a->j, a->a);;CHKERRCUSP(ierr);
      cuspstruct->mat->setCPRowIndices(ridx, m);

      /*
        INODES : Determine the inode data structure for the GPU.
        This only really matters for the CSR format.
      */
      if (a->inode.use) {
        PetscInt * temp;
        ierr   = PetscMalloc((a->inode.node_count+1)*sizeof(PetscInt), &temp);CHKERRQ(ierr);
        temp[0]=0;
        PetscInt nodeMax=0, nnzPerRowMax=0;
        for (int i = 0; i<a->inode.node_count; i++) {
          temp[i+1]= a->inode.size[i]+temp[i];
          if (a->inode.size[i] > nodeMax) nodeMax = a->inode.size[i];
        }
        /* Determine the maximum number of nonzeros in a row. */
        cuspstruct->nonzerorow = 0;
        for (int j = 0; j<A->rmap->n; j++) {
          cuspstruct->nonzerorow += ((a->i[j+1]-a->i[j])>0);
          if (a->i[j+1]-a->i[j] > nnzPerRowMax) nnzPerRowMax = a->i[j+1]-a->i[j];
        }
        /* Set the Inode data ... only relevant for CSR really */
        cuspstruct->mat->setInodeData(temp, a->inode.node_count+1, nnzPerRowMax, nodeMax, a->inode.node_count);
        ierr = PetscFree(temp);CHKERRQ(ierr);
      }
      if (!a->compressedrow.use) {
        ierr = PetscFree(ii);CHKERRQ(ierr);
        ierr = PetscFree(ridx);CHKERRQ(ierr);
      }

#else

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
#endif
      cuspstruct->tempvec = new CUSPARRAY;
      cuspstruct->tempvec->resize(m);
    } catch(char *ex) {
      SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"CUSP error: %s", ex);
    }
    ierr = WaitForGPU();CHKERRCUSP(ierr);

    A->valid_GPU_matrix = PETSC_CUSP_BOTH;

    ierr = PetscLogEventEnd(MAT_CUSPCopyToGPU,A,0,0,0);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatCUSPCopyFromGPU"
PetscErrorCode MatCUSPCopyFromGPU(Mat A, CUSPMATRIX *Agpu)
{
  Mat_SeqAIJCUSP *cuspstruct = (Mat_SeqAIJCUSP*) A->spptr;
  Mat_SeqAIJ     *a          = (Mat_SeqAIJ*) A->data;
  PetscInt       m           = A->rmap->n;
  PetscErrorCode ierr;

#if defined(PETSC_HAVE_TXPETSCGPU)
  CUSPMATRIX *mat;
  ierr = cuspstruct->mat->getCsrMatrix(&mat);CHKERRCUSP(ierr);
#else
  CUSPMATRIX *mat = (CUSPMATRIX*)cuspstruct->mat;
#endif

  PetscFunctionBegin;
  if (A->valid_GPU_matrix == PETSC_CUSP_UNALLOCATED) {
    if (A->valid_GPU_matrix == PETSC_CUSP_UNALLOCATED) {
      try {
        mat = Agpu;
        if (a->compressedrow.use) {
          SETERRQ(PETSC_COMM_WORLD, PETSC_ERR_ARG_WRONG, "Cannot handle row compression for GPU matrices");
        } else {
          PetscInt i;

          if (m+1 != (PetscInt) mat->row_offsets.size()) SETERRQ2(PETSC_COMM_WORLD, PETSC_ERR_ARG_SIZ, "GPU matrix has %d rows, should be %d", mat->row_offsets.size()-1, m);
          a->nz           = mat->values.size();
          a->maxnz        = a->nz; /* Since we allocate exactly the right amount */
          A->preallocated = PETSC_TRUE;
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
          thrust::copy(mat->row_offsets.begin(), mat->row_offsets.end(), a->i);
          thrust::copy(mat->column_indices.begin(), mat->column_indices.end(), a->j);
          thrust::copy(mat->values.begin(), mat->values.end(), a->a);
          /* Setup row lengths */
          if (a->imax) {ierr = PetscFree2(a->imax,a->ilen);CHKERRQ(ierr);}
          ierr = PetscMalloc2(m,PetscInt,&a->imax,m,PetscInt,&a->ilen);CHKERRQ(ierr);
          ierr = PetscLogObjectMemory(A, 2*m*sizeof(PetscInt));CHKERRQ(ierr);
          for (i = 0; i < m; ++i) a->imax[i] = a->ilen[i] = a->i[i+1] - a->i[i];
          /* a->diag?*/
        }
        cuspstruct->tempvec = new CUSPARRAY;
        cuspstruct->tempvec->resize(m);
      } catch(char *ex) {
        SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_LIB, "CUSP error: %s", ex);
      }
    }
    /* This assembly prevents resetting the flag to PETSC_CUSP_CPU and recopying */
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
    ierr = VecCreate(PetscObjectComm((PetscObject)mat),right);CHKERRQ(ierr);
    ierr = VecSetSizes(*right,mat->cmap->n,PETSC_DETERMINE);CHKERRQ(ierr);
    ierr = VecSetBlockSize(*right,mat->rmap->bs);CHKERRQ(ierr);
    ierr = VecSetType(*right,VECSEQCUSP);CHKERRQ(ierr);
    ierr = PetscLayoutReference(mat->cmap,&(*right)->map);CHKERRQ(ierr);
  }
  if (left) {
    ierr = VecCreate(PetscObjectComm((PetscObject)mat),left);CHKERRQ(ierr);
    ierr = VecSetSizes(*left,mat->rmap->n,PETSC_DETERMINE);CHKERRQ(ierr);
    ierr = VecSetBlockSize(*left,mat->rmap->bs);CHKERRQ(ierr);
    ierr = VecSetType(*left,VECSEQCUSP);CHKERRQ(ierr);
    ierr = PetscLayoutReference(mat->rmap,&(*left)->map);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatMult_SeqAIJCUSP"
PetscErrorCode MatMult_SeqAIJCUSP(Mat A,Vec xx,Vec yy)
{
  Mat_SeqAIJ     *a = (Mat_SeqAIJ*)A->data;
  PetscErrorCode ierr;
  Mat_SeqAIJCUSP *cuspstruct = (Mat_SeqAIJCUSP*)A->spptr;
#if !defined(PETSC_HAVE_TXPETSCGPU)
  PetscBool usecprow = a->compressedrow.use;
#endif
  CUSPARRAY *xarray=NULL,*yarray=NULL;

  PetscFunctionBegin;
  /* The line below should not be necessary as it has been moved to MatAssemblyEnd_SeqAIJCUSP
     ierr = MatCUSPCopyToGPU(A);CHKERRQ(ierr); */
  ierr = VecCUSPGetArrayRead(xx,&xarray);CHKERRQ(ierr);
  ierr = VecCUSPGetArrayWrite(yy,&yarray);CHKERRQ(ierr);
  try {
#if defined(PETSC_HAVE_TXPETSCGPU)
    ierr = cuspstruct->mat->multiply(xarray, yarray);CHKERRCUSP(ierr);
#else
    if (usecprow) { /* use compressed row format */
      cusp::multiply(*cuspstruct->mat,*xarray,*cuspstruct->tempvec);
      ierr = VecSet_SeqCUSP(yy,0.0);CHKERRQ(ierr);
      thrust::copy(cuspstruct->tempvec->begin(),cuspstruct->tempvec->end(),thrust::make_permutation_iterator(yarray->begin(),cuspstruct->indices->begin()));
    } else { /* do not use compressed row format */
      cusp::multiply(*cuspstruct->mat,*xarray,*yarray);
    }
#endif

  } catch (char *ex) {
    SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"CUSP error: %s", ex);
  }
  ierr = VecCUSPRestoreArrayRead(xx,&xarray);CHKERRQ(ierr);
  ierr = VecCUSPRestoreArrayWrite(yy,&yarray);CHKERRQ(ierr);
#if defined(PETSC_HAVE_TXPETSCGPU)
  if (!cuspstruct->mat->hasNonZeroStream()) {
    ierr = WaitForGPU();CHKERRCUSP(ierr);
  }
#else
  ierr = WaitForGPU();CHKERRCUSP(ierr);
#endif
  ierr = PetscLogFlops(2.0*a->nz - cuspstruct->nonzerorow);CHKERRQ(ierr);
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
  Mat_SeqAIJCUSP *cuspstruct = (Mat_SeqAIJCUSP*)A->spptr;
  CUSPARRAY      *xarray     = NULL,*yarray=NULL,*zarray=NULL;

  PetscFunctionBegin;
  /* The line below should not be necessary as it has been moved to MatAssemblyEnd_SeqAIJCUSP
     ierr = MatCUSPCopyToGPU(A);CHKERRQ(ierr); */
  try {
    ierr = VecCopy_SeqCUSP(yy,zz);CHKERRQ(ierr);
    ierr = VecCUSPGetArrayRead(xx,&xarray);CHKERRQ(ierr);
    ierr = VecCUSPGetArrayRead(yy,&yarray);CHKERRQ(ierr);
    ierr = VecCUSPGetArrayWrite(zz,&zarray);CHKERRQ(ierr);
#if defined(PETSC_HAVE_TXPETSCGPU)
    ierr = cuspstruct->mat->multiplyAdd(xarray, zarray);CHKERRCUSP(ierr);
#else
    if (a->compressedrow.use) {
      cusp::multiply(*cuspstruct->mat,*xarray, *cuspstruct->tempvec);
      thrust::for_each(
        thrust::make_zip_iterator(
          thrust::make_tuple(cuspstruct->tempvec->begin(),
                             thrust::make_permutation_iterator(zarray->begin(), cuspstruct->indices->begin()))),
        thrust::make_zip_iterator(
          thrust::make_tuple(cuspstruct->tempvec->begin(),
                             thrust::make_permutation_iterator(zarray->begin(),cuspstruct->indices->begin()))) + cuspstruct->tempvec->size(),
        VecCUSPPlusEquals());
    } else {
      cusp::multiply(*cuspstruct->mat,*xarray,*cuspstruct->tempvec);
      thrust::for_each(
        thrust::make_zip_iterator(thrust::make_tuple(
                                    cuspstruct->tempvec->begin(),
                                    zarray->begin())),
        thrust::make_zip_iterator(thrust::make_tuple(
                                    cuspstruct->tempvec->end(),
                                    zarray->end())),
        VecCUSPPlusEquals());
    }
#endif
    ierr = VecCUSPRestoreArrayRead(xx,&xarray);CHKERRQ(ierr);
    ierr = VecCUSPRestoreArrayRead(yy,&yarray);CHKERRQ(ierr);
    ierr = VecCUSPRestoreArrayWrite(zz,&zarray);CHKERRQ(ierr);

  } catch(char *ex) {
    SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"CUSP error: %s", ex);
  }
  ierr = WaitForGPU();CHKERRCUSP(ierr);
  ierr = PetscLogFlops(2.0*a->nz);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatAssemblyEnd_SeqAIJCUSP"
PetscErrorCode MatAssemblyEnd_SeqAIJCUSP(Mat A,MatAssemblyType mode)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatAssemblyEnd_SeqAIJ(A,mode);CHKERRQ(ierr);
  ierr = MatCUSPCopyToGPU(A);CHKERRQ(ierr);
  if (mode == MAT_FLUSH_ASSEMBLY) PetscFunctionReturn(0);
  A->ops->mult    = MatMult_SeqAIJCUSP;
  A->ops->multadd = MatMultAdd_SeqAIJCUSP;
  PetscFunctionReturn(0);
}

/* --------------------------------------------------------------------------------*/
#undef __FUNCT__
#define __FUNCT__ "MatCreateSeqAIJCUSP"
/*@
   MatCreateSeqAIJCUSP - Creates a sparse matrix in AIJ (compressed row) format
   (the default parallel PETSc format).  This matrix will ultimately pushed down
   to NVidia GPUs and use the CUSP library for calculations. For good matrix
   assembly performance the user should preallocate the matrix storage by setting
   the parameter nz (or the array nnz).  By setting these parameters accurately,
   performance during matrix assembly can be increased by more than a factor of 50.


   Collective on MPI_Comm

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

   By default, this format uses inodes (identical nodes) when possible, to
   improve numerical efficiency of matrix-vector products and solves. We
   search for consecutive rows with the same nonzero structure, thereby
   reusing matrix information to achieve increased efficiency.

   Level: intermediate

.seealso: MatCreate(), MatCreateAIJ(), MatSetValues(), MatSeqAIJSetColumnIndices(), MatCreateSeqAIJWithArrays(), MatCreateAIJ()

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

#if defined(PETSC_HAVE_TXPETSCGPU)

#undef __FUNCT__
#define __FUNCT__ "MatDestroy_SeqAIJCUSP"
PetscErrorCode MatDestroy_SeqAIJCUSP(Mat A)
{
  PetscErrorCode ierr;
  Mat_SeqAIJCUSP *cuspstruct = (Mat_SeqAIJCUSP*)A->spptr;

  PetscFunctionBegin;
  if (A->factortype==MAT_FACTOR_NONE) {
    try {
      if (A->valid_GPU_matrix != PETSC_CUSP_UNALLOCATED) {
        delete (GPU_Matrix_Ifc*)(cuspstruct->mat);
      }
      if (cuspstruct->tempvec!=0) delete cuspstruct->tempvec;
      delete cuspstruct;
      A->valid_GPU_matrix = PETSC_CUSP_UNALLOCATED;
    } catch(char *ex) {
      SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"CUSP error: %s", ex);
    }
  }

  /*this next line is because MatDestroy tries to PetscFree spptr if it is not zero, and PetscFree only works if the memory was allocated with PetscNew or PetscMalloc, which don't call the constructor */
  A->spptr = 0;

  ierr = MatDestroy_SeqAIJ(A);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#else

#undef __FUNCT__
#define __FUNCT__ "MatDestroy_SeqAIJCUSP"
PetscErrorCode MatDestroy_SeqAIJCUSP(Mat A)
{
  PetscErrorCode ierr;
  Mat_SeqAIJCUSP *cuspcontainer = (Mat_SeqAIJCUSP*)A->spptr;

  PetscFunctionBegin;
  try {
    if (A->valid_GPU_matrix != PETSC_CUSP_UNALLOCATED) delete (CUSPMATRIX*)(cuspcontainer->mat);
    delete cuspcontainer;
    A->valid_GPU_matrix = PETSC_CUSP_UNALLOCATED;
  } catch(char *ex) {
    SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"CUSP error: %s", ex);
  }
  /*this next line is because MatDestroy tries to PetscFree spptr if it is not zero, and PetscFree only works if the memory was allocated with PetscNew or PetscMalloc, which don't call the constructor */
  A->spptr = 0;
  ierr     = MatDestroy_SeqAIJ(A);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#endif

/*
#undef __FUNCT__
#define __FUNCT__ "MatCreateSeqAIJCUSPFromTriple"
PetscErrorCode MatCreateSeqAIJCUSPFromTriple(MPI_Comm comm, PetscInt m, PetscInt n, PetscInt* i, PetscInt* j, PetscScalar*a, Mat *mat, PetscInt nz, PetscBool idx)
{
  CUSPMATRIX *gpucsr;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (idx) {
  }
  PetscFunctionReturn(0);
}*/

extern PetscErrorCode MatSetValuesBatch_SeqAIJCUSP(Mat, PetscInt, PetscInt, PetscInt*,const PetscScalar*);

#if defined(PETSC_HAVE_TXPETSCGPU)
PETSC_EXTERN PetscErrorCode MatGetFactor_seqaij_cusparse(Mat,MatFactorType,Mat*);
extern PetscErrorCode MatFactorGetSolverPackage_seqaij_cusparse(Mat,const MatSolverPackage*);
#endif

#if defined(PETSC_HAVE_TXPETSCGPU)

#undef __FUNCT__
#define __FUNCT__ "MatCreate_SeqAIJCUSP"
PETSC_EXTERN PetscErrorCode MatCreate_SeqAIJCUSP(Mat B)
{
  PetscErrorCode       ierr;
  MatCUSPStorageFormat format = MAT_CUSP_CSR;

  PetscFunctionBegin;
  ierr            = MatCreate_SeqAIJ(B);CHKERRQ(ierr);
  B->ops->mult    = MatMult_SeqAIJCUSP;
  B->ops->multadd = MatMultAdd_SeqAIJCUSP;

  if (B->factortype==MAT_FACTOR_NONE) {
    /* you cannot check the inode.use flag here since the matrix was just created.*/
    B->spptr                             = new Mat_SeqAIJCUSP;
    ((Mat_SeqAIJCUSP*)B->spptr)->mat     = 0;
    ((Mat_SeqAIJCUSP*)B->spptr)->tempvec = 0;
    ((Mat_SeqAIJCUSP*)B->spptr)->format  = format;
  }
  B->ops->assemblyend    = MatAssemblyEnd_SeqAIJCUSP;
  B->ops->destroy        = MatDestroy_SeqAIJCUSP;
  B->ops->getvecs        = MatGetVecs_SeqAIJCUSP;
  B->ops->setvaluesbatch = MatSetValuesBatch_SeqAIJCUSP;
  B->ops->setfromoptions = MatSetFromOptions_SeqAIJCUSP;

  ierr = PetscObjectChangeTypeName((PetscObject)B,MATSEQAIJCUSP);CHKERRQ(ierr);

  B->valid_GPU_matrix = PETSC_CUSP_UNALLOCATED;
  /* Here we overload MatGetFactor_cusparse_C which enables -pc_factor_mat_solver_package cusparse to work with
     -mat_type aijcusp. That is, an aijcusp matrix can call the cusparse tri solve.
     Note the difference with the implementation in MatCreate_SeqAIJCUSPARSE in ../seqcusparse/aijcusparse.cu */
  ierr = PetscObjectComposeFunction((PetscObject)B,"MatGetFactor_cusparse_C",MatGetFactor_seqaij_cusparse);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)B,"MatCUSPSetFormat_C", MatCUSPSetFormat_SeqAIJCUSP);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#else

#undef __FUNCT__
#define __FUNCT__ "MatCreate_SeqAIJCUSP"
PETSC_EXTERN PetscErrorCode MatCreate_SeqAIJCUSP(Mat B)
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

  ((Mat_SeqAIJCUSP*)B->spptr)->mat     = 0;
  ((Mat_SeqAIJCUSP*)B->spptr)->tempvec = 0;
  ((Mat_SeqAIJCUSP*)B->spptr)->indices = 0;

  B->ops->assemblyend    = MatAssemblyEnd_SeqAIJCUSP;
  B->ops->destroy        = MatDestroy_SeqAIJCUSP;
  B->ops->getvecs        = MatGetVecs_SeqAIJCUSP;
  B->ops->setvaluesbatch = MatSetValuesBatch_SeqAIJCUSP;

  ierr = PetscObjectComposeFunction((PetscObject)B,"MatCUSPSetFormat_C", MatCUSPSetFormat_SeqAIJCUSP);CHKERRQ(ierr);
  ierr = PetscObjectChangeTypeName((PetscObject)B,MATSEQAIJCUSP);CHKERRQ(ierr);

  B->valid_GPU_matrix = PETSC_CUSP_UNALLOCATED;
  PetscFunctionReturn(0);
}

#endif

/*M
   MATSEQAIJCUSP - MATAIJCUSP = "aijcusp" = "seqaijcusp" - A matrix type to be used for sparse matrices.

   A matrix type type whose data resides on Nvidia GPUs. These matrices are in CSR format by
   default. All matrix calculations are performed using the CUSP library.
   DIA and ELL formats are ONLY available when using the 'txpetscgpu' package.
   Use --download-txpetscgpu to build/install PETSc to use different GPU storage formats
   with CUSP matrix types.

   Options Database Keys:
+  -mat_type aijcusp - sets the matrix type to "seqaijcusp" during a call to MatSetFromOptions()
.  -mat_cusp_storage_format csr - sets the storage format of matrices for MatMult during a call to MatSetFromOptions(). Other options include dia (diagonal) or ell (ellpack). dia and ell only available with 'txpetscgpu' package.
-  -mat_cusp_mult_storage_format csr - sets the storage format of matrices for MatMult during a call to MatSetFromOptions(). Other options include dia (diagonal) or ell (ellpack). dia and ell only available with 'txpetscgpu' package.

  Level: beginner

.seealso: MatCreateSeqAIJCUSP(), MATAIJCUSP, MatCreateAIJCUSP(), MatCUSPSetFormat(), MatCUSPStorageFormat, MatCUSPFormatOperation
M*/

