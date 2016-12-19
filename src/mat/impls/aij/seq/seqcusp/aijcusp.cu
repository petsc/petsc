/*
  Defines the basic matrix operations for the AIJ (compressed row)
  matrix storage format.
*/
#define PETSC_SKIP_COMPLEX
#define PETSC_SKIP_SPINLOCK

#include <petscconf.h>
#include <../src/mat/impls/aij/seq/aij.h>          /*I "petscmat.h" I*/
#include <petscbt.h>
#include <../src/vec/vec/impls/dvecimpl.h>
#include <petsc/private/vecimpl.h>
#undef VecType
#include <../src/mat/impls/aij/seq/seqcusp/cuspmatimpl.h>
#include <../src/vec/vec/impls/seq/seqcusp/cuspvecimpl.h>

const char *const MatCUSPStorageFormats[] = {"CSR","DIA","ELL","MatCUSPStorageFormat","MAT_CUSP_",0};

PETSC_INTERN PetscErrorCode MatGetFactor_seqaij_petsc(Mat,MatFactorType,Mat*);

PetscErrorCode MatCUSPSetStream(Mat A,const cudaStream_t stream)
{
  Mat_SeqAIJCUSP *cuspstruct = (Mat_SeqAIJCUSP*)A->spptr;

  PetscFunctionBegin;
  cuspstruct->stream = stream;
  PetscFunctionReturn(0);
}

PetscErrorCode MatCUSPSetFormat_SeqAIJCUSP(Mat A,MatCUSPFormatOperation op,MatCUSPStorageFormat format)
{
  Mat_SeqAIJCUSP *cuspMat = (Mat_SeqAIJCUSP*)A->spptr;

  PetscFunctionBegin;
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
  PetscFunctionReturn(0);
}

/*@
   MatCUSPSetFormat - Sets the storage format of CUSP matrices for a particular
   operation. Only the MatMult operation can use different GPU storage formats
   for AIJCUSP matrices.

   Not Collective

   Input Parameters:
+  A - Matrix of type SEQAIJCUSP
.  op - MatCUSPFormatOperation. SEQAIJCUSP matrices support MAT_CUSP_MULT and MAT_CUSP_ALL. MPIAIJCUSP matrices support MAT_CUSP_MULT_DIAG, MAT_CUSP_MULT_OFFDIAG, and MAT_CUSP_ALL.
-  format - MatCUSPStorageFormat (one of MAT_CUSP_CSR, MAT_CUSP_DIA, MAT_CUSP_ELL)

   Output Parameter:

   Level: intermediate

.seealso: MatCUSPStorageFormat, MatCUSPFormatOperation
@*/
PetscErrorCode MatCUSPSetFormat(Mat A,MatCUSPFormatOperation op,MatCUSPStorageFormat format)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(A, MAT_CLASSID,1);
  ierr = PetscTryMethod(A, "MatCUSPSetFormat_C",(Mat,MatCUSPFormatOperation,MatCUSPStorageFormat),(A,op,format));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode MatSetFromOptions_SeqAIJCUSP(PetscOptionItems *PetscOptionsObject,Mat A)
{
  Mat_SeqAIJCUSP       *cuspMat = (Mat_SeqAIJCUSP*)A->spptr;
  PetscErrorCode       ierr;
  MatCUSPStorageFormat format;
  PetscBool            flg;

  PetscFunctionBegin;
  ierr = PetscOptionsHead(PetscOptionsObject,"SeqAIJCUSP options");CHKERRQ(ierr);
  ierr = PetscObjectOptionsBegin((PetscObject)A);
  ierr = PetscOptionsEnum("-mat_cusp_mult_storage_format","sets storage format of (seq)aijcusp gpu matrices for SpMV",
                          "MatCUSPSetFormat",MatCUSPStorageFormats,(PetscEnum)cuspMat->format,(PetscEnum*)&format,&flg);CHKERRQ(ierr);
  if (flg) {
    ierr = MatCUSPSetFormat(A,MAT_CUSP_MULT,format);CHKERRQ(ierr);
  }
  ierr = PetscOptionsEnum("-mat_cusp_storage_format","sets storage format of (seq)aijcusp gpu matrices for SpMV",
                          "MatCUSPSetFormat",MatCUSPStorageFormats,(PetscEnum)cuspMat->format,(PetscEnum*)&format,&flg);CHKERRQ(ierr);
  if (flg) {
    ierr = MatCUSPSetFormat(A,MAT_CUSP_ALL,format);CHKERRQ(ierr);
  }
  ierr = PetscOptionsEnd();CHKERRQ(ierr);
  PetscFunctionReturn(0);

}

PetscErrorCode MatCUSPCopyToGPU(Mat A)
{

  Mat_SeqAIJCUSP *cuspstruct = (Mat_SeqAIJCUSP*)A->spptr;
  Mat_SeqAIJ     *a          = (Mat_SeqAIJ*)A->data;
  PetscInt       m           = A->rmap->n,*ii,*ridx;
  CUSPMATRIX     *mat;
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
        if (cuspstruct->format==MAT_CUSP_ELL)
          delete (CUSPMATRIXELL *) cuspstruct->mat;
        else if (cuspstruct->format==MAT_CUSP_DIA)
          delete (CUSPMATRIXDIA *) cuspstruct->mat;
        else
          delete (CUSPMATRIX *) cuspstruct->mat;
      } catch(char *ex) {
        SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"CUSP error: %s", ex);
      }
    }
    try {
      cuspstruct->nonzerorow=0;
      for (int j = 0; j<m; j++) cuspstruct->nonzerorow += ((a->i[j+1]-a->i[j])>0);
      if (a->compressedrow.use) {
        m    = a->compressedrow.nrows;
        ii   = a->compressedrow.i;
        ridx = a->compressedrow.rindex;
      } else {
        /* Forcing compressed row on the GPU */
        int k=0;
        ierr = PetscMalloc1(cuspstruct->nonzerorow+1, &ii);CHKERRQ(ierr);
        ierr = PetscMalloc1(cuspstruct->nonzerorow, &ridx);CHKERRQ(ierr);
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

      /* now build matrix */
      mat = new CUSPMATRIX;
      mat->resize(m,A->cmap->n,a->nz);
      mat->row_offsets.assign(ii,ii+m+1);
      mat->column_indices.assign(a->j,a->j+a->nz);
      mat->values.assign(a->a,a->a+a->nz);

      /* convert to other formats if selected */
      if (a->compressedrow.use || cuspstruct->format==MAT_CUSP_CSR) {
        cuspstruct->mat = mat;
        cuspstruct->format = MAT_CUSP_CSR;
      } else {
        if (cuspstruct->format==MAT_CUSP_ELL) {
          CUSPMATRIXELL *ellMat = new CUSPMATRIXELL(*mat);
          cuspstruct->mat = ellMat;
        } else {
          CUSPMATRIXDIA *diaMat = new CUSPMATRIXDIA(*mat);
          cuspstruct->mat = diaMat;
        }
        delete (CUSPMATRIX*) mat;
      }

      /* assign the compressed row indices */
      if (cuspstruct->indices) delete (CUSPINTARRAYGPU*)cuspstruct->indices;
      cuspstruct->indices = new CUSPINTARRAYGPU;
      cuspstruct->indices->assign(ridx,ridx+m);

      /* free the temporaries */
      if (!a->compressedrow.use) {
        ierr = PetscFree(ii);CHKERRQ(ierr);
        ierr = PetscFree(ridx);CHKERRQ(ierr);
      }
      if (cuspstruct->tempvec) delete (CUSPARRAY*)cuspstruct->tempvec;
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

PetscErrorCode MatCUSPCopyFromGPU(Mat A, CUSPMATRIX *Agpu)
{
  Mat_SeqAIJCUSP *cuspstruct = (Mat_SeqAIJCUSP*) A->spptr;
  Mat_SeqAIJ     *a          = (Mat_SeqAIJ*) A->data;
  PetscInt       m           = A->rmap->n;
  CUSPMATRIX *mat;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  /* if the data is stored in non-CSR format, create a temporary */
  if (cuspstruct->format==MAT_CUSP_ELL) {
    mat = new CUSPMATRIX(*((CUSPMATRIXELL*)cuspstruct->mat));
  } else if (cuspstruct->format==MAT_CUSP_DIA) {
    mat = new CUSPMATRIX(*((CUSPMATRIXDIA*)cuspstruct->mat));
  } else {
    mat = (CUSPMATRIX*) cuspstruct->mat;
  }

  if (A->valid_GPU_matrix == PETSC_CUSP_UNALLOCATED) {
    if (A->valid_GPU_matrix == PETSC_CUSP_UNALLOCATED) {
      try {
        mat = Agpu;
        if (a->compressedrow.use) SETERRQ(PETSC_COMM_WORLD, PETSC_ERR_ARG_WRONG, "Cannot handle row compression for GPU matrices");
        else {
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
          ierr = PetscMalloc3(a->nz,&a->a,a->nz,&a->j,m+1,&a->i);CHKERRQ(ierr);
          ierr = PetscLogObjectMemory((PetscObject)A, a->nz*(sizeof(PetscScalar)+sizeof(PetscInt))+(m+1)*sizeof(PetscInt));CHKERRQ(ierr);

          a->singlemalloc = PETSC_TRUE;
          thrust::copy(mat->row_offsets.begin(), mat->row_offsets.end(), a->i);
          thrust::copy(mat->column_indices.begin(), mat->column_indices.end(), a->j);
          thrust::copy(mat->values.begin(), mat->values.end(), a->a);
          /* Setup row lengths */
          if (a->imax) {ierr = PetscFree2(a->imax,a->ilen);CHKERRQ(ierr);}
          ierr = PetscMalloc2(m,&a->imax,m,&a->ilen);CHKERRQ(ierr);
          ierr = PetscLogObjectMemory((PetscObject)A, 2*m*sizeof(PetscInt));CHKERRQ(ierr);
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

    /* delete the temporary */
    if (cuspstruct->format==MAT_CUSP_ELL || cuspstruct->format==MAT_CUSP_DIA)
      delete (CUSPMATRIX*) mat;
  } else SETERRQ(PETSC_COMM_WORLD, PETSC_ERR_ARG_WRONG, "Only valid for unallocated GPU matrices");
  PetscFunctionReturn(0);
}

PetscErrorCode MatCreateVecs_SeqAIJCUSP(Mat mat, Vec *right, Vec *left)
{
  PetscErrorCode ierr;
  PetscInt rbs,cbs;

  PetscFunctionBegin;
  ierr = MatGetBlockSizes(mat,&rbs,&cbs);CHKERRQ(ierr);
  if (right) {
    ierr = VecCreate(PetscObjectComm((PetscObject)mat),right);CHKERRQ(ierr);
    ierr = VecSetSizes(*right,mat->cmap->n,PETSC_DETERMINE);CHKERRQ(ierr);
    ierr = VecSetBlockSize(*right,cbs);CHKERRQ(ierr);
    ierr = VecSetType(*right,VECSEQCUSP);CHKERRQ(ierr);
    ierr = PetscLayoutReference(mat->cmap,&(*right)->map);CHKERRQ(ierr);
  }
  if (left) {
    ierr = VecCreate(PetscObjectComm((PetscObject)mat),left);CHKERRQ(ierr);
    ierr = VecSetSizes(*left,mat->rmap->n,PETSC_DETERMINE);CHKERRQ(ierr);
    ierr = VecSetBlockSize(*left,rbs);CHKERRQ(ierr);
    ierr = VecSetType(*left,VECSEQCUSP);CHKERRQ(ierr);
    ierr = PetscLayoutReference(mat->rmap,&(*left)->map);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

PetscErrorCode MatMult_SeqAIJCUSP(Mat A,Vec xx,Vec yy)
{
  Mat_SeqAIJ       *a = (Mat_SeqAIJ*)A->data;
  PetscErrorCode   ierr;
  Mat_SeqAIJCUSP   *cuspstruct = (Mat_SeqAIJCUSP*)A->spptr;
  PetscBool        usecprow = a->compressedrow.use;
  CUSPARRAY        *xarray=NULL,*yarray=NULL;
  static PetscBool cite = PETSC_FALSE;

  PetscFunctionBegin;
  /*
    DM 1/28/2014: As of version 0.4.0 cusp does not handle the case of
    zero matrices well.  It produces segfaults on some platforms.
    Therefore we manually check for the case of a zero matrix here.
  */
  if (a->nz == 0) {
    PetscFunctionReturn(0);
  }
  ierr = PetscCitationsRegister("@incollection{msk2013,\n  author = {Victor Minden and Barry F. Smith and Matthew G. Knepley},\n  title = {Preliminary Implementation of {PETSc} Using {GPUs}},\n  booktitle = {GPU Solutions to Multi-scale Problems in Science and Engineering},\n  series = {Lecture Notes in Earth System Sciences},\n  editor = {David A. Yuen and Long Wang and Xuebin Chi and Lennart Johnsson and Wei Ge and Yaolin Shi},\n  publisher = {Springer Berlin Heidelberg},\n  pages = {131--140},\n  year = {2013},\n}\n",&cite);CHKERRQ(ierr);
  ierr = VecCUSPGetArrayRead(xx,&xarray);CHKERRQ(ierr);
  ierr = VecCUSPGetArrayWrite(yy,&yarray);CHKERRQ(ierr);
  try {
    if (usecprow) {
      /* use compressed row format */
      CUSPMATRIX *mat = (CUSPMATRIX*)cuspstruct->mat;
      cusp::multiply(*mat,*xarray,*cuspstruct->tempvec);
      ierr = VecSet_SeqCUSP(yy,0.0);CHKERRQ(ierr);
      thrust::copy(cuspstruct->tempvec->begin(),cuspstruct->tempvec->end(),thrust::make_permutation_iterator(yarray->begin(),cuspstruct->indices->begin()));
    } else {
      /* do not use compressed row format */
      if (cuspstruct->format==MAT_CUSP_ELL) {
        CUSPMATRIXELL *mat = (CUSPMATRIXELL*)cuspstruct->mat;
        cusp::multiply(*mat,*xarray,*yarray);
      } else if (cuspstruct->format==MAT_CUSP_DIA) {
        CUSPMATRIXDIA *mat = (CUSPMATRIXDIA*)cuspstruct->mat;
        cusp::multiply(*mat,*xarray,*yarray);
      } else {
        CUSPMATRIX *mat = (CUSPMATRIX*)cuspstruct->mat;
        cusp::multiply(*mat,*xarray,*yarray);
      }
    }

  } catch (char *ex) {
    SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"CUSP error: %s", ex);
  }
  ierr = VecCUSPRestoreArrayRead(xx,&xarray);CHKERRQ(ierr);
  ierr = VecCUSPRestoreArrayWrite(yy,&yarray);CHKERRQ(ierr);
  if (!cuspstruct->stream) {
    ierr = WaitForGPU();CHKERRCUSP(ierr);
  }
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

PetscErrorCode MatMultAdd_SeqAIJCUSP(Mat A,Vec xx,Vec yy,Vec zz)
{
  Mat_SeqAIJ     *a = (Mat_SeqAIJ*)A->data;
  PetscErrorCode ierr;
  PetscBool      usecprow = a->compressedrow.use;
  Mat_SeqAIJCUSP *cuspstruct = (Mat_SeqAIJCUSP*)A->spptr;
  CUSPARRAY      *xarray     = NULL,*yarray=NULL,*zarray=NULL;

  PetscFunctionBegin;
  /*
    DM 1/28/2014: As of version 0.4.0 cusp does not handle the case of
    zero matrices well.  It produces segfaults on some platforms.
    Therefore we manually check for the case of a zero matrix here.
  */
  if (a->nz == 0) {
    PetscFunctionReturn(0);
  }
  try {
    ierr = VecCopy_SeqCUSP(yy,zz);CHKERRQ(ierr);
    ierr = VecCUSPGetArrayRead(xx,&xarray);CHKERRQ(ierr);
    ierr = VecCUSPGetArrayRead(yy,&yarray);CHKERRQ(ierr);
    ierr = VecCUSPGetArrayWrite(zz,&zarray);CHKERRQ(ierr);

    if (usecprow) {
      /* use compressed row format */
      CUSPMATRIX *mat = (CUSPMATRIX*)cuspstruct->mat;
      cusp::multiply(*mat,*xarray,*cuspstruct->tempvec);
      thrust::for_each(thrust::make_zip_iterator(thrust::make_tuple(cuspstruct->tempvec->begin(),
                                                                    thrust::make_permutation_iterator(zarray->begin(), cuspstruct->indices->begin()))),
                       thrust::make_zip_iterator(thrust::make_tuple(cuspstruct->tempvec->end(),
                                                                    thrust::make_permutation_iterator(zarray->end(),cuspstruct->indices->end()))),
                       VecCUSPPlusEquals());
    } else {

      if (cuspstruct->format==MAT_CUSP_ELL) {
        CUSPMATRIXELL *mat = (CUSPMATRIXELL*)cuspstruct->mat;
        cusp::multiply(*mat,*xarray,*cuspstruct->tempvec);
      } else if (cuspstruct->format==MAT_CUSP_DIA) {
        CUSPMATRIXDIA *mat = (CUSPMATRIXDIA*)cuspstruct->mat;
        cusp::multiply(*mat,*xarray,*cuspstruct->tempvec);
      } else {
        CUSPMATRIX *mat = (CUSPMATRIX*)cuspstruct->mat;
        cusp::multiply(*mat,*xarray,*cuspstruct->tempvec);
      }

      if (zarray->size() == cuspstruct->indices->size()) {
        thrust::for_each(thrust::make_zip_iterator(thrust::make_tuple(cuspstruct->tempvec->begin(),zarray->begin())),
            thrust::make_zip_iterator(thrust::make_tuple(cuspstruct->tempvec->end(),zarray->end())),
            VecCUSPPlusEquals());
      } else {
        thrust::for_each(thrust::make_zip_iterator(thrust::make_tuple(cuspstruct->tempvec->begin(),
                thrust::make_permutation_iterator(zarray->begin(), cuspstruct->indices->begin()))),
            thrust::make_zip_iterator(thrust::make_tuple(cuspstruct->tempvec->end(),
                thrust::make_permutation_iterator(zarray->end(),cuspstruct->indices->end()))),
            VecCUSPPlusEquals());
      }
    }
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

PetscErrorCode MatDestroy_SeqAIJCUSP(Mat A)
{
  PetscErrorCode ierr;
  Mat_SeqAIJCUSP *cuspcontainer = (Mat_SeqAIJCUSP*)A->spptr;

  PetscFunctionBegin;
  try {
    if (A->valid_GPU_matrix != PETSC_CUSP_UNALLOCATED) {
      if (cuspcontainer->format==MAT_CUSP_ELL) {
        delete (CUSPMATRIXELL*)(cuspcontainer->mat);
      } else if (cuspcontainer->format==MAT_CUSP_DIA) {
        delete (CUSPMATRIXDIA*)(cuspcontainer->mat);
      } else {
        delete (CUSPMATRIX*)(cuspcontainer->mat);
      }
      if (cuspcontainer->indices) delete (CUSPINTARRAYGPU*)cuspcontainer->indices;
      if (cuspcontainer->tempvec) delete (CUSPARRAY*)cuspcontainer->tempvec;
    }
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

extern PetscErrorCode MatSetValuesBatch_SeqAIJCUSP(Mat, PetscInt, PetscInt, PetscInt*,const PetscScalar*);

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

  if (B->factortype==MAT_FACTOR_NONE) {
    ((Mat_SeqAIJCUSP*)B->spptr)->mat     = 0;
    ((Mat_SeqAIJCUSP*)B->spptr)->tempvec = 0;
    ((Mat_SeqAIJCUSP*)B->spptr)->indices = 0;
    ((Mat_SeqAIJCUSP*)B->spptr)->nonzerorow = 0;
    ((Mat_SeqAIJCUSP*)B->spptr)->stream = 0;
    ((Mat_SeqAIJCUSP*)B->spptr)->format = MAT_CUSP_CSR;
  }

  B->ops->assemblyend    = MatAssemblyEnd_SeqAIJCUSP;
  B->ops->destroy        = MatDestroy_SeqAIJCUSP;
  B->ops->getvecs        = MatCreateVecs_SeqAIJCUSP;
  B->ops->setvaluesbatch = MatSetValuesBatch_SeqAIJCUSP;
  B->ops->setfromoptions = MatSetFromOptions_SeqAIJCUSP;

  ierr = PetscObjectComposeFunction((PetscObject)B,"MatCUSPSetFormat_C", MatCUSPSetFormat_SeqAIJCUSP);CHKERRQ(ierr);
  ierr = PetscObjectChangeTypeName((PetscObject)B,MATSEQAIJCUSP);CHKERRQ(ierr);

  B->valid_GPU_matrix = PETSC_CUSP_UNALLOCATED;
  PetscFunctionReturn(0);
}

/*M
   MATSEQAIJCUSP - MATAIJCUSP = "aijcusp" = "seqaijcusp" - A matrix type to be used for sparse matrices.

   A matrix type type whose data resides on Nvidia GPUs. These matrices are in CSR format by
   default. All matrix calculations are performed using the CUSP library. DIA and ELL formats are 
   also available.

   Options Database Keys:
+  -mat_type aijcusp - sets the matrix type to "seqaijcusp" during a call to MatSetFromOptions()
.  -mat_cusp_storage_format csr - sets the storage format of matrices for MatMult during a call to MatSetFromOptions(). Other options include dia (diagonal) or ell (ellpack).
-  -mat_cusp_mult_storage_format csr - sets the storage format of matrices for MatMult during a call to MatSetFromOptions(). Other options include dia (diagonal) or ell (ellpack).

  Level: beginner

.seealso: MatCreateSeqAIJCUSP(), MATAIJCUSP, MatCreateAIJCUSP(), MatCUSPSetFormat(), MatCUSPStorageFormat, MatCUSPFormatOperation
M*/
PETSC_EXTERN PetscErrorCode MatSolverPackageRegister_CUSP(void)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatSolverPackageRegister(MATSOLVERPETSC, MATSEQAIJCUSP,    MAT_FACTOR_LU,MatGetFactor_seqaij_petsc);CHKERRQ(ierr);
  ierr = MatSolverPackageRegister(MATSOLVERPETSC, MATSEQAIJCUSP,    MAT_FACTOR_CHOLESKY,MatGetFactor_seqaij_petsc);CHKERRQ(ierr);
  ierr = MatSolverPackageRegister(MATSOLVERPETSC, MATSEQAIJCUSP,    MAT_FACTOR_ILU,MatGetFactor_seqaij_petsc);CHKERRQ(ierr);
  ierr = MatSolverPackageRegister(MATSOLVERPETSC, MATSEQAIJCUSP,    MAT_FACTOR_ICC,MatGetFactor_seqaij_petsc);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

