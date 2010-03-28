#define PETSCMAT_DLL

/*
  Defines a matrix-vector product for the MATSEQAIJCRL matrix class.
  This class is derived from the MATSEQAIJ class and retains the 
  compressed row storage (aka Yale sparse matrix format) but augments 
  it with a column oriented storage that is more efficient for 
  matrix vector products on Vector machines.

  CRL stands for constant row length (that is the same number of columns
  is kept (padded with zeros) for each row of the sparse matrix.
*/
#include "../src/mat/impls/aij/seq/crl/crl.h"

#undef __FUNCT__
#define __FUNCT__ "MatDestroy_SeqCRL"
PetscErrorCode MatDestroy_SeqCRL(Mat A)
{
  PetscErrorCode ierr;
  Mat_CRL        *crl = (Mat_CRL *) A->spptr;

  /* Free everything in the Mat_CRL data structure. */
  ierr = PetscFree2(crl->acols,crl->icols);CHKERRQ(ierr);
  ierr = PetscFree(crl);CHKERRQ(ierr);
  A->spptr = 0;

  ierr = PetscObjectChangeTypeName( (PetscObject)A, MATSEQAIJ);CHKERRQ(ierr);
  ierr = MatDestroy_SeqAIJ(A);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode MatDuplicate_CRL(Mat A, MatDuplicateOption op, Mat *M) 
{
  PetscFunctionBegin;
  SETERRQ(PETSC_ERR_SUP,"Cannot duplicate CRL matrices yet");    
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SeqCRL_create_crl"
PetscErrorCode SeqCRL_create_crl(Mat A)
{
  Mat_SeqAIJ     *a = (Mat_SeqAIJ *)(A)->data;
  Mat_CRL        *crl = (Mat_CRL*) A->spptr;
  PetscInt       m = A->rmap->n;  /* Number of rows in the matrix. */
  PetscInt       *aj = a->j;  /* From the CSR representation; points to the beginning  of each row. */
  PetscInt       i, j,rmax = a->rmax,*icols, *ilen = a->ilen;
  MatScalar      *aa = a->a;
  PetscScalar    *acols;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  crl->nz   = a->nz;
  crl->m    = A->rmap->n;
  crl->rmax = rmax;
  ierr = PetscFree2(crl->acols,crl->icols);CHKERRQ(ierr);
  ierr = PetscMalloc2(rmax*m,PetscScalar,&crl->acols,rmax*m,PetscInt,&crl->icols);CHKERRQ(ierr);
  acols = crl->acols;
  icols = crl->icols;
  for (i=0; i<m; i++) {
    for (j=0; j<ilen[i]; j++) {
      acols[j*m+i] = *aa++;
      icols[j*m+i] = *aj++;
    }
    for (;j<rmax; j++) { /* empty column entries */
      acols[j*m+i] = 0.0;
      icols[j*m+i] = (j) ? icols[(j-1)*m+i] : 0;  /* handle case where row is EMPTY */
    }
  }
  ierr = PetscInfo2(A,"Percentage of 0's introduced for vectorized multiply %G. Rmax= %D\n",1.0-((double)a->nz)/((double)(rmax*m)),rmax);
  PetscFunctionReturn(0);
}

extern PetscErrorCode MatAssemblyEnd_SeqAIJ(Mat,MatAssemblyType);

#undef __FUNCT__
#define __FUNCT__ "MatAssemblyEnd_SeqCRL"
PetscErrorCode MatAssemblyEnd_SeqCRL(Mat A, MatAssemblyType mode)
{
  PetscErrorCode ierr;
  Mat_SeqAIJ     *a = (Mat_SeqAIJ*)A->data;

  PetscFunctionBegin;
  a->inode.use = PETSC_FALSE;
  ierr = MatAssemblyEnd_SeqAIJ(A,mode);CHKERRQ(ierr);
  if (mode == MAT_FLUSH_ASSEMBLY) PetscFunctionReturn(0);

  /* Now calculate the permutation and grouping information. */
  ierr = SeqCRL_create_crl(A);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#include "../src/mat/impls/aij/seq/crl/ftn-kernels/fmultcrl.h"

#undef __FUNCT__  
#define __FUNCT__ "MatMult_CRL"
/*
    Shared by both sequential and parallel versions of CRL matrix: MATMPICRL and MATSEQCRL
    - the scatter is used only in the parallel version

*/
PetscErrorCode MatMult_CRL(Mat A,Vec xx,Vec yy)
{
  Mat_CRL        *crl = (Mat_CRL*) A->spptr;
  PetscInt       m = crl->m;  /* Number of rows in the matrix. */
  PetscInt       rmax = crl->rmax,*icols = crl->icols;
  PetscScalar    *acols = crl->acols;
  PetscErrorCode ierr;
  PetscScalar    *x,*y;
#if !defined(PETSC_USE_FORTRAN_KERNEL_MULTCRL)
  PetscInt       i,j,ii;
#endif


#if defined(PETSC_HAVE_PRAGMA_DISJOINT)
#pragma disjoint(*x,*y,*aa)
#endif

  PetscFunctionBegin;
  if (crl->xscat) {
    ierr = VecCopy(xx,crl->xwork);CHKERRQ(ierr);
    /* get remote values needed for local part of multiply */
    ierr = VecScatterBegin(crl->xscat,xx,crl->fwork,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
    ierr = VecScatterEnd(crl->xscat,xx,crl->fwork,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
    xx = crl->xwork;
  };

  ierr = VecGetArray(xx,&x);CHKERRQ(ierr);
  ierr = VecGetArray(yy,&y);CHKERRQ(ierr);

#if defined(PETSC_USE_FORTRAN_KERNEL_MULTCRL)
  fortranmultcrl_(&m,&rmax,x,y,icols,acols);
#else

  /* first column */
  for (j=0; j<m; j++) { 
    y[j] = acols[j]*x[icols[j]];
  }

  /* other columns */
#if defined(PETSC_HAVE_CRAYC)
#pragma _CRI preferstream
#endif
  for (i=1; i<rmax; i++) {
    ii = i*m;
#if defined(PETSC_HAVE_CRAYC)
#pragma _CRI prefervector
#endif
    for (j=0; j<m; j++) { 
      y[j] = y[j] + acols[ii+j]*x[icols[ii+j]];
    }
  }
#if defined(PETSC_HAVE_CRAYC)
#pragma _CRI ivdep
#endif

#endif
  ierr = PetscLogFlops(2.0*crl->nz - m);CHKERRQ(ierr);
  ierr = VecRestoreArray(xx,&x);CHKERRQ(ierr);
  ierr = VecRestoreArray(yy,&y);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


/* MatConvert_SeqAIJ_SeqCRL converts a SeqAIJ matrix into a 
 * SeqCRL matrix.  This routine is called by the MatCreate_SeqCRL() 
 * routine, but can also be used to convert an assembled SeqAIJ matrix 
 * into a SeqCRL one. */
EXTERN_C_BEGIN
#undef __FUNCT__
#define __FUNCT__ "MatConvert_SeqAIJ_SeqCRL"
PetscErrorCode PETSCMAT_DLLEXPORT MatConvert_SeqAIJ_SeqCRL(Mat A,const MatType type,MatReuse reuse,Mat *newmat)
{
  PetscErrorCode ierr;
  Mat            B = *newmat;
  Mat_CRL        *crl;

  PetscFunctionBegin;
  if (reuse == MAT_INITIAL_MATRIX) {
    ierr = MatDuplicate(A,MAT_COPY_VALUES,&B);CHKERRQ(ierr);
  }

  ierr = PetscNewLog(B,Mat_CRL,&crl);CHKERRQ(ierr);
  B->spptr = (void *) crl;

  /* Set function pointers for methods that we inherit from AIJ but override. */
  B->ops->duplicate   = MatDuplicate_CRL;
  B->ops->assemblyend = MatAssemblyEnd_SeqCRL;
  B->ops->destroy     = MatDestroy_SeqCRL;
  B->ops->mult        = MatMult_CRL;

  /* If A has already been assembled, compute the permutation. */
  if (A->assembled) {
    ierr = SeqCRL_create_crl(B);CHKERRQ(ierr);
  }
  ierr = PetscObjectChangeTypeName((PetscObject)B,MATSEQCRL);CHKERRQ(ierr);
  *newmat = B;
  PetscFunctionReturn(0);
}
EXTERN_C_END


#undef __FUNCT__
#define __FUNCT__ "MatCreateSeqCRL"
/*@C
   MatCreateSeqCRL - Creates a sparse matrix of type SEQCRL.
   This type inherits from AIJ, but stores some additional
   information that is used to allow better vectorization of 
   the matrix-vector product. At the cost of increased storage, the AIJ formatted 
   matrix can be copied to a format in which pieces of the matrix are 
   stored in ELLPACK format, allowing the vectorized matrix multiply 
   routine to use stride-1 memory accesses.  As with the AIJ type, it is 
   important to preallocate matrix storage in order to get good assembly 
   performance.
   
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

   Notes:
   If nnz is given then nz is ignored

   Level: intermediate

.keywords: matrix, cray, sparse, parallel

.seealso: MatCreate(), MatCreateMPICSRPERM(), MatSetValues()
@*/
PetscErrorCode PETSCMAT_DLLEXPORT MatCreateSeqCRL(MPI_Comm comm,PetscInt m,PetscInt n,PetscInt nz,const PetscInt nnz[],Mat *A)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatCreate(comm,A);CHKERRQ(ierr);
  ierr = MatSetSizes(*A,m,n,m,n);CHKERRQ(ierr);
  ierr = MatSetType(*A,MATSEQCRL);CHKERRQ(ierr);
  ierr = MatSeqAIJSetPreallocation_SeqAIJ(*A,nz,(PetscInt*)nnz);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


EXTERN_C_BEGIN
#undef __FUNCT__
#define __FUNCT__ "MatCreate_SeqCRL"
PetscErrorCode PETSCMAT_DLLEXPORT MatCreate_SeqCRL(Mat A)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatSetType(A,MATSEQAIJ);CHKERRQ(ierr);
  ierr = MatConvert_SeqAIJ_SeqCRL(A,MATSEQCRL,MAT_REUSE_MATRIX,&A);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
EXTERN_C_END

