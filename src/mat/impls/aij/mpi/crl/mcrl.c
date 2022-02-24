
/*
  Defines a matrix-vector product for the MATMPIAIJCRL matrix class.
  This class is derived from the MATMPIAIJ class and retains the
  compressed row storage (aka Yale sparse matrix format) but augments
  it with a column oriented storage that is more efficient for
  matrix vector products on Vector machines.

  CRL stands for constant row length (that is the same number of columns
  is kept (padded with zeros) for each row of the sparse matrix.

   See src/mat/impls/aij/seq/crl/crl.c for the sequential version
*/

#include <../src/mat/impls/aij/mpi/mpiaij.h>
#include <../src/mat/impls/aij/seq/crl/crl.h>

PetscErrorCode MatDestroy_MPIAIJCRL(Mat A)
{
  Mat_AIJCRL     *aijcrl = (Mat_AIJCRL*) A->spptr;

  PetscFunctionBegin;
  if (aijcrl) {
    CHKERRQ(PetscFree2(aijcrl->acols,aijcrl->icols));
    CHKERRQ(VecDestroy(&aijcrl->fwork));
    CHKERRQ(VecDestroy(&aijcrl->xwork));
    CHKERRQ(PetscFree(aijcrl->array));
  }
  CHKERRQ(PetscFree(A->spptr));

  CHKERRQ(PetscObjectChangeTypeName((PetscObject)A, MATMPIAIJ));
  CHKERRQ(MatDestroy_MPIAIJ(A));
  PetscFunctionReturn(0);
}

PetscErrorCode MatMPIAIJCRL_create_aijcrl(Mat A)
{
  Mat_MPIAIJ     *a      = (Mat_MPIAIJ*)(A)->data;
  Mat_SeqAIJ     *Aij    = (Mat_SeqAIJ*)(a->A->data), *Bij = (Mat_SeqAIJ*)(a->B->data);
  Mat_AIJCRL     *aijcrl = (Mat_AIJCRL*) A->spptr;
  PetscInt       m       = A->rmap->n; /* Number of rows in the matrix. */
  PetscInt       nd      = a->A->cmap->n; /* number of columns in diagonal portion */
  PetscInt       *aj     = Aij->j,*bj = Bij->j; /* From the CSR representation; points to the beginning  of each row. */
  PetscInt       i, j,rmax = 0,*icols, *ailen = Aij->ilen, *bilen = Bij->ilen;
  PetscScalar    *aa = Aij->a,*ba = Bij->a,*acols,*array;

  PetscFunctionBegin;
  /* determine the row with the most columns */
  for (i=0; i<m; i++) {
    rmax = PetscMax(rmax,ailen[i]+bilen[i]);
  }
  aijcrl->nz   = Aij->nz+Bij->nz;
  aijcrl->m    = A->rmap->n;
  aijcrl->rmax = rmax;

  CHKERRQ(PetscFree2(aijcrl->acols,aijcrl->icols));
  CHKERRQ(PetscMalloc2(rmax*m,&aijcrl->acols,rmax*m,&aijcrl->icols));
  acols = aijcrl->acols;
  icols = aijcrl->icols;
  for (i=0; i<m; i++) {
    for (j=0; j<ailen[i]; j++) {
      acols[j*m+i] = *aa++;
      icols[j*m+i] = *aj++;
    }
    for (; j<ailen[i]+bilen[i]; j++) {
      acols[j*m+i] = *ba++;
      icols[j*m+i] = nd + *bj++;
    }
    for (; j<rmax; j++) { /* empty column entries */
      acols[j*m+i] = 0.0;
      icols[j*m+i] = (j) ? icols[(j-1)*m+i] : 0;  /* handle case where row is EMPTY */
    }
  }
  CHKERRQ(PetscInfo(A,"Percentage of 0's introduced for vectorized multiply %g\n",1.0-((double)(aijcrl->nz))/((double)(rmax*m))));

  CHKERRQ(PetscFree(aijcrl->array));
  CHKERRQ(PetscMalloc1(a->B->cmap->n+nd,&array));
  /* xwork array is actually B->n+nd long, but we define xwork this length so can copy into it */
  CHKERRQ(VecDestroy(&aijcrl->xwork));
  CHKERRQ(VecCreateMPIWithArray(PetscObjectComm((PetscObject)A),1,nd,PETSC_DECIDE,array,&aijcrl->xwork));
  CHKERRQ(VecDestroy(&aijcrl->fwork));
  CHKERRQ(VecCreateSeqWithArray(PETSC_COMM_SELF,1,a->B->cmap->n,array+nd,&aijcrl->fwork));

  aijcrl->array = array;
  aijcrl->xscat = a->Mvctx;
  PetscFunctionReturn(0);
}

PetscErrorCode MatAssemblyEnd_MPIAIJCRL(Mat A, MatAssemblyType mode)
{
  Mat_MPIAIJ     *a   = (Mat_MPIAIJ*)A->data;
  Mat_SeqAIJ     *Aij = (Mat_SeqAIJ*)(a->A->data), *Bij = (Mat_SeqAIJ*)(a->A->data);

  PetscFunctionBegin;
  Aij->inode.use = PETSC_FALSE;
  Bij->inode.use = PETSC_FALSE;

  CHKERRQ(MatAssemblyEnd_MPIAIJ(A,mode));
  if (mode == MAT_FLUSH_ASSEMBLY) PetscFunctionReturn(0);

  /* Now calculate the permutation and grouping information. */
  CHKERRQ(MatMPIAIJCRL_create_aijcrl(A));
  PetscFunctionReturn(0);
}

extern PetscErrorCode MatMult_AIJCRL(Mat,Vec,Vec);
extern PetscErrorCode MatDuplicate_AIJCRL(Mat,MatDuplicateOption,Mat*);

/* MatConvert_MPIAIJ_MPIAIJCRL converts a MPIAIJ matrix into a
 * MPIAIJCRL matrix.  This routine is called by the MatCreate_MPIAIJCRL()
 * routine, but can also be used to convert an assembled MPIAIJ matrix
 * into a MPIAIJCRL one. */

PETSC_INTERN PetscErrorCode MatConvert_MPIAIJ_MPIAIJCRL(Mat A,MatType type,MatReuse reuse,Mat *newmat)
{
  Mat            B = *newmat;
  Mat_AIJCRL     *aijcrl;

  PetscFunctionBegin;
  if (reuse == MAT_INITIAL_MATRIX) {
    CHKERRQ(MatDuplicate(A,MAT_COPY_VALUES,&B));
  }

  CHKERRQ(PetscNewLog(B,&aijcrl));
  B->spptr = (void*) aijcrl;

  /* Set function pointers for methods that we inherit from AIJ but override. */
  B->ops->duplicate   = MatDuplicate_AIJCRL;
  B->ops->assemblyend = MatAssemblyEnd_MPIAIJCRL;
  B->ops->destroy     = MatDestroy_MPIAIJCRL;
  B->ops->mult        = MatMult_AIJCRL;

  /* If A has already been assembled, compute the permutation. */
  if (A->assembled) {
    CHKERRQ(MatMPIAIJCRL_create_aijcrl(B));
  }
  CHKERRQ(PetscObjectChangeTypeName((PetscObject)B,MATMPIAIJCRL));
  *newmat = B;
  PetscFunctionReturn(0);
}

/*@C
   MatCreateMPIAIJCRL - Creates a sparse matrix of type MPIAIJCRL.
   This type inherits from AIJ, but stores some additional
   information that is used to allow better vectorization of
   the matrix-vector product. At the cost of increased storage, the AIJ formatted
   matrix can be copied to a format in which pieces of the matrix are
   stored in ELLPACK format, allowing the vectorized matrix multiply
   routine to use stride-1 memory accesses.  As with the AIJ type, it is
   important to preallocate matrix storage in order to get good assembly
   performance.

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

   Notes:
   If nnz is given then nz is ignored

   Level: intermediate

.seealso: MatCreate(), MatCreateMPIAIJPERM(), MatSetValues()
@*/
PetscErrorCode  MatCreateMPIAIJCRL(MPI_Comm comm,PetscInt m,PetscInt n,PetscInt nz,const PetscInt nnz[],PetscInt onz,const PetscInt onnz[],Mat *A)
{
  PetscFunctionBegin;
  CHKERRQ(MatCreate(comm,A));
  CHKERRQ(MatSetSizes(*A,m,n,m,n));
  CHKERRQ(MatSetType(*A,MATMPIAIJCRL));
  CHKERRQ(MatMPIAIJSetPreallocation_MPIAIJ(*A,nz,(PetscInt*)nnz,onz,(PetscInt*)onnz));
  PetscFunctionReturn(0);
}

PETSC_EXTERN PetscErrorCode MatCreate_MPIAIJCRL(Mat A)
{
  PetscFunctionBegin;
  CHKERRQ(MatSetType(A,MATMPIAIJ));
  CHKERRQ(MatConvert_MPIAIJ_MPIAIJCRL(A,MATMPIAIJCRL,MAT_INPLACE_MATRIX,&A));
  PetscFunctionReturn(0);
}
