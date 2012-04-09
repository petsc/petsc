
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

extern PetscErrorCode MatDestroy_MPIAIJ(Mat);

#undef __FUNCT__
#define __FUNCT__ "MatDestroy_MPIAIJCRL"
PetscErrorCode MatDestroy_MPIAIJCRL(Mat A)
{
  PetscErrorCode ierr;
  Mat_AIJCRL     *aijcrl = (Mat_AIJCRL *) A->spptr;

  /* Free everything in the Mat_AIJCRL data structure. */
  if (aijcrl) {
    ierr = PetscFree2(aijcrl->acols,aijcrl->icols);CHKERRQ(ierr);
    ierr = VecDestroy(&aijcrl->fwork);CHKERRQ(ierr);
    ierr = VecDestroy(&aijcrl->xwork);CHKERRQ(ierr);
    ierr = PetscFree(aijcrl->array);CHKERRQ(ierr);
  }
  ierr = PetscFree(A->spptr);CHKERRQ(ierr);

  ierr = PetscObjectChangeTypeName( (PetscObject)A, MATMPIAIJ);CHKERRQ(ierr);
  ierr = MatDestroy_MPIAIJ(A);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatMPIAIJCRL_create_aijcrl"
PetscErrorCode MatMPIAIJCRL_create_aijcrl(Mat A)
{
  Mat_MPIAIJ     *a = (Mat_MPIAIJ *)(A)->data;
  Mat_SeqAIJ     *Aij = (Mat_SeqAIJ*)(a->A->data), *Bij = (Mat_SeqAIJ*)(a->B->data);
  Mat_AIJCRL     *aijcrl = (Mat_AIJCRL*) A->spptr;
  PetscInt       m = A->rmap->n;  /* Number of rows in the matrix. */
  PetscInt       nd = a->A->cmap->n; /* number of columns in diagonal portion */
  PetscInt       *aj = Aij->j,*bj = Bij->j;  /* From the CSR representation; points to the beginning  of each row. */
  PetscInt       i, j,rmax = 0,*icols, *ailen = Aij->ilen, *bilen = Bij->ilen;
  PetscScalar    *aa = Aij->a,*ba = Bij->a,*acols,*array;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  /* determine the row with the most columns */
  for (i=0; i<m; i++) {
    rmax = PetscMax(rmax,ailen[i]+bilen[i]);
  }
  aijcrl->nz   = Aij->nz+Bij->nz;
  aijcrl->m    = A->rmap->n;
  aijcrl->rmax = rmax;
  ierr = PetscFree2(aijcrl->acols,aijcrl->icols);CHKERRQ(ierr);
  ierr = PetscMalloc2(rmax*m,PetscScalar,&aijcrl->acols,rmax*m,PetscInt,&aijcrl->icols);CHKERRQ(ierr);
  acols = aijcrl->acols;
  icols = aijcrl->icols;
  for (i=0; i<m; i++) {
    for (j=0; j<ailen[i]; j++) {
      acols[j*m+i] = *aa++;
      icols[j*m+i] = *aj++;
    }
    for (;j<ailen[i]+bilen[i]; j++) {
      acols[j*m+i] = *ba++;
      icols[j*m+i] = nd + *bj++;
    }
    for (;j<rmax; j++) { /* empty column entries */
      acols[j*m+i] = 0.0;
      icols[j*m+i] = (j) ? icols[(j-1)*m+i] : 0;  /* handle case where row is EMPTY */
    }
  }
  ierr = PetscInfo1(A,"Percentage of 0's introduced for vectorized multiply %g\n",1.0-((double)(aijcrl->nz))/((double)(rmax*m)));

  ierr = PetscFree(aijcrl->array);CHKERRQ(ierr);
  ierr = PetscMalloc((a->B->cmap->n+nd)*sizeof(PetscScalar),&array);CHKERRQ(ierr);
  /* xwork array is actually B->n+nd long, but we define xwork this length so can copy into it */
  ierr = VecDestroy(&aijcrl->xwork);CHKERRQ(ierr);
  ierr = VecCreateMPIWithArray(((PetscObject)A)->comm,1,nd,PETSC_DECIDE,array,&aijcrl->xwork);CHKERRQ(ierr);
  ierr = VecDestroy(&aijcrl->fwork);CHKERRQ(ierr);
  ierr = VecCreateSeqWithArray(PETSC_COMM_SELF,1,a->B->cmap->n,array+nd,&aijcrl->fwork);CHKERRQ(ierr);
  aijcrl->array = array;
  aijcrl->xscat = a->Mvctx;
  PetscFunctionReturn(0);
}

extern PetscErrorCode MatAssemblyEnd_MPIAIJ(Mat,MatAssemblyType);

#undef __FUNCT__
#define __FUNCT__ "MatAssemblyEnd_MPIAIJCRL"
PetscErrorCode MatAssemblyEnd_MPIAIJCRL(Mat A, MatAssemblyType mode)
{
  PetscErrorCode ierr;
  Mat_MPIAIJ     *a = (Mat_MPIAIJ*)A->data;
  Mat_SeqAIJ     *Aij = (Mat_SeqAIJ*)(a->A->data), *Bij = (Mat_SeqAIJ*)(a->A->data);

  PetscFunctionBegin;
  Aij->inode.use = PETSC_FALSE;
  Bij->inode.use = PETSC_FALSE;
  ierr = MatAssemblyEnd_MPIAIJ(A,mode);CHKERRQ(ierr);
  if (mode == MAT_FLUSH_ASSEMBLY) PetscFunctionReturn(0);

  /* Now calculate the permutation and grouping information. */
  ierr = MatMPIAIJCRL_create_aijcrl(A);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

extern PetscErrorCode MatMult_AIJCRL(Mat,Vec,Vec);
extern PetscErrorCode MatDuplicate_AIJCRL(Mat,MatDuplicateOption,Mat*); 

/* MatConvert_MPIAIJ_MPIAIJCRL converts a MPIAIJ matrix into a 
 * MPIAIJCRL matrix.  This routine is called by the MatCreate_MPIAIJCRL() 
 * routine, but can also be used to convert an assembled MPIAIJ matrix 
 * into a MPIAIJCRL one. */
EXTERN_C_BEGIN
#undef __FUNCT__
#define __FUNCT__ "MatConvert_MPIAIJ_MPIAIJCRL"
PetscErrorCode  MatConvert_MPIAIJ_MPIAIJCRL(Mat A,const MatType type,MatReuse reuse,Mat *newmat)
{
  PetscErrorCode ierr;
  Mat            B = *newmat;
  Mat_AIJCRL     *aijcrl;

  PetscFunctionBegin;
  if (reuse == MAT_INITIAL_MATRIX) {
    ierr = MatDuplicate(A,MAT_COPY_VALUES,&B);CHKERRQ(ierr);
  }

  ierr = PetscNewLog(B,Mat_AIJCRL,&aijcrl);CHKERRQ(ierr);
  B->spptr = (void *) aijcrl;

  /* Set function pointers for methods that we inherit from AIJ but override. */
  B->ops->duplicate   = MatDuplicate_AIJCRL;
  B->ops->assemblyend = MatAssemblyEnd_MPIAIJCRL;
  B->ops->destroy     = MatDestroy_MPIAIJCRL;
  B->ops->mult        = MatMult_AIJCRL;

  /* If A has already been assembled, compute the permutation. */
  if (A->assembled) {
    ierr = MatMPIAIJCRL_create_aijcrl(B);CHKERRQ(ierr);
  }
  ierr = PetscObjectChangeTypeName((PetscObject)B,MATMPIAIJCRL);CHKERRQ(ierr);
  *newmat = B;
  PetscFunctionReturn(0);
}
EXTERN_C_END


#undef __FUNCT__
#define __FUNCT__ "MatCreateMPIAIJCRL"
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

.seealso: MatCreate(), MatCreateMPIAIJPERM(), MatSetValues()
@*/
PetscErrorCode  MatCreateMPIAIJCRL(MPI_Comm comm,PetscInt m,PetscInt n,PetscInt nz,const PetscInt nnz[],PetscInt onz,const PetscInt onnz[],Mat *A)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatCreate(comm,A);CHKERRQ(ierr);
  ierr = MatSetSizes(*A,m,n,m,n);CHKERRQ(ierr);
  ierr = MatSetType(*A,MATMPIAIJCRL);CHKERRQ(ierr);
  ierr = MatMPIAIJSetPreallocation_MPIAIJ(*A,nz,(PetscInt*)nnz,onz,(PetscInt*)onnz);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


EXTERN_C_BEGIN
#undef __FUNCT__
#define __FUNCT__ "MatCreate_MPIAIJCRL"
PetscErrorCode  MatCreate_MPIAIJCRL(Mat A)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatSetType(A,MATMPIAIJ);CHKERRQ(ierr);
  ierr = MatConvert_MPIAIJ_MPIAIJCRL(A,MATMPIAIJCRL,MAT_REUSE_MATRIX,&A);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
EXTERN_C_END

