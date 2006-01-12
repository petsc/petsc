#define PETSCMAT_DLL

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

#include "src/mat/impls/aij/mpi/mpiaij.h"
#include "src/mat/impls/aij/seq/crl/crl.h"

#undef __FUNCT__
#define __FUNCT__ "MatDestroy_MPICRL"
PetscErrorCode MatDestroy_MPICRL(Mat A)
{
  PetscErrorCode ierr;
  Mat_CRL        *crl = (Mat_CRL *) A->spptr;

  /* We are going to convert A back into a MPIAIJ matrix, since we are 
   * eventually going to use MatDestroy_MPIAIJ() to destroy everything 
   * that is not specific to CRL.
   * In preparation for this, reset the operations pointers in A to 
   * their MPIAIJ versions. */
  A->ops->assemblyend = crl->AssemblyEnd;
  A->ops->destroy     = crl->MatDestroy;
  A->ops->duplicate   = crl->MatDuplicate;

  /* Free everything in the Mat_CRL data structure. */
  if (crl->icols) {
    ierr = PetscFree2(crl->acols,crl->icols);CHKERRQ(ierr);
  }
  if (crl->fwork) {
    ierr = VecDestroy(crl->fwork);CHKERRQ(ierr);
  }
  if (crl->xwork) {
    ierr = VecDestroy(crl->xwork);CHKERRQ(ierr);
  }
  if (crl->array) {
    ierr = PetscFree(crl->array);CHKERRQ(ierr);
  }
  /* Free the Mat_CRL struct itself. */
  ierr = PetscFree(crl);CHKERRQ(ierr);

  /* Change the type of A back to MPIAIJ and use MatDestroy_MPIAIJ() 
   * to destroy everything that remains. */
  ierr = PetscObjectChangeTypeName( (PetscObject)A, MATMPIAIJ);CHKERRQ(ierr);
  /* Note that I don't call MatSetType().  I believe this is because that 
   * is only to be called when *building* a matrix. */
  ierr = (*A->ops->destroy)(A);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MPICRL_create_crl"
PetscErrorCode MPICRL_create_crl(Mat A)
{
  Mat_MPIAIJ     *a = (Mat_MPIAIJ *)(A)->data;
  Mat_SeqAIJ     *Aij = (Mat_SeqAIJ*)(a->A->data), *Bij = (Mat_SeqAIJ*)(a->B->data);
  Mat_CRL        *crl = (Mat_CRL*) A->spptr;
  PetscInt       m = A->m;  /* Number of rows in the matrix. */
  PetscInt       nd = a->A->n; /* number of columns in diagonal portion */
  PetscInt       *aj = Aij->j,*bj = Bij->j;  /* From the CSR representation; points to the beginning  of each row. */
  PetscInt       i, j,rmax = 0,*icols, *ailen = Aij->ilen, *bilen = Bij->ilen;
  PetscScalar    *aa = Aij->a,*ba = Bij->a,*acols,*array;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  /* determine the row with the most columns */
  for (i=0; i<m; i++) {
    rmax = PetscMax(rmax,ailen[i]+bilen[i]);
  }
  crl->nz   = Aij->nz+Bij->nz;
  crl->m    = A->m;
  crl->rmax = rmax;
  ierr  = PetscMalloc2(rmax*m,PetscScalar,&crl->acols,rmax*m,PetscInt,&crl->icols);CHKERRQ(ierr);
  acols = crl->acols;
  icols = crl->icols;
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
  ierr = PetscInfo1(A,"Percentage of 0's introduced for vectorized multiply %g\n",1.0-((double)(crl->nz))/((double)(rmax*m)));

  ierr = PetscMalloc((a->B->n+nd)*sizeof(PetscScalar),&array);CHKERRQ(ierr);
  /* xwork array is actually B->n+nd long, but we define xwork this length so can copy into it */
  ierr = VecCreateMPIWithArray(A->comm,nd,PETSC_DECIDE,array,&crl->xwork);CHKERRQ(ierr);
  ierr = VecCreateSeqWithArray(PETSC_COMM_SELF,a->B->n,array+nd,&crl->fwork);CHKERRQ(ierr);
  crl->array = array;
  crl->xscat = a->Mvctx;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatAssemblyEnd_MPICRL"
PetscErrorCode MatAssemblyEnd_MPICRL(Mat A, MatAssemblyType mode)
{
  PetscErrorCode ierr;
  Mat_CRL        *crl = (Mat_CRL*) A->spptr;
  Mat_MPIAIJ     *a = (Mat_MPIAIJ*)A->data;
  Mat_SeqAIJ     *Aij = (Mat_SeqAIJ*)(a->A->data), *Bij = (Mat_SeqAIJ*)(a->A->data);

  PetscFunctionBegin;
  if (mode == MAT_FLUSH_ASSEMBLY) PetscFunctionReturn(0);
  
  /* Since a MATMPICRL matrix is really just a MATMPIAIJ with some 
   * extra information, call the AssemblyEnd routine for a MATMPIAIJ. 
   * I'm not sure if this is the best way to do this, but it avoids 
   * a lot of code duplication.
   * I also note that currently MATMPICRL doesn't know anything about 
   * the Mat_CompressedRow data structure that MPIAIJ now uses when there 
   * are many zero rows.  If the MPIAIJ assembly end routine decides to use 
   * this, this may break things.  (Don't know... haven't looked at it.) */
  Aij->inode.use = PETSC_FALSE;
  Bij->inode.use = PETSC_FALSE;
  (*crl->AssemblyEnd)(A, mode);

  /* Now calculate the permutation and grouping information. */
  ierr = MPICRL_create_crl(A);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

extern PetscErrorCode MatMult_CRL(Mat,Vec,Vec);
extern PetscErrorCode MatDuplicate_CRL(Mat,MatDuplicateOption,Mat*); 

/* MatConvert_MPIAIJ_MPICRL converts a MPIAIJ matrix into a 
 * MPICRL matrix.  This routine is called by the MatCreate_MPICRL() 
 * routine, but can also be used to convert an assembled MPIAIJ matrix 
 * into a MPICRL one. */
EXTERN_C_BEGIN
#undef __FUNCT__
#define __FUNCT__ "MatConvert_MPIAIJ_MPICRL"
PetscErrorCode PETSCMAT_DLLEXPORT MatConvert_MPIAIJ_MPICRL(Mat A,MatType type,MatReuse reuse,Mat *newmat)
{
  /* This routine is only called to convert to MATMPICRL
   * from MATMPIAIJ, so we can ignore 'MatType Type'. */
  PetscErrorCode ierr;
  Mat            B = *newmat;
  Mat_CRL        *crl;

  PetscFunctionBegin;
  if (reuse == MAT_INITIAL_MATRIX) {
    ierr = MatDuplicate(A,MAT_COPY_VALUES,&B);CHKERRQ(ierr);
  }

  ierr = PetscNew(Mat_CRL,&crl);CHKERRQ(ierr);
  B->spptr = (void *) crl;

  /* Save a pointer to the original MPIAIJ assembly end routine, because we 
   * will want to use it later in the CRL assembly end routine. 
   * Also, save a pointer to the original MPIAIJ Destroy routine, because we 
   * will want to use it in the CRL destroy routine. */
  crl->AssemblyEnd  = A->ops->assemblyend;
  crl->MatDestroy   = A->ops->destroy;
  crl->MatDuplicate = A->ops->duplicate;

  /* Set function pointers for methods that we inherit from AIJ but 
   * override. */
  B->ops->duplicate   = MatDuplicate_CRL;
  B->ops->assemblyend = MatAssemblyEnd_MPICRL;
  B->ops->destroy     = MatDestroy_MPICRL;
  B->ops->mult        = MatMult_CRL;

  /* If A has already been assembled, compute the permutation. */
  if (A->assembled == PETSC_TRUE) {
    ierr = MPICRL_create_crl(B);CHKERRQ(ierr);
  }
  ierr = PetscObjectChangeTypeName((PetscObject)B,MATMPICRL);CHKERRQ(ierr);
  *newmat = B;
  PetscFunctionReturn(0);
}
EXTERN_C_END


#undef __FUNCT__
#define __FUNCT__ "MatCreateMPICRL"
/*@C
   MatCreateMPICRL - Creates a sparse matrix of type MPICRL.
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
PetscErrorCode PETSCMAT_DLLEXPORT MatCreateMPICRL(MPI_Comm comm,PetscInt m,PetscInt n,PetscInt nz,const PetscInt nnz[],PetscInt onz,const PetscInt onnz[],Mat *A)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatCreate(comm,A);CHKERRQ(ierr);
  ierr = MatSetSizes(*A,m,n,m,n);CHKERRQ(ierr);
  ierr = MatSetType(*A,MATMPICRL);CHKERRQ(ierr);
  ierr = MatMPIAIJSetPreallocation_MPIAIJ(*A,nz,(PetscInt*)nnz,onz,(PetscInt*)onnz);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


EXTERN_C_BEGIN
#undef __FUNCT__
#define __FUNCT__ "MatCreate_MPICRL"
PetscErrorCode PETSCMAT_DLLEXPORT MatCreate_MPICRL(Mat A)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  /* Change the type name before calling MatSetType() to force proper construction of MPIAIJ 
     and MATMPICRL types. */
  ierr = PetscObjectChangeTypeName((PetscObject)A,MATMPICRL);CHKERRQ(ierr);
  ierr = MatSetType(A,MATMPIAIJ);CHKERRQ(ierr);
  ierr = MatConvert_MPIAIJ_MPICRL(A,MATMPICRL,MAT_REUSE_MATRIX,&A);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
EXTERN_C_END

