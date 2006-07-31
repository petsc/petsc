#define PETSCMAT_DLL

/* 
        Provides an interface to the IBM RS6000 Essl sparse solver

*/
#include "src/mat/impls/aij/seq/aij.h"
/* #include <essl.h> This doesn't work!  */

EXTERN_C_BEGIN
void dgss(int*,int*,double*,int*,int*,int*,double*,double*,int*);
void dgsf(int*,int*,int*,double*,int*,int*,int*,int*,double*,double*,double*,int*);
EXTERN_C_END

typedef struct {
  int         n,nz;
  PetscScalar *a;
  int         *ia;
  int         *ja;
  int         lna;
  int         iparm[5];
  PetscReal   rparm[5];
  PetscReal   oparm[5];
  PetscScalar *aux;
  int         naux;

  PetscErrorCode (*MatDuplicate)(Mat,MatDuplicateOption,Mat*);
  PetscErrorCode (*MatAssemblyEnd)(Mat,MatAssemblyType);
  PetscErrorCode (*MatLUFactorSymbolic)(Mat,IS,IS,MatFactorInfo*,Mat*);
  PetscErrorCode (*MatDestroy)(Mat);
  PetscTruth CleanUpESSL;
} Mat_Essl;

EXTERN PetscErrorCode MatDuplicate_Essl(Mat,MatDuplicateOption,Mat*);

EXTERN_C_BEGIN
#undef __FUNCT__
#define __FUNCT__ "MatConvert_Essl_SeqAIJ"
PetscErrorCode PETSCMAT_DLLEXPORT MatConvert_Essl_SeqAIJ(Mat A,MatType type,MatReuse reuse,Mat *newmat) {
  PetscErrorCode ierr;
  Mat            B=*newmat;
  Mat_Essl       *essl=(Mat_Essl*)A->spptr;
  
  PetscFunctionBegin;
  if (reuse == MAT_INITIAL_MATRIX) {
    ierr = MatDuplicate(A,MAT_COPY_VALUES,&B);
  }
  B->ops->duplicate        = essl->MatDuplicate;
  B->ops->assemblyend      = essl->MatAssemblyEnd;
  B->ops->lufactorsymbolic = essl->MatLUFactorSymbolic;
  B->ops->destroy          = essl->MatDestroy;

  /* free the Essl datastructures */
  ierr = PetscFree(essl);CHKERRQ(ierr);

  ierr = PetscObjectComposeFunction((PetscObject)B,"MatConvert_seqaij_essl_C","",PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)B,"MatConvert_essl_seqaij_C","",PETSC_NULL);CHKERRQ(ierr);

  ierr = PetscObjectChangeTypeName((PetscObject)B,MATSEQAIJ);CHKERRQ(ierr);
  *newmat = B;
  PetscFunctionReturn(0);
}
EXTERN_C_END  

#undef __FUNCT__  
#define __FUNCT__ "MatDestroy_Essl"
PetscErrorCode MatDestroy_Essl(Mat A) 
{
  PetscErrorCode ierr;
  Mat_Essl       *essl=(Mat_Essl*)A->spptr;

  PetscFunctionBegin;
  if (essl->CleanUpESSL) {
    ierr = PetscFree(essl->a);CHKERRQ(ierr);
  }
  ierr = MatConvert_Essl_SeqAIJ(A,MATSEQAIJ,MAT_REUSE_MATRIX,&A);
  ierr = (*A->ops->destroy)(A);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatSolve_Essl"
PetscErrorCode MatSolve_Essl(Mat A,Vec b,Vec x) {
  Mat_Essl       *essl = (Mat_Essl*)A->spptr;
  PetscScalar    *xx;
  PetscErrorCode ierr;
  int            m,zero = 0;

  PetscFunctionBegin;
  ierr = VecGetLocalSize(b,&m);CHKERRQ(ierr);
  ierr = VecCopy(b,x);CHKERRQ(ierr);
  ierr = VecGetArray(x,&xx);CHKERRQ(ierr);
  dgss(&zero,&A->cmap.n,essl->a,essl->ia,essl->ja,&essl->lna,xx,essl->aux,&essl->naux);
  ierr = VecRestoreArray(x,&xx);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatLUFactorNumeric_Essl"
PetscErrorCode MatLUFactorNumeric_Essl(Mat A,MatFactorInfo *info,Mat *F) {
  Mat_SeqAIJ     *aa=(Mat_SeqAIJ*)(A)->data;
  Mat_Essl       *essl=(Mat_Essl*)(*F)->spptr;
  PetscErrorCode ierr;
  int            i,one = 1;

  PetscFunctionBegin;
  /* copy matrix data into silly ESSL data structure (1-based Frotran style) */
  for (i=0; i<A->rmap.n+1; i++) essl->ia[i] = aa->i[i] + 1;
  for (i=0; i<aa->nz; i++) essl->ja[i]  = aa->j[i] + 1;
 
  ierr = PetscMemcpy(essl->a,aa->a,(aa->nz)*sizeof(PetscScalar));CHKERRQ(ierr);
  
  /* set Essl options */
  essl->iparm[0] = 1; 
  essl->iparm[1] = 5;
  essl->iparm[2] = 1;
  essl->iparm[3] = 0;
  essl->rparm[0] = 1.e-12;
  essl->rparm[1] = 1.0;
  ierr = PetscOptionsGetReal(A->prefix,"-matessl_lu_threshold",&essl->rparm[1],PETSC_NULL);CHKERRQ(ierr);

  dgsf(&one,&A->rmap.n,&essl->nz,essl->a,essl->ia,essl->ja,&essl->lna,essl->iparm,
               essl->rparm,essl->oparm,essl->aux,&essl->naux);

  (*F)->assembled = PETSC_TRUE;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatLUFactorSymbolic_Essl"
PetscErrorCode MatLUFactorSymbolic_Essl(Mat A,IS r,IS c,MatFactorInfo *info,Mat *F) {
  Mat            B;
  Mat_SeqAIJ     *a = (Mat_SeqAIJ*)A->data;
  PetscErrorCode ierr;
  int            len;
  Mat_Essl       *essl;
  PetscReal      f = 1.0;

  PetscFunctionBegin;
  if (A->cmap.N != A->rmap.N) SETERRQ(PETSC_ERR_ARG_SIZ,"matrix must be square"); 
  ierr = MatCreate(A->comm,&B);CHKERRQ(ierr);
  ierr = MatSetSizes(B,PETSC_DECIDE,PETSC_DECIDE,A->rmap.n,A->cmap.n);CHKERRQ(ierr);
  ierr = MatSetType(B,A->type_name);CHKERRQ(ierr);
  ierr = MatSeqAIJSetPreallocation(B,0,PETSC_NULL);CHKERRQ(ierr);

  B->ops->solve           = MatSolve_Essl;
  B->ops->lufactornumeric = MatLUFactorNumeric_Essl;
  B->factor               = FACTOR_LU;
  
  essl = (Mat_Essl *)(B->spptr);

  /* allocate the work arrays required by ESSL */
  f = info->fill;
  essl->nz   = a->nz;
  essl->lna  = (int)a->nz*f;
  essl->naux = 100 + 10*A->rmap.n;

  /* since malloc is slow on IBM we try a single malloc */
  len               = essl->lna*(2*sizeof(int)+sizeof(PetscScalar)) + essl->naux*sizeof(PetscScalar);
  ierr              = PetscMalloc(len,&essl->a);CHKERRQ(ierr);
  essl->aux         = essl->a + essl->lna;
  essl->ia          = (int*)(essl->aux + essl->naux);
  essl->ja          = essl->ia + essl->lna;
  essl->CleanUpESSL = PETSC_TRUE;

  ierr = PetscLogObjectMemory(B,len+sizeof(Mat_Essl));CHKERRQ(ierr);
  *F = B;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatAssemblyEnd_Essl"
PetscErrorCode MatAssemblyEnd_Essl(Mat A,MatAssemblyType mode) 
{
  PetscErrorCode ierr;
  Mat_Essl       *essl=(Mat_Essl*)(A->spptr);

  PetscFunctionBegin;
  ierr = (*essl->MatAssemblyEnd)(A,mode);CHKERRQ(ierr);

  essl->MatLUFactorSymbolic = A->ops->lufactorsymbolic;
  A->ops->lufactorsymbolic  = MatLUFactorSymbolic_Essl;
  ierr = PetscInfo(0,"Using ESSL for LU factorization and solves\n");CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

EXTERN_C_BEGIN
#undef __FUNCT__
#define __FUNCT__ "MatConvert_SeqAIJ_Essl"
PetscErrorCode PETSCMAT_DLLEXPORT MatConvert_SeqAIJ_Essl(Mat A,MatType type,MatReuse reuse,Mat *newmat) 
{
  Mat            B=*newmat;
  PetscErrorCode ierr;
  Mat_Essl       *essl;

  PetscFunctionBegin;
  if (reuse == MAT_INITIAL_MATRIX) {
    ierr = MatDuplicate(A,MAT_COPY_VALUES,&B);CHKERRQ(ierr);
  }

  ierr                      = PetscNew(Mat_Essl,&essl);CHKERRQ(ierr);
  essl->MatDuplicate        = A->ops->duplicate;
  essl->MatAssemblyEnd      = A->ops->assemblyend;
  essl->MatLUFactorSymbolic = A->ops->lufactorsymbolic;
  essl->MatDestroy          = A->ops->destroy;
  essl->CleanUpESSL         = PETSC_FALSE;

  B->spptr                  = (void*)essl;
  B->ops->duplicate         = MatDuplicate_Essl;
  B->ops->assemblyend       = MatAssemblyEnd_Essl;
  B->ops->lufactorsymbolic  = MatLUFactorSymbolic_Essl;
  B->ops->destroy           = MatDestroy_Essl;

  ierr = PetscObjectComposeFunctionDynamic((PetscObject)B,"MatConvert_seqaij_essl_C",
                                           "MatConvert_SeqAIJ_Essl",MatConvert_SeqAIJ_Essl);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)B,"MatConvert_essl_seqaij_C",
                                           "MatConvert_Essl_SeqAIJ",MatConvert_Essl_SeqAIJ);CHKERRQ(ierr);
  ierr = PetscObjectChangeTypeName((PetscObject)B,type);CHKERRQ(ierr);

  *newmat = B;
  PetscFunctionReturn(0);
}
EXTERN_C_END

#undef __FUNCT__
#define __FUNCT__ "MatDuplicate_Essl"
PetscErrorCode MatDuplicate_Essl(Mat A, MatDuplicateOption op, Mat *M) 
{
  PetscErrorCode ierr;
  Mat_Essl       *lu = (Mat_Essl *)A->spptr;

  PetscFunctionBegin;
  ierr = (*lu->MatDuplicate)(A,op,M);CHKERRQ(ierr);
  ierr = PetscMemcpy((*M)->spptr,lu,sizeof(Mat_Essl));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*MC
  MATESSL - MATESSL = "essl" - A matrix type providing direct solvers (LU) for sequential matrices 
  via the external package ESSL.

  If ESSL is installed (see the manual for
  instructions on how to declare the existence of external packages),
  a matrix type can be constructed which invokes ESSL solvers.
  After calling MatCreate(...,A), simply call MatSetType(A,MATESSL).
  This matrix type is only supported for double precision real.

  This matrix inherits from MATSEQAIJ.  As a result, MatSeqAIJSetPreallocation is 
  supported for this matrix type.  One can also call MatConvert for an inplace conversion to or from 
  the MATSEQAIJ type without data copy.

  Options Database Keys:
. -mat_type essl - sets the matrix type to "essl" during a call to MatSetFromOptions()

   Level: beginner

.seealso: PCLU
M*/

EXTERN_C_BEGIN
#undef __FUNCT__
#define __FUNCT__ "MatCreate_Essl"
PetscErrorCode PETSCMAT_DLLEXPORT MatCreate_Essl(Mat A) 
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatSetType(A,MATSEQAIJ);CHKERRQ(ierr);
  ierr = MatConvert_SeqAIJ_Essl(A,MATESSL,MAT_REUSE_MATRIX,&A);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
EXTERN_C_END
