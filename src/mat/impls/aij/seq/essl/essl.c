/*$Id: essl.c,v 1.49 2001/08/07 03:02:47 balay Exp $*/

/* 
        Provides an interface to the IBM RS6000 Essl sparse solver

*/
#include "src/mat/impls/aij/seq/aij.h"
/* #include <essl.h> This doesn't work!  */

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

  int (*MatAssemblyEnd)(Mat,MatAssemblyType);
  int (*MatLUFactorSymbolic)(Mat,IS,IS,MatFactorInfo*,Mat*);
  int (*MatDestroy)(Mat);
} Mat_SeqAIJ_Essl;

EXTERN int MatLUFactorSymbolic_SeqAIJ_Essl(Mat,IS,IS,MatFactorInfo*,Mat*);

EXTERN_C_BEGIN
#undef __FUNCT__
#define __FUNCT__ "MatConvert_Essl_SeqAIJ"
int MatConvert_Essl_SeqAIJ(Mat A,MatType type,Mat *newmat) {
  int             ierr;
  Mat             B=*newmat;
  Mat_SeqAIJ_Essl *essl = (Mat_SeqAIJ_Essl*)A->spptr;
  
  PetscFunctionBegin;
  if (B != A) {
    ierr = MatDuplicate(A,MAT_COPY_VALUES,&B);
  } else {
    /* free the Essl datastructures */
    ierr = PetscFree(essl->a);CHKERRQ(ierr);
    ierr = PetscFree(essl);CHKERRQ(ierr);
    ierr = PetscObjectChangeTypeName((PetscObject)B,MATSEQAIJ);CHKERRQ(ierr);
  }
  *newmat = B;
  PetscFunctionReturn(0);
}
EXTERN_C_END  

#undef __FUNCT__  
#define __FUNCT__ "MatDestroy_SeqAIJ_Essl"
int MatDestroy_SeqAIJ_Essl(Mat A)
{
  int             ierr;
  Mat_SeqAIJ_Essl *essl = (Mat_SeqAIJ_Essl*)A->spptr;
  int             (*destroy)(Mat)=essl->MatDestroy;

  PetscFunctionBegin;
  ierr = MatConvert_Essl_SeqAIJ(A,MATSEQAIJ,&A);
  ierr = (*destroy)(A);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatAssemblyEnd_SeqAIJ_Essl"
int MatAssemblyEnd_SeqAIJ_Essl(Mat A,MatAssemblyType mode) {
  int             ierr;
  Mat_SeqAIJ_Essl *essl=(Mat_SeqAIJ_Essl*)(A->spptr);

  PetscFunctionBegin;
  ierr = (*essl->MatAssemblyEnd)(A,mode);CHKERRQ(ierr);

  essl->MatLUFactorSymbolic = A->ops->lufactorsymbolic;
  A->ops->lufactorsymbolic  = MatLUFactorSymbolic_SeqAIJ_Essl;
  PetscLogInfo(0,"Using ESSL for SeqAIJ LU factorization and solves");
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatSolve_SeqAIJ_Essl"
int MatSolve_SeqAIJ_Essl(Mat A,Vec b,Vec x)
{
  Mat_SeqAIJ_Essl *essl = (Mat_SeqAIJ_Essl*)A->spptr;
  PetscScalar     *xx;
  int             ierr,m,zero = 0;

  PetscFunctionBegin;
  ierr = VecGetLocalSize(b,&m);CHKERRQ(ierr);
  ierr = VecCopy(b,x);CHKERRQ(ierr);
  ierr = VecGetArray(x,&xx);CHKERRQ(ierr);
  dgss(&zero,&A->n,essl->a,essl->ia,essl->ja,&essl->lna,xx,essl->aux,&essl->naux);
  ierr = VecRestoreArray(x,&xx);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatLUFactorNumeric_SeqAIJ_Essl"
int MatLUFactorNumeric_SeqAIJ_Essl(Mat A,Mat *F)
{
  Mat_SeqAIJ      *aa = (Mat_SeqAIJ*)(A)->data;
  Mat_SeqAIJ_Essl *essl = (Mat_SeqAIJ_Essl*)(*F)->spptr;
  int             i,ierr,one = 1;

  PetscFunctionBegin;
  /* copy matrix data into silly ESSL data structure (1-based Frotran style) */
  for (i=0; i<A->m+1; i++) essl->ia[i] = aa->i[i] + 1;
  for (i=0; i<aa->nz; i++) essl->ja[i]  = aa->j[i] + 1;
 
  ierr = PetscMemcpy(essl->a,aa->a,(aa->nz)*sizeof(PetscScalar));CHKERRQ(ierr);
  
  /* set Essl options */
  essl->iparm[0] = 1; 
  essl->iparm[1] = 5;
  essl->iparm[2] = 1;
  essl->iparm[3] = 0;
  essl->rparm[0] = 1.e-12;
  essl->rparm[1] = A->lupivotthreshold;

  dgsf(&one,&A->m,&essl->nz,essl->a,essl->ia,essl->ja,&essl->lna,essl->iparm,
               essl->rparm,essl->oparm,essl->aux,&essl->naux);

  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatLUFactorSymbolic_SeqAIJ_Essl"
int MatLUFactorSymbolic_SeqAIJ_Essl(Mat A,IS r,IS c,MatFactorInfo *info,Mat *F)
{
  Mat             B;
  Mat_SeqAIJ      *a = (Mat_SeqAIJ*)A->data;
  int             ierr,len;
  Mat_SeqAIJ_Essl *essl;
  PetscReal       f = 1.0;

  PetscFunctionBegin;
  if (A->N != A->M) SETERRQ(PETSC_ERR_ARG_SIZ,"matrix must be square"); 
  ierr = MatCreate(A->comm,PETSC_DECIDE,PETSC_DECIDE,A->m,A->n,&B);CHKERRQ(ierr);
  ierr = MatSetType(B,MATESSL);CHKERRQ(ierr);
  ierr = MatSeqAIJSetPreallocation(B,0,PETSC_NULL);CHKERRQ(ierr);

  B->ops->solve           = MatSolve_SeqAIJ_Essl;
  B->ops->lufactornumeric = MatLUFactorNumeric_SeqAIJ_Essl;
  B->factor               = FACTOR_LU;
  
  essl = (Mat_SeqAIJ_Essl *)(B->spptr);

  /* allocate the work arrays required by ESSL */
  f = info->fill;
  essl->nz   = a->nz;
  essl->lna  = (int)a->nz*f;
  essl->naux = 100 + 10*A->m;

  /* since malloc is slow on IBM we try a single malloc */
  len        = essl->lna*(2*sizeof(int)+sizeof(PetscScalar)) + essl->naux*sizeof(PetscScalar);
  ierr       = PetscMalloc(len,&essl->a);CHKERRQ(ierr);
  essl->aux  = essl->a + essl->lna;
  essl->ia   = (int*)(essl->aux + essl->naux);
  essl->ja   = essl->ia + essl->lna;

  PetscLogObjectMemory(B,len+sizeof(Mat_SeqAIJ_Essl));
  *F = B;
  PetscFunctionReturn(0);
}

EXTERN_C_BEGIN
#undef __FUNCT__
#define __FUNCT__ "MatConvert_SeqAIJ_Essl"
int MatConvert_SeqAIJ_Essl(Mat A,MatType type,Mat *newmat) {
  Mat             B=*newmat;
  int             ierr;
  Mat_SeqAIJ_Essl *essl;

  PetscFunctionBegin;

  if (B != A) {
    ierr = MatDuplicate(A,MAT_COPY_VALUES,&B);CHKERRQ(ierr);
  }

  ierr                      = PetscNew(Mat_SeqAIJ_Essl,&essl);CHKERRQ(ierr);
  essl->MatAssemblyEnd      = A->ops->assemblyend;
  essl->MatLUFactorSymbolic = A->ops->lufactorsymbolic;
  essl->MatDestroy          = A->ops->destroy;
  B->spptr                  = (void *)essl;

  B->ops->assemblyend       = MatAssemblyEnd_SeqAIJ_Essl;
  B->ops->lufactorsymbolic  = MatLUFactorSymbolic_SeqAIJ_Essl;
  B->ops->destroy           = MatDestroy_SeqAIJ_Essl;

  ierr = PetscObjectComposeFunctionDynamic((PetscObject)B,"MatConvert_seqaij_essl_C",
                                           "MatConvert_SeqAIJ_Essl",MatConvert_SeqAIJ_Essl);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)B,"MatConvert_essl_seqaij_C",
                                           "MatConvert_Essl_SeqAIJ",MatConvert_Essl_SeqAIJ);CHKERRQ(ierr);
  ierr = PetscObjectChangeTypeName((PetscObject)B,type);CHKERRQ(ierr);
  *newmat = B;
  PetscFunctionReturn(0);
}
EXTERN_C_END

EXTERN_C_BEGIN
#undef __FUNCT__
#define __FUNCT__ "MatCreate_SeqAIJ_Essl"
int MatCreate_SeqAIJ_Essl(Mat A) {
  int             ierr;

  PetscFunctionBegin;
  ierr = MatSetType(A,MATSEQAIJ);
  ierr = MatConvert_SeqAIJ_Essl(A,MATESSL,&A);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
EXTERN_C_END
