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

  int (*MatDestroy)(Mat);
} Mat_SeqAIJ_Essl;

#undef __FUNCT__  
#define __FUNCT__ "MatDestroy_SeqAIJ_Essl"
int MatDestroy_SeqAIJ_Essl(Mat A)
{
  Mat_SeqAIJ_Essl *essl = (Mat_SeqAIJ_Essl*)A->spptr;
  int             ierr,(*destroy)(Mat);

  PetscFunctionBegin;
  /* free the Essl datastructures */
  destroy = essl->MatDestroy;
  ierr = PetscFree(essl->a);CHKERRQ(ierr);
  ierr = (*destroy)(A);CHKERRQ(ierr);
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
  Mat_SeqAIJ      *a = (Mat_SeqAIJ*)(*F)->data; 
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
  Mat_SeqAIJ      *a = (Mat_SeqAIJ*)A->data,*b;
  int             ierr,*ridx,*cidx,i,len;
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

#undef __FUNCT__  
#define __FUNCT__ "MatUseEssl_SeqAIJ"
int MatUseEssl_SeqAIJ(Mat A)
{
  PetscFunctionBegin;
  A->ops->lufactorsymbolic = MatLUFactorSymbolic_SeqAIJ_Essl;
  PetscLogInfo(0,"Using ESSL for SeqAIJ LU factorization and solves");
  PetscFunctionReturn(0);
}

EXTERN_C_BEGIN
#undef __FUNCT__
#define __FUNCT__ "MatCreate_SeqAIJ_Essl"
int MatCreate_SeqAIJ_Essl(Mat A) {
  int             ierr;
  Mat_SeqAIJ_Essl *essl;

  PetscFunctionBegin;
  ierr = MatSetType(A,MATSEQAIJ);
  ierr = MatUseEssl_SeqAIJ(A);

  ierr             = PetscNew(Mat_SeqAIJ_Essl,&essl);CHKERRQ(ierr);
  essl->MatDestroy = A->ops->destroy;
  A->spptr         = (void *)essl;

  A->ops->destroy  = MatDestroy_SeqAIJ_Essl;
  PetscFunctionReturn(0);
}
EXTERN_C_END
