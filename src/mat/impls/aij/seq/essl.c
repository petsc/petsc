/*$Id: essl.c,v 1.42 2000/05/16 22:56:32 bsmith Exp bsmith $*/

/* 
        Provides an interface to the IBM RS6000 Essl sparse solver

*/
#include "src/mat/impls/aij/seq/aij.h"

#if defined(PETSC_HAVE_ESSL) && !defined(__cplusplus)
/* #include <essl.h> This doesn't work!  */

typedef struct {
   int       n,nz;
   Scalar    *a;
   int       *ia;
   int       *ja;
   int       lna;
   int       iparm[5];
   PetscReal rparm[5];
   PetscReal oparm[5];
   Scalar    *aux;
   int       naux;
} Mat_SeqAIJ_Essl;


EXTERN int MatDestroy_SeqAIJ(Mat);

#undef __FUNC__  
#define __FUNC__ /*<a name="MatDestroy_SeqAIJ_Essl"></a>*/"MatDestroy_SeqAIJ_Essl"
int MatDestroy_SeqAIJ_Essl(Mat A)
{
  Mat_SeqAIJ      *a = (Mat_SeqAIJ*)A->data;
  Mat_SeqAIJ_Essl *essl = (Mat_SeqAIJ_Essl*)a->spptr;
  int             ierr;

  PetscFunctionBegin;
  /* free the Essl datastructures */
  ierr = PetscFree(essl->a);CHKERRQ(ierr);
  ierr = MatDestroy_SeqAIJ(A);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ /*<a name="MatSolve_SeqAIJ_Essl"></a>*/"MatSolve_SeqAIJ_Essl"
int MatSolve_SeqAIJ_Essl(Mat A,Vec b,Vec x)
{
  Mat_SeqAIJ      *a = (Mat_SeqAIJ*)A->data;
  Mat_SeqAIJ_Essl *essl = (Mat_SeqAIJ_Essl*)a->spptr;
  Scalar          *xx;
  int             ierr,m,zero = 0;

  PetscFunctionBegin;
  ierr = VecGetLocalSize(b,&m);CHKERRQ(ierr);
  ierr = VecCopy(b,x);CHKERRQ(ierr);
  ierr = VecGetArray(x,&xx);CHKERRQ(ierr);
  dgss(&zero,&a->n,essl->a,essl->ia,essl->ja,&essl->lna,xx,essl->aux,&essl->naux);
  ierr = VecRestoreArray(x,&xx);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ /*<a name="MatLUFactorNumeric_SeqAIJ_Essl"></a>*/"MatLUFactorNumeric_SeqAIJ_Essl"
int MatLUFactorNumeric_SeqAIJ_Essl(Mat A,Mat *F)
{
  Mat_SeqAIJ      *a = (Mat_SeqAIJ*)(*F)->data;
  Mat_SeqAIJ      *aa = (Mat_SeqAIJ*)(A)->data;
  Mat_SeqAIJ_Essl *essl = (Mat_SeqAIJ_Essl*)a->spptr;
  int             i,ierr,one = 1;

  PetscFunctionBegin;
  /* copy matrix data into silly ESSL data structure */
  if (!a->indexshift) {
    for (i=0; i<aa->m+1; i++) essl->ia[i] = aa->i[i] + 1;
    for (i=0; i<aa->nz; i++) essl->ja[i]  = aa->j[i] + 1;
  } else {
    ierr = PetscMemcpy(essl->ia,aa->i,(aa->m+1)*sizeof(int));CHKERRQ(ierr);
    ierr = PetscMemcpy(essl->ja,aa->j,(aa->nz)*sizeof(int));CHKERRQ(ierr);
  }
  ierr = PetscMemcpy(essl->a,aa->a,(aa->nz)*sizeof(Scalar));CHKERRQ(ierr);
  
  /* set Essl options */
  essl->iparm[0] = 1; 
  essl->iparm[1] = 5;
  essl->iparm[2] = 1;
  essl->iparm[3] = 0;
  essl->rparm[0] = 1.e-12;
  essl->rparm[1] = A->lupivotthreshold;

  dgsf(&one,&aa->m,&essl->nz,essl->a,essl->ia,essl->ja,&essl->lna,essl->iparm,
               essl->rparm,essl->oparm,essl->aux,&essl->naux);

  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ /*<a name="MatLUFactorSymbolic_SeqAIJ_Essl"></a>*/"MatLUFactorSymbolic_SeqAIJ_Essl"
int MatLUFactorSymbolic_SeqAIJ_Essl(Mat A,IS r,IS c,MatLUInfo,Mat *F)
{
  Mat             B;
  Mat_SeqAIJ      *a = (Mat_SeqAIJ*)A->data,*b;
  int             ierr,*ridx,*cidx,i,len;
  Mat_SeqAIJ_Essl *essl;
  PetscReal       f = 1.0;

  PetscFunctionBegin;
  if (A->N != A->M) SETERRQ(PETSC_ERR_ARG_SIZ,0,"matrix must be square"); 
  ierr           = MatCreateSeqAIJ(A->comm,a->m,a->n,0,PETSC_NULL,F);CHKERRQ(ierr);
  B                       = *F;
  B->ops->solve           = MatSolve_SeqAIJ_Essl;
  B->ops->destroy         = MatDestroy_SeqAIJ_Essl;
  B->ops->lufactornumeric = MatLUFactorNumeric_SeqAIJ_Essl;
  B->factor               = FACTOR_LU;
  b                       = (Mat_SeqAIJ*)B->data;
  essl                    = PetscNew(Mat_SeqAIJ_Essl);CHKPTRQ(essl);
  b->spptr                = (void*)essl;

  /* allocate the work arrays required by ESSL */
  if (info) f = info->fill;
  essl->nz   = a->nz;
  essl->lna  = (int)a->nz*f;
  essl->naux = 100 + 10*a->m;

  /* since malloc is slow on IBM we try a single malloc */
  len        = essl->lna*(2*sizeof(int)+sizeof(Scalar)) + essl->naux*sizeof(Scalar);
  essl->a    = (Scalar*)PetscMalloc(len);CHKPTRQ(essl->a);
  essl->aux  = essl->a + essl->lna;
  essl->ia   = (int*)(essl->aux + essl->naux);
  essl->ja   = essl->ia + essl->lna;

  PLogObjectMemory(B,len+sizeof(Mat_SeqAIJ_Essl));
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ /*<a name="MatUseEssl_SeqAIJ"></a>*/"MatUseEssl_SeqAIJ"
int MatUseEssl_SeqAIJ(Mat A)
{
  PetscFunctionBegin;
  A->ops->lufactorsymbolic = MatLUFactorSymbolic_SeqAIJ_Essl;
  PLogInfo(0,"Using ESSL for SeqAIJ LU factorization and solves");
  PetscFunctionReturn(0);
}

#else

#undef __FUNC__  
#define __FUNC__ /*<a name=""></a>*/"MatUseEssl_SeqAIJ"
int MatUseEssl_SeqAIJ(Mat A)
{
  PetscFunctionBegin;
  PetscFunctionReturn(0);
}

#endif


