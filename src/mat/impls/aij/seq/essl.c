
#ifndef lint
static char vcid[] = "$Id: essl.c,v 1.1 1995/09/20 01:56:56 bsmith Exp bsmith $";
#endif

/* 
        Provides an interface to the IBM RS6000 Essl sparse solver

*/
#include "aij.h"

#if defined(HAVE_ESSL) && !defined(__cplusplus)
/* #include <essl.h>  */
#include <math.h>

typedef struct {
   int    n,nz;
   Scalar *a;
   int    *ia;
   int    *ja;
   int    lna;
   int    iparm[5];
   double rparm[5];
   double oparm[5];
   Scalar *aux;
   int    naux;
} Mat_SeqAIJ_EsslLU;


extern int MatDestroy_SeqAIJ(PetscObject);

static int MatGetReordering_SeqAIJ_EsslLU(Mat mat,MatOrdering type,
                                               IS *rperm,IS *cperm)
{
  *perm = *cperm = 0;
  return 0;
}

static int MatDestroy_SeqAIJ_EsslLU(PetscObject obj)
{
  Mat               A = (Mat) obj;
  Mat_SeqAIJ        *a = (Mat_SeqAIJ*) A->data;
  Mat_SeqAIJ_EsslLU *essl = (Mat_SeqAIJ_EsslLU*) a->spptr;

  /* free the EsslLU datastructures */
  PETSCFREE(essl->a);
  PETSCFREE(essl->ia);
  PETSCFREE(essl->ja);
  PETSCFREE(essl->aux);
  PETSCFREE(essl);

  return MatDestroy_SeqAIJ(obj);
}

static int MatSolve_SeqAIJ_EsslLU(Mat A,Vec b,Vec x)
{
  Mat_SeqAIJ         *a = (Mat_SeqAIJ*) A->data;
  Mat_SeqAIJ_EsslLU *essl = (Mat_SeqAIJ_EsslLU*) a->spptr;
  Scalar             *xx;
  int                ierr,m, zero = 0;

  ierr = VecGetSize(b,&m); CHKERRQ(ierr);
  ierr = VecCopy(b,x); CHKERRQ(ierr);
  ierr = VecGetArray(x,&xx); CHKERRQ(ierr);

  dgss(&zero, &a->n, essl->a, essl->ia, essl->ja,&essl->lna,xx,essl->aux,
              &essl->naux);

  return 0;
}

static int MatLUFactorSymbolic_SeqAIJ_EsslLU(Mat A,IS r,IS c,double f,Mat *F)
{
  Mat                B;
  Mat_SeqAIJ         *a = (Mat_SeqAIJ*) A->data, *b;
  int                ierr, *ridx, *cidx,i;
  Mat_SeqAIJ_EsslLU *essl;

  if (a->m != a->n) 
    SETERRQ(1,"MatLUFactorSymbolic_SeqAIJ_EsslLU: matrix ust be square"); 
  ierr          = MatCreateSeqAIJ(A->comm,a->m,a->n,0,0,F); CHKERRQ(ierr);
  B             = *F;
  B->ops.solve  = MatSolve_SeqAIJ_EsslLU;
  B->destroy    = MatDestroy_SeqAIJ_EsslLU;
  B->factor     = FACTOR_LU;
  b             = (Mat_SeqAIJ*) B->data;
  essl          = PETSCNEW(Mat_SeqAIJ_EsslLU); CHKPTRQ(essl);
  b->spptr      = (void*) essl;


  /* allocate the work arrays required by ESSL */
  essl->nz   = a->nz;
  essl->lna  = a->nz*sqrt((double)(a->nz));
  essl->ia   = (int*) PETSCMALLOC(essl->lna*sizeof(int)); CHKPTRQ(essl->ia);
  essl->ja   = (int*) PETSCMALLOC(essl->lna*sizeof(int)); CHKPTRQ(essl->ja);
  essl->a    = (Scalar*) PETSCMALLOC(essl->lna*sizeof(Scalar)); 
               CHKPTRQ(essl->a);
  essl->naux = 100 + 10*a->m;
  essl->aux  = (Scalar*) PETSCMALLOC(essl->naux*sizeof(Scalar)); 
               CHKPTRQ(essl->aux);

  return 0;
}

static int MatLUFactorNumeric_SeqAIJ_EsslLU(Mat A,Mat *F)
{
  Mat_SeqAIJ        *a = (Mat_SeqAIJ*) (*F)->data;
  Mat_SeqAIJ        *aa = (Mat_SeqAIJ*) (A)->data;
  Mat_SeqAIJ_EsslLU *essl = (Mat_SeqAIJ_EsslLU *) a->spptr;
  int               i,ierr, one = 1;

  /* copy matrix data into silly ESSL data structure */
  if (!a->indexshift) {
    for ( i=0; i<aa->m+1; i++ ) essl->ia[i] = aa->i[i] + 1;
    for ( i=0; i<aa->nz; i++ ) essl->ja[i]  = aa->j[i] + 1;
  }
  else {
    PETSCMEMCPY(essl->ia,aa->i,(aa->m+1)*sizeof(int));
    PETSCMEMCPY(essl->ja,aa->j,(aa->nz)*sizeof(int));
  }
  PETSCMEMCPY(essl->a,aa->a,(aa->nz)*sizeof(Scalar));
  
  /* set Essl options */
  essl->iparm[0] = 0; /* use defaults */

  dgsf(&one,&aa->m,&essl->nz,essl->a,essl->ia,essl->ja,&essl->lna,essl->iparm,
               essl->rparm,essl->oparm,essl->aux,&essl->naux);

  return 0;
}

int MatUseEsslLU_SeqAIJ(Mat A)
{
  PETSCVALIDHEADERSPECIFIC(A,MAT_COOKIE);  
  if (A->type != MATSEQAIJ) return 0;

  A->ops.lufactorsymbolic = MatLUFactorSymbolic_SeqAIJ_EsslLU;
  A->ops.lufactornumeric  = MatLUFactorNumeric_SeqAIJ_EsslLU;
  A->ops.lugetreordering  = MatGetReordering_SeqAIJ_EsslLU;

  return 0;
}

#else

int MatUseEsslLU_SeqAIJ(Mat A)
{
  return 0;
}


#endif
