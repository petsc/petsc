
#ifndef lint
static char vcid[] = "$Id: superlu.c,v 1.1 1995/09/13 02:22:32 bsmith Exp bsmith $";
#endif

/* 
        Provides an interface to the SuperLU sparse factorization 
  package of Jim Demmel, John Gilbert and Xiaoye Li.
*/
#include "aij.h"
#include "vec/vecimpl.h"
#include "inline/spops.h"

#if defined(HAVE_SUPERLU)
#include "dsp_defs.h"
#include "util.h"

typedef struct {
  SuperMatrix AC,L,U;
  int         *perm_r,*perm_c;
  int         *etree;
} Mat_SeqAIJ_SuperLU;

extern LUStat_t LUStat;

extern int MatDestroy_SeqAIJ(PetscObject);

static int MatDestroy_SeqAIJ_SuperLU(PetscObject obj)
{
  Mat                A = (Mat) obj;
  Mat_SeqAIJ         *a = (Mat_SeqAIJ*) A->data;
  Mat_SeqAIJ_SuperLU *super = (Mat_SeqAIJ_SuperLU*) a->spptr;

  /* free the SuperLU datastructures */
  Destroy_CompCol_Permuted(&super->AC);
  PETSCFREE(super->etree);
  PETSCFREE(super);

  return MatDestroy_SeqAIJ(obj);
}

static int MatSolve_SeqAIJ_SuperLU(Mat A,Vec b,Vec x)
{
  Mat_SeqAIJ         *a = (Mat_SeqAIJ*) A->data;
  Mat_SeqAIJ_SuperLU *super = (Mat_SeqAIJ_SuperLU*) a->spptr;
  Scalar             *xx;
  int                ierr,m;
  SuperMatrix        BB;

  ierr = VecGetSize(b,&m); CHKERRQ(ierr);
  ierr = VecCopy(b,x); CHKERRQ(ierr);
  ierr = VecGetArray(x,&xx); CHKERRQ(ierr);
  dCreate_Dense_Matrix(&BB,m,1,xx,m,DN,D,GE);

  dgstrs ("T", &super->L, &super->U, super->perm_r, super->perm_c,&BB, &ierr);

  PLogFlops(LUStat.ops[SOLVE]);
  return 0;
}

static int MatLUFactorSymbolic_SeqAIJ_SuperLU(Mat A,IS r,IS c,double f,Mat *F)
{
  Mat                B;
  Mat_SeqAIJ         *a = (Mat_SeqAIJ*) A->data, *b;
  int                ierr, *ridx, *cidx,i;
  SuperMatrix        Asuper;
  Mat_SeqAIJ_SuperLU *super;

  ierr = MatCreateSeqAIJ(A->comm,a->m,a->n,0,0,F); CHKERRQ(ierr);
  B = *F;
  B->ops.solve  = MatSolve_SeqAIJ_SuperLU;
  B->destroy    = MatDestroy_SeqAIJ_SuperLU;
  B->factor     = FACTOR_LU;
  b = (Mat_SeqAIJ*) B->data;
  super = PETSCNEW(Mat_SeqAIJ_SuperLU); CHKPTRQ(super);
  b->spptr = (void*) super;

  /* note that SuperLU is column oriented, thus we pass the row and 
     column pointers backwards */
  /* shift row and column indices to start at 0 */
  if (a->indexshift) {
    for ( i=0; i<a->m+1; i++ ) a->i[i]--;
    for ( i=0; i<a->nz; i++ ) a->j[i]--;
  }

  dCreate_CompCol_Matrix(&Asuper,a->m,a->n,a->nz,a->a,a->j,a->i,NC,D,GE);
    
  ierr = ISGetIndices(r,&ridx); CHKERRQ(ierr);
  ierr = ISGetIndices(c,&cidx); CHKERRQ(ierr);
  super->etree = (int*) PETSCMALLOC(a->m*sizeof(int)); CHKPTRQ(super->etree);

  StatInit(sp_ienv(1)); 

  sp_preorder("N", &Asuper, cidx, super->etree, &super->AC);

  super->perm_r = ridx; 
  super->perm_c = cidx;

  /* shift I and J pointers back */
  if (a->indexshift) {
    for ( i=0; i<a->m+1; i++ ) a->i[i]++;
    for ( i=0; i<a->nz; i++ ) a->j[i]++;
  }
  return 0;
}

static int MatLUFactorNumeric_SeqAIJ_SuperLU(Mat A,Mat *F)
{
  Mat_SeqAIJ         *a = (Mat_SeqAIJ*) (*F)->data;
  Mat_SeqAIJ         *aa = (Mat_SeqAIJ*) (A)->data;
  Mat_SeqAIJ_SuperLU *super = (Mat_SeqAIJ_SuperLU *) a->spptr;
  int                i,ierr, *ridx = super->perm_r;

  if (a->indexshift) {
    for ( i=0; i<aa->m+1; i++ ) aa->i[i]--;
    for ( i=0; i<aa->nz; i++ ) aa->j[i]--;
  }

  dgstrf("N",&super->AC,1.0,0.0,sp_ienv(2),sp_ienv(1),super->etree,0,0,ridx,
         &super->L,&super->U,&ierr);  CHKERRQ(ierr);

  PLogFlops(LUStat.ops[FACT]);

  if (a->indexshift) {
    for ( i=0; i<aa->m+1; i++ ) aa->i[i]++;
    for ( i=0; i<aa->nz; i++ ) aa->j[i]++;
  }
  return 0;
}

int MatUseSuperLU(Mat A)
{
  PETSCVALIDHEADERSPECIFIC(A,MAT_COOKIE);  
  if (A->type != MATSEQAIJ) return 0;

  A->ops.lufactorsymbolic = MatLUFactorSymbolic_SeqAIJ_SuperLU;
  A->ops.lufactornumeric  = MatLUFactorNumeric_SeqAIJ_SuperLU;

  return 0;
}

#else

int MatUseSuperLU(Mat A)
{
  return 0;
}


#endif
