
#ifndef lint
static char vcid[] = "$Id: dxml.c,v 1.2 1995/09/30 19:28:44 bsmith Exp bsmith $";
#endif

/* 
        Provides an interface to the DEC Alpha DXML library
     At the moment the DXNL library only offers sparse matrix vector product.
*/
#include "aij.h"

#if defined(HAVE_DXML) && !defined(__cplusplus)

static int MatMult_SeqAIJ_DXML(Mat A,Vec x,Vec y)
{
  Mat_SeqAIJ         *a = (Mat_SeqAIJ*) A->data;
  Scalar             *xx,*yy;
  int                ierr, zero = 0;

  ierr = VecGetArray(x,&xx); CHKERRQ(ierr);
  ierr = VecGetArray(y,&yy); CHKERRQ(ierr);
  dmatvec_genr_(&zero,a->a,a->i,a->j,&a->nz,0,xx,yy,&a->m);
  PLogFlops(2*a->nz - a->m);
  return 0;
}


int MatUseDXML_SeqAIJ(Mat A)
{
  PETSCVALIDHEADERSPECIFIC(A,MAT_COOKIE);  
  if (A->type != MATSEQAIJ) return 0;
  A->ops.mult    = MatMult_SeqAIJ_DXML;
  return 0;
}

#else

int MatUseDXML_SeqAIJ(Mat A)
{
  return 0;
}


#endif
