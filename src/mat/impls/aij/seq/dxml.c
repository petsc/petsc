#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: dxml.c,v 1.9 1997/02/22 02:25:00 bsmith Exp balay $";
#endif

/* 
        Provides an interface to the DEC Alpha DXML library
     At the moment the DXNL library only offers sparse matrix vector product.
*/
#include "src/mat/impls/aij/seq/aij.h"
#include "src/vec/vecimpl.h"

#if defined(HAVE_DXML) && !defined(__cplusplus)

#undef __FUNC__  
#define __FUNC__ "MatMult_SeqAIJ_DXML"
static int MatMult_SeqAIJ_DXML(Mat A,Vec x,Vec y)
{
  Mat_SeqAIJ         *a = (Mat_SeqAIJ*) A->data;
  Scalar             *xx,*yy;
  int                ierr, zero = 0;

  VecGetArray_Fast(x,xx);
  VecGetArray_Fast(y,yy);
  dmatvec_genr_(&zero,a->a,a->i,a->j,&a->nz,0,xx,yy,&a->m);
  PLogFlops(2*a->nz - a->m);
  return 0;
}


#undef __FUNC__  
#define __FUNC__ "MatUseDXML_SeqAIJ" /* ADIC Ignore */
int MatUseDXML_SeqAIJ(Mat A)
{
  PetscValidHeaderSpecific(A,MAT_COOKIE);  
  if (A->type != MATSEQAIJ) return 0;
  A->ops.mult    = MatMult_SeqAIJ_DXML;
  return 0;
}

#else

#undef __FUNC__  
#define __FUNC__ "MatUseDXML_SeqAIJ" /* ADIC Ignore */
int MatUseDXML_SeqAIJ(Mat A)
{
  return 0;
}


#endif
