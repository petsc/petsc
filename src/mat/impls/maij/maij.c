/*$Id: maij.c,v 1.1 2000/05/17 19:44:12 bsmith Exp bsmith $*/
/*
    Defines the basic matrix operations for the MAIJ  matrix storage format.
  This format is used for restriction and interpolation operations for 
  multicomponent problems. It interpolates each component the same way
  independently.

     We provide:
         MatMult()
         MatMultTranspose()
         MatMultTransposeAdd()
         MatMultAdd()
          and
         MatCreateMAIJ(Mat,dof,Mat*)
*/

#include "src/mat/impls/aij/seq/aij.h"

typedef struct {
  int dof;               /* number of components */
  Mat AIJ;               /* representation of interpolation for one component */
} Mat_SeqMAIJ;

#undef __FUNC__  
#define __FUNC__ /*<a name="MatDestroy_SeqMAIJ"></a>*/"MatDestroy_SeqMAIJ" 
int MatDestroy_SeqMAIJ(Mat A)
{
  int         ierr;
  Mat_SeqMAIJ *b = (Mat_SeqMAIJ*)A->data;

  PetscFunctionBegin;
  if (b->AIJ) {
    ierr = MatDestroy(b->AIJ);CHKERRQ(ierr);
  }
  ierr = PetscFree(b);CHKERRQ(ierr);
  PetscHeaderDestroy(A);
  PetscFunctionReturn(0);
}

EXTERN_C_BEGIN
#undef __FUNC__  
#define __FUNC__ /*<a name="MatCreate_SeqMAIJ"></a>*/"MatCreate_SeqMAIJ" 
int MatCreate_SeqMAIJ(Mat A)
{
  int         ierr;
  Mat_SeqMAIJ *b;

  PetscFunctionBegin;
  A->data             = (void*)(b = PetscNew(Mat_SeqMAIJ));CHKPTRQ(b);
  ierr = PetscMemzero(b,sizeof(Mat_SeqMAIJ));CHKERRQ(ierr);
  ierr = PetscMemzero(A->ops,sizeof(struct _MatOps));CHKERRQ(ierr);
  A->factor           = 0;
  A->mapping          = 0;
  b->AIJ = 0;
  b->dof = 0;  
  PetscFunctionReturn(0);
}
EXTERN_C_END

EXTERN int MatMult_SeqAIJ(Mat,Vec,Vec);
EXTERN int MatMultTranspose_SeqAIJ(Mat,Vec,Vec);
EXTERN int MatMultTransposeAdd_SeqAIJ(Mat,Vec,Vec,Vec);
EXTERN int matmulttransposeadd_seqaijMatMultAdd_SeqAIJ(Mat,Vec,Vec,Vec);

#undef __FUNC__  
#define __FUNC__ /*<a name="MatMult_SeqMAIJ_1"></a>*/"MatMult_SeqMAIJ_1"
int MatMult_SeqMAIJ_1(Mat A,Vec xx,Vec yy)
{
  Mat_SeqMAIJ *b = (Mat_SeqMAIJ*)A->data;
  int         ierr;
  PetscFunctionBegin;
  ierr = MatMult_SeqAIJ(b->AIJ,xx,yy);
  PetscFunctionReturn(0);
}
#undef __FUNC__  
#define __FUNC__ /*<a name="MatMultTranspose_SeqMAIJ_1"></a>*/"MatMultTranspose_SeqMAIJ_1"
int MatMultTranspose_SeqMAIJ_1(Mat A,Vec xx,Vec yy)
{
  Mat_SeqMAIJ *b = (Mat_SeqMAIJ*)A->data;
  int         ierr;
  PetscFunctionBegin;
  ierr = MatMultTranspose_SeqAIJ(b->AIJ,xx,yy);
  PetscFunctionReturn(0);
}
#undef __FUNC__  
#define __FUNC__ /*<a name="MatMultAdd_SeqMAIJ_1"></a>*/"MatMultAdd_SeqMAIJ_1"
int MatMultAdd_SeqMAIJ_1(Mat A,Vec xx,Vec yy,Vec zz)
{
  Mat_SeqMAIJ *b = (Mat_SeqMAIJ*)A->data;
  int         ierr;
  PetscFunctionBegin;
  ierr = MatMultAdd_SeqAIJ(b->AIJ,xx,yy,zz);
  PetscFunctionReturn(0);
}
#undef __FUNC__  
#define __FUNC__ /*<a name="MatMultTransposeAdd_SeqMAIJ_1"></a>*/"MatMultTransposeAdd_SeqMAIJ_1"
int MatMultTransposeAdd_SeqMAIJ_1(Mat A,Vec xx,Vec yy,Vec zz)
{
  Mat_SeqMAIJ *b = (Mat_SeqMAIJ*)A->data;
  int         ierr;
  PetscFunctionBegin;
  ierr = MatMultTransposeAdd_SeqAIJ(b->AIJ,xx,yy,zz);
  PetscFunctionReturn(0);
}

/* --------------------------------------------------------------------------------------*/
#undef __FUNC__  
#define __FUNC__ /*<a name="MatMult_SeqMAIJ_2"></a>*/"MatMult_SeqMAIJ_2"
int MatMult_SeqMAIJ_2(Mat A,Vec xx,Vec yy)
{
  Mat_SeqMAIJ *b = (Mat_SeqMAIJ*)A->data;
  int         ierr;
  PetscFunctionBegin;

  PetscFunctionReturn(0);
}
#undef __FUNC__  
#define __FUNC__ /*<a name="MatMultTranspose_SeqMAIJ_2"></a>*/"MatMultTranspose_SeqMAIJ_2"
int MatMultTranspose_SeqMAIJ_2(Mat A,Vec xx,Vec yy)
{
  Mat_SeqMAIJ *b = (Mat_SeqMAIJ*)A->data;
  int         ierr;
  PetscFunctionBegin;

  PetscFunctionReturn(0);
}
#undef __FUNC__  
#define __FUNC__ /*<a name="MatMultAdd_SeqMAIJ_2"></a>*/"MatMultAdd_SeqMAIJ_2"
int MatMultAdd_SeqMAIJ_2(Mat A,Vec xx,Vec yy,Vec zz)
{
  Mat_SeqMAIJ *b = (Mat_SeqMAIJ*)A->data;
  int         ierr;
  PetscFunctionBegin;

  PetscFunctionReturn(0);
}
#undef __FUNC__  
#define __FUNC__ /*<a name="MatMultTransposeAdd_SeqMAIJ_2"></a>*/"MatMultTransposeAdd_SeqMAIJ_2"
int MatMultTransposeAdd_SeqMAIJ_2(Mat A,Vec xx,Vec yy,Vec zz)
{
  Mat_SeqMAIJ *b = (Mat_SeqMAIJ*)A->data;
  int         ierr;
  PetscFunctionBegin;

  PetscFunctionReturn(0);
}
/* --------------------------------------------------------------------------------------*/

#undef __FUNC__  
#define __FUNC__ /*<a name="MatCreateMAIJ"></a>*/"MatCreateMAIJ" 
int MatCreateMAIJ(Mat A,int dof,Mat *maij)
{
  int         ierr;
  Mat_SeqMAIJ *b;
  Mat         B;

  PetscFunctionBegin;
  ierr = MATCreate(A->comm,dof*A->m,dof*A->n,dof*A->M,dof*A->N,&B);CHKERRQ(ierr);
  ierr = MatSetType(B,MATSEQMAIJ);CHKERRQ(ierr);

  B->assembled    = PETSC_TRUE;
  B->ops->destroy = MatDestroy_SeqMAIJ;
  b = (Mat_SeqMAIJ*)B->data;

  b->AIJ = A;
  b->dof = dof;
  ierr = PetscObjectReference((PetscObject)A);CHKERRQ(ierr);
  if (dof == 1) {
    B->ops->mult             = MatMult_SeqMAIJ_1;
    B->ops->multadd          = MatMultAdd_SeqMAIJ_1;
    B->ops->multtranspose    = MatMultTranspose_SeqMAIJ_1;
    B->ops->multtransposeadd = MatMultTransposeAdd_SeqMAIJ_1;
  } else if (dof == 2) {
    B->ops->mult             = MatMult_SeqMAIJ_2;
    B->ops->multadd          = MatMultAdd_SeqMAIJ_2;
    B->ops->multtranspose    = MatMultTranspose_SeqMAIJ_2;
    B->ops->multtransposeadd = MatMultTransposeAdd_SeqMAIJ_2;
  } else {
    SETERRQ1(1,1,"Cannot handle a dof of %d\n",dof);
  }
  *maij = B;
  PetscFunctionReturn(0);
}












