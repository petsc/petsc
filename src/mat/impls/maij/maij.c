/*$Id: maij.c,v 1.2 2000/05/27 03:52:46 bsmith Exp bsmith $*/
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

/* ----------------------------------------------------------------------------------*/
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
  Mat_SeqAIJ  *a = (Mat_SeqAIJ*)b->AIJ->data;
  Scalar      *x,*y,*v,sum1, sum2;
  int         ierr,m = a->m,*idx,shift = a->indexshift,*ii;
  int         n,i,jrow,j;

  PetscFunctionBegin;
  ierr = VecGetArray(xx,&x);CHKERRQ(ierr);
  ierr = VecGetArray(yy,&y);CHKERRQ(ierr);
  x    = x + shift;    /* shift for Fortran start by 1 indexing */
  idx  = a->j;
  v    = a->a;
  ii   = a->i;

  v    += shift; /* shift for Fortran start by 1 indexing */
  idx  += shift;
  for (i=0; i<m; i++) {
    jrow = ii[i];
    n    = ii[i+1] - jrow;
    sum1  = 0.0;
    sum2  = 0.0;
    for (j=0; j<n; j++) {
      sum1 += v[jrow]*x[2*idx[jrow]];
      sum2 += v[jrow]*x[2*idx[jrow]+1];
      jrow++;
     }
    y[2*i]   = sum1;
    y[2*i+1] = sum2;
  }

  PLogFlops(4*a->nz - 2*m);
  ierr = VecRestoreArray(xx,&x);CHKERRQ(ierr);
  ierr = VecRestoreArray(yy,&y);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ /*<a name="MatMultTranspose_SeqMAIJ_2"></a>*/"MatMultTranspose_SeqMAIJ_2"
int MatMultTranspose_SeqMAIJ_2(Mat A,Vec xx,Vec yy)
{
  Mat_SeqMAIJ *b = (Mat_SeqMAIJ*)A->data;
  Mat_SeqAIJ  *a = (Mat_SeqAIJ*)b->AIJ->data;
  Scalar     *x,*y,*v,alpha1,alpha2,zero = 0.0;
  int        ierr,m = a->m,n,i,*idx,shift = a->indexshift;

  PetscFunctionBegin; 
  ierr = VecSet(&zero,yy);CHKERRQ(ierr);
  ierr = VecGetArray(xx,&x);CHKERRQ(ierr);
  ierr = VecGetArray(yy,&y);CHKERRQ(ierr);
  y = y + shift; /* shift for Fortran start by 1 indexing */
  for (i=0; i<m; i++) {
    idx    = a->j + a->i[i] + shift;
    v      = a->a + a->i[i] + shift;
    n      = a->i[i+1] - a->i[i];
    alpha1 = x[2*i];
    alpha2 = x[2*i+1];
    while (n-->0) {y[2*(*idx)] += alpha1*(*v); y[2*(*idx)+1] += alpha2*(*v); idx++; v++;}
  }
  PLogFlops(4*a->nz - 2*a->n);
  ierr = VecRestoreArray(xx,&x);CHKERRQ(ierr);
  ierr = VecRestoreArray(yy,&y);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ /*<a name="MatMultAdd_SeqMAIJ_2"></a>*/"MatMultAdd_SeqMAIJ_2"
int MatMultAdd_SeqMAIJ_2(Mat A,Vec xx,Vec yy,Vec zz)
{
  Mat_SeqMAIJ *b = (Mat_SeqMAIJ*)A->data;
  Mat_SeqAIJ  *a = (Mat_SeqAIJ*)b->AIJ->data;
  Scalar      *x,*y,*v,sum1, sum2;
  int         ierr,m = a->m,*idx,shift = a->indexshift,*ii;
  int         n,i,jrow,j;

  PetscFunctionBegin;
  ierr = VecGetArray(xx,&x);CHKERRQ(ierr);
  ierr = VecGetArray(yy,&y);CHKERRQ(ierr);
  x    = x + shift;    /* shift for Fortran start by 1 indexing */
  idx  = a->j;
  v    = a->a;
  ii   = a->i;

  v    += shift; /* shift for Fortran start by 1 indexing */
  idx  += shift;
  for (i=0; i<m; i++) {
    jrow = ii[i];
    n    = ii[i+1] - jrow;
    sum1  = 0.0;
    sum2  = 0.0;
    for (j=0; j<n; j++) {
      sum1 += v[jrow]*x[2*idx[jrow]];
      sum2 += v[jrow]*x[2*idx[jrow]+1];
      jrow++;
     }
    y[2*i]   += sum1;
    y[2*i+1] += sum2;
  }

  PLogFlops(4*a->nz - 2*m);
  ierr = VecRestoreArray(xx,&x);CHKERRQ(ierr);
  ierr = VecRestoreArray(yy,&y);CHKERRQ(ierr);
  PetscFunctionReturn(0);
  PetscFunctionReturn(0);
}
#undef __FUNC__  
#define __FUNC__ /*<a name="MatMultTransposeAdd_SeqMAIJ_2"></a>*/"MatMultTransposeAdd_SeqMAIJ_2"
int MatMultTransposeAdd_SeqMAIJ_2(Mat A,Vec xx,Vec yy,Vec zz)
{
  Mat_SeqMAIJ *b = (Mat_SeqMAIJ*)A->data;
  Mat_SeqAIJ  *a = (Mat_SeqAIJ*)b->AIJ->data;
  Scalar     *x,*y,*v,alpha1,alpha2;
  int        ierr,m = a->m,n,i,*idx,shift = a->indexshift;

  PetscFunctionBegin; 
  ierr = VecGetArray(xx,&x);CHKERRQ(ierr);
  ierr = VecGetArray(yy,&y);CHKERRQ(ierr);
  y = y + shift; /* shift for Fortran start by 1 indexing */
  for (i=0; i<m; i++) {
    idx   = a->j + a->i[i] + shift;
    v     = a->a + a->i[i] + shift;
    n     = a->i[i+1] - a->i[i];
    alpha1 = x[2*i];
    alpha2 = x[2*i+1];
    while (n-->0) {y[2*(*idx)] += alpha1*(*v); y[2*(*idx)+1] += alpha2*(*v); idx++; v++;}
  }
  PLogFlops(4*a->nz - 2*a->n);
  ierr = VecRestoreArray(xx,&x);CHKERRQ(ierr);
  ierr = VecRestoreArray(yy,&y);CHKERRQ(ierr);
  PetscFunctionReturn(0);
  PetscFunctionReturn(0);
}
/* --------------------------------------------------------------------------------------*/
#undef __FUNC__  
#define __FUNC__ /*<a name="MatMult_SeqMAIJ_3"></a>*/"MatMult_SeqMAIJ_3"
int MatMult_SeqMAIJ_3(Mat A,Vec xx,Vec yy)
{
  Mat_SeqMAIJ *b = (Mat_SeqMAIJ*)A->data;
  Mat_SeqAIJ  *a = (Mat_SeqAIJ*)b->AIJ->data;
  Scalar      *x,*y,*v,sum1, sum2, sum3;
  int         ierr,m = a->m,*idx,shift = a->indexshift,*ii;
  int         n,i,jrow,j;

  PetscFunctionBegin;
  ierr = VecGetArray(xx,&x);CHKERRQ(ierr);
  ierr = VecGetArray(yy,&y);CHKERRQ(ierr);
  x    = x + shift;    /* shift for Fortran start by 1 indexing */
  idx  = a->j;
  v    = a->a;
  ii   = a->i;

  v    += shift; /* shift for Fortran start by 1 indexing */
  idx  += shift;
  for (i=0; i<m; i++) {
    jrow = ii[i];
    n    = ii[i+1] - jrow;
    sum1  = 0.0;
    sum2  = 0.0;
    sum3  = 0.0;
    for (j=0; j<n; j++) {
      sum1 += v[jrow]*x[3*idx[jrow]];
      sum2 += v[jrow]*x[3*idx[jrow]+1];
      sum3 += v[jrow]*x[3*idx[jrow]+2];
      jrow++;
     }
    y[3*i]   = sum1;
    y[3*i+1] = sum2;
    y[3*i+2] = sum3;
  }

  PLogFlops(6*a->nz - 3*m);
  ierr = VecRestoreArray(xx,&x);CHKERRQ(ierr);
  ierr = VecRestoreArray(yy,&y);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ /*<a name="MatMultTranspose_SeqMAIJ_3"></a>*/"MatMultTranspose_SeqMAIJ_3"
int MatMultTranspose_SeqMAIJ_3(Mat A,Vec xx,Vec yy)
{
  Mat_SeqMAIJ *b = (Mat_SeqMAIJ*)A->data;
  Mat_SeqAIJ  *a = (Mat_SeqAIJ*)b->AIJ->data;
  Scalar     *x,*y,*v,alpha1,alpha2,alpha3,zero = 0.0;
  int        ierr,m = a->m,n,i,*idx,shift = a->indexshift;

  PetscFunctionBegin; 
  ierr = VecSet(&zero,yy);CHKERRQ(ierr);
  ierr = VecGetArray(xx,&x);CHKERRQ(ierr);
  ierr = VecGetArray(yy,&y);CHKERRQ(ierr);
  y = y + shift; /* shift for Fortran start by 1 indexing */
  for (i=0; i<m; i++) {
    idx    = a->j + a->i[i] + shift;
    v      = a->a + a->i[i] + shift;
    n      = a->i[i+1] - a->i[i];
    alpha1 = x[3*i];
    alpha2 = x[3*i+1];
    alpha3 = x[3*i+2];
    while (n-->0) {
      y[3*(*idx)]   += alpha1*(*v);
      y[3*(*idx)+1] += alpha2*(*v);
      y[3*(*idx)+2] += alpha3*(*v);
      idx++; v++;
    }
  }
  PLogFlops(6*a->nz - 3*a->n);
  ierr = VecRestoreArray(xx,&x);CHKERRQ(ierr);
  ierr = VecRestoreArray(yy,&y);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ /*<a name="MatMultAdd_SeqMAIJ_3"></a>*/"MatMultAdd_SeqMAIJ_3"
int MatMultAdd_SeqMAIJ_3(Mat A,Vec xx,Vec yy,Vec zz)
{
  Mat_SeqMAIJ *b = (Mat_SeqMAIJ*)A->data;
  Mat_SeqAIJ  *a = (Mat_SeqAIJ*)b->AIJ->data;
  Scalar      *x,*y,*v,sum1, sum2, sum3;
  int         ierr,m = a->m,*idx,shift = a->indexshift,*ii;
  int         n,i,jrow,j;

  PetscFunctionBegin;
  ierr = VecGetArray(xx,&x);CHKERRQ(ierr);
  ierr = VecGetArray(yy,&y);CHKERRQ(ierr);
  x    = x + shift;    /* shift for Fortran start by 1 indexing */
  idx  = a->j;
  v    = a->a;
  ii   = a->i;

  v    += shift; /* shift for Fortran start by 1 indexing */
  idx  += shift;
  for (i=0; i<m; i++) {
    jrow = ii[i];
    n    = ii[i+1] - jrow;
    sum1  = 0.0;
    sum2  = 0.0;
    sum3  = 0.0;
    for (j=0; j<n; j++) {
      sum1 += v[jrow]*x[3*idx[jrow]];
      sum2 += v[jrow]*x[3*idx[jrow]+1];
      sum3 += v[jrow]*x[3*idx[jrow]+2];
      jrow++;
     }
    y[3*i]   += sum1;
    y[3*i+1] += sum2;
    y[3*i+2] += sum3;
  }

  PLogFlops(6*a->nz);
  ierr = VecRestoreArray(xx,&x);CHKERRQ(ierr);
  ierr = VecRestoreArray(yy,&y);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
#undef __FUNC__  
#define __FUNC__ /*<a name="MatMultTransposeAdd_SeqMAIJ_3"></a>*/"MatMultTransposeAdd_SeqMAIJ_3"
int MatMultTransposeAdd_SeqMAIJ_3(Mat A,Vec xx,Vec yy,Vec zz)
{
  Mat_SeqMAIJ *b = (Mat_SeqMAIJ*)A->data;
  Mat_SeqAIJ  *a = (Mat_SeqAIJ*)b->AIJ->data;
  Scalar     *x,*y,*v,alpha1,alpha2,alpha3;
  int        ierr,m = a->m,n,i,*idx,shift = a->indexshift;

  PetscFunctionBegin; 
  ierr = VecGetArray(xx,&x);CHKERRQ(ierr);
  ierr = VecGetArray(yy,&y);CHKERRQ(ierr);
  y = y + shift; /* shift for Fortran start by 1 indexing */
  for (i=0; i<m; i++) {
    idx    = a->j + a->i[i] + shift;
    v      = a->a + a->i[i] + shift;
    n      = a->i[i+1] - a->i[i];
    alpha1 = x[3*i];
    alpha2 = x[3*i+1];
    alpha3 = x[3*i+2];
    while (n-->0) {
      y[3*(*idx)]   += alpha1*(*v);
      y[3*(*idx)+1] += alpha2*(*v);
      y[3*(*idx)+2] += alpha3*(*v);
      idx++; v++;
    }
  }
  PLogFlops(6*a->nz);
  ierr = VecRestoreArray(xx,&x);CHKERRQ(ierr);
  ierr = VecRestoreArray(yy,&y);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* ------------------------------------------------------------------------------*/
#undef __FUNC__  
#define __FUNC__ /*<a name="MatMult_SeqMAIJ_4"></a>*/"MatMult_SeqMAIJ_4"
int MatMult_SeqMAIJ_4(Mat A,Vec xx,Vec yy)
{
  Mat_SeqMAIJ *b = (Mat_SeqMAIJ*)A->data;
  Mat_SeqAIJ  *a = (Mat_SeqAIJ*)b->AIJ->data;
  Scalar      *x,*y,*v,sum1, sum2, sum3, sum4;
  int         ierr,m = a->m,*idx,shift = a->indexshift,*ii;
  int         n,i,jrow,j;

  PetscFunctionBegin;
  ierr = VecGetArray(xx,&x);CHKERRQ(ierr);
  ierr = VecGetArray(yy,&y);CHKERRQ(ierr);
  x    = x + shift;    /* shift for Fortran start by 1 indexing */
  idx  = a->j;
  v    = a->a;
  ii   = a->i;

  v    += shift; /* shift for Fortran start by 1 indexing */
  idx  += shift;
  for (i=0; i<m; i++) {
    jrow = ii[i];
    n    = ii[i+1] - jrow;
    sum1  = 0.0;
    sum2  = 0.0;
    sum3  = 0.0;
    sum4  = 0.0;
    for (j=0; j<n; j++) {
      sum1 += v[jrow]*x[4*idx[jrow]];
      sum2 += v[jrow]*x[4*idx[jrow]+1];
      sum3 += v[jrow]*x[4*idx[jrow]+2];
      sum4 += v[jrow]*x[4*idx[jrow]+3];
      jrow++;
     }
    y[4*i]   = sum1;
    y[4*i+1] = sum2;
    y[4*i+2] = sum3;
    y[4*i+3] = sum4;
  }

  PLogFlops(8*a->nz - 4*m);
  ierr = VecRestoreArray(xx,&x);CHKERRQ(ierr);
  ierr = VecRestoreArray(yy,&y);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ /*<a name="MatMultTranspose_SeqMAIJ_4"></a>*/"MatMultTranspose_SeqMAIJ_4"
int MatMultTranspose_SeqMAIJ_4(Mat A,Vec xx,Vec yy)
{
  Mat_SeqMAIJ *b = (Mat_SeqMAIJ*)A->data;
  Mat_SeqAIJ  *a = (Mat_SeqAIJ*)b->AIJ->data;
  Scalar     *x,*y,*v,alpha1,alpha2,alpha3,alpha4,zero = 0.0;
  int        ierr,m = a->m,n,i,*idx,shift = a->indexshift;

  PetscFunctionBegin; 
  ierr = VecSet(&zero,yy);CHKERRQ(ierr);
  ierr = VecGetArray(xx,&x);CHKERRQ(ierr);
  ierr = VecGetArray(yy,&y);CHKERRQ(ierr);
  y = y + shift; /* shift for Fortran start by 1 indexing */
  for (i=0; i<m; i++) {
    idx    = a->j + a->i[i] + shift;
    v      = a->a + a->i[i] + shift;
    n      = a->i[i+1] - a->i[i];
    alpha1 = x[4*i];
    alpha2 = x[4*i+1];
    alpha3 = x[4*i+2];
    alpha4 = x[4*i+3];
    while (n-->0) {
      y[4*(*idx)]   += alpha1*(*v);
      y[4*(*idx)+1] += alpha2*(*v);
      y[4*(*idx)+2] += alpha3*(*v);
      y[4*(*idx)+3] += alpha4*(*v);
      idx++; v++;
    }
  }
  PLogFlops(8*a->nz - 4*a->n);
  ierr = VecRestoreArray(xx,&x);CHKERRQ(ierr);
  ierr = VecRestoreArray(yy,&y);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ /*<a name="MatMultAdd_SeqMAIJ_4"></a>*/"MatMultAdd_SeqMAIJ_4"
int MatMultAdd_SeqMAIJ_4(Mat A,Vec xx,Vec yy,Vec zz)
{
return 0;
}
#undef __FUNC__  
#define __FUNC__ /*<a name="MatMultTransposeAdd_SeqMAIJ_4"></a>*/"MatMultTransposeAdd_SeqMAIJ_4"
int MatMultTransposeAdd_SeqMAIJ_4(Mat A,Vec xx,Vec yy,Vec zz)
{
return 0;
}

/* ---------------------------------------------------------------------------------- */
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
  } else if (dof == 3) {
    B->ops->mult             = MatMult_SeqMAIJ_3;
    B->ops->multadd          = MatMultAdd_SeqMAIJ_3;
    B->ops->multtranspose    = MatMultTranspose_SeqMAIJ_3;
    B->ops->multtransposeadd = MatMultTransposeAdd_SeqMAIJ_3;
  } else if (dof == 4) {
    B->ops->mult             = MatMult_SeqMAIJ_4;
    B->ops->multadd          = MatMultAdd_SeqMAIJ_4;
    B->ops->multtranspose    = MatMultTranspose_SeqMAIJ_4;
    B->ops->multtransposeadd = MatMultTransposeAdd_SeqMAIJ_4;
  } else {
    SETERRQ1(1,1,"Cannot handle a dof of %d\n",dof);
  }
  *maij = B;
  PetscFunctionReturn(0);
}












