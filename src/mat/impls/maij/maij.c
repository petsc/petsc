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

     This single directory handles both the sequential and parallel codes
*/

#include "src/mat/impls/maij/maij.h"
#include "src/mat/utils/freespace.h"
#include "vecimpl.h"

#undef __FUNCT__  
#define __FUNCT__ "MatMAIJGetAIJ" 
PetscErrorCode MatMAIJGetAIJ(Mat A,Mat *B)
{
  PetscErrorCode ierr;
  PetscTruth     ismpimaij,isseqmaij;

  PetscFunctionBegin;
  ierr = PetscTypeCompare((PetscObject)A,MATMPIMAIJ,&ismpimaij);CHKERRQ(ierr);  
  ierr = PetscTypeCompare((PetscObject)A,MATSEQMAIJ,&isseqmaij);CHKERRQ(ierr);  
  if (ismpimaij) {
    Mat_MPIMAIJ *b = (Mat_MPIMAIJ*)A->data;

    *B = b->A;
  } else if (isseqmaij) {
    Mat_SeqMAIJ *b = (Mat_SeqMAIJ*)A->data;

    *B = b->AIJ;
  } else {
    *B = A;
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatMAIJRedimension" 
PetscErrorCode MatMAIJRedimension(Mat A,PetscInt dof,Mat *B)
{
  PetscErrorCode ierr;
  Mat            Aij;

  PetscFunctionBegin;
  ierr = MatMAIJGetAIJ(A,&Aij);CHKERRQ(ierr);
  ierr = MatCreateMAIJ(Aij,dof,B);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatDestroy_SeqMAIJ" 
PetscErrorCode MatDestroy_SeqMAIJ(Mat A)
{
  PetscErrorCode ierr;
  Mat_SeqMAIJ    *b = (Mat_SeqMAIJ*)A->data;

  PetscFunctionBegin;
  if (b->AIJ) {
    ierr = MatDestroy(b->AIJ);CHKERRQ(ierr);
  }
  ierr = PetscFree(b);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatView_SeqMAIJ" 
PetscErrorCode MatView_SeqMAIJ(Mat A,PetscViewer viewer)
{
  PetscErrorCode ierr;
  Mat            B;

  PetscFunctionBegin;
  ierr = MatConvert(A,MATSEQAIJ,MAT_INITIAL_MATRIX,&B);
  ierr = MatView(B,viewer);CHKERRQ(ierr);
  ierr = MatDestroy(B);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatView_MPIMAIJ" 
PetscErrorCode MatView_MPIMAIJ(Mat A,PetscViewer viewer)
{
  PetscErrorCode ierr;
  Mat            B;

  PetscFunctionBegin;
  ierr = MatConvert(A,MATMPIAIJ,MAT_INITIAL_MATRIX,&B);
  ierr = MatView(B,viewer);CHKERRQ(ierr);
  ierr = MatDestroy(B);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatDestroy_MPIMAIJ" 
PetscErrorCode MatDestroy_MPIMAIJ(Mat A)
{
  PetscErrorCode ierr;
  Mat_MPIMAIJ    *b = (Mat_MPIMAIJ*)A->data;

  PetscFunctionBegin;
  if (b->AIJ) {
    ierr = MatDestroy(b->AIJ);CHKERRQ(ierr);
  }
  if (b->OAIJ) {
    ierr = MatDestroy(b->OAIJ);CHKERRQ(ierr);
  }
  if (b->A) {
    ierr = MatDestroy(b->A);CHKERRQ(ierr);
  }
  if (b->ctx) {
    ierr = VecScatterDestroy(b->ctx);CHKERRQ(ierr);
  }
  if (b->w) {
    ierr = VecDestroy(b->w);CHKERRQ(ierr);
  }
  ierr = PetscFree(b);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*MC
  MATMAIJ - MATMAIJ = "maij" - A matrix type to be used for restriction and interpolation operations for 
  multicomponent problems, interpolating or restricting each component the same way independently.
  The matrix type is based on MATSEQAIJ for sequential matrices, and MATMPIAIJ for distributed matrices.

  Operations provided:
. MatMult
. MatMultTranspose
. MatMultAdd
. MatMultTransposeAdd

  Level: advanced

.seealso: MatCreateSeqDense
M*/

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "MatCreate_MAIJ" 
PetscErrorCode MatCreate_MAIJ(Mat A)
{
  PetscErrorCode ierr;
  Mat_MPIMAIJ    *b;

  PetscFunctionBegin;
  ierr     = PetscNew(Mat_MPIMAIJ,&b);CHKERRQ(ierr);
  A->data  = (void*)b;
  ierr = PetscMemzero(A->ops,sizeof(struct _MatOps));CHKERRQ(ierr);
  A->factor           = 0;
  A->mapping          = 0;

  b->AIJ  = 0;
  b->dof  = 0;  
  b->OAIJ = 0;
  b->ctx  = 0;
  b->w    = 0;  
  PetscFunctionReturn(0);
}
EXTERN_C_END

/* --------------------------------------------------------------------------------------*/
#undef __FUNCT__  
#define __FUNCT__ "MatMult_SeqMAIJ_2"
PetscErrorCode MatMult_SeqMAIJ_2(Mat A,Vec xx,Vec yy)
{
  Mat_SeqMAIJ    *b = (Mat_SeqMAIJ*)A->data;
  Mat_SeqAIJ     *a = (Mat_SeqAIJ*)b->AIJ->data;
  PetscScalar    *x,*y,*v,sum1, sum2;
  PetscErrorCode ierr;
  PetscInt       m = b->AIJ->m,*idx,*ii;
  PetscInt       n,i,jrow,j;

  PetscFunctionBegin;
  ierr = VecGetArray(xx,&x);CHKERRQ(ierr);
  ierr = VecGetArray(yy,&y);CHKERRQ(ierr);
  idx  = a->j;
  v    = a->a;
  ii   = a->i;

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

  PetscLogFlops(4*a->nz - 2*m);
  ierr = VecRestoreArray(xx,&x);CHKERRQ(ierr);
  ierr = VecRestoreArray(yy,&y);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatMultTranspose_SeqMAIJ_2"
PetscErrorCode MatMultTranspose_SeqMAIJ_2(Mat A,Vec xx,Vec yy)
{
  Mat_SeqMAIJ    *b = (Mat_SeqMAIJ*)A->data;
  Mat_SeqAIJ     *a = (Mat_SeqAIJ*)b->AIJ->data;
  PetscScalar    *x,*y,*v,alpha1,alpha2,zero = 0.0;
  PetscErrorCode ierr;
  PetscInt       m = b->AIJ->m,n,i,*idx;

  PetscFunctionBegin; 
  ierr = VecSet(&zero,yy);CHKERRQ(ierr);
  ierr = VecGetArray(xx,&x);CHKERRQ(ierr);
  ierr = VecGetArray(yy,&y);CHKERRQ(ierr);
 
  for (i=0; i<m; i++) {
    idx    = a->j + a->i[i] ;
    v      = a->a + a->i[i] ;
    n      = a->i[i+1] - a->i[i];
    alpha1 = x[2*i];
    alpha2 = x[2*i+1];
    while (n-->0) {y[2*(*idx)] += alpha1*(*v); y[2*(*idx)+1] += alpha2*(*v); idx++; v++;}
  }
  PetscLogFlops(4*a->nz - 2*b->AIJ->n);
  ierr = VecRestoreArray(xx,&x);CHKERRQ(ierr);
  ierr = VecRestoreArray(yy,&y);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatMultAdd_SeqMAIJ_2"
PetscErrorCode MatMultAdd_SeqMAIJ_2(Mat A,Vec xx,Vec yy,Vec zz)
{
  Mat_SeqMAIJ    *b = (Mat_SeqMAIJ*)A->data;
  Mat_SeqAIJ     *a = (Mat_SeqAIJ*)b->AIJ->data;
  PetscScalar    *x,*y,*v,sum1, sum2;
  PetscErrorCode ierr;
  PetscInt       m = b->AIJ->m,*idx,*ii;
  PetscInt       n,i,jrow,j;

  PetscFunctionBegin;
  if (yy != zz) {ierr = VecCopy(yy,zz);CHKERRQ(ierr);}
  ierr = VecGetArray(xx,&x);CHKERRQ(ierr);
  ierr = VecGetArray(zz,&y);CHKERRQ(ierr);
  idx  = a->j;
  v    = a->a;
  ii   = a->i;

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

  PetscLogFlops(4*a->nz - 2*m);
  ierr = VecRestoreArray(xx,&x);CHKERRQ(ierr);
  ierr = VecRestoreArray(zz,&y);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
#undef __FUNCT__  
#define __FUNCT__ "MatMultTransposeAdd_SeqMAIJ_2"
PetscErrorCode MatMultTransposeAdd_SeqMAIJ_2(Mat A,Vec xx,Vec yy,Vec zz)
{
  Mat_SeqMAIJ    *b = (Mat_SeqMAIJ*)A->data;
  Mat_SeqAIJ     *a = (Mat_SeqAIJ*)b->AIJ->data;
  PetscScalar    *x,*y,*v,alpha1,alpha2;
  PetscErrorCode ierr;
  PetscInt       m = b->AIJ->m,n,i,*idx;

  PetscFunctionBegin; 
  if (yy != zz) {ierr = VecCopy(yy,zz);CHKERRQ(ierr);}
  ierr = VecGetArray(xx,&x);CHKERRQ(ierr);
  ierr = VecGetArray(zz,&y);CHKERRQ(ierr);
 
  for (i=0; i<m; i++) {
    idx   = a->j + a->i[i] ;
    v     = a->a + a->i[i] ;
    n     = a->i[i+1] - a->i[i];
    alpha1 = x[2*i];
    alpha2 = x[2*i+1];
    while (n-->0) {y[2*(*idx)] += alpha1*(*v); y[2*(*idx)+1] += alpha2*(*v); idx++; v++;}
  }
  PetscLogFlops(4*a->nz - 2*b->AIJ->n);
  ierr = VecRestoreArray(xx,&x);CHKERRQ(ierr);
  ierr = VecRestoreArray(zz,&y);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
/* --------------------------------------------------------------------------------------*/
#undef __FUNCT__  
#define __FUNCT__ "MatMult_SeqMAIJ_3"
PetscErrorCode MatMult_SeqMAIJ_3(Mat A,Vec xx,Vec yy)
{
  Mat_SeqMAIJ    *b = (Mat_SeqMAIJ*)A->data;
  Mat_SeqAIJ     *a = (Mat_SeqAIJ*)b->AIJ->data;
  PetscScalar    *x,*y,*v,sum1, sum2, sum3;
  PetscErrorCode ierr;
  PetscInt       m = b->AIJ->m,*idx,*ii;
  PetscInt       n,i,jrow,j;

  PetscFunctionBegin;
  ierr = VecGetArray(xx,&x);CHKERRQ(ierr);
  ierr = VecGetArray(yy,&y);CHKERRQ(ierr);
  idx  = a->j;
  v    = a->a;
  ii   = a->i;

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

  PetscLogFlops(6*a->nz - 3*m);
  ierr = VecRestoreArray(xx,&x);CHKERRQ(ierr);
  ierr = VecRestoreArray(yy,&y);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatMultTranspose_SeqMAIJ_3"
PetscErrorCode MatMultTranspose_SeqMAIJ_3(Mat A,Vec xx,Vec yy)
{
  Mat_SeqMAIJ    *b = (Mat_SeqMAIJ*)A->data;
  Mat_SeqAIJ     *a = (Mat_SeqAIJ*)b->AIJ->data;
  PetscScalar    *x,*y,*v,alpha1,alpha2,alpha3,zero = 0.0;
  PetscErrorCode ierr;
  PetscInt       m = b->AIJ->m,n,i,*idx;

  PetscFunctionBegin; 
  ierr = VecSet(&zero,yy);CHKERRQ(ierr);
  ierr = VecGetArray(xx,&x);CHKERRQ(ierr);
  ierr = VecGetArray(yy,&y);CHKERRQ(ierr);
  
  for (i=0; i<m; i++) {
    idx    = a->j + a->i[i];
    v      = a->a + a->i[i];
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
  PetscLogFlops(6*a->nz - 3*b->AIJ->n);
  ierr = VecRestoreArray(xx,&x);CHKERRQ(ierr);
  ierr = VecRestoreArray(yy,&y);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatMultAdd_SeqMAIJ_3"
PetscErrorCode MatMultAdd_SeqMAIJ_3(Mat A,Vec xx,Vec yy,Vec zz)
{
  Mat_SeqMAIJ    *b = (Mat_SeqMAIJ*)A->data;
  Mat_SeqAIJ     *a = (Mat_SeqAIJ*)b->AIJ->data;
  PetscScalar    *x,*y,*v,sum1, sum2, sum3;
  PetscErrorCode ierr;
  PetscInt       m = b->AIJ->m,*idx,*ii;
  PetscInt       n,i,jrow,j;

  PetscFunctionBegin;
  if (yy != zz) {ierr = VecCopy(yy,zz);CHKERRQ(ierr);}
  ierr = VecGetArray(xx,&x);CHKERRQ(ierr);
  ierr = VecGetArray(zz,&y);CHKERRQ(ierr);
  idx  = a->j;
  v    = a->a;
  ii   = a->i;

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

  PetscLogFlops(6*a->nz);
  ierr = VecRestoreArray(xx,&x);CHKERRQ(ierr);
  ierr = VecRestoreArray(zz,&y);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
#undef __FUNCT__  
#define __FUNCT__ "MatMultTransposeAdd_SeqMAIJ_3"
PetscErrorCode MatMultTransposeAdd_SeqMAIJ_3(Mat A,Vec xx,Vec yy,Vec zz)
{
  Mat_SeqMAIJ    *b = (Mat_SeqMAIJ*)A->data;
  Mat_SeqAIJ     *a = (Mat_SeqAIJ*)b->AIJ->data;
  PetscScalar    *x,*y,*v,alpha1,alpha2,alpha3;
  PetscErrorCode ierr;
  PetscInt       m = b->AIJ->m,n,i,*idx;

  PetscFunctionBegin; 
  if (yy != zz) {ierr = VecCopy(yy,zz);CHKERRQ(ierr);}
  ierr = VecGetArray(xx,&x);CHKERRQ(ierr);
  ierr = VecGetArray(zz,&y);CHKERRQ(ierr);
  for (i=0; i<m; i++) {
    idx    = a->j + a->i[i] ;
    v      = a->a + a->i[i] ;
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
  PetscLogFlops(6*a->nz);
  ierr = VecRestoreArray(xx,&x);CHKERRQ(ierr);
  ierr = VecRestoreArray(zz,&y);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* ------------------------------------------------------------------------------*/
#undef __FUNCT__  
#define __FUNCT__ "MatMult_SeqMAIJ_4"
PetscErrorCode MatMult_SeqMAIJ_4(Mat A,Vec xx,Vec yy)
{
  Mat_SeqMAIJ    *b = (Mat_SeqMAIJ*)A->data;
  Mat_SeqAIJ     *a = (Mat_SeqAIJ*)b->AIJ->data;
  PetscScalar    *x,*y,*v,sum1, sum2, sum3, sum4;
  PetscErrorCode ierr;
  PetscInt       m = b->AIJ->m,*idx,*ii;
  PetscInt       n,i,jrow,j;

  PetscFunctionBegin;
  ierr = VecGetArray(xx,&x);CHKERRQ(ierr);
  ierr = VecGetArray(yy,&y);CHKERRQ(ierr);
  idx  = a->j;
  v    = a->a;
  ii   = a->i;

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

  PetscLogFlops(8*a->nz - 4*m);
  ierr = VecRestoreArray(xx,&x);CHKERRQ(ierr);
  ierr = VecRestoreArray(yy,&y);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatMultTranspose_SeqMAIJ_4"
PetscErrorCode MatMultTranspose_SeqMAIJ_4(Mat A,Vec xx,Vec yy)
{
  Mat_SeqMAIJ    *b = (Mat_SeqMAIJ*)A->data;
  Mat_SeqAIJ     *a = (Mat_SeqAIJ*)b->AIJ->data;
  PetscScalar    *x,*y,*v,alpha1,alpha2,alpha3,alpha4,zero = 0.0;
  PetscErrorCode ierr;
  PetscInt       m = b->AIJ->m,n,i,*idx;

  PetscFunctionBegin; 
  ierr = VecSet(&zero,yy);CHKERRQ(ierr);
  ierr = VecGetArray(xx,&x);CHKERRQ(ierr);
  ierr = VecGetArray(yy,&y);CHKERRQ(ierr);
  for (i=0; i<m; i++) {
    idx    = a->j + a->i[i] ;
    v      = a->a + a->i[i] ;
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
  PetscLogFlops(8*a->nz - 4*b->AIJ->n);
  ierr = VecRestoreArray(xx,&x);CHKERRQ(ierr);
  ierr = VecRestoreArray(yy,&y);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatMultAdd_SeqMAIJ_4"
PetscErrorCode MatMultAdd_SeqMAIJ_4(Mat A,Vec xx,Vec yy,Vec zz)
{
  Mat_SeqMAIJ    *b = (Mat_SeqMAIJ*)A->data;
  Mat_SeqAIJ     *a = (Mat_SeqAIJ*)b->AIJ->data;
  PetscScalar    *x,*y,*v,sum1, sum2, sum3, sum4;
  PetscErrorCode ierr;
  PetscInt       m = b->AIJ->m,*idx,*ii;
  PetscInt       n,i,jrow,j;

  PetscFunctionBegin;
  if (yy != zz) {ierr = VecCopy(yy,zz);CHKERRQ(ierr);}
  ierr = VecGetArray(xx,&x);CHKERRQ(ierr);
  ierr = VecGetArray(zz,&y);CHKERRQ(ierr);
  idx  = a->j;
  v    = a->a;
  ii   = a->i;

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
    y[4*i]   += sum1;
    y[4*i+1] += sum2;
    y[4*i+2] += sum3;
    y[4*i+3] += sum4;
  }

  PetscLogFlops(8*a->nz - 4*m);
  ierr = VecRestoreArray(xx,&x);CHKERRQ(ierr);
  ierr = VecRestoreArray(zz,&y);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
#undef __FUNCT__  
#define __FUNCT__ "MatMultTransposeAdd_SeqMAIJ_4"
PetscErrorCode MatMultTransposeAdd_SeqMAIJ_4(Mat A,Vec xx,Vec yy,Vec zz)
{
  Mat_SeqMAIJ    *b = (Mat_SeqMAIJ*)A->data;
  Mat_SeqAIJ     *a = (Mat_SeqAIJ*)b->AIJ->data;
  PetscScalar    *x,*y,*v,alpha1,alpha2,alpha3,alpha4;
  PetscErrorCode ierr;
  PetscInt       m = b->AIJ->m,n,i,*idx;

  PetscFunctionBegin; 
  if (yy != zz) {ierr = VecCopy(yy,zz);CHKERRQ(ierr);}
  ierr = VecGetArray(xx,&x);CHKERRQ(ierr);
  ierr = VecGetArray(zz,&y);CHKERRQ(ierr);
  
  for (i=0; i<m; i++) {
    idx    = a->j + a->i[i] ;
    v      = a->a + a->i[i] ;
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
  PetscLogFlops(8*a->nz - 4*b->AIJ->n);
  ierr = VecRestoreArray(xx,&x);CHKERRQ(ierr);
  ierr = VecRestoreArray(zz,&y);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
/* ------------------------------------------------------------------------------*/

#undef __FUNCT__  
#define __FUNCT__ "MatMult_SeqMAIJ_5"
PetscErrorCode MatMult_SeqMAIJ_5(Mat A,Vec xx,Vec yy)
{
  Mat_SeqMAIJ    *b = (Mat_SeqMAIJ*)A->data;
  Mat_SeqAIJ     *a = (Mat_SeqAIJ*)b->AIJ->data;
  PetscScalar    *x,*y,*v,sum1, sum2, sum3, sum4, sum5;
  PetscErrorCode ierr;
  PetscInt       m = b->AIJ->m,*idx,*ii;
  PetscInt       n,i,jrow,j;

  PetscFunctionBegin;
  ierr = VecGetArray(xx,&x);CHKERRQ(ierr);
  ierr = VecGetArray(yy,&y);CHKERRQ(ierr);
  idx  = a->j;
  v    = a->a;
  ii   = a->i;

  for (i=0; i<m; i++) {
    jrow = ii[i];
    n    = ii[i+1] - jrow;
    sum1  = 0.0;
    sum2  = 0.0;
    sum3  = 0.0;
    sum4  = 0.0;
    sum5  = 0.0;
    for (j=0; j<n; j++) {
      sum1 += v[jrow]*x[5*idx[jrow]];
      sum2 += v[jrow]*x[5*idx[jrow]+1];
      sum3 += v[jrow]*x[5*idx[jrow]+2];
      sum4 += v[jrow]*x[5*idx[jrow]+3];
      sum5 += v[jrow]*x[5*idx[jrow]+4];
      jrow++;
     }
    y[5*i]   = sum1;
    y[5*i+1] = sum2;
    y[5*i+2] = sum3;
    y[5*i+3] = sum4;
    y[5*i+4] = sum5;
  }

  PetscLogFlops(10*a->nz - 5*m);
  ierr = VecRestoreArray(xx,&x);CHKERRQ(ierr);
  ierr = VecRestoreArray(yy,&y);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatMultTranspose_SeqMAIJ_5"
PetscErrorCode MatMultTranspose_SeqMAIJ_5(Mat A,Vec xx,Vec yy)
{
  Mat_SeqMAIJ    *b = (Mat_SeqMAIJ*)A->data;
  Mat_SeqAIJ     *a = (Mat_SeqAIJ*)b->AIJ->data;
  PetscScalar    *x,*y,*v,alpha1,alpha2,alpha3,alpha4,alpha5,zero = 0.0;
  PetscErrorCode ierr;
  PetscInt       m = b->AIJ->m,n,i,*idx;

  PetscFunctionBegin; 
  ierr = VecSet(&zero,yy);CHKERRQ(ierr);
  ierr = VecGetArray(xx,&x);CHKERRQ(ierr);
  ierr = VecGetArray(yy,&y);CHKERRQ(ierr);
  
  for (i=0; i<m; i++) {
    idx    = a->j + a->i[i] ;
    v      = a->a + a->i[i] ;
    n      = a->i[i+1] - a->i[i];
    alpha1 = x[5*i];
    alpha2 = x[5*i+1];
    alpha3 = x[5*i+2];
    alpha4 = x[5*i+3];
    alpha5 = x[5*i+4];
    while (n-->0) {
      y[5*(*idx)]   += alpha1*(*v);
      y[5*(*idx)+1] += alpha2*(*v);
      y[5*(*idx)+2] += alpha3*(*v);
      y[5*(*idx)+3] += alpha4*(*v);
      y[5*(*idx)+4] += alpha5*(*v);
      idx++; v++;
    }
  }
  PetscLogFlops(10*a->nz - 5*b->AIJ->n);
  ierr = VecRestoreArray(xx,&x);CHKERRQ(ierr);
  ierr = VecRestoreArray(yy,&y);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatMultAdd_SeqMAIJ_5"
PetscErrorCode MatMultAdd_SeqMAIJ_5(Mat A,Vec xx,Vec yy,Vec zz)
{
  Mat_SeqMAIJ    *b = (Mat_SeqMAIJ*)A->data;
  Mat_SeqAIJ     *a = (Mat_SeqAIJ*)b->AIJ->data;
  PetscScalar    *x,*y,*v,sum1, sum2, sum3, sum4, sum5;
  PetscErrorCode ierr;
  PetscInt       m = b->AIJ->m,*idx,*ii;
  PetscInt       n,i,jrow,j;

  PetscFunctionBegin;
  if (yy != zz) {ierr = VecCopy(yy,zz);CHKERRQ(ierr);}
  ierr = VecGetArray(xx,&x);CHKERRQ(ierr);
  ierr = VecGetArray(zz,&y);CHKERRQ(ierr);
  idx  = a->j;
  v    = a->a;
  ii   = a->i;

  for (i=0; i<m; i++) {
    jrow = ii[i];
    n    = ii[i+1] - jrow;
    sum1  = 0.0;
    sum2  = 0.0;
    sum3  = 0.0;
    sum4  = 0.0;
    sum5  = 0.0;
    for (j=0; j<n; j++) {
      sum1 += v[jrow]*x[5*idx[jrow]];
      sum2 += v[jrow]*x[5*idx[jrow]+1];
      sum3 += v[jrow]*x[5*idx[jrow]+2];
      sum4 += v[jrow]*x[5*idx[jrow]+3];
      sum5 += v[jrow]*x[5*idx[jrow]+4];
      jrow++;
     }
    y[5*i]   += sum1;
    y[5*i+1] += sum2;
    y[5*i+2] += sum3;
    y[5*i+3] += sum4;
    y[5*i+4] += sum5;
  }

  PetscLogFlops(10*a->nz);
  ierr = VecRestoreArray(xx,&x);CHKERRQ(ierr);
  ierr = VecRestoreArray(zz,&y);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatMultTransposeAdd_SeqMAIJ_5"
PetscErrorCode MatMultTransposeAdd_SeqMAIJ_5(Mat A,Vec xx,Vec yy,Vec zz)
{
  Mat_SeqMAIJ    *b = (Mat_SeqMAIJ*)A->data;
  Mat_SeqAIJ     *a = (Mat_SeqAIJ*)b->AIJ->data;
  PetscScalar    *x,*y,*v,alpha1,alpha2,alpha3,alpha4,alpha5;
  PetscErrorCode ierr;
  PetscInt       m = b->AIJ->m,n,i,*idx;

  PetscFunctionBegin; 
  if (yy != zz) {ierr = VecCopy(yy,zz);CHKERRQ(ierr);}
  ierr = VecGetArray(xx,&x);CHKERRQ(ierr);
  ierr = VecGetArray(zz,&y);CHKERRQ(ierr);
 
  for (i=0; i<m; i++) {
    idx    = a->j + a->i[i] ;
    v      = a->a + a->i[i] ;
    n      = a->i[i+1] - a->i[i];
    alpha1 = x[5*i];
    alpha2 = x[5*i+1];
    alpha3 = x[5*i+2];
    alpha4 = x[5*i+3];
    alpha5 = x[5*i+4];
    while (n-->0) {
      y[5*(*idx)]   += alpha1*(*v);
      y[5*(*idx)+1] += alpha2*(*v);
      y[5*(*idx)+2] += alpha3*(*v);
      y[5*(*idx)+3] += alpha4*(*v);
      y[5*(*idx)+4] += alpha5*(*v);
      idx++; v++;
    }
  }
  PetscLogFlops(10*a->nz);
  ierr = VecRestoreArray(xx,&x);CHKERRQ(ierr);
  ierr = VecRestoreArray(zz,&y);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* ------------------------------------------------------------------------------*/
#undef __FUNCT__  
#define __FUNCT__ "MatMult_SeqMAIJ_6"
PetscErrorCode MatMult_SeqMAIJ_6(Mat A,Vec xx,Vec yy)
{
  Mat_SeqMAIJ    *b = (Mat_SeqMAIJ*)A->data;
  Mat_SeqAIJ     *a = (Mat_SeqAIJ*)b->AIJ->data;
  PetscScalar    *x,*y,*v,sum1, sum2, sum3, sum4, sum5, sum6;
  PetscErrorCode ierr;
  PetscInt       m = b->AIJ->m,*idx,*ii;
  PetscInt       n,i,jrow,j;

  PetscFunctionBegin;
  ierr = VecGetArray(xx,&x);CHKERRQ(ierr);
  ierr = VecGetArray(yy,&y);CHKERRQ(ierr);
  idx  = a->j;
  v    = a->a;
  ii   = a->i;

  for (i=0; i<m; i++) {
    jrow = ii[i];
    n    = ii[i+1] - jrow;
    sum1  = 0.0;
    sum2  = 0.0;
    sum3  = 0.0;
    sum4  = 0.0;
    sum5  = 0.0;
    sum6  = 0.0;
    for (j=0; j<n; j++) {
      sum1 += v[jrow]*x[6*idx[jrow]];
      sum2 += v[jrow]*x[6*idx[jrow]+1];
      sum3 += v[jrow]*x[6*idx[jrow]+2];
      sum4 += v[jrow]*x[6*idx[jrow]+3];
      sum5 += v[jrow]*x[6*idx[jrow]+4];
      sum6 += v[jrow]*x[6*idx[jrow]+5];
      jrow++;
     }
    y[6*i]   = sum1;
    y[6*i+1] = sum2;
    y[6*i+2] = sum3;
    y[6*i+3] = sum4;
    y[6*i+4] = sum5;
    y[6*i+5] = sum6;
  }

  PetscLogFlops(12*a->nz - 6*m);
  ierr = VecRestoreArray(xx,&x);CHKERRQ(ierr);
  ierr = VecRestoreArray(yy,&y);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatMultTranspose_SeqMAIJ_6"
PetscErrorCode MatMultTranspose_SeqMAIJ_6(Mat A,Vec xx,Vec yy)
{
  Mat_SeqMAIJ    *b = (Mat_SeqMAIJ*)A->data;
  Mat_SeqAIJ     *a = (Mat_SeqAIJ*)b->AIJ->data;
  PetscScalar    *x,*y,*v,alpha1,alpha2,alpha3,alpha4,alpha5,alpha6,zero = 0.0;
  PetscErrorCode ierr;
  PetscInt       m = b->AIJ->m,n,i,*idx;

  PetscFunctionBegin; 
  ierr = VecSet(&zero,yy);CHKERRQ(ierr);
  ierr = VecGetArray(xx,&x);CHKERRQ(ierr);
  ierr = VecGetArray(yy,&y);CHKERRQ(ierr);

  for (i=0; i<m; i++) {
    idx    = a->j + a->i[i] ;
    v      = a->a + a->i[i] ;
    n      = a->i[i+1] - a->i[i];
    alpha1 = x[6*i];
    alpha2 = x[6*i+1];
    alpha3 = x[6*i+2];
    alpha4 = x[6*i+3];
    alpha5 = x[6*i+4];
    alpha6 = x[6*i+5];
    while (n-->0) {
      y[6*(*idx)]   += alpha1*(*v);
      y[6*(*idx)+1] += alpha2*(*v);
      y[6*(*idx)+2] += alpha3*(*v);
      y[6*(*idx)+3] += alpha4*(*v);
      y[6*(*idx)+4] += alpha5*(*v);
      y[6*(*idx)+5] += alpha6*(*v);
      idx++; v++;
    }
  }
  PetscLogFlops(12*a->nz - 6*b->AIJ->n);
  ierr = VecRestoreArray(xx,&x);CHKERRQ(ierr);
  ierr = VecRestoreArray(yy,&y);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatMultAdd_SeqMAIJ_6"
PetscErrorCode MatMultAdd_SeqMAIJ_6(Mat A,Vec xx,Vec yy,Vec zz)
{
  Mat_SeqMAIJ    *b = (Mat_SeqMAIJ*)A->data;
  Mat_SeqAIJ     *a = (Mat_SeqAIJ*)b->AIJ->data;
  PetscScalar    *x,*y,*v,sum1, sum2, sum3, sum4, sum5, sum6;
  PetscErrorCode ierr;
  PetscInt       m = b->AIJ->m,*idx,*ii;
  PetscInt       n,i,jrow,j;

  PetscFunctionBegin;
  if (yy != zz) {ierr = VecCopy(yy,zz);CHKERRQ(ierr);}
  ierr = VecGetArray(xx,&x);CHKERRQ(ierr);
  ierr = VecGetArray(zz,&y);CHKERRQ(ierr);
  idx  = a->j;
  v    = a->a;
  ii   = a->i;

  for (i=0; i<m; i++) {
    jrow = ii[i];
    n    = ii[i+1] - jrow;
    sum1  = 0.0;
    sum2  = 0.0;
    sum3  = 0.0;
    sum4  = 0.0;
    sum5  = 0.0;
    sum6  = 0.0;
    for (j=0; j<n; j++) {
      sum1 += v[jrow]*x[6*idx[jrow]];
      sum2 += v[jrow]*x[6*idx[jrow]+1];
      sum3 += v[jrow]*x[6*idx[jrow]+2];
      sum4 += v[jrow]*x[6*idx[jrow]+3];
      sum5 += v[jrow]*x[6*idx[jrow]+4];
      sum6 += v[jrow]*x[6*idx[jrow]+5];
      jrow++;
     }
    y[6*i]   += sum1;
    y[6*i+1] += sum2;
    y[6*i+2] += sum3;
    y[6*i+3] += sum4;
    y[6*i+4] += sum5;
    y[6*i+5] += sum6;
  }

  PetscLogFlops(12*a->nz);
  ierr = VecRestoreArray(xx,&x);CHKERRQ(ierr);
  ierr = VecRestoreArray(zz,&y);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatMultTransposeAdd_SeqMAIJ_6"
PetscErrorCode MatMultTransposeAdd_SeqMAIJ_6(Mat A,Vec xx,Vec yy,Vec zz)
{
  Mat_SeqMAIJ    *b = (Mat_SeqMAIJ*)A->data;
  Mat_SeqAIJ     *a = (Mat_SeqAIJ*)b->AIJ->data;
  PetscScalar    *x,*y,*v,alpha1,alpha2,alpha3,alpha4,alpha5,alpha6;
  PetscErrorCode ierr;
  PetscInt       m = b->AIJ->m,n,i,*idx;

  PetscFunctionBegin; 
  if (yy != zz) {ierr = VecCopy(yy,zz);CHKERRQ(ierr);}
  ierr = VecGetArray(xx,&x);CHKERRQ(ierr);
  ierr = VecGetArray(zz,&y);CHKERRQ(ierr);
 
  for (i=0; i<m; i++) {
    idx    = a->j + a->i[i] ;
    v      = a->a + a->i[i] ;
    n      = a->i[i+1] - a->i[i];
    alpha1 = x[6*i];
    alpha2 = x[6*i+1];
    alpha3 = x[6*i+2];
    alpha4 = x[6*i+3];
    alpha5 = x[6*i+4];
    alpha6 = x[6*i+5];
    while (n-->0) {
      y[6*(*idx)]   += alpha1*(*v);
      y[6*(*idx)+1] += alpha2*(*v);
      y[6*(*idx)+2] += alpha3*(*v);
      y[6*(*idx)+3] += alpha4*(*v);
      y[6*(*idx)+4] += alpha5*(*v);
      y[6*(*idx)+5] += alpha6*(*v);
      idx++; v++;
    }
  }
  PetscLogFlops(12*a->nz);
  ierr = VecRestoreArray(xx,&x);CHKERRQ(ierr);
  ierr = VecRestoreArray(zz,&y);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* ------------------------------------------------------------------------------*/
#undef __FUNCT__  
#define __FUNCT__ "MatMult_SeqMAIJ_8"
PetscErrorCode MatMult_SeqMAIJ_8(Mat A,Vec xx,Vec yy)
{
  Mat_SeqMAIJ    *b = (Mat_SeqMAIJ*)A->data;
  Mat_SeqAIJ     *a = (Mat_SeqAIJ*)b->AIJ->data;
  PetscScalar    *x,*y,*v,sum1, sum2, sum3, sum4, sum5, sum6, sum7, sum8;
  PetscErrorCode ierr;
  PetscInt       m = b->AIJ->m,*idx,*ii;
  PetscInt       n,i,jrow,j;

  PetscFunctionBegin;
  ierr = VecGetArray(xx,&x);CHKERRQ(ierr);
  ierr = VecGetArray(yy,&y);CHKERRQ(ierr);
  idx  = a->j;
  v    = a->a;
  ii   = a->i;

  for (i=0; i<m; i++) {
    jrow = ii[i];
    n    = ii[i+1] - jrow;
    sum1  = 0.0;
    sum2  = 0.0;
    sum3  = 0.0;
    sum4  = 0.0;
    sum5  = 0.0;
    sum6  = 0.0;
    sum7  = 0.0;
    sum8  = 0.0;
    for (j=0; j<n; j++) {
      sum1 += v[jrow]*x[8*idx[jrow]];
      sum2 += v[jrow]*x[8*idx[jrow]+1];
      sum3 += v[jrow]*x[8*idx[jrow]+2];
      sum4 += v[jrow]*x[8*idx[jrow]+3];
      sum5 += v[jrow]*x[8*idx[jrow]+4];
      sum6 += v[jrow]*x[8*idx[jrow]+5];
      sum7 += v[jrow]*x[8*idx[jrow]+6];
      sum8 += v[jrow]*x[8*idx[jrow]+7];
      jrow++;
     }
    y[8*i]   = sum1;
    y[8*i+1] = sum2;
    y[8*i+2] = sum3;
    y[8*i+3] = sum4;
    y[8*i+4] = sum5;
    y[8*i+5] = sum6;
    y[8*i+6] = sum7;
    y[8*i+7] = sum8;
  }

  PetscLogFlops(16*a->nz - 8*m);
  ierr = VecRestoreArray(xx,&x);CHKERRQ(ierr);
  ierr = VecRestoreArray(yy,&y);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatMultTranspose_SeqMAIJ_8"
PetscErrorCode MatMultTranspose_SeqMAIJ_8(Mat A,Vec xx,Vec yy)
{
  Mat_SeqMAIJ    *b = (Mat_SeqMAIJ*)A->data;
  Mat_SeqAIJ     *a = (Mat_SeqAIJ*)b->AIJ->data;
  PetscScalar    *x,*y,*v,alpha1,alpha2,alpha3,alpha4,alpha5,alpha6,alpha7,alpha8,zero = 0.0;
  PetscErrorCode ierr;
  PetscInt       m = b->AIJ->m,n,i,*idx;

  PetscFunctionBegin; 
  ierr = VecSet(&zero,yy);CHKERRQ(ierr);
  ierr = VecGetArray(xx,&x);CHKERRQ(ierr);
  ierr = VecGetArray(yy,&y);CHKERRQ(ierr);

  for (i=0; i<m; i++) {
    idx    = a->j + a->i[i] ;
    v      = a->a + a->i[i] ;
    n      = a->i[i+1] - a->i[i];
    alpha1 = x[8*i];
    alpha2 = x[8*i+1];
    alpha3 = x[8*i+2];
    alpha4 = x[8*i+3];
    alpha5 = x[8*i+4];
    alpha6 = x[8*i+5];
    alpha7 = x[8*i+6];
    alpha8 = x[8*i+7];
    while (n-->0) {
      y[8*(*idx)]   += alpha1*(*v);
      y[8*(*idx)+1] += alpha2*(*v);
      y[8*(*idx)+2] += alpha3*(*v);
      y[8*(*idx)+3] += alpha4*(*v);
      y[8*(*idx)+4] += alpha5*(*v);
      y[8*(*idx)+5] += alpha6*(*v);
      y[8*(*idx)+6] += alpha7*(*v);
      y[8*(*idx)+7] += alpha8*(*v);
      idx++; v++;
    }
  }
  PetscLogFlops(16*a->nz - 8*b->AIJ->n);
  ierr = VecRestoreArray(xx,&x);CHKERRQ(ierr);
  ierr = VecRestoreArray(yy,&y);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatMultAdd_SeqMAIJ_8"
PetscErrorCode MatMultAdd_SeqMAIJ_8(Mat A,Vec xx,Vec yy,Vec zz)
{
  Mat_SeqMAIJ    *b = (Mat_SeqMAIJ*)A->data;
  Mat_SeqAIJ     *a = (Mat_SeqAIJ*)b->AIJ->data;
  PetscScalar    *x,*y,*v,sum1, sum2, sum3, sum4, sum5, sum6, sum7, sum8;
  PetscErrorCode ierr;
  PetscInt       m = b->AIJ->m,*idx,*ii;
  PetscInt       n,i,jrow,j;

  PetscFunctionBegin;
  if (yy != zz) {ierr = VecCopy(yy,zz);CHKERRQ(ierr);}
  ierr = VecGetArray(xx,&x);CHKERRQ(ierr);
  ierr = VecGetArray(zz,&y);CHKERRQ(ierr);
  idx  = a->j;
  v    = a->a;
  ii   = a->i;

  for (i=0; i<m; i++) {
    jrow = ii[i];
    n    = ii[i+1] - jrow;
    sum1  = 0.0;
    sum2  = 0.0;
    sum3  = 0.0;
    sum4  = 0.0;
    sum5  = 0.0;
    sum6  = 0.0;
    sum7  = 0.0;
    sum8  = 0.0;
    for (j=0; j<n; j++) {
      sum1 += v[jrow]*x[8*idx[jrow]];
      sum2 += v[jrow]*x[8*idx[jrow]+1];
      sum3 += v[jrow]*x[8*idx[jrow]+2];
      sum4 += v[jrow]*x[8*idx[jrow]+3];
      sum5 += v[jrow]*x[8*idx[jrow]+4];
      sum6 += v[jrow]*x[8*idx[jrow]+5];
      sum7 += v[jrow]*x[8*idx[jrow]+6];
      sum8 += v[jrow]*x[8*idx[jrow]+7];
      jrow++;
     }
    y[8*i]   += sum1;
    y[8*i+1] += sum2;
    y[8*i+2] += sum3;
    y[8*i+3] += sum4;
    y[8*i+4] += sum5;
    y[8*i+5] += sum6;
    y[8*i+6] += sum7;
    y[8*i+7] += sum8;
  }

  PetscLogFlops(16*a->nz);
  ierr = VecRestoreArray(xx,&x);CHKERRQ(ierr);
  ierr = VecRestoreArray(zz,&y);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatMultTransposeAdd_SeqMAIJ_8"
PetscErrorCode MatMultTransposeAdd_SeqMAIJ_8(Mat A,Vec xx,Vec yy,Vec zz)
{
  Mat_SeqMAIJ    *b = (Mat_SeqMAIJ*)A->data;
  Mat_SeqAIJ     *a = (Mat_SeqAIJ*)b->AIJ->data;
  PetscScalar    *x,*y,*v,alpha1,alpha2,alpha3,alpha4,alpha5,alpha6,alpha7,alpha8;
  PetscErrorCode ierr;
  PetscInt       m = b->AIJ->m,n,i,*idx;

  PetscFunctionBegin; 
  if (yy != zz) {ierr = VecCopy(yy,zz);CHKERRQ(ierr);}
  ierr = VecGetArray(xx,&x);CHKERRQ(ierr);
  ierr = VecGetArray(zz,&y);CHKERRQ(ierr);
  for (i=0; i<m; i++) {
    idx    = a->j + a->i[i] ;
    v      = a->a + a->i[i] ;
    n      = a->i[i+1] - a->i[i];
    alpha1 = x[8*i];
    alpha2 = x[8*i+1];
    alpha3 = x[8*i+2];
    alpha4 = x[8*i+3];
    alpha5 = x[8*i+4];
    alpha6 = x[8*i+5];
    alpha7 = x[8*i+6];
    alpha8 = x[8*i+7];
    while (n-->0) {
      y[8*(*idx)]   += alpha1*(*v);
      y[8*(*idx)+1] += alpha2*(*v);
      y[8*(*idx)+2] += alpha3*(*v);
      y[8*(*idx)+3] += alpha4*(*v);
      y[8*(*idx)+4] += alpha5*(*v);
      y[8*(*idx)+5] += alpha6*(*v);
      y[8*(*idx)+6] += alpha7*(*v);
      y[8*(*idx)+7] += alpha8*(*v);
      idx++; v++;
    }
  }
  PetscLogFlops(16*a->nz);
  ierr = VecRestoreArray(xx,&x);CHKERRQ(ierr);
  ierr = VecRestoreArray(zz,&y);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* ------------------------------------------------------------------------------*/
#undef __FUNCT__  
#define __FUNCT__ "MatMult_SeqMAIJ_9"
PetscErrorCode MatMult_SeqMAIJ_9(Mat A,Vec xx,Vec yy)
{
  Mat_SeqMAIJ    *b = (Mat_SeqMAIJ*)A->data;
  Mat_SeqAIJ     *a = (Mat_SeqAIJ*)b->AIJ->data;
  PetscScalar    *x,*y,*v,sum1, sum2, sum3, sum4, sum5, sum6, sum7, sum8, sum9;
  PetscErrorCode ierr;
  PetscInt       m = b->AIJ->m,*idx,*ii;
  PetscInt       n,i,jrow,j;

  PetscFunctionBegin;
  ierr = VecGetArray(xx,&x);CHKERRQ(ierr);
  ierr = VecGetArray(yy,&y);CHKERRQ(ierr);
  idx  = a->j;
  v    = a->a;
  ii   = a->i;

  for (i=0; i<m; i++) {
    jrow = ii[i];
    n    = ii[i+1] - jrow;
    sum1  = 0.0;
    sum2  = 0.0;
    sum3  = 0.0;
    sum4  = 0.0;
    sum5  = 0.0;
    sum6  = 0.0;
    sum7  = 0.0;
    sum8  = 0.0;
    sum9  = 0.0;
    for (j=0; j<n; j++) {
      sum1 += v[jrow]*x[9*idx[jrow]];
      sum2 += v[jrow]*x[9*idx[jrow]+1];
      sum3 += v[jrow]*x[9*idx[jrow]+2];
      sum4 += v[jrow]*x[9*idx[jrow]+3];
      sum5 += v[jrow]*x[9*idx[jrow]+4];
      sum6 += v[jrow]*x[9*idx[jrow]+5];
      sum7 += v[jrow]*x[9*idx[jrow]+6];
      sum8 += v[jrow]*x[9*idx[jrow]+7];
      sum9 += v[jrow]*x[9*idx[jrow]+8];
      jrow++;
     }
    y[9*i]   = sum1;
    y[9*i+1] = sum2;
    y[9*i+2] = sum3;
    y[9*i+3] = sum4;
    y[9*i+4] = sum5;
    y[9*i+5] = sum6;
    y[9*i+6] = sum7;
    y[9*i+7] = sum8;
    y[9*i+8] = sum9;
  }

  PetscLogFlops(18*a->nz - 9*m);
  ierr = VecRestoreArray(xx,&x);CHKERRQ(ierr);
  ierr = VecRestoreArray(yy,&y);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatMultTranspose_SeqMAIJ_9"
PetscErrorCode MatMultTranspose_SeqMAIJ_9(Mat A,Vec xx,Vec yy)
{
  Mat_SeqMAIJ    *b = (Mat_SeqMAIJ*)A->data;
  Mat_SeqAIJ     *a = (Mat_SeqAIJ*)b->AIJ->data;
  PetscScalar    *x,*y,*v,alpha1,alpha2,alpha3,alpha4,alpha5,alpha6,alpha7,alpha8,alpha9,zero = 0.0;
  PetscErrorCode ierr;
  PetscInt       m = b->AIJ->m,n,i,*idx;

  PetscFunctionBegin; 
  ierr = VecSet(&zero,yy);CHKERRQ(ierr);
  ierr = VecGetArray(xx,&x);CHKERRQ(ierr);
  ierr = VecGetArray(yy,&y);CHKERRQ(ierr);

  for (i=0; i<m; i++) {
    idx    = a->j + a->i[i] ;
    v      = a->a + a->i[i] ;
    n      = a->i[i+1] - a->i[i];
    alpha1 = x[9*i];
    alpha2 = x[9*i+1];
    alpha3 = x[9*i+2];
    alpha4 = x[9*i+3];
    alpha5 = x[9*i+4];
    alpha6 = x[9*i+5];
    alpha7 = x[9*i+6];
    alpha8 = x[9*i+7];
    alpha9 = x[9*i+8];
    while (n-->0) {
      y[9*(*idx)]   += alpha1*(*v);
      y[9*(*idx)+1] += alpha2*(*v);
      y[9*(*idx)+2] += alpha3*(*v);
      y[9*(*idx)+3] += alpha4*(*v);
      y[9*(*idx)+4] += alpha5*(*v);
      y[9*(*idx)+5] += alpha6*(*v);
      y[9*(*idx)+6] += alpha7*(*v);
      y[9*(*idx)+7] += alpha8*(*v);
      y[9*(*idx)+8] += alpha9*(*v);
      idx++; v++;
    }
  }
  PetscLogFlops(18*a->nz - 9*b->AIJ->n);
  ierr = VecRestoreArray(xx,&x);CHKERRQ(ierr);
  ierr = VecRestoreArray(yy,&y);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatMultAdd_SeqMAIJ_9"
PetscErrorCode MatMultAdd_SeqMAIJ_9(Mat A,Vec xx,Vec yy,Vec zz)
{
  Mat_SeqMAIJ    *b = (Mat_SeqMAIJ*)A->data;
  Mat_SeqAIJ     *a = (Mat_SeqAIJ*)b->AIJ->data;
  PetscScalar    *x,*y,*v,sum1, sum2, sum3, sum4, sum5, sum6, sum7, sum8, sum9;
  PetscErrorCode ierr;
  PetscInt       m = b->AIJ->m,*idx,*ii;
  PetscInt       n,i,jrow,j;

  PetscFunctionBegin;
  if (yy != zz) {ierr = VecCopy(yy,zz);CHKERRQ(ierr);}
  ierr = VecGetArray(xx,&x);CHKERRQ(ierr);
  ierr = VecGetArray(zz,&y);CHKERRQ(ierr);
  idx  = a->j;
  v    = a->a;
  ii   = a->i;

  for (i=0; i<m; i++) {
    jrow = ii[i];
    n    = ii[i+1] - jrow;
    sum1  = 0.0;
    sum2  = 0.0;
    sum3  = 0.0;
    sum4  = 0.0;
    sum5  = 0.0;
    sum6  = 0.0;
    sum7  = 0.0;
    sum8  = 0.0;
    sum9  = 0.0;
    for (j=0; j<n; j++) {
      sum1 += v[jrow]*x[9*idx[jrow]];
      sum2 += v[jrow]*x[9*idx[jrow]+1];
      sum3 += v[jrow]*x[9*idx[jrow]+2];
      sum4 += v[jrow]*x[9*idx[jrow]+3];
      sum5 += v[jrow]*x[9*idx[jrow]+4];
      sum6 += v[jrow]*x[9*idx[jrow]+5];
      sum7 += v[jrow]*x[9*idx[jrow]+6];
      sum8 += v[jrow]*x[9*idx[jrow]+7];
      sum9 += v[jrow]*x[9*idx[jrow]+8];
      jrow++;
     }
    y[9*i]   += sum1;
    y[9*i+1] += sum2;
    y[9*i+2] += sum3;
    y[9*i+3] += sum4;
    y[9*i+4] += sum5;
    y[9*i+5] += sum6;
    y[9*i+6] += sum7;
    y[9*i+7] += sum8;
    y[9*i+8] += sum9;
  }

  PetscLogFlops(18*a->nz);
  ierr = VecRestoreArray(xx,&x);CHKERRQ(ierr);
  ierr = VecRestoreArray(zz,&y);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatMultTransposeAdd_SeqMAIJ_9"
PetscErrorCode MatMultTransposeAdd_SeqMAIJ_9(Mat A,Vec xx,Vec yy,Vec zz)
{
  Mat_SeqMAIJ    *b = (Mat_SeqMAIJ*)A->data;
  Mat_SeqAIJ     *a = (Mat_SeqAIJ*)b->AIJ->data;
  PetscScalar    *x,*y,*v,alpha1,alpha2,alpha3,alpha4,alpha5,alpha6,alpha7,alpha8,alpha9;
  PetscErrorCode ierr;
  PetscInt       m = b->AIJ->m,n,i,*idx;

  PetscFunctionBegin; 
  if (yy != zz) {ierr = VecCopy(yy,zz);CHKERRQ(ierr);}
  ierr = VecGetArray(xx,&x);CHKERRQ(ierr);
  ierr = VecGetArray(zz,&y);CHKERRQ(ierr);
  for (i=0; i<m; i++) {
    idx    = a->j + a->i[i] ;
    v      = a->a + a->i[i] ;
    n      = a->i[i+1] - a->i[i];
    alpha1 = x[9*i];
    alpha2 = x[9*i+1];
    alpha3 = x[9*i+2];
    alpha4 = x[9*i+3];
    alpha5 = x[9*i+4];
    alpha6 = x[9*i+5];
    alpha7 = x[9*i+6];
    alpha8 = x[9*i+7];
    alpha9 = x[9*i+8];
    while (n-->0) {
      y[9*(*idx)]   += alpha1*(*v);
      y[9*(*idx)+1] += alpha2*(*v);
      y[9*(*idx)+2] += alpha3*(*v);
      y[9*(*idx)+3] += alpha4*(*v);
      y[9*(*idx)+4] += alpha5*(*v);
      y[9*(*idx)+5] += alpha6*(*v);
      y[9*(*idx)+6] += alpha7*(*v);
      y[9*(*idx)+7] += alpha8*(*v);
      y[9*(*idx)+8] += alpha9*(*v);
      idx++; v++;
    }
  }
  PetscLogFlops(18*a->nz);
  ierr = VecRestoreArray(xx,&x);CHKERRQ(ierr);
  ierr = VecRestoreArray(zz,&y);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*--------------------------------------------------------------------------------------------*/
#undef __FUNCT__  
#define __FUNCT__ "MatMult_SeqMAIJ_16"
PetscErrorCode MatMult_SeqMAIJ_16(Mat A,Vec xx,Vec yy)
{
  Mat_SeqMAIJ    *b = (Mat_SeqMAIJ*)A->data;
  Mat_SeqAIJ     *a = (Mat_SeqAIJ*)b->AIJ->data;
  PetscScalar    *x,*y,*v,sum1, sum2, sum3, sum4, sum5, sum6, sum7, sum8;
  PetscScalar    sum9, sum10, sum11, sum12, sum13, sum14, sum15, sum16;
  PetscErrorCode ierr;
  PetscInt       m = b->AIJ->m,*idx,*ii;
  PetscInt       n,i,jrow,j;

  PetscFunctionBegin;
  ierr = VecGetArray(xx,&x);CHKERRQ(ierr);
  ierr = VecGetArray(yy,&y);CHKERRQ(ierr);
  idx  = a->j;
  v    = a->a;
  ii   = a->i;

  for (i=0; i<m; i++) {
    jrow = ii[i];
    n    = ii[i+1] - jrow;
    sum1  = 0.0;
    sum2  = 0.0;
    sum3  = 0.0;
    sum4  = 0.0;
    sum5  = 0.0;
    sum6  = 0.0;
    sum7  = 0.0;
    sum8  = 0.0;
    sum9  = 0.0;
    sum10 = 0.0;
    sum11 = 0.0;
    sum12 = 0.0;
    sum13 = 0.0;
    sum14 = 0.0;
    sum15 = 0.0;
    sum16 = 0.0;
    for (j=0; j<n; j++) {
      sum1  += v[jrow]*x[16*idx[jrow]];
      sum2  += v[jrow]*x[16*idx[jrow]+1];
      sum3  += v[jrow]*x[16*idx[jrow]+2];
      sum4  += v[jrow]*x[16*idx[jrow]+3];
      sum5  += v[jrow]*x[16*idx[jrow]+4];
      sum6  += v[jrow]*x[16*idx[jrow]+5];
      sum7  += v[jrow]*x[16*idx[jrow]+6];
      sum8  += v[jrow]*x[16*idx[jrow]+7];
      sum9  += v[jrow]*x[16*idx[jrow]+8];
      sum10 += v[jrow]*x[16*idx[jrow]+9];
      sum11 += v[jrow]*x[16*idx[jrow]+10];
      sum12 += v[jrow]*x[16*idx[jrow]+11];
      sum13 += v[jrow]*x[16*idx[jrow]+12];
      sum14 += v[jrow]*x[16*idx[jrow]+13];
      sum15 += v[jrow]*x[16*idx[jrow]+14];
      sum16 += v[jrow]*x[16*idx[jrow]+15];
      jrow++;
     }
    y[16*i]    = sum1;
    y[16*i+1]  = sum2;
    y[16*i+2]  = sum3;
    y[16*i+3]  = sum4;
    y[16*i+4]  = sum5;
    y[16*i+5]  = sum6;
    y[16*i+6]  = sum7;
    y[16*i+7]  = sum8;
    y[16*i+8]  = sum9;
    y[16*i+9]  = sum10;
    y[16*i+10] = sum11;
    y[16*i+11] = sum12;
    y[16*i+12] = sum13;
    y[16*i+13] = sum14;
    y[16*i+14] = sum15;
    y[16*i+15] = sum16;
  }

  PetscLogFlops(32*a->nz - 16*m);
  ierr = VecRestoreArray(xx,&x);CHKERRQ(ierr);
  ierr = VecRestoreArray(yy,&y);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatMultTranspose_SeqMAIJ_16"
PetscErrorCode MatMultTranspose_SeqMAIJ_16(Mat A,Vec xx,Vec yy)
{
  Mat_SeqMAIJ    *b = (Mat_SeqMAIJ*)A->data;
  Mat_SeqAIJ     *a = (Mat_SeqAIJ*)b->AIJ->data;
  PetscScalar    *x,*y,*v,alpha1,alpha2,alpha3,alpha4,alpha5,alpha6,alpha7,alpha8,zero = 0.0;
  PetscScalar    alpha9,alpha10,alpha11,alpha12,alpha13,alpha14,alpha15,alpha16;
  PetscErrorCode ierr;
  PetscInt       m = b->AIJ->m,n,i,*idx;

  PetscFunctionBegin; 
  ierr = VecSet(&zero,yy);CHKERRQ(ierr);
  ierr = VecGetArray(xx,&x);CHKERRQ(ierr);
  ierr = VecGetArray(yy,&y);CHKERRQ(ierr);

  for (i=0; i<m; i++) {
    idx    = a->j + a->i[i] ;
    v      = a->a + a->i[i] ;
    n      = a->i[i+1] - a->i[i];
    alpha1  = x[16*i];
    alpha2  = x[16*i+1];
    alpha3  = x[16*i+2];
    alpha4  = x[16*i+3];
    alpha5  = x[16*i+4];
    alpha6  = x[16*i+5];
    alpha7  = x[16*i+6];
    alpha8  = x[16*i+7];
    alpha9  = x[16*i+8];
    alpha10 = x[16*i+9];
    alpha11 = x[16*i+10];
    alpha12 = x[16*i+11];
    alpha13 = x[16*i+12];
    alpha14 = x[16*i+13];
    alpha15 = x[16*i+14];
    alpha16 = x[16*i+15];
    while (n-->0) {
      y[16*(*idx)]    += alpha1*(*v);
      y[16*(*idx)+1]  += alpha2*(*v);
      y[16*(*idx)+2]  += alpha3*(*v);
      y[16*(*idx)+3]  += alpha4*(*v);
      y[16*(*idx)+4]  += alpha5*(*v);
      y[16*(*idx)+5]  += alpha6*(*v);
      y[16*(*idx)+6]  += alpha7*(*v);
      y[16*(*idx)+7]  += alpha8*(*v);
      y[16*(*idx)+8]  += alpha9*(*v);
      y[16*(*idx)+9]  += alpha10*(*v);
      y[16*(*idx)+10] += alpha11*(*v);
      y[16*(*idx)+11] += alpha12*(*v);
      y[16*(*idx)+12] += alpha13*(*v);
      y[16*(*idx)+13] += alpha14*(*v);
      y[16*(*idx)+14] += alpha15*(*v);
      y[16*(*idx)+15] += alpha16*(*v);
      idx++; v++;
    }
  }
  PetscLogFlops(32*a->nz - 16*b->AIJ->n);
  ierr = VecRestoreArray(xx,&x);CHKERRQ(ierr);
  ierr = VecRestoreArray(yy,&y);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatMultAdd_SeqMAIJ_16"
PetscErrorCode MatMultAdd_SeqMAIJ_16(Mat A,Vec xx,Vec yy,Vec zz)
{
  Mat_SeqMAIJ    *b = (Mat_SeqMAIJ*)A->data;
  Mat_SeqAIJ     *a = (Mat_SeqAIJ*)b->AIJ->data;
  PetscScalar    *x,*y,*v,sum1, sum2, sum3, sum4, sum5, sum6, sum7, sum8;
  PetscScalar    sum9, sum10, sum11, sum12, sum13, sum14, sum15, sum16;
  PetscErrorCode ierr;
  PetscInt       m = b->AIJ->m,*idx,*ii;
  PetscInt       n,i,jrow,j;

  PetscFunctionBegin;
  if (yy != zz) {ierr = VecCopy(yy,zz);CHKERRQ(ierr);}
  ierr = VecGetArray(xx,&x);CHKERRQ(ierr);
  ierr = VecGetArray(zz,&y);CHKERRQ(ierr);
  idx  = a->j;
  v    = a->a;
  ii   = a->i;

  for (i=0; i<m; i++) {
    jrow = ii[i];
    n    = ii[i+1] - jrow;
    sum1  = 0.0;
    sum2  = 0.0;
    sum3  = 0.0;
    sum4  = 0.0;
    sum5  = 0.0;
    sum6  = 0.0;
    sum7  = 0.0;
    sum8  = 0.0;
    sum9  = 0.0;
    sum10 = 0.0;
    sum11 = 0.0;
    sum12 = 0.0;
    sum13 = 0.0;
    sum14 = 0.0;
    sum15 = 0.0;
    sum16 = 0.0;
    for (j=0; j<n; j++) {
      sum1  += v[jrow]*x[16*idx[jrow]];
      sum2  += v[jrow]*x[16*idx[jrow]+1];
      sum3  += v[jrow]*x[16*idx[jrow]+2];
      sum4  += v[jrow]*x[16*idx[jrow]+3];
      sum5  += v[jrow]*x[16*idx[jrow]+4];
      sum6  += v[jrow]*x[16*idx[jrow]+5];
      sum7  += v[jrow]*x[16*idx[jrow]+6];
      sum8  += v[jrow]*x[16*idx[jrow]+7];
      sum9  += v[jrow]*x[16*idx[jrow]+8];
      sum10 += v[jrow]*x[16*idx[jrow]+9];
      sum11 += v[jrow]*x[16*idx[jrow]+10];
      sum12 += v[jrow]*x[16*idx[jrow]+11];
      sum13 += v[jrow]*x[16*idx[jrow]+12];
      sum14 += v[jrow]*x[16*idx[jrow]+13];
      sum15 += v[jrow]*x[16*idx[jrow]+14];
      sum16 += v[jrow]*x[16*idx[jrow]+15];
      jrow++;
     }
    y[16*i]    += sum1;
    y[16*i+1]  += sum2;
    y[16*i+2]  += sum3;
    y[16*i+3]  += sum4;
    y[16*i+4]  += sum5;
    y[16*i+5]  += sum6;
    y[16*i+6]  += sum7;
    y[16*i+7]  += sum8;
    y[16*i+8]  += sum9;
    y[16*i+9]  += sum10;
    y[16*i+10] += sum11;
    y[16*i+11] += sum12;
    y[16*i+12] += sum13;
    y[16*i+13] += sum14;
    y[16*i+14] += sum15;
    y[16*i+15] += sum16;
  }

  PetscLogFlops(32*a->nz);
  ierr = VecRestoreArray(xx,&x);CHKERRQ(ierr);
  ierr = VecRestoreArray(zz,&y);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatMultTransposeAdd_SeqMAIJ_16"
PetscErrorCode MatMultTransposeAdd_SeqMAIJ_16(Mat A,Vec xx,Vec yy,Vec zz)
{
  Mat_SeqMAIJ    *b = (Mat_SeqMAIJ*)A->data;
  Mat_SeqAIJ     *a = (Mat_SeqAIJ*)b->AIJ->data;
  PetscScalar    *x,*y,*v,alpha1,alpha2,alpha3,alpha4,alpha5,alpha6,alpha7,alpha8;
  PetscScalar    alpha9,alpha10,alpha11,alpha12,alpha13,alpha14,alpha15,alpha16;
  PetscErrorCode ierr;
  PetscInt       m = b->AIJ->m,n,i,*idx;

  PetscFunctionBegin; 
  if (yy != zz) {ierr = VecCopy(yy,zz);CHKERRQ(ierr);}
  ierr = VecGetArray(xx,&x);CHKERRQ(ierr);
  ierr = VecGetArray(zz,&y);CHKERRQ(ierr);
  for (i=0; i<m; i++) {
    idx    = a->j + a->i[i] ;
    v      = a->a + a->i[i] ;
    n      = a->i[i+1] - a->i[i];
    alpha1 = x[16*i];
    alpha2 = x[16*i+1];
    alpha3 = x[16*i+2];
    alpha4 = x[16*i+3];
    alpha5 = x[16*i+4];
    alpha6 = x[16*i+5];
    alpha7 = x[16*i+6];
    alpha8 = x[16*i+7];
    alpha9  = x[16*i+8];
    alpha10 = x[16*i+9];
    alpha11 = x[16*i+10];
    alpha12 = x[16*i+11];
    alpha13 = x[16*i+12];
    alpha14 = x[16*i+13];
    alpha15 = x[16*i+14];
    alpha16 = x[16*i+15];
    while (n-->0) {
      y[16*(*idx)]   += alpha1*(*v);
      y[16*(*idx)+1] += alpha2*(*v);
      y[16*(*idx)+2] += alpha3*(*v);
      y[16*(*idx)+3] += alpha4*(*v);
      y[16*(*idx)+4] += alpha5*(*v);
      y[16*(*idx)+5] += alpha6*(*v);
      y[16*(*idx)+6] += alpha7*(*v);
      y[16*(*idx)+7] += alpha8*(*v);
      y[16*(*idx)+8]  += alpha9*(*v);
      y[16*(*idx)+9]  += alpha10*(*v);
      y[16*(*idx)+10] += alpha11*(*v);
      y[16*(*idx)+11] += alpha12*(*v);
      y[16*(*idx)+12] += alpha13*(*v);
      y[16*(*idx)+13] += alpha14*(*v);
      y[16*(*idx)+14] += alpha15*(*v);
      y[16*(*idx)+15] += alpha16*(*v);
      idx++; v++;
    }
  }
  PetscLogFlops(32*a->nz);
  ierr = VecRestoreArray(xx,&x);CHKERRQ(ierr);
  ierr = VecRestoreArray(zz,&y);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*===================================================================================*/
#undef __FUNCT__  
#define __FUNCT__ "MatMult_MPIMAIJ_dof"
PetscErrorCode MatMult_MPIMAIJ_dof(Mat A,Vec xx,Vec yy)
{
  Mat_MPIMAIJ    *b = (Mat_MPIMAIJ*)A->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  /* start the scatter */
  ierr = VecScatterBegin(xx,b->w,INSERT_VALUES,SCATTER_FORWARD,b->ctx);CHKERRQ(ierr);
  ierr = (*b->AIJ->ops->mult)(b->AIJ,xx,yy);CHKERRQ(ierr);
  ierr = VecScatterEnd(xx,b->w,INSERT_VALUES,SCATTER_FORWARD,b->ctx);CHKERRQ(ierr);
  ierr = (*b->OAIJ->ops->multadd)(b->OAIJ,b->w,yy,yy);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatMultTranspose_MPIMAIJ_dof"
PetscErrorCode MatMultTranspose_MPIMAIJ_dof(Mat A,Vec xx,Vec yy)
{
  Mat_MPIMAIJ    *b = (Mat_MPIMAIJ*)A->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = (*b->OAIJ->ops->multtranspose)(b->OAIJ,xx,b->w);CHKERRQ(ierr);
  ierr = (*b->AIJ->ops->multtranspose)(b->AIJ,xx,yy);CHKERRQ(ierr);
  ierr = VecScatterBegin(b->w,yy,ADD_VALUES,SCATTER_REVERSE,b->ctx);CHKERRQ(ierr);
  ierr = VecScatterEnd(b->w,yy,ADD_VALUES,SCATTER_REVERSE,b->ctx);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatMultAdd_MPIMAIJ_dof"
PetscErrorCode MatMultAdd_MPIMAIJ_dof(Mat A,Vec xx,Vec yy,Vec zz)
{
  Mat_MPIMAIJ    *b = (Mat_MPIMAIJ*)A->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  /* start the scatter */
  ierr = VecScatterBegin(xx,b->w,INSERT_VALUES,SCATTER_FORWARD,b->ctx);CHKERRQ(ierr);
  ierr = (*b->AIJ->ops->multadd)(b->AIJ,xx,yy,zz);CHKERRQ(ierr);
  ierr = VecScatterEnd(xx,b->w,INSERT_VALUES,SCATTER_FORWARD,b->ctx);CHKERRQ(ierr);
  ierr = (*b->OAIJ->ops->multadd)(b->OAIJ,b->w,zz,zz);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatMultTransposeAdd_MPIMAIJ_dof"
PetscErrorCode MatMultTransposeAdd_MPIMAIJ_dof(Mat A,Vec xx,Vec yy,Vec zz)
{
  Mat_MPIMAIJ    *b = (Mat_MPIMAIJ*)A->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = (*b->OAIJ->ops->multtranspose)(b->OAIJ,xx,b->w);CHKERRQ(ierr);
  ierr = VecScatterBegin(b->w,zz,ADD_VALUES,SCATTER_REVERSE,b->ctx);CHKERRQ(ierr);
  ierr = (*b->AIJ->ops->multtransposeadd)(b->AIJ,xx,yy,zz);CHKERRQ(ierr);
  ierr = VecScatterEnd(b->w,zz,ADD_VALUES,SCATTER_REVERSE,b->ctx);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatPtAPSymbolic_SeqAIJ_SeqMAIJ"
PetscErrorCode MatPtAPSymbolic_SeqAIJ_SeqMAIJ(Mat A,Mat PP,PetscReal fill,Mat *C) 
{
  /* This routine requires testing -- but it's getting better. */
  PetscErrorCode ierr;
  FreeSpaceList  free_space=PETSC_NULL,current_space=PETSC_NULL;
  Mat_SeqMAIJ    *pp=(Mat_SeqMAIJ*)PP->data;
  Mat            P=pp->AIJ;
  Mat_SeqAIJ     *a=(Mat_SeqAIJ*)A->data,*p=(Mat_SeqAIJ*)P->data,*c;
  PetscInt       *pti,*ptj,*ptJ,*ai=a->i,*aj=a->j,*ajj,*pi=p->i,*pj=p->j,*pjj;
  PetscInt       *ci,*cj,*ptadenserow,*ptasparserow,*denserow,*sparserow,*ptaj;
  PetscInt       an=A->N,am=A->M,pn=P->N,pm=P->M,ppdof=pp->dof,cn;
  PetscInt       i,j,k,dof,pshift,ptnzi,arow,anzj,ptanzi,prow,pnzj,cnzi;
  MatScalar      *ca;

  PetscFunctionBegin;  
  /* Start timer */
  ierr = PetscLogEventBegin(MAT_PtAPSymbolic,A,PP,0,0);CHKERRQ(ierr);

  /* Get ij structure of P^T */
  ierr = MatGetSymbolicTranspose_SeqAIJ(P,&pti,&ptj);CHKERRQ(ierr);

  cn = pn*ppdof;
  /* Allocate ci array, arrays for fill computation and */
  /* free space for accumulating nonzero column info */
  ierr = PetscMalloc((cn+1)*sizeof(PetscInt),&ci);CHKERRQ(ierr);
  ci[0] = 0;

  /* Work arrays for rows of P^T*A */
  ierr = PetscMalloc((2*cn+2*an+1)*sizeof(PetscInt),&ptadenserow);CHKERRQ(ierr);
  ierr = PetscMemzero(ptadenserow,(2*cn+2*an+1)*sizeof(PetscInt));CHKERRQ(ierr);
  ptasparserow = ptadenserow  + an;
  denserow     = ptasparserow + an;
  sparserow    = denserow     + cn;

  /* Set initial free space to be nnz(A) scaled by aspect ratio of P. */
  /* This should be reasonable if sparsity of PtAP is similar to that of A. */
  /* Note, aspect ratio of P is the same as the aspect ratio of SeqAIJ inside P */
  ierr          = GetMoreSpace((ai[am]/pm)*pn,&free_space);
  current_space = free_space;

  /* Determine symbolic info for each row of C: */
  for (i=0;i<pn;i++) {
    ptnzi  = pti[i+1] - pti[i];
    ptJ    = ptj + pti[i];
    for (dof=0;dof<ppdof;dof++) {
      ptanzi = 0;
      /* Determine symbolic row of PtA: */
      for (j=0;j<ptnzi;j++) {
        /* Expand ptJ[j] by block size and shift by dof to get the right row of A */
        arow = ptJ[j]*ppdof + dof;
        /* Nonzeros of P^T*A will be in same locations as any element of A in that row */
        anzj = ai[arow+1] - ai[arow];
        ajj  = aj + ai[arow];
        for (k=0;k<anzj;k++) {
          if (!ptadenserow[ajj[k]]) {
            ptadenserow[ajj[k]]    = -1;
            ptasparserow[ptanzi++] = ajj[k];
          }
        }
      }
      /* Using symbolic info for row of PtA, determine symbolic info for row of C: */
      ptaj = ptasparserow;
      cnzi   = 0;
      for (j=0;j<ptanzi;j++) {
        /* Get offset within block of P */
        pshift = *ptaj%ppdof;
        /* Get block row of P */
        prow = (*ptaj++)/ppdof; /* integer division */
        /* P has same number of nonzeros per row as the compressed form */
        pnzj = pi[prow+1] - pi[prow];
        pjj  = pj + pi[prow];
        for (k=0;k<pnzj;k++) {
          /* Locations in C are shifted by the offset within the block */
          /* Note: we cannot use PetscLLAdd here because of the additional offset for the write location */
          if (!denserow[pjj[k]*ppdof+pshift]) {
            denserow[pjj[k]*ppdof+pshift] = -1;
            sparserow[cnzi++]             = pjj[k]*ppdof+pshift;
          }
        }
      }

      /* sort sparserow */
      ierr = PetscSortInt(cnzi,sparserow);CHKERRQ(ierr);
      
      /* If free space is not available, make more free space */
      /* Double the amount of total space in the list */
      if (current_space->local_remaining<cnzi) {
        ierr = GetMoreSpace(current_space->total_array_size,&current_space);CHKERRQ(ierr);
      }

      /* Copy data into free space, and zero out denserows */
      ierr = PetscMemcpy(current_space->array,sparserow,cnzi*sizeof(PetscInt));CHKERRQ(ierr);
      current_space->array           += cnzi;
      current_space->local_used      += cnzi;
      current_space->local_remaining -= cnzi;

      for (j=0;j<ptanzi;j++) {
        ptadenserow[ptasparserow[j]] = 0;
      }
      for (j=0;j<cnzi;j++) {
        denserow[sparserow[j]] = 0;
      }
      /* Aside: Perhaps we should save the pta info for the numerical factorization. */
      /*        For now, we will recompute what is needed. */ 
      ci[i*ppdof+1+dof] = ci[i*ppdof+dof] + cnzi;
    }
  }
  /* nnz is now stored in ci[ptm], column indices are in the list of free space */
  /* Allocate space for cj, initialize cj, and */
  /* destroy list of free space and other temporary array(s) */
  ierr = PetscMalloc((ci[cn]+1)*sizeof(PetscInt),&cj);CHKERRQ(ierr);
  ierr = MakeSpaceContiguous(&free_space,cj);CHKERRQ(ierr);
  ierr = PetscFree(ptadenserow);CHKERRQ(ierr);
    
  /* Allocate space for ca */
  ierr = PetscMalloc((ci[cn]+1)*sizeof(MatScalar),&ca);CHKERRQ(ierr);
  ierr = PetscMemzero(ca,(ci[cn]+1)*sizeof(MatScalar));CHKERRQ(ierr);
  
  /* put together the new matrix */
  ierr = MatCreateSeqAIJWithArrays(A->comm,cn,cn,ci,cj,ca,C);CHKERRQ(ierr);

  /* MatCreateSeqAIJWithArrays flags matrix so PETSc doesn't free the user's arrays. */
  /* Since these are PETSc arrays, change flags to free them as necessary. */
  c = (Mat_SeqAIJ *)((*C)->data);
  c->freedata = PETSC_TRUE;
  c->nonew    = 0;

  /* Clean up. */
  ierr = MatRestoreSymbolicTranspose_SeqAIJ(P,&pti,&ptj);CHKERRQ(ierr);

  ierr = PetscLogEventEnd(MAT_PtAPSymbolic,A,PP,0,0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatPtAPNumeric_SeqAIJ_SeqMAIJ"
PetscErrorCode MatPtAPNumeric_SeqAIJ_SeqMAIJ(Mat A,Mat PP,Mat C) 
{
  /* This routine requires testing -- first draft only */
  PetscErrorCode ierr;
  PetscInt       flops=0;
  Mat_SeqMAIJ    *pp=(Mat_SeqMAIJ*)PP->data;
  Mat            P=pp->AIJ;
  Mat_SeqAIJ     *a  = (Mat_SeqAIJ *) A->data;
  Mat_SeqAIJ     *p  = (Mat_SeqAIJ *) P->data;
  Mat_SeqAIJ     *c  = (Mat_SeqAIJ *) C->data;
  PetscInt       *ai=a->i,*aj=a->j,*apj,*apjdense,*pi=p->i,*pj=p->j,*pJ=p->j,*pjj;
  PetscInt       *ci=c->i,*cj=c->j,*cjj;
  PetscInt       am=A->M,cn=C->N,cm=C->M,ppdof=pp->dof;
  PetscInt       i,j,k,pshift,poffset,anzi,pnzi,apnzj,nextap,pnzj,prow,crow;
  MatScalar      *aa=a->a,*apa,*pa=p->a,*pA=p->a,*paj,*ca=c->a,*caj;

  PetscFunctionBegin;
  /* Allocate temporary array for storage of one row of A*P */
  ierr = PetscMalloc(cn*(sizeof(MatScalar)+2*sizeof(PetscInt)),&apa);CHKERRQ(ierr);
  ierr = PetscMemzero(apa,cn*(sizeof(MatScalar)+2*sizeof(PetscInt)));CHKERRQ(ierr);

  apj      = (PetscInt *)(apa + cn);
  apjdense = apj + cn;

  /* Clear old values in C */
  ierr = PetscMemzero(ca,ci[cm]*sizeof(MatScalar));CHKERRQ(ierr);

  for (i=0;i<am;i++) {
    /* Form sparse row of A*P */
    anzi  = ai[i+1] - ai[i];
    apnzj = 0;
    for (j=0;j<anzi;j++) {
      /* Get offset within block of P */
      pshift = *aj%ppdof;
      /* Get block row of P */
      prow   = *aj++/ppdof; /* integer division */
      pnzj = pi[prow+1] - pi[prow];
      pjj  = pj + pi[prow];
      paj  = pa + pi[prow];
      for (k=0;k<pnzj;k++) {
        poffset = pjj[k]*ppdof+pshift;
        if (!apjdense[poffset]) {
          apjdense[poffset] = -1; 
          apj[apnzj++]      = poffset;
        }
        apa[poffset] += (*aa)*paj[k];
      }
      flops += 2*pnzj;
      aa++;
    }

    /* Sort the j index array for quick sparse axpy. */
    /* Note: a array does not need sorting as it is in dense storage locations. */
    ierr = PetscSortInt(apnzj,apj);CHKERRQ(ierr);

    /* Compute P^T*A*P using outer product (P^T)[:,j]*(A*P)[j,:]. */
    prow    = i/ppdof; /* integer division */
    pshift  = i%ppdof; 
    poffset = pi[prow];
    pnzi = pi[prow+1] - poffset;
    /* Reset pJ and pA so we can traverse the same row of P 'dof' times. */
    pJ   = pj+poffset;
    pA   = pa+poffset;
    for (j=0;j<pnzi;j++) {
      crow   = (*pJ)*ppdof+pshift;
      cjj    = cj + ci[crow];
      caj    = ca + ci[crow];
      pJ++;
      /* Perform sparse axpy operation.  Note cjj includes apj. */
      for (k=0,nextap=0;nextap<apnzj;k++) {
        if (cjj[k]==apj[nextap]) {
          caj[k] += (*pA)*apa[apj[nextap++]];
        }
      }
      flops += 2*apnzj;
      pA++;
    }

    /* Zero the current row info for A*P */
    for (j=0;j<apnzj;j++) {
      apa[apj[j]]      = 0.;
      apjdense[apj[j]] = 0;
    }
  }

  /* Assemble the final matrix and clean up */
  ierr = MatAssemblyBegin(C,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(C,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = PetscFree(apa);CHKERRQ(ierr);
  ierr = PetscLogFlops(flops);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatConvert_SeqMAIJ_SeqAIJ"
PetscErrorCode MatConvert_SeqMAIJ_SeqAIJ(Mat A,const MatType newtype,MatReuse reuse,Mat *newmat)
{
  Mat_SeqMAIJ       *b = (Mat_SeqMAIJ*)A->data;
  Mat               a = b->AIJ,B;
  Mat_SeqAIJ        *aij = (Mat_SeqAIJ*)a->data;
  PetscErrorCode    ierr;
  PetscInt          m,n,i,ncols,*ilen,nmax = 0,*icols,j,k,ii,dof = b->dof;
  PetscInt          *cols;
  PetscScalar       *vals;

  PetscFunctionBegin;
  ierr = MatGetSize(a,&m,&n);CHKERRQ(ierr);    
  ierr = PetscMalloc(dof*m*sizeof(int),&ilen);CHKERRQ(ierr);
  for (i=0; i<m; i++) {
    nmax = PetscMax(nmax,aij->ilen[i]);
    for (j=0; j<dof; j++) {
      ilen[dof*i+j] = aij->ilen[i];
    }
  }
  ierr = MatCreateSeqAIJ(PETSC_COMM_SELF,dof*m,dof*n,0,ilen,&B);CHKERRQ(ierr);
  ierr = MatSetOption(B,MAT_COLUMNS_SORTED);CHKERRQ(ierr);
  ierr = PetscFree(ilen);CHKERRQ(ierr);
  ierr = PetscMalloc(nmax*sizeof(PetscInt),&icols);CHKERRQ(ierr);
  ii   = 0;
  for (i=0; i<m; i++) {
    ierr = MatGetRow_SeqAIJ(a,i,&ncols,&cols,&vals);CHKERRQ(ierr);
    for (j=0; j<dof; j++) {
      for (k=0; k<ncols; k++) {
        icols[k] = dof*cols[k]+j;
      }
      ierr = MatSetValues_SeqAIJ(B,1,&ii,ncols,icols,vals,INSERT_VALUES);CHKERRQ(ierr);
      ii++;
    }
    ierr = MatRestoreRow_SeqAIJ(a,i,&ncols,&cols,&vals);CHKERRQ(ierr);
  }
  ierr = PetscFree(icols);CHKERRQ(ierr);
  ierr = MatAssemblyBegin(B,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(B,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

  /* Fake support for "inplace" convert. */
  if (reuse == MAT_REUSE_MATRIX) {
    ierr = MatDestroy(A);CHKERRQ(ierr);
  }
  *newmat = B;
  PetscFunctionReturn(0);
}

#include "src/mat/impls/aij/mpi/mpiaij.h"
#undef __FUNCT__  
#define __FUNCT__ "MatConvert_MPIMAIJ_MPIAIJ"
PetscErrorCode MatConvert_MPIMAIJ_MPIAIJ(Mat A,const MatType newtype,MatReuse reuse,Mat *newmat)
{
  Mat_MPIMAIJ       *maij = (Mat_MPIMAIJ*)A->data;
  Mat               MatAIJ  = ((Mat_SeqMAIJ*)maij->AIJ->data)->AIJ,B;
  Mat               MatOAIJ = ((Mat_SeqMAIJ*)maij->OAIJ->data)->AIJ;
  Mat_SeqAIJ        *AIJ = (Mat_SeqAIJ*) MatAIJ->data;
  Mat_SeqAIJ        *OAIJ =(Mat_SeqAIJ*) MatOAIJ->data;
  Mat_MPIAIJ        *mpiaij = (Mat_MPIAIJ*) maij->A->data;
  PetscInt          dof = maij->dof,i,j,*dnz,*onz,nmax = 0,onmax = 0,*oicols,*icols,ncols,*cols,oncols,*ocols;
  PetscInt          rstart,cstart,*garray,ii,k;
  PetscErrorCode    ierr;
  PetscScalar       *vals,*ovals;

  PetscFunctionBegin;
  ierr = PetscMalloc2(A->m,PetscInt,&dnz,A->m,PetscInt,&onz);CHKERRQ(ierr);
  for (i=0; i<A->m/dof; i++) {
    nmax  = PetscMax(nmax,AIJ->ilen[i]);
    onmax = PetscMax(onmax,OAIJ->ilen[i]);
    for (j=0; j<dof; j++) {
      dnz[dof*i+j] = AIJ->ilen[i];
      onz[dof*i+j] = OAIJ->ilen[i];
    }
  }
  ierr = MatCreateMPIAIJ(A->comm,A->m,A->n,A->M,A->N,0,dnz,0,onz,&B);CHKERRQ(ierr);
  ierr = MatSetOption(B,MAT_COLUMNS_SORTED);CHKERRQ(ierr);
  ierr = PetscFree2(dnz,onz);CHKERRQ(ierr);

  ierr   = PetscMalloc2(nmax,PetscInt,&icols,onmax,PetscInt,&oicols);CHKERRQ(ierr);
  rstart = dof*mpiaij->rstart;
  cstart = dof*mpiaij->cstart;
  garray = mpiaij->garray;

  ii = rstart;
  for (i=0; i<A->m/dof; i++) {
    ierr = MatGetRow_SeqAIJ(MatAIJ,i,&ncols,&cols,&vals);CHKERRQ(ierr);
    ierr = MatGetRow_SeqAIJ(MatOAIJ,i,&oncols,&ocols,&ovals);CHKERRQ(ierr);
    for (j=0; j<dof; j++) {
      for (k=0; k<ncols; k++) {
        icols[k] = cstart + dof*cols[k]+j;
      }
      for (k=0; k<oncols; k++) {
        oicols[k] = dof*garray[ocols[k]]+j;
      }
      ierr = MatSetValues_MPIAIJ(B,1,&ii,ncols,icols,vals,INSERT_VALUES);CHKERRQ(ierr);
      ierr = MatSetValues_MPIAIJ(B,1,&ii,oncols,oicols,ovals,INSERT_VALUES);CHKERRQ(ierr);
      ii++;
    }
    ierr = MatRestoreRow_SeqAIJ(MatAIJ,i,&ncols,&cols,&vals);CHKERRQ(ierr);
    ierr = MatRestoreRow_SeqAIJ(MatOAIJ,i,&oncols,&ocols,&ovals);CHKERRQ(ierr);
  }
  ierr = PetscFree2(icols,oicols);CHKERRQ(ierr);

  ierr = MatAssemblyBegin(B,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(B,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

  /* Fake support for "inplace" convert. */
  if (reuse == MAT_REUSE_MATRIX) {
    ierr = MatDestroy(A);CHKERRQ(ierr);
  }
  *newmat = B;
  PetscFunctionReturn(0);
}



/* ---------------------------------------------------------------------------------- */
/*MC
  MatCreateMAIJ - Creates a matrix type providing restriction and interpolation 
  operations for multicomponent problems.  It interpolates each component the same
  way independently.  The matrix type is based on MATSEQAIJ for sequential matrices,
  and MATMPIAIJ for distributed matrices.

  Operations provided:
+ MatMult
. MatMultTranspose
. MatMultAdd
. MatMultTransposeAdd
- MatView

  Level: advanced

M*/
#undef __FUNCT__  
#define __FUNCT__ "MatCreateMAIJ" 
PetscErrorCode MatCreateMAIJ(Mat A,PetscInt dof,Mat *maij)
{
  PetscErrorCode ierr;
  PetscMPIInt    size;
  PetscInt       n;
  Mat_MPIMAIJ    *b;
  Mat            B;

  PetscFunctionBegin;
  ierr   = PetscObjectReference((PetscObject)A);CHKERRQ(ierr);

  if (dof == 1) {
    *maij = A;
  } else {
    ierr = MatCreate(A->comm,dof*A->m,dof*A->n,dof*A->M,dof*A->N,&B);CHKERRQ(ierr);
    B->assembled    = PETSC_TRUE;

    ierr = MPI_Comm_size(A->comm,&size);CHKERRQ(ierr);
    if (size == 1) {
      ierr = MatSetType(B,MATSEQMAIJ);CHKERRQ(ierr);
      B->ops->destroy = MatDestroy_SeqMAIJ;
      B->ops->view    = MatView_SeqMAIJ;
      b      = (Mat_MPIMAIJ*)B->data;
      b->dof = dof;
      b->AIJ = A;
      if (dof == 2) {
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
      } else if (dof == 5) {
        B->ops->mult             = MatMult_SeqMAIJ_5;
        B->ops->multadd          = MatMultAdd_SeqMAIJ_5;
        B->ops->multtranspose    = MatMultTranspose_SeqMAIJ_5;
        B->ops->multtransposeadd = MatMultTransposeAdd_SeqMAIJ_5;
      } else if (dof == 6) {
        B->ops->mult             = MatMult_SeqMAIJ_6;
        B->ops->multadd          = MatMultAdd_SeqMAIJ_6;
        B->ops->multtranspose    = MatMultTranspose_SeqMAIJ_6;
        B->ops->multtransposeadd = MatMultTransposeAdd_SeqMAIJ_6;
      } else if (dof == 8) {
        B->ops->mult             = MatMult_SeqMAIJ_8;
        B->ops->multadd          = MatMultAdd_SeqMAIJ_8;
        B->ops->multtranspose    = MatMultTranspose_SeqMAIJ_8;
        B->ops->multtransposeadd = MatMultTransposeAdd_SeqMAIJ_8;
      } else if (dof == 9) {
        B->ops->mult             = MatMult_SeqMAIJ_9;
        B->ops->multadd          = MatMultAdd_SeqMAIJ_9;
        B->ops->multtranspose    = MatMultTranspose_SeqMAIJ_9;
        B->ops->multtransposeadd = MatMultTransposeAdd_SeqMAIJ_9;
      } else if (dof == 16) {
        B->ops->mult             = MatMult_SeqMAIJ_16;
        B->ops->multadd          = MatMultAdd_SeqMAIJ_16;
        B->ops->multtranspose    = MatMultTranspose_SeqMAIJ_16;
        B->ops->multtransposeadd = MatMultTransposeAdd_SeqMAIJ_16;
      } else {
        SETERRQ1(PETSC_ERR_SUP,"Cannot handle a dof of %D. Send request for code to petsc-maint@mcs.anl.gov\n",dof);
      }
      B->ops->ptapsymbolic_seqaij = MatPtAPSymbolic_SeqAIJ_SeqMAIJ;
      B->ops->ptapnumeric_seqaij  = MatPtAPNumeric_SeqAIJ_SeqMAIJ;
      ierr = PetscObjectComposeFunctionDynamic((PetscObject)B,"MatConvert_seqmaij_seqaij_C","MatConvert_SeqMAIJ_SeqAIJ",MatConvert_SeqMAIJ_SeqAIJ);CHKERRQ(ierr);
    } else {
      Mat_MPIAIJ *mpiaij = (Mat_MPIAIJ *)A->data;
      IS         from,to;
      Vec        gvec;
      PetscInt   *garray,i;

      ierr = MatSetType(B,MATMPIMAIJ);CHKERRQ(ierr);
      B->ops->destroy = MatDestroy_MPIMAIJ;
      B->ops->view    = MatView_MPIMAIJ;
      b      = (Mat_MPIMAIJ*)B->data;
      b->dof = dof;
      b->A   = A;
      ierr = MatCreateMAIJ(mpiaij->A,dof,&b->AIJ);CHKERRQ(ierr);
      ierr = MatCreateMAIJ(mpiaij->B,dof,&b->OAIJ);CHKERRQ(ierr);

      ierr = VecGetSize(mpiaij->lvec,&n);CHKERRQ(ierr);
      ierr = VecCreateSeq(PETSC_COMM_SELF,n*dof,&b->w);CHKERRQ(ierr);
      ierr = VecSetBlockSize(b->w,dof);CHKERRQ(ierr);

      /* create two temporary Index sets for build scatter gather */
      ierr = PetscMalloc((n+1)*sizeof(PetscInt),&garray);CHKERRQ(ierr);
      for (i=0; i<n; i++) garray[i] = dof*mpiaij->garray[i];
      ierr = ISCreateBlock(A->comm,dof,n,garray,&from);CHKERRQ(ierr);
      ierr = PetscFree(garray);CHKERRQ(ierr);
      ierr = ISCreateStride(PETSC_COMM_SELF,n*dof,0,1,&to);CHKERRQ(ierr);

      /* create temporary global vector to generate scatter context */
      ierr = VecCreateMPI(A->comm,dof*A->n,dof*A->N,&gvec);CHKERRQ(ierr);
      ierr = VecSetBlockSize(gvec,dof);CHKERRQ(ierr);

      /* generate the scatter context */
      ierr = VecScatterCreate(gvec,from,b->w,to,&b->ctx);CHKERRQ(ierr);

      ierr = ISDestroy(from);CHKERRQ(ierr);
      ierr = ISDestroy(to);CHKERRQ(ierr);
      ierr = VecDestroy(gvec);CHKERRQ(ierr);

      B->ops->mult             = MatMult_MPIMAIJ_dof;
      B->ops->multtranspose    = MatMultTranspose_MPIMAIJ_dof;
      B->ops->multadd          = MatMultAdd_MPIMAIJ_dof;
      B->ops->multtransposeadd = MatMultTransposeAdd_MPIMAIJ_dof;
      ierr = PetscObjectComposeFunctionDynamic((PetscObject)B,"MatConvert_mpimaij_mpiaij_C","MatConvert_MPIMAIJ_MPIAIJ",MatConvert_MPIMAIJ_MPIAIJ);CHKERRQ(ierr);
    }
    *maij = B;
    ierr = MatView_Private(B);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}












