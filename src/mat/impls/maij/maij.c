
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

#include <../src/mat/impls/maij/maij.h> /*I "petscmat.h" I*/
#include <../src/mat/utils/freespace.h>
#include <petsc-private/vecimpl.h>

#undef __FUNCT__  
#define __FUNCT__ "MatMAIJGetAIJ"
/*@C
   MatMAIJGetAIJ - Get the AIJ matrix describing the blockwise action of the MAIJ matrix

   Not Collective, but if the MAIJ matrix is parallel, the AIJ matrix is also parallel

   Input Parameter:
.  A - the MAIJ matrix

   Output Parameter:
.  B - the AIJ matrix

   Level: advanced

   Notes: The reference count on the AIJ matrix is not increased so you should not destroy it.

.seealso: MatCreateMAIJ()
@*/
PetscErrorCode  MatMAIJGetAIJ(Mat A,Mat *B)
{
  PetscErrorCode ierr;
  PetscBool      ismpimaij,isseqmaij;

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
/*@C
   MatMAIJRedimension - Get an MAIJ matrix with the same action, but for a different block size

   Logically Collective

   Input Parameter:
+  A - the MAIJ matrix
-  dof - the block size for the new matrix

   Output Parameter:
.  B - the new MAIJ matrix

   Level: advanced

.seealso: MatCreateMAIJ()
@*/
PetscErrorCode  MatMAIJRedimension(Mat A,PetscInt dof,Mat *B)
{
  PetscErrorCode ierr;
  Mat            Aij = PETSC_NULL;

  PetscFunctionBegin;
  PetscValidLogicalCollectiveInt(A,dof,2);
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
  ierr = MatDestroy(&b->AIJ);CHKERRQ(ierr);
  ierr = PetscFree(A->data);CHKERRQ(ierr);
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
  ierr = MatDestroy(&B);CHKERRQ(ierr);
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
  ierr = MatDestroy(&B);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatDestroy_MPIMAIJ"
PetscErrorCode MatDestroy_MPIMAIJ(Mat A)
{
  PetscErrorCode ierr;
  Mat_MPIMAIJ    *b = (Mat_MPIMAIJ*)A->data;

  PetscFunctionBegin;
  ierr = MatDestroy(&b->AIJ);CHKERRQ(ierr);
  ierr = MatDestroy(&b->OAIJ);CHKERRQ(ierr);
  ierr = MatDestroy(&b->A);CHKERRQ(ierr);
  ierr = VecScatterDestroy(&b->ctx);CHKERRQ(ierr);
  ierr = VecDestroy(&b->w);CHKERRQ(ierr);
  ierr = PetscFree(A->data);CHKERRQ(ierr);
  ierr = PetscObjectChangeTypeName((PetscObject)A,0);CHKERRQ(ierr);
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

.seealso: MatMAIJGetAIJ(), MatMAIJRedimension(), MatCreateMAIJ()
M*/

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "MatCreate_MAIJ" 
PetscErrorCode  MatCreate_MAIJ(Mat A)
{
  PetscErrorCode ierr;
  Mat_MPIMAIJ    *b;
  PetscMPIInt    size;

  PetscFunctionBegin;
  ierr     = PetscNewLog(A,Mat_MPIMAIJ,&b);CHKERRQ(ierr);
  A->data  = (void*)b;
  ierr = PetscMemzero(A->ops,sizeof(struct _MatOps));CHKERRQ(ierr);

  b->AIJ  = 0;
  b->dof  = 0;  
  b->OAIJ = 0;
  b->ctx  = 0;
  b->w    = 0; 
  ierr = MPI_Comm_size(((PetscObject)A)->comm,&size);CHKERRQ(ierr);
  if (size == 1){
    ierr = PetscObjectChangeTypeName((PetscObject)A,MATSEQMAIJ);CHKERRQ(ierr);
  } else {
    ierr = PetscObjectChangeTypeName((PetscObject)A,MATMPIMAIJ);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}
EXTERN_C_END

/* --------------------------------------------------------------------------------------*/
#undef __FUNCT__  
#define __FUNCT__ "MatMult_SeqMAIJ_2"
PetscErrorCode MatMult_SeqMAIJ_2(Mat A,Vec xx,Vec yy)
{
  Mat_SeqMAIJ       *b = (Mat_SeqMAIJ*)A->data;
  Mat_SeqAIJ        *a = (Mat_SeqAIJ*)b->AIJ->data;
  const PetscScalar *x,*v;
  PetscScalar       *y, sum1, sum2;
  PetscErrorCode    ierr;
  PetscInt          nonzerorow=0,n,i,jrow,j;
  const PetscInt    m = b->AIJ->rmap->n,*idx,*ii;

  PetscFunctionBegin;
  ierr = VecGetArrayRead(xx,&x);CHKERRQ(ierr);
  ierr = VecGetArray(yy,&y);CHKERRQ(ierr);
  idx  = a->j;
  v    = a->a;
  ii   = a->i;

  for (i=0; i<m; i++) {
    jrow = ii[i];
    n    = ii[i+1] - jrow;
    sum1  = 0.0;
    sum2  = 0.0;
    nonzerorow += (n>0);
    for (j=0; j<n; j++) {
      sum1 += v[jrow]*x[2*idx[jrow]];
      sum2 += v[jrow]*x[2*idx[jrow]+1];
      jrow++;
     }
    y[2*i]   = sum1;
    y[2*i+1] = sum2;
  }

  ierr = PetscLogFlops(4.0*a->nz - 2.0*nonzerorow);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(xx,&x);CHKERRQ(ierr);
  ierr = VecRestoreArray(yy,&y);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatMultTranspose_SeqMAIJ_2"
PetscErrorCode MatMultTranspose_SeqMAIJ_2(Mat A,Vec xx,Vec yy)
{
  Mat_SeqMAIJ       *b = (Mat_SeqMAIJ*)A->data;
  Mat_SeqAIJ        *a = (Mat_SeqAIJ*)b->AIJ->data;
  const PetscScalar *x,*v;
  PetscScalar       *y,alpha1,alpha2;
  PetscErrorCode    ierr;
  PetscInt          n,i;
  const PetscInt    m = b->AIJ->rmap->n,*idx;

  PetscFunctionBegin; 
  ierr = VecSet(yy,0.0);CHKERRQ(ierr);
  ierr = VecGetArrayRead(xx,&x);CHKERRQ(ierr);
  ierr = VecGetArray(yy,&y);CHKERRQ(ierr);
 
  for (i=0; i<m; i++) {
    idx    = a->j + a->i[i] ;
    v      = a->a + a->i[i] ;
    n      = a->i[i+1] - a->i[i];
    alpha1 = x[2*i];
    alpha2 = x[2*i+1];
    while (n-->0) {y[2*(*idx)] += alpha1*(*v); y[2*(*idx)+1] += alpha2*(*v); idx++; v++;}
  }
  ierr = PetscLogFlops(4.0*a->nz);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(xx,&x);CHKERRQ(ierr);
  ierr = VecRestoreArray(yy,&y);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatMultAdd_SeqMAIJ_2"
PetscErrorCode MatMultAdd_SeqMAIJ_2(Mat A,Vec xx,Vec yy,Vec zz)
{
  Mat_SeqMAIJ       *b = (Mat_SeqMAIJ*)A->data;
  Mat_SeqAIJ        *a = (Mat_SeqAIJ*)b->AIJ->data;
  const PetscScalar *x,*v; 
  PetscScalar       *y,sum1, sum2;
  PetscErrorCode    ierr;
  PetscInt          n,i,jrow,j;
  const PetscInt    m = b->AIJ->rmap->n,*idx,*ii;

  PetscFunctionBegin;
  if (yy != zz) {ierr = VecCopy(yy,zz);CHKERRQ(ierr);}
  ierr = VecGetArrayRead(xx,&x);CHKERRQ(ierr);
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

  ierr = PetscLogFlops(4.0*a->nz);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(xx,&x);CHKERRQ(ierr);
  ierr = VecRestoreArray(zz,&y);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
#undef __FUNCT__  
#define __FUNCT__ "MatMultTransposeAdd_SeqMAIJ_2"
PetscErrorCode MatMultTransposeAdd_SeqMAIJ_2(Mat A,Vec xx,Vec yy,Vec zz)
{
  Mat_SeqMAIJ       *b = (Mat_SeqMAIJ*)A->data;
  Mat_SeqAIJ        *a = (Mat_SeqAIJ*)b->AIJ->data;
  const PetscScalar *x,*v;
  PetscScalar       *y,alpha1,alpha2;
  PetscErrorCode    ierr;
  PetscInt          n,i;
  const PetscInt    m = b->AIJ->rmap->n,*idx;

  PetscFunctionBegin; 
  if (yy != zz) {ierr = VecCopy(yy,zz);CHKERRQ(ierr);}
  ierr = VecGetArrayRead(xx,&x);CHKERRQ(ierr);
  ierr = VecGetArray(zz,&y);CHKERRQ(ierr);
 
  for (i=0; i<m; i++) {
    idx   = a->j + a->i[i] ;
    v     = a->a + a->i[i] ;
    n     = a->i[i+1] - a->i[i];
    alpha1 = x[2*i];
    alpha2 = x[2*i+1];
    while (n-->0) {y[2*(*idx)] += alpha1*(*v); y[2*(*idx)+1] += alpha2*(*v); idx++; v++;}
  }
  ierr = PetscLogFlops(4.0*a->nz);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(xx,&x);CHKERRQ(ierr);
  ierr = VecRestoreArray(zz,&y);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
/* --------------------------------------------------------------------------------------*/
#undef __FUNCT__  
#define __FUNCT__ "MatMult_SeqMAIJ_3"
PetscErrorCode MatMult_SeqMAIJ_3(Mat A,Vec xx,Vec yy)
{
  Mat_SeqMAIJ       *b = (Mat_SeqMAIJ*)A->data;
  Mat_SeqAIJ        *a = (Mat_SeqAIJ*)b->AIJ->data;
  const PetscScalar *x,*v; 
  PetscScalar       *y,sum1, sum2, sum3;
  PetscErrorCode    ierr;
  PetscInt          nonzerorow=0,n,i,jrow,j;
  const PetscInt    m = b->AIJ->rmap->n,*idx,*ii;

  PetscFunctionBegin;
  ierr = VecGetArrayRead(xx,&x);CHKERRQ(ierr);
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
    nonzerorow += (n>0);
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

  ierr = PetscLogFlops(6.0*a->nz - 3.0*nonzerorow);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(xx,&x);CHKERRQ(ierr);
  ierr = VecRestoreArray(yy,&y);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatMultTranspose_SeqMAIJ_3"
PetscErrorCode MatMultTranspose_SeqMAIJ_3(Mat A,Vec xx,Vec yy)
{
  Mat_SeqMAIJ       *b = (Mat_SeqMAIJ*)A->data;
  Mat_SeqAIJ        *a = (Mat_SeqAIJ*)b->AIJ->data;
  const PetscScalar *x,*v;
  PetscScalar       *y,alpha1,alpha2,alpha3;
  PetscErrorCode    ierr;
  PetscInt          n,i;
  const PetscInt    m = b->AIJ->rmap->n,*idx;

  PetscFunctionBegin; 
  ierr = VecSet(yy,0.0);CHKERRQ(ierr);
  ierr = VecGetArrayRead(xx,&x);CHKERRQ(ierr);
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
  ierr = PetscLogFlops(6.0*a->nz);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(xx,&x);CHKERRQ(ierr);
  ierr = VecRestoreArray(yy,&y);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatMultAdd_SeqMAIJ_3"
PetscErrorCode MatMultAdd_SeqMAIJ_3(Mat A,Vec xx,Vec yy,Vec zz)
{
  Mat_SeqMAIJ       *b = (Mat_SeqMAIJ*)A->data;
  Mat_SeqAIJ        *a = (Mat_SeqAIJ*)b->AIJ->data;
  const PetscScalar *x,*v;
  PetscScalar       *y,sum1, sum2, sum3;
  PetscErrorCode    ierr;
  PetscInt          n,i,jrow,j;
  const PetscInt    m = b->AIJ->rmap->n,*idx,*ii;

  PetscFunctionBegin;
  if (yy != zz) {ierr = VecCopy(yy,zz);CHKERRQ(ierr);}
  ierr = VecGetArrayRead(xx,&x);CHKERRQ(ierr);
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

  ierr = PetscLogFlops(6.0*a->nz);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(xx,&x);CHKERRQ(ierr);
  ierr = VecRestoreArray(zz,&y);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
#undef __FUNCT__  
#define __FUNCT__ "MatMultTransposeAdd_SeqMAIJ_3"
PetscErrorCode MatMultTransposeAdd_SeqMAIJ_3(Mat A,Vec xx,Vec yy,Vec zz)
{
  Mat_SeqMAIJ       *b = (Mat_SeqMAIJ*)A->data;
  Mat_SeqAIJ        *a = (Mat_SeqAIJ*)b->AIJ->data;
  const PetscScalar *x,*v;
  PetscScalar       *y,alpha1,alpha2,alpha3;
  PetscErrorCode    ierr;
  PetscInt          n,i;
  const PetscInt    m = b->AIJ->rmap->n,*idx;

  PetscFunctionBegin; 
  if (yy != zz) {ierr = VecCopy(yy,zz);CHKERRQ(ierr);}
  ierr = VecGetArrayRead(xx,&x);CHKERRQ(ierr);
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
  ierr = PetscLogFlops(6.0*a->nz);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(xx,&x);CHKERRQ(ierr);
  ierr = VecRestoreArray(zz,&y);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* ------------------------------------------------------------------------------*/
#undef __FUNCT__  
#define __FUNCT__ "MatMult_SeqMAIJ_4"
PetscErrorCode MatMult_SeqMAIJ_4(Mat A,Vec xx,Vec yy)
{
  Mat_SeqMAIJ       *b = (Mat_SeqMAIJ*)A->data;
  Mat_SeqAIJ        *a = (Mat_SeqAIJ*)b->AIJ->data;
  const PetscScalar *x,*v;
  PetscScalar       *y,sum1, sum2, sum3, sum4;
  PetscErrorCode    ierr;
  PetscInt          nonzerorow=0,n,i,jrow,j;
  const PetscInt    m = b->AIJ->rmap->n,*idx,*ii;

  PetscFunctionBegin;
  ierr = VecGetArrayRead(xx,&x);CHKERRQ(ierr);
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
    nonzerorow += (n>0);
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

  ierr = PetscLogFlops(8.0*a->nz - 4.0*nonzerorow);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(xx,&x);CHKERRQ(ierr);
  ierr = VecRestoreArray(yy,&y);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatMultTranspose_SeqMAIJ_4"
PetscErrorCode MatMultTranspose_SeqMAIJ_4(Mat A,Vec xx,Vec yy)
{
  Mat_SeqMAIJ       *b = (Mat_SeqMAIJ*)A->data;
  Mat_SeqAIJ        *a = (Mat_SeqAIJ*)b->AIJ->data;
  const PetscScalar *x,*v;
  PetscScalar       *y,alpha1,alpha2,alpha3,alpha4;
  PetscErrorCode    ierr;
  PetscInt          n,i;
  const PetscInt    m = b->AIJ->rmap->n,*idx;

  PetscFunctionBegin; 
  ierr = VecSet(yy,0.0);CHKERRQ(ierr);
  ierr = VecGetArrayRead(xx,&x);CHKERRQ(ierr);
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
  ierr = PetscLogFlops(8.0*a->nz);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(xx,&x);CHKERRQ(ierr);
  ierr = VecRestoreArray(yy,&y);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatMultAdd_SeqMAIJ_4"
PetscErrorCode MatMultAdd_SeqMAIJ_4(Mat A,Vec xx,Vec yy,Vec zz)
{
  Mat_SeqMAIJ       *b = (Mat_SeqMAIJ*)A->data;
  Mat_SeqAIJ        *a = (Mat_SeqAIJ*)b->AIJ->data;
  const PetscScalar *x,*v; 
  PetscScalar       *y,sum1, sum2, sum3, sum4;
  PetscErrorCode    ierr;
  PetscInt          n,i,jrow,j;
  const PetscInt    m = b->AIJ->rmap->n,*idx,*ii;

  PetscFunctionBegin;
  if (yy != zz) {ierr = VecCopy(yy,zz);CHKERRQ(ierr);}
  ierr = VecGetArrayRead(xx,&x);CHKERRQ(ierr);
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

  ierr = PetscLogFlops(8.0*a->nz);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(xx,&x);CHKERRQ(ierr);
  ierr = VecRestoreArray(zz,&y);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
#undef __FUNCT__  
#define __FUNCT__ "MatMultTransposeAdd_SeqMAIJ_4"
PetscErrorCode MatMultTransposeAdd_SeqMAIJ_4(Mat A,Vec xx,Vec yy,Vec zz)
{
  Mat_SeqMAIJ       *b = (Mat_SeqMAIJ*)A->data;
  Mat_SeqAIJ        *a = (Mat_SeqAIJ*)b->AIJ->data;
  const PetscScalar *x,*v;
  PetscScalar       *y,alpha1,alpha2,alpha3,alpha4;
  PetscErrorCode    ierr;
  PetscInt          n,i;
  const PetscInt    m = b->AIJ->rmap->n,*idx;

  PetscFunctionBegin; 
  if (yy != zz) {ierr = VecCopy(yy,zz);CHKERRQ(ierr);}
  ierr = VecGetArrayRead(xx,&x);CHKERRQ(ierr);
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
  ierr = PetscLogFlops(8.0*a->nz);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(xx,&x);CHKERRQ(ierr);
  ierr = VecRestoreArray(zz,&y);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
/* ------------------------------------------------------------------------------*/

#undef __FUNCT__  
#define __FUNCT__ "MatMult_SeqMAIJ_5"
PetscErrorCode MatMult_SeqMAIJ_5(Mat A,Vec xx,Vec yy)
{
  Mat_SeqMAIJ       *b = (Mat_SeqMAIJ*)A->data;
  Mat_SeqAIJ        *a = (Mat_SeqAIJ*)b->AIJ->data;
  const PetscScalar *x,*v; 
  PetscScalar       *y,sum1, sum2, sum3, sum4, sum5;
  PetscErrorCode    ierr;
  PetscInt          nonzerorow=0,n,i,jrow,j;
  const PetscInt    m = b->AIJ->rmap->n,*idx,*ii;

  PetscFunctionBegin;
  ierr = VecGetArrayRead(xx,&x);CHKERRQ(ierr);
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
    nonzerorow += (n>0);
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

  ierr = PetscLogFlops(10.0*a->nz - 5.0*nonzerorow);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(xx,&x);CHKERRQ(ierr);
  ierr = VecRestoreArray(yy,&y);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatMultTranspose_SeqMAIJ_5"
PetscErrorCode MatMultTranspose_SeqMAIJ_5(Mat A,Vec xx,Vec yy)
{
  Mat_SeqMAIJ       *b = (Mat_SeqMAIJ*)A->data;
  Mat_SeqAIJ        *a = (Mat_SeqAIJ*)b->AIJ->data;
  const PetscScalar *x,*v;
  PetscScalar       *y,alpha1,alpha2,alpha3,alpha4,alpha5;
  PetscErrorCode    ierr;
  PetscInt          n,i;
  const PetscInt    m = b->AIJ->rmap->n,*idx;

  PetscFunctionBegin; 
  ierr = VecSet(yy,0.0);CHKERRQ(ierr);
  ierr = VecGetArrayRead(xx,&x);CHKERRQ(ierr);
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
  ierr = PetscLogFlops(10.0*a->nz);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(xx,&x);CHKERRQ(ierr);
  ierr = VecRestoreArray(yy,&y);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatMultAdd_SeqMAIJ_5"
PetscErrorCode MatMultAdd_SeqMAIJ_5(Mat A,Vec xx,Vec yy,Vec zz)
{
  Mat_SeqMAIJ       *b = (Mat_SeqMAIJ*)A->data;
  Mat_SeqAIJ        *a = (Mat_SeqAIJ*)b->AIJ->data;
  const PetscScalar *x,*v;
  PetscScalar       *y,sum1, sum2, sum3, sum4, sum5;
  PetscErrorCode    ierr;
  PetscInt          n,i,jrow,j;
  const PetscInt    m = b->AIJ->rmap->n,*idx,*ii;

  PetscFunctionBegin;
  if (yy != zz) {ierr = VecCopy(yy,zz);CHKERRQ(ierr);}
  ierr = VecGetArrayRead(xx,&x);CHKERRQ(ierr);
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

  ierr = PetscLogFlops(10.0*a->nz);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(xx,&x);CHKERRQ(ierr);
  ierr = VecRestoreArray(zz,&y);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatMultTransposeAdd_SeqMAIJ_5"
PetscErrorCode MatMultTransposeAdd_SeqMAIJ_5(Mat A,Vec xx,Vec yy,Vec zz)
{
  Mat_SeqMAIJ       *b = (Mat_SeqMAIJ*)A->data;
  Mat_SeqAIJ        *a = (Mat_SeqAIJ*)b->AIJ->data;
  const PetscScalar *x,*v;
  PetscScalar       *y,alpha1,alpha2,alpha3,alpha4,alpha5;
  PetscErrorCode    ierr;
  PetscInt          n,i;
  const PetscInt    m = b->AIJ->rmap->n,*idx;

  PetscFunctionBegin; 
  if (yy != zz) {ierr = VecCopy(yy,zz);CHKERRQ(ierr);}
  ierr = VecGetArrayRead(xx,&x);CHKERRQ(ierr);
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
  ierr = PetscLogFlops(10.0*a->nz);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(xx,&x);CHKERRQ(ierr);
  ierr = VecRestoreArray(zz,&y);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* ------------------------------------------------------------------------------*/
#undef __FUNCT__  
#define __FUNCT__ "MatMult_SeqMAIJ_6"
PetscErrorCode MatMult_SeqMAIJ_6(Mat A,Vec xx,Vec yy)
{
  Mat_SeqMAIJ       *b = (Mat_SeqMAIJ*)A->data;
  Mat_SeqAIJ        *a = (Mat_SeqAIJ*)b->AIJ->data;
  const PetscScalar *x,*v;
  PetscScalar       *y,sum1, sum2, sum3, sum4, sum5, sum6;
  PetscErrorCode    ierr;
  PetscInt          nonzerorow=0,n,i,jrow,j;
  const PetscInt    m = b->AIJ->rmap->n,*idx,*ii;

  PetscFunctionBegin;
  ierr = VecGetArrayRead(xx,&x);CHKERRQ(ierr);
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
    nonzerorow += (n>0);
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

  ierr = PetscLogFlops(12.0*a->nz - 6.0*nonzerorow);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(xx,&x);CHKERRQ(ierr);
  ierr = VecRestoreArray(yy,&y);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatMultTranspose_SeqMAIJ_6"
PetscErrorCode MatMultTranspose_SeqMAIJ_6(Mat A,Vec xx,Vec yy)
{
  Mat_SeqMAIJ       *b = (Mat_SeqMAIJ*)A->data;
  Mat_SeqAIJ        *a = (Mat_SeqAIJ*)b->AIJ->data;
  const PetscScalar *x,*v;
  PetscScalar       *y,alpha1,alpha2,alpha3,alpha4,alpha5,alpha6;
  PetscErrorCode    ierr;
  PetscInt          n,i;
  const PetscInt    m = b->AIJ->rmap->n,*idx;

  PetscFunctionBegin; 
  ierr = VecSet(yy,0.0);CHKERRQ(ierr);
  ierr = VecGetArrayRead(xx,&x);CHKERRQ(ierr);
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
  ierr = PetscLogFlops(12.0*a->nz);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(xx,&x);CHKERRQ(ierr);
  ierr = VecRestoreArray(yy,&y);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatMultAdd_SeqMAIJ_6"
PetscErrorCode MatMultAdd_SeqMAIJ_6(Mat A,Vec xx,Vec yy,Vec zz)
{
  Mat_SeqMAIJ       *b = (Mat_SeqMAIJ*)A->data;
  Mat_SeqAIJ        *a = (Mat_SeqAIJ*)b->AIJ->data;
  const PetscScalar *x,*v;
  PetscScalar       *y,sum1, sum2, sum3, sum4, sum5, sum6;
  PetscErrorCode    ierr;
  PetscInt          n,i,jrow,j;
  const PetscInt    m = b->AIJ->rmap->n,*idx,*ii;

  PetscFunctionBegin;
  if (yy != zz) {ierr = VecCopy(yy,zz);CHKERRQ(ierr);}
  ierr = VecGetArrayRead(xx,&x);CHKERRQ(ierr);
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

  ierr = PetscLogFlops(12.0*a->nz);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(xx,&x);CHKERRQ(ierr);
  ierr = VecRestoreArray(zz,&y);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatMultTransposeAdd_SeqMAIJ_6"
PetscErrorCode MatMultTransposeAdd_SeqMAIJ_6(Mat A,Vec xx,Vec yy,Vec zz)
{
  Mat_SeqMAIJ       *b = (Mat_SeqMAIJ*)A->data;
  Mat_SeqAIJ        *a = (Mat_SeqAIJ*)b->AIJ->data;
  const PetscScalar *x,*v;
  PetscScalar       *y,alpha1,alpha2,alpha3,alpha4,alpha5,alpha6;
  PetscErrorCode    ierr;
  PetscInt          n,i;
  const PetscInt    m = b->AIJ->rmap->n,*idx;

  PetscFunctionBegin; 
  if (yy != zz) {ierr = VecCopy(yy,zz);CHKERRQ(ierr);}
  ierr = VecGetArrayRead(xx,&x);CHKERRQ(ierr);
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
  ierr = PetscLogFlops(12.0*a->nz);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(xx,&x);CHKERRQ(ierr);
  ierr = VecRestoreArray(zz,&y);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* ------------------------------------------------------------------------------*/
#undef __FUNCT__  
#define __FUNCT__ "MatMult_SeqMAIJ_7"
PetscErrorCode MatMult_SeqMAIJ_7(Mat A,Vec xx,Vec yy)
{
  Mat_SeqMAIJ       *b = (Mat_SeqMAIJ*)A->data;
  Mat_SeqAIJ        *a = (Mat_SeqAIJ*)b->AIJ->data;
  const PetscScalar *x,*v;
  PetscScalar       *y,sum1, sum2, sum3, sum4, sum5, sum6, sum7;
  PetscErrorCode    ierr;
  const PetscInt    m = b->AIJ->rmap->n,*idx,*ii;
  PetscInt          nonzerorow=0,n,i,jrow,j;

  PetscFunctionBegin;
  ierr = VecGetArrayRead(xx,&x);CHKERRQ(ierr);
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
    nonzerorow += (n>0);
    for (j=0; j<n; j++) {
      sum1 += v[jrow]*x[7*idx[jrow]];
      sum2 += v[jrow]*x[7*idx[jrow]+1];
      sum3 += v[jrow]*x[7*idx[jrow]+2];
      sum4 += v[jrow]*x[7*idx[jrow]+3];
      sum5 += v[jrow]*x[7*idx[jrow]+4];
      sum6 += v[jrow]*x[7*idx[jrow]+5];
      sum7 += v[jrow]*x[7*idx[jrow]+6];
      jrow++;
     }
    y[7*i]   = sum1;
    y[7*i+1] = sum2;
    y[7*i+2] = sum3;
    y[7*i+3] = sum4;
    y[7*i+4] = sum5;
    y[7*i+5] = sum6;
    y[7*i+6] = sum7;
  }

  ierr = PetscLogFlops(14.0*a->nz - 7.0*nonzerorow);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(xx,&x);CHKERRQ(ierr);
  ierr = VecRestoreArray(yy,&y);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatMultTranspose_SeqMAIJ_7"
PetscErrorCode MatMultTranspose_SeqMAIJ_7(Mat A,Vec xx,Vec yy)
{
  Mat_SeqMAIJ       *b = (Mat_SeqMAIJ*)A->data;
  Mat_SeqAIJ        *a = (Mat_SeqAIJ*)b->AIJ->data;
  const PetscScalar *x,*v;
  PetscScalar       *y,alpha1,alpha2,alpha3,alpha4,alpha5,alpha6,alpha7;
  PetscErrorCode    ierr;
  const PetscInt    m = b->AIJ->rmap->n,*idx;
  PetscInt          n,i;

  PetscFunctionBegin; 
  ierr = VecSet(yy,0.0);CHKERRQ(ierr);
  ierr = VecGetArrayRead(xx,&x);CHKERRQ(ierr);
  ierr = VecGetArray(yy,&y);CHKERRQ(ierr);

  for (i=0; i<m; i++) {
    idx    = a->j + a->i[i] ;
    v      = a->a + a->i[i] ;
    n      = a->i[i+1] - a->i[i];
    alpha1 = x[7*i];
    alpha2 = x[7*i+1];
    alpha3 = x[7*i+2];
    alpha4 = x[7*i+3];
    alpha5 = x[7*i+4];
    alpha6 = x[7*i+5];
    alpha7 = x[7*i+6];
    while (n-->0) {
      y[7*(*idx)]   += alpha1*(*v);
      y[7*(*idx)+1] += alpha2*(*v);
      y[7*(*idx)+2] += alpha3*(*v);
      y[7*(*idx)+3] += alpha4*(*v);
      y[7*(*idx)+4] += alpha5*(*v);
      y[7*(*idx)+5] += alpha6*(*v);
      y[7*(*idx)+6] += alpha7*(*v);
      idx++; v++;
    }
  }
  ierr = PetscLogFlops(14.0*a->nz);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(xx,&x);CHKERRQ(ierr);
  ierr = VecRestoreArray(yy,&y);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatMultAdd_SeqMAIJ_7"
PetscErrorCode MatMultAdd_SeqMAIJ_7(Mat A,Vec xx,Vec yy,Vec zz)
{
  Mat_SeqMAIJ       *b = (Mat_SeqMAIJ*)A->data;
  Mat_SeqAIJ        *a = (Mat_SeqAIJ*)b->AIJ->data;
  const PetscScalar *x,*v;
  PetscScalar       *y,sum1, sum2, sum3, sum4, sum5, sum6, sum7;
  PetscErrorCode    ierr;
  const PetscInt    m = b->AIJ->rmap->n,*idx,*ii;
  PetscInt          n,i,jrow,j;

  PetscFunctionBegin;
  if (yy != zz) {ierr = VecCopy(yy,zz);CHKERRQ(ierr);}
  ierr = VecGetArrayRead(xx,&x);CHKERRQ(ierr);
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
    for (j=0; j<n; j++) {
      sum1 += v[jrow]*x[7*idx[jrow]];
      sum2 += v[jrow]*x[7*idx[jrow]+1];
      sum3 += v[jrow]*x[7*idx[jrow]+2];
      sum4 += v[jrow]*x[7*idx[jrow]+3];
      sum5 += v[jrow]*x[7*idx[jrow]+4];
      sum6 += v[jrow]*x[7*idx[jrow]+5];
      sum7 += v[jrow]*x[7*idx[jrow]+6];
      jrow++;
     }
    y[7*i]   += sum1;
    y[7*i+1] += sum2;
    y[7*i+2] += sum3;
    y[7*i+3] += sum4;
    y[7*i+4] += sum5;
    y[7*i+5] += sum6;
    y[7*i+6] += sum7;
  }

  ierr = PetscLogFlops(14.0*a->nz);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(xx,&x);CHKERRQ(ierr);
  ierr = VecRestoreArray(zz,&y);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatMultTransposeAdd_SeqMAIJ_7"
PetscErrorCode MatMultTransposeAdd_SeqMAIJ_7(Mat A,Vec xx,Vec yy,Vec zz)
{
  Mat_SeqMAIJ       *b = (Mat_SeqMAIJ*)A->data;
  Mat_SeqAIJ        *a = (Mat_SeqAIJ*)b->AIJ->data;
  const PetscScalar *x,*v;
  PetscScalar       *y,alpha1,alpha2,alpha3,alpha4,alpha5,alpha6,alpha7;
  PetscErrorCode    ierr;
  const PetscInt    m = b->AIJ->rmap->n,*idx;
  PetscInt          n,i;

  PetscFunctionBegin; 
  if (yy != zz) {ierr = VecCopy(yy,zz);CHKERRQ(ierr);}
  ierr = VecGetArrayRead(xx,&x);CHKERRQ(ierr);
  ierr = VecGetArray(zz,&y);CHKERRQ(ierr);
  for (i=0; i<m; i++) {
    idx    = a->j + a->i[i] ;
    v      = a->a + a->i[i] ;
    n      = a->i[i+1] - a->i[i];
    alpha1 = x[7*i];
    alpha2 = x[7*i+1];
    alpha3 = x[7*i+2];
    alpha4 = x[7*i+3];
    alpha5 = x[7*i+4];
    alpha6 = x[7*i+5];
    alpha7 = x[7*i+6];
    while (n-->0) {
      y[7*(*idx)]   += alpha1*(*v);
      y[7*(*idx)+1] += alpha2*(*v);
      y[7*(*idx)+2] += alpha3*(*v);
      y[7*(*idx)+3] += alpha4*(*v);
      y[7*(*idx)+4] += alpha5*(*v);
      y[7*(*idx)+5] += alpha6*(*v);
      y[7*(*idx)+6] += alpha7*(*v);
      idx++; v++;
    }
  }
  ierr = PetscLogFlops(14.0*a->nz);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(xx,&x);CHKERRQ(ierr);
  ierr = VecRestoreArray(zz,&y);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatMult_SeqMAIJ_8"
PetscErrorCode MatMult_SeqMAIJ_8(Mat A,Vec xx,Vec yy)
{
  Mat_SeqMAIJ       *b = (Mat_SeqMAIJ*)A->data;
  Mat_SeqAIJ        *a = (Mat_SeqAIJ*)b->AIJ->data;
  const PetscScalar *x,*v;
  PetscScalar       *y,sum1, sum2, sum3, sum4, sum5, sum6, sum7, sum8;
  PetscErrorCode    ierr;
  const PetscInt    m = b->AIJ->rmap->n,*idx,*ii;
  PetscInt          nonzerorow=0,n,i,jrow,j;

  PetscFunctionBegin;
  ierr = VecGetArrayRead(xx,&x);CHKERRQ(ierr);
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
    nonzerorow += (n>0);
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

  ierr = PetscLogFlops(16.0*a->nz - 8.0*nonzerorow);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(xx,&x);CHKERRQ(ierr);
  ierr = VecRestoreArray(yy,&y);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatMultTranspose_SeqMAIJ_8"
PetscErrorCode MatMultTranspose_SeqMAIJ_8(Mat A,Vec xx,Vec yy)
{
  Mat_SeqMAIJ       *b = (Mat_SeqMAIJ*)A->data;
  Mat_SeqAIJ        *a = (Mat_SeqAIJ*)b->AIJ->data;
  const PetscScalar *x,*v;
  PetscScalar       *y,alpha1,alpha2,alpha3,alpha4,alpha5,alpha6,alpha7,alpha8;
  PetscErrorCode    ierr;
  const PetscInt    m = b->AIJ->rmap->n,*idx;
  PetscInt          n,i;

  PetscFunctionBegin; 
  ierr = VecSet(yy,0.0);CHKERRQ(ierr);
  ierr = VecGetArrayRead(xx,&x);CHKERRQ(ierr);
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
  ierr = PetscLogFlops(16.0*a->nz);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(xx,&x);CHKERRQ(ierr);
  ierr = VecRestoreArray(yy,&y);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatMultAdd_SeqMAIJ_8"
PetscErrorCode MatMultAdd_SeqMAIJ_8(Mat A,Vec xx,Vec yy,Vec zz)
{
  Mat_SeqMAIJ       *b = (Mat_SeqMAIJ*)A->data;
  Mat_SeqAIJ        *a = (Mat_SeqAIJ*)b->AIJ->data;
  const PetscScalar *x,*v;
  PetscScalar       *y,sum1, sum2, sum3, sum4, sum5, sum6, sum7, sum8;
  PetscErrorCode    ierr;
  const PetscInt    m = b->AIJ->rmap->n,*idx,*ii;
  PetscInt          n,i,jrow,j;

  PetscFunctionBegin;
  if (yy != zz) {ierr = VecCopy(yy,zz);CHKERRQ(ierr);}
  ierr = VecGetArrayRead(xx,&x);CHKERRQ(ierr);
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

  ierr = PetscLogFlops(16.0*a->nz);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(xx,&x);CHKERRQ(ierr);
  ierr = VecRestoreArray(zz,&y);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatMultTransposeAdd_SeqMAIJ_8"
PetscErrorCode MatMultTransposeAdd_SeqMAIJ_8(Mat A,Vec xx,Vec yy,Vec zz)
{
  Mat_SeqMAIJ       *b = (Mat_SeqMAIJ*)A->data;
  Mat_SeqAIJ        *a = (Mat_SeqAIJ*)b->AIJ->data;
  const PetscScalar *x,*v;
  PetscScalar       *y,alpha1,alpha2,alpha3,alpha4,alpha5,alpha6,alpha7,alpha8;
  PetscErrorCode    ierr;
  const PetscInt    m = b->AIJ->rmap->n,*idx;
  PetscInt          n,i;

  PetscFunctionBegin; 
  if (yy != zz) {ierr = VecCopy(yy,zz);CHKERRQ(ierr);}
  ierr = VecGetArrayRead(xx,&x);CHKERRQ(ierr);
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
  ierr = PetscLogFlops(16.0*a->nz);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(xx,&x);CHKERRQ(ierr);
  ierr = VecRestoreArray(zz,&y);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* ------------------------------------------------------------------------------*/
#undef __FUNCT__  
#define __FUNCT__ "MatMult_SeqMAIJ_9"
PetscErrorCode MatMult_SeqMAIJ_9(Mat A,Vec xx,Vec yy)
{
  Mat_SeqMAIJ       *b = (Mat_SeqMAIJ*)A->data;
  Mat_SeqAIJ        *a = (Mat_SeqAIJ*)b->AIJ->data;
  const PetscScalar *x,*v;
  PetscScalar       *y,sum1, sum2, sum3, sum4, sum5, sum6, sum7, sum8, sum9;
  PetscErrorCode    ierr;
  const PetscInt    m = b->AIJ->rmap->n,*idx,*ii;
  PetscInt          nonzerorow=0,n,i,jrow,j;

  PetscFunctionBegin;
  ierr = VecGetArrayRead(xx,&x);CHKERRQ(ierr);
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
    nonzerorow += (n>0);
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

  ierr = PetscLogFlops(18.0*a->nz - 9*nonzerorow);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(xx,&x);CHKERRQ(ierr);
  ierr = VecRestoreArray(yy,&y);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* ------------------------------------------------------------------------------*/

#undef __FUNCT__  
#define __FUNCT__ "MatMultTranspose_SeqMAIJ_9"
PetscErrorCode MatMultTranspose_SeqMAIJ_9(Mat A,Vec xx,Vec yy)
{
  Mat_SeqMAIJ       *b = (Mat_SeqMAIJ*)A->data;
  Mat_SeqAIJ        *a = (Mat_SeqAIJ*)b->AIJ->data;
  const PetscScalar *x,*v;
  PetscScalar       *y,alpha1,alpha2,alpha3,alpha4,alpha5,alpha6,alpha7,alpha8,alpha9;
  PetscErrorCode    ierr;
  const PetscInt    m = b->AIJ->rmap->n,*idx;
  PetscInt          n,i;

  PetscFunctionBegin; 
  ierr = VecSet(yy,0.0);CHKERRQ(ierr);
  ierr = VecGetArrayRead(xx,&x);CHKERRQ(ierr);
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
  ierr = PetscLogFlops(18.0*a->nz);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(xx,&x);CHKERRQ(ierr);
  ierr = VecRestoreArray(yy,&y);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatMultAdd_SeqMAIJ_9"
PetscErrorCode MatMultAdd_SeqMAIJ_9(Mat A,Vec xx,Vec yy,Vec zz)
{
  Mat_SeqMAIJ       *b = (Mat_SeqMAIJ*)A->data;
  Mat_SeqAIJ        *a = (Mat_SeqAIJ*)b->AIJ->data;
  const PetscScalar *x,*v;
  PetscScalar       *y,sum1, sum2, sum3, sum4, sum5, sum6, sum7, sum8, sum9;
  PetscErrorCode    ierr;
  const PetscInt    m = b->AIJ->rmap->n,*idx,*ii;
  PetscInt          n,i,jrow,j;

  PetscFunctionBegin;
  if (yy != zz) {ierr = VecCopy(yy,zz);CHKERRQ(ierr);}
  ierr = VecGetArrayRead(xx,&x);CHKERRQ(ierr);
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

  ierr = PetscLogFlops(18*a->nz);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(xx,&x);CHKERRQ(ierr);
  ierr = VecRestoreArray(zz,&y);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatMultTransposeAdd_SeqMAIJ_9"
PetscErrorCode MatMultTransposeAdd_SeqMAIJ_9(Mat A,Vec xx,Vec yy,Vec zz)
{
  Mat_SeqMAIJ       *b = (Mat_SeqMAIJ*)A->data;
  Mat_SeqAIJ        *a = (Mat_SeqAIJ*)b->AIJ->data;
  const PetscScalar *x,*v;
  PetscScalar       *y,alpha1,alpha2,alpha3,alpha4,alpha5,alpha6,alpha7,alpha8,alpha9;
  PetscErrorCode    ierr;
  const PetscInt    m = b->AIJ->rmap->n,*idx;
  PetscInt          n,i;

  PetscFunctionBegin; 
  if (yy != zz) {ierr = VecCopy(yy,zz);CHKERRQ(ierr);}
  ierr = VecGetArrayRead(xx,&x);CHKERRQ(ierr);
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
  ierr = PetscLogFlops(18.0*a->nz);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(xx,&x);CHKERRQ(ierr);
  ierr = VecRestoreArray(zz,&y);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
#undef __FUNCT__  
#define __FUNCT__ "MatMult_SeqMAIJ_10"
PetscErrorCode MatMult_SeqMAIJ_10(Mat A,Vec xx,Vec yy)
{
  Mat_SeqMAIJ       *b = (Mat_SeqMAIJ*)A->data;
  Mat_SeqAIJ        *a = (Mat_SeqAIJ*)b->AIJ->data;
  const PetscScalar *x,*v;
  PetscScalar       *y,sum1, sum2, sum3, sum4, sum5, sum6, sum7, sum8, sum9, sum10;
  PetscErrorCode    ierr;
  const PetscInt    m = b->AIJ->rmap->n,*idx,*ii;
  PetscInt          nonzerorow=0,n,i,jrow,j;

  PetscFunctionBegin;
  ierr = VecGetArrayRead(xx,&x);CHKERRQ(ierr);
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
    nonzerorow += (n>0);
    for (j=0; j<n; j++) {
      sum1  += v[jrow]*x[10*idx[jrow]];
      sum2  += v[jrow]*x[10*idx[jrow]+1];
      sum3  += v[jrow]*x[10*idx[jrow]+2];
      sum4  += v[jrow]*x[10*idx[jrow]+3];
      sum5  += v[jrow]*x[10*idx[jrow]+4];
      sum6  += v[jrow]*x[10*idx[jrow]+5];
      sum7  += v[jrow]*x[10*idx[jrow]+6];
      sum8  += v[jrow]*x[10*idx[jrow]+7];
      sum9  += v[jrow]*x[10*idx[jrow]+8];
      sum10 += v[jrow]*x[10*idx[jrow]+9];
      jrow++;
     }
    y[10*i]   = sum1;
    y[10*i+1] = sum2;
    y[10*i+2] = sum3;
    y[10*i+3] = sum4;
    y[10*i+4] = sum5;
    y[10*i+5] = sum6;
    y[10*i+6] = sum7;
    y[10*i+7] = sum8;
    y[10*i+8] = sum9;
    y[10*i+9] = sum10;
  }

  ierr = PetscLogFlops(20.0*a->nz - 10.0*nonzerorow);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(xx,&x);CHKERRQ(ierr);
  ierr = VecRestoreArray(yy,&y);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatMultAdd_SeqMAIJ_10"
PetscErrorCode MatMultAdd_SeqMAIJ_10(Mat A,Vec xx,Vec yy,Vec zz)
{
  Mat_SeqMAIJ       *b = (Mat_SeqMAIJ*)A->data;
  Mat_SeqAIJ        *a = (Mat_SeqAIJ*)b->AIJ->data;
  const PetscScalar *x,*v;
  PetscScalar       *y,sum1, sum2, sum3, sum4, sum5, sum6, sum7, sum8, sum9, sum10;
  PetscErrorCode    ierr;
  const PetscInt    m = b->AIJ->rmap->n,*idx,*ii;
  PetscInt          n,i,jrow,j;

  PetscFunctionBegin;
  if (yy != zz) {ierr = VecCopy(yy,zz);CHKERRQ(ierr);}
  ierr = VecGetArrayRead(xx,&x);CHKERRQ(ierr);
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
    for (j=0; j<n; j++) {
      sum1  += v[jrow]*x[10*idx[jrow]];
      sum2  += v[jrow]*x[10*idx[jrow]+1];
      sum3  += v[jrow]*x[10*idx[jrow]+2];
      sum4  += v[jrow]*x[10*idx[jrow]+3];
      sum5  += v[jrow]*x[10*idx[jrow]+4];
      sum6  += v[jrow]*x[10*idx[jrow]+5];
      sum7  += v[jrow]*x[10*idx[jrow]+6];
      sum8  += v[jrow]*x[10*idx[jrow]+7];
      sum9  += v[jrow]*x[10*idx[jrow]+8];
      sum10 += v[jrow]*x[10*idx[jrow]+9];
      jrow++;
     }
    y[10*i]   += sum1;
    y[10*i+1] += sum2;
    y[10*i+2] += sum3;
    y[10*i+3] += sum4;
    y[10*i+4] += sum5;
    y[10*i+5] += sum6;
    y[10*i+6] += sum7;
    y[10*i+7] += sum8;
    y[10*i+8] += sum9;
    y[10*i+9] += sum10;
  }

  ierr = PetscLogFlops(20.0*a->nz);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(xx,&x);CHKERRQ(ierr);
  ierr = VecRestoreArray(yy,&y);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatMultTranspose_SeqMAIJ_10"
PetscErrorCode MatMultTranspose_SeqMAIJ_10(Mat A,Vec xx,Vec yy)
{
  Mat_SeqMAIJ       *b = (Mat_SeqMAIJ*)A->data;
  Mat_SeqAIJ        *a = (Mat_SeqAIJ*)b->AIJ->data;
  const PetscScalar *x,*v;
  PetscScalar       *y,alpha1,alpha2,alpha3,alpha4,alpha5,alpha6,alpha7,alpha8,alpha9,alpha10;
  PetscErrorCode    ierr;
  const PetscInt    m = b->AIJ->rmap->n,*idx;
  PetscInt          n,i;

  PetscFunctionBegin; 
  ierr = VecSet(yy,0.0);CHKERRQ(ierr);
  ierr = VecGetArrayRead(xx,&x);CHKERRQ(ierr);
  ierr = VecGetArray(yy,&y);CHKERRQ(ierr);

  for (i=0; i<m; i++) {
    idx    = a->j + a->i[i] ;
    v      = a->a + a->i[i] ;
    n      = a->i[i+1] - a->i[i];
    alpha1 = x[10*i];
    alpha2 = x[10*i+1];
    alpha3 = x[10*i+2];
    alpha4 = x[10*i+3];
    alpha5 = x[10*i+4];
    alpha6 = x[10*i+5];
    alpha7 = x[10*i+6];
    alpha8 = x[10*i+7];
    alpha9 = x[10*i+8];
    alpha10 = x[10*i+9];
    while (n-->0) {
      y[10*(*idx)]   += alpha1*(*v);
      y[10*(*idx)+1] += alpha2*(*v);
      y[10*(*idx)+2] += alpha3*(*v);
      y[10*(*idx)+3] += alpha4*(*v);
      y[10*(*idx)+4] += alpha5*(*v);
      y[10*(*idx)+5] += alpha6*(*v);
      y[10*(*idx)+6] += alpha7*(*v);
      y[10*(*idx)+7] += alpha8*(*v);
      y[10*(*idx)+8] += alpha9*(*v);
      y[10*(*idx)+9] += alpha10*(*v);
      idx++; v++;
    }
  }
  ierr = PetscLogFlops(20.0*a->nz);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(xx,&x);CHKERRQ(ierr);
  ierr = VecRestoreArray(yy,&y);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatMultTransposeAdd_SeqMAIJ_10"
PetscErrorCode MatMultTransposeAdd_SeqMAIJ_10(Mat A,Vec xx,Vec yy,Vec zz)
{
  Mat_SeqMAIJ       *b = (Mat_SeqMAIJ*)A->data;
  Mat_SeqAIJ        *a = (Mat_SeqAIJ*)b->AIJ->data;
  const PetscScalar *x,*v;
  PetscScalar       *y,alpha1,alpha2,alpha3,alpha4,alpha5,alpha6,alpha7,alpha8,alpha9,alpha10;
  PetscErrorCode    ierr;
  const PetscInt    m = b->AIJ->rmap->n,*idx;
  PetscInt          n,i;

  PetscFunctionBegin; 
  if (yy != zz) {ierr = VecCopy(yy,zz);CHKERRQ(ierr);}
  ierr = VecGetArrayRead(xx,&x);CHKERRQ(ierr);
  ierr = VecGetArray(zz,&y);CHKERRQ(ierr);
  for (i=0; i<m; i++) {
    idx    = a->j + a->i[i] ;
    v      = a->a + a->i[i] ;
    n      = a->i[i+1] - a->i[i];
    alpha1 = x[10*i];
    alpha2 = x[10*i+1];
    alpha3 = x[10*i+2];
    alpha4 = x[10*i+3];
    alpha5 = x[10*i+4];
    alpha6 = x[10*i+5];
    alpha7 = x[10*i+6];
    alpha8 = x[10*i+7];
    alpha9 = x[10*i+8];
    alpha10 = x[10*i+9];
    while (n-->0) {
      y[10*(*idx)]   += alpha1*(*v);
      y[10*(*idx)+1] += alpha2*(*v);
      y[10*(*idx)+2] += alpha3*(*v);
      y[10*(*idx)+3] += alpha4*(*v);
      y[10*(*idx)+4] += alpha5*(*v);
      y[10*(*idx)+5] += alpha6*(*v);
      y[10*(*idx)+6] += alpha7*(*v);
      y[10*(*idx)+7] += alpha8*(*v);
      y[10*(*idx)+8] += alpha9*(*v);
      y[10*(*idx)+9] += alpha10*(*v);
      idx++; v++;
    }
  }
  ierr = PetscLogFlops(20.0*a->nz);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(xx,&x);CHKERRQ(ierr);
  ierr = VecRestoreArray(zz,&y);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


/*--------------------------------------------------------------------------------------------*/
#undef __FUNCT__  
#define __FUNCT__ "MatMult_SeqMAIJ_11"
PetscErrorCode MatMult_SeqMAIJ_11(Mat A,Vec xx,Vec yy)
{
  Mat_SeqMAIJ       *b = (Mat_SeqMAIJ*)A->data;
  Mat_SeqAIJ        *a = (Mat_SeqAIJ*)b->AIJ->data;
  const PetscScalar *x,*v;
  PetscScalar       *y,sum1, sum2, sum3, sum4, sum5, sum6, sum7, sum8, sum9, sum10, sum11;
  PetscErrorCode    ierr;
  const PetscInt    m = b->AIJ->rmap->n,*idx,*ii;
  PetscInt          nonzerorow=0,n,i,jrow,j;

  PetscFunctionBegin;
  ierr = VecGetArrayRead(xx,&x);CHKERRQ(ierr);
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
    nonzerorow += (n>0);
    for (j=0; j<n; j++) {
      sum1  += v[jrow]*x[11*idx[jrow]];
      sum2  += v[jrow]*x[11*idx[jrow]+1];
      sum3  += v[jrow]*x[11*idx[jrow]+2];
      sum4  += v[jrow]*x[11*idx[jrow]+3];
      sum5  += v[jrow]*x[11*idx[jrow]+4];
      sum6  += v[jrow]*x[11*idx[jrow]+5];
      sum7  += v[jrow]*x[11*idx[jrow]+6];
      sum8  += v[jrow]*x[11*idx[jrow]+7];
      sum9  += v[jrow]*x[11*idx[jrow]+8];
      sum10 += v[jrow]*x[11*idx[jrow]+9];
      sum11 += v[jrow]*x[11*idx[jrow]+10];
      jrow++;
     }
    y[11*i]   = sum1;
    y[11*i+1] = sum2;
    y[11*i+2] = sum3;
    y[11*i+3] = sum4;
    y[11*i+4] = sum5;
    y[11*i+5] = sum6;
    y[11*i+6] = sum7;
    y[11*i+7] = sum8;
    y[11*i+8] = sum9;
    y[11*i+9] = sum10;
    y[11*i+10] = sum11;
  }

  ierr = PetscLogFlops(22*a->nz - 11*nonzerorow);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(xx,&x);CHKERRQ(ierr);
  ierr = VecRestoreArray(yy,&y);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatMultAdd_SeqMAIJ_11"
PetscErrorCode MatMultAdd_SeqMAIJ_11(Mat A,Vec xx,Vec yy,Vec zz)
{
  Mat_SeqMAIJ       *b = (Mat_SeqMAIJ*)A->data;
  Mat_SeqAIJ        *a = (Mat_SeqAIJ*)b->AIJ->data;
  const PetscScalar *x,*v;
  PetscScalar       *y,sum1, sum2, sum3, sum4, sum5, sum6, sum7, sum8, sum9, sum10, sum11;
  PetscErrorCode    ierr;
  const PetscInt    m = b->AIJ->rmap->n,*idx,*ii;
  PetscInt          n,i,jrow,j;

  PetscFunctionBegin;
  if (yy != zz) {ierr = VecCopy(yy,zz);CHKERRQ(ierr);}
  ierr = VecGetArrayRead(xx,&x);CHKERRQ(ierr);
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
    for (j=0; j<n; j++) {
      sum1  += v[jrow]*x[11*idx[jrow]];
      sum2  += v[jrow]*x[11*idx[jrow]+1];
      sum3  += v[jrow]*x[11*idx[jrow]+2];
      sum4  += v[jrow]*x[11*idx[jrow]+3];
      sum5  += v[jrow]*x[11*idx[jrow]+4];
      sum6  += v[jrow]*x[11*idx[jrow]+5];
      sum7  += v[jrow]*x[11*idx[jrow]+6];
      sum8  += v[jrow]*x[11*idx[jrow]+7];
      sum9  += v[jrow]*x[11*idx[jrow]+8];
      sum10 += v[jrow]*x[11*idx[jrow]+9];
      sum11 += v[jrow]*x[11*idx[jrow]+10];
      jrow++;
     }
    y[11*i]   += sum1;
    y[11*i+1] += sum2;
    y[11*i+2] += sum3;
    y[11*i+3] += sum4;
    y[11*i+4] += sum5;
    y[11*i+5] += sum6;
    y[11*i+6] += sum7;
    y[11*i+7] += sum8;
    y[11*i+8] += sum9;
    y[11*i+9] += sum10;
    y[11*i+10] += sum11;
  }

  ierr = PetscLogFlops(22*a->nz);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(xx,&x);CHKERRQ(ierr);
  ierr = VecRestoreArray(yy,&y);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatMultTranspose_SeqMAIJ_11"
PetscErrorCode MatMultTranspose_SeqMAIJ_11(Mat A,Vec xx,Vec yy)
{
  Mat_SeqMAIJ       *b = (Mat_SeqMAIJ*)A->data;
  Mat_SeqAIJ        *a = (Mat_SeqAIJ*)b->AIJ->data;
  const PetscScalar *x,*v;
  PetscScalar       *y,alpha1,alpha2,alpha3,alpha4,alpha5,alpha6,alpha7,alpha8,alpha9,alpha10,alpha11;
  PetscErrorCode    ierr;
  const PetscInt    m = b->AIJ->rmap->n,*idx;
  PetscInt          n,i;

  PetscFunctionBegin; 
  ierr = VecSet(yy,0.0);CHKERRQ(ierr);
  ierr = VecGetArrayRead(xx,&x);CHKERRQ(ierr);
  ierr = VecGetArray(yy,&y);CHKERRQ(ierr);

  for (i=0; i<m; i++) {
    idx    = a->j + a->i[i] ;
    v      = a->a + a->i[i] ;
    n      = a->i[i+1] - a->i[i];
    alpha1 = x[11*i];
    alpha2 = x[11*i+1];
    alpha3 = x[11*i+2];
    alpha4 = x[11*i+3];
    alpha5 = x[11*i+4];
    alpha6 = x[11*i+5];
    alpha7 = x[11*i+6];
    alpha8 = x[11*i+7];
    alpha9 = x[11*i+8];
    alpha10 = x[11*i+9];
    alpha11 = x[11*i+10];
    while (n-->0) {
      y[11*(*idx)]   += alpha1*(*v);
      y[11*(*idx)+1] += alpha2*(*v);
      y[11*(*idx)+2] += alpha3*(*v);
      y[11*(*idx)+3] += alpha4*(*v);
      y[11*(*idx)+4] += alpha5*(*v);
      y[11*(*idx)+5] += alpha6*(*v);
      y[11*(*idx)+6] += alpha7*(*v);
      y[11*(*idx)+7] += alpha8*(*v);
      y[11*(*idx)+8] += alpha9*(*v);
      y[11*(*idx)+9] += alpha10*(*v);
      y[11*(*idx)+10] += alpha11*(*v);
      idx++; v++;
    }
  }
  ierr = PetscLogFlops(22*a->nz);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(xx,&x);CHKERRQ(ierr);
  ierr = VecRestoreArray(yy,&y);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatMultTransposeAdd_SeqMAIJ_11"
PetscErrorCode MatMultTransposeAdd_SeqMAIJ_11(Mat A,Vec xx,Vec yy,Vec zz)
{
  Mat_SeqMAIJ       *b = (Mat_SeqMAIJ*)A->data;
  Mat_SeqAIJ        *a = (Mat_SeqAIJ*)b->AIJ->data;
  const PetscScalar *x,*v;
  PetscScalar       *y,alpha1,alpha2,alpha3,alpha4,alpha5,alpha6,alpha7,alpha8,alpha9,alpha10,alpha11;
  PetscErrorCode    ierr;
  const PetscInt    m = b->AIJ->rmap->n,*idx;
  PetscInt          n,i;

  PetscFunctionBegin; 
  if (yy != zz) {ierr = VecCopy(yy,zz);CHKERRQ(ierr);}
  ierr = VecGetArrayRead(xx,&x);CHKERRQ(ierr);
  ierr = VecGetArray(zz,&y);CHKERRQ(ierr);
  for (i=0; i<m; i++) {
    idx    = a->j + a->i[i] ;
    v      = a->a + a->i[i] ;
    n      = a->i[i+1] - a->i[i];
    alpha1 = x[11*i];
    alpha2 = x[11*i+1];
    alpha3 = x[11*i+2];
    alpha4 = x[11*i+3];
    alpha5 = x[11*i+4];
    alpha6 = x[11*i+5];
    alpha7 = x[11*i+6];
    alpha8 = x[11*i+7];
    alpha9 = x[11*i+8];
    alpha10 = x[11*i+9];
    alpha11 = x[11*i+10];
    while (n-->0) {
      y[11*(*idx)]   += alpha1*(*v);
      y[11*(*idx)+1] += alpha2*(*v);
      y[11*(*idx)+2] += alpha3*(*v);
      y[11*(*idx)+3] += alpha4*(*v);
      y[11*(*idx)+4] += alpha5*(*v);
      y[11*(*idx)+5] += alpha6*(*v);
      y[11*(*idx)+6] += alpha7*(*v);
      y[11*(*idx)+7] += alpha8*(*v);
      y[11*(*idx)+8] += alpha9*(*v);
      y[11*(*idx)+9] += alpha10*(*v);
      y[11*(*idx)+10] += alpha11*(*v);
      idx++; v++;
    }
  }
  ierr = PetscLogFlops(22*a->nz);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(xx,&x);CHKERRQ(ierr);
  ierr = VecRestoreArray(zz,&y);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


/*--------------------------------------------------------------------------------------------*/
#undef __FUNCT__  
#define __FUNCT__ "MatMult_SeqMAIJ_16"
PetscErrorCode MatMult_SeqMAIJ_16(Mat A,Vec xx,Vec yy)
{
  Mat_SeqMAIJ       *b = (Mat_SeqMAIJ*)A->data;
  Mat_SeqAIJ        *a = (Mat_SeqAIJ*)b->AIJ->data;
  const PetscScalar *x,*v;
  PetscScalar        *y,sum1, sum2, sum3, sum4, sum5, sum6, sum7, sum8;
  PetscScalar        sum9, sum10, sum11, sum12, sum13, sum14, sum15, sum16;
  PetscErrorCode     ierr;
  const PetscInt     m = b->AIJ->rmap->n,*idx,*ii;
  PetscInt           nonzerorow=0,n,i,jrow,j;

  PetscFunctionBegin;
  ierr = VecGetArrayRead(xx,&x);CHKERRQ(ierr);
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
    nonzerorow += (n>0);
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

  ierr = PetscLogFlops(32.0*a->nz - 16.0*nonzerorow);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(xx,&x);CHKERRQ(ierr);
  ierr = VecRestoreArray(yy,&y);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatMultTranspose_SeqMAIJ_16"
PetscErrorCode MatMultTranspose_SeqMAIJ_16(Mat A,Vec xx,Vec yy)
{
  Mat_SeqMAIJ       *b = (Mat_SeqMAIJ*)A->data;
  Mat_SeqAIJ        *a = (Mat_SeqAIJ*)b->AIJ->data;
  const PetscScalar *x,*v;
  PetscScalar       *y,alpha1,alpha2,alpha3,alpha4,alpha5,alpha6,alpha7,alpha8;
  PetscScalar       alpha9,alpha10,alpha11,alpha12,alpha13,alpha14,alpha15,alpha16;
  PetscErrorCode    ierr;
  const PetscInt    m = b->AIJ->rmap->n,*idx;
  PetscInt          n,i;

  PetscFunctionBegin; 
  ierr = VecSet(yy,0.0);CHKERRQ(ierr);
  ierr = VecGetArrayRead(xx,&x);CHKERRQ(ierr);
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
  ierr = PetscLogFlops(32.0*a->nz);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(xx,&x);CHKERRQ(ierr);
  ierr = VecRestoreArray(yy,&y);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatMultAdd_SeqMAIJ_16"
PetscErrorCode MatMultAdd_SeqMAIJ_16(Mat A,Vec xx,Vec yy,Vec zz)
{
  Mat_SeqMAIJ       *b = (Mat_SeqMAIJ*)A->data;
  Mat_SeqAIJ        *a = (Mat_SeqAIJ*)b->AIJ->data;
  const PetscScalar *x,*v;
  PetscScalar       *y,sum1, sum2, sum3, sum4, sum5, sum6, sum7, sum8;
  PetscScalar       sum9, sum10, sum11, sum12, sum13, sum14, sum15, sum16;
  PetscErrorCode    ierr;
  const PetscInt    m = b->AIJ->rmap->n,*idx,*ii;
  PetscInt          n,i,jrow,j;

  PetscFunctionBegin;
  if (yy != zz) {ierr = VecCopy(yy,zz);CHKERRQ(ierr);}
  ierr = VecGetArrayRead(xx,&x);CHKERRQ(ierr);
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

  ierr = PetscLogFlops(32.0*a->nz);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(xx,&x);CHKERRQ(ierr);
  ierr = VecRestoreArray(zz,&y);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatMultTransposeAdd_SeqMAIJ_16"
PetscErrorCode MatMultTransposeAdd_SeqMAIJ_16(Mat A,Vec xx,Vec yy,Vec zz)
{
  Mat_SeqMAIJ       *b = (Mat_SeqMAIJ*)A->data;
  Mat_SeqAIJ        *a = (Mat_SeqAIJ*)b->AIJ->data;
  const PetscScalar *x,*v;
  PetscScalar       *y,alpha1,alpha2,alpha3,alpha4,alpha5,alpha6,alpha7,alpha8;
  PetscScalar       alpha9,alpha10,alpha11,alpha12,alpha13,alpha14,alpha15,alpha16;
  PetscErrorCode    ierr;
  const PetscInt    m = b->AIJ->rmap->n,*idx;
  PetscInt          n,i;

  PetscFunctionBegin; 
  if (yy != zz) {ierr = VecCopy(yy,zz);CHKERRQ(ierr);}
  ierr = VecGetArrayRead(xx,&x);CHKERRQ(ierr);
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
  ierr = PetscLogFlops(32.0*a->nz);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(xx,&x);CHKERRQ(ierr);
  ierr = VecRestoreArray(zz,&y);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*--------------------------------------------------------------------------------------------*/
#undef __FUNCT__  
#define __FUNCT__ "MatMult_SeqMAIJ_18"
PetscErrorCode MatMult_SeqMAIJ_18(Mat A,Vec xx,Vec yy)
{
  Mat_SeqMAIJ       *b = (Mat_SeqMAIJ*)A->data;
  Mat_SeqAIJ        *a = (Mat_SeqAIJ*)b->AIJ->data;
  const PetscScalar *x,*v;
  PetscScalar       *y,sum1, sum2, sum3, sum4, sum5, sum6, sum7, sum8;
  PetscScalar       sum9, sum10, sum11, sum12, sum13, sum14, sum15, sum16, sum17, sum18;
  PetscErrorCode    ierr;
  const PetscInt    m = b->AIJ->rmap->n,*idx,*ii;
  PetscInt          nonzerorow=0,n,i,jrow,j;

  PetscFunctionBegin;
  ierr = VecGetArrayRead(xx,&x);CHKERRQ(ierr);
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
    sum17 = 0.0;
    sum18 = 0.0;
    nonzerorow += (n>0);
    for (j=0; j<n; j++) {
      sum1  += v[jrow]*x[18*idx[jrow]];
      sum2  += v[jrow]*x[18*idx[jrow]+1];
      sum3  += v[jrow]*x[18*idx[jrow]+2];
      sum4  += v[jrow]*x[18*idx[jrow]+3];
      sum5  += v[jrow]*x[18*idx[jrow]+4];
      sum6  += v[jrow]*x[18*idx[jrow]+5];
      sum7  += v[jrow]*x[18*idx[jrow]+6];
      sum8  += v[jrow]*x[18*idx[jrow]+7];
      sum9  += v[jrow]*x[18*idx[jrow]+8];
      sum10 += v[jrow]*x[18*idx[jrow]+9];
      sum11 += v[jrow]*x[18*idx[jrow]+10];
      sum12 += v[jrow]*x[18*idx[jrow]+11];
      sum13 += v[jrow]*x[18*idx[jrow]+12];
      sum14 += v[jrow]*x[18*idx[jrow]+13];
      sum15 += v[jrow]*x[18*idx[jrow]+14];
      sum16 += v[jrow]*x[18*idx[jrow]+15];
      sum17 += v[jrow]*x[18*idx[jrow]+16];
      sum18 += v[jrow]*x[18*idx[jrow]+17];
      jrow++;
     }
    y[18*i]    = sum1;
    y[18*i+1]  = sum2;
    y[18*i+2]  = sum3;
    y[18*i+3]  = sum4;
    y[18*i+4]  = sum5;
    y[18*i+5]  = sum6;
    y[18*i+6]  = sum7;
    y[18*i+7]  = sum8;
    y[18*i+8]  = sum9;
    y[18*i+9]  = sum10;
    y[18*i+10] = sum11;
    y[18*i+11] = sum12;
    y[18*i+12] = sum13;
    y[18*i+13] = sum14;
    y[18*i+14] = sum15;
    y[18*i+15] = sum16;
    y[18*i+16] = sum17;
    y[18*i+17] = sum18;
  }

  ierr = PetscLogFlops(36.0*a->nz - 18.0*nonzerorow);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(xx,&x);CHKERRQ(ierr);
  ierr = VecRestoreArray(yy,&y);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatMultTranspose_SeqMAIJ_18"
PetscErrorCode MatMultTranspose_SeqMAIJ_18(Mat A,Vec xx,Vec yy)
{
  Mat_SeqMAIJ       *b = (Mat_SeqMAIJ*)A->data;
  Mat_SeqAIJ        *a = (Mat_SeqAIJ*)b->AIJ->data;
  const PetscScalar *x,*v;
  PetscScalar       *y,alpha1,alpha2,alpha3,alpha4,alpha5,alpha6,alpha7,alpha8;
  PetscScalar       alpha9,alpha10,alpha11,alpha12,alpha13,alpha14,alpha15,alpha16,alpha17,alpha18;
  PetscErrorCode    ierr;
  const PetscInt    m = b->AIJ->rmap->n,*idx;
  PetscInt          n,i;

  PetscFunctionBegin; 
  ierr = VecSet(yy,0.0);CHKERRQ(ierr);
  ierr = VecGetArrayRead(xx,&x);CHKERRQ(ierr);
  ierr = VecGetArray(yy,&y);CHKERRQ(ierr);

  for (i=0; i<m; i++) {
    idx    = a->j + a->i[i] ;
    v      = a->a + a->i[i] ;
    n      = a->i[i+1] - a->i[i];
    alpha1  = x[18*i];
    alpha2  = x[18*i+1];
    alpha3  = x[18*i+2];
    alpha4  = x[18*i+3];
    alpha5  = x[18*i+4];
    alpha6  = x[18*i+5];
    alpha7  = x[18*i+6];
    alpha8  = x[18*i+7];
    alpha9  = x[18*i+8];
    alpha10 = x[18*i+9];
    alpha11 = x[18*i+10];
    alpha12 = x[18*i+11];
    alpha13 = x[18*i+12];
    alpha14 = x[18*i+13];
    alpha15 = x[18*i+14];
    alpha16 = x[18*i+15];
    alpha17 = x[18*i+16];
    alpha18 = x[18*i+17];
    while (n-->0) {
      y[18*(*idx)]    += alpha1*(*v);
      y[18*(*idx)+1]  += alpha2*(*v);
      y[18*(*idx)+2]  += alpha3*(*v);
      y[18*(*idx)+3]  += alpha4*(*v);
      y[18*(*idx)+4]  += alpha5*(*v);
      y[18*(*idx)+5]  += alpha6*(*v);
      y[18*(*idx)+6]  += alpha7*(*v);
      y[18*(*idx)+7]  += alpha8*(*v);
      y[18*(*idx)+8]  += alpha9*(*v);
      y[18*(*idx)+9]  += alpha10*(*v);
      y[18*(*idx)+10] += alpha11*(*v);
      y[18*(*idx)+11] += alpha12*(*v);
      y[18*(*idx)+12] += alpha13*(*v);
      y[18*(*idx)+13] += alpha14*(*v);
      y[18*(*idx)+14] += alpha15*(*v);
      y[18*(*idx)+15] += alpha16*(*v);
      y[18*(*idx)+16] += alpha17*(*v);
      y[18*(*idx)+17] += alpha18*(*v);
      idx++; v++;
    }
  }
  ierr = PetscLogFlops(36.0*a->nz);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(xx,&x);CHKERRQ(ierr);
  ierr = VecRestoreArray(yy,&y);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatMultAdd_SeqMAIJ_18"
PetscErrorCode MatMultAdd_SeqMAIJ_18(Mat A,Vec xx,Vec yy,Vec zz)
{
  Mat_SeqMAIJ       *b = (Mat_SeqMAIJ*)A->data;
  Mat_SeqAIJ        *a = (Mat_SeqAIJ*)b->AIJ->data;
  const PetscScalar *x,*v;
  PetscScalar       *y,sum1, sum2, sum3, sum4, sum5, sum6, sum7, sum8;
  PetscScalar       sum9, sum10, sum11, sum12, sum13, sum14, sum15, sum16, sum17, sum18;
  PetscErrorCode    ierr;
  const PetscInt    m = b->AIJ->rmap->n,*idx,*ii;
  PetscInt          n,i,jrow,j;

  PetscFunctionBegin;
  if (yy != zz) {ierr = VecCopy(yy,zz);CHKERRQ(ierr);}
  ierr = VecGetArrayRead(xx,&x);CHKERRQ(ierr);
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
    sum17 = 0.0;
    sum18 = 0.0;
    for (j=0; j<n; j++) {
      sum1  += v[jrow]*x[18*idx[jrow]];
      sum2  += v[jrow]*x[18*idx[jrow]+1];
      sum3  += v[jrow]*x[18*idx[jrow]+2];
      sum4  += v[jrow]*x[18*idx[jrow]+3];
      sum5  += v[jrow]*x[18*idx[jrow]+4];
      sum6  += v[jrow]*x[18*idx[jrow]+5];
      sum7  += v[jrow]*x[18*idx[jrow]+6];
      sum8  += v[jrow]*x[18*idx[jrow]+7];
      sum9  += v[jrow]*x[18*idx[jrow]+8];
      sum10 += v[jrow]*x[18*idx[jrow]+9];
      sum11 += v[jrow]*x[18*idx[jrow]+10];
      sum12 += v[jrow]*x[18*idx[jrow]+11];
      sum13 += v[jrow]*x[18*idx[jrow]+12];
      sum14 += v[jrow]*x[18*idx[jrow]+13];
      sum15 += v[jrow]*x[18*idx[jrow]+14];
      sum16 += v[jrow]*x[18*idx[jrow]+15];
      sum17 += v[jrow]*x[18*idx[jrow]+16];
      sum18 += v[jrow]*x[18*idx[jrow]+17];
      jrow++;
     }
    y[18*i]    += sum1;
    y[18*i+1]  += sum2;
    y[18*i+2]  += sum3;
    y[18*i+3]  += sum4;
    y[18*i+4]  += sum5;
    y[18*i+5]  += sum6;
    y[18*i+6]  += sum7;
    y[18*i+7]  += sum8;
    y[18*i+8]  += sum9;
    y[18*i+9]  += sum10;
    y[18*i+10] += sum11;
    y[18*i+11] += sum12;
    y[18*i+12] += sum13;
    y[18*i+13] += sum14;
    y[18*i+14] += sum15;
    y[18*i+15] += sum16;
    y[18*i+16] += sum17;
    y[18*i+17] += sum18;
  }

  ierr = PetscLogFlops(36.0*a->nz);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(xx,&x);CHKERRQ(ierr);
  ierr = VecRestoreArray(zz,&y);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatMultTransposeAdd_SeqMAIJ_18"
PetscErrorCode MatMultTransposeAdd_SeqMAIJ_18(Mat A,Vec xx,Vec yy,Vec zz)
{
  Mat_SeqMAIJ       *b = (Mat_SeqMAIJ*)A->data;
  Mat_SeqAIJ        *a = (Mat_SeqAIJ*)b->AIJ->data;
  const PetscScalar *x,*v;
  PetscScalar       *y,alpha1,alpha2,alpha3,alpha4,alpha5,alpha6,alpha7,alpha8;
  PetscScalar       alpha9,alpha10,alpha11,alpha12,alpha13,alpha14,alpha15,alpha16,alpha17,alpha18;
  PetscErrorCode    ierr;
  const PetscInt    m = b->AIJ->rmap->n,*idx;
  PetscInt          n,i;

  PetscFunctionBegin; 
  if (yy != zz) {ierr = VecCopy(yy,zz);CHKERRQ(ierr);}
  ierr = VecGetArrayRead(xx,&x);CHKERRQ(ierr);
  ierr = VecGetArray(zz,&y);CHKERRQ(ierr);
  for (i=0; i<m; i++) {
    idx    = a->j + a->i[i] ;
    v      = a->a + a->i[i] ;
    n      = a->i[i+1] - a->i[i];
    alpha1 = x[18*i];
    alpha2 = x[18*i+1];
    alpha3 = x[18*i+2];
    alpha4 = x[18*i+3];
    alpha5 = x[18*i+4];
    alpha6 = x[18*i+5];
    alpha7 = x[18*i+6];
    alpha8 = x[18*i+7];
    alpha9  = x[18*i+8];
    alpha10 = x[18*i+9];
    alpha11 = x[18*i+10];
    alpha12 = x[18*i+11];
    alpha13 = x[18*i+12];
    alpha14 = x[18*i+13];
    alpha15 = x[18*i+14];
    alpha16 = x[18*i+15];
    alpha17 = x[18*i+16];
    alpha18 = x[18*i+17];
    while (n-->0) {
      y[18*(*idx)]   += alpha1*(*v);
      y[18*(*idx)+1] += alpha2*(*v);
      y[18*(*idx)+2] += alpha3*(*v);
      y[18*(*idx)+3] += alpha4*(*v);
      y[18*(*idx)+4] += alpha5*(*v);
      y[18*(*idx)+5] += alpha6*(*v);
      y[18*(*idx)+6] += alpha7*(*v);
      y[18*(*idx)+7] += alpha8*(*v);
      y[18*(*idx)+8]  += alpha9*(*v);
      y[18*(*idx)+9]  += alpha10*(*v);
      y[18*(*idx)+10] += alpha11*(*v);
      y[18*(*idx)+11] += alpha12*(*v);
      y[18*(*idx)+12] += alpha13*(*v);
      y[18*(*idx)+13] += alpha14*(*v);
      y[18*(*idx)+14] += alpha15*(*v);
      y[18*(*idx)+15] += alpha16*(*v);
      y[18*(*idx)+16] += alpha17*(*v);
      y[18*(*idx)+17] += alpha18*(*v);
      idx++; v++;
    }
  }
  ierr = PetscLogFlops(36.0*a->nz);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(xx,&x);CHKERRQ(ierr);
  ierr = VecRestoreArray(zz,&y);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatMult_SeqMAIJ_N"
PetscErrorCode MatMult_SeqMAIJ_N(Mat A,Vec xx,Vec yy)
{
  Mat_SeqMAIJ       *b = (Mat_SeqMAIJ*)A->data;
  Mat_SeqAIJ        *a = (Mat_SeqAIJ*)b->AIJ->data;
  const PetscScalar *x,*v;
  PetscScalar       *y,*sums;
  PetscErrorCode    ierr;
  const PetscInt    m = b->AIJ->rmap->n,*idx,*ii;
  PetscInt          n,i,jrow,j,dof = b->dof,k;

  PetscFunctionBegin;
  ierr = VecGetArrayRead(xx,&x);CHKERRQ(ierr);
  ierr = VecSet(yy,0.0);CHKERRQ(ierr);
  ierr = VecGetArray(yy,&y);CHKERRQ(ierr);
  idx  = a->j;
  v    = a->a;
  ii   = a->i;

  for (i=0; i<m; i++) {
    jrow = ii[i];
    n    = ii[i+1] - jrow;
    sums = y + dof*i;
    for (j=0; j<n; j++) {
      for (k=0; k<dof; k++) {
        sums[k]  += v[jrow]*x[dof*idx[jrow]+k];
      }
      jrow++;
    }
  }

  ierr = PetscLogFlops(2.0*dof*a->nz);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(xx,&x);CHKERRQ(ierr);
  ierr = VecRestoreArray(yy,&y);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatMultAdd_SeqMAIJ_N"
PetscErrorCode MatMultAdd_SeqMAIJ_N(Mat A,Vec xx,Vec yy,Vec zz)
{
  Mat_SeqMAIJ       *b = (Mat_SeqMAIJ*)A->data;
  Mat_SeqAIJ        *a = (Mat_SeqAIJ*)b->AIJ->data;
  const PetscScalar *x,*v;
  PetscScalar       *y,*sums;
  PetscErrorCode    ierr;
  const PetscInt    m = b->AIJ->rmap->n,*idx,*ii;
  PetscInt          n,i,jrow,j,dof = b->dof,k;

  PetscFunctionBegin;
  if (yy != zz) {ierr = VecCopy(yy,zz);CHKERRQ(ierr);}
  ierr = VecGetArrayRead(xx,&x);CHKERRQ(ierr);
  ierr = VecGetArray(zz,&y);CHKERRQ(ierr);
  idx  = a->j;
  v    = a->a;
  ii   = a->i;

  for (i=0; i<m; i++) {
    jrow = ii[i];
    n    = ii[i+1] - jrow;
    sums = y + dof*i;
    for (j=0; j<n; j++) {
      for (k=0; k<dof; k++) {
        sums[k]  += v[jrow]*x[dof*idx[jrow]+k];
      }
      jrow++;
    }
  }

  ierr = PetscLogFlops(2.0*dof*a->nz);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(xx,&x);CHKERRQ(ierr);
  ierr = VecRestoreArray(zz,&y);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatMultTranspose_SeqMAIJ_N"
PetscErrorCode MatMultTranspose_SeqMAIJ_N(Mat A,Vec xx,Vec yy)
{
  Mat_SeqMAIJ       *b = (Mat_SeqMAIJ*)A->data;
  Mat_SeqAIJ        *a = (Mat_SeqAIJ*)b->AIJ->data;
  const PetscScalar *x,*v,*alpha;
  PetscScalar       *y;
  PetscErrorCode    ierr;
  const PetscInt    m = b->AIJ->rmap->n,*idx,dof = b->dof;
  PetscInt          n,i,k;

  PetscFunctionBegin; 
  ierr = VecGetArrayRead(xx,&x);CHKERRQ(ierr);
  ierr = VecSet(yy,0.0);CHKERRQ(ierr);
  ierr = VecGetArray(yy,&y);CHKERRQ(ierr);
  for (i=0; i<m; i++) {
    idx    = a->j + a->i[i] ;
    v      = a->a + a->i[i] ;
    n      = a->i[i+1] - a->i[i];
    alpha  = x + dof*i;
    while (n-->0) {
      for (k=0; k<dof; k++) {
        y[dof*(*idx)+k] += alpha[k]*(*v);
      }
      idx++; v++;
    }
  }
  ierr = PetscLogFlops(2.0*dof*a->nz);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(xx,&x);CHKERRQ(ierr);
  ierr = VecRestoreArray(yy,&y);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatMultTransposeAdd_SeqMAIJ_N"
PetscErrorCode MatMultTransposeAdd_SeqMAIJ_N(Mat A,Vec xx,Vec yy,Vec zz)
{
  Mat_SeqMAIJ       *b = (Mat_SeqMAIJ*)A->data;
  Mat_SeqAIJ        *a = (Mat_SeqAIJ*)b->AIJ->data;
  const PetscScalar *x,*v,*alpha;
  PetscScalar       *y;
  PetscErrorCode    ierr;
  const PetscInt    m = b->AIJ->rmap->n,*idx,dof = b->dof;
  PetscInt          n,i,k;

  PetscFunctionBegin; 
  if (yy != zz) {ierr = VecCopy(yy,zz);CHKERRQ(ierr);}
  ierr = VecGetArrayRead(xx,&x);CHKERRQ(ierr);
  ierr = VecGetArray(zz,&y);CHKERRQ(ierr);
  for (i=0; i<m; i++) {
    idx    = a->j + a->i[i] ;
    v      = a->a + a->i[i] ;
    n      = a->i[i+1] - a->i[i];
    alpha  = x + dof*i;
    while (n-->0) {
      for (k=0; k<dof; k++) {
        y[dof*(*idx)+k] += alpha[k]*(*v);
      }
      idx++; v++;
    }
  }
  ierr = PetscLogFlops(2.0*dof*a->nz);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(xx,&x);CHKERRQ(ierr);
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
  ierr = VecScatterBegin(b->ctx,xx,b->w,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = (*b->AIJ->ops->mult)(b->AIJ,xx,yy);CHKERRQ(ierr);
  ierr = VecScatterEnd(b->ctx,xx,b->w,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
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
  ierr = VecScatterBegin(b->ctx,b->w,yy,ADD_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
  ierr = VecScatterEnd(b->ctx,b->w,yy,ADD_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
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
  ierr = VecScatterBegin(b->ctx,xx,b->w,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = (*b->AIJ->ops->multadd)(b->AIJ,xx,yy,zz);CHKERRQ(ierr);
  ierr = VecScatterEnd(b->ctx,xx,b->w,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
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
  ierr = VecScatterBegin(b->ctx,b->w,zz,ADD_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
  ierr = (*b->AIJ->ops->multtransposeadd)(b->AIJ,xx,yy,zz);CHKERRQ(ierr);
  ierr = VecScatterEnd(b->ctx,b->w,zz,ADD_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* ----------------------------------------------------------------*/
#undef __FUNCT__
#define __FUNCT__ "MatPtAPSymbolic_SeqAIJ_SeqMAIJ"
PetscErrorCode MatPtAPSymbolic_SeqAIJ_SeqMAIJ(Mat A,Mat PP,PetscReal fill,Mat *C) 
{
  /* This routine requires testing -- but it's getting better. */
  PetscErrorCode     ierr;
  PetscFreeSpaceList free_space=PETSC_NULL,current_space=PETSC_NULL;
  Mat_SeqMAIJ        *pp=(Mat_SeqMAIJ*)PP->data;
  Mat                P=pp->AIJ;
  Mat_SeqAIJ         *a=(Mat_SeqAIJ*)A->data,*p=(Mat_SeqAIJ*)P->data,*c;
  PetscInt           *pti,*ptj,*ptJ;
  PetscInt           *ci,*cj,*ptadenserow,*ptasparserow,*denserow,*sparserow,*ptaj;
  const PetscInt     an=A->cmap->N,am=A->rmap->N,pn=P->cmap->N,pm=P->rmap->N,ppdof=pp->dof;
  PetscInt           i,j,k,dof,pshift,ptnzi,arow,anzj,ptanzi,prow,pnzj,cnzi,cn;
  MatScalar          *ca;
  const PetscInt     *pi = p->i,*pj = p->j,*pjj,*ai=a->i,*aj=a->j,*ajj;

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
  ierr = PetscMalloc4(an,PetscInt,&ptadenserow,an,PetscInt,&ptasparserow,cn,PetscInt,&denserow,cn,PetscInt,&sparserow);CHKERRQ(ierr);
  ierr = PetscMemzero(ptadenserow,an*sizeof(PetscInt));CHKERRQ(ierr);
  ierr = PetscMemzero(denserow,cn*sizeof(PetscInt));CHKERRQ(ierr);

  /* Set initial free space to be nnz(A) scaled by aspect ratio of P. */
  /* This should be reasonable if sparsity of PtAP is similar to that of A. */
  /* Note, aspect ratio of P is the same as the aspect ratio of SeqAIJ inside P */
  ierr          = PetscFreeSpaceGet((ai[am]/pm)*pn,&free_space);
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
        ierr = PetscFreeSpaceGet(cnzi+current_space->total_array_size,&current_space);CHKERRQ(ierr);
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
  ierr = PetscFreeSpaceContiguous(&free_space,cj);CHKERRQ(ierr);
  ierr = PetscFree4(ptadenserow,ptasparserow,denserow,sparserow);CHKERRQ(ierr);
    
  /* Allocate space for ca */
  ierr = PetscMalloc((ci[cn]+1)*sizeof(MatScalar),&ca);CHKERRQ(ierr);
  ierr = PetscMemzero(ca,(ci[cn]+1)*sizeof(MatScalar));CHKERRQ(ierr);
  
  /* put together the new matrix */
  ierr = MatCreateSeqAIJWithArrays(((PetscObject)A)->comm,cn,cn,ci,cj,ca,C);CHKERRQ(ierr);

  /* MatCreateSeqAIJWithArrays flags matrix so PETSc doesn't free the user's arrays. */
  /* Since these are PETSc arrays, change flags to free them as necessary. */
  c          = (Mat_SeqAIJ *)((*C)->data);
  c->free_a  = PETSC_TRUE;
  c->free_ij = PETSC_TRUE;
  c->nonew   = 0;

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
  PetscErrorCode  ierr;
  Mat_SeqMAIJ     *pp=(Mat_SeqMAIJ*)PP->data;
  Mat             P=pp->AIJ;
  Mat_SeqAIJ      *a  = (Mat_SeqAIJ *) A->data;
  Mat_SeqAIJ      *p  = (Mat_SeqAIJ *) P->data;
  Mat_SeqAIJ      *c  = (Mat_SeqAIJ *) C->data;
  const PetscInt  *ai=a->i,*aj=a->j,*pi=p->i,*pj=p->j,*pJ=p->j,*pjj;
  const PetscInt  *ci=c->i,*cj=c->j,*cjj;
  const PetscInt  am=A->rmap->N,cn=C->cmap->N,cm=C->rmap->N,ppdof=pp->dof;
  PetscInt        i,j,k,pshift,poffset,anzi,pnzi,apnzj,nextap,pnzj,prow,crow,*apj,*apjdense;
  const MatScalar *aa=a->a,*pa=p->a,*pA=p->a,*paj;
  MatScalar       *ca=c->a,*caj,*apa;

  PetscFunctionBegin;
  /* Allocate temporary array for storage of one row of A*P */
  ierr = PetscMalloc3(cn,MatScalar,&apa,cn,PetscInt,&apj,cn,PetscInt,&apjdense);CHKERRQ(ierr);
  ierr = PetscMemzero(apa,cn*sizeof(MatScalar));CHKERRQ(ierr);
  ierr = PetscMemzero(apj,cn*sizeof(PetscInt));CHKERRQ(ierr);
  ierr = PetscMemzero(apjdense,cn*sizeof(PetscInt));CHKERRQ(ierr);

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
      ierr = PetscLogFlops(2.0*pnzj);CHKERRQ(ierr);
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
      ierr = PetscLogFlops(2.0*apnzj);CHKERRQ(ierr);
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
  ierr = PetscFree3(apa,apj,apjdense);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatPtAPSymbolic_MPIAIJ_MPIMAIJ"
PetscErrorCode MatPtAPSymbolic_MPIAIJ_MPIMAIJ(Mat A,Mat PP,PetscReal fill,Mat *C) 
{
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  /* MatPtAPSymbolic_MPIAIJ_MPIMAIJ() is not implemented yet. Convert PP to mpiaij format */
  ierr = MatConvert(PP,MATMPIAIJ,MAT_REUSE_MATRIX,&PP);CHKERRQ(ierr);
  ierr =(*PP->ops->ptapsymbolic)(A,PP,fill,C);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatPtAPNumeric_MPIAIJ_MPIMAIJ"
PetscErrorCode MatPtAPNumeric_MPIAIJ_MPIMAIJ(Mat A,Mat PP,Mat C) 
{
  PetscFunctionBegin;
  SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"MatPtAPNumeric is not implemented for MPIMAIJ matrix yet");
  PetscFunctionReturn(0);
}

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "MatConvert_SeqMAIJ_SeqAIJ"
PetscErrorCode  MatConvert_SeqMAIJ_SeqAIJ(Mat A, MatType newtype,MatReuse reuse,Mat *newmat)
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
  ierr = PetscMalloc(dof*m*sizeof(PetscInt),&ilen);CHKERRQ(ierr);
  for (i=0; i<m; i++) {
    nmax = PetscMax(nmax,aij->ilen[i]);
    for (j=0; j<dof; j++) {
      ilen[dof*i+j] = aij->ilen[i];
    }
  }
  ierr = MatCreateSeqAIJ(PETSC_COMM_SELF,dof*m,dof*n,0,ilen,&B);CHKERRQ(ierr);
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

  if (reuse == MAT_REUSE_MATRIX) {
    ierr = MatHeaderReplace(A,B);CHKERRQ(ierr);
  } else {
    *newmat = B;
  }
  PetscFunctionReturn(0);
}
EXTERN_C_END

#include <../src/mat/impls/aij/mpi/mpiaij.h>

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "MatConvert_MPIMAIJ_MPIAIJ"
PetscErrorCode  MatConvert_MPIMAIJ_MPIAIJ(Mat A, MatType newtype,MatReuse reuse,Mat *newmat)
{
  Mat_MPIMAIJ       *maij = (Mat_MPIMAIJ*)A->data;
  Mat               MatAIJ  = ((Mat_SeqMAIJ*)maij->AIJ->data)->AIJ,B;
  Mat               MatOAIJ = ((Mat_SeqMAIJ*)maij->OAIJ->data)->AIJ;
  Mat_SeqAIJ        *AIJ = (Mat_SeqAIJ*) MatAIJ->data;
  Mat_SeqAIJ        *OAIJ =(Mat_SeqAIJ*) MatOAIJ->data;
  Mat_MPIAIJ        *mpiaij = (Mat_MPIAIJ*) maij->A->data;
  PetscInt          dof = maij->dof,i,j,*dnz = PETSC_NULL,*onz = PETSC_NULL,nmax = 0,onmax = 0;
  PetscInt          *oicols = PETSC_NULL,*icols = PETSC_NULL,ncols,*cols = PETSC_NULL,oncols,*ocols = PETSC_NULL;
  PetscInt          rstart,cstart,*garray,ii,k;
  PetscErrorCode    ierr;
  PetscScalar       *vals,*ovals;

  PetscFunctionBegin;
  ierr = PetscMalloc2(A->rmap->n,PetscInt,&dnz,A->rmap->n,PetscInt,&onz);CHKERRQ(ierr);
  for (i=0; i<A->rmap->n/dof; i++) {
    nmax  = PetscMax(nmax,AIJ->ilen[i]);
    onmax = PetscMax(onmax,OAIJ->ilen[i]);
    for (j=0; j<dof; j++) {
      dnz[dof*i+j] = AIJ->ilen[i];
      onz[dof*i+j] = OAIJ->ilen[i];
    }
  }
  ierr = MatCreateAIJ(((PetscObject)A)->comm,A->rmap->n,A->cmap->n,A->rmap->N,A->cmap->N,0,dnz,0,onz,&B);CHKERRQ(ierr);
  ierr = PetscFree2(dnz,onz);CHKERRQ(ierr);

  ierr   = PetscMalloc2(nmax,PetscInt,&icols,onmax,PetscInt,&oicols);CHKERRQ(ierr);
  rstart = dof*maij->A->rmap->rstart;
  cstart = dof*maij->A->cmap->rstart;
  garray = mpiaij->garray;

  ii = rstart;
  for (i=0; i<A->rmap->n/dof; i++) {
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

  if (reuse == MAT_REUSE_MATRIX) {
    PetscInt refct = ((PetscObject)A)->refct; /* save ((PetscObject)A)->refct */
    ((PetscObject)A)->refct = 1;
    ierr = MatHeaderReplace(A,B);CHKERRQ(ierr);
    ((PetscObject)A)->refct = refct; /* restore ((PetscObject)A)->refct */
  } else {
    *newmat = B;
  }
  PetscFunctionReturn(0);
}
EXTERN_C_END

#undef __FUNCT__  
#define __FUNCT__ "MatGetSubMatrix_MAIJ"
PetscErrorCode  MatGetSubMatrix_MAIJ(Mat mat,IS isrow,IS iscol,MatReuse cll,Mat *newmat)
{
  PetscErrorCode ierr;
  Mat            A;

  PetscFunctionBegin;
  ierr = MatConvert(mat,MATAIJ,MAT_INITIAL_MATRIX,&A);CHKERRQ(ierr);
  ierr = MatGetSubMatrix(A,isrow,iscol,cll,newmat);CHKERRQ(ierr);
  ierr = MatDestroy(&A);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* ---------------------------------------------------------------------------------- */
#undef __FUNCT__  
#define __FUNCT__ "MatCreateMAIJ"
/*@C
  MatCreateMAIJ - Creates a matrix type providing restriction and interpolation
  operations for multicomponent problems.  It interpolates each component the same
  way independently.  The matrix type is based on MATSEQAIJ for sequential matrices,
  and MATMPIAIJ for distributed matrices.

  Collective

  Input Parameters:
+ A - the AIJ matrix describing the action on blocks
- dof - the block size (number of components per node)

  Output Parameter:
. maij - the new MAIJ matrix

  Operations provided:
+ MatMult
. MatMultTranspose
. MatMultAdd
. MatMultTransposeAdd
- MatView

  Level: advanced

.seealso: MatMAIJGetAIJ(), MatMAIJRedimension(), MATMAIJ
@*/
PetscErrorCode  MatCreateMAIJ(Mat A,PetscInt dof,Mat *maij)
{
  PetscErrorCode ierr;
  PetscMPIInt    size;
  PetscInt       n;
  Mat            B;

  PetscFunctionBegin;
  ierr   = PetscObjectReference((PetscObject)A);CHKERRQ(ierr);

  if (dof == 1) {
    *maij = A;
  } else {
    ierr = MatCreate(((PetscObject)A)->comm,&B);CHKERRQ(ierr);
    ierr = MatSetSizes(B,dof*A->rmap->n,dof*A->cmap->n,dof*A->rmap->N,dof*A->cmap->N);CHKERRQ(ierr);
    ierr = PetscLayoutSetBlockSize(B->rmap,dof);CHKERRQ(ierr);
    ierr = PetscLayoutSetBlockSize(B->cmap,dof);CHKERRQ(ierr);
    ierr = PetscLayoutSetUp(B->rmap);CHKERRQ(ierr);
    ierr = PetscLayoutSetUp(B->cmap);CHKERRQ(ierr);
    B->assembled    = PETSC_TRUE;

    ierr = MPI_Comm_size(((PetscObject)A)->comm,&size);CHKERRQ(ierr);
    if (size == 1) {
      Mat_SeqMAIJ    *b;

      ierr = MatSetType(B,MATSEQMAIJ);CHKERRQ(ierr);
      B->ops->destroy = MatDestroy_SeqMAIJ;
      B->ops->view    = MatView_SeqMAIJ;
      b      = (Mat_SeqMAIJ*)B->data;
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
      } else if (dof == 7) {
        B->ops->mult             = MatMult_SeqMAIJ_7;
        B->ops->multadd          = MatMultAdd_SeqMAIJ_7;
        B->ops->multtranspose    = MatMultTranspose_SeqMAIJ_7;
        B->ops->multtransposeadd = MatMultTransposeAdd_SeqMAIJ_7;
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
      } else if (dof == 10) {
        B->ops->mult             = MatMult_SeqMAIJ_10;
        B->ops->multadd          = MatMultAdd_SeqMAIJ_10;
        B->ops->multtranspose    = MatMultTranspose_SeqMAIJ_10;
        B->ops->multtransposeadd = MatMultTransposeAdd_SeqMAIJ_10;
      } else if (dof == 11) {
        B->ops->mult             = MatMult_SeqMAIJ_11;
        B->ops->multadd          = MatMultAdd_SeqMAIJ_11;
        B->ops->multtranspose    = MatMultTranspose_SeqMAIJ_11;
        B->ops->multtransposeadd = MatMultTransposeAdd_SeqMAIJ_11;
      } else if (dof == 16) {
        B->ops->mult             = MatMult_SeqMAIJ_16;
        B->ops->multadd          = MatMultAdd_SeqMAIJ_16;
        B->ops->multtranspose    = MatMultTranspose_SeqMAIJ_16;
        B->ops->multtransposeadd = MatMultTransposeAdd_SeqMAIJ_16;
      } else if (dof == 18) {
        B->ops->mult             = MatMult_SeqMAIJ_18;
        B->ops->multadd          = MatMultAdd_SeqMAIJ_18;
        B->ops->multtranspose    = MatMultTranspose_SeqMAIJ_18;
        B->ops->multtransposeadd = MatMultTransposeAdd_SeqMAIJ_18;
      } else {
        B->ops->mult             = MatMult_SeqMAIJ_N;
        B->ops->multadd          = MatMultAdd_SeqMAIJ_N;
        B->ops->multtranspose    = MatMultTranspose_SeqMAIJ_N;
        B->ops->multtransposeadd = MatMultTransposeAdd_SeqMAIJ_N;
      } 
      B->ops->ptapsymbolic_seqaij = MatPtAPSymbolic_SeqAIJ_SeqMAIJ;
      B->ops->ptapnumeric_seqaij  = MatPtAPNumeric_SeqAIJ_SeqMAIJ;
      ierr = PetscObjectComposeFunctionDynamic((PetscObject)B,"MatConvert_seqmaij_seqaij_C","MatConvert_SeqMAIJ_SeqAIJ",MatConvert_SeqMAIJ_SeqAIJ);CHKERRQ(ierr);
    } else {
      Mat_MPIAIJ  *mpiaij = (Mat_MPIAIJ *)A->data;
      Mat_MPIMAIJ *b;
      IS          from,to;
      Vec         gvec;

      ierr = MatSetType(B,MATMPIMAIJ);CHKERRQ(ierr);
      B->ops->destroy = MatDestroy_MPIMAIJ;
      B->ops->view    = MatView_MPIMAIJ;
      b      = (Mat_MPIMAIJ*)B->data;
      b->dof = dof;
      b->A   = A;
      ierr = MatCreateMAIJ(mpiaij->A,dof,&b->AIJ);CHKERRQ(ierr);
      ierr = MatCreateMAIJ(mpiaij->B,dof,&b->OAIJ);CHKERRQ(ierr);

      ierr = VecGetSize(mpiaij->lvec,&n);CHKERRQ(ierr);
      ierr = VecCreate(PETSC_COMM_SELF,&b->w);CHKERRQ(ierr);
      ierr = VecSetSizes(b->w,n*dof,n*dof);CHKERRQ(ierr);
      ierr = VecSetBlockSize(b->w,dof);CHKERRQ(ierr);
      ierr = VecSetType(b->w,VECSEQ);CHKERRQ(ierr);

      /* create two temporary Index sets for build scatter gather */
      ierr = ISCreateBlock(((PetscObject)A)->comm,dof,n,mpiaij->garray,PETSC_COPY_VALUES,&from);CHKERRQ(ierr);
      ierr = ISCreateStride(PETSC_COMM_SELF,n*dof,0,1,&to);CHKERRQ(ierr);

      /* create temporary global vector to generate scatter context */
      ierr = VecCreateMPIWithArray(((PetscObject)A)->comm,dof,dof*A->cmap->n,dof*A->cmap->N,PETSC_NULL,&gvec);CHKERRQ(ierr);

      /* generate the scatter context */
      ierr = VecScatterCreate(gvec,from,b->w,to,&b->ctx);CHKERRQ(ierr);

      ierr = ISDestroy(&from);CHKERRQ(ierr);
      ierr = ISDestroy(&to);CHKERRQ(ierr);
      ierr = VecDestroy(&gvec);CHKERRQ(ierr);

      B->ops->mult                = MatMult_MPIMAIJ_dof;
      B->ops->multtranspose       = MatMultTranspose_MPIMAIJ_dof;
      B->ops->multadd             = MatMultAdd_MPIMAIJ_dof;
      B->ops->multtransposeadd    = MatMultTransposeAdd_MPIMAIJ_dof;
      B->ops->ptapsymbolic_mpiaij = MatPtAPSymbolic_MPIAIJ_MPIMAIJ;
      B->ops->ptapnumeric_mpiaij  = MatPtAPNumeric_MPIAIJ_MPIMAIJ;
      ierr = PetscObjectComposeFunctionDynamic((PetscObject)B,"MatConvert_mpimaij_mpiaij_C","MatConvert_MPIMAIJ_MPIAIJ",MatConvert_MPIMAIJ_MPIAIJ);CHKERRQ(ierr);
    }
    B->ops->getsubmatrix        = MatGetSubMatrix_MAIJ;
    ierr = MatSetUp(B);CHKERRQ(ierr);
    *maij = B;
    ierr = MatView_Private(B);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}
