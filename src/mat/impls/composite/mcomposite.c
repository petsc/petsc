#define PETSCMAT_DLL

#include "private/matimpl.h"        /*I "petscmat.h" I*/

typedef struct _Mat_CompositeLink *Mat_CompositeLink;
struct _Mat_CompositeLink {
  Mat               mat;
  Vec               work;
  Mat_CompositeLink next,prev;
};
  
typedef struct {
  MatCompositeType  type;
  Mat_CompositeLink head,tail;
  Vec               work;
  PetscScalar       scale;        /* scale factor supplied with MatScale() */
  Vec               left,right;   /* left and right diagonal scaling provided with MatDiagonalScale() */
  Vec               leftwork,rightwork;
} Mat_Composite;

#undef __FUNCT__  
#define __FUNCT__ "MatDestroy_Composite"
PetscErrorCode MatDestroy_Composite(Mat mat)
{
  PetscErrorCode   ierr;
  Mat_Composite    *shell = (Mat_Composite*)mat->data;
  Mat_CompositeLink next = shell->head,oldnext;

  PetscFunctionBegin;
  while (next) {
    ierr = MatDestroy(next->mat);CHKERRQ(ierr);
    if (next->work && (!next->next || next->work != next->next->work)) {
      ierr = VecDestroy(next->work);CHKERRQ(ierr);
    }
    oldnext = next;
    next     = next->next;
    ierr     = PetscFree(oldnext);CHKERRQ(ierr);
  }
  if (shell->work) {ierr = VecDestroy(shell->work);CHKERRQ(ierr);}
  if (shell->left) {ierr = VecDestroy(shell->left);CHKERRQ(ierr);}
  if (shell->right) {ierr = VecDestroy(shell->right);CHKERRQ(ierr);}
  if (shell->leftwork) {ierr = VecDestroy(shell->leftwork);CHKERRQ(ierr);}
  if (shell->rightwork) {ierr = VecDestroy(shell->rightwork);CHKERRQ(ierr);}
  ierr      = PetscFree(shell);CHKERRQ(ierr);
  mat->data = 0;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatMult_Composite_Multiplicative"
PetscErrorCode MatMult_Composite_Multiplicative(Mat A,Vec x,Vec y)
{
  Mat_Composite     *shell = (Mat_Composite*)A->data;  
  Mat_CompositeLink next = shell->head;
  PetscErrorCode    ierr;
  Vec               in,out;

  PetscFunctionBegin;
  if (!next) SETERRQ(PETSC_ERR_ARG_WRONGSTATE,"Must provide at least one matrix with MatCompositeAddMat()");
  in = x;
  if (shell->right) {
    if (!shell->rightwork) {
      ierr = VecDuplicate(shell->right,&shell->rightwork);CHKERRQ(ierr);
    }
    ierr = VecPointwiseMult(shell->rightwork,shell->right,in);CHKERRQ(ierr);
    in   = shell->rightwork;
  }
  while (next->next) {
    if (!next->work) { /* should reuse previous work if the same size */
      ierr = MatGetVecs(next->mat,PETSC_NULL,&next->work);CHKERRQ(ierr);
    }
    out = next->work;
    ierr = MatMult(next->mat,in,out);CHKERRQ(ierr);
    in   = out;
    next = next->next;
  }
  ierr = MatMult(next->mat,in,y);CHKERRQ(ierr);
  if (shell->left) {
    ierr = VecPointwiseMult(y,shell->left,y);CHKERRQ(ierr);
  }
  ierr = VecScale(y,shell->scale);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatMultTranspose_Composite_Multiplicative"
PetscErrorCode MatMultTranspose_Composite_Multiplicative(Mat A,Vec x,Vec y)
{
  Mat_Composite     *shell = (Mat_Composite*)A->data;  
  Mat_CompositeLink tail = shell->tail;
  PetscErrorCode    ierr;
  Vec               in,out;

  PetscFunctionBegin;
  if (!tail) SETERRQ(PETSC_ERR_ARG_WRONGSTATE,"Must provide at least one matrix with MatCompositeAddMat()");
  in = x;
  if (shell->left) {
    if (!shell->leftwork) {
      ierr = VecDuplicate(shell->left,&shell->leftwork);CHKERRQ(ierr);
    }
    ierr = VecPointwiseMult(shell->leftwork,shell->left,in);CHKERRQ(ierr);
    in   = shell->leftwork;
  }
  while (tail->prev) {
    if (!tail->prev->work) { /* should reuse previous work if the same size */
      ierr = MatGetVecs(tail->mat,PETSC_NULL,&tail->prev->work);CHKERRQ(ierr);
    }
    out = tail->prev->work;
    ierr = MatMultTranspose(tail->mat,in,out);CHKERRQ(ierr);
    in   = out;
    tail = tail->prev;
  }
  ierr = MatMultTranspose(tail->mat,in,y);CHKERRQ(ierr);
  if (shell->right) {
    ierr = VecPointwiseMult(y,shell->right,y);CHKERRQ(ierr);
  }
  ierr = VecScale(y,shell->scale);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatMult_Composite"
PetscErrorCode MatMult_Composite(Mat A,Vec x,Vec y)
{
  Mat_Composite     *shell = (Mat_Composite*)A->data;  
  Mat_CompositeLink next = shell->head;
  PetscErrorCode    ierr;
  Vec               in;

  PetscFunctionBegin;
  if (!next) SETERRQ(PETSC_ERR_ARG_WRONGSTATE,"Must provide at least one matrix with MatCompositeAddMat()");
  in = x;
  if (shell->right) {
    if (!shell->rightwork) {
      ierr = VecDuplicate(shell->right,&shell->rightwork);CHKERRQ(ierr);
    }
    ierr = VecPointwiseMult(shell->rightwork,shell->right,in);CHKERRQ(ierr);
    in   = shell->rightwork;
  }
  ierr = MatMult(next->mat,in,y);CHKERRQ(ierr);
  while ((next = next->next)) {
    ierr = MatMultAdd(next->mat,in,y,y);CHKERRQ(ierr);
  }
  if (shell->left) {
    ierr = VecPointwiseMult(y,shell->left,y);CHKERRQ(ierr);
  }
  ierr = VecScale(y,shell->scale);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatMultTranspose_Composite"
PetscErrorCode MatMultTranspose_Composite(Mat A,Vec x,Vec y)
{
  Mat_Composite     *shell = (Mat_Composite*)A->data;  
  Mat_CompositeLink next = shell->head;
  PetscErrorCode    ierr;
  Vec               in;

  PetscFunctionBegin;
  if (!next) SETERRQ(PETSC_ERR_ARG_WRONGSTATE,"Must provide at least one matrix with MatCompositeAddMat()");
  in = x;
  if (shell->left) {
    if (!shell->leftwork) {
      ierr = VecDuplicate(shell->left,&shell->leftwork);CHKERRQ(ierr);
    }
    ierr = VecPointwiseMult(shell->leftwork,shell->left,in);CHKERRQ(ierr);
    in   = shell->leftwork;
  }
  ierr = MatMultTranspose(next->mat,in,y);CHKERRQ(ierr);
  while ((next = next->next)) {
    ierr = MatMultTransposeAdd(next->mat,in,y,y);CHKERRQ(ierr);
  }
  if (shell->right) {
    ierr = VecPointwiseMult(y,shell->right,y);CHKERRQ(ierr);
  }
  ierr = VecScale(y,shell->scale);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatGetDiagonal_Composite"
PetscErrorCode MatGetDiagonal_Composite(Mat A,Vec v)
{
  Mat_Composite     *shell = (Mat_Composite*)A->data;  
  Mat_CompositeLink next = shell->head;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  if (!next) SETERRQ(PETSC_ERR_ARG_WRONGSTATE,"Must provide at least one matrix with MatCompositeAddMat()");
  if (shell->right || shell->left) SETERRQ(PETSC_ERR_SUP,"Cannot get diagonal if left or right scaling");

  ierr = MatGetDiagonal(next->mat,v);CHKERRQ(ierr);
  if (next->next && !shell->work) {
    ierr = VecDuplicate(v,&shell->work);CHKERRQ(ierr);
  }
  while ((next = next->next)) {
    ierr = MatGetDiagonal(next->mat,shell->work);CHKERRQ(ierr);
    ierr = VecAXPY(v,1.0,shell->work);CHKERRQ(ierr);
  }
  ierr = VecScale(v,shell->scale);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatAssemblyEnd_Composite"
PetscErrorCode MatAssemblyEnd_Composite(Mat Y,MatAssemblyType t)
{
  PetscErrorCode ierr;
  PetscTruth     flg = PETSC_FALSE;

  PetscFunctionBegin;
  ierr = PetscOptionsGetTruth(((PetscObject)Y)->prefix,"-mat_composite_merge",&flg,PETSC_NULL);CHKERRQ(ierr);
  if (flg) {
    ierr = MatCompositeMerge(Y);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatScale_Composite"
PetscErrorCode MatScale_Composite(Mat inA,PetscScalar alpha)
{
  Mat_Composite  *a = (Mat_Composite*)inA->data;

  PetscFunctionBegin;
  a->scale *= alpha;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatDiagonalScale_Composite"
PetscErrorCode MatDiagonalScale_Composite(Mat inA,Vec left,Vec right)
{
  Mat_Composite  *a = (Mat_Composite*)inA->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (left) {
    if (!a->left) {
      ierr = VecDuplicate(left,&a->left);CHKERRQ(ierr);
      ierr = VecCopy(left,a->left);CHKERRQ(ierr);
    } else {
      ierr = VecPointwiseMult(a->left,left,a->left);CHKERRQ(ierr);
    }
  }
  if (right) {
    if (!a->right) {
      ierr = VecDuplicate(right,&a->right);CHKERRQ(ierr);
      ierr = VecCopy(right,a->right);CHKERRQ(ierr);
    } else {
      ierr = VecPointwiseMult(a->right,right,a->right);CHKERRQ(ierr);
    }
  }
  PetscFunctionReturn(0);
}

static struct _MatOps MatOps_Values = {0,
       0,
       0,
       MatMult_Composite,
       0,
/* 5*/ MatMultTranspose_Composite,
       0,
       0,
       0,
       0,
/*10*/ 0,
       0,
       0,
       0,
       0,
/*15*/ 0,
       0,
       MatGetDiagonal_Composite,
       MatDiagonalScale_Composite,
       0,
/*20*/ 0,
       MatAssemblyEnd_Composite,
       0,
       0,
/*24*/ 0,
       0,
       0,
       0,
       0,
/*29*/ 0,
       0,
       0,
       0,
       0,
/*34*/ 0,
       0,
       0,
       0,
       0,
/*39*/ 0,
       0,
       0,
       0,
       0,
/*44*/ 0,
       MatScale_Composite,
       0,
       0,
       0,
/*49*/ 0,
       0,
       0,
       0,
       0,
/*54*/ 0,
       0,
       0,
       0,
       0,
/*59*/ 0,
       MatDestroy_Composite,
       0,
       0,
       0,
/*64*/ 0,
       0,
       0,
       0,
       0,
/*69*/ 0,
       0,
       0,
       0,
       0,
/*74*/ 0,
       0,
       0,
       0,
       0,
/*79*/ 0,
       0,
       0,
       0,
       0,
/*84*/ 0,
       0,
       0,
       0,
       0,
/*89*/ 0,
       0,
       0,
       0,
       0,
/*94*/ 0,
       0,
       0,
       0};

/*MC
   MATCOMPOSITE - A matrix defined by the sum (or product) of one or more matrices (all matrices are of same size and parallel layout).

   Notes: to use the product of the matrices call MatCompositeSetType(mat,MAT_COMPOSITE_MULTIPLICATIVE);

  Level: advanced

.seealso: MatCreateComposite(), MatCompositeAddMat(), MatSetType(), MatCompositeMerge(), MatCompositeSetType(), MatCompositeType
M*/

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "MatCreate_Composite"
PetscErrorCode PETSCMAT_DLLEXPORT MatCreate_Composite(Mat A)
{
  Mat_Composite  *b;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscNewLog(A,Mat_Composite,&b);CHKERRQ(ierr);
  A->data = (void*)b;
  ierr = PetscMemcpy(A->ops,&MatOps_Values,sizeof(struct _MatOps));CHKERRQ(ierr);

  ierr = PetscLayoutSetBlockSize(A->rmap,1);CHKERRQ(ierr);
  ierr = PetscLayoutSetBlockSize(A->cmap,1);CHKERRQ(ierr);
  ierr = PetscLayoutSetUp(A->rmap);CHKERRQ(ierr);
  ierr = PetscLayoutSetUp(A->cmap);CHKERRQ(ierr);

  A->assembled     = PETSC_TRUE;
  A->preallocated  = PETSC_FALSE;
  b->type          = MAT_COMPOSITE_ADDITIVE;
  b->scale         = 1.0;
  ierr = PetscObjectChangeTypeName((PetscObject)A,MATCOMPOSITE);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
EXTERN_C_END

#undef __FUNCT__  
#define __FUNCT__ "MatCreateComposite"
/*@C
   MatCreateComposite - Creates a matrix as the sum of zero or more matrices

  Collective on MPI_Comm

   Input Parameters:
+  comm - MPI communicator
.  nmat - number of matrices to put in
-  mats - the matrices

   Output Parameter:
.  A - the matrix

   Level: advanced

   Notes:
     Alternative construction 
$       MatCreate(comm,&mat);
$       MatSetSizes(mat,m,n,M,N);
$       MatSetType(mat,MATCOMPOSITE);
$       MatCompositeAddMat(mat,mats[0]);
$       ....
$       MatCompositeAddMat(mat,mats[nmat-1]);
$       MatAssemblyBegin(mat,MAT_FINAL_ASSEMBLY);
$       MatAssemblyEnd(mat,MAT_FINAL_ASSEMBLY);

     For the multiplicative form the product is mat[nmat-1]*mat[nmat-2]*....*mat[0]

.seealso: MatDestroy(), MatMult(), MatCompositeAddMat(), MatCompositeMerge(), MatCompositeSetType(), MatCompositeType

@*/
PetscErrorCode PETSCMAT_DLLEXPORT MatCreateComposite(MPI_Comm comm,PetscInt nmat,const Mat *mats,Mat *mat)
{
  PetscErrorCode ierr;
  PetscInt       m,n,M,N,i;
  
  PetscFunctionBegin;
  if (nmat < 1) SETERRQ(PETSC_ERR_ARG_OUTOFRANGE,"Must pass in at least one matrix");
  PetscValidPointer(mat,3);

  ierr = MatGetLocalSize(mats[0],&m,&n);CHKERRQ(ierr);
  ierr = MatGetSize(mats[0],&M,&N);CHKERRQ(ierr);
  ierr = MatCreate(comm,mat);CHKERRQ(ierr);
  ierr = MatSetSizes(*mat,m,n,M,N);CHKERRQ(ierr);
  ierr = MatSetType(*mat,MATCOMPOSITE);CHKERRQ(ierr);
  for (i=0; i<nmat; i++) {
    ierr = MatCompositeAddMat(*mat,mats[i]);CHKERRQ(ierr);
  }
  ierr = MatAssemblyBegin(*mat,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(*mat,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatCompositeAddMat"
/*@
    MatCompositeAddMat - add another matrix to a composite matrix

   Collective on Mat

    Input Parameters:
+   mat - the composite matrix
-   smat - the partial matrix

   Level: advanced

.seealso: MatCreateComposite()
@*/
PetscErrorCode PETSCMAT_DLLEXPORT MatCompositeAddMat(Mat mat,Mat smat)
{
  Mat_Composite     *shell;
  PetscErrorCode    ierr;
  Mat_CompositeLink ilink,next;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(mat,MAT_COOKIE,1);
  PetscValidHeaderSpecific(smat,MAT_COOKIE,2);
  ierr        = PetscNewLog(mat,struct _Mat_CompositeLink,&ilink);CHKERRQ(ierr);
  ilink->next = 0;
  ierr        = PetscObjectReference((PetscObject)smat);CHKERRQ(ierr);
  ilink->mat  = smat;

  shell = (Mat_Composite*)mat->data;
  next = shell->head;
  if (!next) {
    shell->head  = ilink;
  } else {
    while (next->next) {
      next = next->next;
    }
    next->next      = ilink;
    ilink->prev     = next;
  }
  shell->tail = ilink;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatCompositeSetType"
/*@C
   MatCompositeSetType - Indicates if the matrix is defined as the sum of a set of matrices or the product

  Collective on MPI_Comm

   Input Parameters:
.  mat - the composite matrix


   Level: advanced

   Notes:
      The MatType of the resulting matrix will be the same as the MatType of the FIRST
    matrix in the composite matrix.

.seealso: MatDestroy(), MatMult(), MatCompositeAddMat(), MatCreateComposite(), MATCOMPOSITE

@*/
PetscErrorCode PETSCMAT_DLLEXPORT MatCompositeSetType(Mat mat,MatCompositeType type)
{
  Mat_Composite  *b = (Mat_Composite*)mat->data;  
  PetscTruth     flg;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscTypeCompare((PetscObject)mat,MATCOMPOSITE,&flg);CHKERRQ(ierr);
  if (!flg) SETERRQ(PETSC_ERR_ARG_WRONGSTATE,"Can only use with composite matrix");
  if (type == MAT_COMPOSITE_MULTIPLICATIVE) {
    mat->ops->getdiagonal   = 0;
    mat->ops->mult          = MatMult_Composite_Multiplicative;
    mat->ops->multtranspose = MatMultTranspose_Composite_Multiplicative;
    b->type                 = MAT_COMPOSITE_MULTIPLICATIVE;
  } else {
    mat->ops->getdiagonal   = MatGetDiagonal_Composite;
    mat->ops->mult          = MatMult_Composite;
    mat->ops->multtranspose = MatMultTranspose_Composite;
    b->type                 = MAT_COMPOSITE_ADDITIVE;
  }
  PetscFunctionReturn(0);
}


#undef __FUNCT__  
#define __FUNCT__ "MatCompositeMerge"
/*@C
   MatCompositeMerge - Given a composite matrix, replaces it with a "regular" matrix
     by summing all the matrices inside the composite matrix.

  Collective on MPI_Comm

   Input Parameters:
.  mat - the composite matrix


   Options Database:
.  -mat_composite_merge  (you must call MatAssemblyBegin()/MatAssemblyEnd() to have this checked)

   Level: advanced

   Notes:
      The MatType of the resulting matrix will be the same as the MatType of the FIRST
    matrix in the composite matrix.

.seealso: MatDestroy(), MatMult(), MatCompositeAddMat(), MatCreateComposite(), MATCOMPOSITE

@*/
PetscErrorCode PETSCMAT_DLLEXPORT MatCompositeMerge(Mat mat)
{
  Mat_Composite     *shell = (Mat_Composite*)mat->data;  
  Mat_CompositeLink next = shell->head, prev = shell->tail;
  PetscErrorCode    ierr;
  Mat               tmat,newmat;

  PetscFunctionBegin;
  if (!next) SETERRQ(PETSC_ERR_ARG_WRONGSTATE,"Must provide at least one matrix with MatCompositeAddMat()");

  PetscFunctionBegin;
  if (shell->type == MAT_COMPOSITE_ADDITIVE) {
    ierr = MatDuplicate(next->mat,MAT_COPY_VALUES,&tmat);CHKERRQ(ierr);
    while ((next = next->next)) {
      ierr = MatAXPY(tmat,1.0,next->mat,DIFFERENT_NONZERO_PATTERN);CHKERRQ(ierr);
    }
  } else {
    ierr = MatDuplicate(next->mat,MAT_COPY_VALUES,&tmat);CHKERRQ(ierr);
    while ((prev = prev->prev)) {
      ierr = MatMatMult(tmat,prev->mat,MAT_INITIAL_MATRIX,PETSC_DECIDE,&newmat);CHKERRQ(ierr);
      ierr = MatDestroy(tmat);CHKERRQ(ierr);
      tmat = newmat;
    }
  }
  ierr = MatHeaderReplace(mat,tmat);CHKERRQ(ierr);
  ierr = MatDiagonalScale(mat,shell->left,shell->right);CHKERRQ(ierr);
  ierr = MatScale(mat,shell->scale);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
