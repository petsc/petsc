
#include <petsc/private/matimpl.h>        /*I "petscmat.h" I*/

const char *const MatCompositeMergeTypes[] = {"left","right","MatCompositeMergeType","MAT_COMPOSITE_",0};

typedef struct _Mat_CompositeLink *Mat_CompositeLink;
struct _Mat_CompositeLink {
  Mat               mat;
  Vec               work;
  Mat_CompositeLink next,prev;
};

typedef struct {
  MatCompositeType      type;
  Mat_CompositeLink     head,tail;
  Vec                   work;
  PetscScalar           scale;        /* scale factor supplied with MatScale() */
  Vec                   left,right;   /* left and right diagonal scaling provided with MatDiagonalScale() */
  Vec                   leftwork,rightwork;
  PetscInt              nmat;
  PetscBool             merge;
  MatCompositeMergeType mergetype;
  MatStructure          structure;
} Mat_Composite;

PetscErrorCode MatDestroy_Composite(Mat mat)
{
  PetscErrorCode    ierr;
  Mat_Composite     *shell = (Mat_Composite*)mat->data;
  Mat_CompositeLink next   = shell->head,oldnext;

  PetscFunctionBegin;
  while (next) {
    ierr = MatDestroy(&next->mat);CHKERRQ(ierr);
    if (next->work && (!next->next || next->work != next->next->work)) {
      ierr = VecDestroy(&next->work);CHKERRQ(ierr);
    }
    oldnext = next;
    next    = next->next;
    ierr    = PetscFree(oldnext);CHKERRQ(ierr);
  }
  ierr = VecDestroy(&shell->work);CHKERRQ(ierr);
  ierr = VecDestroy(&shell->left);CHKERRQ(ierr);
  ierr = VecDestroy(&shell->right);CHKERRQ(ierr);
  ierr = VecDestroy(&shell->leftwork);CHKERRQ(ierr);
  ierr = VecDestroy(&shell->rightwork);CHKERRQ(ierr);
  ierr = PetscFree(mat->data);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode MatMult_Composite_Multiplicative(Mat A,Vec x,Vec y)
{
  Mat_Composite     *shell = (Mat_Composite*)A->data;
  Mat_CompositeLink next   = shell->head;
  PetscErrorCode    ierr;
  Vec               in,out;

  PetscFunctionBegin;
  if (!next) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"Must provide at least one matrix with MatCompositeAddMat()");
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
      ierr = MatCreateVecs(next->mat,NULL,&next->work);CHKERRQ(ierr);
    }
    out  = next->work;
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

PetscErrorCode MatMultTranspose_Composite_Multiplicative(Mat A,Vec x,Vec y)
{
  Mat_Composite     *shell = (Mat_Composite*)A->data;
  Mat_CompositeLink tail   = shell->tail;
  PetscErrorCode    ierr;
  Vec               in,out;

  PetscFunctionBegin;
  if (!tail) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"Must provide at least one matrix with MatCompositeAddMat()");
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
      ierr = MatCreateVecs(tail->mat,NULL,&tail->prev->work);CHKERRQ(ierr);
    }
    out  = tail->prev->work;
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

PetscErrorCode MatMult_Composite(Mat A,Vec x,Vec y)
{
  Mat_Composite     *shell = (Mat_Composite*)A->data;
  Mat_CompositeLink next   = shell->head;
  PetscErrorCode    ierr;
  Vec               in;

  PetscFunctionBegin;
  if (!next) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"Must provide at least one matrix with MatCompositeAddMat()");
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

PetscErrorCode MatMultTranspose_Composite(Mat A,Vec x,Vec y)
{
  Mat_Composite     *shell = (Mat_Composite*)A->data;
  Mat_CompositeLink next   = shell->head;
  PetscErrorCode    ierr;
  Vec               in;

  PetscFunctionBegin;
  if (!next) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"Must provide at least one matrix with MatCompositeAddMat()");
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

PetscErrorCode MatMultAdd_Composite(Mat A,Vec x,Vec y,Vec z)
{
  Mat_Composite     *shell = (Mat_Composite*)A->data;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  if (y != z) {
    ierr = MatMult(A,x,z);CHKERRQ(ierr);
    ierr = VecAXPY(z,1.0,y);CHKERRQ(ierr);
  } else {
    if (!shell->leftwork) {
      ierr = VecDuplicate(z,&shell->leftwork);CHKERRQ(ierr);
    }
    ierr = MatMult(A,x,shell->leftwork);CHKERRQ(ierr);
    ierr = VecCopy(y,z);CHKERRQ(ierr);
    ierr = VecAXPY(z,1.0,shell->leftwork);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

PetscErrorCode MatMultTransposeAdd_Composite(Mat A,Vec x,Vec y, Vec z)
{
  Mat_Composite     *shell = (Mat_Composite*)A->data;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  if (y != z) {
    ierr = MatMultTranspose(A,x,z);CHKERRQ(ierr);
    ierr = VecAXPY(z,1.0,y);CHKERRQ(ierr);
  } else {
    if (!shell->rightwork) {
      ierr = VecDuplicate(z,&shell->rightwork);CHKERRQ(ierr);
    }
    ierr = MatMultTranspose(A,x,shell->rightwork);CHKERRQ(ierr);
    ierr = VecCopy(y,z);CHKERRQ(ierr);
    ierr = VecAXPY(z,1.0,shell->rightwork);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

PetscErrorCode MatGetDiagonal_Composite(Mat A,Vec v)
{
  Mat_Composite     *shell = (Mat_Composite*)A->data;
  Mat_CompositeLink next   = shell->head;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  if (!next) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"Must provide at least one matrix with MatCompositeAddMat()");
  if (shell->right || shell->left) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"Cannot get diagonal if left or right scaling");

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

PetscErrorCode MatAssemblyEnd_Composite(Mat Y,MatAssemblyType t)
{
  Mat_Composite     *shell = (Mat_Composite*)Y->data;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  if (shell->merge) {
    ierr = MatCompositeMerge(Y);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

PetscErrorCode MatScale_Composite(Mat inA,PetscScalar alpha)
{
  Mat_Composite *a = (Mat_Composite*)inA->data;

  PetscFunctionBegin;
  a->scale *= alpha;
  PetscFunctionReturn(0);
}

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

PetscErrorCode MatSetFromOptions_Composite(PetscOptionItems *PetscOptionsObject,Mat A)
{
  Mat_Composite  *a = (Mat_Composite*)A->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscOptionsHead(PetscOptionsObject,"MATCOMPOSITE options");CHKERRQ(ierr);
  ierr = PetscOptionsBool("-mat_composite_merge","Merge at MatAssemblyEnd","MatCompositeMerge",a->merge,&a->merge,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsEnum("-mat_composite_merge_type","Set composite merge direction","MatCompositeSetMergeType",MatCompositeMergeTypes,(PetscEnum)a->mergetype,(PetscEnum*)&a->mergetype,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsTail();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@
   MatCreateComposite - Creates a matrix as the sum or product of one or more matrices

  Collective

   Input Parameters:
+  comm - MPI communicator
.  nmat - number of matrices to put in
-  mats - the matrices

   Output Parameter:
.  mat - the matrix

   Options Database Keys:
+  -mat_composite_merge - merge in MatAssemblyEnd()
-  -mat_composite_merge_type - set merge direction

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

.seealso: MatDestroy(), MatMult(), MatCompositeAddMat(), MatCompositeGetMat(), MatCompositeMerge(), MatCompositeSetType(), MATCOMPOSITE

@*/
PetscErrorCode MatCreateComposite(MPI_Comm comm,PetscInt nmat,const Mat *mats,Mat *mat)
{
  PetscErrorCode ierr;
  PetscInt       m,n,M,N,i;

  PetscFunctionBegin;
  if (nmat < 1) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Must pass in at least one matrix");
  PetscValidPointer(mat,3);

  ierr = MatGetLocalSize(mats[0],PETSC_IGNORE,&n);CHKERRQ(ierr);
  ierr = MatGetLocalSize(mats[nmat-1],&m,PETSC_IGNORE);CHKERRQ(ierr);
  ierr = MatGetSize(mats[0],PETSC_IGNORE,&N);CHKERRQ(ierr);
  ierr = MatGetSize(mats[nmat-1],&M,PETSC_IGNORE);CHKERRQ(ierr);
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


static PetscErrorCode MatCompositeAddMat_Composite(Mat mat,Mat smat)
{
  Mat_Composite     *shell = (Mat_Composite*)mat->data;
  Mat_CompositeLink ilink,next = shell->head;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  ierr        = PetscNewLog(mat,&ilink);CHKERRQ(ierr);
  ilink->next = 0;
  ierr        = PetscObjectReference((PetscObject)smat);CHKERRQ(ierr);
  ilink->mat  = smat;

  if (!next) shell->head = ilink;
  else {
    while (next->next) {
      next = next->next;
    }
    next->next  = ilink;
    ilink->prev = next;
  }
  shell->tail =  ilink;
  shell->nmat += 1;
  PetscFunctionReturn(0);
}

/*@
    MatCompositeAddMat - Add another matrix to a composite matrix.

   Collective on Mat

    Input Parameters:
+   mat - the composite matrix
-   smat - the partial matrix

   Level: advanced

.seealso: MatCreateComposite(), MatCompositeGetMat(), MATCOMPOSITE
@*/
PetscErrorCode MatCompositeAddMat(Mat mat,Mat smat)
{
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(mat,MAT_CLASSID,1);
  PetscValidHeaderSpecific(smat,MAT_CLASSID,2);
  ierr = PetscUseMethod(mat,"MatCompositeAddMat_C",(Mat,Mat),(mat,smat));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode MatCompositeSetType_Composite(Mat mat,MatCompositeType type)
{
  Mat_Composite  *b = (Mat_Composite*)mat->data;

  PetscFunctionBegin;
  b->type = type;
  if (type == MAT_COMPOSITE_MULTIPLICATIVE) {
    mat->ops->getdiagonal   = 0;
    mat->ops->mult          = MatMult_Composite_Multiplicative;
    mat->ops->multtranspose = MatMultTranspose_Composite_Multiplicative;
  } else {
    mat->ops->getdiagonal   = MatGetDiagonal_Composite;
    mat->ops->mult          = MatMult_Composite;
    mat->ops->multtranspose = MatMultTranspose_Composite;
  }
  PetscFunctionReturn(0);
}

/*@
   MatCompositeSetType - Indicates if the matrix is defined as the sum of a set of matrices or the product.

   Logically Collective on Mat

   Input Parameters:
.  mat - the composite matrix

   Level: advanced

.seealso: MatDestroy(), MatMult(), MatCompositeAddMat(), MatCreateComposite(), MatCompositeGetType(), MATCOMPOSITE

@*/
PetscErrorCode MatCompositeSetType(Mat mat,MatCompositeType type)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(mat,MAT_CLASSID,1);
  PetscValidLogicalCollectiveEnum(mat,type,2);
  ierr = PetscUseMethod(mat,"MatCompositeSetType_C",(Mat,MatCompositeType),(mat,type));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode MatCompositeGetType_Composite(Mat mat,MatCompositeType *type)
{
  Mat_Composite  *b = (Mat_Composite*)mat->data;

  PetscFunctionBegin;
  *type = b->type;
  PetscFunctionReturn(0);
}

/*@
   MatCompositeGetType - Returns type of composite.

   Not Collective

   Input Parameter:
.  mat - the composite matrix

   Output Parameter:
.  type - type of composite

   Level: advanced

.seealso: MatCreateComposite(), MatCompositeSetType(), MATCOMPOSITE

@*/
PetscErrorCode MatCompositeGetType(Mat mat,MatCompositeType *type)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(mat,MAT_CLASSID,1);
  PetscValidPointer(type,2);
  ierr = PetscUseMethod(mat,"MatCompositeGetType_C",(Mat,MatCompositeType*),(mat,type));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode MatCompositeSetMatStructure_Composite(Mat mat,MatStructure str)
{
  Mat_Composite  *b = (Mat_Composite*)mat->data;

  PetscFunctionBegin;
  b->structure = str;
  PetscFunctionReturn(0);
}

/*@
   MatCompositeSetMatStructure - Indicates structure of matrices in the composite matrix.

   Not Collective

   Input Parameters:
+  mat - the composite matrix
-  str - either SAME_NONZERO_PATTERN, DIFFERENT_NONZERO_PATTERN (default) or SUBSET_NONZERO_PATTERN

   Level: advanced

   Notes:
    Information about the matrices structure is used in MatCompositeMerge() for additive composite matrix.

.seealso: MatAXPY(), MatCreateComposite(), MatCompositeMerge() MatCompositeGetMatStructure(), MATCOMPOSITE

@*/
PetscErrorCode MatCompositeSetMatStructure(Mat mat,MatStructure str)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(mat,MAT_CLASSID,1);
  ierr = PetscUseMethod(mat,"MatCompositeSetMatStructure_C",(Mat,MatStructure),(mat,str));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode MatCompositeGetMatStructure_Composite(Mat mat,MatStructure *str)
{
  Mat_Composite  *b = (Mat_Composite*)mat->data;

  PetscFunctionBegin;
  *str = b->structure;
  PetscFunctionReturn(0);
}

/*@
   MatCompositeGetMatStructure - Returns the structure of matrices in the composite matrix.

   Not Collective

   Input Parameter:
.  mat - the composite matrix

   Output Parameter:
.  str - structure of the matrices

   Level: advanced

.seealso: MatCreateComposite(), MatCompositeSetMatStructure(), MATCOMPOSITE

@*/
PetscErrorCode MatCompositeGetMatStructure(Mat mat,MatStructure *str)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(mat,MAT_CLASSID,1);
  PetscValidPointer(str,2);
  ierr = PetscUseMethod(mat,"MatCompositeGetMatStructure_C",(Mat,MatStructure*),(mat,str));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode MatCompositeSetMergeType_Composite(Mat mat,MatCompositeMergeType type)
{
  Mat_Composite     *shell = (Mat_Composite*)mat->data;

  PetscFunctionBegin;
  shell->mergetype = type;
  PetscFunctionReturn(0);
}

/*@
   MatCompositeSetMergeType - Sets order of MatCompositeMerge().

   Logically Collective on Mat

   Input Parameter:
+  mat - the composite matrix
-  type - MAT_COMPOSITE_MERGE RIGHT (default) to start merge from right with the first added matrix (mat[0]),
          MAT_COMPOSITE_MERGE_LEFT to start merge from left with the last added matrix (mat[nmat-1])

   Level: advanced

   Notes:
    The resulting matrix is the same regardles of the MergeType. Only the order of operation is changed.
    If set to MAT_COMPOSITE_MERGE_RIGHT the order of the merge is mat[nmat-1]*(mat[nmat-2]*(...*(mat[1]*mat[0])))
    otherwise the order is (((mat[nmat-1]*mat[nmat-2])*mat[nmat-3])*...)*mat[0].

.seealso: MatCreateComposite(), MatCompositeMerge(), MATCOMPOSITE

@*/
PetscErrorCode MatCompositeSetMergeType(Mat mat,MatCompositeMergeType type)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(mat,MAT_CLASSID,1);
  PetscValidLogicalCollectiveEnum(mat,type,2);
  ierr = PetscUseMethod(mat,"MatCompositeSetMergeType_C",(Mat,MatCompositeMergeType),(mat,type));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode MatCompositeMerge_Composite(Mat mat)
{
  Mat_Composite     *shell = (Mat_Composite*)mat->data;
  Mat_CompositeLink next   = shell->head, prev = shell->tail;
  PetscErrorCode    ierr;
  Mat               tmat,newmat;
  Vec               left,right;
  PetscScalar       scale;

  PetscFunctionBegin;
  if (!next) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"Must provide at least one matrix with MatCompositeAddMat()");

  PetscFunctionBegin;
  if (shell->type == MAT_COMPOSITE_ADDITIVE) {
    if (shell->mergetype == MAT_COMPOSITE_MERGE_RIGHT) {
      ierr = MatDuplicate(next->mat,MAT_COPY_VALUES,&tmat);CHKERRQ(ierr);
      while ((next = next->next)) {
        ierr = MatAXPY(tmat,1.0,next->mat,shell->structure);CHKERRQ(ierr);
      }
    } else {
      ierr = MatDuplicate(prev->mat,MAT_COPY_VALUES,&tmat);CHKERRQ(ierr);
      while ((prev = prev->prev)) {
        ierr = MatAXPY(tmat,1.0,prev->mat,shell->structure);CHKERRQ(ierr);
      }
    }
  } else {
    if (shell->mergetype == MAT_COMPOSITE_MERGE_RIGHT) {
      ierr = MatDuplicate(next->mat,MAT_COPY_VALUES,&tmat);CHKERRQ(ierr);
      while ((next = next->next)) {
        ierr = MatMatMult(next->mat,tmat,MAT_INITIAL_MATRIX,PETSC_DECIDE,&newmat);CHKERRQ(ierr);
        ierr = MatDestroy(&tmat);CHKERRQ(ierr);
        tmat = newmat;
      }
    } else {
      ierr = MatDuplicate(prev->mat,MAT_COPY_VALUES,&tmat);CHKERRQ(ierr);
      while ((prev = prev->prev)) {
        ierr = MatMatMult(tmat,prev->mat,MAT_INITIAL_MATRIX,PETSC_DECIDE,&newmat);CHKERRQ(ierr);
        ierr = MatDestroy(&tmat);CHKERRQ(ierr);
        tmat = newmat;
      }
    }
  }

  scale = shell->scale;
  if ((left = shell->left)) {ierr = PetscObjectReference((PetscObject)left);CHKERRQ(ierr);}
  if ((right = shell->right)) {ierr = PetscObjectReference((PetscObject)right);CHKERRQ(ierr);}

  ierr = MatHeaderReplace(mat,&tmat);CHKERRQ(ierr);

  ierr = MatDiagonalScale(mat,left,right);CHKERRQ(ierr);
  ierr = MatScale(mat,scale);CHKERRQ(ierr);
  ierr = VecDestroy(&left);CHKERRQ(ierr);
  ierr = VecDestroy(&right);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@
   MatCompositeMerge - Given a composite matrix, replaces it with a "regular" matrix
     by summing or computing the product of all the matrices inside the composite matrix.

  Collective

   Input Parameters:
.  mat - the composite matrix


   Options Database Keys:
+  -mat_composite_merge - merge in MatAssemblyEnd()
-  -mat_composite_merge_type - set merge direction

   Level: advanced

   Notes:
      The MatType of the resulting matrix will be the same as the MatType of the FIRST
    matrix in the composite matrix.

.seealso: MatDestroy(), MatMult(), MatCompositeAddMat(), MatCreateComposite(), MatCompositeSetMatStructure(), MatCompositeSetMergeType(), MATCOMPOSITE

@*/
PetscErrorCode MatCompositeMerge(Mat mat)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(mat,MAT_CLASSID,1);
  ierr = PetscUseMethod(mat,"MatCompositeMerge_C",(Mat),(mat));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode MatCompositeGetNumberMat_Composite(Mat mat,PetscInt *nmat)
{
  Mat_Composite     *shell = (Mat_Composite*)mat->data;

  PetscFunctionBegin;
  *nmat = shell->nmat;
  PetscFunctionReturn(0);
}

/*@
   MatCompositeGetNumberMat - Returns the number of matrices in the composite matrix.

   Not Collective

   Input Parameter:
.  mat - the composite matrix

   Output Parameter:
.  nmat - number of matrices in the composite matrix

   Level: advanced

.seealso: MatCreateComposite(), MatCompositeGetMat(), MATCOMPOSITE

@*/
PetscErrorCode MatCompositeGetNumberMat(Mat mat,PetscInt *nmat)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(mat,MAT_CLASSID,1);
  PetscValidPointer(nmat,2);
  ierr = PetscUseMethod(mat,"MatCompositeGetNumberMat_C",(Mat,PetscInt*),(mat,nmat));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode MatCompositeGetMat_Composite(Mat mat,PetscInt i,Mat *Ai)
{
  Mat_Composite     *shell = (Mat_Composite*)mat->data;
  Mat_CompositeLink ilink;
  PetscInt          k;

  PetscFunctionBegin;
  if (i >= shell->nmat) SETERRQ2(PetscObjectComm((PetscObject)mat),PETSC_ERR_ARG_OUTOFRANGE,"index out of range: %d >= %d",i,shell->nmat);
  ilink = shell->head;
  for (k=0; k<i; k++) {
    ilink = ilink->next;
  }
  *Ai = ilink->mat;
  PetscFunctionReturn(0);
}

/*@
   MatCompositeGetMat - Returns the ith matrix from the composite matrix.

   Logically Collective on Mat

   Input Parameter:
+  mat - the composite matrix
-  i - the number of requested matrix

   Output Parameter:
.  Ai - ith matrix in composite

   Level: advanced

.seealso: MatCreateComposite(), MatCompositeGetNumberMat(), MatCompositeAddMat(), MATCOMPOSITE

@*/
PetscErrorCode MatCompositeGetMat(Mat mat,PetscInt i,Mat *Ai)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(mat,MAT_CLASSID,1);
  PetscValidLogicalCollectiveInt(mat,i,2);
  PetscValidPointer(Ai,3);
  ierr = PetscUseMethod(mat,"MatCompositeGetMat_C",(Mat,PetscInt,Mat*),(mat,i,Ai));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static struct _MatOps MatOps_Values = {0,
                                       0,
                                       0,
                                       MatMult_Composite,
                                       MatMultAdd_Composite,
                                /*  5*/ MatMultTranspose_Composite,
                                       MatMultTransposeAdd_Composite,
                                       0,
                                       0,
                                       0,
                                /* 10*/ 0,
                                       0,
                                       0,
                                       0,
                                       0,
                                /* 15*/ 0,
                                       0,
                                       MatGetDiagonal_Composite,
                                       MatDiagonalScale_Composite,
                                       0,
                                /* 20*/ 0,
                                       MatAssemblyEnd_Composite,
                                       0,
                                       0,
                               /* 24*/ 0,
                                       0,
                                       0,
                                       0,
                                       0,
                               /* 29*/ 0,
                                       0,
                                       0,
                                       0,
                                       0,
                               /* 34*/ 0,
                                       0,
                                       0,
                                       0,
                                       0,
                               /* 39*/ 0,
                                       0,
                                       0,
                                       0,
                                       0,
                               /* 44*/ 0,
                                       MatScale_Composite,
                                       MatShift_Basic,
                                       0,
                                       0,
                               /* 49*/ 0,
                                       0,
                                       0,
                                       0,
                                       0,
                               /* 54*/ 0,
                                       0,
                                       0,
                                       0,
                                       0,
                               /* 59*/ 0,
                                       MatDestroy_Composite,
                                       0,
                                       0,
                                       0,
                               /* 64*/ 0,
                                       0,
                                       0,
                                       0,
                                       0,
                               /* 69*/ 0,
                                       0,
                                       0,
                                       0,
                                       0,
                               /* 74*/ 0,
                                       0,
                                       MatSetFromOptions_Composite,
                                       0,
                                       0,
                               /* 79*/ 0,
                                       0,
                                       0,
                                       0,
                                       0,
                               /* 84*/ 0,
                                       0,
                                       0,
                                       0,
                                       0,
                               /* 89*/ 0,
                                       0,
                                       0,
                                       0,
                                       0,
                               /* 94*/ 0,
                                       0,
                                       0,
                                       0,
                                       0,
                                /*99*/ 0,
                                       0,
                                       0,
                                       0,
                                       0,
                               /*104*/ 0,
                                       0,
                                       0,
                                       0,
                                       0,
                               /*109*/ 0,
                                       0,
                                       0,
                                       0,
                                       0,
                               /*114*/ 0,
                                       0,
                                       0,
                                       0,
                                       0,
                               /*119*/ 0,
                                       0,
                                       0,
                                       0,
                                       0,
                               /*124*/ 0,
                                       0,
                                       0,
                                       0,
                                       0,
                               /*129*/ 0,
                                       0,
                                       0,
                                       0,
                                       0,
                               /*134*/ 0,
                                       0,
                                       0,
                                       0,
                                       0,
                               /*139*/ 0,
                                       0,
                                       0
};

/*MC
   MATCOMPOSITE - A matrix defined by the sum (or product) of one or more matrices.
    The matrices need to have a correct size and parallel layout for the sum or product to be valid.

   Notes:
    to use the product of the matrices call MatCompositeSetType(mat,MAT_COMPOSITE_MULTIPLICATIVE);

  Level: advanced

.seealso: MatCreateComposite(), MatCompositeAddMat(), MatSetType(), MatCompositeSetType(), MatCompositeGetType(), MatCompositeSetMatStructure(), MatCompositeGetMatStructure(), MatCompositeMerge(), MatCompositeSetMergeType(), MatCompositeGetNumberMat(), MatCompositeGetMat()
M*/

PETSC_EXTERN PetscErrorCode MatCreate_Composite(Mat A)
{
  Mat_Composite  *b;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr    = PetscNewLog(A,&b);CHKERRQ(ierr);
  A->data = (void*)b;
  ierr    = PetscMemcpy(A->ops,&MatOps_Values,sizeof(struct _MatOps));CHKERRQ(ierr);

  ierr = PetscLayoutSetUp(A->rmap);CHKERRQ(ierr);
  ierr = PetscLayoutSetUp(A->cmap);CHKERRQ(ierr);

  A->assembled    = PETSC_TRUE;
  A->preallocated = PETSC_TRUE;
  b->type         = MAT_COMPOSITE_ADDITIVE;
  b->scale        = 1.0;
  b->nmat         = 0;
  b->merge        = PETSC_FALSE;
  b->mergetype    = MAT_COMPOSITE_MERGE_RIGHT;
  b->structure    = DIFFERENT_NONZERO_PATTERN;

  ierr = PetscObjectComposeFunction((PetscObject)A,"MatCompositeAddMat_C",MatCompositeAddMat_Composite);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)A,"MatCompositeSetType_C",MatCompositeSetType_Composite);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)A,"MatCompositeGetType_C",MatCompositeGetType_Composite);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)A,"MatCompositeSetMergeType_C",MatCompositeSetMergeType_Composite);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)A,"MatCompositeSetMatStructure_C",MatCompositeSetMatStructure_Composite);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)A,"MatCompositeGetMatStructure_C",MatCompositeGetMatStructure_Composite);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)A,"MatCompositeMerge_C",MatCompositeMerge_Composite);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)A,"MatCompositeGetNumberMat_C",MatCompositeGetNumberMat_Composite);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)A,"MatCompositeGetMat_C",MatCompositeGetMat_Composite);CHKERRQ(ierr);

  ierr = PetscObjectChangeTypeName((PetscObject)A,MATCOMPOSITE);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

