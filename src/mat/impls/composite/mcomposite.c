#define PETSCMAT_DLL

#include "src/mat/matimpl.h"        /*I "petscmat.h" I*/

typedef struct _Mat_CompositeLink *Mat_CompositeLink;
struct _Mat_CompositeLink {
  Mat               mat;
  Mat_CompositeLink next;
};
  
typedef struct {
  Mat_CompositeLink head;
  Vec               work;
} Mat_Composite;

#undef __FUNCT__  
#define __FUNCT__ "MatDestroy_Composite"
PetscErrorCode MatDestroy_Composite(Mat mat)
{
  PetscErrorCode   ierr;
  Mat_Composite    *shell = (Mat_Composite*)mat->data;;
  Mat_CompositeLink next = shell->head;

  PetscFunctionBegin;
  while (next) {
    ierr = MatDestroy(next->mat);CHKERRQ(ierr);
    next = next->next;
  }
  if (shell->work) {ierr = VecDestroy(shell->work);CHKERRQ(ierr);}
  ierr      = PetscFree(shell);CHKERRQ(ierr);
  mat->data = 0;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatMult_Composite"
PetscErrorCode MatMult_Composite(Mat A,Vec x,Vec y)
{
  Mat_Composite     *shell = (Mat_Composite*)A->data;  
  Mat_CompositeLink next = shell->head;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  if (!next) SETERRQ(PETSC_ERR_ARG_WRONGSTATE,"Must provide at least one matrix with MatCompositeAddMat()");
  ierr = MatMult(next->mat,x,y);CHKERRQ(ierr);
  while ((next = next->next)) {
    ierr = MatMultAdd(next->mat,x,y,y);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatMultTranspose_Composite"
PetscErrorCode MatMultTranspose_Composite(Mat A,Vec x,Vec y)
{
  Mat_Composite     *shell = (Mat_Composite*)A->data;  
  Mat_CompositeLink next = shell->head;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  if (!next) SETERRQ(PETSC_ERR_ARG_WRONGSTATE,"Must provide at least one matrix with MatCompositeAddMat()");
  ierr = MatMultTranspose(next->mat,x,y);CHKERRQ(ierr);
  while ((next = next->next)) {
    ierr = MatMultTransposeAdd(next->mat,x,y,y);CHKERRQ(ierr);
  }
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
  ierr = MatGetDiagonal(next->mat,v);CHKERRQ(ierr);
  if (next->next && !shell->work) {
    ierr = VecDuplicate(v,&shell->work);CHKERRQ(ierr);
  }
  while ((next = next->next)) {
    ierr = MatGetDiagonal(next->mat,shell->work);CHKERRQ(ierr);
    ierr = VecAXPY(v,1.0,shell->work);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatAssemblyEnd_Composite"
PetscErrorCode MatAssemblyEnd_Composite(Mat Y,MatAssemblyType t)
{
  PetscFunctionBegin;
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
       0,
       0,
/*20*/ 0,
       MatAssemblyEnd_Composite,
       0,
       0,
       0,
/*25*/ 0,
       0,
       0,
       0,
       0,
/*30*/ 0,
       0,
       0,
       0,
       0,
/*35*/ 0,
       0,
       0,
       0,
       0,
/*40*/ 0,
       0,
       0,
       0,
       0,
/*45*/ 0,
       0,
       0,
       0,
       0,
/*50*/ 0,
       0,
       0,
       0,
       0,
/*55*/ 0,
       0,
       0,
       0,
       0,
/*60*/ 0,
       MatDestroy_Composite,
       0,
       0,
       0,
/*65*/ 0,
       0,
       0,
       0,
       0,
/*70*/ 0,
       0,
       0,
       0,
       0,
/*75*/ 0,
       0,
       0,
       0,
       0,
/*80*/ 0,
       0,
       0,
       0,
       0,
/*85*/ 0,
       0,
       0,
       0,
       0,
/*90*/ 0,
       0,
       0,
       0,
       0,
/*95*/ 0,
       0,
       0,
       0};

/*MC
   MATCOMPOSITE - A matrix defined by the sum of one or more matrices (all matrices are of same size and parallel layout)

  Level: advanced

.seealso: MatCreateComposite
M*/

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "MatCreate_Composite"
PetscErrorCode PETSCMAT_DLLEXPORT MatCreate_Composite(Mat A)
{
  Mat_Composite  *b;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscMemcpy(A->ops,&MatOps_Values,sizeof(struct _MatOps));CHKERRQ(ierr);

  ierr = PetscNew(Mat_Composite,&b);CHKERRQ(ierr);
  A->data = (void*)b;

  ierr = PetscMapInitialize(A->comm,&A->rmap);CHKERRQ(ierr);
  ierr = PetscMapInitialize(A->comm,&A->cmap);CHKERRQ(ierr);

  A->assembled     = PETSC_TRUE;
  A->preallocated  = PETSC_FALSE;
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

.seealso: MatDestroy(), MatMult(), MatCompositeAddMat()

@*/
PetscErrorCode PETSCMAT_DLLEXPORT MatCreateComposite(MPI_Comm comm,PetscInt nmat,const Mat *mats,Mat *mat)
{
  PetscErrorCode ierr;
  PetscInt       m,n,M,N,i;

  PetscFunctionBegin;
  if (nmat < 1) SETERRQ(PETSC_ERR_ARG_OUTOFRANGE,"Must pass in at least one matrix");
  PetscValidPointer(mat,3);CHKERRQ(ierr);

  ierr = MatGetLocalSize(mats[0],&m,&n);CHKERRQ(ierr);
  ierr = MatGetSize(mats[0],&M,&N);CHKERRQ(ierr);
  ierr = MatCreate(comm,mat);CHKERRQ(ierr);
  ierr = MatSetSizes(*mat,m,n,M,N);CHKERRQ(ierr);
  ierr = MatSetType(*mat,MATCOMPOSITE);CHKERRQ(ierr);
  for (i=0; i<nmat; i++) {
    ierr = MatCompositeAddMat(*mat,mats[i]);CHKERRQ(ierr);
  }
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
  Mat_Composite     *shell = (Mat_Composite*)mat->data;
  PetscErrorCode    ierr;
  Mat_CompositeLink ilink,next;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(mat,MAT_COOKIE,1);
  PetscValidHeaderSpecific(smat,MAT_COOKIE,2);
  ierr       = PetscNew(struct _Mat_CompositeLink,&ilink);CHKERRQ(ierr);
  ilink->next = 0;
  ilink->mat  = smat;
  ierr        = PetscObjectReference((PetscObject)smat);CHKERRQ(ierr);

  next = shell->head;
  if (!next) {
    shell->head  = ilink;
  } else {
    while (next->next) {
      next = next->next;
    }
    next->next      = ilink;
  }
  PetscFunctionReturn(0);
}

