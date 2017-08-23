
/*
   This provides a simple shell for Fortran (and C programmers) to
  create a very simple matrix class for use with KSP without coding
  much of anything.
*/

#include <petsc/private/matimpl.h>        /*I "petscmat.h" I*/

struct _MatShellOps {
  /*   0 */
  PetscErrorCode (*mult)(Mat,Vec,Vec);
  /*   5 */
  PetscErrorCode (*multtranspose)(Mat,Vec,Vec);
  /*  10 */
  /*  15 */
  PetscErrorCode (*getdiagonal)(Mat,Vec);
  /*  20 */
  /*  24 */
  /*  29 */
  /*  34 */
  /*  39 */
  PetscErrorCode (*copy)(Mat,Mat,MatStructure);
  /*  44 */
  PetscErrorCode (*diagonalset)(Mat,Vec,InsertMode);
  /*  49 */
  /*  54 */
  /*  59 */
  PetscErrorCode (*destroy)(Mat);
  /*  64 */
  /*  69 */
  /*  74 */
  /*  79 */
  /*  84 */
  /*  89 */
  /*  94 */
  /*  99 */
  /* 104 */
  /* 109 */
  /* 114 */
  /* 119 */
  /* 124 */
  /* 129 */
  /* 134 */
  /* 139 */
  /* 144 */
};

typedef struct {
  struct _MatShellOps ops[1];

  PetscScalar vscale,vshift;
  Vec         dshift;
  Vec         left,right;
  Vec         dshift_owned,left_owned,right_owned;
  Vec         left_work,right_work;
  Vec         left_add_work,right_add_work;
  PetscBool   usingscaled;
  void        *ctx;
} Mat_Shell;
/*
 The most general expression for the matrix is

 S = L (a A + B) R

 where
 A is the matrix defined by the user's function
 a is a scalar multiple
 L is left scaling
 R is right scaling
 B is a diagonal shift defined by
   diag(dshift) if the vector dshift is non-NULL
   vscale*identity otherwise

 The following identities apply:

 Scale by c:
  c [L (a A + B) R] = L [(a c) A + c B] R

 Shift by c:
  [L (a A + B) R] + c = L [a A + (B + c Linv Rinv)] R

 Diagonal scaling is achieved by simply multiplying with existing L and R vectors

 In the data structure:

 vscale=1.0  means no special scaling will be applied
 dshift=NULL means a constant diagonal shift (fall back to vshift)
 vshift=0.0  means no constant diagonal shift, note that vshift is only used if dshift is NULL
*/

static PetscErrorCode MatMult_Shell(Mat,Vec,Vec);
static PetscErrorCode MatMultTranspose_Shell(Mat,Vec,Vec);
static PetscErrorCode MatGetDiagonal_Shell(Mat,Vec);
static PetscErrorCode MatCopy_Shell(Mat,Mat,MatStructure);

static PetscErrorCode MatShellUseScaledMethods(Mat Y)
{
  Mat_Shell *shell = (Mat_Shell*)Y->data;

  PetscFunctionBegin;
  if (shell->usingscaled) PetscFunctionReturn(0);
  shell->ops->mult  = Y->ops->mult;
  Y->ops->mult = MatMult_Shell;
  if (Y->ops->multtranspose) {
    shell->ops->multtranspose  = Y->ops->multtranspose;
    Y->ops->multtranspose = MatMultTranspose_Shell;
  }
  if (Y->ops->getdiagonal) {
    shell->ops->getdiagonal  = Y->ops->getdiagonal;
    Y->ops->getdiagonal = MatGetDiagonal_Shell;
  }
  if (Y->ops->copy) {
    shell->ops->copy  = Y->ops->copy;
    Y->ops->copy = MatCopy_Shell;
  }
  shell->usingscaled = PETSC_TRUE;
  PetscFunctionReturn(0);
}

static PetscErrorCode MatShellPreScaleLeft(Mat A,Vec x,Vec *xx)
{
  Mat_Shell      *shell = (Mat_Shell*)A->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  *xx = NULL;
  if (!shell->left) {
    *xx = x;
  } else {
    if (!shell->left_work) {ierr = VecDuplicate(shell->left,&shell->left_work);CHKERRQ(ierr);}
    ierr = VecPointwiseMult(shell->left_work,x,shell->left);CHKERRQ(ierr);
    *xx  = shell->left_work;
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode MatShellPreScaleRight(Mat A,Vec x,Vec *xx)
{
  Mat_Shell      *shell = (Mat_Shell*)A->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  *xx = NULL;
  if (!shell->right) {
    *xx = x;
  } else {
    if (!shell->right_work) {ierr = VecDuplicate(shell->right,&shell->right_work);CHKERRQ(ierr);}
    ierr = VecPointwiseMult(shell->right_work,x,shell->right);CHKERRQ(ierr);
    *xx  = shell->right_work;
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode MatShellPostScaleLeft(Mat A,Vec x)
{
  Mat_Shell      *shell = (Mat_Shell*)A->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (shell->left) {ierr = VecPointwiseMult(x,x,shell->left);CHKERRQ(ierr);}
  PetscFunctionReturn(0);
}

static PetscErrorCode MatShellPostScaleRight(Mat A,Vec x)
{
  Mat_Shell      *shell = (Mat_Shell*)A->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (shell->right) {ierr = VecPointwiseMult(x,x,shell->right);CHKERRQ(ierr);}
  PetscFunctionReturn(0);
}

static PetscErrorCode MatShellShiftAndScale(Mat A,Vec X,Vec Y)
{
  Mat_Shell      *shell = (Mat_Shell*)A->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (shell->dshift) {          /* get arrays because there is no VecPointwiseMultAdd() */
    PetscInt          i,m;
    const PetscScalar *x,*d;
    PetscScalar       *y;
    ierr = VecGetLocalSize(X,&m);CHKERRQ(ierr);
    ierr = VecGetArrayRead(shell->dshift,&d);CHKERRQ(ierr);
    ierr = VecGetArrayRead(X,&x);CHKERRQ(ierr);
    ierr = VecGetArray(Y,&y);CHKERRQ(ierr);
    for (i=0; i<m; i++) y[i] = shell->vscale*y[i] + d[i]*x[i];
    ierr = VecRestoreArrayRead(shell->dshift,&d);CHKERRQ(ierr);
    ierr = VecRestoreArrayRead(X,&x);CHKERRQ(ierr);
    ierr = VecRestoreArray(Y,&y);CHKERRQ(ierr);
  } else if (PetscAbsScalar(shell->vshift) != 0) {
    ierr = VecAXPBY(Y,shell->vshift,shell->vscale,X);CHKERRQ(ierr);
  } else if (shell->vscale != 1.0) {
    ierr = VecScale(Y,shell->vscale);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

/*@
    MatShellGetContext - Returns the user-provided context associated with a shell matrix.

    Not Collective

    Input Parameter:
.   mat - the matrix, should have been created with MatCreateShell()

    Output Parameter:
.   ctx - the user provided context

    Level: advanced

   Fortran Notes: To use this from Fortran you must write a Fortran interface definition for this
    function that tells Fortran the Fortran derived data type that you are passing in as the ctx argument.

.keywords: matrix, shell, get, context

.seealso: MatCreateShell(), MatShellSetOperation(), MatShellSetContext()
@*/
PetscErrorCode  MatShellGetContext(Mat mat,void *ctx)
{
  PetscErrorCode ierr;
  PetscBool      flg;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(mat,MAT_CLASSID,1);
  PetscValidPointer(ctx,2);
  ierr = PetscObjectTypeCompare((PetscObject)mat,MATSHELL,&flg);CHKERRQ(ierr);
  if (flg) *(void**)ctx = ((Mat_Shell*)(mat->data))->ctx;
  else SETERRQ(PetscObjectComm((PetscObject)mat),PETSC_ERR_SUP,"Cannot get context from non-shell matrix");
  PetscFunctionReturn(0);
}

PetscErrorCode MatDestroy_Shell(Mat mat)
{
  PetscErrorCode ierr;
  Mat_Shell      *shell = (Mat_Shell*)mat->data;

  PetscFunctionBegin;
  if (shell->ops->destroy) {
    ierr = (*shell->ops->destroy)(mat);CHKERRQ(ierr);
  }
  ierr = VecDestroy(&shell->left_owned);CHKERRQ(ierr);
  ierr = VecDestroy(&shell->right_owned);CHKERRQ(ierr);
  ierr = VecDestroy(&shell->dshift_owned);CHKERRQ(ierr);
  ierr = VecDestroy(&shell->left_work);CHKERRQ(ierr);
  ierr = VecDestroy(&shell->right_work);CHKERRQ(ierr);
  ierr = VecDestroy(&shell->left_add_work);CHKERRQ(ierr);
  ierr = VecDestroy(&shell->right_add_work);CHKERRQ(ierr);
  ierr = PetscFree(mat->data);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode MatCopy_Shell(Mat A,Mat B,MatStructure str)
{
  Mat_Shell       *shellA = (Mat_Shell*)A->data,*shellB = (Mat_Shell*)B->data;
  PetscBool       matflg,shellflg;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  ierr = PetscObjectTypeCompare((PetscObject)B,MATSHELL,&matflg);CHKERRQ(ierr);
  if(!matflg) { SETERRQ(PetscObjectComm((PetscObject)A),PETSC_ERR_ARG_NOTSAMETYPE,"Output matrix must be a MATSHELL"); }

  ierr = MatShellUseScaledMethods(B);CHKERRQ(ierr);
  ierr = PetscMemcmp(A->ops,B->ops,sizeof(struct _MatOps),&matflg);CHKERRQ(ierr);
  ierr = PetscMemcmp(shellA->ops,shellB->ops,sizeof(struct _MatShellOps),&shellflg);CHKERRQ(ierr);
  if (!matflg || !shellflg) { SETERRQ(PetscObjectComm((PetscObject)A),PETSC_ERR_ARG_INCOMP,"Matrices cannot be copied because they have different operations"); }

  ierr = (*shellA->ops->copy)(A,B,str);CHKERRQ(ierr);
  shellB->vscale = shellA->vscale;
  shellB->vshift = shellA->vshift;
  if (shellA->dshift_owned) {
    if (!shellB->dshift_owned) {
      ierr = VecDuplicate(shellA->dshift_owned,&shellB->dshift_owned);CHKERRQ(ierr);
    }
    ierr = VecCopy(shellA->dshift_owned,shellB->dshift_owned);CHKERRQ(ierr);
    shellB->dshift = shellB->dshift_owned;
  } else {
    shellB->dshift = NULL;
  }
  if (shellA->left_owned) {
    if (!shellB->left_owned) {
      ierr = VecDuplicate(shellA->left_owned,&shellB->left_owned);CHKERRQ(ierr);
    }
    ierr = VecCopy(shellA->left_owned,shellB->left_owned);CHKERRQ(ierr);
    shellB->left = shellB->left_owned;
  } else {
    shellB->left = NULL;
  }
  if (shellA->right_owned) {
    if (!shellB->right_owned) {
      ierr = VecDuplicate(shellA->right_owned,&shellB->right_owned);CHKERRQ(ierr);
    }
    ierr = VecCopy(shellA->right_owned,shellB->right_owned);CHKERRQ(ierr);
    shellB->right = shellB->right_owned;
  } else {
    shellB->right = NULL;
  }
  PetscFunctionReturn(0);
}

PetscErrorCode MatMult_Shell(Mat A,Vec x,Vec y)
{
  Mat_Shell        *shell = (Mat_Shell*)A->data;
  PetscErrorCode   ierr;
  Vec              xx;
  PetscObjectState instate,outstate;

  PetscFunctionBegin;
  ierr = MatShellPreScaleRight(A,x,&xx);CHKERRQ(ierr);
  ierr = PetscObjectStateGet((PetscObject)y, &instate);CHKERRQ(ierr);
  ierr = (*shell->ops->mult)(A,xx,y);CHKERRQ(ierr);
  ierr = PetscObjectStateGet((PetscObject)y, &outstate);CHKERRQ(ierr);
  if (instate == outstate) {
    /* increase the state of the output vector since the user did not update its state themself as should have been done */
    ierr = PetscObjectStateIncrease((PetscObject)y);CHKERRQ(ierr);
  }
  ierr = MatShellShiftAndScale(A,xx,y);CHKERRQ(ierr);
  ierr = MatShellPostScaleLeft(A,y);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode MatMultAdd_Shell(Mat A,Vec x,Vec y,Vec z)
{
  Mat_Shell      *shell = (Mat_Shell*)A->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (y == z) {
    if (!shell->right_add_work) {ierr = VecDuplicate(z,&shell->right_add_work);CHKERRQ(ierr);}
    ierr = MatMult(A,x,shell->right_add_work);CHKERRQ(ierr);
    ierr = VecAXPY(z,1.0,shell->right_add_work);CHKERRQ(ierr);
  } else {
    ierr = MatMult(A,x,z);CHKERRQ(ierr);
    ierr = VecAXPY(z,1.0,y);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

PetscErrorCode MatMultTranspose_Shell(Mat A,Vec x,Vec y)
{
  Mat_Shell        *shell = (Mat_Shell*)A->data;
  PetscErrorCode   ierr;
  Vec              xx;
  PetscObjectState instate,outstate;

  PetscFunctionBegin;
  ierr = MatShellPreScaleLeft(A,x,&xx);CHKERRQ(ierr);
  ierr = PetscObjectStateGet((PetscObject)y, &instate);CHKERRQ(ierr);
  ierr = (*shell->ops->multtranspose)(A,xx,y);CHKERRQ(ierr);
  ierr = PetscObjectStateGet((PetscObject)y, &outstate);CHKERRQ(ierr);
  if (instate == outstate) {
    /* increase the state of the output vector since the user did not update its state themself as should have been done */
    ierr = PetscObjectStateIncrease((PetscObject)y);CHKERRQ(ierr);
  }
  ierr = MatShellShiftAndScale(A,xx,y);CHKERRQ(ierr);
  ierr = MatShellPostScaleRight(A,y);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode MatMultTransposeAdd_Shell(Mat A,Vec x,Vec y,Vec z)
{
  Mat_Shell      *shell = (Mat_Shell*)A->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (y == z) {
    if (!shell->left_add_work) {ierr = VecDuplicate(z,&shell->left_add_work);CHKERRQ(ierr);}
    ierr = MatMultTranspose(A,x,shell->left_add_work);CHKERRQ(ierr);
    ierr = VecWAXPY(z,1.0,shell->left_add_work,y);CHKERRQ(ierr);
  } else {
    ierr = MatMultTranspose(A,x,z);CHKERRQ(ierr);
    ierr = VecAXPY(z,1.0,y);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

PetscErrorCode MatGetDiagonal_Shell(Mat A,Vec v)
{
  Mat_Shell      *shell = (Mat_Shell*)A->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = (*shell->ops->getdiagonal)(A,v);CHKERRQ(ierr);
  ierr = VecScale(v,shell->vscale);CHKERRQ(ierr);
  if (shell->dshift) {
    ierr = VecPointwiseMult(v,v,shell->dshift);CHKERRQ(ierr);
  } else {
    ierr = VecShift(v,shell->vshift);CHKERRQ(ierr);
  }
  if (shell->left)  {ierr = VecPointwiseMult(v,v,shell->left);CHKERRQ(ierr);}
  if (shell->right) {ierr = VecPointwiseMult(v,v,shell->right);CHKERRQ(ierr);}
  PetscFunctionReturn(0);
}

PetscErrorCode MatShift_Shell(Mat Y,PetscScalar a)
{
  Mat_Shell      *shell = (Mat_Shell*)Y->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (shell->left || shell->right || shell->dshift) {
    if (!shell->dshift) {
      if (!shell->dshift_owned) {ierr = VecDuplicate(shell->left ? shell->left : shell->right, &shell->dshift_owned);CHKERRQ(ierr);}
      shell->dshift = shell->dshift_owned;
      ierr          = VecSet(shell->dshift,shell->vshift+a);CHKERRQ(ierr);
    } else {ierr = VecScale(shell->dshift,a);CHKERRQ(ierr);}
    if (shell->left)  {ierr = VecPointwiseDivide(shell->dshift,shell->dshift,shell->left);CHKERRQ(ierr);}
    if (shell->right) {ierr = VecPointwiseDivide(shell->dshift,shell->dshift,shell->right);CHKERRQ(ierr);}
  } else shell->vshift += a;
  ierr = MatShellUseScaledMethods(Y);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode MatDiagonalSet_Shell(Mat A,Vec D,InsertMode ins)
{
  Mat_Shell      *shell = (Mat_Shell*)A->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (shell->vscale != 1.0 || shell->left || shell->right) {
    SETERRQ(PetscObjectComm((PetscObject)A),PETSC_ERR_ARG_WRONGSTATE, "Operation not supported on scaled matrix");
  }

  if (shell->ops->diagonalset) {
    ierr = (*shell->ops->diagonalset)(A,D,ins);CHKERRQ(ierr);
  }
  shell->vshift = 0.0;
  if (shell->dshift) { ierr = VecZeroEntries(shell->dshift);CHKERRQ(ierr); }
  PetscFunctionReturn(0);
}

PetscErrorCode MatScale_Shell(Mat Y,PetscScalar a)
{
  Mat_Shell      *shell = (Mat_Shell*)Y->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  shell->vscale *= a;
  if (shell->dshift) {
    ierr = VecScale(shell->dshift,a);CHKERRQ(ierr);
  } else shell->vshift *= a;
  ierr = MatShellUseScaledMethods(Y);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode MatDiagonalScale_Shell(Mat Y,Vec left,Vec right)
{
  Mat_Shell      *shell = (Mat_Shell*)Y->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (left) {
    if (!shell->left_owned) {ierr = VecDuplicate(left,&shell->left_owned);CHKERRQ(ierr);}
    if (shell->left) {
      ierr = VecPointwiseMult(shell->left,shell->left,left);CHKERRQ(ierr);
    } else {
      shell->left = shell->left_owned;
      ierr        = VecCopy(left,shell->left);CHKERRQ(ierr);
    }
  }
  if (right) {
    if (!shell->right_owned) {ierr = VecDuplicate(right,&shell->right_owned);CHKERRQ(ierr);}
    if (shell->right) {
      ierr = VecPointwiseMult(shell->right,shell->right,right);CHKERRQ(ierr);
    } else {
      shell->right = shell->right_owned;
      ierr         = VecCopy(right,shell->right);CHKERRQ(ierr);
    }
  }
  ierr = MatShellUseScaledMethods(Y);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode MatAssemblyEnd_Shell(Mat Y,MatAssemblyType t)
{
  Mat_Shell *shell = (Mat_Shell*)Y->data;

  PetscFunctionBegin;
  if (t == MAT_FINAL_ASSEMBLY) {
    shell->vshift = 0.0;
    shell->vscale = 1.0;
    shell->dshift = NULL;
    shell->left   = NULL;
    shell->right  = NULL;
    if (shell->ops->mult) {
      Y->ops->mult = shell->ops->mult;
      shell->ops->mult  = NULL;
    }
    if (shell->ops->multtranspose) {
      Y->ops->multtranspose = shell->ops->multtranspose;
      shell->ops->multtranspose = NULL;
    }
    if (shell->ops->getdiagonal) {
      Y->ops->getdiagonal = shell->ops->getdiagonal;
      shell->ops->getdiagonal = NULL;
    }
    if (shell->ops->copy) {
      Y->ops->copy = shell->ops->copy;
      shell->ops->copy = NULL;
    }
    shell->usingscaled = PETSC_FALSE;
  }
  PetscFunctionReturn(0);
}

PETSC_INTERN PetscErrorCode MatConvert_Shell(Mat, MatType,MatReuse,Mat*);

static PetscErrorCode MatMissingDiagonal_Shell(Mat A,PetscBool  *missing,PetscInt *d)
{
  PetscFunctionBegin;
  *missing = PETSC_FALSE;
  PetscFunctionReturn(0);
}

static struct _MatOps MatOps_Values = {0,
                                       0,
                                       0,
                                       0,
                                /* 4*/ 0,
                                       0,
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
                                       0,
                                       MatDiagonalScale_Shell,
                                       0,
                                /*20*/ 0,
                                       MatAssemblyEnd_Shell,
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
                                       MatScale_Shell,
                                       MatShift_Shell,
                                       MatDiagonalSet_Shell,
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
                                       MatDestroy_Shell,
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
                                       MatConvert_Shell,
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
                                       MatMissingDiagonal_Shell,
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
   MATSHELL - MATSHELL = "shell" - A matrix type to be used to define your own matrix type -- perhaps matrix free.

  Level: advanced

.seealso: MatCreateShell
M*/

PETSC_EXTERN PetscErrorCode MatCreate_Shell(Mat A)
{
  Mat_Shell      *b;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscMemcpy(A->ops,&MatOps_Values,sizeof(struct _MatOps));CHKERRQ(ierr);

  ierr    = PetscNewLog(A,&b);CHKERRQ(ierr);
  A->data = (void*)b;

  ierr = PetscLayoutSetUp(A->rmap);CHKERRQ(ierr);
  ierr = PetscLayoutSetUp(A->cmap);CHKERRQ(ierr);

  b->ctx           = 0;
  b->vshift        = 0.0;
  b->vscale        = 1.0;
  A->assembled     = PETSC_TRUE;
  A->preallocated  = PETSC_FALSE;

  ierr = PetscObjectChangeTypeName((PetscObject)A,MATSHELL);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
   MatCreateShell - Creates a new matrix class for use with a user-defined
   private data storage format.

  Collective on MPI_Comm

   Input Parameters:
+  comm - MPI communicator
.  m - number of local rows (must be given)
.  n - number of local columns (must be given)
.  M - number of global rows (may be PETSC_DETERMINE)
.  N - number of global columns (may be PETSC_DETERMINE)
-  ctx - pointer to data needed by the shell matrix routines

   Output Parameter:
.  A - the matrix

   Level: advanced

  Usage:
$    extern int mult(Mat,Vec,Vec);
$    MatCreateShell(comm,m,n,M,N,ctx,&mat);
$    MatShellSetOperation(mat,MATOP_MULT,(void(*)(void))mult);
$    [ Use matrix for operations that have been set ]
$    MatDestroy(mat);

   Notes:
   The shell matrix type is intended to provide a simple class to use
   with KSP (such as, for use with matrix-free methods). You should not
   use the shell type if you plan to define a complete matrix class.

   Fortran Notes: To use this from Fortran with a ctx you must write an interface definition for this
    function and for MatShellGetContext() that tells Fortran the Fortran derived data type you are passing
    in as the ctx argument.

   PETSc requires that matrices and vectors being used for certain
   operations are partitioned accordingly.  For example, when
   creating a shell matrix, A, that supports parallel matrix-vector
   products using MatMult(A,x,y) the user should set the number
   of local matrix rows to be the number of local elements of the
   corresponding result vector, y. Note that this is information is
   required for use of the matrix interface routines, even though
   the shell matrix may not actually be physically partitioned.
   For example,

$
$     Vec x, y
$     extern int mult(Mat,Vec,Vec);
$     Mat A
$
$     VecCreateMPI(comm,PETSC_DECIDE,M,&y);
$     VecCreateMPI(comm,PETSC_DECIDE,N,&x);
$     VecGetLocalSize(y,&m);
$     VecGetLocalSize(x,&n);
$     MatCreateShell(comm,m,n,M,N,ctx,&A);
$     MatShellSetOperation(mat,MATOP_MULT,(void(*)(void))mult);
$     MatMult(A,x,y);
$     MatDestroy(A);
$     VecDestroy(y); VecDestroy(x);
$

.keywords: matrix, shell, create

.seealso: MatShellSetOperation(), MatHasOperation(), MatShellGetContext(), MatShellSetContext()
@*/
PetscErrorCode  MatCreateShell(MPI_Comm comm,PetscInt m,PetscInt n,PetscInt M,PetscInt N,void *ctx,Mat *A)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatCreate(comm,A);CHKERRQ(ierr);
  ierr = MatSetSizes(*A,m,n,M,N);CHKERRQ(ierr);
  ierr = MatSetType(*A,MATSHELL);CHKERRQ(ierr);
  ierr = MatShellSetContext(*A,ctx);CHKERRQ(ierr);
  ierr = MatSetUp(*A);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@
    MatShellSetContext - sets the context for a shell matrix

   Logically Collective on Mat

    Input Parameters:
+   mat - the shell matrix
-   ctx - the context

   Level: advanced

   Fortran Notes: To use this from Fortran you must write a Fortran interface definition for this
    function that tells Fortran the Fortran derived data type that you are passing in as the ctx argument.

.seealso: MatCreateShell(), MatShellGetContext(), MatShellGetOperation()
@*/
PetscErrorCode  MatShellSetContext(Mat mat,void *ctx)
{
  Mat_Shell      *shell = (Mat_Shell*)mat->data;
  PetscErrorCode ierr;
  PetscBool      flg;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(mat,MAT_CLASSID,1);
  ierr = PetscObjectTypeCompare((PetscObject)mat,MATSHELL,&flg);CHKERRQ(ierr);
  if (flg) {
    shell->ctx = ctx;
  } else SETERRQ(PetscObjectComm((PetscObject)mat),PETSC_ERR_SUP,"Cannot attach context to non-shell matrix");
  PetscFunctionReturn(0);
}

/*@C
    MatShellSetOperation - Allows user to set a matrix operation for
                           a shell matrix.

   Logically Collective on Mat

    Input Parameters:
+   mat - the shell matrix
.   op - the name of the operation
-   f - the function that provides the operation.

   Level: advanced

    Usage:
$      extern PetscErrorCode usermult(Mat,Vec,Vec);
$      ierr = MatCreateShell(comm,m,n,M,N,ctx,&A);
$      ierr = MatShellSetOperation(A,MATOP_MULT,(void(*)(void))usermult);

    Notes:
    See the file include/petscmat.h for a complete list of matrix
    operations, which all have the form MATOP_<OPERATION>, where
    <OPERATION> is the name (in all capital letters) of the
    user interface routine (e.g., MatMult() -> MATOP_MULT).

    All user-provided functions (execept for MATOP_DESTROY) should have the same calling
    sequence as the usual matrix interface routines, since they
    are intended to be accessed via the usual matrix interface
    routines, e.g.,
$       MatMult(Mat,Vec,Vec) -> usermult(Mat,Vec,Vec)

    In particular each function MUST return an error code of 0 on success and
    nonzero on failure.

    Within each user-defined routine, the user should call
    MatShellGetContext() to obtain the user-defined context that was
    set by MatCreateShell().

    Fortran Notes: For MatCreateVecs() the user code should check if the input left or right matrix is -1 and in that case not
       generate a matrix. See src/mat/examples/tests/ex120f.F

.keywords: matrix, shell, set, operation

.seealso: MatCreateShell(), MatShellGetContext(), MatShellGetOperation(), MatShellSetContext()
@*/
PetscErrorCode  MatShellSetOperation(Mat mat,MatOperation op,void (*f)(void))
{
  PetscErrorCode ierr;
  PetscBool      flg;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(mat,MAT_CLASSID,1);
  switch (op) {
  case MATOP_DESTROY:
    ierr = PetscObjectTypeCompare((PetscObject)mat,MATSHELL,&flg);CHKERRQ(ierr);
    if (flg) {
      Mat_Shell *shell = (Mat_Shell*)mat->data;
      shell->ops->destroy = (PetscErrorCode (*)(Mat))f;
    } else mat->ops->destroy = (PetscErrorCode (*)(Mat))f;
    break;
  case MATOP_DIAGONAL_SET:
    ierr = PetscObjectTypeCompare((PetscObject)mat,MATSHELL,&flg);CHKERRQ(ierr);
    if (flg) {
      Mat_Shell *shell = (Mat_Shell*)mat->data;
      shell->ops->diagonalset = (PetscErrorCode (*)(Mat,Vec,InsertMode))f;
    } else {
      mat->ops->diagonalset = (PetscErrorCode (*)(Mat,Vec,InsertMode))f;
    }
    break;
  case MATOP_VIEW:
    mat->ops->view = (PetscErrorCode (*)(Mat,PetscViewer))f;
    break;
  case MATOP_MULT:
    mat->ops->mult = (PetscErrorCode (*)(Mat,Vec,Vec))f;
    if (!mat->ops->multadd) {
      ierr = PetscObjectTypeCompare((PetscObject)mat,MATSHELL,&flg);CHKERRQ(ierr);
      if (flg) mat->ops->multadd = MatMultAdd_Shell;
    }
    break;
  case MATOP_MULT_TRANSPOSE:
    mat->ops->multtranspose = (PetscErrorCode (*)(Mat,Vec,Vec))f;
    if (!mat->ops->multtransposeadd) {
      ierr = PetscObjectTypeCompare((PetscObject)mat,MATSHELL,&flg);CHKERRQ(ierr);
      if (flg) mat->ops->multtransposeadd = MatMultTransposeAdd_Shell;
    }
    break;
  default:
    (((void(**)(void))mat->ops)[op]) = f;
  }
  PetscFunctionReturn(0);
}

/*@C
    MatShellGetOperation - Gets a matrix function for a shell matrix.

    Not Collective

    Input Parameters:
+   mat - the shell matrix
-   op - the name of the operation

    Output Parameter:
.   f - the function that provides the operation.

    Level: advanced

    Notes:
    See the file include/petscmat.h for a complete list of matrix
    operations, which all have the form MATOP_<OPERATION>, where
    <OPERATION> is the name (in all capital letters) of the
    user interface routine (e.g., MatMult() -> MATOP_MULT).

    All user-provided functions have the same calling
    sequence as the usual matrix interface routines, since they
    are intended to be accessed via the usual matrix interface
    routines, e.g.,
$       MatMult(Mat,Vec,Vec) -> usermult(Mat,Vec,Vec)

    Within each user-defined routine, the user should call
    MatShellGetContext() to obtain the user-defined context that was
    set by MatCreateShell().

.keywords: matrix, shell, set, operation

.seealso: MatCreateShell(), MatShellGetContext(), MatShellSetOperation(), MatShellSetContext()
@*/
PetscErrorCode  MatShellGetOperation(Mat mat,MatOperation op,void(**f)(void))
{
  PetscErrorCode ierr;
  PetscBool      flg;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(mat,MAT_CLASSID,1);
  switch (op) {
  case MATOP_DESTROY:
    ierr = PetscObjectTypeCompare((PetscObject)mat,MATSHELL,&flg);CHKERRQ(ierr);
    if (flg) {
      Mat_Shell *shell = (Mat_Shell*)mat->data;
      *f = (void (*)(void))shell->ops->destroy;
    } else {
      *f = (void (*)(void))mat->ops->destroy;
    }
    break;
  case MATOP_DIAGONAL_SET:
    ierr = PetscObjectTypeCompare((PetscObject)mat,MATSHELL,&flg);CHKERRQ(ierr);
    if (flg) {
      Mat_Shell *shell = (Mat_Shell*)mat->data;
      *f = (void (*)(void))shell->ops->diagonalset;
    } else {
      *f = (void (*)(void))mat->ops->diagonalset;
    }
    break;
  case MATOP_VIEW:
    *f = (void (*)(void))mat->ops->view;
    break;
  default:
    *f = (((void (**)(void))mat->ops)[op]);
  }
  PetscFunctionReturn(0);
}

