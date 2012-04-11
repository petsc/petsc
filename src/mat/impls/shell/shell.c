
/*
   This provides a simple shell for Fortran (and C programmers) to 
  create a very simple matrix class for use with KSP without coding 
  much of anything.
*/

#include <petsc-private/matimpl.h>        /*I "petscmat.h" I*/
#include <petsc-private/vecimpl.h>  

typedef struct {
  PetscErrorCode (*destroy)(Mat);
  PetscErrorCode (*mult)(Mat,Vec,Vec);
  PetscErrorCode (*multtranspose)(Mat,Vec,Vec);
  PetscErrorCode (*getdiagonal)(Mat,Vec);
  PetscScalar    vscale,vshift;
  Vec            dshift;
  Vec            left,right;
  Vec            dshift_owned,left_owned,right_owned;
  Vec            left_work,right_work;
  PetscBool      usingscaled;
  void           *ctx;
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

#undef __FUNCT__  
#define __FUNCT__ "MatShellUseScaledMethods"
static PetscErrorCode MatShellUseScaledMethods(Mat Y)
{
  Mat_Shell *shell = (Mat_Shell*)Y->data;

  PetscFunctionBegin;
  if (shell->usingscaled) PetscFunctionReturn(0);
  shell->mult  = Y->ops->mult;
  Y->ops->mult = MatMult_Shell;
  if (Y->ops->multtranspose) {
    shell->multtranspose  = Y->ops->multtranspose;
    Y->ops->multtranspose = MatMultTranspose_Shell;
  }
  if (Y->ops->getdiagonal) {
    shell->getdiagonal  = Y->ops->getdiagonal;
    Y->ops->getdiagonal = MatGetDiagonal_Shell;
  }
  shell->usingscaled = PETSC_TRUE;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatShellPreScaleLeft"
static PetscErrorCode MatShellPreScaleLeft(Mat A,Vec x,Vec *xx)
{
  Mat_Shell *shell = (Mat_Shell*)A->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  *xx = PETSC_NULL;
  if (!shell->left) {
    *xx = x;
  } else {
    if (!shell->left_work) {ierr = VecDuplicate(shell->left,&shell->left_work);CHKERRQ(ierr);}
    ierr = VecPointwiseMult(shell->left_work,x,shell->left);CHKERRQ(ierr);
    *xx = shell->left_work;
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatShellPreScaleRight"
static PetscErrorCode MatShellPreScaleRight(Mat A,Vec x,Vec *xx)
{
  Mat_Shell *shell = (Mat_Shell*)A->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  *xx = PETSC_NULL;
  if (!shell->right) {
    *xx = x;
  } else {
    if (!shell->right_work) {ierr = VecDuplicate(shell->right,&shell->right_work);CHKERRQ(ierr);}
    ierr = VecPointwiseMult(shell->right_work,x,shell->right);CHKERRQ(ierr);
    *xx = shell->right_work;
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatShellPostScaleLeft"
static PetscErrorCode MatShellPostScaleLeft(Mat A,Vec x)
{
  Mat_Shell *shell = (Mat_Shell*)A->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (shell->left) {ierr = VecPointwiseMult(x,x,shell->left);CHKERRQ(ierr);}
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatShellPostScaleRight"
static PetscErrorCode MatShellPostScaleRight(Mat A,Vec x)
{
  Mat_Shell *shell = (Mat_Shell*)A->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (shell->right) {ierr = VecPointwiseMult(x,x,shell->right);CHKERRQ(ierr);}
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatShellShiftAndScale"
static PetscErrorCode MatShellShiftAndScale(Mat A,Vec X,Vec Y)
{
  Mat_Shell *shell = (Mat_Shell*)A->data;
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

#undef __FUNCT__  
#define __FUNCT__ "MatShellGetContext"
/*@
    MatShellGetContext - Returns the user-provided context associated with a shell matrix.

    Not Collective

    Input Parameter:
.   mat - the matrix, should have been created with MatCreateShell()

    Output Parameter:
.   ctx - the user provided context

    Level: advanced

    Notes:
    This routine is intended for use within various shell matrix routines,
    as set with MatShellSetOperation().
    
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
  ierr = PetscTypeCompare((PetscObject)mat,MATSHELL,&flg);CHKERRQ(ierr);
  if (flg) *(void**)ctx = ((Mat_Shell*)(mat->data))->ctx;
  else SETERRQ(((PetscObject)mat)->comm,PETSC_ERR_SUP,"Cannot get context from non-shell matrix");
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatDestroy_Shell"
PetscErrorCode MatDestroy_Shell(Mat mat)
{
  PetscErrorCode ierr;
  Mat_Shell      *shell = (Mat_Shell*)mat->data;

  PetscFunctionBegin;
  if (shell->destroy) {
    ierr = (*shell->destroy)(mat);CHKERRQ(ierr);
  }
  ierr = VecDestroy(&shell->left_owned);CHKERRQ(ierr);
  ierr = VecDestroy(&shell->right_owned);CHKERRQ(ierr);
  ierr = VecDestroy(&shell->dshift_owned);CHKERRQ(ierr);
  ierr = VecDestroy(&shell->left_work);CHKERRQ(ierr);
  ierr = VecDestroy(&shell->right_work);CHKERRQ(ierr);
  ierr = PetscFree(mat->data);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatMult_Shell"
PetscErrorCode MatMult_Shell(Mat A,Vec x,Vec y)
{
  Mat_Shell      *shell = (Mat_Shell*)A->data;
  PetscErrorCode ierr;
  Vec            xx;

  PetscFunctionBegin;
  ierr = MatShellPreScaleRight(A,x,&xx);CHKERRQ(ierr);
  ierr = (*shell->mult)(A,xx,y);CHKERRQ(ierr);
  ierr = MatShellShiftAndScale(A,xx,y);CHKERRQ(ierr);
  ierr = MatShellPostScaleLeft(A,y);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatMultTranspose_Shell"
PetscErrorCode MatMultTranspose_Shell(Mat A,Vec x,Vec y)
{
  Mat_Shell      *shell = (Mat_Shell*)A->data;
  PetscErrorCode ierr;
  Vec            xx;

  PetscFunctionBegin;
  ierr = MatShellPreScaleLeft(A,x,&xx);CHKERRQ(ierr);
  ierr = (*shell->multtranspose)(A,xx,y);CHKERRQ(ierr);
  ierr = MatShellShiftAndScale(A,xx,y);CHKERRQ(ierr);
  ierr = MatShellPostScaleRight(A,y);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatGetDiagonal_Shell"
PetscErrorCode MatGetDiagonal_Shell(Mat A,Vec v)
{
  Mat_Shell      *shell = (Mat_Shell*)A->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = (*shell->getdiagonal)(A,v);CHKERRQ(ierr);
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

#undef __FUNCT__  
#define __FUNCT__ "MatShift_Shell"
PetscErrorCode MatShift_Shell(Mat Y,PetscScalar a)
{
  Mat_Shell *shell = (Mat_Shell*)Y->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (shell->left || shell->right || shell->dshift) {
    if (!shell->dshift) {
      if (!shell->dshift_owned) {ierr = VecDuplicate(shell->left ? shell->left : shell->right, &shell->dshift_owned);CHKERRQ(ierr);}
      shell->dshift = shell->dshift_owned;
      ierr = VecSet(shell->dshift,shell->vshift+a);CHKERRQ(ierr);
    } else {ierr = VecScale(shell->dshift,a);CHKERRQ(ierr);}
    if (shell->left)  {ierr = VecPointwiseDivide(shell->dshift,shell->dshift,shell->left);CHKERRQ(ierr);}
    if (shell->right) {ierr = VecPointwiseDivide(shell->dshift,shell->dshift,shell->right);CHKERRQ(ierr);}
  } else shell->vshift += a;
  ierr = MatShellUseScaledMethods(Y);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatScale_Shell"
PetscErrorCode MatScale_Shell(Mat Y,PetscScalar a)
{
  Mat_Shell      *shell = (Mat_Shell*)Y->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  shell->vscale *= a;
  if (shell->dshift) {ierr = VecScale(shell->dshift,a);CHKERRQ(ierr);}
  else shell->vshift *= a;
  ierr = MatShellUseScaledMethods(Y);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatDiagonalScale_Shell"
static PetscErrorCode MatDiagonalScale_Shell(Mat Y,Vec left,Vec right)
{
  Mat_Shell *shell = (Mat_Shell*)Y->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (left) {
    if (!shell->left_owned) {ierr = VecDuplicate(left,&shell->left_owned);CHKERRQ(ierr);}
    if (shell->left) {ierr = VecPointwiseMult(shell->left,shell->left,left);CHKERRQ(ierr);}
    else {
      shell->left = shell->left_owned;
      ierr = VecCopy(left,shell->left);CHKERRQ(ierr);
    }
  }
  if (right) {
    if (!shell->right_owned) {ierr = VecDuplicate(right,&shell->right_owned);CHKERRQ(ierr);}
    if (shell->right) {ierr = VecPointwiseMult(shell->right,shell->right,right);CHKERRQ(ierr);}
    else {
      shell->right = shell->right_owned;
      ierr = VecCopy(right,shell->right);CHKERRQ(ierr);
    }
  }
  ierr = MatShellUseScaledMethods(Y);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatAssemblyEnd_Shell"
PetscErrorCode MatAssemblyEnd_Shell(Mat Y,MatAssemblyType t)
{
  Mat_Shell *shell = (Mat_Shell*)Y->data;  

  PetscFunctionBegin;
  if (t == MAT_FINAL_ASSEMBLY) {
    shell->vshift      = 0.0;
    shell->vscale      = 1.0;
    shell->dshift      = PETSC_NULL;
    shell->left        = PETSC_NULL;
    shell->right       = PETSC_NULL;
    if (shell->mult) {
      Y->ops->mult = shell->mult;
      shell->mult  = PETSC_NULL;
    }
    if (shell->multtranspose) {
      Y->ops->multtranspose = shell->multtranspose;
      shell->multtranspose  = PETSC_NULL;
    }
    if (shell->getdiagonal) {
      Y->ops->getdiagonal = shell->getdiagonal;
      shell->getdiagonal  = PETSC_NULL;
    }
    shell->usingscaled = PETSC_FALSE;
  }
  PetscFunctionReturn(0);
}

extern PetscErrorCode MatConvert_Shell(Mat, const MatType,MatReuse,Mat*);

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
       0};

/*MC
   MATSHELL - MATSHELL = "shell" - A matrix type to be used to define your own matrix type -- perhaps matrix free.

  Level: advanced

.seealso: MatCreateShell
M*/

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "MatCreate_Shell"
PetscErrorCode  MatCreate_Shell(Mat A)
{
  Mat_Shell      *b;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscMemcpy(A->ops,&MatOps_Values,sizeof(struct _MatOps));CHKERRQ(ierr);

  ierr = PetscNewLog(A,Mat_Shell,&b);CHKERRQ(ierr);
  A->data = (void*)b;

  ierr = PetscLayoutSetUp(A->rmap);CHKERRQ(ierr);
  ierr = PetscLayoutSetUp(A->cmap);CHKERRQ(ierr);

  b->ctx           = 0;
  b->vshift        = 0.0;
  b->vscale        = 1.0;
  b->mult          = 0;
  b->multtranspose = 0;
  b->getdiagonal   = 0;
  A->assembled     = PETSC_TRUE;
  A->preallocated  = PETSC_FALSE;
  ierr = PetscObjectChangeTypeName((PetscObject)A,MATSHELL);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
EXTERN_C_END

#undef __FUNCT__  
#define __FUNCT__ "MatCreateShell"
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

   Fortran Notes: The context can only be an integer or a PetscObject
      unfortunately it cannot be a Fortran array or derived type.

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

#undef __FUNCT__  
#define __FUNCT__ "MatShellSetContext"
/*@
    MatShellSetContext - sets the context for a shell matrix

   Logically Collective on Mat

    Input Parameters:
+   mat - the shell matrix
-   ctx - the context

   Level: advanced

   Fortran Notes: The context can only be an integer or a PetscObject
      unfortunately it cannot be a Fortran array or derived type.

.seealso: MatCreateShell(), MatShellGetContext(), MatShellGetOperation()
@*/
PetscErrorCode  MatShellSetContext(Mat mat,void *ctx)
{
  Mat_Shell      *shell = (Mat_Shell*)mat->data;
  PetscErrorCode ierr;
  PetscBool      flg;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(mat,MAT_CLASSID,1);
  ierr = PetscTypeCompare((PetscObject)mat,MATSHELL,&flg);CHKERRQ(ierr);
  if (flg) {
    shell->ctx = ctx;
  } else SETERRQ(((PetscObject)mat)->comm,PETSC_ERR_SUP,"Cannot attach context to non-shell matrix");
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatShellSetOperation"
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

    Fortran Notes: For MatGetVecs() the user code should check if the input left or right matrix is -1 and in that case not
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
  if (op == MATOP_DESTROY) {
    ierr = PetscTypeCompare((PetscObject)mat,MATSHELL,&flg);CHKERRQ(ierr);
    if (flg) {
       Mat_Shell *shell = (Mat_Shell*)mat->data;
       shell->destroy                 = (PetscErrorCode (*)(Mat)) f;
    } else mat->ops->destroy          = (PetscErrorCode (*)(Mat)) f;
  } 
  else if (op == MATOP_VIEW) mat->ops->view  = (PetscErrorCode (*)(Mat,PetscViewer)) f;
  else                       (((void(**)(void))mat->ops)[op]) = f;

  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatShellGetOperation"
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
  if (op == MATOP_DESTROY) {
    ierr = PetscTypeCompare((PetscObject)mat,MATSHELL,&flg);CHKERRQ(ierr);
    if (flg) {
      Mat_Shell *shell = (Mat_Shell*)mat->data;
      *f = (void(*)(void))shell->destroy;
    } else {
      *f = (void(*)(void))mat->ops->destroy;
    }
  } else if (op == MATOP_VIEW) {
    *f = (void(*)(void))mat->ops->view;
  } else {
    *f = (((void(**)(void))mat->ops)[op]);
  }

  PetscFunctionReturn(0);
}

