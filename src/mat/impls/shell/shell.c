#define PETSCMAT_DLL

/*
   This provides a simple shell for Fortran (and C programmers) to 
  create a very simple matrix class for use with KSP without coding 
  much of anything.
*/

#include "private/matimpl.h"        /*I "petscmat.h" I*/
#include "private/vecimpl.h"  

typedef struct {
  PetscErrorCode (*destroy)(Mat);
  PetscErrorCode (*mult)(Mat,Vec,Vec);
  PetscErrorCode (*multtranspose)(Mat,Vec,Vec);
  PetscErrorCode (*getdiagonal)(Mat,Vec);
  PetscTruth     scale,shift;
  PetscScalar    vscale,vshift;
  void           *ctx;
} Mat_Shell;      

#undef __FUNCT__  
#define __FUNCT__ "MatShellGetContext"
/*@C
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
PetscErrorCode PETSCMAT_DLLEXPORT MatShellGetContext(Mat mat,void **ctx)
{
  PetscErrorCode ierr;
  PetscTruth     flg;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(mat,MAT_COOKIE,1);
  PetscValidPointer(ctx,2); 
  ierr = PetscTypeCompare((PetscObject)mat,MATSHELL,&flg);CHKERRQ(ierr);
  if (!flg) *ctx = 0; 
  else      *ctx = ((Mat_Shell*)(mat->data))->ctx; 
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatDestroy_Shell"
PetscErrorCode MatDestroy_Shell(Mat mat)
{
  PetscErrorCode ierr;
  Mat_Shell      *shell;

  PetscFunctionBegin;
  shell = (Mat_Shell*)mat->data;
  if (shell->destroy) {ierr = (*shell->destroy)(mat);CHKERRQ(ierr);}
  ierr = PetscFree(shell);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatMult_Shell"
PetscErrorCode MatMult_Shell(Mat A,Vec x,Vec y)
{
  Mat_Shell      *shell = (Mat_Shell*)A->data;  
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = (*shell->mult)(A,x,y);CHKERRQ(ierr);
  if (shell->shift && shell->scale) {
    ierr = VecAXPBY(y,shell->vshift,shell->vscale,x);CHKERRQ(ierr);
  } else if (shell->scale) {
    ierr = VecScale(y,shell->vscale);CHKERRQ(ierr);
  } else {
    ierr = VecAXPY(y,shell->vshift,x);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatMultTranspose_Shell"
PetscErrorCode MatMultTranspose_Shell(Mat A,Vec x,Vec y)
{
  Mat_Shell      *shell = (Mat_Shell*)A->data;  
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = (*shell->multtranspose)(A,x,y);CHKERRQ(ierr);
  if (shell->shift && shell->scale) {
    ierr = VecAXPBY(y,shell->vshift,shell->vscale,x);CHKERRQ(ierr);
  } else if (shell->scale) {
    ierr = VecScale(y,shell->vscale);CHKERRQ(ierr);
  } else {
    ierr = VecAXPY(y,shell->vshift,x);CHKERRQ(ierr);
  }
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
  if (shell->scale) {
    ierr = VecScale(v,shell->vscale);CHKERRQ(ierr);
  }
  if (shell->shift) {
    ierr = VecShift(v,shell->vshift);CHKERRQ(ierr);
  } 
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatShift_Shell"
PetscErrorCode MatShift_Shell(Mat Y,PetscScalar a)
{
  Mat_Shell *shell = (Mat_Shell*)Y->data;  

  PetscFunctionBegin;
  if (shell->scale || shell->shift) {
    shell->vshift += a;
  } else {
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
    shell->vshift = a;
  }
  shell->shift  =  PETSC_TRUE;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatScale_Shell"
PetscErrorCode MatScale_Shell(Mat Y,PetscScalar a)
{
  Mat_Shell *shell = (Mat_Shell*)Y->data;  

  PetscFunctionBegin;
  if (shell->scale || shell->shift) {
    shell->vscale *= a;
  } else {
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
    shell->vscale = a;
  }
  shell->scale  =  PETSC_TRUE;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatAssemblyEnd_Shell"
PetscErrorCode MatAssemblyEnd_Shell(Mat Y,MatAssemblyType t)
{
  Mat_Shell *shell = (Mat_Shell*)Y->data;  

  PetscFunctionBegin;
  if ((shell->shift || shell->scale) && t == MAT_FINAL_ASSEMBLY) {
    shell->scale  = PETSC_FALSE;
    shell->shift  = PETSC_FALSE;
    shell->vshift = 0.0;
    shell->vscale = 1.0;
    Y->ops->mult          = shell->mult;
    Y->ops->multtranspose = shell->multtranspose;
    Y->ops->getdiagonal   = shell->getdiagonal;
  }
  PetscFunctionReturn(0);
}

EXTERN PetscErrorCode MatConvert_Shell(Mat, const MatType,MatReuse,Mat*);

#undef __FUNCT__  
#define __FUNCT__ "MatSetBlockSize_Shell"
PetscErrorCode MatSetBlockSize_Shell(Mat A,PetscInt bs)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscLayoutSetBlockSize(A->rmap,bs);CHKERRQ(ierr);
  ierr = PetscLayoutSetBlockSize(A->cmap,bs);CHKERRQ(ierr);
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
       0,
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
/*49*/ MatSetBlockSize_Shell,
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
PetscErrorCode PETSCMAT_DLLEXPORT MatCreate_Shell(Mat A)
{
  Mat_Shell      *b;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscMemcpy(A->ops,&MatOps_Values,sizeof(struct _MatOps));CHKERRQ(ierr);

  ierr = PetscNewLog(A,Mat_Shell,&b);CHKERRQ(ierr);
  A->data = (void*)b;

  if (A->rmap->n == PETSC_DECIDE || A->cmap->n == PETSC_DECIDE) {
    SETERRQ(PETSC_ERR_ARG_WRONG,"Must give local row and column count for matrix");
  }

  ierr = PetscLayoutSetBlockSize(A->rmap,1);CHKERRQ(ierr);
  ierr = PetscLayoutSetBlockSize(A->cmap,1);CHKERRQ(ierr);
  ierr = PetscLayoutSetUp(A->rmap);CHKERRQ(ierr);
  ierr = PetscLayoutSetUp(A->cmap);CHKERRQ(ierr);

  b->ctx           = 0;
  b->scale         = PETSC_FALSE;
  b->shift         = PETSC_FALSE;
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
PetscErrorCode PETSCMAT_DLLEXPORT MatCreateShell(MPI_Comm comm,PetscInt m,PetscInt n,PetscInt M,PetscInt N,void *ctx,Mat *A)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatCreate(comm,A);CHKERRQ(ierr);
  ierr = MatSetSizes(*A,m,n,M,N);CHKERRQ(ierr);
  
  ierr = MatSetType(*A,MATSHELL);CHKERRQ(ierr);
  ierr = MatShellSetContext(*A,ctx);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatShellSetContext"
/*@
    MatShellSetContext - sets the context for a shell matrix

   Collective on Mat

    Input Parameters:
+   mat - the shell matrix
-   ctx - the context

   Level: advanced

   Fortran Notes: The context can only be an integer or a PetscObject
      unfortunately it cannot be a Fortran array or derived type.

.seealso: MatCreateShell(), MatShellGetContext(), MatShellGetOperation()
@*/
PetscErrorCode PETSCMAT_DLLEXPORT MatShellSetContext(Mat mat,void *ctx)
{
  Mat_Shell      *shell = (Mat_Shell*)mat->data;
  PetscErrorCode ierr;
  PetscTruth     flg;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(mat,MAT_COOKIE,1);
  ierr = PetscTypeCompare((PetscObject)mat,MATSHELL,&flg);CHKERRQ(ierr);
  if (flg) {
    shell->ctx = ctx;
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatShellSetOperation"
/*@C
    MatShellSetOperation - Allows user to set a matrix operation for
                           a shell matrix.

   Collective on Mat

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

    All user-provided functions should have the same calling
    sequence as the usual matrix interface routines, since they
    are intended to be accessed via the usual matrix interface
    routines, e.g., 
$       MatMult(Mat,Vec,Vec) -> usermult(Mat,Vec,Vec)

    In particular each function MUST return an error code of 0 on success and 
    nonzero on failure.

    Within each user-defined routine, the user should call
    MatShellGetContext() to obtain the user-defined context that was
    set by MatCreateShell().

.keywords: matrix, shell, set, operation

.seealso: MatCreateShell(), MatShellGetContext(), MatShellGetOperation(), MatShellSetContext()
@*/
PetscErrorCode PETSCMAT_DLLEXPORT MatShellSetOperation(Mat mat,MatOperation op,void (*f)(void))
{
  PetscErrorCode ierr;
  PetscTruth     flg;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(mat,MAT_COOKIE,1);
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
PetscErrorCode PETSCMAT_DLLEXPORT MatShellGetOperation(Mat mat,MatOperation op,void(**f)(void))
{
  PetscErrorCode ierr;
  PetscTruth     flg;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(mat,MAT_COOKIE,1);
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

