#ifndef lint
static char vcid[] = "$Id: shell.c,v 1.19 1995/09/30 19:28:52 bsmith Exp bsmith $";
#endif

/*
   This provides a simple shell for Fortran (and C programmers) to 
  create a very simple matrix class for use with KSP without coding 
  mush of anything.
*/

#include "petsc.h"
#include "matimpl.h"        /*I "mat.h" I*/
#include "vec/vecimpl.h"  

typedef struct {
  int  m, n;                       /* rows, columns */
  int  (*mult)(void*,Vec,Vec);
  int  (*multtransadd)(void*,Vec,Vec,Vec);
  int  (*destroy)(void*);
  int  (*getsize)(void*,int*,int*);
  void *ctx;
} Mat_Shell;      

static int MatGetSize_Shell(Mat mat,int *m,int *n)
{
  Mat_Shell *shell = (Mat_Shell *) mat->data;
  *m = shell->m; *n = shell->n;
  return 0;
}

static int MatMult_Shell(Mat mat,Vec x,Vec y)
{
  Mat_Shell *shell = (Mat_Shell *) mat->data;
  return (*shell->mult)(shell->ctx,x,y);
}

static int MatMultTransAdd_Shell(Mat mat,Vec x,Vec y,Vec z)
{
  Mat_Shell *shell = (Mat_Shell *) mat->data;
  return (*shell->multtransadd)(shell->ctx,x,y,z);
}

static int MatDestroy_Shell(PetscObject obj)
{
  int      ierr;
  Mat      mat = (Mat) obj;
  Mat_Shell *shell;
  shell = (Mat_Shell *) mat->data;
  if (shell->destroy) {ierr = (*shell->destroy)(shell->ctx);CHKERRQ(ierr);}
  PETSCFREE(shell); 
  PLogObjectDestroy(mat);
  PETSCHEADERDESTROY(mat);
  return 0;
}
  
static struct _MatOps MatOps = {0,0,
       0, 
       MatMult_Shell,0,0,MatMultTransAdd_Shell,
       0,0,0,0,
       0,0,
       0,
       0,
       0,0,0,
       0,
       0,0,0,
       0,0,
       0,
       0,0,0,0,
       0,0,MatGetSize_Shell,
       0,0,0,0,
       0,0,0,0 };

/*@C
   MatShellCreate - Creates a new matrix class for use with a user-defined
   private data storage format. 

   Input Parameters:
.  comm - MPI communicator
.  m - number of rows
.  n - number of columns
.  ctx - pointer to data needed by matrix-vector multiplication routine(s)

   Output Parameter:
.  mat - the matrix

   Notes:
   The shell matrix type is intended to provide a simple class to use
   with KSP (such as, for use with matrix-free methods). You should not
   use the shell type if you plan to define a complete matrix class.

  Usage:
$   MatShellCreate(m,n,ctx,&mat);
$   MatShellSetMult(mat,mult);

.keywords: matrix, shell, create

.seealso: MatShellSetMult(), MatShellSetMultTransAdd()
@*/
int MatShellCreate(MPI_Comm comm,int m,int n,void *ctx,Mat *mat)
{
  Mat      newmat;
  Mat_Shell *shell;
  PETSCHEADERCREATE(newmat,_Mat,MAT_COOKIE,MATSHELL,comm);
  PLogObjectCreate(newmat);
  *mat           = newmat;
  newmat->factor = 0;
  newmat->destroy= MatDestroy_Shell;
  PetscMemcpy(&newmat->ops,&MatOps,sizeof(struct _MatOps));
  shell          = PETSCNEW(Mat_Shell); CHKPTRQ(shell);
  PetscMemzero(shell,sizeof(Mat_Shell));
  newmat->data   = (void *) shell;
  shell->mult    = 0;
  shell->m       = m;
  shell->n       = n;
  shell->ctx     = ctx;
  return 0;
}

/*@C
   MatShellSetMult - Sets the routine for computing the matrix-vector product.

   Input Parameters:
.  mat - the matrix associated with this operation, created 
         with MatShellCreate()
.  mult - the user-defined routine

   Calling sequence of mult:
   int mult (void *ptr,Vec xin,Vec xout)
.  ptr - the application context for matrix data
.  xin - input vector
.  xout - output vector

.keywords: matrix, multiply, shell, set

.seealso: MatShellCreate(), MatShellSetMultTransAdd()
@*/
int MatShellSetMult(Mat mat,int (*mult)(void*,Vec,Vec))
{
  Mat_Shell *shell;
  PETSCVALIDHEADERSPECIFIC(mat,MAT_COOKIE);
  shell = (Mat_Shell *) mat->data;
  shell->mult = mult;
  return 0;
}
/*@C
   MatShellSetMultTransAdd - Sets the routine for computing v3 = v2 + A' * v1.

   Input Parameters:
.  mat - the matrix associated with this operation, created 
         with MatShellCreate()
.  mult - the user-defined routine

   Calling sequence of mult:
   int mult (void *ptr,Vec v1,Vec v2,Vec v3)
.  ptr - the application context for matrix data
.  v1, v2 - the input vectors
.  v3 - the result

.keywords: matrix, multiply, transpose

.seealso: MatShellCreate(), MatShellSetMult()
@*/
int MatShellSetMultTransAdd(Mat mat,int (*mult)(void*,Vec,Vec,Vec))
{
  Mat_Shell *shell;
  PETSCVALIDHEADERSPECIFIC(mat,MAT_COOKIE);
  shell               = (Mat_Shell *) mat->data;
  shell->multtransadd = mult;
  return 0;
}
/*@C
   MatShellSetDestroy - Set the routine to use to destroy the 
        private contents of your MatShell.

   Input Parameters:
.  mat - the matrix associated with this operation, created 
         with MatShellCreate()
.  destroy - the user-defined routine

   Calling sequence of mult:
   int destroy (void *ptr)
.  ptr - the application context for matrix data

.keywords: matrix, destroy, shell, set

.seealso: MatShellCreate()
@*/
int MatShellSetDestroy(Mat mat,int (*destroy)(void*))
{
  Mat_Shell *shell;
  PETSCVALIDHEADERSPECIFIC(mat,MAT_COOKIE);
  shell = (Mat_Shell *) mat->data;
  shell->destroy = destroy;
  return 0;
}


