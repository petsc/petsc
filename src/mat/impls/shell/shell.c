#ifndef lint
static char vcid[] = "$Id: shell.c,v 1.26 1996/03/19 21:26:12 bsmith Exp bsmith $";
#endif

/*
   This provides a simple shell for Fortran (and C programmers) to 
  create a very simple matrix class for use with KSP without coding 
  much of anything.
*/

#include "petsc.h"
#include "matimpl.h"        /*I "mat.h" I*/
#include "vec/vecimpl.h"  

typedef struct {
  int  m, n;                       /* rows, columns */
  int  (*destroy)(void*);
  void *ctx;
} Mat_Shell;      

/*@
    MatShellGetContext - Returns the user provided context associated 
         with a MatShell.

  Input Parameter:
.   mat - the matrix, should have been created with MatCreateShell()

  Output Parameter:
.   ctx - the user provided context

@*/
int MatShellGetContext(Mat mat,void **ctx)
{
  PetscValidHeaderSpecific(mat,MAT_COOKIE); 
  if (mat->type != MATSHELL) *ctx = 0; 
  else                       *ctx = ((Mat_Shell *) (mat->data))->ctx; 
  return 0;
}

static int MatGetSize_Shell(Mat mat,int *m,int *n)
{
  Mat_Shell *shell = (Mat_Shell *) mat->data;
  *m = shell->m; *n = shell->n;
  return 0;
}

static int MatDestroy_Shell(PetscObject obj)
{
  int       ierr;
  Mat       mat = (Mat) obj;
  Mat_Shell *shell;

  shell = (Mat_Shell *) mat->data;
  if (shell->destroy) {ierr = (*shell->destroy)(shell->ctx);CHKERRQ(ierr);}
  PetscFree(shell); 
  PLogObjectDestroy(mat);
  PetscHeaderDestroy(mat);
  return 0;
}
  
static struct _MatOps MatOps = {0,0,
       0, 
       0,0,0,0,
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
   MatCreateShell - Creates a new matrix class for use with a user-defined
   private data storage format. 

   Input Parameters:
.  comm - MPI communicator
.  m - number of rows
.  n - number of columns
.  ctx - pointer to data needed by the matrix routines

   Output Parameter:
.  mat - the matrix

   Notes:
   The shell matrix type is intended to provide a simple class to use
   with KSP (such as, for use with matrix-free methods). You should not
   use the shell type if you plan to define a complete matrix class.

  Usage:
$   MatCreateShell(m,n,ctx,&mat);
$   MatSetOperation(mat,MAT_MULT,mult);

.keywords: matrix, shell, create

.seealso: MatSetOperation(), MatHasOperation(), MatShellGetContext()
@*/
int MatCreateShell(MPI_Comm comm,int m,int n,void *ctx,Mat *mat)
{
  Mat       newmat;
  Mat_Shell *shell;

  PetscHeaderCreate(newmat,_Mat,MAT_COOKIE,MATSHELL,comm);
  PLogObjectCreate(newmat);
  *mat              = newmat;
  newmat->factor    = 0;
  newmat->destroy   = MatDestroy_Shell;
  newmat->assembled = PETSC_TRUE;
  PetscMemcpy(&newmat->ops,&MatOps,sizeof(struct _MatOps));

  shell          = PetscNew(Mat_Shell); CHKPTRQ(shell);
  PetscMemzero(shell,sizeof(Mat_Shell));
  newmat->data   = (void *) shell;
  shell->m       = m;
  shell->n       = n;
  shell->ctx     = ctx;
  return 0;
}

/*@
    MatShellSetOperation - Allows use to set a matrix operation for a shell matrix.

  Input Parameters:
.   mat - the shell matrix
.   op - the name of the operation
.   f - the function that provides the operation.

  Usage:
   extern int mult(Mat,Vec,Vec);
   ierr = MatCreateShell(comm,m,m,ctx,&A);
   ierr = MatSetOperation(A,MAT_MULT,mult);

   In the user provided function, use MatShellGetContext() to obtain the 
context passed into MatCreateShell().
@*/
int MatShellSetOperation(Mat mat,MatOperation op, void *f)
{
  PetscValidHeaderSpecific(mat,MAT_COOKIE);

  if (op == MAT_DESTROY)   mat->destroy              = (int (*)(PetscObject)) f;
  else if (op == MAT_VIEW) mat->view                 = (int (*)(PetscObject,Viewer)) f;
  else                     (((void **)&mat->ops)[op])= f;
  return 0;
}

