#ifndef lint
static char vcid[] = "$Id: shell.c,v 1.9 1995/05/11 03:46:44 curfman Exp bsmith $";
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
  int  m,n;
  int  (*mult)(void *,Vec,Vec);
  int  (*multtransadd)(void*,Vec,Vec,Vec);
  void *ctx;
} MatShell;      

static int MatShellMult(Mat mat,Vec x,Vec y)
{
  MatShell *shell;
  shell = (MatShell *) mat->data;
  return (*shell->mult)(shell->ctx,x,y);
}
static int MatShellMultTransAdd(Mat mat,Vec x,Vec y,Vec z)
{
  MatShell *shell;
  shell = (MatShell *) mat->data;
  return (*shell->multtransadd)(shell->ctx,x,y,z);
}
static int MatShellDestroy(PetscObject obj)
{
  Mat      mat = (Mat) obj;
  MatShell *shell;
  shell = (MatShell *) mat->data;
  PETSCFREE(shell); 
  PLogObjectDestroy(mat);
  PETSCHEADERDESTROY(mat);
  return 0;
}
  
static struct _MatOps MatOps = {0,0,
       0, 
       MatShellMult,0,0,MatShellMultTransAdd,
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
       0,0 };

/*@
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
int MatShellCreate(MPI_Comm comm,int m, int n, void *ctx,Mat *mat)
{
  Mat      newmat;
  MatShell *shell;
  PETSCHEADERCREATE(newmat,_Mat,MAT_COOKIE,MATSHELL,comm);
  PLogObjectCreate(newmat);
  *mat           = newmat;
  newmat->factor = 0;
  newmat->destroy= MatShellDestroy;
  newmat->ops    = &MatOps;
  shell          = PETSCNEW(MatShell); CHKPTRQ(shell);
  newmat->data   = (void *) shell;
  shell->mult    = 0;
  shell->m       = m;
  shell->n       = n;
  shell->ctx     = ctx;
  return 0;
}

/*@
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
int MatShellSetMult(Mat mat, int (*mult)(void*,Vec,Vec))
{
  MatShell *shell;
  VALIDHEADER(mat,MAT_COOKIE);
  shell = (MatShell *) mat->data;
  shell->mult = mult;
  return 0;
}
/*@
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
int MatShellSetMultTransAdd(Mat mat, int (*mult)(void*,Vec,Vec,Vec))
{
  MatShell *shell;
  VALIDHEADER(mat,MAT_COOKIE);
  shell               = (MatShell *) mat->data;
  shell->multtransadd = mult;
  return 0;
}


