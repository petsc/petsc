
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
  FREE(shell); 
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
   MatShellCreate - creates a new matrix class for use with your 
          own private data storage format. This is intended to 
          provide a simple class to use with KSP. You should 
          not use this if you plan to make a complete class.

  Input Parameters:
.  m,n - number of rows and columns in matrix
.  ctx - pointer to your data needed by matrix multiply.

  Output Parameters:
.  mat - the matrix

  Keywords: matrix, shell

  Usage:
.             int (*mult)(void *,Vec,Vec);
.             MatShellCreate(m,n,ctx,&mat);
.             MatShellSetMult(mat,mult);

@*/
int MatShellCreate(int m, int n, void *ctx,Mat *mat)
{
  Mat      newmat;
  MatShell *shell;
  PETSCHEADERCREATE(newmat,_Mat,MAT_COOKIE,MATSHELL,MPI_COMM_WORLD);
  PLogObjectCreate(newmat);
  *mat           = newmat;
  newmat->factor = 0;
  newmat->row    = 0;
  newmat->col    = 0;
  newmat->destroy= MatShellDestroy;
  newmat->ops    = &MatOps;
  shell          = NEW(MatShell); CHKPTR(shell);
  newmat->data   = (void *) shell;
  shell->mult    = 0;
  shell->m       = m;
  shell->n       = n;
  shell->ctx     = ctx;
  return 0;
}

/*@
   MatShellSetMult - sets routine to use as matrix vector multiply.

  Input Parameters:
.  mat - the matrix to add the operation to, created with MatShellCreate()
.  mult - the matrix vector multiply routine.

  Keywords: matrix, multiply
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
   MatShellSetMultTransAdd - sets routine to use as matrix vector multiply.

  Input Parameters:
.  mat - the matrix to add the operation to, created with MatShellCreate()
.  mult - the matrix vector multiply routine.

  Keywords: matrix, multiply, transpose
@*/
int MatShellSetMultTransAdd(Mat mat, int (*mult)(void*,Vec,Vec,Vec))
{
  MatShell *shell;
  VALIDHEADER(mat,MAT_COOKIE);
  shell               = (MatShell *) mat->data;
  shell->multtransadd = mult;
  return 0;
}


