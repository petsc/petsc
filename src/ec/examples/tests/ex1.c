
#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: ex1.c,v 1.2 1997/08/27 19:08:26 curfman Exp curfman $";
#endif

static char help[] = 
"Reads a PETSc matrixfrom a file and computes its eigenvalues.\n\\n";

/*T
   Concepts: EC^Computing eigenvalues - loading a binary matrix;
   Routines: ECCreate(); ECSetOperators(); ECSetFromOptions();
   Routines: ECSolve(); ECSetUp(); ECView();
   Routines: MatLoad(); VecLoad();
   Routines: ViewerFileOpenBinary(); ViewerDestroy();
   Processors: n
T*/

/* 
  Include "ec.h" so that we can use the eigenvalue routines.  Note that this file
  automatically includes:
     petsc.h  - base PETSc routines   vec.h    - vectors
     sys.h    - system routines       mat.h    - matrices
     is.h     - index sets            viewer.h - viewers 
*/
#include "ec.h"
#include <stdio.h>

int main(int argc,char **args)
{
  EC         ec;               /* eigenvalue computation context */
  MatType    mtype;            /* matrix format */
  Mat        A;                /* matrix */
  Viewer     fd;               /* viewer */
  char       file[128];     /* input file name */
  int        ierr, flg;
  PetscTruth set;

  PetscInitialize(&argc,&args,(char *)0,help);

  /* 
     Determine files from which we read the matrix
     (matrix and right-hand-side vector).
  */
  ierr = OptionsGetString(PETSC_NULL,"-f",file,127,&flg); CHKERRA(ierr);
  if (!flg) SETERRA(1,0,"Must indicate binary file containing a matrix with the -f option");


  /* - - - - - - - - - - - New Stage - - - - - - - - - - - - -
                           Load matrix
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  /* 
     Open binary file.  Note that we use BINARY_RDONLY to indicate
     reading from this file.
  */
  ierr = ViewerFileOpenBinary(MPI_COMM_WORLD,file,BINARY_RDONLY,&fd);CHKERRA(ierr);

  /* 
     Determine matrix format to be used (specified at runtime).
     See the manpage for MatLoad() for available formats.
  */
  ierr = MatGetTypeFromOptions(MPI_COMM_WORLD,0,&mtype,&set); CHKERRQ(ierr);

  /*
     Load the matrix; then destroy the viewer.
  */
  ierr = MatLoad(fd,mtype,&A); CHKERRA(ierr);
  ierr = ViewerDestroy(fd); CHKERRA(ierr);

  /*
       Create eigenvalue computer; set operators; set runtime options.
  */
  ierr = ECCreate(MPI_COMM_WORLD,EC_EIGENVALUE,&ec); CHKERRA(ierr);
  ierr = ECSetOperators(ec,A,PETSC_NULL); CHKERRA(ierr);
  ierr = ECSetFromOptions(ec); CHKERRA(ierr);

  /* 
       Here we explicitly call ECSetUp(); if you do not call it,
    it will automatically be called in ECSolve().
  */
  ierr = ECSetUp(ec); CHKERRA(ierr);

  /*
       Compute the eigenvalues
  */
  ierr = ECSolve(ec); CHKERRA(ierr);

  ierr = MatDestroy(A); CHKERRA(ierr); 
  ierr = ECDestroy(ec); CHKERRA(ierr); 

  PetscFinalize();
  return 0;
}

