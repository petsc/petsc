/*$Id: ex62.c,v 1.11 1999/11/24 21:54:09 bsmith Exp bsmith $*/

static char help[] = "Tests the use of MatSolveTranspose().\n\n";

#include "mat.h"

#undef __FUNC__
#define __FUNC__ "main"
int main(int argc,char **args)
{
  Mat        C,A;
  int        i,j,m,ierr,size;
  IS         row,col;
  Vec        x,u,b;
  double     norm;
  Viewer     fd;
  MatType    mtype = MATSEQAIJ;
  char       file[128];
  Scalar     one = 1.0,mone = -1.0;
  PetscTruth flg;

  PetscInitialize(&argc,&args,(char *)0,help);
  ierr = MPI_Comm_size(PETSC_COMM_WORLD,&size);CHKERRA(ierr);
  if (size > 1) SETERRA(1,1,"Can only run on one processor");

  ierr = OptionsGetString(PETSC_NULL,"-f",file,127,&flg);CHKERRA(ierr);
  if (!flg) SETERRA(1,0,"Must indicate binary file with the -f option");
  /* 
     Open binary file.  Note that we use BINARY_RDONLY to indicate
     reading from this file.
  */
  ierr = ViewerBinaryOpen(PETSC_COMM_WORLD,file,BINARY_RDONLY,&fd);CHKERRA(ierr);

  /* 
     Determine matrix format to be used (specified at runtime).
     See the manpage for MatLoad() for available formats.
  */
  ierr = MatGetTypeFromOptions(PETSC_COMM_WORLD,0,&mtype,PETSC_NULL);CHKERRQ(ierr);

  /*
     Load the matrix and vector; then destroy the viewer.
  */
  ierr = MatLoad(fd,mtype,&C);CHKERRA(ierr);
  ierr = VecLoad(fd,&u);CHKERRA(ierr);
  ierr = ViewerDestroy(fd);CHKERRA(ierr);

  ierr = VecDuplicate(u,&x);CHKERRA(ierr);
  ierr = VecDuplicate(u,&b);CHKERRA(ierr);

  ierr = MatMultTranspose(C,u,b);CHKERRA(ierr);

  /* Set default ordering to be Quotient Minimum Degree; also read
     orderings from the options database */
  ierr = MatGetOrdering(C,MATORDERING_QMD,&row,&col);CHKERRA(ierr);

  ierr = MatLUFactorSymbolic(C,row,col,1.0,&A);CHKERRA(ierr);
  ierr = MatLUFactorNumeric(C,&A);CHKERRA(ierr);
  ierr = MatSolveTranspose(A,b,x);CHKERRA(ierr);

  ierr = VecAXPY(&mone,u,x);CHKERRA(ierr);
  ierr = VecNorm(x,NORM_2,&norm);CHKERRA(ierr);
  ierr = PetscPrintf(PETSC_COMM_SELF,"Norm of error %g\n",norm);CHKERRA(ierr);

  ierr = ISDestroy(row);CHKERRA(ierr);
  ierr = ISDestroy(col);CHKERRA(ierr);
  ierr = VecDestroy(u);CHKERRA(ierr);
  ierr = VecDestroy(x);CHKERRA(ierr);
  ierr = VecDestroy(b);CHKERRA(ierr);
  ierr = MatDestroy(C);CHKERRA(ierr);
  ierr = MatDestroy(A);CHKERRA(ierr);
  PetscFinalize();
  return 0;
}
