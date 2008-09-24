
static char help[] = "Tests the use of MatSolveTranspose().\n\n";

#include "petscmat.h"

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc,char **args)
{
  Mat            C,A;
  PetscErrorCode ierr;
  PetscMPIInt    size;
  IS             row,col;
  Vec            x,u,b;
  PetscReal      norm;
  PetscViewer    fd;
  char           type[256],file[PETSC_MAX_PATH_LEN];
  PetscScalar    mone = -1.0;
  PetscTruth     flg;

  PetscInitialize(&argc,&args,(char *)0,help);
  ierr = MPI_Comm_size(PETSC_COMM_WORLD,&size);CHKERRQ(ierr);
  if (size > 1) SETERRQ(1,"Can only run on one processor");

  ierr = PetscOptionsGetString(PETSC_NULL,"-f",file,PETSC_MAX_PATH_LEN-1,&flg);CHKERRQ(ierr);
  if (!flg) SETERRQ(1,"Must indicate binary file with the -f option");
  /* 
     Open binary file.  Note that we use FILE_MODE_READ to indicate
     reading from this file.
  */
  ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,file,FILE_MODE_READ,&fd);CHKERRQ(ierr);

  /* 
     Determine matrix format to be used (specified at runtime).
     See the manpage for MatLoad() for available formats.
  */
  ierr = PetscStrcpy(type,MATSEQAIJ);CHKERRQ(ierr);
  ierr = PetscOptionsGetString(PETSC_NULL,"-mat_type",type,256,PETSC_NULL);CHKERRQ(ierr);

  /*
     Load the matrix and vector; then destroy the viewer.
  */
  ierr = MatLoad(fd,type,&C);CHKERRQ(ierr);
  ierr = VecLoad(fd,PETSC_NULL,&u);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(fd);CHKERRQ(ierr);

  ierr = VecDuplicate(u,&x);CHKERRQ(ierr);
  ierr = VecDuplicate(u,&b);CHKERRQ(ierr);

  ierr = MatMultTranspose(C,u,b);CHKERRQ(ierr);

  /* Set default ordering to be Quotient Minimum Degree; also read
     orderings from the options database */
  ierr = MatGetOrdering(C,MATORDERING_QMD,&row,&col);CHKERRQ(ierr);

  ierr = MatGetFactor(C,MAT_SOLVER_PETSC,MAT_FACTOR_LU,&A);CHKERRQ(ierr);
  ierr = MatLUFactorSymbolic(A,C,row,col,PETSC_NULL);CHKERRQ(ierr);
  ierr = MatLUFactorNumeric(A,C,PETSC_NULL);CHKERRQ(ierr);
  ierr = MatSolveTranspose(A,b,x);CHKERRQ(ierr);

  ierr = VecAXPY(x,-1.0,u);CHKERRQ(ierr);
  ierr = VecNorm(x,NORM_2,&norm);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_SELF,"Norm of error %G\n",norm);CHKERRQ(ierr);

  ierr = ISDestroy(row);CHKERRQ(ierr);
  ierr = ISDestroy(col);CHKERRQ(ierr);
  ierr = VecDestroy(u);CHKERRQ(ierr);
  ierr = VecDestroy(x);CHKERRQ(ierr);
  ierr = VecDestroy(b);CHKERRQ(ierr);
  ierr = MatDestroy(C);CHKERRQ(ierr);
  ierr = MatDestroy(A);CHKERRQ(ierr);
  ierr = PetscFinalize();CHKERRQ(ierr);
  return 0;
}
