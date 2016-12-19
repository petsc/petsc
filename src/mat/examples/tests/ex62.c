
static char help[] = "Tests the use of MatSolveTranspose().\n\n";

#include <petscmat.h>

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
  PetscBool      flg;

  ierr = PetscInitialize(&argc,&args,(char*)0,help);if (ierr) return ierr;
  ierr = MPI_Comm_size(PETSC_COMM_WORLD,&size);CHKERRQ(ierr);
  if (size > 1) SETERRQ(PETSC_COMM_WORLD,1,"Can only run on one processor");

  ierr = PetscOptionsGetString(NULL,NULL,"-f",file,PETSC_MAX_PATH_LEN,&flg);CHKERRQ(ierr);
  if (!flg) SETERRQ(PETSC_COMM_WORLD,1,"Must indicate binary file with the -f option");
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
  ierr = PetscOptionsGetString(NULL,NULL,"-mat_type",type,256,NULL);CHKERRQ(ierr);

  /*
     Load the matrix and vector; then destroy the viewer.
  */
  ierr = MatCreate(PETSC_COMM_WORLD,&C);CHKERRQ(ierr);
  ierr = MatSetType(C,type);CHKERRQ(ierr);
  ierr = MatLoad(C,fd);CHKERRQ(ierr);
  ierr = VecCreate(PETSC_COMM_WORLD,&u);CHKERRQ(ierr);
  ierr = VecLoad(u,fd);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&fd);CHKERRQ(ierr);

  ierr = VecDuplicate(u,&x);CHKERRQ(ierr);
  ierr = VecDuplicate(u,&b);CHKERRQ(ierr);

  ierr = MatMultTranspose(C,u,b);CHKERRQ(ierr);

  /* Set default ordering to be Quotient Minimum Degree; also read
     orderings from the options database */
  ierr = MatGetOrdering(C,MATORDERINGQMD,&row,&col);CHKERRQ(ierr);

  ierr = MatGetFactor(C,MATSOLVERPETSC,MAT_FACTOR_LU,&A);CHKERRQ(ierr);
  ierr = MatLUFactorSymbolic(A,C,row,col,NULL);CHKERRQ(ierr);
  ierr = MatLUFactorNumeric(A,C,NULL);CHKERRQ(ierr);
  ierr = MatSolveTranspose(A,b,x);CHKERRQ(ierr);

  ierr = VecAXPY(x,-1.0,u);CHKERRQ(ierr);
  ierr = VecNorm(x,NORM_2,&norm);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_SELF,"Norm of error %g\n",(double)norm);CHKERRQ(ierr);

  ierr = ISDestroy(&row);CHKERRQ(ierr);
  ierr = ISDestroy(&col);CHKERRQ(ierr);
  ierr = VecDestroy(&u);CHKERRQ(ierr);
  ierr = VecDestroy(&x);CHKERRQ(ierr);
  ierr = VecDestroy(&b);CHKERRQ(ierr);
  ierr = MatDestroy(&C);CHKERRQ(ierr);
  ierr = MatDestroy(&A);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
}
