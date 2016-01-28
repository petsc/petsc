static char help[] = "Test MatPtAP,  MatMatMatMult\n\
Reads PETSc matrix A and P, then comput Pt*A*P \n\
Input parameters include\n\
  -fA <input_file> -fP <input_file>: second files to load (projection) \n\n";

#include <petscmat.h>

#undef WRITEFILE
#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc,char **args)
{
  Mat            A,P,C,R,RAP;
  PetscViewer    fd;
  char           file[2][PETSC_MAX_PATH_LEN];
  PetscBool      flg;
  PetscErrorCode ierr;
  PetscReal      fill=2.0,norm;

  PetscInitialize(&argc,&args,(char*)0,help);
#if defined(WRITEFILE)
  {
    PetscViewer viewer;
    ierr = PetscPrintf(PETSC_COMM_WORLD,"writing matrix A in binary to A.dat ...\n");CHKERRQ(ierr);
    ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,"A.dat",FILE_MODE_WRITE,&viewer);CHKERRQ(ierr);
    ierr = MatView(A,viewer);CHKERRQ(ierr);
    ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);

    ierr = PetscPrintf(PETSC_COMM_WORLD,"writing matrix P in binary to P.dat ...\n");CHKERRQ(ierr);
    ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,"P.dat",FILE_MODE_WRITE,&viewer);CHKERRQ(ierr);
    ierr = MatView(P,viewer);CHKERRQ(ierr);
    ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);
  }
#endif

  /* read the two matrices, A (square) and P (projection) */
  ierr = PetscOptionsGetString(NULL,NULL,"-fA",file[0],PETSC_MAX_PATH_LEN,&flg);CHKERRQ(ierr);
  if (!flg) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_USER,"Must indicate binary file with the -fA options");
  ierr = PetscOptionsGetString(NULL,NULL,"-fP",file[1],PETSC_MAX_PATH_LEN,&flg);CHKERRQ(ierr);
  if (!flg) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_USER,"Must indicate binary file with the -fP options");

  /* Load matrices */
  ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,file[0],FILE_MODE_READ,&fd);CHKERRQ(ierr);
  ierr = MatCreate(PETSC_COMM_WORLD,&A);CHKERRQ(ierr);
  ierr = MatLoad(A,fd);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&fd);CHKERRQ(ierr);
  /* ierr = MatGetSize(A,&m,&n);CHKERRQ(ierr); */

  ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,file[1],FILE_MODE_READ,&fd);CHKERRQ(ierr);
  ierr = MatCreate(PETSC_COMM_WORLD,&P);CHKERRQ(ierr);
  ierr = MatLoad(P,fd);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&fd);CHKERRQ(ierr);

  ierr = MatPtAP(A,P,MAT_INITIAL_MATRIX,fill,&C);CHKERRQ(ierr);
  ierr = MatPtAP(A,P,MAT_REUSE_MATRIX,fill,&C);CHKERRQ(ierr);
  /* ierr = MatView(C,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr); */

  /* Test PtAP = RAP */
  ierr = MatTranspose(P,MAT_INITIAL_MATRIX,&R);CHKERRQ(ierr);
  ierr = MatMatMatMult(R,A,P,MAT_INITIAL_MATRIX,2.0,&RAP);CHKERRQ(ierr);
  ierr = MatAXPY(RAP,-1.0,C,DIFFERENT_NONZERO_PATTERN);CHKERRQ(ierr);
 
  ierr = MatNorm(RAP,NORM_FROBENIUS,&norm);CHKERRQ(ierr);
  if (norm > 1.e-14) printf("norm(PtAP - RAP)= %g\n",norm);
 
  ierr = MatDestroy(&R);CHKERRQ(ierr);
  ierr = MatDestroy(&RAP);CHKERRQ(ierr);
  ierr = MatDestroy(&C);CHKERRQ(ierr);
  ierr = MatDestroy(&P);CHKERRQ(ierr);
  ierr = MatDestroy(&A);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return 0;
}
