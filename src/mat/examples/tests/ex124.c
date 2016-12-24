static char help[] = "Check the difference of the two matrices \n\
Reads PETSc matrix A and B, then check B=A-B \n\
Input parameters include\n\
  -fA <input_file> -fB <input_file> \n\n";

#include <petscmat.h>

#undef WRITEFILE
PetscInt main(PetscInt argc,char **args)
{
  Mat            A,B;
  PetscViewer    fd;
  char           file[2][PETSC_MAX_PATH_LEN];
  PetscBool      flg;
  PetscErrorCode ierr;
  PetscMPIInt    size;
  PetscInt       ma,na,mb,nb;

  ierr = PetscInitialize(&argc,&args,(char*)0,help);if (ierr) return ierr;
  ierr = MPI_Comm_size(PETSC_COMM_WORLD,&size);CHKERRQ(ierr);
  if (size != 1) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_SUP,"This is a uniprocessor example only!");

  /* read the two matrices, A and B */
  ierr = PetscOptionsGetString(NULL,NULL,"-fA",file[0],PETSC_MAX_PATH_LEN,&flg);CHKERRQ(ierr);
  if (!flg) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_USER,"Must indicate binary file with the -fA options");
  ierr = PetscOptionsGetString(NULL,NULL,"-fB",file[1],PETSC_MAX_PATH_LEN,&flg);CHKERRQ(ierr);
  if (!flg) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_USER,"Must indicate binary file with the -fP options");

  /* Load matrices */
  ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,file[0],FILE_MODE_READ,&fd);CHKERRQ(ierr);
  ierr = MatCreate(PETSC_COMM_WORLD,&A);CHKERRQ(ierr);
  ierr = MatLoad(A,fd);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&fd);CHKERRQ(ierr);
  printf("\n A:\n");
  printf("----------------------\n");
  ierr = MatView(A,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  ierr = MatGetSize(A,&ma,&na);CHKERRQ(ierr);

  ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,file[1],FILE_MODE_READ,&fd);CHKERRQ(ierr);
  ierr = MatCreate(PETSC_COMM_WORLD,&B);CHKERRQ(ierr);
  ierr = MatLoad(B,fd);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&fd);CHKERRQ(ierr);
  printf("\n B:\n");
  printf("----------------------\n");
  ierr = MatView(B,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  ierr = MatGetSize(B,&mb,&nb);CHKERRQ(ierr);

  /* Compute B = -A + B */
  if (ma != mb || na != nb) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_SIZ,"nonconforming matrix size");
  ierr = MatAXPY(B,-1.0,A,DIFFERENT_NONZERO_PATTERN);CHKERRQ(ierr);
  printf("\n B - A:\n");
  printf("----------------------\n");
  ierr = MatView(B,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);

  ierr = MatDestroy(&B);CHKERRQ(ierr);
  ierr = MatDestroy(&A);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
}
