
static char help[] = "Tests sequtial and parallel MatMatMult().\n\
Input arguments are:\n\
  -f0 <input_file> -f1 <input_file> -f2 <input_file> -f3 <input_file> : file to load\n\n";
/* ex94 -f0 $D/small -f1 $D/small -f2 $D/arco6 -f3 $D/arco6 */

#include "petscmat.h"

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc,char **args)
{
  Mat          A,B,C;
  Vec          x,y1,y2;
  PetscViewer  viewer;
  int          i,ierr,m,n;
  PetscReal    norm,norm_tmp,tol=1.e-10,none = -1.0;
  PetscRandom  rand;
  char         file[4][128];
  PetscTruth   flg,preload = PETSC_TRUE;

  PetscInitialize(&argc,&args,(char *)0,help);

  /*  Load the matrices A and B */
  ierr = PetscOptionsGetString(PETSC_NULL,"-f0",file[0],127,&flg);CHKERRQ(ierr);
  if (!flg) SETERRQ(1,"Must indicate a file name for small matrix A with the -f0 option.");
  ierr = PetscOptionsGetString(PETSC_NULL,"-f1",file[1],127,&flg);CHKERRQ(ierr);
  if (!flg) SETERRQ(1,"Must indicate a file name for small matrix B with the -f1 option.");
  ierr = PetscOptionsGetString(PETSC_NULL,"-f2",file[2],127,&flg);CHKERRQ(ierr);
  if (!flg) {
    preload = PETSC_FALSE;
  } else {
    ierr = PetscOptionsGetString(PETSC_NULL,"-f3",file[3],127,&flg);CHKERRQ(ierr);
    if (!flg) SETERRQ(1,"Must indicate a file name for test matrix B with the -f3 option."); 
  }

  PreLoadBegin(preload,"Load system");
  /* printf("... load A \n"); */
  ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,file[2*PreLoadIt],PETSC_FILE_RDONLY,&viewer);CHKERRQ(ierr);
  ierr = MatLoad(viewer,MATAIJ,&A);CHKERRQ(ierr); 
  ierr = PetscViewerDestroy(viewer);CHKERRQ(ierr);

  /* printf("... load B \n"); */
  ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,file[2*PreLoadIt+1],PETSC_FILE_RDONLY,&viewer);CHKERRQ(ierr);
  ierr = MatLoad(viewer,MATAIJ,&B);CHKERRQ(ierr); 
  ierr = PetscViewerDestroy(viewer);CHKERRQ(ierr);
 
  /* Create vectors */
  ierr = VecCreate(PETSC_COMM_WORLD,&x);CHKERRQ(ierr);
  ierr = MatGetLocalSize(B,&m,&n);CHKERRQ(ierr);
  ierr = VecSetSizes(x,m,PETSC_DECIDE);CHKERRQ(ierr);
  ierr = VecSetFromOptions(x);CHKERRQ(ierr);

  ierr = VecCreate(PETSC_COMM_WORLD,&y1);CHKERRQ(ierr);
  ierr = MatGetLocalSize(A,&m,&n);CHKERRQ(ierr);
  ierr = VecSetSizes(y1,m,PETSC_DECIDE);CHKERRQ(ierr);
  ierr = VecSetFromOptions(y1);CHKERRQ(ierr);
  ierr = VecDuplicate(y1,&y2);CHKERRQ(ierr);

  /* Test MatMatMult */
  /* printf("... call  MatMatMult() \n"); */
  ierr = MatMatMult(A,B,&C);CHKERRQ(ierr);
  
  ierr = PetscRandomCreate(PETSC_COMM_WORLD,RANDOM_DEFAULT,&rand);CHKERRQ(ierr);
  norm = 0.0;
  int size;
  ierr = MPI_Comm_size(PETSC_COMM_WORLD,&size);CHKERRQ(ierr);
  if (size == 1){
  for (i=0; i<40; i++) {
    ierr = VecSetRandom(rand,x);CHKERRQ(ierr);
    ierr = MatMult(B,x,y1);CHKERRQ(ierr);  
    ierr = MatMult(A,y1,y2);CHKERRQ(ierr);  /* y2 = A*B*x */
    ierr = MatMult(C,x,y1);CHKERRQ(ierr);   /* y1 = C*x   */
    ierr = VecAXPY(&none,y2,y1);CHKERRQ(ierr);
    ierr = VecNorm(y1,NORM_2,&norm_tmp);CHKERRQ(ierr);
    if (norm_tmp > norm) norm = norm_tmp;
  }
  if (norm > tol) {
    ierr = PetscPrintf(PETSC_COMM_SELF,"Error: MatMatMult(), |y1 - y2|: %g\n",norm);CHKERRQ(ierr);
  }
  ierr = MatDestroy(C);CHKERRQ(ierr);
  }
  ierr = VecDestroy(x);CHKERRQ(ierr);
  ierr = VecDestroy(y1);CHKERRQ(ierr);
  ierr = VecDestroy(y2);CHKERRQ(ierr);
  ierr = PetscRandomDestroy(rand);CHKERRQ(ierr);
  
  ierr = MatDestroy(A);CHKERRQ(ierr);
  ierr = MatDestroy(B);CHKERRQ(ierr);

  PreLoadEnd();
  ierr = PetscFinalize();CHKERRQ(ierr);

  return 0;
}

