
static char help[] = "Tests sequtial and parallel MatMatMult().\n\
Input arguments are:\n\
  -f0 <input_file> -f1 <input_file> -f2 <input_file> -f3 <input_file> : file to load\n\n";
/* ex94 -f0 $D/small -f1 $D/small -f2 $D/arco4 -f3 $D/arco4 */

#include "petscmat.h"

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc,char **args)
{
  Mat          A,B,P,C;
  Vec          x,y1,y2;
  PetscViewer  viewer;
  int          i,ierr,m,n,size,am,j,idxn[10];
  PetscReal    norm,norm_tmp,tol=1.e-10,none = -1.0,fill=4;
  PetscRandom  rand;
  char         file[4][128];
  PetscTruth   flg,preload = PETSC_TRUE;
  PetscScalar  a[10],rval;
  PetscTruth   Test_MatMatMult=PETSC_TRUE;

  PetscInitialize(&argc,&args,(char *)0,help);
  ierr = MPI_Comm_size(PETSC_COMM_WORLD,&size);CHKERRQ(ierr);

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
 
  /* Create vectors y1 and y2 that are compatible with A */
  ierr = VecCreate(PETSC_COMM_WORLD,&y1);CHKERRQ(ierr);
  ierr = MatGetLocalSize(A,&am,PETSC_NULL);CHKERRQ(ierr);
  ierr = VecSetSizes(y1,am,PETSC_DECIDE);CHKERRQ(ierr);
  ierr = VecSetFromOptions(y1);CHKERRQ(ierr);
  ierr = VecDuplicate(y1,&y2);CHKERRQ(ierr);

  ierr = PetscRandomCreate(PETSC_COMM_WORLD,RANDOM_DEFAULT,&rand);CHKERRQ(ierr);

  /* Test MatMatMult() */
  /*-------------------*/
  if (Test_MatMatMult){
  /* printf("... call  MatMatMult() \n"); */
  ierr = MatMatMult(A,B,MAT_INITIAL_MATRIX,fill,&C);CHKERRQ(ierr);
  
  /* Create vector x that is compatible with B */
  ierr = VecCreate(PETSC_COMM_WORLD,&x);CHKERRQ(ierr);
  ierr = MatGetLocalSize(B,&m,&n);CHKERRQ(ierr);
  ierr = VecSetSizes(x,n,PETSC_DECIDE);CHKERRQ(ierr);
  ierr = VecSetFromOptions(x);CHKERRQ(ierr);

  norm = 0.0;
  for (i=0; i<4; i++) {
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
  ierr = VecDestroy(x);CHKERRQ(ierr);
  } /* if (Test_MatMatMult) */

  /* Test MatMatMultTranspose() */

  /* Test MatSeqAIJPtAP() */
  /*----------------------*/
  if (size > 1) SETERRQ(1,"MatSeqAIJPtAP() is not writtern for size > 1 yet.");
  /* create P -- seq for the momnet */
 
  n = am/2;
  ierr = MatCreate(PETSC_COMM_SELF,am,n,PETSC_DECIDE,PETSC_DECIDE,&P);CHKERRQ(ierr); 
  ierr = MatSetType(P,MATSEQAIJ);CHKERRQ(ierr);
  ierr = MatSeqAIJSetPreallocation(P,10,PETSC_NULL);CHKERRQ(ierr);
  for (i=0; i<10; i++){
    ierr = PetscRandomGetValue(rand,&a[i]);CHKERRQ(ierr);
  }
  for (i=0; i<am; i++){
    for (j=0; j<10; j++){
      ierr = PetscRandomGetValue(rand,&rval);CHKERRQ(ierr);
      idxn[j] = (int)(PetscRealPart(rval)*n);
      /* printf("%d, j: %d, a: %g\n",i,idxn[j],a[j]); */
    }
    ierr = MatSetValues(P,1,&i,10,idxn,a,ADD_VALUES);CHKERRQ(ierr);
  }
  ierr = MatAssemblyBegin(P,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(P,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  printf(" P is assembled...\n");

  ierr = MatMatMult(A,P,MAT_INITIAL_MATRIX,fill,&C);CHKERRQ(ierr);
  ierr = MatDestroy(C);CHKERRQ(ierr);

  ierr = MatSeqAIJPtAP(A,P,&C);CHKERRQ(ierr);

  /* Create vector x that is compatible with P */
  ierr = VecCreate(PETSC_COMM_WORLD,&x);CHKERRQ(ierr);
  ierr = MatGetLocalSize(P,&m,&n);CHKERRQ(ierr);
  ierr = VecSetSizes(x,n,PETSC_DECIDE);CHKERRQ(ierr);
  ierr = VecSetFromOptions(x);CHKERRQ(ierr);
  
  Vec y3,y4;
  ierr = VecCreate(PETSC_COMM_WORLD,&y3);CHKERRQ(ierr);
  ierr = VecSetSizes(y3,n,PETSC_DECIDE);CHKERRQ(ierr);
  ierr = VecSetFromOptions(y3);CHKERRQ(ierr);
  ierr = VecDuplicate(y3,&y4);CHKERRQ(ierr);

  norm = 0.0;
  for (i=0; i<4; i++) {
    ierr = VecSetRandom(rand,x);CHKERRQ(ierr);
    ierr = MatMult(P,x,y1);CHKERRQ(ierr);  
    ierr = MatMult(A,y1,y2);CHKERRQ(ierr);  /* y2 = A*P*x */

    ierr = MatMultTranspose(P,y2,y3);CHKERRQ(ierr); /* y3 = Pt*A*P*x */
    ierr = MatMult(C,x,y4);CHKERRQ(ierr);   /* y3 = C*x   */
    ierr = VecAXPY(&none,y3,y4);CHKERRQ(ierr);
    ierr = VecNorm(y4,NORM_2,&norm_tmp);CHKERRQ(ierr);
    if (norm_tmp > norm) norm = norm_tmp;
  }
  tol = 0.0;
  if (norm > tol) {
    ierr = PetscPrintf(PETSC_COMM_SELF,"Error: MatSeqAIJPtAP(), |y1 - y2|: %g\n",norm);CHKERRQ(ierr);
  }
  

  ierr = MatDestroy(P);CHKERRQ(ierr);
  ierr = MatDestroy(C);CHKERRQ(ierr);
  ierr = VecDestroy(y3);CHKERRQ(ierr);
  ierr = VecDestroy(y4);CHKERRQ(ierr);

  /* Destroy objects */
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

