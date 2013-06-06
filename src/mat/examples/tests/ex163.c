
static char help[] = "Tests MatTransposeMatMult() on MatLoad() matrix \n\n";

#include <petscmat.h>

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc,char **args)
{
  Mat            A,C,Bdense;
  PetscErrorCode ierr;
  PetscViewer    fd;              /* viewer */
  char           file[PETSC_MAX_PATH_LEN]; /* input file name */
  PetscBool      flg,viewmats=PETSC_FALSE;
  PetscMPIInt    rank;
  PetscReal      fill=1.0;
  PetscInt       m,n,i,j,BN=10;

  PetscInitialize(&argc,&args,(char*)0,help);
  ierr = MPI_Comm_rank(PETSC_COMM_WORLD,&rank);CHKERRQ(ierr);

  /* Determine file from which we read the matrix A */
  ierr = PetscOptionsGetString(NULL,"-f",file,PETSC_MAX_PATH_LEN,&flg);CHKERRQ(ierr);
  if (!flg) SETERRQ(PETSC_COMM_WORLD,1,"Must indicate binary file with the -f option");

  /* Load matrix A */
  ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,file,FILE_MODE_READ,&fd);CHKERRQ(ierr);
  ierr = MatCreate(PETSC_COMM_WORLD,&A);CHKERRQ(ierr);
  ierr = MatLoad(A,fd);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&fd);CHKERRQ(ierr);

  /* Print (for testing only) */
  ierr = PetscOptionsHasName(NULL, "-view_mats", &viewmats);CHKERRQ(ierr);
  if (viewmats) {
    if (!rank) printf("A_aij:\n");
    ierr = MatView(A,0);CHKERRQ(ierr);
  }

  /* Test MatTransposeMatMult_aij_aij() */
  ierr = MatTransposeMatMult(A,A,MAT_INITIAL_MATRIX,fill,&C);CHKERRQ(ierr);
  if (viewmats) {
    if (!rank) printf("\nC = A_aij^T * A_aij:\n");
    ierr = MatView(C,0);CHKERRQ(ierr);
  }
  ierr = MatDestroy(&C);CHKERRQ(ierr);
  ierr = MatGetLocalSize(A,&m,&n);CHKERRQ(ierr);
  
  /* create a dense matrix Bdense */
  PetscInt rstart,rend;
  ierr = MatCreate(PETSC_COMM_WORLD,&Bdense);CHKERRQ(ierr);
  ierr = MatSetSizes(Bdense,m,BN,PETSC_DECIDE,PETSC_DECIDE);CHKERRQ(ierr);
  ierr = MatSetType(Bdense,MATDENSE);CHKERRQ(ierr);
  ierr = MatSetFromOptions(Bdense);CHKERRQ(ierr);
  ierr = MatSetUp(Bdense);CHKERRQ(ierr);
  ierr = MatGetOwnershipRange(Bdense,&rstart,&rend);CHKERRQ(ierr);
  printf("[%d] rstart/end %d %d\n",rank,rstart,rend);

  PetscRandom    rand;
  PetscScalar    array[m*BN],rval;
  PetscInt       rows[m],cols[BN];

  for (i=0; i<m; i++) rows[i] = rstart + i;
  ierr = PetscRandomCreate(PETSC_COMM_WORLD,&rand);CHKERRQ(ierr);
  ierr = PetscRandomSetFromOptions(rand);CHKERRQ(ierr);
  for (j=0; j<BN; j++) {
    cols[j] = j;
    for (i=0; i<m; i++) {
      ierr = PetscRandomGetValue(rand,&rval);CHKERRQ(ierr);
      array[m*j+i] = rval;
    }
  }
  ierr = MatSetValues(Bdense,m,rows,BN,cols,array,INSERT_VALUES);CHKERRQ(ierr);
  ierr = MatAssemblyBegin(Bdense,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(Bdense,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = PetscRandomDestroy(&rand);CHKERRQ(ierr);
  if (viewmats) {
    if (!rank) printf("\nBdense:\n");
    ierr = MatView(Bdense,0);CHKERRQ(ierr);
  }

  /* Test MatTransposeMatMult_aij_dense() */
  ierr = MatTransposeMatMult(A,Bdense,MAT_INITIAL_MATRIX,fill,&C);CHKERRQ(ierr);

  /* Free data structures */
  ierr = MatDestroy(&A);CHKERRQ(ierr);
  ierr = MatDestroy(&C);CHKERRQ(ierr);
  ierr = MatDestroy(&Bdense);CHKERRQ(ierr);

  ierr = PetscFinalize();
  return 0;
}
