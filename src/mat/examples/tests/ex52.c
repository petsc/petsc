
static char help[] = "Tests the vatious routines in MatMPIBAIJ format.\n";


#include "petscmat.h"

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc,char **args)
{
  Mat          A;
  int          m=2,ierr,bs=1,M,row,col,rank,size,start,end;
  PetscScalar  data=100;

  PetscInitialize(&argc,&args,(char *)0,help);
  ierr = MPI_Comm_rank(PETSC_COMM_WORLD,&rank);CHKERRQ(ierr);
  ierr = MPI_Comm_size(PETSC_COMM_WORLD,&size);CHKERRQ(ierr);
  /* Test MatSetValues() and MatGetValues() */
  ierr = PetscOptionsGetInt(PETSC_NULL,"-mat_block_size",&bs,PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(PETSC_NULL,"-mat_size",&m,PETSC_NULL);CHKERRQ(ierr);

  M    = m*bs*size;
  ierr = MatCreateMPIBAIJ(PETSC_COMM_WORLD,bs,PETSC_DECIDE,PETSC_DECIDE,M,M,PETSC_DECIDE,PETSC_NULL,PETSC_DECIDE,PETSC_NULL,&A);CHKERRQ(ierr);

  ierr = MatGetOwnershipRange(A,&start,&end);CHKERRQ(ierr);
  
  for (row=start; row<end; row++) {
    for (col=start; col<end; col++,data+=1) {
      ierr = MatSetValues(A,1,&row,1,&col,&data,INSERT_VALUES);CHKERRQ(ierr);
    }
  }
  ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

  /* off proc assembly */
  data = 5.0;
  row = (M+start-1)%M;
  for (col=0; col<M; col++) {
    ierr = MatSetValues(A,1,&row,1,&col,&data,ADD_VALUES);CHKERRQ(ierr);
  } 
  ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

  ierr = MatView(A,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);

  ierr = MatDestroy(A);
  
  ierr = PetscFinalize();CHKERRQ(ierr);
  return 0;
}
 
