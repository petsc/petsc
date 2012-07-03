
static char help[] = "Tests sequential and parallel MatGetRow() and MatRestoreRow().\n";

#include <petscmat.h>

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc,char **args)
{
  Mat               C; 
  PetscInt          i,j,m = 5,n = 5,Ii,J,nz,rstart,rend;
  PetscErrorCode    ierr;
  PetscMPIInt       rank;
  const PetscInt    *idx;
  PetscScalar       v;
  const PetscScalar *values;

  PetscInitialize(&argc,&args,(char *)0,help);

  /* Create the matrix for the five point stencil, YET AGAIN */
  ierr = MatCreate(PETSC_COMM_WORLD,&C);CHKERRQ(ierr); 
  ierr = MatSetSizes(C,PETSC_DECIDE,PETSC_DECIDE,m*n,m*n);CHKERRQ(ierr); 
  ierr = MatSetFromOptions(C);CHKERRQ(ierr);
  ierr = MatSetUp(C);CHKERRQ(ierr);
  for (i=0; i<m; i++) {
    for (j=0; j<n; j++) {
      v = -1.0;  Ii = j + n*i;
      if (i>0)   {J = Ii - n; ierr = MatSetValues(C,1,&Ii,1,&J,&v,INSERT_VALUES);CHKERRQ(ierr);}
      if (i<m-1) {J = Ii + n; ierr = MatSetValues(C,1,&Ii,1,&J,&v,INSERT_VALUES);CHKERRQ(ierr);}
      if (j>0)   {J = Ii - 1; ierr = MatSetValues(C,1,&Ii,1,&J,&v,INSERT_VALUES);CHKERRQ(ierr);}
      if (j<n-1) {J = Ii + 1; ierr = MatSetValues(C,1,&Ii,1,&J,&v,INSERT_VALUES);CHKERRQ(ierr);}
      v = 4.0; ierr = MatSetValues(C,1,&Ii,1,&Ii,&v,INSERT_VALUES);CHKERRQ(ierr);
    }
  }
  ierr = MatAssemblyBegin(C,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(C,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatView(C,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr); 

  ierr = MPI_Comm_rank(PETSC_COMM_WORLD,&rank);CHKERRQ(ierr);
  ierr = MatGetOwnershipRange(C,&rstart,&rend);CHKERRQ(ierr);
  for (i=rstart; i<rend; i++){
    ierr = MatGetRow(C,i,&nz,&idx,&values);CHKERRQ(ierr);
    if (!rank){
#if defined(PETSC_USE_COMPLEX)
      for (j=0; j<nz; j++) {
        ierr = PetscPrintf(PETSC_COMM_SELF,"%D %G ",idx[j],PetscRealPart(values[j]));CHKERRQ(ierr);
      }
#else
      for (j=0; j<nz; j++) {
        ierr = PetscPrintf(PETSC_COMM_SELF,"%D %G ",idx[j],values[j]);CHKERRQ(ierr);}
#endif
      ierr = PetscPrintf(PETSC_COMM_SELF,"\n");CHKERRQ(ierr);
    }
    ierr = MatRestoreRow(C,i,&nz,&idx,&values);CHKERRQ(ierr);
  }

  ierr = MatDestroy(&C);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return 0;
}
