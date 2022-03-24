
static char help[] = "Tests sequential and parallel MatGetRow() and MatRestoreRow().\n";

#include <petscmat.h>

int main(int argc,char **args)
{
  Mat               C;
  PetscInt          i,j,m = 5,n = 5,Ii,J,nz,rstart,rend;
  PetscMPIInt       rank;
  const PetscInt    *idx;
  PetscScalar       v;
  const PetscScalar *values;

  CHKERRQ(PetscInitialize(&argc,&args,(char*)0,help));
  /* Create the matrix for the five point stencil, YET AGAIN */
  CHKERRQ(MatCreate(PETSC_COMM_WORLD,&C));
  CHKERRQ(MatSetSizes(C,PETSC_DECIDE,PETSC_DECIDE,m*n,m*n));
  CHKERRQ(MatSetFromOptions(C));
  CHKERRQ(MatSetUp(C));
  for (i=0; i<m; i++) {
    for (j=0; j<n; j++) {
      v = -1.0;  Ii = j + n*i;
      if (i>0)   {J = Ii - n; CHKERRQ(MatSetValues(C,1,&Ii,1,&J,&v,INSERT_VALUES));}
      if (i<m-1) {J = Ii + n; CHKERRQ(MatSetValues(C,1,&Ii,1,&J,&v,INSERT_VALUES));}
      if (j>0)   {J = Ii - 1; CHKERRQ(MatSetValues(C,1,&Ii,1,&J,&v,INSERT_VALUES));}
      if (j<n-1) {J = Ii + 1; CHKERRQ(MatSetValues(C,1,&Ii,1,&J,&v,INSERT_VALUES));}
      v = 4.0; CHKERRQ(MatSetValues(C,1,&Ii,1,&Ii,&v,INSERT_VALUES));
    }
  }
  CHKERRQ(MatAssemblyBegin(C,MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatAssemblyEnd(C,MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatView(C,PETSC_VIEWER_STDOUT_WORLD));

  CHKERRMPI(MPI_Comm_rank(PETSC_COMM_WORLD,&rank));
  CHKERRQ(MatGetOwnershipRange(C,&rstart,&rend));
  for (i=rstart; i<rend; i++) {
    CHKERRQ(MatGetRow(C,i,&nz,&idx,&values));
    if (rank == 0) {
      for (j=0; j<nz; j++) {
        CHKERRQ(PetscPrintf(PETSC_COMM_SELF,"%" PetscInt_FMT " %g ",idx[j],(double)PetscRealPart(values[j])));
      }
      CHKERRQ(PetscPrintf(PETSC_COMM_SELF,"\n"));
    }
    CHKERRQ(MatRestoreRow(C,i,&nz,&idx,&values));
  }

  CHKERRQ(MatDestroy(&C));
  CHKERRQ(PetscFinalize());
  return 0;
}

/*TEST

   test:

TEST*/
