
static char help[] = "Tests MatTranspose() and MatEqual() for MPIAIJ matrices.\n\n";

#include <petscmat.h>

int main(int argc,char **argv)
{
  Mat            A,B;
  PetscInt       m = 7,n,i,rstart,rend,cols[3];
  PetscScalar    v[3];
  PetscBool      equal;
  const char     *eq[2];

  CHKERRQ(PetscInitialize(&argc,&argv,(char*)0,help));
  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-m",&m,NULL));
  n    = m;

  /* ------- Assemble matrix, --------- */

  CHKERRQ(MatCreateAIJ(PETSC_COMM_WORLD,PETSC_DECIDE,PETSC_DECIDE,m,n,0,0,0,0,&A));
  CHKERRQ(MatSetOption(A,MAT_NEW_NONZERO_LOCATION_ERR,PETSC_FALSE));
  CHKERRQ(MatGetOwnershipRange(A,&rstart,&rend));
  if (!rstart) {
    cols[0] = 0;
    cols[1] = 1;
    v[0]    = 2.0; v[1] = -1.0;
    CHKERRQ(MatSetValues(A,1,&rstart,2,cols,v,INSERT_VALUES));
    rstart++;
  }
  if (rend == m) {
    rend--;
    cols[0] = rend-1;
    cols[1] = rend;
    v[0]    = -1.0; v[1] = 2.0;
    CHKERRQ(MatSetValues(A,1,&rend,2,cols,v,INSERT_VALUES));
  }
  v[0] = -1.0; v[1] = 2.0; v[2] = -1.0;
  for (i=rstart; i<rend; i++) {
    cols[0] = i-1;
    cols[1] = i;
    cols[2] = i+1;
    CHKERRQ(MatSetValues(A,1,&i,3,cols,v,INSERT_VALUES));
  }
  CHKERRQ(MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY));

  CHKERRQ(MatTranspose(A,MAT_INITIAL_MATRIX,&B));

  CHKERRQ(MatEqual(A,B,&equal));

  eq[0] = "not equal";
  eq[1] = "equal";
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"Matrices are %s\n",eq[equal]));

  CHKERRQ(MatTranspose(A,MAT_REUSE_MATRIX,&B));
  CHKERRQ(MatEqual(A,B,&equal));
  if (!equal) CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"MatTranspose with MAT_REUSE_MATRIX failed"));

  /* Free data structures */
  CHKERRQ(MatDestroy(&A));
  CHKERRQ(MatDestroy(&B));

  CHKERRQ(PetscFinalize());
  return 0;
}

/*TEST

    test:

    test:
      suffix: 2
      nsize: 2
      output_file: output/ex58_1.out

TEST*/
