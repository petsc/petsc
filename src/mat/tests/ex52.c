
static char help[] = "Tests various routines in MatMPIBAIJ format.\n";

#include <petscmat.h>

int main(int argc,char **args)
{
  Mat            A;
  PetscInt       m=2,bs=1,M,row,col,start,end,i,j,k;
  PetscMPIInt    rank,size;
  PetscScalar    data=100;
  PetscBool      flg;

  CHKERRQ(PetscInitialize(&argc,&args,(char*)0,help));
  CHKERRMPI(MPI_Comm_rank(PETSC_COMM_WORLD,&rank));
  CHKERRMPI(MPI_Comm_size(PETSC_COMM_WORLD,&size));

  /* Test MatSetValues() and MatGetValues() */
  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-mat_block_size",&bs,NULL));
  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-mat_size",&m,NULL));

  M    = m*bs*size;
  CHKERRQ(MatCreateBAIJ(PETSC_COMM_WORLD,bs,PETSC_DECIDE,PETSC_DECIDE,M,M,PETSC_DECIDE,NULL,PETSC_DECIDE,NULL,&A));

  CHKERRQ(MatGetOwnershipRange(A,&start,&end));
  CHKERRQ(PetscOptionsHasName(NULL,NULL,"-column_oriented",&flg));
  if (flg) {
    CHKERRQ(MatSetOption(A,MAT_ROW_ORIENTED,PETSC_FALSE));
  }

  /* inproc assembly */
  for (row=start; row<end; row++) {
    for (col=start; col<end; col++,data+=1) {
      CHKERRQ(MatSetValues(A,1,&row,1,&col,&data,INSERT_VALUES));
    }
  }
  CHKERRQ(MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY));

  /* offproc assembly */
  data = 5.0;
  row  = (M+start-1)%M;
  for (col=0; col<M; col++) {
    CHKERRQ(MatSetValues(A,1,&row,1,&col,&data,ADD_VALUES));
  }
  CHKERRQ(MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY));

  /* Test MatSetValuesBlocked() */
  CHKERRQ(PetscOptionsHasName(NULL,NULL,"-test_setvaluesblocked",&flg));
  if (flg) {
    PetscScalar *bval;
    row /= bs;
    col  = start/bs;
    CHKERRQ(PetscMalloc1(bs*bs,&bval));
    k = 1;
    /* row oriented - default */
    for (i=0; i<bs; i++) {
      for (j=0; j<bs; j++) {
        bval[i*bs+j] = (PetscScalar)k; k++;
      }
    }
    CHKERRQ(MatSetValuesBlocked(A,1,&row,1,&col,bval,INSERT_VALUES));
    CHKERRQ(MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY));
    CHKERRQ(MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY));
    CHKERRQ(PetscFree(bval));
  }

  CHKERRQ(MatView(A,PETSC_VIEWER_STDOUT_WORLD));
  CHKERRQ(MatDestroy(&A));
  CHKERRQ(PetscFinalize());
  return 0;
}

/*TEST

   test:
      suffix: 1
      nsize: 3
      args: -mat_block_size 2 -test_setvaluesblocked

   test:
      suffix: 2
      nsize: 3
      args: -mat_block_size 2 -test_setvaluesblocked -column_oriented

   test:
      suffix: 3
      nsize: 3
      args: -mat_block_size 1 -test_setvaluesblocked

   test:
      suffix: 4
      nsize: 3
      args: -mat_block_size 1 -test_setvaluesblocked -column_oriented

TEST*/
