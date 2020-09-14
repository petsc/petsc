
static char help[] = "Tests MatGetRowMax(), MatGetRowMin(), MatGetRowMaxAbs()\n";

#include <petscmat.h>

#define M 5
#define N 6

int main(int argc,char **args)
{
  Mat            A;
  Vec            min,max,maxabs,e;
  PetscInt       m,n,j,col,cols[N];
  PetscInt       imin[M],imax[M],imaxabs[M],indices[N],row,testcase=0;
  PetscScalar    values[N];
  PetscErrorCode ierr;
  MatType        type;
  PetscMPIInt    size,rank;
  PetscBool      doTest=PETSC_TRUE;
  PetscReal      enorm;

  ierr = PetscInitialize(&argc,&args,(char*)0,help);if (ierr) return ierr;
  ierr = MPI_Comm_size(PETSC_COMM_WORLD,&size);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(PETSC_COMM_WORLD,&rank);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(NULL,NULL,"-testcase",&testcase,NULL);CHKERRQ(ierr);

  ierr = MatCreate(PETSC_COMM_WORLD,&A);CHKERRQ(ierr);
  if (testcase == 1) { /* proc[0] holds entire A and other processes have no entry */
    if (!rank) {
      ierr = MatSetSizes(A,M,N,PETSC_DECIDE,PETSC_DECIDE);CHKERRQ(ierr);
    } else {
      ierr = MatSetSizes(A,0,0,PETSC_DECIDE,PETSC_DECIDE);CHKERRQ(ierr);
    }
    testcase = 0;
  } else {
    ierr = MatSetSizes(A,PETSC_DECIDE,PETSC_DECIDE,M,N);CHKERRQ(ierr);
  }
  ierr = MatSetFromOptions(A);CHKERRQ(ierr);
  ierr = MatSetUp(A);CHKERRQ(ierr);

  if (!rank) { /* proc[0] sets matrix A */
    for (j=0; j<N; j++) indices[j] = j;
    switch (testcase) {
    case 1: /* see testcast 0 */
      break;
    case 2:
      row = 0;
      values[0]  = -2.0; values[1] = -2.0; values[2] = -2.0; values[3] = -4.0; values[4] = 0.0; values[5] = 1.0;
      ierr = MatSetValues(A,1,&row,N,indices,values,INSERT_VALUES);CHKERRQ(ierr);
      row = 2;
      indices[0] = 0;    indices[1] = 3;    indices[2] = 5;
      values[0]  = -2.0; values[1]  = -2.0; values[2]  = -2.0;
      ierr = MatSetValues(A,1,&row,3,indices,values,INSERT_VALUES);CHKERRQ(ierr);
      row = 3;
      indices[0] = 0;    indices[1] = 1;    indices[2] = 4;
      values[0]  = -2.0; values[1]  = -2.0; values[2]  = -2.0;
      ierr = MatSetValues(A,1,&row,3,indices,values,INSERT_VALUES);CHKERRQ(ierr);
      row = 4;
      indices[0] = 0;    indices[1] = 1;    indices[2] = 2;
      values[0]  = -2.0; values[1]  = -2.0; values[2]  = -2.0;
      ierr = MatSetValues(A,1,&row,3,indices,values,INSERT_VALUES);CHKERRQ(ierr);
      break;
    case 3:
      row = 0;
      values[0]  = -2.0; values[1] = -2.0; values[2] = -2.0;
      ierr = MatSetValues(A,1,&row,3,indices+1,values,INSERT_VALUES);CHKERRQ(ierr);
      row = 1;
      values[0]  = -2.0; values[1] = -2.0; values[2] = -2.0;
      ierr = MatSetValues(A,1,&row,3,indices,values,INSERT_VALUES);CHKERRQ(ierr);
      row = 2;
      values[0]  = -2.0; values[1] = -2.0; values[2]  = -2.0;
      ierr = MatSetValues(A,1,&row,3,indices,values,INSERT_VALUES);CHKERRQ(ierr);
      row = 3;
      values[0]  = -2.0; values[1] = -2.0; values[2]  = -2.0; values[3] = -1.0;
      ierr = MatSetValues(A,1,&row,4,indices,values,INSERT_VALUES);CHKERRQ(ierr);
      row = 4;
      values[0]  = -2.0; values[1] = -2.0; values[2]  = -2.0; values[3] = -1.0;
      ierr = MatSetValues(A,1,&row,4,indices,values,INSERT_VALUES);CHKERRQ(ierr);
      break;

    default:
      row  = 0;
      values[0]  = -1.0; values[1] = 0.0; values[2] = 1.0; values[3] = 3.0; values[4] = 4.0; values[5] = -5.0;
      ierr = MatSetValues(A,1,&row,N,indices,values,INSERT_VALUES);CHKERRQ(ierr);
      row  = 1;
      ierr = MatSetValues(A,1,&row,3,indices,values,INSERT_VALUES);CHKERRQ(ierr);
      row  = 3;
      ierr = MatSetValues(A,1,&row,1,indices+4,values+4,INSERT_VALUES);CHKERRQ(ierr);
      row  = 4;
      ierr = MatSetValues(A,1,&row,2,indices+4,values+4,INSERT_VALUES);CHKERRQ(ierr);
    }
  }
  ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatView(A,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);

  ierr = MatGetLocalSize(A, &m,&n);CHKERRQ(ierr);
  ierr = VecCreate(PETSC_COMM_WORLD,&min);CHKERRQ(ierr);
  ierr = VecSetSizes(min,m,PETSC_DECIDE);CHKERRQ(ierr);
  ierr = VecSetFromOptions(min);CHKERRQ(ierr);
  ierr = VecDuplicate(min,&max);CHKERRQ(ierr);
  ierr = VecDuplicate(min,&maxabs);CHKERRQ(ierr);
  ierr = VecDuplicate(min,&e);CHKERRQ(ierr);

  /* Test MatGetRowMax() */
  ierr = PetscPrintf(PETSC_COMM_WORLD,"\n Row Maximums\n");CHKERRQ(ierr);
  ierr = MatGetRowMax(A,max,NULL);CHKERRQ(ierr);
  ierr = MatGetRowMax(A,max,imax);CHKERRQ(ierr);
  ierr = VecView(max,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  ierr = VecGetLocalSize(max,&n);CHKERRQ(ierr);
  ierr = PetscIntView(n,imax,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);

  /* Test MatGetRowMin() */
  ierr = MatScale(A,-1.0);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"\n Row Minimums\n");CHKERRQ(ierr);
  ierr = MatGetRowMin(A,min,NULL);CHKERRQ(ierr);
  ierr = MatGetRowMin(A,min,imin);CHKERRQ(ierr);

  ierr = VecWAXPY(e,1.0,max,min);CHKERRQ(ierr); /* e = max + min */
  ierr = VecNorm(e,NORM_INFINITY,&enorm);CHKERRQ(ierr);
  if (enorm > PETSC_MACHINE_EPSILON) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_PLIB,"max+min > PETSC_MACHINE_EPSILON ");
  for (j = 0; j < n; j++) if (imin[j] != imax[j]) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_PLIB,"imin != imax");

  exit(1);

  /* Test MatGetRowMin, MatGetRowMax and MatGetRowMaxAbs */
  if (size == 1) {
    ierr = MatGetRowMin(A,min,imin);CHKERRQ(ierr);
    ierr = MatGetRowMax(A,max,imax);CHKERRQ(ierr);
    ierr = MatGetRowMaxAbs(A,maxabs,imaxabs);CHKERRQ(ierr);

    ierr = PetscPrintf(PETSC_COMM_WORLD,"Row Minimums\n");CHKERRQ(ierr);
    ierr = VecView(min,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
    ierr = PetscIntView(5,imin,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Row Maximums\n");CHKERRQ(ierr);
    ierr = VecView(max,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
    ierr = PetscIntView(5,imax,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Row Maximum Absolute Values\n");CHKERRQ(ierr);
    ierr = VecView(maxabs,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
    ierr = PetscIntView(5,imaxabs,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);

  } else {
    ierr = MatGetType(A,&type);CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_WORLD,"\nMatrix type: %s\n",type);CHKERRQ(ierr);
    /* AIJ */
    ierr = PetscObjectTypeCompare((PetscObject)A,MATMPIAIJ,&doTest);CHKERRQ(ierr);
    if (doTest) {
      ierr = MatGetRowMaxAbs(A,maxabs,NULL);CHKERRQ(ierr);
      ierr = MatGetRowMaxAbs(A,maxabs,imaxabs);CHKERRQ(ierr);
      ierr = PetscPrintf(PETSC_COMM_WORLD,"Row Maximum Absolute Values:\n");CHKERRQ(ierr);
      ierr = VecView(maxabs,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
    }
    /* BAIJ */
    ierr = PetscObjectTypeCompare((PetscObject)A,MATMPIBAIJ,&doTest);CHKERRQ(ierr);
    if (doTest) {
      ierr = MatGetRowMaxAbs(A,maxabs,NULL);CHKERRQ(ierr);
      ierr = MatGetRowMaxAbs(A,maxabs,imaxabs);CHKERRQ(ierr);
      ierr = PetscPrintf(PETSC_COMM_WORLD,"Row Maximum Absolute Values:\n");CHKERRQ(ierr);
      ierr = VecView(maxabs,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
    }
  }

  if (size == 1) {
    ierr = MatConvert(A,MATDENSE,MAT_INPLACE_MATRIX,&A);CHKERRQ(ierr);

    ierr = MatGetRowMin(A,min,imin);CHKERRQ(ierr);
    ierr = MatGetRowMax(A,max,imax);CHKERRQ(ierr);
    ierr = MatGetRowMaxAbs(A,maxabs,imaxabs);CHKERRQ(ierr);

    ierr = MatView(A,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Row Minimums\n");CHKERRQ(ierr);
    ierr = VecView(min,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
    ierr = PetscIntView(5,imin,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Row Maximums\n");CHKERRQ(ierr);
    ierr = VecView(max,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
    ierr = PetscIntView(5,imax,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Row Maximum Absolute Values\n");CHKERRQ(ierr);
    ierr = VecView(maxabs,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
    ierr = PetscIntView(5,imaxabs,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  }

  ierr = VecDestroy(&min);CHKERRQ(ierr);
  ierr = VecDestroy(&max);CHKERRQ(ierr);
  ierr = VecDestroy(&maxabs);CHKERRQ(ierr);
  ierr = VecDestroy(&e);CHKERRQ(ierr);
  ierr = MatDestroy(&A);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
}



/*TEST

   test:
      output_file: output/ex114.out

   test:
      suffix: 2
      nsize: 2

   test:
      suffix: 3
      nsize: 2
      args: -mat_type baij

TEST*/
