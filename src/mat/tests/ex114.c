
static char help[] = "Tests MatGetRowMax(), MatGetRowMin(), MatGetRowMaxAbs()\n";

#include <petscmat.h>

#define M 5
#define N 6

int main(int argc,char **args)
{
  Mat            A;
  Vec            min,max,maxabs,e;
  PetscInt       m,n,j,imin[M],imax[M],imaxabs[M],indices[N],row,testcase=0;
  PetscScalar    values[N];
  PetscErrorCode ierr;
  PetscMPIInt    size,rank;
  PetscReal      enorm;

  ierr = PetscInitialize(&argc,&args,(char*)0,help);if (ierr) return ierr;
  ierr = MPI_Comm_size(PETSC_COMM_WORLD,&size);CHKERRMPI(ierr);
  ierr = MPI_Comm_rank(PETSC_COMM_WORLD,&rank);CHKERRMPI(ierr);
  ierr = PetscOptionsGetInt(NULL,NULL,"-testcase",&testcase,NULL);CHKERRQ(ierr);

  ierr = MatCreate(PETSC_COMM_WORLD,&A);CHKERRQ(ierr);
  if (testcase == 1) { /* proc[0] holds entire A and other processes have no entry */
    if (rank == 0) {
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

  if (rank == 0) { /* proc[0] sets matrix A */
    for (j=0; j<N; j++) indices[j] = j;
    switch (testcase) {
    case 1: /* see testcast 0 */
      break;
    case 2:
      row = 0;
      values[0]  = -2.0; values[1] = -2.0; values[2] = -2.0; values[3] = -4.0; values[4] = 1.0; values[5] = 1.0;
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

  /* MatGetRowMax() */
  ierr = PetscPrintf(PETSC_COMM_WORLD,"\n MatGetRowMax\n");CHKERRQ(ierr);
  ierr = MatGetRowMax(A,max,NULL);CHKERRQ(ierr);
  ierr = MatGetRowMax(A,max,imax);CHKERRQ(ierr);
  ierr = VecView(max,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  ierr = VecGetLocalSize(max,&n);CHKERRQ(ierr);
  ierr = PetscIntView(n,imax,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);

  /* MatGetRowMin() */
  ierr = MatScale(A,-1.0);CHKERRQ(ierr);
  ierr = MatView(A,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"\n MatGetRowMin\n");CHKERRQ(ierr);
  ierr = MatGetRowMin(A,min,NULL);CHKERRQ(ierr);
  ierr = MatGetRowMin(A,min,imin);CHKERRQ(ierr);

  ierr = VecWAXPY(e,1.0,max,min);CHKERRQ(ierr); /* e = max + min */
  ierr = VecNorm(e,NORM_INFINITY,&enorm);CHKERRQ(ierr);
  if (enorm > PETSC_MACHINE_EPSILON) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_PLIB,"max+min > PETSC_MACHINE_EPSILON ");
  for (j = 0; j < n; j++) {
    if (imin[j] != imax[j]) SETERRQ3(PETSC_COMM_SELF,PETSC_ERR_PLIB,"imin[%" PetscInt_FMT "] %" PetscInt_FMT " != imax %" PetscInt_FMT,j,imin[j],imax[j]);
  }

  /* MatGetRowMaxAbs() */
  ierr = PetscPrintf(PETSC_COMM_WORLD,"\n MatGetRowMaxAbs\n");CHKERRQ(ierr);
  ierr = MatGetRowMaxAbs(A,maxabs,NULL);CHKERRQ(ierr);
  ierr = MatGetRowMaxAbs(A,maxabs,imaxabs);CHKERRQ(ierr);
  ierr = VecView(maxabs,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  ierr = PetscIntView(n,imaxabs,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);

  /* MatGetRowMinAbs() */
  ierr = PetscPrintf(PETSC_COMM_WORLD,"\n MatGetRowMinAbs\n");CHKERRQ(ierr);
  ierr = MatGetRowMinAbs(A,min,NULL);CHKERRQ(ierr);
  ierr = MatGetRowMinAbs(A,min,imin);CHKERRQ(ierr);
  ierr = VecView(min,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  ierr = PetscIntView(n,imin,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);

  if (size == 1) {
    /* Test MatGetRowMax, MatGetRowMin and MatGetRowMaxAbs for SeqDense and MPIBAIJ matrix */
    Mat Adense;
    Vec max_d,maxabs_d;
    ierr = VecDuplicate(min,&max_d);CHKERRQ(ierr);
    ierr = VecDuplicate(min,&maxabs_d);CHKERRQ(ierr);

    ierr = MatScale(A,-1.0);CHKERRQ(ierr);
    ierr = MatConvert(A,MATDENSE,MAT_INITIAL_MATRIX,&Adense);CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_WORLD,"MatGetRowMax for seqdense matrix\n");CHKERRQ(ierr);
    ierr = MatGetRowMax(Adense,max_d,imax);CHKERRQ(ierr);

    ierr = VecWAXPY(e,-1.0,max,max_d);CHKERRQ(ierr); /* e = -max + max_d */
    ierr = VecNorm(e,NORM_INFINITY,&enorm);CHKERRQ(ierr);
    if (enorm > PETSC_MACHINE_EPSILON) SETERRQ1(PETSC_COMM_WORLD,PETSC_ERR_PLIB,"norm(-max + max_d) %g > PETSC_MACHINE_EPSILON",(double)enorm);

    ierr = MatScale(Adense,-1.0);CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_WORLD,"MatGetRowMin for seqdense matrix\n");CHKERRQ(ierr);
    ierr = MatGetRowMin(Adense,min,imin);CHKERRQ(ierr);

    ierr = VecWAXPY(e,1.0,max,min);CHKERRQ(ierr); /* e = max + min */
    ierr = VecNorm(e,NORM_INFINITY,&enorm);CHKERRQ(ierr);
    if (enorm > PETSC_MACHINE_EPSILON) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_PLIB,"max+min > PETSC_MACHINE_EPSILON ");
    for (j = 0; j < n; j++) {
      if (imin[j] != imax[j]) {
        SETERRQ3(PETSC_COMM_SELF,PETSC_ERR_PLIB,"imin[%" PetscInt_FMT "] %" PetscInt_FMT " != imax %" PetscInt_FMT " for seqdense matrix",j,imin[j],imax[j]);
      }
    }

    ierr = PetscPrintf(PETSC_COMM_WORLD,"MatGetRowMaxAbs for seqdense matrix\n");CHKERRQ(ierr);
    ierr = MatGetRowMaxAbs(Adense,maxabs_d,imaxabs);CHKERRQ(ierr);
    ierr = VecWAXPY(e,-1.0,maxabs,maxabs_d);CHKERRQ(ierr); /* e = -maxabs + maxabs_d */
    ierr = VecNorm(e,NORM_INFINITY,&enorm);CHKERRQ(ierr);
    if (enorm > PETSC_MACHINE_EPSILON) SETERRQ1(PETSC_COMM_WORLD,PETSC_ERR_PLIB,"norm(-maxabs + maxabs_d) %g > PETSC_MACHINE_EPSILON",(double)enorm);

    ierr = MatDestroy(&Adense);CHKERRQ(ierr);
    ierr = VecDestroy(&max_d);CHKERRQ(ierr);
    ierr = VecDestroy(&maxabs_d);CHKERRQ(ierr);
  }

  { /* BAIJ matrix */
    Mat               B;
    Vec               maxabsB,maxabsB2;
    PetscInt          bs=2,*imaxabsB,*imaxabsB2,rstart,rend,cstart,cend,ncols,col,Brows[2],Bcols[2];
    const PetscInt    *cols;
    const PetscScalar *vals,*vals2;
    PetscScalar       Bvals[4];

    ierr = PetscMalloc2(M,&imaxabsB,bs*M,&imaxabsB2);CHKERRQ(ierr);

    /* bs = 1 */
    ierr = MatConvert(A,MATMPIBAIJ,MAT_INITIAL_MATRIX,&B);CHKERRQ(ierr);
    ierr = VecDuplicate(min,&maxabsB);CHKERRQ(ierr);
    ierr = MatGetRowMaxAbs(B,maxabsB,NULL);CHKERRQ(ierr);
    ierr = MatGetRowMaxAbs(B,maxabsB,imaxabsB);CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_WORLD,"\n MatGetRowMaxAbs for BAIJ matrix\n");CHKERRQ(ierr);
    ierr = VecWAXPY(e,-1.0,maxabs,maxabsB);CHKERRQ(ierr); /* e = -maxabs + maxabsB */
    ierr = VecNorm(e,NORM_INFINITY,&enorm);CHKERRQ(ierr);
    if (enorm > PETSC_MACHINE_EPSILON) SETERRQ1(PETSC_COMM_WORLD,PETSC_ERR_PLIB,"norm(-maxabs + maxabs_d) %g > PETSC_MACHINE_EPSILON",(double)enorm);

    for (j = 0; j < n; j++) {
      if (imaxabs[j] != imaxabsB[j]) SETERRQ3(PETSC_COMM_SELF,PETSC_ERR_PLIB,"imaxabs[%" PetscInt_FMT "] %" PetscInt_FMT " != imaxabsB %" PetscInt_FMT,j,imin[j],imax[j]);
    }
    ierr = MatDestroy(&B);CHKERRQ(ierr);

    /* Test bs = 2: Create B with bs*bs block structure of A */
    ierr = VecCreate(PETSC_COMM_WORLD,&maxabsB2);CHKERRQ(ierr);
    ierr = VecSetSizes(maxabsB2,bs*m,PETSC_DECIDE);CHKERRQ(ierr);
    ierr = VecSetFromOptions(maxabsB2);CHKERRQ(ierr);

    ierr = MatGetOwnershipRange(A,&rstart,&rend);CHKERRQ(ierr);
    ierr = MatGetOwnershipRangeColumn(A,&cstart,&cend);CHKERRQ(ierr);
    ierr = MatCreate(PETSC_COMM_WORLD,&B);CHKERRQ(ierr);
    ierr = MatSetSizes(B,bs*(rend-rstart),bs*(cend-cstart),PETSC_DECIDE,PETSC_DECIDE);CHKERRQ(ierr);
    ierr = MatSetFromOptions(B);CHKERRQ(ierr);
    ierr = MatSetUp(B);CHKERRQ(ierr);

    for (row=rstart; row<rend; row++) {
      ierr = MatGetRow(A,row,&ncols,&cols,&vals);CHKERRQ(ierr);
      for (col=0; col<ncols; col++) {
        for (j=0; j<bs; j++) {
          Brows[j] = bs*row + j;
          Bcols[j] = bs*cols[col]+j;
        }
        for (j=0; j<bs*bs; j++) Bvals[j] = vals[col];
        ierr = MatSetValues(B,bs,Brows,bs,Bcols,Bvals,INSERT_VALUES);CHKERRQ(ierr);
      }
      ierr = MatRestoreRow(A,row,&ncols,&cols,&vals);CHKERRQ(ierr);
    }
    ierr = MatAssemblyBegin(B,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatAssemblyEnd(B,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

    ierr = MatGetRowMaxAbs(B,maxabsB2,imaxabsB2);CHKERRQ(ierr);

    /* Check maxabsB2 and imaxabsB2 */
    ierr = VecGetArrayRead(maxabsB,&vals);CHKERRQ(ierr);
    ierr = VecGetArrayRead(maxabsB2,&vals2);CHKERRQ(ierr);
    for (row=0; row<m; row++) {
      if (PetscAbsScalar(vals[row] - vals2[bs*row]) > PETSC_MACHINE_EPSILON)
        SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_PLIB,"row %" PetscInt_FMT " maxabsB != maxabsB2",row);
    }
    ierr = VecRestoreArrayRead(maxabsB,&vals);CHKERRQ(ierr);
    ierr = VecRestoreArrayRead(maxabsB2,&vals2);CHKERRQ(ierr);

    for (col=0; col<n; col++) {
      if (imaxabsB[col] != imaxabsB2[bs*col]/bs)
        SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_PLIB,"col %" PetscInt_FMT " imaxabsB != imaxabsB2",col);
    }
    ierr = VecDestroy(&maxabsB);CHKERRQ(ierr);
    ierr = MatDestroy(&B);CHKERRQ(ierr);
    ierr = VecDestroy(&maxabsB2);CHKERRQ(ierr);
    ierr = PetscFree2(imaxabsB,imaxabsB2);CHKERRQ(ierr);
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
      args: -testcase 1
      output_file: output/ex114.out

   test:
      suffix: 3
      args: -testcase 2
      output_file: output/ex114_3.out

   test:
      suffix: 4
      args: -testcase 3
      output_file: output/ex114_4.out

   test:
      suffix: 5
      nsize: 3
      args: -testcase 0
      output_file: output/ex114_5.out

   test:
      suffix: 6
      nsize: 3
      args: -testcase 1
      output_file: output/ex114_6.out

   test:
      suffix: 7
      nsize: 3
      args: -testcase 2
      output_file: output/ex114_7.out

   test:
      suffix: 8
      nsize: 3
      args: -testcase 3
      output_file: output/ex114_8.out

TEST*/
