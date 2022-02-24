
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
  CHKERRMPI(MPI_Comm_size(PETSC_COMM_WORLD,&size));
  CHKERRMPI(MPI_Comm_rank(PETSC_COMM_WORLD,&rank));
  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-testcase",&testcase,NULL));

  CHKERRQ(MatCreate(PETSC_COMM_WORLD,&A));
  if (testcase == 1) { /* proc[0] holds entire A and other processes have no entry */
    if (rank == 0) {
      CHKERRQ(MatSetSizes(A,M,N,PETSC_DECIDE,PETSC_DECIDE));
    } else {
      CHKERRQ(MatSetSizes(A,0,0,PETSC_DECIDE,PETSC_DECIDE));
    }
    testcase = 0;
  } else {
    CHKERRQ(MatSetSizes(A,PETSC_DECIDE,PETSC_DECIDE,M,N));
  }
  CHKERRQ(MatSetFromOptions(A));
  CHKERRQ(MatSetUp(A));

  if (rank == 0) { /* proc[0] sets matrix A */
    for (j=0; j<N; j++) indices[j] = j;
    switch (testcase) {
    case 1: /* see testcast 0 */
      break;
    case 2:
      row = 0;
      values[0]  = -2.0; values[1] = -2.0; values[2] = -2.0; values[3] = -4.0; values[4] = 1.0; values[5] = 1.0;
      CHKERRQ(MatSetValues(A,1,&row,N,indices,values,INSERT_VALUES));
      row = 2;
      indices[0] = 0;    indices[1] = 3;    indices[2] = 5;
      values[0]  = -2.0; values[1]  = -2.0; values[2]  = -2.0;
      CHKERRQ(MatSetValues(A,1,&row,3,indices,values,INSERT_VALUES));
      row = 3;
      indices[0] = 0;    indices[1] = 1;    indices[2] = 4;
      values[0]  = -2.0; values[1]  = -2.0; values[2]  = -2.0;
      CHKERRQ(MatSetValues(A,1,&row,3,indices,values,INSERT_VALUES));
      row = 4;
      indices[0] = 0;    indices[1] = 1;    indices[2] = 2;
      values[0]  = -2.0; values[1]  = -2.0; values[2]  = -2.0;
      CHKERRQ(MatSetValues(A,1,&row,3,indices,values,INSERT_VALUES));
      break;
    case 3:
      row = 0;
      values[0]  = -2.0; values[1] = -2.0; values[2] = -2.0;
      CHKERRQ(MatSetValues(A,1,&row,3,indices+1,values,INSERT_VALUES));
      row = 1;
      values[0]  = -2.0; values[1] = -2.0; values[2] = -2.0;
      CHKERRQ(MatSetValues(A,1,&row,3,indices,values,INSERT_VALUES));
      row = 2;
      values[0]  = -2.0; values[1] = -2.0; values[2]  = -2.0;
      CHKERRQ(MatSetValues(A,1,&row,3,indices,values,INSERT_VALUES));
      row = 3;
      values[0]  = -2.0; values[1] = -2.0; values[2]  = -2.0; values[3] = -1.0;
      CHKERRQ(MatSetValues(A,1,&row,4,indices,values,INSERT_VALUES));
      row = 4;
      values[0]  = -2.0; values[1] = -2.0; values[2]  = -2.0; values[3] = -1.0;
      CHKERRQ(MatSetValues(A,1,&row,4,indices,values,INSERT_VALUES));
      break;

    default:
      row  = 0;
      values[0]  = -1.0; values[1] = 0.0; values[2] = 1.0; values[3] = 3.0; values[4] = 4.0; values[5] = -5.0;
      CHKERRQ(MatSetValues(A,1,&row,N,indices,values,INSERT_VALUES));
      row  = 1;
      CHKERRQ(MatSetValues(A,1,&row,3,indices,values,INSERT_VALUES));
      row  = 3;
      CHKERRQ(MatSetValues(A,1,&row,1,indices+4,values+4,INSERT_VALUES));
      row  = 4;
      CHKERRQ(MatSetValues(A,1,&row,2,indices+4,values+4,INSERT_VALUES));
    }
  }
  CHKERRQ(MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatView(A,PETSC_VIEWER_STDOUT_WORLD));

  CHKERRQ(MatGetLocalSize(A, &m,&n));
  CHKERRQ(VecCreate(PETSC_COMM_WORLD,&min));
  CHKERRQ(VecSetSizes(min,m,PETSC_DECIDE));
  CHKERRQ(VecSetFromOptions(min));
  CHKERRQ(VecDuplicate(min,&max));
  CHKERRQ(VecDuplicate(min,&maxabs));
  CHKERRQ(VecDuplicate(min,&e));

  /* MatGetRowMax() */
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"\n MatGetRowMax\n"));
  CHKERRQ(MatGetRowMax(A,max,NULL));
  CHKERRQ(MatGetRowMax(A,max,imax));
  CHKERRQ(VecView(max,PETSC_VIEWER_STDOUT_WORLD));
  CHKERRQ(VecGetLocalSize(max,&n));
  CHKERRQ(PetscIntView(n,imax,PETSC_VIEWER_STDOUT_WORLD));

  /* MatGetRowMin() */
  CHKERRQ(MatScale(A,-1.0));
  CHKERRQ(MatView(A,PETSC_VIEWER_STDOUT_WORLD));
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"\n MatGetRowMin\n"));
  CHKERRQ(MatGetRowMin(A,min,NULL));
  CHKERRQ(MatGetRowMin(A,min,imin));

  CHKERRQ(VecWAXPY(e,1.0,max,min)); /* e = max + min */
  CHKERRQ(VecNorm(e,NORM_INFINITY,&enorm));
  PetscCheckFalse(enorm > PETSC_MACHINE_EPSILON,PETSC_COMM_WORLD,PETSC_ERR_PLIB,"max+min > PETSC_MACHINE_EPSILON ");
  for (j = 0; j < n; j++) {
    PetscCheckFalse(imin[j] != imax[j],PETSC_COMM_SELF,PETSC_ERR_PLIB,"imin[%" PetscInt_FMT "] %" PetscInt_FMT " != imax %" PetscInt_FMT,j,imin[j],imax[j]);
  }

  /* MatGetRowMaxAbs() */
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"\n MatGetRowMaxAbs\n"));
  CHKERRQ(MatGetRowMaxAbs(A,maxabs,NULL));
  CHKERRQ(MatGetRowMaxAbs(A,maxabs,imaxabs));
  CHKERRQ(VecView(maxabs,PETSC_VIEWER_STDOUT_WORLD));
  CHKERRQ(PetscIntView(n,imaxabs,PETSC_VIEWER_STDOUT_WORLD));

  /* MatGetRowMinAbs() */
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"\n MatGetRowMinAbs\n"));
  CHKERRQ(MatGetRowMinAbs(A,min,NULL));
  CHKERRQ(MatGetRowMinAbs(A,min,imin));
  CHKERRQ(VecView(min,PETSC_VIEWER_STDOUT_WORLD));
  CHKERRQ(PetscIntView(n,imin,PETSC_VIEWER_STDOUT_WORLD));

  if (size == 1) {
    /* Test MatGetRowMax, MatGetRowMin and MatGetRowMaxAbs for SeqDense and MPIBAIJ matrix */
    Mat Adense;
    Vec max_d,maxabs_d;
    CHKERRQ(VecDuplicate(min,&max_d));
    CHKERRQ(VecDuplicate(min,&maxabs_d));

    CHKERRQ(MatScale(A,-1.0));
    CHKERRQ(MatConvert(A,MATDENSE,MAT_INITIAL_MATRIX,&Adense));
    CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"MatGetRowMax for seqdense matrix\n"));
    CHKERRQ(MatGetRowMax(Adense,max_d,imax));

    CHKERRQ(VecWAXPY(e,-1.0,max,max_d)); /* e = -max + max_d */
    CHKERRQ(VecNorm(e,NORM_INFINITY,&enorm));
    PetscCheckFalse(enorm > PETSC_MACHINE_EPSILON,PETSC_COMM_WORLD,PETSC_ERR_PLIB,"norm(-max + max_d) %g > PETSC_MACHINE_EPSILON",(double)enorm);

    CHKERRQ(MatScale(Adense,-1.0));
    CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"MatGetRowMin for seqdense matrix\n"));
    CHKERRQ(MatGetRowMin(Adense,min,imin));

    CHKERRQ(VecWAXPY(e,1.0,max,min)); /* e = max + min */
    CHKERRQ(VecNorm(e,NORM_INFINITY,&enorm));
    PetscCheckFalse(enorm > PETSC_MACHINE_EPSILON,PETSC_COMM_WORLD,PETSC_ERR_PLIB,"max+min > PETSC_MACHINE_EPSILON ");
    for (j = 0; j < n; j++) {
      if (imin[j] != imax[j]) {
        SETERRQ(PETSC_COMM_SELF,PETSC_ERR_PLIB,"imin[%" PetscInt_FMT "] %" PetscInt_FMT " != imax %" PetscInt_FMT " for seqdense matrix",j,imin[j],imax[j]);
      }
    }

    CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"MatGetRowMaxAbs for seqdense matrix\n"));
    CHKERRQ(MatGetRowMaxAbs(Adense,maxabs_d,imaxabs));
    CHKERRQ(VecWAXPY(e,-1.0,maxabs,maxabs_d)); /* e = -maxabs + maxabs_d */
    CHKERRQ(VecNorm(e,NORM_INFINITY,&enorm));
    PetscCheckFalse(enorm > PETSC_MACHINE_EPSILON,PETSC_COMM_WORLD,PETSC_ERR_PLIB,"norm(-maxabs + maxabs_d) %g > PETSC_MACHINE_EPSILON",(double)enorm);

    CHKERRQ(MatDestroy(&Adense));
    CHKERRQ(VecDestroy(&max_d));
    CHKERRQ(VecDestroy(&maxabs_d));
  }

  { /* BAIJ matrix */
    Mat               B;
    Vec               maxabsB,maxabsB2;
    PetscInt          bs=2,*imaxabsB,*imaxabsB2,rstart,rend,cstart,cend,ncols,col,Brows[2],Bcols[2];
    const PetscInt    *cols;
    const PetscScalar *vals,*vals2;
    PetscScalar       Bvals[4];

    CHKERRQ(PetscMalloc2(M,&imaxabsB,bs*M,&imaxabsB2));

    /* bs = 1 */
    CHKERRQ(MatConvert(A,MATMPIBAIJ,MAT_INITIAL_MATRIX,&B));
    CHKERRQ(VecDuplicate(min,&maxabsB));
    CHKERRQ(MatGetRowMaxAbs(B,maxabsB,NULL));
    CHKERRQ(MatGetRowMaxAbs(B,maxabsB,imaxabsB));
    CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"\n MatGetRowMaxAbs for BAIJ matrix\n"));
    CHKERRQ(VecWAXPY(e,-1.0,maxabs,maxabsB)); /* e = -maxabs + maxabsB */
    CHKERRQ(VecNorm(e,NORM_INFINITY,&enorm));
    PetscCheckFalse(enorm > PETSC_MACHINE_EPSILON,PETSC_COMM_WORLD,PETSC_ERR_PLIB,"norm(-maxabs + maxabs_d) %g > PETSC_MACHINE_EPSILON",(double)enorm);

    for (j = 0; j < n; j++) {
      PetscCheckFalse(imaxabs[j] != imaxabsB[j],PETSC_COMM_SELF,PETSC_ERR_PLIB,"imaxabs[%" PetscInt_FMT "] %" PetscInt_FMT " != imaxabsB %" PetscInt_FMT,j,imin[j],imax[j]);
    }
    CHKERRQ(MatDestroy(&B));

    /* Test bs = 2: Create B with bs*bs block structure of A */
    CHKERRQ(VecCreate(PETSC_COMM_WORLD,&maxabsB2));
    CHKERRQ(VecSetSizes(maxabsB2,bs*m,PETSC_DECIDE));
    CHKERRQ(VecSetFromOptions(maxabsB2));

    CHKERRQ(MatGetOwnershipRange(A,&rstart,&rend));
    CHKERRQ(MatGetOwnershipRangeColumn(A,&cstart,&cend));
    CHKERRQ(MatCreate(PETSC_COMM_WORLD,&B));
    CHKERRQ(MatSetSizes(B,bs*(rend-rstart),bs*(cend-cstart),PETSC_DECIDE,PETSC_DECIDE));
    CHKERRQ(MatSetFromOptions(B));
    CHKERRQ(MatSetUp(B));

    for (row=rstart; row<rend; row++) {
      CHKERRQ(MatGetRow(A,row,&ncols,&cols,&vals));
      for (col=0; col<ncols; col++) {
        for (j=0; j<bs; j++) {
          Brows[j] = bs*row + j;
          Bcols[j] = bs*cols[col]+j;
        }
        for (j=0; j<bs*bs; j++) Bvals[j] = vals[col];
        CHKERRQ(MatSetValues(B,bs,Brows,bs,Bcols,Bvals,INSERT_VALUES));
      }
      CHKERRQ(MatRestoreRow(A,row,&ncols,&cols,&vals));
    }
    CHKERRQ(MatAssemblyBegin(B,MAT_FINAL_ASSEMBLY));
    CHKERRQ(MatAssemblyEnd(B,MAT_FINAL_ASSEMBLY));

    CHKERRQ(MatGetRowMaxAbs(B,maxabsB2,imaxabsB2));

    /* Check maxabsB2 and imaxabsB2 */
    CHKERRQ(VecGetArrayRead(maxabsB,&vals));
    CHKERRQ(VecGetArrayRead(maxabsB2,&vals2));
    for (row=0; row<m; row++) {
      if (PetscAbsScalar(vals[row] - vals2[bs*row]) > PETSC_MACHINE_EPSILON)
        SETERRQ(PETSC_COMM_SELF,PETSC_ERR_PLIB,"row %" PetscInt_FMT " maxabsB != maxabsB2",row);
    }
    CHKERRQ(VecRestoreArrayRead(maxabsB,&vals));
    CHKERRQ(VecRestoreArrayRead(maxabsB2,&vals2));

    for (col=0; col<n; col++) {
      if (imaxabsB[col] != imaxabsB2[bs*col]/bs)
        SETERRQ(PETSC_COMM_SELF,PETSC_ERR_PLIB,"col %" PetscInt_FMT " imaxabsB != imaxabsB2",col);
    }
    CHKERRQ(VecDestroy(&maxabsB));
    CHKERRQ(MatDestroy(&B));
    CHKERRQ(VecDestroy(&maxabsB2));
    CHKERRQ(PetscFree2(imaxabsB,imaxabsB2));
  }

  CHKERRQ(VecDestroy(&min));
  CHKERRQ(VecDestroy(&max));
  CHKERRQ(VecDestroy(&maxabs));
  CHKERRQ(VecDestroy(&e));
  CHKERRQ(MatDestroy(&A));
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
