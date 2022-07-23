
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
  PetscMPIInt    size,rank;
  PetscReal      enorm;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc,&args,(char*)0,help));
  PetscCallMPI(MPI_Comm_size(PETSC_COMM_WORLD,&size));
  PetscCallMPI(MPI_Comm_rank(PETSC_COMM_WORLD,&rank));
  PetscCall(PetscOptionsGetInt(NULL,NULL,"-testcase",&testcase,NULL));

  PetscCall(MatCreate(PETSC_COMM_WORLD,&A));
  if (testcase == 1) { /* proc[0] holds entire A and other processes have no entry */
    if (rank == 0) {
      PetscCall(MatSetSizes(A,M,N,PETSC_DECIDE,PETSC_DECIDE));
    } else {
      PetscCall(MatSetSizes(A,0,0,PETSC_DECIDE,PETSC_DECIDE));
    }
    testcase = 0;
  } else {
    PetscCall(MatSetSizes(A,PETSC_DECIDE,PETSC_DECIDE,M,N));
  }
  PetscCall(MatSetFromOptions(A));
  PetscCall(MatSetUp(A));

  if (rank == 0) { /* proc[0] sets matrix A */
    for (j=0; j<N; j++) indices[j] = j;
    switch (testcase) {
    case 1: /* see testcast 0 */
      break;
    case 2:
      row = 0;
      values[0]  = -2.0; values[1] = -2.0; values[2] = -2.0; values[3] = -4.0; values[4] = 1.0; values[5] = 1.0;
      PetscCall(MatSetValues(A,1,&row,N,indices,values,INSERT_VALUES));
      row = 2;
      indices[0] = 0;    indices[1] = 3;    indices[2] = 5;
      values[0]  = -2.0; values[1]  = -2.0; values[2]  = -2.0;
      PetscCall(MatSetValues(A,1,&row,3,indices,values,INSERT_VALUES));
      row = 3;
      indices[0] = 0;    indices[1] = 1;    indices[2] = 4;
      values[0]  = -2.0; values[1]  = -2.0; values[2]  = -2.0;
      PetscCall(MatSetValues(A,1,&row,3,indices,values,INSERT_VALUES));
      row = 4;
      indices[0] = 0;    indices[1] = 1;    indices[2] = 2;
      values[0]  = -2.0; values[1]  = -2.0; values[2]  = -2.0;
      PetscCall(MatSetValues(A,1,&row,3,indices,values,INSERT_VALUES));
      break;
    case 3:
      row = 0;
      values[0]  = -2.0; values[1] = -2.0; values[2] = -2.0;
      PetscCall(MatSetValues(A,1,&row,3,indices+1,values,INSERT_VALUES));
      row = 1;
      values[0]  = -2.0; values[1] = -2.0; values[2] = -2.0;
      PetscCall(MatSetValues(A,1,&row,3,indices,values,INSERT_VALUES));
      row = 2;
      values[0]  = -2.0; values[1] = -2.0; values[2]  = -2.0;
      PetscCall(MatSetValues(A,1,&row,3,indices,values,INSERT_VALUES));
      row = 3;
      values[0]  = -2.0; values[1] = -2.0; values[2]  = -2.0; values[3] = -1.0;
      PetscCall(MatSetValues(A,1,&row,4,indices,values,INSERT_VALUES));
      row = 4;
      values[0]  = -2.0; values[1] = -2.0; values[2]  = -2.0; values[3] = -1.0;
      PetscCall(MatSetValues(A,1,&row,4,indices,values,INSERT_VALUES));
      break;

    default:
      row  = 0;
      values[0]  = -1.0; values[1] = 0.0; values[2] = 1.0; values[3] = 3.0; values[4] = 4.0; values[5] = -5.0;
      PetscCall(MatSetValues(A,1,&row,N,indices,values,INSERT_VALUES));
      row  = 1;
      PetscCall(MatSetValues(A,1,&row,3,indices,values,INSERT_VALUES));
      row  = 3;
      PetscCall(MatSetValues(A,1,&row,1,indices+4,values+4,INSERT_VALUES));
      row  = 4;
      PetscCall(MatSetValues(A,1,&row,2,indices+4,values+4,INSERT_VALUES));
    }
  }
  PetscCall(MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY));
  PetscCall(MatView(A,PETSC_VIEWER_STDOUT_WORLD));

  PetscCall(MatGetLocalSize(A, &m,&n));
  PetscCall(VecCreate(PETSC_COMM_WORLD,&min));
  PetscCall(VecSetSizes(min,m,PETSC_DECIDE));
  PetscCall(VecSetFromOptions(min));
  PetscCall(VecDuplicate(min,&max));
  PetscCall(VecDuplicate(min,&maxabs));
  PetscCall(VecDuplicate(min,&e));

  /* MatGetRowMax() */
  PetscCall(PetscPrintf(PETSC_COMM_WORLD,"\n MatGetRowMax\n"));
  PetscCall(MatGetRowMax(A,max,NULL));
  PetscCall(MatGetRowMax(A,max,imax));
  PetscCall(VecView(max,PETSC_VIEWER_STDOUT_WORLD));
  PetscCall(VecGetLocalSize(max,&n));
  PetscCall(PetscIntView(n,imax,PETSC_VIEWER_STDOUT_WORLD));

  /* MatGetRowMin() */
  PetscCall(MatScale(A,-1.0));
  PetscCall(MatView(A,PETSC_VIEWER_STDOUT_WORLD));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD,"\n MatGetRowMin\n"));
  PetscCall(MatGetRowMin(A,min,NULL));
  PetscCall(MatGetRowMin(A,min,imin));

  PetscCall(VecWAXPY(e,1.0,max,min)); /* e = max + min */
  PetscCall(VecNorm(e,NORM_INFINITY,&enorm));
  PetscCheck(enorm <= PETSC_MACHINE_EPSILON,PETSC_COMM_WORLD,PETSC_ERR_PLIB,"max+min > PETSC_MACHINE_EPSILON ");
  for (j = 0; j < n; j++) {
    PetscCheck(imin[j] == imax[j],PETSC_COMM_SELF,PETSC_ERR_PLIB,"imin[%" PetscInt_FMT "] %" PetscInt_FMT " != imax %" PetscInt_FMT,j,imin[j],imax[j]);
  }

  /* MatGetRowMaxAbs() */
  PetscCall(PetscPrintf(PETSC_COMM_WORLD,"\n MatGetRowMaxAbs\n"));
  PetscCall(MatGetRowMaxAbs(A,maxabs,NULL));
  PetscCall(MatGetRowMaxAbs(A,maxabs,imaxabs));
  PetscCall(VecView(maxabs,PETSC_VIEWER_STDOUT_WORLD));
  PetscCall(PetscIntView(n,imaxabs,PETSC_VIEWER_STDOUT_WORLD));

  /* MatGetRowMinAbs() */
  PetscCall(PetscPrintf(PETSC_COMM_WORLD,"\n MatGetRowMinAbs\n"));
  PetscCall(MatGetRowMinAbs(A,min,NULL));
  PetscCall(MatGetRowMinAbs(A,min,imin));
  PetscCall(VecView(min,PETSC_VIEWER_STDOUT_WORLD));
  PetscCall(PetscIntView(n,imin,PETSC_VIEWER_STDOUT_WORLD));

  if (size == 1) {
    /* Test MatGetRowMax, MatGetRowMin and MatGetRowMaxAbs for SeqDense and MPIBAIJ matrix */
    Mat Adense;
    Vec max_d,maxabs_d;
    PetscCall(VecDuplicate(min,&max_d));
    PetscCall(VecDuplicate(min,&maxabs_d));

    PetscCall(MatScale(A,-1.0));
    PetscCall(MatConvert(A,MATDENSE,MAT_INITIAL_MATRIX,&Adense));
    PetscCall(PetscPrintf(PETSC_COMM_WORLD,"MatGetRowMax for seqdense matrix\n"));
    PetscCall(MatGetRowMax(Adense,max_d,imax));

    PetscCall(VecWAXPY(e,-1.0,max,max_d)); /* e = -max + max_d */
    PetscCall(VecNorm(e,NORM_INFINITY,&enorm));
    PetscCheck(enorm <= PETSC_MACHINE_EPSILON,PETSC_COMM_WORLD,PETSC_ERR_PLIB,"norm(-max + max_d) %g > PETSC_MACHINE_EPSILON",(double)enorm);

    PetscCall(MatScale(Adense,-1.0));
    PetscCall(PetscPrintf(PETSC_COMM_WORLD,"MatGetRowMin for seqdense matrix\n"));
    PetscCall(MatGetRowMin(Adense,min,imin));

    PetscCall(VecWAXPY(e,1.0,max,min)); /* e = max + min */
    PetscCall(VecNorm(e,NORM_INFINITY,&enorm));
    PetscCheck(enorm <= PETSC_MACHINE_EPSILON,PETSC_COMM_WORLD,PETSC_ERR_PLIB,"max+min > PETSC_MACHINE_EPSILON ");
    for (j = 0; j < n; j++) {
      if (imin[j] != imax[j]) {
        SETERRQ(PETSC_COMM_SELF,PETSC_ERR_PLIB,"imin[%" PetscInt_FMT "] %" PetscInt_FMT " != imax %" PetscInt_FMT " for seqdense matrix",j,imin[j],imax[j]);
      }
    }

    PetscCall(PetscPrintf(PETSC_COMM_WORLD,"MatGetRowMaxAbs for seqdense matrix\n"));
    PetscCall(MatGetRowMaxAbs(Adense,maxabs_d,imaxabs));
    PetscCall(VecWAXPY(e,-1.0,maxabs,maxabs_d)); /* e = -maxabs + maxabs_d */
    PetscCall(VecNorm(e,NORM_INFINITY,&enorm));
    PetscCheck(enorm <= PETSC_MACHINE_EPSILON,PETSC_COMM_WORLD,PETSC_ERR_PLIB,"norm(-maxabs + maxabs_d) %g > PETSC_MACHINE_EPSILON",(double)enorm);

    PetscCall(MatDestroy(&Adense));
    PetscCall(VecDestroy(&max_d));
    PetscCall(VecDestroy(&maxabs_d));
  }

  { /* BAIJ matrix */
    Mat               B;
    Vec               maxabsB,maxabsB2;
    PetscInt          bs=2,*imaxabsB,*imaxabsB2,rstart,rend,cstart,cend,ncols,col,Brows[2],Bcols[2];
    const PetscInt    *cols;
    const PetscScalar *vals,*vals2;
    PetscScalar       Bvals[4];

    PetscCall(PetscMalloc2(M,&imaxabsB,bs*M,&imaxabsB2));

    /* bs = 1 */
    PetscCall(MatConvert(A,MATMPIBAIJ,MAT_INITIAL_MATRIX,&B));
    PetscCall(VecDuplicate(min,&maxabsB));
    PetscCall(MatGetRowMaxAbs(B,maxabsB,NULL));
    PetscCall(MatGetRowMaxAbs(B,maxabsB,imaxabsB));
    PetscCall(PetscPrintf(PETSC_COMM_WORLD,"\n MatGetRowMaxAbs for BAIJ matrix\n"));
    PetscCall(VecWAXPY(e,-1.0,maxabs,maxabsB)); /* e = -maxabs + maxabsB */
    PetscCall(VecNorm(e,NORM_INFINITY,&enorm));
    PetscCheck(enorm <= PETSC_MACHINE_EPSILON,PETSC_COMM_WORLD,PETSC_ERR_PLIB,"norm(-maxabs + maxabs_d) %g > PETSC_MACHINE_EPSILON",(double)enorm);

    for (j = 0; j < n; j++) {
      PetscCheck(imaxabs[j] == imaxabsB[j],PETSC_COMM_SELF,PETSC_ERR_PLIB,"imaxabs[%" PetscInt_FMT "] %" PetscInt_FMT " != imaxabsB %" PetscInt_FMT,j,imin[j],imax[j]);
    }
    PetscCall(MatDestroy(&B));

    /* Test bs = 2: Create B with bs*bs block structure of A */
    PetscCall(VecCreate(PETSC_COMM_WORLD,&maxabsB2));
    PetscCall(VecSetSizes(maxabsB2,bs*m,PETSC_DECIDE));
    PetscCall(VecSetFromOptions(maxabsB2));

    PetscCall(MatGetOwnershipRange(A,&rstart,&rend));
    PetscCall(MatGetOwnershipRangeColumn(A,&cstart,&cend));
    PetscCall(MatCreate(PETSC_COMM_WORLD,&B));
    PetscCall(MatSetSizes(B,bs*(rend-rstart),bs*(cend-cstart),PETSC_DECIDE,PETSC_DECIDE));
    PetscCall(MatSetFromOptions(B));
    PetscCall(MatSetUp(B));

    for (row=rstart; row<rend; row++) {
      PetscCall(MatGetRow(A,row,&ncols,&cols,&vals));
      for (col=0; col<ncols; col++) {
        for (j=0; j<bs; j++) {
          Brows[j] = bs*row + j;
          Bcols[j] = bs*cols[col]+j;
        }
        for (j=0; j<bs*bs; j++) Bvals[j] = vals[col];
        PetscCall(MatSetValues(B,bs,Brows,bs,Bcols,Bvals,INSERT_VALUES));
      }
      PetscCall(MatRestoreRow(A,row,&ncols,&cols,&vals));
    }
    PetscCall(MatAssemblyBegin(B,MAT_FINAL_ASSEMBLY));
    PetscCall(MatAssemblyEnd(B,MAT_FINAL_ASSEMBLY));

    PetscCall(MatGetRowMaxAbs(B,maxabsB2,imaxabsB2));

    /* Check maxabsB2 and imaxabsB2 */
    PetscCall(VecGetArrayRead(maxabsB,&vals));
    PetscCall(VecGetArrayRead(maxabsB2,&vals2));
    for (row=0; row<m; row++) {
      if (PetscAbsScalar(vals[row] - vals2[bs*row]) > PETSC_MACHINE_EPSILON)
        SETERRQ(PETSC_COMM_SELF,PETSC_ERR_PLIB,"row %" PetscInt_FMT " maxabsB != maxabsB2",row);
    }
    PetscCall(VecRestoreArrayRead(maxabsB,&vals));
    PetscCall(VecRestoreArrayRead(maxabsB2,&vals2));

    for (col=0; col<n; col++) {
      if (imaxabsB[col] != imaxabsB2[bs*col]/bs)
        SETERRQ(PETSC_COMM_SELF,PETSC_ERR_PLIB,"col %" PetscInt_FMT " imaxabsB != imaxabsB2",col);
    }
    PetscCall(VecDestroy(&maxabsB));
    PetscCall(MatDestroy(&B));
    PetscCall(VecDestroy(&maxabsB2));
    PetscCall(PetscFree2(imaxabsB,imaxabsB2));
  }

  PetscCall(VecDestroy(&min));
  PetscCall(VecDestroy(&max));
  PetscCall(VecDestroy(&maxabs));
  PetscCall(VecDestroy(&e));
  PetscCall(MatDestroy(&A));
  PetscCall(PetscFinalize());
  return 0;
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
