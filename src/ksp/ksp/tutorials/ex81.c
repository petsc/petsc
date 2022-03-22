#include <petsc.h>

static char help[] = "Solves a linear system with a MatNest and nested fields.\n\n";

#define Q 5 /* everything is hardwired for a 5x5 MatNest for now */

int main(int argc,char **args)
{
  KSP                ksp;
  PC                 pc;
  Mat                array[Q*Q],A,a;
  Vec                b,x,sub;
  IS                 rows[Q];
  PetscInt           i,j,*outer,M,N,found=Q;
  PetscMPIInt        size;
  PetscBool          flg;
  PetscRandom        rctx;

  PetscCall(PetscInitialize(&argc,&args,NULL,help));
  PetscCallMPI(MPI_Comm_size(PETSC_COMM_WORLD,&size));
  PetscCall(PetscMalloc1(found,&outer));
  PetscCall(PetscOptionsGetIntArray(NULL,NULL,"-outer_fieldsplit_sizes",outer,&found,&flg));
  PetscCall(PetscRandomCreate(PETSC_COMM_WORLD,&rctx));
  if (flg) {
    PetscCheck(found != 1,PETSC_COMM_WORLD,PETSC_ERR_USER,"Must supply more than one field");
    j = 0;
    for (i=0; i<found; ++i) j += outer[i];
    PetscCheck(j == Q,PETSC_COMM_WORLD,PETSC_ERR_USER,"Sum of outer fieldsplit sizes (%D) greater than number of blocks in MatNest (%D)",j,Q);
  }
  PetscCall(KSPCreate(PETSC_COMM_WORLD,&ksp));
  size = PetscMax(3,size);
  for (i=0; i<Q*Q; ++i) array[i] = NULL;
  for (i=0; i<Q; ++i) {
    if (i == 0) {
      PetscCall(MatCreateAIJ(PETSC_COMM_WORLD,PETSC_DECIDE,PETSC_DECIDE,size,size,1,NULL,0,NULL,array+(Q+1)*i));
    } else if (i == 1 || i == 3) {
      PetscCall(MatCreateSBAIJ(PETSC_COMM_WORLD,2,PETSC_DECIDE,PETSC_DECIDE,size,size,1,NULL,0,NULL,array+(Q+1)*i));
    } else if (i == 2 || i == 4) {
      PetscCall(MatCreateBAIJ(PETSC_COMM_WORLD,2,PETSC_DECIDE,PETSC_DECIDE,size,size,1,NULL,0,NULL,array+(Q+1)*i));
    }
    PetscCall(MatAssemblyBegin(array[(Q+1)*i],MAT_FINAL_ASSEMBLY));
    PetscCall(MatAssemblyEnd(array[(Q+1)*i],MAT_FINAL_ASSEMBLY));
    PetscCall(MatShift(array[(Q+1)*i],100+i+1));
    if (i == 3) {
      PetscCall(MatDuplicate(array[(Q+1)*i],MAT_COPY_VALUES,&a));
      PetscCall(MatDestroy(array+(Q+1)*i));
      PetscCall(MatCreateHermitianTranspose(a,array+(Q+1)*i));
      PetscCall(MatDestroy(&a));
    }
    size *= 2;
  }
  PetscCall(MatGetSize(array[0],&M,NULL));
  for (i=2; i<Q; ++i) {
    PetscCall(MatGetSize(array[(Q+1)*i],NULL,&N));
    if (i != Q-1) {
      PetscCall(MatCreateAIJ(PETSC_COMM_WORLD,PETSC_DECIDE,PETSC_DECIDE,i==3?N:M,i==3?M:N,0,NULL,0,NULL,array+i));
    } else {
      PetscCall(MatCreateDense(PETSC_COMM_WORLD,PETSC_DECIDE,PETSC_DECIDE,M,N,NULL,array+i));
    }
    PetscCall(MatAssemblyBegin(array[i],MAT_FINAL_ASSEMBLY));
    PetscCall(MatAssemblyEnd(array[i],MAT_FINAL_ASSEMBLY));
    PetscCall(MatSetRandom(array[i],rctx));
    if (i == 3) {
      PetscCall(MatDuplicate(array[i],MAT_COPY_VALUES,&a));
      PetscCall(MatDestroy(array+i));
      PetscCall(MatCreateHermitianTranspose(a,array+i));
      PetscCall(MatDestroy(&a));
    }
  }
  PetscCall(MatGetSize(array[0],NULL,&N));
  for (i=2; i<Q; i+=2) {
    PetscCall(MatGetSize(array[(Q+1)*i],&M,NULL));
    if (i != Q-1) {
      PetscCall(MatCreateAIJ(PETSC_COMM_WORLD,PETSC_DECIDE,PETSC_DECIDE,M,N,2,NULL,2,NULL,array+Q*i));
    } else {
      PetscCall(MatCreateDense(PETSC_COMM_WORLD,PETSC_DECIDE,PETSC_DECIDE,N,M,NULL,array+Q*i));
    }
    PetscCall(MatAssemblyBegin(array[Q*i],MAT_FINAL_ASSEMBLY));
    PetscCall(MatAssemblyEnd(array[Q*i],MAT_FINAL_ASSEMBLY));
    PetscCall(MatSetRandom(array[Q*i],rctx));
    if (i == Q-1) {
      PetscCall(MatDuplicate(array[Q*i],MAT_COPY_VALUES,&a));
      PetscCall(MatDestroy(array+Q*i));
      PetscCall(MatCreateHermitianTranspose(a,array+Q*i));
      PetscCall(MatDestroy(&a));
    }
  }
  PetscCall(MatGetSize(array[(Q+1)*3],&M,NULL));
  for (i=1; i<3; ++i) {
    PetscCall(MatGetSize(array[(Q+1)*i],NULL,&N));
    PetscCall(MatCreateAIJ(PETSC_COMM_WORLD,PETSC_DECIDE,PETSC_DECIDE,M,N,2,NULL,2,NULL,array+Q*3+i));
    PetscCall(MatAssemblyBegin(array[Q*3+i],MAT_FINAL_ASSEMBLY));
    PetscCall(MatAssemblyEnd(array[Q*3+i],MAT_FINAL_ASSEMBLY));
    PetscCall(MatSetRandom(array[Q*3+i],rctx));
  }
  PetscCall(MatGetSize(array[(Q+1)*1],NULL,&N));
  PetscCall(MatGetSize(array[(Q+1)*(Q-1)],&M,NULL));
  PetscCall(MatCreateBAIJ(PETSC_COMM_WORLD,2,PETSC_DECIDE,PETSC_DECIDE,M,N,0,NULL,0,NULL,&a));
  PetscCall(MatAssemblyBegin(a,MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(a,MAT_FINAL_ASSEMBLY));
  PetscCall(MatCreateHermitianTranspose(a,array+Q+Q-1));
  PetscCall(MatDestroy(&a));
  PetscCall(MatDestroy(array+Q*Q-1));
  PetscCall(MatCreateNest(PETSC_COMM_WORLD,Q,NULL,Q,NULL,array,&A));
  for (i=0; i<Q; ++i) {
    PetscCall(MatDestroy(array+(Q+1)*i));
  }
  for (i=2; i<Q; ++i) {
    PetscCall(MatDestroy(array+i));
    PetscCall(MatDestroy(array+Q*i));
  }
  for (i=1; i<3; ++i) {
    PetscCall(MatDestroy(array+Q*3+i));
  }
  PetscCall(MatDestroy(array+Q+Q-1));
  PetscCall(KSPSetOperators(ksp,A,A));
  PetscCall(MatNestGetISs(A,rows,NULL));
  PetscCall(KSPGetPC(ksp,&pc));
  PetscCall(PCSetType(pc,PCFIELDSPLIT));
  M = 0;
  for (j=0; j<found; ++j) {
    IS expand1,expand2;
    PetscCall(ISDuplicate(rows[M],&expand1));
    for (i=1; i<outer[j]; ++i) {
      PetscCall(ISExpand(expand1,rows[M+i],&expand2));
      PetscCall(ISDestroy(&expand1));
      expand1 = expand2;
    }
    M += outer[j];
    PetscCall(PCFieldSplitSetIS(pc,NULL,expand1));
    PetscCall(ISDestroy(&expand1));
  }
  PetscCall(KSPSetFromOptions(ksp));
  flg = PETSC_FALSE;
  PetscCall(PetscOptionsGetBool(NULL,NULL,"-test_matmult",&flg,NULL));
  if (flg) {
    Mat D, E, F, H;
    PetscCall(MatConvert(A,MATDENSE,MAT_INITIAL_MATRIX,&D));
    PetscCall(MatMultEqual(A,D,10,&flg));
    PetscCheck(flg,PETSC_COMM_SELF,PETSC_ERR_PLIB,"MatDense != MatNest");
    PetscCall(MatZeroEntries(D));
    PetscCall(MatConvert(A,MATDENSE,MAT_REUSE_MATRIX,&D));
    PetscCall(MatMultEqual(A,D,10,&flg));
    PetscCheck(flg,PETSC_COMM_SELF,PETSC_ERR_PLIB,"MatDense != MatNest");
    PetscCall(MatConvert(D,MATAIJ,MAT_INITIAL_MATRIX,&E));
    PetscCall(MatMultEqual(E,D,10,&flg));
    PetscCheck(flg,PETSC_COMM_SELF,PETSC_ERR_PLIB,"MatDense != MatAIJ");
    PetscCall(MatZeroEntries(E));
    PetscCall(MatConvert(D,MATAIJ,MAT_REUSE_MATRIX,&E));
    PetscCall(MatMultEqual(E,D,10,&flg));
    PetscCheck(flg,PETSC_COMM_SELF,PETSC_ERR_PLIB,"MatDense != MatAIJ");
    PetscCall(MatConvert(E,MATDENSE,MAT_INPLACE_MATRIX,&E));
    PetscCall(MatMultEqual(A,E,10,&flg));
    PetscCheck(flg,PETSC_COMM_SELF,PETSC_ERR_PLIB,"MatAIJ != MatNest");
    PetscCall(MatConvert(D,MATAIJ,MAT_INPLACE_MATRIX,&D));
    PetscCall(MatMultEqual(A,D,10,&flg));
    PetscCheck(flg,PETSC_COMM_SELF,PETSC_ERR_PLIB,"MatDense != MatNest");
    PetscCall(MatDestroy(&E));
    PetscCall(MatCreateHermitianTranspose(D,&E));
    PetscCall(MatConvert(E,MATAIJ,MAT_INITIAL_MATRIX,&F));
    PetscCall(MatMultEqual(E,F,10,&flg));
    PetscCheck(flg,PETSC_COMM_SELF,PETSC_ERR_PLIB,"MatHermitianTranspose != MatAIJ");
    PetscCall(MatZeroEntries(F));
    PetscCall(MatConvert(E,MATAIJ,MAT_REUSE_MATRIX,&F));
    PetscCall(MatMultEqual(E,F,10,&flg));
    PetscCheck(flg,PETSC_COMM_SELF,PETSC_ERR_PLIB,"MatHermitianTranspose != MatAIJ");
    PetscCall(MatDestroy(&F));
    PetscCall(MatConvert(E,MATAIJ,MAT_INPLACE_MATRIX,&E));
    PetscCall(MatCreateHermitianTranspose(D,&H));
    PetscCall(MatMultEqual(E,H,10,&flg));
    PetscCheck(flg,PETSC_COMM_SELF,PETSC_ERR_PLIB,"MatHermitianTranspose != MatAIJ");
    PetscCall(MatDestroy(&H));
    PetscCall(MatDestroy(&E));
    PetscCall(MatDestroy(&D));
  }
  PetscCall(KSPSetUp(ksp));
  PetscCall(MatCreateVecs(A,&b,&x));
  PetscCall(VecSetRandom(b,rctx));
  PetscCall(VecGetSubVector(b,rows[Q-1],&sub));
  PetscCall(VecSet(sub,0.0));
  PetscCall(VecRestoreSubVector(b,rows[Q-1],&sub));
  PetscCall(KSPSolve(ksp,b,x));
  PetscCall(VecDestroy(&b));
  PetscCall(VecDestroy(&x));
  PetscCall(PetscRandomDestroy(&rctx));
  PetscCall(MatDestroy(&A));
  PetscCall(KSPDestroy(&ksp));
  PetscCall(PetscFree(outer));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

   test:
      nsize: {{1 3}shared output}
      suffix: wo_explicit_schur
      filter: sed -e "s/seq/mpi/g" -e "s/CONVERGED_ATOL/CONVERGED_RTOL/g" -e "s/ 1 MPI processes/ 3 MPI processes/g" -e "s/iterations [4-5]/iterations 4/g"
      args: -outer_fieldsplit_sizes {{1,2,2 2,1,2 2,2,1}separate output} -ksp_view_mat -pc_type fieldsplit -ksp_converged_reason -fieldsplit_pc_type jacobi -test_matmult

   test:
      nsize: {{1 3}shared output}
      suffix: w_explicit_schur
      filter: sed -e "s/seq/mpi/g" -e "s/CONVERGED_ATOL/CONVERGED_RTOL/g" -e "s/ 1 MPI processes/ 3 MPI processes/g" -e "s/iterations [1-2]/iterations 1/g"
      args: -outer_fieldsplit_sizes {{1,4 2,3 3,2 4,1}separate output} -ksp_view_mat -pc_type fieldsplit -ksp_converged_reason -fieldsplit_pc_type jacobi -pc_fieldsplit_type schur -pc_fieldsplit_schur_fact_type full -pc_fieldsplit_schur_precondition full

TEST*/
