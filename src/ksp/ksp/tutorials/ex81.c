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
  PetscErrorCode     ierr;

  ierr = PetscInitialize(&argc,&args,NULL,help);if (ierr) return ierr;
  CHKERRMPI(MPI_Comm_size(PETSC_COMM_WORLD,&size));
  CHKERRQ(PetscMalloc1(found,&outer));
  CHKERRQ(PetscOptionsGetIntArray(NULL,NULL,"-outer_fieldsplit_sizes",outer,&found,&flg));
  CHKERRQ(PetscRandomCreate(PETSC_COMM_WORLD,&rctx));
  if (flg) {
    PetscCheckFalse(found == 1,PETSC_COMM_WORLD,PETSC_ERR_USER,"Must supply more than one field");
    j = 0;
    for (i=0; i<found; ++i) j += outer[i];
    PetscCheckFalse(j != Q,PETSC_COMM_WORLD,PETSC_ERR_USER,"Sum of outer fieldsplit sizes (%D) greater than number of blocks in MatNest (%D)",j,Q);
  }
  CHKERRQ(KSPCreate(PETSC_COMM_WORLD,&ksp));
  size = PetscMax(3,size);
  for (i=0; i<Q*Q; ++i) array[i] = NULL;
  for (i=0; i<Q; ++i) {
    if (i == 0) {
      CHKERRQ(MatCreateAIJ(PETSC_COMM_WORLD,PETSC_DECIDE,PETSC_DECIDE,size,size,1,NULL,0,NULL,array+(Q+1)*i));
    } else if (i == 1 || i == 3) {
      CHKERRQ(MatCreateSBAIJ(PETSC_COMM_WORLD,2,PETSC_DECIDE,PETSC_DECIDE,size,size,1,NULL,0,NULL,array+(Q+1)*i));
    } else if (i == 2 || i == 4) {
      CHKERRQ(MatCreateBAIJ(PETSC_COMM_WORLD,2,PETSC_DECIDE,PETSC_DECIDE,size,size,1,NULL,0,NULL,array+(Q+1)*i));
    }
    CHKERRQ(MatAssemblyBegin(array[(Q+1)*i],MAT_FINAL_ASSEMBLY));
    CHKERRQ(MatAssemblyEnd(array[(Q+1)*i],MAT_FINAL_ASSEMBLY));
    CHKERRQ(MatShift(array[(Q+1)*i],100+i+1));
    if (i == 3) {
      CHKERRQ(MatDuplicate(array[(Q+1)*i],MAT_COPY_VALUES,&a));
      CHKERRQ(MatDestroy(array+(Q+1)*i));
      CHKERRQ(MatCreateHermitianTranspose(a,array+(Q+1)*i));
      CHKERRQ(MatDestroy(&a));
    }
    size *= 2;
  }
  CHKERRQ(MatGetSize(array[0],&M,NULL));
  for (i=2; i<Q; ++i) {
    CHKERRQ(MatGetSize(array[(Q+1)*i],NULL,&N));
    if (i != Q-1) {
      CHKERRQ(MatCreateAIJ(PETSC_COMM_WORLD,PETSC_DECIDE,PETSC_DECIDE,i==3?N:M,i==3?M:N,0,NULL,0,NULL,array+i));
    } else {
      CHKERRQ(MatCreateDense(PETSC_COMM_WORLD,PETSC_DECIDE,PETSC_DECIDE,M,N,NULL,array+i));
    }
    CHKERRQ(MatAssemblyBegin(array[i],MAT_FINAL_ASSEMBLY));
    CHKERRQ(MatAssemblyEnd(array[i],MAT_FINAL_ASSEMBLY));
    CHKERRQ(MatSetRandom(array[i],rctx));
    if (i == 3) {
      CHKERRQ(MatDuplicate(array[i],MAT_COPY_VALUES,&a));
      CHKERRQ(MatDestroy(array+i));
      CHKERRQ(MatCreateHermitianTranspose(a,array+i));
      CHKERRQ(MatDestroy(&a));
    }
  }
  CHKERRQ(MatGetSize(array[0],NULL,&N));
  for (i=2; i<Q; i+=2) {
    CHKERRQ(MatGetSize(array[(Q+1)*i],&M,NULL));
    if (i != Q-1) {
      CHKERRQ(MatCreateAIJ(PETSC_COMM_WORLD,PETSC_DECIDE,PETSC_DECIDE,M,N,2,NULL,2,NULL,array+Q*i));
    } else {
      CHKERRQ(MatCreateDense(PETSC_COMM_WORLD,PETSC_DECIDE,PETSC_DECIDE,N,M,NULL,array+Q*i));
    }
    CHKERRQ(MatAssemblyBegin(array[Q*i],MAT_FINAL_ASSEMBLY));
    CHKERRQ(MatAssemblyEnd(array[Q*i],MAT_FINAL_ASSEMBLY));
    CHKERRQ(MatSetRandom(array[Q*i],rctx));
    if (i == Q-1) {
      CHKERRQ(MatDuplicate(array[Q*i],MAT_COPY_VALUES,&a));
      CHKERRQ(MatDestroy(array+Q*i));
      CHKERRQ(MatCreateHermitianTranspose(a,array+Q*i));
      CHKERRQ(MatDestroy(&a));
    }
  }
  CHKERRQ(MatGetSize(array[(Q+1)*3],&M,NULL));
  for (i=1; i<3; ++i) {
    CHKERRQ(MatGetSize(array[(Q+1)*i],NULL,&N));
    CHKERRQ(MatCreateAIJ(PETSC_COMM_WORLD,PETSC_DECIDE,PETSC_DECIDE,M,N,2,NULL,2,NULL,array+Q*3+i));
    CHKERRQ(MatAssemblyBegin(array[Q*3+i],MAT_FINAL_ASSEMBLY));
    CHKERRQ(MatAssemblyEnd(array[Q*3+i],MAT_FINAL_ASSEMBLY));
    CHKERRQ(MatSetRandom(array[Q*3+i],rctx));
  }
  CHKERRQ(MatGetSize(array[(Q+1)*1],NULL,&N));
  CHKERRQ(MatGetSize(array[(Q+1)*(Q-1)],&M,NULL));
  CHKERRQ(MatCreateBAIJ(PETSC_COMM_WORLD,2,PETSC_DECIDE,PETSC_DECIDE,M,N,0,NULL,0,NULL,&a));
  CHKERRQ(MatAssemblyBegin(a,MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatAssemblyEnd(a,MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatCreateHermitianTranspose(a,array+Q+Q-1));
  CHKERRQ(MatDestroy(&a));
  CHKERRQ(MatDestroy(array+Q*Q-1));
  CHKERRQ(MatCreateNest(PETSC_COMM_WORLD,Q,NULL,Q,NULL,array,&A));
  for (i=0; i<Q; ++i) {
    CHKERRQ(MatDestroy(array+(Q+1)*i));
  }
  for (i=2; i<Q; ++i) {
    CHKERRQ(MatDestroy(array+i));
    CHKERRQ(MatDestroy(array+Q*i));
  }
  for (i=1; i<3; ++i) {
    CHKERRQ(MatDestroy(array+Q*3+i));
  }
  CHKERRQ(MatDestroy(array+Q+Q-1));
  CHKERRQ(KSPSetOperators(ksp,A,A));
  CHKERRQ(MatNestGetISs(A,rows,NULL));
  CHKERRQ(KSPGetPC(ksp,&pc));
  CHKERRQ(PCSetType(pc,PCFIELDSPLIT));
  M = 0;
  for (j=0; j<found; ++j) {
    IS expand1,expand2;
    CHKERRQ(ISDuplicate(rows[M],&expand1));
    for (i=1; i<outer[j]; ++i) {
      CHKERRQ(ISExpand(expand1,rows[M+i],&expand2));
      CHKERRQ(ISDestroy(&expand1));
      expand1 = expand2;
    }
    M += outer[j];
    CHKERRQ(PCFieldSplitSetIS(pc,NULL,expand1));
    CHKERRQ(ISDestroy(&expand1));
  }
  CHKERRQ(KSPSetFromOptions(ksp));
  flg = PETSC_FALSE;
  CHKERRQ(PetscOptionsGetBool(NULL,NULL,"-test_matmult",&flg,NULL));
  if (flg) {
    Mat D, E, F, H;
    CHKERRQ(MatConvert(A,MATDENSE,MAT_INITIAL_MATRIX,&D));
    CHKERRQ(MatMultEqual(A,D,10,&flg));
    PetscCheckFalse(!flg,PETSC_COMM_SELF,PETSC_ERR_PLIB,"MatDense != MatNest");
    CHKERRQ(MatZeroEntries(D));
    CHKERRQ(MatConvert(A,MATDENSE,MAT_REUSE_MATRIX,&D));
    CHKERRQ(MatMultEqual(A,D,10,&flg));
    PetscCheckFalse(!flg,PETSC_COMM_SELF,PETSC_ERR_PLIB,"MatDense != MatNest");
    CHKERRQ(MatConvert(D,MATAIJ,MAT_INITIAL_MATRIX,&E));
    CHKERRQ(MatMultEqual(E,D,10,&flg));
    PetscCheckFalse(!flg,PETSC_COMM_SELF,PETSC_ERR_PLIB,"MatDense != MatAIJ");
    CHKERRQ(MatZeroEntries(E));
    CHKERRQ(MatConvert(D,MATAIJ,MAT_REUSE_MATRIX,&E));
    CHKERRQ(MatMultEqual(E,D,10,&flg));
    PetscCheckFalse(!flg,PETSC_COMM_SELF,PETSC_ERR_PLIB,"MatDense != MatAIJ");
    CHKERRQ(MatConvert(E,MATDENSE,MAT_INPLACE_MATRIX,&E));
    CHKERRQ(MatMultEqual(A,E,10,&flg));
    PetscCheckFalse(!flg,PETSC_COMM_SELF,PETSC_ERR_PLIB,"MatAIJ != MatNest");
    CHKERRQ(MatConvert(D,MATAIJ,MAT_INPLACE_MATRIX,&D));
    CHKERRQ(MatMultEqual(A,D,10,&flg));
    PetscCheckFalse(!flg,PETSC_COMM_SELF,PETSC_ERR_PLIB,"MatDense != MatNest");
    CHKERRQ(MatDestroy(&E));
    CHKERRQ(MatCreateHermitianTranspose(D,&E));
    CHKERRQ(MatConvert(E,MATAIJ,MAT_INITIAL_MATRIX,&F));
    CHKERRQ(MatMultEqual(E,F,10,&flg));
    PetscCheckFalse(!flg,PETSC_COMM_SELF,PETSC_ERR_PLIB,"MatHermitianTranspose != MatAIJ");
    CHKERRQ(MatZeroEntries(F));
    CHKERRQ(MatConvert(E,MATAIJ,MAT_REUSE_MATRIX,&F));
    CHKERRQ(MatMultEqual(E,F,10,&flg));
    PetscCheckFalse(!flg,PETSC_COMM_SELF,PETSC_ERR_PLIB,"MatHermitianTranspose != MatAIJ");
    CHKERRQ(MatDestroy(&F));
    CHKERRQ(MatConvert(E,MATAIJ,MAT_INPLACE_MATRIX,&E));
    CHKERRQ(MatCreateHermitianTranspose(D,&H));
    CHKERRQ(MatMultEqual(E,H,10,&flg));
    PetscCheckFalse(!flg,PETSC_COMM_SELF,PETSC_ERR_PLIB,"MatHermitianTranspose != MatAIJ");
    CHKERRQ(MatDestroy(&H));
    CHKERRQ(MatDestroy(&E));
    CHKERRQ(MatDestroy(&D));
  }
  CHKERRQ(KSPSetUp(ksp));
  CHKERRQ(MatCreateVecs(A,&b,&x));
  CHKERRQ(VecSetRandom(b,rctx));
  CHKERRQ(VecGetSubVector(b,rows[Q-1],&sub));
  CHKERRQ(VecSet(sub,0.0));
  CHKERRQ(VecRestoreSubVector(b,rows[Q-1],&sub));
  CHKERRQ(KSPSolve(ksp,b,x));
  CHKERRQ(VecDestroy(&b));
  CHKERRQ(VecDestroy(&x));
  CHKERRQ(PetscRandomDestroy(&rctx));
  CHKERRQ(MatDestroy(&A));
  CHKERRQ(KSPDestroy(&ksp));
  CHKERRQ(PetscFree(outer));
  ierr = PetscFinalize();
  return ierr;
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
