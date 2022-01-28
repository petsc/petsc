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
  ierr = MPI_Comm_size(PETSC_COMM_WORLD,&size);CHKERRMPI(ierr);
  ierr = PetscMalloc1(found,&outer);CHKERRQ(ierr);
  ierr = PetscOptionsGetIntArray(NULL,NULL,"-outer_fieldsplit_sizes",outer,&found,&flg);CHKERRQ(ierr);
  ierr = PetscRandomCreate(PETSC_COMM_WORLD,&rctx);CHKERRQ(ierr);
  if (flg) {
    if (found == 1) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_USER,"Must supply more than one field");
    j = 0;
    for (i=0; i<found; ++i) j += outer[i];
    if (j != Q) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_USER,"Sum of outer fieldsplit sizes (%D) greater than number of blocks in MatNest (%D)",j,Q);
  }
  ierr = KSPCreate(PETSC_COMM_WORLD,&ksp);CHKERRQ(ierr);
  size = PetscMax(3,size);
  for (i=0; i<Q*Q; ++i) array[i] = NULL;
  for (i=0; i<Q; ++i) {
    if (i == 0) {
      ierr = MatCreateAIJ(PETSC_COMM_WORLD,PETSC_DECIDE,PETSC_DECIDE,size,size,1,NULL,0,NULL,array+(Q+1)*i);CHKERRQ(ierr);
    } else if (i == 1 || i == 3) {
      ierr = MatCreateSBAIJ(PETSC_COMM_WORLD,2,PETSC_DECIDE,PETSC_DECIDE,size,size,1,NULL,0,NULL,array+(Q+1)*i);CHKERRQ(ierr);
    } else if (i == 2 || i == 4) {
      ierr = MatCreateBAIJ(PETSC_COMM_WORLD,2,PETSC_DECIDE,PETSC_DECIDE,size,size,1,NULL,0,NULL,array+(Q+1)*i);CHKERRQ(ierr);
    }
    ierr = MatAssemblyBegin(array[(Q+1)*i],MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatAssemblyEnd(array[(Q+1)*i],MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatShift(array[(Q+1)*i],100+i+1);CHKERRQ(ierr);
    if (i == 3) {
      ierr = MatDuplicate(array[(Q+1)*i],MAT_COPY_VALUES,&a);CHKERRQ(ierr);
      ierr = MatDestroy(array+(Q+1)*i);CHKERRQ(ierr);
      ierr = MatCreateHermitianTranspose(a,array+(Q+1)*i);CHKERRQ(ierr);
      ierr = MatDestroy(&a);CHKERRQ(ierr);
    }
    size *= 2;
  }
  ierr = MatGetSize(array[0],&M,NULL);CHKERRQ(ierr);
  for (i=2; i<Q; ++i) {
    ierr = MatGetSize(array[(Q+1)*i],NULL,&N);CHKERRQ(ierr);
    if (i != Q-1) {
      ierr = MatCreateAIJ(PETSC_COMM_WORLD,PETSC_DECIDE,PETSC_DECIDE,i==3?N:M,i==3?M:N,0,NULL,0,NULL,array+i);CHKERRQ(ierr);
    } else {
      ierr = MatCreateDense(PETSC_COMM_WORLD,PETSC_DECIDE,PETSC_DECIDE,M,N,NULL,array+i);CHKERRQ(ierr);
    }
    ierr = MatAssemblyBegin(array[i],MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatAssemblyEnd(array[i],MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatSetRandom(array[i],rctx);CHKERRQ(ierr);
    if (i == 3) {
      ierr = MatDuplicate(array[i],MAT_COPY_VALUES,&a);CHKERRQ(ierr);
      ierr = MatDestroy(array+i);CHKERRQ(ierr);
      ierr = MatCreateHermitianTranspose(a,array+i);CHKERRQ(ierr);
      ierr = MatDestroy(&a);CHKERRQ(ierr);
    }
  }
  ierr = MatGetSize(array[0],NULL,&N);CHKERRQ(ierr);
  for (i=2; i<Q; i+=2) {
    ierr = MatGetSize(array[(Q+1)*i],&M,NULL);CHKERRQ(ierr);
    if (i != Q-1) {
      ierr = MatCreateAIJ(PETSC_COMM_WORLD,PETSC_DECIDE,PETSC_DECIDE,M,N,2,NULL,2,NULL,array+Q*i);CHKERRQ(ierr);
    } else {
      ierr = MatCreateDense(PETSC_COMM_WORLD,PETSC_DECIDE,PETSC_DECIDE,N,M,NULL,array+Q*i);CHKERRQ(ierr);
    }
    ierr = MatAssemblyBegin(array[Q*i],MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatAssemblyEnd(array[Q*i],MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatSetRandom(array[Q*i],rctx);CHKERRQ(ierr);
    if (i == Q-1) {
      ierr = MatDuplicate(array[Q*i],MAT_COPY_VALUES,&a);CHKERRQ(ierr);
      ierr = MatDestroy(array+Q*i);CHKERRQ(ierr);
      ierr = MatCreateHermitianTranspose(a,array+Q*i);CHKERRQ(ierr);
      ierr = MatDestroy(&a);CHKERRQ(ierr);
    }
  }
  ierr = MatGetSize(array[(Q+1)*3],&M,NULL);CHKERRQ(ierr);
  for (i=1; i<3; ++i) {
    ierr = MatGetSize(array[(Q+1)*i],NULL,&N);CHKERRQ(ierr);
    ierr = MatCreateAIJ(PETSC_COMM_WORLD,PETSC_DECIDE,PETSC_DECIDE,M,N,2,NULL,2,NULL,array+Q*3+i);CHKERRQ(ierr);
    ierr = MatAssemblyBegin(array[Q*3+i],MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatAssemblyEnd(array[Q*3+i],MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatSetRandom(array[Q*3+i],rctx);CHKERRQ(ierr);
  }
  ierr = MatGetSize(array[(Q+1)*1],NULL,&N);CHKERRQ(ierr);
  ierr = MatGetSize(array[(Q+1)*(Q-1)],&M,NULL);CHKERRQ(ierr);
  ierr = MatCreateBAIJ(PETSC_COMM_WORLD,2,PETSC_DECIDE,PETSC_DECIDE,M,N,0,NULL,0,NULL,&a);CHKERRQ(ierr);
  ierr = MatAssemblyBegin(a,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(a,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatCreateHermitianTranspose(a,array+Q+Q-1);CHKERRQ(ierr);
  ierr = MatDestroy(&a);CHKERRQ(ierr);
  ierr = MatDestroy(array+Q*Q-1);CHKERRQ(ierr);
  ierr = MatCreateNest(PETSC_COMM_WORLD,Q,NULL,Q,NULL,array,&A);CHKERRQ(ierr);
  for (i=0; i<Q; ++i) {
    ierr = MatDestroy(array+(Q+1)*i);CHKERRQ(ierr);
  }
  for (i=2; i<Q; ++i) {
    ierr = MatDestroy(array+i);CHKERRQ(ierr);
    ierr = MatDestroy(array+Q*i);CHKERRQ(ierr);
  }
  for (i=1; i<3; ++i) {
    ierr = MatDestroy(array+Q*3+i);CHKERRQ(ierr);
  }
  ierr = MatDestroy(array+Q+Q-1);CHKERRQ(ierr);
  ierr = KSPSetOperators(ksp,A,A);CHKERRQ(ierr);
  ierr = MatNestGetISs(A,rows,NULL);CHKERRQ(ierr);
  ierr = KSPGetPC(ksp,&pc);CHKERRQ(ierr);
  ierr = PCSetType(pc,PCFIELDSPLIT);CHKERRQ(ierr);
  M = 0;
  for (j=0; j<found; ++j) {
    IS expand1,expand2;
    ierr = ISDuplicate(rows[M],&expand1);CHKERRQ(ierr);
    for (i=1; i<outer[j]; ++i) {
      ierr = ISExpand(expand1,rows[M+i],&expand2);CHKERRQ(ierr);
      ierr = ISDestroy(&expand1);CHKERRQ(ierr);
      expand1 = expand2;
    }
    M += outer[j];
    ierr = PCFieldSplitSetIS(pc,NULL,expand1);CHKERRQ(ierr);
    ierr = ISDestroy(&expand1);CHKERRQ(ierr);
  }
  ierr = KSPSetFromOptions(ksp);CHKERRQ(ierr);
  flg = PETSC_FALSE;
  ierr = PetscOptionsGetBool(NULL,NULL,"-test_matmult",&flg,NULL);CHKERRQ(ierr);
  if (flg) {
    Mat D, E, F, H;
    ierr = MatConvert(A,MATDENSE,MAT_INITIAL_MATRIX,&D);CHKERRQ(ierr);
    ierr = MatMultEqual(A,D,10,&flg);CHKERRQ(ierr);
    if (!flg) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_PLIB,"MatDense != MatNest");
    ierr = MatZeroEntries(D);CHKERRQ(ierr);
    ierr = MatConvert(A,MATDENSE,MAT_REUSE_MATRIX,&D);CHKERRQ(ierr);
    ierr = MatMultEqual(A,D,10,&flg);CHKERRQ(ierr);
    if (!flg) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_PLIB,"MatDense != MatNest");
    ierr = MatConvert(D,MATAIJ,MAT_INITIAL_MATRIX,&E);CHKERRQ(ierr);
    ierr = MatMultEqual(E,D,10,&flg);CHKERRQ(ierr);
    if (!flg) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_PLIB,"MatDense != MatAIJ");
    ierr = MatZeroEntries(E);CHKERRQ(ierr);
    ierr = MatConvert(D,MATAIJ,MAT_REUSE_MATRIX,&E);CHKERRQ(ierr);
    ierr = MatMultEqual(E,D,10,&flg);CHKERRQ(ierr);
    if (!flg) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_PLIB,"MatDense != MatAIJ");
    ierr = MatConvert(E,MATDENSE,MAT_INPLACE_MATRIX,&E);CHKERRQ(ierr);
    ierr = MatMultEqual(A,E,10,&flg);CHKERRQ(ierr);
    if (!flg) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_PLIB,"MatAIJ != MatNest");
    ierr = MatConvert(D,MATAIJ,MAT_INPLACE_MATRIX,&D);CHKERRQ(ierr);
    ierr = MatMultEqual(A,D,10,&flg);CHKERRQ(ierr);
    if (!flg) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_PLIB,"MatDense != MatNest");
    ierr = MatDestroy(&E);CHKERRQ(ierr);
    ierr = MatCreateHermitianTranspose(D,&E);CHKERRQ(ierr);
    ierr = MatConvert(E,MATAIJ,MAT_INITIAL_MATRIX,&F);CHKERRQ(ierr);
    ierr = MatMultEqual(E,F,10,&flg);CHKERRQ(ierr);
    if (!flg) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_PLIB,"MatHermitianTranspose != MatAIJ");
    ierr = MatZeroEntries(F);CHKERRQ(ierr);
    ierr = MatConvert(E,MATAIJ,MAT_REUSE_MATRIX,&F);CHKERRQ(ierr);
    ierr = MatMultEqual(E,F,10,&flg);CHKERRQ(ierr);
    if (!flg) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_PLIB,"MatHermitianTranspose != MatAIJ");
    ierr = MatDestroy(&F);CHKERRQ(ierr);
    ierr = MatConvert(E,MATAIJ,MAT_INPLACE_MATRIX,&E);CHKERRQ(ierr);
    ierr = MatCreateHermitianTranspose(D,&H);CHKERRQ(ierr);
    ierr = MatMultEqual(E,H,10,&flg);CHKERRQ(ierr);
    if (!flg) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_PLIB,"MatHermitianTranspose != MatAIJ");
    ierr = MatDestroy(&H);CHKERRQ(ierr);
    ierr = MatDestroy(&E);CHKERRQ(ierr);
    ierr = MatDestroy(&D);CHKERRQ(ierr);
  }
  ierr = KSPSetUp(ksp);CHKERRQ(ierr);
  ierr = MatCreateVecs(A,&b,&x);CHKERRQ(ierr);
  ierr = VecSetRandom(b,rctx);CHKERRQ(ierr);
  ierr = VecGetSubVector(b,rows[Q-1],&sub);CHKERRQ(ierr);
  ierr = VecSet(sub,0.0);CHKERRQ(ierr);
  ierr = VecRestoreSubVector(b,rows[Q-1],&sub);CHKERRQ(ierr);
  ierr = KSPSolve(ksp,b,x);CHKERRQ(ierr);
  ierr = VecDestroy(&b);CHKERRQ(ierr);
  ierr = VecDestroy(&x);CHKERRQ(ierr);
  ierr = PetscRandomDestroy(&rctx);CHKERRQ(ierr);
  ierr = MatDestroy(&A);CHKERRQ(ierr);
  ierr = KSPDestroy(&ksp);CHKERRQ(ierr);
  ierr = PetscFree(outer);CHKERRQ(ierr);
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
