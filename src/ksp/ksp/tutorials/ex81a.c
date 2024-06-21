#include <petsc.h>

static char help[] = "Solves a linear system with a MatNest and PCFIELDSPLIT with fields defined from the command line.\n\n";
/* similar to ex81.c except the PCFIELDSPLIT fields are defined from the command line with fields instead of hardwired IS from the MATNEST */

#define Q 5 /* everything is hardwired for a 5x5 MatNest for now */

int main(int argc, char **args)
{
  KSP         ksp;
  PC          pc;
  Mat         array[Q * Q], A, a;
  Vec         b, x, sub;
  IS          rows[Q];
  PetscInt    i, M, N;
  PetscMPIInt size;
  PetscRandom rctx;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &args, NULL, help));
  PetscCallMPI(MPI_Comm_size(PETSC_COMM_WORLD, &size));
  PetscCall(PetscRandomCreate(PETSC_COMM_WORLD, &rctx));
  PetscCall(KSPCreate(PETSC_COMM_WORLD, &ksp));
  size = PetscMax(3, size);
  for (i = 0; i < Q * Q; ++i) array[i] = NULL;
  for (i = 0; i < Q; ++i) {
    if (i == 0) {
      PetscCall(MatCreateAIJ(PETSC_COMM_WORLD, PETSC_DECIDE, PETSC_DECIDE, size, size, 1, NULL, 0, NULL, array + (Q + 1) * i));
    } else if (i == 1 || i == 3) {
      PetscCall(MatCreateSBAIJ(PETSC_COMM_WORLD, 2, PETSC_DECIDE, PETSC_DECIDE, size, size, 1, NULL, 0, NULL, array + (Q + 1) * i));
    } else if (i == 2 || i == 4) {
      PetscCall(MatCreateBAIJ(PETSC_COMM_WORLD, 2, PETSC_DECIDE, PETSC_DECIDE, size, size, 1, NULL, 0, NULL, array + (Q + 1) * i));
    }
    PetscCall(MatAssemblyBegin(array[(Q + 1) * i], MAT_FINAL_ASSEMBLY));
    PetscCall(MatAssemblyEnd(array[(Q + 1) * i], MAT_FINAL_ASSEMBLY));
    PetscCall(MatShift(array[(Q + 1) * i], 100 + i + 1));
    if (i == 3) {
      PetscCall(MatDuplicate(array[(Q + 1) * i], MAT_COPY_VALUES, &a));
      PetscCall(MatDestroy(array + (Q + 1) * i));
      PetscCall(MatCreateHermitianTranspose(a, array + (Q + 1) * i));
      PetscCall(MatDestroy(&a));
    }
    size *= 2;
  }
  PetscCall(MatGetSize(array[0], &M, NULL));
  for (i = 2; i < Q; ++i) {
    PetscCall(MatGetSize(array[(Q + 1) * i], NULL, &N));
    if (i != Q - 1) {
      PetscCall(MatCreateAIJ(PETSC_COMM_WORLD, PETSC_DECIDE, PETSC_DECIDE, i == 3 ? N : M, i == 3 ? M : N, 0, NULL, 0, NULL, array + i));
    } else {
      PetscCall(MatCreateDense(PETSC_COMM_WORLD, PETSC_DECIDE, PETSC_DECIDE, M, N, NULL, array + i));
    }
    PetscCall(MatAssemblyBegin(array[i], MAT_FINAL_ASSEMBLY));
    PetscCall(MatAssemblyEnd(array[i], MAT_FINAL_ASSEMBLY));
    PetscCall(MatSetRandom(array[i], rctx));
    if (i == 3) {
      PetscCall(MatDuplicate(array[i], MAT_COPY_VALUES, &a));
      PetscCall(MatDestroy(array + i));
      PetscCall(MatCreateHermitianTranspose(a, array + i));
      PetscCall(MatDestroy(&a));
    }
  }
  PetscCall(MatGetSize(array[0], NULL, &N));
  for (i = 2; i < Q; i += 2) {
    PetscCall(MatGetSize(array[(Q + 1) * i], &M, NULL));
    if (i != Q - 1) {
      PetscCall(MatCreateAIJ(PETSC_COMM_WORLD, PETSC_DECIDE, PETSC_DECIDE, M, N, 2, NULL, 2, NULL, array + Q * i));
    } else {
      PetscCall(MatCreateDense(PETSC_COMM_WORLD, PETSC_DECIDE, PETSC_DECIDE, N, M, NULL, array + Q * i));
    }
    PetscCall(MatAssemblyBegin(array[Q * i], MAT_FINAL_ASSEMBLY));
    PetscCall(MatAssemblyEnd(array[Q * i], MAT_FINAL_ASSEMBLY));
    PetscCall(MatSetRandom(array[Q * i], rctx));
    if (i == Q - 1) {
      PetscCall(MatDuplicate(array[Q * i], MAT_COPY_VALUES, &a));
      PetscCall(MatDestroy(array + Q * i));
      PetscCall(MatCreateHermitianTranspose(a, array + Q * i));
      PetscCall(MatDestroy(&a));
    }
  }
  PetscCall(MatGetSize(array[(Q + 1) * 3], &M, NULL));
  for (i = 1; i < 3; ++i) {
    PetscCall(MatGetSize(array[(Q + 1) * i], NULL, &N));
    PetscCall(MatCreateAIJ(PETSC_COMM_WORLD, PETSC_DECIDE, PETSC_DECIDE, M, N, 2, NULL, 2, NULL, array + Q * 3 + i));
    PetscCall(MatAssemblyBegin(array[Q * 3 + i], MAT_FINAL_ASSEMBLY));
    PetscCall(MatAssemblyEnd(array[Q * 3 + i], MAT_FINAL_ASSEMBLY));
    PetscCall(MatSetRandom(array[Q * 3 + i], rctx));
  }
  PetscCall(MatGetSize(array[(Q + 1) * 1], NULL, &N));
  PetscCall(MatGetSize(array[(Q + 1) * (Q - 1)], &M, NULL));
  PetscCall(MatCreateBAIJ(PETSC_COMM_WORLD, 2, PETSC_DECIDE, PETSC_DECIDE, M, N, 0, NULL, 0, NULL, &a));
  PetscCall(MatAssemblyBegin(a, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(a, MAT_FINAL_ASSEMBLY));
  PetscCall(MatCreateHermitianTranspose(a, array + Q + Q - 1));
  PetscCall(MatDestroy(&a));
  PetscCall(MatDestroy(array + Q * Q - 1));
  PetscCall(MatCreateNest(PETSC_COMM_WORLD, Q, NULL, Q, NULL, array, &A));
  for (i = 0; i < Q; ++i) PetscCall(MatDestroy(array + (Q + 1) * i));
  for (i = 2; i < Q; ++i) {
    PetscCall(MatDestroy(array + i));
    PetscCall(MatDestroy(array + Q * i));
  }
  for (i = 1; i < 3; ++i) PetscCall(MatDestroy(array + Q * 3 + i));
  PetscCall(MatDestroy(array + Q + Q - 1));
  PetscCall(KSPSetOperators(ksp, A, A));
  PetscCall(MatNestGetISs(A, rows, NULL));
  PetscCall(KSPGetPC(ksp, &pc));
  PetscCall(PCSetType(pc, PCFIELDSPLIT));
  PetscCall(KSPSetFromOptions(ksp));
  PetscCall(MatCreateVecs(A, &b, &x));
  PetscCall(VecSetRandom(b, rctx));
  PetscCall(VecGetSubVector(b, rows[Q - 1], &sub));
  PetscCall(VecSet(sub, 0.0));
  PetscCall(VecRestoreSubVector(b, rows[Q - 1], &sub));
  PetscCall(KSPSolve(ksp, b, x));
  PetscCall(VecDestroy(&b));
  PetscCall(VecDestroy(&x));
  PetscCall(PetscRandomDestroy(&rctx));
  PetscCall(MatDestroy(&A));
  PetscCall(KSPDestroy(&ksp));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

   test:
      nsize: 3
      suffix: 1
      requires: !complex !single
      filter: sed  -e "s/CONVERGED_ATOL/CONVERGED_RTOL/g" -e "s/iterations [4-5]/iterations 4/g" -e "s/hermitiantranspose/transpose/g"
      args: -pc_fieldsplit_0_fields 0,1 -pc_fieldsplit_1_fields 2,3 -pc_fieldsplit_2_fields 4 -pc_type fieldsplit -ksp_converged_reason -fieldsplit_pc_type jacobi -ksp_view

   test:
      suffix: 2
      nsize: 3
      requires: !complex !single
      filter: sed  -e "s/CONVERGED_ATOL/CONVERGED_RTOL/g" -e "s/iterations [4-5]/iterations 4/g" -e "s/hermitiantranspose/transpose/g"
      args: -pc_type fieldsplit -ksp_converged_reason -fieldsplit_pc_type jacobi -ksp_view

TEST*/
