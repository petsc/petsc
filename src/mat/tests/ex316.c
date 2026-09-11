static char help[] = "Tests MatISFixLocalEmpty() with a blocked local to global mapping.\n\n";

#include <petscmat.h>

int main(int argc, char **args)
{
  Mat                    A, B;
  ISLocalToGlobalMapping map, rl2g, cl2g;
  PetscScalar            v;
  PetscInt              *idxs, N, i, j, rbs, cbs, bs = 2, nl = 4;
  PetscMPIInt            rank, size;
  PetscBool              partial = PETSC_FALSE, drop, flg;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &args, NULL, help));
  PetscCallMPI(MPI_Comm_rank(PETSC_COMM_WORLD, &rank));
  PetscCallMPI(MPI_Comm_size(PETSC_COMM_WORLD, &size));
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-bs", &bs, NULL));
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-nl", &nl, NULL));
  PetscCall(PetscOptionsGetBool(NULL, NULL, "-partial", &partial, NULL));

  /* subdomains sharing a block of degrees of freedom, as in a finite element decomposition */
  PetscCall(PetscMalloc1(nl, &idxs));
  for (i = 0; i < nl; i++) idxs[i] = rank * (nl - 1) + i;
  PetscCall(ISLocalToGlobalMappingCreate(PETSC_COMM_WORLD, bs, nl, idxs, PETSC_OWN_POINTER, &map));

  N = (size * (nl - 1) + 1) * bs;
  PetscCall(MatCreateIS(PETSC_COMM_WORLD, bs, PETSC_DECIDE, PETSC_DECIDE, N, N, map, map, &A));
  PetscCall(MatISSetPreallocation(A, 3, NULL, 3, NULL));

  /* leave the last local block without any entry, so that MatISFixLocalEmpty() drops it;
     with -partial the last subdomain instead leaves a single degree of freedom out, which
     breaks the block and must make every process give up the block size
  */
  for (i = 0; i < nl * bs; i++) {
    drop = (PetscBool)(i / bs == nl - 1);
    if (partial && rank == size - 1) drop = (PetscBool)(i == nl * bs - 1);
    if (drop) continue;
    v = 2.0;
    PetscCall(MatSetValuesLocal(A, 1, &i, 1, &i, &v, ADD_VALUES));
    for (j = i - 1; j <= i + 1; j += 2) {
      if (j < 0 || j >= (partial && rank == size - 1 ? nl * bs - 1 : (nl - 1) * bs)) continue;
      v = -1.0;
      PetscCall(MatSetValuesLocal(A, 1, &i, 1, &j, &v, ADD_VALUES));
    }
  }
  PetscCall(MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY));
  PetscCall(MatConvert(A, MATAIJ, MAT_INITIAL_MATRIX, &B));

  PetscCall(MatGetLocalToGlobalMapping(A, &rl2g, &cl2g));
  PetscCall(ISLocalToGlobalMappingGetBlockSize(rl2g, &rbs));
  PetscCall(ISLocalToGlobalMappingGetBlockSize(cl2g, &cbs));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Block sizes before MatISFixLocalEmpty %" PetscInt_FMT " %" PetscInt_FMT "\n", rbs, cbs));

  PetscCall(MatISFixLocalEmpty(A, PETSC_TRUE));
  PetscCall(MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY));

  PetscCall(MatGetLocalToGlobalMapping(A, &rl2g, &cl2g));
  PetscCall(ISLocalToGlobalMappingGetBlockSize(rl2g, &rbs));
  PetscCall(ISLocalToGlobalMappingGetBlockSize(cl2g, &cbs));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Block sizes after MatISFixLocalEmpty %" PetscInt_FMT " %" PetscInt_FMT "\n", rbs, cbs));

  PetscCall(MatMultEqual(A, B, 5, &flg));
  PetscCheck(flg, PETSC_COMM_WORLD, PETSC_ERR_PLIB, "MatISFixLocalEmpty changed the operator");

  PetscCall(MatDestroy(&A));
  PetscCall(MatDestroy(&B));
  PetscCall(ISLocalToGlobalMappingDestroy(&map));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

   test:
      suffix: 1
      nsize: 1

   test:
      suffix: 2
      nsize: 3

   test:
      suffix: 1_partial
      nsize: 1
      args: -partial

   test:
      suffix: 2_partial
      nsize: 3
      args: -partial

TEST*/
