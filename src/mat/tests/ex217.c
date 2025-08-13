static char help[] = "Tests MatGetCurrentMemType for gpu type matrices both bound and unbound to cpu";

#include <petscmat.h>
#include <../src/mat/impls/aij/mpi/mpiaij.h>

int main(int argc, char **argv)
{
  Mat          A;
  PetscMemType memtype;
  MatType      mattype;
  PetscBool    ishypre, iskokkos, iscuda, iship;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, NULL, help));
  PetscCall(MatCreate(PETSC_COMM_WORLD, &A));
  PetscCall(MatSetSizes(A, PETSC_DECIDE, PETSC_DECIDE, 1, 1));
  PetscCall(MatSetFromOptions(A));
  PetscCall(MatGetType(A, &mattype));
  PetscCall(PetscObjectTypeCompareAny((PetscObject)A, &iscuda, MATMPIAIJCUSPARSE, MATSEQAIJCUSPARSE, ""));
  PetscCall(PetscObjectTypeCompareAny((PetscObject)A, &iship, MATMPIAIJHIPSPARSE, MATSEQAIJHIPSPARSE, ""));
  PetscCall(PetscObjectTypeCompareAny((PetscObject)A, &iskokkos, MATMPIAIJKOKKOS, MATSEQAIJKOKKOS, ""));
  PetscCall(PetscObjectTypeCompare((PetscObject)A, MATHYPRE, &ishypre));
#if defined(PETSC_HAVE_HYPRE)
  PetscCall(MatHYPRESetPreallocation(A, 1, NULL, 1, NULL));
#endif
  PetscCall(MatSeqAIJSetPreallocation(A, 1, NULL));
  PetscCall(MatMPIAIJSetPreallocation(A, 1, NULL, 1, NULL));

  PetscCall(MatGetCurrentMemType(A, &memtype));
  if (iscuda) PetscCheck(memtype == PETSC_MEMTYPE_CUDA, PETSC_COMM_WORLD, PETSC_ERR_PLIB, "wrong memory type");
  else if (iship) PetscCheck(memtype == PETSC_MEMTYPE_HIP, PETSC_COMM_WORLD, PETSC_ERR_PLIB, "wrong memory type");
  else if (iskokkos) PetscCheck(memtype == PETSC_MEMTYPE_KOKKOS, PETSC_COMM_WORLD, PETSC_ERR_PLIB, "wrong memory type");
  else if (ishypre) PetscCheck(PetscDefined(HAVE_HYPRE_DEVICE) ? memtype == PETSC_MEMTYPE_DEVICE : memtype == PETSC_MEMTYPE_HOST, PETSC_COMM_WORLD, PETSC_ERR_PLIB, "wrong memory type");
  else PetscCheck(memtype == PETSC_MEMTYPE_HOST, PETSC_COMM_WORLD, PETSC_ERR_PLIB, "wrong memory type");

  // Kokkos doesn't currently implement MatBindToCPU
  if (!iskokkos) {
    PetscCall(MatBindToCPU(A, PETSC_TRUE));
    PetscCall(MatGetCurrentMemType(A, &memtype));
    PetscCheck(memtype == PETSC_MEMTYPE_HOST, PETSC_COMM_WORLD, PETSC_ERR_PLIB, "wrong memory type");
  }
  PetscCall(MatDestroy(&A));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

   test:
     suffix: seqaij
     args: -mat_type aij
     output_file: output/empty.out

   test:
     suffix: mpiaij
     nsize: 2
     args: -mat_type aij
     output_file: output/empty.out

   test:
     requires: cuda
     suffix: seqaijcusparse
     args: -mat_type aijcusparse
     output_file: output/empty.out

   test:
     requires: cuda
     suffix: mpiaijcusparse
     nsize: 2
     args: -mat_type aijcusparse
     output_file: output/empty.out

   test:
     requires: hip
     suffix: seqaijhipsparse
     args: -mat_type aijhipsparse
     output_file: output/empty.out

   test:
     requires: hip
     suffix: mpiaijhipsparse
     nsize: 2
     args: -mat_type aijhipsparse
     output_file: output/empty.out

   test:
     requires: kokkos_kernels
     suffix: seqaijkokkos
     args: -mat_type aijkokkos
     output_file: output/empty.out

   test:
     requires: kokkos_kernels
     suffix: mpiaijkokkos
     nsize: 2
     args: -mat_type aijkokkos
     output_file: output/empty.out

   test:
     requires: hypre
     suffix: hypre
     args: -mat_type hypre
     output_file: output/empty.out

   test:
     requires: hypre
     suffix: hypre_parallel
     nsize: 2
     args: -mat_type hypre
     output_file: output/empty.out

TEST*/
