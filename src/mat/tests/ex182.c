static char help[] = "Tests using MatShift() to create a constant diagonal matrix\n\n";

#include <petscmat.h>

int main(int argc, char **argv)
{
  Mat           A, F;
  MatFactorInfo info;
  PetscInt      m = 10;
  IS            perm;
  PetscMPIInt   size;
  PetscBool     issbaij;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, (char *)0, help));
  PetscCallMPI(MPI_Comm_size(PETSC_COMM_WORLD, &size));

  PetscCall(MatCreate(PETSC_COMM_WORLD, &A));
  PetscCall(MatSetSizes(A, PETSC_DECIDE, PETSC_DECIDE, m, m));
  PetscCall(MatSetFromOptions(A));
  PetscCall(MatSetUp(A));
  PetscCall(MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY));

  PetscCall(MatShift(A, 1.0));

  PetscCall(PetscObjectTypeCompare((PetscObject)A, MATSEQSBAIJ, &issbaij));
  if (size == 1 && !issbaij) {
    PetscCall(MatGetFactor(A, MATSOLVERPETSC, MAT_FACTOR_LU, &F));
    PetscCall(MatFactorInfoInitialize(&info));
    PetscCall(ISCreateStride(PETSC_COMM_SELF, m, 0, 1, &perm));
    PetscCall(MatLUFactorSymbolic(F, A, perm, perm, &info));
    PetscCall(MatLUFactorNumeric(F, A, &info));
    PetscCall(MatDestroy(&F));
    PetscCall(ISDestroy(&perm));
  }
  PetscCall(MatDestroy(&A));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

   test:
      requires: defined(PETSC_USE_INFO)
      args: -info
      filter: grep malloc | sort -b

   test:
      suffix: 2
      nsize: 2
      requires: defined(PETSC_USE_INFO)
      args: -info ex182info
      filter: grep -h malloc "ex182info.0" | sort -b

   test:
      suffix: 3
      requires: defined(PETSC_USE_INFO)
      args: -info -mat_type baij
      filter: grep malloc | sort -b

   test:
      suffix: 4
      nsize: 2
      requires: defined(PETSC_USE_INFO)
      args: -info ex182info -mat_type baij
      filter: grep -h malloc "ex182info.1" | sort -b

   test:
      suffix: 5
      requires: defined(PETSC_USE_INFO)
      args: -info -mat_type sbaij
      filter: grep malloc | sort  -b

   test:
      suffix: 6
      nsize: 2
      requires: defined(PETSC_USE_INFO)
      args: -info ex182info -mat_type sbaij
      filter: grep -h malloc "ex182info.0" | sort -b

   test:
     suffix: 7
     nsize: 1
     requires: defined(PETSC_USE_INFO)
     args: -info ex182info
     filter: grep -h malloc "ex182info.0" | grep -v Running | sort -b

   test:
     suffix: 8
     nsize: 2
     requires: defined(PETSC_USE_INFO)
     args: -info ex182info:mat
     filter: grep -h malloc "ex182info.1" | sort -b

   test:
     suffix: 9
     nsize: 1
     requires: defined(PETSC_USE_INFO)
     args: -info ex182info:sys
     filter: grep -h -ve Running -ve MPI_Comm -ve Initialize -ve communicator -ve HostName -ve PetscDetermineInitialFPTrap -ve libpetscbamg "ex182info.0" | sort -b

   test:
     suffix: 10
     nsize: 1
     requires: defined(PETSC_USE_INFO)
     args: -info :~sys
     filter: grep -h malloc | sort -b

   test:
     suffix: 11
     nsize: 2
     requires: defined(PETSC_USE_INFO)
     args: -info :~sys,mat
     filter: sort -b

   test:
     suffix: 12
     nsize: 2
     requires: defined(PETSC_USE_INFO)
     args: -info ex182info:sys,mat
     filter: grep -h -ve Running -ve MPI_Comm -ve Initialize -ve communicator -ve HostName -ve PetscDetermineInitialFPTrap -ve libpetscbamg "ex182info.1" | sort -b

   test:
     suffix: 13
     nsize: 2
     requires: defined(PETSC_USE_INFO)
     args: -info ex182info:mat:~self
     filter: grep -h "ex182info.1" | sort -b

   test:
     suffix: 14
     nsize: 2
     requires: defined(PETSC_USE_INFO)
     args: -info ex182info::~self
     filter: grep -h -ve Running -ve MPI_Comm -ve Initialize -ve communicator -ve HostName -ve PetscDetermineInitialFPTrap "ex182info.1" | sort -b

   test:
     suffix: 15
     nsize: 2
     requires: defined(PETSC_USE_INFO)
     args: -info ex182info::self
     filter: grep -h -ve Running -ve MPI_Comm -ve Initialize -ve communicator -ve HostName -ve PetscDetermineInitialFPTrap -ve libpetscbamg "ex182info.1" | sort -b

TEST*/
