static char help[] = "Tests using MatShift() to create a constant diagonal matrix\n\n";

#include <petscmat.h>

int main(int argc,char **argv)
{
  Mat            A,F;
  MatFactorInfo  info;
  PetscErrorCode ierr;
  PetscInt       m = 10;
  IS             perm;
  PetscMPIInt    size;
  PetscBool      issbaij;

  ierr = PetscInitialize(&argc,&argv,(char*) 0,help);if (ierr) return ierr;
  ierr = MPI_Comm_size(PETSC_COMM_WORLD,&size);CHKERRQ(ierr);

  ierr = MatCreate(PETSC_COMM_WORLD,&A);CHKERRQ(ierr);
  ierr = MatSetSizes(A,PETSC_DECIDE,PETSC_DECIDE,m,m);CHKERRQ(ierr);
  ierr = MatSetFromOptions(A);CHKERRQ(ierr);
  ierr = MatSetUp(A);CHKERRQ(ierr);
  ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

  ierr = MatShift(A,1.0);CHKERRQ(ierr);

  ierr = PetscObjectTypeCompare((PetscObject)A,MATSEQSBAIJ,&issbaij);CHKERRQ(ierr);
  if (size == 1 && !issbaij) {
    ierr = MatGetFactor(A,MATSOLVERPETSC,MAT_FACTOR_LU,&F);CHKERRQ(ierr);
    ierr = MatFactorInfoInitialize(&info);CHKERRQ(ierr);
    ierr = ISCreateStride(PETSC_COMM_SELF,m,0,1,&perm);CHKERRQ(ierr);
    ierr = MatLUFactorSymbolic(F,A,perm,perm,&info);CHKERRQ(ierr);
    ierr = MatLUFactorNumeric(F,A,&info);CHKERRQ(ierr);
    ierr = MatDestroy(&F);CHKERRQ(ierr);
    ierr = ISDestroy(&perm);CHKERRQ(ierr);
  }
  ierr = MatDestroy(&A);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
}

/*TEST

   test:
      requires: define(PETSC_USE_INFO)
      args: -info
      filter: grep malloc | sort -b

   test:
      suffix: 2
      nsize: 2
      requires: define(PETSC_USE_INFO)
      args: -info ex182info
      filter: grep -h malloc "ex182info.0" | sort -b

   test:
      suffix: 3
      requires: define(PETSC_USE_INFO)
      args: -info -mat_type baij
      filter: grep malloc | sort -b

   test:
      suffix: 4
      nsize: 2
      requires: define(PETSC_USE_INFO)
      args: -info ex182info -mat_type baij
      filter: grep -h malloc "ex182info.1" | sort -b

   test:
      suffix: 5
      requires: define(PETSC_USE_INFO)
      args: -info -mat_type sbaij
      filter: grep malloc | sort  -b

   test:
      suffix: 6
      nsize: 2
      requires: define(PETSC_USE_INFO)
      args: -info ex182info -mat_type sbaij
      filter: grep -h malloc "ex182info.0" | sort -b

   test:
     suffix: 7
     nsize: 1
     requires: define(PETSC_USE_INFO)
     args: -info ex182info
     filter: grep -h malloc "ex182info.0" | grep -v Running | sort -b

   test:
     suffix: 8
     nsize: 2
     requires: define(PETSC_USE_INFO)
     args: -info ex182info:mat
     filter: grep -h malloc "ex182info.1" | sort -b

   test:
     suffix: 9
     nsize: 1
     requires: define(PETSC_USE_INFO)
     args: -info ex182info:sys
     filter: grep -h -ve Running -ve MPI_Comm -ve Initialize -ve communicator -ve HostName -ve PetscDetermineInitalFPTrap "ex182info.0" | sort -b

   test:
     suffix: 10
     nsize: 1
     requires: define(PETSC_USE_INFO)
     args: -info :~sys
     filter: grep -h malloc | sort -b

   test:
     suffix: 11
     nsize: 2
     requires: define(PETSC_USE_INFO)
     args: -info :~sys,mat
     filter: sort -b

   test:
     suffix: 12
     nsize: 2
     requires: define(PETSC_USE_INFO)
     args: -info ex182info:sys,mat
     filter: grep -h -ve Running -ve MPI_Comm -ve Initialize -ve communicator -ve HostName -ve PetscDetermineInitalFPTrap "ex182info.1" | sort -b

   test:
     suffix: 13
     nsize: 2
     requires: define(PETSC_USE_INFO)
     args: -info ex182info:mat:~self
     filter: grep -h "ex182info.1" | sort -b

   test:
     suffix: 14
     nsize: 2
     requires: define(PETSC_USE_INFO)
     args: -info ex182info::~self
     filter: grep -h -ve Running -ve MPI_Comm -ve Initialize -ve communicator -ve HostName -ve PetscDetermineInitalFPTrap "ex182info.1" | sort -b

   test:
     suffix: 15
     nsize: 2
     requires: define(PETSC_USE_INFO)
     args: -info ex182info::self
     filter: grep -h -ve Running -ve MPI_Comm -ve Initialize -ve communicator -ve HostName -ve PetscDetermineInitalFPTrap "ex182info.1" | sort -b

TEST*/
