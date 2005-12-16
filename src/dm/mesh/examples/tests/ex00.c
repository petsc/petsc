static char help[] = "Tests basic ALE memory management and logging.\n\n";

#include "petscda.h"

#include <Sieve.hh>

#undef __FUNCT__
#define __FUNCT__ "MemTest"
PetscErrorCode MemTest()
{
  ALE::Obj<ALE::Point_set> pointset;
  ALE::Obj<ALE::PreSieve>  presieve;
  ALE::Obj<ALE::Sieve>     sieve;
  ALE::Obj<ALE::Point>     point;
  ALE::Obj<ALE::Point>     point2(ALE::Point(0, 1));

  PetscFunctionBegin;
  point.create(ALE::Point(0, 0));
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "ConeTest"
PetscErrorCode ConeTest()
{
  ALE::Obj<ALE::def::Sieve> sieve;
  ALE::def::Point           base(0, -1);

  PetscFunctionBegin;
  for(int i = 0; i < 1; i++) {
    ALE::def::Point point(0, i);

    sieve.addCone(point);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc, char *argv[])
{
  PetscTruth     memTest, coneTest;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscInitialize(&argc, &argv, (char *) 0, help);CHKERRQ(ierr);
  
  MPI_Comm  comm = PETSC_COMM_WORLD;
  ierr = PetscOptionsBegin(comm, "", "Options for ALE memory management and logging testing", "Mesh");
    memTest = PETSC_TRUE;
    ierr = PetscOptionsTruth("-mem_test", "Perform the mem test", "ex0.c", PETSC_TRUE, &memTest, PETSC_NULL);CHKERRQ(ierr);
    coneTest = PETSC_TRUE;
    ierr = PetscOptionsTruth("-cone_test", "Perform the cone test", "ex0.c", PETSC_TRUE, &coneTest, PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsEnd();
  if (memTest) {
    ierr = MemTest();CHKERRQ(ierr);
  }
  if (coneTest) {
    ierr = ConeTest();CHKERRQ(ierr);
  }
  ierr = PetscFinalize();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
