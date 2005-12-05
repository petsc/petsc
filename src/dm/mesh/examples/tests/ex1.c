static char help[] = "Tests basic Sieve operations.\n\n";

#include "petscda.h"

#include <IndexBundle.hh>

#undef __FUNCT__
#define __FUNCT__ "PreSieveConeTest"
PetscErrorCode PreSieveConeTest(MPI_Comm comm)
{
  ALE::Obj<ALE::PreSieve> topology(new ALE::PreSieve(comm));
  ALE::Point              vertexA(0, 0);
  ALE::Point              vertexB(0, 1);
  ALE::Point              edge(0, 2);

  PetscFunctionBegin;
  topology->addCone(vertexA, edge);
  topology->addCone(vertexB, edge);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SieveConeTest"
PetscErrorCode SieveConeTest(MPI_Comm comm)
{
  ALE::Obj<ALE::Sieve> topology;
  ALE::Point           vertexA(0, 0);
  ALE::Point           vertexB(0, 1);
  ALE::Point           edge(0, 2);

  PetscFunctionBegin;
  topology.create(ALE::Sieve(comm));
  topology->addCone(vertexA, edge);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc, char *argv[])
{
  MPI_Comm       comm;
  PetscTruth     coneTests;
  PetscTruth     presieveTests;
  PetscTruth     sieveTests;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscInitialize(&argc, &argv, (char *) 0, help);CHKERRQ(ierr);
  ierr = PetscOptionsBegin(comm, "", "Options for Sieve testing", "Mesh");
    presieveTests = PETSC_TRUE;
    ierr = PetscOptionsTruth("-presieve_tests", "Perform the PreSieve test", "ex1.c", PETSC_TRUE, &presieveTests, PETSC_NULL);CHKERRQ(ierr);
    sieveTests = PETSC_TRUE;
    ierr = PetscOptionsTruth("-sieve_tests", "Perform the Sieve test", "ex1.c", PETSC_TRUE, &sieveTests, PETSC_NULL);CHKERRQ(ierr);
    coneTests = PETSC_TRUE;
    ierr = PetscOptionsTruth("-cone_tests", "Perform the cone test", "ex1.c", PETSC_TRUE, &coneTests, PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsEnd();
  comm = PETSC_COMM_WORLD;
  if (presieveTests && coneTests) {
    ierr = PreSieveConeTest(comm);CHKERRQ(ierr);
  }
  if (sieveTests && coneTests) {
    ierr = SieveConeTest(comm);CHKERRQ(ierr);
  }
  ierr = PetscFinalize();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
