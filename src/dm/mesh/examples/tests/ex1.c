static char help[] = "Tests basic Sieve operations.\n\n";

#include "petscda.h"

#include <IndexBundle.hh>

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc, char *argv[])
{
  MPI_Comm       comm;
  ALE::Sieve    *topology;
  ALE::Point     vertexA(0, 0);
  ALE::Point     vertexB(0, 1);
  ALE::Point     edge(0, 2);
  PetscTruth     coneTest;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscInitialize(&argc, &argv, (char *) 0, help);CHKERRQ(ierr);
  ierr = PetscOptionsBegin(comm, "", "Options for Sieve testing", "Mesh");
    coneTest = PETSC_TRUE;
    ierr = PetscOptionsTruth("-cone_test", "Perform the cone test", "ex1.c", PETSC_TRUE, &coneTest, PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsEnd();
  comm = PETSC_COMM_WORLD;

  topology = new ALE::Sieve(comm);
  if (coneTest) {
    topology->addCone(vertexA, edge);
  }
  delete topology;

  ierr = PetscFinalize();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
