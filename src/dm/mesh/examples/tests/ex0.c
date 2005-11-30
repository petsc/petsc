static char help[] = "Tests basic ALE memory management and logging.\n\n";

#include "petscda.h"

#include <Sieve.hh>

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc, char *argv[])
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscInitialize(&argc, &argv, (char *) 0, help);CHKERRQ(ierr);
  
  MPI_Comm  comm = PETSC_COMM_WORLD;
  ierr = PetscOptionsBegin(comm, "", "Options for ALE memory management and logging testing", "Mesh");
    //PetscTruth memTest = PETSC_TRUE;
    // ierr = PetscOptionsTruth("-mem_test", "Perform the mem test", "ex0.c", PETSC_TRUE, &memTest, PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsEnd();

  ALE::Obj<ALE::Point_set> pointset;
  ALE::Obj<ALE::PreSieve>  presieve;
  ALE::Obj<ALE::Sieve>     sieve;
  ALE::Obj<ALE::Point>     point; point.create(ALE::Point(0, 0));



  ierr = PetscFinalize();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
