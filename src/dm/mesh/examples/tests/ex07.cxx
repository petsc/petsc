/*T
   Concepts: ArrowContainer, Predicate
   Processors: 1
T*/

/*
  Tests Predicate-enabled ArrowContainers.
*/

static char help[] = "Constructs and views test Predicate-enabled ArrowContainers.\n\n";

#include <Predicate.hh>

PetscErrorCode testWindowedArrowContainer();

typedef ALE::X::SifterDef::ArrowContainer<ALE::X::UnicolorArrowSet> WindowedArrowContainer;

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc, char *argv[])
{
  MPI_Comm       comm;
  PetscTruth     flag;
  PetscInt       verbosity;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscInitialize(&argc, &argv, (char *) 0, help);CHKERRQ(ierr);
  verbosity = 1;
  ierr = PetscOptionsGetInt(PETSC_NULL, "-verbosity", &verbosity, &flag);CHKERRQ(ierr);
  comm = PETSC_COMM_WORLD;
  ierr = testWindowedArrowContainer();                                   CHKERRQ(ierr);


  ierr = PetscFinalize();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}/* main() */


#undef __FUNCT__
#define __FUNCT__ "testWindowedArrowContainer"
PetscErrorCode testWindowedArrowContainer()
{
  PetscInt       verbosity;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  verbosity = 1;
  ierr = PetscOptionsGetInt(PETSC_NULL, "-verbosity", &verbosity, &flag);CHKERRQ(ierr);

  WindowedArrowContainer wac;

  
  


  PetscFunctionReturn(0);
}/* testWindowedArrowContainer() */


