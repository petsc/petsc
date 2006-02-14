/*T
   Concepts: BiGraph, ParDelta
   Processors: 1
T*/

/*
  Tests TargetArrow sequence -- a sequence that wraps an array of (source,color) arrows over a given target
  and presents it as an sequence of Arrows; used in ParDelta.fusion().
  Note: may not fail in parallel, but is not designed to run that way.
*/

static char help[] = "Constructs and views test Arrow sequences involved in fusion.\n\n";

#include <Delta.hh>

typedef ALE::def::Point                                         Point;
typedef ALE::def::Arrow<Point, Point, Point>                    PointArrow;
typedef ALE::Two::TargetArrowArraySequence<Point, Point, Point> PointTargetArrowSequence;
typedef PointTargetArrowSequence::target_arrow_type             PointTargetArrow;

PetscErrorCode   testPointTargetArrowSequence();

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc, char *argv[])
{
  MPI_Comm       comm;
  PetscTruth     flag;
  PetscInt       verbosity;
  PetscTruth     sequenceOnly;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscInitialize(&argc, &argv, (char *) 0, help);CHKERRQ(ierr);
  verbosity = 1;
  ierr = PetscOptionsGetInt(PETSC_NULL, "-verbosity", &verbosity, &flag); CHKERRQ(ierr);
  comm = PETSC_COMM_WORLD;

  ierr = testPointTargetArrowSequence();                                          CHKERRQ(ierr);


  ierr = PetscFinalize();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}/* main() */

#undef  __FUNCT__
#define __FUNCT__ "testPointTargetArrowSequence"
PetscErrorCode testPointTargetArrowSequence() {
  PetscInt debug;
  PetscTruth flag;
  PetscErrorCode ierr;

  // Allocate a raw array of n PointArrows 
  int n = 10;
  ierr = PetscOptionsGetInt(PETSC_NULL, "-sequenceSize", &n, &flag); CHKERRQ(ierr);
  if(n < 0) {
    SETERRQ1(1, "Invalid PointTargetArrowSequence size: %d", n);
  }
  
  PointTargetArrow *aa = (PointTargetArrow*) malloc(sizeof(PointTargetArrow)*n);
  // Fill in the array
  for(int i = 0; i < n; i++) {
    aa[i] = PointTargetArrow(Point(i+1,i+1), Point(-(i+1),-(i+1)));
  }
  // Wrap it in a PointTargetArrowSequence with the target (0,0).
  PointTargetArrowSequence aas(Point(0,0),aa,n);

  // View the sequence
  std::cout << __FUNCT__ << ": viewing a PointTargetArrowSequence of " << aas.size() << " PointTargetArrows" << std::endl;
  if(aas.empty()) {
    std::cout << __FUNCT__ << ": sequence IS empty" << std::endl;
  }
  if(!aas.empty()) {
    std::cout << __FUNCT__ << ": sequence NOT empty" << std::endl;
  }
  std::cout << "[";
  for(PointTargetArrowSequence::iterator ai = aas.begin(); ai != aas.end(); ai++) {
    std::cout << " " << *ai;
  }
  std::cout << "]" << std::endl;

  PetscFunctionReturn(0);
}/* testPointTargetArrowSequence() */
