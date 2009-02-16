/*T
   Concepts: BiGraph, ParDelta, wrapping
   Processors: 1
T*/

/*
  Tests ParDelta::ConeArraySequence -- a sequence that wraps an array of (source,color) "arrows" over a given target
  and presents it as an sequence of Arrows; used in ParDelta::fusion().
  Note: this test may not fail in parallel, but is not designed to run that way.

  Tests ParDelta::Flip  -- a class that wraps a BiGraph, implements a subset of the BiGraph interface  and redirects 
  select methods to the underlying BiGraph while reversing the input arrows.
*/

static char help[] = "Constructs and views test ParDelta::ConeArraySequences and ParDelta::Flip involved in ParDelta.\n\n";

#include <ParDelta.hh>

typedef ALE::def::Point                                   Point;
typedef ALE::Two::Arrow<Point, Point, Point>              PointArrow;
typedef ALE::Two::ConeArraySequence<PointArrow>           PointConeArraySequence;
typedef PointConeArraySequence::cone_arrow_type           PointConeArrow;

PetscErrorCode   testPointConeArraySequence();
PetscErrorCode   viewPointConeArraySequence(PointConeArraySequence& seq, const char* label = NULL);

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
  ierr = testPointConeArraySequence();                                   CHKERRQ(ierr);


  ierr = PetscFinalize();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}/* main() */

#undef  __FUNCT__
#define __FUNCT__ "testPointConeArraySequence"
PetscErrorCode testPointConeArraySequence() {
  PetscTruth flag;
  PetscErrorCode ierr;

  // Allocate a raw array of n PointConeArrows 
  int n = 10;
  ierr = PetscOptionsGetInt(PETSC_NULL, "-sequenceSize", &n, &flag);CHKERRQ(ierr);
  if(n < 0) {
    SETERRQ1(1, "Invalid PointConeArraySequence size: %d", n);
  }
  
  PointConeArrow *aa = (PointConeArrow*) malloc(sizeof(PointConeArrow)*n);

  // Fill in the array
  for(int i = 0; i < n; i++) {
    aa[i] = PointConeArrow(Point(i+1,i+1), Point(-(i+1),-(i+1)));
  }
  // Wrap it in a PointConeArraySequence with the target (0,0).
  PointConeArraySequence aas(aa,n,Point(0,0));

  // View the sequence
  ierr = viewPointConeArraySequence(aas, "'Manual'");CHKERRQ(ierr);

  // Fill in the array using the 'PointConeArrow::place' method
  for(int i = 0; i < n; i++) {
    PointArrow a(Point(i+1,i+1), Point(0,0), Point(-(i+1),-(i+1)));
    PointConeArraySequence::cone_arrow_type::place(aa+i,a);
  }
  // Wrap it in a PointConeArraySequence with the target (0,0).
  aas = PointConeArraySequence(aa,n,Point(0,0));

  // View the sequence
  ierr = viewPointConeArraySequence(aas, "'Auto'");CHKERRQ(ierr);

  PetscFunctionReturn(0);
}/* testPointConeArraySequence() */


#undef  __FUNCT__
#define __FUNCT__ "viewPointConeArraySequence"
PetscErrorCode viewPointConeArraySequence(PointConeArraySequence& aas, const char* label) {
  PetscFunctionBegin;

  std::cout << __FUNCT__ << ": viewing a PointConeArraySequence ";
  if(label != NULL) {
    std::cout << label;
  }
  std::cout << " of " << aas.size() << " PointConeArrows" << std::endl;
  if(aas.empty()) {
    std::cout << __FUNCT__ << ": sequence IS empty" << std::endl;
  }
  if(!aas.empty()) {
    std::cout << __FUNCT__ << ": sequence NOT empty" << std::endl;
  }
  std::cout << "[";
  for(PointConeArraySequence::iterator ai = aas.begin(); ai != aas.end(); ai++) {
    std::cout << " " << *ai;
  }
  std::cout << "]" << std::endl;

  PetscFunctionReturn(0);
}/* viewPointConeArraySequence() */
