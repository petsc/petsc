/*T
   Concepts: BiGraph, ParDelta, Flip.
   Processors: 1
T*/

/*
  Tests ParDelta::Flip  -- a class that wraps a BiGraph, implements a subset of the BiGraph interface  and redirects 
  select methods to the underlying BiGraph while reversing the input arrows.
*/

static char help[] = "Constructs and views a test BiGraph and then wraps it in a ParDelta::Flip and views the Flip.\n\n";

#include <ParDelta.hh>

typedef ALE::Two::BiGraph<int,ALE::def::Point,int>    PointBiGraph;
typedef ALE::Two::ParSupportDelta<PointBiGraph>       PointParDelta;
typedef ALE::Two::Flip<PointBiGraph>                  PointFlip;

PetscErrorCode   testHatFlip();
void viewFlip(const ALE::Obj<PointFlip>& flip, const char* name);

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

  ierr = testHatFlip();CHKERRQ(ierr);

  ierr = PetscFinalize();CHKERRQ(ierr);
  PetscFunctionReturn(0);

}/* main() */

#undef  __FUNCT__
#define __FUNCT__ "testHat"
PetscErrorCode testHatFlip() {
  PetscErrorCode ierr;
  int debug;
  PetscTruth flag;
  MPI_Comm comm = PETSC_COMM_SELF;
  PetscFunctionBegin;

  debug = 0;
  ierr = PetscOptionsGetInt(PETSC_NULL, "-debug", &debug, &flag);CHKERRQ(ierr);
  ierr = PetscPrintf(comm, "%s: using debug value of %d\n", __FUNCT__, debug);CHKERRQ(ierr);

  // Construct an empty hat bigraph
  ALE::Obj<PointBiGraph> hat = PointBiGraph(comm, debug);

  // Flip the hat
  ALE::Obj<PointFlip> flip = PointFlip(hat);

  // Add arrows to the flipped hat
  // Add three arrows from a single cap point 0 to global points (-1,i) for i = 0,1,2 with color 1
  for(int i = 0; i < 3; i++) {
    flip->addArrow(ALE::def::Point(-1,i), 0, 1);
  }

  // View the flip
  viewFlip(flip, "Hat flip");

  // View the hat
  hat->view("Hat bigraph");


  PetscFunctionReturn(0);
}/* testBiGraphHat() */


#undef  __FUNCT__
#define __FUNCT__ "viewFlip"
void viewFlip(const ALE::Obj<PointFlip>& flip, const char* name) {
  
  // View the cones for all base points
  std::cout << name << " cones:" << std::endl;
  PointFlip::traits::baseSequence base = flip->base();
  for(PointFlip::traits::baseSequence::traits::iterator i = base.begin(); i != base.end(); i++) {
    PointFlip::traits::coneSequence cone = flip->cone(*i);
    std::cout << *i << ": ";
    cone.view(std::cout, true); 
  }

  // View the supports for all cap points
  std::cout << name << " supports:" << std::endl;
  PointFlip::traits::capSequence cap = flip->cap();
  for(PointFlip::traits::capSequence::traits::iterator i = cap.begin(); i != cap.end(); i++) {
    PointFlip::traits::supportSequence supp = flip->support(*i);
    std::cout << *i << ": ";
    supp.view(std::cout, true); 
  }
}/* viewConesAndSupports() */

