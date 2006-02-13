/*T
   Concepts: BiGraph
   Processors: multiple
T*/

/*
  Create a series of parallel BiGraphs suitable for testing the Delta routines .
*/

static char help[] = "Constructs a series of parallel BiGraphs and performs Delta routines.\n\n";

#include <Delta.hh>
#include <ALE.hh>


typedef ALE::Two::BiGraph<int,ALE::Two::Rec<int>,ALE::def::Point,ALE::Two::Rec<ALE::def::Point>,int> PointBiGraph;
typedef ALE::Two::RightConeDuplicationFuser<PointBiGraph,PointBiGraph>    PointConeFuser;
typedef ALE::Two::ParDelta<PointBiGraph,PointConeFuser>                   PointParDelter;
typedef PointParDelter::overlap_type                                      PointOverlap;
typedef PointParDelter::delta_type                                        PointConeDelta;

PetscErrorCode   testBiGraphHat(MPI_Comm comm);
void             viewConesAndSupports(const ALE::Obj<PointBiGraph>& bg, const char* name);

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
  ierr = PetscOptionsGetInt(PETSC_NULL, "-verbosity", &verbosity, &flag); CHKERRQ(ierr);
  comm = PETSC_COMM_WORLD;

  ierr = testBiGraphHat(comm); CHKERRQ(ierr);

  ierr = PetscFinalize();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}/* main() */

#undef  __FUNCT__
#define __FUNCT__ "testBiGraphHat"
PetscErrorCode testBiGraphHat(MPI_Comm comm) {
  int rank;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  ALE::Obj<PointBiGraph> bg = PointBiGraph(comm);

  ierr = MPI_Comm_rank(comm, &rank); CHKERRQ(ierr);
  
  // Add three arrows from a single cap point rank to global points with the indices 2*rank, 2*rank+1, 2*rank+2 
  for(int i = 0; i < 3; i++) {
    bg->addArrow(rank, ALE::def::Point(-1,2*rank+i), -rank);
  }
  
  // View
  bg->view(std::cout, "Hat bigraph");

  // View cones and supports
  viewConesAndSupports(bg, "Hat bigraph");
  
  // Construct a Delta object and a base overlap object
  PointParDelter delter(bg, 0);
  ALE::Obj<PointOverlap>   overlap = delter.overlap();
  // View
  overlap->view(std::cout, "Hat overlap");

  ALE::Obj<PointConeDelta> delta   = delter.delta(overlap);
  // View
  delta->view(std::cout, "Hat cone delta");



  PetscFunctionReturn(0);
}/* testBiGraphHat() */

#undef  __FUNCT__
#define __FUNCT__ "viewConesAndSupports"
void viewConesAndSupports(const ALE::Obj<PointBiGraph>& bg, const char* name) {
  
  // View the cones for all base points
  std::cout << name << " cones:" << std::endl;
  ALE::Obj<PointBiGraph::traits::baseSequence> base = bg->base();
  for(PointBiGraph::traits::baseSequence::traits::iterator i = base->begin(); i != base->end(); i++) {
    ALE::Obj<PointBiGraph::traits::coneSequence> cone = bg->cone(*i);
    std::cout << *i << ": ";
    cone->view(std::cout, true); 
  }

  // View the supports for all cap points
  std::cout << name << " supports:" << std::endl;
  ALE::Obj<PointBiGraph::traits::capSequence> cap = bg->cap();
  for(PointBiGraph::traits::capSequence::traits::iterator i = cap->begin(); i != cap->end(); i++) {
    ALE::Obj<PointBiGraph::traits::supportSequence> supp = bg->support(*i);
    std::cout << *i << ": ";
    supp->view(std::cout, true); 
  }
}/* viewConesAndSupports() */
