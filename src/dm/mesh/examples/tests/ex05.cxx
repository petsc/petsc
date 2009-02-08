/*T
   Concepts: BiGraph
   Processors: multiple
T*/

/*
  Create a series of parallel BiGraphs suitable for testing the Delta routines.
*/

static char help[] = "Constructs a series of parallel BiGraphs and performs ParDelta routines.\n\n";

#include <ParDelta.hh>
#include <ALE.hh>


typedef ALE::Two::BiGraph<int,ALE::def::Point,int>     PointBiGraph;
typedef ALE::Two::BiGraph<ALE::def::Point,int,int>     PointBiGraphFlip;
typedef ALE::Two::ParConeDelta<PointBiGraph>           PointParConeDelta;
typedef ALE::Two::ParSupportDelta<PointBiGraphFlip>    PointParSupportDelta;

PetscErrorCode   testHat(MPI_Comm comm);
PetscErrorCode   testSkewedHat(MPI_Comm comm);
PetscErrorCode   testSkewedHatFlip(MPI_Comm comm);

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

  ierr = testHat(comm);CHKERRQ(ierr);
  ierr = testSkewedHat(comm);CHKERRQ(ierr);
  ierr = testSkewedHatFlip(comm);CHKERRQ(ierr);

  ierr = PetscFinalize();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}/* main() */

#undef  __FUNCT__
#define __FUNCT__ "testHat"
PetscErrorCode testHat(MPI_Comm comm) {
  int rank;
  PetscErrorCode ierr;
  int debug;
  PetscTruth flag;
  PetscFunctionBegin;

  ierr = MPI_Comm_rank(comm, &rank);CHKERRQ(ierr);
  
  debug = 0;
  ierr = PetscOptionsGetInt(PETSC_NULL, "-debug", &debug, &flag);CHKERRQ(ierr);
  ierr = PetscPrintf(comm, "%s: using debug value of %d\n", __FUNCT__, debug);CHKERRQ(ierr);

  ALE::Obj<PointBiGraph> bg = PointBiGraph(comm, debug);

  // Add three arrows from a single cap point rank to global points with the indices 2*rank, 2*rank+1, 2*rank+2 
  for(int i = 0; i < 3; i++) {
    bg->addArrow(rank, ALE::def::Point(-1,2*rank+i), -rank);
  }
  
  // View
  bg->view("Hat bigraph", true);

  // Compute a base overlap object using the static method PointParDelta::computeOverlap
  ALE::Obj<PointParConeDelta::overlap_type>    overlap = PointParConeDelta::overlap(bg);
  // View
  overlap->view("Hat overlap", true);

  // Compute the fusion over the overlap using the static method PointParDelta::computeFusion
  ALE::Obj<PointParConeDelta::fusion_type>    fusion = PointParConeDelta::fusion(bg, overlap);
  // View
  fusion->view("Hat cone fusion", true);

  PetscFunctionReturn(0);
}/* testHat() */

#undef  __FUNCT__
#define __FUNCT__ "testSkewedHat"
PetscErrorCode testSkewedHat(MPI_Comm comm) {
  int rank;
  PetscErrorCode ierr;
  int debug;
  PetscTruth flag;
  PetscFunctionBegin;

  ierr = MPI_Comm_rank(comm, &rank);CHKERRQ(ierr);
  
  debug = 0;
  ierr = PetscOptionsGetInt(PETSC_NULL, "-debug", &debug, &flag);CHKERRQ(ierr);
  ierr = PetscPrintf(comm, "%s: using debug value of %d\n", __FUNCT__, debug);CHKERRQ(ierr);

  ALE::Obj<PointBiGraph> bg = PointBiGraph(comm, debug);

  // Add two arrows from a single cap point 'rank' to global points with the indices 2*rank, 2*rank+1
  // as well as a single base point 2*(rank+1)
  for(int i = 0; i < 2; i++) {
    bg->addArrow(rank, ALE::def::Point(-1,2*rank+i), -rank);
  }
  bg->addBasePoint(ALE::def::Point(-1,2*(rank+1)));
  
  // View
  bg->view("SkewedHat bigraph");

  // Compute a base overlap object using the static method PointParDelta::computeOverlap
  ALE::Obj<PointParConeDelta::overlap_type>    overlap = PointParConeDelta::overlap(bg);
  // View
  overlap->view("SkewedHat base overlap");

  // Compute the fusion over the overlap using the static method PointParDelta::computeFusion
  ALE::Obj<PointParConeDelta::fusion_type>    fusion = PointParConeDelta::fusion(bg, overlap);
  // View
  fusion->view("SkewedHat cone fusion");

  PetscFunctionReturn(0);
}/* testSkewedHat() */


#undef  __FUNCT__
#define __FUNCT__ "testSkewedHatFlip"
PetscErrorCode testSkewedHatFlip(MPI_Comm comm) {
  int rank;
  PetscErrorCode ierr;
  int debug;
  PetscTruth flag;
  PetscFunctionBegin;

  ierr = MPI_Comm_rank(comm, &rank);CHKERRQ(ierr);
  
  debug = 0;
  ierr = PetscOptionsGetInt(PETSC_NULL, "-debug", &debug, &flag);CHKERRQ(ierr);
  ierr = PetscPrintf(comm, "%s: using debug value of %d\n", __FUNCT__, debug);CHKERRQ(ierr);

  ALE::Obj<PointBiGraphFlip> bg = PointBiGraphFlip(comm, debug);

  // Add two arrows from global points with the indices 2*rank, 2*rank+1 to a single cap point 'rank'
  // as well as a single cap point 2*(rank+1)
  for(int i = 0; i < 2; i++) {
    bg->addArrow(ALE::def::Point(-1,2*rank+i), rank, -rank);
  }
  bg->addCapPoint(ALE::def::Point(-1,2*(rank+1)));
  
  // View
  bg->view("SkewedHatFlip bigraph");

  // Compute a base overlap object using the static method PointParSupportDelta::computeOverlap
  ALE::Obj<PointParSupportDelta::overlap_type>   overlap = PointParSupportDelta::overlap(bg);
  // View
  overlap->view("SkewedHatFlip cap overlap");

  // Compute the fusion over the overlap using the static method PointParDelta::computeFusion
  ALE::Obj<PointParSupportDelta::fusion_type>    fusion = PointParSupportDelta::fusion(bg, overlap);
  // View
  fusion->view("SkewedHatFlip support fusion");

  PetscFunctionReturn(0);
}/* testSkewedHatFlip() */
