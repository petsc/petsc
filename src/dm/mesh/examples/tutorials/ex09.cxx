/*T
   Concepts: Sifter
   Processors: multiple
T*/

/*
  Tests ParDelta::computeOverlap for two different Sifters
*/

static char help[] = "Constructs a series of parallel Sifters and performs ParDelta routines.\n\n";

#include <ParDelta.hh>

typedef ALE::Sifter<int,ALE::Point,int>       PointSifter;
typedef ALE::Sifter<ALE::Point,int,int>       PointSifterFlip;
typedef ALE::ParConeDelta<PointSifter>        PointParConeDelta;
typedef ALE::ParSupportDelta<PointSifterFlip> PointParSupportDelta;

PetscErrorCode testHat(MPI_Comm comm, int debug);
PetscErrorCode testSkewedHat(MPI_Comm comm);
PetscErrorCode testSkewedHatFlip(MPI_Comm comm);

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc, char *argv[])
{
  MPI_Comm       comm;
  PetscInt       debug = 0;
  PetscTruth     flag;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscInitialize(&argc, &argv, (char *) 0, help);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(PETSC_NULL, "-debug", &debug, &flag);CHKERRQ(ierr);
  comm = PETSC_COMM_WORLD;

  ierr = testHat(comm, debug);CHKERRQ(ierr);
  //ierr = testSkewedHat(comm);CHKERRQ(ierr);
  //ierr = testSkewedHatFlip(comm);CHKERRQ(ierr);

  ierr = PetscFinalize();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef  __FUNCT__
#define __FUNCT__ "testHat"
PetscErrorCode testHat(MPI_Comm comm, int debug) {
  ALE::Obj<PointSifter> bgA  = PointSifter(comm, debug);
  ALE::Obj<PointSifter> bgB  = PointSifter(comm, debug);
  int                   size = bgA->commSize();
  int                   rank = bgA->commRank();
  int                revRank = size - (rank+1);
  int                 prefix = 0;

  PetscFunctionBegin;
  // Add three arrows from a single cap point rank to global points with the indices 2*rank, 2*rank+1, 2*rank+2 
  for(int i = 0; i < 3; i++) {
    bgA->addArrow(rank, ALE::Point(prefix, 2*rank+i), -rank);
  }
  // Reverse the above graph
  for(int i = 0; i < 3; i++) {
    bgB->addArrow(rank, ALE::Point(prefix, 2*revRank+i), -revRank);
  }
  
  // View
  bgA->view("Hat sifter", true);
  bgB->view("Reverse Hat sifter", true);

  // Compute a base overlap object using the static method PointParDelta::computeOverlap
  PointParConeDelta::setDebug(debug);
  ALE::Obj<PointParConeDelta::bioverlap_type> overlap = PointParConeDelta::overlap(bgA, bgB);
  // View
  overlap->view("Hat-ReverseHat overlap", true);

  // Compute the fusion over the overlap using the static method PointParDelta::computeFusion
  ALE::Obj<PointParConeDelta::fusion_type> fusion = PointParConeDelta::fusion(bgA, bgB, overlap);
  // View
  fusion->view("Hat-ReverseHat cone fusion", true);

  PetscFunctionReturn(0);
}

#if 0
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
}

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
}
#endif
