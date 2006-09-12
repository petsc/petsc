//Use Coarsener.h to implement the creation of an entire array of topologies for use in multigrid methods.

#include "Coarsener.h"
 

using ALE::Obj;
char baseFile[2048]; //stores the base file name.
double c_factor;
int debug;

#undef __FUNCT__
#define __FUNCT__ "ProcessOptions"
PetscErrorCode ProcessOptions(MPI_Comm comm)
{
  PetscErrorCode ierr;

  ierr = PetscStrcpy(baseFile, "data/ex1_2d");CHKERRQ(ierr);
  c_factor = 2; //default
  debug = 0;
  ierr = PetscOptionsBegin(comm, "", "Options for mesh loading", "DMMG");CHKERRQ(ierr);
    ierr = PetscOptionsInt("-debug", "The debugging level", "ex1.c", debug, &debug, PETSC_NULL);CHKERRQ(ierr);
    ierr = PetscOptionsString("-base_file", "The base filename for mesh files", "ex_coarsen", "ex_coarsen", baseFile, 2048, PETSC_NULL);CHKERRQ(ierr);
    ierr = PetscOptionsReal("-coarsen", "The coarsening factor", "ex_coarsen.c", c_factor, &c_factor, PETSC_NULL);    
   // ierr = PetscOptionsReal("-generate", "Generate the mesh with refinement limit placed after this.", "ex_coarsen.c", r_factor, &r_factor, PETSC_NULL);
  ierr = PetscOptionsEnd();
 PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "CreateMesh"
PetscErrorCode CreateMesh(MPI_Comm comm, Obj<ALE::Mesh>& mesh)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  ALE::LogStage stage = ALE::LogStageRegister("MeshCreation");
  ALE::LogStagePush(stage);
  ierr = PetscPrintf(comm, "Creating mesh\n");CHKERRQ(ierr);
    mesh = ALE::PCICE::Builder::readMesh(comm, 2, baseFile, true, true, debug);
   ALE::Coarsener::IdentifyBoundary(mesh, 2);
  ALE::LogStagePop(stage);
  Obj<ALE::Mesh::topology_type> topology = mesh->getTopologyNew();
  ierr = PetscPrintf(comm, "  Read %d elements\n", topology->heightStratum(0, 0)->size());CHKERRQ(ierr);
  ierr = PetscPrintf(comm, "  Read %d vertices\n", topology->depthStratum(0, 0)->size());CHKERRQ(ierr);
  if (0) {
    topology->view("Serial topology");
  }
  PetscFunctionReturn(0);
}

int main(int argc, char *argv[])
{
  MPI_Comm       comm;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscInitialize(&argc, &argv, (char *) 0, NULL);CHKERRQ(ierr);
  comm = PETSC_COMM_WORLD;

  try {
    Obj<ALE::Mesh> mesh;
    ierr = ProcessOptions(comm);CHKERRQ(ierr);
    ierr = CreateMesh(comm, mesh);CHKERRQ(ierr);
    ierr = ALE::Coarsener::CreateSpacingFunction(mesh,2);CHKERRQ(ierr);
    ierr = ALE::Coarsener::CreateCoarsenedHierarchy(mesh, 2, 6, c_factor);CHKERRQ(ierr);
    
//    ierr = OutputMesh(coarseMesh);CHKERRQ(ierr);
  } catch (ALE::Exception e) {
    std::cout << e << std::endl;
  }
  ierr = PetscFinalize();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
