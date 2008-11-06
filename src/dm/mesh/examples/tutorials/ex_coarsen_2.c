//Use Coarsener.h to implement the creation of an entire array of topologies for use in multigrid methods.

#include "Coarsener.h"
//#include "tree_mis.h"
#include "../src/dm/mesh/meshvtk.h"
#include "fast_coarsen.h"

using ALE::Obj;

typedef struct {
  int        debug;              // The debugging level
  PetscTruth useZeroBase;        // Use zero-based indexing
  char       baseFilename[2048]; // The base filename for mesh files
  PetscInt   levels;             // The number of levels in the hierarchy
  PetscReal  coarseFactor;       // The maximum coarsening factor
  PetscReal  zScale;             // The relative spread of levels for visualization
  PetscTruth outputVTK;          // Output the mesh in VTK
} Options;

#undef __FUNCT__
#define __FUNCT__ "ProcessOptions"
PetscErrorCode ProcessOptions(MPI_Comm comm, Options *options)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  options->debug        = 0;
  options->useZeroBase  = PETSC_TRUE;
  ierr = PetscStrcpy(options->baseFilename, "data/coarsen_mesh");CHKERRQ(ierr);
  options->levels       = 6;
  options->coarseFactor = 1.41;
  options->zScale       = 1.0;
  options->outputVTK    = PETSC_TRUE;

  ierr = PetscOptionsBegin(comm, "", "Options for mesh coarsening", "DMMG");CHKERRQ(ierr);
    ierr = PetscOptionsInt("-debug", "The debugging level", "ex_coarsen", options->debug, &options->debug, PETSC_NULL);CHKERRQ(ierr);
    ierr = PetscOptionsTruth("-use_zero_base", "Use zero-based indexing", "ex1.c", options->useZeroBase, &options->useZeroBase, PETSC_NULL);CHKERRQ(ierr);
    ierr = PetscOptionsString("-base_file", "The base filename for mesh files", "ex_coarsen", options->baseFilename, options->baseFilename, 2048, PETSC_NULL);CHKERRQ(ierr);
    ierr = PetscOptionsInt("-levels", "The number of coarse levels", "ex_coarsen.c", options->levels, &options->levels, PETSC_NULL);    
    ierr = PetscOptionsReal("-coarsen", "The maximum coarsening factor", "ex_coarsen.c", options->coarseFactor, &options->coarseFactor, PETSC_NULL);    
    ierr = PetscOptionsReal("-z_scale", "The relative spread of levels for visualization", "ex_coarsen.c", options->zScale, &options->zScale, PETSC_NULL);    
    ierr = PetscOptionsTruth("-output_vtk", "Output the mesh in VTK", "ex1.c", options->outputVTK, &options->outputVTK, PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsEnd();
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "CreateMesh"
PetscErrorCode CreateMesh(MPI_Comm comm, Obj<ALE::Mesh>& mesh, Options *options)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  ALE::LogStage stage = ALE::LogStageRegister("MeshCreation");
  ALE::LogStagePush(stage);
  ierr = PetscPrintf(comm, "Creating mesh\n");CHKERRQ(ierr);
  mesh = ALE::PCICE::Builder::readMesh(comm, 2, options->baseFilename, options->useZeroBase, true, options->debug);
  ALE::Coarsener::IdentifyBoundary(mesh, 2);
  ALE::Coarsener::make_coarsest_boundary(mesh, 2, options->levels + 1);
  ALE::LogStagePop(stage);
  Obj<ALE::Mesh::topology_type> topology = mesh->getTopology();
  ierr = PetscPrintf(comm, "  Read %d elements\n", topology->heightStratum(0, 0)->size());CHKERRQ(ierr);
  ierr = PetscPrintf(comm, "  Read %d vertices\n", topology->depthStratum(0, 0)->size());CHKERRQ(ierr);
  if (options->debug) {
    topology->view("Serial topology");
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "OutputVTK"
PetscErrorCode OutputVTK(const Obj<ALE::Mesh>& mesh, Options *options)
{
  PetscViewer    viewer;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (options->outputVTK) {
    ALE::LogStage stage = ALE::LogStageRegister("VTKOutput");
    ALE::LogStagePush(stage);
    ierr = PetscPrintf(mesh->comm(), "Creating VTK mesh files\n");CHKERRQ(ierr);
    ierr = PetscViewerCreate(mesh->comm(), &viewer);CHKERRQ(ierr);
    ierr = PetscViewerSetType(viewer, PETSC_VIEWER_ASCII);CHKERRQ(ierr);
    ierr = PetscViewerSetFormat(viewer, PETSC_VIEWER_ASCII_VTK);CHKERRQ(ierr);
    ierr = PetscViewerFileSetName(viewer, "testMesh.vtk");CHKERRQ(ierr);
    ierr = VTKViewer::writeHeader(viewer);CHKERRQ(ierr);
    ierr = VTKViewer::writeHierarchyVertices(mesh, viewer, options->zScale);CHKERRQ(ierr);
    ierr = VTKViewer::writeHierarchyElements(mesh, viewer);CHKERRQ(ierr);
    ierr = PetscViewerDestroy(viewer);CHKERRQ(ierr);
    const ALE::Mesh::topology_type::sheaf_type& patches = mesh->getTopology()->getPatches();
#if 0
    for(ALE::Mesh::topology_type::sheaf_type::iterator p_iter = patches.begin(); p_iter != patches.end(); ++p_iter) {
      ostringstream filename;

      filename << "coarseMesh." << *p_iter << ".vtk";
      ierr = PetscViewerCreate(mesh->comm(), &viewer);CHKERRQ(ierr);
      ierr = PetscViewerSetType(viewer, PETSC_VIEWER_ASCII);CHKERRQ(ierr);
      ierr = PetscViewerSetFormat(viewer, PETSC_VIEWER_ASCII_VTK);CHKERRQ(ierr);
      ierr = PetscViewerFileSetName(viewer, filename.str().c_str());CHKERRQ(ierr);
      ierr = VTKViewer::writeHeader(viewer);CHKERRQ(ierr);
      ierr = VTKViewer::writeVertices(mesh, *p_iter, viewer);CHKERRQ(ierr);
      ierr = VTKViewer::writeElements(mesh, *p_iter, viewer);CHKERRQ(ierr);
      //ierr = FieldView_Sieve(mesh, "spacing", viewer);CHKERRQ(ierr);
      ierr = PetscViewerDestroy(viewer);CHKERRQ(ierr);
    }
#endif
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc, char *argv[])
{
  MPI_Comm       comm;
  Options        options;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscInitialize(&argc, &argv, (char *) 0, NULL);CHKERRQ(ierr);
  comm = PETSC_COMM_WORLD;

  try {
    Obj<ALE::Mesh> mesh;

    ierr = ProcessOptions(comm, &options);CHKERRQ(ierr);
    ierr = CreateMesh(comm, mesh, &options);CHKERRQ(ierr);
    ierr = ALE::Coarsener::CreateSpacingFunction(mesh, 2);CHKERRQ(ierr);
    ierr = ALE::Coarsener::CreateCoarsenedHierarchyNew(mesh, 2, options.levels, options.coarseFactor);CHKERRQ(ierr);
    Obj<ALE::Mesh::sieve_type> sieve = new ALE::Mesh::sieve_type(mesh->comm(), 0);
    mesh->getTopology()->setPatch(options.levels+1, sieve);
    mesh->getTopology()->stratify();
    ierr = OutputVTK(mesh, &options);CHKERRQ(ierr);
  } catch (ALE::Exception e) {
    std::cout << e << std::endl;
  }
  ierr = PetscFinalize();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
