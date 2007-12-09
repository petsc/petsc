//Use Hierarchy.hh to implement the creation of an entire array of topologies for use in multigrid methods.



#include <petscmesh.hh>
#include <petscmesh_viewers.hh>
#include <petscmesh_formats.hh>
#include <petscdmmg.h>
//TEST compile without triangle and tetgen
//#undef PETSC_HAVE_TRIANGLE
//#undef PETSC_HAVE_TETGEN
#include "Generator.hh"
#include "Hierarchy.hh"
#include "Hierarchy_New.hh"


using ALE::Obj;


typedef struct {
  int        dim;                // The mesh dimension
  int        debug;              // The debugging level
  PetscTruth useZeroBase;        // Use zero-based indexing
  char       baseFilename[2048]; // The base filename for mesh files
  PetscInt   levels;             // The number of levels in the hierarchy
  PetscReal  coarseFactor;       // The maximum coarsening factor
  PetscReal  zScale;             // The relative spread of levels for visualization
  PetscTruth outputVTK;          // Output the mesh in VTK
  PetscTruth generate;           // Generate the mesh rather than reading it in
  PetscReal  curvatureCutoff;     // the cutoff for the curvature
} Options;

#undef __FUNCT__
#define __FUNCT__ "ProcessOptions"
PetscErrorCode ProcessOptions(MPI_Comm comm, Options *options)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  options->dim          = 2;
  options->debug        = 0;
  options->useZeroBase  = PETSC_TRUE;
  ierr = PetscStrcpy(options->baseFilename, "data/coarsen_mesh");CHKERRQ(ierr);
  options->levels       = 3;
  options->coarseFactor = 2.;
  options->zScale       = 1.0;
  options->outputVTK    = PETSC_TRUE;
  options->curvatureCutoff = 1.5;
  options->generate = PETSC_TRUE;

    ierr = PetscOptionsInt("-dim", "The mesh dimension", "ex_coarsen_3.c", options->dim, &options->dim, PETSC_NULL);    
    ierr = PetscOptionsBegin(comm, "", "Options for mesh coarsening", "DMMG");CHKERRQ(ierr);
    ierr = PetscOptionsInt("-debug", "The debugging level", "ex_coarsen_3", options->debug, &options->debug, PETSC_NULL);CHKERRQ(ierr);
    ierr = PetscOptionsTruth("-use_zero_base", "Use zero-based indexing", "ex_coarsen_3.c", options->useZeroBase, &options->useZeroBase, PETSC_NULL);CHKERRQ(ierr);
    ierr = PetscOptionsString("-base_file", "The base filename for mesh files", "ex_coarsen_3.c", options->baseFilename, options->baseFilename, 2048, PETSC_NULL);CHKERRQ(ierr);
    ierr = PetscOptionsInt("-levels", "The number of coarse levels", "ex_coarsen_3.c", options->levels, &options->levels, PETSC_NULL);    
    ierr = PetscOptionsReal("-coarsen", "The maximum coarsening factor", "ex_coarsen_3.c", options->coarseFactor, &options->coarseFactor, PETSC_NULL);   
    ierr = PetscOptionsReal("-curvature", "The automatic inclusion threshhold for the curvature", "ex_coarsen_3.c", options->curvatureCutoff, &options->curvatureCutoff, PETSC_NULL); 
    ierr = PetscOptionsTruth("-generate", "Generate the mesh rather than reading it in.", "ex_coarsen.c", options->generate, &options->generate, PETSC_NULL);
    ierr = PetscOptionsReal("-z_scale", "The relative spread of levels for visualization", "ex_coarsen_3.c", options->zScale, &options->zScale, PETSC_NULL);    
    ierr = PetscOptionsTruth("-output_vtk", "Output the mesh in VTK", "ex_coarsen_3.c", options->outputVTK, &options->outputVTK, PETSC_NULL);CHKERRQ(ierr);
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
  if (options->generate) {
    if (options->dim == 3) {
      double lower[3] = {0.0, 0.0, 0.0};
      double upper[3] = {1.0, 1.0, 1.0};
      double offset[3] = {0.5, 0.5, 0.5};
      ALE::Obj<ALE::Mesh> mb = ALE::MeshBuilder::createKyoceraCornerBoundary(comm, lower, upper, offset);
      mesh = ALE::Generator::refineMesh(ALE::Generator::generateMesh(mb, PETSC_TRUE), 0.0001, PETSC_TRUE);
    } else if (options->dim == 2) {
      double lower[2] = {0.0, 0.0};
      double upper[2] = {1.0, 1.0};
      double offset[2] = {0.5, 0.5};
      ALE::Obj<ALE::Mesh> mb = ALE::MeshBuilder::createReentrantBoundary(comm, lower, upper, offset);
      mesh = ALE::Generator::refineMesh(ALE::Generator::generateMesh(mb, options->debug), 0.001, PETSC_TRUE);
      //mesh = ALE::Generator::generateMesh(mb, options->debug);
    }
  } else {
    mesh = ALE::PCICE::Builder::readMesh(comm, options->dim, options->baseFilename, options->useZeroBase, true, options->debug);
  }
  //ALE::Coarsener::IdentifyBoundary(mesh, 2);
  //ALE::Coarsener::make_coarsest_boundary(mesh, 2, options->levels + 1);
  ALE::LogStagePop(stage);
  ierr = PetscPrintf(comm, "  Read %d elements\n", mesh->heightStratum(0)->size());CHKERRQ(ierr);
  ierr = PetscPrintf(comm, "  Read %d edges/faces\n", mesh->heightStratum(1)->size());CHKERRQ(ierr);
  ierr = PetscPrintf(comm, "  Read %d vertices\n", mesh->depthStratum(0)->size());CHKERRQ(ierr);
  if (options->debug) {
   // topology->view("Serial topology");
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "OutputVTK"
PetscErrorCode OutputVTK(ALE::Obj<ALE::Mesh> mesh, Options *options, std::string outname)
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
    ierr = PetscViewerFileSetName(viewer, outname.c_str());CHKERRQ(ierr);
    ierr = VTKViewer::writeHeader(viewer);CHKERRQ(ierr);
    ierr = VTKViewer::writeVertices(mesh, viewer);
    ierr = VTKViewer::writeElements(mesh, viewer);
//    ierr = VTKViewer::writeHierarchyVertices(mesh, viewer, options->zScale);CHKERRQ(ierr);
//    ierr = VTKViewer::writeHierarchyElements(mesh, viewer);CHKERRQ(ierr);
    ierr = PetscViewerDestroy(viewer);CHKERRQ(ierr);
    //const ALE::Mesh::topology_type::sheaf_type& patches = mesh->getTopology()->getPatches();
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
    ierr = ProcessOptions(comm, &options);CHKERRQ(ierr);
    Obj<ALE::Mesh> m;
    Mesh mesh;
    ierr = MeshCreate(comm, &mesh);
    m = new ALE::Mesh(comm, options.debug);
    ierr = CreateMesh(comm, m, &options);CHKERRQ(ierr);
    ierr = MeshSetMesh(mesh, m);
    //ierr = MeshIDBoundary(mesh);
    //create the spacing function on the original mesh
    
    m->markBoundaryCells("marker");
    PetscPrintf(m->comm(), "marked the boundary cells\n");

 
  Obj<ALE::Mesh> coarsened_mesh = Hierarchy_coarsenMesh(m, 1.4);
 
    char vtkfilename[256];
    sprintf(vtkfilename, "testMesh.vtk");
    ierr = OutputVTK(coarsened_mesh, &options, vtkfilename);CHKERRQ(ierr);
 
  } catch (ALE::Exception e) {
    std::cout << e << std::endl;
  }
  ierr = PetscFinalize();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

