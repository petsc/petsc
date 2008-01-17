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

    ierr = PetscOptionsInt("-dim", "The mesh dimension", "ex_coarsen_3.c", options->dim, &options->dim, PETSC_NULL);    
    ierr = PetscOptionsBegin(comm, "", "Options for mesh coarsening", "DMMG");CHKERRQ(ierr);
    ierr = PetscOptionsInt("-debug", "The debugging level", "ex_coarsen_3", options->debug, &options->debug, PETSC_NULL);CHKERRQ(ierr);
    ierr = PetscOptionsTruth("-use_zero_base", "Use zero-based indexing", "ex_coarsen_3.c", options->useZeroBase, &options->useZeroBase, PETSC_NULL);CHKERRQ(ierr);
    ierr = PetscOptionsString("-base_file", "The base filename for mesh files", "ex_coarsen_3.c", options->baseFilename, options->baseFilename, 2048, PETSC_NULL);CHKERRQ(ierr);
    ierr = PetscOptionsInt("-levels", "The number of coarse levels", "ex_coarsen_3.c", options->levels, &options->levels, PETSC_NULL);    
    ierr = PetscOptionsReal("-coarsen", "The maximum coarsening factor", "ex_coarsen_3.c", options->coarseFactor, &options->coarseFactor, PETSC_NULL);   
    ierr = PetscOptionsReal("-curvature", "The automatic inclusion threshhold for the curvature", "ex_coarsen_3.c", options->curvatureCutoff, &options->curvatureCutoff, PETSC_NULL); 
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
  mesh = ALE::PCICE::Builder::readMesh(comm, options->dim, options->baseFilename, options->useZeroBase, true, options->debug);
  //MeshIDBoundary(mesh);
  //mesh->markBoundaryCells("marker");
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
PetscErrorCode OutputVTK(Mesh m, Options *options, std::string outname)
{
  PetscViewer    viewer;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (options->outputVTK) {
    Obj<ALE::Mesh> mesh;
    ierr = MeshGetMesh(m, mesh);
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
    Obj<ALE::Mesh> mesh;
    Mesh mesh_set[options.levels];
    for (int i = 0; i < options.levels; i++) {
      ierr = MeshCreate(comm, &mesh_set[i]);CHKERRQ(ierr);
    };
    ierr = CreateMesh(comm, mesh, &options);CHKERRQ(ierr);
    
    MeshSetMesh(mesh_set[0], mesh);
    ierr = MeshIDBoundary(mesh_set[0]);
    mesh->markBoundaryCells("marker");
    ierr = PetscPrintf(mesh->comm(), "%d boundary vertices, %d boundary cells\n", mesh->getLabelStratum("marker", 1)->size(), mesh->getLabelStratum("marker", 2)->size());
    ierr = MeshSpacingFunction(mesh_set[0]);
    //ierr = MeshIDBoundary(mesh_set[0]);
//    mesh->createLabel("marker");
//    mesh->markBoundaryCells("marker", 1, 2, false);
    MeshCreateHierarchyLabel_Link(mesh_set[0], options.coarseFactor, options.levels, &mesh_set[1],NULL,options.curvatureCutoff );
    //ierr = MeshCoarsenMesh(m, pow(options.coarseFactor, 2), &n);
    //ierr = MeshGetMesh(n, mesh);
    //ierr = MeshLocateInMesh(m, n);
   // Obj<ALE::Mesh::sieve_type> sieve = new ALE::Mesh::sieve_type(mesh->comm(), 0);
   // mesh->getTopology()->setPatch(options.levels, sieve);
   // mesh->getTopology()->stratify();
    char vtkfilename[128];
    for (int i = 0; i < options.levels; i++) {
      sprintf(vtkfilename, "testMesh%d.vtk", i);
      ierr = OutputVTK(mesh_set[i], &options, vtkfilename);CHKERRQ(ierr);
    }
  } catch (ALE::Exception e) {
    std::cout << e << std::endl;
  }
  ierr = PetscFinalize();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

