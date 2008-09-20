static char help[] = "Mesh Refinement Tests.\n\n";

#include <petscmesh_viewers.hh>

using ALE::Obj;

typedef struct {
  PetscInt debug;
  PetscInt numLevels; // The number of refinement levels
} Options;

#undef __FUNCT__
#define __FUNCT__ "ProcessOptions"
PetscErrorCode ProcessOptions(MPI_Comm comm, Options *options)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  options->debug     = 0;
  options->numLevels = 1;

  ierr = PetscOptionsBegin(comm, "", "Options for the Sieve package tests", "Sieve");CHKERRQ(ierr);
    ierr = PetscOptionsInt("-debug", "Debugging flag", "refineTests", options->debug, &options->debug, PETSC_NULL);CHKERRQ(ierr);
    ierr = PetscOptionsInt("-num_levels", "The number of refinement levels", "refineTests", options->numLevels, &options->numLevels, PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsEnd();
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TetrahedronTest"
PetscErrorCode TetrahedronTest(const Options *options)
{
  typedef ALE::IMesh            mesh_type;
  typedef mesh_type::sieve_type sieve_type;
  typedef mesh_type::point_type point_type;
  typedef std::pair<point_type,point_type> edge_type;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  // Create a single tetrahedron
  Obj<mesh_type>  mesh  = new mesh_type(PETSC_COMM_WORLD, 3, options->debug);
  Obj<sieve_type> sieve = new sieve_type(mesh->comm(), options->debug);
  int    cone[4]    = {1, 2, 3, 4};
  int    support[1] = {0};
  double coords[12] = {0.0, 0.0, 0.0,
                       0.0, 1.0, 0.0,
                       1.0, 0.0, 0.0,
                       0.0, 0.0, 1.0};
  double v0[3], J[9], invJ[9], detJ;
  std::map<edge_type, point_type> edge2vertex;

  sieve->setChart(sieve_type::chart_type(0, 5));
  sieve->setConeSize(0, 4);
  for(int v = 1; v < 5; ++v) {sieve->setSupportSize(v, 1);}
  sieve->allocate();
  sieve->setCone(cone, 0);
  for(int v = 1; v < 5; ++v) {sieve->setSupport(v, support);}
  mesh->setSieve(sieve);
  mesh->stratify();
  ALE::SieveBuilder<mesh_type>::buildCoordinates(mesh, mesh->getDimension(), coords);
  mesh->computeElementGeometry(mesh->getRealSection("coordinates"), 0, v0, J, invJ, detJ);
  if (detJ <= 0.0) {SETERRQ1(PETSC_ERR_LIB, "Inverted element, detJ %g", detJ);}

  for(int l = 0; l < options->numLevels; ++l) {
    Obj<mesh_type>  newMesh  = new mesh_type(mesh->comm(), 3, options->debug);
    Obj<sieve_type> newSieve = new sieve_type(newMesh->comm(), options->debug);

    newMesh->setSieve(newSieve);
    ALE::MeshBuilder<mesh_type>::refineTetrahedra(*mesh, *newMesh, edge2vertex);
    edge2vertex.clear();
    if (options->debug) {
      PetscViewer viewer;

      ierr = PetscViewerCreate(PETSC_COMM_WORLD, &viewer);CHKERRQ(ierr);
      ierr = PetscViewerSetType(viewer, PETSC_VIEWER_ASCII);CHKERRQ(ierr);
      ierr = PetscViewerSetFormat(viewer, PETSC_VIEWER_ASCII_VTK);CHKERRQ(ierr);
      ierr = PetscViewerFileSetName(viewer, "refineTest1.vtk");CHKERRQ(ierr);
      ierr = VTKViewer::writeHeader(viewer);CHKERRQ(ierr);
      ierr = VTKViewer::writeVertices(newMesh, viewer);CHKERRQ(ierr);
      ierr = VTKViewer::writeElements(newMesh, viewer);CHKERRQ(ierr);
      ierr = PetscViewerDestroy(viewer);CHKERRQ(ierr);
    }
    for(int c = 0; c < pow(8, l+1); ++c) {
      newMesh->computeElementGeometry(newMesh->getRealSection("coordinates"), c, v0, J, invJ, detJ);
      if (detJ <= 0.0) {SETERRQ1(PETSC_ERR_LIB, "Inverted element, detJ %g", detJ);}
    }
    mesh = newMesh;
    newMesh = PETSC_NULL;
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "RunUnitTests"
PetscErrorCode RunUnitTests(const Options *options)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = TetrahedronTest(options);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc, char *argv[])
{
  Options        options;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscInitialize(&argc, &argv, (char *) 0, help);CHKERRQ(ierr);
  ierr = ProcessOptions(PETSC_COMM_WORLD, &options);CHKERRQ(ierr);
  ierr = RunUnitTests(&options);CHKERRQ(ierr);
  ierr = PetscFinalize();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
