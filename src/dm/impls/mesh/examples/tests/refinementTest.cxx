static char help[] = "Mesh Refinement Tests.\n\n";

#include <petscmesh_viewers.hh>
#include <list>

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

template<typename Point>
class Edge : public std::pair<Point, Point> {
public:
  Edge() : std::pair<Point, Point>() {};
  Edge(const Point l) : std::pair<Point, Point>(l, l) {};
  Edge(const Point l, const Point r) : std::pair<Point, Point>(l, r) {};
  ~Edge() {};
  friend std::ostream& operator<<(std::ostream& stream, const Edge& edge) {
    stream << "(" << edge.first << ", " << edge.second << ")";
    return stream;
  };
};

#undef __FUNCT__
#define __FUNCT__ "SerialTetrahedronTest"
PetscErrorCode SerialTetrahedronTest(const Options *options)
{
  typedef PETSC_MESH_TYPE       mesh_type;
  typedef mesh_type::sieve_type sieve_type;
  typedef mesh_type::point_type point_type;
  typedef Edge<point_type>      edge_type;
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

  if (mesh->commSize() > 1) {PetscFunctionReturn(0);}
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
  if (detJ <= 0.0) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB, "Inverted element, detJ %g", detJ);

  for(int l = 0; l < options->numLevels; ++l) {
    Obj<mesh_type>  newMesh  = new mesh_type(mesh->comm(), 3, options->debug);
    Obj<sieve_type> newSieve = new sieve_type(newMesh->comm(), options->debug);

    newMesh->setSieve(newSieve);
    //ALE::MeshBuilder<mesh_type>::refineTetrahedra(*mesh, *newMesh, edge2vertex);
    edge2vertex.clear();
    if (options->debug) {
      PetscViewer viewer;

      ierr = PetscViewerCreate(PETSC_COMM_WORLD, &viewer);CHKERRQ(ierr);
      ierr = PetscViewerSetType(viewer, PETSCVIEWERASCII);CHKERRQ(ierr);
      ierr = PetscViewerSetFormat(viewer, PETSC_VIEWER_ASCII_VTK);CHKERRQ(ierr);
      ierr = PetscViewerFileSetName(viewer, "refineTest1.vtk");CHKERRQ(ierr);
      ierr = VTKViewer::writeHeader(newMesh, viewer);CHKERRQ(ierr);
      ierr = VTKViewer::writeVertices(newMesh, viewer);CHKERRQ(ierr);
      ierr = VTKViewer::writeElements(newMesh, viewer);CHKERRQ(ierr);
      ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);
    }
    for(int c = 0; c < pow(8, l+1); ++c) {
      newMesh->computeElementGeometry(newMesh->getRealSection("coordinates"), c, v0, J, invJ, detJ);
      if (detJ <= 0.0) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB, "Inverted element, detJ %g", detJ);
    }
    mesh = newMesh;
    newMesh = PETSC_NULL;
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SerialSplitDoubletTetrahedronTest"
PetscErrorCode SerialSplitDoubletTetrahedronTest(const Options *options)
{
  typedef PETSC_MESH_TYPE       mesh_type;
  typedef mesh_type::sieve_type sieve_type;
  typedef mesh_type::point_type point_type;
  typedef Edge<point_type>      edge_type;
  typedef ALE::MeshBuilder<mesh_type>::CellRefiner<mesh_type, edge_type> refiner_type;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  // Create a tetrahedron doublet with a cohesive cell
  Obj<mesh_type>  mesh  = new mesh_type(PETSC_COMM_WORLD, 3, options->debug);
  Obj<sieve_type> sieve = new sieve_type(mesh->comm(), options->debug);
  int    cone[17]   = {3, 4, 5, 6, 8, 10, 9, 7, 3, 4, 5, 8, 9, 10, 11, 12, 13};
  int    support[3] = {0, 1, 2};
  double coords[33] = {0.0, 0.0, 0.0,
                       0.0, 1.0, 0.0,
                       1.0, 0.0, 0.0,
                       0.0, 0.0, 1.0,
                       0.0, 0.0, -1.0,
                       0.0, 0.0, 0.0,
                       0.0, 1.0, 0.0,
                       1.0, 0.0, 0.0,
                       0.0, 0.0, 0.0,
                       0.0, 1.0, 0.0,
                       1.0, 0.0, 0.0};
  double v0[3], J[9], invJ[9], detJ;

  if (mesh->commSize() > 1) {PetscFunctionReturn(0);}
  sieve->setChart(sieve_type::chart_type(0, 14));
  sieve->setConeSize(0, 4);
  sieve->setConeSize(1, 4);
  sieve->setConeSize(2, 9);
  for(int v = 3;  v < 6;  ++v) {sieve->setSupportSize(v, 2);}
  for(int v = 6;  v < 8;  ++v) {sieve->setSupportSize(v, 1);}
  for(int v = 8;  v < 11; ++v) {sieve->setSupportSize(v, 2);}
  for(int v = 11; v < 14; ++v) {sieve->setSupportSize(v, 1);}
  sieve->allocate();
  sieve->setCone(&cone[0], 0);
  sieve->setCone(&cone[4], 1);
  sieve->setCone(&cone[8], 2);
  for(int v = 3;  v < 6;  ++v) {sieve->setSupport(v, &support[0]);}
  sieve->setSupport(6, &support[0]);
  sieve->setSupport(7, &support[1]);
  for(int v = 8;  v < 11; ++v) {sieve->setSupport(v, &support[1]);}
  for(int v = 11; v < 14; ++v) {sieve->setSupport(v, &support[2]);}
  mesh->setSieve(sieve);
  mesh->stratify();
  ALE::SieveBuilder<mesh_type>::buildCoordinates(mesh, mesh->getDimension(), coords);
  for(int e = 0; e < 2; ++e) {
    mesh->computeElementGeometry(mesh->getRealSection("coordinates"), e, v0, J, invJ, detJ);
    if (detJ <= 0.0) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_LIB, "Inverted element %d, detJ %g", e, detJ);
  }
  refiner_type refiner(*mesh);

  for(int l = 0; l < options->numLevels; ++l) {
    Obj<mesh_type>  newMesh  = new mesh_type(mesh->comm(), 3, options->debug);
    Obj<sieve_type> newSieve = new sieve_type(newMesh->comm(), options->debug);

    newMesh->setSieve(newSieve);
    //ALE::MeshBuilder<mesh_type>::refineTetrahedra(*mesh, *newMesh, refiner);
    if (options->debug) {
      PetscViewer viewer;

      ierr = PetscViewerCreate(PETSC_COMM_WORLD, &viewer);CHKERRQ(ierr);
      ierr = PetscViewerSetType(viewer, PETSCVIEWERASCII);CHKERRQ(ierr);
      ierr = PetscViewerSetFormat(viewer, PETSC_VIEWER_ASCII_VTK);CHKERRQ(ierr);
      ierr = PetscViewerFileSetName(viewer, "refineTest1.vtk");CHKERRQ(ierr);
      ierr = VTKViewer::writeHeader(newMesh, viewer);CHKERRQ(ierr);
      ierr = VTKViewer::writeVertices(newMesh, viewer);CHKERRQ(ierr);
      ierr = VTKViewer::writeElements(newMesh, viewer);CHKERRQ(ierr);
      ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);
    }
    for(int c = 0; c < pow(8, l+1); ++c) {
      newMesh->computeElementGeometry(newMesh->getRealSection("coordinates"), c, v0, J, invJ, detJ);
      if (detJ <= 0.0) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB, "Inverted element, detJ %g", detJ);
    }
    mesh = newMesh;
    newMesh = PETSC_NULL;
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "ParallelTetrahedronTest"
PetscErrorCode ParallelTetrahedronTest(const Options *options)
{
  typedef PETSC_MESH_TYPE       mesh_type;
  typedef mesh_type::sieve_type sieve_type;
  typedef mesh_type::point_type point_type;
  typedef Edge<point_type>      edge_type;
  const int      debug = options->debug;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  // Create a single tetrahedron
  Obj<mesh_type>  serialMesh  = new mesh_type(PETSC_COMM_WORLD, 3, options->debug);
  Obj<sieve_type> serialSieve = new sieve_type(serialMesh->comm(), options->debug);
  int    cone[8]    = {2, 3, 4, 5, 3, 4, 5, 6};
  int    support[2] = {0, 1};
  double coords[15] = {0.0, 0.0, 0.0,
                       0.0, 1.0, 0.0,
                       1.0, 0.0, 0.0,
                       0.0, 0.0, 1.0,
                       1.0, 1.0, 1.0};
  double v0[3], J[9], invJ[9], detJ;

  if (serialMesh->commSize() != 2) {PetscFunctionReturn(0);}
  if (!serialMesh->commRank()) {
    serialSieve->setChart(sieve_type::chart_type(0, 7));
    serialSieve->setConeSize(0, 4);
    serialSieve->setConeSize(1, 4);
    serialSieve->setSupportSize(2, 1);
    for(int v = 3; v < 6; ++v) {serialSieve->setSupportSize(v, 2);}
    serialSieve->setSupportSize(6, 1);
  } else {
    serialSieve->setChart(sieve_type::chart_type(0, 0));
  }
  serialSieve->allocate();
  if (!serialMesh->commRank()) {
    serialSieve->setCone(&cone[0], 0);
    serialSieve->setCone(&cone[4], 1);
    for(int v = 2; v < 6; ++v) {serialSieve->setSupport(v, support);}
    serialSieve->setSupport(6, &support[1]);
  }
  serialMesh->setSieve(serialSieve);
  serialMesh->stratify();
  ALE::SieveBuilder<mesh_type>::buildCoordinates(serialMesh, serialMesh->getDimension(), coords);
  for(int c = 0; c < (int) serialMesh->heightStratum(0)->size(); ++c) {
    serialMesh->computeElementGeometry(serialMesh->getRealSection("coordinates"), c, v0, J, invJ, detJ);
    if (detJ <= 0.0) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB, "Inverted element, detJ %g", detJ);
  }
  Obj<mesh_type>  mesh  = new mesh_type(serialMesh->comm(), serialMesh->getDimension(), options->debug);
  Obj<sieve_type> sieve = new sieve_type(mesh->comm(), options->debug);

  mesh->setSieve(sieve);
  ALE::DistributionNew<mesh_type>::distributeMeshAndSectionsV(serialMesh, mesh);
  if (debug) {mesh->view("Parallel Mesh");}
  for(int c = 0; c < (int) mesh->heightStratum(0)->size(); ++c) {
    mesh->computeElementGeometry(mesh->getRealSection("coordinates"), c, v0, J, invJ, detJ);
    if (detJ <= 0.0) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB, "Inverted element, detJ %g", detJ);
  }

  for(int l = 0; l < options->numLevels; ++l) {
    Obj<mesh_type>  newMesh  = new mesh_type(mesh->comm(), 3, options->debug);
    Obj<sieve_type> newSieve = new sieve_type(newMesh->comm(), options->debug);

    newMesh->setSieve(newSieve);
    //ALE::MeshBuilder<mesh_type>::refineTetrahedra(*mesh, *newMesh, edge2vertex);
    if (debug) {newMesh->view("Refined Parallel Mesh");}
    if (options->debug) {
      PetscViewer viewer;

      ierr = PetscViewerCreate(PETSC_COMM_WORLD, &viewer);CHKERRQ(ierr);
      ierr = PetscViewerSetType(viewer, PETSCVIEWERASCII);CHKERRQ(ierr);
      ierr = PetscViewerSetFormat(viewer, PETSC_VIEWER_ASCII_VTK);CHKERRQ(ierr);
      ierr = PetscViewerFileSetName(viewer, "refineTest1.vtk");CHKERRQ(ierr);
      ierr = VTKViewer::writeHeader(newMesh, viewer);CHKERRQ(ierr);
      ierr = VTKViewer::writeVertices(newMesh, viewer);CHKERRQ(ierr);
      ierr = VTKViewer::writeElements(newMesh, viewer);CHKERRQ(ierr);
      ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);
    }
    for(int c = 0; c < pow(8, l+1); ++c) {
      newMesh->computeElementGeometry(newMesh->getRealSection("coordinates"), c, v0, J, invJ, detJ);
      if (detJ <= 0.0) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB, "Inverted element, detJ %g", detJ);
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
  ierr = SerialTetrahedronTest(options);CHKERRQ(ierr);
  ierr = SerialSplitDoubletTetrahedronTest(options);CHKERRQ(ierr);
  ierr = ParallelTetrahedronTest(options);CHKERRQ(ierr);
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
  try {
    ierr = RunUnitTests(&options);CHKERRQ(ierr);
  } catch(ALE::Exception e) {
    std::cerr << "ERROR: " << e.msg() << std::endl;
  }
  ierr = PetscFinalize();
  PetscFunctionReturn(0);
}
