static char help[] = "This is experimental.\n\n";

#define ALE_HAVE_CXX_ABI

#include <Mesh.hh>
#include <petscmesh_viewers.hh>

#define NUM_CELLS    66
#define NUM_CORNERS  3
#define NUM_VERTICES 47
#define SPACE_DIM    2

const int adj[NUM_CELLS*NUM_CORNERS] = {
   1, 15, 19,
  13, 15, 35,
  25, 37, 42,
  32, 44, 34,
   2, 39, 20,
   0, 32, 26,
  24, 25, 42,
  30, 42, 40,
  10, 12, 14,
   0, 44, 32,
   2, 20, 31,
   9, 38, 22,
  23, 30, 41,
  37, 46, 40,
  27, 32, 34,
  28, 30, 40,
  13, 35, 33,
  12, 19, 24,
   5, 18, 29,
  37, 40, 42,
  25, 36, 37,
   9, 33, 38,
  18, 36, 33,
   0,  7, 23,
   7, 14, 24,
   9, 22, 20,
   7, 24, 42,
   5, 46, 37,
  20, 22, 31,
  16, 28, 40,
  15, 45, 35,
  13, 36, 25,
   7, 42, 30,
  33, 35, 38,
  13, 33, 36,
   5, 37, 36,
   0, 26, 14,
   0, 14,  7,
  12, 24, 14,
   6, 35, 45,
  10, 14, 26,
  10, 26, 11,
   8, 32, 27,
  20, 39, 43,
   9, 18, 33,
  17, 29, 43,
   1,  3, 15,
   8, 26, 32,
   5, 36, 18,
   3, 45, 15,
   1,  4, 21,
   6, 38, 35,
  13, 25, 19,
   1, 12,  4,
   8, 11, 26,
   7, 30, 23,
  18, 43, 29,
   0, 23, 44,
  17, 43, 39,
   9, 43, 18,
  16, 40, 46,
   1, 19, 12,
  28, 41, 30,
  13, 19, 15,
   9, 20, 43,
  19, 25, 24};

const double coordinates[NUM_VERTICES*SPACE_DIM] = {
  2.0,    2.0,
  3.5,    5.0,
  9.0,    4.0,
  4.0,    6.0,
  2.5,    5.0,
  6.0,    2.0,
  6.0,    6.0,
  3.0,    2.0,
  0.5,    3.0,
  7.0,    4.0,
  2.0,    4.0,
  1.0,    4.0,
  3.0,    4.0,
  5.0,    4.0,
  2.5,    3.0,
  4.5,    5.0,
  5.5,    0.0,
  8.0,    2.0,
  6.5,    3.0,
  4.0,    4.0,
  8.0,    4.0,
  3.0,    6.0,
  7.5,    5.0,
  2.5,    1.0,
  3.5,    3.0,
  4.5,    3.0,
  1.5,    3.0,
  0.0,    2.0,
  4.5,    0.0,
  7.0,    2.0,
  3.5,    1.0,
  8.5,    5.0,
  1.0,    2.0,
  6.0,    4.0,
  0.5,    1.0,
  5.5,    5.0,
  5.5,    3.0,
  5.0,    2.0,
  6.5,    5.0,
  8.5,    3.0,
  4.5,    1.0,
  3.0,    0.0,
  4.0,    2.0,
  7.5,    3.0,
  1.5,    1.0,
  5.0,    6.0,
  5.5,    1.0};

template<typename Mesh_>
PetscErrorCode CreateMesh(ALE::Obj<Mesh_>& mesh)
{
  ALE::Obj<ALE::Mesh::sieve_type> s = new ALE::Mesh::sieve_type(mesh->comm(), mesh->debug());
  std::map<ALE::Mesh::point_type,ALE::Mesh::point_type> renumbering;
  const int  meshDim     = 2;
  const int  numCells    = NUM_CELLS;
  const int  numCorners  = NUM_CORNERS;
  const int *cells       = adj;
  const int  numVertices = NUM_VERTICES;
  const bool interpolate = false;
  const bool renumber    = false;
  const int  spaceDim    = SPACE_DIM;

  PetscFunctionBegin;
  if (mesh->commRank() == 0) {
    // Can optimize input
    ALE::SieveBuilder<ALE::Mesh>::buildTopology(s, meshDim, numCells, (int *) cells, numVertices, interpolate, numCorners);
    ALE::ISieveConverter::convertSieve(*s, *mesh->getSieve(), renumbering, renumber);
  } else {
    mesh->getSieve()->setChart(typename Mesh_::sieve_type::chart_type());
    mesh->getSieve()->allocate();
  }
  mesh->getSieve()->view("Sieve");

  // Can optimize stratification
  mesh->stratify();
  ALE::SieveBuilder<Mesh_>::buildCoordinates(mesh, spaceDim, coordinates);
  mesh->view("Mesh");
  PetscFunctionReturn(0);
}

template<typename Mesh_>
PetscErrorCode WriteVTK(ALE::Obj<Mesh_>& mesh) {
  const std::string& filename = "watsonTest.vtk";

  PetscFunctionBegin;
  try {
    PetscViewer    viewer;
    PetscErrorCode ierr;

    ierr = PetscViewerCreate(mesh->comm(), &viewer);CHKERRQ(ierr);
    ierr = PetscViewerSetType(viewer, PETSC_VIEWER_ASCII);CHKERRQ(ierr);
    ierr = PetscViewerSetFormat(viewer, PETSC_VIEWER_ASCII_VTK);CHKERRQ(ierr);
    ierr = PetscViewerFileSetName(viewer, filename.c_str());CHKERRQ(ierr);

    ierr = VTKViewer::writeHeader(viewer);CHKERRQ(ierr);
    ierr = VTKViewer::writeVertices(mesh, viewer);CHKERRQ(ierr);
    ierr = VTKViewer::writeElements(mesh, viewer);CHKERRQ(ierr);
  } catch (const std::exception& err) {
    std::ostringstream msg;
    msg << "Error while preparing for writing data to VTK file " << filename << ".\n" << err.what();
    SETERRQ(PETSC_ERR_PLIB, msg.str().c_str());
  } catch (const ALE::Exception& err) {
    std::ostringstream msg;
    msg << "Error while preparing for writing data to VTK file " << filename << ".\n" << err.msg();
    SETERRQ(PETSC_ERR_PLIB, msg.str().c_str());
  } catch (...) { 
    std::ostringstream msg;
    msg << "Unknown error while preparing for writing data to VTK file " << filename << ".\n";
    SETERRQ(PETSC_ERR_PLIB, msg.str().c_str());
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc, char *argv[])
{
  PetscInt       debug = 0;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscInitialize(&argc, &argv, (char *) 0, help);CHKERRQ(ierr);
  try {
    MPI_Comm comm = PETSC_COMM_WORLD;
    ALE::Obj<PETSC_MESH_TYPE>             mesh  = new PETSC_MESH_TYPE(comm, debug);
    ALE::Obj<PETSC_MESH_TYPE::sieve_type> sieve = new PETSC_MESH_TYPE::sieve_type(comm, debug);

    mesh->setSieve(sieve);
    ierr = CreateMesh(mesh);CHKERRQ(ierr);
    ierr = WriteVTK(mesh);CHKERRQ(ierr);
  } catch(ALE::Exception e) {
    std::cerr << e << std::endl;
  }
  ierr = PetscFinalize();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
