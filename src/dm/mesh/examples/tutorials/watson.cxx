static char help[] = "This is experimental.\n\n";

#define ALE_HAVE_CXX_ABI

#include <Mesh.hh>
#include <petscmesh_viewers.hh>

#define NUM_CELLS_0    33
#define NUM_CELLS_1    33
#define NUM_CORNERS    3
#define NUM_VERTICES_0 39 
#define NUM_VERTICES_1 42
#define SPACE_DIM      2

const int numCells[2] = {NUM_CELLS_0, NUM_CELLS_1};

const int adj0[NUM_CELLS_0*NUM_CORNERS] = {
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
   7, 42, 30};
const int adj1[NUM_CELLS_1*NUM_CORNERS] = {
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

const int *adjs[2] = {adj0, adj1};

const int numVertices[2] = {NUM_VERTICES_0, NUM_VERTICES_1};

const double coordinates0[NUM_VERTICES_0*SPACE_DIM] = {
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
  3.5,    3.0};
const double coordinates1[NUM_VERTICES_1*SPACE_DIM] = {
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

const double *coordinatesP[2] = {coordinates0, coordinates1};

#define NUM_CELLS    66
#define NUM_VERTICES 47

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
  const int  numCorners  = NUM_CORNERS;
  const bool interpolate = false;
  const int  spaceDim    = SPACE_DIM;

  PetscFunctionBegin;
  if (mesh->commSize() == 25) {
    const bool renumber = true;
    const int  rank     = mesh->commRank();

    ALE::SieveBuilder<ALE::Mesh>::buildTopology_private(s, meshDim, numCells[rank], (int *) adjs[rank], numVertices[rank], interpolate, numCorners);
    s->view("Old Sieve");
    ALE::ISieveConverter::convertSieve(*s, *mesh->getSieve(), renumbering, renumber);
  } else {
    const int  numCells    = NUM_CELLS;
    const int *cells       = adj;
    const int  numVertices = NUM_VERTICES;
    const bool renumber    = false;

    if (mesh->commRank() == 0) {
      // Can optimize input
      ALE::SieveBuilder<ALE::Mesh>::buildTopology(s, meshDim, numCells, (int *) cells, numVertices, interpolate, numCorners);
      ALE::ISieveConverter::convertSieve(*s, *mesh->getSieve(), renumbering, renumber);
    } else {
      mesh->getSieve()->setChart(typename Mesh_::sieve_type::chart_type());
      mesh->getSieve()->allocate();
    }
  }
  mesh->getSieve()->view("Sieve");

  // Can optimize stratification
  mesh->stratify();
  mesh->getLabel("depth")->view("Depth");
  if (mesh->commSize() == 25) {
    ALE::SieveBuilder<Mesh_>::buildCoordinates(mesh, spaceDim, coordinatesP[mesh->commRank()]);
  } else {
    ALE::SieveBuilder<Mesh_>::buildCoordinates(mesh, spaceDim, coordinates);
  }

  if (mesh->commSize() == 25) {
    //   Can I figure this out in a nicer way?
    ALE::SetFromMap<std::map<typename Mesh_::point_type,typename Mesh_::point_type> > globalPoints(renumbering);
    ALE::OverlapBuilder<>::constructOverlap(globalPoints, renumbering, mesh->getSendOverlap(), mesh->getRecvOverlap());
    mesh->setCalculatedOverlap(true);
  }
  mesh->view("Mesh");
  PetscFunctionReturn(0);
}

template<typename Mesh_>
PetscErrorCode DistributeMesh(ALE::Obj<Mesh_>& mesh, ALE::Obj<Mesh_>& newMesh)
{
  PetscFunctionBegin;
  typedef ALE::DistributionNew<Mesh_>                 DistributionType;
  typedef typename Mesh_::point_type                  point_type;
  typedef typename DistributionType::partitioner_type partitioner_type;
  typedef typename DistributionType::partition_type   partition_type;

  const ALE::Obj<typename Mesh_::send_overlap_type> sendMeshOverlap = new typename Mesh_::send_overlap_type(mesh->comm(), mesh->debug());
  const ALE::Obj<typename Mesh_::recv_overlap_type> recvMeshOverlap = new typename Mesh_::recv_overlap_type(mesh->comm(), mesh->debug());

  // IMESH_TODO This might be unnecessary, since the overlap for
  //   submeshes is just the restriction of the overlaps
  //   std::map<point_type,point_type> renumbering;
  typename Mesh_::renumbering_type& renumbering = newMesh->getRenumbering();
  // Distribute the mesh
  ALE::Obj<partition_type> partition = DistributionType::distributeMeshV(mesh, newMesh, renumbering, sendMeshOverlap, recvMeshOverlap);
  // Check overlap
  int localSendOverlapSize = 0, sendOverlapSize;
  int localRecvOverlapSize = 0, recvOverlapSize;
  const int commSize = sendMeshOverlap->commSize();
  for (int p = 0; p < commSize; ++p) {
    localSendOverlapSize += sendMeshOverlap->cone(p)->size();
    localRecvOverlapSize += recvMeshOverlap->support(p)->size();
  }
  MPI_Allreduce(&localSendOverlapSize, &sendOverlapSize, 1, MPI_INT, MPI_SUM, sendMeshOverlap->comm());
  MPI_Allreduce(&localRecvOverlapSize, &recvOverlapSize, 1, MPI_INT, MPI_SUM, recvMeshOverlap->comm());
  if (sendOverlapSize != recvOverlapSize) {
    std::cout <<"["<<sendMeshOverlap->commRank()<<"]: Size mismatch " << sendOverlapSize << " != " << recvOverlapSize << std::endl;
    sendMeshOverlap->view("Send Overlap");
    recvMeshOverlap->view("Recv Overlap");
    throw ALE::Exception("Invalid Overlap");
  }

  // Distribute the coordinates
  const ALE::Obj<typename Mesh_::real_section_type>& coordinates         = mesh->getRealSection("coordinates");
  const ALE::Obj<typename Mesh_::real_section_type>& parallelCoordinates = newMesh->getRealSection("coordinates");

  newMesh->setupCoordinates(parallelCoordinates);
  DistributionType::distributeSection(coordinates, partition, renumbering, sendMeshOverlap, recvMeshOverlap, parallelCoordinates);

  // Create the parallel overlap
  ALE::Obj<typename Mesh_::send_overlap_type> sendParallelMeshOverlap = newMesh->getSendOverlap();
  ALE::Obj<typename Mesh_::recv_overlap_type> recvParallelMeshOverlap = newMesh->getRecvOverlap();

  //   Can I figure this out in a nicer way?
  ALE::SetFromMap<std::map<point_type,point_type> > globalPoints(renumbering);
  ALE::OverlapBuilder<>::constructOverlap(globalPoints, renumbering, sendParallelMeshOverlap, recvParallelMeshOverlap);
  newMesh->setCalculatedOverlap(true);

  newMesh->view("Parallel Mesh");
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "CreatePartition"
// Creates a field whose value is the processor rank on each element
template<typename Mesh_>
PetscErrorCode CreatePartition(ALE::Obj<Mesh_>& mesh, SectionInt *partition)
{
  ALE::Obj<typename Mesh_::int_section_type> s;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = SectionIntCreate(mesh->comm(), partition);CHKERRQ(ierr);
  ierr = SectionIntSetBundle(*partition, mesh);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) *partition, "partition");CHKERRQ(ierr);
  ierr = SectionIntGetSection(*partition, s);CHKERRQ(ierr);
  const Obj<typename Mesh_::label_sequence>&     cells = mesh->heightStratum(0);
  const typename Mesh_::label_sequence::iterator end   = cells->end();
  const int                                      rank  = mesh->commRank();

  s->setChart(typename Mesh_::sieve_type::chart_type(0, 66));
  s->setFiberDimension(cells, 1);
  s->allocatePoint();
  for(typename ALE::Mesh::label_sequence::iterator c_iter = cells->begin(); c_iter != end; ++c_iter) {
    ierr = SectionIntUpdate(*partition, *c_iter, &rank, ADD_VALUES);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

template<typename Mesh_>
PetscErrorCode WriteVTK(ALE::Obj<Mesh_>& mesh) {
  const std::string& filename = "watsonTest.vtk";

  PetscFunctionBegin;
  try {
    SectionInt     partition;
    PetscViewer    viewer;
    PetscErrorCode ierr;

    ierr = PetscViewerCreate(mesh->comm(), &viewer);CHKERRQ(ierr);
    ierr = PetscViewerSetType(viewer, PETSC_VIEWER_ASCII);CHKERRQ(ierr);
    ierr = PetscViewerSetFormat(viewer, PETSC_VIEWER_ASCII_VTK);CHKERRQ(ierr);
    ierr = PetscViewerFileSetName(viewer, filename.c_str());CHKERRQ(ierr);

    ierr = VTKViewer::writeHeader(viewer);CHKERRQ(ierr);
    ierr = VTKViewer::writeVertices(mesh, viewer);CHKERRQ(ierr);
    ierr = VTKViewer::writeElements(mesh, viewer);CHKERRQ(ierr);

    ierr = PetscViewerSetFormat(viewer, PETSC_VIEWER_ASCII_VTK_CELL);CHKERRQ(ierr);
    ierr = CreatePartition(mesh, &partition);CHKERRQ(ierr);
    ierr = SectionIntView(partition, viewer);CHKERRQ(ierr);
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
    ALE::Obj<PETSC_MESH_TYPE>             mesh     = new PETSC_MESH_TYPE(comm, 2, debug);
    ALE::Obj<PETSC_MESH_TYPE::sieve_type> sieve    = new PETSC_MESH_TYPE::sieve_type(comm, debug);
    ALE::Obj<PETSC_MESH_TYPE>             newMesh  = new PETSC_MESH_TYPE(comm, 2, debug);
    ALE::Obj<PETSC_MESH_TYPE::sieve_type> newSieve = new PETSC_MESH_TYPE::sieve_type(comm, debug);

    mesh->setSieve(sieve);
    ierr = CreateMesh(mesh);CHKERRQ(ierr);
    newMesh->setSieve(newSieve);
    if (mesh->commSize() != 25) {
      ierr = DistributeMesh(mesh, newMesh);CHKERRQ(ierr);
      ierr = WriteVTK(newMesh);CHKERRQ(ierr);
    } else {
      ierr = WriteVTK(mesh);CHKERRQ(ierr);
    }
  } catch(ALE::Exception e) {
    std::cerr << e << std::endl;
  }
  ierr = PetscFinalize();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
