#define ALE_MEM_LOGGING

#include <petscmat.h>
#include <petscmesh.hh>
#include <Mesh.hh>
#include <Generator.hh>
#include "unitTests.hh"

#include <cppunit/extensions/TestFactoryRegistry.h>
#include <cppunit/extensions/HelperMacros.h>

#include <iostream>
#include <fstream>

class FunctionTestDistribution : public CppUnit::TestFixture
{
  CPPUNIT_TEST_SUITE(FunctionTestDistribution);

  CPPUNIT_TEST(testDistributeMesh2DUninterpolated);
  CPPUNIT_TEST(testOldDistributeMesh2DUninterpolated);
#ifdef BAD_TEST
  CPPUNIT_TEST(testPreallocationMesh2DUninterpolated);
#endif
  CPPUNIT_TEST(testOldPreallocationMesh3DUninterpolated);

  CPPUNIT_TEST_SUITE_END();
public:
  typedef ALE::Mesh                    mesh_type;
  typedef mesh_type::sieve_type        sieve_type;
  typedef mesh_type::point_type        point_type;
  typedef mesh_type::real_section_type real_section_type;
protected:
  ALE::Obj<mesh_type> _mesh;
  int                 _debug; // The debugging level
  PetscInt            _iters; // The number of test repetitions
  PetscInt            _size;  // The interval size
  std::map<point_type,point_type> _renumbering;
public:
  PetscErrorCode processOptions() {
    PetscErrorCode ierr;

    this->_debug = 0;
    this->_iters = 1;
    this->_size  = 10;

    PetscFunctionBegin;
    ierr = PetscOptionsBegin(PETSC_COMM_WORLD, "", "Options for interval section stress test", "ISection");CHKERRQ(ierr);
      ierr = PetscOptionsInt("-debug", "The debugging level", "isection.c", this->_debug, &this->_debug, PETSC_NULL);CHKERRQ(ierr);
      ierr = PetscOptionsInt("-iterations", "The number of test repetitions", "isection.c", this->_iters, &this->_iters, PETSC_NULL);CHKERRQ(ierr);
      ierr = PetscOptionsInt("-size", "The interval size", "isection.c", this->_size, &this->_size, PETSC_NULL);CHKERRQ(ierr);
    ierr = PetscOptionsEnd();CHKERRQ(ierr);
    PetscFunctionReturn(0);
  };

  /// Setup data.
  void setUp(void) {
    this->processOptions();
  };

  /// Tear down data.
  void tearDown(void) {};

  void createMesh(const int dim, const bool interpolate) {
    ALE::Obj<mesh_type> mB;
    ALE::Obj<mesh_type> m;

    if (dim == 2) {
      double lower[2] = {0.0, 0.0};
      double upper[2] = {1.0, 1.0};
      int    faces[2] = {2, 2};

      mB = ALE::MeshBuilder<mesh_type>::createSquareBoundary(PETSC_COMM_WORLD, lower, upper, faces, this->_debug);
    }
    this->_mesh = ALE::Generator<mesh_type>::generateMesh(mB, interpolate);
  };

  void readMesh(const char filename[], const int dim, const bool interpolate) {
    int           spaceDim    = 0;
    double       *coordinates = PETSC_NULL;
    std::ifstream f;

    this->_mesh = new mesh_type(PETSC_COMM_WORLD, dim);
    ALE::Obj<sieve_type> sieve = new sieve_type(this->_mesh->comm());
    this->_mesh->setSieve(sieve);
    if (!sieve->commRank()) {
      int  numCells    = 0;
      int  numCorners  = 0;
      int  numVertices = 0;
      int *cells;

      f.open(filename);
      f >> numCells;
      f >> numCorners;
      cells = new int[numCells*numCorners];
      for(int k = 0; k < numCells*numCorners; ++k) f >> cells[k];
      f >> numVertices;
      f >> spaceDim;
      coordinates = new double[numVertices*spaceDim];
      for(int k = 0; k < numVertices*spaceDim; ++k) f >> coordinates[k];
      f.close();
      ALE::SieveBuilder<mesh_type>::buildTopology(sieve, dim, numCells, cells, numVertices, interpolate, numCorners, -1,
                                                  this->_mesh->getArrowSection("orientation"));
    } else {
      this->_mesh->getArrowSection("orientation");
    }
    this->_mesh->stratify();
    ALE::SieveBuilder<mesh_type>::buildCoordinates(this->_mesh, spaceDim, coordinates);
  };

  void setupSection(const char filename[], const int numCells, real_section_type& section) {
    std::ifstream f;
    int           numBC;
    int          *numPoints;
    int         **constDof;
    int         **points;

    f.open(filename);
    f >> numBC;
    numPoints = new int[numBC];
    constDof  = new int*[numBC];
    points    = new int*[numBC];
    for(int bc = 0; bc < numBC; ++bc) {
      int  numConstraints;

      f >> numConstraints;
      constDof[bc] = new int[numConstraints];
      for(int c = 0; c < numConstraints; ++c) {
        f >> constDof[bc][c];
      }
      f >> numPoints[bc];
      points[bc] = new int[numPoints[bc]];
      for(int p = 0; p < numPoints[bc]; ++p) {
        f >> points[bc][p];
        points[bc][p] += numCells;
        if (section.hasPoint(points[bc][p])) section.setConstraintDimension(points[bc][p], numConstraints);
      }
    }
    section.allocatePoint();
    for(int bc = 0; bc < numBC; ++bc) {
      for(int p = 0; p < numPoints[bc]; ++p) {
        if (section.hasPoint(points[bc][p])) section.setConstraintDof(points[bc][p], constDof[bc]);
      }
      delete [] constDof[bc];
      delete [] points[bc];
    }
    delete [] numPoints;
    delete [] constDof;
    delete [] points;
    f.close();
  };

  void checkMesh(const ALE::Obj<mesh_type>& mesh, const char basename[]) {
    const ALE::Obj<sieve_type>& sieve = mesh->getSieve();
    ostringstream filename;
    filename << "data/" << basename << mesh->commSize() << "_p" << mesh->commRank() << ".mesh";
    std::ifstream f;

//     mesh->view("Mesh");
//     for(std::map<point_type,point_type>::const_iterator r_iter = _renumbering.begin(); r_iter != _renumbering.end(); ++r_iter) {
//       std::cout <<"["<<mesh->commRank()<<"]: renumbering " << r_iter->first << " --> " << r_iter->second << std::endl;
//     }
    f.open(filename.str().c_str());
    // Check cones
    int numCones = 0, totConeSize = 0;
    f >> numCones;
    int *coneRoots = new int[numCones];
    for(int c = 0; c < numCones; ++c) f >> coneRoots[c];
    int *coneSizes = new int[numCones];
    for(int c = 0; c < numCones; totConeSize += coneSizes[c], ++c) f >> coneSizes[c];
    int *cones = new int[totConeSize];
    for(int c = 0; c < totConeSize; ++c) f >> cones[c];

    const ALE::Obj<sieve_type::traits::baseSequence> base = sieve->base();
    int b = 0, k = 0;

    CPPUNIT_ASSERT_EQUAL(numCones, (int) base->size());
    for(sieve_type::traits::baseSequence::iterator p_iter = base->begin(); p_iter != base->end(); ++p_iter, ++b) {
      CPPUNIT_ASSERT_EQUAL(this->_renumbering[coneRoots[b]], *p_iter);
      const ALE::Obj<sieve_type::coneSequence> cone = sieve->cone(*p_iter);

      CPPUNIT_ASSERT_EQUAL(coneSizes[b], (int) cone->size());
      for(sieve_type::coneSequence::iterator c_iter = cone->begin(); c_iter != cone->end(); ++c_iter, ++k) {
        CPPUNIT_ASSERT_EQUAL(this->_renumbering[cones[k]], *c_iter);
      }
    }
    delete [] coneRoots;
    delete [] coneSizes;
    delete [] cones;

    // Check supports
    int numSupports = 0, totSupportSize = 0;
    f >> numSupports;
    int *supportRoots = new int[numSupports];
    for(int c = 0; c < numSupports; ++c) f >> supportRoots[c];
    int *supportSizes = new int[numSupports];
    for(int c = 0; c < numSupports; totSupportSize += supportSizes[c], ++c) f >> supportSizes[c];
    int *supports = new int[totSupportSize];
    for(int c = 0; c < totSupportSize; ++c) f >> supports[c];

    const ALE::Obj<sieve_type::traits::capSequence> cap = sieve->cap();
    int c = 0;

    k = 0;
    CPPUNIT_ASSERT_EQUAL(numSupports, (int) cap->size());
    for(sieve_type::traits::capSequence::iterator p_iter = cap->begin(); p_iter != cap->end(); ++p_iter, ++c) {
      CPPUNIT_ASSERT_EQUAL(this->_renumbering[supportRoots[c]], *p_iter);
      const ALE::Obj<sieve_type::supportSequence> support = sieve->support(*p_iter);

      CPPUNIT_ASSERT_EQUAL(supportSizes[c], (int) support->size());
      for(sieve_type::supportSequence::iterator s_iter = support->begin(); s_iter != support->end(); ++s_iter, ++k) {
        CPPUNIT_ASSERT_EQUAL(this->_renumbering[supports[k]], *s_iter);
      }
    }
    delete [] supportRoots;
    delete [] supportSizes;
    delete [] supports;
    // Check coordinates
    const ALE::Obj<real_section_type>&         coordinates = mesh->getRealSection("coordinates");
    const ALE::Obj<mesh_type::label_sequence>& vertices    = mesh->depthStratum(0);
    const int                                  dim         = mesh->getDimension();
    size_t                                     numVertices;

    f >> numVertices;
    CPPUNIT_ASSERT_EQUAL(numVertices, vertices->size());
    for(mesh_type::label_sequence::const_iterator v_iter = vertices->begin(); v_iter != vertices->end(); ++v_iter) {
      const real_section_type::value_type *coords = coordinates->restrictPoint(*v_iter);
      point_type vertex;

      f >> vertex;
      CPPUNIT_ASSERT_EQUAL(this->_renumbering[vertex], *v_iter);
      CPPUNIT_ASSERT_EQUAL(dim, coordinates->getFiberDimension(*v_iter));
      for(int d = 0; d < dim; ++d) {
        real_section_type::value_type coord;

        f >> coord;
        CPPUNIT_ASSERT_DOUBLES_EQUAL(coord, coords[d], 1.0e-12);
      }
    }
    // Check overlap
    const ALE::Obj<mesh_type::send_overlap_type>&                      sendOverlap = mesh->getSendOverlap();
    const ALE::Obj<mesh_type::recv_overlap_type>&                      recvOverlap = mesh->getRecvOverlap();
    const ALE::Obj<mesh_type::send_overlap_type::traits::capSequence>  sendCap     = sendOverlap->cap();
    const ALE::Obj<mesh_type::recv_overlap_type::traits::baseSequence> recvBase    = recvOverlap->base();
    size_t                                                             numPoints;

    f >> numPoints;
    CPPUNIT_ASSERT_EQUAL(numPoints, sendCap->size());
    for(mesh_type::send_overlap_type::traits::capSequence::iterator p_iter = sendCap->begin(); p_iter != sendCap->end(); ++p_iter) {
      const ALE::Obj<mesh_type::send_overlap_type::supportSequence>& ranks = sendOverlap->support(*p_iter);
      point_type           point, remotePoint;
      mesh_type::rank_type rank;

      for(mesh_type::send_overlap_type::supportSequence::iterator r_iter = ranks->begin(); r_iter != ranks->end(); ++r_iter) {
        f >> point;
        CPPUNIT_ASSERT_EQUAL(point, *p_iter);
        f >> remotePoint;
        CPPUNIT_ASSERT_EQUAL(remotePoint, r_iter.color());
        f >> rank;
        CPPUNIT_ASSERT_EQUAL(rank, *r_iter);
      }
    }
    CPPUNIT_ASSERT_EQUAL(numPoints, recvBase->size());
    for(mesh_type::recv_overlap_type::traits::baseSequence::iterator p_iter = recvBase->begin(); p_iter != recvBase->end(); ++p_iter) {
      const ALE::Obj<mesh_type::recv_overlap_type::coneSequence>& ranks = recvOverlap->cone(*p_iter);
      point_type           point, remotePoint;
      mesh_type::rank_type rank;

      for(mesh_type::recv_overlap_type::coneSequence::iterator r_iter = ranks->begin(); r_iter != ranks->end(); ++r_iter) {
        f >> point;
        CPPUNIT_ASSERT_EQUAL(point, *p_iter);
        f >> remotePoint;
        CPPUNIT_ASSERT_EQUAL(remotePoint, r_iter.color());
        f >> rank;
        CPPUNIT_ASSERT_EQUAL(rank, *r_iter);
      }
    }
    f.close();
  };

  void checkMatrix(Mat A, PetscInt dnz[], PetscInt onz[], const char basename[]) {
    MPI_Comm comm;
    int      commSize, commRank;
    PetscObjectGetComm((PetscObject) A, &comm);
    MPI_Comm_size(comm, &commSize);
    MPI_Comm_rank(comm, &commRank);
    ostringstream filename;
    filename << "data/" << basename << commSize << "_p" << commRank << ".mat";
    std::ifstream f;

    f.open(filename.str().c_str());
    // Check preallocation
    int localSize;
    f >> localSize;
    int *diagonal    = new int[localSize];
    int *offdiagonal = new int[localSize];
    for(int i = 0; i < localSize; ++i) {
      f >> diagonal[i];
      f >> offdiagonal[i];
    }
    f.close();
    PetscInt m, n;

    MatGetLocalSize(A, &m, &n);
    CPPUNIT_ASSERT_EQUAL(localSize, m);
    for(int i = 0; i < localSize; ++i) {
      CPPUNIT_ASSERT_EQUAL(diagonal[i],    dnz[i]);
      CPPUNIT_ASSERT_EQUAL(offdiagonal[i], onz[i]);
    }
    delete [] diagonal;
    delete [] offdiagonal;
  };

  void testDistributeMesh2DUninterpolated(void) {
    this->createMesh(2, false);

    typedef ALE::Partitioner<>::part_type                rank_type;
    typedef ALE::Sifter<point_type,rank_type,point_type> mesh_send_overlap_type;
    typedef ALE::Sifter<rank_type,point_type,point_type> mesh_recv_overlap_type;
    typedef ALE::DistributionNew<mesh_type>              distribution_type;
    typedef distribution_type::partition_type            partition_type;
    ALE::Obj<mesh_type>                    parallelMesh    = new mesh_type(this->_mesh->comm(), this->_mesh->getDimension(), this->_mesh->debug());
    ALE::Obj<mesh_type::sieve_type>        parallelSieve   = new mesh_type::sieve_type(this->_mesh->comm(), this->_mesh->debug());
    const ALE::Obj<mesh_send_overlap_type> sendMeshOverlap = new mesh_send_overlap_type(this->_mesh->comm(), this->_mesh->debug());
    const ALE::Obj<mesh_recv_overlap_type> recvMeshOverlap = new mesh_recv_overlap_type(this->_mesh->comm(), this->_mesh->debug());

    parallelMesh->setSieve(parallelSieve);
    ALE::Obj<partition_type> partition = distribution_type::distributeMesh(this->_mesh, parallelMesh, this->_renumbering, sendMeshOverlap, recvMeshOverlap);
    const ALE::Obj<real_section_type>& coordinates         = this->_mesh->getRealSection("coordinates");
    const ALE::Obj<real_section_type>& parallelCoordinates = parallelMesh->getRealSection("coordinates");

    parallelMesh->setupCoordinates(parallelCoordinates);
    distribution_type::distributeSection(coordinates, partition, this->_renumbering, sendMeshOverlap, recvMeshOverlap, parallelCoordinates);
    ALE::SetFromMap<std::map<point_type,point_type> > globalPoints(this->_renumbering);

    ALE::OverlapBuilder<>::constructOverlap(globalPoints, this->_renumbering, parallelMesh->getSendOverlap(), parallelMesh->getRecvOverlap());
    this->checkMesh(parallelMesh, "2DUninterpolatedDist");
  };

  void testOldDistributeMesh2DUninterpolated(void) {
    this->createMesh(2, false);
    typedef ALE::Distribution<mesh_type> distribution_type;

    ALE::Obj<mesh_type> parallelMesh = distribution_type::distributeMesh(this->_mesh);
    parallelMesh->constructOverlap();
    const ALE::Obj<ALE::Mesh::sieve_type::traits::baseSequence> base = parallelMesh->getSieve()->base();
    const ALE::Obj<ALE::Mesh::sieve_type::traits::capSequence>  cap  = parallelMesh->getSieve()->cap();
    const int min = std::min(*std::min_element(base->begin(), base->end()), *std::min_element(cap->begin(), cap->end()));
    const int max = std::max(*std::max_element(base->begin(), base->end()), *std::max_element(cap->begin(), cap->end()));

    for(int i = min; i <= max; ++i) {
      this->_renumbering[i] = i;
    }
    this->checkMesh(parallelMesh, "2DUninterpolatedOldDist");
  };

#ifdef BAD_TEST
  void testPreallocationMesh2DUninterpolated(void) {
    this->createMesh(2, false);

    typedef ALE::Partitioner<>::part_type                rank_type;
    typedef ALE::Sifter<point_type,rank_type,point_type> mesh_send_overlap_type;
    typedef ALE::Sifter<rank_type,point_type,point_type> mesh_recv_overlap_type;
    typedef ALE::DistributionNew<mesh_type>              distribution_type;
    typedef distribution_type::partition_type            partition_type;
    ALE::Obj<mesh_type>                    parallelMesh    = new mesh_type(this->_mesh->comm(), this->_mesh->getDimension(), this->_mesh->debug());
    ALE::Obj<mesh_type::sieve_type>        parallelSieve   = new mesh_type::sieve_type(this->_mesh->comm(), this->_mesh->debug());
    const ALE::Obj<mesh_send_overlap_type> sendMeshOverlap = new mesh_send_overlap_type(this->_mesh->comm(), this->_mesh->debug());
    const ALE::Obj<mesh_recv_overlap_type> recvMeshOverlap = new mesh_recv_overlap_type(this->_mesh->comm(), this->_mesh->debug());

    parallelMesh->setSieve(parallelSieve);
    ALE::Obj<partition_type> partition = distribution_type::distributeMesh(this->_mesh, parallelMesh, this->_renumbering, sendMeshOverlap, recvMeshOverlap);
    const ALE::Obj<real_section_type>& coordinates         = this->_mesh->getRealSection("coordinates");
    const ALE::Obj<real_section_type>& parallelCoordinates = parallelMesh->getRealSection("coordinates");

    parallelMesh->setupCoordinates(parallelCoordinates);
    distribution_type::distributeSection(coordinates, partition, this->_renumbering, sendMeshOverlap, recvMeshOverlap, parallelCoordinates);
    ALE::SetFromMap<std::map<point_type,point_type> > globalPoints(this->_renumbering);

    ALE::OverlapBuilder<>::constructOverlap(globalPoints, this->_renumbering, parallelMesh->getSendOverlap(), parallelMesh->getRecvOverlap());
    parallelMesh->setCalculatedOverlap(true);
    const ALE::Obj<mesh_type::order_type>& globalOrder = parallelMesh->getFactory()->getGlobalOrder(parallelMesh, "default", parallelCoordinates);
    const int                              localSize   = globalOrder->getLocalSize();
    const int                              globalSize  = globalOrder->getGlobalSize();
    PetscInt      *dnz;
    PetscInt      *onz;
    Mat            A;
    PetscErrorCode ierr;

    ierr = MatCreate(parallelMesh->comm(), &A);
    ierr = MatSetSizes(A, localSize, localSize, globalSize, globalSize);
    ierr = MatSetType(A, MATAIJ);
    ierr = MatSetFromOptions(A);
    ierr = PetscMalloc2(localSize, PetscInt, &dnz, localSize, PetscInt, &onz);
    ierr = preallocateOperator(parallelMesh, 1, parallelCoordinates->getAtlas(), globalOrder, dnz, onz, A);
    CPPUNIT_ASSERT_EQUAL(0, ierr);
    this->checkMatrix(A, dnz, onz, "2DUninterpolatedPreallocate");
    ierr = PetscFree2(dnz, onz);
  };
#endif

  void testOldPreallocationMesh3DUninterpolated(void) {
    this->readMesh("data/3DHex.mesh", 3, false);
    int numLocalCells = this->_mesh->heightStratum(0)->size(), numCells;
    typedef ALE::Distribution<mesh_type> distribution_type;

    MPI_Allreduce(&numLocalCells, &numCells, 1, MPI_INT, MPI_MAX, this->_mesh->comm());
    ALE::Obj<mesh_type> parallelMesh = distribution_type::distributeMesh(this->_mesh, 0, "chaco");
    parallelMesh->constructOverlap();
    const ALE::Obj<real_section_type>&     section     = parallelMesh->getRealSection("default2");
    section->setFiberDimension(parallelMesh->depthStratum(0), 3);
    this->setupSection("data/3DHex.bc", numCells, *section);
    const ALE::Obj<mesh_type::order_type>& globalOrder = parallelMesh->getFactory()->getGlobalOrder(parallelMesh, "default", section);
    const int                              localSize   = globalOrder->getLocalSize();
    const int                              globalSize  = globalOrder->getGlobalSize();
    PetscInt      *dnz;
    PetscInt      *onz;
    Mat            A;
    PetscErrorCode ierr;

    ierr = MatCreate(parallelMesh->comm(), &A);
    ierr = MatSetSizes(A, localSize, localSize, globalSize, globalSize);
    ierr = MatSetType(A, MATAIJ);
    ierr = MatSetFromOptions(A);
    ierr = PetscMalloc2(localSize, PetscInt, &dnz, localSize, PetscInt, &onz);
    ierr = preallocateOperator(parallelMesh, 1, section->getAtlas(), globalOrder, dnz, onz, A);
    CPPUNIT_ASSERT_EQUAL(0, ierr);
    this->checkMatrix(A, dnz, onz, "3DUninterpolatedPreallocate");
    ierr = PetscFree2(dnz, onz);
  };
};

#undef __FUNCT__
#define __FUNCT__ "RegisterDistributionFunctionSuite"
PetscErrorCode RegisterDistributionFunctionSuite() {
  CPPUNIT_TEST_SUITE_REGISTRATION(FunctionTestDistribution);
  PetscFunctionBegin;
  PetscFunctionReturn(0);
}
