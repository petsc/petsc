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

class FunctionTestIDistribution : public CppUnit::TestFixture
{
  CPPUNIT_TEST_SUITE(FunctionTestIDistribution);

  CPPUNIT_TEST(testDistributeMesh2DUninterpolated);
#if 0
  CPPUNIT_TEST(testPreallocationMesh2DUninterpolated);
  CPPUNIT_TEST(testPreallocationMesh3DUninterpolated);
#endif
  CPPUNIT_TEST(testDistributeLabel);

  CPPUNIT_TEST_SUITE_END();
public:
  typedef ALE::IMesh<>                 mesh_type;
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
    const ALE::Obj<sieve_type> sieve = new sieve_type(PETSC_COMM_WORLD, this->_debug);
    ALE::Obj<ALE::Mesh>        mB;
    ALE::Obj<ALE::Mesh>        m;

    if (dim == 2) {
      double lower[2] = {0.0, 0.0};
      double upper[2] = {1.0, 1.0};
      int    faces[2] = {2, 2};

      mB = ALE::MeshBuilder<ALE::Mesh>::createSquareBoundary(PETSC_COMM_WORLD, lower, upper, faces, this->_debug);
    }
    m           = ALE::Generator<ALE::Mesh>::generateMesh(mB, interpolate);
    this->_mesh = new mesh_type(PETSC_COMM_WORLD, dim, this->_debug);
    this->_mesh->setSieve(sieve);
    const ALE::Obj<ALE::Mesh::sieve_type::traits::baseSequence> base = m->getSieve()->base();
    const ALE::Obj<ALE::Mesh::sieve_type::traits::capSequence>  cap  = m->getSieve()->cap();

    if (!sieve->commRank()) {
      sieve->setChart(sieve_type::chart_type(std::min(*std::min_element(base->begin(), base->end()), *std::min_element(cap->begin(), cap->end())),
                                             std::max(*std::max_element(base->begin(), base->end()), *std::max_element(cap->begin(), cap->end()))));
    }
    ALE::ISieveConverter::convertMesh(*m, *this->_mesh, this->_renumbering, false);
    this->_renumbering.clear();
    if (this->_debug > 1) this->_mesh->view("Mesh");
  };

  void readMesh(const char filename[], const int dim, const bool interpolate) {
    int           spaceDim    = 0;
    double       *coordinates = PETSC_NULL;
    std::ifstream f;

    ALE::Obj<ALE::Mesh>             m = new ALE::Mesh(PETSC_COMM_WORLD, dim);
    ALE::Obj<ALE::Mesh::sieve_type> s = new ALE::Mesh::sieve_type(m->comm());
    m->setSieve(s);
    if (!m->commRank()) {
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
      ALE::SieveBuilder<ALE::Mesh>::buildTopology(s, dim, numCells, cells, numVertices, interpolate, numCorners, -1,
                                                  m->getArrowSection("orientation"));
    } else {
      m->getArrowSection("orientation");
    }
    m->stratify();
    ALE::SieveBuilder<ALE::Mesh>::buildCoordinates(m, spaceDim, coordinates);

    this->_mesh = new mesh_type(PETSC_COMM_WORLD, dim, this->_debug);
    const ALE::Obj<sieve_type> sieve = new sieve_type(PETSC_COMM_WORLD, this->_debug);
    this->_mesh->setSieve(sieve);
    const ALE::Obj<ALE::Mesh::sieve_type::traits::baseSequence> base = m->getSieve()->base();
    const ALE::Obj<ALE::Mesh::sieve_type::traits::capSequence>  cap  = m->getSieve()->cap();

    if (!sieve->commRank()) {
      sieve->setChart(sieve_type::chart_type(std::min(*std::min_element(base->begin(), base->end()), *std::min_element(cap->begin(), cap->end())),
                                             std::max(*std::max_element(base->begin(), base->end()), *std::max_element(cap->begin(), cap->end()))));
    }
    ALE::ISieveConverter::convertMesh(*m, *this->_mesh, this->_renumbering, false);
    this->_renumbering.clear();
  };

  void setupSection(const char filename[], const int numCells, mesh_type::renumbering_type& renumbering, real_section_type& section) {
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
        if (renumbering.size()) {
          if (renumbering.find(points[bc][p]) == renumbering.end()) {
            continue;
          }
          points[bc][p] = renumbering[points[bc][p]];
        }
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
    // How do I check distribution_type::partition_type::graph_partitioner_type?
#ifdef PETSC_HAVE_CHACO
    std::string partName = "chaco";
#elif defined(PETSC_HAVE_PARMETIS)
    std::string partName = "parmetis";
#else
    std::string partName = "simple";
#endif

    if (mesh->commSize() == 1) {
      filename << "data/" << basename << mesh->commSize() << "_p" << mesh->commRank() << ".mesh";
    } else {
      filename << "data/" << basename << mesh->commSize() << "_" << partName << "_p" << mesh->commRank() << ".mesh";
    }
    std::ifstream f;

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

    CPPUNIT_ASSERT_EQUAL(numCones, sieve->getBaseSize());
    ALE::ISieveVisitor::PointRetriever<sieve_type> baseV(numCones);
    ALE::ISieveVisitor::PointRetriever<sieve_type> cV(sieve->getMaxConeSize());

    sieve->base(baseV);
    const sieve_type::point_type *base = baseV.getPoints();

    for(int b = 0, k = 0; b < (int) baseV.getSize(); ++b) {
      CPPUNIT_ASSERT_EQUAL(this->_renumbering[coneRoots[b]], base[b]);
      sieve->cone(base[b], cV);
      const sieve_type::point_type *cone     = cV.getPoints();
      const int                     coneSize = cV.getSize();

      CPPUNIT_ASSERT_EQUAL(coneSizes[b], coneSize);
      for(int c = 0; c < coneSize; ++c, ++k) {
        CPPUNIT_ASSERT_EQUAL(this->_renumbering[cones[k]], cone[c]);
      }
      cV.clear();
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

    CPPUNIT_ASSERT_EQUAL(numSupports, sieve->getCapSize());
    ALE::ISieveVisitor::PointRetriever<sieve_type> capV(numSupports);
    ALE::ISieveVisitor::PointRetriever<sieve_type> sV(sieve->getMaxSupportSize());

    sieve->cap(capV);
    const sieve_type::point_type *cap = capV.getPoints();

    for(int c = 0, k = 0; c < (int) capV.getSize(); ++c) {
      CPPUNIT_ASSERT_EQUAL(this->_renumbering[supportRoots[c]], cap[c]);
      sieve->support(cap[c], sV);
      const sieve_type::point_type *support     = sV.getPoints();
      const int                     supportSize = sV.getSize();

      CPPUNIT_ASSERT_EQUAL(supportSizes[c], supportSize);
      for(int s = 0; s < supportSize; ++s, ++k) {
        CPPUNIT_ASSERT_EQUAL(this->_renumbering[supports[k]], support[s]);
      }
      sV.clear();
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

  void checkLabel(const ALE::Obj<mesh_type::label_type>& label, const char basename[]) {
    ostringstream filename;
    // How do I check distribution_type::partition_type::graph_partitioner_type?
#ifdef PETSC_HAVE_CHACO
    std::string partName = "chaco";
#elif defined(PETSC_HAVE_PARMETIS)
    std::string partName = "parmetis";
#else
    std::string partName = "simple";
#endif

    if (label->commSize() == 1) {
      filename << "data/" << basename << label->commSize() << "_p" << label->commRank() << ".label";
    } else {
      filename << "data/" << basename << label->commSize() << "_" << partName << "_p" << label->commRank() << ".label";
    }
    std::ifstream f;

    f.open(filename.str().c_str());
    // Check label
    int numPoints = 0;
    f >> numPoints;
    int *points = new point_type[numPoints];
    int *values = new int[numPoints];
    for(int p = 0; p < numPoints; ++p) {
      f >> points[p];
      f >> values[p];
    }

    CPPUNIT_ASSERT_EQUAL(numPoints, label->getBaseSize());
    for(int i = 0; i < numPoints; ++i) {
      CPPUNIT_ASSERT_EQUAL(1, label->getConeSize(points[i]));
      CPPUNIT_ASSERT_EQUAL(values[i], *label->cone(points[i])->begin());
    }
    delete [] points;
    delete [] values;
    f.close();
  };

  void checkOrder(mesh_type::order_type& globalOrder, const char basename[]) {
    ostringstream filename;
    filename << "data/" << basename << globalOrder.commSize() << "_p" << globalOrder.commRank() << ".order";
    std::ifstream f;

    f.open(filename.str().c_str());
    size_t localSize;
    f >> localSize;
    int *ordering = new int[localSize*3];
    for(int i = 0; i < (int) localSize; ++i) {
      f >> ordering[i*3+0]; // point
      f >> ordering[i*3+1]; // offset
      f >> ordering[i*3+2]; // dim
    }
    f.close();

    const mesh_type::order_type::chart_type& chart = globalOrder.getChart();
    int i = 0;

    CPPUNIT_ASSERT_EQUAL(localSize, chart.size());
    for(mesh_type::order_type::chart_type::const_iterator p_iter = chart.begin(); p_iter != chart.end(); ++p_iter, ++i) {
      CPPUNIT_ASSERT_EQUAL(ordering[i*3+0], *p_iter);
      CPPUNIT_ASSERT_EQUAL(ordering[i*3+1], globalOrder.restrictPoint(*p_iter)[0].prefix);
      CPPUNIT_ASSERT_EQUAL(ordering[i*3+2], globalOrder.restrictPoint(*p_iter)[0].index);
    }
    delete [] ordering;
  };

  void checkMatrix(Mat A, PetscInt dnz[], PetscInt onz[], const char basename[], real_section_type& section, mesh_type::order_type& globalOrder) {
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
#if 1
      if (diagonal[i] != dnz[i]) {
        mesh_type::point_type p = -1;
        for(real_section_type::chart_type::const_iterator c_iter = section.getChart().begin(); c_iter != section.getChart().end(); ++c_iter) {
          const int idx  = globalOrder.getIndex(*c_iter);
          const int size = section.getConstrainedFiberDimension(*c_iter);

          if ((i >= idx) && (i < idx+size)) {
            p = *c_iter;
            break;
          }
        }
        mesh_type::point_type gP = -1;
        for(mesh_type::renumbering_type::const_iterator r_iter = this->_renumbering.begin(); r_iter != this->_renumbering.end(); ++r_iter) {
          if (r_iter->second == p) {
            gP = r_iter->first;
            break;
          }
        }
        std::cerr << "["<<this->_mesh->commRank()<<"]: Local row " << i << " local point " << p << " global point " << gP << " expected dnz: " << diagonal[i] << " actual dnz: " << dnz[i] << std::endl;
      }
#else
      CPPUNIT_ASSERT_EQUAL(diagonal[i],    dnz[i]);
#endif
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
    ALE::Obj<partition_type> partition = distribution_type::distributeMeshV(this->_mesh, parallelMesh, this->_renumbering, sendMeshOverlap, recvMeshOverlap);
    const ALE::Obj<real_section_type>& coordinates         = this->_mesh->getRealSection("coordinates");
    const ALE::Obj<real_section_type>& parallelCoordinates = parallelMesh->getRealSection("coordinates");

    parallelMesh->setupCoordinates(parallelCoordinates);
    distribution_type::distributeSection(coordinates, partition, this->_renumbering, sendMeshOverlap, recvMeshOverlap, parallelCoordinates);
    if (this->_debug) {
      parallelMesh->view("Parallel Mesh");
      for(mesh_type::renumbering_type::const_iterator r_iter = this->_renumbering.begin(); r_iter != this->_renumbering.end(); ++r_iter) {
	std::cout << "renumbering["<<r_iter->first<<"]: " << r_iter->second << std::endl;
      }
    }
    ALE::SetFromMap<std::map<point_type,point_type> > globalPoints(this->_renumbering);

    ALE::OverlapBuilder<>::constructOverlap(globalPoints, this->_renumbering, parallelMesh->getSendOverlap(), parallelMesh->getRecvOverlap());
    this->checkMesh(parallelMesh, "2DUninterpolatedIDist");
  };

  void testDistributeLabel(void) {
    this->createMesh(2, false);
    const Obj<mesh_type::label_type>& origLabel = this->_mesh->createLabel("test");
    for(point_type p = this->_mesh->getSieve()->getChart().min(); p < this->_mesh->getSieve()->getChart().max()/2; ++p) {
      this->_mesh->setValue(origLabel, p, 15);
    }
    for(point_type p = this->_mesh->getSieve()->getChart().max()/2 + 1; p < this->_mesh->getSieve()->getChart().max(); ++p) {
      this->_mesh->setValue(origLabel, p, 37);
    }
    if (this->_debug) {origLabel->view("Serial Label");}

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
    ALE::Obj<partition_type> partition = distribution_type::distributeMeshV(this->_mesh, parallelMesh, this->_renumbering, sendMeshOverlap, recvMeshOverlap);
    const Obj<mesh_type::label_type>& newLabel = parallelMesh->createLabel("test");

    distribution_type::distributeLabelV(parallelMesh->getSieve(), origLabel, partition, this->_renumbering, sendMeshOverlap, recvMeshOverlap, newLabel);
    if (this->_debug) {
      newLabel->view("Parallel Label");
      for(mesh_type::renumbering_type::const_iterator r_iter = this->_renumbering.begin(); r_iter != this->_renumbering.end(); ++r_iter) {
	std::cout << "renumbering["<<r_iter->first<<"]: " << r_iter->second << std::endl;
      }
    }
    this->checkLabel(newLabel, "2DUninterpolated");
  };

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
    ALE::Obj<partition_type> partition = distribution_type::distributeMeshV(this->_mesh, parallelMesh, this->_renumbering, sendMeshOverlap, recvMeshOverlap);
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
    //ierr = preallocateOperator(parallelMesh, parallelMesh->getDimension(), parallelCoordinates->getAtlas(), globalOrder, dnz, onz, A);
    ierr = preallocateOperator(parallelMesh, 1, parallelCoordinates->getAtlas(), globalOrder, dnz, onz, A);
    CPPUNIT_ASSERT_EQUAL(0, ierr);
    this->checkMatrix(A, dnz, onz, "2DUninterpolatedPreallocate", *parallelCoordinates, *globalOrder);
    ierr = PetscFree2(dnz, onz);
  };

  void testPreallocationMesh3DUninterpolated(void) {
    this->readMesh("data/3DHex.mesh", 3, false);
    int numLocalCells = this->_mesh->heightStratum(0)->size(), numCells;

    MPI_Allreduce(&numLocalCells, &numCells, 1, MPI_INT, MPI_MAX, this->_mesh->comm());
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
    ALE::Obj<partition_type> partition = distribution_type::distributeMeshV(this->_mesh, parallelMesh, this->_renumbering, sendMeshOverlap, recvMeshOverlap);
    ALE::SetFromMap<std::map<point_type,point_type> > globalPoints(this->_renumbering);

    ALE::OverlapBuilder<>::constructOverlap(globalPoints, this->_renumbering, parallelMesh->getSendOverlap(), parallelMesh->getRecvOverlap());
    parallelMesh->setCalculatedOverlap(true);
    const ALE::Obj<real_section_type>&         section     = parallelMesh->getRealSection("default2");
    const ALE::Obj<mesh_type::label_sequence>& vertices    = parallelMesh->depthStratum(0);
    try {
      section->setChart(real_section_type::chart_type(*vertices->begin(), *vertices->begin() + vertices->size()));
      section->setFiberDimension(parallelMesh->depthStratum(0), 3);
      this->setupSection("data/3DHex.bc", numCells, this->_renumbering, *section);
    } catch(ALE::Exception e) {
      std::cerr << "ERROR: " << e << std::endl;
    }
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
    this->checkMatrix(A, dnz, onz, "3DUninterpolatedPreallocate", *section, *globalOrder);
    this->checkOrder(*globalOrder, "3DUninterpolatedPreallocate");
    ierr = PetscFree2(dnz, onz);
  };
};

#undef __FUNCT__
#define __FUNCT__ "RegisterIDistributionFunctionSuite"
PetscErrorCode RegisterIDistributionFunctionSuite() {
  CPPUNIT_TEST_SUITE_REGISTRATION(FunctionTestIDistribution);
  PetscFunctionBegin;
  PetscFunctionReturn(0);
}
