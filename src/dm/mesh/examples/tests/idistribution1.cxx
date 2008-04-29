#include <petsc.h>
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

  CPPUNIT_TEST_SUITE_END();
public:
  typedef ALE::IMesh            mesh_type;
  typedef mesh_type::sieve_type sieve_type;
  typedef mesh_type::point_type point_type;
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

      mB = ALE::MeshBuilder::createSquareBoundary(PETSC_COMM_WORLD, lower, upper, faces, this->_debug);
    }
    m           = ALE::Generator::generateMesh(mB, interpolate);
    this->_mesh = new mesh_type(PETSC_COMM_WORLD, dim, this->_debug);
    this->_mesh->setSieve(sieve);
    const ALE::Obj<ALE::Mesh::sieve_type::traits::baseSequence> base = m->getSieve()->base();
    const ALE::Obj<ALE::Mesh::sieve_type::traits::capSequence>  cap  = m->getSieve()->cap();

    if (!sieve->commRank()) {
      sieve->setChart(sieve_type::chart_type(std::min(*std::min_element(base->begin(), base->end()), *std::min_element(cap->begin(), cap->end())),
                                             std::max(*std::max_element(base->begin(), base->end()), *std::max_element(cap->begin(), cap->end()))));
    }
    ALE::ISieveConverter::convertSieve(*m->getSieve(), *this->_mesh->getSieve(), this->_renumbering);
    this->_renumbering.clear();
    this->_mesh->stratify();
  };

  void checkMesh(const ALE::Obj<mesh_type>& mesh, const char basename[]) {
    const ALE::Obj<sieve_type>& sieve = mesh->getSieve();
    ostringstream filename;
    filename << "data/" << basename << mesh->commSize() << "_p" << mesh->commRank() << ".mesh";
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

    for(int b = 0, k = 0; b < baseV.getSize(); ++b) {
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

    for(int c = 0, k = 0; c < capV.getSize(); ++c) {
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
    // Check overlap
    f.close();
  };

  void testDistributeMesh2DUninterpolated(void) {
    this->createMesh(2, false);

    try {
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
    partition->view("Partition");
    typedef mesh_type::real_section_type real_section_type;
    const ALE::Obj<real_section_type>& coordinates         = this->_mesh->getRealSection("coordinates");
    const ALE::Obj<real_section_type>& parallelCoordinates = parallelMesh->getRealSection("coordinates");

    parallelMesh->setupCoordinates(parallelCoordinates);
    distribution_type::distributeSection(coordinates, partition, this->_renumbering, sendMeshOverlap, recvMeshOverlap, parallelCoordinates);
    ALE::SetFromMap<std::map<point_type,point_type> > globalPoints(this->_renumbering);

    ALE::OverlapBuilder<>::constructOverlap(globalPoints, this->_renumbering, this->_mesh->getSendOverlap(), this->_mesh->getRecvOverlap());
    this->checkMesh(parallelMesh, "2DUninterpolatedDist");
    } catch(ALE::Exception e) {
      std::cerr << "ERROR: " << e << std::endl;
    }
  };
};

#undef __FUNCT__
#define __FUNCT__ "RegisterIDistributionFunctionSuite"
PetscErrorCode RegisterIDistributionFunctionSuite() {
  CPPUNIT_TEST_SUITE_REGISTRATION(FunctionTestIDistribution);
  PetscFunctionBegin;
  PetscFunctionReturn(0);
}
