#define ALE_MEM_LOGGING

#include <petscsys.h>
#include <petsclog.hh>
#include <petscmesh_formats.hh>
#include <Mesh.hh>
#include <Generator.hh>
#include <Selection.hh>
#include "unitTests.hh"

#include <boost/pool/pool_alloc.hpp>

#include <cppunit/extensions/TestFactoryRegistry.h>
#include <cppunit/extensions/HelperMacros.h>

class FunctionTestIMesh : public CppUnit::TestFixture
{
  CPPUNIT_TEST_SUITE(FunctionTestIMesh);

  CPPUNIT_TEST(testSerialization);
  CPPUNIT_TEST(testStratify);
  CPPUNIT_TEST(testStratifyLine);

  CPPUNIT_TEST_SUITE_END();
public:
  typedef ALE::IMesh<>          mesh_type;
  typedef mesh_type::point_type point_type;
protected:
  ALE::Obj<mesh_type> _mesh;
  int                 _debug;        // The debugging level
  PetscInt            _iters;        // The number of test repetitions
  PetscInt            _size;         // The interval size
  PetscTruth          _interpolate;  // Flag for mesh interpolation
  PetscTruth          _onlyParallel; // Shut off serial tests
  ALE::Obj<ALE::Mesh>             _m;
  std::map<point_type,point_type> _renumbering;
public:
  PetscErrorCode processOptions() {
    PetscErrorCode ierr;

    this->_debug        = 0;
    this->_iters        = 1;
    this->_size         = 10;
    this->_interpolate  = PETSC_FALSE;
    this->_onlyParallel = PETSC_FALSE;

    PetscFunctionBegin;
    ierr = PetscOptionsBegin(PETSC_COMM_WORLD, "", "Options for interval mesh stress test", "IMesh");CHKERRQ(ierr);
      ierr = PetscOptionsInt("-debug", "The debugging level", "imesh.c", this->_debug, &this->_debug, PETSC_NULL);CHKERRQ(ierr);
      ierr = PetscOptionsInt("-iterations", "The number of test repetitions", "imesh.c", this->_iters, &this->_iters, PETSC_NULL);CHKERRQ(ierr);
      ierr = PetscOptionsInt("-size", "The interval size", "imesh.c", this->_size, &this->_size, PETSC_NULL);CHKERRQ(ierr);
      ierr = PetscOptionsTruth("-interpolate", "Flag for mesh interpolation", "imesh.c", this->_interpolate, &this->_interpolate, PETSC_NULL);CHKERRQ(ierr);
      ierr = PetscOptionsTruth("-only_parallel", "Shut off serial tests", "isieve.c", this->_onlyParallel, &this->_onlyParallel, PETSC_NULL);CHKERRQ(ierr);
    ierr = PetscOptionsEnd();CHKERRQ(ierr);
    PetscFunctionReturn(0);
  };

  /// Setup data.
  void setUp(void) {
    try {
    this->processOptions();
    double                    lower[3]    = {0.0, 0.0, 0.0};
    double                    upper[3]    = {1.0, 1.0, 1.0};
    int                       faces[3]    = {3, 3, 3};
    bool                      interpolate = this->_interpolate;
    const ALE::Obj<ALE::Mesh> mB          = ALE::MeshBuilder<ALE::Mesh>::createCubeBoundary(PETSC_COMM_WORLD, lower, upper, faces, this->_debug);
    this->_m    = ALE::Generator<ALE::Mesh>::generateMesh(mB, interpolate);
    this->_mesh = new mesh_type(mB->comm(), 3, this->_debug);
    ALE::Obj<mesh_type::sieve_type> sieve = new mesh_type::sieve_type(this->_mesh->comm(), 0, 119, this->_debug);

    this->_mesh->setSieve(sieve);
    ALE::ISieveConverter::convertMesh(*this->_m, *this->_mesh, this->_renumbering);
    if (this->_mesh->commSize() > 1) {
      ALE::Obj<mesh_type>             newMesh  = new mesh_type(PETSC_COMM_WORLD, this->_mesh->getDimension(), this->_debug);
      ALE::Obj<mesh_type::sieve_type> newSieve = new mesh_type::sieve_type(newMesh->comm(), this->_debug);

      newMesh->setSieve(newSieve);
      ALE::DistributionNew<mesh_type>::distributeMeshAndSectionsV(this->_mesh, newMesh);
      this->_mesh = newMesh;
    }
    } catch (ALE::Exception e) {
      std::cerr << e << std::endl;
      CPPUNIT_FAIL(e.msg());
    }
  };

  /// Tear down data.
  void tearDown(void) {};

  template<typename Label>
  static void checkLabel(Label& labelA, Label& labelB) {
    typename Label::capSequence cap = labelA.cap();

    CPPUNIT_ASSERT_EQUAL(cap.size(), labelB.cap().size());
    for(typename Label::capSequence::iterator p_iter = cap.begin(); p_iter != cap.end(); ++p_iter) {
      const Obj<typename Label::traits::supportSequence>&     supportA = labelA.support(*p_iter);
      const Obj<typename Label::traits::supportSequence>&     supportB = labelB.support(*p_iter);
      typename Label::traits::supportSequence::const_iterator s_iterA  = supportA->begin();
      typename Label::traits::supportSequence::const_iterator s_iterB  = supportB->begin();

      CPPUNIT_ASSERT_EQUAL(supportA->size(), supportB->size());
      for(; s_iterA != supportA->end(); ++s_iterA, ++s_iterB) {
        CPPUNIT_ASSERT_EQUAL(*s_iterA, *s_iterB);
      }
    }
    // Could also check cones
  };

  template<typename Section>
  static void checkSection(Section& sectionA, Section& sectionB) {
    // Check atlas
    checkSection(*sectionA.getAtlas(), *sectionB.getAtlas());
    // Check values
    typedef typename Section::point_type point_type;
    typedef typename Section::value_type value_type;
    point_type min = sectionA.getChart().min();
    point_type max = sectionA.getChart().max();

    CPPUNIT_ASSERT_EQUAL(min, sectionB.getChart().min());
    CPPUNIT_ASSERT_EQUAL(max, sectionB.getChart().max());
    for(point_type p = min; p < max; ++p) {
      const int         dim     = sectionA.getFiberDimension(p);
      const value_type *valuesA = sectionA.restrictPoint(p);
      const value_type *valuesB = sectionB.restrictPoint(p);

      CPPUNIT_ASSERT_EQUAL(dim, sectionB.getFiberDimension(p));
      CPPUNIT_ASSERT(valuesA != NULL);
      CPPUNIT_ASSERT(valuesB != NULL);
      for(int d = 0; d < dim; ++d) {
        CPPUNIT_ASSERT_EQUAL(valuesA[d], valuesB[d]);
      }
    }
  };

  template<typename Point_, typename Value_>
  static void checkSection(ALE::IConstantSection<Point_, Value_>& sectionA, ALE::IConstantSection<Point_, Value_>& sectionB) {
    CPPUNIT_ASSERT_EQUAL(sectionA.getChart().min(), sectionB.getChart().min());
    CPPUNIT_ASSERT_EQUAL(sectionA.getChart().max(), sectionB.getChart().max());
    CPPUNIT_ASSERT_EQUAL(sectionA.restrictPoint(sectionA.getChart().min())[0], sectionB.restrictPoint(sectionB.getChart().min())[0]);
    CPPUNIT_ASSERT_EQUAL(sectionA.getDefaultValue(), sectionB.getDefaultValue());
  };


  template<typename ISieve>
  static void checkSieve(ISieve& sieveA, const ISieve& sieveB) {
    ALE::ISieveVisitor::PointRetriever<ISieve> baseV(sieveA.getBaseSize());

    sieveA.base(baseV);
    const typename ISieve::point_type *base = baseV.getPoints();
    for(int b = 0; b < (int) baseV.getSize(); ++b) {
      ALE::ISieveVisitor::PointRetriever<ISieve>    retrieverA((int) pow((double) sieveA.getMaxConeSize(), 3));
      ALE::ISieveVisitor::PointRetriever<ISieve>    retrieverB((int) pow((double) sieveB.getMaxConeSize(), 3));

      sieveA.cone(base[b], retrieverA);
      sieveB.cone(base[b], retrieverB);
      const typename ISieve::point_type *coneA = retrieverA.getPoints();
      const typename ISieve::point_type *coneB = retrieverB.getPoints();

      CPPUNIT_ASSERT_EQUAL(retrieverA.getSize(), retrieverB.getSize());
      for(int c = 0; c < (int) retrieverA.getSize(); ++c) {
        CPPUNIT_ASSERT_EQUAL(coneA[c], coneB[c]);
      }
      CPPUNIT_ASSERT_EQUAL(sieveA.orientedCones(), sieveB.orientedCones());
      if (sieveA.orientedCones()) {
        retrieverA.clear();
        retrieverB.clear();
        sieveA.orientedCone(base[b], retrieverA);
        sieveB.orientedCone(base[b], retrieverB);
        const typename ALE::ISieveVisitor::PointRetriever<ISieve>::oriented_point_type *oConeA = retrieverA.getOrientedPoints();
        const typename ALE::ISieveVisitor::PointRetriever<ISieve>::oriented_point_type *oConeB = retrieverB.getOrientedPoints();

        CPPUNIT_ASSERT_EQUAL(retrieverA.getOrientedSize(), retrieverB.getOrientedSize());
        for(int c = 0; c < (int) retrieverA.getOrientedSize(); ++c) {
          CPPUNIT_ASSERT_EQUAL(oConeA[c].second, oConeB[c].second);
        }
      }
    }
    ALE::ISieveVisitor::PointRetriever<ISieve> capV(sieveA.getCapSize());

    sieveA.cap(capV);
    const typename ISieve::point_type *cap = capV.getPoints();
    for(int c = 0; c < (int) capV.getSize(); ++c) {
      ALE::ISieveVisitor::PointRetriever<ISieve> retrieverA((int) pow((double) sieveA.getMaxSupportSize(), 3));
      ALE::ISieveVisitor::PointRetriever<ISieve> retrieverB((int) pow((double) sieveB.getMaxSupportSize(), 3));

      sieveA.support(cap[c], retrieverA);
      sieveB.support(cap[c], retrieverB);
      const typename ISieve::point_type *supportA = retrieverA.getPoints();
      const typename ISieve::point_type *supportB = retrieverB.getPoints();

      CPPUNIT_ASSERT_EQUAL(retrieverA.getSize(), retrieverB.getSize());
      for(int s = 0; s < (int) retrieverA.getSize(); ++s) {
        CPPUNIT_ASSERT_EQUAL(supportA[s], supportB[s]);
      }
    }
  };

  template<typename Mesh>
  static void checkMesh(Mesh& meshA, Mesh& meshB) {
    checkSieve(*meshA.getSieve(), *meshB.getSieve());
    const typename Mesh::labels_type&          labelsA = meshA.getLabels();
    const typename Mesh::labels_type&          labelsB = meshB.getLabels();
    typename Mesh::labels_type::const_iterator l_iterA = labelsA.begin();
    typename Mesh::labels_type::const_iterator l_iterB = labelsB.begin();

    CPPUNIT_ASSERT_EQUAL(labelsA.size(), labelsB.size());
    for(; l_iterA != labelsA.end(); ++l_iterA, ++l_iterB) {
      CPPUNIT_ASSERT_EQUAL(l_iterA->first, l_iterB->first);
      checkLabel(*l_iterA->second, *l_iterB->second);
    }
    // Check sections
    Obj<std::set<std::string> >                    realNamesA = meshA.getRealSections();
    Obj<std::set<std::string> >                    realNamesB = meshB.getRealSections();
    typename std::set<std::string>::const_iterator r_iterA    = realNamesA->begin();
    typename std::set<std::string>::const_iterator r_iterB    = realNamesB->begin();

    CPPUNIT_ASSERT_EQUAL(realNamesA->size(), realNamesB->size());
    for(; r_iterA != realNamesA->end(); ++r_iterA, ++r_iterB) {
      CPPUNIT_ASSERT_EQUAL(*r_iterA, *r_iterB);
      checkSection(*meshA.getRealSection(*r_iterA), *meshB.getRealSection(*r_iterB));
    }
    Obj<std::set<std::string> >                    intNamesA = meshA.getIntSections();
    Obj<std::set<std::string> >                    intNamesB = meshB.getIntSections();
    typename std::set<std::string>::const_iterator i_iterA   = intNamesA->begin();
    typename std::set<std::string>::const_iterator i_iterB   = intNamesB->begin();

    CPPUNIT_ASSERT_EQUAL(intNamesA->size(), intNamesB->size());
    for(; i_iterA != intNamesA->end(); ++i_iterA, ++i_iterB) {
      CPPUNIT_ASSERT_EQUAL(*i_iterA, *i_iterB);
      checkSection(*meshA.getIntSection(*i_iterA), *meshB.getIntSection(*i_iterB));
    }
    // Check overlap
  };

  void testSerialization() {
    ALE::Obj<mesh_type> newMesh = new mesh_type(PETSC_COMM_WORLD, 3, this->_debug);
    const char         *filename = "meshTest.sav";

    if (newMesh->commSize() > 1) {
      typedef ALE::DistributionNew<mesh_type> distribution_type;
      ALE::Obj<mesh_type>             parallelMesh  = new mesh_type(this->_mesh->comm(), this->_mesh->getDimension(), this->_mesh->debug());
      ALE::Obj<mesh_type::sieve_type> parallelSieve = new mesh_type::sieve_type(this->_mesh->comm(), this->_mesh->debug());

      distribution_type::distributeMeshAndSectionsV(this->_mesh, parallelMesh);
      this->_mesh = parallelMesh;
    }
    ALE::MeshSerializer::writeMesh(filename, *this->_mesh);
    ALE::MeshSerializer::loadMesh(filename, *newMesh);
    unlink(filename);
    checkMesh(*this->_mesh, *newMesh);
  };

  void testStratify() {
    if (this->_mesh->commSize() > 1) {
      if (this->_onlyParallel) return;
      CPPUNIT_FAIL("This test is not yet parallel");
    }
    this->_mesh->stratify();
    const ALE::Obj<ALE::Mesh::label_type>& h1 = this->_m->getLabel("height");
    const ALE::Obj<mesh_type::label_type>& h2 = this->_mesh->getLabel("height");

    for(int h = 0; h < 4; ++h) {
      CPPUNIT_ASSERT_EQUAL(h1->support(h)->size(), h2->support(h)->size());
      const ALE::Obj<ALE::Mesh::label_sequence>& points1 = h1->support(h);
      const ALE::Obj<mesh_type::label_sequence>& points2 = h2->support(h);
      ALE::Mesh::label_sequence::iterator        p_iter1 = points1->begin();
      mesh_type::label_sequence::iterator        p_iter2 = points2->begin();
      ALE::Mesh::label_sequence::iterator        end1    = points1->end();

      while(p_iter1 != end1) {
        CPPUNIT_ASSERT_EQUAL(this->_renumbering[*p_iter1], *p_iter2);
        ++p_iter1;
        ++p_iter2;
      }
    }
    const ALE::Obj<ALE::Mesh::label_type>& d1 = this->_m->getLabel("depth");
    const ALE::Obj<mesh_type::label_type>& d2 = this->_mesh->getLabel("depth");

    for(int d = 0; d < 4; ++d) {
      CPPUNIT_ASSERT_EQUAL(d1->support(d)->size(), d2->support(d)->size());
      const ALE::Obj<ALE::Mesh::label_sequence>& points1 = d1->support(d);
      const ALE::Obj<mesh_type::label_sequence>& points2 = d2->support(d);
      ALE::Mesh::label_sequence::iterator        p_iter1 = points1->begin();
      mesh_type::label_sequence::iterator        p_iter2 = points2->begin();
      ALE::Mesh::label_sequence::iterator        end1    = points1->end();

      while(p_iter1 != end1) {
        CPPUNIT_ASSERT_EQUAL(this->_renumbering[*p_iter1], *p_iter2);
        ++p_iter1;
        ++p_iter2;
      }
    }
  };

  // Sieve mesh
  // 2 ----- 3 ----- 4
  //     0       1
  void testStratifyLine() {
    typedef ALE::IMesh<ALE::LabelSifter<int, mesh_type::point_type> > submesh_type;
    ALE::Obj<mesh_type::sieve_type> sieve = new mesh_type::sieve_type(PETSC_COMM_WORLD, 0, 5, this->_debug);
    this->_mesh = new mesh_type(PETSC_COMM_WORLD, 1, this->_debug);
    double     coords[3] = {-1.0, 0.0, 1.0};
    point_type cone[2];

    if (this->_mesh->commSize() > 1) {
      if (this->_onlyParallel) return;
      CPPUNIT_FAIL("This test is not yet parallel");
    }
    this->_mesh->setSieve(sieve);
    sieve->setConeSize(0, 2);
    sieve->setConeSize(1, 2);
    sieve->setSupportSize(2, 1);
    sieve->setSupportSize(3, 2);
    sieve->setSupportSize(4, 1);
    sieve->allocate();
    cone[0] = 2; cone[1] = 3;
    sieve->setCone(cone, 0);
    cone[0] = 3; cone[1] = 4;
    sieve->setCone(cone, 1);
    sieve->symmetrize();
    this->_mesh->stratify();
    ALE::SieveBuilder<mesh_type>::buildCoordinates(this->_mesh, 1, coords);

    const Obj<mesh_type::int_section_type>& bcSection = this->_mesh->getIntSection("bc0");
    int one = 1.0;

    bcSection->setChart(sieve->getChart());
    bcSection->setFiberDimension(2, 1);
    bcSection->setFiberDimension(4, 1);
    bcSection->allocatePoint();
    bcSection->updatePoint(2, &one);
    bcSection->updatePoint(4, &one);
    Obj<submesh_type> boundaryMesh = ALE::Selection<mesh_type>::submeshV<submesh_type>(this->_mesh, bcSection);
    //boundaryMesh->view("Boundary Mesh");
  };
};

#undef __FUNCT__
#define __FUNCT__ "RegisterIMeshFunctionSuite"
PetscErrorCode RegisterIMeshFunctionSuite() {
  CPPUNIT_TEST_SUITE_REGISTRATION(FunctionTestIMesh);
  PetscFunctionBegin;
  PetscFunctionReturn(0);
}

class StressTestIMesh : public CppUnit::TestFixture
{
  CPPUNIT_TEST_SUITE(StressTestIMesh);

  CPPUNIT_TEST(testClosure);
#if 0
  // Did not give any improvement (in fact was worse)
  CPPUNIT_TEST(testLabelAllocator);
#endif

  CPPUNIT_TEST_SUITE_END();
public:
  typedef ALE::IMesh<>          mesh_type;
  typedef mesh_type::sieve_type sieve_type;
  typedef mesh_type::point_type point_type;
protected:
  ALE::Obj<mesh_type> _mesh;
  std::map<point_type,point_type> _renumbering;
  int                 _debug; // The debugging level
  PetscInt            _iters; // The number of test repetitions
  PetscInt            _size;  // The number of mesh points
  PetscTruth          _interpolate;  // Flag for mesh interpolation
  ALE::Obj<ALE::Mesh> _m;
public:
  PetscErrorCode processOptions() {
    PetscErrorCode ierr;

    this->_debug       = 0;
    this->_iters       = 1;
    this->_size        = 1000;
    this->_interpolate = PETSC_FALSE;

    PetscFunctionBegin;
    ierr = PetscOptionsBegin(PETSC_COMM_WORLD, "", "Options for interval section stress test", "ISection");CHKERRQ(ierr);
      ierr = PetscOptionsInt("-debug", "The debugging level", "isection.c", this->_debug, &this->_debug, PETSC_NULL);CHKERRQ(ierr);
      ierr = PetscOptionsInt("-iterations", "The number of test repetitions", "isection.c", this->_iters, &this->_iters, PETSC_NULL);CHKERRQ(ierr);
      ierr = PetscOptionsInt("-size", "The number of points", "isection.c", this->_size, &this->_size, PETSC_NULL);CHKERRQ(ierr);
      ierr = PetscOptionsTruth("-interpolate", "Flag for mesh interpolation", "imesh.c", this->_interpolate, &this->_interpolate, PETSC_NULL);CHKERRQ(ierr);
    ierr = PetscOptionsEnd();CHKERRQ(ierr);
    PetscFunctionReturn(0);
  };

  /// Setup data.
  void setUp(void) {
    this->processOptions();
    try {
      double                    lower[3]    = {0.0, 0.0, 0.0};
      double                    upper[3]    = {1.0, 1.0, 1.0};
      int                       faces[3]    = {3, 3, 3};
      bool                      interpolate = this->_interpolate;
      const ALE::Obj<ALE::Mesh> mB          = ALE::MeshBuilder<ALE::Mesh>::createCubeBoundary(PETSC_COMM_WORLD, lower, upper, faces, this->_debug);
      this->_m    = ALE::Generator<ALE::Mesh>::generateMesh(mB, interpolate);
      this->_mesh = new mesh_type(mB->comm(), 3, this->_debug);
      ALE::Obj<mesh_type::sieve_type> sieve = new mesh_type::sieve_type(this->_mesh->comm(), 0, 119, this->_debug);

      this->_mesh->setSieve(sieve);
      ALE::ISieveConverter::convertMesh(*this->_m, *this->_mesh, this->_renumbering);
      if (this->_mesh->commSize() > 1) {
	ALE::Obj<mesh_type>             newMesh  = new mesh_type(PETSC_COMM_WORLD, this->_mesh->getDimension(), this->_debug);
	ALE::Obj<mesh_type::sieve_type> newSieve = new mesh_type::sieve_type(newMesh->comm(), this->_debug);

	newMesh->setSieve(newSieve);
	ALE::DistributionNew<mesh_type>::distributeMeshAndSectionsV(this->_mesh, newMesh);
	this->_mesh = newMesh;
      }
    } catch (ALE::Exception e) {
      std::cerr << e << std::endl;
      CPPUNIT_FAIL(e.msg());
    }
  };

  /// Tear down data.
  void tearDown(void) {};

  void testLabelAllocator(void) {
    // Q_2 mesh, 50x50x50, 64s, few labels
    typedef ALE::LabelSifter<int,int> LabelALE;
    typedef ALE::NewSifterDef::ArrowContainer<int,int>::traits::arrow_type arrow_type;
    typedef ALE::LabelSifter<int,int,boost::pool_allocator<arrow_type> > LabelPool;
    PETSc::Log::Stage("LabelTests").push();
    PETSc::Log::Event("ALEAllocatorTestTotal").begin();
    {
      LabelALE  labelALE(PETSC_COMM_WORLD, this->_debug);

      PETSc::Log::Event("ALEAllocatorTest").begin();
      for(int val = 0; val < this->_iters; ++val) {
	for(int p = val*this->_size + this->_iters; p < (val+1)*this->_size + this->_iters; ++p) {
	  labelALE.setCone(val, p);
	}
      }
      PETSc::Log::Event("ALEAllocatorTest").end();
      CPPUNIT_ASSERT_EQUAL(this->_iters*this->_size, labelALE.size());
    }
    PETSc::Log::Event("ALEAllocatorTestTotal").end();
    PETSc::Log::Event("PoolAllocatorTestTotal").begin();
    {
      boost::pool_allocator<arrow_type> pool;
      LabelPool labelPool(PETSC_COMM_WORLD, pool, this->_debug);

      arrow_type *a = pool.allocate(this->_iters*this->_size);
      pool.deallocate(a, this->_iters*this->_size);
      PETSc::Log::Event("PoolAllocatorTest").begin();
      for(int val = 0; val < this->_iters; ++val) {
	for(int p = val*this->_size + this->_iters; p < (val+1)*this->_size + this->_iters; ++p) {
	  labelPool.setCone(val, p);
	}
      }
      PETSc::Log::Event("PoolAllocatorTest").end();
      CPPUNIT_ASSERT_EQUAL(this->_iters*this->_size, labelPool.size());
    }
    PETSc::Log::Event("PoolAllocatorTestTotal").end();
    PETSc::Log::Stage("LabelTests").pop();
    StageLog             stageLog;
    EventPerfLog         eventLog;
    const PetscLogDouble maxTimePerInsertion = 2.0e-5;
    PetscErrorCode       ierr;

    ierr = PetscLogGetStageLog(&stageLog);CHKERRXX(ierr);
    ierr = StageLogGetEventPerfLog(stageLog, PETSc::Log::Stage("LabelTests").getId(), &eventLog);CHKERRXX(ierr);
    EventPerfInfo eventInfoALE = eventLog->eventInfo[PETSc::Log::Event("ALEAllocatorTest").getId()];
    EventPerfInfo eventInfoALETotal = eventLog->eventInfo[PETSc::Log::Event("ALEAllocatorTestTotal").getId()];

    CPPUNIT_ASSERT_EQUAL(eventInfoALE.count, 1);
    CPPUNIT_ASSERT_EQUAL((int) eventInfoALE.flops, 0);
    if (this->_debug) {
      ierr = PetscPrintf(PETSC_COMM_WORLD, "\nInsertion time: %g  Total time: %g  Average time per insertion: %gs\n", eventInfoALE.time, eventInfoALETotal.time, eventInfoALE.time/(this->_iters*this->_size));CHKERRXX(ierr);
    }
    CPPUNIT_ASSERT((eventInfoALE.time < maxTimePerInsertion * this->_size * this->_iters));
    EventPerfInfo eventInfoPool = eventLog->eventInfo[PETSc::Log::Event("PoolAllocatorTest").getId()];
    EventPerfInfo eventInfoPoolTotal = eventLog->eventInfo[PETSc::Log::Event("PoolAllocatorTestTotal").getId()];

    CPPUNIT_ASSERT_EQUAL(eventInfoPool.count, 1);
    CPPUNIT_ASSERT_EQUAL((int) eventInfoPool.flops, 0);
    if (this->_debug) {
      ierr = PetscPrintf(PETSC_COMM_WORLD, "\nInsertion time: %g  Total time: %g  Average time per insertion: %gs\n", eventInfoPool.time, eventInfoPoolTotal.time, eventInfoPool.time/(this->_iters*this->_size));CHKERRXX(ierr);
    }
    CPPUNIT_ASSERT((eventInfoPool.time < maxTimePerInsertion * this->_size * this->_iters));
  };

  void testClosure(void) {
    const double maxTimePerClosure = 2.0e-5;
    long         count             = 0;

    ALE::LogStage  stage = ALE::LogStageRegister("Mesh Closure Test");
    PetscLogEvent  closureEvent;
    PetscErrorCode ierr;

    ierr = PetscLogEventRegister("Closure", PETSC_OBJECT_COOKIE,&closureEvent);
    ALE::LogStagePush(stage);
    ierr = PetscLogEventBegin(closureEvent,0,0,0,0);
    const ALE::Obj<mesh_type::label_sequence>& cells      = this->_mesh->heightStratum(0);
    const mesh_type::label_sequence::iterator  cellsBegin = cells->begin();
    const mesh_type::label_sequence::iterator  cellsEnd   = cells->end();
    double coords[12];
    ALE::ISieveVisitor::RestrictVisitor<mesh_type::real_section_type> coordsVisitor(*this->_mesh->getRealSection("coordinates"), 12, coords);

    for(int r = 0; r < this->_iters; r++) {
      for(mesh_type::label_sequence::iterator c_iter = cellsBegin; c_iter != cellsEnd; ++c_iter) {
	coordsVisitor.clear();
	this->_mesh->restrictClosure(*c_iter, coordsVisitor);
	  count += coordsVisitor.getSize();
      }
    }
    ierr = PetscLogEventEnd(closureEvent,0,0,0,0);
    ALE::LogStagePop(stage);
    CPPUNIT_ASSERT_EQUAL(count, (long) cells->size()*12*this->_iters);
    StageLog     stageLog;
    EventPerfLog eventLog;
    const long   numClosures = cells->size() * this->_iters;

    ierr = PetscLogGetStageLog(&stageLog);
    ierr = StageLogGetEventPerfLog(stageLog, stage, &eventLog);
    EventPerfInfo eventInfo = eventLog->eventInfo[closureEvent];

    CPPUNIT_ASSERT_EQUAL(eventInfo.count, 1);
    CPPUNIT_ASSERT_EQUAL((int) eventInfo.flops, 0);
    if (this->_debug) {
      ierr = PetscPrintf(this->_mesh->comm(), "Closures: %d Average time per closure: %gs\n", numClosures, eventInfo.time/numClosures);
    }
    CPPUNIT_ASSERT((eventInfo.time < maxTimePerClosure * numClosures));
  };
};

#undef __FUNCT__
#define __FUNCT__ "RegisterIMeshStressSuite"
PetscErrorCode RegisterIMeshStressSuite() {
  CPPUNIT_TEST_SUITE_REGISTRATION(StressTestIMesh);
  PetscFunctionBegin;
  PetscFunctionReturn(0);
}

class MemoryTestIMesh : public CppUnit::TestFixture
{
  CPPUNIT_TEST_SUITE(MemoryTestIMesh);

  CPPUNIT_TEST(testPCICE);

  CPPUNIT_TEST_SUITE_END();
public:
  typedef ALE::IMesh<>          mesh_type;
  typedef mesh_type::sieve_type sieve_type;
  typedef mesh_type::point_type point_type;
protected:
  ALE::Obj<mesh_type> _mesh;
  int                 _debug; // The debugging level
  PetscInt            _iters; // The number of test repetitions
public:
  PetscErrorCode processOptions() {
    PetscErrorCode ierr;

    this->_debug = 0;
    this->_iters = 1;

    PetscFunctionBegin;
    ierr = PetscOptionsBegin(PETSC_COMM_WORLD, "", "Options for interval section stress test", "ISection");CHKERRQ(ierr);
      ierr = PetscOptionsInt("-debug", "The debugging level", "isection.c", this->_debug, &this->_debug, PETSC_NULL);CHKERRQ(ierr);
      ierr = PetscOptionsInt("-iterations", "The number of test repetitions", "isection.c", this->_iters, &this->_iters, PETSC_NULL);CHKERRQ(ierr);
    ierr = PetscOptionsEnd();CHKERRQ(ierr);
    PetscFunctionReturn(0);
  };

  /// Setup data.
  void setUp(void) {
    this->processOptions();
  };

  /// Tear down data.
  void tearDown(void) {};

  void testPCICE(void) {
    ALE::MemoryLogger& logger      = ALE::MemoryLogger::singleton();
    const char        *nameOld     = "PCICE Old";
    const char        *name        = "PCICE";
    const bool         interpolate = false;
    PetscLogDouble     osMemStart, osMemSemiEnd, osMemEnd, osMemOld, osMem;

    PetscMemoryGetCurrentUsage(&osMemStart);
    logger.setDebug(this->_debug);
    logger.stagePush(nameOld);
    {
      const int                             dim   = 2;
      ALE::Obj<mesh_type>                   mesh  = new mesh_type(PETSC_COMM_WORLD, dim, this->_debug);
      ALE::Obj<sieve_type>                  sieve = new sieve_type(mesh->comm(), this->_debug);
      const ALE::Obj<ALE::Mesh>             m     = new ALE::Mesh(mesh->comm(), dim, this->_debug);
      const ALE::Obj<ALE::Mesh::sieve_type> s     = new ALE::Mesh::sieve_type(mesh->comm(), this->_debug);
      int                                  *cells         = NULL;
      double                               *coordinates   = NULL;
      //const std::string&                    coordFilename = "../tutorials/data/ex1_2d.nodes";
      //const std::string&                    adjFilename   = "../tutorials/data/ex1_2d.lcon";
      const std::string&                    coordFilename = "data/3d.nodes";
      const std::string&                    adjFilename   = "data/3d.lcon";
      const bool                            useZeroBase   = true;
      int                                   numCells = 0, numVertices = 0, numCorners = dim+1;
      PetscErrorCode                        ierr;

      ALE::PCICE::Builder::readConnectivity(mesh->comm(), adjFilename, numCorners, useZeroBase, numCells, &cells);
      ALE::PCICE::Builder::readCoordinates(mesh->comm(), coordFilename, dim, numVertices, &coordinates);
      ALE::SieveBuilder<ALE::Mesh>::buildTopology(s, dim, numCells, cells, numVertices, interpolate, numCorners, -1, m->getArrowSection("orientation"));
      m->setSieve(s);
      m->stratify();
      PetscMemoryGetCurrentUsage(&osMemOld);
      logger.stagePush(name);
      mesh->setSieve(sieve);
      mesh_type::renumbering_type renumbering;
      ALE::ISieveConverter::convertSieve(*s, *sieve, renumbering, false);
      mesh->stratify();
      ALE::ISieveConverter::convertOrientation(*s, *sieve, renumbering, m->getArrowSection("orientation").ptr());
      ALE::SieveBuilder<PETSC_MESH_TYPE>::buildCoordinates(mesh, dim, coordinates);
      if (cells)       {ierr = PetscFree(cells);}
      if (coordinates) {ierr = PetscFree(coordinates);}
      logger.stagePop();
      PetscMemoryGetCurrentUsage(&osMem);
    }
    mesh_type::MeshNumberingFactory::singleton(PETSC_COMM_WORLD, this->_debug, true);
    ALE::Mesh::MeshNumberingFactory::singleton(PETSC_COMM_WORLD, this->_debug, true);
    logger.stagePop();
    PetscMemoryGetCurrentUsage(&osMemSemiEnd);
    // Reallocate difference
    const int diffSize = (int)((osMemOld-osMemStart)-(osMem-osMemOld))*0.8;
    char     *tmp      = new char[diffSize];
    std::cout << "Allocated " << diffSize << " bytes to check for empty space" << std::endl;
    for(int i = 0; i < diffSize; ++i) {
      tmp[i] = i;
    }
    PetscMemoryGetCurrentUsage(&osMemEnd);
    std::cout << std::endl << nameOld << " " << logger.getNumAllocations(nameOld) << " allocations " << logger.getAllocationTotal(nameOld) << " bytes" << std::endl;
    std::cout << std::endl << nameOld << " " << logger.getNumDeallocations(nameOld) << " deallocations " << logger.getDeallocationTotal(nameOld) << " bytes" << std::endl;
    std::cout << std::endl << name << " " << logger.getNumAllocations(name) << " allocations " << logger.getAllocationTotal(name) << " bytes" << std::endl;
    std::cout << std::endl << name << " " << logger.getNumDeallocations(name) << " deallocations " << logger.getDeallocationTotal(name) << " bytes" << std::endl;
    std::cout << std::endl << "osMemOld: " << osMemOld-osMemStart << " osMem: " << osMem-osMemOld << " osMemSemiEnd: " << osMemSemiEnd-osMemStart << " osMemEnd: " << osMemEnd-osMemStart << std::endl;
    std::cout << std::endl << osMemStart<<" "<<osMemOld <<" "<<osMem<<" "<<osMemSemiEnd<<" "<<osMemEnd << std::endl;
    CPPUNIT_ASSERT_EQUAL(logger.getNumAllocations(nameOld)+logger.getNumAllocations(name), logger.getNumDeallocations(nameOld)+logger.getNumDeallocations(name));
    CPPUNIT_ASSERT_EQUAL(logger.getAllocationTotal(nameOld)+logger.getAllocationTotal(name), logger.getDeallocationTotal(nameOld)+logger.getDeallocationTotal(name));
  };
};

#undef __FUNCT__
#define __FUNCT__ "RegisterIMeshMemorySuite"
PetscErrorCode RegisterIMeshMemorySuite() {
  CPPUNIT_TEST_SUITE_REGISTRATION(MemoryTestIMesh);
  PetscFunctionBegin;
  PetscFunctionReturn(0);
}
