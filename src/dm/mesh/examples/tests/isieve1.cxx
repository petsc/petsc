#define ALE_MEM_LOGGING

#include <petscsys.h>
#include <ISieve.hh>
#include <Mesh.hh>
#include <Generator.hh>
#include "unitTests.hh"

#include <cppunit/extensions/TestFactoryRegistry.h>
#include <cppunit/extensions/HelperMacros.h>

class FunctionTestISieve : public CppUnit::TestFixture
{
  CPPUNIT_TEST_SUITE(FunctionTestISieve);

  CPPUNIT_TEST(testBase);
  CPPUNIT_TEST(testConversion);
  CPPUNIT_TEST(testSerializationTriangularUninterpolated);
#if 0
  CPPUNIT_TEST(testSerializationTriangularInterpolated);
  CPPUNIT_TEST(testSerializationTetrahedralUninterpolated);
  CPPUNIT_TEST(testSerializationTetrahedralInterpolated);
#endif
  CPPUNIT_TEST(testConstruction);
  CPPUNIT_TEST(testTriangularUninterpolatedOrientedClosure);
  CPPUNIT_TEST(testTriangularInterpolatedOrientedClosure);
  CPPUNIT_TEST(testTetrahedralUninterpolatedOrientedClosure);
  CPPUNIT_TEST(testTetrahedralInterpolatedOrientedClosure);

  CPPUNIT_TEST_SUITE_END();
public:
  typedef ALE::IFSieve<int> sieve_type;
protected:
  ALE::Obj<sieve_type> _sieve;
  int                  _debug; // The debugging level
  PetscInt             _iters; // The number of test repetitions
  PetscInt             _size;  // The interval size
  PetscTruth           _onlyParallel; // Shut off serial tests
public:
  PetscErrorCode processOptions() {
    PetscErrorCode ierr;

    this->_debug = 0;
    this->_iters = 1;
    this->_size  = 10;
    this->_onlyParallel = PETSC_FALSE;

    PetscFunctionBegin;
    ierr = PetscOptionsBegin(PETSC_COMM_WORLD, "", "Options for interval section stress test", "ISieve");CHKERRQ(ierr);
      ierr = PetscOptionsInt("-debug", "The debugging level", "isieve.c", this->_debug, &this->_debug, PETSC_NULL);CHKERRQ(ierr);
      ierr = PetscOptionsInt("-iterations", "The number of test repetitions", "isieve.c", this->_iters, &this->_iters, PETSC_NULL);CHKERRQ(ierr);
      ierr = PetscOptionsInt("-size", "The interval size", "isieve.c", this->_size, &this->_size, PETSC_NULL);CHKERRQ(ierr);
      ierr = PetscOptionsTruth("-only_parallel", "Shut off serial tests", "isieve.c", this->_onlyParallel, &this->_onlyParallel, PETSC_NULL);CHKERRQ(ierr);
    ierr = PetscOptionsEnd();CHKERRQ(ierr);
    PetscFunctionReturn(0);
  };

  /// Setup data.
  void setUp(void) {
    this->processOptions();
    this->_sieve = new sieve_type(PETSC_COMM_WORLD, 0, this->_size*3+1, this->_debug);
    try {
      ALE::Test::SifterBuilder::createHatISifter<sieve_type>(PETSC_COMM_WORLD, *this->_sieve, this->_size, this->_debug);
    } catch (ALE::Exception e) {
      std::cerr << "ERROR: " << e << std::endl;
    }
  };

  /// Tear down data.
  void tearDown(void) {};

  template<typename Sieve, typename ISieve, typename Renumbering>
  void checkSieve(Sieve& sieve, const ISieve& isieve, Renumbering& renumbering, bool orderedSupports) {
    const ALE::Obj<typename Sieve::baseSequence>& base = sieve.base();

    for(typename Sieve::baseSequence::iterator b_iter = base->begin(); b_iter != base->end(); ++b_iter) {
      const ALE::Obj<typename Sieve::coneSequence>& cone = sieve.cone(*b_iter);
      ALE::ISieveVisitor::PointRetriever<ISieve>    retriever((int) pow((double) isieve.getMaxConeSize(), 3));

      isieve.cone(renumbering[*b_iter], retriever);
      const typename ISieve::point_type *icone = retriever.getPoints();
      int i = 0;

      CPPUNIT_ASSERT_EQUAL(cone->size(), retriever.getSize());
      for(typename Sieve::coneSequence::iterator c_iter = cone->begin(); c_iter != cone->end(); ++c_iter, ++i) {
        CPPUNIT_ASSERT_EQUAL(renumbering[*c_iter], icone[i]);
      }
    }
    const ALE::Obj<typename Sieve::capSequence>& cap = sieve.cap();

    for(typename Sieve::capSequence::iterator c_iter = cap->begin(); c_iter != cap->end(); ++c_iter) {
      const ALE::Obj<typename Sieve::supportSequence>& support = sieve.support(*c_iter);
      ALE::ISieveVisitor::PointRetriever<ISieve> retriever((int) pow((double) isieve.getMaxSupportSize(), 3));

      isieve.support(renumbering[*c_iter], retriever);
      const typename ISieve::point_type *isupport = retriever.getPoints();
      int i = 0;

      CPPUNIT_ASSERT_EQUAL(support->size(), retriever.getSize());
      if (orderedSupports) {
        for(typename Sieve::supportSequence::iterator s_iter = support->begin(); s_iter != support->end(); ++s_iter, ++i) {
          CPPUNIT_ASSERT_EQUAL(renumbering[*s_iter], isupport[i]);
        }
      } else {
        std::set<typename ISieve::point_type> isupportSet(&isupport[0], &isupport[retriever.getSize()]);
        std::set<typename Sieve::point_type>  supportSet;

        for(typename Sieve::supportSequence::iterator s_iter = support->begin(); s_iter != support->end(); ++s_iter, ++i) {
          supportSet.insert(renumbering[*s_iter]);
        }
        CPPUNIT_ASSERT(supportSet == isupportSet);
      }
    }
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

  void createTriangularMesh(bool interpolate, ALE::Obj<ALE::Mesh>& m, std::map<ALE::Mesh::point_type,sieve_type::point_type>& renumbering) {
    double lower[2] = {0.0, 0.0};
    double upper[2] = {1.0, 1.0};
    int    edges[2] = {2, 2};

    const ALE::Obj<ALE::Mesh> mB = ALE::MeshBuilder<ALE::Mesh>::createSquareBoundary(PETSC_COMM_WORLD, lower, upper, edges, 0);
    mB->getFactory()->clear(); // Necessary since we get pointer aliasing
    m = ALE::Generator<ALE::Mesh>::generateMesh(mB, interpolate);
    ALE::ISieveConverter::convertSieve(*m->getSieve(), *this->_sieve, renumbering);
    ALE::ISieveConverter::convertOrientation(*m->getSieve(), *this->_sieve, renumbering, m->getArrowSection("orientation").ptr());
    if (this->_debug > 1) {
      m->view("Square Mesh");
      this->_sieve->view("Square Sieve");
    }
  };

  void createTetrahedralMesh(bool interpolate, ALE::Obj<ALE::Mesh>& m, std::map<ALE::Mesh::point_type,sieve_type::point_type>& renumbering) {
    double lower[3] = {0.0, 0.0, 0.0};
    double upper[3] = {1.0, 1.0, 1.0};
    int    faces[3] = {1, 1, 1};

    const ALE::Obj<ALE::Mesh> mB = ALE::MeshBuilder<ALE::Mesh>::createCubeBoundary(PETSC_COMM_WORLD, lower, upper, faces, 0);
    mB->getFactory()->clear(); // Necessary since we get pointer aliasing
    m  = ALE::Generator<ALE::Mesh>::generateMesh(mB, interpolate);
    ALE::ISieveConverter::convertSieve(*m->getSieve(), *this->_sieve, renumbering);
    ALE::ISieveConverter::convertOrientation(*m->getSieve(), *this->_sieve, renumbering, m->getArrowSection("orientation").ptr());
    if (this->_debug > 1) {
      m->view("Cube Mesh");
      this->_sieve->view("Cube Sieve");
    }
  };

  void testBase(void) {
  };

  void testConversion(void) {
    typedef ALE::Mesh::sieve_type Sieve;
    typedef sieve_type            ISieve;
    double lower[2] = {0.0, 0.0};
    double upper[2] = {1.0, 1.0};
    int    edges[2] = {2, 2};
    const ALE::Obj<ALE::Mesh> m = ALE::MeshBuilder<ALE::Mesh>::createSquareBoundary(PETSC_COMM_WORLD, lower, upper, edges, 0);
    std::map<ALE::Mesh::point_type,sieve_type::point_type> renumbering;

    if (m->commSize() > 1) {
      if (this->_onlyParallel) return;
      CPPUNIT_FAIL("This test is not yet parallel");
    }
    ALE::ISieveConverter::convertSieve(*m->getSieve(), *this->_sieve, renumbering);
    this->checkSieve(*m->getSieve(), *this->_sieve, renumbering, true);
  };

  void testSerialization() {
    ALE::Obj<sieve_type> newSieve = new sieve_type(PETSC_COMM_WORLD, 0, this->_size*3+1, this->_debug);
    const char          *filename = "sieveTest.sav";

    ALE::ISieveSerializer::writeSieve(filename, *this->_sieve);
    ALE::ISieveSerializer::loadSieve(filename, *newSieve);
    unlink(filename);
    checkSieve(*this->_sieve, *newSieve);
  };

  void testSerializationTriangular(bool interpolate) {
    ALE::Obj<ALE::Mesh> m;
    std::map<ALE::Mesh::point_type,sieve_type::point_type> renumbering;

    createTriangularMesh(interpolate, m, renumbering);
    testSerialization();
  };

  void testSerializationTriangularInterpolated() {
    testSerializationTriangular(true);
  };

  void testSerializationTriangularUninterpolated() {
    testSerializationTriangular(false);
  };

  void testSerializationTetrahedral(bool interpolate) {
    ALE::Obj<ALE::Mesh> m;
    std::map<ALE::Mesh::point_type,sieve_type::point_type> renumbering;

    createTetrahedralMesh(interpolate, m, renumbering);
    testSerialization();
  };

  void testSerializationTetrahedralInterpolated() {
    testSerializationTetrahedral(true);
  };

  void testSerializationTetrahedralUninterpolated() {
    testSerializationTetrahedral(false);
  };

  void testConstruction(void) {
    const int          numCells    = 8;
    const int          numCorners  = 3;
    const int          numVertices = 9;
    const int          numPoints   = numCells+numVertices;
    const int          cones[24]   = {0,  3, 1,   3,  4, 1,   4,  2, 1,   4,  5,  2,   6,  4,  3,   6,  7,  4,   7,  5,  4,   7,  8,  5};
    const int          icones[24]  = {8, 11, 9,  11, 12, 9,  12, 10, 9,  12, 13, 10,  14, 12, 11,  14, 15, 12,  15, 13, 12,  15, 16, 13};

    {
      ALE::Obj<ALE::Mesh::sieve_type> s = new ALE::Mesh::sieve_type(PETSC_COMM_WORLD, this->_debug);
      ALE::SieveBuilder<ALE::Mesh>::buildTopology(s, 2, numCells, const_cast<int *>(cones), numVertices, false, numCorners);

      if (s->commSize() > 1) {
        if (this->_onlyParallel) return;
        CPPUNIT_FAIL("This test is not yet parallel");
      }
      ALE::Obj<sieve_type> sieve = new sieve_type(PETSC_COMM_WORLD, 0, numPoints, this->_debug);
      for(int c = 0; c < numCells; ++c) {
        sieve->setConeSize(c, numCorners);
      }
      sieve->symmetrizeSizes(numCells, numCorners, icones);
      sieve->allocate();
      for(int c = 0; c < numCells; ++c) {
        sieve->setCone(&icones[c*numCorners], c);
      }
      sieve->symmetrize();

      std::map<ALE::Mesh::point_type,sieve_type::point_type> renumbering;
      for(int i = 0; i < numPoints; ++i) renumbering[i] = i;
      this->checkSieve(*s, *sieve, renumbering, false);
    }
  };

  template<typename Mesh, typename Renumbering>
  void testOrientedClosure(const ALE::Obj<Mesh>& flexibleMesh, Renumbering& renumbering) {
    if (flexibleMesh->commSize() > 1) {
      if (this->_onlyParallel) return;
      CPPUNIT_FAIL("This test is not yet parallel");
    }
    typedef typename Mesh::sieve_type                  Sieve;
    typedef ALE::SieveAlg<Mesh>                        sieve_alg_type;
    typedef typename sieve_alg_type::orientedConeArray oConeArray;
    typedef sieve_type                                 ISieve;
    typedef ALE::ISieveVisitor::PointRetriever<ISieve> Visitor;

    const ALE::Obj<typename Sieve::baseSequence>& base        = flexibleMesh->getSieve()->base();
    const int                                     depth       = flexibleMesh->depth();
    const int                                     closureSize = std::max(0, (int) pow(this->_sieve->getConeSize(*base->begin()), depth)+1);
    Visitor                                       retriever(closureSize, true);

    for(typename Sieve::baseSequence::iterator b_iter = base->begin(); b_iter != base->end(); ++b_iter) {
      const ALE::Obj<oConeArray>& closure = sieve_alg_type::orientedClosure(flexibleMesh, *b_iter);

      retriever.clear();
      ALE::ISieveTraversal<ISieve>::orientedClosure(*this->_sieve, renumbering[*b_iter], retriever);
      const Visitor::oriented_point_type *icone = retriever.getOrientedPoints();
      int                                 ic    = 0;

      CPPUNIT_ASSERT_EQUAL(closure->size(), retriever.getOrientedSize());
      if (this->_debug) {std::cout << "Closure of " << *b_iter <<":"<< renumbering[*b_iter] << std::endl;}
      for(typename oConeArray::iterator c_iter = closure->begin(); c_iter != closure->end(); ++c_iter, ++ic) {
        if (this->_debug) {std::cout << "  point " << icone[ic].first << "  " << c_iter->first<<":"<<renumbering[c_iter->first] << std::endl;}
        CPPUNIT_ASSERT_EQUAL(renumbering[c_iter->first], icone[ic].first);
        if (this->_debug) {std::cout << "  order " << icone[ic].second << "  " << c_iter->second << std::endl;}
        CPPUNIT_ASSERT_EQUAL(c_iter->second, icone[ic].second);
      }
    }
  };

  void testTriangularOrientedClosure(bool interpolate) {
    ALE::Obj<ALE::Mesh> m;
    std::map<ALE::Mesh::point_type,sieve_type::point_type> renumbering;

    try {
      createTriangularMesh(interpolate, m, renumbering);
      testOrientedClosure(m, renumbering);
    } catch (ALE::Exception e) {
      std::cerr << e << std::endl;
      CPPUNIT_FAIL(e.msg());
    }
  };

  void testTriangularInterpolatedOrientedClosure() {
    testTriangularOrientedClosure(true);
  };

  void testTriangularUninterpolatedOrientedClosure() {
    testTriangularOrientedClosure(false);
  };

  void testTetrahedralOrientedClosure(bool interpolate) {
    ALE::Obj<ALE::Mesh> m;
    std::map<ALE::Mesh::point_type,sieve_type::point_type> renumbering;

    createTetrahedralMesh(interpolate, m, renumbering);
    testOrientedClosure(m, renumbering);
  };

  void testTetrahedralInterpolatedOrientedClosure() {
    testTetrahedralOrientedClosure(true);
  };

  void testTetrahedralUninterpolatedOrientedClosure() {
    testTetrahedralOrientedClosure(false);
  };
};

#undef __FUNCT__
#define __FUNCT__ "RegisterISieveFunctionSuite"
PetscErrorCode RegisterISieveFunctionSuite() {
  CPPUNIT_TEST_SUITE_REGISTRATION(FunctionTestISieve);
  PetscFunctionBegin;
  PetscFunctionReturn(0);
}

class MemoryTestISieve : public CppUnit::TestFixture
{
  CPPUNIT_TEST_SUITE(MemoryTestISieve);

  CPPUNIT_TEST(testTriangularInterpolatedSieve);
  CPPUNIT_TEST(testTriangularUninterpolatedSieve);
  CPPUNIT_TEST(testConversion);

  CPPUNIT_TEST_SUITE_END();
public:
  typedef ALE::IFSieve<int> sieve_type;
protected:
  ALE::Obj<sieve_type> _sieve;
  int                  _debug; // The debugging level
  PetscInt             _iters; // The number of test repetitions
public:
  PetscErrorCode processOptions() {
    PetscErrorCode ierr;

    this->_debug = 0;
    this->_iters = 1;

    PetscFunctionBegin;
    ierr = PetscOptionsBegin(PETSC_COMM_WORLD, "", "Options for interval section stress test", "ISieve");CHKERRQ(ierr);
      ierr = PetscOptionsInt("-debug", "The debugging level", "isieve.c", this->_debug, &this->_debug, PETSC_NULL);CHKERRQ(ierr);
      ierr = PetscOptionsInt("-iterations", "The number of test repetitions", "isieve.c", this->_iters, &this->_iters, PETSC_NULL);CHKERRQ(ierr);
    ierr = PetscOptionsEnd();CHKERRQ(ierr);
    PetscFunctionReturn(0);
  };

  /// Setup data.
  void setUp(void) {
    this->processOptions();
  };

  /// Tear down data.
  void tearDown(void) {};

  void testTriangularInterpolatedSieve(void) {
    ALE::MemoryLogger& logger      = ALE::MemoryLogger::singleton();
    const char        *name        = "ISieve I";
    const int          numCells    = 8;
    const int          numCorners  = 3;
    const int          numVertices = 9;
    const int          numEdges    = 16;
    const int          numPoints   = numCells+numVertices+numEdges;
    const int          cones[56]   = {19, 20, 17,  24, 21, 20,  22, 18, 21,  25, 23, 22,  27, 24, 26,  31, 28, 27,  29, 25, 28,  32, 30, 29,
                                      8, 9,  9, 10,  8, 11,  9, 11,  9, 12,  10, 12,  10, 13,  11, 12,  12, 13,  11, 14,  12, 14,  12, 15,  13, 15,  13, 16,  14, 15,  15, 16};

    logger.setDebug(this->_debug);
    logger.stagePush(name);
    {
      ALE::Obj<sieve_type> sieve = new sieve_type(PETSC_COMM_WORLD, 0, numPoints, this->_debug);

      for(int c = 0; c < numCells; ++c) {
        sieve->setConeSize(c, numCorners);
      }
      for(int e = numCells+numVertices; e < numCells+numVertices+numEdges; ++e) {
        sieve->setConeSize(e, 2);
      }
      sieve->symmetrizeSizes(numCells, numCorners, cones);
      sieve->symmetrizeSizes(numEdges, 2, &cones[numCells*numCorners]);
      sieve->allocate();
      for(int c = 0; c < numCells; ++c) {
        sieve->setCone(&cones[c*numCorners], c);
      }
      for(int e = 0; e < numEdges; ++e) {
        sieve->setCone(&cones[e*2+numCells*numCorners], e);
      }
      sieve->symmetrize();
    }
    logger.stagePop();
    const int numArrows = numCells*numCorners+numEdges*2;
    const int bytes     = 4 /*Obj*/ + (numPoints+1)*4 /*coneOffsets*/ + (numPoints+1)*4 /*supportOffsets*/ +
      (numArrows)*4 /*cones*/ + numArrows*4 /*coneOrientations*/ + numArrows*4 /*supports*/ + (numPoints+1)*4 /*offsets*/;
    CPPUNIT_ASSERT_EQUAL_MESSAGE("Invalid number of allocations", 7, logger.getNumAllocations(name));
    CPPUNIT_ASSERT_EQUAL_MESSAGE("Invalid number of deallocations", 7, logger.getNumDeallocations(name));
    CPPUNIT_ASSERT_EQUAL_MESSAGE("Invalid number of bytes allocated", bytes, logger.getAllocationTotal(name));
    CPPUNIT_ASSERT_EQUAL_MESSAGE("Invalid number of bytes deallocated", bytes, logger.getDeallocationTotal(name));
  };

  void testTriangularUninterpolatedSieve(void) {
    ALE::MemoryLogger& logger      = ALE::MemoryLogger::singleton();
    const char        *name        = "ISieve II";
    const int          numCells    = 8;
    const int          numCorners  = 3;
    const int          numVertices = 9;
    const int          numPoints   = numCells+numVertices;
    const int          cones[24]   = {8, 11, 9,  11, 12, 9,  12, 10, 9,  12, 13, 10,  14, 12, 11,  14, 15, 12,  15, 13, 12,  15, 16, 13};

    logger.setDebug(this->_debug);
    logger.stagePush(name);
    {
      ALE::Obj<sieve_type> sieve = new sieve_type(PETSC_COMM_WORLD, 0, numPoints, this->_debug);

      for(int c = 0; c < numCells; ++c) {
        sieve->setConeSize(c, numCorners);
      }
      sieve->symmetrizeSizes(numCells, numCorners, cones);
      sieve->allocate();
      for(int c = 0; c < numCells; ++c) {
        sieve->setCone(&cones[c*numCorners], c);
      }
      sieve->symmetrize();
    }
    logger.stagePop();
    const int numArrows = numCells*numCorners;
    const int bytes     = 4 /*Obj*/ + (numPoints+1)*4 /*coneOffsets*/ + (numPoints+1)*4 /*supportOffsets*/ +
      numArrows*4 /*cones*/ + numArrows*4 /*coneOrientations*/ + numArrows*4 /*supports*/ + (numPoints+1)*4 /*offsets*/;
    CPPUNIT_ASSERT_EQUAL_MESSAGE("Invalid number of allocations", 7, logger.getNumAllocations(name));
    CPPUNIT_ASSERT_EQUAL_MESSAGE("Invalid number of deallocations", 7, logger.getNumDeallocations(name));
    CPPUNIT_ASSERT_EQUAL_MESSAGE("Invalid number of bytes allocated", bytes, logger.getAllocationTotal(name));
    CPPUNIT_ASSERT_EQUAL_MESSAGE("Invalid number of bytes deallocated", bytes, logger.getDeallocationTotal(name));
  };

  void testConversion(void) {
    ALE::MemoryLogger& logger      = ALE::MemoryLogger::singleton();
    const char        *name        = "ISieve III";
    const char        *nameOld     = "Sieve III";
    const int          numCells    = 8;
    const int          numCorners  = 3;
    const int          numVertices = 9;
    const int          numPoints   = numCells+numVertices;
    const int          cones[24]   = {0, 3, 1,  3, 4, 1,  4, 2, 1,  4, 5, 2,  6, 4, 3,  6, 7, 4,  7, 5, 4,  7, 8, 5};

    logger.setDebug(this->_debug);
    logger.stagePush(name);
    {
      logger.stagePush(nameOld);
      ALE::Obj<ALE::Mesh::sieve_type> s = new ALE::Mesh::sieve_type(PETSC_COMM_WORLD, this->_debug);
      ALE::SieveBuilder<ALE::Mesh>::buildTopology(s, 2, numCells, const_cast<int *>(cones), numVertices, false, numCorners);
      logger.stagePop();

      ALE::Obj<sieve_type> sieve = new sieve_type(PETSC_COMM_WORLD, this->_debug);
      std::map<ALE::Mesh::point_type,sieve_type::point_type> renumbering;

      ALE::ISieveConverter::convertSieve(*s, *sieve, renumbering);
    }
    logger.stagePop();
    std::cout << std::endl << nameOld << " " << logger.getNumAllocations(nameOld) << " allocations " << logger.getAllocationTotal(nameOld) << " bytes" << std::endl;
    std::cout << std::endl << name << " " << logger.getNumAllocations(name) << " allocations " << logger.getAllocationTotal(name) << " bytes" << std::endl;
    const int numArrows = numCells*numCorners;
    const int bytes     = 4 /*Obj*/ + (numPoints+1)*4 /*coneOffsets*/ + (numPoints+1)*4 /*supportOffsets*/ +
      numArrows*4 /*cones*/ + numArrows*4 /*coneOrientations*/ + numArrows*4 /*supports*/ +
      4 /*Obj*/ + 8 /*baseSeq*/ + 4 /*Obj*/ + 8 /*capSeq*/;
    CPPUNIT_ASSERT_EQUAL_MESSAGE("Invalid number of allocations", 10, logger.getNumAllocations(name));
    CPPUNIT_ASSERT_EQUAL_MESSAGE("Invalid number of bytes allocated", bytes, logger.getAllocationTotal(name));

    std::cout << std::endl << name << " " << logger.getNumDeallocations(name) << " deallocations " << logger.getDeallocationTotal(name) << " bytes" << std::endl;
    CPPUNIT_ASSERT_EQUAL_MESSAGE("Invalid number of deallocations", 10+80, logger.getNumDeallocations(name)+logger.getNumDeallocations(nameOld));
    CPPUNIT_ASSERT_EQUAL_MESSAGE("Invalid number of bytes deallocated", bytes+5720, logger.getDeallocationTotal(name)+logger.getDeallocationTotal(nameOld));
  };
};

#undef __FUNCT__
#define __FUNCT__ "RegisterISieveMemorySuite"
PetscErrorCode RegisterISieveMemorySuite() {
  CPPUNIT_TEST_SUITE_REGISTRATION(MemoryTestISieve);
  PetscFunctionBegin;
  PetscFunctionReturn(0);
}
