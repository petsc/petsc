#include <petsc.h>
#include <Sieve.hh>
#include <SieveAlgorithms.hh>
#include <Mesh.hh>
#include <Generator.hh>
#include "unitTests.hh"

#include <cppunit/extensions/TestFactoryRegistry.h>
#include <cppunit/extensions/HelperMacros.h>

typedef ALE::Mesh                    mesh_type;
typedef ALE::Mesh::point_type        point_type;
typedef ALE::Mesh::sieve_type        sieve_type;
typedef ALE::Mesh::real_section_type section_type;
typedef ALE::SieveAlg<mesh_type>     sieveAlg_type;

class TestGeneralMeshSection : public CppUnit::TestFixture
{
protected:
  ALE::Obj<mesh_type>    _mesh;
  ALE::Obj<sieve_type>   _sieve;
  ALE::Obj<section_type> _section;
  int                    _debug; // The debugging level
  PetscInt               _iters; // The number of test repetitions

public :
  virtual std::string getName() {return "General";};

  PetscErrorCode processOptions() {
    PetscErrorCode ierr;

    this->_debug = 0;
    this->_iters = 10000;

    PetscFunctionBegin;
    ierr = PetscOptionsBegin(PETSC_COMM_WORLD, "", "Options for square sieve stress test", "Sieve");CHKERRQ(ierr);
      ierr = PetscOptionsInt("-debug", "The debugging level", "sieve1.c", this->_debug, &this->_debug, PETSC_NULL);CHKERRQ(ierr);
      ierr = PetscOptionsInt("-iterations", "The number of test repetitions", "sieve1.c", this->_iters, &this->_iters, PETSC_NULL);CHKERRQ(ierr);
    ierr = PetscOptionsEnd();CHKERRQ(ierr);
    PetscFunctionReturn(0);
  };

  /// Tear down data.
  void tearDown(void) {};

  /// Test restrict().
  void testRestrict(const std::string testName, const double maxTimePerRestrict) {
    const ALE::Obj<mesh_type::label_sequence>& cells = this->_mesh->heightStratum(0);
    const int   numCells  = cells->size();
    std::string stageName = this->getName()+" Restrict Test";
    std::string eventName = testName+" Restrict";

    ALE::LogStage  stage = ALE::LogStageRegister(stageName.c_str());
    PetscEvent     restrictEvent;
    PetscErrorCode ierr;

    ierr = PetscLogEventRegister(&restrictEvent, eventName.c_str(), PETSC_OBJECT_COOKIE);
    ALE::LogStagePush(stage);
    ierr = PetscLogEventBegin(restrictEvent,0,0,0,0);
    for(int r = 0; r < this->_iters; r++) {
      for(mesh_type::label_sequence::iterator c_iter = cells->begin(); c_iter != cells->end(); ++c_iter) {
        const double *restrict = this->_mesh->restrict(this->_section, *c_iter);
      }
    }
    ierr = PetscLogEventEnd(restrictEvent,0,0,0,0);
    ALE::LogStagePop(stage);
    StageLog     stageLog;
    EventPerfLog eventLog;

    ierr = PetscLogGetStageLog(&stageLog);
    ierr = StageLogGetEventPerfLog(stageLog, stage, &eventLog);
    EventPerfInfo eventInfo = eventLog->eventInfo[restrictEvent];

    CPPUNIT_ASSERT_EQUAL(eventInfo.count, 1);
    CPPUNIT_ASSERT_EQUAL((int) eventInfo.flops, 0);
    if (this->_debug) {
      ierr = PetscPrintf(this->_sieve->comm(), " Average time per restrict: %gs\n", eventInfo.time/(numCells*this->_iters));
    }
    CPPUNIT_ASSERT((eventInfo.time <  maxTimePerRestrict * numCells * this->_iters));
  };

  /// Test restrict() with a precomputed atlas.
  void testRestrictPrecomputed(const std::string testName, const double maxTimePerRestrict) {
    const ALE::Obj<mesh_type::label_sequence>& cells = this->_mesh->heightStratum(0);
    const int   numCells  = cells->size();
    std::string stageName = this->getName()+" PrecompRestrict Test";
    std::string eventName = testName+" PrecompRestrict";

    const int tag = this->_mesh->calculateCustomAtlas(this->_section, cells);

    ALE::LogStage  stage = ALE::LogStageRegister(stageName.c_str());
    PetscEvent     restrictEvent;
    PetscErrorCode ierr;

    ierr = PetscLogEventRegister(&restrictEvent, eventName.c_str(), PETSC_OBJECT_COOKIE);
    ALE::LogStagePush(stage);
    ierr = PetscLogEventBegin(restrictEvent,0,0,0,0);
    for(int r = 0; r < this->_iters; r++) {
      for(int c = 0; c < numCells; ++c) {
        const double *restrict = this->_mesh->restrict(this->_section, tag, c);
      }
    }
    ierr = PetscLogEventEnd(restrictEvent,0,0,0,0);
    ALE::LogStagePop(stage);
    StageLog     stageLog;
    EventPerfLog eventLog;

    ierr = PetscLogGetStageLog(&stageLog);
    ierr = StageLogGetEventPerfLog(stageLog, stage, &eventLog);
    EventPerfInfo eventInfo = eventLog->eventInfo[restrictEvent];

    CPPUNIT_ASSERT_EQUAL(eventInfo.count, 1);
    CPPUNIT_ASSERT_EQUAL((int) eventInfo.flops, 0);
    if (this->_debug) {
      ierr = PetscPrintf(this->_sieve->comm(), " Average time per precomputed restrict: %gs\n", eventInfo.time/(numCells*this->_iters));
    }
    CPPUNIT_ASSERT((eventInfo.time <  maxTimePerRestrict * numCells * this->_iters));
  };
};

class TestSquareMeshSection : public TestGeneralMeshSection
{
  CPPUNIT_TEST_SUITE(TestSquareMeshSection);

  CPPUNIT_TEST(testLinearRestrict);
  CPPUNIT_TEST(testLinearRestrictPrecomp);
  CPPUNIT_TEST(testCubicRestrict);
  CPPUNIT_TEST(testCubicRestrictPrecomp);

  CPPUNIT_TEST_SUITE_END();
public:
  virtual std::string getName() {return "Square";};

  /// Setup data.
  void setUp(void) {
    double lower[2] = {0.0, 0.0};
    double upper[2] = {1.0, 1.0};
    int    edges[2] = {2, 2};

    this->_mesh    = PETSC_NULL;
    this->_sieve   = PETSC_NULL;
    this->_section = PETSC_NULL;
    this->processOptions();
    const ALE::Obj<mesh_type> mB = ALE::MeshBuilder::createSquareBoundary(PETSC_COMM_WORLD, lower, upper, edges, 0);
    this->_mesh  = ALE::Generator::generateMesh(mB, true);
    this->_sieve = this->_mesh->getSieve();
  };

  void testLinearRestrict(void) {
    this->_section = this->_mesh->getRealSection("coordinates");
    this->testRestrict("Linear", 2.0e-4);
  }

  void testLinearRestrictPrecomp(void) {
    this->_section = this->_mesh->getRealSection("coordinates");
    this->testRestrictPrecomputed("LinearPrecomp", 2.0e-4);
  }

  void testCubicRestrict(void) {
    this->_section = this->_mesh->getRealSection("test");
    this->_section->setFiberDimension(this->_mesh->depthStratum(0), 1);
    this->_section->setFiberDimension(this->_mesh->depthStratum(1), 2);
    this->_section->setFiberDimension(this->_mesh->depthStratum(2), 1);
    this->_mesh->allocate(this->_section);
    this->testRestrict("Cubic", 1.0e-3);
  }

  void testCubicRestrictPrecomp(void) {
    this->_section = this->_mesh->getRealSection("test");
    this->_section->setFiberDimension(this->_mesh->depthStratum(0), 1);
    this->_section->setFiberDimension(this->_mesh->depthStratum(1), 2);
    this->_section->setFiberDimension(this->_mesh->depthStratum(2), 1);
    this->_mesh->allocate(this->_section);
    this->testRestrictPrecomputed("CubicPrecomp", 1.0e-3);
  }
};

class TestBigSquareMeshSection : public TestGeneralMeshSection
{
  CPPUNIT_TEST_SUITE(TestBigSquareMeshSection);

  CPPUNIT_TEST(testNSRestrict);
  CPPUNIT_TEST(testNSRestrictPrecomp);

  CPPUNIT_TEST_SUITE_END();
public:
  virtual std::string getName() {return "Big Square";};

  /// Setup data.
  void setUp(void) {
    double lower[2] = {0.0, 0.0};
    double upper[2] = {1.0, 1.0};
    int    edges[2] = {32, 32};

    this->_mesh    = PETSC_NULL;
    this->_sieve   = PETSC_NULL;
    this->_section = PETSC_NULL;
    this->processOptions();
    const ALE::Obj<mesh_type> mB = ALE::MeshBuilder::createSquareBoundary(PETSC_COMM_WORLD, lower, upper, edges, 0);
    this->_mesh  = ALE::Generator::generateMesh(mB, true);
    this->_sieve = this->_mesh->getSieve();
  };

  void testNSRestrict(void) {
    this->_section = this->_mesh->getRealSection("test");
    this->_section->setFiberDimension(this->_mesh->depthStratum(0), 4);
    this->_section->setFiberDimension(this->_mesh->depthStratum(1), 3);
    this->_mesh->allocate(this->_section);
    this->testRestrict("NS", 3.0e-4);
  }

  void testNSRestrictPrecomp(void) {
    this->_section = this->_mesh->getRealSection("test");
    this->_section->setFiberDimension(this->_mesh->depthStratum(0), 4);
    this->_section->setFiberDimension(this->_mesh->depthStratum(1), 3);
    this->_mesh->allocate(this->_section);
    this->testRestrictPrecomputed("NSPrecomp", 3.0e-4);
  }
};

class TestCubeMeshSection : public TestGeneralMeshSection
{
  CPPUNIT_TEST_SUITE(TestCubeMeshSection);

  CPPUNIT_TEST(testLinearRestrict);
  CPPUNIT_TEST(testLinearRestrictPrecomp);
  CPPUNIT_TEST(testCubicRestrict);
  CPPUNIT_TEST(testCubicRestrictPrecomp);

  CPPUNIT_TEST_SUITE_END();
public:
  virtual std::string getName() {return "Cube";};

  /// Setup data.
  void setUp(void) {
    double lower[3] = {0.0, 0.0, 0.0};
    double upper[3] = {1.0, 1.0, 1.0};
    int    faces[3] = {1, 1, 1};

    this->processOptions();
    ALE::Obj<mesh_type> mB = ALE::MeshBuilder::createCubeBoundary(PETSC_COMM_WORLD, lower, upper, faces, 0);
    this->_mesh  = ALE::Generator::generateMesh(mB, true);
    this->_sieve = this->_mesh->getSieve();
  };

  void testLinearRestrict(void) {
    this->_section = this->_mesh->getRealSection("coordinates");
    this->testRestrict("Linear", 5.0e-4);
  }

  void testLinearRestrictPrecomp(void) {
    this->_section = this->_mesh->getRealSection("coordinates");
    this->testRestrictPrecomputed("LinearPrecomp", 5.0e-4);
  }

  void testCubicRestrict(void) {
    this->_section = this->_mesh->getRealSection("test");
    this->_section->setFiberDimension(this->_mesh->depthStratum(0), 1);
    this->_section->setFiberDimension(this->_mesh->depthStratum(1), 2);
    this->_section->setFiberDimension(this->_mesh->depthStratum(2), 1);
    this->_mesh->allocate(this->_section);
    this->testRestrict("Cubic", 5.0e-4);
  }

  void testCubicRestrictPrecomp(void) {
    this->_section = this->_mesh->getRealSection("test");
    this->_section->setFiberDimension(this->_mesh->depthStratum(0), 1);
    this->_section->setFiberDimension(this->_mesh->depthStratum(1), 2);
    this->_section->setFiberDimension(this->_mesh->depthStratum(2), 1);
    this->_mesh->allocate(this->_section);
    this->testRestrictPrecomputed("CubicPrecomp", 5.0e-4);
  }
};

class TestCubeMeshSectionNonInterp : public TestGeneralMeshSection
{
  CPPUNIT_TEST_SUITE(TestCubeMeshSectionNonInterp);

  CPPUNIT_TEST(testLinearRestrict);
  CPPUNIT_TEST(testLinearRestrictPrecomp);

  CPPUNIT_TEST_SUITE_END();
public:
  virtual std::string getName() {return "Non-Interpolated Cube";};

  /// Setup data.
  void setUp(void) {
    double lower[3] = {0.0, 0.0, 0.0};
    double upper[3] = {1.0, 1.0, 1.0};
    int    faces[3] = {1, 1, 1};

    this->processOptions();
    ALE::Obj<mesh_type> mB = ALE::MeshBuilder::createCubeBoundary(PETSC_COMM_WORLD, lower, upper, faces, 0);
    this->_mesh  = ALE::Generator::generateMesh(mB, false);
    this->_sieve = this->_mesh->getSieve();
  };

  void testLinearRestrict(void) {
    this->_section = this->_mesh->getRealSection("coordinates");
    this->testRestrict("Linear", 3.0e-5);
  }

  void testLinearRestrictPrecomp(void) {
    this->_section = this->_mesh->getRealSection("coordinates");
    this->testRestrictPrecomputed("LinearPrecomp", 3.0e-5);
  }
};

class TestBigCubeMeshSection : public TestGeneralMeshSection
{
  CPPUNIT_TEST_SUITE(TestBigCubeMeshSection);

  CPPUNIT_TEST(testNSRestrict);
  CPPUNIT_TEST(testNSRestrictPrecomp);

  CPPUNIT_TEST_SUITE_END();
public:
  virtual std::string getName() {return "Big Cube";};

  /// Setup data.
  void setUp(void) {
    double lower[3] = {0.0, 0.0, 0.0};
    double upper[3] = {1.0, 1.0, 1.0};
    int    faces[3] = {1, 1, 1};

    this->processOptions();
    ALE::Obj<mesh_type> mB = ALE::MeshBuilder::createCubeBoundary(PETSC_COMM_WORLD, lower, upper, faces, 0);
    this->_mesh  = ALE::Generator::refineMesh(ALE::Generator::generateMesh(mB, true), 1.0/6000, true);
    this->_sieve = this->_mesh->getSieve();
  };

  void testNSRestrict(void) {
    this->_section = this->_mesh->getRealSection("test");
    this->_section->setFiberDimension(this->_mesh->depthStratum(0), 4);
    this->_section->setFiberDimension(this->_mesh->depthStratum(1), 3);
    this->_mesh->allocate(this->_section);
    this->testRestrict("NS", 8.0e-4);
  }

  void testNSRestrictPrecomp(void) {
    this->_section = this->_mesh->getRealSection("test");
    this->_section->setFiberDimension(this->_mesh->depthStratum(0), 4);
    this->_section->setFiberDimension(this->_mesh->depthStratum(1), 3);
    this->_mesh->allocate(this->_section);
    this->testRestrictPrecomputed("NSPrecomp", 8.0e-4);
  }
};

#undef __FUNCT__
#define __FUNCT__ "RegisterSectionSuite"
PetscErrorCode RegisterSectionSuite() {
  CPPUNIT_TEST_SUITE_REGISTRATION(TestSquareMeshSection);
  CPPUNIT_TEST_SUITE_REGISTRATION(TestBigSquareMeshSection);
  CPPUNIT_TEST_SUITE_REGISTRATION(TestCubeMeshSection);
  CPPUNIT_TEST_SUITE_REGISTRATION(TestCubeMeshSectionNonInterp);
  CPPUNIT_TEST_SUITE_REGISTRATION(TestBigCubeMeshSection);
  PetscFunctionBegin;
  PetscFunctionReturn(0);
}
