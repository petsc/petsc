#include <petsc.h>
#include <Sieve.hh>
#include <SieveAlgorithms.hh>
#include <Mesh.hh>
#include <Generator.hh>
#include "unitTests.hh"

#include <cppunit/extensions/TestFactoryRegistry.h>
#include <cppunit/extensions/HelperMacros.h>

typedef int                              point_type;
typedef ALE::Sieve<point_type, int, int> sieve_type;
typedef ALE::Bundle<sieve_type>          bundle_type;
typedef ALE::Mesh                        mesh_type;
typedef ALE::SieveAlg<bundle_type>       sieveAlg_type;
typedef ALE::SieveAlg<mesh_type>         sieveAlgM_type;

class TestHatSieve : public CppUnit::TestFixture
{
  CPPUNIT_TEST_SUITE(TestHatSieve);

  CPPUNIT_TEST(testClosure);
  CPPUNIT_TEST(testStar);

  CPPUNIT_TEST_SUITE_END();
protected:
  ALE::Obj<bundle_type> _bundle;
  ALE::Obj<sieve_type>  _sieve;
  int                   _debug; // The debugging level
  PetscInt              _iters; // The number of test repetitions

public :
  PetscErrorCode processOptions() {
    PetscErrorCode ierr;

    this->_debug = 0;
    this->_iters = 10000;

    PetscFunctionBegin;
    ierr = PetscOptionsBegin(PETSC_COMM_WORLD, "", "Options for sieve stress test", "Sieve");CHKERRQ(ierr);
      ierr = PetscOptionsInt("-debug", "The debugging level", "sieve1.c", this->_debug, &this->_debug, PETSC_NULL);CHKERRQ(ierr);
      ierr = PetscOptionsInt("-iterations", "The number of test repetitions", "sieve1.c", this->_iters, &this->_iters, PETSC_NULL);CHKERRQ(ierr);
    ierr = PetscOptionsEnd();CHKERRQ(ierr);
    PetscFunctionReturn(0);
  };
  /// Setup data.
  void setUp(void) {
    this->processOptions();
    this->_bundle = new bundle_type(PETSC_COMM_WORLD, this->_debug);
    this->_sieve  = ALE::Test::SifterBuilder::createHatSifter<sieve_type>(this->_bundle->comm());
    this->_bundle->setSieve(this->_sieve);
    this->_bundle->getArrowSection("orientation");
  };

  /// Tear down data.
  void tearDown(void) {};

  /// Test closure().
  void testClosure(void) {
    const ALE::Obj<sieve_type::traits::baseSequence>& base = this->_sieve->base();
    const int    baseSize          = base->size();
    const double maxTimePerClosure = 2.0e-5;
    const long   numClosurePoints  = (long) baseSize*4;
    long         count             = 0;

    ALE::LogStage  stage = ALE::LogStageRegister("Hat Closure Test");
    PetscEvent     closureEvent;
    PetscErrorCode ierr;

    ierr = PetscLogEventRegister(&closureEvent, "Closure", PETSC_OBJECT_COOKIE);
    ALE::LogStagePush(stage);
    ierr = PetscLogEventBegin(closureEvent,0,0,0,0);
    for(int r = 0; r < this->_iters; r++) {
      for(sieve_type::traits::baseSequence::iterator b_iter = base->begin(); b_iter != base->end(); ++b_iter) {
        const ALE::Obj<sieveAlg_type::coneArray>& closure = sieveAlg_type::closure(this->_bundle, *b_iter);

        for(sieveAlg_type::coneArray::iterator c_iter = closure->begin(); c_iter != closure->end(); ++c_iter) {
          count++;
        }
      }
    }
    ierr = PetscLogEventEnd(closureEvent,0,0,0,0);
    ALE::LogStagePop(stage);
    CPPUNIT_ASSERT_EQUAL(count, numClosurePoints*this->_iters);
    StageLog     stageLog;
    EventPerfLog eventLog;

    ierr = PetscLogGetStageLog(&stageLog);
    ierr = StageLogGetEventPerfLog(stageLog, stage, &eventLog);
    EventPerfInfo eventInfo = eventLog->eventInfo[closureEvent];

    CPPUNIT_ASSERT_EQUAL(eventInfo.count, 1);
    CPPUNIT_ASSERT_EQUAL((int) eventInfo.flops, 0);
    if (this->_debug) {
      ierr = PetscPrintf(this->_sieve->comm(), "Average time per closure: %gs\n", eventInfo.time/(this->_iters*baseSize));
    }
    CPPUNIT_ASSERT((eventInfo.time < maxTimePerClosure * baseSize * this->_iters));
  };

  /// Test star().
  void testStar(void) {
    const ALE::Obj<sieve_type::traits::capSequence>& cap = this->_sieve->cap();
    const int    capSize        = cap->size();
    const double maxTimePerStar = 2.0e-5;
    const long   numStarPoints  = (long) ((capSize - 1)*3)/2 + capSize;
    long         count          = 0;

    PetscFunctionBegin;
    ALE::LogStage  stage = ALE::LogStageRegister("Hat Star Test");
    PetscEvent     starEvent;
    PetscErrorCode ierr;

    ierr = PetscLogEventRegister(&starEvent, "Star", PETSC_OBJECT_COOKIE);
    ALE::LogStagePush(stage);
    ierr = PetscLogEventBegin(starEvent,0,0,0,0);
    for(int r = 0; r < this->_iters; r++) {
      for(sieve_type::traits::baseSequence::iterator c_iter = cap->begin(); c_iter != cap->end(); ++c_iter) {
        const ALE::Obj<sieveAlg_type::supportArray>& star = sieveAlg_type::star(this->_bundle, *c_iter);

        for(sieveAlg_type::supportArray::iterator s_iter = star->begin(); s_iter != star->end(); ++s_iter) {
          count++;
        }
      }
    }
    ierr = PetscLogEventEnd(starEvent,0,0,0,0);
    ALE::LogStagePop(stage);
    CPPUNIT_ASSERT_EQUAL(count, numStarPoints*this->_iters);
    StageLog     stageLog;
    EventPerfLog eventLog;

    ierr = PetscLogGetStageLog(&stageLog);
    ierr = StageLogGetEventPerfLog(stageLog, stage, &eventLog);
    EventPerfInfo eventInfo = eventLog->eventInfo[starEvent];

    CPPUNIT_ASSERT_EQUAL(eventInfo.count, 1);
    CPPUNIT_ASSERT_EQUAL((int) eventInfo.flops, 0);
    if (this->_debug) {
      ierr = PetscPrintf(this->_sieve->comm(), "Average time per star: %gs\n", eventInfo.time/(this->_iters*capSize));
    }
    CPPUNIT_ASSERT((eventInfo.time < maxTimePerStar * capSize * this->_iters));
  };
};

class TestGeneralSieve : public CppUnit::TestFixture
{
protected:
  ALE::Obj<mesh_type>  _bundle;
  ALE::Obj<sieve_type> _sieve;
  int                  _debug; // The debugging level
  PetscInt             _iters; // The number of test repetitions

public :
  virtual std::string getName() {return "General";};

  /// Tear down data.
  void tearDown(void) {};

  /// Test closure().
  void testClosure(void) {
    const ALE::Obj<sieve_type::traits::baseSequence>& base = this->_sieve->base();
    const int    baseSize          = base->size();
    const double maxTimePerClosure = 8.0e-5;
    long         count             = 0;
    std::string  stageName         = this->getName()+" Closure Test";

    ALE::LogStage  stage = ALE::LogStageRegister(stageName.c_str());
    PetscEvent     closureEvent;
    PetscErrorCode ierr;

    ierr = PetscLogEventRegister(&closureEvent, "Closure", PETSC_OBJECT_COOKIE);
    ALE::LogStagePush(stage);
    ierr = PetscLogEventBegin(closureEvent,0,0,0,0);
    for(int r = 0; r < this->_iters; r++) {
      for(sieve_type::traits::baseSequence::iterator b_iter = base->begin(); b_iter != base->end(); ++b_iter) {
        const ALE::Obj<sieveAlgM_type::coneArray>& closure = sieveAlgM_type::closure(this->_bundle, *b_iter);

        for(sieveAlgM_type::coneArray::iterator c_iter = closure->begin(); c_iter != closure->end(); ++c_iter) {
          count++;
        }
      }
    }
    ierr = PetscLogEventEnd(closureEvent,0,0,0,0);
    ALE::LogStagePop(stage);
    StageLog     stageLog;
    EventPerfLog eventLog;

    ierr = PetscLogGetStageLog(&stageLog);
    ierr = StageLogGetEventPerfLog(stageLog, stage, &eventLog);
    EventPerfInfo eventInfo = eventLog->eventInfo[closureEvent];

    CPPUNIT_ASSERT_EQUAL(eventInfo.count, 1);
    CPPUNIT_ASSERT_EQUAL((int) eventInfo.flops, 0);
    if (this->_debug) {
      ierr = PetscPrintf(this->_sieve->comm(), "Average time per closure: %gs\n", eventInfo.time/(baseSize*this->_iters));
      ierr = PetscPrintf(this->_sieve->comm(), "Average time per arrow in closure: %gs\n", eventInfo.time/count);
    }
    CPPUNIT_ASSERT((eventInfo.time <  maxTimePerClosure * baseSize * this->_iters));
  };

  /// Test star().
  void testStar(void) {
    const ALE::Obj<sieve_type::traits::capSequence>& cap = this->_sieve->cap();
    const int    capSize        = cap->size();
    const double maxTimePerStar = 8.0e-5;
    long         count          = 0;
    std::string  stageName      = this->getName()+" Star Test";

    ALE::LogStage  stage = ALE::LogStageRegister(stageName.c_str());
    PetscEvent     starEvent;
    PetscErrorCode ierr;

    ierr = PetscLogEventRegister(&starEvent, "Star", PETSC_OBJECT_COOKIE);
    ALE::LogStagePush(stage);
    ierr = PetscLogEventBegin(starEvent,0,0,0,0);
    for(int r = 0; r < this->_iters; r++) {
      for(sieve_type::traits::baseSequence::iterator c_iter = cap->begin(); c_iter != cap->end(); ++c_iter) {
        const ALE::Obj<sieveAlgM_type::supportArray>& star = sieveAlgM_type::star(this->_bundle, *c_iter);

        for(sieveAlgM_type::supportArray::iterator s_iter = star->begin(); s_iter != star->end(); ++s_iter) {
          count++;
        }
      }
    }
    ierr = PetscLogEventEnd(starEvent,0,0,0,0);
    ALE::LogStagePop(stage);
    StageLog     stageLog;
    EventPerfLog eventLog;

    ierr = PetscLogGetStageLog(&stageLog);
    ierr = StageLogGetEventPerfLog(stageLog, stage, &eventLog);
    EventPerfInfo eventInfo = eventLog->eventInfo[starEvent];

    CPPUNIT_ASSERT_EQUAL(eventInfo.count, 1);
    CPPUNIT_ASSERT_EQUAL((int) eventInfo.flops, 0);
    if (this->_debug) {
      ierr = PetscPrintf(this->_sieve->comm(), "Average time per star: %gs\n", eventInfo.time/(capSize*this->_iters));
      ierr = PetscPrintf(this->_sieve->comm(), "Average time per arrow in star: %gs\n", eventInfo.time/count);
    }
    CPPUNIT_ASSERT((eventInfo.time < maxTimePerStar * capSize * this->_iters));
  };
};

class TestSquareSieve : public TestGeneralSieve
{
  CPPUNIT_TEST_SUITE(TestSquareSieve);

  CPPUNIT_TEST(testClosure);
  CPPUNIT_TEST(testStar);

  CPPUNIT_TEST_SUITE_END();
public:
  virtual std::string getName() {return "Square";};

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
  /// Setup data.
  void setUp(void) {
    double lower[2] = {0.0, 0.0};
    double upper[2] = {1.0, 1.0};
    int    edges[2] = {2, 2};

    this->processOptions();
    ALE::Obj<mesh_type> mB = ALE::MeshBuilder::createSquareBoundary(PETSC_COMM_WORLD, lower, upper, edges, 0);
    this->_bundle = ALE::Generator::generateMesh(mB, true);
    this->_sieve  = this->_bundle->getSieve();
  };
};

class TestCubeSieve : public TestGeneralSieve
{
  CPPUNIT_TEST_SUITE(TestCubeSieve);

  CPPUNIT_TEST(testClosure);
  CPPUNIT_TEST(testStar);

  CPPUNIT_TEST_SUITE_END();
public:
  virtual std::string getName() {return "Cube";};

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
  /// Setup data.
  void setUp(void) {
    double lower[3] = {0.0, 0.0, 0.0};
    double upper[3] = {1.0, 1.0, 1.0};
    int    faces[3] = {1, 1, 1};

    this->processOptions();
    ALE::Obj<mesh_type> mB = ALE::MeshBuilder::createCubeBoundary(PETSC_COMM_WORLD, lower, upper, faces, 0);
    this->_bundle = ALE::Generator::generateMesh(mB, true);
    this->_sieve  = this->_bundle->getSieve();
  };
};

#undef __FUNCT__
#define __FUNCT__ "RegisterSieveSuite"
PetscErrorCode RegisterSieveSuite() {
  CPPUNIT_TEST_SUITE_REGISTRATION(TestHatSieve);
  CPPUNIT_TEST_SUITE_REGISTRATION(TestSquareSieve);
  CPPUNIT_TEST_SUITE_REGISTRATION(TestCubeSieve);
  PetscFunctionBegin;
  PetscFunctionReturn(0);
}
