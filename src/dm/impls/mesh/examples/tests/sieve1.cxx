#define ALE_MEM_LOGGING

#include <petscsys.h>
#include <../src/sys/plog/logimpl.h>
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

class StressTestHatSieve : public CppUnit::TestFixture
{
  CPPUNIT_TEST_SUITE(StressTestHatSieve);

  CPPUNIT_TEST(testClosure);
  CPPUNIT_TEST(testStar);

  CPPUNIT_TEST_SUITE_END();
protected:
  ALE::Obj<bundle_type> _bundle;
  ALE::Obj<sieve_type>  _sieve;
  int                   _debug; // The debugging level
  PetscInt              _iters; // The number of test repetitions

public :
  #undef __FUNCT__
  #define __FUNCT__ "processOptions"
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
    PetscLogEvent  closureEvent;
    PetscErrorCode ierr;

    ierr = PetscLogEventRegister("Closure", PETSC_OBJECT_CLASSID,&closureEvent);
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
    PetscStageLog     stageLog;
    PetscEventPerfLog eventLog;

    ierr = PetscLogGetStageLog(&stageLog);
    ierr = PetscStageLogGetEventPerfLog(stageLog, stage, &eventLog);
    PetscEventPerfInfo eventInfo = eventLog->eventInfo[closureEvent];

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

    ALE::LogStage  stage = ALE::LogStageRegister("Hat Star Test");
    PetscLogEvent  starEvent;
    PetscErrorCode ierr;

    ierr = PetscLogEventRegister("Star", PETSC_OBJECT_CLASSID,&starEvent);
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
    PetscStageLog     stageLog;
    PetscEventPerfLog eventLog;

    ierr = PetscLogGetStageLog(&stageLog);
    ierr = PetscStageLogGetEventPerfLog(stageLog, stage, &eventLog);
    PetscEventPerfInfo eventInfo = eventLog->eventInfo[starEvent];

    CPPUNIT_ASSERT_EQUAL(eventInfo.count, 1);
    CPPUNIT_ASSERT_EQUAL((int) eventInfo.flops, 0);
    if (this->_debug) {
      ierr = PetscPrintf(this->_sieve->comm(), "Average time per star: %gs\n", eventInfo.time/(this->_iters*capSize));
    }
    CPPUNIT_ASSERT((eventInfo.time < maxTimePerStar * capSize * this->_iters));
  };
};

class StressTestGeneralSieve : public CppUnit::TestFixture
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
    PetscLogEvent  closureEvent;
    PetscErrorCode ierr;

    ierr = PetscLogEventRegister("Closure", PETSC_OBJECT_CLASSID,&closureEvent);
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
    PetscStageLog     stageLog;
    PetscEventPerfLog eventLog;

    ierr = PetscLogGetStageLog(&stageLog);
    ierr = PetscStageLogGetEventPerfLog(stageLog, stage, &eventLog);
    PetscEventPerfInfo eventInfo = eventLog->eventInfo[closureEvent];

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
    PetscLogEvent  starEvent;
    PetscErrorCode ierr;

    ierr = PetscLogEventRegister("Star", PETSC_OBJECT_CLASSID,&starEvent);
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
    PetscStageLog     stageLog;
    PetscEventPerfLog eventLog;

    ierr = PetscLogGetStageLog(&stageLog);
    ierr = PetscStageLogGetEventPerfLog(stageLog, stage, &eventLog);
    PetscEventPerfInfo eventInfo = eventLog->eventInfo[starEvent];

    CPPUNIT_ASSERT_EQUAL(eventInfo.count, 1);
    CPPUNIT_ASSERT_EQUAL((int) eventInfo.flops, 0);
    if (this->_debug) {
      ierr = PetscPrintf(this->_sieve->comm(), "Average time per star: %gs\n", eventInfo.time/(capSize*this->_iters));
      ierr = PetscPrintf(this->_sieve->comm(), "Average time per arrow in star: %gs\n", eventInfo.time/count);
    }
    CPPUNIT_ASSERT((eventInfo.time < maxTimePerStar * capSize * this->_iters));
  };
};

class StressTestSquareSieve : public StressTestGeneralSieve
{
  CPPUNIT_TEST_SUITE(StressTestSquareSieve);

  CPPUNIT_TEST(testClosure);
  CPPUNIT_TEST(testStar);

  CPPUNIT_TEST_SUITE_END();
public:
  virtual std::string getName() {return "Square";};

  #undef __FUNCT__
  #define __FUNCT__ "processOptions"
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
    ALE::Obj<mesh_type> mB = ALE::MeshBuilder<mesh_type>::createSquareBoundary(PETSC_COMM_WORLD, lower, upper, edges, 0);
    this->_bundle = ALE::Generator<mesh_type>::generateMesh(mB, true);
    this->_sieve  = this->_bundle->getSieve();
  };
};

class StressTestCubeSieve : public StressTestGeneralSieve
{
  CPPUNIT_TEST_SUITE(StressTestCubeSieve);

  CPPUNIT_TEST(testClosure);
  CPPUNIT_TEST(testStar);

  CPPUNIT_TEST_SUITE_END();
public:
  virtual std::string getName() {return "Cube";};

  #undef __FUNCT__
  #define __FUNCT__ "processOptions"
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
    ALE::Obj<mesh_type> mB = ALE::MeshBuilder<mesh_type>::createCubeBoundary(PETSC_COMM_WORLD, lower, upper, faces, 0);
    this->_bundle = ALE::Generator<mesh_type>::generateMesh(mB, true);
    this->_sieve  = this->_bundle->getSieve();
  };
};

#undef __FUNCT__
#define __FUNCT__ "RegisterSieveStressSuite"
PetscErrorCode RegisterSieveStressSuite() {
  CPPUNIT_TEST_SUITE_REGISTRATION(StressTestHatSieve);
  CPPUNIT_TEST_SUITE_REGISTRATION(StressTestSquareSieve);
  CPPUNIT_TEST_SUITE_REGISTRATION(StressTestCubeSieve);
  PetscFunctionBegin;
  PetscFunctionReturn(0);
}

class FunctionTestHatSieve : public CppUnit::TestFixture
{
  CPPUNIT_TEST_SUITE(FunctionTestHatSieve);

  CPPUNIT_TEST(testJoin);

  CPPUNIT_TEST_SUITE_END();
protected:
  ALE::Obj<bundle_type> _bundle;
  ALE::Obj<sieve_type>  _sieve;
  int                   _debug; // The debugging level
  PetscInt              _iters; // The number of test repetitions

public :
  #undef __FUNCT__
  #define __FUNCT__ "processOptions"
  PetscErrorCode processOptions() {
    PetscErrorCode ierr;

    this->_debug = 0;
    this->_iters = 1;

    PetscFunctionBegin;
    ierr = PetscOptionsBegin(PETSC_COMM_WORLD, "", "Options for sieve functionality test", "Sieve");CHKERRQ(ierr);
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

  /// Test join().
  ///   THIS IS BROKEN
  void testJoin(void) {
    const ALE::Obj<sieve_type::traits::baseSequence>& base  = this->_sieve->base();
    sieve_type::point_type                            prior = *base->begin();
    sieve_type::traits::baseSequence::iterator        begin = base->begin();
    const sieve_type::traits::baseSequence::iterator  end   = base->end();

    this->_sieve->view("");
    ++begin;
    for(int r = 0; r < this->_iters; r++) {
      for(sieve_type::traits::baseSequence::iterator b_iter = begin; b_iter != end; ++b_iter) {
        std::cout << "Joining " << prior << " and " << *b_iter << std::endl;
        const ALE::Obj<sieve_type::supportSet> cells = this->_sieve->nJoin(prior, *b_iter, 1);

        for(sieve_type::supportSet::iterator s_iter = cells->begin(); s_iter != cells->end(); ++s_iter) {
          std::cout << "    --> " << *s_iter << std::endl;
        }
        CPPUNIT_ASSERT_EQUAL((int) cells->size(), 1);
        std::cout << "  --> " << *cells->begin() << " should be " << (*b_iter)/2 << std::endl;
        CPPUNIT_ASSERT_EQUAL(*cells->begin(), (*b_iter)/2);
        prior = *b_iter;
      }
    }
  };
};

#undef __FUNCT__
#define __FUNCT__ "RegisterSieveFunctionSuite"
PetscErrorCode RegisterSieveFunctionSuite() {
  CPPUNIT_TEST_SUITE_REGISTRATION(FunctionTestHatSieve);
  PetscFunctionBegin;
  PetscFunctionReturn(0);
}
