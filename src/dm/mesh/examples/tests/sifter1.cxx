#include <petsc.h>
#include "unitTests.hh"

#include <cppunit/extensions/TestFactoryRegistry.h>
#include <cppunit/extensions/HelperMacros.h>

typedef ALE::Point                               point_type;
typedef ALE::Sifter<point_type, point_type, int> sifter_type;

typedef struct {
  int      debug; // The debugging level
  PetscInt iters; // The number of test repetitions
} Options;

#undef __FUNCT__
#define __FUNCT__ "ProcessOptions"
static PetscErrorCode ProcessOptions(MPI_Comm comm, Options *options)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  options->debug = 0;
  options->iters = 10000;

  ierr = PetscOptionsBegin(comm, "", "Options for sifter stress test", "Sieve");CHKERRQ(ierr);
    ierr = PetscOptionsInt("-debug", "The debugging level", "sifter1.c", options->debug, &options->debug, PETSC_NULL);CHKERRQ(ierr);
    ierr = PetscOptionsInt("-iterations", "The number of test repetitions", "sifter1.c", options->iters, &options->iters, PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsEnd();
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "ConeTest"
PetscErrorCode ConeTest(const ALE::Obj<sifter_type>& sifter, Options *options)
{
  ALE::Obj<sifter_type::traits::baseSequence> base = sifter->base();
  long numConeArrows = (long) base->size()*3;
  long count = 0;

  PetscFunctionBegin;
  ALE::LogStage  stage = ALE::LogStageRegister("Cone Test");
  PetscEvent     coneEvent;
  PetscErrorCode ierr;

  ierr = PetscLogEventRegister(&coneEvent, "Cone", PETSC_OBJECT_COOKIE);
  ALE::LogStagePush(stage);
  ierr = PetscLogEventBegin(coneEvent,0,0,0,0);
  for(int r = 0; r < options->iters; r++) {
    for(sifter_type::traits::baseSequence::iterator b_iter = base->begin(); b_iter != base->end(); ++b_iter) {
      const ALE::Obj<sifter_type::traits::coneSequence>& cone = sifter->cone(*b_iter);

      for(sifter_type::traits::coneSequence::iterator c_iter = cone->begin(); c_iter != cone->end(); ++c_iter) {
        count++;
      }
    }
  }
  ierr = PetscLogEventEnd(coneEvent,0,0,0,0);
  ALE::LogStagePop(stage);
  CPPUNIT_ASSERT_EQUAL(count, numConeArrows*options->iters);
  StageLog     stageLog;
  EventPerfLog eventLog;

  ierr = PetscLogGetStageLog(&stageLog);CHKERRQ(ierr);
  ierr = StageLogGetEventPerfLog(stageLog, stage, &eventLog);CHKERRQ(ierr);
  EventPerfInfo eventInfo = eventLog->eventInfo[coneEvent];

  CPPUNIT_ASSERT_EQUAL(eventInfo.count, 1);
  CPPUNIT_ASSERT_EQUAL((int) eventInfo.flops, 0);
  if (options->debug) {
    ierr = PetscPrintf(sifter->comm(), "Average time per cone: %gs\n", eventInfo.time/(options->iters*base->size()));CHKERRQ(ierr);
  }
  CPPUNIT_ASSERT((eventInfo.time < 2.0 * options->iters / 5000));
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SupportTest"
PetscErrorCode SupportTest(const ALE::Obj<sifter_type>& sifter, Options *options)
{
  ALE::Obj<sifter_type::traits::capSequence> cap = sifter->cap();
  long numSupportArrows = (long) ((cap->size() - 1)*3)/2;
  long count = 0;

  PetscFunctionBegin;
  ALE::LogStage  stage = ALE::LogStageRegister("Support Test");
  PetscEvent     supportEvent;
  PetscErrorCode ierr;

  ierr = PetscLogEventRegister(&supportEvent, "Support", PETSC_OBJECT_COOKIE);
  ALE::LogStagePush(stage);
  ierr = PetscLogEventBegin(supportEvent,0,0,0,0);
  for(int r = 0; r < options->iters; r++) {
    for(sifter_type::traits::baseSequence::iterator c_iter = cap->begin(); c_iter != cap->end(); ++c_iter) {
      const ALE::Obj<sifter_type::traits::supportSequence>& support = sifter->support(*c_iter);

      for(sifter_type::traits::supportSequence::iterator s_iter = support->begin(); s_iter != support->end(); ++s_iter) {
        count++;
      }
    }
  }
  ierr = PetscLogEventEnd(supportEvent,0,0,0,0);
  ALE::LogStagePop(stage);
  CPPUNIT_ASSERT_EQUAL(count, numSupportArrows*options->iters);
  StageLog     stageLog;
  EventPerfLog eventLog;

  ierr = PetscLogGetStageLog(&stageLog);CHKERRQ(ierr);
  ierr = StageLogGetEventPerfLog(stageLog, stage, &eventLog);CHKERRQ(ierr);
  EventPerfInfo eventInfo = eventLog->eventInfo[supportEvent];

  CPPUNIT_ASSERT_EQUAL(eventInfo.count, 1);
  CPPUNIT_ASSERT_EQUAL((int) eventInfo.flops, 0);
  if (options->debug) {
    ierr = PetscPrintf(sifter->comm(), "Average time per support: %gs\n", eventInfo.time/(options->iters*cap->size()));CHKERRQ(ierr);
  }
  CPPUNIT_ASSERT((eventInfo.time < 2.0 * options->iters / 5000));
  PetscFunctionReturn(0);
}

class TestSifter : public CppUnit::TestFixture
{
  CPPUNIT_TEST_SUITE(TestSifter);

  CPPUNIT_TEST(testCone);
  CPPUNIT_TEST(testSupport);

  CPPUNIT_TEST_SUITE_END();
protected:
  ALE::Obj<sifter_type> _sifter;
  Options               _options;

public :
  /// Setup data.
  void setUp(void) {
    ProcessOptions(PETSC_COMM_WORLD, &this->_options);
    this->_sifter = ALE::Test::SifterBuilder::createHatSifter<sifter_type>(PETSC_COMM_WORLD);
  };

  /// Tear down data.
  void tearDown(void) {};

  /// Test cone().
  void testCone(void) {
    ConeTest(this->_sifter, &this->_options);
  };

  /// Test support().
  void testSupport(void) {
    SupportTest(this->_sifter, &this->_options);
  };
};


#undef __FUNCT__
#define __FUNCT__ "RegisterSifterSuite"
PetscErrorCode RegisterSifterSuite() {
  CPPUNIT_TEST_SUITE_REGISTRATION (TestSifter);
  PetscFunctionBegin;
  PetscFunctionReturn(0);
}

#if 0

#undef __FUNCT__
#define __FUNCT__ "RunUnitTests"
PetscErrorCode RunUnitTests()
{ // main
  CppUnit::TestResultCollector result;

  PetscFunctionBegin;
  try {
    // Create event manager and test controller
    CppUnit::TestResult controller;

    // Add listener to collect test results
    controller.addListener(&result);

    // Add listener to show progress as tests run
    CppUnit::BriefTestProgressListener progress;
    controller.addListener(&progress);

    // Add top suite to test runner
    CppUnit::TestRunner runner;
    runner.addTest(CppUnit::TestFactoryRegistry::getRegistry().makeTest());
    runner.run(controller);

    // Print tests
    CppUnit::TextOutputter outputter(&result, std::cerr);
    outputter.write();
  } catch (...) {
    abort();
  }

  PetscFunctionReturn(result.wasSuccessful() ? 0 : 1);
}

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc, char *argv[])
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscInitialize(&argc, &argv, (char *) 0, help);CHKERRQ(ierr);
  ierr = PetscLogBegin();CHKERRQ(ierr);
  ierr = RunUnitTests();CHKERRQ(ierr);
  ierr = PetscFinalize();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#endif
