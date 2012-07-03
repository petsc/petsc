#define ALE_MEM_LOGGING

#include <petscsys.h>
#include <../src/sys/plog/logimpl.h>
#include <IField.hh>
#include "unitTests.hh"

#include <cppunit/extensions/TestFactoryRegistry.h>
#include <cppunit/extensions/HelperMacros.h>

template<typename Section_>
class StressTestSection : public CppUnit::TestFixture
{
public:
  typedef Section_ section_type;
protected:
  ALE::Obj<section_type> _section;
  int                    _debug; // The debugging level
  PetscInt               _iters; // The number of test repetitions
  PetscInt               _size;  // The interval size
public :
  virtual std::string getName() {return "General";};

  #undef __FUNCT__
  #define __FUNCT__ "processOptions"
  PetscErrorCode processOptions() {
    PetscErrorCode ierr;

    this->_debug = 0;
    this->_iters = 10000;
    this->_size  = 10000;

    PetscFunctionBegin;
    ierr = PetscOptionsBegin(PETSC_COMM_WORLD, "", "Options for interval section stress test", "ISection");CHKERRQ(ierr);
      ierr = PetscOptionsInt("-debug", "The debugging level", "isection.c", this->_debug, &this->_debug, PETSC_NULL);CHKERRQ(ierr);
      ierr = PetscOptionsInt("-iterations", "The number of test repetitions", "isection.c", this->_iters, &this->_iters, PETSC_NULL);CHKERRQ(ierr);
      ierr = PetscOptionsInt("-size", "The interval size", "isection.c", this->_size, &this->_size, PETSC_NULL);CHKERRQ(ierr);
    ierr = PetscOptionsEnd();CHKERRQ(ierr);
    PetscFunctionReturn(0);
  };

  /// Tear down data.
  void tearDown(void) {};

  /// Test restrict().
  void testRestrictPoint(const std::string testName, const double maxTimePerRestrict) {
    const typename section_type::chart_type& chart = this->_section->getChart();
    const int   numPoints = chart.size();
    std::string stageName = this->getName()+" RestrictPoint Test";
    std::string eventName = testName+" RestrictPoint";

    ALE::LogStage  stage = ALE::LogStageRegister(stageName.c_str());
    PetscLogEvent  restrictEvent;
    PetscErrorCode ierr;

    ierr = PetscLogEventRegister(eventName.c_str(), PETSC_OBJECT_CLASSID,&restrictEvent);
    ALE::LogStagePush(stage);
    ierr = PetscLogEventBegin(restrictEvent,0,0,0,0);
    for(int r = 0; r < this->_iters; r++) {
      for(typename section_type::chart_type::const_iterator c_iter = chart.begin(); c_iter != chart.end(); ++c_iter) {
        const typename section_type::value_type *restrict = this->_section->restrictPoint(*c_iter);
        CPPUNIT_ASSERT(restrict != NULL);
      }
    }
    ierr = PetscLogEventEnd(restrictEvent,0,0,0,0);
    ALE::LogStagePop(stage);
    PetscStageLog     stageLog;
    PetscEventPerfLog eventLog;

    ierr = PetscLogGetStageLog(&stageLog);
    ierr = PetscStageLogGetEventPerfLog(stageLog, stage, &eventLog);
    PetscEventPerfInfo eventInfo = eventLog->eventInfo[restrictEvent];

    CPPUNIT_ASSERT_EQUAL(eventInfo.count, 1);
    CPPUNIT_ASSERT_EQUAL((int) eventInfo.flops, 0);
    if (this->_debug) {
      ierr = PetscPrintf(this->_section->comm(), " Average time per restrictPoint: %gs\n", eventInfo.time/(numPoints*this->_iters));
    }
    CPPUNIT_ASSERT((eventInfo.time <  maxTimePerRestrict * numPoints * this->_iters));
  };
};

class StressTestIConstantSection : public StressTestSection<ALE::IConstantSection<int, double> >
{
  CPPUNIT_TEST_SUITE(StressTestIConstantSection);

  CPPUNIT_TEST(testConstantRestrictPoint);

  CPPUNIT_TEST_SUITE_END();
public:
  virtual std::string getName() {return "Constant";};

  /// Setup data.
  void setUp(void) {
    this->processOptions();
    this->_section = new section_type(PETSC_COMM_WORLD, 0, this->_size, 1.0, this->_debug);
  };

  void testConstantRestrictPoint(void) {
    this->testRestrictPoint("Constant", 1.5e-6);
  }
};

class StressTestIUniformSection : public StressTestSection<ALE::IUniformSection<int, double> >
{
  CPPUNIT_TEST_SUITE(StressTestIUniformSection);

  CPPUNIT_TEST(testUniformRestrictPoint);

  CPPUNIT_TEST_SUITE_END();
public:
  virtual std::string getName() {return "Uniform";};

  /// Setup data.
  void setUp(void) {
    this->processOptions();
    this->_section = new section_type(PETSC_COMM_WORLD, 0, this->_size, this->_debug);
    this->_section->allocatePoint();
  };

  void testUniformRestrictPoint(void) {
    this->testRestrictPoint("Uniform", 1.5e-6);
  }
};

#undef __FUNCT__
#define __FUNCT__ "RegisterISectionStressSuite"
PetscErrorCode RegisterISectionStressSuite() {
  CPPUNIT_TEST_SUITE_REGISTRATION(StressTestIConstantSection);
  CPPUNIT_TEST_SUITE_REGISTRATION(StressTestIUniformSection);
  PetscFunctionBegin;
  PetscFunctionReturn(0);
}
