static char help[] = "Sifter Base Tests.\n\n";

#include <petsc.h>
#include "xsifterTest.hh"

typedef ALE::Test::XSifter::arrow_type  arrow_type;
typedef ALE::Test::XSifter::xsifter_type xsifter_type;
typedef ALE::Test::XSifter::RealBase RealBase;


#undef __FUNCT__
#define __FUNCT__ "BaseRangeTest"
PetscErrorCode BaseRangeTest(const ALE::Obj<xsifter_type>& xsifter, ALE::Test::XSifter::Options options, const char* xsifterName = NULL)
{
  ALE::Obj<xsifter_type::BaseSequence> base = xsifter->base();

  PetscFunctionBegin;
  ALE::LogStage stage = ALE::LogStageRegister("Base Test");
  ALE::LogStagePush(stage);
  if(xsifterName != NULL) {
    std::cout << xsifterName << ": ";
  }
  xsifter_type::BaseSequence::iterator bbegin, bend;
  bbegin = base->begin();
  bend   = base->end();
  std::cout << "Base: *bbegin: " << *bbegin << ", *bend: " << *bend << std::endl;
  ALE::LogStagePop(stage);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "BaseTest"
PetscErrorCode BaseTest(const ALE::Obj<xsifter_type>& xsifter, ALE::Test::XSifter::Options options, const char* xsifterName = NULL)
{
  
  ALE::Obj<xsifter_type::BaseSequence> base;
  if(options.predicate < 0) {
    base = xsifter->base();
  }
  else {
    base = xsifter->base((xsifter_type::predicate_type) options.predicate);
  }


  PetscFunctionBegin;
  ALE::LogStage stage = ALE::LogStageRegister("Base Test");
  ALE::LogStagePush(stage);
  if(xsifterName != NULL) {
    std::cout << xsifterName << ": ";
  }
  RealBase realBase;
  for(xsifter_type::iterator ii = xsifter->begin(); ii != xsifter->end(); ++ii){
    if(options.predicate < 0 || (xsifter_type::predicate_type) options.predicate == ii->predicate()) {
      realBase.insert(ii->target());
    }
  }
  xsifter_type::BaseSequence::iterator bbegin, bend, itor;
  bbegin = base->begin();
  bend   = base->end();
  itor   = bbegin;
  if(options.predicate < 0) {
    std::cout << "Base: " << std::endl;
  }
  else {
    std::cout << "Base (predicate: " << options.predicate << "): " << std::endl;
  }
  int counter = 0;
  for(; itor != bend; ++itor) {   
    std::cout << *itor << ", ";
    counter++;
  }
  std::cout << std::endl;
  if(options.predicate < 0) {
    std::cout << "realBase: " << std::endl;
  }
  else {
    std::cout << "realBase (predicate: " << options.predicate << "): " << std::endl;
  }
  for(RealBase::iterator ii = realBase.begin(); ii!=realBase.end(); ++ii) {   
    std::cout << *ii << ", ";
  }
  std::cout << std::endl;
  ALE::LogStagePop(stage);
  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc, char *argv[])
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscInitialize(&argc, &argv, (char *) 0, help); CHKERRQ(ierr);
  {
    ALE::Test::XSifter::Options        options;
    ALE::Obj<xsifter_type> xsifterFork = ALE::Test::XSifter::createForkXSifter(PETSC_COMM_SELF, options);
    ierr = BaseRangeTest(xsifterFork, options, "Fork XSifter"); CHKERRQ(ierr);
    ALE::Obj<xsifter_type> xsifterHat = ALE::Test::XSifter::createHatXSifter(PETSC_COMM_SELF, options);
    ierr = BaseRangeTest(xsifterHat, options, "Hat XSifter"); CHKERRQ(ierr);
    //
    ierr = BaseTest(xsifterFork, options, "Fork XSifter"); CHKERRQ(ierr);
    ierr = BaseTest(xsifterHat, options, "Hat XSifter"); CHKERRQ(ierr);
  }
  ierr = PetscFinalize();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
