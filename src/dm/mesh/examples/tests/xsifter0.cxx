static char help[] = "Sifter Basic Functionality Tests.\n\n";

#include <petsc.h>
#include "xsifterTest.hh"

typedef ALE::Test::arrow_type  arrow_type;
typedef ALE::Test::xsifter_type xsifter_type;


#undef __FUNCT__
#define __FUNCT__ "BasicBaseTest"
PetscErrorCode BasicBaseTest(const ALE::Obj<xsifter_type>& xsifter, ALE::Test::Options options)
{
  ALE::Obj<xsifter_type::BaseSequence> base = xsifter->base();

  PetscFunctionBegin;
  ALE::LogStage stage = ALE::LogStageRegister("Base Test");
  ALE::LogStagePush(stage);
  std::cout << "Basic base:" << std::endl;
  xsifter_type::BaseSequence::iterator begin, end, itor;
  begin = base->begin();
  end   = base->end();
  itor = begin;
  std::cout << *itor;
  for(; itor != end; ++itor) {
    std::cout << ", " << *itor;
  }
  ALE::LogStagePop(stage);
  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc, char *argv[])
{
  ALE::Test::Options        options;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscInitialize(&argc, &argv, (char *) 0, help);CHKERRQ(ierr);
  {
    ALE::Test::Options options(PETSC_COMM_WORLD);
    ALE::Obj<xsifter_type> xsifter = ALE::Test::XSifterTest::createForkXSifter(PETSC_COMM_WORLD);

    ierr = BasicBaseTest(xsifter, options);CHKERRQ(ierr);
  }
  ierr = PetscFinalize();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
