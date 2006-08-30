static char help[] = "Sifter Basic Functionality Tests.\n\n";

#include <petsc.h>
#include "xsifterTest.hh"

typedef ALE_X::Test::arrow_type  arrow_type;
typedef ALE_X::Test::sifter_type sifter_type;


#undef __FUNCT__
#define __FUNCT__ "BasicBaseTest"
PetscErrorCode BasicBaseTest(const ALE::Obj<sifter_type>& sifter, ALE_X::Test::Options options)
{
  ALE::Obj<sifter_type::BaseSequence> base = sifter->base();

  PetscFunctionBegin;
  ALE::LogStage stage = ALE::LogStageRegister("Base Test");
  ALE::LogStagePush(stage);
  std::cout << "Basic base:" << std::endl;
  sifter_type::BaseSequence::iterator begin, end, itor;
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
  ALE_X::Test::Options        options;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscInitialize(&argc, &argv, (char *) 0, help);CHKERRQ(ierr);
  {
    ALE_X::Test::Options options(PETSC_COMM_WORLD);
    ALE::Obj<sifter_type> sifter = ALE_X::Test::SifterTest::createForkSifter(PETSC_COMM_WORLD);

    ierr = BasicBaseTest(sifter, options);CHKERRQ(ierr);
  }
  ierr = PetscFinalize();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
