static char help[] = "Sifter Basic Functionality Tests.\n\n";

#include <petsc.h>
#include "xsifterTest.hh"

typedef ALE_X::Test::arrow_type  arrow_type;
typedef ALE_X::Test::sifter_type sifter_type;


#undef __FUNCT__
#define __FUNCT__ "BasicConeTest"
PetscErrorCode BasicConeTest(const ALE_X::Obj<sifter_type>& sifter, Options options)
{
  ALE_X::Obj<sifter_type::ConeSequence> cone = sifter_type::ConeSequence();
  long count = 0;

  PetscFunctionBegin;
  ALE::LogStage stage = ALE::LogStageRegister(__FUNCT__);
  ALE::LogStagePush(stage);
  // CONTINUE: 1) fix test to retrieve the base, then the cones and send them to cout; 
  //           2) fix default comms in "main" to be PETSC_COMM_SELF
  for(int r = 0; r < options.iters; r++) {
    for(sifter_type::traits::baseSequence::iterator b_iter = base->begin(); b_iter != base->end(); ++b_iter) {
      const ALE::Obj<sifter_type::traits::coneSequence>& cone = sifter->cone(*b_iter);

      for(sifter_type::traits::coneSequence::iterator c_iter = cone->begin(); c_iter != cone->end(); ++c_iter) {
        count++;
      }
    }
  }
  ALE::LogStagePop(stage);
  if (count != numConeArrows*options->iters) {
    SETERRQ2(PETSC_ERR_ARG_SIZ, "Cap count should be %d, not %d\n", numConeArrows*options->iters, count);
  }
  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc, char *argv[])
{
  Options        options;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscInitialize(&argc, &argv, (char *) 0, help);CHKERRQ(ierr);
  {
    ALE_X::Test::Options options(PETSC_COMM_WORLD);
    ALE::Obj<sifter_type> sifter = ALE_X::Test::SifterTest::createForkSifter(PETSC_COMM_WORLD);

    ierr = BasicConeTest(sifter, options);CHKERRQ(ierr);
  }
  ierr = PetscFinalize();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
