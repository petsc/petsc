static char help[] = "Sifter Performance Stress Tests.\n\n";

#include <petsc.h>
#include "sifterTest.hh"

typedef ALE::Test::Point       Point;
typedef ALE::Test::sifter_type sifter_type;

typedef struct {
  int      debug; // The debugging level
  PetscInt iters; // The number of test repetitions
} Options;

#undef __FUNCT__
#define __FUNCT__ "ConeTest"
PetscErrorCode ConeTest(const ALE::Obj<sifter_type>& sifter, Options *options)
{
  ALE::Obj<sifter_type::traits::baseSequence> base = sifter->base();
  long numConeArrows = (long) base->size()*3;
  long count = 0;

  PetscFunctionBegin;
  ALE::LogStage stage = ALE::LogStageRegister("Cone Test");
  ALE::LogStagePush(stage);
  for(int r = 0; r < options->iters; r++) {
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
#define __FUNCT__ "SupportTest"
PetscErrorCode SupportTest(const ALE::Obj<sifter_type>& sifter, Options *options)
{
  ALE::Obj<sifter_type::traits::capSequence> cap = sifter->cap();
  long numSupportArrows = (long) ((cap->size() - 1)*3)/2;
  long count = 0;

  PetscFunctionBegin;
  ALE::LogStage stage = ALE::LogStageRegister("Support Test");
  ALE::LogStagePush(stage);
  for(int r = 0; r < options->iters; r++) {
    for(sifter_type::traits::baseSequence::iterator c_iter = cap->begin(); c_iter != cap->end(); ++c_iter) {
      const ALE::Obj<sifter_type::traits::supportSequence>& support = sifter->support(*c_iter);

      for(sifter_type::traits::supportSequence::iterator s_iter = support->begin(); s_iter != support->end(); ++s_iter) {
        count++;
      }
    }
  }
  ALE::LogStagePop(stage);
  if (count != numSupportArrows*options->iters) {
    SETERRQ2(PETSC_ERR_ARG_SIZ, "Cap count should be %d, not %d\n", numSupportArrows*options->iters, count);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "ProcessOptions"
PetscErrorCode ProcessOptions(MPI_Comm comm, Options *options)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  options->debug = 0;
  options->iters = 10000;

  ierr = PetscOptionsBegin(comm, "", "Options for sifter stress test", "Sieve");CHKERRQ(ierr);
    ierr = PetscOptionsInt("-debug", "The debugging level", "sifter1.c", 0, &options->debug, PETSC_NULL);CHKERRQ(ierr);
    ierr = PetscOptionsInt("-iterations", "The number of test repetitions", "sifter1.c", options->iters, &options->iters, PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsEnd();
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
  ierr = ProcessOptions(PETSC_COMM_WORLD, &options);CHKERRQ(ierr);
  {
    ALE::Obj<sifter_type> sifter = ALE::Test::SifterTest::createHatSifter(PETSC_COMM_WORLD);

    ierr = ConeTest(sifter, &options);CHKERRQ(ierr);
    ierr = SupportTest(sifter, &options);CHKERRQ(ierr);
  }
  ierr = PetscFinalize();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
