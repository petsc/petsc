static char help[] = "Demonstrates restricting the -help output to specific manual sections with -help mansec.\n\n";

#include <petscsys.h>
#include <petscoptions.h>

int main(int argc, char **argv)
{
  PetscReal   r1 = 0., r2 = -1.;
  PetscReal   s1 = 0., s2 = -1.;
  PetscInt    i1 = 0, i2 = -1;
  PetscRandom rnd;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, NULL, help));

  /* the last argument of PetscOptionsBegin() is the manual section that -help matches against */
  PetscOptionsBegin(PETSC_COMM_WORLD, NULL, "Scaled units options 1", "Sec1");
  PetscCall(PetscOptionsReal("-r1", "r1: real", "ManPage1", r1, &r1, NULL));
  PetscCall(PetscOptionsReal("-s1", "s1: real", "ManPage1", s1, &s1, NULL));
  PetscCall(PetscOptionsInt("-i1", "i1: int", "ManPage1", i1, &i1, NULL));
  PetscOptionsEnd();

  PetscOptionsBegin(PETSC_COMM_WORLD, NULL, "Scaled units options 2", "Sec2");
  PetscCall(PetscOptionsReal("-r2", "r2: real", "ManPage2", r2, &r2, NULL));
  PetscCall(PetscOptionsReal("-s2", "s2: real", "ManPage2", s2, &s2, NULL));
  PetscCall(PetscOptionsInt("-i2", "i2: int", "ManPage2", i2, &i2, NULL));
  PetscOptionsEnd();

  /* PetscRandomSetFromOptions() opens its block with PetscObjectOptionsBegin(), which takes the
     manual section from the object; a PetscRandom carries the Sys section */
  PetscCall(PetscRandomCreate(PETSC_COMM_WORLD, &rnd));
  PetscCall(PetscRandomSetFromOptions(rnd));
  PetscCall(PetscRandomDestroy(&rnd));

  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

   # "-help mansec" restricts the help output to the options blocks in the listed manual section(s);
   # filter to the options blocks to avoid the version banner
   testset:
      filter: grep -E -e "^Scaled units options|\(ManPage[12]\)"
      test:
         suffix: help_sec1
         args: -r1 2 -help Sec1
      test:
         suffix: help_sec1_sec2
         args: -r1 2 -help Sec1,Sec2
      test:
         suffix: help
         args: -help
      # manual sections are matched without regard to case, as PETSc option names are
      test:
         suffix: help_sec1_case
         args: -r1 2 -help sEc1
         output_file: output/ex21_help_sec1.out
      # a man page name is not a manual section, so it selects nothing
      test:
         suffix: help_manpage
         args: -help ManPage1
         output_file: output/empty.out
      # a logical true value is not a manual section; it is equivalent to a bare -help
      test:
         suffix: help_true
         args: -help 1
         output_file: output/ex21_help.out
      # a logical false value turns the help output off; the program runs as if -help had not been given
      test:
         suffix: help_false
         args: -help false
         output_file: output/empty.out

   # blocks opened with PetscObjectOptionsBegin() take their manual section from the object
   test:
      suffix: help_object
      args: -help Sys
      filter: grep -E -e "^Random number generator|\(PetscRandomSetSeed\)"

   # the viewer options of an object are documented in the Viewer section, not in the section of the
   # object that creates them, so they are not selected along with it
   test:
      suffix: help_sys_no_viewer
      args: -help Sys
      filter: grep -E -e "^Viewer \("
      output_file: output/empty.out
   test:
      suffix: help_viewer
      args: -help Viewer
      filter: grep -E -e "^Viewer \(-random_view\)"

TEST*/
