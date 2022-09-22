
static char help[] = "Tests string options with spaces";

#include <petscsys.h>

int main(int argc, char **argv)
{
  char      option2[20], option3[30];
  PetscBool flg;
  PetscInt  option1;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, "ex19options", help));
  PetscCall(PetscOptionsGetInt(NULL, 0, "-option1", &option1, &flg));
  PetscCall(PetscOptionsGetString(NULL, 0, "-option2", option2, sizeof(option2), &flg));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "%s\n", option2));
  PetscCall(PetscOptionsGetString(NULL, 0, "-option3", option3, sizeof(option3), &flg));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "%s\n", option3));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

   test:
     localrunfiles: ex19options

TEST*/
