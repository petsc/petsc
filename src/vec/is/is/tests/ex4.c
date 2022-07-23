
static char help[] = "Tests ISToGeneral().\n\n";

#include <petscis.h>

int main(int argc,char **argv)
{
  PetscInt       step = 2;
  IS             is;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc,&argv,(char*)0,help));

  PetscCall(PetscOptionsGetInt(NULL,NULL,"-step",&step,NULL));
  PetscCall(ISCreateStride(PETSC_COMM_SELF,10,0,step,&is));

  PetscCall(ISToGeneral(is));

  PetscCall(ISDestroy(&is));

  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

   test:
     output_file: output/ex1_1.out

TEST*/
