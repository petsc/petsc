
static char help[] = "Tests ISToGeneral().\n\n";

#include <petscis.h>

int main(int argc,char **argv)
{
  PetscInt       step = 2;
  IS             is;

  CHKERRQ(PetscInitialize(&argc,&argv,(char*)0,help));

  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-step",&step,NULL));
  CHKERRQ(ISCreateStride(PETSC_COMM_SELF,10,0,step,&is));

  CHKERRQ(ISToGeneral(is));

  CHKERRQ(ISDestroy(&is));

  CHKERRQ(PetscFinalize());
  return 0;
}

/*TEST

   test:
     output_file: output/ex1_1.out

TEST*/
