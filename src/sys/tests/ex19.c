
static char help[] = "Tests string options with spaces";

#include <petscsys.h>

int main(int argc,char **argv)
{
  char           option2[20],option3[30];
  PetscBool      flg;
  PetscInt       option1;

  CHKERRQ(PetscInitialize(&argc,&argv,"ex19options",help));
  CHKERRQ(PetscOptionsGetInt(NULL,0,"-option1",&option1,&flg));
  CHKERRQ(PetscOptionsGetString(NULL,0,"-option2",option2,sizeof(option2),&flg));
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"%s\n",option2));
  CHKERRQ(PetscOptionsGetString(NULL,0,"-option3",option3,sizeof(option3),&flg));
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"%s\n",option3));
  CHKERRQ(PetscFinalize());
  return 0;
}

/*TEST

   test:
     localrunfiles: ex19options

TEST*/
