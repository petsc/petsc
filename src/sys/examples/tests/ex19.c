
static char help[] = "Tests string options with spaces";

#include <petscsys.h>

int main(int argc,char **argv)
{
  PetscErrorCode ierr;
  char           option2[20],option3[30];
  PetscBool      flg;
  PetscInt       option1;

  ierr = PetscInitialize(&argc,&argv,"ex19options",help);if (ierr) return ierr;
  ierr = PetscOptionsGetInt(NULL,0,"-option1",&option1,&flg);CHKERRQ(ierr);
  ierr = PetscOptionsGetString(NULL,0,"-option2",option2,20,&flg);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"%s\n",option2);CHKERRQ(ierr);
  ierr = PetscOptionsGetString(NULL,0,"-option3",option3,30,&flg);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"%s\n",option3);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
}


/*TEST

   test:
     localrunfiles: ex19options

TEST*/
