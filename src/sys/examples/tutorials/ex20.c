
static char help[] = "Demonstrates PetscOptionsPush()/PetscOptionsPop().\n\n";

#include <petscsys.h>
#include <petscoptions.h>
int main(int argc,char **argv)
{
  PetscErrorCode ierr;
  PetscOptions   opt1,opt2;
  PetscInt       int1,int2;
  PetscBool      flg1,flg2,flga,match;
  char           str[16];

  ierr = PetscInitialize(&argc,&argv,(char*)0,help);if (ierr) return ierr;

  ierr = PetscOptionsCreate(&opt1);CHKERRQ(ierr);
  ierr = PetscOptionsInsertString(opt1,"-testa a");CHKERRQ(ierr);
  ierr = PetscOptionsPush(opt1);CHKERRQ(ierr);
  ierr = PetscOptionsSetValue(NULL,"-test1","1");CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(NULL,NULL,"-test1",&int1,&flg1);CHKERRQ(ierr);
  if (!flg1 || int1 != 1) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_PLIB,"Unable to locate option test1 or it has the wrong value");
  ierr = PetscOptionsGetString(NULL,NULL,"-testa",str,sizeof(str),&flga);CHKERRQ(ierr);
  ierr = PetscStrcmp(str,"a",&match);CHKERRQ(ierr);
  if (!flga|| !match)  SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_PLIB,"Unable to locate option testa or it has the wrong value");
  ierr = PetscOptionsCreate(&opt2);CHKERRQ(ierr);
  ierr = PetscOptionsPush(opt2);CHKERRQ(ierr);
  ierr = PetscOptionsSetValue(NULL,"-test2","2");CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(NULL,NULL,"-test2",&int2,&flg2);CHKERRQ(ierr);
  if (!flg2 || int2 != 2) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_PLIB,"Unable to locate option test2 or it has the wrong value");
  ierr = PetscOptionsGetInt(NULL,NULL,"-test1",&int1,&flg1);CHKERRQ(ierr);
  if (flg1) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_PLIB,"Able to access test1 from a different options database");

  ierr = PetscOptionsPop();CHKERRQ(ierr);
  ierr = PetscOptionsPop();CHKERRQ(ierr);
  ierr = PetscOptionsDestroy(&opt2);CHKERRQ(ierr);
  ierr = PetscOptionsDestroy(&opt1);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
}


/*TEST

   test:

TEST*/
