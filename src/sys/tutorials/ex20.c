
static char help[] = "Demonstrates PetscOptionsPush()/PetscOptionsPop().\n\n";

#include <petscsys.h>
#include <petscoptions.h>
int main(int argc,char **argv)
{
  PetscOptions   opt1,opt2;
  PetscInt       int1,int2;
  PetscBool      flg1,flg2,flga,match;
  char           str[16];

  CHKERRQ(PetscInitialize(&argc,&argv,(char*)0,help));

  CHKERRQ(PetscOptionsCreate(&opt1));
  CHKERRQ(PetscOptionsInsertString(opt1,"-testa a"));
  CHKERRQ(PetscOptionsPush(opt1));
  CHKERRQ(PetscOptionsSetValue(NULL,"-test1","1"));
  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-test1",&int1,&flg1));
  PetscCheckFalse(!flg1 || int1 != 1,PETSC_COMM_WORLD,PETSC_ERR_PLIB,"Unable to locate option test1 or it has the wrong value");
  CHKERRQ(PetscOptionsGetString(NULL,NULL,"-testa",str,sizeof(str),&flga));
  CHKERRQ(PetscStrcmp(str,"a",&match));
  PetscCheckFalse(!flga|| !match,PETSC_COMM_WORLD,PETSC_ERR_PLIB,"Unable to locate option testa or it has the wrong value");
  CHKERRQ(PetscOptionsCreate(&opt2));
  CHKERRQ(PetscOptionsPush(opt2));
  CHKERRQ(PetscOptionsSetValue(NULL,"-test2","2"));
  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-test2",&int2,&flg2));
  PetscCheckFalse(!flg2 || int2 != 2,PETSC_COMM_WORLD,PETSC_ERR_PLIB,"Unable to locate option test2 or it has the wrong value");
  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-test1",&int1,&flg1));
  PetscCheck(!flg1,PETSC_COMM_WORLD,PETSC_ERR_PLIB,"Able to access test1 from a different options database");

  CHKERRQ(PetscOptionsPop());
  CHKERRQ(PetscOptionsPop());
  CHKERRQ(PetscOptionsDestroy(&opt2));
  CHKERRQ(PetscOptionsDestroy(&opt1));
  CHKERRQ(PetscFinalize());
  return 0;
}

/*TEST

   test:

TEST*/
