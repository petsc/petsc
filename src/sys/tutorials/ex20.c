
static char help[] = "Demonstrates PetscOptionsPush()/PetscOptionsPop().\n\n";

#include <petscsys.h>
#include <petscoptions.h>
int main(int argc,char **argv)
{
  PetscOptions   opt1,opt2;
  PetscInt       int1,int2;
  PetscBool      flg1,flg2,flga,match;
  char           str[16];

  PetscCall(PetscInitialize(&argc,&argv,(char*)0,help));

  PetscCall(PetscOptionsCreate(&opt1));
  PetscCall(PetscOptionsInsertString(opt1,"-testa a"));
  PetscCall(PetscOptionsPush(opt1));
  PetscCall(PetscOptionsSetValue(NULL,"-test1","1"));
  PetscCall(PetscOptionsGetInt(NULL,NULL,"-test1",&int1,&flg1));
  PetscCheck(flg1 && int1 == 1,PETSC_COMM_WORLD,PETSC_ERR_PLIB,"Unable to locate option test1 or it has the wrong value");
  PetscCall(PetscOptionsGetString(NULL,NULL,"-testa",str,sizeof(str),&flga));
  PetscCall(PetscStrcmp(str,"a",&match));
  PetscCheck(flga && match,PETSC_COMM_WORLD,PETSC_ERR_PLIB,"Unable to locate option testa or it has the wrong value");
  PetscCall(PetscOptionsCreate(&opt2));
  PetscCall(PetscOptionsPush(opt2));
  PetscCall(PetscOptionsSetValue(NULL,"-test2","2"));
  PetscCall(PetscOptionsGetInt(NULL,NULL,"-test2",&int2,&flg2));
  PetscCheck(flg2 && int2 == 2,PETSC_COMM_WORLD,PETSC_ERR_PLIB,"Unable to locate option test2 or it has the wrong value");
  PetscCall(PetscOptionsGetInt(NULL,NULL,"-test1",&int1,&flg1));
  PetscCheck(!flg1,PETSC_COMM_WORLD,PETSC_ERR_PLIB,"Able to access test1 from a different options database");

  PetscCall(PetscOptionsPop());
  PetscCall(PetscOptionsPop());
  PetscCall(PetscOptionsDestroy(&opt2));
  PetscCall(PetscOptionsDestroy(&opt1));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

   test:

TEST*/
