
#include <petscsys.h>

int main(int argc,char **argv)
{
  char           shorturl[64];

  CHKERRQ(PetscInitialize(&argc,&argv,NULL,NULL));
  CHKERRQ(PetscURLShorten("http://www.google.com",shorturl,64));
  CHKERRQ(PetscPrintf(PETSC_COMM_SELF,"Long url %s short url %s\n","http://www.google.com",shorturl));
  CHKERRQ(PetscFinalize());
  return 0;
}

/*TEST

   build:
     requires: ssl

   test:

TEST*/
