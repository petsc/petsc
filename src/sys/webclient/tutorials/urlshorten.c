
#include <petscsys.h>

int main(int argc,char **argv)
{
  char           shorturl[64];

  PetscCall(PetscInitialize(&argc,&argv,NULL,NULL));
  PetscCall(PetscURLShorten("http://www.google.com",shorturl,64));
  PetscCall(PetscPrintf(PETSC_COMM_SELF,"Long url %s short url %s\n","http://www.google.com",shorturl));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

   build:
     requires: ssl

   test:

TEST*/
