

#include <petscsys.h>

int main(int argc,char **argv)
{
  PetscErrorCode ierr;
  char           shorturl[64];

  ierr = PetscInitialize(&argc,&argv,NULL,NULL);if (ierr) return ierr;
  ierr = PetscURLShorten("http://www.google.com",shorturl,64);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_SELF,"Long url %s short url %s\n","http://www.google.com",shorturl);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
}

/*TEST

   build:
     requires: ssl

   test:

TEST*/

