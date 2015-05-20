

#include <petscsys.h>

int main(int argc,char **argv)
{
  PetscErrorCode ierr;
  char           shorturl[64];

  PetscInitialize(&argc,&argv,NULL,NULL);
  ierr = PetscURLShorten("http://www.google.com",shorturl,64);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_SELF,"Long url %s short url %s\n","http://www.google.com",shorturl);CHKERRQ(ierr);
  PetscFinalize();
  return 0;
}


