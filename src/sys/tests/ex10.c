
static char help[] = "Tests PetscArraymove()/PetscMemmove()\n";

#include <petscsys.h>

int main(int argc,char **argv)
{
  PetscInt       i,*a,*b;
  PetscErrorCode ierr;

  ierr = PetscInitialize(&argc,&argv,(char*)0,help);if (ierr) return ierr;

  CHKERRQ(PetscMalloc1(10,&a));
  CHKERRQ(PetscMalloc1(20,&b));

  /*
      Nonoverlapping regions
  */
  for (i=0; i<20; i++) b[i] = i;
  CHKERRQ(PetscArraymove(a,b,10));
  CHKERRQ(PetscIntView(10,a,NULL));

  CHKERRQ(PetscFree(a));

  /*
     |        |                |       |
     b        a               b+15    b+20
                              a+10    a+15
  */
  a    = b + 5;
  CHKERRQ(PetscArraymove(a,b,15));
  CHKERRQ(PetscIntView(15,a,NULL));
  CHKERRQ(PetscFree(b));

  /*
     |       |                    |       |
     a       b                   a+20   a+25
                                        b+20
  */
  CHKERRQ(PetscMalloc1(25,&a));
  b    = a + 5;
  for (i=0; i<20; i++) b[i] = i;
  CHKERRQ(PetscArraymove(a,b,20));
  CHKERRQ(PetscIntView(20,a,NULL));
  CHKERRQ(PetscFree(a));

  ierr = PetscFinalize();
  return ierr;
}

/*TEST

   test:

TEST*/
