
static char help[] = "Tests PetscArraymove()/PetscMemmove()\n";

#include <petscsys.h>

int main(int argc,char **argv)
{
  PetscInt       i,*a,*b;
  PetscErrorCode ierr;

  ierr = PetscInitialize(&argc,&argv,(char*)0,help);if (ierr) return ierr;

  ierr = PetscMalloc1(10,&a);CHKERRQ(ierr);
  ierr = PetscMalloc1(20,&b);CHKERRQ(ierr);

  /*
      Nonoverlapping regions
  */
  for (i=0; i<20; i++) b[i] = i;
  ierr = PetscArraymove(a,b,10);CHKERRQ(ierr);
  ierr = PetscIntView(10,a,NULL);CHKERRQ(ierr);

  ierr = PetscFree(a);CHKERRQ(ierr);

  /*
     |        |                |       |
     b        a               b+15    b+20
                              a+10    a+15
  */
  a    = b + 5;
  ierr = PetscArraymove(a,b,15);CHKERRQ(ierr);
  ierr = PetscIntView(15,a,NULL);CHKERRQ(ierr);
  ierr = PetscFree(b);CHKERRQ(ierr);

  /*
     |       |                    |       |
     a       b                   a+20   a+25
                                        b+20
  */
  ierr = PetscMalloc1(25,&a);CHKERRQ(ierr);
  b    = a + 5;
  for (i=0; i<20; i++) b[i] = i;
  ierr = PetscArraymove(a,b,20);CHKERRQ(ierr);
  ierr = PetscIntView(20,a,NULL);CHKERRQ(ierr);
  ierr = PetscFree(a);CHKERRQ(ierr);

  ierr = PetscFinalize();
  return ierr;
}

/*TEST

   test:

TEST*/
