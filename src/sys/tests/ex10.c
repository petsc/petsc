
static char help[] = "Tests PetscArraymove()/PetscMemmove()\n";

#include <petscsys.h>

int main(int argc,char **argv)
{
  PetscInt       i,*a,*b;

  PetscCall(PetscInitialize(&argc,&argv,(char*)0,help));

  PetscCall(PetscMalloc1(10,&a));
  PetscCall(PetscMalloc1(20,&b));

  /*
      Nonoverlapping regions
  */
  for (i=0; i<20; i++) b[i] = i;
  PetscCall(PetscArraymove(a,b,10));
  PetscCall(PetscIntView(10,a,NULL));

  PetscCall(PetscFree(a));

  /*
     |        |                |       |
     b        a               b+15    b+20
                              a+10    a+15
  */
  a    = b + 5;
  PetscCall(PetscArraymove(a,b,15));
  PetscCall(PetscIntView(15,a,NULL));
  PetscCall(PetscFree(b));

  /*
     |       |                    |       |
     a       b                   a+20   a+25
                                        b+20
  */
  PetscCall(PetscMalloc1(25,&a));
  b    = a + 5;
  for (i=0; i<20; i++) b[i] = i;
  PetscCall(PetscArraymove(a,b,20));
  PetscCall(PetscIntView(20,a,NULL));
  PetscCall(PetscFree(a));

  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

   test:

TEST*/
