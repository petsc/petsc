#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: ex10.c,v 1.3 1997/08/14 16:42:17 bsmith Exp bsmith $";
#endif

/* 
   Tests PetscMemmove()
*/

#include "petsc.h"

#undef __FUNC__
#define __FUNC__ "main"
int main(int argc,char **argv)
{
  int i, *a,*b;
  PetscInitialize(&argc,&argv,(char *)0,0);

  a = (int *) PetscMalloc( 10*sizeof(int) ); CHKPTRA(a);
  b = (int *) PetscMalloc( 20*sizeof(int) ); CHKPTRA(a);

  /*
      Nonoverlapping regions
  */
  for (i=0; i<20; i++) b[i] = i;
  PetscMemmove(a,b,10*sizeof(int));
  PetscIntView(10,a,0);

  PetscFree(a);

  /*
     |        |                |       |
     b        a               b+15    b+20
                              a+10    a+15
  */
  a = b + 5;
  PetscMemmove(a,b,15*sizeof(int));
  PetscIntView(15,a,0);
  PetscFree(b);

  /*
     |       |                    |       |
     a       b                   a+20   a+25
                                        b+20
  */
  a = (int*) PetscMalloc( 25*sizeof(int) ); CHKPTRA(a);
  b = a + 5;
  for (i=0; i<20; i++) b[i] = i;
  PetscMemmove(a,b,20*sizeof(int));
  PetscIntView(20,a,0);
  PetscFree(a);

  PetscFinalize();
  return 0;
}
 
