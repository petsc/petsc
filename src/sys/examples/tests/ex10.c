#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: ex10.c,v 1.4 1999/03/19 21:17:16 bsmith Exp balay $";
#endif

/* 
   Tests PetscMemmove()
*/

#include "petsc.h"

#undef __FUNC__
#define __FUNC__ "main"
int main(int argc,char **argv)
{
  int i, *a,*b,ierr;
  PetscInitialize(&argc,&argv,(char *)0,0);

  a = (int *) PetscMalloc( 10*sizeof(int) );CHKPTRA(a);
  b = (int *) PetscMalloc( 20*sizeof(int) );CHKPTRA(a);

  /*
      Nonoverlapping regions
  */
  for (i=0; i<20; i++) b[i] = i;
  ierr = PetscMemmove(a,b,10*sizeof(int));CHKERRA(ierr);
  PetscIntView(10,a,0);

  PetscFree(a);

  /*
     |        |                |       |
     b        a               b+15    b+20
                              a+10    a+15
  */
  a = b + 5;
  ierr = PetscMemmove(a,b,15*sizeof(int));CHKERRA(ierr);
  PetscIntView(15,a,0);
  PetscFree(b);

  /*
     |       |                    |       |
     a       b                   a+20   a+25
                                        b+20
  */
  a = (int*) PetscMalloc( 25*sizeof(int) );CHKPTRA(a);
  b = a + 5;
  for (i=0; i<20; i++) b[i] = i;
  ierr = PetscMemmove(a,b,20*sizeof(int));CHKERRA(ierr);
  PetscIntView(20,a,0);
  PetscFree(a);

  PetscFinalize();
  return 0;
}
 
