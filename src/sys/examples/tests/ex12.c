/*$Id: ex12.c,v 1.10 2000/05/05 22:14:28 balay Exp balay $*/

static char help[] = "Tests timing PetscSortInt().\n\n";

#include "petsc.h"
#include "petscsys.h"

#undef __FUNC__
#define __FUNC__ "main"
int main(int argc,char **argv)
{
  int         ierr,i,n = 1000,*values,event;
  PetscRandom rand;
  Scalar      value;

  PetscInitialize(&argc,&argv,(char *)0,help);
  ierr = OptionsGetInt(PETSC_NULL,"-n",&n,PETSC_NULL);CHKERRA(ierr);
  
  ierr = PetscRandomCreate(PETSC_COMM_SELF,RANDOM_DEFAULT,&rand);CHKERRA(ierr);

  values = (int*)PetscMalloc(n*sizeof(int));CHKPTRA(values);
  for (i=0; i<n; i++) {
    ierr = PetscRandomGetValue(rand,&value);CHKERRA(ierr);
    values[i] = (int)(n*PetscRealPart(value) + 2.0);
  }
  ierr = PetscSortInt(n,values);CHKERRA(ierr);

  ierr = PLogEventRegister(&event,"Sort",PETSC_NULL);CHKERRA(ierr);
  ierr = PLogEventBegin(event,0,0,0,0);CHKERRA(ierr);
  values = (int*)PetscMalloc(n*sizeof(int));CHKPTRA(values);
  for (i=0; i<n; i++) {
    ierr = PetscRandomGetValue(rand,&value);CHKERRA(ierr);
    values[i] = (int)(n*PetscRealPart(value) + 2.0);
  }
  ierr = PetscSortInt(n,values);CHKERRA(ierr);
  ierr = PLogEventEnd(event,0,0,0,0);CHKERRA(ierr);

  for (i=1; i<n; i++) {
    if (values[i] < values[i-1]) SETERRA(1,1,"Values not sorted");
  }
  ierr = PetscFree(values);CHKERRA(ierr);
  ierr = PetscRandomDestroy(rand);CHKERRA(ierr);

  PetscFinalize();
  return 0;
}
 
