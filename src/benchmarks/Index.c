
#include "stdio.h"
#include "petsc.h"
#include "sys.h"

int main( int argc, char **argv)
{
  double    t1, t2, value;
  int       i, ierr,z[10000], intval, tmp;
  int       x[10000],y[10000];
  SYRandom  r;

  PetscInitialize(&argc, &argv,0,0,0);
  ierr = SYRandomCreate( MPI_COMM_SELF,RANDOM_DEFAULT,&r); CHKERRQ(ierr);

  /* Take care of paging effects */
  t1 = PetscGetTime(); 
  
  for (i=0; i<10000; i++) {
    x[i] = i;
    y[i] = i;
    z[i] = i;
  }
 
  /* Form the random set of integers */
  for (i=0; i<10000; i++) {
    ierr   = SYRandomGetValue(r, &value); CHKERRQ(ierr);
    intval = (int)(value*10000.0);
    tmp    = z[i];
    z[i]   = z[intval];
    z[intval] = tmp;
  }


  t1 = PetscGetTime(); 
  for (i=0; i<1000; i++) {  x[i] = y[z[i]]; }
  t2 = PetscGetTime(); 
  fprintf(stderr,"%-19s : %e sec\n","x[i] = y[idx[i]]",(t2-t1)/sizeof(int)*1000.0);


  t1 = PetscGetTime(); 
  for (i=0; i<1000; i++) {  x[z[i]] = y[i]; }
  t2 = PetscGetTime(); 
  fprintf(stderr,"%-19s : %e sec\n","x[z[i]] = y[i]",(t2-t1)/(sizeof(int)*1000.0));

  t1 = PetscGetTime(); 
  for (i=0; i<1000; i++) {  x[z[i]] = y[z[i]]; }
  t2 = PetscGetTime(); 
  fprintf(stderr,"%-19s : %e sec\n","x[z[i]] = y[z[i]]",(t2-t1)/(sizeof(int)*1000.0));

  PetscFinalize();
  return 0;
}
