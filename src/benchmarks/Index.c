#ifndef lint
static char vcid[] = "$Id: PetscTime.c,v 1.6 1996/03/19 21:30:13 bsmith Exp $";
#endif

#include "stdio.h"
#include "petsc.h"
#include "sys.h"

int BlastCache();

int main( int argc, char **argv)
{
  PetscInitialize(&argc, &argv,0,0);
 
  test1();
  test2();

  PetscFinalize();
  return 0;
}

int test1()
{
  double    t1, t2, value;
  int       i, ierr,*z,*zi, intval, tmp;
  Scalar    *x, *y;
  PetscRandom  r;

  ierr = PetscRandomCreate(MPI_COMM_SELF,RANDOM_DEFAULT,&r); CHKERRQ(ierr);
  x    = (Scalar *)PetscMalloc(20000*sizeof(Scalar)); CHKPTRA(x);
  y    = (Scalar *)PetscMalloc(20000*sizeof(Scalar)); CHKPTRA(y);

  z     = (int *)PetscMalloc(2000*sizeof(int)); CHKPTRA(z); 
  zi    = (int *)PetscMalloc(2000*sizeof(int)); CHKPTRA(zi); 



  /* Take care of paging effects */
  t1 = PetscGetTime(); 
  
   /* Form the random set of integers */
  for (i=0; i<2000; i++) {
    ierr   = PetscRandomGetValue(r, &value); CHKERRQ(ierr);
    intval = (int)(value*20000.0);
    z[i]   = intval;
  }

  for (i=0; i<2000; i++) {
    ierr   = PetscRandomGetValue(r, &value); CHKERRQ(ierr);
    intval = (int)(value*20000.0);
    z[i]   = intval;
  }
  fprintf(stderr,"Done setup\n");

  BlastCache(); 

  t1 = PetscGetTime(); 
  for (i=0; i<2000; i++) {  x[i] = y[i]; }
  t2 = PetscGetTime(); 
  fprintf(stderr,"%-19s : %e sec\n","x[i] = y[i]",(t2-t1)/2000.0);

  BlastCache(); 

  t1 = PetscGetTime(); 
  for (i=0; i<2000; i++) {  x[i] = y[z[i]]; }
  t2 = PetscGetTime(); 
  fprintf(stderr,"%-19s : %e sec\n","x[i] = y[idx[i]]",(t2-t1)/2000.0);

  BlastCache();

  t1 = PetscGetTime(); 
  for (i=0; i<2000; i++) {  x[z[i]] = y[i]; }
  t2 = PetscGetTime(); 
  fprintf(stderr,"%-19s : %e sec\n","x[z[i]] = y[i]",(t2-t1)/2000.0);

  BlastCache();

  t1 = PetscGetTime(); 
  for (i=0; i<2000; i++) {  x[z[i]] = y[zi[i]]; }
  t2 = PetscGetTime(); 
  fprintf(stderr,"%-19s : %e sec\n","x[z[i]] = y[zi[i]]",(t2-t1)/2000.0);

  PetscFree(z);
  PetscFree(zi);
  PetscFree(x);
  PetscFree(y);
  PetscRandomDestroy(r);
  return 0;
}

int test2()
{
  double    t1, t2, value;
  int       i, ierr,z[2000],zi[20000], intval, tmp;
  Scalar    x[20000], y[20000];
  PetscRandom  r;

  ierr = PetscRandomCreate(MPI_COMM_SELF,RANDOM_DEFAULT,&r); CHKERRQ(ierr);

  /* Take care of paging effects */
  t1 = PetscGetTime(); 
  
  for (i=0; i<2000; i++) {
    zi[i] = i;
    z[i]  = i;
  }
 
  for (i=0; i<20000; i++) {
    x[i] = i;
    y[i]  = i;
  }

   /* Form the random set of integers */
  for (i=0; i<2000; i++) {
    ierr   = PetscRandomGetValue(r, &value); CHKERRQ(ierr);
    intval = (int)(value*20000.0);
    tmp    = z[i];
    z[i]   = z[intval];
    z[intval] = tmp;
  }

  for (i=0; i<2000; i++) {
    ierr   = PetscRandomGetValue(r, &value); CHKERRQ(ierr);
    intval = (int)(value*20000.0);
    tmp    = zi[i];
    zi[i]  = zi[intval];
    zi[intval] = tmp;
  }
  fprintf(stderr,"Done setup\n");

  /* BlastCache();  */

  t1 = PetscGetTime(); 
  for (i=0; i<2000; i++) {  x[i] = y[i]; }
  t2 = PetscGetTime(); 
  fprintf(stderr,"%-19s : %e sec\n","x[i] = y[i]",(t2-t1)/2000.0);

  /* BlastCache();  */

  t1 = PetscGetTime(); 
  for (i=0; i<2000; i++) {  y[i] = x[z[i]]; }
  t2 = PetscGetTime(); 
  fprintf(stderr,"%-19s : %e sec\n","x[i] = y[idx[i]]",(t2-t1)/2000.0);

  /* BlastCache(); */

  t1 = PetscGetTime(); 
  for (i=0; i<2000; i++) {  x[z[i]] = y[i]; }
  t2 = PetscGetTime(); 
  fprintf(stderr,"%-19s : %e sec\n","x[z[i]] = y[i]",(t2-t1)/2000.0);

  /* BlastCache(); */

  t1 = PetscGetTime(); 
  for (i=0; i<2000; i++) {  y[z[i]] = x[zi[i]]; }
  t2 = PetscGetTime(); 
  fprintf(stderr,"%-19s : %e sec\n","x[z[i]] = y[zi[i]]",(t2-t1)/2000.0);


  PetscRandomDestroy(r);
  return 0;
}

int BlastCache()
{
  int    i,n = 1000000;
  Scalar *x,*y,*z,*a, *b;

  x = (Scalar *) PetscMalloc(5*n); CHKPTRA(x);
  y = x + n;
  z = y + n;
  a = z + n;
  b = a + n;

  for ( i=0; i<n; i++ ) {
    a[i] = 3.0*x[i] + 2.0*y[i] + 3.3*z[i] - 25.*b[i];
  }
  for ( i=0; i<n; i++ ) {
    b[i] = 3.0*x[i] + 2.0*y[i] + 3.3*a[i] - 25.*b[i];
  }
  for ( i=0; i<n; i++ ) {
    z[i] = 3.0*x[i] + 2.0*y[i] + 3.3*a[i] - 25.*b[i];
  }
  PetscFree(x);
  return 0;
}
