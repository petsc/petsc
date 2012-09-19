
#include <petscsys.h>

extern int BlastCache(void);
extern int test1(void);
extern int test2(void);

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc,char **argv)
{
  PetscErrorCode ierr;

  PetscInitialize(&argc,&argv,0,0);

  ierr = test1();CHKERRQ(ierr);
  ierr = test2();CHKERRQ(ierr);

  ierr = PetscFinalize();
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "test1"
int test1(void)
{
  PetscLogDouble  t1,t2;
  double      value;
  int         i,ierr,*z,*zi,intval;
  PetscScalar *x,*y;
  PetscRandom r;

  ierr = PetscRandomCreate(PETSC_COMM_SELF,&r);CHKERRQ(ierr);
  ierr = PetscRandomSetFromOptions(r);CHKERRQ(ierr);
  ierr = PetscMalloc(20000*sizeof(PetscScalar),&x);CHKERRQ(ierr);
  ierr = PetscMalloc(20000*sizeof(PetscScalar),&y);CHKERRQ(ierr);

  ierr = PetscMalloc(2000*sizeof(int),&z);CHKERRQ(ierr);
  ierr = PetscMalloc(2000*sizeof(int),&zi);CHKERRQ(ierr);



  /* Take care of paging effects */
  ierr = PetscGetTime(&t1);CHKERRQ(ierr);

   /* Form the random set of integers */
  for (i=0; i<2000; i++) {
    ierr   = PetscRandomGetValue(r,&value);CHKERRQ(ierr);
    intval = (int)(value*20000.0);
    z[i]   = intval;
  }

  for (i=0; i<2000; i++) {
    ierr    = PetscRandomGetValue(r,&value);CHKERRQ(ierr);
    intval  = (int)(value*20000.0);
    zi[i]   = intval;
  }
  /* fprintf(stdout,"Done setup\n"); */

  ierr = BlastCache();CHKERRQ(ierr);

  ierr = PetscGetTime(&t1);CHKERRQ(ierr);
  for (i=0; i<2000; i++) {  x[i] = y[i]; }
  ierr = PetscGetTime(&t2);CHKERRQ(ierr);
  fprintf(stdout,"%-27s : %e sec\n","x[i] = y[i]",(t2-t1)/2000.0);

  ierr = BlastCache();CHKERRQ(ierr);

  ierr = PetscGetTime(&t1);CHKERRQ(ierr);
  for (i=0; i<500; i+=4) {
    x[i]   = y[z[i]];
    x[1+i] = y[z[1+i]];
    x[2+i] = y[z[2+i]];
    x[3+i] = y[z[3+i]];
  }
  ierr = PetscGetTime(&t2);CHKERRQ(ierr);
  fprintf(stdout,"%-27s : %e sec\n","x[i] = y[idx[i]] - unroll 4",(t2-t1)/2000.0);

  ierr = BlastCache();CHKERRQ(ierr);

  ierr = PetscGetTime(&t1);CHKERRQ(ierr)
  for (i=0; i<2000; i++) {  x[i] = y[z[i]]; }
  ierr = PetscGetTime(&t2);CHKERRQ(ierr);
  fprintf(stdout,"%-27s : %e sec\n","x[i] = y[idx[i]]",(t2-t1)/2000.0);

  ierr = BlastCache();CHKERRQ(ierr);

  ierr = PetscGetTime(&t1);CHKERRQ(ierr);
  for (i=0; i<1000; i+=2) {  x[i] = y[z[i]];  x[1+i] = y[z[1+i]]; }
  ierr = PetscGetTime(&t2);CHKERRQ(ierr);
  fprintf(stdout,"%-27s : %e sec\n","x[i] = y[idx[i]] - unroll 2",(t2-t1)/2000.0);

  ierr = BlastCache();CHKERRQ(ierr);

  ierr = PetscGetTime(&t1);CHKERRQ(ierr);
  for (i=0; i<2000; i++) {  x[z[i]] = y[i]; }
  ierr = PetscGetTime(&t2);CHKERRQ(ierr);
  fprintf(stdout,"%-27s : %e sec\n","x[z[i]] = y[i]",(t2-t1)/2000.0);

  ierr = BlastCache();CHKERRQ(ierr);

  ierr = PetscGetTime(&t1);CHKERRQ(ierr);
  for (i=0; i<2000; i++) {  x[z[i]] = y[zi[i]]; }
  ierr = PetscGetTime(&t2);CHKERRQ(ierr);
  fprintf(stdout,"%-27s : %e sec\n","x[z[i]] = y[zi[i]]",(t2-t1)/2000.0);

  ierr = PetscMemcpy(x,y,10);CHKERRQ(ierr);
  ierr = PetscMemcpy(z,zi,10);CHKERRQ(ierr);
  ierr = PetscFree(z);CHKERRQ(ierr);
  ierr = PetscFree(zi);CHKERRQ(ierr);
  ierr = PetscFree(x);CHKERRQ(ierr);
  ierr = PetscFree(y);CHKERRQ(ierr);
  ierr = PetscRandomDestroy(&r);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "test2"
int test2(void)
{
  PetscLogDouble   t1,t2;
  double       value;
  int          i,ierr,z[20000],zi[20000],intval,tmp;
  PetscScalar  x[20000],y[20000];
  PetscRandom  r;

  ierr = PetscRandomCreate(PETSC_COMM_SELF,&r);CHKERRQ(ierr);
  ierr = PetscRandomSetFromOptions(r);CHKERRQ(ierr);

  /* Take care of paging effects */
  ierr = PetscGetTime(&t1);CHKERRQ(ierr);

  for (i=0; i<20000; i++) {
    x[i]  = i;
    y[i]  = i;
    z[i]  = i;
    zi[i] = i;
  }

   /* Form the random set of integers */
  for (i=0; i<20000; i++) {
    ierr   = PetscRandomGetValue(r,&value);CHKERRQ(ierr);
    intval = (int)(value*20000.0);
    tmp    = z[i];
    z[i]   = z[intval];
    z[intval] = tmp;
  }

  for (i=0; i<20000; i++) {
    ierr   = PetscRandomGetValue(r,&value);CHKERRQ(ierr);
    intval = (int)(value*20000.0);
    tmp    = zi[i];
    zi[i]  = zi[intval];
    zi[intval] = tmp;
  }
  /* fprintf(stdout,"Done setup\n"); */

  /* ierr = BlastCache();CHKERRQ(ierr); */

  ierr = PetscGetTime(&t1);CHKERRQ(ierr);
  for (i=0; i<2000; i++) {  x[i] = y[i]; }
  ierr = PetscGetTime(&t2);CHKERRQ(ierr);
  fprintf(stdout,"%-27s : %e sec\n","x[i] = y[i]",(t2-t1)/2000.0);

  /* ierr = BlastCache();CHKERRQ(ierr); */

  ierr = PetscGetTime(&t1);CHKERRQ(ierr);
  for (i=0; i<2000; i++) {  y[i] = x[z[i]]; }
  ierr = PetscGetTime(&t2);CHKERRQ(ierr);
  fprintf(stdout,"%-27s : %e sec\n","x[i] = y[idx[i]]",(t2-t1)/2000.0);

  /* ierr = BlastCache();CHKERRQ(ierr); */

  ierr = PetscGetTime(&t1);CHKERRQ(ierr);
  for (i=0; i<2000; i++) {  x[z[i]] = y[i]; }
  ierr = PetscGetTime(&t2);CHKERRQ(ierr);
  fprintf(stdout,"%-27s : %e sec\n","x[z[i]] = y[i]",(t2-t1)/2000.0);

  /* ierr = BlastCache();CHKERRQ(ierr); */

  ierr = PetscGetTime(&t1);CHKERRQ(ierr);
  for (i=0; i<2000; i++) {  y[z[i]] = x[zi[i]]; }
  ierr = PetscGetTime(&t2);CHKERRQ(ierr);
  fprintf(stdout,"%-27s : %e sec\n","x[z[i]] = y[zi[i]]",(t2-t1)/2000.0);


  ierr = PetscRandomDestroy(&r);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "BlastCache"
int BlastCache(void)
{
  int    i,ierr,n = 1000000;
  PetscScalar *x,*y,*z,*a,*b;

  ierr = PetscMalloc(5*n*sizeof(PetscScalar),&x);CHKERRQ(ierr);
  y = x + n;
  z = y + n;
  a = z + n;
  b = a + n;

  for (i=0; i<n; i++) {
    a[i] = (PetscScalar) i;
    y[i] = (PetscScalar) i;
    z[i] = (PetscScalar) i;
    b[i] = (PetscScalar) i;
    x[i] = (PetscScalar) i;
  }

  for (i=0; i<n; i++) {
    a[i] = 3.0*x[i] + 2.0*y[i] + 3.3*z[i] - 25.*b[i];
  }
  for (i=0; i<n; i++) {
    b[i] = 3.0*x[i] + 2.0*y[i] + 3.3*a[i] - 25.*b[i];
  }
  for (i=0; i<n; i++) {
    z[i] = 3.0*x[i] + 2.0*y[i] + 3.3*a[i] - 25.*b[i];
  }
  ierr = PetscFree(x);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
