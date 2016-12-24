
#include <petscsys.h>
#include <petsctime.h>

int main(int argc,char **argv)
{
  PetscLogDouble x,y,z;
  PetscInt       i;
  PetscErrorCode ierr;
  PetscScalar    *A,*B;

  ierr = PetscInitialize(&argc,&argv,0,0);if (ierr) return ierr;
  ierr = PetscMalloc1(8000000,&A);CHKERRQ(ierr);
  ierr = PetscMalloc1(8000000,&B);CHKERRQ(ierr);

  for (i=0; i<8000000; i++) {
    A[i] = i%61897;
    B[i] = i%61897;
  }
  /* To take care of paging effects */
  ierr = PetscMemcpy(A,B,sizeof(PetscScalar)*0);CHKERRQ(ierr);
  ierr = PetscTime(&x);CHKERRQ(ierr);

  ierr = PetscTime(&x);CHKERRQ(ierr);
  /*
  PetscMemcpy(A,B,sizeof(PetscScalar)*8000000);
  PetscMemcpy(A,B,sizeof(PetscScalar)*8000000);
  PetscMemcpy(A,B,sizeof(PetscScalar)*8000000);
  PetscMemcpy(A,B,sizeof(PetscScalar)*8000000);
  PetscMemcpy(A,B,sizeof(PetscScalar)*8000000);
  PetscMemcpy(A,B,sizeof(PetscScalar)*8000000);
  PetscMemcpy(A,B,sizeof(PetscScalar)*8000000);
  PetscMemcpy(A,B,sizeof(PetscScalar)*8000000);
  PetscMemcpy(A,B,sizeof(PetscScalar)*8000000);
  PetscMemcpy(A,B,sizeof(PetscScalar)*8000000); */
  { int j;
    for (j = 0; j<10; j++)
      for (i=0; i<8000000; i++) B[i] = A[i];}

  ierr = PetscTime(&y);CHKERRQ(ierr);
  ierr = PetscMemcpy(A,B,sizeof(PetscScalar)*0);CHKERRQ(ierr);
  ierr = PetscMemcpy(A,B,sizeof(PetscScalar)*0);CHKERRQ(ierr);
  ierr = PetscMemcpy(A,B,sizeof(PetscScalar)*0);CHKERRQ(ierr);
  ierr = PetscMemcpy(A,B,sizeof(PetscScalar)*0);CHKERRQ(ierr);
  ierr = PetscMemcpy(A,B,sizeof(PetscScalar)*0);CHKERRQ(ierr);
  ierr = PetscMemcpy(A,B,sizeof(PetscScalar)*0);CHKERRQ(ierr);
  ierr = PetscMemcpy(A,B,sizeof(PetscScalar)*0);CHKERRQ(ierr);
  ierr = PetscMemcpy(A,B,sizeof(PetscScalar)*0);CHKERRQ(ierr);
  ierr = PetscMemcpy(A,B,sizeof(PetscScalar)*0);CHKERRQ(ierr);
  ierr = PetscMemcpy(A,B,sizeof(PetscScalar)*0);CHKERRQ(ierr);
  ierr = PetscTime(&z);CHKERRQ(ierr);

  fprintf(stdout,"%s : \n","PetscMemcpy");
  fprintf(stdout,"    %-15s : %e MB/s\n","Bandwidth",10.0*8*8/(y-x));
  fprintf(stdout,"    %-15s : %e sec\n","Latency",(z-y)/10.0);
  fprintf(stdout,"    %-15s : %e sec\n","Per PetscScalar",(2*y-x-z)/8000000.0);

  ierr = PetscFree(A);CHKERRQ(ierr);
  ierr = PetscFree(B);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
}
