
#include <petscsys.h>
#include <petsctime.h>

int main(int argc,char **argv)
{
  PetscLogDouble x,y;
  PetscErrorCode ierr;
  PetscScalar    *A,*B;

  ierr = PetscInitialize(&argc,&argv,0,0);if (ierr) return ierr;
  ierr = PetscCalloc1(8000000,&A);CHKERRQ(ierr);
  ierr = PetscMalloc1(8000000,&B);CHKERRQ(ierr);

  for (i=0; i<8000000; i++) {
    A[i] = i%61897;
    B[i] = i%61897;
  }
  /* To take care of paging effects */
  ierr = PetscArraycpy(A,B,8000000);CHKERRQ(ierr);
  ierr = PetscTime(&x);CHKERRQ(ierr);
  ierr = PetscArraycpy(A,B,8000000);CHKERRQ(ierr);
  ierr = PetscTime(&x);CHKERRQ(ierr);

  fprintf(stdout,"%s : \n","PetscMemcpy");
  fprintf(stdout,"    %-15s : %e MB/s\n","Bandwidth",10.0*8*8/(y-x));

  ierr = PetscFree(A);CHKERRQ(ierr);
  ierr = PetscFree(B);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
}
