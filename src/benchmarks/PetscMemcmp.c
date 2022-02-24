
#include <petscsys.h>
#include <petsctime.h>

int main(int argc,char **argv)
{
  PetscLogDouble x,y,z;
  PetscScalar    A[10000],B[10000];
  PetscInt       i;
  PetscErrorCode ierr;
  PetscBool      flg;

  ierr = PetscInitialize(&argc,&argv,0,0);if (ierr) return ierr;
  for (i=0; i<10000; i++) {
    A[i] = i%61897;
    B[i] = i%61897;
  }
  /* To take care of paging effects */
  PetscMemcmp(A,B,sizeof(PetscScalar)*0,&flg);
  CHKERRQ(PetscTime(&x));

  CHKERRQ(PetscTime(&x));
  PetscMemcmp(A,B,sizeof(PetscScalar)*10000,&flg);
  PetscMemcmp(A,B,sizeof(PetscScalar)*10000,&flg);
  PetscMemcmp(A,B,sizeof(PetscScalar)*10000,&flg);
  PetscMemcmp(A,B,sizeof(PetscScalar)*10000,&flg);
  PetscMemcmp(A,B,sizeof(PetscScalar)*10000,&flg);
  PetscMemcmp(A,B,sizeof(PetscScalar)*10000,&flg);
  PetscMemcmp(A,B,sizeof(PetscScalar)*10000,&flg);
  PetscMemcmp(A,B,sizeof(PetscScalar)*10000,&flg);
  PetscMemcmp(A,B,sizeof(PetscScalar)*10000,&flg);
  PetscMemcmp(A,B,sizeof(PetscScalar)*10000,&flg);
  CHKERRQ(PetscTime(&y));
  PetscMemcmp(A,B,sizeof(PetscScalar)*0,&flg);
  PetscMemcmp(A,B,sizeof(PetscScalar)*0,&flg);
  PetscMemcmp(A,B,sizeof(PetscScalar)*0,&flg);
  PetscMemcmp(A,B,sizeof(PetscScalar)*0,&flg);
  PetscMemcmp(A,B,sizeof(PetscScalar)*0,&flg);
  PetscMemcmp(A,B,sizeof(PetscScalar)*0,&flg);
  PetscMemcmp(A,B,sizeof(PetscScalar)*0,&flg);
  PetscMemcmp(A,B,sizeof(PetscScalar)*0,&flg);
  PetscMemcmp(A,B,sizeof(PetscScalar)*0,&flg);
  PetscMemcmp(A,B,sizeof(PetscScalar)*0,&flg);
  CHKERRQ(PetscTime(&z));

  fprintf(stdout,"%s : \n","PetscMemcmp");
  fprintf(stdout,"    %-15s : %e sec\n","Latency",(z-y)/10.0);
  fprintf(stdout,"    %-15s : %e sec\n","Per PetscScalar",(2*y-x-z)/100000);

  ierr = PetscFinalize();
  return ierr;
}
