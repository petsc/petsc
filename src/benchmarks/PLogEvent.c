
#include <petscsys.h>
#include <petsctime.h>

int main(int argc,char **argv)
{
  PetscLogDouble x,y;
  PetscLogEvent  e1;
  PetscErrorCode ierr;
  PetscBool      flg;

  ierr = PetscInitialize(&argc,&argv,0,0);if (ierr) return ierr;
  PetscLogEventRegister("*DummyEvent",0,&e1);
  /* To take care of the paging effects */
  CHKERRQ(PetscTime(&x));
  CHKERRQ(PetscLogEventBegin(e1,&x,0,0,0));
  CHKERRQ(PetscLogEventEnd(e1,&x,0,0,0));

  CHKERRQ(PetscTime(&x));
  /* 10 Occurrences of the dummy event */
  CHKERRQ(PetscLogEventBegin(e1,&x,0,0,0));
  CHKERRQ(PetscLogEventEnd(e1,&x,0,0,0));
  CHKERRQ(PetscLogEventBegin(e1,&x,&y,0,0));
  CHKERRQ(PetscLogEventEnd(e1,&x,&y,0,0));
  CHKERRQ(PetscLogEventBegin(e1,&y,0,0,0));
  CHKERRQ(PetscLogEventEnd(e1,&y,0,0,0));
  CHKERRQ(PetscLogEventBegin(e1,&x,0,0,0));
  CHKERRQ(PetscLogEventEnd(e1,&x,0,0,0));
  CHKERRQ(PetscLogEventBegin(e1,&x,&y,0,0));
  CHKERRQ(PetscLogEventEnd(e1,&x,&y,0,0));
  CHKERRQ(PetscLogEventBegin(e1,&y,0,0,0));
  CHKERRQ(PetscLogEventEnd(e1,&y,0,0,0));
  CHKERRQ(PetscLogEventBegin(e1,&x,0,0,0));
  CHKERRQ(PetscLogEventEnd(e1,&x,0,0,0));
  CHKERRQ(PetscLogEventBegin(e1,&x,&y,0,0));
  CHKERRQ(PetscLogEventEnd(e1,&x,&y,0,0));
  CHKERRQ(PetscLogEventBegin(e1,&y,0,0,0));
  CHKERRQ(PetscLogEventEnd(e1,&y,0,0,0));
  CHKERRQ(PetscLogEventBegin(e1,&x,&e1,0,0));
  CHKERRQ(PetscLogEventEnd(e1,&x,&e1,0,0));

  CHKERRQ(PetscTime(&y));
  fprintf(stderr,"%-15s : %e sec, with options : ","PetscLogEvent",(y-x)/10.0);

  CHKERRQ(PetscOptionsHasName(NULL,"-log",&flg));
  if (flg) fprintf(stderr,"-log ");
  CHKERRQ(PetscOptionsHasName(NULL,"-log_all",&flg));
  if (flg) fprintf(stderr,"-log_all ");
  CHKERRQ(PetscOptionsHasName(NULL,"-log_view",&flg));
  if (flg) fprintf(stderr,"-log_view ");
  CHKERRQ(PetscOptionsHasName(NULL,"-log_mpe",&flg));
  if (flg) fprintf(stderr,"-log_mpe ");

  fprintf(stderr,"\n");

  ierr = PetscFinalize();
  return ierr;
}
