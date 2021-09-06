
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
  ierr = PetscTime(&x);CHKERRQ(ierr);
  ierr = PetscLogEventBegin(e1,&x,0,0,0);CHKERRQ(ierr);
  ierr = PetscLogEventEnd(e1,&x,0,0,0);CHKERRQ(ierr);

  ierr = PetscTime(&x);CHKERRQ(ierr);
  /* 10 Occurrences of the dummy event */
  ierr = PetscLogEventBegin(e1,&x,0,0,0);CHKERRQ(ierr);
  ierr = PetscLogEventEnd(e1,&x,0,0,0);CHKERRQ(ierr);
  ierr = PetscLogEventBegin(e1,&x,&y,0,0);CHKERRQ(ierr);
  ierr = PetscLogEventEnd(e1,&x,&y,0,0);CHKERRQ(ierr);
  ierr = PetscLogEventBegin(e1,&y,0,0,0);CHKERRQ(ierr);
  ierr = PetscLogEventEnd(e1,&y,0,0,0);CHKERRQ(ierr);
  ierr = PetscLogEventBegin(e1,&x,0,0,0);CHKERRQ(ierr);
  ierr = PetscLogEventEnd(e1,&x,0,0,0);CHKERRQ(ierr);
  ierr = PetscLogEventBegin(e1,&x,&y,0,0);CHKERRQ(ierr);
  ierr = PetscLogEventEnd(e1,&x,&y,0,0);CHKERRQ(ierr);
  ierr = PetscLogEventBegin(e1,&y,0,0,0);CHKERRQ(ierr);
  ierr = PetscLogEventEnd(e1,&y,0,0,0);CHKERRQ(ierr);
  ierr = PetscLogEventBegin(e1,&x,0,0,0);CHKERRQ(ierr);
  ierr = PetscLogEventEnd(e1,&x,0,0,0);CHKERRQ(ierr);
  ierr = PetscLogEventBegin(e1,&x,&y,0,0);CHKERRQ(ierr);
  ierr = PetscLogEventEnd(e1,&x,&y,0,0);CHKERRQ(ierr);
  ierr = PetscLogEventBegin(e1,&y,0,0,0);CHKERRQ(ierr);
  ierr = PetscLogEventEnd(e1,&y,0,0,0);CHKERRQ(ierr);
  ierr = PetscLogEventBegin(e1,&x,&e1,0,0);CHKERRQ(ierr);
  ierr = PetscLogEventEnd(e1,&x,&e1,0,0);CHKERRQ(ierr);

  ierr = PetscTime(&y);CHKERRQ(ierr);
  fprintf(stderr,"%-15s : %e sec, with options : ","PetscLogEvent",(y-x)/10.0);

  ierr = PetscOptionsHasName(NULL,"-log",&flg);CHKERRQ(ierr);
  if (flg) fprintf(stderr,"-log ");
  ierr = PetscOptionsHasName(NULL,"-log_all",&flg);CHKERRQ(ierr);
  if (flg) fprintf(stderr,"-log_all ");
  ierr = PetscOptionsHasName(NULL,"-log_view",&flg);CHKERRQ(ierr);
  if (flg) fprintf(stderr,"-log_view ");
  ierr = PetscOptionsHasName(NULL,"-log_mpe",&flg);CHKERRQ(ierr);
  if (flg) fprintf(stderr,"-log_mpe ");

  fprintf(stderr,"\n");

  ierr = PetscFinalize();
  return ierr;
}
