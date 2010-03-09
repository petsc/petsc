
#include "petscsys.h"

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc,char **argv)
{
  PetscLogDouble x,y;
  PetscLogEvent  e1;
  PetscErrorCode ierr;
  PetscTruth     flg;

  PetscInitialize(&argc,&argv,0,0);
  PetscLogEventRegister("*DummyEvent",0,&e1);
  /* To take care of the paging effects */
  ierr = PetscGetTime(&x);CHKERRQ(ierr);
  ierr = PetscLogEventBegin(e1,&x,0,0,0);CHKERRQ(ierr);
  ierr = PetscLogEventEnd(e1,&x,0,0,0);CHKERRQ(ierr);

  ierr = PetscGetTime(&x);CHKERRQ(ierr);
  /* 10 Occurences of the dummy event */
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

  ierr = PetscGetTime(&y);CHKERRQ(ierr);
  fprintf(stderr,"%-15s : %e sec, with options : ","PetscLogEvent",(y-x)/10.0);

  if(PetscOptionsHasName(PETSC_NULL,"-log",&flg),flg) fprintf(stderr,"-log ");
  if(PetscOptionsHasName(PETSC_NULL,"-log_all",&flg),flg) fprintf(stderr,"-log_all ");
  if(PetscOptionsHasName(PETSC_NULL,"-log_summary",&flg),flg) fprintf(stderr,"-log_summary ");
  if(PetscOptionsHasName(PETSC_NULL,"-log_mpe",&flg),flg) fprintf(stderr,"-log_mpe ");
  
  fprintf(stderr,"\n");

  ierr = PetscFinalize();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
