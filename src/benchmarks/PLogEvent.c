/*$Id: PLogEvent.c,v 1.21 2001/01/17 22:28:38 bsmith Exp balay $*/

#include "petsc.h"

#undef __FUNC__
#define __FUNC__ "main"
int main(int argc,char **argv)
{
  PetscLogDouble x,y;
  int        e1,ierr;
  PetscTruth flg;

  PetscInitialize(&argc,&argv,0,0);
  PetscLogEventRegister(&e1,"*DummyEvent","red:");
  /* To take care of the paging effects */
  ierr = PetscGetTime(&x);CHKERRQ(ierr);
  ierr = PetscLogEventBegin(e1,&x,0,0,0);CHKERRQ(ierr);
  ierr = PetscLogEventEnd(e1,&x,0,0,0);  CHKERRQ(ierr);

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
  fprintf(stdout,"%-15s : %e sec, with options : ","PetscLogEvent",(y-x)/10.0);

  if(PetscOptionsHasName(PETSC_NULL,"-log",&flg),flg) fprintf(stdout,"-log ");
  if(PetscOptionsHasName(PETSC_NULL,"-log_all",&flg),flg) fprintf(stdout,"-log_all ");
  if(PetscOptionsHasName(PETSC_NULL,"-log_summary",&flg),flg) fprintf(stdout,"-log_summary ");
  if(PetscOptionsHasName(PETSC_NULL,"-log_mpe",&flg),flg) fprintf(stdout,"-log_mpe ");
  
  fprintf(stdout,"\n");

  ierr = PetscFinalize();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
