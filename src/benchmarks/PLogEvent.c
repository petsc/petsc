/*$Id: PLogEvent.c,v 1.19 2000/11/28 17:32:38 bsmith Exp bsmith $*/

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
  ierr = PetscGetTime(&x);CHKERRA(ierr);
  ierr = PetscLogEventBegin(e1,&x,0,0,0);CHKERRA(ierr);
  ierr = PetscLogEventEnd(e1,&x,0,0,0);  CHKERRA(ierr);

  ierr = PetscGetTime(&x);CHKERRA(ierr);
  /* 10 Occurences of the dummy event */
  ierr = PetscLogEventBegin(e1,&x,0,0,0);CHKERRA(ierr);
  ierr = PetscLogEventEnd(e1,&x,0,0,0);CHKERRA(ierr);
  ierr = PetscLogEventBegin(e1,&x,&y,0,0);CHKERRA(ierr);
  ierr = PetscLogEventEnd(e1,&x,&y,0,0);CHKERRA(ierr);
  ierr = PetscLogEventBegin(e1,&y,0,0,0);CHKERRA(ierr);
  ierr = PetscLogEventEnd(e1,&y,0,0,0);CHKERRA(ierr);
  ierr = PetscLogEventBegin(e1,&x,0,0,0);CHKERRA(ierr);
  ierr = PetscLogEventEnd(e1,&x,0,0,0);CHKERRA(ierr);
  ierr = PetscLogEventBegin(e1,&x,&y,0,0);CHKERRA(ierr);
  ierr = PetscLogEventEnd(e1,&x,&y,0,0);CHKERRA(ierr);
  ierr = PetscLogEventBegin(e1,&y,0,0,0);CHKERRA(ierr);
  ierr = PetscLogEventEnd(e1,&y,0,0,0);CHKERRA(ierr);
  ierr = PetscLogEventBegin(e1,&x,0,0,0);CHKERRA(ierr);
  ierr = PetscLogEventEnd(e1,&x,0,0,0);CHKERRA(ierr);
  ierr = PetscLogEventBegin(e1,&x,&y,0,0);CHKERRA(ierr);
  ierr = PetscLogEventEnd(e1,&x,&y,0,0);CHKERRA(ierr);
  ierr = PetscLogEventBegin(e1,&y,0,0,0);CHKERRA(ierr);
  ierr = PetscLogEventEnd(e1,&y,0,0,0);CHKERRA(ierr);
  ierr = PetscLogEventBegin(e1,&x,&e1,0,0);CHKERRA(ierr);
  ierr = PetscLogEventEnd(e1,&x,&e1,0,0);CHKERRA(ierr);

  ierr = PetscGetTime(&y);CHKERRA(ierr);
  fprintf(stdout,"%-15s : %e sec, with options : ","PetscLogEvent",(y-x)/10.0);

  if(PetscOptionsHasName(PETSC_NULL,"-log",&flg),flg) fprintf(stdout,"-log ");
  if(PetscOptionsHasName(PETSC_NULL,"-log_all",&flg),flg) fprintf(stdout,"-log_all ");
  if(PetscOptionsHasName(PETSC_NULL,"-log_summary",&flg),flg) fprintf(stdout,"-log_summary ");
  if(PetscOptionsHasName(PETSC_NULL,"-log_mpe",&flg),flg) fprintf(stdout,"-log_mpe ");
  
  fprintf(stdout,"\n");

  PetscFinalize();
  PetscFunctionReturn(0);
}
