/*$Id: PLogEvent.c,v 1.18 2000/09/06 22:19:15 balay Exp bsmith $*/

#include "petsc.h"

#undef __FUNC__
#define __FUNC__ "main"
int main(int argc,char **argv)
{
  PLogDouble x,y;
  int        e1,ierr;
  PetscTruth flg;

  PetscInitialize(&argc,&argv,0,0);
  PLogEventRegister(&e1,"*DummyEvent","red:");
  /* To take care of the paging effects */
  ierr = PetscGetTime(&x);CHKERRA(ierr);
  ierr = PLogEventBegin(e1,&x,0,0,0);CHKERRA(ierr);
  ierr = PLogEventEnd(e1,&x,0,0,0);  CHKERRA(ierr);

  ierr = PetscGetTime(&x);CHKERRA(ierr);
  /* 10 Occurences of the dummy event */
  ierr = PLogEventBegin(e1,&x,0,0,0);CHKERRA(ierr);
  ierr = PLogEventEnd(e1,&x,0,0,0);CHKERRA(ierr);
  ierr = PLogEventBegin(e1,&x,&y,0,0);CHKERRA(ierr);
  ierr = PLogEventEnd(e1,&x,&y,0,0);CHKERRA(ierr);
  ierr = PLogEventBegin(e1,&y,0,0,0);CHKERRA(ierr);
  ierr = PLogEventEnd(e1,&y,0,0,0);CHKERRA(ierr);
  ierr = PLogEventBegin(e1,&x,0,0,0);CHKERRA(ierr);
  ierr = PLogEventEnd(e1,&x,0,0,0);CHKERRA(ierr);
  ierr = PLogEventBegin(e1,&x,&y,0,0);CHKERRA(ierr);
  ierr = PLogEventEnd(e1,&x,&y,0,0);CHKERRA(ierr);
  ierr = PLogEventBegin(e1,&y,0,0,0);CHKERRA(ierr);
  ierr = PLogEventEnd(e1,&y,0,0,0);CHKERRA(ierr);
  ierr = PLogEventBegin(e1,&x,0,0,0);CHKERRA(ierr);
  ierr = PLogEventEnd(e1,&x,0,0,0);CHKERRA(ierr);
  ierr = PLogEventBegin(e1,&x,&y,0,0);CHKERRA(ierr);
  ierr = PLogEventEnd(e1,&x,&y,0,0);CHKERRA(ierr);
  ierr = PLogEventBegin(e1,&y,0,0,0);CHKERRA(ierr);
  ierr = PLogEventEnd(e1,&y,0,0,0);CHKERRA(ierr);
  ierr = PLogEventBegin(e1,&x,&e1,0,0);CHKERRA(ierr);
  ierr = PLogEventEnd(e1,&x,&e1,0,0);CHKERRA(ierr);

  ierr = PetscGetTime(&y);CHKERRA(ierr);
  fprintf(stdout,"%-15s : %e sec, with options : ","PLogEvent",(y-x)/10.0);

  if(OptionsHasName(PETSC_NULL,"-log",&flg),flg) fprintf(stdout,"-log ");
  if(OptionsHasName(PETSC_NULL,"-log_all",&flg),flg) fprintf(stdout,"-log_all ");
  if(OptionsHasName(PETSC_NULL,"-log_summary",&flg),flg) fprintf(stdout,"-log_summary ");
  if(OptionsHasName(PETSC_NULL,"-log_mpe",&flg),flg) fprintf(stdout,"-log_mpe ");
  
  fprintf(stdout,"\n");

  PetscFinalize();
  PetscFunctionReturn(0);
}
