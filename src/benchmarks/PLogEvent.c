/*$Id: PLogEvent.c,v 1.14 1999/05/04 20:38:02 balay Exp bsmith $*/

#include "petsc.h"

#undef __FUNC__
#define __FUNC__ "main"
int main( int argc, char **argv)
{
  PLogDouble x, y;
  int        e1, flg,ierr;

  PetscInitialize(&argc, &argv,0,0);
  PLogEventRegister(&e1,"*DummyEvent", "red:");
  /* To take care of the paging effects */
  ierr = PetscGetTime(&x);CHKERRA(ierr);
  PLogEventBegin(e1,&x,0,0,0);
  PLogEventEnd(e1,&x,0,0,0);  

  ierr = PetscGetTime(&x);CHKERRA(ierr);
  /* 10 Occurences of the dummy event */
  PLogEventBegin(e1,&x,0,0,0);
  PLogEventEnd(e1,&x,0,0,0);  
  PLogEventBegin(e1,&x,&y,0,0);
  PLogEventEnd(e1,&x,&y,0,0);  
  PLogEventBegin(e1,&y,0,0,0);
  PLogEventEnd(e1,&y,0,0,0);  
  PLogEventBegin(e1,&x,0,0,0);
  PLogEventEnd(e1,&x,0,0,0);  
  PLogEventBegin(e1,&x,&y,0,0);
  PLogEventEnd(e1,&x,&y,0,0);  
  PLogEventBegin(e1,&y,0,0,0);
  PLogEventEnd(e1,&y,0,0,0);  
  PLogEventBegin(e1,&x,0,0,0);
  PLogEventEnd(e1,&x,0,0,0);  
  PLogEventBegin(e1,&x,&y,0,0);
  PLogEventEnd(e1,&x,&y,0,0);  
  PLogEventBegin(e1,&y,0,0,0);
  PLogEventEnd(e1,&y,0,0,0);  
  PLogEventBegin(e1,&x,&e1,0,0);
  PLogEventEnd(e1,&x,&e1,0,0);  

  ierr = PetscGetTime(&y);CHKERRA(ierr);
  fprintf(stderr,"%-15s : %e sec , with options : ","PLogEvent",(y-x)/10.0);

  if(OptionsHasName(PETSC_NULL,"-log",&flg),flg) fprintf(stderr,"-log ");
  if(OptionsHasName(PETSC_NULL,"-log_all",&flg),flg) fprintf(stderr,"-log_all ");
  if(OptionsHasName(PETSC_NULL,"-log_summary",&flg),flg) fprintf(stderr,"-log_summary ");
  if(OptionsHasName(PETSC_NULL,"-log_mpe",&flg),flg) fprintf(stderr,"-log_mpe ");
  
  fprintf(stderr,"\n");

  PetscFinalize();
  PetscFunctionReturn(0);
}
