#ifndef lint
static char vcid[] = "$Id: MPI_Wtime.c,v 1.3 1996/03/06 17:40:32 balay Exp $";
#endif

#include "stdio.h"
#include "petsc.h"

int main( int argc, char **argv)
{
  double x, y;
  int e1, flg;

  PetscInitialize(&argc, &argv,0,0,0);
  PLogEventRegister(&e1,"*DummyEvent     ", "red");
  /* To take care of the paging effects */
  PLogEventBegin(e1,&x,0,0,0);
  PLogEventEnd(e1,&x,0,0,0);  
  x = PetscGetTime();

  x = PetscGetTime();
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

  y = PetscGetTime();
  fprintf(stderr,"%-15s : %e sec , with options : ","PLogEvent",(y-x)/10.0);

  if(OptionsHasName(PETSC_NULL,"-log",&flg),flg) fprintf(stderr,"-log ");
  if(OptionsHasName(PETSC_NULL,"-log_all",&flg),flg) fprintf(stderr,"-log_all ");
  if(OptionsHasName(PETSC_NULL,"-log_summary",&flg),flg) fprintf(stderr,"-log_summary ");
  if(OptionsHasName(PETSC_NULL,"-log_mpe",&flg),flg) fprintf(stderr,"-log_mpe ");
  
  fprintf(stderr,"\n");

  PetscFinalize();
  return 0;
}
