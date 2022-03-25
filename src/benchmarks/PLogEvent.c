
#include <petscsys.h>
#include <petsctime.h>

int main(int argc,char **argv)
{
  PetscLogDouble x,y;
  PetscLogEvent  e1;
  PetscBool      flg;

  PetscCall(PetscInitialize(&argc,&argv,0,0));
  PetscLogEventRegister("*DummyEvent",0,&e1);
  /* To take care of the paging effects */
  PetscCall(PetscTime(&x));
  PetscCall(PetscLogEventBegin(e1,&x,0,0,0));
  PetscCall(PetscLogEventEnd(e1,&x,0,0,0));

  PetscCall(PetscTime(&x));
  /* 10 Occurrences of the dummy event */
  PetscCall(PetscLogEventBegin(e1,&x,0,0,0));
  PetscCall(PetscLogEventEnd(e1,&x,0,0,0));
  PetscCall(PetscLogEventBegin(e1,&x,&y,0,0));
  PetscCall(PetscLogEventEnd(e1,&x,&y,0,0));
  PetscCall(PetscLogEventBegin(e1,&y,0,0,0));
  PetscCall(PetscLogEventEnd(e1,&y,0,0,0));
  PetscCall(PetscLogEventBegin(e1,&x,0,0,0));
  PetscCall(PetscLogEventEnd(e1,&x,0,0,0));
  PetscCall(PetscLogEventBegin(e1,&x,&y,0,0));
  PetscCall(PetscLogEventEnd(e1,&x,&y,0,0));
  PetscCall(PetscLogEventBegin(e1,&y,0,0,0));
  PetscCall(PetscLogEventEnd(e1,&y,0,0,0));
  PetscCall(PetscLogEventBegin(e1,&x,0,0,0));
  PetscCall(PetscLogEventEnd(e1,&x,0,0,0));
  PetscCall(PetscLogEventBegin(e1,&x,&y,0,0));
  PetscCall(PetscLogEventEnd(e1,&x,&y,0,0));
  PetscCall(PetscLogEventBegin(e1,&y,0,0,0));
  PetscCall(PetscLogEventEnd(e1,&y,0,0,0));
  PetscCall(PetscLogEventBegin(e1,&x,&e1,0,0));
  PetscCall(PetscLogEventEnd(e1,&x,&e1,0,0));

  PetscCall(PetscTime(&y));
  fprintf(stderr,"%-15s : %e sec, with options : ","PetscLogEvent",(y-x)/10.0);

  PetscCall(PetscOptionsHasName(NULL,"-log",&flg));
  if (flg) fprintf(stderr,"-log ");
  PetscCall(PetscOptionsHasName(NULL,"-log_all",&flg));
  if (flg) fprintf(stderr,"-log_all ");
  PetscCall(PetscOptionsHasName(NULL,"-log_view",&flg));
  if (flg) fprintf(stderr,"-log_view ");
  PetscCall(PetscOptionsHasName(NULL,"-log_mpe",&flg));
  if (flg) fprintf(stderr,"-log_mpe ");

  fprintf(stderr,"\n");

  PetscCall(PetscFinalize());
  return 0;
}
