
#include "pc.h"
#include <stdio.h>
#include "options.h"


int main(int argc,char **args)
{
  PC  pc;
  int ierr;

  OptionsCreate(argc,args,0,0);
  ierr = PCCreate(&pc); CHKERR(ierr);

  ierr = PCSetMethod(pc,PCNONE); CHKERR(ierr);
  ierr = PCSetUp(pc); CHKPTR(ierr);
  ierr = PCDestroy(pc); CHKERR(ierr);
  return 0;
}
    


