
static char help[] = "Tests creation of PC context.\n";

#include "pc.h"
#include "vec.h"
#include "mat.h"
#include <stdio.h>
#include "options.h"


int main(int argc,char **args)
{
  PC  pc;
  int ierr, n = 5;
  Vec u;
  Mat mat;

  PetscInitialize(&argc,&args,0,0);
  if (OptionsHasName(0,0,"-help")) fprintf(stderr,"%s",help);
  ierr = PCCreate(&pc);				CHKERR(ierr);
  ierr = PCSetMethod(pc,PCNONE);		CHKERR(ierr);

  /* Vector and matrix must be set before PCSetUp */
  ierr = VecCreateSequential(n,&u);		CHKERR(ierr);
  ierr = PCSetVector(pc,u);			CHKERR(ierr);
  ierr = MatCreateSequentialAIJ(n,n,3,0,&mat);	CHKERR(ierr);
  ierr = PCSetMat(pc,mat);			CHKERR(ierr);

  ierr = PCSetUp(pc);				CHKERR(ierr);

  ierr = VecDestroy(u);				CHKERR(ierr);
  ierr = MatDestroy(mat);			CHKERR(ierr);
  ierr = PCDestroy(pc);				CHKERR(ierr);
  PetscFinalize();
  return 0;
}
    


