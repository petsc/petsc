
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
  ierr = PCCreate(MPI_COMM_WORLD,&pc);		CHKERRA(ierr);
  ierr = PCSetMethod(pc,PCNONE);		CHKERRA(ierr);

  /* Vector and matrix must be set before PCSetUp */
  ierr = VecCreateSequential(n,&u);		CHKERRA(ierr);
  ierr = PCSetVector(pc,u);			CHKERRA(ierr);
  ierr = MatCreateSequentialAIJ(MPI_COMM_SELF,n,n,3,0,&mat);	CHKERRA(ierr);
  ierr = PCSetOperators(pc,mat,mat,0);		CHKERRA(ierr);

  ierr = PCSetUp(pc);				CHKERRA(ierr);

  ierr = VecDestroy(u);				CHKERRA(ierr);
  ierr = MatDestroy(mat);			CHKERRA(ierr);
  ierr = PCDestroy(pc);				CHKERRA(ierr);
  PetscFinalize();
  return 0;
}
    


