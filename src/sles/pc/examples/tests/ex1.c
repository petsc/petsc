#ifndef lint
static char vcid[] = "$Id: ex1.c,v 1.16 1995/08/30 23:47:40 curfman Exp bsmith $";
#endif

static char help[] = "This example tests the creation of a PC context.\n\n";

#include "pc.h"
#include "petsc.h"
#include <stdio.h>

int main(int argc,char **args)
{
  PC  pc;
  int ierr, n = 5;
  Vec u;
  Mat mat;

  PetscInitialize(&argc,&args,0,0);
  if (OptionsHasName(0,"-help")) fprintf(stdout,"%s",help);
  ierr = PCCreate(MPI_COMM_WORLD,&pc); CHKERRA(ierr);
  ierr = PCSetMethod(pc,PCNONE); CHKERRA(ierr);

  /* Vector and matrix must be set before calling PCSetUp */
  ierr = VecCreateSeq(MPI_COMM_SELF,n,&u);CHKERRA(ierr);
  ierr = PCSetVector(pc,u); CHKERRA(ierr);
  ierr = MatCreateSeqAIJ(MPI_COMM_SELF,n,n,3,0,&mat); CHKERRA(ierr);
  ierr = PCSetOperators(pc,mat,mat,ALLMAT_DIFFERENT_NONZERO_PATTERN);
         CHKERRA(ierr);
  ierr = PCSetUp(pc); CHKERRA(ierr);

  ierr = VecDestroy(u);	CHKERRA(ierr);
  ierr = MatDestroy(mat); CHKERRA(ierr);
  ierr = PCDestroy(pc);	CHKERRA(ierr);
  PetscFinalize();
  return 0;
}
    


