
/*
    Demonstrates the use of fast Richardson for SOR.
    Also tests MatRelax routines.
*/

#include "vec.h"
#include "mat.h"
#include "ksp.h"
#include "pc.h"
#include <stdio.h>
#include "options.h"

int main(int argc,char **args)
{
  Mat       mat;
  Vec       b,u;
  PC        pc;
  int       ierr, n = 5, i, col[3];
  Scalar    value[3], zero = 0.0;

  PetscInitialize(&argc,&args,0,0);
  ierr = VecCreateSequential(n,&b);     CHKERRA(ierr);
  ierr = VecCreateSequential(n,&u);     CHKERRA(ierr);

  ierr = MatCreateSequentialDense(MPI_COMM_SELF,n,n,&mat); CHKERRA(ierr);
  value[0] = -1.0; value[1] = 2.0; value[2] = -1.0;
  for (i=1; i<n-1; i++ ) {
    col[0] = i-1; col[1] = i; col[2] = i+1;
    ierr = MatSetValues(mat,1,&i,3,col,value,InsertValues); CHKERRA(ierr);
  }
  i = n - 1; col[0] = n - 2; col[1] = n - 1;
  ierr = MatSetValues(mat,1,&i,2,col,value,InsertValues); CHKERRA(ierr);
  i = 0; col[0] = 0; col[1] = 1; value[0] = 2.0; value[1] = -1.0;
  ierr = MatSetValues(mat,1,&i,2,col,value,InsertValues); CHKERRA(ierr);
  ierr = MatBeginAssembly(mat,FINAL_ASSEMBLY); CHKERRA(ierr);
  ierr = MatEndAssembly(mat,FINAL_ASSEMBLY); CHKERRA(ierr);

  ierr = PCCreate(MPI_COMM_WORLD,&pc); CHKERRA(ierr);
  ierr = PCSetMethod(pc,PCSOR); CHKERRA(ierr);
  PCSetFromOptions(pc);
  ierr = PCSetOperators(pc,mat,mat,0); CHKERRA(ierr);
  ierr = PCSetVector(pc,u);   CHKERRA(ierr);
  ierr = PCSetUp(pc); CHKERRA(ierr);


  value[0] = 1.0;
  for ( i=0; i<n; i++ ) {
    ierr = VecSet(&zero,u);               CHKERRA(ierr);
    ierr = VecSetValues(u,1,&i,value,InsertValues); CHKERRA(ierr);
    ierr = PCApply(pc,u,b);   CHKERRA(ierr);
    VecView(b,STDOUT_VIEWER);
  }

  MatDestroy(mat);
  PCDestroy(pc);
  VecDestroy(u);
  VecDestroy(b);
  PetscFinalize();
  return 0;
}
    


