
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
  Vec       b,ustar,u;
  PC        pc;
  int       ierr, n = 5, i, its, col[3];
  Scalar    value[3], one = 1.0, zero = 0.0;

  PetscInitialize(&argc,&args,0,0);
  ierr = VecCreateSequential(n,&b);     CHKERR(ierr);
  ierr = VecCreateSequential(n,&u);     CHKERR(ierr);

  ierr = MatCreateSequentialDense(n,n,&mat); CHKERR(ierr);
  value[0] = -1.0; value[1] = 2.0; value[2] = -1.0;
  for (i=1; i<n-1; i++ ) {
    col[0] = i-1; col[1] = i; col[2] = i+1;
    ierr = MatSetValues(mat,1,&i,3,col,value,InsertValues); CHKERR(ierr);
  }
  i = n - 1; col[0] = n - 2; col[1] = n - 1;
  ierr = MatSetValues(mat,1,&i,2,col,value,InsertValues); CHKERR(ierr);
  i = 0; col[0] = 0; col[1] = 1; value[0] = 2.0; value[1] = -1.0;
  ierr = MatSetValues(mat,1,&i,2,col,value,InsertValues); CHKERR(ierr);
  ierr = MatBeginAssembly(mat); CHKERR(ierr);
  ierr = MatEndAssembly(mat); CHKERR(ierr);

  ierr = PCCreate(&pc); CHKERR(ierr);
  ierr = PCSetMethod(pc,PCSOR); CHKERR(ierr);
  PCSetFromOptions(pc);
  ierr = PCSetMat(pc,mat); CHKERR(ierr);
  ierr = PCSetVector(pc,u);   CHKERR(ierr);
  ierr = PCSetUp(pc); CHKERR(ierr);


  value[0] = 1.0;
  for ( i=0; i<n; i++ ) {
    ierr = VecSet(&zero,u);               CHKERR(ierr);
    ierr = VecSetValues(u,1,&i,value,InsertValues); CHKERR(ierr);
    ierr = PCApply(pc,u,b);   CHKERR(ierr);
    VecView(b,0);
  }

  MatDestroy(mat);
  PCDestroy(pc);
  VecDestroy(u);
  VecDestroy(b);
  PetscFinalize();
  return 0;
}
    


