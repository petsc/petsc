
static char help[] = "Tests PC and KSP on tridiagonal matrix\n";

#include "vec.h"
#include "mat.h"
#include "ksp.h"
#include "pc.h"
#include "stdio.h"
#include "options.h"

int main(int argc,char **args)
{
  Mat       mat;
  Vec       b,ustar,u;
  PC        pc;
  KSP       ksp;
  int       ierr, n = 10, i, its, col[3];
  Scalar    value[3], mone = -1.0, norm, one = 1.0, zero = 0.0;
  KSPMETHOD kspmethod;
  PCMETHOD  pcmethod;
  char      *kspname, *pcname;

  OptionsCreate(&argc,&args,0,0);
  if (OptionsHasName(0,0,"-help")) fprintf(stderr,"%s",help);

  ierr = VecCreateSequential(n,&b);     CHKERR(ierr);
  ierr = VecCreateSequential(n,&ustar); CHKERR(ierr);
  ierr = VecCreateSequential(n,&u);     CHKERR(ierr);
  ierr = VecSet(&one,ustar);            CHKERR(ierr);
  ierr = VecSet(&zero,u);               CHKERR(ierr);

  ierr = MatCreateSequentialAIJ(n,n,3,0,&mat); CHKERR(ierr);
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


  ierr = MatMult(mat,ustar,b); CHKERR(ierr);

  ierr = PCCreate(&pc); CHKERR(ierr);
  ierr = PCSetMethod(pc,PCNONE); CHKERR(ierr);
  PCSetFromOptions(pc);
  ierr = PCSetMat(pc,mat); CHKERR(ierr);
  ierr = PCSetVector(pc,u);   CHKERR(ierr);
  ierr = PCSetUp(pc); CHKERR(ierr);

  ierr = KSPCreate(&ksp); CHKERR(ierr);
  ierr = KSPSetMethod(ksp,KSPRICHARDSON); CHKERR(ierr);
  KSPSetFromOptions(ksp);
  ierr = KSPSetSolution(ksp,u); CHKERR(ierr);
  ierr = KSPSetRhs(ksp,b); CHKERR(ierr);
  ierr = KSPSetAmult(ksp,mat); CHKERR(ierr);
  ierr = KSPSetBinv(ksp,pc); CHKERR(ierr);
  ierr = KSPSetUp(ksp); CHKERR(ierr);

  KSPGetMethodFromContext(ksp,&kspmethod);
  KSPGetMethodName(kspmethod,&kspname);
  PCGetMethodFromContext(pc,&pcmethod);
  PCGetMethodName(pcmethod,&pcname);
  
  printf("Running %s with %s preconditioning\n",kspname,pcname);
  ierr = KSPSolve(ksp,&its); CHKERR(ierr);
  ierr = VecAXPY(&mone,ustar,u); CHKERR(ierr);
  ierr = VecNorm(u,&norm);
  fprintf(stdout,"Number of iterations %d 2 norm error %g\n",its,norm);

  ierr = KSPDestroy(ksp); CHKERR(ierr);
  ierr = VecDestroy(u); CHKERR(ierr);
  ierr = VecDestroy(ustar); CHKERR(ierr);
  ierr = VecDestroy(b); CHKERR(ierr);
  ierr = MatDestroy(mat); CHKERR(ierr);
  ierr = PCDestroy(pc); CHKERR(ierr);

  PetscFinalize();
  return 0;
}
    


