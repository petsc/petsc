
/*
  MatView_SeqAIJ
  MatMultAdd_SeqAIJ

  MatMult_MPIAIJ


  Mat_SeqAIJ
  Mat_MPIAIJ

*/

static char help[] = "MatMPIAJ";

#include "petscksp.h"

EXTERN PetscErrorCode PETSCKSP_DLLEXPORT PCCreate_Schur(PC);

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc,char **args)
{
  Vec            x,b,res;                 /* approx solution, RHS */
  Mat            A;                       /* linear system matrix */
  KSP            ksp;                    /* linear solver context */
  PC             pc;                      /* PC context */
  PetscInt       m = 4,n = 4;          /* mesh dimensions in x- and y- directions */
  PetscInt       i,j,Ii,J,Istart,Iend;
  PetscErrorCode ierr;
  PetscInt       solve=1,solveT=0;
  PetscMPIInt    size,rank;
  PetscScalar    v, one = 1.0, zero=0.0;
  PetscReal      norm;

  PetscViewer vw_s,vw_w;

  PetscInt r,c,R,C;

  PetscInitialize(&argc,&args,(char *)0,help);
  ierr = MPI_Comm_size(PETSC_COMM_WORLD,&size);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(PETSC_COMM_WORLD,&rank);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(PETSC_NULL,"-m",&m,PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(PETSC_NULL,"-n",&n,PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(PETSC_NULL,"-solve",&solve,PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(PETSC_NULL,"-solveT",&solveT,PETSC_NULL);CHKERRQ(ierr);
  
  vw_s = PETSC_VIEWER_STDOUT_SELF;
  vw_w = PETSC_VIEWER_STDOUT_WORLD;
  
  /* 
     Assemble the matrix for the five point stencil, YET AGAIN 
  */
  ierr = MatCreate(PETSC_COMM_WORLD,&A);CHKERRQ(ierr);
  ierr = MatSetSizes(A,PETSC_DECIDE,PETSC_DECIDE,m*n,m*n);CHKERRQ(ierr);
  ierr = MatSetFromOptions(A);CHKERRQ(ierr);
  ierr = MatGetOwnershipRange(A,&Istart,&Iend);CHKERRQ(ierr);
#if 1
  for (Ii=Istart; Ii<Iend; Ii++) { 
    v = -1.0; i = Ii/n; j = Ii - i*n;  
    if (i>0)   {J = Ii - n; ierr = MatSetValues(A,1,&Ii,1,&J,&v,INSERT_VALUES);CHKERRQ(ierr);}
    if (i<m-1) {J = Ii + n; ierr = MatSetValues(A,1,&Ii,1,&J,&v,INSERT_VALUES);CHKERRQ(ierr);}
    if (j>0)   {J = Ii - 1; ierr = MatSetValues(A,1,&Ii,1,&J,&v,INSERT_VALUES);CHKERRQ(ierr);}
    if (j<n-1) {J = Ii + 1; ierr = MatSetValues(A,1,&Ii,1,&J,&v,INSERT_VALUES);CHKERRQ(ierr);}
    v = 4.0; ierr = MatSetValues(A,1,&Ii,1,&Ii,&v,INSERT_VALUES);CHKERRQ(ierr);
  }
#endif
  ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

  {
    MatGetLocalSize(A,&r,&c);
    MatGetSize(A,&R,&C);
    ierr = PetscSynchronizedPrintf(PETSC_COMM_WORLD,
				   "[%d] r: %d, c: %d, R: %d, C: %d\n",
				  rank,r,c,R,C);CHKERRQ(ierr);
    PetscSynchronizedFlush(PETSC_COMM_WORLD);
  }
  if (0) {
    PetscInt n, *colmap;
    Mat Ad, Ao; 
    ierr = MatMPIAIJGetSeqAIJ(A,&Ad,&Ao,&colmap);CHKERRQ(ierr);
    MatGetSize(Ao,0,&n);

    PetscSequentialPhaseBegin(PETSC_COMM_WORLD,1);
    PetscPrintf(PETSC_COMM_SELF,"[%d] ------\n", rank);

    MatGetLocalSize(Ad,&r,&c);
    MatGetSize(Ad,&R,&C);
    ierr = PetscPrintf(PETSC_COMM_SELF,
		       "[%d] r: %d, c: %d, R: %d, C: %d\n",
		       rank,r,c,R,C);CHKERRQ(ierr);
    MatView(Ad,vw_s);
    PetscPrintf(PETSC_COMM_SELF,"\n");

    MatGetLocalSize(Ao,&r,&c);
    MatGetSize(Ao,&R,&C);
    ierr = PetscPrintf(PETSC_COMM_SELF,
		       "[%d] r: %d, c: %d, R: %d, C: %d\n",
		       rank,r,c,R,C);CHKERRQ(ierr);
    MatView(Ao,vw_s);
    PetscPrintf(PETSC_COMM_SELF,"\n");


    if (colmap) PetscIntView(n,colmap,vw_s);
    PetscPrintf(PETSC_COMM_SELF,"\n");
    PetscSynchronizedFlush(PETSC_COMM_SELF);
    PetscSequentialPhaseEnd(PETSC_COMM_WORLD,1);
    
  }

  ierr = PCRegisterDynamic("schur",0,"PCCreate_Schur",PCCreate_Schur);CHKERRQ(ierr);
  
  MatGetVecs(A,&x,&b);
  VecDuplicate(b,&res);

  KSPCreate(PETSC_COMM_WORLD,&ksp);
  KSPGetPC(ksp,&pc);
  KSPSetType(ksp,"fgmres");
  PCSetType(pc,"schur");
  KSPSetOperators(ksp,A,A,SAME_NONZERO_PATTERN);
  KSPSetFromOptions(ksp);

  KSPSetUp(ksp);

  for (i=0; i<solve; i++) {
    VecSet(x,zero);
    VecSet(b,one);
    KSPSetOperators(ksp,A,A,SAME_NONZERO_PATTERN);
    KSPSolve(ksp,b,x);
    MatMult(A,x,res);
    VecAYPX(res,-1,b);
    VecNorm(res,NORM_2,&norm);
    ierr = PetscPrintf(PETSC_COMM_WORLD,"||b-A*x||: %g\n",norm);CHKERRQ(ierr);
  } 

  for (i=0; i<solveT; i++) {
    VecSet(x,zero);
    VecSet(b,one);
    KSPSolveTranspose(ksp,b,x);
    MatMultTranspose(A,x,res);
    VecAYPX(res,-1,b);
    VecNorm(res,NORM_2,&norm);
    ierr = PetscPrintf(PETSC_COMM_WORLD,"||b-A'*x||: %g\n",norm);CHKERRQ(ierr);
  } 

  MatDestroy(A);
  VecDestroy(x);
  VecDestroy(b);
  VecDestroy(res);
  KSPDestroy(ksp);

  ierr = PetscFinalize();CHKERRQ(ierr);
  return 0;
}
