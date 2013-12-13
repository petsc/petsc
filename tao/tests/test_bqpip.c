#include "tao.h"
typedef struct {
  Vec b;
  Mat H;
}AppCtx;

PetscErrorCode formfg(TaoSolver,Vec,PetscReal*,Vec,void*);
PetscErrorCode formh(TaoSolver,Vec,Mat*,Mat*,MatStructure*,void*);



#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc,char **argv)
{
  PetscErrorCode ierr;
  Vec        x,xl,xu,xl2;
  PetscInt   testnumber=4;
  PetscReal h1[]={7.0041870379596727e+05,
		-8.3224463710973843e+05,
		-5.2666356047707784e+05,
		-8.3224463710973843e+05,
		9.9981119327620871e+05,
		6.4966052261797199e+05,
		-5.2666356047707784e+05,
		6.4966052261797199e+05,
		5.4293716095589777e+05};
  PetscReal h2[]={1.5912330965418366e+06,
		  -2.4858764503055345e+06,
		  -1.8359886998944925e+05,
		  -2.4858764503055345e+06,
		  3.9642193845191249e+06,
		  3.0813954151966132e+05,
		  -1.8359886998944925e+05,
		  3.0813954151966132e+05,
		  3.3439154547860053e+04};
  PetscReal h3[]={3.6417056765638507e+05,
		  -6.1515583991605288e+05,
		  -4.7143914245777630e+05,
		  -6.1515583991605288e+05,
		  1.0698635183323724e+06,
		  8.6368774543820019e+05,
		  -4.7143914245777630e+05,
		  8.6368774543820019e+05,
		  7.7510985588455945e+05};
  PetscReal h4[]={-1.44339e+04,
		  1.54854e+04,
		  1.54854e+04,
		  -1.60218e+04};
  PetscReal b1[]={-33535.5,40755.2,19244.7};
  PetscReal b2[]={-50344.8,78886.6,3994.66};
  PetscReal b3[]={-24323.8,40608.4,24496.6};
  PetscReal b4[]={-471668.0,235620.0};
  PetscReal u1 = 0.015;
  PetscReal u2 = 0.05;
  PetscReal u3 = 0.025;
  PetscReal u4= 0.4;
  PetscReal l1 = -1e20;
  PetscReal l2 = -0.05;
  PetscReal l3 = -0.025;
  PetscReal l4 = -0.4;
  TaoSolver  tao;     
  KSP ksp;
  PC pc;
  AppCtx     user; 

  /* Initialize TAO and PETSc */
  TaoInitialize(&argc,&argv,(char*)0,(char*)0);
  ierr = PetscOptionsGetInt(PETSC_NULL,"-test",&testnumber,PETSC_NULL);
  if (testnumber == 4) {
    ierr = VecCreateSeq(PETSC_COMM_SELF,2,&x); CHKERRQ(ierr);
  } else {
    ierr = VecCreateSeq(PETSC_COMM_SELF,3,&x); CHKERRQ(ierr);
  }
  ierr = VecDuplicate(x,&xl); CHKERRQ(ierr);
  ierr = VecDuplicate(x,&xl2); CHKERRQ(ierr);
  ierr = VecDuplicate(x,&xu); CHKERRQ(ierr);
  ierr = VecSet(x,0); CHKERRQ(ierr);

  if (testnumber == 1) {
    ierr = VecCreateSeqWithArray(PETSC_COMM_SELF,3,b1,&user.b); 
    CHKERRQ(ierr);
    ierr = MatCreateSeqDense(PETSC_COMM_SELF,3,3,h1,&user.H);
    CHKERRQ(ierr);
    ierr = VecSet(xl,l1); CHKERRQ(ierr);
    ierr = VecSet(xu,u1); CHKERRQ(ierr);
  } else if (testnumber == 2) {
    ierr = VecCreateSeqWithArray(PETSC_COMM_SELF,3,b2,&user.b);
    CHKERRQ(ierr);
    ierr = MatCreateSeqDense(PETSC_COMM_SELF,3,3,h2,&user.H);
    CHKERRQ(ierr);
    ierr = VecSet(xl,l2); CHKERRQ(ierr);
    ierr = VecSet(xu,u2); CHKERRQ(ierr);

  } else if (testnumber == 3) {
    ierr = VecCreateSeqWithArray(PETSC_COMM_SELF,3,b3,&user.b);
    CHKERRQ(ierr);
    ierr = MatCreateSeqDense(PETSC_COMM_SELF,3,3,h3,&user.H);
    CHKERRQ(ierr);
    ierr = VecSet(xl,l3); CHKERRQ(ierr);
    ierr = VecSet(xu,u3); CHKERRQ(ierr);
  } else if (testnumber == 4) {
    ierr = VecCreateSeqWithArray(PETSC_COMM_SELF,2,b4,&user.b);
    CHKERRQ(ierr);
    ierr = MatCreateSeqDense(PETSC_COMM_SELF,2,2,h4,&user.H);
    CHKERRQ(ierr);
    ierr = VecSet(xl,l4); CHKERRQ(ierr);
    ierr = VecSet(xu,u4); CHKERRQ(ierr);
  }

  
  ierr = TaoCreate(PETSC_COMM_SELF,&tao); CHKERRQ(ierr);
  ierr = TaoSetType(tao,"tao_bqpip"); CHKERRQ(ierr);
  ierr = TaoSetInitialVector(tao,x); CHKERRQ(ierr);
  ierr = TaoSetVariableBounds(tao,xl,xu); CHKERRQ(ierr);
  ierr = TaoSetObjectiveAndGradientRoutine(tao,formfg,&user); CHKERRQ(ierr);
  ierr = TaoSetHessianRoutine(tao,user.H,user.H,formh,&user); CHKERRQ(ierr);
  ierr = TaoSetFromOptions(tao); CHKERRQ(ierr);
  ierr = TaoGetKSP(tao,&ksp); CHKERRQ(ierr);
  if (ksp) {
    ierr = KSPGetPC(ksp,&pc); CHKERRQ(ierr);
    ierr = PCSetType(pc,PCNONE); CHKERRQ(ierr);
  }
  ierr = TaoSolve(tao); CHKERRQ(ierr);
  ierr = VecView(x,0); CHKERRQ(ierr);


  /* solve 2nd time */
  ierr = VecCopy(xl,xl2); CHKERRQ(ierr);
  ierr = VecDestroy(&xl); CHKERRQ(ierr);
  ierr = TaoResetStatistics(tao); CHKERRQ(ierr);
  ierr = VecSet(x,0); CHKERRQ(ierr);
  ierr = TaoSetInitialVector(tao,x); CHKERRQ(ierr);
  ierr = TaoSetVariableBounds(tao,xl2,xu); CHKERRQ(ierr);
  ierr = TaoSolve(tao); CHKERRQ(ierr);

  ierr = MatDestroy(&user.H); CHKERRQ(ierr);
  ierr = VecDestroy(&user.b); CHKERRQ(ierr);
  ierr = VecDestroy(&x); CHKERRQ(ierr);
  ierr = VecDestroy(&xl2); CHKERRQ(ierr);
  ierr = VecDestroy(&xu); CHKERRQ(ierr);
  ierr = TaoDestroy(&tao); CHKERRQ(ierr);
  TaoFinalize();
  return 0;
}


#undef __FUNCT__ 
#define __FUNCT__ "formh"
PetscErrorCode formh(TaoSolver tao, Vec v, Mat *H, Mat *Hpre, MatStructure *flag, void *ctx)
{
  PetscFunctionBegin;
  *flag = SAME_NONZERO_PATTERN;
  PetscFunctionReturn(0);
}
#undef __FUNCT__ 
#define __FUNCT__ "formfg"
PetscErrorCode  formfg(TaoSolver subtao, Vec x, PetscReal *f, Vec g, void *ctx)
{
  AppCtx *user = (AppCtx*)ctx;
  PetscReal d1,d2;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  /* g = A*x  (add b later)*/
  ierr = MatMult(user->H,x,g); CHKERRQ(ierr);


  /* f = 1/2 * x'*(Ax) + b'*x  */
  ierr = VecDot(x,g,&d1); CHKERRQ(ierr);
  ierr = VecDot(user->b,x,&d2); CHKERRQ(ierr);
  *f = 0.5 *d1 + d2;

  /* now  g = g + b */
  ierr = VecAXPY(g, 1.0, user->b); CHKERRQ(ierr);

  PetscFunctionReturn(0);
}
