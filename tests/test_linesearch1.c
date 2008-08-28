#include "taolinesearch.h"

typedef struct {
    int n;
    PetscScalar alpha;
} AppCtx;
PetscErrorCode viewme(Vec v); 

PetscErrorCode MyFuncGrad(TaoLineSearch ls, Vec X, PetscScalar *f, Vec G, void *ctx);

int main(int argc, char *argv[])
{
    TaoLineSearch ls;
    TaoLineSearchTerminationReason reason;
    Vec x,g,s;
    PetscScalar f,step;
    AppCtx user;
    int ierr;
    
	
    ierr = PetscInitialize(&argc, &argv,0,0); CHKERRQ(ierr);

    user.n=2; user.alpha = 99.0;
    ierr = VecCreate(PETSC_COMM_WORLD,&x); CHKERRQ(ierr);
    ierr = VecSetSizes(x,PETSC_DECIDE,2); CHKERRQ(ierr);
    ierr = VecSetFromOptions(x); CHKERRQ(ierr);

    ierr = VecDuplicate(x,&g); CHKERRQ(ierr);
    ierr = VecDuplicate(x,&s); CHKERRQ(ierr);

    ierr = VecSet(x,0.0); CHKERRQ(ierr);
    ierr = VecSet(s,0.0); CHKERRQ(ierr);
    f = 1.0;
    ierr = VecSetValue(g,0,-1.0, INSERT_VALUES); CHKERRQ(ierr);
    ierr = VecSetValue(g,1,0.0, INSERT_VALUES);CHKERRQ(ierr);
    ierr = VecAssemblyBegin(g); CHKERRQ(ierr);
    ierr = VecAssemblyEnd(g); CHKERRQ(ierr);
    ierr = VecAXPY(s,-1.0,g); CHKERRQ(ierr);

    ierr = TaoLineSearchCreate(PETSC_COMM_WORLD,&ls); CHKERRQ(ierr);
//    ierr = TaoLineSearchSetType(ls,"unit"); CHKERRQ(ierr);
    ierr = TaoLineSearchSetFromOptions(ls); CHKERRQ(ierr);
    ierr = TaoLineSearchSetObjectiveAndGradient(ls,MyFuncGrad,(void*)&user); 
    CHKERRQ(ierr);
    ierr = TaoLineSearchApply(ls,x,&f,g,s,&step,&reason); CHKERRQ(ierr);
    ierr = TaoLineSearchView(ls,PETSC_VIEWER_STDOUT_WORLD); CHKERRQ(ierr);

    ierr = PetscPrintf(PETSC_COMM_WORLD,"Status: %d\n",reason); CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Step length: %g\n",step); CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_WORLD,"New Obj value: %g\n",f); CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_WORLD,"New gradient:\n"); CHKERRQ(ierr);
    ierr = VecView(g,PETSC_VIEWER_STDOUT_WORLD); CHKERRQ(ierr);

    
    ierr = TaoLineSearchDestroy(ls); CHKERRQ(ierr);
    ierr = VecDestroy(x); CHKERRQ(ierr);
    ierr = VecDestroy(s); CHKERRQ(ierr);
    ierr = VecDestroy(g); CHKERRQ(ierr);
    ierr = PetscFinalize();
    return(0);
}


PetscErrorCode MyFuncGrad(TaoLineSearch ls, Vec X, PetscScalar *f, Vec G, void *ctx)
{
    AppCtx *user = (AppCtx*)ctx;
    PetscErrorCode    ierr;
    PetscInt i,nn=user->n/2;
    double ff=0,t1,t2,alpha=user->alpha;
    PetscScalar *x,*g;

    /* Get pointers to vector data */
    ierr = VecGetArray(X,&x); CHKERRQ(ierr);
    ierr = VecGetArray(G,&g); CHKERRQ(ierr);

    /* Compute G(X) */
    for (i=0; i<nn; i++){
	t1 = x[2*i+1]-x[2*i]*x[2*i]; t2= 1-x[2*i];
	ff += alpha*t1*t1 + t2*t2;
	g[2*i] = -4*alpha*t1*x[2*i]-2.0*t2;
	g[2*i+1] = 2*alpha*t1;
    }

    /* Restore vectors */
    ierr = VecRestoreArray(X,&x); CHKERRQ(ierr);
    ierr = VecRestoreArray(G,&g); CHKERRQ(ierr);
    *f=ff;

    ierr = PetscLogFlops(nn*15); CHKERRQ(ierr);
    return 0;
}

#undef __FUNCT__
#define __FUNCT__ "viewme"
PetscErrorCode viewme(Vec v) 
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(v,VEC_COOKIE,1);
  ierr = VecView(v,PETSC_VIEWER_STDOUT_SELF); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
