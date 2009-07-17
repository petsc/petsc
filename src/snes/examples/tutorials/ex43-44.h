/*
  Accelerates Newton's method by solving a small problem defined by those elements with large residual plus one level of overlap

  This is a toy code for playing around

  Counts residual entries as small if they are less then .2 times the maximum
  Decides to solve a reduced problem if the number of large entries is less than 20 percent of all entries (and this has been true for criteria_reduce iterations)
*/
#include "petscsnes.h"

extern PetscErrorCode FormFunctionSub(SNES,Vec,Vec,void*);
extern PetscErrorCode FormJacobianSub(SNES,Vec,Mat*,Mat*,MatStructure*,void*);


typedef struct {
  Vec        xwork,fwork;
  VecScatter scatter;
  SNES       snes;
  IS         is;
} SubCtx;

#undef __FUNCT__
#define __FUNCT__ "FormFunctionSub"
PetscErrorCode FormFunctionSub(SNES snes,Vec x,Vec f,void *ictx)
{
  PetscErrorCode ierr;
  SubCtx         *ctx = (SubCtx*) ictx;

  PetscFunctionBegin;
  ierr = VecScatterBegin(ctx->scatter,x,ctx->xwork,INSERT_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
  ierr = VecScatterEnd(ctx->scatter,x,ctx->xwork,INSERT_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
  ierr = SNESComputeFunction(ctx->snes,ctx->xwork,ctx->fwork);CHKERRQ(ierr);
  ierr = VecScatterBegin(ctx->scatter,ctx->fwork,f,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = VecScatterEnd(ctx->scatter,ctx->fwork,f,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "FormJaocbianSub"
PetscErrorCode FormJacobianSub(SNES snes,Vec x,Mat *A, Mat *B, MatStructure *str,void *ictx)
{
  PetscErrorCode ierr;
  SubCtx         *ctx = (SubCtx*) ictx;
  Mat            As,Bs;

  PetscFunctionBegin;
  ierr = VecScatterBegin(ctx->scatter,x,ctx->xwork,INSERT_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
  ierr = VecScatterEnd(ctx->scatter,x,ctx->xwork,INSERT_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
  ierr = SNESGetJacobian(ctx->snes,&As,&Bs,PETSC_NULL,PETSC_NULL);CHKERRQ(ierr);
  ierr = SNESComputeJacobian(ctx->snes,ctx->xwork,&As,&Bs,str);CHKERRQ(ierr);
  if (*B) {
    ierr = MatGetSubMatrix(Bs,ctx->is,ctx->is,MAT_REUSE_MATRIX,B);CHKERRQ(ierr);
  } else {
    ierr = MatGetSubMatrix(Bs,ctx->is,ctx->is,MAT_INITIAL_MATRIX,B);CHKERRQ(ierr);
  }
  if (!*A) {
    *A = *B;
    ierr = PetscObjectReference((PetscObject)*A);CHKERRQ(ierr);
  }
  if (*A != *B) {
    ierr = MatAssemblyBegin(*A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatAssemblyEnd(*A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SolveSubproblem"
PetscErrorCode SolveSubproblem(SNES snes)
{
  PetscErrorCode ierr;
  Vec            residual,solution;
  PetscReal      rmax;
  PetscInt       i,n,cnt,*indices;
  PetscScalar    *r;
  SNES           snessub;
  Vec            x,f;
  SubCtx         ctx;
  Mat            mat;

  PetscFunctionBegin;
  ctx.snes = snes;
  ierr = SNESGetSolution(snes,&solution);CHKERRQ(ierr);
  ierr = SNESGetFunction(snes,&residual,0,0);CHKERRQ(ierr);
  ierr = VecNorm(residual,NORM_INFINITY,&rmax);CHKERRQ(ierr);
  ierr = VecGetLocalSize(residual,&n);CHKERRQ(ierr);
  ierr = VecGetArray(residual,&r);CHKERRQ(ierr);
  cnt  = 0;
  for (i=0; i<n; i++) {
    if (PetscAbsScalar(r[i]) > .20*rmax ) cnt++;
  }
  ierr = PetscMalloc(cnt*sizeof(PetscInt),&indices);CHKERRQ(ierr);
  cnt  = 0;
  for (i=0; i<n; i++) {
    if (PetscAbsScalar(r[i]) > .20*rmax ) indices[cnt++] = i;
  }
  if (cnt > .2*n) PetscFunctionReturn(0);

  printf("number in subproblem %d\n",cnt);CHKERRQ(ierr);
  ierr = VecRestoreArray(residual,&r);CHKERRQ(ierr);
  ierr = ISCreateGeneral(PETSC_COMM_WORLD,cnt,indices,&ctx.is);CHKERRQ(ierr);
  ierr = PetscFree(indices);CHKERRQ(ierr);

  ierr = SNESGetJacobian(snes,0,&mat,0,0);CHKERRQ(ierr);
  ierr = MatIncreaseOverlap(mat,1,&ctx.is,2);CHKERRQ(ierr);
  ierr = ISSort(ctx.is);CHKERRQ(ierr);
  ierr = ISGetLocalSize(ctx.is,&cnt);CHKERRQ(ierr);
  printf("number in subproblem %d\n",cnt);CHKERRQ(ierr);
  ierr = VecCreate(PETSC_COMM_WORLD,&x);CHKERRQ(ierr);
  ierr = VecSetSizes(x,cnt,cnt);CHKERRQ(ierr);
  ierr = VecSetType(x,VECSEQ);CHKERRQ(ierr);
  ierr = VecDuplicate(x,&f);CHKERRQ(ierr);
  ierr = VecScatterCreate(solution,ctx.is,x,PETSC_NULL,&ctx.scatter);CHKERRQ(ierr);

  ierr = VecDuplicate(solution,&ctx.xwork);CHKERRQ(ierr);
  ierr = VecCopy(solution,ctx.xwork);CHKERRQ(ierr);
  ierr = VecDuplicate(residual,&ctx.fwork);CHKERRQ(ierr);
  ierr = VecScatterBegin(ctx.scatter,solution,x,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = VecScatterEnd(ctx.scatter,solution,x,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = SNESCreate(PETSC_COMM_WORLD,&snessub);CHKERRQ(ierr);
  ierr = SNESSetOptionsPrefix(snessub,"sub_");CHKERRQ(ierr);
  ierr = SNESSetFunction(snessub,f,FormFunctionSub,(void*)&ctx);CHKERRQ(ierr);
  ierr = SNESSetJacobian(snessub,0,0,FormJacobianSub,(void*)&ctx);CHKERRQ(ierr);
  ierr = SNESSetFromOptions(snessub);CHKERRQ(ierr);
  ierr = SNESSolve(snessub,PETSC_NULL,x);CHKERRQ(ierr);
  ierr = VecScatterBegin(ctx.scatter,x,solution,INSERT_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
  ierr = VecScatterEnd(ctx.scatter,x,solution,INSERT_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);

  ierr = ISDestroy(ctx.is);CHKERRQ(ierr);
  ierr = VecDestroy(ctx.xwork);CHKERRQ(ierr);
  ierr = VecDestroy(ctx.fwork);CHKERRQ(ierr);
  ierr = VecScatterDestroy(ctx.scatter);CHKERRQ(ierr);
  ierr = VecDestroy(x);CHKERRQ(ierr);
  ierr = VecDestroy(f);CHKERRQ(ierr);
  ierr = SNESDestroy(snessub);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


extern PetscErrorCode PETSCSNES_DLLEXPORT SNESMonitorRange_Private(SNES,PetscInt,PetscReal*);
static PetscInt CountGood = 0;

#undef __FUNCT__  
#define __FUNCT__ "MonitorRange"
PetscErrorCode PETSCSNES_DLLEXPORT MonitorRange(SNES snes,PetscInt it,PetscReal rnorm,void *dummy)
{
  PetscErrorCode          ierr;
  PetscReal               perc;

  ierr = SNESMonitorRange_Private(snes,it,&perc);CHKERRQ(ierr);
  if (perc < .20) CountGood++;
  else CountGood = 0;
  PetscFunctionReturn(0);
}
