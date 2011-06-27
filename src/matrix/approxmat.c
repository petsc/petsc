#include "approxmat.h"   /*I "approxmat.h" */




#undef __FUNCT__
#define __FUNCT__ "MatCreateAPPROX"
/*@C
  MatCreateAPPROX - Creates a limited memory matrix BFGS matrix for multiplying -- not solving

  Collective on A

  Input Parameters:
  . comm -- MPI Communicator
  . n -- local size of vectors
  . N -- global size of vectors
  Output Parameters:
  . A - New Approx matrix

  Level: developer

@*/
extern PetscErrorCode MatCreateAPPROX(MPI_Comm comm, PetscInt n, PetscInt N, Mat *A)
{
  MatApproxCtx *ctx;
  PetscErrorCode info;
  PetscInt nhistory;

  PetscFunctionBegin;

  // create data structure and populate with default values
  info = PetscMalloc(sizeof(MatApproxCtx),(void**)&ctx); CHKERRQ(info);
  ctx->lm=5;


  info = PetscMalloc((ctx->lm+1)*sizeof(PetscReal),(void**)&ctx->rho); 
  CHKERRQ(info);
  nhistory = 1;

  // Finish initializations
  ctx->lmnow = 0;
  ctx->iter = 0;
  ctx->nupdates = 0;
  ctx->nrejects = 0;

  ctx->Gprev = 0;
  ctx->Xprev = 0;

    
  info = MatCreateShell(comm, n, n, N, N, ctx, A); CHKERRQ(info);
  info = MatShellSetOperation(*A,MATOP_DESTROY,(void(*)(void))MatDestroy_APPROX);
  CHKERRQ(info);
  info = MatShellSetOperation(*A,MATOP_VIEW,(void(*)(void))MatView_APPROX);
  CHKERRQ(info);
  info = MatShellSetOperation(*A,MATOP_MULT,(void(*)(void))MatMult_APPROX);
  CHKERRQ(info);

  PetscFunctionReturn(0);
}



  
  
#undef __FUNCT__
#define __FUNCT__ "MatMult_APPROX"
extern PetscErrorCode MatMult_APPROX(Mat A, Vec x, Vec y) 
{
    PetscReal      ytx, bstx, bsts;
    PetscInt       i;
    MatApproxCtx     *shell;
    PetscErrorCode info;

    PetscFunctionBegin;
    PetscValidHeaderSpecific(A,MAT_CLASSID,1);
    PetscValidHeaderSpecific(x,VEC_CLASSID,2);
    PetscValidHeaderSpecific(y,VEC_CLASSID,3);
    info = MatShellGetContext(A,(void**)&shell); CHKERRQ(info);
    if (shell->lmnow < 1) {
	shell->rho[0] = 1.0;
    }

    info = VecCopy(x,y); CHKERRQ(info);
    for (i = 0; i < shell->lmnow; ++i) {
      info = VecDot(x,shell->Y[i],&ytx); CHKERRQ(info);
      info = VecDot(x,shell->Bs[i],&bstx); CHKERRQ(info);
      info = VecDot(shell->S[i],shell->Bs[i],&bsts); CHKERRQ(info);
      info = VecAXPY(y,shell->rho[i]*ytx,shell->Y[i]); CHKERRQ(info);
      info = VecAXPY(y,-bstx/bsts,shell->Bs[i]); CHKERRQ(info);
    }
    PetscFunctionReturn(0);
}



  
#undef __FUNCT__
#define __FUNCT__ "MatView_APPROX"
extern PetscErrorCode MatView_APPROX(Mat A, PetscViewer pv)
{
    PetscBool isascii;
    PetscErrorCode info;
    MatApproxCtx *lmP;
    PetscFunctionBegin;

    info = MatShellGetContext(A,(void**)&lmP); CHKERRQ(info);

    info = PetscTypeCompare((PetscObject)pv,PETSCVIEWERASCII,&isascii); CHKERRQ(info);
    if (isascii) {
	info = PetscViewerASCIIPrintf(pv,"Approx Matrix\n"); CHKERRQ(info);
	info = PetscViewerASCIIPrintf(pv," Number of vectors: %d\n",lmP->lm); CHKERRQ(info);
	info = PetscViewerASCIIPrintf(pv," updates: %d\n",lmP->nupdates); CHKERRQ(info);
	info = PetscViewerASCIIPrintf(pv," rejects: %d\n",lmP->nrejects); CHKERRQ(info);
	
    }
    else {
	SETERRQ(PETSC_COMM_SELF,1,"non-ascii viewers not implemented for MatApprox\n");
    }

    PetscFunctionReturn(0);
    
}

#undef __FUNCT__
#define __FUNCT__ "MatDestroy_APPROX"
extern PetscErrorCode MatDestroy_APPROX(Mat M)
{
    MatApproxCtx     *ctx;
    PetscErrorCode ierr;
    PetscFunctionBegin;

    ierr = MatShellGetContext(M,(void**)&ctx); CHKERRQ(ierr);
    if (ctx->allocated) {
      ierr = PetscObjectDereference((PetscObject)ctx->Xprev); CHKERRQ(ierr);
      ierr = PetscObjectDereference((PetscObject)ctx->Gprev); CHKERRQ(ierr);

      ierr = VecDestroyVecs(ctx->lm+1,&ctx->S); CHKERRQ(ierr);
      ierr = VecDestroyVecs(ctx->lm+1,&ctx->Y); CHKERRQ(ierr);
      ierr = VecDestroyVecs(ctx->lm+1,&ctx->Bs); CHKERRQ(ierr);
      ierr = VecDestroy(&ctx->W); CHKERRQ(ierr);
    }
    ierr = PetscFree(ctx->rho); CHKERRQ(ierr);
    ierr = PetscFree(ctx); CHKERRQ(ierr);

    PetscFunctionReturn(0);
	
}



#undef __FUNCT__
#define __FUNCT__ "MatApproxReset"
extern PetscErrorCode MatApproxReset(Mat M)
{
    PetscErrorCode ierr;
    MatApproxCtx *ctx;
    PetscInt i;
    PetscFunctionBegin;
    ierr = MatShellGetContext(M,(void**)&ctx); CHKERRQ(ierr);
    ctx->Gprev = ctx->Y[ctx->lm];
    ctx->Xprev = ctx->S[ctx->lm];
    ierr = PetscObjectReference((PetscObject)ctx->Gprev);
    ierr = PetscObjectReference((PetscObject)ctx->Xprev);
    for (i=0; i<ctx->lm; ++i) {
      ctx->rho[i] = 0.0;
    }
    ctx->rho[0] = 1.0;
    

    ctx->iter=0;
    ctx->nupdates=0;
    ctx->lmnow=0;

    
    PetscFunctionReturn(0);
}



#undef __FUNCT__
#define __FUNCT__ "MatApproxUpdate"
extern PetscErrorCode MatApproxUpdate(Mat M, Vec x, Vec g)
{
  MatApproxCtx *ctx;
  PetscReal rhotemp, rhotol;
  PetscReal y0temp;
  PetscErrorCode ierr;
  PetscInt i;
  PetscBool same;

  PetscFunctionBegin;

  PetscValidHeaderSpecific(x,VEC_CLASSID,2); 
  PetscValidHeaderSpecific(g,VEC_CLASSID,3);
  ierr = PetscTypeCompare((PetscObject)M,MATSHELL,&same); CHKERRQ(ierr);
  if (!same) {SETERRQ(PETSC_COMM_SELF,1,"Matrix M is not type MatApprox");}
  ierr = MatShellGetContext(M,(void**)&ctx); CHKERRQ(ierr);
  if (!ctx->allocated) {
      ierr = MatApproxAllocateVectors(M, x);  CHKERRQ(ierr);
  }
  
  if (0 == ctx->iter) {
    ierr = MatApproxReset(M); CHKERRQ(ierr);
  } 
  else {
    ierr = VecAYPX(ctx->Gprev,-1.0,g); CHKERRQ(ierr);
    ierr = VecAYPX(ctx->Xprev,-1.0,x); CHKERRQ(ierr);
    ierr = VecDot(ctx->Gprev,ctx->Xprev,&rhotemp); CHKERRQ(ierr);
    ierr = VecDot(ctx->Gprev,ctx->Gprev,&y0temp); CHKERRQ(ierr);
    rhotol = 0;
    if (rhotemp > rhotol) {
      ierr = MatMult(M,ctx->Xprev,ctx->W); CHKERRQ(ierr);
      ++ctx->nupdates;

      ctx->lmnow = PetscMin(ctx->lmnow+1, ctx->lm);
      ierr=PetscObjectDereference((PetscObject)ctx->S[ctx->lm]); CHKERRQ(ierr);
      ierr=PetscObjectDereference((PetscObject)ctx->Y[ctx->lm]); CHKERRQ(ierr);
      for (i = ctx->lm-1; i >= 0; --i) {
	ctx->S[i+1] = ctx->S[i];
	ctx->Y[i+1] = ctx->Y[i];
	ierr = VecCopy(ctx->Bs[i],ctx->Bs[i+1]); CHKERRQ(ierr);
	ctx->rho[i+1] = ctx->rho[i];
      }
      ctx->S[0] = ctx->Xprev;
      ctx->Y[0] = ctx->Gprev;
      ierr = VecCopy(ctx->W, ctx->Bs[0]); CHKERRQ(ierr);

      PetscObjectReference((PetscObject)ctx->S[0]);
      PetscObjectReference((PetscObject)ctx->Y[0]);
      PetscObjectReference((PetscObject)ctx->Bs[0]);
      ctx->rho[0] = 1.0 / rhotemp;

      ierr = PetscObjectDereference((PetscObject)ctx->Xprev); CHKERRQ(ierr);
      ierr = PetscObjectDereference((PetscObject)ctx->Gprev); CHKERRQ(ierr);
      ctx->Xprev = ctx->S[ctx->lm]; 
      ctx->Gprev = ctx->Y[ctx->lm];

      ierr = PetscObjectReference((PetscObject)ctx->S[ctx->lm]); CHKERRQ(ierr);
      ierr = PetscObjectReference((PetscObject)ctx->Y[ctx->lm]); CHKERRQ(ierr);


    } 
    else { 
      ++ctx->nrejects;
    }
  }
  
  ++ctx->iter;
  ierr = VecCopy(x, ctx->Xprev); CHKERRQ(ierr);
  ierr = VecCopy(g, ctx->Gprev); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "MatApproxAllocateVectors"
extern PetscErrorCode MatApproxAllocateVectors(Mat m, Vec v)
{
    PetscErrorCode ierr;
    MatApproxCtx *ctx;
    PetscBool same;

    PetscFunctionBegin;
    PetscValidHeaderSpecific(m,MAT_CLASSID,1);
    PetscValidHeaderSpecific(v,VEC_CLASSID,2);
    ierr = PetscTypeCompare((PetscObject)m,MATSHELL,&same); CHKERRQ(ierr);
    if (!same) {
	SETERRQ(PETSC_COMM_SELF,1,"Matrix m is not type MatApprox");
    }
    ierr = MatShellGetContext(m,(void**)&ctx); CHKERRQ(ierr);
    

    // Perform allocations
    ierr = VecDuplicateVecs(v,ctx->lm+1,&ctx->S); CHKERRQ(ierr);
    ierr = VecDuplicateVecs(v,ctx->lm+1,&ctx->Y); CHKERRQ(ierr);
    ierr = VecDuplicateVecs(v,ctx->lm+1,&ctx->Bs); CHKERRQ(ierr);
    ierr = VecDuplicate(v,&ctx->W); CHKERRQ(ierr);
    ctx->allocated = PETSC_TRUE;

    PetscFunctionReturn(0);
}


