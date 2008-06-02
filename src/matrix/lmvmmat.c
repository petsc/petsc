#include "lmvmmat.h"   /*I "lmvmmat.h" */


/* These lists are used for setting options */
static const char *Scale_Table[64] = {
    "none","scalar","broyden"
};

static const char *Rescale_Table[64] = {
    "none","scalar","gl"
};

static const char *Limit_Table[64] = {
    "none","average","relative","absolute"
};


#undef __FUNCT__
#define __FUNCT__ "MatCreateLMVM"
/*@C
  MatCreateLMVM - Creates a limited memory matrix for lmvm algorithms.

  Collective on v

  Input Parameters:
  . comm -- MPI Communicator
  . N -- size of Hessian matrix being approximated
  Output Parameters:
  . A - New LMVM matrix

  Level: developer

@*/
EXTERN PetscErrorCode MatCreateLMVM(MPI_Comm comm, PetscInt n, PetscInt N, Mat *A)
{
    MatLMVMCtx *ctx;
    PetscErrorCode info;
    PetscInt nhistory;

    PetscFunctionBegin;

    // create data structure and populate with default values
    info = PetscMalloc(sizeof(MatLMVMCtx),(void**)&ctx); CHKERRQ(info);
    ctx->lm=5;
    ctx->eps=0.0;
    ctx->limitType=MatLMVM_Limit_None;
    ctx->rScaleType = MatLMVM_Rescale_Scalar;
    ctx->s_alpha = 1.0;
    ctx->r_alpha = 1.0;
    ctx->r_beta = 0.5;
    ctx->mu = 1.0;
    ctx->nu = 100.0;

    ctx->phi = 0.125;		

    ctx->scalar_history = 1;
    ctx->rescale_history = 1;

    ctx->delta_min = 1e-7;
    ctx->delta_max = 100.0;

    // Begin configuration
    PetscOptionsInt("-tao_lmm_vectors", "vectors to use for approximation", "", ctx->lm, &ctx->lm, 0);
    PetscOptionsReal("-tao_lmm_limit_mu", "mu limiting factor", "", ctx->mu, &ctx->mu, 0);
    PetscOptionsReal("-tao_lmm_limit_nu", "nu limiting factor", "", ctx->nu, &ctx->nu, 0);
    PetscOptionsReal("-tao_lmm_broyden_phi", "phi factor for Broyden scaling", "", ctx->phi, &ctx->phi, 0);
    PetscOptionsReal("-tao_lmm_scalar_alpha", "alpha factor for scalar scaling", "",ctx->s_alpha, &ctx->s_alpha, 0);
    PetscOptionsReal("-tao_lmm_rescale_alpha", "alpha factor for rescaling diagonal", "", ctx->r_alpha, &ctx->r_alpha, 0);
    PetscOptionsReal("-tao_lmm_rescale_beta", "beta factor for rescaling diagonal", "", ctx->r_beta, &ctx->r_beta, 0);
    PetscOptionsInt("-tao_lmm_scalar_history", "amount of history for scalar scaling", "", ctx->scalar_history, &ctx->scalar_history, 0);
    PetscOptionsInt("-tao_lmm_rescale_history", "amount of history for rescaling diagonal", "", ctx->rescale_history, &ctx->rescale_history, 0);
    PetscOptionsReal("-tao_lmm_eps", "rejection tolerance", "", ctx->eps, &ctx->eps, 0);
    PetscOptionsEList("-tao_lmm_scale_type", "scale type", "", Scale_Table, MatLMVM_Scale_Types, Scale_Table[ctx->scaleType], &ctx->scaleType, 0);
    PetscOptionsEList("-tao_lmm_rescale_type", "rescale type", "", Rescale_Table, MatLMVM_Rescale_Types, Rescale_Table[ctx->rScaleType], &ctx->rScaleType, 0);
    PetscOptionsEList("-tao_lmm_limit_type", "limit type", "", Limit_Table, MatLMVM_Limit_Types, Limit_Table[ctx->limitType], &ctx->limitType, 0);
    PetscOptionsReal("-tao_lmm_delta_min", "minimum delta value", "", ctx->delta_min, &ctx->delta_min, 0);
    PetscOptionsReal("-tao_lmm_delta_max", "maximum delta value", "", ctx->delta_max, &ctx->delta_max, 0);

    // Complete configuration
    ctx->rescale_history = PetscMin(ctx->rescale_history, ctx->lm);


    info = PetscMalloc((ctx->lm+1)*sizeof(PetscReal),(void**)&ctx->rho); 
                       CHKERRQ(info);
    info = PetscMalloc((ctx->lm+1)*sizeof(PetscReal),(void**)&ctx->beta); 
                       CHKERRQ(info);

    nhistory = PetscMax(ctx->scalar_history,1);
    info = PetscMalloc(nhistory*sizeof(PetscReal),(void**)&ctx->yy_history); 
                       CHKERRQ(info);
    info = PetscMalloc(nhistory*sizeof(PetscReal),(void**)&ctx->ys_history);
                       CHKERRQ(info);
    info = PetscMalloc(nhistory*sizeof(PetscReal),(void**)&ctx->ss_history); 
                       CHKERRQ(info);

    nhistory = PetscMax(ctx->rescale_history,1);
    info = PetscMalloc(nhistory*sizeof(PetscReal),(void**)&ctx->yy_rhistory);
                       CHKERRQ(info);
    info = PetscMalloc(nhistory*sizeof(PetscReal),(void**)&ctx->ys_rhistory);
                       CHKERRQ(info);
    info = PetscMalloc(nhistory*sizeof(PetscReal),(void**)&ctx->ss_rhistory); 
                       CHKERRQ(info);


    // Finish initializations
    ctx->lmnow = 0;
    ctx->iter = 0;
    ctx->updates = 0;
    ctx->rejects = 0;
    ctx->delta = 1.0;

    ctx->Gprev = 0;
    ctx->Xprev = 0;

    ctx->scale = 0;

    ctx->H0 = 0;
    ctx->useDefaultH0=PETSC_TRUE;
    
    info = MatCreateShell(comm, n, n, N, N, ctx, A); CHKERRQ(info);
    info = MatShellSetOperation(*A,MATOP_SOLVE,(void(*)(void))MatSolve_LMVM);
    CHKERRQ(info);
    info = MatShellSetOperation(*A,MATOP_DESTROY,(void(*)(void))MatDestroy_LMVM);
    CHKERRQ(info);
    info = MatShellSetOperation(*A,MATOP_VIEW,(void(*)(void))MatView_LMVM);
    
    PetscFunctionReturn(0);
}


  
  
#undef __FUNCT__
#define __FUNCT__ "MatSolve_LMVM"
EXTERN PetscErrorCode MatSolve_LMVM(Mat A, Vec b, Vec x) 
{
    PetscReal      sq, yq, dd;
    PetscInt       ll;
    PetscTruth     scaled;
    MatLMVMCtx     *shell;
    PetscErrorCode info;

    PetscFunctionBegin;
    PetscValidHeaderSpecific(A,MAT_COOKIE,1);
    PetscValidHeaderSpecific(b,VEC_COOKIE,2);
    PetscValidHeaderSpecific(x,VEC_COOKIE,3);
    info = MatShellGetContext(A,(void**)&shell); CHKERRQ(info);
    if (shell->lmnow < 1) {
	shell->rho[0] = 1.0;
    }

    info = VecCopy(b,x); CHKERRQ(info);
    for (ll = 0; ll < shell->lmnow; ++ll) {
	info = VecDot(x,shell->S[ll],&sq); CHKERRQ(info);
	shell->beta[ll] = sq * shell->rho[ll];
	info = VecAXPY(x,-shell->beta[ll],shell->Y[ll]); CHKERRQ(info);
    }

    scaled = PETSC_FALSE;
    if (!scaled && shell->useDefaultH0) {
	info = MatSolve(shell->H0,x,shell->U); CHKERRQ(info);
	info = VecDot(x,shell->U,&dd); CHKERRQ(info);
	if ((dd > 0.0) && !TaoInfOrNaN(dd)) {
	    // Accept Hessian solve
	    info = VecCopy(shell->U,x); CHKERRQ(info);
	    scaled = PETSC_TRUE;
	}
    }

    if (!scaled && shell->useScale) {
	info = VecPointwiseMult(shell->U,x,shell->scale); CHKERRQ(info);
	info = VecDot(x,shell->U,&dd); CHKERRQ(info);
	if ((dd > 0.0) && !TaoInfOrNaN(dd)) {
	    // Accept scaling
	    info = VecCopy(shell->U,x); CHKERRQ(info);	
	    scaled = PETSC_TRUE;
	}
    }
  
    if (!scaled) {
	switch(shell->scaleType) {
	    case MatLMVM_Scale_None:
		break;

	    case MatLMVM_Scale_Scalar:
		info = VecScale(x,shell->sigma); CHKERRQ(info);
		break;
  
	    case MatLMVM_Scale_Broyden:
		info = VecPointwiseMult(x,x,shell->D); CHKERRQ(info);
		break;
	}
    } 

    for (ll = shell->lmnow-1; ll >= 0; --ll) {
	info = VecDot(x,shell->Y[ll],&yq); CHKERRQ(info);
	info = VecAXPY(x,shell->beta[ll]-yq*shell->rho[ll],shell->S[ll]);
	CHKERRQ(info);
    }
    PetscFunctionReturn(0);
}

  
#undef __FUNCT__
#define __FUNCT__ "MatView_LMVM"
EXTERN PetscErrorCode MatView_LMVM(Mat A, PetscViewer pv)
{
    PetscTruth isascii;
    PetscErrorCode info;
    PetscFunctionBegin;

    info = PetscTypeCompare((PetscObject)pv,PETSC_VIEWER_ASCII,&isascii); CHKERRQ(info);
    if (isascii) {
	info = PetscViewerASCIIPrintf(pv,"View matrix not implemented.\n");
	CHKERRQ(info);
    }
    else {
	SETERRQ(1,"non-ascii viewers not implemented yet for MatLMVM\n");
    }

    PetscFunctionReturn(0);
    
}

#undef __FUNCT__
#define __FUNCT__ "MatDestroy_LMVM"
EXTERN PetscErrorCode MatDestroy_LMVM(Mat A)
{
    MatLMVMCtx     *ctx;
    PetscErrorCode info;
    PetscFunctionBegin;
    info = MatShellGetContext(A,(void**)&ctx); CHKERRQ(info);
    info = PetscFree(ctx); CHKERRQ(info);
    PetscFunctionReturn(0);
	
}


#undef __FUNCT__
#define __FUNCT__ "MatLMVMReset"
EXTERN PetscErrorCode MatLMVMReset()
{
    PetscFunctionBegin;
    PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "MatLMVMUpdate"
EXTERN PetscErrorCode MatLMVMUpdate(Vec x, Vec g)
{
    PetscFunctionBegin;
    PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatLMVMSetDelta"
EXTERN PetscErrorCode MatLMVMSetDelta(PetscReal d)
{
    PetscFunctionBegin;
    PetscFunctionReturn(0);
    
}

#undef __FUNCT__
#define __FUNCT__ "MatLMVMSetScale"
EXTERN PetscErrorCode MatLMVMSetScale(Vec s)
{
    PetscFunctionBegin;
    PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatLMVMGetRejects"
EXTERN PetscErrorCode MatLMVMGetRejects()
{
    PetscFunctionBegin;
    PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatLMVMSetH0"
EXTERN PetscErrorCode MatLMVMSetH0(Mat A)
{
    PetscFunctionBegin;
    PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatLMVMGetX0"
EXTERN PetscErrorCode MatLMVMGetX0(Vec x)
{
    PetscFunctionBegin;
    PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatLMVMRefine"
EXTERN PetscErrorCode MatLMVMRefine(Mat coarse, Mat op, Vec fineX, Vec fineG)
{
    PetscErrorCode info;
    PetscTruth same;
    PetscFunctionBegin;
    PetscValidHeaderSpecific(coarse,MAT_COOKIE,1);
    PetscValidHeaderSpecific(op,MAT_COOKIE,2);
    PetscValidHeaderSpecific(fineX,VEC_COOKIE,3);
    PetscValidHeaderSpecific(fineG,VEC_COOKIE,4);
    info = PetscTypeCompare((PetscObject)coarse,MATSHELL,&same); CHKERRQ(info);
    if (!same) {
	SETERRQ(1,"Matrix m is not type MatLMVM");
    }
    info = PetscTypeCompare((PetscObject)op,MATSHELL,&same); CHKERRQ(info);
    if (!same) {
	SETERRQ(1,"Matrix m is not type MatLMVM");
    }
    PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatLMVMAllocateVectors"
EXTERN PetscErrorCode MatLMVMAllocateVectors(Mat m, Vec v)
{
    PetscErrorCode info;
    MatLMVMCtx *ctx;
    PetscTruth same;

    PetscFunctionBegin;
    PetscValidHeaderSpecific(m,MAT_COOKIE,1);
    PetscValidHeaderSpecific(v,VEC_COOKIE,2);
    info = PetscTypeCompare((PetscObject)m,MATSHELL,&same); CHKERRQ(info);
    if (!same) {
	SETERRQ(1,"Matrix m is not type MatLMVM");
    }
    

    // Perform allocations
    
    info = VecDuplicateVecs(v,ctx->lm+1,&ctx->S); CHKERRQ(info);
    info = VecDuplicateVecs(v,ctx->lm+1,&ctx->Y); CHKERRQ(info);
    info = VecDuplicate(v,&ctx->D); CHKERRQ(info);
    info = VecDuplicate(v,&ctx->U); CHKERRQ(info);
    info = VecDuplicate(v,&ctx->V); CHKERRQ(info);
    info = VecDuplicate(v,&ctx->W); CHKERRQ(info);
    info = VecDuplicate(v,&ctx->P); CHKERRQ(info);
    info = VecDuplicate(v,&ctx->Q); CHKERRQ(info);

    PetscFunctionReturn(0);
}
