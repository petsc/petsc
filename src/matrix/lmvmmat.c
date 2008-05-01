#include "lmvmmat.h"   /*I "lmvmmat.h" */

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
EXTERN PetscErrorCode MatCreateLMVM(MPI_Comm comm, PetscInt N, Mat *A)
{
    MatLMVMCtx ctx;
    PetscErrorCode info;

    PetscFunctionBegin;
    
    info = MatCreateShell(comm, N, N, &ctx, &B); CHKERRQ(info);
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
	info = VecDot(x,S[ll],&sq); CHKERRQ(info);
	beta[ll] = sq * rho[ll];
	info = VecAxpy(x,-shell->beta[ll],shell->Y[ll]); CHKERRQ(info);
    }

    shell->scaled = PETSC_FALSE;
    if (!sheoll->scaled && shell->useDefaultH0) {
	info = MatSolve(H0,x,shell->U); CHKERRQ(info);
	info = VecDot(x,shell->U,&dd); CHKERRQ(info);
	if ((dd > 0.0) && !TaoInfOrNaN(dd)) {
	    // Accept Hessian solve
	    info = VecCopy(shell->U,x); CHKERRQ(info);
	    shell->scaled = PETSC_TRUE;
	}
    }

    if (!shell->scaled && shell->useScale) {
	invo = VecPointwiseMult(shell->U,dx,shell->scale); CHKERRQ(info);
	info = VecDot(x,shell->U,&dd); CHKERRQ(info);
	if ((dd > 0.0) && !TaoInfOrNaN(dd)) {
	    // Accept scaling
	    info = VecCopy(shell->U,x); CHKERRQ(info);	
	    shell->scaled = TAO_TRUE;
	}
    }
  
    if (!shell->scaled) {
	switch(shell->scaleType) {
	    case LMVMMat_Scale_None:
		break;

	    case LMVMMat_Scale_Scalar:
		info = VecScale(x,shell->sigma); CHKERRQ(info);
		break;
  
	    case LMVMMat_Scale_Broyden:
		info = VecPointwiseMult(x,x,D); CHKERRQ(info);
		break;
	}
    } 

    for (ll = shell->lmnow-1; ll >= 0; --ll) {
	info = VecDot(x,shell->Y[ll],&yq); CHKERRQ(info);
	info = VecAxpy(x,shell->beta[ll]-yq*shell->rho[ll],shell->S[ll]);
	CHKERRQ(info);
    }
    TaoFunctionReturn(0);
}

  
#undef __FUNCT__
#define __FUNCT__ "MatView_LMVM"
EXTERN PetscErrorCode MatView_LMVM(Mat A, PetscViewer pv)
{
    
}

#undef __FUNCT__
#define __FUNCT__ "MatDestroy_LMVM"
EXTERN PetscErrorCode MatDestroy_LMVM(Mat A)
{
    
}


#undef __FUNCT__
#define __FUNCT__ "MatLMVMReset"
EXTERN PetscErrorCode MatLMVMReset()
{
}


#undef __FUNCT__
#define __FUNCT__ "MatLMVMUpdate"
EXTERN PetscErrorCode MatLMVMUpdate(Vec x, Vec g)
{
}

#undef __FUNCT__
#define __FUNCT__ "MatLMVMSetDelta"
EXTERN PetscErrorCode MatLMVMSetDelta(PetscReal d)
{
    
}

#undef __FUNCT__
#define __FUNCT__ "MatLMVMSetScale"
EXTERN PetscErrorCode MatLMVMSetScale(Vec s)
{
}

#undef __FUNCT__
#define __FUNCT__ "MatLMVMGetRejects"
EXTERN PetscErrorCode MatLMVMGetRejects()
{
}

#undef __FUNCT__
#define __FUNCT__ "MatLMVMSetH0"
EXTERN PetscErrorCode MatLMVMSetH0(Mat A)
{
}

#undef __FUNCT__
#define __FUNCT__ "MatLMVMGetX0"
EXTERN PetscErrorCode MatLMVMGetX0(Vec x)
{
}

#undef __FUNCT__
#define __FUNCT__ "MatLMVMRefine"
EXTERN PetscErrorCode MatLMVMRefine(Mat coarse, Mat op, Vec fineX, Vec fineG)
{
}
