
#include "petscksp.h"
#include "private/kspimpl.h"

typedef struct {
  PetscInt       restart;
  PetscInt       n_restarts;
  PetscScalar    *val;
  Vec            *VV, *SS;
  Vec            R;

  PetscErrorCode (*modifypc)(KSP,PetscInt,PetscReal,void*);  /* function to modify the preconditioner*/
  PetscErrorCode (*modifypc_destroy)(void*);                 /* function to destroy the user context for the modifypc function */
  void            *modifypc_ctx;                             /* user defined data for the modifypc function */ 
} KSP_GCR;

#undef __FUNCT__  
#define __FUNCT__ "KSPSolve_GCR_cycle"
PetscErrorCode KSPSolve_GCR_cycle( KSP ksp )
{
  KSP_GCR        *ctx = (KSP_GCR*)ksp->data;
  PetscErrorCode ierr;
  PetscScalar    nrm,r_dot_v;
  Mat            A, B;
  PC             pc;
  Vec            s,v,r;
  PetscReal      norm_r;
  PetscInt       k, i, restart;
  Vec            b,x;        
  PetscReal      res;
                
  PetscFunctionBegin;
  restart = ctx->restart;
  ierr = KSPGetPC( ksp, &pc );CHKERRQ(ierr);
  ierr = KSPGetOperators( ksp, &A, &B, 0 );CHKERRQ(ierr);
        
  x = ksp->vec_sol;
  b = ksp->vec_rhs;
  r = ctx->R;

  for ( k=0; k<restart; k++ ) {
    v = ctx->VV[k];
    s = ctx->SS[k];
    if (ctx->modifypc) {
      ierr = (*ctx->modifypc)(ksp,ksp->its,ksp->rnorm,ctx->modifypc_ctx);CHKERRQ(ierr);
    }
		
    ierr = PCApply( pc, r, s ); CHKERRQ(ierr); /* s = B^{-1} r */
    ierr = MatMult( A, s, v ); CHKERRQ(ierr);  /* v = A s */
		
    ierr = VecMDot( v,k, ctx->VV, ctx->val ); CHKERRQ(ierr);
    
    for( i=0; i<=k; i++ ) {
      ierr = VecAXPY( v, -ctx->val[i], ctx->VV[i] ); CHKERRQ(ierr); /* v = v - alpha_i v_i */
      ierr = VecAXPY( s, -ctx->val[i], ctx->SS[i] ); CHKERRQ(ierr); /* s = s - alpha_i s_i */
    }
		
    ierr = VecDotNorm2(r,v,&r_dot_v,&nrm);CHKERRQ(ierr);
    nrm     = sqrt(nrm);
    r_dot_v = r_dot_v/nrm;
    ierr = VecScale( v, 1.0/nrm ); CHKERRQ(ierr);
    ierr = VecScale( s, 1.0/nrm ); CHKERRQ(ierr);
    ierr = VecAXPY( x,  r_dot_v, s ); CHKERRQ(ierr);
    ierr = VecAXPY( r, -r_dot_v, v ); CHKERRQ(ierr);
    if (ksp->its > ksp->chknorm  ) {
      ierr = VecNorm( r, NORM_2, &norm_r ); CHKERRQ(ierr);
    }		
    /* update the local counter and the global counter */
    ksp->its++;
    res = norm_r;
    ksp->rnorm = res;
		
    KSPLogResidualHistory(ksp,res);
    KSPMonitor(ksp,ksp->its,res);
		
    if( ksp->its > ksp->chknorm  ) {
      ierr = (*ksp->converged)(ksp,ksp->its,res,&ksp->reason,ksp->cnvP);CHKERRQ(ierr);
      if (ksp->reason) break;
    }
    
    if( ksp->its >= ksp->max_it ) {
      ksp->reason = KSP_CONVERGED_ITS;
      break;
    }
  }
  ctx->n_restarts++;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "KSPSolve_GCR"
PetscErrorCode KSPSolve_GCR( KSP ksp )
{
  KSP_GCR        *ctx = (KSP_GCR*)ksp->data;
  PetscErrorCode ierr;
  Mat            A, B;
  Vec            r,b,x;
  PetscReal      norm_r;
        
  PetscFunctionBegin;
  ierr = KSPGetOperators( ksp, &A, &B, PETSC_NULL );CHKERRQ(ierr);
  x = ksp->vec_sol;
  b = ksp->vec_rhs;
  r = ctx->R;
        
  /* compute initial residual */
  ierr = MatMult( A, x, r );CHKERRQ(ierr);
  ierr = VecAYPX( r, -1.0, b );CHKERRQ(ierr); /* r = b - A x  */
  ierr = VecNorm( r, NORM_2, &norm_r );CHKERRQ(ierr);
        
  ksp->its = 0;
  ksp->rnorm0 = norm_r;
        
  KSPLogResidualHistory(ksp,ksp->rnorm0);
  KSPMonitor(ksp,ksp->its,ksp->rnorm0);
  ierr = (*ksp->converged)(ksp,ksp->its,ksp->rnorm0,&ksp->reason,ksp->cnvP);CHKERRQ(ierr);
  if (ksp->reason) PetscFunctionReturn(0);
        
  do {
    ierr = KSPSolve_GCR_cycle( ksp );CHKERRQ(ierr);
    if (ksp->reason) break; /* catch case when convergence occurs inside the cycle */
  } while( ksp->its < ksp->max_it );CHKERRQ(ierr);
  if( ksp->its >= ksp->max_it) {
    ksp->reason = KSP_DIVERGED_ITS;
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "KSPView_GCR"
PetscErrorCode KSPView_GCR( KSP ksp, PetscViewer viewer )
{
  KSP_GCR        *ctx = (KSP_GCR*)ksp->data;
  PetscErrorCode ierr;
  PetscTruth     iascii;

  PetscFunctionBegin;
  PetscTypeCompare((PetscObject)viewer,PETSC_VIEWER_ASCII,&iascii);
  if (iascii) {
    ierr = PetscViewerASCIIPrintf(viewer,"  GCR: restart = %D \n", ctx->restart );CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"  GCR: restarts performed = %D \n", ctx->n_restarts );CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}


#undef __FUNCT__  
#define __FUNCT__ "KSPSetUp_GCR"
PetscErrorCode KSPSetUp_GCR( KSP ksp )
{
  KSP_GCR        *ctx = (KSP_GCR*)ksp->data;
  PetscErrorCode ierr;
  Mat            A;
        
  PetscFunctionBegin;
  if (ksp->pc_side == PC_LEFT) {SETERRQ(PETSC_ERR_SUP,"No left preconditioning for GCR");}
  else if (ksp->pc_side == PC_SYMMETRIC) {SETERRQ(PETSC_ERR_SUP,"No symmetric preconditioning for GCR");}

  ierr = KSPGetOperators( ksp, &A, 0, 0 );CHKERRQ(ierr);
  ierr = MatGetVecs( A, &ctx->R, PETSC_NULL );CHKERRQ(ierr);
  ierr = VecDuplicateVecs( ctx->R, ctx->restart, &ctx->VV );CHKERRQ(ierr);
  ierr = VecDuplicateVecs( ctx->R, ctx->restart, &ctx->SS );CHKERRQ(ierr);
        
  ierr = PetscMalloc( sizeof(PetscScalar)*ctx->restart, &ctx->val );CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "KSPDestroy_GCR"
PetscErrorCode KSPDestroy_GCR( KSP ksp )
{
  PetscErrorCode ierr;
  KSP_GCR        *ctx = (KSP_GCR*)ksp->data;
        
  PetscFunctionBegin;
  ierr = VecDestroy( ctx->R );CHKERRQ(ierr);
  ierr = VecDestroyVecs( ctx->VV, ctx->restart );CHKERRQ(ierr);
  ierr = VecDestroyVecs( ctx->SS, ctx->restart );CHKERRQ(ierr);
  ierr = PetscFree( ctx->val );CHKERRQ(ierr);
  if (ctx->modifypc_destroy) {
    ierr = (*ctx->modifypc_destroy)(ctx->modifypc_ctx);CHKERRQ(ierr);
  }
  ierr = KSPDefaultDestroy(ksp);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "KSPSetFromOptions_GCR"
PetscErrorCode KSPSetFromOptions_GCR(KSP ksp)
{
  PetscErrorCode  ierr;
  KSP_GCR         *ctx = (KSP_GCR *)ksp->data;
  PetscInt        restart;
  PetscTruth      flg;
        
  PetscFunctionBegin;
  ierr = PetscOptionsHead("KSP GCR options");CHKERRQ(ierr);
    ierr = PetscOptionsInt("-ksp_gcr_restart","Number of Krylov search directions","KSPGCRSetRestart",ctx->restart,&restart,&flg);CHKERRQ(ierr);
    if (flg) { ierr = KSPGCRSetRestart(ksp,restart);CHKERRQ(ierr); }
  ierr = PetscOptionsTail();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "KSPGCRSetModifyPC_GCR" 
PetscErrorCode PETSCKSP_DLLEXPORT KSPGCRSetModifyPC_GCR(KSP ksp,PetscErrorCode (*function)(KSP,PetscInt,PetscReal,void*),void *data,PetscErrorCode (*destroy)(void*))
{
  KSP_GCR         *ctx = (KSP_GCR *)ksp->data;
	
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ksp,KSP_COOKIE,1);
  ctx->modifypc         = function;
  ctx->modifypc_destroy = destroy;
  ctx->modifypc_ctx     = data;
  PetscFunctionReturn(0);
}
EXTERN_C_END

#undef __FUNCT__  
#define __FUNCT__ "KSPGCRSetModifyPC" 
/*@C
 KSPGCRSetModifyPC - Sets the routine used by GCR to modify the preconditioner.
 
 Collective on KSP
 
 Input Parameters:
 +  ksp      - iterative context obtained from KSPCreate()
 .  function - user defined function to modify the preconditioner
 .  ctx      - user provided contex for the modify preconditioner function
 -  destroy  - the function to use to destroy the user provided application context.
 
 Calling Sequence of function:
  PetscErrorCode function ( KSP ksp, PetscInt n, PetscReal rnorm, void *ctx )
 
 ksp   - iterative context 
 n     - the total number of GCR iterations that have occurred     
 rnorm - 2-norm residual value
 ctx   - the user provided application context
 
 Level: intermediate
 
 Notes:
 The default modifypc routine is KSPGCRModifyPCNoChange()
 
 .seealso: KSPGCRModifyPCNoChange()
 
 @*/
PetscErrorCode PETSCKSP_DLLEXPORT KSPGCRSetModifyPC(KSP ksp,PetscErrorCode (*function)(KSP,PetscInt,PetscReal,void*),void *data,PetscErrorCode (*destroy)(void*))
{
  PetscErrorCode ierr,(*f)(KSP,PetscErrorCode (*)(KSP,PetscInt,PetscReal,void*),void *data,PetscErrorCode (*)(void*));
  
  PetscFunctionBegin;
  ierr = PetscObjectQueryFunction((PetscObject)ksp,"KSPGCRSetModifyPC_C",(void (**)(void))&f);CHKERRQ(ierr);
  if (f) {
    ierr = (*f)(ksp,function,data,destroy);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "KSPGCRSetRestart_GCR" 
PetscErrorCode KSPGCRSetRestart_GCR(KSP ksp,PetscInt restart)
{
  KSP_GCR *ctx;
        
  PetscFunctionBegin;
  ctx = (KSP_GCR *)ksp->data;
  ctx->restart = restart;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "KSPGCRSetRestart" 
PetscErrorCode  KSPGCRSetRestart(KSP ksp, PetscInt restart)
{     
  PetscErrorCode ierr;

  PetscFunctionBegin;  
  ierr = PetscTryMethod(ksp,"KSPGCRSetRestart_C",(KSP,PetscInt),(ksp,restart));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "KSPBuildSolution_GCR" 
PetscErrorCode  KSPBuildSolution_GCR(KSP ksp, Vec v, Vec *V)
{       
  PetscErrorCode ierr;
  KSP_GCR         *ctx;
  Vec             x;
        
  PetscFunctionBegin;
  x = ksp->vec_sol;
  ctx = (KSP_GCR *)ksp->data;
  if (v) {
    ierr = VecCopy( x, v );CHKERRQ(ierr);
    if (V) *V = v;
  } else if (V) {
    *V = ksp->vec_sol;
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "KSPBuildResidual_GCR" 
PetscErrorCode  KSPBuildResidual_GCR(KSP ksp, Vec t, Vec v, Vec *V)
{       
  PetscErrorCode ierr;
  KSP_GCR         *ctx;
        
  PetscFunctionBegin;
  ctx = (KSP_GCR *)ksp->data;
  if (v) {
    ierr = VecCopy( ctx->R, v );CHKERRQ(ierr);
    if (V) *V = v;
  } else if (V) {
    *V = ctx->R;
  }
  PetscFunctionReturn(0);
}

/*MC
     KSPGCR - Implements the preconditioned Generalized Conjugate Residual method.  
 
 
   Options Database Keys:
+   -ksp_gcr_restart <restart> - the number of stored vectors to orthogonalize against
 
   Level: beginner
 
    Notes: The GCR Krylov method supports non-symmetric matrices and permits the use of a preconditioner 
           which may vary from one iteration to the next. Users can can define a method to vary the 
           preconditioner between iterates via KSPGCRSetModifyPC().
           Restarts are solves with x0 not equal to zero. When a restart occurs, the initial starting 
           solution is given by the current estimate for x which was obtained by the last restart 
           iterations of the GCR algorithm.
           Unlike GMRES and FGMRES, when using GCR, the solution and residual vector can be directly accessed at any iterate,
           with zero computational cost, via a call to KSPBuildSolution() and KSPBuildResidual() respectively.
           This implementation of GCR will only apply the stopping condition test whenever ksp->its > ksp->chknorm, 
           where ksp->chknorm is specified via the command line argument -ksp_check_norm_iteration or via 
           the function KSPSetCheckNormIteration().
           The method implemented requires the storage of 2 x restart + 1 vectors, twice as much as GMRES.
           Support only for right preconditioning.

    Contributed by Dave May
 
    References:
           S. C. Eisenstat, H. C. Elman, and H. C. Schultz. Variational iterative methods for 
           non-symmetric systems of linear equations. SIAM J. Numer. Anal., 20, 345-357, 1983
 
 
.seealso:  KSPCreate(), KSPSetType(), KSPType (for list of available types), KSP,
           KSPGCRSetRestart(), KSPGCRSetModifyPC(), KSPGMRES, KSPFGMRES
 
M*/
EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "KSPCreate_GCR"
PetscErrorCode KSPCreate_GCR(KSP ksp)
{
  PetscErrorCode ierr;
  KSP_GCR        *ctx;

  PetscFunctionBegin;
  ierr = PetscNewLog(ksp,KSP_GCR,&ctx);CHKERRQ(ierr);
  ctx->restart                   = 30;
  ctx->n_restarts                = 0;
  ksp->data                      = (void*)ctx;
  if (ksp->pc_side != PC_RIGHT) {
     ierr = PetscInfo(ksp,"WARNING! Setting PC_SIDE for GCR to right!\n");CHKERRQ(ierr);
  }
  ksp->pc_side                   = PC_RIGHT;
        
  ksp->ops->setup                = KSPSetUp_GCR;
  ksp->ops->solve                = KSPSolve_GCR;
  ksp->ops->destroy              = KSPDestroy_GCR;
  ksp->ops->view                 = KSPView_GCR;
  ksp->ops->setfromoptions       = KSPSetFromOptions_GCR;
  ksp->ops->buildsolution        = KSPBuildSolution_GCR;
  ksp->ops->buildresidual        = KSPBuildResidual_GCR;
  
  ierr = PetscObjectComposeFunctionDynamic(  (PetscObject)ksp, "KSPGCRSetRestart_C",
				      "KSPGCRSetRestart_GCR",KSPGCRSetRestart_GCR );CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)ksp,"KSPGCRSetModifyPC_C",
					   "KSPGCRSetModifyPC_GCR",KSPGCRSetModifyPC_GCR);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
EXTERN_C_END





