
#include "petsc.h"
#include "petscvec.h"
#include "petscmat.h"
#include "petscksp.h"

#include "private/kspimpl.h"


typedef struct {
  PetscTruth recycle;
  PetscInt   restart;
  PetscInt   n_restarts;
  PetscReal  *val;
  Vec        *VV, *SS;
  Vec        R;
} KSP_GCR;

#undef __FUNCT__  
#define __FUNCT__ "KSPSolve_GCRRestart"
PetscErrorCode KSPSolve_GCRRestart( KSP ksp )
{
  KSP_GCR        *ctx = (KSP_GCR*)ksp->data;
  PetscErrorCode ierr;
  PetscReal      r_dot_v, nrm;
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
  v = ctx->VV[0];
  s = ctx->SS[0];
        
  k=0;
  do {
    ierr = PCApply( pc, r, s );CHKERRQ(ierr); // s = P^{-1} r
    ierr = MatMult( A, s, v );CHKERRQ(ierr);
                
    if( k>0 ) {
      ierr = VecMDot( v,k, &ctx->VV[1], &ctx->val[1] );CHKERRQ(ierr);
    }
    
    for( i=1; i<=k; i++ ) {
      ierr = VecAXPY( v, -ctx->val[i], ctx->VV[i] );CHKERRQ(ierr); // v = v - alpha_i v_i
      ierr = VecAXPY( s, -ctx->val[i], ctx->SS[i] );CHKERRQ(ierr); // s = s - alpha_i s_i
    }
    
    ierr = VecNorm( v, NORM_2, &nrm );CHKERRQ(ierr);
    ierr = VecScale( v, 1.0/nrm );CHKERRQ(ierr);
    ierr = VecScale( s, 1.0/nrm );CHKERRQ(ierr);
    ierr = VecDot( r, v, &r_dot_v );CHKERRQ(ierr);
    ierr = VecAXPY( x,  r_dot_v, s );CHKERRQ(ierr);
    ierr = VecAXPY( r, -r_dot_v, v );CHKERRQ(ierr);
    ierr = VecNorm( r, NORM_2, &norm_r );CHKERRQ(ierr);
                
    /* update the local counter and the global counter */
    k++;
    ksp->its++;
    res = norm_r;
    ksp->rnorm = res;
                
    KSPLogResidualHistory(ksp,res);
    KSPMonitor(ksp,ksp->its,res);
                
    ierr = (*ksp->converged)(ksp,ksp->its,res,&ksp->reason,ksp->cnvP);CHKERRQ(ierr);
    if (ksp->reason) break;
                
    if( ksp->its >= ksp->max_it ) {
      ksp->reason = KSP_CONVERGED_ITS;
      break;
    }
    ierr = VecCopy( v, ctx->VV[k] );CHKERRQ(ierr);
    ierr = VecCopy( s, ctx->SS[k] );CHKERRQ(ierr);
  } while( k < restart-1 );
  ctx->n_restarts++;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "KSPSolve_GCR_2"
PetscErrorCode KSPSolve_GCR_2( KSP ksp )
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
  ierr = VecAYPX( r, -1.0, b );CHKERRQ(ierr); // r = b - A x 
  ierr = VecNorm( r, NORM_2, &norm_r );CHKERRQ(ierr);
        
  ksp->its = 0;
  ksp->rnorm0 = norm_r;
        
  KSPLogResidualHistory(ksp,ksp->rnorm0);
  KSPMonitor(ksp,ksp->its,ksp->rnorm0);
  ierr = (*ksp->converged)(ksp,ksp->its,ksp->rnorm0,&ksp->reason,ksp->cnvP);CHKERRQ(ierr);
  if (ksp->reason) PetscFunctionReturn(0);
        
  do {
    ierr = KSPSolve_GCRRestart( ksp );CHKERRQ(ierr);
    if (ksp->reason) break; /* catch case when convergence occurs inside the cycle */
  } while( ksp->its < ksp->max_it );CHKERRQ(ierr);
  if( ksp->its >= ksp->max_it) {
    ksp->reason = KSP_DIVERGED_ITS;
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "KSPSolve_GCR"
PetscErrorCode KSPSolve_GCR( KSP ksp )
{
  KSP_GCR        *ctx = (KSP_GCR*)ksp->data;
  PetscErrorCode ierr;
  PetscReal      r_dot_v, nrm;
  Mat            A, B;
  PC             pc;
  Vec            s,v,r;
  PetscReal      norm_r0, norm_r;
  PetscInt       k, i, restart;
  PetscScalar    eps, abs_eps;
  Vec            b,x;        
  PetscReal      res;
                
  PetscFunctionBegin;
  restart = ksp->max_it;
  eps = ksp->rtol;
  abs_eps = ksp->abstol;
  ierr = KSPGetPC( ksp, &pc );CHKERRQ(ierr);
  ierr = KSPGetOperators( ksp, &A, &B, 0 );CHKERRQ(ierr);
        
  x = ksp->vec_sol;
  b = ksp->vec_rhs;
  r = ctx->R;
  v = ctx->VV[0];
  s = ctx->SS[0];
        
  ierr = MatMult( A, x, r );CHKERRQ(ierr);
  ierr = VecAYPX( r, -1.0, b );CHKERRQ(ierr); // r = b - A x 
  ierr = VecNorm( r, NORM_2, &norm_r0 );CHKERRQ(ierr);

  k=0;
  ksp->its = k;
  res = norm_r0;
  ksp->rnorm = res;
  
  KSPLogResidualHistory(ksp,res);
  KSPMonitor(ksp,ksp->its,res);
    
  ierr = (*ksp->converged)(ksp,ksp->its,res,&ksp->reason,ksp->cnvP);CHKERRQ(ierr);
  if (ksp->reason) PetscFunctionReturn(0);
        
  do {
    ierr = PCApply( pc, r, s );CHKERRQ(ierr); // s = P^{-1} r
    ierr = MatMult( A, s, v );CHKERRQ(ierr);
                
    if( k>0 ) {
      ierr = VecMDot( v,k, &ctx->VV[1], &ctx->val[1] );CHKERRQ(ierr);
    }
    for( i=1; i<=k; i++ ) {
      ierr = VecAXPY( v, -ctx->val[i], ctx->VV[i] );CHKERRQ(ierr); // v = v - alpha_i v_i
      ierr = VecAXPY( s, -ctx->val[i], ctx->SS[i] );CHKERRQ(ierr); // s = s - alpha_i ss_i
    }
    
    ierr = VecNorm( v, NORM_2, &nrm );CHKERRQ(ierr);
    ierr = VecScale( v, 1.0/nrm );CHKERRQ(ierr);
    ierr = VecScale( s, 1.0/nrm );CHKERRQ(ierr);
    ierr = VecDot( r, v, &r_dot_v );CHKERRQ(ierr);
    ierr = VecAXPY( x,  r_dot_v, s );CHKERRQ(ierr);
    ierr = VecAXPY( r, -r_dot_v, v );CHKERRQ(ierr);
    ierr = VecNorm( r, NORM_2, &norm_r );CHKERRQ(ierr);
    k++;
    ksp->its = k;
    res = norm_r;
    ksp->rnorm = res;
    
    KSPLogResidualHistory(ksp,res);
    KSPMonitor(ksp,ksp->its,res);
                
    ierr = (*ksp->converged)(ksp,ksp->its,res,&ksp->reason,ksp->cnvP);CHKERRQ(ierr);
    if (ksp->reason) break;
                
    ierr = VecCopy( v, ctx->VV[k] );CHKERRQ(ierr);
    ierr = VecCopy( s, ctx->SS[k] );CHKERRQ(ierr);
  } while( k<ksp->max_it );
  if (k >= ksp->max_it) {
    ksp->reason = KSP_DIVERGED_ITS;
  }
  ctx->n_restarts++;
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
    if (ctx->recycle) {
      ierr = PetscViewerASCIIPrintf(viewer,"  GCR: Using recycled solution\n" );CHKERRQ(ierr);
    }
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
  PetscTruth      flg, recycle;
        
  PetscFunctionBegin;
  ierr = PetscOptionsHead("KSP GCR options");CHKERRQ(ierr);
    ierr = PetscOptionsInt("-ksp_gcr_restart","Number of Krylov search directions","KSPGCRSetRestart",ctx->restart,&restart,&flg);CHKERRQ(ierr);
    if (flg) { ierr = KSPGCRSetRestart(ksp,restart);CHKERRQ(ierr); }
    
    ierr = PetscOptionsTruth("-ksp_gcr_recycle","Recycle solution","KSPGCRSetRecycleSolution_GCR",ctx->recycle,&recycle,&flg);CHKERRQ(ierr);
    if (flg) { ierr = KSPGCRSetRecycleSolution(ksp,recycle);CHKERRQ(ierr); }
  ierr = PetscOptionsTail();CHKERRQ(ierr);
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
#define __FUNCT__ "KSPGCRSetRecycleSolution_GCR" 
PetscErrorCode KSPGCRSetRecycleSolution_GCR(KSP ksp,PetscTruth recycle)
{
  KSP_GCR *ctx;
        
  PetscFunctionBegin;
  ctx = (KSP_GCR *)ksp->data;
  ctx->recycle = recycle;

  if (ctx->recycle) {
    ksp->ops->solve = KSPSolve_GCR_2;
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "KSPGCRSetRecycleSolution" 
PetscErrorCode  KSPGCRSetRecycleSolution(KSP ksp, PetscTruth recycle)
{ 
  PetscErrorCode ierr;
      
  ierr = PetscTryMethod(ksp,"KSPGCRSetRecycleSolution_C",(KSP,PetscTruth),(ksp,recycle));CHKERRQ(ierr);
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

#undef __FUNCT__  
#define __FUNCT__ "KSPCreate_GCR"
PetscErrorCode KSPCreate_GCR(KSP ksp)
{
  PetscErrorCode ierr;
  KSP_GCR         *ctx;

  PetscFunctionBegin;
  ierr = PetscNewLog(ksp,KSP_GCR,&ctx);CHKERRQ(ierr);
  ctx->restart                   = 30;
  ctx->n_restarts                = 0;
  ksp->data                      = (void*)ctx;
  ksp->pc_side                   = PC_RIGHT;
        
  ksp->ops->setup                = KSPSetUp_GCR;
  ksp->ops->solve                = KSPSolve_GCR;
  ksp->ops->destroy              = KSPDestroy_GCR;
  ksp->ops->view                 = KSPView_GCR;
  ksp->ops->setfromoptions       = KSPSetFromOptions_GCR;
  ksp->ops->buildsolution        = KSPBuildSolution_GCR;
  ksp->ops->buildresidual        = KSPBuildResidual_GCR;
  
  ierr = PetscObjectComposeFunctionDynamic(  (PetscObject)ksp, "KSPGCRSetRecycleSolution_C",
				      "KSPGCRSetRecycleSolution_GCR",KSPGCRSetRecycleSolution_GCR );CHKERRQ(ierr);
  
  ierr = PetscObjectComposeFunctionDynamic(  (PetscObject)ksp, "KSPGCRSetRestart_C",
				      "KSPGCRSetRestart_GCR",KSPGCRSetRestart_GCR );CHKERRQ(ierr);
  PetscFunctionReturn(0);
}






