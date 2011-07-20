
#include <../src/ksp/ksp/impls/ngmres/ngmresimpl.h>       /*I "petscksp.h" I*/

static PetscErrorCode    BuildNGmresSoln(PetscScalar*,Vec,KSP,PetscInt,PetscInt);
/*
     KSPSetUp_NGMRES - Sets up the workspace needed by the NGMRES method. 

      This is called once, usually automatically by KSPSolve() or KSPSetUp()
*/
#undef __FUNCT__  
#define __FUNCT__ "KSPSetUp_NGMRES"
PetscErrorCode KSPSetUp_NGMRES(KSP ksp)
{

  PetscInt       hh;
  PetscInt       msize;
  KSP_NGMRES     *ngmres = (KSP_NGMRES*)ksp->data;
  PetscErrorCode ierr;
  PetscBool      diagonalscale;

  PetscFunctionBegin;
  /* 
       This implementation of NGMRES only handles left preconditioning
     so generate an error otherwise.
  */
  if (ksp->pc_side == PC_RIGHT) SETERRQ(((PetscObject)ksp)->comm,PETSC_ERR_SUP,"No right preconditioning for KSPNGMRES");
  else if (ksp->pc_side == PC_SYMMETRIC) SETERRQ(((PetscObject)ksp)->comm,PETSC_ERR_SUP,"No symmetric preconditioning for KSPNGMRES");
  ierr    = PCGetDiagonalScale(ksp->pc,&diagonalscale);CHKERRQ(ierr);
  if (diagonalscale) SETERRQ1(((PetscObject)ksp)->comm,PETSC_ERR_SUP,"Krylov method %s does not support diagonal scaling",((PetscObject)ksp)->type_name);
  ngmres->beta  = 1.0;
  msize         = ngmres->msize;  /* restart size */
  hh            = msize * msize;
  
  //  ierr = PetscMalloc1(hh,PetscScalar,&ngmres->hh_origin,bb,PetscScalar,&ngmres->bb_origin,rs,PetscScalar,&ngmres->rs_origin,cc,PetscScalar,&ngmres->cc_origin,cc,PetscScalar,&ngmres->ss_origin);CHKERRQ(ierr);
  ierr = PetscMalloc2(hh,PetscScalar,&ngmres->hh_origin,msize,PetscScalar,&ngmres->nrs);CHKERRQ(ierr);

  ierr = PetscMemzero(ngmres->hh_origin,hh*sizeof(PetscScalar));CHKERRQ(ierr);
  ierr = PetscMemzero(ngmres->nrs,msize*sizeof(PetscScalar));CHKERRQ(ierr);

  ierr = PetscLogObjectMemory(ksp,(hh+msize)*sizeof(PetscScalar));CHKERRQ(ierr);

  ierr = KSPGetVecs(ksp,ngmres->msize,&ngmres->v,ngmres->msize*3,&ngmres->w);CHKERRQ(ierr);
  
  ierr = KSPDefaultGetWork(ksp,3);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*
       KSPSolve_NGMRES - This routine actually applies the method


   Input Parameter:
.     ksp - the Krylov space object that was set to use conjugate gradient, by, for 
            example, KSPCreate(MPI_Comm,KSP *ksp); KSPSetType(ksp,KSPNGMRES);
*/
#undef __FUNCT__  
#define __FUNCT__ "KSPSolve_NGMRES"
PetscErrorCode  KSPSolve_NGMRES(KSP ksp)
{
  PetscErrorCode ierr;
  PetscInt       i,j,k,l,flag,it;
  KSP_NGMRES    *ngmres = (KSP_NGMRES*)ksp->data;
  Mat            Amat;
  Vec            X,F,R, B,Fold, Xold,temp,*dX = ngmres->w,*dF = ngmres->w+ngmres->msize;
  PetscScalar    *nrs=ngmres->nrs;
  PetscReal      gnorm;
  

  PetscFunctionBegin;
  X             = ksp->vec_sol;
  B             = ksp->vec_rhs;
  F             = ksp->work[0];
  Fold          = ksp->work[1];
  R             = ksp->work[2];
  ierr = VecDuplicate(X,&Fold);CHKERRQ(ierr);
  ierr = VecDuplicate(X,&Xold);CHKERRQ(ierr);
  ierr = VecDuplicate(X,&temp);CHKERRQ(ierr);


  ierr = PCGetOperators(ksp->pc,&Amat,PETSC_NULL,PETSC_NULL);CHKERRQ(ierr);

  if (!ksp->guess_zero) {
    ierr = KSP_MatMult(ksp,Amat,X,R);CHKERRQ(ierr);            /*     r <- b - Ax     */
    ierr = VecAYPX(R,-1.0,B);CHKERRQ(ierr);
  } else { 
    ierr = VecCopy(B,R);CHKERRQ(ierr);                         /*     r <- b (x is 0) */
  } 
  ierr = PCApply(ksp->pc,R,F);CHKERRQ(ierr);                /*     F = B r */
  if (ksp->normtype == KSP_NORM_UNPRECONDITIONED) {
    ierr = VecNorm(R,NORM_2,&gnorm);CHKERRQ(ierr);          
  } else if (ksp->normtype == KSP_NORM_PRECONDITIONED) {
    ierr = VecNorm(F,NORM_2,&gnorm);CHKERRQ(ierr);          
  } else SETERRQ(((PetscObject)ksp)->comm,PETSC_ERR_SUP,"NormType not supported");
  KSPLogResidualHistory(ksp,gnorm);
  ierr = KSPMonitor(ksp,0,gnorm);CHKERRQ(ierr);
  ierr = (*ksp->converged)(ksp,0,gnorm,&ksp->reason,ksp->cnvP);CHKERRQ(ierr); 


 /* for k=0 */

  k=0;
  ierr= VecCopy(X,Xold); CHKERRQ(ierr); /* Xbar_0= X_0 */
  ierr= VecCopy(F,Fold); CHKERRQ(ierr); /* Fbar_0 = f_0= B(b-Ax_0) */
  ierr= VecWAXPY(X, ngmres->beta, Fold, Xold); CHKERRQ(ierr);       /*X_1 = X_bar + beta * Fbar */

  /* to calculate f_1 */
  ierr = KSP_MatMult(ksp,Amat,X,R);CHKERRQ(ierr);            /*     r <- b - Ax     */
  ierr = VecAYPX(R,-1.0,B);CHKERRQ(ierr); 
  ierr = PCApply(ksp->pc,R,F);CHKERRQ(ierr);                /*     F= f_1 = B(b-Ax) */

  /* calculate dX and dF for k=0 */
  ierr= VecWAXPY(dX[k],-1.0, Xold, X); CHKERRQ(ierr); /* dX= X_1 - X_0 */
  ierr= VecWAXPY(dF[k],-1.0, Fold, F); CHKERRQ(ierr); /* dF= f_1 - f_0 */
    
 
  ierr= VecCopy(X,Xold); CHKERRQ(ierr); /* Xbar_0= X_0 */
  ierr= VecCopy(F,Fold); CHKERRQ(ierr); /* Fbar_0 = f_0= B(b-Ax_0) */

  flag=0;

  for (k=1; k<ksp->max_it; k += 1) {  /* begin the iteration */     
    l=ngmres->msize;
    if(k<l) l=k;
    it=l-1;
    ierr = BuildNGmresSoln(nrs,Fold,ksp,it,flag);CHKERRQ(ierr);
   
    /* to obtain the solution at k+1 step */
    ierr= VecCopy(Xold,X); CHKERRQ(ierr); /* X=Xold+Fold-(dX + dF) *nrd */
     ierr= VecAXPY(X,1.0,Fold); CHKERRQ(ierr); /* X= Xold+Fold */
    for(i=0;i<l;i++){      /*X= Xold+Fold- (dX+dF*beta) *innerb */
      ierr= VecAXPY(X,-nrs[i], dX[i]);CHKERRQ(ierr);
      ierr= VecAXPY(X,-nrs[i]*ngmres->beta, dF[i]);CHKERRQ(ierr);
    }


    /* to test with GMRES */
    //ierr= VecCopy(Xold,temp); CHKERRQ(ierr); /* X=Xold-(dX ) *nrd */
    /*for(i=0;i<l;i++){      
      ierr= VecAXPY(temp,-nrs[i], dX[i]);CHKERRQ(ierr);
    }
    ierr = KSP_MatMult(ksp,Amat,temp,R);CHKERRQ(ierr);            
    ierr = VecAYPX(R,-1.0,B);CHKERRQ(ierr); 
    ierr = PCApply(ksp->pc,R,F);CHKERRQ(ierr);                
     ierr = VecNorm(R,NORM_2,&gnorm);CHKERRQ(ierr); 
     printf("gmres residual norm=%g\n",gnorm);
    */

    /* to calculate f_k+1 */
    ierr = KSP_MatMult(ksp,Amat,X,R);CHKERRQ(ierr);            /*     r <- b - Ax     */
    ierr = VecAYPX(R,-1.0,B);CHKERRQ(ierr); 
    ierr = PCApply(ksp->pc,R,F);CHKERRQ(ierr);                /*     F= f_1 = B(b-Ax) */
   

    /* check the convegence */
    
    if (ksp->normtype == KSP_NORM_UNPRECONDITIONED) {
        ierr = VecNorm(R,NORM_2,&gnorm);CHKERRQ(ierr);          
      } else if (ksp->normtype == KSP_NORM_PRECONDITIONED) {
        ierr = VecNorm(F,NORM_2,&gnorm);CHKERRQ(ierr);          
      } else SETERRQ(((PetscObject)ksp)->comm,PETSC_ERR_SUP,"NormType not supported");
      KSPLogResidualHistory(ksp,gnorm);
      //printf("k=%d",k);
      KSPMonitor(ksp,k,gnorm);
      ksp->its=k;
      ierr = (*ksp->converged)(ksp,k,gnorm,&ksp->reason,ksp->cnvP);CHKERRQ(ierr); 
      if (ksp->reason) PetscFunctionReturn(0);


    /* calculate dX and dF for k=0 */
      if( k>l) {/* we need to replace the old vectors */
	flag=1;
	for(i=0;i<l-1;i++){
	  ierr= VecCopy(dX[i+1],dX[i]); CHKERRQ(ierr); /* X=Xold+Fold-(dX + dF) *nrd */
	  ierr= VecCopy(dF[i+1],dF[i]); CHKERRQ(ierr); /* X=Xold+Fold-(dX + dF) *nrd */
          for(j=0;j<l;j++)
	    *HH(j,i)=*HH(j,i+1);
	}
      }
      ierr= VecWAXPY(dX[l],-1.0, Xold, X); CHKERRQ(ierr); /* dX= X_1 - X_0 */
      ierr= VecWAXPY(dF[l],-1.0, Fold, F); CHKERRQ(ierr); /* dF= f_1 - f_0 */
      ierr= VecCopy(X,Xold); CHKERRQ(ierr);
      ierr= VecCopy(F,Fold); CHKERRQ(ierr);
  
  }
  ksp->reason = KSP_DIVERGED_ITS;
  PetscFunctionReturn(0);
}
#undef __FUNCT__  
#define __FUNCT__ "KSPReset_NGMRES" 
PetscErrorCode KSPReset_NGMRES(KSP ksp)
{
  KSP_NGMRES    *ngmres = (KSP_NGMRES*)ksp->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = VecDestroyVecs(ngmres->msize,&ngmres->v);CHKERRQ(ierr);
  ierr = VecDestroyVecs(ngmres->msize,&ngmres->w);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*
       KSPDestroy_NGMRES - Frees all memory space used by the Krylov method

*/
#undef __FUNCT__  
#define __FUNCT__ "KSPDestroy_NGMRES" 
PetscErrorCode KSPDestroy_NGMRES(KSP ksp)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = KSPReset_NGMRES(ksp);CHKERRQ(ierr);
  ierr = KSPDefaultDestroy(ksp);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*
     KSPView_NGMRES - Prints information about the current Krylov method being used

      Currently this only prints information to a file (or stdout) about the 
      symmetry of the problem. If your Krylov method has special options or 
      flags that information should be printed here.

*/
#undef __FUNCT__  
#define __FUNCT__ "KSPView_NGMRES" 
PetscErrorCode KSPView_NGMRES(KSP ksp,PetscViewer viewer)
{
  KSP_NGMRES    *ngmres = (KSP_NGMRES *)ksp->data; 
  PetscErrorCode ierr;
  PetscBool      iascii;

  PetscFunctionBegin;
  ierr = PetscTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&iascii);CHKERRQ(ierr);
  if (iascii) {
    ierr = PetscViewerASCIIPrintf(viewer,"  Size of space %d\n",ngmres->msize);CHKERRQ(ierr);
  } else {
    SETERRQ1(((PetscObject)ksp)->comm,PETSC_ERR_SUP,"Viewer type %s not supported for KSP ngmres",((PetscObject)viewer)->type_name);
  }
  PetscFunctionReturn(0);
}

/*
    KSPSetFromOptions_NGMRES - Checks the options database for options related to the  method
*/ 
#undef __FUNCT__  
#define __FUNCT__ "KSPSetFromOptions_NGMRES"
PetscErrorCode KSPSetFromOptions_NGMRES(KSP ksp)
{
  PetscErrorCode ierr;
  KSP_NGMRES    *ngmres = (KSP_NGMRES *)ksp->data;

  PetscFunctionBegin;
  ierr = PetscOptionsHead("KSP NGMRES options");CHKERRQ(ierr);
    ierr = PetscOptionsInt("-ksp_gmres_restart","Number of directions","None",ngmres->msize,&ngmres->msize,PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsTail();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*
    KSPCreate_NGMRES - Creates the data structure for the Krylov method NGMRES and sets the 
       function pointers for all the routines it needs to call (KSPSolve_NGMRES() etc)

    It must be wrapped in EXTERN_C_BEGIN to be dynamically linkable in C++
*/
/*MC
     KSPNGMRES - The preconditioned conjugate gradient (NGMRES) iterative method

   Level: beginner

   Notes: Supports only left preconditioning

.seealso:  KSPCreate(), KSPSetType(), KSPType (for list of available types), KSP

M*/
EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "KSPCreate_NGMRES"
PetscErrorCode  KSPCreate_NGMRES(KSP ksp)
{
  PetscErrorCode ierr;
  KSP_NGMRES     *ngmres;

  PetscFunctionBegin;
  ierr      = PetscNewLog(ksp,KSP_NGMRES,&ngmres);CHKERRQ(ierr);
  ksp->data = (void*)ngmres;
  ngmres->msize = 30;
  ngmres->csize = 0;

 if (ksp->pc_side != PC_LEFT) {
    ierr = PetscInfo(ksp,"WARNING! Setting PC_SIDE for NGMRES to left!\n");CHKERRQ(ierr);
  }
  ksp->pc_side                   = PC_LEFT;

  /*
       Sets the functions that are associated with this data structure 
       (in C++ this is the same as defining virtual functions)
  */
  ksp->ops->setup                = KSPSetUp_NGMRES;
  ksp->ops->solve                = KSPSolve_NGMRES;
  ksp->ops->reset                = KSPReset_NGMRES;
  ksp->ops->destroy              = KSPDestroy_NGMRES;
  ksp->ops->view                 = KSPView_NGMRES;
  ksp->ops->setfromoptions       = KSPSetFromOptions_NGMRES;
  ksp->ops->buildsolution        = KSPDefaultBuildSolution;
  ksp->ops->buildresidual        = KSPDefaultBuildResidual;
  PetscFunctionReturn(0);
}
EXTERN_C_END




/*
    BuildNGmresSoln - create the solution of the least square problem to determine the coefficients

    Input parameters:
        nrs - the solution
	Fold  - the RHS vector
        flag - =1, we need to replace the old vector by new one, =0, we still add new vector 
     This is an internal routine that knows about the NGMRES internals.
 */
#undef __FUNCT__  
#define __FUNCT__ "BuildNGmresSoln"
static PetscErrorCode BuildNGmresSoln(PetscScalar* nrs, Vec Fold, KSP ksp,PetscInt it, PetscInt flag)
{
  PetscScalar    tt,temps;
  PetscErrorCode ierr;
  PetscInt       i,ii,j,l;
  KSP_NGMRES      *ngmres = (KSP_NGMRES *)(ksp->data);
  Vec *dF=ngmres->w+ngmres->msize, *Q=ngmres->w+ngmres->msize*2,temp;
  PetscReal      gam;
  PetscScalar    a,b,c,s;
 
  PetscFunctionBegin;
  ierr = VecDuplicate(Fold,&temp);CHKERRQ(ierr);
  l=it+1;
    
  /* Solve for solution vector that minimizes the residual */

  if(flag==1) { // we need to replace the old vector and need to modify the QR factors, use Givens rotation
      for(i=0;i<it;i++){
	/* calculate the Givens rotation */
	a=*HH(i,i);
	b=*HH(i+1,i);
#if defined(PETSC_USE_COMPLEX)
	gam = 1.0/ PetScSqrtScalar(PetscConj(a) * a + PetscConj(b) * b);
#else
        gam=1.0/PetscSqrtScalar(a*a+b*b);
#endif
        c= a*gam;
        s= b*gam;
     
#if defined(PETSC_USE_COMPLEX)
	/* update the Q factor */
        ierr= VecCopy(Q[i],temp); CHKERRQ(ierr); 
	ierr = VecAXPBY(temp,s,PetscConj(c),Q[i+1]);CHKERRQ(ierr); /*temp= c*Q[i]+s*Q[i+1] */
        ierr = VecAXPBY(Q[i+1],-s,c,Q[i]);CHKERRQ(ierr); /* Q[i+1]= -s*Q[i] + c*Q[i+1] */
        ierr= VecCopy(temp,Q[i]); CHKERRQ(ierr);   /* Q[i]= c*Q[i] + s*Q[i+1] */
        /* update the R factor */
        for(j=0;j<l;j++){
          a= *HH(i,j);
          b=*HH(i+1,j);
	  temps=PetscConj(c)* a+s* b;           
          *HH(i+1,j)=-s*a+c*b;
          *HH(i,j)=temps;
        } 
#else
	/* update the Q factor */
        ierr= VecCopy(Q[i],temp); CHKERRQ(ierr); 
	ierr = VecAXPBY(temp,s,c,Q[i+1]);CHKERRQ(ierr); /*temp= c*Q[i]+s*Q[i+1] */
        ierr = VecAXPBY(Q[i+1],-s,c,Q[i]);CHKERRQ(ierr); /* Q[i+1]= -s*Q[i] + c*Q[i+1] */
        ierr= VecCopy(temp,Q[i]); CHKERRQ(ierr);   /* Q[i]= c*Q[i] + s*Q[i+1] */
        /* update the R factor */
        for(j=0;j<l;j++){
          a= *HH(i,j);
          b=*HH(i+1,j);
	  temps=c* a+s* b;           
          *HH(i+1,j)=-s*a+c*b;
          *HH(i,j)=temps;
        }
#endif 
      }
    }

    // add a new vector, use modified Gram-Schmidt 
    ierr= VecCopy(dF[it],temp); CHKERRQ(ierr);
    for(i=0;i<it;i++){
      ierr=VecDot(temp,Q[i],HH(i,it));CHKERRQ(ierr); /* h(i,l-1)= dF[l-1]'*Q[i] */
      ierr = VecAXPBY(temp,-*HH(i,it),1.0,Q[i]);CHKERRQ(ierr); /* temp= temp- h(i,l-1)*Q[i] */ 
    }
    ierr=VecCopy(temp,Q[it]);CHKERRQ(ierr); 
    ierr=VecNormalize(Q[it],&a);CHKERRQ(ierr);
    *HH(it,it)=a;
    


    /* modify the RHS with Q'*Fold*/
  
  for(i=0;i<l;i++) 
          ierr=VecDot(Fold,Q[i],ngmres->nrs+i);CHKERRQ(ierr); /* nrs= Fold'*Q[i] */
      
    /* start backsubstitution to solve the least square problem */      

     if (*HH(it,it) != 0.0) {
      nrs[it] =  nrs[it]/ *HH(it,it);
    } else {
    ksp->reason = KSP_DIVERGED_BREAKDOWN;
    ierr = PetscInfo2(ksp,"Likely your matrix or preconditioner is singular. HH(it,it) is identically zero; it = %D nrs(it) = %G",it,10);
    PetscFunctionReturn(0);
  }
  for (ii=1; ii<=it; ii++) {
    i   = it - ii;
    tt  = nrs[i];
    for (j=i+1; j<=it; j++) tt  = tt - *HH(i,j) * nrs[j];
    if (*HH(i,i) == 0.0) {
      ksp->reason = KSP_DIVERGED_BREAKDOWN;
      ierr = PetscInfo1(ksp,"Likely your matrix or preconditioner is singular. HH(k,k) is identically zero; i = %D",i);
      PetscFunctionReturn(0);
    } 
    nrs[i]   = tt / *HH(i,i);
  }

  
  PetscFunctionReturn(0);
}
