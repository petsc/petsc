
#include <../src/ksp/ksp/impls/ngmres/ngmresimpl.h>       /*I "petscksp.h" I*/


/*
     KSPSetUp_NGMRES - Sets up the workspace needed by the NGMRES method. 

      This is called once, usually automatically by KSPSolve() or KSPSetUp()
*/
#undef __FUNCT__  
#define __FUNCT__ "KSPSetUp_NGMRES"
PetscErrorCode KSPSetUp_NGMRES(KSP ksp)
{
  KSP_NGMRES     *cgP = (KSP_NGMRES*)ksp->data;
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
  ierr = KSPGetVecs(ksp,cgP->msize,&cgP->v,cgP->msize,&cgP->w);CHKERRQ(ierr);
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
  PetscInt       i,j,k;
  KSP_NGMRES    *cg = (KSP_NGMRES*)ksp->data;
  Mat            Amat;
  Vec            X,B,R,Pold,P,*V = cg->v,*W = cg->w;
  PetscScalar    gdot;
  PetscReal      gnorm;
  PetscScalar    rdot,abr,A0;
  Vec            y,w;

  PetscFunctionBegin;
  X             = ksp->vec_sol;
  B             = ksp->vec_rhs;
  R             = ksp->work[0];
  Pold          = ksp->work[1];
  P             = ksp->work[2];
  ierr = PCGetOperators(ksp->pc,&Amat,PETSC_NULL,PETSC_NULL);CHKERRQ(ierr);

  if (!ksp->guess_zero) {
    ierr = KSP_MatMult(ksp,Amat,X,R);CHKERRQ(ierr);            /*     r <- b - Ax     */
    ierr = VecAYPX(R,-1.0,B);CHKERRQ(ierr);
  } else { 
    ierr = VecCopy(B,R);CHKERRQ(ierr);                         /*     r <- b (x is 0) */
  } 
  ierr = PCApply(ksp->pc,R,Pold);CHKERRQ(ierr);                /*     p = B r */
  if (ksp->normtype == KSP_NORM_UNPRECONDITIONED) {
    ierr = VecNorm(R,NORM_2,&gnorm);CHKERRQ(ierr);          
  } else if (ksp->normtype == KSP_NORM_PRECONDITIONED) {
    ierr = VecNorm(Pold,NORM_2,&gnorm);CHKERRQ(ierr);          
  } else SETERRQ(((PetscObject)ksp)->comm,PETSC_ERR_SUP,"NormType not supported");
  KSPLogResidualHistory(ksp,gnorm);
  KSPMonitor(ksp,0,gnorm);
  ierr = (*ksp->converged)(ksp,0,gnorm,&ksp->reason,ksp->cnvP);CHKERRQ(ierr); 

  /* determine optimal scale factor -- slow code */
  ierr = VecDuplicate(P,&y);CHKERRQ(ierr);
  ierr = VecDuplicate(P,&w);CHKERRQ(ierr);
  ierr = MatMult(Amat,Pold,y);CHKERRQ(ierr);
  /*ierr = KSP_PCApplyBAorAB(ksp,Pold,y,w);CHKERRQ(ierr);  */    /* y = BAp */
  ierr  = VecDotNorm2(Pold,y,&rdot,&abr);CHKERRQ(ierr);   /*   rdot = (p)^T(BAp); abr = (BAp)^T (BAp) */
  ierr = VecDestroy(&y);CHKERRQ(ierr);
  ierr = VecDestroy(&w);CHKERRQ(ierr);
  A0 = rdot/abr;
  ierr = VecAXPY(X,A0,Pold);CHKERRQ(ierr);             /*   x  <- x + scale p */

    /* fix below */

  for (k=0; k<ksp->max_it; k += cg->msize) {
    for (i=0; i<cg->msize && k+i<ksp->max_it; i++) {
      ierr = KSP_MatMult(ksp,Amat,X,R);CHKERRQ(ierr);            /*     r <- b - Ax     */
      ierr = VecAYPX(R,-1.0,B);CHKERRQ(ierr);

      ierr = PCApply(ksp->pc,R,P);CHKERRQ(ierr);                 /*     p = B r */
      if (ksp->normtype == KSP_NORM_UNPRECONDITIONED) {
        ierr = VecNorm(R,NORM_2,&gnorm);CHKERRQ(ierr);          
      } else if (ksp->normtype == KSP_NORM_PRECONDITIONED) {
        ierr = VecNorm(P,NORM_2,&gnorm);CHKERRQ(ierr);          
      } else SETERRQ(((PetscObject)ksp)->comm,PETSC_ERR_SUP,"NormType not supported");
      KSPLogResidualHistory(ksp,gnorm);
      KSPMonitor(ksp,(1+k+i),gnorm);
      ierr = (*ksp->converged)(ksp,1+k+i,gnorm,&ksp->reason,ksp->cnvP);CHKERRQ(ierr); 
      if (ksp->reason) PetscFunctionReturn(0);

      for (j=0; j<i; j++) {                                     /* r = product_i [I+v(i)w(i)^T]* */
        ierr = VecDot(W[j],P,&gdot);CHKERRQ(ierr);
        ierr = VecAXPY(P,gdot,V[j]);CHKERRQ(ierr);
      }
      ierr = VecCopy(Pold,W[i]);CHKERRQ(ierr);                   /* W[i] = Pold */

      ierr = VecAXPY(Pold,-1.0,P);CHKERRQ(ierr);                 /* V[i] =       P           */
      ierr = VecDot(W[i],Pold,&gdot);CHKERRQ(ierr);             /*        ----------------- */
      ierr = VecCopy(P,V[i]);CHKERRQ(ierr);                      /*         W[i]'*(Pold - P)    */
      ierr = VecScale(V[i],1.0/gdot);CHKERRQ(ierr);

      ierr = VecDot(W[i],P,&gdot);CHKERRQ(ierr);                /* P = (I + V[i]*W[i]')*P  */
      ierr = VecAXPY(P,gdot,V[i]);CHKERRQ(ierr);
      ierr = VecCopy(P,Pold);CHKERRQ(ierr);

      ierr = VecAXPY(X,1.0,P);CHKERRQ(ierr);                    /* X = X + P */
    }
  }
  ksp->reason = KSP_DIVERGED_ITS;
  PetscFunctionReturn(0);
}
#undef __FUNCT__  
#define __FUNCT__ "KSPReset_NGMRES" 
PetscErrorCode KSPReset_NGMRES(KSP ksp)
{
  KSP_NGMRES    *cg = (KSP_NGMRES*)ksp->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = VecDestroyVecs(cg->msize,&cg->v);CHKERRQ(ierr);
  ierr = VecDestroyVecs(cg->msize,&cg->w);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*
       KSPDestroy_NGMRES - Frees all memory space used by the Krylov method

*/
#undef __FUNCT__  
#define __FUNCT__ "KSPDestroy_NGMRES" 
PetscErrorCode KSPDestroy_NGMRES(KSP *ksp)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = KSPReset_NGMRES(*ksp);CHKERRQ(ierr);
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
  KSP_NGMRES    *cg = (KSP_NGMRES *)ksp->data; 
  PetscErrorCode ierr;
  PetscBool      iascii;

  PetscFunctionBegin;
  ierr = PetscTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&iascii);CHKERRQ(ierr);
  if (iascii) {
    ierr = PetscViewerASCIIPrintf(viewer,"  Size of space %d\n",cg->msize);CHKERRQ(ierr);
  } else {
    SETERRQ1(((PetscObject)ksp)->comm,PETSC_ERR_SUP,"Viewer type %s not supported for KSP cg",((PetscObject)viewer)->type_name);
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
  KSP_NGMRES    *cg = (KSP_NGMRES *)ksp->data;

  PetscFunctionBegin;
  ierr = PetscOptionsHead("KSP NGMRES options");CHKERRQ(ierr);
    ierr = PetscOptionsInt("-ksp_gmres_restart","Number of directions","None",cg->msize,&cg->msize,PETSC_NULL);CHKERRQ(ierr);
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
  KSP_NGMRES     *cg;

  PetscFunctionBegin;
  ierr      = PetscNewLog(ksp,KSP_NGMRES,&cg);CHKERRQ(ierr);
  ksp->data = (void*)cg;
  cg->msize = 30;
  cg->csize = 0;

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




