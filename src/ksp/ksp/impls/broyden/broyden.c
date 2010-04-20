#define PETSCKSP_DLL

#include "../src/ksp/ksp/impls/broyden/broydenimpl.h"       /*I "petscksp.h" I*/


/*
     KSPSetUp_Broyden - Sets up the workspace needed by the Broyden method. 

      This is called once, usually automatically by KSPSolve() or KSPSetUp()
     but can be called directly by KSPSetUp()
*/
#undef __FUNCT__  
#define __FUNCT__ "KSPSetUp_Broyden"
PetscErrorCode KSPSetUp_Broyden(KSP ksp)
{
  KSP_Broyden    *cgP = (KSP_Broyden*)ksp->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  /* 
       This implementation of Broyden only handles left preconditioning
     so generate an error otherwise.
  */
  if (ksp->pc_side == PC_RIGHT) {
    SETERRQ(PETSC_ERR_SUP,"No right preconditioning for KSPBroyden");
  } else if (ksp->pc_side == PC_SYMMETRIC) {
    SETERRQ(PETSC_ERR_SUP,"No symmetric preconditioning for KSPBroyden");
  }
  ierr = KSPGetVecs(ksp,cgP->msize,&cgP->v,cgP->msize,&cgP->w);CHKERRQ(ierr);
  ierr = KSPDefaultGetWork(ksp,3);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*
       KSPSolve_Broyden - This routine actually applies the method


   Input Parameter:
.     ksp - the Krylov space object that was set to use conjugate gradient, by, for 
            example, KSPCreate(MPI_Comm,KSP *ksp); KSPSetType(ksp,KSPBROYDEN);
*/
#undef __FUNCT__  
#define __FUNCT__ "KSPSolve_Broyden"
PetscErrorCode  KSPSolve_Broyden(KSP ksp)
{
  PetscErrorCode ierr;
  PetscInt       i,j,k;
  KSP_Broyden    *cg = (KSP_Broyden*)ksp->data;
  Mat            Amat;
  Vec            X,B,R,Pold,P,*V = cg->v,*W = cg->w;
  PetscScalar    gdot;
  PetscReal      gnorm;

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
  } else SETERRQ(PETSC_ERR_SUP,"NormType not supported");
  KSPLogResidualHistory(ksp,gnorm);
  KSPMonitor(ksp,0,gnorm);
  ierr = (*ksp->converged)(ksp,0,gnorm,&ksp->reason,ksp->cnvP);CHKERRQ(ierr); 

  ierr = VecAXPY(X,1.0,Pold);CHKERRQ(ierr);                    /*     x = x + p */

  for (k=0; k<ksp->max_it; k += cg->msize) {
    for (i=0; i<cg->msize && k+i<ksp->max_it; i++) {
      ierr = KSP_MatMult(ksp,Amat,X,R);CHKERRQ(ierr);            /*     r <- b - Ax     */
      ierr = VecAYPX(R,-1.0,B);CHKERRQ(ierr);

      ierr = PCApply(ksp->pc,R,P);CHKERRQ(ierr);                 /*     p = B r */
      if (ksp->normtype == KSP_NORM_UNPRECONDITIONED) {
        ierr = VecNorm(R,NORM_2,&gnorm);CHKERRQ(ierr);          
      } else if (ksp->normtype == KSP_NORM_PRECONDITIONED) {
        ierr = VecNorm(P,NORM_2,&gnorm);CHKERRQ(ierr);          
      } else SETERRQ(PETSC_ERR_SUP,"NormType not supported");
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
/*
       KSPDestroy_Broyden - Frees all memory space used by the Krylov method

*/
#undef __FUNCT__  
#define __FUNCT__ "KSPDestroy_Broyden" 
PetscErrorCode KSPDestroy_Broyden(KSP ksp)
{
  KSP_Broyden    *cg = (KSP_Broyden*)ksp->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = VecDestroyVecs(cg->v,cg->msize);CHKERRQ(ierr);
  ierr = VecDestroyVecs(cg->w,cg->msize);CHKERRQ(ierr);
  ierr = KSPDefaultDestroy(ksp);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*
     KSPView_Broyden - Prints information about the current Krylov method being used

      Currently this only prints information to a file (or stdout) about the 
      symmetry of the problem. If your Krylov method has special options or 
      flags that information should be printed here.

*/
#undef __FUNCT__  
#define __FUNCT__ "KSPView_Broyden" 
PetscErrorCode KSPView_Broyden(KSP ksp,PetscViewer viewer)
{
  KSP_Broyden    *cg = (KSP_Broyden *)ksp->data; 
  PetscErrorCode ierr;
  PetscTruth     iascii;

  PetscFunctionBegin;
  ierr = PetscTypeCompare((PetscObject)viewer,PETSC_VIEWER_ASCII,&iascii);CHKERRQ(ierr);
  if (iascii) {
    ierr = PetscViewerASCIIPrintf(viewer,"  Size of space %d\n",cg->msize);CHKERRQ(ierr);
  } else {
    SETERRQ1(PETSC_ERR_SUP,"Viewer type %s not supported for KSP cg",((PetscObject)viewer)->type_name);
  }
  PetscFunctionReturn(0);
}

/*
    KSPSetFromOptions_Broyden - Checks the options database for options related to the  method
*/ 
#undef __FUNCT__  
#define __FUNCT__ "KSPSetFromOptions_Broyden"
PetscErrorCode KSPSetFromOptions_Broyden(KSP ksp)
{
  PetscErrorCode ierr;
  KSP_Broyden    *cg = (KSP_Broyden *)ksp->data;

  PetscFunctionBegin;
  ierr = PetscOptionsHead("KSP Broyden options");CHKERRQ(ierr);
    ierr = PetscOptionsInt("-ksp_broyden_restart","Number of directions","None",cg->msize,&cg->msize,PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsTail();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*
    KSPCreate_Broyden - Creates the data structure for the Krylov method Broyden and sets the 
       function pointers for all the routines it needs to call (KSPSolve_Broyden() etc)

    It must be wrapped in EXTERN_C_BEGIN to be dynamically linkable in C++
*/
/*MC
     KSPBROYDEN - The preconditioned conjugate gradient (Broyden) iterative method

   Level: beginner

   Notes: Supports only left preconditioning

.seealso:  KSPCreate(), KSPSetType(), KSPType (for list of available types), KSP

M*/
EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "KSPCreate_Broyden"
PetscErrorCode PETSCKSP_DLLEXPORT KSPCreate_Broyden(KSP ksp)
{
  PetscErrorCode ierr;
  KSP_Broyden    *cg;

  PetscFunctionBegin;
  ierr = PetscNewLog(ksp,KSP_Broyden,&cg);CHKERRQ(ierr);
  cg->msize                      = 30;
  cg->csize                      = 0;

  ksp->data                      = (void*)cg;
 if (ksp->pc_side != PC_LEFT) {
    ierr = PetscInfo(ksp,"WARNING! Setting PC_SIDE for Broyden to left!\n");CHKERRQ(ierr);
  }
  ksp->pc_side                   = PC_LEFT;

  /*
       Sets the functions that are associated with this data structure 
       (in C++ this is the same as defining virtual functions)
  */
  ksp->ops->setup                = KSPSetUp_Broyden;
  ksp->ops->solve                = KSPSolve_Broyden;
  ksp->ops->destroy              = KSPDestroy_Broyden;
  ksp->ops->view                 = KSPView_Broyden;
  ksp->ops->setfromoptions       = KSPSetFromOptions_Broyden;
  ksp->ops->buildsolution        = KSPDefaultBuildSolution;
  ksp->ops->buildresidual        = KSPDefaultBuildResidual;
  PetscFunctionReturn(0);
}
EXTERN_C_END




