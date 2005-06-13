#define PETSCMAT_DLL
/*
    ADIC based nonlinear operator object that can be used with FAS

    This does not really belong in the matrix directories but since it 
    was cloned off of Mat_DAAD I'm leaving it here until I have a better place

*/

#include "petscda.h"
#include "petscsys.h"
EXTERN_C_BEGIN
#include "adic/ad_utils.h"
EXTERN_C_END

#include "src/dm/da/daimpl.h"

struct NLF_DAAD {
  DA         da;
  void       *ctx;
  Vec        residual;
  int        newton_its;
};



/*
      Solves the one dimensional equation using Newton's method 
*/
#undef __FUNCT__  
#define __FUNCT__ "NLFNewton_DAAD"
PetscErrorCode NLFNewton_DAAD(NLF A,DALocalInfo *info,MatStencil *stencil,void *ad_vu,PetscScalar *ad_vustart,int nI,int gI,PetscScalar residual)
{
  PetscErrorCode ierr;
  PetscInt       cnt = A->newton_its;
  PetscScalar    ad_f[2],J,f;

  PetscFunctionBegin;
  ad_vustart[1+2*gI] = 1.0;

  do {
    /* compute the function and Jacobian */        
    ierr = (*A->da->adicmf_lfi)(info,stencil,ad_vu,ad_f,A->ctx);CHKERRQ(ierr);
    J    = -ad_f[1];
    f    = -ad_f[0] + residual;
    ad_vustart[2*gI] =  ad_vustart[2*gI] - f/J;
  } while (--cnt > 0 && PetscAbsScalar(f) > 1.e-14);

  ad_vustart[1+2*gI] = 0.0;
  PetscFunctionReturn(0);
}


/*
        Nonlinear relax on all the equations with an initial guess in xx
*/
EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "NLFMatRelax_DAAD"
PetscErrorCode PETSCMAT_DLLEXPORT NLFRelax_DAAD(NLF A,MatSORType flag,int its,Vec xx)
{
  PetscErrorCode ierr;
  PetscInt       j,gtdof,nI,gI;
  PetscScalar    *avu,*av,*ad_vustart,*residual;
  Vec            localxx;
  DALocalInfo    info;
  MatStencil     stencil;
  void*          *ad_vu;

  PetscFunctionBegin;
  if (its <= 0) SETERRQ1(PETSC_ERR_ARG_WRONG,"Relaxation requires global its %D positive",its);

  ierr = DAGetLocalVector(A->da,&localxx);CHKERRQ(ierr);
  /* get space for derivative object.  */
  ierr = DAGetAdicMFArray(A->da,PETSC_TRUE,(void **)&ad_vu,(void**)&ad_vustart,&gtdof);CHKERRQ(ierr);
  ierr = VecGetArray(A->residual,&residual);CHKERRQ(ierr);


  /* tell ADIC we will be computing one dimensional Jacobians */
  PetscADResetIndep();
  ierr = PetscADIncrementTotalGradSize(1);CHKERRQ(ierr);
  PetscADSetIndepDone();

  ierr = DAGetLocalInfo(A->da,&info);CHKERRQ(ierr);
  while (its--) {

    /* get initial solution properly ghosted */
    ierr = DAGlobalToLocalBegin(A->da,xx,INSERT_VALUES,localxx);CHKERRQ(ierr);
    ierr = DAGlobalToLocalEnd(A->da,xx,INSERT_VALUES,localxx);CHKERRQ(ierr);

    /* copy input vector into derivative object */
    ierr = VecGetArray(localxx,&avu);CHKERRQ(ierr);
    for (j=0; j<gtdof; j++) {
      ad_vustart[2*j]   = avu[j];
      ad_vustart[2*j+1] = 0.0;
    }
    ierr = VecRestoreArray(localxx,&avu);CHKERRQ(ierr);

    if (flag & SOR_FORWARD_SWEEP || flag & SOR_LOCAL_FORWARD_SWEEP){
      nI = 0;
      for (stencil.k = info.zs; stencil.k<info.zs+info.zm; stencil.k++) {
        for (stencil.j = info.ys; stencil.j<info.ys+info.ym; stencil.j++) {
          for (stencil.i = info.xs; stencil.i<info.xs+info.xm; stencil.i++) {
            for (stencil.c = 0; stencil.c<info.dof; stencil.c++) {
              gI   = stencil.c + (stencil.i - info.gxs)*info.dof + (stencil.j - info.gys)*info.dof*info.gxm + (stencil.k - info.gzs)*info.dof*info.gxm*info.gym;
              ierr = NLFNewton_DAAD(A,&info,&stencil,ad_vu,ad_vustart,nI,gI,residual[nI]);CHKERRQ(ierr);
              nI++;
            }
          }
        }
      }
    }
    if (flag & SOR_BACKWARD_SWEEP || flag & SOR_LOCAL_BACKWARD_SWEEP){
      nI = info.dof*info.xm*info.ym*info.zm - 1;
      for (stencil.k = info.zs+info.zm-1; stencil.k>=info.zs; stencil.k--) {
        for (stencil.j = info.ys+info.ym-1; stencil.j>=info.ys; stencil.j--) {
          for (stencil.i = info.xs+info.xm-1; stencil.i>=info.xs; stencil.i--) {
            for (stencil.c = info.dof-1; stencil.c>=0; stencil.c--) {
              gI   = stencil.c + (stencil.i - info.gxs)*info.dof + (stencil.j - info.gys)*info.dof*info.gxm + (stencil.k - info.gzs)*info.dof*info.gxm*info.gym;
              ierr = NLFNewton_DAAD(A,&info,&stencil,ad_vu,ad_vustart,nI,gI,residual[nI]);CHKERRQ(ierr);
              nI--;
            }
          }
        }
      }
    }

    /* copy solution back into ghosted vector from derivative object */
    ierr = VecGetArray(localxx,&av);CHKERRQ(ierr);
    for (j=0; j<gtdof; j++) {
      av[j] = ad_vustart[2*j];
    }
    ierr = VecRestoreArray(localxx,&av);CHKERRQ(ierr);
    /* stick relaxed solution back into global solution */
    ierr = DALocalToGlobal(A->da,localxx,INSERT_VALUES,xx);CHKERRQ(ierr);
  }


  ierr = VecRestoreArray(A->residual,&residual);CHKERRQ(ierr);
  ierr = DARestoreLocalVector(A->da,&localxx);CHKERRQ(ierr);
  ierr = DARestoreAdicMFArray(A->da,PETSC_TRUE,(void **)&ad_vu,(void**)&ad_vustart,&gtdof);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
EXTERN_C_END


#undef __FUNCT__  
#define __FUNCT__ "NLFDestroy_DAAD"
PetscErrorCode NLFDestroy_DAAD(NLF A)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = DADestroy(A->da);CHKERRQ(ierr);
  ierr = PetscFree(A);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "NLFDAADSetDA_DAAD"
PetscErrorCode PETSCMAT_DLLEXPORT NLFDAADSetDA_DAAD(NLF A,DA da)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  A->da = da;
  ierr = PetscObjectReference((PetscObject)da);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
EXTERN_C_END

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "NLFDAADSetNewtonIterations_DAAD"
PetscErrorCode PETSCMAT_DLLEXPORT NLFDAADSetNewtonIterations_DAAD(NLF A,int its)
{
  PetscFunctionBegin;
  A->newton_its = its;
  PetscFunctionReturn(0);
}
EXTERN_C_END

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "NLFDAADSetResidual_DAAD"
PetscErrorCode PETSCMAT_DLLEXPORT NLFDAADSetResidual_DAAD(NLF A,Vec residual)
{
  PetscFunctionBegin;
  A->residual = residual;
  PetscFunctionReturn(0);
}
EXTERN_C_END


EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "NLFDAADSetCtx_DAAD"
PetscErrorCode PETSCMAT_DLLEXPORT NLFDAADSetCtx_DAAD(NLF A,void *ctx)
{
  PetscFunctionBegin;
  A->ctx = ctx;
  PetscFunctionReturn(0);
}
EXTERN_C_END

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "NLFCreate_DAAD"
PetscErrorCode PETSCMAT_DLLEXPORT NLFCreate_DAAD(NLF *A)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr    = PetscNew(struct NLF_DAAD,A);CHKERRQ(ierr);
  (*A)->da         = 0;
  (*A)->ctx        = 0;
  (*A)->newton_its = 2;
  PetscFunctionReturn(0);
}
EXTERN_C_END



