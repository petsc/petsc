#define PETSCKSP_DLL

#include "include/private/kspimpl.h"

#undef __FUNCT__  
#define __FUNCT__ "KSPSetUp_IBCGS"
static PetscErrorCode KSPSetUp_IBCGS(KSP ksp)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (ksp->pc_side == PC_SYMMETRIC) {
    SETERRQ(PETSC_ERR_SUP,"no symmetric preconditioning for KSPIBCGS");
  }
  ierr = KSPDefaultGetWork(ksp,9);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* 
    The code below "cheats" from PETSc style
       1) VecRestoreArray() is called immediately after VecGetArray() and the array values are still accessed
       2) The vector operations on done directly on the arrays instead of with VecXXXX() calls
*/
#undef __FUNCT__  
#define __FUNCT__ "KSPSolve_IBCGS"
static PetscErrorCode  KSPSolve_IBCGS(KSP ksp)
{
  PetscErrorCode ierr;
  PetscInt       i,N;
  PetscReal      rnorm;
  PetscScalar    insums[6],outsums[6];
  PetscScalar    sigman_2, sigman_1, sigman, pin_1, pin, phin_1, phin;
  PetscScalar    taun_1, taun, rhon_1, rhon, alphan_1, alphan, omegan_1, omegan;
  PetscScalar    *r0, *rn_1,*rn,*xn_1, *xn, *f0, *vn_1, *vn,*zn_1, *zn, *qn_1, *qn, *b, *un_1, *un;
  /* the rest do not have to keep n_1 values */
  PetscScalar    kappan, thetan, etan, gamman, betan, deltan;
  PetscScalar    *tn, *sn;
  Vec            R0,Rn_1,Rn,Xn_1,Xn,F0,Vn_1,Vn,Zn_1,Zn,Qn_1,Qn,Tn,Sn,B,Un_1,Un;
  Mat            A;

  PetscFunctionBegin;
  ierr = PCGetOperators(ksp->pc,&A,PETSC_NULL,PETSC_NULL);CHKERRQ(ierr);
  ierr = VecGetLocalSize(ksp->vec_sol,&N);CHKERRQ(ierr);
  Xn_1 = Xn = ksp->vec_sol;ierr = VecGetArray(Xn_1,&xn_1);CHKERRQ(ierr);ierr = VecRestoreArray(Xn_1,&xn_1);CHKERRQ(ierr);xn = xn_1;
  B         = ksp->vec_rhs;ierr = VecGetArray(B,&b);ierr = VecRestoreArray(B,&b);CHKERRQ(ierr);
  R0        = ksp->work[0];ierr = VecGetArray(R0,&r0);CHKERRQ(ierr);ierr = VecRestoreArray(R0,&r0);
  Rn_1 = Rn = ksp->work[1];ierr = VecGetArray(Rn_1,&rn_1);CHKERRQ(ierr);ierr = VecRestoreArray(Rn_1,&rn_1);CHKERRQ(ierr);rn = rn_1;
  Un_1 = Un = ksp->work[2];ierr = VecGetArray(Un_1,&un_1);CHKERRQ(ierr);ierr = VecRestoreArray(Un_1,&un_1);CHKERRQ(ierr);un = un_1;
  F0        = ksp->work[3];ierr = VecGetArray(F0,&f0);CHKERRQ(ierr);ierr = VecRestoreArray(F0,&f0);CHKERRQ(ierr);
  Vn_1 = Vn = ksp->work[4];ierr = VecGetArray(Vn_1,&vn_1);CHKERRQ(ierr);ierr = VecRestoreArray(Vn_1,&vn_1);CHKERRQ(ierr);vn = vn_1;
  Zn_1 = Zn = ksp->work[5];ierr = VecGetArray(Zn_1,&zn_1);CHKERRQ(ierr);ierr = VecRestoreArray(Zn_1,&zn_1);CHKERRQ(ierr);zn = zn_1;
  Qn_1 = Qn = ksp->work[6];ierr = VecGetArray(Qn_1,&qn_1);CHKERRQ(ierr);ierr = VecRestoreArray(Qn_1,&qn_1);CHKERRQ(ierr);qn = qn_1;
  Tn        = ksp->work[7];ierr = VecGetArray(Tn,&tn);CHKERRQ(ierr);ierr = VecRestoreArray(Tn,&tn);CHKERRQ(ierr);
  Sn        = ksp->work[8];ierr = VecGetArray(Sn,&sn);CHKERRQ(ierr);ierr = VecRestoreArray(Sn,&sn);CHKERRQ(ierr);

  /* r0 = rn_1 = b - A*xn_1; */
  /* ierr = KSP_PCApplyBAorAB(ksp,Xn_1,Rn_1,Tn);CHKERRQ(ierr);
     ierr = VecAYPX(Rn_1,-1.0,B);CHKERRQ(ierr); */
  ierr = KSPInitialResidual(ksp,Xn_1,Tn,Sn,Rn_1,B);CHKERRQ(ierr);

  ierr = VecNorm(Rn_1,NORM_2,&rnorm);CHKERRQ(ierr);
  KSPMonitor(ksp,0,rnorm);
  ierr = (*ksp->converged)(ksp,0,rnorm,&ksp->reason,ksp->cnvP);CHKERRQ(ierr);   
  if (ksp->reason) PetscFunctionReturn(0);

  ierr = VecCopy(Rn_1,R0);CHKERRQ(ierr);

  /* un_1 = A*rn_1; */
  ierr = KSP_PCApplyBAorAB(ksp,Rn_1,Un_1,Tn);CHKERRQ(ierr);
  
  /* f0   = A'*rn_1; */
  ierr = KSP_PCApplyBAorABTranspose(ksp,Rn_1,F0,Tn);CHKERRQ(ierr);

  /*qn_1 = vn_1 = zn_1 = 0.0; */
  ierr = VecSet(Qn_1,0.0);CHKERRQ(ierr);
  ierr = VecSet(Vn_1,0.0);CHKERRQ(ierr);
  ierr = VecSet(Zn_1,0.0);CHKERRQ(ierr);

  sigman_2 = pin_1 = phin_1 = taun_1 = 0.0;

  ierr = VecDot(R0,R0,&phin_1);CHKERRQ(ierr); 

  /* sigman_1 = rn_1'un_1  */
  ierr = VecDot(R0,Un_1,&sigman_1);CHKERRQ(ierr); 

  rhon_1 = alphan_1 = omegan_1 = 1.0;

  for (ksp->its = 1; ksp->its<ksp->max_it+1; ksp->its++) {
    rhon   = phin_1 - omegan_1*sigman_2 + omegan_1*alphan_1*pin_1;
    //    if (rhon == 0.0) SETERRQ1(PETSC_ERR_CONV_FAILED,"rhon is zero, iteration %D",n);
    if (ksp->its == 1) deltan = rhon;
    else deltan = rhon/taun_1;
    betan  = deltan/omegan_1;
    taun   = sigman_1 + betan*taun_1  - deltan*pin_1;
    if (taun == 0.0) SETERRQ1(PETSC_ERR_CONV_FAILED,"taun is zero, iteration %D",ksp->its);
    alphan = rhon/taun;
    printf("phin_1 rhon deltan betan taun alphan %g %g %g %g %g %g\n",phin_1,rhon,deltan,betan,taun,alphan);

    /*  
        zn = alphan*rn_1 + betan*zn_1 - alphan*deltan*vn_1
        vn = un_1 + betan*vn_1 - deltan*qn_1
        sn = rn_1 - alphan*vn
    */
    for (i=0; i<N; i++) {
      zn[i] = alphan*rn_1[i] + (alphan/alphan_1)*betan*zn_1[i] - alphan*deltan*vn_1[i];
      vn[i] = un_1[i] + betan*vn_1[i] - deltan*qn_1[i];
      sn[i] = rn_1[i] - alphan*vn[i];
    }

    /*
        qn = A*vn
    */
    ierr = KSP_PCApplyBAorAB(ksp,Vn,Qn,Tn);CHKERRQ(ierr);

    /*
        tn = un_1 - alphan*qn
    */
    ierr = VecWAXPY(Tn,-alphan,Qn,Un_1);CHKERRQ(ierr);
      

    /*
        phin = r0'sn
        pin  = r0'qn
        gamman = f0'sn
        etan   = f0'tn
        thetan = sn'tn
        kappan = tn'tn
    */
    phin = pin = gamman = etan = thetan = kappan = 0.0;
    for (i=0; i<N; i++) {
      phin += r0[i]*sn[i];
      pin  += r0[i]*qn[i];
      gamman += f0[i]*sn[i];
      etan   += f0[i]*tn[i];
      thetan += sn[i]*tn[i];
      kappan += tn[i]*tn[i];
    }
    insums[0] = phin;
    insums[1] = pin;
    insums[2] = gamman;
    insums[3] = etan;
    insums[4] = thetan;
    insums[5] = kappan;
    ierr = MPI_Allreduce(insums,outsums,6,MPIU_SCALAR,MPI_SUM,ksp->hdr.comm);CHKERRQ(ierr);
    phin     = outsums[0];
    pin      = outsums[1];
    gamman   = outsums[2];
    etan     = outsums[3];
    thetan   = outsums[4];
    kappan   = outsums[5];

    if (kappan == 0.0) SETERRQ1(PETSC_ERR_CONV_FAILED,"kappan is zero, iteration %D",ksp->its);
    if (thetan == 0.0) SETERRQ1(PETSC_ERR_CONV_FAILED,"thetan is zero, iteration %D",ksp->its);
    omegan = thetan/kappan;
    sigman = gamman - omegan*etan;

VecView(Xn,0);
 printf("omega %g\n",omegan);
    /*
        rn = sn - omegan*tn
        xn = xn_1 + zn + omegan*sn
    */
    rnorm = 0.0;
    for (i=0; i<N; i++) {
      rn[i] = sn[i] - omegan*tn[i];
      rnorm += PetscRealPart(PetscConj(rn[i])*rn[i]);
      xn[i] += zn[i] + omegan*sn[i];
    }
    rnorm = sqrt(rnorm);

VecView(Xn,0);

    /* Test for convergence */
    KSPMonitor(ksp,ksp->its,rnorm);
    ierr = (*ksp->converged)(ksp,ksp->its,rnorm,&ksp->reason,ksp->cnvP);CHKERRQ(ierr);   
    if (ksp->reason) break;
 
    /* un = A*rn */
    ierr = KSP_PCApplyBAorAB(ksp,Rn,Un,Tn);CHKERRQ(ierr);   

    /* Update n-1 locations with n locations */
    sigman_2 = sigman_1;
    sigman_1 = sigman;
    pin_1    = pin;
    phin_1   = phin;
    alphan_1 = alphan;
    taun_1   = taun;
    rhon_1   = rhon;
    omegan_1 = omegan;
  }
  if (ksp->its >= ksp->max_it) {
    ksp->reason = KSP_DIVERGED_ITS;
  }
  ierr = KSPUnwindPreconditioner(ksp,Xn,Tn);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


/*MC
     KSPIBCGS - Implements the IBiCGStab (Improved Stabilized version of BiConjugate Gradient Squared) method
            in an alternative form to have only a single global reduction operation instead of the usual 3 (or 4)

   Options Database Keys:
.   see KSPSolve()

   Level: beginner

   Notes: Reference: The Improved BiCGStab Method for Large and Sparse Unsymmetric Linear Systems on Parallel Distributed Memory
                     Architectures. L. T. Yand and R. Brent, Proceedings of the Fifth International Conference on Algorithms and 
                     Architectures for Parallel Processing, 2002, IEEE.
          See KSPBCGSL for additional stabilization

          Unlike the Bi-CG-stab algorithm, this requires one multiplication be the transpose of the operator
           before the iteration starts.

.seealso:  KSPCreate(), KSPSetType(), KSPType (for list of available types), KSP, KSPBICG, KSPBCGSL, KSPIBCGS
M*/
EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "KSPCreate_IBCGS"
PetscErrorCode PETSCKSP_DLLEXPORT KSPCreate_IBCGS(KSP ksp)
{
  PetscFunctionBegin;
  ksp->data                 = (void*)0;
  ksp->pc_side              = PC_LEFT;
  ksp->ops->setup           = KSPSetUp_IBCGS;
  ksp->ops->solve           = KSPSolve_IBCGS;
  ksp->ops->destroy         = KSPDefaultDestroy;
  ksp->ops->buildsolution   = KSPDefaultBuildSolution;
  ksp->ops->buildresidual   = KSPDefaultBuildResidual;
  ksp->ops->setfromoptions  = 0;
  ksp->ops->view            = 0;
  PetscFunctionReturn(0);
}
EXTERN_C_END
