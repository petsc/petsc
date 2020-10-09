#include <petsctaolinesearch.h>
#include <../src/tao/constrained/impls/ipm/ipm.h> /*I "ipm.h" I*/

/*
   x,d in R^n
   f in R
   nb = mi + nlb+nub
   s in R^nb is slack vector CI(x) / x-XL / -x+XU
   bin in R^mi (tao->constraints_inequality)
   beq in R^me (tao->constraints_equality)
   lamdai in R^nb (ipmP->lamdai)
   lamdae in R^me (ipmP->lamdae)
   Jeq in R^(me x n) (tao->jacobian_equality)
   Jin in R^(mi x n) (tao->jacobian_inequality)
   Ai in  R^(nb x n) (ipmP->Ai)
   H in R^(n x n) (tao->hessian)
   min f=(1/2)*x'*H*x + d'*x
   s.t.  CE(x) == 0
         CI(x) >= 0
         x >= tao->XL
         -x >= -tao->XU
*/

static PetscErrorCode IPMComputeKKT(Tao tao);
static PetscErrorCode IPMPushInitialPoint(Tao tao);
static PetscErrorCode IPMEvaluate(Tao tao);
static PetscErrorCode IPMUpdateK(Tao tao);
static PetscErrorCode IPMUpdateAi(Tao tao);
static PetscErrorCode IPMGatherRHS(Tao tao,Vec,Vec,Vec,Vec,Vec);
static PetscErrorCode IPMScatterStep(Tao tao,Vec,Vec,Vec,Vec,Vec);
static PetscErrorCode IPMInitializeBounds(Tao tao);

static PetscErrorCode TaoSolve_IPM(Tao tao)
{
  PetscErrorCode     ierr;
  TAO_IPM            *ipmP = (TAO_IPM*)tao->data;
  PetscInt           its,i;
  PetscScalar        stepsize=1.0;
  PetscScalar        step_s,step_l,alpha,tau,sigma,phi_target;

  PetscFunctionBegin;
  /* Push initial point away from bounds */
  ierr = IPMInitializeBounds(tao);CHKERRQ(ierr);
  ierr = IPMPushInitialPoint(tao);CHKERRQ(ierr);
  ierr = VecCopy(tao->solution,ipmP->rhs_x);CHKERRQ(ierr);
  ierr = IPMEvaluate(tao);CHKERRQ(ierr);
  ierr = IPMComputeKKT(tao);CHKERRQ(ierr);

  tao->reason = TAO_CONTINUE_ITERATING;
  ierr = TaoLogConvergenceHistory(tao,ipmP->kkt_f,ipmP->phi,0.0,tao->ksp_its);CHKERRQ(ierr);
  ierr = TaoMonitor(tao,tao->niter,ipmP->kkt_f,ipmP->phi,0.0,1.0);CHKERRQ(ierr);
  ierr = (*tao->ops->convergencetest)(tao,tao->cnvP);CHKERRQ(ierr);

  while (tao->reason == TAO_CONTINUE_ITERATING) {
    /* Call general purpose update function */
    if (tao->ops->update) {
      ierr = (*tao->ops->update)(tao, tao->niter, tao->user_update);CHKERRQ(ierr);
    }

    tao->ksp_its=0;
    ierr = IPMUpdateK(tao);CHKERRQ(ierr);
    /*
       rhs.x    = -rd
       rhs.lame = -rpe
       rhs.lami = -rpi
       rhs.com  = -com
    */

    ierr = VecCopy(ipmP->rd,ipmP->rhs_x);CHKERRQ(ierr);
    if (ipmP->me > 0) {
      ierr = VecCopy(ipmP->rpe,ipmP->rhs_lamdae);CHKERRQ(ierr);
    }
    if (ipmP->nb > 0) {
      ierr = VecCopy(ipmP->rpi,ipmP->rhs_lamdai);CHKERRQ(ierr);
      ierr = VecCopy(ipmP->complementarity,ipmP->rhs_s);CHKERRQ(ierr);
    }
    ierr = IPMGatherRHS(tao,ipmP->bigrhs,ipmP->rhs_x,ipmP->rhs_lamdae,ipmP->rhs_lamdai,ipmP->rhs_s);CHKERRQ(ierr);
    ierr = VecScale(ipmP->bigrhs,-1.0);CHKERRQ(ierr);

    /* solve K * step = rhs */
    ierr = KSPSetOperators(tao->ksp,ipmP->K,ipmP->K);CHKERRQ(ierr);
    ierr = KSPSolve(tao->ksp,ipmP->bigrhs,ipmP->bigstep);CHKERRQ(ierr);

    ierr = IPMScatterStep(tao,ipmP->bigstep,tao->stepdirection,ipmP->ds,ipmP->dlamdae,ipmP->dlamdai);CHKERRQ(ierr);
    ierr = KSPGetIterationNumber(tao->ksp,&its);CHKERRQ(ierr);
    tao->ksp_its += its;
    tao->ksp_tot_its+=its;
     /* Find distance along step direction to closest bound */
    if (ipmP->nb > 0) {
      ierr = VecStepBoundInfo(ipmP->s,ipmP->ds,ipmP->Zero_nb,ipmP->Inf_nb,&step_s,NULL,NULL);CHKERRQ(ierr);
      ierr = VecStepBoundInfo(ipmP->lamdai,ipmP->dlamdai,ipmP->Zero_nb,ipmP->Inf_nb,&step_l,NULL,NULL);CHKERRQ(ierr);
      alpha = PetscMin(step_s,step_l);
      alpha = PetscMin(alpha,1.0);
      ipmP->alpha1 = alpha;
    } else {
      ipmP->alpha1 = alpha = 1.0;
    }

    /* x_aff = x + alpha*d */
    ierr = VecCopy(tao->solution,ipmP->save_x);CHKERRQ(ierr);
    if (ipmP->me > 0) {
      ierr = VecCopy(ipmP->lamdae,ipmP->save_lamdae);CHKERRQ(ierr);
    }
    if (ipmP->nb > 0) {
      ierr = VecCopy(ipmP->lamdai,ipmP->save_lamdai);CHKERRQ(ierr);
      ierr = VecCopy(ipmP->s,ipmP->save_s);CHKERRQ(ierr);
    }

    ierr = VecAXPY(tao->solution,alpha,tao->stepdirection);CHKERRQ(ierr);
    if (ipmP->me > 0) {
      ierr = VecAXPY(ipmP->lamdae,alpha,ipmP->dlamdae);CHKERRQ(ierr);
    }
    if (ipmP->nb > 0) {
      ierr = VecAXPY(ipmP->lamdai,alpha,ipmP->dlamdai);CHKERRQ(ierr);
      ierr = VecAXPY(ipmP->s,alpha,ipmP->ds);CHKERRQ(ierr);
    }

    /* Recompute kkt to find centering parameter sigma = (new_mu/old_mu)^3 */
    if (ipmP->mu == 0.0) {
      sigma = 0.0;
    } else {
      sigma = 1.0/ipmP->mu;
    }
    ierr = IPMComputeKKT(tao);CHKERRQ(ierr);
    sigma *= ipmP->mu;
    sigma*=sigma*sigma;

    /* revert kkt info */
    ierr = VecCopy(ipmP->save_x,tao->solution);CHKERRQ(ierr);
    if (ipmP->me > 0) {
      ierr = VecCopy(ipmP->save_lamdae,ipmP->lamdae);CHKERRQ(ierr);
    }
    if (ipmP->nb > 0) {
      ierr = VecCopy(ipmP->save_lamdai,ipmP->lamdai);CHKERRQ(ierr);
      ierr = VecCopy(ipmP->save_s,ipmP->s);CHKERRQ(ierr);
    }
    ierr = IPMComputeKKT(tao);CHKERRQ(ierr);

    /* update rhs with new complementarity vector */
    if (ipmP->nb > 0) {
      ierr = VecCopy(ipmP->complementarity,ipmP->rhs_s);CHKERRQ(ierr);
      ierr = VecScale(ipmP->rhs_s,-1.0);CHKERRQ(ierr);
      ierr = VecShift(ipmP->rhs_s,sigma*ipmP->mu);CHKERRQ(ierr);
    }
    ierr = IPMGatherRHS(tao,ipmP->bigrhs,NULL,NULL,NULL,ipmP->rhs_s);CHKERRQ(ierr);

    /* solve K * step = rhs */
    ierr = KSPSetOperators(tao->ksp,ipmP->K,ipmP->K);CHKERRQ(ierr);
    ierr = KSPSolve(tao->ksp,ipmP->bigrhs,ipmP->bigstep);CHKERRQ(ierr);

    ierr = IPMScatterStep(tao,ipmP->bigstep,tao->stepdirection,ipmP->ds,ipmP->dlamdae,ipmP->dlamdai);CHKERRQ(ierr);
    ierr = KSPGetIterationNumber(tao->ksp,&its);CHKERRQ(ierr);
    tao->ksp_its += its;
    tao->ksp_tot_its+=its;
    if (ipmP->nb > 0) {
      /* Get max step size and apply frac-to-boundary */
      tau = PetscMax(ipmP->taumin,1.0-ipmP->mu);
      tau = PetscMin(tau,1.0);
      if (tau != 1.0) {
        ierr = VecScale(ipmP->s,tau);CHKERRQ(ierr);
        ierr = VecScale(ipmP->lamdai,tau);CHKERRQ(ierr);
      }
      ierr = VecStepBoundInfo(ipmP->s,ipmP->ds,ipmP->Zero_nb,ipmP->Inf_nb,&step_s,NULL,NULL);CHKERRQ(ierr);
      ierr = VecStepBoundInfo(ipmP->lamdai,ipmP->dlamdai,ipmP->Zero_nb,ipmP->Inf_nb,&step_l,NULL,NULL);CHKERRQ(ierr);
      if (tau != 1.0) {
        ierr = VecCopy(ipmP->save_s,ipmP->s);CHKERRQ(ierr);
        ierr = VecCopy(ipmP->save_lamdai,ipmP->lamdai);CHKERRQ(ierr);
      }
      alpha = PetscMin(step_s,step_l);
      alpha = PetscMin(alpha,1.0);
    } else {
      alpha = 1.0;
    }
    ipmP->alpha2 = alpha;
    /* TODO make phi_target meaningful */
    phi_target = ipmP->dec * ipmP->phi;
    for (i=0; i<11;i++) {
      ierr = VecAXPY(tao->solution,alpha,tao->stepdirection);CHKERRQ(ierr);
      if (ipmP->nb > 0) {
        ierr = VecAXPY(ipmP->s,alpha,ipmP->ds);CHKERRQ(ierr);
        ierr = VecAXPY(ipmP->lamdai,alpha,ipmP->dlamdai);CHKERRQ(ierr);
      }
      if (ipmP->me > 0) {
        ierr = VecAXPY(ipmP->lamdae,alpha,ipmP->dlamdae);CHKERRQ(ierr);
      }

      /* update dual variables */
      if (ipmP->me > 0) {
        ierr = VecCopy(ipmP->lamdae,tao->DE);CHKERRQ(ierr);
      }

      ierr = IPMEvaluate(tao);CHKERRQ(ierr);
      ierr = IPMComputeKKT(tao);CHKERRQ(ierr);
      if (ipmP->phi <= phi_target) break;
      alpha /= 2.0;
    }

    ierr = TaoLogConvergenceHistory(tao,ipmP->kkt_f,ipmP->phi,0.0,tao->ksp_its);CHKERRQ(ierr);
    ierr = TaoMonitor(tao,tao->niter,ipmP->kkt_f,ipmP->phi,0.0,stepsize);CHKERRQ(ierr);
    ierr = (*tao->ops->convergencetest)(tao,tao->cnvP);CHKERRQ(ierr);
    tao->niter++;
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode TaoSetup_IPM(Tao tao)
{
  TAO_IPM        *ipmP = (TAO_IPM*)tao->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ipmP->nb = ipmP->mi = ipmP->me = 0;
  ipmP->K = NULL;
  ierr = VecGetSize(tao->solution,&ipmP->n);CHKERRQ(ierr);
  if (!tao->gradient) {
    ierr = VecDuplicate(tao->solution, &tao->gradient);CHKERRQ(ierr);
    ierr = VecDuplicate(tao->solution, &tao->stepdirection);CHKERRQ(ierr);
    ierr = VecDuplicate(tao->solution, &ipmP->rd);CHKERRQ(ierr);
    ierr = VecDuplicate(tao->solution, &ipmP->rhs_x);CHKERRQ(ierr);
    ierr = VecDuplicate(tao->solution, &ipmP->work);CHKERRQ(ierr);
    ierr = VecDuplicate(tao->solution, &ipmP->save_x);CHKERRQ(ierr);
  }
  if (tao->constraints_equality) {
    ierr = VecGetSize(tao->constraints_equality,&ipmP->me);CHKERRQ(ierr);
    ierr = VecDuplicate(tao->constraints_equality,&ipmP->lamdae);CHKERRQ(ierr);
    ierr = VecDuplicate(tao->constraints_equality,&ipmP->dlamdae);CHKERRQ(ierr);
    ierr = VecDuplicate(tao->constraints_equality,&ipmP->rhs_lamdae);CHKERRQ(ierr);
    ierr = VecDuplicate(tao->constraints_equality,&ipmP->save_lamdae);CHKERRQ(ierr);
    ierr = VecDuplicate(tao->constraints_equality,&ipmP->rpe);CHKERRQ(ierr);
    ierr = VecDuplicate(tao->constraints_equality,&tao->DE);CHKERRQ(ierr);
  }
  if (tao->constraints_inequality) {
    ierr = VecDuplicate(tao->constraints_inequality,&tao->DI);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode IPMInitializeBounds(Tao tao)
{
  TAO_IPM        *ipmP = (TAO_IPM*)tao->data;
  Vec            xtmp;
  PetscInt       xstart,xend;
  PetscInt       ucstart,ucend; /* user ci */
  PetscInt       ucestart,uceend; /* user ce */
  PetscInt       sstart = 0 ,send = 0;
  PetscInt       bigsize;
  PetscInt       i,counter,nloc;
  PetscInt       *cind,*xind,*ucind,*uceind,*stepind;
  VecType        vtype;
  const PetscInt *xli,*xui;
  PetscInt       xl_offset,xu_offset;
  IS             bigxl,bigxu,isuc,isc,isx,sis,is1;
  PetscErrorCode ierr;
  MPI_Comm       comm;

  PetscFunctionBegin;
  cind=xind=ucind=uceind=stepind=NULL;
  ipmP->mi=0;
  ipmP->nxlb=0;
  ipmP->nxub=0;
  ipmP->nb=0;
  ipmP->nslack=0;

  ierr = VecDuplicate(tao->solution,&xtmp);CHKERRQ(ierr);
  if (!tao->XL && !tao->XU && tao->ops->computebounds) {
    ierr = TaoComputeVariableBounds(tao);CHKERRQ(ierr);
  }
  if (tao->XL) {
    ierr = VecSet(xtmp,PETSC_NINFINITY);CHKERRQ(ierr);
    ierr = VecWhichGreaterThan(tao->XL,xtmp,&ipmP->isxl);CHKERRQ(ierr);
    ierr = ISGetSize(ipmP->isxl,&ipmP->nxlb);CHKERRQ(ierr);
  } else {
    ipmP->nxlb=0;
  }
  if (tao->XU) {
    ierr = VecSet(xtmp,PETSC_INFINITY);CHKERRQ(ierr);
    ierr = VecWhichLessThan(tao->XU,xtmp,&ipmP->isxu);CHKERRQ(ierr);
    ierr = ISGetSize(ipmP->isxu,&ipmP->nxub);CHKERRQ(ierr);
  } else {
    ipmP->nxub=0;
  }
  ierr = VecDestroy(&xtmp);CHKERRQ(ierr);
  if (tao->constraints_inequality) {
    ierr = VecGetSize(tao->constraints_inequality,&ipmP->mi);CHKERRQ(ierr);
  } else {
    ipmP->mi = 0;
  }
  ipmP->nb = ipmP->nxlb + ipmP->nxub + ipmP->mi;

  ierr = PetscObjectGetComm((PetscObject)tao->solution,&comm);CHKERRQ(ierr);

  bigsize = ipmP->n+2*ipmP->nb+ipmP->me;
  ierr = PetscMalloc1(bigsize,&stepind);CHKERRQ(ierr);
  ierr = PetscMalloc1(ipmP->n,&xind);CHKERRQ(ierr);
  ierr = PetscMalloc1(ipmP->me,&uceind);CHKERRQ(ierr);
  ierr = VecGetOwnershipRange(tao->solution,&xstart,&xend);CHKERRQ(ierr);

  if (ipmP->nb > 0) {
    ierr = VecCreate(comm,&ipmP->s);CHKERRQ(ierr);
    ierr = VecSetSizes(ipmP->s,PETSC_DECIDE,ipmP->nb);CHKERRQ(ierr);
    ierr = VecSetFromOptions(ipmP->s);CHKERRQ(ierr);
    ierr = VecDuplicate(ipmP->s,&ipmP->ds);CHKERRQ(ierr);
    ierr = VecDuplicate(ipmP->s,&ipmP->rhs_s);CHKERRQ(ierr);
    ierr = VecDuplicate(ipmP->s,&ipmP->complementarity);CHKERRQ(ierr);
    ierr = VecDuplicate(ipmP->s,&ipmP->ci);CHKERRQ(ierr);

    ierr = VecDuplicate(ipmP->s,&ipmP->lamdai);CHKERRQ(ierr);
    ierr = VecDuplicate(ipmP->s,&ipmP->dlamdai);CHKERRQ(ierr);
    ierr = VecDuplicate(ipmP->s,&ipmP->rhs_lamdai);CHKERRQ(ierr);
    ierr = VecDuplicate(ipmP->s,&ipmP->save_lamdai);CHKERRQ(ierr);

    ierr = VecDuplicate(ipmP->s,&ipmP->save_s);CHKERRQ(ierr);
    ierr = VecDuplicate(ipmP->s,&ipmP->rpi);CHKERRQ(ierr);
    ierr = VecDuplicate(ipmP->s,&ipmP->Zero_nb);CHKERRQ(ierr);
    ierr = VecSet(ipmP->Zero_nb,0.0);CHKERRQ(ierr);
    ierr = VecDuplicate(ipmP->s,&ipmP->One_nb);CHKERRQ(ierr);
    ierr = VecSet(ipmP->One_nb,1.0);CHKERRQ(ierr);
    ierr = VecDuplicate(ipmP->s,&ipmP->Inf_nb);CHKERRQ(ierr);
    ierr = VecSet(ipmP->Inf_nb,PETSC_INFINITY);CHKERRQ(ierr);

    ierr = PetscMalloc1(ipmP->nb,&cind);CHKERRQ(ierr);
    ierr = PetscMalloc1(ipmP->mi,&ucind);CHKERRQ(ierr);
    ierr = VecGetOwnershipRange(ipmP->s,&sstart,&send);CHKERRQ(ierr);

    if (ipmP->mi > 0) {
      ierr = VecGetOwnershipRange(tao->constraints_inequality,&ucstart,&ucend);CHKERRQ(ierr);
      counter=0;
      for (i=ucstart;i<ucend;i++) {
        cind[counter++] = i;
      }
      ierr = ISCreateGeneral(comm,counter,cind,PETSC_COPY_VALUES,&isuc);CHKERRQ(ierr);
      ierr = ISCreateGeneral(comm,counter,cind,PETSC_COPY_VALUES,&isc);CHKERRQ(ierr);
      ierr = VecScatterCreate(tao->constraints_inequality,isuc,ipmP->ci,isc,&ipmP->ci_scat);CHKERRQ(ierr);

      ierr = ISDestroy(&isuc);CHKERRQ(ierr);
      ierr = ISDestroy(&isc);CHKERRQ(ierr);
    }
    /* need to know how may xbound indices are on each process */
    /* TODO better way */
    if (ipmP->nxlb) {
      ierr = ISAllGather(ipmP->isxl,&bigxl);CHKERRQ(ierr);
      ierr = ISGetIndices(bigxl,&xli);CHKERRQ(ierr);
      /* find offsets for this processor */
      xl_offset = ipmP->mi;
      for (i=0;i<ipmP->nxlb;i++) {
        if (xli[i] < xstart) {
          xl_offset++;
        } else break;
      }
      ierr = ISRestoreIndices(bigxl,&xli);CHKERRQ(ierr);

      ierr = ISGetIndices(ipmP->isxl,&xli);CHKERRQ(ierr);
      ierr = ISGetLocalSize(ipmP->isxl,&nloc);CHKERRQ(ierr);
      for (i=0;i<nloc;i++) {
        xind[i] = xli[i];
        cind[i] = xl_offset+i;
      }

      ierr = ISCreateGeneral(comm,nloc,xind,PETSC_COPY_VALUES,&isx);CHKERRQ(ierr);
      ierr = ISCreateGeneral(comm,nloc,cind,PETSC_COPY_VALUES,&isc);CHKERRQ(ierr);
      ierr = VecScatterCreate(tao->XL,isx,ipmP->ci,isc,&ipmP->xl_scat);CHKERRQ(ierr);
      ierr = ISDestroy(&isx);CHKERRQ(ierr);
      ierr = ISDestroy(&isc);CHKERRQ(ierr);
      ierr = ISDestroy(&bigxl);CHKERRQ(ierr);
    }

    if (ipmP->nxub) {
      ierr = ISAllGather(ipmP->isxu,&bigxu);CHKERRQ(ierr);
      ierr = ISGetIndices(bigxu,&xui);CHKERRQ(ierr);
      /* find offsets for this processor */
      xu_offset = ipmP->mi + ipmP->nxlb;
      for (i=0;i<ipmP->nxub;i++) {
        if (xui[i] < xstart) {
          xu_offset++;
        } else break;
      }
      ierr = ISRestoreIndices(bigxu,&xui);CHKERRQ(ierr);

      ierr = ISGetIndices(ipmP->isxu,&xui);CHKERRQ(ierr);
      ierr = ISGetLocalSize(ipmP->isxu,&nloc);CHKERRQ(ierr);
      for (i=0;i<nloc;i++) {
        xind[i] = xui[i];
        cind[i] = xu_offset+i;
      }

      ierr = ISCreateGeneral(comm,nloc,xind,PETSC_COPY_VALUES,&isx);CHKERRQ(ierr);
      ierr = ISCreateGeneral(comm,nloc,cind,PETSC_COPY_VALUES,&isc);CHKERRQ(ierr);
      ierr = VecScatterCreate(tao->XU,isx,ipmP->ci,isc,&ipmP->xu_scat);CHKERRQ(ierr);
      ierr = ISDestroy(&isx);CHKERRQ(ierr);
      ierr = ISDestroy(&isc);CHKERRQ(ierr);
      ierr = ISDestroy(&bigxu);CHKERRQ(ierr);
    }
  }
  ierr = VecCreate(comm,&ipmP->bigrhs);CHKERRQ(ierr);
  ierr = VecGetType(tao->solution,&vtype);CHKERRQ(ierr);
  ierr = VecSetType(ipmP->bigrhs,vtype);CHKERRQ(ierr);
  ierr = VecSetSizes(ipmP->bigrhs,PETSC_DECIDE,bigsize);CHKERRQ(ierr);
  ierr = VecSetFromOptions(ipmP->bigrhs);CHKERRQ(ierr);
  ierr = VecDuplicate(ipmP->bigrhs,&ipmP->bigstep);CHKERRQ(ierr);

  /* create scatters for step->x and x->rhs */
  for (i=xstart;i<xend;i++) {
    stepind[i-xstart] = i;
    xind[i-xstart] = i;
  }
  ierr = ISCreateGeneral(comm,xend-xstart,stepind,PETSC_COPY_VALUES,&sis);CHKERRQ(ierr);
  ierr = ISCreateGeneral(comm,xend-xstart,xind,PETSC_COPY_VALUES,&is1);CHKERRQ(ierr);
  ierr = VecScatterCreate(ipmP->bigstep,sis,tao->solution,is1,&ipmP->step1);CHKERRQ(ierr);
  ierr = VecScatterCreate(tao->solution,is1,ipmP->bigrhs,sis,&ipmP->rhs1);CHKERRQ(ierr);
  ierr = ISDestroy(&sis);CHKERRQ(ierr);
  ierr = ISDestroy(&is1);CHKERRQ(ierr);

  if (ipmP->nb > 0) {
    for (i=sstart;i<send;i++) {
      stepind[i-sstart] = i+ipmP->n;
      cind[i-sstart] = i;
    }
    ierr = ISCreateGeneral(comm,send-sstart,stepind,PETSC_COPY_VALUES,&sis);CHKERRQ(ierr);
    ierr = ISCreateGeneral(comm,send-sstart,cind,PETSC_COPY_VALUES,&is1);CHKERRQ(ierr);
    ierr = VecScatterCreate(ipmP->bigstep,sis,ipmP->s,is1,&ipmP->step2);CHKERRQ(ierr);
    ierr = ISDestroy(&sis);CHKERRQ(ierr);

    for (i=sstart;i<send;i++) {
      stepind[i-sstart] = i+ipmP->n+ipmP->me;
      cind[i-sstart] = i;
    }
    ierr = ISCreateGeneral(comm,send-sstart,stepind,PETSC_COPY_VALUES,&sis);CHKERRQ(ierr);
    ierr = VecScatterCreate(ipmP->s,is1,ipmP->bigrhs,sis,&ipmP->rhs3);CHKERRQ(ierr);
    ierr = ISDestroy(&sis);CHKERRQ(ierr);
    ierr = ISDestroy(&is1);CHKERRQ(ierr);
  }

  if (ipmP->me > 0) {
    ierr = VecGetOwnershipRange(tao->constraints_equality,&ucestart,&uceend);CHKERRQ(ierr);
    for (i=ucestart;i<uceend;i++) {
      stepind[i-ucestart] = i + ipmP->n+ipmP->nb;
      uceind[i-ucestart] = i;
    }

    ierr = ISCreateGeneral(comm,uceend-ucestart,stepind,PETSC_COPY_VALUES,&sis);CHKERRQ(ierr);
    ierr = ISCreateGeneral(comm,uceend-ucestart,uceind,PETSC_COPY_VALUES,&is1);CHKERRQ(ierr);
    ierr = VecScatterCreate(ipmP->bigstep,sis,tao->constraints_equality,is1,&ipmP->step3);CHKERRQ(ierr);
    ierr = ISDestroy(&sis);CHKERRQ(ierr);

    for (i=ucestart;i<uceend;i++) {
      stepind[i-ucestart] = i + ipmP->n;
    }

    ierr = ISCreateGeneral(comm,uceend-ucestart,stepind,PETSC_COPY_VALUES,&sis);CHKERRQ(ierr);
    ierr = VecScatterCreate(tao->constraints_equality,is1,ipmP->bigrhs,sis,&ipmP->rhs2);CHKERRQ(ierr);
    ierr = ISDestroy(&sis);CHKERRQ(ierr);
    ierr = ISDestroy(&is1);CHKERRQ(ierr);
  }

  if (ipmP->nb > 0) {
    for (i=sstart;i<send;i++) {
      stepind[i-sstart] = i + ipmP->n + ipmP->nb + ipmP->me;
      cind[i-sstart] = i;
    }
    ierr = ISCreateGeneral(comm,send-sstart,cind,PETSC_COPY_VALUES,&is1);CHKERRQ(ierr);
    ierr = ISCreateGeneral(comm,send-sstart,stepind,PETSC_COPY_VALUES,&sis);CHKERRQ(ierr);
    ierr = VecScatterCreate(ipmP->bigstep,sis,ipmP->s,is1,&ipmP->step4);CHKERRQ(ierr);
    ierr = VecScatterCreate(ipmP->s,is1,ipmP->bigrhs,sis,&ipmP->rhs4);CHKERRQ(ierr);
    ierr = ISDestroy(&sis);CHKERRQ(ierr);
    ierr = ISDestroy(&is1);CHKERRQ(ierr);
  }

  ierr = PetscFree(stepind);CHKERRQ(ierr);
  ierr = PetscFree(cind);CHKERRQ(ierr);
  ierr = PetscFree(ucind);CHKERRQ(ierr);
  ierr = PetscFree(uceind);CHKERRQ(ierr);
  ierr = PetscFree(xind);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode TaoDestroy_IPM(Tao tao)
{
  TAO_IPM        *ipmP = (TAO_IPM*)tao->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = VecDestroy(&ipmP->rd);CHKERRQ(ierr);
  ierr = VecDestroy(&ipmP->rpe);CHKERRQ(ierr);
  ierr = VecDestroy(&ipmP->rpi);CHKERRQ(ierr);
  ierr = VecDestroy(&ipmP->work);CHKERRQ(ierr);
  ierr = VecDestroy(&ipmP->lamdae);CHKERRQ(ierr);
  ierr = VecDestroy(&ipmP->lamdai);CHKERRQ(ierr);
  ierr = VecDestroy(&ipmP->s);CHKERRQ(ierr);
  ierr = VecDestroy(&ipmP->ds);CHKERRQ(ierr);
  ierr = VecDestroy(&ipmP->ci);CHKERRQ(ierr);

  ierr = VecDestroy(&ipmP->rhs_x);CHKERRQ(ierr);
  ierr = VecDestroy(&ipmP->rhs_lamdae);CHKERRQ(ierr);
  ierr = VecDestroy(&ipmP->rhs_lamdai);CHKERRQ(ierr);
  ierr = VecDestroy(&ipmP->rhs_s);CHKERRQ(ierr);

  ierr = VecDestroy(&ipmP->save_x);CHKERRQ(ierr);
  ierr = VecDestroy(&ipmP->save_lamdae);CHKERRQ(ierr);
  ierr = VecDestroy(&ipmP->save_lamdai);CHKERRQ(ierr);
  ierr = VecDestroy(&ipmP->save_s);CHKERRQ(ierr);

  ierr = VecScatterDestroy(&ipmP->step1);CHKERRQ(ierr);
  ierr = VecScatterDestroy(&ipmP->step2);CHKERRQ(ierr);
  ierr = VecScatterDestroy(&ipmP->step3);CHKERRQ(ierr);
  ierr = VecScatterDestroy(&ipmP->step4);CHKERRQ(ierr);

  ierr = VecScatterDestroy(&ipmP->rhs1);CHKERRQ(ierr);
  ierr = VecScatterDestroy(&ipmP->rhs2);CHKERRQ(ierr);
  ierr = VecScatterDestroy(&ipmP->rhs3);CHKERRQ(ierr);
  ierr = VecScatterDestroy(&ipmP->rhs4);CHKERRQ(ierr);

  ierr = VecScatterDestroy(&ipmP->ci_scat);CHKERRQ(ierr);
  ierr = VecScatterDestroy(&ipmP->xl_scat);CHKERRQ(ierr);
  ierr = VecScatterDestroy(&ipmP->xu_scat);CHKERRQ(ierr);

  ierr = VecDestroy(&ipmP->dlamdai);CHKERRQ(ierr);
  ierr = VecDestroy(&ipmP->dlamdae);CHKERRQ(ierr);
  ierr = VecDestroy(&ipmP->Zero_nb);CHKERRQ(ierr);
  ierr = VecDestroy(&ipmP->One_nb);CHKERRQ(ierr);
  ierr = VecDestroy(&ipmP->Inf_nb);CHKERRQ(ierr);
  ierr = VecDestroy(&ipmP->complementarity);CHKERRQ(ierr);

  ierr = VecDestroy(&ipmP->bigrhs);CHKERRQ(ierr);
  ierr = VecDestroy(&ipmP->bigstep);CHKERRQ(ierr);
  ierr = MatDestroy(&ipmP->Ai);CHKERRQ(ierr);
  ierr = MatDestroy(&ipmP->K);CHKERRQ(ierr);
  ierr = ISDestroy(&ipmP->isxu);CHKERRQ(ierr);
  ierr = ISDestroy(&ipmP->isxl);CHKERRQ(ierr);
  ierr = PetscFree(tao->data);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode TaoSetFromOptions_IPM(PetscOptionItems *PetscOptionsObject,Tao tao)
{
  TAO_IPM        *ipmP = (TAO_IPM*)tao->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscOptionsHead(PetscOptionsObject,"IPM method for constrained optimization");CHKERRQ(ierr);
  ierr = PetscOptionsBool("-tao_ipm_monitorkkt","monitor kkt status",NULL,ipmP->monitorkkt,&ipmP->monitorkkt,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-tao_ipm_pushs","parameter to push initial slack variables away from bounds",NULL,ipmP->pushs,&ipmP->pushs,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-tao_ipm_pushnu","parameter to push initial (inequality) dual variables away from bounds",NULL,ipmP->pushnu,&ipmP->pushnu,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsTail();CHKERRQ(ierr);
  ierr = KSPSetFromOptions(tao->ksp);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode TaoView_IPM(Tao tao, PetscViewer viewer)
{
  return 0;
}

/* IPMObjectiveAndGradient()
   f = d'x + 0.5 * x' * H * x
   rd = H*x + d + Ae'*lame - Ai'*lami
   rpe = Ae*x - be
   rpi = Ai*x - yi - bi
   mu = yi' * lami/mi;
   com = yi.*lami

   phi = ||rd|| + ||rpe|| + ||rpi|| + ||com||
*/
/*
static PetscErrorCode IPMObjective(TaoLineSearch ls, Vec X, PetscReal *f, void *tptr)
{
  Tao tao = (Tao)tptr;
  TAO_IPM *ipmP = (TAO_IPM*)tao->data;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  ierr = IPMComputeKKT(tao);CHKERRQ(ierr);
  *f = ipmP->phi;
  PetscFunctionReturn(0);
}
*/

/*
   f = d'x + 0.5 * x' * H * x
   rd = H*x + d + Ae'*lame - Ai'*lami
       Ai =   jac_ineq
               I (w/lb)
              -I (w/ub)

   rpe = ce
   rpi = ci - s;
   com = s.*lami
   mu = yi' * lami/mi;

   phi = ||rd|| + ||rpe|| + ||rpi|| + ||com||
*/
static PetscErrorCode IPMComputeKKT(Tao tao)
{
  TAO_IPM        *ipmP = (TAO_IPM *)tao->data;
  PetscScalar    norm;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = VecCopy(tao->gradient,ipmP->rd);CHKERRQ(ierr);

  if (ipmP->me > 0) {
    /* rd = gradient + Ae'*lamdae */
    ierr = MatMultTranspose(tao->jacobian_equality,ipmP->lamdae,ipmP->work);CHKERRQ(ierr);
    ierr = VecAXPY(ipmP->rd, 1.0, ipmP->work);CHKERRQ(ierr);

    /* rpe = ce(x) */
    ierr = VecCopy(tao->constraints_equality,ipmP->rpe);CHKERRQ(ierr);
  }
  if (ipmP->nb > 0) {
    /* rd = rd - Ai'*lamdai */
    ierr = MatMultTranspose(ipmP->Ai,ipmP->lamdai,ipmP->work);CHKERRQ(ierr);
    ierr = VecAXPY(ipmP->rd, -1.0, ipmP->work);CHKERRQ(ierr);

    /* rpi = cin - s */
    ierr = VecCopy(ipmP->ci,ipmP->rpi);CHKERRQ(ierr);
    ierr = VecAXPY(ipmP->rpi, -1.0, ipmP->s);CHKERRQ(ierr);

    /* com = s .* lami */
    ierr = VecPointwiseMult(ipmP->complementarity, ipmP->s,ipmP->lamdai);CHKERRQ(ierr);
  }
  /* phi = ||rd; rpe; rpi; com|| */
  ierr = VecDot(ipmP->rd,ipmP->rd,&norm);CHKERRQ(ierr);
  ipmP->phi = norm;
  if (ipmP->me > 0) {
    ierr = VecDot(ipmP->rpe,ipmP->rpe,&norm);CHKERRQ(ierr);
    ipmP->phi += norm;
  }
  if (ipmP->nb > 0) {
    ierr = VecDot(ipmP->rpi,ipmP->rpi,&norm);CHKERRQ(ierr);
    ipmP->phi += norm;
    ierr = VecDot(ipmP->complementarity,ipmP->complementarity,&norm);CHKERRQ(ierr);
    ipmP->phi += norm;
    /* mu = s'*lami/nb */
    ierr = VecDot(ipmP->s,ipmP->lamdai,&ipmP->mu);CHKERRQ(ierr);
    ipmP->mu /= ipmP->nb;
  } else {
    ipmP->mu = 1.0;
  }

  ipmP->phi = PetscSqrtScalar(ipmP->phi);
  PetscFunctionReturn(0);
}

/* evaluate user info at current point */
PetscErrorCode IPMEvaluate(Tao tao)
{
  TAO_IPM        *ipmP = (TAO_IPM *)tao->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = TaoComputeObjectiveAndGradient(tao,tao->solution,&ipmP->kkt_f,tao->gradient);CHKERRQ(ierr);
  ierr = TaoComputeHessian(tao,tao->solution,tao->hessian,tao->hessian_pre);CHKERRQ(ierr);
  if (ipmP->me > 0) {
    ierr = TaoComputeEqualityConstraints(tao,tao->solution,tao->constraints_equality);CHKERRQ(ierr);
    ierr = TaoComputeJacobianEquality(tao,tao->solution,tao->jacobian_equality,tao->jacobian_equality_pre);CHKERRQ(ierr);
  }
  if (ipmP->mi > 0) {
    ierr = TaoComputeInequalityConstraints(tao,tao->solution,tao->constraints_inequality);CHKERRQ(ierr);
    ierr = TaoComputeJacobianInequality(tao,tao->solution,tao->jacobian_inequality,tao->jacobian_inequality_pre);CHKERRQ(ierr);
  }
  if (ipmP->nb > 0) {
    /* Ai' =   jac_ineq | I (w/lb) | -I (w/ub)  */
    ierr = IPMUpdateAi(tao);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

/* Push initial point away from bounds */
PetscErrorCode IPMPushInitialPoint(Tao tao)
{
  TAO_IPM        *ipmP = (TAO_IPM *)tao->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = TaoComputeVariableBounds(tao);CHKERRQ(ierr);
  if (tao->XL && tao->XU) {
    ierr = VecMedian(tao->XL, tao->solution, tao->XU, tao->solution);CHKERRQ(ierr);
  }
  if (ipmP->nb > 0) {
    ierr = VecSet(ipmP->s,ipmP->pushs);CHKERRQ(ierr);
    ierr = VecSet(ipmP->lamdai,ipmP->pushnu);CHKERRQ(ierr);
    if (ipmP->mi > 0) {
      ierr = VecSet(tao->DI,ipmP->pushnu);CHKERRQ(ierr);
    }
  }
  if (ipmP->me > 0) {
    ierr = VecSet(tao->DE,1.0);CHKERRQ(ierr);
    ierr = VecSet(ipmP->lamdae,1.0);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

PetscErrorCode IPMUpdateAi(Tao tao)
{
  /* Ai =     Ji
              I (w/lb)
             -I (w/ub) */

  /* Ci =    user->ci
             Xi - lb (w/lb)
             -Xi + ub (w/ub)  */

  TAO_IPM           *ipmP = (TAO_IPM *)tao->data;
  MPI_Comm          comm;
  PetscInt          i;
  PetscScalar       newval;
  PetscInt          newrow,newcol,ncols;
  const PetscScalar *vals;
  const PetscInt    *cols;
  PetscInt          astart,aend,jstart,jend;
  PetscInt          *nonzeros;
  PetscInt          r2,r3,r4;
  PetscMPIInt       size;
  PetscErrorCode    ierr;
  Vec               solu;
  PetscInt          nloc;

  PetscFunctionBegin;
  r2 = ipmP->mi;
  r3 = r2 + ipmP->nxlb;
  r4 = r3 + ipmP->nxub;

  if (!ipmP->nb) PetscFunctionReturn(0);

  /* Create Ai matrix if it doesn't exist yet */
  if (!ipmP->Ai) {
    comm = ((PetscObject)(tao->solution))->comm;
    ierr = MPI_Comm_size(comm,&size);CHKERRQ(ierr);
    if (size == 1) {
      ierr = PetscMalloc1(ipmP->nb,&nonzeros);CHKERRQ(ierr);
      for (i=0;i<ipmP->mi;i++) {
        ierr = MatGetRow(tao->jacobian_inequality,i,&ncols,NULL,NULL);CHKERRQ(ierr);
        nonzeros[i] = ncols;
        ierr = MatRestoreRow(tao->jacobian_inequality,i,&ncols,NULL,NULL);CHKERRQ(ierr);
      }
      for (i=r2;i<r4;i++) {
        nonzeros[i] = 1;
      }
    }
    ierr = MatCreate(comm,&ipmP->Ai);CHKERRQ(ierr);
    ierr = MatSetType(ipmP->Ai,MATAIJ);CHKERRQ(ierr);

    ierr = TaoGetSolutionVector(tao,&solu);CHKERRQ(ierr);
    ierr = VecGetLocalSize(solu,&nloc);CHKERRQ(ierr);
    ierr = MatSetSizes(ipmP->Ai,PETSC_DECIDE,nloc,ipmP->nb,PETSC_DECIDE);CHKERRQ(ierr);
    ierr = MatSetFromOptions(ipmP->Ai);CHKERRQ(ierr);
    ierr = MatMPIAIJSetPreallocation(ipmP->Ai,ipmP->nb,NULL,ipmP->nb,NULL);CHKERRQ(ierr);
    ierr = MatSeqAIJSetPreallocation(ipmP->Ai,PETSC_DEFAULT,nonzeros);CHKERRQ(ierr);
    if (size ==1) {
      ierr = PetscFree(nonzeros);CHKERRQ(ierr);
    }
  }

  /* Copy values from user jacobian to Ai */
  ierr = MatGetOwnershipRange(ipmP->Ai,&astart,&aend);CHKERRQ(ierr);

  /* Ai w/lb */
  if (ipmP->mi) {
    ierr = MatZeroEntries(ipmP->Ai);CHKERRQ(ierr);
    ierr = MatGetOwnershipRange(tao->jacobian_inequality,&jstart,&jend);CHKERRQ(ierr);
    for (i=jstart;i<jend;i++) {
      ierr = MatGetRow(tao->jacobian_inequality,i,&ncols,&cols,&vals);CHKERRQ(ierr);
      newrow = i;
      ierr = MatSetValues(ipmP->Ai,1,&newrow,ncols,cols,vals,INSERT_VALUES);CHKERRQ(ierr);
      ierr = MatRestoreRow(tao->jacobian_inequality,i,&ncols,&cols,&vals);CHKERRQ(ierr);
    }
  }

  /* I w/ xlb */
  if (ipmP->nxlb) {
    for (i=0;i<ipmP->nxlb;i++) {
      if (i>=astart && i<aend) {
        newrow = i+r2;
        newcol = i;
        newval = 1.0;
        ierr = MatSetValues(ipmP->Ai,1,&newrow,1,&newcol,&newval,INSERT_VALUES);CHKERRQ(ierr);
      }
    }
  }
  if (ipmP->nxub) {
    /* I w/ xub */
    for (i=0;i<ipmP->nxub;i++) {
      if (i>=astart && i<aend) {
      newrow = i+r3;
      newcol = i;
      newval = -1.0;
      ierr = MatSetValues(ipmP->Ai,1,&newrow,1,&newcol,&newval,INSERT_VALUES);CHKERRQ(ierr);
      }
    }
  }

  ierr = MatAssemblyBegin(ipmP->Ai,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(ipmP->Ai,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  CHKMEMQ;

  ierr = VecSet(ipmP->ci,0.0);CHKERRQ(ierr);

  /* user ci */
  if (ipmP->mi > 0) {
    ierr = VecScatterBegin(ipmP->ci_scat,tao->constraints_inequality,ipmP->ci,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
    ierr = VecScatterEnd(ipmP->ci_scat,tao->constraints_inequality,ipmP->ci,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  }
  if (!ipmP->work){
    VecDuplicate(tao->solution,&ipmP->work);
  }
  ierr = VecCopy(tao->solution,ipmP->work);CHKERRQ(ierr);
  if (tao->XL) {
    ierr = VecAXPY(ipmP->work,-1.0,tao->XL);CHKERRQ(ierr);

    /* lower bounds on variables */
    if (ipmP->nxlb > 0) {
      ierr = VecScatterBegin(ipmP->xl_scat,ipmP->work,ipmP->ci,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
      ierr = VecScatterEnd(ipmP->xl_scat,ipmP->work,ipmP->ci,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
    }
  }
  if (tao->XU) {
    /* upper bounds on variables */
    ierr = VecCopy(tao->solution,ipmP->work);CHKERRQ(ierr);
    ierr = VecScale(ipmP->work,-1.0);CHKERRQ(ierr);
    ierr = VecAXPY(ipmP->work,1.0,tao->XU);CHKERRQ(ierr);
    if (ipmP->nxub > 0) {
      ierr = VecScatterBegin(ipmP->xu_scat,ipmP->work,ipmP->ci,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
      ierr = VecScatterEnd(ipmP->xu_scat,ipmP->work,ipmP->ci,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
    }
  }
  PetscFunctionReturn(0);
}

/* create K = [ Hlag , 0 , Ae', -Ai'];
              [Ae , 0,   0  , 0];
              [Ai ,-I,   0 ,  0];
              [ 0 , S ,  0,   Y ];  */
PetscErrorCode IPMUpdateK(Tao tao)
{
  TAO_IPM         *ipmP = (TAO_IPM *)tao->data;
  MPI_Comm        comm;
  PetscMPIInt     size;
  PetscErrorCode  ierr;
  PetscInt        i,j,row;
  PetscInt        ncols,newcol,newcols[2],newrow;
  const PetscInt  *cols;
  const PetscReal *vals;
  const PetscReal *l,*y;
  PetscReal       *newvals;
  PetscReal       newval;
  PetscInt        subsize;
  const PetscInt  *indices;
  PetscInt        *nonzeros,*d_nonzeros,*o_nonzeros;
  PetscInt        bigsize;
  PetscInt        r1,r2,r3;
  PetscInt        c1,c2,c3;
  PetscInt        klocalsize;
  PetscInt        hstart,hend,kstart,kend;
  PetscInt        aistart,aiend,aestart,aeend;
  PetscInt        sstart,send;

  PetscFunctionBegin;
  comm = ((PetscObject)(tao->solution))->comm;
  ierr = MPI_Comm_size(comm,&size);CHKERRQ(ierr);
  ierr = IPMUpdateAi(tao);CHKERRQ(ierr);

  /* allocate workspace */
  subsize = PetscMax(ipmP->n,ipmP->nb);
  subsize = PetscMax(ipmP->me,subsize);
  subsize = PetscMax(2,subsize);
  ierr = PetscMalloc1(subsize,(PetscInt**)&indices);CHKERRQ(ierr);
  ierr = PetscMalloc1(subsize,&newvals);CHKERRQ(ierr);

  r1 = c1 = ipmP->n;
  r2 = r1 + ipmP->me;  c2 = c1 + ipmP->nb;
  r3 = c3 = r2 + ipmP->nb;

  bigsize = ipmP->n+2*ipmP->nb+ipmP->me;
  ierr = VecGetOwnershipRange(ipmP->bigrhs,&kstart,&kend);CHKERRQ(ierr);
  ierr = MatGetOwnershipRange(tao->hessian,&hstart,&hend);CHKERRQ(ierr);
  klocalsize = kend-kstart;
  if (!ipmP->K) {
    if (size == 1) {
      ierr = PetscMalloc1(kend-kstart,&nonzeros);CHKERRQ(ierr);
      for (i=0;i<bigsize;i++) {
        if (i<r1) {
          ierr = MatGetRow(tao->hessian,i,&ncols,NULL,NULL);CHKERRQ(ierr);
          nonzeros[i] = ncols;
          ierr = MatRestoreRow(tao->hessian,i,&ncols,NULL,NULL);CHKERRQ(ierr);
          nonzeros[i] += ipmP->me+ipmP->nb;
        } else if (i<r2) {
          nonzeros[i-kstart] = ipmP->n;
        } else if (i<r3) {
          nonzeros[i-kstart] = ipmP->n+1;
        } else if (i<bigsize) {
          nonzeros[i-kstart] = 2;
        }
      }
      ierr = MatCreate(comm,&ipmP->K);CHKERRQ(ierr);
      ierr = MatSetType(ipmP->K,MATSEQAIJ);CHKERRQ(ierr);
      ierr = MatSetSizes(ipmP->K,klocalsize,klocalsize,PETSC_DETERMINE,PETSC_DETERMINE);CHKERRQ(ierr);
      ierr = MatSeqAIJSetPreallocation(ipmP->K,0,nonzeros);CHKERRQ(ierr);
      ierr = MatSetFromOptions(ipmP->K);CHKERRQ(ierr);
      ierr = PetscFree(nonzeros);CHKERRQ(ierr);
    } else {
      ierr = PetscMalloc1(kend-kstart,&d_nonzeros);CHKERRQ(ierr);
      ierr = PetscMalloc1(kend-kstart,&o_nonzeros);CHKERRQ(ierr);
      for (i=kstart;i<kend;i++) {
        if (i<r1) {
          /* TODO fix preallocation for mpi mats */
          d_nonzeros[i-kstart] = PetscMin(ipmP->n+ipmP->me+ipmP->nb,kend-kstart);
          o_nonzeros[i-kstart] = PetscMin(ipmP->n+ipmP->me+ipmP->nb,bigsize-(kend-kstart));
        } else if (i<r2) {
          d_nonzeros[i-kstart] = PetscMin(ipmP->n,kend-kstart);
          o_nonzeros[i-kstart] = PetscMin(ipmP->n,bigsize-(kend-kstart));
        } else if (i<r3) {
          d_nonzeros[i-kstart] = PetscMin(ipmP->n+2,kend-kstart);
          o_nonzeros[i-kstart] = PetscMin(ipmP->n+2,bigsize-(kend-kstart));
        } else {
          d_nonzeros[i-kstart] = PetscMin(2,kend-kstart);
          o_nonzeros[i-kstart] = PetscMin(2,bigsize-(kend-kstart));
        }
      }
      ierr = MatCreate(comm,&ipmP->K);CHKERRQ(ierr);
      ierr = MatSetType(ipmP->K,MATMPIAIJ);CHKERRQ(ierr);
      ierr = MatSetSizes(ipmP->K,klocalsize,klocalsize,PETSC_DETERMINE,PETSC_DETERMINE);CHKERRQ(ierr);
      ierr = MatMPIAIJSetPreallocation(ipmP->K,0,d_nonzeros,0,o_nonzeros);CHKERRQ(ierr);
      ierr = PetscFree(d_nonzeros);CHKERRQ(ierr);
      ierr = PetscFree(o_nonzeros);CHKERRQ(ierr);
      ierr = MatSetFromOptions(ipmP->K);CHKERRQ(ierr);
    }
  }

  ierr = MatZeroEntries(ipmP->K);CHKERRQ(ierr);
  /* Copy H */
  for (i=hstart;i<hend;i++) {
    ierr = MatGetRow(tao->hessian,i,&ncols,&cols,&vals);CHKERRQ(ierr);
    if (ncols > 0) {
      ierr = MatSetValues(ipmP->K,1,&i,ncols,cols,vals,INSERT_VALUES);CHKERRQ(ierr);
    }
    ierr = MatRestoreRow(tao->hessian,i,&ncols,&cols,&vals);CHKERRQ(ierr);
  }

  /* Copy Ae and Ae' */
  if (ipmP->me > 0) {
    ierr = MatGetOwnershipRange(tao->jacobian_equality,&aestart,&aeend);CHKERRQ(ierr);
    for (i=aestart;i<aeend;i++) {
      ierr = MatGetRow(tao->jacobian_equality,i,&ncols,&cols,&vals);CHKERRQ(ierr);
      if (ncols > 0) {
        /*Ae*/
        row = i+r1;
        ierr = MatSetValues(ipmP->K,1,&row,ncols,cols,vals,INSERT_VALUES);CHKERRQ(ierr);
        /*Ae'*/
        for (j=0;j<ncols;j++) {
          newcol = i + c2;
          newrow = cols[j];
          newval = vals[j];
          ierr = MatSetValues(ipmP->K,1,&newrow,1,&newcol,&newval,INSERT_VALUES);CHKERRQ(ierr);
        }
      }
      ierr = MatRestoreRow(tao->jacobian_equality,i,&ncols,&cols,&vals);CHKERRQ(ierr);
    }
  }

  if (ipmP->nb > 0) {
    ierr = MatGetOwnershipRange(ipmP->Ai,&aistart,&aiend);CHKERRQ(ierr);
    /* Copy Ai,and Ai' */
    for (i=aistart;i<aiend;i++) {
      row = i+r2;
      ierr = MatGetRow(ipmP->Ai,i,&ncols,&cols,&vals);CHKERRQ(ierr);
      if (ncols > 0) {
        /*Ai*/
        ierr = MatSetValues(ipmP->K,1,&row,ncols,cols,vals,INSERT_VALUES);CHKERRQ(ierr);
        /*-Ai'*/
        for (j=0;j<ncols;j++) {
          newcol = i + c3;
          newrow = cols[j];
          newval = -vals[j];
          ierr = MatSetValues(ipmP->K,1,&newrow,1,&newcol,&newval,INSERT_VALUES);CHKERRQ(ierr);
        }
      }
      ierr = MatRestoreRow(ipmP->Ai,i,&ncols,&cols,&vals);CHKERRQ(ierr);
    }

    /* -I */
    for (i=kstart;i<kend;i++) {
      if (i>=r2 && i<r3) {
        newrow = i;
        newcol = i-r2+c1;
        newval = -1.0;
        MatSetValues(ipmP->K,1,&newrow,1,&newcol,&newval,INSERT_VALUES);CHKERRQ(ierr);
      }
    }

    /* Copy L,Y */
    ierr = VecGetOwnershipRange(ipmP->s,&sstart,&send);CHKERRQ(ierr);
    ierr = VecGetArrayRead(ipmP->lamdai,&l);CHKERRQ(ierr);
    ierr = VecGetArrayRead(ipmP->s,&y);CHKERRQ(ierr);

    for (i=sstart;i<send;i++) {
      newcols[0] = c1+i;
      newcols[1] = c3+i;
      newvals[0] = l[i-sstart];
      newvals[1] = y[i-sstart];
      newrow = r3+i;
      ierr = MatSetValues(ipmP->K,1,&newrow,2,newcols,newvals,INSERT_VALUES);CHKERRQ(ierr);
    }

    ierr = VecRestoreArrayRead(ipmP->lamdai,&l);CHKERRQ(ierr);
    ierr = VecRestoreArrayRead(ipmP->s,&y);CHKERRQ(ierr);
  }

  ierr = PetscFree(indices);CHKERRQ(ierr);
  ierr = PetscFree(newvals);CHKERRQ(ierr);
  ierr = MatAssemblyBegin(ipmP->K,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(ipmP->K,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode IPMGatherRHS(Tao tao,Vec RHS,Vec X1,Vec X2,Vec X3,Vec X4)
{
  TAO_IPM        *ipmP = (TAO_IPM *)tao->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  /* rhs = [x1      (n)
            x2     (me)
            x3     (nb)
            x4     (nb)] */
  if (X1) {
    ierr = VecScatterBegin(ipmP->rhs1,X1,RHS,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
    ierr = VecScatterEnd(ipmP->rhs1,X1,RHS,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  }
  if (ipmP->me > 0 && X2) {
    ierr = VecScatterBegin(ipmP->rhs2,X2,RHS,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
    ierr = VecScatterEnd(ipmP->rhs2,X2,RHS,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  }
  if (ipmP->nb > 0) {
    if (X3) {
      ierr = VecScatterBegin(ipmP->rhs3,X3,RHS,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
      ierr = VecScatterEnd(ipmP->rhs3,X3,RHS,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
    }
    if (X4) {
      ierr = VecScatterBegin(ipmP->rhs4,X4,RHS,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
      ierr = VecScatterEnd(ipmP->rhs4,X4,RHS,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
    }
  }
  PetscFunctionReturn(0);
}

PetscErrorCode IPMScatterStep(Tao tao, Vec STEP, Vec X1, Vec X2, Vec X3, Vec X4)
{
  TAO_IPM        *ipmP = (TAO_IPM *)tao->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  CHKMEMQ;
  /*        [x1    (n)
             x2    (nb) may be 0
             x3    (me) may be 0
             x4    (nb) may be 0 */
  if (X1) {
    ierr = VecScatterBegin(ipmP->step1,STEP,X1,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
    ierr = VecScatterEnd(ipmP->step1,STEP,X1,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  }
  if (X2 && ipmP->nb > 0) {
    ierr = VecScatterBegin(ipmP->step2,STEP,X2,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
    ierr = VecScatterEnd(ipmP->step2,STEP,X2,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  }
  if (X3 && ipmP->me > 0) {
    ierr = VecScatterBegin(ipmP->step3,STEP,X3,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
    ierr = VecScatterEnd(ipmP->step3,STEP,X3,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  }
  if (X4 && ipmP->nb > 0) {
    ierr = VecScatterBegin(ipmP->step4,STEP,X4,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
    ierr = VecScatterEnd(ipmP->step4,STEP,X4,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  }
  CHKMEMQ;
  PetscFunctionReturn(0);
}

/*MC
  TAOIPM - Interior point algorithm for generally constrained optimization.

  Option Database Keys:
+   -tao_ipm_pushnu - parameter to push initial dual variables away from bounds
-   -tao_ipm_pushs - parameter to push initial slack variables away from bounds

  Notes:
    This algorithm is more of a place-holder for future constrained optimization algorithms and should not yet be used for large problems or production code.
  Level: beginner

M*/

PETSC_EXTERN PetscErrorCode TaoCreate_IPM(Tao tao)
{
  TAO_IPM        *ipmP;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  tao->ops->setup = TaoSetup_IPM;
  tao->ops->solve = TaoSolve_IPM;
  tao->ops->view = TaoView_IPM;
  tao->ops->setfromoptions = TaoSetFromOptions_IPM;
  tao->ops->destroy = TaoDestroy_IPM;
  /* tao->ops->computedual = TaoComputeDual_IPM; */

  ierr = PetscNewLog(tao,&ipmP);CHKERRQ(ierr);
  tao->data = (void*)ipmP;

  /* Override default settings (unless already changed) */
  if (!tao->max_it_changed) tao->max_it = 200;
  if (!tao->max_funcs_changed) tao->max_funcs = 500;

  ipmP->dec = 10000; /* line search critera */
  ipmP->taumin = 0.995;
  ipmP->monitorkkt = PETSC_FALSE;
  ipmP->pushs = 100;
  ipmP->pushnu = 100;
  ierr = KSPCreate(((PetscObject)tao)->comm, &tao->ksp);CHKERRQ(ierr);
  ierr = PetscObjectIncrementTabLevel((PetscObject)tao->ksp, (PetscObject)tao, 1);CHKERRQ(ierr);
  ierr = KSPSetOptionsPrefix(tao->ksp, tao->hdr.prefix);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
