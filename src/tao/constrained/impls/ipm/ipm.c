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
  TAO_IPM            *ipmP = (TAO_IPM*)tao->data;
  PetscInt           its,i;
  PetscScalar        stepsize=1.0;
  PetscScalar        step_s,step_l,alpha,tau,sigma,phi_target;

  PetscFunctionBegin;
  /* Push initial point away from bounds */
  PetscCall(IPMInitializeBounds(tao));
  PetscCall(IPMPushInitialPoint(tao));
  PetscCall(VecCopy(tao->solution,ipmP->rhs_x));
  PetscCall(IPMEvaluate(tao));
  PetscCall(IPMComputeKKT(tao));

  tao->reason = TAO_CONTINUE_ITERATING;
  PetscCall(TaoLogConvergenceHistory(tao,ipmP->kkt_f,ipmP->phi,0.0,tao->ksp_its));
  PetscCall(TaoMonitor(tao,tao->niter,ipmP->kkt_f,ipmP->phi,0.0,1.0));
  PetscCall((*tao->ops->convergencetest)(tao,tao->cnvP));

  while (tao->reason == TAO_CONTINUE_ITERATING) {
    /* Call general purpose update function */
    if (tao->ops->update) {
      PetscCall((*tao->ops->update)(tao, tao->niter, tao->user_update));
    }

    tao->ksp_its=0;
    PetscCall(IPMUpdateK(tao));
    /*
       rhs.x    = -rd
       rhs.lame = -rpe
       rhs.lami = -rpi
       rhs.com  = -com
    */

    PetscCall(VecCopy(ipmP->rd,ipmP->rhs_x));
    if (ipmP->me > 0) {
      PetscCall(VecCopy(ipmP->rpe,ipmP->rhs_lamdae));
    }
    if (ipmP->nb > 0) {
      PetscCall(VecCopy(ipmP->rpi,ipmP->rhs_lamdai));
      PetscCall(VecCopy(ipmP->complementarity,ipmP->rhs_s));
    }
    PetscCall(IPMGatherRHS(tao,ipmP->bigrhs,ipmP->rhs_x,ipmP->rhs_lamdae,ipmP->rhs_lamdai,ipmP->rhs_s));
    PetscCall(VecScale(ipmP->bigrhs,-1.0));

    /* solve K * step = rhs */
    PetscCall(KSPSetOperators(tao->ksp,ipmP->K,ipmP->K));
    PetscCall(KSPSolve(tao->ksp,ipmP->bigrhs,ipmP->bigstep));

    PetscCall(IPMScatterStep(tao,ipmP->bigstep,tao->stepdirection,ipmP->ds,ipmP->dlamdae,ipmP->dlamdai));
    PetscCall(KSPGetIterationNumber(tao->ksp,&its));
    tao->ksp_its += its;
    tao->ksp_tot_its+=its;
     /* Find distance along step direction to closest bound */
    if (ipmP->nb > 0) {
      PetscCall(VecStepBoundInfo(ipmP->s,ipmP->ds,ipmP->Zero_nb,ipmP->Inf_nb,&step_s,NULL,NULL));
      PetscCall(VecStepBoundInfo(ipmP->lamdai,ipmP->dlamdai,ipmP->Zero_nb,ipmP->Inf_nb,&step_l,NULL,NULL));
      alpha = PetscMin(step_s,step_l);
      alpha = PetscMin(alpha,1.0);
      ipmP->alpha1 = alpha;
    } else {
      ipmP->alpha1 = alpha = 1.0;
    }

    /* x_aff = x + alpha*d */
    PetscCall(VecCopy(tao->solution,ipmP->save_x));
    if (ipmP->me > 0) {
      PetscCall(VecCopy(ipmP->lamdae,ipmP->save_lamdae));
    }
    if (ipmP->nb > 0) {
      PetscCall(VecCopy(ipmP->lamdai,ipmP->save_lamdai));
      PetscCall(VecCopy(ipmP->s,ipmP->save_s));
    }

    PetscCall(VecAXPY(tao->solution,alpha,tao->stepdirection));
    if (ipmP->me > 0) {
      PetscCall(VecAXPY(ipmP->lamdae,alpha,ipmP->dlamdae));
    }
    if (ipmP->nb > 0) {
      PetscCall(VecAXPY(ipmP->lamdai,alpha,ipmP->dlamdai));
      PetscCall(VecAXPY(ipmP->s,alpha,ipmP->ds));
    }

    /* Recompute kkt to find centering parameter sigma = (new_mu/old_mu)^3 */
    if (ipmP->mu == 0.0) {
      sigma = 0.0;
    } else {
      sigma = 1.0/ipmP->mu;
    }
    PetscCall(IPMComputeKKT(tao));
    sigma *= ipmP->mu;
    sigma*=sigma*sigma;

    /* revert kkt info */
    PetscCall(VecCopy(ipmP->save_x,tao->solution));
    if (ipmP->me > 0) {
      PetscCall(VecCopy(ipmP->save_lamdae,ipmP->lamdae));
    }
    if (ipmP->nb > 0) {
      PetscCall(VecCopy(ipmP->save_lamdai,ipmP->lamdai));
      PetscCall(VecCopy(ipmP->save_s,ipmP->s));
    }
    PetscCall(IPMComputeKKT(tao));

    /* update rhs with new complementarity vector */
    if (ipmP->nb > 0) {
      PetscCall(VecCopy(ipmP->complementarity,ipmP->rhs_s));
      PetscCall(VecScale(ipmP->rhs_s,-1.0));
      PetscCall(VecShift(ipmP->rhs_s,sigma*ipmP->mu));
    }
    PetscCall(IPMGatherRHS(tao,ipmP->bigrhs,NULL,NULL,NULL,ipmP->rhs_s));

    /* solve K * step = rhs */
    PetscCall(KSPSetOperators(tao->ksp,ipmP->K,ipmP->K));
    PetscCall(KSPSolve(tao->ksp,ipmP->bigrhs,ipmP->bigstep));

    PetscCall(IPMScatterStep(tao,ipmP->bigstep,tao->stepdirection,ipmP->ds,ipmP->dlamdae,ipmP->dlamdai));
    PetscCall(KSPGetIterationNumber(tao->ksp,&its));
    tao->ksp_its += its;
    tao->ksp_tot_its+=its;
    if (ipmP->nb > 0) {
      /* Get max step size and apply frac-to-boundary */
      tau = PetscMax(ipmP->taumin,1.0-ipmP->mu);
      tau = PetscMin(tau,1.0);
      if (tau != 1.0) {
        PetscCall(VecScale(ipmP->s,tau));
        PetscCall(VecScale(ipmP->lamdai,tau));
      }
      PetscCall(VecStepBoundInfo(ipmP->s,ipmP->ds,ipmP->Zero_nb,ipmP->Inf_nb,&step_s,NULL,NULL));
      PetscCall(VecStepBoundInfo(ipmP->lamdai,ipmP->dlamdai,ipmP->Zero_nb,ipmP->Inf_nb,&step_l,NULL,NULL));
      if (tau != 1.0) {
        PetscCall(VecCopy(ipmP->save_s,ipmP->s));
        PetscCall(VecCopy(ipmP->save_lamdai,ipmP->lamdai));
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
      PetscCall(VecAXPY(tao->solution,alpha,tao->stepdirection));
      if (ipmP->nb > 0) {
        PetscCall(VecAXPY(ipmP->s,alpha,ipmP->ds));
        PetscCall(VecAXPY(ipmP->lamdai,alpha,ipmP->dlamdai));
      }
      if (ipmP->me > 0) {
        PetscCall(VecAXPY(ipmP->lamdae,alpha,ipmP->dlamdae));
      }

      /* update dual variables */
      if (ipmP->me > 0) {
        PetscCall(VecCopy(ipmP->lamdae,tao->DE));
      }

      PetscCall(IPMEvaluate(tao));
      PetscCall(IPMComputeKKT(tao));
      if (ipmP->phi <= phi_target) break;
      alpha /= 2.0;
    }

    PetscCall(TaoLogConvergenceHistory(tao,ipmP->kkt_f,ipmP->phi,0.0,tao->ksp_its));
    PetscCall(TaoMonitor(tao,tao->niter,ipmP->kkt_f,ipmP->phi,0.0,stepsize));
    PetscCall((*tao->ops->convergencetest)(tao,tao->cnvP));
    tao->niter++;
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode TaoSetup_IPM(Tao tao)
{
  TAO_IPM        *ipmP = (TAO_IPM*)tao->data;

  PetscFunctionBegin;
  ipmP->nb = ipmP->mi = ipmP->me = 0;
  ipmP->K = NULL;
  PetscCall(VecGetSize(tao->solution,&ipmP->n));
  if (!tao->gradient) {
    PetscCall(VecDuplicate(tao->solution, &tao->gradient));
    PetscCall(VecDuplicate(tao->solution, &tao->stepdirection));
    PetscCall(VecDuplicate(tao->solution, &ipmP->rd));
    PetscCall(VecDuplicate(tao->solution, &ipmP->rhs_x));
    PetscCall(VecDuplicate(tao->solution, &ipmP->work));
    PetscCall(VecDuplicate(tao->solution, &ipmP->save_x));
  }
  if (tao->constraints_equality) {
    PetscCall(VecGetSize(tao->constraints_equality,&ipmP->me));
    PetscCall(VecDuplicate(tao->constraints_equality,&ipmP->lamdae));
    PetscCall(VecDuplicate(tao->constraints_equality,&ipmP->dlamdae));
    PetscCall(VecDuplicate(tao->constraints_equality,&ipmP->rhs_lamdae));
    PetscCall(VecDuplicate(tao->constraints_equality,&ipmP->save_lamdae));
    PetscCall(VecDuplicate(tao->constraints_equality,&ipmP->rpe));
    PetscCall(VecDuplicate(tao->constraints_equality,&tao->DE));
  }
  if (tao->constraints_inequality) {
    PetscCall(VecDuplicate(tao->constraints_inequality,&tao->DI));
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
  MPI_Comm       comm;

  PetscFunctionBegin;
  cind=xind=ucind=uceind=stepind=NULL;
  ipmP->mi=0;
  ipmP->nxlb=0;
  ipmP->nxub=0;
  ipmP->nb=0;
  ipmP->nslack=0;

  PetscCall(VecDuplicate(tao->solution,&xtmp));
  PetscCall(TaoComputeVariableBounds(tao));
  if (tao->XL) {
    PetscCall(VecSet(xtmp,PETSC_NINFINITY));
    PetscCall(VecWhichGreaterThan(tao->XL,xtmp,&ipmP->isxl));
    PetscCall(ISGetSize(ipmP->isxl,&ipmP->nxlb));
  } else {
    ipmP->nxlb=0;
  }
  if (tao->XU) {
    PetscCall(VecSet(xtmp,PETSC_INFINITY));
    PetscCall(VecWhichLessThan(tao->XU,xtmp,&ipmP->isxu));
    PetscCall(ISGetSize(ipmP->isxu,&ipmP->nxub));
  } else {
    ipmP->nxub=0;
  }
  PetscCall(VecDestroy(&xtmp));
  if (tao->constraints_inequality) {
    PetscCall(VecGetSize(tao->constraints_inequality,&ipmP->mi));
  } else {
    ipmP->mi = 0;
  }
  ipmP->nb = ipmP->nxlb + ipmP->nxub + ipmP->mi;

  PetscCall(PetscObjectGetComm((PetscObject)tao->solution,&comm));

  bigsize = ipmP->n+2*ipmP->nb+ipmP->me;
  PetscCall(PetscMalloc1(bigsize,&stepind));
  PetscCall(PetscMalloc1(ipmP->n,&xind));
  PetscCall(PetscMalloc1(ipmP->me,&uceind));
  PetscCall(VecGetOwnershipRange(tao->solution,&xstart,&xend));

  if (ipmP->nb > 0) {
    PetscCall(VecCreate(comm,&ipmP->s));
    PetscCall(VecSetSizes(ipmP->s,PETSC_DECIDE,ipmP->nb));
    PetscCall(VecSetFromOptions(ipmP->s));
    PetscCall(VecDuplicate(ipmP->s,&ipmP->ds));
    PetscCall(VecDuplicate(ipmP->s,&ipmP->rhs_s));
    PetscCall(VecDuplicate(ipmP->s,&ipmP->complementarity));
    PetscCall(VecDuplicate(ipmP->s,&ipmP->ci));

    PetscCall(VecDuplicate(ipmP->s,&ipmP->lamdai));
    PetscCall(VecDuplicate(ipmP->s,&ipmP->dlamdai));
    PetscCall(VecDuplicate(ipmP->s,&ipmP->rhs_lamdai));
    PetscCall(VecDuplicate(ipmP->s,&ipmP->save_lamdai));

    PetscCall(VecDuplicate(ipmP->s,&ipmP->save_s));
    PetscCall(VecDuplicate(ipmP->s,&ipmP->rpi));
    PetscCall(VecDuplicate(ipmP->s,&ipmP->Zero_nb));
    PetscCall(VecSet(ipmP->Zero_nb,0.0));
    PetscCall(VecDuplicate(ipmP->s,&ipmP->One_nb));
    PetscCall(VecSet(ipmP->One_nb,1.0));
    PetscCall(VecDuplicate(ipmP->s,&ipmP->Inf_nb));
    PetscCall(VecSet(ipmP->Inf_nb,PETSC_INFINITY));

    PetscCall(PetscMalloc1(ipmP->nb,&cind));
    PetscCall(PetscMalloc1(ipmP->mi,&ucind));
    PetscCall(VecGetOwnershipRange(ipmP->s,&sstart,&send));

    if (ipmP->mi > 0) {
      PetscCall(VecGetOwnershipRange(tao->constraints_inequality,&ucstart,&ucend));
      counter=0;
      for (i=ucstart;i<ucend;i++) {
        cind[counter++] = i;
      }
      PetscCall(ISCreateGeneral(comm,counter,cind,PETSC_COPY_VALUES,&isuc));
      PetscCall(ISCreateGeneral(comm,counter,cind,PETSC_COPY_VALUES,&isc));
      PetscCall(VecScatterCreate(tao->constraints_inequality,isuc,ipmP->ci,isc,&ipmP->ci_scat));

      PetscCall(ISDestroy(&isuc));
      PetscCall(ISDestroy(&isc));
    }
    /* need to know how may xbound indices are on each process */
    /* TODO better way */
    if (ipmP->nxlb) {
      PetscCall(ISAllGather(ipmP->isxl,&bigxl));
      PetscCall(ISGetIndices(bigxl,&xli));
      /* find offsets for this processor */
      xl_offset = ipmP->mi;
      for (i=0;i<ipmP->nxlb;i++) {
        if (xli[i] < xstart) {
          xl_offset++;
        } else break;
      }
      PetscCall(ISRestoreIndices(bigxl,&xli));

      PetscCall(ISGetIndices(ipmP->isxl,&xli));
      PetscCall(ISGetLocalSize(ipmP->isxl,&nloc));
      for (i=0;i<nloc;i++) {
        xind[i] = xli[i];
        cind[i] = xl_offset+i;
      }

      PetscCall(ISCreateGeneral(comm,nloc,xind,PETSC_COPY_VALUES,&isx));
      PetscCall(ISCreateGeneral(comm,nloc,cind,PETSC_COPY_VALUES,&isc));
      PetscCall(VecScatterCreate(tao->XL,isx,ipmP->ci,isc,&ipmP->xl_scat));
      PetscCall(ISDestroy(&isx));
      PetscCall(ISDestroy(&isc));
      PetscCall(ISDestroy(&bigxl));
    }

    if (ipmP->nxub) {
      PetscCall(ISAllGather(ipmP->isxu,&bigxu));
      PetscCall(ISGetIndices(bigxu,&xui));
      /* find offsets for this processor */
      xu_offset = ipmP->mi + ipmP->nxlb;
      for (i=0;i<ipmP->nxub;i++) {
        if (xui[i] < xstart) {
          xu_offset++;
        } else break;
      }
      PetscCall(ISRestoreIndices(bigxu,&xui));

      PetscCall(ISGetIndices(ipmP->isxu,&xui));
      PetscCall(ISGetLocalSize(ipmP->isxu,&nloc));
      for (i=0;i<nloc;i++) {
        xind[i] = xui[i];
        cind[i] = xu_offset+i;
      }

      PetscCall(ISCreateGeneral(comm,nloc,xind,PETSC_COPY_VALUES,&isx));
      PetscCall(ISCreateGeneral(comm,nloc,cind,PETSC_COPY_VALUES,&isc));
      PetscCall(VecScatterCreate(tao->XU,isx,ipmP->ci,isc,&ipmP->xu_scat));
      PetscCall(ISDestroy(&isx));
      PetscCall(ISDestroy(&isc));
      PetscCall(ISDestroy(&bigxu));
    }
  }
  PetscCall(VecCreate(comm,&ipmP->bigrhs));
  PetscCall(VecGetType(tao->solution,&vtype));
  PetscCall(VecSetType(ipmP->bigrhs,vtype));
  PetscCall(VecSetSizes(ipmP->bigrhs,PETSC_DECIDE,bigsize));
  PetscCall(VecSetFromOptions(ipmP->bigrhs));
  PetscCall(VecDuplicate(ipmP->bigrhs,&ipmP->bigstep));

  /* create scatters for step->x and x->rhs */
  for (i=xstart;i<xend;i++) {
    stepind[i-xstart] = i;
    xind[i-xstart] = i;
  }
  PetscCall(ISCreateGeneral(comm,xend-xstart,stepind,PETSC_COPY_VALUES,&sis));
  PetscCall(ISCreateGeneral(comm,xend-xstart,xind,PETSC_COPY_VALUES,&is1));
  PetscCall(VecScatterCreate(ipmP->bigstep,sis,tao->solution,is1,&ipmP->step1));
  PetscCall(VecScatterCreate(tao->solution,is1,ipmP->bigrhs,sis,&ipmP->rhs1));
  PetscCall(ISDestroy(&sis));
  PetscCall(ISDestroy(&is1));

  if (ipmP->nb > 0) {
    for (i=sstart;i<send;i++) {
      stepind[i-sstart] = i+ipmP->n;
      cind[i-sstart] = i;
    }
    PetscCall(ISCreateGeneral(comm,send-sstart,stepind,PETSC_COPY_VALUES,&sis));
    PetscCall(ISCreateGeneral(comm,send-sstart,cind,PETSC_COPY_VALUES,&is1));
    PetscCall(VecScatterCreate(ipmP->bigstep,sis,ipmP->s,is1,&ipmP->step2));
    PetscCall(ISDestroy(&sis));

    for (i=sstart;i<send;i++) {
      stepind[i-sstart] = i+ipmP->n+ipmP->me;
      cind[i-sstart] = i;
    }
    PetscCall(ISCreateGeneral(comm,send-sstart,stepind,PETSC_COPY_VALUES,&sis));
    PetscCall(VecScatterCreate(ipmP->s,is1,ipmP->bigrhs,sis,&ipmP->rhs3));
    PetscCall(ISDestroy(&sis));
    PetscCall(ISDestroy(&is1));
  }

  if (ipmP->me > 0) {
    PetscCall(VecGetOwnershipRange(tao->constraints_equality,&ucestart,&uceend));
    for (i=ucestart;i<uceend;i++) {
      stepind[i-ucestart] = i + ipmP->n+ipmP->nb;
      uceind[i-ucestart] = i;
    }

    PetscCall(ISCreateGeneral(comm,uceend-ucestart,stepind,PETSC_COPY_VALUES,&sis));
    PetscCall(ISCreateGeneral(comm,uceend-ucestart,uceind,PETSC_COPY_VALUES,&is1));
    PetscCall(VecScatterCreate(ipmP->bigstep,sis,tao->constraints_equality,is1,&ipmP->step3));
    PetscCall(ISDestroy(&sis));

    for (i=ucestart;i<uceend;i++) {
      stepind[i-ucestart] = i + ipmP->n;
    }

    PetscCall(ISCreateGeneral(comm,uceend-ucestart,stepind,PETSC_COPY_VALUES,&sis));
    PetscCall(VecScatterCreate(tao->constraints_equality,is1,ipmP->bigrhs,sis,&ipmP->rhs2));
    PetscCall(ISDestroy(&sis));
    PetscCall(ISDestroy(&is1));
  }

  if (ipmP->nb > 0) {
    for (i=sstart;i<send;i++) {
      stepind[i-sstart] = i + ipmP->n + ipmP->nb + ipmP->me;
      cind[i-sstart] = i;
    }
    PetscCall(ISCreateGeneral(comm,send-sstart,cind,PETSC_COPY_VALUES,&is1));
    PetscCall(ISCreateGeneral(comm,send-sstart,stepind,PETSC_COPY_VALUES,&sis));
    PetscCall(VecScatterCreate(ipmP->bigstep,sis,ipmP->s,is1,&ipmP->step4));
    PetscCall(VecScatterCreate(ipmP->s,is1,ipmP->bigrhs,sis,&ipmP->rhs4));
    PetscCall(ISDestroy(&sis));
    PetscCall(ISDestroy(&is1));
  }

  PetscCall(PetscFree(stepind));
  PetscCall(PetscFree(cind));
  PetscCall(PetscFree(ucind));
  PetscCall(PetscFree(uceind));
  PetscCall(PetscFree(xind));
  PetscFunctionReturn(0);
}

static PetscErrorCode TaoDestroy_IPM(Tao tao)
{
  TAO_IPM        *ipmP = (TAO_IPM*)tao->data;

  PetscFunctionBegin;
  PetscCall(VecDestroy(&ipmP->rd));
  PetscCall(VecDestroy(&ipmP->rpe));
  PetscCall(VecDestroy(&ipmP->rpi));
  PetscCall(VecDestroy(&ipmP->work));
  PetscCall(VecDestroy(&ipmP->lamdae));
  PetscCall(VecDestroy(&ipmP->lamdai));
  PetscCall(VecDestroy(&ipmP->s));
  PetscCall(VecDestroy(&ipmP->ds));
  PetscCall(VecDestroy(&ipmP->ci));

  PetscCall(VecDestroy(&ipmP->rhs_x));
  PetscCall(VecDestroy(&ipmP->rhs_lamdae));
  PetscCall(VecDestroy(&ipmP->rhs_lamdai));
  PetscCall(VecDestroy(&ipmP->rhs_s));

  PetscCall(VecDestroy(&ipmP->save_x));
  PetscCall(VecDestroy(&ipmP->save_lamdae));
  PetscCall(VecDestroy(&ipmP->save_lamdai));
  PetscCall(VecDestroy(&ipmP->save_s));

  PetscCall(VecScatterDestroy(&ipmP->step1));
  PetscCall(VecScatterDestroy(&ipmP->step2));
  PetscCall(VecScatterDestroy(&ipmP->step3));
  PetscCall(VecScatterDestroy(&ipmP->step4));

  PetscCall(VecScatterDestroy(&ipmP->rhs1));
  PetscCall(VecScatterDestroy(&ipmP->rhs2));
  PetscCall(VecScatterDestroy(&ipmP->rhs3));
  PetscCall(VecScatterDestroy(&ipmP->rhs4));

  PetscCall(VecScatterDestroy(&ipmP->ci_scat));
  PetscCall(VecScatterDestroy(&ipmP->xl_scat));
  PetscCall(VecScatterDestroy(&ipmP->xu_scat));

  PetscCall(VecDestroy(&ipmP->dlamdai));
  PetscCall(VecDestroy(&ipmP->dlamdae));
  PetscCall(VecDestroy(&ipmP->Zero_nb));
  PetscCall(VecDestroy(&ipmP->One_nb));
  PetscCall(VecDestroy(&ipmP->Inf_nb));
  PetscCall(VecDestroy(&ipmP->complementarity));

  PetscCall(VecDestroy(&ipmP->bigrhs));
  PetscCall(VecDestroy(&ipmP->bigstep));
  PetscCall(MatDestroy(&ipmP->Ai));
  PetscCall(MatDestroy(&ipmP->K));
  PetscCall(ISDestroy(&ipmP->isxu));
  PetscCall(ISDestroy(&ipmP->isxl));
  PetscCall(PetscFree(tao->data));
  PetscFunctionReturn(0);
}

static PetscErrorCode TaoSetFromOptions_IPM(PetscOptionItems *PetscOptionsObject,Tao tao)
{
  TAO_IPM        *ipmP = (TAO_IPM*)tao->data;

  PetscFunctionBegin;
  PetscOptionsHeadBegin(PetscOptionsObject,"IPM method for constrained optimization");
  PetscCall(PetscOptionsBool("-tao_ipm_monitorkkt","monitor kkt status",NULL,ipmP->monitorkkt,&ipmP->monitorkkt,NULL));
  PetscCall(PetscOptionsReal("-tao_ipm_pushs","parameter to push initial slack variables away from bounds",NULL,ipmP->pushs,&ipmP->pushs,NULL));
  PetscCall(PetscOptionsReal("-tao_ipm_pushnu","parameter to push initial (inequality) dual variables away from bounds",NULL,ipmP->pushnu,&ipmP->pushnu,NULL));
  PetscOptionsHeadEnd();
  PetscCall(KSPSetFromOptions(tao->ksp));
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
  PetscFunctionBegin;
  PetscCall(IPMComputeKKT(tao));
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

  PetscFunctionBegin;
  PetscCall(VecCopy(tao->gradient,ipmP->rd));

  if (ipmP->me > 0) {
    /* rd = gradient + Ae'*lamdae */
    PetscCall(MatMultTranspose(tao->jacobian_equality,ipmP->lamdae,ipmP->work));
    PetscCall(VecAXPY(ipmP->rd, 1.0, ipmP->work));

    /* rpe = ce(x) */
    PetscCall(VecCopy(tao->constraints_equality,ipmP->rpe));
  }
  if (ipmP->nb > 0) {
    /* rd = rd - Ai'*lamdai */
    PetscCall(MatMultTranspose(ipmP->Ai,ipmP->lamdai,ipmP->work));
    PetscCall(VecAXPY(ipmP->rd, -1.0, ipmP->work));

    /* rpi = cin - s */
    PetscCall(VecCopy(ipmP->ci,ipmP->rpi));
    PetscCall(VecAXPY(ipmP->rpi, -1.0, ipmP->s));

    /* com = s .* lami */
    PetscCall(VecPointwiseMult(ipmP->complementarity, ipmP->s,ipmP->lamdai));
  }
  /* phi = ||rd; rpe; rpi; com|| */
  PetscCall(VecDot(ipmP->rd,ipmP->rd,&norm));
  ipmP->phi = norm;
  if (ipmP->me > 0) {
    PetscCall(VecDot(ipmP->rpe,ipmP->rpe,&norm));
    ipmP->phi += norm;
  }
  if (ipmP->nb > 0) {
    PetscCall(VecDot(ipmP->rpi,ipmP->rpi,&norm));
    ipmP->phi += norm;
    PetscCall(VecDot(ipmP->complementarity,ipmP->complementarity,&norm));
    ipmP->phi += norm;
    /* mu = s'*lami/nb */
    PetscCall(VecDot(ipmP->s,ipmP->lamdai,&ipmP->mu));
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

  PetscFunctionBegin;
  PetscCall(TaoComputeObjectiveAndGradient(tao,tao->solution,&ipmP->kkt_f,tao->gradient));
  PetscCall(TaoComputeHessian(tao,tao->solution,tao->hessian,tao->hessian_pre));
  if (ipmP->me > 0) {
    PetscCall(TaoComputeEqualityConstraints(tao,tao->solution,tao->constraints_equality));
    PetscCall(TaoComputeJacobianEquality(tao,tao->solution,tao->jacobian_equality,tao->jacobian_equality_pre));
  }
  if (ipmP->mi > 0) {
    PetscCall(TaoComputeInequalityConstraints(tao,tao->solution,tao->constraints_inequality));
    PetscCall(TaoComputeJacobianInequality(tao,tao->solution,tao->jacobian_inequality,tao->jacobian_inequality_pre));
  }
  if (ipmP->nb > 0) {
    /* Ai' =   jac_ineq | I (w/lb) | -I (w/ub)  */
    PetscCall(IPMUpdateAi(tao));
  }
  PetscFunctionReturn(0);
}

/* Push initial point away from bounds */
PetscErrorCode IPMPushInitialPoint(Tao tao)
{
  TAO_IPM        *ipmP = (TAO_IPM *)tao->data;

  PetscFunctionBegin;
  PetscCall(TaoComputeVariableBounds(tao));
  PetscCall(VecMedian(tao->XL, tao->solution, tao->XU, tao->solution));
  if (ipmP->nb > 0) {
    PetscCall(VecSet(ipmP->s,ipmP->pushs));
    PetscCall(VecSet(ipmP->lamdai,ipmP->pushnu));
    if (ipmP->mi > 0) {
      PetscCall(VecSet(tao->DI,ipmP->pushnu));
    }
  }
  if (ipmP->me > 0) {
    PetscCall(VecSet(tao->DE,1.0));
    PetscCall(VecSet(ipmP->lamdae,1.0));
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
    PetscCallMPI(MPI_Comm_size(comm,&size));
    if (size == 1) {
      PetscCall(PetscMalloc1(ipmP->nb,&nonzeros));
      for (i=0;i<ipmP->mi;i++) {
        PetscCall(MatGetRow(tao->jacobian_inequality,i,&ncols,NULL,NULL));
        nonzeros[i] = ncols;
        PetscCall(MatRestoreRow(tao->jacobian_inequality,i,&ncols,NULL,NULL));
      }
      for (i=r2;i<r4;i++) {
        nonzeros[i] = 1;
      }
    }
    PetscCall(MatCreate(comm,&ipmP->Ai));
    PetscCall(MatSetType(ipmP->Ai,MATAIJ));

    PetscCall(TaoGetSolution(tao,&solu));
    PetscCall(VecGetLocalSize(solu,&nloc));
    PetscCall(MatSetSizes(ipmP->Ai,PETSC_DECIDE,nloc,ipmP->nb,PETSC_DECIDE));
    PetscCall(MatSetFromOptions(ipmP->Ai));
    PetscCall(MatMPIAIJSetPreallocation(ipmP->Ai,ipmP->nb,NULL,ipmP->nb,NULL));
    PetscCall(MatSeqAIJSetPreallocation(ipmP->Ai,PETSC_DEFAULT,nonzeros));
    if (size ==1) {
      PetscCall(PetscFree(nonzeros));
    }
  }

  /* Copy values from user jacobian to Ai */
  PetscCall(MatGetOwnershipRange(ipmP->Ai,&astart,&aend));

  /* Ai w/lb */
  if (ipmP->mi) {
    PetscCall(MatZeroEntries(ipmP->Ai));
    PetscCall(MatGetOwnershipRange(tao->jacobian_inequality,&jstart,&jend));
    for (i=jstart;i<jend;i++) {
      PetscCall(MatGetRow(tao->jacobian_inequality,i,&ncols,&cols,&vals));
      newrow = i;
      PetscCall(MatSetValues(ipmP->Ai,1,&newrow,ncols,cols,vals,INSERT_VALUES));
      PetscCall(MatRestoreRow(tao->jacobian_inequality,i,&ncols,&cols,&vals));
    }
  }

  /* I w/ xlb */
  if (ipmP->nxlb) {
    for (i=0;i<ipmP->nxlb;i++) {
      if (i>=astart && i<aend) {
        newrow = i+r2;
        newcol = i;
        newval = 1.0;
        PetscCall(MatSetValues(ipmP->Ai,1,&newrow,1,&newcol,&newval,INSERT_VALUES));
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
      PetscCall(MatSetValues(ipmP->Ai,1,&newrow,1,&newcol,&newval,INSERT_VALUES));
      }
    }
  }

  PetscCall(MatAssemblyBegin(ipmP->Ai,MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(ipmP->Ai,MAT_FINAL_ASSEMBLY));
  CHKMEMQ;

  PetscCall(VecSet(ipmP->ci,0.0));

  /* user ci */
  if (ipmP->mi > 0) {
    PetscCall(VecScatterBegin(ipmP->ci_scat,tao->constraints_inequality,ipmP->ci,INSERT_VALUES,SCATTER_FORWARD));
    PetscCall(VecScatterEnd(ipmP->ci_scat,tao->constraints_inequality,ipmP->ci,INSERT_VALUES,SCATTER_FORWARD));
  }
  if (!ipmP->work) {
    VecDuplicate(tao->solution,&ipmP->work);
  }
  PetscCall(VecCopy(tao->solution,ipmP->work));
  if (tao->XL) {
    PetscCall(VecAXPY(ipmP->work,-1.0,tao->XL));

    /* lower bounds on variables */
    if (ipmP->nxlb > 0) {
      PetscCall(VecScatterBegin(ipmP->xl_scat,ipmP->work,ipmP->ci,INSERT_VALUES,SCATTER_FORWARD));
      PetscCall(VecScatterEnd(ipmP->xl_scat,ipmP->work,ipmP->ci,INSERT_VALUES,SCATTER_FORWARD));
    }
  }
  if (tao->XU) {
    /* upper bounds on variables */
    PetscCall(VecCopy(tao->solution,ipmP->work));
    PetscCall(VecScale(ipmP->work,-1.0));
    PetscCall(VecAXPY(ipmP->work,1.0,tao->XU));
    if (ipmP->nxub > 0) {
      PetscCall(VecScatterBegin(ipmP->xu_scat,ipmP->work,ipmP->ci,INSERT_VALUES,SCATTER_FORWARD));
      PetscCall(VecScatterEnd(ipmP->xu_scat,ipmP->work,ipmP->ci,INSERT_VALUES,SCATTER_FORWARD));
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
  PetscCallMPI(MPI_Comm_size(comm,&size));
  PetscCall(IPMUpdateAi(tao));

  /* allocate workspace */
  subsize = PetscMax(ipmP->n,ipmP->nb);
  subsize = PetscMax(ipmP->me,subsize);
  subsize = PetscMax(2,subsize);
  PetscCall(PetscMalloc1(subsize,(PetscInt**)&indices));
  PetscCall(PetscMalloc1(subsize,&newvals));

  r1 = c1 = ipmP->n;
  r2 = r1 + ipmP->me;  c2 = c1 + ipmP->nb;
  r3 = c3 = r2 + ipmP->nb;

  bigsize = ipmP->n+2*ipmP->nb+ipmP->me;
  PetscCall(VecGetOwnershipRange(ipmP->bigrhs,&kstart,&kend));
  PetscCall(MatGetOwnershipRange(tao->hessian,&hstart,&hend));
  klocalsize = kend-kstart;
  if (!ipmP->K) {
    if (size == 1) {
      PetscCall(PetscMalloc1(kend-kstart,&nonzeros));
      for (i=0;i<bigsize;i++) {
        if (i<r1) {
          PetscCall(MatGetRow(tao->hessian,i,&ncols,NULL,NULL));
          nonzeros[i] = ncols;
          PetscCall(MatRestoreRow(tao->hessian,i,&ncols,NULL,NULL));
          nonzeros[i] += ipmP->me+ipmP->nb;
        } else if (i<r2) {
          nonzeros[i-kstart] = ipmP->n;
        } else if (i<r3) {
          nonzeros[i-kstart] = ipmP->n+1;
        } else if (i<bigsize) {
          nonzeros[i-kstart] = 2;
        }
      }
      PetscCall(MatCreate(comm,&ipmP->K));
      PetscCall(MatSetType(ipmP->K,MATSEQAIJ));
      PetscCall(MatSetSizes(ipmP->K,klocalsize,klocalsize,PETSC_DETERMINE,PETSC_DETERMINE));
      PetscCall(MatSeqAIJSetPreallocation(ipmP->K,0,nonzeros));
      PetscCall(MatSetFromOptions(ipmP->K));
      PetscCall(PetscFree(nonzeros));
    } else {
      PetscCall(PetscMalloc1(kend-kstart,&d_nonzeros));
      PetscCall(PetscMalloc1(kend-kstart,&o_nonzeros));
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
      PetscCall(MatCreate(comm,&ipmP->K));
      PetscCall(MatSetType(ipmP->K,MATMPIAIJ));
      PetscCall(MatSetSizes(ipmP->K,klocalsize,klocalsize,PETSC_DETERMINE,PETSC_DETERMINE));
      PetscCall(MatMPIAIJSetPreallocation(ipmP->K,0,d_nonzeros,0,o_nonzeros));
      PetscCall(PetscFree(d_nonzeros));
      PetscCall(PetscFree(o_nonzeros));
      PetscCall(MatSetFromOptions(ipmP->K));
    }
  }

  PetscCall(MatZeroEntries(ipmP->K));
  /* Copy H */
  for (i=hstart;i<hend;i++) {
    PetscCall(MatGetRow(tao->hessian,i,&ncols,&cols,&vals));
    if (ncols > 0) {
      PetscCall(MatSetValues(ipmP->K,1,&i,ncols,cols,vals,INSERT_VALUES));
    }
    PetscCall(MatRestoreRow(tao->hessian,i,&ncols,&cols,&vals));
  }

  /* Copy Ae and Ae' */
  if (ipmP->me > 0) {
    PetscCall(MatGetOwnershipRange(tao->jacobian_equality,&aestart,&aeend));
    for (i=aestart;i<aeend;i++) {
      PetscCall(MatGetRow(tao->jacobian_equality,i,&ncols,&cols,&vals));
      if (ncols > 0) {
        /*Ae*/
        row = i+r1;
        PetscCall(MatSetValues(ipmP->K,1,&row,ncols,cols,vals,INSERT_VALUES));
        /*Ae'*/
        for (j=0;j<ncols;j++) {
          newcol = i + c2;
          newrow = cols[j];
          newval = vals[j];
          PetscCall(MatSetValues(ipmP->K,1,&newrow,1,&newcol,&newval,INSERT_VALUES));
        }
      }
      PetscCall(MatRestoreRow(tao->jacobian_equality,i,&ncols,&cols,&vals));
    }
  }

  if (ipmP->nb > 0) {
    PetscCall(MatGetOwnershipRange(ipmP->Ai,&aistart,&aiend));
    /* Copy Ai,and Ai' */
    for (i=aistart;i<aiend;i++) {
      row = i+r2;
      PetscCall(MatGetRow(ipmP->Ai,i,&ncols,&cols,&vals));
      if (ncols > 0) {
        /*Ai*/
        PetscCall(MatSetValues(ipmP->K,1,&row,ncols,cols,vals,INSERT_VALUES));
        /*-Ai'*/
        for (j=0;j<ncols;j++) {
          newcol = i + c3;
          newrow = cols[j];
          newval = -vals[j];
          PetscCall(MatSetValues(ipmP->K,1,&newrow,1,&newcol,&newval,INSERT_VALUES));
        }
      }
      PetscCall(MatRestoreRow(ipmP->Ai,i,&ncols,&cols,&vals));
    }

    /* -I */
    for (i=kstart;i<kend;i++) {
      if (i>=r2 && i<r3) {
        newrow = i;
        newcol = i-r2+c1;
        newval = -1.0;
        PetscCall(MatSetValues(ipmP->K,1,&newrow,1,&newcol,&newval,INSERT_VALUES));
      }
    }

    /* Copy L,Y */
    PetscCall(VecGetOwnershipRange(ipmP->s,&sstart,&send));
    PetscCall(VecGetArrayRead(ipmP->lamdai,&l));
    PetscCall(VecGetArrayRead(ipmP->s,&y));

    for (i=sstart;i<send;i++) {
      newcols[0] = c1+i;
      newcols[1] = c3+i;
      newvals[0] = l[i-sstart];
      newvals[1] = y[i-sstart];
      newrow = r3+i;
      PetscCall(MatSetValues(ipmP->K,1,&newrow,2,newcols,newvals,INSERT_VALUES));
    }

    PetscCall(VecRestoreArrayRead(ipmP->lamdai,&l));
    PetscCall(VecRestoreArrayRead(ipmP->s,&y));
  }

  PetscCall(PetscFree(indices));
  PetscCall(PetscFree(newvals));
  PetscCall(MatAssemblyBegin(ipmP->K,MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(ipmP->K,MAT_FINAL_ASSEMBLY));
  PetscFunctionReturn(0);
}

PetscErrorCode IPMGatherRHS(Tao tao,Vec RHS,Vec X1,Vec X2,Vec X3,Vec X4)
{
  TAO_IPM        *ipmP = (TAO_IPM *)tao->data;

  PetscFunctionBegin;
  /* rhs = [x1      (n)
            x2     (me)
            x3     (nb)
            x4     (nb)] */
  if (X1) {
    PetscCall(VecScatterBegin(ipmP->rhs1,X1,RHS,INSERT_VALUES,SCATTER_FORWARD));
    PetscCall(VecScatterEnd(ipmP->rhs1,X1,RHS,INSERT_VALUES,SCATTER_FORWARD));
  }
  if (ipmP->me > 0 && X2) {
    PetscCall(VecScatterBegin(ipmP->rhs2,X2,RHS,INSERT_VALUES,SCATTER_FORWARD));
    PetscCall(VecScatterEnd(ipmP->rhs2,X2,RHS,INSERT_VALUES,SCATTER_FORWARD));
  }
  if (ipmP->nb > 0) {
    if (X3) {
      PetscCall(VecScatterBegin(ipmP->rhs3,X3,RHS,INSERT_VALUES,SCATTER_FORWARD));
      PetscCall(VecScatterEnd(ipmP->rhs3,X3,RHS,INSERT_VALUES,SCATTER_FORWARD));
    }
    if (X4) {
      PetscCall(VecScatterBegin(ipmP->rhs4,X4,RHS,INSERT_VALUES,SCATTER_FORWARD));
      PetscCall(VecScatterEnd(ipmP->rhs4,X4,RHS,INSERT_VALUES,SCATTER_FORWARD));
    }
  }
  PetscFunctionReturn(0);
}

PetscErrorCode IPMScatterStep(Tao tao, Vec STEP, Vec X1, Vec X2, Vec X3, Vec X4)
{
  TAO_IPM        *ipmP = (TAO_IPM *)tao->data;

  PetscFunctionBegin;
  CHKMEMQ;
  /*        [x1    (n)
             x2    (nb) may be 0
             x3    (me) may be 0
             x4    (nb) may be 0 */
  if (X1) {
    PetscCall(VecScatterBegin(ipmP->step1,STEP,X1,INSERT_VALUES,SCATTER_FORWARD));
    PetscCall(VecScatterEnd(ipmP->step1,STEP,X1,INSERT_VALUES,SCATTER_FORWARD));
  }
  if (X2 && ipmP->nb > 0) {
    PetscCall(VecScatterBegin(ipmP->step2,STEP,X2,INSERT_VALUES,SCATTER_FORWARD));
    PetscCall(VecScatterEnd(ipmP->step2,STEP,X2,INSERT_VALUES,SCATTER_FORWARD));
  }
  if (X3 && ipmP->me > 0) {
    PetscCall(VecScatterBegin(ipmP->step3,STEP,X3,INSERT_VALUES,SCATTER_FORWARD));
    PetscCall(VecScatterEnd(ipmP->step3,STEP,X3,INSERT_VALUES,SCATTER_FORWARD));
  }
  if (X4 && ipmP->nb > 0) {
    PetscCall(VecScatterBegin(ipmP->step4,STEP,X4,INSERT_VALUES,SCATTER_FORWARD));
    PetscCall(VecScatterEnd(ipmP->step4,STEP,X4,INSERT_VALUES,SCATTER_FORWARD));
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

  PetscFunctionBegin;
  tao->ops->setup = TaoSetup_IPM;
  tao->ops->solve = TaoSolve_IPM;
  tao->ops->view = TaoView_IPM;
  tao->ops->setfromoptions = TaoSetFromOptions_IPM;
  tao->ops->destroy = TaoDestroy_IPM;
  /* tao->ops->computedual = TaoComputeDual_IPM; */

  PetscCall(PetscNewLog(tao,&ipmP));
  tao->data = (void*)ipmP;

  /* Override default settings (unless already changed) */
  if (!tao->max_it_changed) tao->max_it = 200;
  if (!tao->max_funcs_changed) tao->max_funcs = 500;

  ipmP->dec = 10000; /* line search critera */
  ipmP->taumin = 0.995;
  ipmP->monitorkkt = PETSC_FALSE;
  ipmP->pushs = 100;
  ipmP->pushnu = 100;
  PetscCall(KSPCreate(((PetscObject)tao)->comm, &tao->ksp));
  PetscCall(PetscObjectIncrementTabLevel((PetscObject)tao->ksp, (PetscObject)tao, 1));
  PetscCall(KSPSetOptionsPrefix(tao->ksp, tao->hdr.prefix));
  PetscFunctionReturn(0);
}
