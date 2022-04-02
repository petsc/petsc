#include <petsc/private/kspimpl.h>
#include <petsc/private/vecimpl.h>

#define offset(j)      PetscMax(((j) - (2*l)), 0)
#define shift(i,j)     ((i) - offset((j)))
#define G(i,j)         (plcg->G[((j)*(2*l+1))+(shift((i),(j))) ])
#define G_noshift(i,j) (plcg->G[((j)*(2*l+1))+(i)])
#define alpha(i)       (plcg->alpha[(i)])
#define gamma(i)       (plcg->gamma[(i)])
#define delta(i)       (plcg->delta[(i)])
#define sigma(i)       (plcg->sigma[(i)])
#define req(i)         (plcg->req[(i)])

typedef struct KSP_CG_PIPE_L_s KSP_CG_PIPE_L;
struct KSP_CG_PIPE_L_s {
  PetscInt    l;          /* pipeline depth */
  Vec         *Z;         /* Z vectors (shifted base) */
  Vec         *U;         /* U vectors (unpreconditioned shifted base) */
  Vec         *V;         /* V vectors (original base) */
  Vec         *Q;         /* Q vectors (auxiliary bases) */
  Vec         p;          /* work vector */
  PetscScalar *G;         /* such that Z = VG (band matrix)*/
  PetscScalar *gamma,*delta,*alpha;
  PetscReal   lmin,lmax;  /* min and max eigen values estimates to compute base shifts */
  PetscReal   *sigma;     /* base shifts */
  MPI_Request *req;       /* request array for asynchronous global collective */
  PetscBool   show_rstrt; /* flag to show restart information in output (default: not shown) */
};

/*
  KSPSetUp_PIPELCG - Sets up the workspace needed by the PIPELCG method.

  This is called once, usually automatically by KSPSolve() or KSPSetUp()
  but can be called directly by KSPSetUp()
*/
static PetscErrorCode KSPSetUp_PIPELCG(KSP ksp)
{
  KSP_CG_PIPE_L  *plcg = (KSP_CG_PIPE_L*)ksp->data;
  PetscInt       l=plcg->l,max_it=ksp->max_it;
  MPI_Comm       comm;

  PetscFunctionBegin;
  comm = PetscObjectComm((PetscObject)ksp);
  PetscCheck(max_it >= 1,comm,PETSC_ERR_ARG_OUTOFRANGE,"%s: max_it argument must be positive.",((PetscObject)ksp)->type_name);
  PetscCheck(l >= 1,comm,PETSC_ERR_ARG_OUTOFRANGE,"%s: pipel argument must be positive.",((PetscObject)ksp)->type_name);
  PetscCheck(l <= max_it,comm,PETSC_ERR_ARG_OUTOFRANGE,"%s: pipel argument must be less than max_it.",((PetscObject)ksp)->type_name);

  PetscCall(KSPSetWorkVecs(ksp,1)); /* get work vectors needed by PIPELCG */
  plcg->p = ksp->work[0];

  PetscCall(VecDuplicateVecs(plcg->p,PetscMax(3,l+1),&plcg->Z));
  PetscCall(VecDuplicateVecs(plcg->p,3,&plcg->U));
  PetscCall(VecDuplicateVecs(plcg->p,3,&plcg->V));
  PetscCall(VecDuplicateVecs(plcg->p,3*(l-1)+1,&plcg->Q));
  PetscCall(PetscCalloc1(2,&plcg->alpha));
  PetscCall(PetscCalloc1(l,&plcg->sigma));

  PetscFunctionReturn(0);
}

static PetscErrorCode KSPReset_PIPELCG(KSP ksp)
{
  KSP_CG_PIPE_L  *plcg = (KSP_CG_PIPE_L*)ksp->data;
  PetscInt       l=plcg->l;

  PetscFunctionBegin;
  PetscCall(PetscFree(plcg->sigma));
  PetscCall(PetscFree(plcg->alpha));
  PetscCall(VecDestroyVecs(PetscMax(3,l+1),&plcg->Z));
  PetscCall(VecDestroyVecs(3,&plcg->U));
  PetscCall(VecDestroyVecs(3,&plcg->V));
  PetscCall(VecDestroyVecs(3*(l-1)+1,&plcg->Q));
  PetscFunctionReturn(0);
}

static PetscErrorCode KSPDestroy_PIPELCG(KSP ksp)
{
  PetscFunctionBegin;
  PetscCall(KSPReset_PIPELCG(ksp));
  PetscCall(KSPDestroyDefault(ksp));
  PetscFunctionReturn(0);
}

static PetscErrorCode KSPSetFromOptions_PIPELCG(PetscOptionItems *PetscOptionsObject,KSP ksp)
{
  KSP_CG_PIPE_L  *plcg = (KSP_CG_PIPE_L*)ksp->data;
  PetscBool      flag=PETSC_FALSE;

  PetscFunctionBegin;
  PetscOptionsHeadBegin(PetscOptionsObject,"KSP PIPELCG options");
  PetscCall(PetscOptionsInt("-ksp_pipelcg_pipel","Pipeline length","",plcg->l,&plcg->l,&flag));
  if (!flag) plcg->l = 1;
  PetscCall(PetscOptionsReal("-ksp_pipelcg_lmin","Estimate for smallest eigenvalue","",plcg->lmin,&plcg->lmin,&flag));
  if (!flag) plcg->lmin = 0.0;
  PetscCall(PetscOptionsReal("-ksp_pipelcg_lmax","Estimate for largest eigenvalue","",plcg->lmax,&plcg->lmax,&flag));
  if (!flag) plcg->lmax = 0.0;
  PetscCall(PetscOptionsBool("-ksp_pipelcg_monitor","Output information on restarts when they occur? (default: 0)","",plcg->show_rstrt,&plcg->show_rstrt,&flag));
  if (!flag) plcg->show_rstrt = PETSC_FALSE;
  PetscOptionsHeadEnd();
  PetscFunctionReturn(0);
}

static PetscErrorCode MPIPetsc_Iallreduce(void *sendbuf,void *recvbuf,PetscMPIInt count,MPI_Datatype datatype,MPI_Op op,MPI_Comm comm,MPI_Request *request)
{
  PetscFunctionBegin;
#if defined(PETSC_HAVE_MPI_NONBLOCKING_COLLECTIVES)
  PetscCallMPI(MPI_Iallreduce(sendbuf,recvbuf,count,datatype,op,comm,request));
#else
  PetscCall(MPIU_Allreduce(sendbuf,recvbuf,count,datatype,op,comm));
  *request = MPI_REQUEST_NULL;
#endif
  PetscFunctionReturn(0);
}

static PetscErrorCode KSPView_PIPELCG(KSP ksp,PetscViewer viewer)
{
  KSP_CG_PIPE_L  *plcg = (KSP_CG_PIPE_L*)ksp->data;
  PetscBool      iascii=PETSC_FALSE,isstring=PETSC_FALSE;

  PetscFunctionBegin;
  PetscCall(PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&iascii));
  PetscCall(PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERSTRING,&isstring));
  if (iascii) {
    PetscCall(PetscViewerASCIIPrintf(viewer,"  Pipeline depth: %D\n", plcg->l));
    PetscCall(PetscViewerASCIIPrintf(viewer,"  Minimal eigenvalue estimate %g\n",plcg->lmin));
    PetscCall(PetscViewerASCIIPrintf(viewer,"  Maximal eigenvalue estimate %g\n",plcg->lmax));
  } else if (isstring) {
    PetscCall(PetscViewerStringSPrintf(viewer,"  Pipeline depth: %D\n", plcg->l));
    PetscCall(PetscViewerStringSPrintf(viewer,"  Minimal eigenvalue estimate %g\n",plcg->lmin));
    PetscCall(PetscViewerStringSPrintf(viewer,"  Maximal eigenvalue estimate %g\n",plcg->lmax));
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode KSPSolve_InnerLoop_PIPELCG(KSP ksp)
{
  KSP_CG_PIPE_L  *plcg = (KSP_CG_PIPE_L*)ksp->data;
  Mat            A=NULL,Pmat=NULL;
  PetscInt       it=0,max_it=ksp->max_it,l=plcg->l,i=0,j=0,k=0;
  PetscInt       start=0,middle=0,end=0;
  Vec            *Z=plcg->Z,*U=plcg->U,*V=plcg->V,*Q=plcg->Q;
  Vec            x=NULL,p=NULL,temp=NULL;
  PetscScalar    sum_dummy=0.0,eta=0.0,zeta=0.0,lambda=0.0;
  PetscReal      dp=0.0,tmp=0.0,beta=0.0,invbeta2=0.0;
  MPI_Comm       comm;

  PetscFunctionBegin;
  x   = ksp->vec_sol;
  p   = plcg->p;

  comm = PetscObjectComm((PetscObject)ksp);
  PetscCall(PCGetOperators(ksp->pc,&A,&Pmat));

  for (it = 0; it < max_it+l; ++it) {
    /* ----------------------------------- */
    /* Multiplication  z_{it+1} =  Az_{it} */
    /* ----------------------------------- */
    /* Shift the U vector pointers */
    temp = U[2];
    for (i = 2; i>0; i--) {
      U[i] = U[i-1];
    }
    U[0] = temp;
    if (it < l) {
      /* SpMV and Sigma-shift and Prec */
      PetscCall(MatMult(A,Z[l-it],U[0]));
      PetscCall(VecAXPY(U[0],-sigma(it),U[1]));
      PetscCall(KSP_PCApply(ksp,U[0],Z[l-it-1]));
      if (it < l-1) {
        PetscCall(VecCopy(Z[l-it-1],Q[3*it]));
      }
    } else {
      /* Shift the Z vector pointers */
      temp = Z[PetscMax(l,2)];
      for (i = PetscMax(l,2); i > 0; --i) {
        Z[i] = Z[i-1];
      }
      Z[0] = temp;
      /* SpMV and Prec */
      PetscCall(MatMult(A,Z[1],U[0]));
      PetscCall(KSP_PCApply(ksp,U[0],Z[0]));
    }

    /* ----------------------------------- */
    /* Adjust the G matrix */
    /* ----------------------------------- */
    if (it >= l) {
      if (it == l) {
        /* MPI_Wait for G(0,0),scale V0 and Z and U and Q vectors with 1/beta */
        PetscCallMPI(MPI_Wait(&req(0),MPI_STATUS_IGNORE));
        beta = PetscSqrtReal(PetscRealPart(G(0,0)));
        G(0,0) = 1.0;
        PetscCall(VecAXPY(V[0],1.0/beta,p)); /* this assumes V[0] to be zero initially */
        for (j = 0; j <= PetscMax(l,2); ++j) {
          PetscCall(VecScale(Z[j],1.0/beta));
        }
        for (j = 0; j <= 2; ++j) {
          PetscCall(VecScale(U[j],1.0/beta));
        }
        for (j = 0; j < l-1; ++j) {
          PetscCall(VecScale(Q[3*j],1.0/beta));
        }
      }

      /* MPI_Wait until the dot products,started l iterations ago,are completed */
      PetscCallMPI(MPI_Wait(&req(it-l+1),MPI_STATUS_IGNORE));
      if (it >= 2*l) {
        for (j = PetscMax(0,it-3*l+1); j <= it-2*l; j++) {
          G(j,it-l+1) = G(it-2*l+1,j+l); /* exploit symmetry in G matrix */
        }
      }

      if (it <= 2*l-1) {
        invbeta2 = 1.0 / (beta * beta);
        /* Scale columns 1 up to l of G with 1/beta^2 */
        for (j = PetscMax(it-3*l+1,0); j <= it-l+1; ++j) {
          G(j,it-l+1) *= invbeta2;
        }
      }

      for (j = PetscMax(it-2*l+2,0); j <= it-l; ++j) {
        sum_dummy = 0.0;
        for (k = PetscMax(it-3*l+1,0); k <= j-1; ++k) {
          sum_dummy = sum_dummy + G(k,j) * G(k,it-l+1);
        }
        G(j,it-l+1) = (G(j,it-l+1) - sum_dummy) / G(j,j);
      }

      sum_dummy = 0.0;
      for (k = PetscMax(it-3*l+1,0); k <= it-l; ++k) {
        sum_dummy = sum_dummy + G(k,it-l+1) * G(k,it-l+1);
      }

      tmp = PetscRealPart(G(it-l+1,it-l+1) - sum_dummy);
      /* Breakdown check */
      if (tmp < 0) {
        if (plcg->show_rstrt) {
          PetscCall(PetscPrintf(comm,"Sqrt breakdown in iteration %D: sqrt argument is %e. Iteration was restarted.\n",ksp->its+1,(double)tmp));
        }
        /* End hanging dot-products in the pipeline before exiting for-loop */
        start = it-l+2;
        end = PetscMin(it+1,max_it+1);  /* !warning! 'it' can actually be greater than 'max_it' */
        for (i = start; i < end; ++i) {
          PetscCallMPI(MPI_Wait(&req(i),MPI_STATUS_IGNORE));
        }
        break;
      }
      G(it-l+1,it-l+1) = PetscSqrtReal(tmp);

      if (it < 2*l) {
        if (it == l) {
          gamma(it-l) = (G(it-l,it-l+1) + sigma(it-l) * G(it-l,it-l)) / G(it-l,it-l);
        } else {
          gamma(it-l) = (G(it-l,it-l+1) + sigma(it-l) * G(it-l,it-l)
                         - delta(it-l-1) * G(it-l-1,it-l)) / G(it-l,it-l);
        }
        delta(it-l) = G(it-l+1,it-l+1) / G(it-l,it-l);
      } else {
        gamma(it-l) = (G(it-l,it-l) * gamma(it-2*l)
                       + G(it-l,it-l+1) * delta(it-2*l)
                       - G(it-l-1,it-l) * delta(it-l-1)) / G(it-l,it-l);
        delta(it-l) = (G(it-l+1,it-l+1) * delta(it-2*l)) / G(it-l,it-l);
      }

      /* -------------------------------------------------- */
      /* Recursively compute the next V, Q, Z and U vectors */
      /* -------------------------------------------------- */
      /* Shift the V vector pointers */
      temp = V[2];
      for (i = 2; i>0; i--) {
        V[i] = V[i-1];
      }
      V[0] = temp;

      /* Recurrence V vectors */
      if (l == 1) {
        PetscCall(VecCopy(Z[1],V[0]));
      } else {
        PetscCall(VecCopy(Q[0],V[0]));
      }
      if (it == l) {
        PetscCall(VecAXPY(V[0],sigma(0)-gamma(it-l),V[1]));
      } else {
        alpha(0) = sigma(0)-gamma(it-l);
        alpha(1) = -delta(it-l-1);
        PetscCall(VecMAXPY(V[0],2,&alpha(0),&V[1]));
      }
      PetscCall(VecScale(V[0],1.0/delta(it-l)));

      /* Recurrence Q vectors */
      for (j = 0; j < l-1; ++j) {
        /* Shift the Q vector pointers */
        temp = Q[3*j+2];
        for (i = 2; i>0; i--) {
          Q[3*j+i] = Q[3*j+i-1];
        }
        Q[3*j] = temp;

        if (j < l-2) {
          PetscCall(VecCopy(Q[3*(j+1)],Q[3*j]));
        } else {
          PetscCall(VecCopy(Z[1],Q[3*j]));
        }
        if (it == l) {
          PetscCall(VecAXPY(Q[3*j],sigma(j+1)-gamma(it-l),Q[3*j+1]));
        } else {
          alpha(0) = sigma(j+1)-gamma(it-l);
          alpha(1) = -delta(it-l-1);
          PetscCall(VecMAXPY(Q[3*j],2,&alpha(0),&Q[3*j+1]));
        }
        PetscCall(VecScale(Q[3*j],1.0/delta(it-l)));
      }

      /* Recurrence Z and U vectors */
      if (it == l) {
        PetscCall(VecAXPY(Z[0],-gamma(it-l),Z[1]));
        PetscCall(VecAXPY(U[0],-gamma(it-l),U[1]));
      } else {
        alpha(0) = -gamma(it-l);
        alpha(1) = -delta(it-l-1);
        PetscCall(VecMAXPY(Z[0],2,&alpha(0),&Z[1]));
        PetscCall(VecMAXPY(U[0],2,&alpha(0),&U[1]));
      }
      PetscCall(VecScale(Z[0],1.0/delta(it-l)));
      PetscCall(VecScale(U[0],1.0/delta(it-l)));
    }

    /* ---------------------------------------- */
    /* Compute and communicate the dot products */
    /* ---------------------------------------- */
    if (it < l) {
      for (j = 0; j < it+2; ++j) {
        PetscCall((*U[0]->ops->dot_local)(U[0],Z[l-j],&G(j,it+1))); /* dot-products (U[0],Z[j]) */
      }
      PetscCall(MPIPetsc_Iallreduce(MPI_IN_PLACE,&G(0,it+1),it+2,MPIU_SCALAR,MPIU_SUM,comm,&req(it+1)));
    } else if ((it >= l) && (it < max_it)) {
      middle = it-l+2;
      end = it+2;
      PetscCall((*U[0]->ops->dot_local)(U[0],V[0],&G(it-l+1,it+1))); /* dot-product (U[0],V[0]) */
      for (j = middle; j < end; ++j) {
        PetscCall((*U[0]->ops->dot_local)(U[0],plcg->Z[it+1-j],&G(j,it+1))); /* dot-products (U[0],Z[j]) */
      }
      PetscCall(MPIPetsc_Iallreduce(MPI_IN_PLACE,&G(it-l+1,it+1),l+1,MPIU_SCALAR,MPIU_SUM,comm,&req(it+1)));
    }

    /* ----------------------------------------- */
    /* Compute solution vector and residual norm */
    /* ----------------------------------------- */
    if (it >= l) {
      if (it == l) {
        if (ksp->its != 0) {
          ++ ksp->its;
        }
        eta  = gamma(0);
        zeta = beta;
        PetscCall(VecCopy(V[1],p));
        PetscCall(VecScale(p,1.0/eta));
        PetscCall(VecAXPY(x,zeta,p));
        dp   = beta;
      } else if (it > l) {
        k = it-l;
        ++ ksp->its;
        lambda = delta(k-1)/eta;
        eta  = gamma(k) - lambda * delta(k-1);
        zeta = -lambda * zeta;
        PetscCall(VecScale(p,-delta(k-1)/eta));
        PetscCall(VecAXPY(p,1.0/eta,V[1]));
        PetscCall(VecAXPY(x,zeta,p));
        dp   = PetscAbsScalar(zeta);
      }
      ksp->rnorm = dp;
      PetscCall(KSPLogResidualHistory(ksp,dp));
      PetscCall(KSPMonitor(ksp,ksp->its,dp));
      PetscCall((*ksp->converged)(ksp,ksp->its,dp,&ksp->reason,ksp->cnvP));

      if (ksp->its >= max_it && !ksp->reason) ksp->reason = KSP_DIVERGED_ITS;
      if (ksp->reason) {
        /* End hanging dot-products in the pipeline before exiting for-loop */
        start = it-l+2;
        end = PetscMin(it+2,max_it+1); /* !warning! 'it' can actually be greater than 'max_it' */
        for (i = start; i < end; ++i) {
          PetscCallMPI(MPI_Wait(&req(i),MPI_STATUS_IGNORE));
        }
        break;
      }
    }
  } /* End inner for loop */
  PetscFunctionReturn(0);
}

static PetscErrorCode KSPSolve_ReInitData_PIPELCG(KSP ksp)
{
  KSP_CG_PIPE_L  *plcg = (KSP_CG_PIPE_L*)ksp->data;
  PetscInt       i=0,j=0,l=plcg->l,max_it=ksp->max_it;

  PetscFunctionBegin;
  for (i = 0; i < PetscMax(3,l+1); ++i) {
    PetscCall(VecSet(plcg->Z[i],0.0));
  }
  for (i = 1; i < 3; ++i) {
    PetscCall(VecSet(plcg->U[i],0.0));
  }
  for (i = 0; i < 3; ++i) {
    PetscCall(VecSet(plcg->V[i],0.0));
  }
  for (i = 0; i < 3*(l-1)+1; ++i) {
    PetscCall(VecSet(plcg->Q[i],0.0));
  }
  for (j = 0; j < (max_it+1); ++j) {
    gamma(j) = 0.0;
    delta(j) = 0.0;
    for (i = 0; i < (2*l+1); ++i) {
      G_noshift(i,j) = 0.0;
    }
  }
  PetscFunctionReturn(0);
}

/*
  KSPSolve_PIPELCG - This routine actually applies the pipelined(l) conjugate gradient method
*/
static PetscErrorCode KSPSolve_PIPELCG(KSP ksp)
{
  KSP_CG_PIPE_L  *plcg = (KSP_CG_PIPE_L*)ksp->data;
  Mat            A=NULL,Pmat=NULL;
  Vec            b=NULL,x=NULL,p=NULL;
  PetscInt       max_it=ksp->max_it,l=plcg->l;
  PetscInt       i=0,outer_it=0,curr_guess_zero=0;
  PetscReal      lmin=plcg->lmin,lmax=plcg->lmax;
  PetscBool      diagonalscale=PETSC_FALSE;
  MPI_Comm       comm;

  PetscFunctionBegin;
  comm = PetscObjectComm((PetscObject)ksp);
  PetscCall(PCGetDiagonalScale(ksp->pc,&diagonalscale));
  if (diagonalscale) {
    SETERRQ(comm,PETSC_ERR_SUP,"Krylov method %s does not support diagonal scaling",((PetscObject)ksp)->type_name);
  }

  x = ksp->vec_sol;
  b = ksp->vec_rhs;
  p = plcg->p;

  PetscCall(PetscCalloc1((max_it+1)*(2*l+1),&plcg->G));
  PetscCall(PetscCalloc1(max_it+1,&plcg->gamma));
  PetscCall(PetscCalloc1(max_it+1,&plcg->delta));
  PetscCall(PetscCalloc1(max_it+1,&plcg->req));

  PetscCall(PCGetOperators(ksp->pc,&A,&Pmat));

  for (i = 0; i < l; ++i) {
    sigma(i) = (0.5*(lmin+lmax) + (0.5*(lmax-lmin) * PetscCosReal(PETSC_PI*(2.0*i+1.0)/(2.0*l))));
  }

  ksp->its = 0;
  outer_it = 0;
  curr_guess_zero = !! ksp->guess_zero;

  while (ksp->its < max_it) { /* OUTER LOOP (gmres-like restart to handle breakdowns) */
    /* RESTART LOOP */
    if (!curr_guess_zero) {
      PetscCall(KSP_MatMult(ksp,A,x,plcg->U[0]));  /* u <- b - Ax */
      PetscCall(VecAYPX(plcg->U[0],-1.0,b));
    } else {
      PetscCall(VecCopy(b,plcg->U[0]));            /* u <- b (x is 0) */
    }
    PetscCall(KSP_PCApply(ksp,plcg->U[0],p));      /* p <- Bu */

    if (outer_it > 0) {
      /* Re-initialize Z,U,V,Q,gamma,delta,G after restart occurred */
      PetscCall(KSPSolve_ReInitData_PIPELCG(ksp));
    }

    PetscCall((*plcg->U[0]->ops->dot_local)(plcg->U[0],p,&G(0,0)));
    PetscCall(MPIPetsc_Iallreduce(MPI_IN_PLACE,&G(0,0),1,MPIU_SCALAR,MPIU_SUM,comm,&req(0)));
    PetscCall(VecCopy(p,plcg->Z[l]));

    PetscCall(KSPSolve_InnerLoop_PIPELCG(ksp));

    if (ksp->reason) break; /* convergence or divergence */
    ++ outer_it;
    curr_guess_zero = 0;
  }

  if (!ksp->reason) ksp->reason = KSP_DIVERGED_ITS;
  PetscCall(PetscFree(plcg->G));
  PetscCall(PetscFree(plcg->gamma));
  PetscCall(PetscFree(plcg->delta));
  PetscCall(PetscFree(plcg->req));
  PetscFunctionReturn(0);
}

/*MC
    KSPPIPELCG - Deep pipelined (length l) Conjugate Gradient method. This method has only a single non-blocking global
    reduction per iteration, compared to 2 blocking reductions for standard CG. The reduction is overlapped by the
    matrix-vector product and preconditioner application of the next l iterations. The pipeline length l is a parameter
    of the method.

    Options Database Keys:
+   -ksp_pipelcg_pipel - pipelined length
.   -ksp_pipelcg_lmin - approximation to the smallest eigenvalue of the preconditioned operator (default: 0.0)
.   -ksp_pipelcg_lmax - approximation to the largest eigenvalue of the preconditioned operator (default: 0.0)
-   -ksp_pipelcg_monitor - output where/why the method restarts when a sqrt breakdown occurs

   see KSPSolve() for additional options

    Level: advanced

    Notes:
    MPI configuration may be necessary for reductions to make asynchronous progress, which is important for
    performance of pipelined methods. See the FAQ on the PETSc website for details.

    Contributed by:
    Siegfried Cools, University of Antwerp, Dept. Mathematics and Computer Science,
    funded by Flemish Research Foundation (FWO) grant number 12H4617N.

    Example usage:
    [*] KSP ex2, no preconditioner, pipel = 2, lmin = 0.0, lmax = 8.0 :
        $mpiexec -n 14 ./ex2 -m 1000 -n 1000 -ksp_type pipelcg -pc_type none -ksp_norm_type natural
        -ksp_rtol 1e-10 -ksp_max_it 1000 -ksp_pipelcg_pipel 2 -ksp_pipelcg_lmin 0.0 -ksp_pipelcg_lmax 8.0 -log_view
    [*] SNES ex48, bjacobi preconditioner, pipel = 3, lmin = 0.0, lmax = 2.0, show restart information :
        $mpiexec -n 14 ./ex48 -M 150 -P 100 -ksp_type pipelcg -pc_type bjacobi -ksp_rtol 1e-10 -ksp_pipelcg_pipel 3
        -ksp_pipelcg_lmin 0.0 -ksp_pipelcg_lmax 2.0 -ksp_pipelcg_monitor -log_view

    References:
+   * - J. Cornelis, S. Cools and W. Vanroose,
        "The Communication-Hiding Conjugate Gradient Method with Deep Pipelines"
        Submitted to SIAM Journal on Scientific Computing (SISC), 2018.
-   * - S. Cools, J. Cornelis and W. Vanroose,
        "Numerically Stable Recurrence Relations for the Communication Hiding Pipelined Conjugate Gradient Method"
        Submitted to IEEE Transactions on Parallel and Distributed Systems, 2019.

.seealso:  KSPCreate(), KSPSetType(), KSPType (for list of available types), KSPCG, KSPPIPECG, KSPPIPECGRR, KSPPGMRES,
    KSPPIPEBCGS, KSPSetPCSide()
M*/
PETSC_EXTERN
PetscErrorCode KSPCreate_PIPELCG(KSP ksp)
{
  KSP_CG_PIPE_L  *plcg = NULL;

  PetscFunctionBegin;
  PetscCall(PetscNewLog(ksp,&plcg));
  ksp->data = (void*)plcg;

  PetscCall(KSPSetSupportedNorm(ksp,KSP_NORM_NONE,PC_LEFT,1));
  PetscCall(KSPSetSupportedNorm(ksp,KSP_NORM_NATURAL,PC_LEFT,2));

  ksp->ops->setup          = KSPSetUp_PIPELCG;
  ksp->ops->solve          = KSPSolve_PIPELCG;
  ksp->ops->reset          = KSPReset_PIPELCG;
  ksp->ops->destroy        = KSPDestroy_PIPELCG;
  ksp->ops->view           = KSPView_PIPELCG;
  ksp->ops->setfromoptions = KSPSetFromOptions_PIPELCG;
  ksp->ops->buildsolution  = KSPBuildSolutionDefault;
  ksp->ops->buildresidual  = KSPBuildResidualDefault;
  PetscFunctionReturn(0);
}
