
static char help[] = "Sensitivity analysis applied in power grid stability analysis of WECC 9 bus system.\n\
This example is based on the 9-bus (node) example given in the book Power\n\
Systems Dynamics and Stability (Chapter 7) by P. Sauer and M. A. Pai.\n\
The power grid in this example consists of 9 buses (nodes), 3 generators,\n\
3 loads, and 9 transmission lines. The network equations are written\n\
in current balance form using rectangular coordinates.\n\n";

/*
   The equations for the stability analysis are described by the DAE. See ex9bus.c for details.
   The system has 'jumps' due to faults, thus the time interval is split into multiple sections, and TSSolve is called for each of them. But TSAdjointSolve only needs to be called once since the whole trajectory has been saved in the forward run.
   The code computes the sensitivity of a final state w.r.t. initial conditions.
*/

#include <petscts.h>
#include <petscdm.h>
#include <petscdmda.h>
#include <petscdmcomposite.h>

#define freq 60
#define w_s (2*PETSC_PI*freq)

/* Sizes and indices */
const PetscInt nbus    = 9; /* Number of network buses */
const PetscInt ngen    = 3; /* Number of generators */
const PetscInt nload   = 3; /* Number of loads */
const PetscInt gbus[3] = {0,1,2}; /* Buses at which generators are incident */
const PetscInt lbus[3] = {4,5,7}; /* Buses at which loads are incident */

/* Generator real and reactive powers (found via loadflow) */
const PetscScalar PG[3] = {0.716786142395021,1.630000000000000,0.850000000000000};
const PetscScalar QG[3] = {0.270702180178785,0.066120127797275,-0.108402221791588};
/* Generator constants */
const PetscScalar H[3]    = {23.64,6.4,3.01};   /* Inertia constant */
const PetscScalar Rs[3]   = {0.0,0.0,0.0}; /* Stator Resistance */
const PetscScalar Xd[3]   = {0.146,0.8958,1.3125};  /* d-axis reactance */
const PetscScalar Xdp[3]  = {0.0608,0.1198,0.1813}; /* d-axis transient reactance */
const PetscScalar Xq[3]   = {0.4360,0.8645,1.2578}; /* q-axis reactance Xq(1) set to 0.4360, value given in text 0.0969 */
const PetscScalar Xqp[3]  = {0.0969,0.1969,0.25}; /* q-axis transient reactance */
const PetscScalar Td0p[3] = {8.96,6.0,5.89}; /* d-axis open circuit time constant */
const PetscScalar Tq0p[3] = {0.31,0.535,0.6}; /* q-axis open circuit time constant */
PetscScalar M[3]; /* M = 2*H/w_s */
PetscScalar D[3]; /* D = 0.1*M */

PetscScalar TM[3]; /* Mechanical Torque */
/* Exciter system constants */
const PetscScalar KA[3] = {20.0,20.0,20.0};  /* Voltage regulartor gain constant */
const PetscScalar TA[3] = {0.2,0.2,0.2};     /* Voltage regulator time constant */
const PetscScalar KE[3] = {1.0,1.0,1.0};     /* Exciter gain constant */
const PetscScalar TE[3] = {0.314,0.314,0.314}; /* Exciter time constant */
const PetscScalar KF[3] = {0.063,0.063,0.063};  /* Feedback stabilizer gain constant */
const PetscScalar TF[3] = {0.35,0.35,0.35};    /* Feedback stabilizer time constant */
const PetscScalar k1[3] = {0.0039,0.0039,0.0039};
const PetscScalar k2[3] = {1.555,1.555,1.555};  /* k1 and k2 for calculating the saturation function SE = k1*exp(k2*Efd) */

PetscScalar Vref[3];
/* Load constants
  We use a composite load model that describes the load and reactive powers at each time instant as follows
  P(t) = \sum\limits_{i=0}^ld_nsegsp \ld_alphap_i*P_D0(\frac{V_m(t)}{V_m0})^\ld_betap_i
  Q(t) = \sum\limits_{i=0}^ld_nsegsq \ld_alphaq_i*Q_D0(\frac{V_m(t)}{V_m0})^\ld_betaq_i
  where
    ld_nsegsp,ld_nsegsq - Number of individual load models for real and reactive power loads
    ld_alphap,ld_alphap - Percentage contribution (weights) or loads
    P_D0                - Real power load
    Q_D0                - Reactive power load
    V_m(t)              - Voltage magnitude at time t
    V_m0                - Voltage magnitude at t = 0
    ld_betap, ld_betaq  - exponents describing the load model for real and reactive part

    Note: All loads have the same characteristic currently.
*/
const PetscScalar PD0[3] = {1.25,0.9,1.0};
const PetscScalar QD0[3] = {0.5,0.3,0.35};
const PetscInt    ld_nsegsp[3] = {3,3,3};
const PetscScalar ld_alphap[3] = {1.0,0.0,0.0};
const PetscScalar ld_betap[3]  = {2.0,1.0,0.0};
const PetscInt    ld_nsegsq[3] = {3,3,3};
const PetscScalar ld_alphaq[3] = {1.0,0.0,0.0};
const PetscScalar ld_betaq[3]  = {2.0,1.0,0.0};

typedef struct {
  DM          dmgen, dmnet; /* DMs to manage generator and network subsystem */
  DM          dmpgrid; /* Composite DM to manage the entire power grid */
  Mat         Ybus; /* Network admittance matrix */
  Vec         V0;  /* Initial voltage vector (Power flow solution) */
  PetscReal   tfaulton,tfaultoff; /* Fault on and off times */
  PetscInt    faultbus; /* Fault bus */
  PetscScalar Rfault;
  PetscReal   t0,tmax;
  PetscInt    neqs_gen,neqs_net,neqs_pgrid;
  PetscBool   alg_flg;
  PetscReal   t;
  IS          is_diff; /* indices for differential equations */
  IS          is_alg; /* indices for algebraic equations */
} Userctx;

/* Converts from machine frame (dq) to network (phase a real,imag) reference frame */
PetscErrorCode dq2ri(PetscScalar Fd,PetscScalar Fq,PetscScalar delta,PetscScalar *Fr, PetscScalar *Fi)
{
  PetscFunctionBegin;
  *Fr =  Fd*PetscSinScalar(delta) + Fq*PetscCosScalar(delta);
  *Fi = -Fd*PetscCosScalar(delta) + Fq*PetscSinScalar(delta);
  PetscFunctionReturn(0);
}

/* Converts from network frame ([phase a real,imag) to machine (dq) reference frame */
PetscErrorCode ri2dq(PetscScalar Fr,PetscScalar Fi,PetscScalar delta,PetscScalar *Fd, PetscScalar *Fq)
{
  PetscFunctionBegin;
  *Fd =  Fr*PetscSinScalar(delta) - Fi*PetscCosScalar(delta);
  *Fq =  Fr*PetscCosScalar(delta) + Fi*PetscSinScalar(delta);
  PetscFunctionReturn(0);
}

PetscErrorCode SetInitialGuess(Vec X,Userctx *user)
{
  Vec            Xgen,Xnet;
  PetscScalar    *xgen,*xnet;
  PetscInt       i,idx=0;
  PetscScalar    Vr,Vi,IGr,IGi,Vm,Vm2;
  PetscScalar    Eqp,Edp,delta;
  PetscScalar    Efd,RF,VR; /* Exciter variables */
  PetscScalar    Id,Iq;  /* Generator dq axis currents */
  PetscScalar    theta,Vd,Vq,SE;

  PetscFunctionBegin;
  M[0] = 2*H[0]/w_s; M[1] = 2*H[1]/w_s; M[2] = 2*H[2]/w_s;
  D[0] = 0.1*M[0]; D[1] = 0.1*M[1]; D[2] = 0.1*M[2];

  CHKERRQ(DMCompositeGetLocalVectors(user->dmpgrid,&Xgen,&Xnet));

  /* Network subsystem initialization */
  CHKERRQ(VecCopy(user->V0,Xnet));

  /* Generator subsystem initialization */
  CHKERRQ(VecGetArray(Xgen,&xgen));
  CHKERRQ(VecGetArray(Xnet,&xnet));

  for (i=0; i < ngen; i++) {
    Vr  = xnet[2*gbus[i]]; /* Real part of generator terminal voltage */
    Vi  = xnet[2*gbus[i]+1]; /* Imaginary part of the generator terminal voltage */
    Vm  = PetscSqrtScalar(Vr*Vr + Vi*Vi); Vm2 = Vm*Vm;
    IGr = (Vr*PG[i] + Vi*QG[i])/Vm2;
    IGi = (Vi*PG[i] - Vr*QG[i])/Vm2;

    delta = PetscAtan2Real(Vi+Xq[i]*IGr,Vr-Xq[i]*IGi); /* Machine angle */

    theta = PETSC_PI/2.0 - delta;

    Id = IGr*PetscCosScalar(theta) - IGi*PetscSinScalar(theta); /* d-axis stator current */
    Iq = IGr*PetscSinScalar(theta) + IGi*PetscCosScalar(theta); /* q-axis stator current */

    Vd = Vr*PetscCosScalar(theta) - Vi*PetscSinScalar(theta);
    Vq = Vr*PetscSinScalar(theta) + Vi*PetscCosScalar(theta);

    Edp = Vd + Rs[i]*Id - Xqp[i]*Iq; /* d-axis transient EMF */
    Eqp = Vq + Rs[i]*Iq + Xdp[i]*Id; /* q-axis transient EMF */

    TM[i] = PG[i];

    /* The generator variables are ordered as [Eqp,Edp,delta,w,Id,Iq] */
    xgen[idx]   = Eqp;
    xgen[idx+1] = Edp;
    xgen[idx+2] = delta;
    xgen[idx+3] = w_s;

    idx = idx + 4;

    xgen[idx]   = Id;
    xgen[idx+1] = Iq;

    idx = idx + 2;

    /* Exciter */
    Efd = Eqp + (Xd[i] - Xdp[i])*Id;
    SE  = k1[i]*PetscExpScalar(k2[i]*Efd);
    VR  =  KE[i]*Efd + SE;
    RF  =  KF[i]*Efd/TF[i];

    xgen[idx]   = Efd;
    xgen[idx+1] = RF;
    xgen[idx+2] = VR;

    Vref[i] = Vm + (VR/KA[i]);

    idx = idx + 3;
  }

  CHKERRQ(VecRestoreArray(Xgen,&xgen));
  CHKERRQ(VecRestoreArray(Xnet,&xnet));

  /* CHKERRQ(VecView(Xgen,0)); */
  CHKERRQ(DMCompositeGather(user->dmpgrid,INSERT_VALUES,X,Xgen,Xnet));
  CHKERRQ(DMCompositeRestoreLocalVectors(user->dmpgrid,&Xgen,&Xnet));
  PetscFunctionReturn(0);
}

/* Computes F = [f(x,y);g(x,y)] */
PetscErrorCode ResidualFunction(SNES snes,Vec X, Vec F, Userctx *user)
{
  Vec            Xgen,Xnet,Fgen,Fnet;
  PetscScalar    *xgen,*xnet,*fgen,*fnet;
  PetscInt       i,idx=0;
  PetscScalar    Vr,Vi,Vm,Vm2;
  PetscScalar    Eqp,Edp,delta,w; /* Generator variables */
  PetscScalar    Efd,RF,VR; /* Exciter variables */
  PetscScalar    Id,Iq;  /* Generator dq axis currents */
  PetscScalar    Vd,Vq,SE;
  PetscScalar    IGr,IGi,IDr,IDi;
  PetscScalar    Zdq_inv[4],det;
  PetscScalar    PD,QD,Vm0,*v0;
  PetscInt       k;

  PetscFunctionBegin;
  CHKERRQ(VecZeroEntries(F));
  CHKERRQ(DMCompositeGetLocalVectors(user->dmpgrid,&Xgen,&Xnet));
  CHKERRQ(DMCompositeGetLocalVectors(user->dmpgrid,&Fgen,&Fnet));
  CHKERRQ(DMCompositeScatter(user->dmpgrid,X,Xgen,Xnet));
  CHKERRQ(DMCompositeScatter(user->dmpgrid,F,Fgen,Fnet));

  /* Network current balance residual IG + Y*V + IL = 0. Only YV is added here.
     The generator current injection, IG, and load current injection, ID are added later
  */
  /* Note that the values in Ybus are stored assuming the imaginary current balance
     equation is ordered first followed by real current balance equation for each bus.
     Thus imaginary current contribution goes in location 2*i, and
     real current contribution in 2*i+1
  */
  CHKERRQ(MatMult(user->Ybus,Xnet,Fnet));

  CHKERRQ(VecGetArray(Xgen,&xgen));
  CHKERRQ(VecGetArray(Xnet,&xnet));
  CHKERRQ(VecGetArray(Fgen,&fgen));
  CHKERRQ(VecGetArray(Fnet,&fnet));

  /* Generator subsystem */
  for (i=0; i < ngen; i++) {
    Eqp   = xgen[idx];
    Edp   = xgen[idx+1];
    delta = xgen[idx+2];
    w     = xgen[idx+3];
    Id    = xgen[idx+4];
    Iq    = xgen[idx+5];
    Efd   = xgen[idx+6];
    RF    = xgen[idx+7];
    VR    = xgen[idx+8];

    /* Generator differential equations */
    fgen[idx]   = (Eqp + (Xd[i] - Xdp[i])*Id - Efd)/Td0p[i];
    fgen[idx+1] = (Edp - (Xq[i] - Xqp[i])*Iq)/Tq0p[i];
    fgen[idx+2] = -w + w_s;
    fgen[idx+3] = (-TM[i] + Edp*Id + Eqp*Iq + (Xqp[i] - Xdp[i])*Id*Iq + D[i]*(w - w_s))/M[i];

    Vr = xnet[2*gbus[i]]; /* Real part of generator terminal voltage */
    Vi = xnet[2*gbus[i]+1]; /* Imaginary part of the generator terminal voltage */

    CHKERRQ(ri2dq(Vr,Vi,delta,&Vd,&Vq));
    /* Algebraic equations for stator currents */

    det = Rs[i]*Rs[i] + Xdp[i]*Xqp[i];

    Zdq_inv[0] = Rs[i]/det;
    Zdq_inv[1] = Xqp[i]/det;
    Zdq_inv[2] = -Xdp[i]/det;
    Zdq_inv[3] = Rs[i]/det;

    fgen[idx+4] = Zdq_inv[0]*(-Edp + Vd) + Zdq_inv[1]*(-Eqp + Vq) + Id;
    fgen[idx+5] = Zdq_inv[2]*(-Edp + Vd) + Zdq_inv[3]*(-Eqp + Vq) + Iq;

    /* Add generator current injection to network */
    CHKERRQ(dq2ri(Id,Iq,delta,&IGr,&IGi));

    fnet[2*gbus[i]]   -= IGi;
    fnet[2*gbus[i]+1] -= IGr;

    Vm = PetscSqrtScalar(Vd*Vd + Vq*Vq);

    SE = k1[i]*PetscExpScalar(k2[i]*Efd);

    /* Exciter differential equations */
    fgen[idx+6] = (KE[i]*Efd + SE - VR)/TE[i];
    fgen[idx+7] = (RF - KF[i]*Efd/TF[i])/TF[i];
    fgen[idx+8] = (VR - KA[i]*RF + KA[i]*KF[i]*Efd/TF[i] - KA[i]*(Vref[i] - Vm))/TA[i];

    idx = idx + 9;
  }

  CHKERRQ(VecGetArray(user->V0,&v0));
  for (i=0; i < nload; i++) {
    Vr  = xnet[2*lbus[i]]; /* Real part of load bus voltage */
    Vi  = xnet[2*lbus[i]+1]; /* Imaginary part of the load bus voltage */
    Vm  = PetscSqrtScalar(Vr*Vr + Vi*Vi); Vm2 = Vm*Vm;
    Vm0 = PetscSqrtScalar(v0[2*lbus[i]]*v0[2*lbus[i]] + v0[2*lbus[i]+1]*v0[2*lbus[i]+1]);
    PD  = QD = 0.0;
    for (k=0; k < ld_nsegsp[i]; k++) PD += ld_alphap[k]*PD0[i]*PetscPowScalar((Vm/Vm0),ld_betap[k]);
    for (k=0; k < ld_nsegsq[i]; k++) QD += ld_alphaq[k]*QD0[i]*PetscPowScalar((Vm/Vm0),ld_betaq[k]);

    /* Load currents */
    IDr = (PD*Vr + QD*Vi)/Vm2;
    IDi = (-QD*Vr + PD*Vi)/Vm2;

    fnet[2*lbus[i]]   += IDi;
    fnet[2*lbus[i]+1] += IDr;
  }
  CHKERRQ(VecRestoreArray(user->V0,&v0));

  CHKERRQ(VecRestoreArray(Xgen,&xgen));
  CHKERRQ(VecRestoreArray(Xnet,&xnet));
  CHKERRQ(VecRestoreArray(Fgen,&fgen));
  CHKERRQ(VecRestoreArray(Fnet,&fnet));

  CHKERRQ(DMCompositeGather(user->dmpgrid,INSERT_VALUES,F,Fgen,Fnet));
  CHKERRQ(DMCompositeRestoreLocalVectors(user->dmpgrid,&Xgen,&Xnet));
  CHKERRQ(DMCompositeRestoreLocalVectors(user->dmpgrid,&Fgen,&Fnet));
  PetscFunctionReturn(0);
}

/* \dot{x} - f(x,y)
     g(x,y) = 0
 */
PetscErrorCode IFunction(TS ts,PetscReal t, Vec X, Vec Xdot, Vec F, Userctx *user)
{
  SNES              snes;
  PetscScalar       *f;
  const PetscScalar *xdot;
  PetscInt          i;

  PetscFunctionBegin;
  user->t = t;

  CHKERRQ(TSGetSNES(ts,&snes));
  CHKERRQ(ResidualFunction(snes,X,F,user));
  CHKERRQ(VecGetArray(F,&f));
  CHKERRQ(VecGetArrayRead(Xdot,&xdot));
  for (i=0;i < ngen;i++) {
    f[9*i]   += xdot[9*i];
    f[9*i+1] += xdot[9*i+1];
    f[9*i+2] += xdot[9*i+2];
    f[9*i+3] += xdot[9*i+3];
    f[9*i+6] += xdot[9*i+6];
    f[9*i+7] += xdot[9*i+7];
    f[9*i+8] += xdot[9*i+8];
  }
  CHKERRQ(VecRestoreArray(F,&f));
  CHKERRQ(VecRestoreArrayRead(Xdot,&xdot));
  PetscFunctionReturn(0);
}

/* This function is used for solving the algebraic system only during fault on and
   off times. It computes the entire F and then zeros out the part corresponding to
   differential equations
 F = [0;g(y)];
*/
PetscErrorCode AlgFunction(SNES snes, Vec X, Vec F, void *ctx)
{
  Userctx        *user=(Userctx*)ctx;
  PetscScalar    *f;
  PetscInt       i;

  PetscFunctionBegin;
  CHKERRQ(ResidualFunction(snes,X,F,user));
  CHKERRQ(VecGetArray(F,&f));
  for (i=0; i < ngen; i++) {
    f[9*i]   = 0;
    f[9*i+1] = 0;
    f[9*i+2] = 0;
    f[9*i+3] = 0;
    f[9*i+6] = 0;
    f[9*i+7] = 0;
    f[9*i+8] = 0;
  }
  CHKERRQ(VecRestoreArray(F,&f));
  PetscFunctionReturn(0);
}

PetscErrorCode PreallocateJacobian(Mat J, Userctx *user)
{
  PetscInt       *d_nnz;
  PetscInt       i,idx=0,start=0;
  PetscInt       ncols;

  PetscFunctionBegin;
  CHKERRQ(PetscMalloc1(user->neqs_pgrid,&d_nnz));
  for (i=0; i<user->neqs_pgrid; i++) d_nnz[i] = 0;
  /* Generator subsystem */
  for (i=0; i < ngen; i++) {

    d_nnz[idx]   += 3;
    d_nnz[idx+1] += 2;
    d_nnz[idx+2] += 2;
    d_nnz[idx+3] += 5;
    d_nnz[idx+4] += 6;
    d_nnz[idx+5] += 6;

    d_nnz[user->neqs_gen+2*gbus[i]]   += 3;
    d_nnz[user->neqs_gen+2*gbus[i]+1] += 3;

    d_nnz[idx+6] += 2;
    d_nnz[idx+7] += 2;
    d_nnz[idx+8] += 5;

    idx = idx + 9;
  }

  start = user->neqs_gen;

  for (i=0; i < nbus; i++) {
    CHKERRQ(MatGetRow(user->Ybus,2*i,&ncols,NULL,NULL));
    d_nnz[start+2*i]   += ncols;
    d_nnz[start+2*i+1] += ncols;
    CHKERRQ(MatRestoreRow(user->Ybus,2*i,&ncols,NULL,NULL));
  }

  CHKERRQ(MatSeqAIJSetPreallocation(J,0,d_nnz));

  CHKERRQ(PetscFree(d_nnz));
  PetscFunctionReturn(0);
}

/*
   J = [-df_dx, -df_dy
        dg_dx, dg_dy]
*/
PetscErrorCode ResidualJacobian(SNES snes,Vec X,Mat J,Mat B,void *ctx)
{
  Userctx           *user=(Userctx*)ctx;
  Vec               Xgen,Xnet;
  PetscScalar       *xgen,*xnet;
  PetscInt          i,idx=0;
  PetscScalar       Vr,Vi,Vm,Vm2;
  PetscScalar       Eqp,Edp,delta; /* Generator variables */
  PetscScalar       Efd; /* Exciter variables */
  PetscScalar       Id,Iq;  /* Generator dq axis currents */
  PetscScalar       Vd,Vq;
  PetscScalar       val[10];
  PetscInt          row[2],col[10];
  PetscInt          net_start=user->neqs_gen;
  PetscScalar       Zdq_inv[4],det;
  PetscScalar       dVd_dVr,dVd_dVi,dVq_dVr,dVq_dVi,dVd_ddelta,dVq_ddelta;
  PetscScalar       dIGr_ddelta,dIGi_ddelta,dIGr_dId,dIGr_dIq,dIGi_dId,dIGi_dIq;
  PetscScalar       dSE_dEfd;
  PetscScalar       dVm_dVd,dVm_dVq,dVm_dVr,dVm_dVi;
  PetscInt          ncols;
  const PetscInt    *cols;
  const PetscScalar *yvals;
  PetscInt          k;
  PetscScalar       PD,QD,Vm0,*v0,Vm4;
  PetscScalar       dPD_dVr,dPD_dVi,dQD_dVr,dQD_dVi;
  PetscScalar       dIDr_dVr,dIDr_dVi,dIDi_dVr,dIDi_dVi;

  PetscFunctionBegin;
  CHKERRQ(MatZeroEntries(B));
  CHKERRQ(DMCompositeGetLocalVectors(user->dmpgrid,&Xgen,&Xnet));
  CHKERRQ(DMCompositeScatter(user->dmpgrid,X,Xgen,Xnet));

  CHKERRQ(VecGetArray(Xgen,&xgen));
  CHKERRQ(VecGetArray(Xnet,&xnet));

  /* Generator subsystem */
  for (i=0; i < ngen; i++) {
    Eqp   = xgen[idx];
    Edp   = xgen[idx+1];
    delta = xgen[idx+2];
    Id    = xgen[idx+4];
    Iq    = xgen[idx+5];
    Efd   = xgen[idx+6];

    /*    fgen[idx]   = (Eqp + (Xd[i] - Xdp[i])*Id - Efd)/Td0p[i]; */
    row[0] = idx;
    col[0] = idx;           col[1] = idx+4;          col[2] = idx+6;
    val[0] = 1/ Td0p[i]; val[1] = (Xd[i] - Xdp[i])/ Td0p[i]; val[2] = -1/Td0p[i];

    CHKERRQ(MatSetValues(J,1,row,3,col,val,INSERT_VALUES));

    /*    fgen[idx+1] = (Edp - (Xq[i] - Xqp[i])*Iq)/Tq0p[i]; */
    row[0] = idx + 1;
    col[0] = idx + 1;       col[1] = idx+5;
    val[0] = 1/Tq0p[i]; val[1] = -(Xq[i] - Xqp[i])/Tq0p[i];
    CHKERRQ(MatSetValues(J,1,row,2,col,val,INSERT_VALUES));

    /*    fgen[idx+2] = - w + w_s; */
    row[0] = idx + 2;
    col[0] = idx + 2; col[1] = idx + 3;
    val[0] = 0;       val[1] = -1;
    CHKERRQ(MatSetValues(J,1,row,2,col,val,INSERT_VALUES));

    /*    fgen[idx+3] = (-TM[i] + Edp*Id + Eqp*Iq + (Xqp[i] - Xdp[i])*Id*Iq + D[i]*(w - w_s))/M[i]; */
    row[0] = idx + 3;
    col[0] = idx; col[1] = idx + 1; col[2] = idx + 3;       col[3] = idx + 4;                  col[4] = idx + 5;
    val[0] = Iq/M[i];  val[1] = Id/M[i];      val[2] = D[i]/M[i]; val[3] = (Edp + (Xqp[i]-Xdp[i])*Iq)/M[i]; val[4] = (Eqp + (Xqp[i] - Xdp[i])*Id)/M[i];
    CHKERRQ(MatSetValues(J,1,row,5,col,val,INSERT_VALUES));

    Vr   = xnet[2*gbus[i]]; /* Real part of generator terminal voltage */
    Vi   = xnet[2*gbus[i]+1]; /* Imaginary part of the generator terminal voltage */
    CHKERRQ(ri2dq(Vr,Vi,delta,&Vd,&Vq));

    det = Rs[i]*Rs[i] + Xdp[i]*Xqp[i];

    Zdq_inv[0] = Rs[i]/det;
    Zdq_inv[1] = Xqp[i]/det;
    Zdq_inv[2] = -Xdp[i]/det;
    Zdq_inv[3] = Rs[i]/det;

    dVd_dVr    = PetscSinScalar(delta); dVd_dVi = -PetscCosScalar(delta);
    dVq_dVr    = PetscCosScalar(delta); dVq_dVi = PetscSinScalar(delta);
    dVd_ddelta = Vr*PetscCosScalar(delta) + Vi*PetscSinScalar(delta);
    dVq_ddelta = -Vr*PetscSinScalar(delta) + Vi*PetscCosScalar(delta);

    /*    fgen[idx+4] = Zdq_inv[0]*(-Edp + Vd) + Zdq_inv[1]*(-Eqp + Vq) + Id; */
    row[0] = idx+4;
    col[0] = idx;         col[1] = idx+1;        col[2] = idx + 2;
    val[0] = -Zdq_inv[1]; val[1] = -Zdq_inv[0];  val[2] = Zdq_inv[0]*dVd_ddelta + Zdq_inv[1]*dVq_ddelta;
    col[3] = idx + 4; col[4] = net_start+2*gbus[i];                     col[5] = net_start + 2*gbus[i]+1;
    val[3] = 1;       val[4] = Zdq_inv[0]*dVd_dVr + Zdq_inv[1]*dVq_dVr; val[5] = Zdq_inv[0]*dVd_dVi + Zdq_inv[1]*dVq_dVi;
    CHKERRQ(MatSetValues(J,1,row,6,col,val,INSERT_VALUES));

    /*  fgen[idx+5] = Zdq_inv[2]*(-Edp + Vd) + Zdq_inv[3]*(-Eqp + Vq) + Iq; */
    row[0] = idx+5;
    col[0] = idx;         col[1] = idx+1;        col[2] = idx + 2;
    val[0] = -Zdq_inv[3]; val[1] = -Zdq_inv[2];  val[2] = Zdq_inv[2]*dVd_ddelta + Zdq_inv[3]*dVq_ddelta;
    col[3] = idx + 5; col[4] = net_start+2*gbus[i];                     col[5] = net_start + 2*gbus[i]+1;
    val[3] = 1;       val[4] = Zdq_inv[2]*dVd_dVr + Zdq_inv[3]*dVq_dVr; val[5] = Zdq_inv[2]*dVd_dVi + Zdq_inv[3]*dVq_dVi;
    CHKERRQ(MatSetValues(J,1,row,6,col,val,INSERT_VALUES));

    dIGr_ddelta = Id*PetscCosScalar(delta) - Iq*PetscSinScalar(delta);
    dIGi_ddelta = Id*PetscSinScalar(delta) + Iq*PetscCosScalar(delta);
    dIGr_dId    = PetscSinScalar(delta);  dIGr_dIq = PetscCosScalar(delta);
    dIGi_dId    = -PetscCosScalar(delta); dIGi_dIq = PetscSinScalar(delta);

    /* fnet[2*gbus[i]]   -= IGi; */
    row[0] = net_start + 2*gbus[i];
    col[0] = idx+2;        col[1] = idx + 4;   col[2] = idx + 5;
    val[0] = -dIGi_ddelta; val[1] = -dIGi_dId; val[2] = -dIGi_dIq;
    CHKERRQ(MatSetValues(J,1,row,3,col,val,INSERT_VALUES));

    /* fnet[2*gbus[i]+1]   -= IGr; */
    row[0] = net_start + 2*gbus[i]+1;
    col[0] = idx+2;        col[1] = idx + 4;   col[2] = idx + 5;
    val[0] = -dIGr_ddelta; val[1] = -dIGr_dId; val[2] = -dIGr_dIq;
    CHKERRQ(MatSetValues(J,1,row,3,col,val,INSERT_VALUES));

    Vm = PetscSqrtScalar(Vd*Vd + Vq*Vq);

    /*    fgen[idx+6] = (KE[i]*Efd + SE - VR)/TE[i]; */
    /*    SE  = k1[i]*PetscExpScalar(k2[i]*Efd); */
    dSE_dEfd = k1[i]*k2[i]*PetscExpScalar(k2[i]*Efd);

    row[0] = idx + 6;
    col[0] = idx + 6;                     col[1] = idx + 8;
    val[0] = (KE[i] + dSE_dEfd)/TE[i];  val[1] = -1/TE[i];
    CHKERRQ(MatSetValues(J,1,row,2,col,val,INSERT_VALUES));

    /* Exciter differential equations */

    /*    fgen[idx+7] = (RF - KF[i]*Efd/TF[i])/TF[i]; */
    row[0] = idx + 7;
    col[0] = idx + 6;       col[1] = idx + 7;
    val[0] = (-KF[i]/TF[i])/TF[i];  val[1] = 1/TF[i];
    CHKERRQ(MatSetValues(J,1,row,2,col,val,INSERT_VALUES));

    /*    fgen[idx+8] = (VR - KA[i]*RF + KA[i]*KF[i]*Efd/TF[i] - KA[i]*(Vref[i] - Vm))/TA[i]; */
    /* Vm = (Vd^2 + Vq^2)^0.5; */
    dVm_dVd    = Vd/Vm; dVm_dVq = Vq/Vm;
    dVm_dVr    = dVm_dVd*dVd_dVr + dVm_dVq*dVq_dVr;
    dVm_dVi    = dVm_dVd*dVd_dVi + dVm_dVq*dVq_dVi;
    row[0]     = idx + 8;
    col[0]     = idx + 6;           col[1] = idx + 7; col[2] = idx + 8;
    val[0]     = (KA[i]*KF[i]/TF[i])/TA[i]; val[1] = -KA[i]/TA[i];  val[2] = 1/TA[i];
    col[3]     = net_start + 2*gbus[i]; col[4] = net_start + 2*gbus[i]+1;
    val[3]     = KA[i]*dVm_dVr/TA[i];         val[4] = KA[i]*dVm_dVi/TA[i];
    CHKERRQ(MatSetValues(J,1,row,5,col,val,INSERT_VALUES));
    idx        = idx + 9;
  }

  for (i=0; i<nbus; i++) {
    CHKERRQ(MatGetRow(user->Ybus,2*i,&ncols,&cols,&yvals));
    row[0] = net_start + 2*i;
    for (k=0; k<ncols; k++) {
      col[k] = net_start + cols[k];
      val[k] = yvals[k];
    }
    CHKERRQ(MatSetValues(J,1,row,ncols,col,val,INSERT_VALUES));
    CHKERRQ(MatRestoreRow(user->Ybus,2*i,&ncols,&cols,&yvals));

    CHKERRQ(MatGetRow(user->Ybus,2*i+1,&ncols,&cols,&yvals));
    row[0] = net_start + 2*i+1;
    for (k=0; k<ncols; k++) {
      col[k] = net_start + cols[k];
      val[k] = yvals[k];
    }
    CHKERRQ(MatSetValues(J,1,row,ncols,col,val,INSERT_VALUES));
    CHKERRQ(MatRestoreRow(user->Ybus,2*i+1,&ncols,&cols,&yvals));
  }

  CHKERRQ(MatAssemblyBegin(J,MAT_FLUSH_ASSEMBLY));
  CHKERRQ(MatAssemblyEnd(J,MAT_FLUSH_ASSEMBLY));

  CHKERRQ(VecGetArray(user->V0,&v0));
  for (i=0; i < nload; i++) {
    Vr      = xnet[2*lbus[i]]; /* Real part of load bus voltage */
    Vi      = xnet[2*lbus[i]+1]; /* Imaginary part of the load bus voltage */
    Vm      = PetscSqrtScalar(Vr*Vr + Vi*Vi); Vm2 = Vm*Vm; Vm4 = Vm2*Vm2;
    Vm0     = PetscSqrtScalar(v0[2*lbus[i]]*v0[2*lbus[i]] + v0[2*lbus[i]+1]*v0[2*lbus[i]+1]);
    PD      = QD = 0.0;
    dPD_dVr = dPD_dVi = dQD_dVr = dQD_dVi = 0.0;
    for (k=0; k < ld_nsegsp[i]; k++) {
      PD      += ld_alphap[k]*PD0[i]*PetscPowScalar((Vm/Vm0),ld_betap[k]);
      dPD_dVr += ld_alphap[k]*ld_betap[k]*PD0[i]*PetscPowScalar((1/Vm0),ld_betap[k])*Vr*PetscPowScalar(Vm,(ld_betap[k]-2));
      dPD_dVi += ld_alphap[k]*ld_betap[k]*PD0[i]*PetscPowScalar((1/Vm0),ld_betap[k])*Vi*PetscPowScalar(Vm,(ld_betap[k]-2));
    }
    for (k=0; k < ld_nsegsq[i]; k++) {
      QD      += ld_alphaq[k]*QD0[i]*PetscPowScalar((Vm/Vm0),ld_betaq[k]);
      dQD_dVr += ld_alphaq[k]*ld_betaq[k]*QD0[i]*PetscPowScalar((1/Vm0),ld_betaq[k])*Vr*PetscPowScalar(Vm,(ld_betaq[k]-2));
      dQD_dVi += ld_alphaq[k]*ld_betaq[k]*QD0[i]*PetscPowScalar((1/Vm0),ld_betaq[k])*Vi*PetscPowScalar(Vm,(ld_betaq[k]-2));
    }

    /*    IDr = (PD*Vr + QD*Vi)/Vm2; */
    /*    IDi = (-QD*Vr + PD*Vi)/Vm2; */

    dIDr_dVr = (dPD_dVr*Vr + dQD_dVr*Vi + PD)/Vm2 - ((PD*Vr + QD*Vi)*2*Vr)/Vm4;
    dIDr_dVi = (dPD_dVi*Vr + dQD_dVi*Vi + QD)/Vm2 - ((PD*Vr + QD*Vi)*2*Vi)/Vm4;

    dIDi_dVr = (-dQD_dVr*Vr + dPD_dVr*Vi - QD)/Vm2 - ((-QD*Vr + PD*Vi)*2*Vr)/Vm4;
    dIDi_dVi = (-dQD_dVi*Vr + dPD_dVi*Vi + PD)/Vm2 - ((-QD*Vr + PD*Vi)*2*Vi)/Vm4;

    /*    fnet[2*lbus[i]]   += IDi; */
    row[0] = net_start + 2*lbus[i];
    col[0] = net_start + 2*lbus[i];  col[1] = net_start + 2*lbus[i]+1;
    val[0] = dIDi_dVr;               val[1] = dIDi_dVi;
    CHKERRQ(MatSetValues(J,1,row,2,col,val,ADD_VALUES));
    /*    fnet[2*lbus[i]+1] += IDr; */
    row[0] = net_start + 2*lbus[i]+1;
    col[0] = net_start + 2*lbus[i];  col[1] = net_start + 2*lbus[i]+1;
    val[0] = dIDr_dVr;               val[1] = dIDr_dVi;
    CHKERRQ(MatSetValues(J,1,row,2,col,val,ADD_VALUES));
  }
  CHKERRQ(VecRestoreArray(user->V0,&v0));

  CHKERRQ(VecRestoreArray(Xgen,&xgen));
  CHKERRQ(VecRestoreArray(Xnet,&xnet));

  CHKERRQ(DMCompositeRestoreLocalVectors(user->dmpgrid,&Xgen,&Xnet));

  CHKERRQ(MatAssemblyBegin(J,MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatAssemblyEnd(J,MAT_FINAL_ASSEMBLY));
  PetscFunctionReturn(0);
}

/*
   J = [I, 0
        dg_dx, dg_dy]
*/
PetscErrorCode AlgJacobian(SNES snes,Vec X,Mat A,Mat B,void *ctx)
{
  Userctx        *user=(Userctx*)ctx;

  PetscFunctionBegin;
  CHKERRQ(ResidualJacobian(snes,X,A,B,ctx));
  CHKERRQ(MatSetOption(A,MAT_KEEP_NONZERO_PATTERN,PETSC_TRUE));
  CHKERRQ(MatZeroRowsIS(A,user->is_diff,1.0,NULL,NULL));
  PetscFunctionReturn(0);
}

/*
   J = [a*I-df_dx, -df_dy
        dg_dx, dg_dy]
*/

PetscErrorCode IJacobian(TS ts,PetscReal t,Vec X,Vec Xdot,PetscReal a,Mat A,Mat B,Userctx *user)
{
  SNES           snes;
  PetscScalar    atmp = (PetscScalar) a;
  PetscInt       i,row;

  PetscFunctionBegin;
  user->t = t;

  CHKERRQ(TSGetSNES(ts,&snes));
  CHKERRQ(ResidualJacobian(snes,X,A,B,user));
  for (i=0;i < ngen;i++) {
    row = 9*i;
    CHKERRQ(MatSetValues(A,1,&row,1,&row,&atmp,ADD_VALUES));
    row  = 9*i+1;
    CHKERRQ(MatSetValues(A,1,&row,1,&row,&atmp,ADD_VALUES));
    row  = 9*i+2;
    CHKERRQ(MatSetValues(A,1,&row,1,&row,&atmp,ADD_VALUES));
    row  = 9*i+3;
    CHKERRQ(MatSetValues(A,1,&row,1,&row,&atmp,ADD_VALUES));
    row  = 9*i+6;
    CHKERRQ(MatSetValues(A,1,&row,1,&row,&atmp,ADD_VALUES));
    row  = 9*i+7;
    CHKERRQ(MatSetValues(A,1,&row,1,&row,&atmp,ADD_VALUES));
    row  = 9*i+8;
    CHKERRQ(MatSetValues(A,1,&row,1,&row,&atmp,ADD_VALUES));
  }
  CHKERRQ(MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY));
  PetscFunctionReturn(0);
}

int main(int argc,char **argv)
{
  TS             ts;
  SNES           snes_alg;
  PetscErrorCode ierr;
  PetscMPIInt    size;
  Userctx        user;
  PetscViewer    Xview,Ybusview;
  Vec            X;
  Mat            J;
  PetscInt       i;
  /* sensitivity context */
  PetscScalar    *y_ptr;
  Vec            lambda[1];
  PetscInt       *idx2;
  Vec            Xdot;
  Vec            F_alg;
  PetscInt       row_loc,col_loc;
  PetscScalar    val;

  ierr = PetscInitialize(&argc,&argv,"petscoptions",help);if (ierr) return ierr;
  CHKERRMPI(MPI_Comm_size(PETSC_COMM_WORLD,&size));
  PetscCheck(size == 1,PETSC_COMM_WORLD,PETSC_ERR_WRONG_MPI_SIZE,"Only for sequential runs");

  user.neqs_gen   = 9*ngen; /* # eqs. for generator subsystem */
  user.neqs_net   = 2*nbus; /* # eqs. for network subsystem   */
  user.neqs_pgrid = user.neqs_gen + user.neqs_net;

  /* Create indices for differential and algebraic equations */
  CHKERRQ(PetscMalloc1(7*ngen,&idx2));
  for (i=0; i<ngen; i++) {
    idx2[7*i]   = 9*i;   idx2[7*i+1] = 9*i+1; idx2[7*i+2] = 9*i+2; idx2[7*i+3] = 9*i+3;
    idx2[7*i+4] = 9*i+6; idx2[7*i+5] = 9*i+7; idx2[7*i+6] = 9*i+8;
  }
  CHKERRQ(ISCreateGeneral(PETSC_COMM_WORLD,7*ngen,idx2,PETSC_COPY_VALUES,&user.is_diff));
  CHKERRQ(ISComplement(user.is_diff,0,user.neqs_pgrid,&user.is_alg));
  CHKERRQ(PetscFree(idx2));

  /* Read initial voltage vector and Ybus */
  CHKERRQ(PetscViewerBinaryOpen(PETSC_COMM_WORLD,"X.bin",FILE_MODE_READ,&Xview));
  CHKERRQ(PetscViewerBinaryOpen(PETSC_COMM_WORLD,"Ybus.bin",FILE_MODE_READ,&Ybusview));

  CHKERRQ(VecCreate(PETSC_COMM_WORLD,&user.V0));
  CHKERRQ(VecSetSizes(user.V0,PETSC_DECIDE,user.neqs_net));
  CHKERRQ(VecLoad(user.V0,Xview));

  CHKERRQ(MatCreate(PETSC_COMM_WORLD,&user.Ybus));
  CHKERRQ(MatSetSizes(user.Ybus,PETSC_DECIDE,PETSC_DECIDE,user.neqs_net,user.neqs_net));
  CHKERRQ(MatSetType(user.Ybus,MATBAIJ));
  /*  CHKERRQ(MatSetBlockSize(user.Ybus,2)); */
  CHKERRQ(MatLoad(user.Ybus,Ybusview));

  /* Set run time options */
  ierr = PetscOptionsBegin(PETSC_COMM_WORLD,NULL,"Transient stability fault options","");CHKERRQ(ierr);
  {
    user.tfaulton  = 1.0;
    user.tfaultoff = 1.2;
    user.Rfault    = 0.0001;
    user.faultbus  = 8;
    CHKERRQ(PetscOptionsReal("-tfaulton","","",user.tfaulton,&user.tfaulton,NULL));
    CHKERRQ(PetscOptionsReal("-tfaultoff","","",user.tfaultoff,&user.tfaultoff,NULL));
    CHKERRQ(PetscOptionsInt("-faultbus","","",user.faultbus,&user.faultbus,NULL));
    user.t0        = 0.0;
    user.tmax      = 5.0;
    CHKERRQ(PetscOptionsReal("-t0","","",user.t0,&user.t0,NULL));
    CHKERRQ(PetscOptionsReal("-tmax","","",user.tmax,&user.tmax,NULL));
  }
  ierr = PetscOptionsEnd();CHKERRQ(ierr);

  CHKERRQ(PetscViewerDestroy(&Xview));
  CHKERRQ(PetscViewerDestroy(&Ybusview));

  /* Create DMs for generator and network subsystems */
  CHKERRQ(DMDACreate1d(PETSC_COMM_WORLD,DM_BOUNDARY_NONE,user.neqs_gen,1,1,NULL,&user.dmgen));
  CHKERRQ(DMSetOptionsPrefix(user.dmgen,"dmgen_"));
  CHKERRQ(DMSetFromOptions(user.dmgen));
  CHKERRQ(DMSetUp(user.dmgen));
  CHKERRQ(DMDACreate1d(PETSC_COMM_WORLD,DM_BOUNDARY_NONE,user.neqs_net,1,1,NULL,&user.dmnet));
  CHKERRQ(DMSetOptionsPrefix(user.dmnet,"dmnet_"));
  CHKERRQ(DMSetFromOptions(user.dmnet));
  CHKERRQ(DMSetUp(user.dmnet));
  /* Create a composite DM packer and add the two DMs */
  CHKERRQ(DMCompositeCreate(PETSC_COMM_WORLD,&user.dmpgrid));
  CHKERRQ(DMSetOptionsPrefix(user.dmpgrid,"pgrid_"));
  CHKERRQ(DMCompositeAddDM(user.dmpgrid,user.dmgen));
  CHKERRQ(DMCompositeAddDM(user.dmpgrid,user.dmnet));

  CHKERRQ(DMCreateGlobalVector(user.dmpgrid,&X));

  CHKERRQ(MatCreate(PETSC_COMM_WORLD,&J));
  CHKERRQ(MatSetSizes(J,PETSC_DECIDE,PETSC_DECIDE,user.neqs_pgrid,user.neqs_pgrid));
  CHKERRQ(MatSetFromOptions(J));
  CHKERRQ(PreallocateJacobian(J,&user));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Create timestepping solver context
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  CHKERRQ(TSCreate(PETSC_COMM_WORLD,&ts));
  CHKERRQ(TSSetProblemType(ts,TS_NONLINEAR));
  CHKERRQ(TSSetType(ts,TSCN));
  CHKERRQ(TSSetIFunction(ts,NULL,(TSIFunction) IFunction,&user));
  CHKERRQ(TSSetIJacobian(ts,J,J,(TSIJacobian)IJacobian,&user));
  CHKERRQ(TSSetApplicationContext(ts,&user));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Set initial conditions
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  CHKERRQ(SetInitialGuess(X,&user));
  /* Just to set up the Jacobian structure */
  CHKERRQ(VecDuplicate(X,&Xdot));
  CHKERRQ(IJacobian(ts,0.0,X,Xdot,0.0,J,J,&user));
  CHKERRQ(VecDestroy(&Xdot));

  /*
    Save trajectory of solution so that TSAdjointSolve() may be used
  */
  CHKERRQ(TSSetSaveTrajectory(ts));

  CHKERRQ(TSSetMaxTime(ts,user.tmax));
  CHKERRQ(TSSetTimeStep(ts,0.01));
  CHKERRQ(TSSetExactFinalTime(ts,TS_EXACTFINALTIME_MATCHSTEP));
  CHKERRQ(TSSetFromOptions(ts));

  user.alg_flg = PETSC_FALSE;
  /* Prefault period */
  CHKERRQ(TSSolve(ts,X));

  /* Create the nonlinear solver for solving the algebraic system */
  /* Note that although the algebraic system needs to be solved only for
     Idq and V, we reuse the entire system including xgen. The xgen
     variables are held constant by setting their residuals to 0 and
     putting a 1 on the Jacobian diagonal for xgen rows
  */
  CHKERRQ(VecDuplicate(X,&F_alg));
  CHKERRQ(SNESCreate(PETSC_COMM_WORLD,&snes_alg));
  CHKERRQ(SNESSetFunction(snes_alg,F_alg,AlgFunction,&user));
  CHKERRQ(MatZeroEntries(J));
  CHKERRQ(SNESSetJacobian(snes_alg,J,J,AlgJacobian,&user));
  CHKERRQ(SNESSetOptionsPrefix(snes_alg,"alg_"));
  CHKERRQ(SNESSetFromOptions(snes_alg));

  /* Apply disturbance - resistive fault at user.faultbus */
  /* This is done by adding shunt conductance to the diagonal location
     in the Ybus matrix */
  row_loc = 2*user.faultbus; col_loc = 2*user.faultbus+1; /* Location for G */
  val     = 1/user.Rfault;
  CHKERRQ(MatSetValues(user.Ybus,1,&row_loc,1,&col_loc,&val,ADD_VALUES));
  row_loc = 2*user.faultbus+1; col_loc = 2*user.faultbus; /* Location for G */
  val     = 1/user.Rfault;
  CHKERRQ(MatSetValues(user.Ybus,1,&row_loc,1,&col_loc,&val,ADD_VALUES));

  CHKERRQ(MatAssemblyBegin(user.Ybus,MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatAssemblyEnd(user.Ybus,MAT_FINAL_ASSEMBLY));

  user.alg_flg = PETSC_TRUE;
  /* Solve the algebraic equations */
  CHKERRQ(SNESSolve(snes_alg,NULL,X));

  /* Disturbance period */
  user.alg_flg = PETSC_FALSE;
  CHKERRQ(TSSetTime(ts,user.tfaulton));
  CHKERRQ(TSSetMaxTime(ts,user.tfaultoff));
  CHKERRQ(TSSolve(ts,X));

  /* Remove the fault */
  row_loc = 2*user.faultbus; col_loc = 2*user.faultbus+1;
  val     = -1/user.Rfault;
  CHKERRQ(MatSetValues(user.Ybus,1,&row_loc,1,&col_loc,&val,ADD_VALUES));
  row_loc = 2*user.faultbus+1; col_loc = 2*user.faultbus;
  val     = -1/user.Rfault;
  CHKERRQ(MatSetValues(user.Ybus,1,&row_loc,1,&col_loc,&val,ADD_VALUES));

  CHKERRQ(MatAssemblyBegin(user.Ybus,MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatAssemblyEnd(user.Ybus,MAT_FINAL_ASSEMBLY));

  CHKERRQ(MatZeroEntries(J));

  user.alg_flg = PETSC_TRUE;

  /* Solve the algebraic equations */
  CHKERRQ(SNESSolve(snes_alg,NULL,X));

  /* Post-disturbance period */
  user.alg_flg = PETSC_TRUE;
  CHKERRQ(TSSetTime(ts,user.tfaultoff));
  CHKERRQ(TSSetMaxTime(ts,user.tmax));
  CHKERRQ(TSSolve(ts,X));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Adjoint model starts here
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  CHKERRQ(TSSetPostStep(ts,NULL));
  CHKERRQ(MatCreateVecs(J,&lambda[0],NULL));
  /*   Set initial conditions for the adjoint integration */
  CHKERRQ(VecZeroEntries(lambda[0]));
  CHKERRQ(VecGetArray(lambda[0],&y_ptr));
  y_ptr[0] = 1.0;
  CHKERRQ(VecRestoreArray(lambda[0],&y_ptr));
  CHKERRQ(TSSetCostGradients(ts,1,lambda,NULL));

  CHKERRQ(TSAdjointSolve(ts));

  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"\n sensitivity wrt initial conditions: \n"));
  CHKERRQ(VecView(lambda[0],PETSC_VIEWER_STDOUT_WORLD));
  CHKERRQ(VecDestroy(&lambda[0]));

  CHKERRQ(SNESDestroy(&snes_alg));
  CHKERRQ(VecDestroy(&F_alg));
  CHKERRQ(MatDestroy(&J));
  CHKERRQ(MatDestroy(&user.Ybus));
  CHKERRQ(VecDestroy(&X));
  CHKERRQ(VecDestroy(&user.V0));
  CHKERRQ(DMDestroy(&user.dmgen));
  CHKERRQ(DMDestroy(&user.dmnet));
  CHKERRQ(DMDestroy(&user.dmpgrid));
  CHKERRQ(ISDestroy(&user.is_diff));
  CHKERRQ(ISDestroy(&user.is_alg));
  CHKERRQ(TSDestroy(&ts));
  ierr = PetscFinalize();
  return ierr;
}

/*TEST

   build:
      requires: double !complex !defined(PETSC_USE_64BIT_INDICES)

   test:
      args: -viewer_binary_skip_info
      localrunfiles: petscoptions X.bin Ybus.bin

   test:
      suffix: 2
      args: -viewer_binary_skip_info -ts_type beuler
      localrunfiles: petscoptions X.bin Ybus.bin

TEST*/
