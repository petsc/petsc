static char help[] = "Application of adjoint sensitivity analysis for power grid stability analysis of WECC 9 bus system.\n\
This example is based on the 9-bus (node) example given in the book Power\n\
Systems Dynamics and Stability (Chapter 7) by P. Sauer and M. A. Pai.\n\
The power grid in this example consists of 9 buses (nodes), 3 generators,\n\
3 loads, and 9 transmission lines. The network equations are written\n\
in current balance form using rectangular coordiantes.\n\n";

/*
  This code demonstrates how to solve a DAE-constrained optimization problem with TAO, TSAdjoint and TS.
  The objectivie is to find optimal parameter PG for each generator to minizie the frequency violations due to faults.
  The problem features discontinuities and a cost function in integral form.
  The gradient is computed with the discrete adjoint of an implicit theta method, see ex9busadj.c for details.
*/

#include <petsctao.h>
#include <petscts.h>
#include <petscdm.h>
#include <petscdmda.h>
#include <petscdmcomposite.h>
#include <petsctime.h>

PetscErrorCode FormFunctionGradient(Tao,Vec,PetscReal*,Vec,void*);

#define freq 60
#define w_s (2*PETSC_PI*freq)

/* Sizes and indices */
const PetscInt nbus    = 9; /* Number of network buses */
const PetscInt ngen    = 3; /* Number of generators */
const PetscInt nload   = 3; /* Number of loads */
const PetscInt gbus[3] = {0,1,2}; /* Buses at which generators are incident */
const PetscInt lbus[3] = {4,5,7}; /* Buses at which loads are incident */

/* Generator real and reactive powers (found via loadflow) */
PetscScalar PG[3] = { 0.69,1.59,0.69};
/* PetscScalar PG[3] = {0.716786142395021,1.630000000000000,0.850000000000000};*/

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
  Mat         Sol; /* Matrix to save solution at each time step */
  PetscInt    stepnum;
  PetscBool   alg_flg;
  PetscReal   t;
  IS          is_diff; /* indices for differential equations */
  IS          is_alg; /* indices for algebraic equations */
  PetscReal   freq_u,freq_l; /* upper and lower frequency limit */
  PetscInt    pow; /* power coefficient used in the cost function */
  PetscBool   jacp_flg;
  Mat         J,Jacp;
  Mat         DRDU,DRDP;
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

/* Saves the solution at each time to a matrix */
PetscErrorCode SaveSolution(TS ts)
{
  PetscErrorCode    ierr;
  Userctx           *user;
  Vec               X;
  PetscScalar       *mat;
  const PetscScalar *x;
  PetscInt          idx;
  PetscReal         t;

  PetscFunctionBegin;
  ierr     = TSGetApplicationContext(ts,&user);CHKERRQ(ierr);
  ierr     = TSGetTime(ts,&t);CHKERRQ(ierr);
  ierr     = TSGetSolution(ts,&X);CHKERRQ(ierr);
  idx      = user->stepnum*(user->neqs_pgrid+1);
  ierr     = MatDenseGetArray(user->Sol,&mat);CHKERRQ(ierr);
  ierr     = VecGetArrayRead(X,&x);CHKERRQ(ierr);
  mat[idx] = t;
  ierr     = PetscArraycpy(mat+idx+1,x,user->neqs_pgrid);CHKERRQ(ierr);
  ierr     = MatDenseRestoreArray(user->Sol,&mat);CHKERRQ(ierr);
  ierr     = VecRestoreArrayRead(X,&x);CHKERRQ(ierr);
  user->stepnum++;
  PetscFunctionReturn(0);
}

PetscErrorCode SetInitialGuess(Vec X,Userctx *user)
{
  PetscErrorCode ierr;
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

  ierr = DMCompositeGetLocalVectors(user->dmpgrid,&Xgen,&Xnet);CHKERRQ(ierr);

  /* Network subsystem initialization */
  ierr = VecCopy(user->V0,Xnet);CHKERRQ(ierr);

  /* Generator subsystem initialization */
  ierr = VecGetArray(Xgen,&xgen);CHKERRQ(ierr);
  ierr = VecGetArray(Xnet,&xnet);CHKERRQ(ierr);

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

  ierr = VecRestoreArray(Xgen,&xgen);CHKERRQ(ierr);
  ierr = VecRestoreArray(Xnet,&xnet);CHKERRQ(ierr);

  /* ierr = VecView(Xgen,0);CHKERRQ(ierr); */
  ierr = DMCompositeGather(user->dmpgrid,INSERT_VALUES,X,Xgen,Xnet);CHKERRQ(ierr);
  ierr = DMCompositeRestoreLocalVectors(user->dmpgrid,&Xgen,&Xnet);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode InitialGuess(Vec X,Userctx *user, const PetscScalar PGv[])
{
  PetscErrorCode ierr;
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

  ierr = DMCompositeGetLocalVectors(user->dmpgrid,&Xgen,&Xnet);CHKERRQ(ierr);

  /* Network subsystem initialization */
  ierr = VecCopy(user->V0,Xnet);CHKERRQ(ierr);

  /* Generator subsystem initialization */
  ierr = VecGetArray(Xgen,&xgen);CHKERRQ(ierr);
  ierr = VecGetArray(Xnet,&xnet);CHKERRQ(ierr);

  for (i=0; i < ngen; i++) {
    Vr  = xnet[2*gbus[i]]; /* Real part of generator terminal voltage */
    Vi  = xnet[2*gbus[i]+1]; /* Imaginary part of the generator terminal voltage */
    Vm  = PetscSqrtScalar(Vr*Vr + Vi*Vi); Vm2 = Vm*Vm;
    IGr = (Vr*PGv[i] + Vi*QG[i])/Vm2;
    IGi = (Vi*PGv[i] - Vr*QG[i])/Vm2;

    delta = PetscAtan2Real(Vi+Xq[i]*IGr,Vr-Xq[i]*IGi); /* Machine angle */

    theta = PETSC_PI/2.0 - delta;

    Id = IGr*PetscCosScalar(theta) - IGi*PetscSinScalar(theta); /* d-axis stator current */
    Iq = IGr*PetscSinScalar(theta) + IGi*PetscCosScalar(theta); /* q-axis stator current */

    Vd = Vr*PetscCosScalar(theta) - Vi*PetscSinScalar(theta);
    Vq = Vr*PetscSinScalar(theta) + Vi*PetscCosScalar(theta);

    Edp = Vd + Rs[i]*Id - Xqp[i]*Iq; /* d-axis transient EMF */
    Eqp = Vq + Rs[i]*Iq + Xdp[i]*Id; /* q-axis transient EMF */

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

    idx = idx + 3;
  }

  ierr = VecRestoreArray(Xgen,&xgen);CHKERRQ(ierr);
  ierr = VecRestoreArray(Xnet,&xnet);CHKERRQ(ierr);

  /* ierr = VecView(Xgen,0);CHKERRQ(ierr); */
  ierr = DMCompositeGather(user->dmpgrid,INSERT_VALUES,X,Xgen,Xnet);CHKERRQ(ierr);
  ierr = DMCompositeRestoreLocalVectors(user->dmpgrid,&Xgen,&Xnet);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode DICDPFiniteDifference(Vec X,Vec *DICDP, Userctx *user)
{
  Vec            Y;
  PetscScalar    PGv[3],eps;
  PetscErrorCode ierr;
  PetscInt       i,j;

  eps = 1.e-7;
  ierr = VecDuplicate(X,&Y);CHKERRQ(ierr);

  for (i=0;i<ngen;i++) {
    for (j=0;j<3;j++) PGv[j] = PG[j];
    PGv[i] = PG[i]+eps;
    ierr = InitialGuess(Y,user,PGv);CHKERRQ(ierr);
    ierr = InitialGuess(X,user,PG);CHKERRQ(ierr);

    ierr = VecAXPY(Y,-1.0,X);CHKERRQ(ierr);
    ierr = VecScale(Y,1./eps);CHKERRQ(ierr);
    ierr = VecCopy(Y,DICDP[i]);CHKERRQ(ierr);
  }
  ierr = VecDestroy(&Y);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


/* Computes F = [-f(x,y);g(x,y)] */
PetscErrorCode ResidualFunction(SNES snes,Vec X, Vec F, Userctx *user)
{
  PetscErrorCode ierr;
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
  ierr = VecZeroEntries(F);CHKERRQ(ierr);
  ierr = DMCompositeGetLocalVectors(user->dmpgrid,&Xgen,&Xnet);CHKERRQ(ierr);
  ierr = DMCompositeGetLocalVectors(user->dmpgrid,&Fgen,&Fnet);CHKERRQ(ierr);
  ierr = DMCompositeScatter(user->dmpgrid,X,Xgen,Xnet);CHKERRQ(ierr);
  ierr = DMCompositeScatter(user->dmpgrid,F,Fgen,Fnet);CHKERRQ(ierr);

  /* Network current balance residual IG + Y*V + IL = 0. Only YV is added here.
     The generator current injection, IG, and load current injection, ID are added later
  */
  /* Note that the values in Ybus are stored assuming the imaginary current balance
     equation is ordered first followed by real current balance equation for each bus.
     Thus imaginary current contribution goes in location 2*i, and
     real current contribution in 2*i+1
  */
  ierr = MatMult(user->Ybus,Xnet,Fnet);CHKERRQ(ierr);

  ierr = VecGetArray(Xgen,&xgen);CHKERRQ(ierr);
  ierr = VecGetArray(Xnet,&xnet);CHKERRQ(ierr);
  ierr = VecGetArray(Fgen,&fgen);CHKERRQ(ierr);
  ierr = VecGetArray(Fnet,&fnet);CHKERRQ(ierr);

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

    ierr = ri2dq(Vr,Vi,delta,&Vd,&Vq);CHKERRQ(ierr);
    /* Algebraic equations for stator currents */
    det = Rs[i]*Rs[i] + Xdp[i]*Xqp[i];

    Zdq_inv[0] = Rs[i]/det;
    Zdq_inv[1] = Xqp[i]/det;
    Zdq_inv[2] = -Xdp[i]/det;
    Zdq_inv[3] = Rs[i]/det;

    fgen[idx+4] = Zdq_inv[0]*(-Edp + Vd) + Zdq_inv[1]*(-Eqp + Vq) + Id;
    fgen[idx+5] = Zdq_inv[2]*(-Edp + Vd) + Zdq_inv[3]*(-Eqp + Vq) + Iq;

    /* Add generator current injection to network */
    ierr = dq2ri(Id,Iq,delta,&IGr,&IGi);CHKERRQ(ierr);

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

  ierr = VecGetArray(user->V0,&v0);CHKERRQ(ierr);
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
  ierr = VecRestoreArray(user->V0,&v0);CHKERRQ(ierr);

  ierr = VecRestoreArray(Xgen,&xgen);CHKERRQ(ierr);
  ierr = VecRestoreArray(Xnet,&xnet);CHKERRQ(ierr);
  ierr = VecRestoreArray(Fgen,&fgen);CHKERRQ(ierr);
  ierr = VecRestoreArray(Fnet,&fnet);CHKERRQ(ierr);

  ierr = DMCompositeGather(user->dmpgrid,INSERT_VALUES,F,Fgen,Fnet);CHKERRQ(ierr);
  ierr = DMCompositeRestoreLocalVectors(user->dmpgrid,&Xgen,&Xnet);CHKERRQ(ierr);
  ierr = DMCompositeRestoreLocalVectors(user->dmpgrid,&Fgen,&Fnet);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* \dot{x} - f(x,y)
     g(x,y) = 0
 */
PetscErrorCode IFunction(TS ts,PetscReal t, Vec X, Vec Xdot, Vec F, Userctx *user)
{
  PetscErrorCode    ierr;
  SNES              snes;
  PetscScalar       *f;
  const PetscScalar *xdot;
  PetscInt          i;

  PetscFunctionBegin;
  user->t = t;

  ierr = TSGetSNES(ts,&snes);CHKERRQ(ierr);
  ierr = ResidualFunction(snes,X,F,user);CHKERRQ(ierr);
  ierr = VecGetArray(F,&f);CHKERRQ(ierr);
  ierr = VecGetArrayRead(Xdot,&xdot);CHKERRQ(ierr);
  for (i=0;i < ngen;i++) {
    f[9*i]   += xdot[9*i];
    f[9*i+1] += xdot[9*i+1];
    f[9*i+2] += xdot[9*i+2];
    f[9*i+3] += xdot[9*i+3];
    f[9*i+6] += xdot[9*i+6];
    f[9*i+7] += xdot[9*i+7];
    f[9*i+8] += xdot[9*i+8];
  }
  ierr = VecRestoreArray(F,&f);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(Xdot,&xdot);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* This function is used for solving the algebraic system only during fault on and
   off times. It computes the entire F and then zeros out the part corresponding to
   differential equations
 F = [0;g(y)];
*/
PetscErrorCode AlgFunction(SNES snes, Vec X, Vec F, void *ctx)
{
  PetscErrorCode ierr;
  Userctx        *user=(Userctx*)ctx;
  PetscScalar    *f;
  PetscInt       i;

  PetscFunctionBegin;
  ierr = ResidualFunction(snes,X,F,user);CHKERRQ(ierr);
  ierr = VecGetArray(F,&f);CHKERRQ(ierr);
  for (i=0; i < ngen; i++) {
    f[9*i]   = 0;
    f[9*i+1] = 0;
    f[9*i+2] = 0;
    f[9*i+3] = 0;
    f[9*i+6] = 0;
    f[9*i+7] = 0;
    f[9*i+8] = 0;
  }
  ierr = VecRestoreArray(F,&f);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode PreallocateJacobian(Mat J, Userctx *user)
{
  PetscErrorCode ierr;
  PetscInt       *d_nnz;
  PetscInt       i,idx=0,start=0;
  PetscInt       ncols;

  PetscFunctionBegin;
  ierr = PetscMalloc1(user->neqs_pgrid,&d_nnz);CHKERRQ(ierr);
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
    ierr = MatGetRow(user->Ybus,2*i,&ncols,NULL,NULL);CHKERRQ(ierr);
    d_nnz[start+2*i]   += ncols;
    d_nnz[start+2*i+1] += ncols;
    ierr = MatRestoreRow(user->Ybus,2*i,&ncols,NULL,NULL);CHKERRQ(ierr);
  }

  ierr = MatSeqAIJSetPreallocation(J,0,d_nnz);CHKERRQ(ierr);
  ierr = PetscFree(d_nnz);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*
   J = [-df_dx, -df_dy
        dg_dx, dg_dy]
*/
PetscErrorCode ResidualJacobian(SNES snes,Vec X,Mat J,Mat B,void *ctx)
{
  PetscErrorCode    ierr;
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
  PetscInt          ncols;
  const PetscInt    *cols;
  const PetscScalar *yvals;
  PetscInt          k;
  PetscScalar       Zdq_inv[4],det;
  PetscScalar       dVd_dVr,dVd_dVi,dVq_dVr,dVq_dVi,dVd_ddelta,dVq_ddelta;
  PetscScalar       dIGr_ddelta,dIGi_ddelta,dIGr_dId,dIGr_dIq,dIGi_dId,dIGi_dIq;
  PetscScalar       dSE_dEfd;
  PetscScalar       dVm_dVd,dVm_dVq,dVm_dVr,dVm_dVi;
  PetscScalar       PD,QD,Vm0,*v0,Vm4;
  PetscScalar       dPD_dVr,dPD_dVi,dQD_dVr,dQD_dVi;
  PetscScalar       dIDr_dVr,dIDr_dVi,dIDi_dVr,dIDi_dVi;

  PetscFunctionBegin;
  ierr  = MatZeroEntries(B);CHKERRQ(ierr);
  ierr  = DMCompositeGetLocalVectors(user->dmpgrid,&Xgen,&Xnet);CHKERRQ(ierr);
  ierr  = DMCompositeScatter(user->dmpgrid,X,Xgen,Xnet);CHKERRQ(ierr);

  ierr = VecGetArray(Xgen,&xgen);CHKERRQ(ierr);
  ierr = VecGetArray(Xnet,&xnet);CHKERRQ(ierr);

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

    ierr = MatSetValues(J,1,row,3,col,val,INSERT_VALUES);CHKERRQ(ierr);

    /*    fgen[idx+1] = (Edp - (Xq[i] - Xqp[i])*Iq)/Tq0p[i]; */
    row[0] = idx + 1;
    col[0] = idx + 1;       col[1] = idx+5;
    val[0] = 1/Tq0p[i]; val[1] = -(Xq[i] - Xqp[i])/Tq0p[i];
    ierr   = MatSetValues(J,1,row,2,col,val,INSERT_VALUES);CHKERRQ(ierr);

    /*    fgen[idx+2] = - w + w_s; */
    row[0] = idx + 2;
    col[0] = idx + 2; col[1] = idx + 3;
    val[0] = 0;       val[1] = -1;
    ierr   = MatSetValues(J,1,row,2,col,val,INSERT_VALUES);CHKERRQ(ierr);

    /*    fgen[idx+3] = (-TM[i] + Edp*Id + Eqp*Iq + (Xqp[i] - Xdp[i])*Id*Iq + D[i]*(w - w_s))/M[i]; */
    row[0] = idx + 3;
    col[0] = idx; col[1] = idx + 1; col[2] = idx + 3;       col[3] = idx + 4;                  col[4] = idx + 5;
    val[0] = Iq/M[i];  val[1] = Id/M[i];      val[2] = D[i]/M[i]; val[3] = (Edp + (Xqp[i]-Xdp[i])*Iq)/M[i]; val[4] = (Eqp + (Xqp[i] - Xdp[i])*Id)/M[i];
    ierr   = MatSetValues(J,1,row,5,col,val,INSERT_VALUES);CHKERRQ(ierr);

    Vr   = xnet[2*gbus[i]]; /* Real part of generator terminal voltage */
    Vi   = xnet[2*gbus[i]+1]; /* Imaginary part of the generator terminal voltage */
    ierr = ri2dq(Vr,Vi,delta,&Vd,&Vq);CHKERRQ(ierr);

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
    ierr   = MatSetValues(J,1,row,6,col,val,INSERT_VALUES);CHKERRQ(ierr);

    /*  fgen[idx+5] = Zdq_inv[2]*(-Edp + Vd) + Zdq_inv[3]*(-Eqp + Vq) + Iq; */
    row[0] = idx+5;
    col[0] = idx;         col[1] = idx+1;        col[2] = idx + 2;
    val[0] = -Zdq_inv[3]; val[1] = -Zdq_inv[2];  val[2] = Zdq_inv[2]*dVd_ddelta + Zdq_inv[3]*dVq_ddelta;
    col[3] = idx + 5; col[4] = net_start+2*gbus[i];                     col[5] = net_start + 2*gbus[i]+1;
    val[3] = 1;       val[4] = Zdq_inv[2]*dVd_dVr + Zdq_inv[3]*dVq_dVr; val[5] = Zdq_inv[2]*dVd_dVi + Zdq_inv[3]*dVq_dVi;
    ierr   = MatSetValues(J,1,row,6,col,val,INSERT_VALUES);CHKERRQ(ierr);

    dIGr_ddelta = Id*PetscCosScalar(delta) - Iq*PetscSinScalar(delta);
    dIGi_ddelta = Id*PetscSinScalar(delta) + Iq*PetscCosScalar(delta);
    dIGr_dId    = PetscSinScalar(delta);  dIGr_dIq = PetscCosScalar(delta);
    dIGi_dId    = -PetscCosScalar(delta); dIGi_dIq = PetscSinScalar(delta);

    /* fnet[2*gbus[i]]   -= IGi; */
    row[0] = net_start + 2*gbus[i];
    col[0] = idx+2;        col[1] = idx + 4;   col[2] = idx + 5;
    val[0] = -dIGi_ddelta; val[1] = -dIGi_dId; val[2] = -dIGi_dIq;
    ierr = MatSetValues(J,1,row,3,col,val,INSERT_VALUES);CHKERRQ(ierr);

    /* fnet[2*gbus[i]+1]   -= IGr; */
    row[0] = net_start + 2*gbus[i]+1;
    col[0] = idx+2;        col[1] = idx + 4;   col[2] = idx + 5;
    val[0] = -dIGr_ddelta; val[1] = -dIGr_dId; val[2] = -dIGr_dIq;
    ierr   = MatSetValues(J,1,row,3,col,val,INSERT_VALUES);CHKERRQ(ierr);

    Vm = PetscSqrtScalar(Vd*Vd + Vq*Vq);

    /*    fgen[idx+6] = (KE[i]*Efd + SE - VR)/TE[i]; */
    /*    SE  = k1[i]*PetscExpScalar(k2[i]*Efd); */
    dSE_dEfd = k1[i]*k2[i]*PetscExpScalar(k2[i]*Efd);

    row[0] = idx + 6;
    col[0] = idx + 6;                     col[1] = idx + 8;
    val[0] = (KE[i] + dSE_dEfd)/TE[i];  val[1] = -1/TE[i];
    ierr   = MatSetValues(J,1,row,2,col,val,INSERT_VALUES);CHKERRQ(ierr);

    /* Exciter differential equations */

    /*    fgen[idx+7] = (RF - KF[i]*Efd/TF[i])/TF[i]; */
    row[0] = idx + 7;
    col[0] = idx + 6;       col[1] = idx + 7;
    val[0] = (-KF[i]/TF[i])/TF[i];  val[1] = 1/TF[i];
    ierr   = MatSetValues(J,1,row,2,col,val,INSERT_VALUES);CHKERRQ(ierr);

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
    ierr       = MatSetValues(J,1,row,5,col,val,INSERT_VALUES);CHKERRQ(ierr);
    idx        = idx + 9;
  }


  for (i=0; i<nbus; i++) {
    ierr   = MatGetRow(user->Ybus,2*i,&ncols,&cols,&yvals);CHKERRQ(ierr);
    row[0] = net_start + 2*i;
    for (k=0; k<ncols; k++) {
      col[k] = net_start + cols[k];
      val[k] = yvals[k];
    }
    ierr = MatSetValues(J,1,row,ncols,col,val,INSERT_VALUES);CHKERRQ(ierr);
    ierr = MatRestoreRow(user->Ybus,2*i,&ncols,&cols,&yvals);CHKERRQ(ierr);

    ierr   = MatGetRow(user->Ybus,2*i+1,&ncols,&cols,&yvals);CHKERRQ(ierr);
    row[0] = net_start + 2*i+1;
    for (k=0; k<ncols; k++) {
      col[k] = net_start + cols[k];
      val[k] = yvals[k];
    }
    ierr = MatSetValues(J,1,row,ncols,col,val,INSERT_VALUES);CHKERRQ(ierr);
    ierr = MatRestoreRow(user->Ybus,2*i+1,&ncols,&cols,&yvals);CHKERRQ(ierr);
  }

  ierr = MatAssemblyBegin(J,MAT_FLUSH_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(J,MAT_FLUSH_ASSEMBLY);CHKERRQ(ierr);


  ierr = VecGetArray(user->V0,&v0);CHKERRQ(ierr);
  for (i=0; i < nload; i++) {
    Vr      = xnet[2*lbus[i]]; /* Real part of load bus voltage */
    Vi      = xnet[2*lbus[i]+1]; /* Imaginary part of the load bus voltage */
    Vm      = PetscSqrtScalar(Vr*Vr + Vi*Vi); Vm2= Vm*Vm; Vm4 = Vm2*Vm2;
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
    ierr   = MatSetValues(J,1,row,2,col,val,ADD_VALUES);CHKERRQ(ierr);
    /*    fnet[2*lbus[i]+1] += IDr; */
    row[0] = net_start + 2*lbus[i]+1;
    col[0] = net_start + 2*lbus[i];  col[1] = net_start + 2*lbus[i]+1;
    val[0] = dIDr_dVr;               val[1] = dIDr_dVi;
    ierr   = MatSetValues(J,1,row,2,col,val,ADD_VALUES);CHKERRQ(ierr);
  }
  ierr = VecRestoreArray(user->V0,&v0);CHKERRQ(ierr);

  ierr = VecRestoreArray(Xgen,&xgen);CHKERRQ(ierr);
  ierr = VecRestoreArray(Xnet,&xnet);CHKERRQ(ierr);

  ierr = DMCompositeRestoreLocalVectors(user->dmpgrid,&Xgen,&Xnet);CHKERRQ(ierr);

  ierr = MatAssemblyBegin(J,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(J,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*
   J = [I, 0
        dg_dx, dg_dy]
*/
PetscErrorCode AlgJacobian(SNES snes,Vec X,Mat A,Mat B,void *ctx)
{
  PetscErrorCode ierr;
  Userctx        *user=(Userctx*)ctx;

  PetscFunctionBegin;
  ierr = ResidualJacobian(snes,X,A,B,ctx);CHKERRQ(ierr);
  ierr = MatSetOption(A,MAT_KEEP_NONZERO_PATTERN,PETSC_TRUE);CHKERRQ(ierr);
  ierr = MatZeroRowsIS(A,user->is_diff,1.0,NULL,NULL);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*
   J = [a*I-df_dx, -df_dy
        dg_dx, dg_dy]
*/

PetscErrorCode IJacobian(TS ts,PetscReal t,Vec X,Vec Xdot,PetscReal a,Mat A,Mat B,Userctx *user)
{
  PetscErrorCode ierr;
  SNES           snes;
  PetscScalar    atmp = (PetscScalar) a;
  PetscInt       i,row;

  PetscFunctionBegin;
  user->t = t;

  ierr = TSGetSNES(ts,&snes);CHKERRQ(ierr);
  ierr = ResidualJacobian(snes,X,A,B,user);CHKERRQ(ierr);
  for (i=0;i < ngen;i++) {
    row = 9*i;
    ierr = MatSetValues(A,1,&row,1,&row,&atmp,ADD_VALUES);CHKERRQ(ierr);
    row  = 9*i+1;
    ierr = MatSetValues(A,1,&row,1,&row,&atmp,ADD_VALUES);CHKERRQ(ierr);
    row  = 9*i+2;
    ierr = MatSetValues(A,1,&row,1,&row,&atmp,ADD_VALUES);CHKERRQ(ierr);
    row  = 9*i+3;
    ierr = MatSetValues(A,1,&row,1,&row,&atmp,ADD_VALUES);CHKERRQ(ierr);
    row  = 9*i+6;
    ierr = MatSetValues(A,1,&row,1,&row,&atmp,ADD_VALUES);CHKERRQ(ierr);
    row  = 9*i+7;
    ierr = MatSetValues(A,1,&row,1,&row,&atmp,ADD_VALUES);CHKERRQ(ierr);
    row  = 9*i+8;
    ierr = MatSetValues(A,1,&row,1,&row,&atmp,ADD_VALUES);CHKERRQ(ierr);
  }
  ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* Matrix JacobianP is constant so that it only needs to be evaluated once */
static PetscErrorCode RHSJacobianP(TS ts,PetscReal t,Vec X,Mat A, void *ctx0)
{
  PetscErrorCode ierr;
  PetscScalar    a;
  PetscInt       row,col;
  Userctx        *ctx=(Userctx*)ctx0;

  PetscFunctionBeginUser;

  if (ctx->jacp_flg) {
    ierr = MatZeroEntries(A);CHKERRQ(ierr);

    for (col=0;col<3;col++) {
      a    = 1.0/M[col];
      row  = 9*col+3;
      ierr = MatSetValues(A,1,&row,1,&col,&a,INSERT_VALUES);CHKERRQ(ierr);
    }

    ctx->jacp_flg = PETSC_FALSE;

    ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode CostIntegrand(TS ts,PetscReal t,Vec U,Vec R,Userctx *user)
{
  PetscErrorCode    ierr;
  const PetscScalar *u;
  PetscInt          idx;
  Vec               Xgen,Xnet;
  PetscScalar       *r,*xgen;
  PetscInt          i;

  PetscFunctionBegin;
  ierr = DMCompositeGetLocalVectors(user->dmpgrid,&Xgen,&Xnet);CHKERRQ(ierr);
  ierr = DMCompositeScatter(user->dmpgrid,U,Xgen,Xnet);CHKERRQ(ierr);

  ierr = VecGetArray(Xgen,&xgen);CHKERRQ(ierr);

  ierr = VecGetArrayRead(U,&u);CHKERRQ(ierr);
  ierr = VecGetArray(R,&r);CHKERRQ(ierr);
  r[0] = 0.;
  idx = 0;
  for (i=0;i<ngen;i++) {
    r[0] += PetscPowScalarInt(PetscMax(0.,PetscMax(xgen[idx+3]/(2.*PETSC_PI)-user->freq_u,user->freq_l-xgen[idx+3]/(2.*PETSC_PI))),user->pow);
    idx  += 9;
  }
  ierr = VecRestoreArrayRead(U,&u);CHKERRQ(ierr);
  ierr = VecRestoreArray(R,&r);CHKERRQ(ierr);
  ierr = DMCompositeRestoreLocalVectors(user->dmpgrid,&Xgen,&Xnet);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode DRDUJacobianTranspose(TS ts,PetscReal t,Vec U,Mat DRDU,Mat B,Userctx *user)
{
  PetscErrorCode ierr;
  Vec            Xgen,Xnet,Dgen,Dnet;
  PetscScalar    *xgen,*dgen;
  PetscInt       i;
  PetscInt       idx;
  Vec            drdu_col;
  PetscScalar    *xarr;

  PetscFunctionBegin;
  ierr = VecDuplicate(U,&drdu_col);CHKERRQ(ierr);
  ierr = MatDenseGetColumn(DRDU,0,&xarr);CHKERRQ(ierr);
  ierr = VecPlaceArray(drdu_col,xarr);CHKERRQ(ierr);
  ierr = DMCompositeGetLocalVectors(user->dmpgrid,&Xgen,&Xnet);CHKERRQ(ierr);
  ierr = DMCompositeGetLocalVectors(user->dmpgrid,&Dgen,&Dnet);CHKERRQ(ierr);
  ierr = DMCompositeScatter(user->dmpgrid,U,Xgen,Xnet);CHKERRQ(ierr);
  ierr = DMCompositeScatter(user->dmpgrid,drdu_col,Dgen,Dnet);CHKERRQ(ierr);

  ierr = VecGetArray(Xgen,&xgen);CHKERRQ(ierr);
  ierr = VecGetArray(Dgen,&dgen);CHKERRQ(ierr);

  idx = 0;
  for (i=0;i<ngen;i++) {
    dgen[idx+3] = 0.;
    if (xgen[idx+3]/(2.*PETSC_PI) > user->freq_u) dgen[idx+3] = user->pow*PetscPowScalarInt(xgen[idx+3]/(2.*PETSC_PI)-user->freq_u,user->pow-1)/(2.*PETSC_PI);
    if (xgen[idx+3]/(2.*PETSC_PI) < user->freq_l) dgen[idx+3] = user->pow*PetscPowScalarInt(user->freq_l-xgen[idx+3]/(2.*PETSC_PI),user->pow-1)/(-2.*PETSC_PI);
    idx += 9;
  }

  ierr = VecRestoreArray(Dgen,&dgen);CHKERRQ(ierr);
  ierr = VecRestoreArray(Xgen,&xgen);CHKERRQ(ierr);
  ierr = DMCompositeGather(user->dmpgrid,INSERT_VALUES,drdu_col,Dgen,Dnet);CHKERRQ(ierr);
  ierr = DMCompositeRestoreLocalVectors(user->dmpgrid,&Dgen,&Dnet);CHKERRQ(ierr);
  ierr = DMCompositeRestoreLocalVectors(user->dmpgrid,&Xgen,&Xnet);CHKERRQ(ierr);
  ierr = VecResetArray(drdu_col);CHKERRQ(ierr);
  ierr = MatDenseRestoreColumn(DRDU,&xarr);CHKERRQ(ierr);
  ierr = VecDestroy(&drdu_col);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode DRDPJacobianTranspose(TS ts,PetscReal t,Vec U,Mat drdp,Userctx *user)
{
  PetscFunctionBegin;
  PetscFunctionReturn(0);
}

PetscErrorCode ComputeSensiP(Vec lambda,Vec mu,Vec *DICDP,Userctx *user)
{
  PetscErrorCode ierr;
  PetscScalar    *x,*y,sensip;
  PetscInt       i;

  PetscFunctionBegin;
  ierr = VecGetArray(lambda,&x);CHKERRQ(ierr);
  ierr = VecGetArray(mu,&y);CHKERRQ(ierr);

  for (i=0;i<3;i++) {
    ierr   = VecDot(lambda,DICDP[i],&sensip);CHKERRQ(ierr);
    sensip = sensip+y[i];
    /* ierr   = PetscPrintf(PETSC_COMM_WORLD,"\n sensitivity wrt %D th parameter: %g \n",i,(double)sensip);CHKERRQ(ierr); */
     y[i] = sensip;
  }
  ierr = VecRestoreArray(mu,&y);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

int main(int argc,char **argv)
{
  Userctx            user;
  Vec                p;
  PetscScalar        *x_ptr;
  PetscErrorCode     ierr;
  PetscMPIInt        size;
  PetscInt           i;
  PetscViewer        Xview,Ybusview;
  PetscInt           *idx2;
  Tao                tao;
  KSP                ksp;
  PC                 pc;
  Vec                lowerb,upperb;

  ierr = PetscInitialize(&argc,&argv,"petscoptions",help);if (ierr) return ierr;
  ierr = MPI_Comm_size(PETSC_COMM_WORLD,&size);CHKERRQ(ierr);
  if (size > 1) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_SUP,"Only for sequential runs");

  user.jacp_flg   = PETSC_TRUE;
  user.neqs_gen   = 9*ngen; /* # eqs. for generator subsystem */
  user.neqs_net   = 2*nbus; /* # eqs. for network subsystem   */
  user.neqs_pgrid = user.neqs_gen + user.neqs_net;

  /* Create indices for differential and algebraic equations */
  ierr = PetscMalloc1(7*ngen,&idx2);CHKERRQ(ierr);
  for (i=0; i<ngen; i++) {
    idx2[7*i]   = 9*i;   idx2[7*i+1] = 9*i+1; idx2[7*i+2] = 9*i+2; idx2[7*i+3] = 9*i+3;
    idx2[7*i+4] = 9*i+6; idx2[7*i+5] = 9*i+7; idx2[7*i+6] = 9*i+8;
  }
  ierr = ISCreateGeneral(PETSC_COMM_WORLD,7*ngen,idx2,PETSC_COPY_VALUES,&user.is_diff);CHKERRQ(ierr);
  ierr = ISComplement(user.is_diff,0,user.neqs_pgrid,&user.is_alg);CHKERRQ(ierr);
  ierr = PetscFree(idx2);CHKERRQ(ierr);

  /* Set run time options */
  ierr = PetscOptionsBegin(PETSC_COMM_WORLD,NULL,"Transient stability fault options","");CHKERRQ(ierr);
  {
    user.tfaulton  = 1.0;
    user.tfaultoff = 1.2;
    user.Rfault    = 0.0001;
    user.faultbus  = 8;
    ierr           = PetscOptionsReal("-tfaulton","","",user.tfaulton,&user.tfaulton,NULL);CHKERRQ(ierr);
    ierr           = PetscOptionsReal("-tfaultoff","","",user.tfaultoff,&user.tfaultoff,NULL);CHKERRQ(ierr);
    ierr           = PetscOptionsInt("-faultbus","","",user.faultbus,&user.faultbus,NULL);CHKERRQ(ierr);
    user.t0        = 0.0;
    user.tmax      = 1.3;
    ierr           = PetscOptionsReal("-t0","","",user.t0,&user.t0,NULL);CHKERRQ(ierr);
    ierr           = PetscOptionsReal("-tmax","","",user.tmax,&user.tmax,NULL);CHKERRQ(ierr);
    user.freq_u    = 61.0;
    user.freq_l    = 59.0;
    user.pow       = 2;
    ierr           = PetscOptionsReal("-frequ","","",user.freq_u,&user.freq_u,NULL);CHKERRQ(ierr);
    ierr           = PetscOptionsReal("-freql","","",user.freq_l,&user.freq_l,NULL);CHKERRQ(ierr);
    ierr           = PetscOptionsInt("-pow","","",user.pow,&user.pow,NULL);CHKERRQ(ierr);

  }
  ierr = PetscOptionsEnd();CHKERRQ(ierr);

  /* Create DMs for generator and network subsystems */
  ierr = DMDACreate1d(PETSC_COMM_WORLD,DM_BOUNDARY_NONE,user.neqs_gen,1,1,NULL,&user.dmgen);CHKERRQ(ierr);
  ierr = DMSetOptionsPrefix(user.dmgen,"dmgen_");CHKERRQ(ierr);
  ierr = DMDACreate1d(PETSC_COMM_WORLD,DM_BOUNDARY_NONE,user.neqs_net,1,1,NULL,&user.dmnet);CHKERRQ(ierr);
  ierr = DMSetOptionsPrefix(user.dmnet,"dmnet_");CHKERRQ(ierr);
  ierr = DMSetFromOptions(user.dmnet);CHKERRQ(ierr);
  ierr = DMSetUp(user.dmnet);CHKERRQ(ierr);
  /* Create a composite DM packer and add the two DMs */
  ierr = DMCompositeCreate(PETSC_COMM_WORLD,&user.dmpgrid);CHKERRQ(ierr);
  ierr = DMSetOptionsPrefix(user.dmpgrid,"pgrid_");CHKERRQ(ierr);
  ierr = DMSetFromOptions(user.dmgen);CHKERRQ(ierr);
  ierr = DMSetUp(user.dmgen);CHKERRQ(ierr);
  ierr = DMCompositeAddDM(user.dmpgrid,user.dmgen);CHKERRQ(ierr);
  ierr = DMCompositeAddDM(user.dmpgrid,user.dmnet);CHKERRQ(ierr);

  /* Read initial voltage vector and Ybus */
  ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,"X.bin",FILE_MODE_READ,&Xview);CHKERRQ(ierr);
  ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,"Ybus.bin",FILE_MODE_READ,&Ybusview);CHKERRQ(ierr);

  ierr = VecCreate(PETSC_COMM_WORLD,&user.V0);CHKERRQ(ierr);
  ierr = VecSetSizes(user.V0,PETSC_DECIDE,user.neqs_net);CHKERRQ(ierr);
  ierr = VecLoad(user.V0,Xview);CHKERRQ(ierr);

  ierr = MatCreate(PETSC_COMM_WORLD,&user.Ybus);CHKERRQ(ierr);
  ierr = MatSetSizes(user.Ybus,PETSC_DECIDE,PETSC_DECIDE,user.neqs_net,user.neqs_net);CHKERRQ(ierr);
  ierr = MatSetType(user.Ybus,MATBAIJ);CHKERRQ(ierr);
  /*  ierr = MatSetBlockSize(ctx->Ybus,2);CHKERRQ(ierr); */
  ierr = MatLoad(user.Ybus,Ybusview);CHKERRQ(ierr);

  ierr = PetscViewerDestroy(&Xview);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&Ybusview);CHKERRQ(ierr);

  /* Allocate space for Jacobians */
  ierr = MatCreate(PETSC_COMM_WORLD,&user.J);CHKERRQ(ierr);
  ierr = MatSetSizes(user.J,PETSC_DECIDE,PETSC_DECIDE,user.neqs_pgrid,user.neqs_pgrid);CHKERRQ(ierr);
  ierr = MatSetFromOptions(user.J);CHKERRQ(ierr);
  ierr = PreallocateJacobian(user.J,&user);CHKERRQ(ierr);

  ierr = MatCreate(PETSC_COMM_WORLD,&user.Jacp);CHKERRQ(ierr);
  ierr = MatSetSizes(user.Jacp,PETSC_DECIDE,PETSC_DECIDE,user.neqs_pgrid,3);CHKERRQ(ierr);
  ierr = MatSetFromOptions(user.Jacp);CHKERRQ(ierr);
  ierr = MatSetUp(user.Jacp);CHKERRQ(ierr);
  ierr = MatZeroEntries(user.Jacp);CHKERRQ(ierr); /* initialize to zeros */

  ierr = MatCreateDense(PETSC_COMM_WORLD,PETSC_DECIDE,PETSC_DECIDE,3,1,NULL,&user.DRDP);CHKERRQ(ierr);
  ierr = MatSetUp(user.DRDP);CHKERRQ(ierr);
  ierr = MatCreateDense(PETSC_COMM_WORLD,PETSC_DECIDE,PETSC_DECIDE,user.neqs_pgrid,1,NULL,&user.DRDU);CHKERRQ(ierr);
  ierr = MatSetUp(user.DRDU);CHKERRQ(ierr);

  /* Create TAO solver and set desired solution method */
  ierr = TaoCreate(PETSC_COMM_WORLD,&tao);CHKERRQ(ierr);
  ierr = TaoSetType(tao,TAOBLMVM);CHKERRQ(ierr);
  /*
     Optimization starts
  */
  /* Set initial solution guess */
  ierr = VecCreateSeq(PETSC_COMM_WORLD,3,&p);CHKERRQ(ierr);
  ierr = VecGetArray(p,&x_ptr);CHKERRQ(ierr);
  x_ptr[0] = PG[0]; x_ptr[1] = PG[1]; x_ptr[2] = PG[2];
  ierr = VecRestoreArray(p,&x_ptr);CHKERRQ(ierr);

  ierr = TaoSetInitialVector(tao,p);CHKERRQ(ierr);
  /* Set routine for function and gradient evaluation */
  ierr = TaoSetObjectiveAndGradientRoutine(tao,FormFunctionGradient,&user);CHKERRQ(ierr);

  /* Set bounds for the optimization */
  ierr = VecDuplicate(p,&lowerb);CHKERRQ(ierr);
  ierr = VecDuplicate(p,&upperb);CHKERRQ(ierr);
  ierr = VecGetArray(lowerb,&x_ptr);CHKERRQ(ierr);
  x_ptr[0] = 0.5; x_ptr[1] = 0.5; x_ptr[2] = 0.5;
  ierr = VecRestoreArray(lowerb,&x_ptr);CHKERRQ(ierr);
  ierr = VecGetArray(upperb,&x_ptr);CHKERRQ(ierr);
  x_ptr[0] = 2.0; x_ptr[1] = 2.0; x_ptr[2] = 2.0;
  ierr = VecRestoreArray(upperb,&x_ptr);CHKERRQ(ierr);
  ierr = TaoSetVariableBounds(tao,lowerb,upperb);CHKERRQ(ierr);

  /* Check for any TAO command line options */
  ierr = TaoSetFromOptions(tao);CHKERRQ(ierr);
  ierr = TaoGetKSP(tao,&ksp);CHKERRQ(ierr);
  if (ksp) {
    ierr = KSPGetPC(ksp,&pc);CHKERRQ(ierr);
    ierr = PCSetType(pc,PCNONE);CHKERRQ(ierr);
  }

  /* SOLVE THE APPLICATION */
  ierr = TaoSolve(tao); CHKERRQ(ierr);

  ierr = VecView(p,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  /* Free TAO data structures */
  ierr = TaoDestroy(&tao);CHKERRQ(ierr);

  ierr = DMDestroy(&user.dmgen);CHKERRQ(ierr);
  ierr = DMDestroy(&user.dmnet);CHKERRQ(ierr);
  ierr = DMDestroy(&user.dmpgrid);CHKERRQ(ierr);
  ierr = ISDestroy(&user.is_diff);CHKERRQ(ierr);
  ierr = ISDestroy(&user.is_alg);CHKERRQ(ierr);

  ierr = MatDestroy(&user.J);CHKERRQ(ierr);
  ierr = MatDestroy(&user.Jacp);CHKERRQ(ierr);
  ierr = MatDestroy(&user.Ybus);CHKERRQ(ierr);
  /* ierr = MatDestroy(&user.Sol);CHKERRQ(ierr); */
  ierr = VecDestroy(&user.V0);CHKERRQ(ierr);
  ierr = VecDestroy(&p);CHKERRQ(ierr);
  ierr = VecDestroy(&lowerb);CHKERRQ(ierr);
  ierr = VecDestroy(&upperb);CHKERRQ(ierr);
  ierr = MatDestroy(&user.DRDU);CHKERRQ(ierr);
  ierr = MatDestroy(&user.DRDP);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
}

/* ------------------------------------------------------------------ */
/*
   FormFunction - Evaluates the function and corresponding gradient.

   Input Parameters:
   tao - the Tao context
   X   - the input vector
   ptr - optional user-defined context, as set by TaoSetObjectiveAndGradientRoutine()

   Output Parameters:
   f   - the newly evaluated function
   G   - the newly evaluated gradient
*/
PetscErrorCode FormFunctionGradient(Tao tao,Vec P,PetscReal *f,Vec G,void *ctx0)
{
  TS             ts,quadts;
  SNES           snes_alg;
  PetscErrorCode ierr;
  Userctx        *ctx = (Userctx*)ctx0;
  Vec            X;
  PetscInt       i;
  /* sensitivity context */
  PetscScalar    *x_ptr;
  Vec            lambda[1],q;
  Vec            mu[1];
  PetscInt       steps1,steps2,steps3;
  Vec            DICDP[3];
  Vec            F_alg;
  PetscInt       row_loc,col_loc;
  PetscScalar    val;
  Vec            Xdot;

  PetscFunctionBegin;
  ierr  = VecGetArrayRead(P,(const PetscScalar**)&x_ptr);CHKERRQ(ierr);
  PG[0] = x_ptr[0];
  PG[1] = x_ptr[1];
  PG[2] = x_ptr[2];
  ierr  = VecRestoreArrayRead(P,(const PetscScalar**)&x_ptr);CHKERRQ(ierr);

  ctx->stepnum = 0;

  ierr = DMCreateGlobalVector(ctx->dmpgrid,&X);CHKERRQ(ierr);

  /* Create matrix to save solutions at each time step */
  /* ierr = MatCreateSeqDense(PETSC_COMM_SELF,ctx->neqs_pgrid+1,1002,NULL,&ctx->Sol);CHKERRQ(ierr); */
  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Create timestepping solver context
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = TSCreate(PETSC_COMM_WORLD,&ts);CHKERRQ(ierr);
  ierr = TSSetProblemType(ts,TS_NONLINEAR);CHKERRQ(ierr);
  ierr = TSSetType(ts,TSCN);CHKERRQ(ierr);
  ierr = TSSetIFunction(ts,NULL,(TSIFunction) IFunction,ctx);CHKERRQ(ierr);
  ierr = TSSetIJacobian(ts,ctx->J,ctx->J,(TSIJacobian)IJacobian,ctx);CHKERRQ(ierr);
  ierr = TSSetApplicationContext(ts,ctx);CHKERRQ(ierr);
  /*   Set RHS JacobianP */
  ierr = TSSetRHSJacobianP(ts,ctx->Jacp,RHSJacobianP,ctx);CHKERRQ(ierr);

  ierr = TSCreateQuadratureTS(ts,PETSC_FALSE,&quadts);CHKERRQ(ierr);
  ierr = TSSetRHSFunction(quadts,NULL,(TSRHSFunction)CostIntegrand,ctx);CHKERRQ(ierr);
  ierr = TSSetRHSJacobian(quadts,ctx->DRDU,ctx->DRDU,(TSRHSJacobian)DRDUJacobianTranspose,ctx);CHKERRQ(ierr);
  ierr = TSSetRHSJacobianP(quadts,ctx->DRDP,(TSRHSJacobianP)DRDPJacobianTranspose,ctx);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Set initial conditions
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = SetInitialGuess(X,ctx);CHKERRQ(ierr);

  /* Approximate DICDP with finite difference, we want to zero out network variables */
  for (i=0;i<3;i++) {
    ierr = VecDuplicate(X,&DICDP[i]);CHKERRQ(ierr);
  }
  ierr = DICDPFiniteDifference(X,DICDP,ctx);CHKERRQ(ierr);

  ierr = VecDuplicate(X,&F_alg);CHKERRQ(ierr);
  ierr = SNESCreate(PETSC_COMM_WORLD,&snes_alg);CHKERRQ(ierr);
  ierr = SNESSetFunction(snes_alg,F_alg,AlgFunction,ctx);CHKERRQ(ierr);
  ierr = MatZeroEntries(ctx->J);CHKERRQ(ierr);
  ierr = SNESSetJacobian(snes_alg,ctx->J,ctx->J,AlgJacobian,ctx);CHKERRQ(ierr);
  ierr = SNESSetOptionsPrefix(snes_alg,"alg_");CHKERRQ(ierr);
  ierr = SNESSetFromOptions(snes_alg);CHKERRQ(ierr);
  ctx->alg_flg = PETSC_TRUE;
  /* Solve the algebraic equations */
  ierr = SNESSolve(snes_alg,NULL,X);CHKERRQ(ierr);

  /* Just to set up the Jacobian structure */
  ierr = VecDuplicate(X,&Xdot);CHKERRQ(ierr);
  ierr = IJacobian(ts,0.0,X,Xdot,0.0,ctx->J,ctx->J,ctx);CHKERRQ(ierr);
  ierr = VecDestroy(&Xdot);CHKERRQ(ierr);

  ctx->stepnum++;

  /*
    Save trajectory of solution so that TSAdjointSolve() may be used
  */
  ierr = TSSetSaveTrajectory(ts);CHKERRQ(ierr);

  ierr = TSSetTimeStep(ts,0.01);CHKERRQ(ierr);
  ierr = TSSetExactFinalTime(ts,TS_EXACTFINALTIME_MATCHSTEP);CHKERRQ(ierr);
  ierr = TSSetFromOptions(ts);CHKERRQ(ierr);
  /* ierr = TSSetPostStep(ts,SaveSolution);CHKERRQ(ierr); */


  /* Prefault period */
  ctx->alg_flg = PETSC_FALSE;
  ierr = TSSetTime(ts,0.0);CHKERRQ(ierr);
  ierr = TSSetMaxTime(ts,ctx->tfaulton);CHKERRQ(ierr);
  ierr = TSSolve(ts,X);CHKERRQ(ierr);
  ierr = TSGetStepNumber(ts,&steps1);CHKERRQ(ierr);

  /* Create the nonlinear solver for solving the algebraic system */
  /* Note that although the algebraic system needs to be solved only for
     Idq and V, we reuse the entire system including xgen. The xgen
     variables are held constant by setting their residuals to 0 and
     putting a 1 on the Jacobian diagonal for xgen rows
  */
  ierr = MatZeroEntries(ctx->J);CHKERRQ(ierr);

  /* Apply disturbance - resistive fault at ctx->faultbus */
  /* This is done by adding shunt conductance to the diagonal location
     in the Ybus matrix */
  row_loc = 2*ctx->faultbus; col_loc = 2*ctx->faultbus+1; /* Location for G */
  val     = 1/ctx->Rfault;
  ierr    = MatSetValues(ctx->Ybus,1,&row_loc,1,&col_loc,&val,ADD_VALUES);CHKERRQ(ierr);
  row_loc = 2*ctx->faultbus+1; col_loc = 2*ctx->faultbus; /* Location for G */
  val     = 1/ctx->Rfault;
  ierr    = MatSetValues(ctx->Ybus,1,&row_loc,1,&col_loc,&val,ADD_VALUES);CHKERRQ(ierr);

  ierr = MatAssemblyBegin(ctx->Ybus,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(ctx->Ybus,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

  ctx->alg_flg = PETSC_TRUE;
  /* Solve the algebraic equations */
  ierr = SNESSolve(snes_alg,NULL,X);CHKERRQ(ierr);

  ctx->stepnum++;

  /* Disturbance period */
  ctx->alg_flg = PETSC_FALSE;
  ierr = TSSetTime(ts,ctx->tfaulton);CHKERRQ(ierr);
  ierr = TSSetMaxTime(ts,ctx->tfaultoff);CHKERRQ(ierr);
  ierr = TSSolve(ts,X);CHKERRQ(ierr);
  ierr = TSGetStepNumber(ts,&steps2);CHKERRQ(ierr);
  steps2 -= steps1;

  /* Remove the fault */
  row_loc = 2*ctx->faultbus; col_loc = 2*ctx->faultbus+1;
  val     = -1/ctx->Rfault;
  ierr    = MatSetValues(ctx->Ybus,1,&row_loc,1,&col_loc,&val,ADD_VALUES);CHKERRQ(ierr);
  row_loc = 2*ctx->faultbus+1; col_loc = 2*ctx->faultbus;
  val     = -1/ctx->Rfault;
  ierr    = MatSetValues(ctx->Ybus,1,&row_loc,1,&col_loc,&val,ADD_VALUES);CHKERRQ(ierr);

  ierr = MatAssemblyBegin(ctx->Ybus,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(ctx->Ybus,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

  ierr = MatZeroEntries(ctx->J);CHKERRQ(ierr);

  ctx->alg_flg = PETSC_TRUE;

  /* Solve the algebraic equations */
  ierr = SNESSolve(snes_alg,NULL,X);CHKERRQ(ierr);

  ctx->stepnum++;

  /* Post-disturbance period */
  ctx->alg_flg = PETSC_TRUE;
  ierr = TSSetTime(ts,ctx->tfaultoff);CHKERRQ(ierr);
  ierr = TSSetMaxTime(ts,ctx->tmax);CHKERRQ(ierr);
  ierr = TSSolve(ts,X);CHKERRQ(ierr);
  ierr = TSGetStepNumber(ts,&steps3);CHKERRQ(ierr);
  steps3 -= steps2;
  steps3 -= steps1;

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Adjoint model starts here
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = TSSetPostStep(ts,NULL);CHKERRQ(ierr);
  ierr = MatCreateVecs(ctx->J,&lambda[0],NULL);CHKERRQ(ierr);
  /*   Set initial conditions for the adjoint integration */
  ierr = VecZeroEntries(lambda[0]);CHKERRQ(ierr);

  ierr = MatCreateVecs(ctx->Jacp,&mu[0],NULL);CHKERRQ(ierr);
  ierr = VecZeroEntries(mu[0]);CHKERRQ(ierr);
  ierr = TSSetCostGradients(ts,1,lambda,mu);CHKERRQ(ierr);

  ierr = TSAdjointSetSteps(ts,steps3);CHKERRQ(ierr);
  ierr = TSAdjointSolve(ts);CHKERRQ(ierr);

  ierr = MatZeroEntries(ctx->J);CHKERRQ(ierr);
  /* Applying disturbance - resistive fault at ctx->faultbus */
  /* This is done by deducting shunt conductance to the diagonal location
     in the Ybus matrix */
  row_loc = 2*ctx->faultbus; col_loc = 2*ctx->faultbus+1; /* Location for G */
  val     = 1./ctx->Rfault;
  ierr    = MatSetValues(ctx->Ybus,1,&row_loc,1,&col_loc,&val,ADD_VALUES);CHKERRQ(ierr);
  row_loc = 2*ctx->faultbus+1; col_loc = 2*ctx->faultbus; /* Location for G */
  val     = 1./ctx->Rfault;
  ierr    = MatSetValues(ctx->Ybus,1,&row_loc,1,&col_loc,&val,ADD_VALUES);CHKERRQ(ierr);

  ierr = MatAssemblyBegin(ctx->Ybus,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(ctx->Ybus,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);


  /*   Set number of steps for the adjoint integration */
  ierr = TSAdjointSetSteps(ts,steps2);CHKERRQ(ierr);
  ierr = TSAdjointSolve(ts);CHKERRQ(ierr);

  ierr = MatZeroEntries(ctx->J);CHKERRQ(ierr);
  /* remove the fault */
  row_loc = 2*ctx->faultbus; col_loc = 2*ctx->faultbus+1; /* Location for G */
  val     = -1./ctx->Rfault;
  ierr    = MatSetValues(ctx->Ybus,1,&row_loc,1,&col_loc,&val,ADD_VALUES);CHKERRQ(ierr);
  row_loc = 2*ctx->faultbus+1; col_loc = 2*ctx->faultbus; /* Location for G */
  val     = -1./ctx->Rfault;
  ierr    = MatSetValues(ctx->Ybus,1,&row_loc,1,&col_loc,&val,ADD_VALUES);CHKERRQ(ierr);

  ierr = MatAssemblyBegin(ctx->Ybus,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(ctx->Ybus,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

  /*   Set number of steps for the adjoint integration */
  ierr = TSAdjointSetSteps(ts,steps1);CHKERRQ(ierr);
  ierr = TSAdjointSolve(ts);CHKERRQ(ierr);

  ierr = ComputeSensiP(lambda[0],mu[0],DICDP,ctx);CHKERRQ(ierr);
  ierr = VecCopy(mu[0],G);CHKERRQ(ierr);

  ierr = TSGetQuadratureTS(ts,NULL,&quadts);CHKERRQ(ierr);
  ierr = TSGetSolution(quadts,&q);CHKERRQ(ierr);
  ierr = VecGetArray(q,&x_ptr);CHKERRQ(ierr);
  *f   = x_ptr[0];
  x_ptr[0] = 0;
  ierr = VecRestoreArray(q,&x_ptr);CHKERRQ(ierr);

  ierr = VecDestroy(&lambda[0]);CHKERRQ(ierr);
  ierr = VecDestroy(&mu[0]);CHKERRQ(ierr);

  ierr = SNESDestroy(&snes_alg);CHKERRQ(ierr);
  ierr = VecDestroy(&F_alg);CHKERRQ(ierr);
  ierr = VecDestroy(&X);CHKERRQ(ierr);
  ierr = TSDestroy(&ts);CHKERRQ(ierr);
  for (i=0;i<3;i++) {
    ierr = VecDestroy(&DICDP[i]);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

/*TEST

   build:
      requires: double !complex !define(PETSC_USE_64BIT_INDICES)

   test:
      args: -viewer_binary_skip_info -tao_monitor -tao_gttol .2
      localrunfiles: petscoptions X.bin Ybus.bin

TEST*/
