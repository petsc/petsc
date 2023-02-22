static char help[] = "Application of adjoint sensitivity analysis for power grid stability analysis of WECC 9 bus system.\n\
This example is based on the 9-bus (node) example given in the book Power\n\
Systems Dynamics and Stability (Chapter 7) by P. Sauer and M. A. Pai.\n\
The power grid in this example consists of 9 buses (nodes), 3 generators,\n\
3 loads, and 9 transmission lines. The network equations are written\n\
in current balance form using rectangular coordinates.\n\n";

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

PetscErrorCode FormFunctionGradient(Tao, Vec, PetscReal *, Vec, void *);

#define freq 60
#define w_s  (2 * PETSC_PI * freq)

/* Sizes and indices */
const PetscInt nbus    = 9;         /* Number of network buses */
const PetscInt ngen    = 3;         /* Number of generators */
const PetscInt nload   = 3;         /* Number of loads */
const PetscInt gbus[3] = {0, 1, 2}; /* Buses at which generators are incident */
const PetscInt lbus[3] = {4, 5, 7}; /* Buses at which loads are incident */

/* Generator real and reactive powers (found via loadflow) */
PetscScalar PG[3] = {0.69, 1.59, 0.69};
/* PetscScalar PG[3] = {0.716786142395021,1.630000000000000,0.850000000000000};*/

const PetscScalar QG[3] = {0.270702180178785, 0.066120127797275, -0.108402221791588};
/* Generator constants */
const PetscScalar H[3]    = {23.64, 6.4, 3.01};       /* Inertia constant */
const PetscScalar Rs[3]   = {0.0, 0.0, 0.0};          /* Stator Resistance */
const PetscScalar Xd[3]   = {0.146, 0.8958, 1.3125};  /* d-axis reactance */
const PetscScalar Xdp[3]  = {0.0608, 0.1198, 0.1813}; /* d-axis transient reactance */
const PetscScalar Xq[3]   = {0.4360, 0.8645, 1.2578}; /* q-axis reactance Xq(1) set to 0.4360, value given in text 0.0969 */
const PetscScalar Xqp[3]  = {0.0969, 0.1969, 0.25};   /* q-axis transient reactance */
const PetscScalar Td0p[3] = {8.96, 6.0, 5.89};        /* d-axis open circuit time constant */
const PetscScalar Tq0p[3] = {0.31, 0.535, 0.6};       /* q-axis open circuit time constant */
PetscScalar       M[3];                               /* M = 2*H/w_s */
PetscScalar       D[3];                               /* D = 0.1*M */

PetscScalar TM[3]; /* Mechanical Torque */
/* Exciter system constants */
const PetscScalar KA[3] = {20.0, 20.0, 20.0};    /* Voltage regulartor gain constant */
const PetscScalar TA[3] = {0.2, 0.2, 0.2};       /* Voltage regulator time constant */
const PetscScalar KE[3] = {1.0, 1.0, 1.0};       /* Exciter gain constant */
const PetscScalar TE[3] = {0.314, 0.314, 0.314}; /* Exciter time constant */
const PetscScalar KF[3] = {0.063, 0.063, 0.063}; /* Feedback stabilizer gain constant */
const PetscScalar TF[3] = {0.35, 0.35, 0.35};    /* Feedback stabilizer time constant */
const PetscScalar k1[3] = {0.0039, 0.0039, 0.0039};
const PetscScalar k2[3] = {1.555, 1.555, 1.555}; /* k1 and k2 for calculating the saturation function SE = k1*exp(k2*Efd) */

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
const PetscScalar PD0[3]       = {1.25, 0.9, 1.0};
const PetscScalar QD0[3]       = {0.5, 0.3, 0.35};
const PetscInt    ld_nsegsp[3] = {3, 3, 3};
const PetscScalar ld_alphap[3] = {1.0, 0.0, 0.0};
const PetscScalar ld_betap[3]  = {2.0, 1.0, 0.0};
const PetscInt    ld_nsegsq[3] = {3, 3, 3};
const PetscScalar ld_alphaq[3] = {1.0, 0.0, 0.0};
const PetscScalar ld_betaq[3]  = {2.0, 1.0, 0.0};

typedef struct {
  DM          dmgen, dmnet;        /* DMs to manage generator and network subsystem */
  DM          dmpgrid;             /* Composite DM to manage the entire power grid */
  Mat         Ybus;                /* Network admittance matrix */
  Vec         V0;                  /* Initial voltage vector (Power flow solution) */
  PetscReal   tfaulton, tfaultoff; /* Fault on and off times */
  PetscInt    faultbus;            /* Fault bus */
  PetscScalar Rfault;
  PetscReal   t0, tmax;
  PetscInt    neqs_gen, neqs_net, neqs_pgrid;
  Mat         Sol; /* Matrix to save solution at each time step */
  PetscInt    stepnum;
  PetscBool   alg_flg;
  PetscReal   t;
  IS          is_diff;        /* indices for differential equations */
  IS          is_alg;         /* indices for algebraic equations */
  PetscReal   freq_u, freq_l; /* upper and lower frequency limit */
  PetscInt    pow;            /* power coefficient used in the cost function */
  PetscBool   jacp_flg;
  Mat         J, Jacp;
  Mat         DRDU, DRDP;
} Userctx;

/* Converts from machine frame (dq) to network (phase a real,imag) reference frame */
PetscErrorCode dq2ri(PetscScalar Fd, PetscScalar Fq, PetscScalar delta, PetscScalar *Fr, PetscScalar *Fi)
{
  PetscFunctionBegin;
  *Fr = Fd * PetscSinScalar(delta) + Fq * PetscCosScalar(delta);
  *Fi = -Fd * PetscCosScalar(delta) + Fq * PetscSinScalar(delta);
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* Converts from network frame ([phase a real,imag) to machine (dq) reference frame */
PetscErrorCode ri2dq(PetscScalar Fr, PetscScalar Fi, PetscScalar delta, PetscScalar *Fd, PetscScalar *Fq)
{
  PetscFunctionBegin;
  *Fd = Fr * PetscSinScalar(delta) - Fi * PetscCosScalar(delta);
  *Fq = Fr * PetscCosScalar(delta) + Fi * PetscSinScalar(delta);
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* Saves the solution at each time to a matrix */
PetscErrorCode SaveSolution(TS ts)
{
  Userctx           *user;
  Vec                X;
  PetscScalar       *mat;
  const PetscScalar *x;
  PetscInt           idx;
  PetscReal          t;

  PetscFunctionBegin;
  PetscCall(TSGetApplicationContext(ts, &user));
  PetscCall(TSGetTime(ts, &t));
  PetscCall(TSGetSolution(ts, &X));
  idx = user->stepnum * (user->neqs_pgrid + 1);
  PetscCall(MatDenseGetArray(user->Sol, &mat));
  PetscCall(VecGetArrayRead(X, &x));
  mat[idx] = t;
  PetscCall(PetscArraycpy(mat + idx + 1, x, user->neqs_pgrid));
  PetscCall(MatDenseRestoreArray(user->Sol, &mat));
  PetscCall(VecRestoreArrayRead(X, &x));
  user->stepnum++;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode SetInitialGuess(Vec X, Userctx *user)
{
  Vec          Xgen, Xnet;
  PetscScalar *xgen, *xnet;
  PetscInt     i, idx = 0;
  PetscScalar  Vr, Vi, IGr, IGi, Vm, Vm2;
  PetscScalar  Eqp, Edp, delta;
  PetscScalar  Efd, RF, VR; /* Exciter variables */
  PetscScalar  Id, Iq;      /* Generator dq axis currents */
  PetscScalar  theta, Vd, Vq, SE;

  PetscFunctionBegin;
  M[0] = 2 * H[0] / w_s;
  M[1] = 2 * H[1] / w_s;
  M[2] = 2 * H[2] / w_s;
  D[0] = 0.1 * M[0];
  D[1] = 0.1 * M[1];
  D[2] = 0.1 * M[2];

  PetscCall(DMCompositeGetLocalVectors(user->dmpgrid, &Xgen, &Xnet));

  /* Network subsystem initialization */
  PetscCall(VecCopy(user->V0, Xnet));

  /* Generator subsystem initialization */
  PetscCall(VecGetArray(Xgen, &xgen));
  PetscCall(VecGetArray(Xnet, &xnet));

  for (i = 0; i < ngen; i++) {
    Vr  = xnet[2 * gbus[i]];     /* Real part of generator terminal voltage */
    Vi  = xnet[2 * gbus[i] + 1]; /* Imaginary part of the generator terminal voltage */
    Vm  = PetscSqrtScalar(Vr * Vr + Vi * Vi);
    Vm2 = Vm * Vm;
    IGr = (Vr * PG[i] + Vi * QG[i]) / Vm2;
    IGi = (Vi * PG[i] - Vr * QG[i]) / Vm2;

    delta = PetscAtan2Real(Vi + Xq[i] * IGr, Vr - Xq[i] * IGi); /* Machine angle */

    theta = PETSC_PI / 2.0 - delta;

    Id = IGr * PetscCosScalar(theta) - IGi * PetscSinScalar(theta); /* d-axis stator current */
    Iq = IGr * PetscSinScalar(theta) + IGi * PetscCosScalar(theta); /* q-axis stator current */

    Vd = Vr * PetscCosScalar(theta) - Vi * PetscSinScalar(theta);
    Vq = Vr * PetscSinScalar(theta) + Vi * PetscCosScalar(theta);

    Edp = Vd + Rs[i] * Id - Xqp[i] * Iq; /* d-axis transient EMF */
    Eqp = Vq + Rs[i] * Iq + Xdp[i] * Id; /* q-axis transient EMF */

    TM[i] = PG[i];

    /* The generator variables are ordered as [Eqp,Edp,delta,w,Id,Iq] */
    xgen[idx]     = Eqp;
    xgen[idx + 1] = Edp;
    xgen[idx + 2] = delta;
    xgen[idx + 3] = w_s;

    idx = idx + 4;

    xgen[idx]     = Id;
    xgen[idx + 1] = Iq;

    idx = idx + 2;

    /* Exciter */
    Efd = Eqp + (Xd[i] - Xdp[i]) * Id;
    SE  = k1[i] * PetscExpScalar(k2[i] * Efd);
    VR  = KE[i] * Efd + SE;
    RF  = KF[i] * Efd / TF[i];

    xgen[idx]     = Efd;
    xgen[idx + 1] = RF;
    xgen[idx + 2] = VR;

    Vref[i] = Vm + (VR / KA[i]);

    idx = idx + 3;
  }

  PetscCall(VecRestoreArray(Xgen, &xgen));
  PetscCall(VecRestoreArray(Xnet, &xnet));

  /* PetscCall(VecView(Xgen,0)); */
  PetscCall(DMCompositeGather(user->dmpgrid, INSERT_VALUES, X, Xgen, Xnet));
  PetscCall(DMCompositeRestoreLocalVectors(user->dmpgrid, &Xgen, &Xnet));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode InitialGuess(Vec X, Userctx *user, const PetscScalar PGv[])
{
  Vec          Xgen, Xnet;
  PetscScalar *xgen, *xnet;
  PetscInt     i, idx = 0;
  PetscScalar  Vr, Vi, IGr, IGi, Vm, Vm2;
  PetscScalar  Eqp, Edp, delta;
  PetscScalar  Efd, RF, VR; /* Exciter variables */
  PetscScalar  Id, Iq;      /* Generator dq axis currents */
  PetscScalar  theta, Vd, Vq, SE;

  PetscFunctionBegin;
  M[0] = 2 * H[0] / w_s;
  M[1] = 2 * H[1] / w_s;
  M[2] = 2 * H[2] / w_s;
  D[0] = 0.1 * M[0];
  D[1] = 0.1 * M[1];
  D[2] = 0.1 * M[2];

  PetscCall(DMCompositeGetLocalVectors(user->dmpgrid, &Xgen, &Xnet));

  /* Network subsystem initialization */
  PetscCall(VecCopy(user->V0, Xnet));

  /* Generator subsystem initialization */
  PetscCall(VecGetArray(Xgen, &xgen));
  PetscCall(VecGetArray(Xnet, &xnet));

  for (i = 0; i < ngen; i++) {
    Vr  = xnet[2 * gbus[i]];     /* Real part of generator terminal voltage */
    Vi  = xnet[2 * gbus[i] + 1]; /* Imaginary part of the generator terminal voltage */
    Vm  = PetscSqrtScalar(Vr * Vr + Vi * Vi);
    Vm2 = Vm * Vm;
    IGr = (Vr * PGv[i] + Vi * QG[i]) / Vm2;
    IGi = (Vi * PGv[i] - Vr * QG[i]) / Vm2;

    delta = PetscAtan2Real(Vi + Xq[i] * IGr, Vr - Xq[i] * IGi); /* Machine angle */

    theta = PETSC_PI / 2.0 - delta;

    Id = IGr * PetscCosScalar(theta) - IGi * PetscSinScalar(theta); /* d-axis stator current */
    Iq = IGr * PetscSinScalar(theta) + IGi * PetscCosScalar(theta); /* q-axis stator current */

    Vd = Vr * PetscCosScalar(theta) - Vi * PetscSinScalar(theta);
    Vq = Vr * PetscSinScalar(theta) + Vi * PetscCosScalar(theta);

    Edp = Vd + Rs[i] * Id - Xqp[i] * Iq; /* d-axis transient EMF */
    Eqp = Vq + Rs[i] * Iq + Xdp[i] * Id; /* q-axis transient EMF */

    /* The generator variables are ordered as [Eqp,Edp,delta,w,Id,Iq] */
    xgen[idx]     = Eqp;
    xgen[idx + 1] = Edp;
    xgen[idx + 2] = delta;
    xgen[idx + 3] = w_s;

    idx = idx + 4;

    xgen[idx]     = Id;
    xgen[idx + 1] = Iq;

    idx = idx + 2;

    /* Exciter */
    Efd = Eqp + (Xd[i] - Xdp[i]) * Id;
    SE  = k1[i] * PetscExpScalar(k2[i] * Efd);
    VR  = KE[i] * Efd + SE;
    RF  = KF[i] * Efd / TF[i];

    xgen[idx]     = Efd;
    xgen[idx + 1] = RF;
    xgen[idx + 2] = VR;

    idx = idx + 3;
  }

  PetscCall(VecRestoreArray(Xgen, &xgen));
  PetscCall(VecRestoreArray(Xnet, &xnet));

  /* PetscCall(VecView(Xgen,0)); */
  PetscCall(DMCompositeGather(user->dmpgrid, INSERT_VALUES, X, Xgen, Xnet));
  PetscCall(DMCompositeRestoreLocalVectors(user->dmpgrid, &Xgen, &Xnet));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode DICDPFiniteDifference(Vec X, Vec *DICDP, Userctx *user)
{
  Vec         Y;
  PetscScalar PGv[3], eps;
  PetscInt    i, j;

  PetscFunctionBegin;
  eps = 1.e-7;
  PetscCall(VecDuplicate(X, &Y));

  for (i = 0; i < ngen; i++) {
    for (j = 0; j < 3; j++) PGv[j] = PG[j];
    PGv[i] = PG[i] + eps;
    PetscCall(InitialGuess(Y, user, PGv));
    PetscCall(InitialGuess(X, user, PG));

    PetscCall(VecAXPY(Y, -1.0, X));
    PetscCall(VecScale(Y, 1. / eps));
    PetscCall(VecCopy(Y, DICDP[i]));
  }
  PetscCall(VecDestroy(&Y));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* Computes F = [-f(x,y);g(x,y)] */
PetscErrorCode ResidualFunction(SNES snes, Vec X, Vec F, Userctx *user)
{
  Vec          Xgen, Xnet, Fgen, Fnet;
  PetscScalar *xgen, *xnet, *fgen, *fnet;
  PetscInt     i, idx = 0;
  PetscScalar  Vr, Vi, Vm, Vm2;
  PetscScalar  Eqp, Edp, delta, w; /* Generator variables */
  PetscScalar  Efd, RF, VR;        /* Exciter variables */
  PetscScalar  Id, Iq;             /* Generator dq axis currents */
  PetscScalar  Vd, Vq, SE;
  PetscScalar  IGr, IGi, IDr, IDi;
  PetscScalar  Zdq_inv[4], det;
  PetscScalar  PD, QD, Vm0, *v0;
  PetscInt     k;

  PetscFunctionBegin;
  PetscCall(VecZeroEntries(F));
  PetscCall(DMCompositeGetLocalVectors(user->dmpgrid, &Xgen, &Xnet));
  PetscCall(DMCompositeGetLocalVectors(user->dmpgrid, &Fgen, &Fnet));
  PetscCall(DMCompositeScatter(user->dmpgrid, X, Xgen, Xnet));
  PetscCall(DMCompositeScatter(user->dmpgrid, F, Fgen, Fnet));

  /* Network current balance residual IG + Y*V + IL = 0. Only YV is added here.
     The generator current injection, IG, and load current injection, ID are added later
  */
  /* Note that the values in Ybus are stored assuming the imaginary current balance
     equation is ordered first followed by real current balance equation for each bus.
     Thus imaginary current contribution goes in location 2*i, and
     real current contribution in 2*i+1
  */
  PetscCall(MatMult(user->Ybus, Xnet, Fnet));

  PetscCall(VecGetArray(Xgen, &xgen));
  PetscCall(VecGetArray(Xnet, &xnet));
  PetscCall(VecGetArray(Fgen, &fgen));
  PetscCall(VecGetArray(Fnet, &fnet));

  /* Generator subsystem */
  for (i = 0; i < ngen; i++) {
    Eqp   = xgen[idx];
    Edp   = xgen[idx + 1];
    delta = xgen[idx + 2];
    w     = xgen[idx + 3];
    Id    = xgen[idx + 4];
    Iq    = xgen[idx + 5];
    Efd   = xgen[idx + 6];
    RF    = xgen[idx + 7];
    VR    = xgen[idx + 8];

    /* Generator differential equations */
    fgen[idx]     = (Eqp + (Xd[i] - Xdp[i]) * Id - Efd) / Td0p[i];
    fgen[idx + 1] = (Edp - (Xq[i] - Xqp[i]) * Iq) / Tq0p[i];
    fgen[idx + 2] = -w + w_s;
    fgen[idx + 3] = (-TM[i] + Edp * Id + Eqp * Iq + (Xqp[i] - Xdp[i]) * Id * Iq + D[i] * (w - w_s)) / M[i];

    Vr = xnet[2 * gbus[i]];     /* Real part of generator terminal voltage */
    Vi = xnet[2 * gbus[i] + 1]; /* Imaginary part of the generator terminal voltage */

    PetscCall(ri2dq(Vr, Vi, delta, &Vd, &Vq));
    /* Algebraic equations for stator currents */
    det = Rs[i] * Rs[i] + Xdp[i] * Xqp[i];

    Zdq_inv[0] = Rs[i] / det;
    Zdq_inv[1] = Xqp[i] / det;
    Zdq_inv[2] = -Xdp[i] / det;
    Zdq_inv[3] = Rs[i] / det;

    fgen[idx + 4] = Zdq_inv[0] * (-Edp + Vd) + Zdq_inv[1] * (-Eqp + Vq) + Id;
    fgen[idx + 5] = Zdq_inv[2] * (-Edp + Vd) + Zdq_inv[3] * (-Eqp + Vq) + Iq;

    /* Add generator current injection to network */
    PetscCall(dq2ri(Id, Iq, delta, &IGr, &IGi));

    fnet[2 * gbus[i]] -= IGi;
    fnet[2 * gbus[i] + 1] -= IGr;

    Vm = PetscSqrtScalar(Vd * Vd + Vq * Vq);

    SE = k1[i] * PetscExpScalar(k2[i] * Efd);

    /* Exciter differential equations */
    fgen[idx + 6] = (KE[i] * Efd + SE - VR) / TE[i];
    fgen[idx + 7] = (RF - KF[i] * Efd / TF[i]) / TF[i];
    fgen[idx + 8] = (VR - KA[i] * RF + KA[i] * KF[i] * Efd / TF[i] - KA[i] * (Vref[i] - Vm)) / TA[i];

    idx = idx + 9;
  }

  PetscCall(VecGetArray(user->V0, &v0));
  for (i = 0; i < nload; i++) {
    Vr  = xnet[2 * lbus[i]];     /* Real part of load bus voltage */
    Vi  = xnet[2 * lbus[i] + 1]; /* Imaginary part of the load bus voltage */
    Vm  = PetscSqrtScalar(Vr * Vr + Vi * Vi);
    Vm2 = Vm * Vm;
    Vm0 = PetscSqrtScalar(v0[2 * lbus[i]] * v0[2 * lbus[i]] + v0[2 * lbus[i] + 1] * v0[2 * lbus[i] + 1]);
    PD = QD = 0.0;
    for (k = 0; k < ld_nsegsp[i]; k++) PD += ld_alphap[k] * PD0[i] * PetscPowScalar((Vm / Vm0), ld_betap[k]);
    for (k = 0; k < ld_nsegsq[i]; k++) QD += ld_alphaq[k] * QD0[i] * PetscPowScalar((Vm / Vm0), ld_betaq[k]);

    /* Load currents */
    IDr = (PD * Vr + QD * Vi) / Vm2;
    IDi = (-QD * Vr + PD * Vi) / Vm2;

    fnet[2 * lbus[i]] += IDi;
    fnet[2 * lbus[i] + 1] += IDr;
  }
  PetscCall(VecRestoreArray(user->V0, &v0));

  PetscCall(VecRestoreArray(Xgen, &xgen));
  PetscCall(VecRestoreArray(Xnet, &xnet));
  PetscCall(VecRestoreArray(Fgen, &fgen));
  PetscCall(VecRestoreArray(Fnet, &fnet));

  PetscCall(DMCompositeGather(user->dmpgrid, INSERT_VALUES, F, Fgen, Fnet));
  PetscCall(DMCompositeRestoreLocalVectors(user->dmpgrid, &Xgen, &Xnet));
  PetscCall(DMCompositeRestoreLocalVectors(user->dmpgrid, &Fgen, &Fnet));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* \dot{x} - f(x,y)
     g(x,y) = 0
 */
PetscErrorCode IFunction(TS ts, PetscReal t, Vec X, Vec Xdot, Vec F, Userctx *user)
{
  SNES               snes;
  PetscScalar       *f;
  const PetscScalar *xdot;
  PetscInt           i;

  PetscFunctionBegin;
  user->t = t;

  PetscCall(TSGetSNES(ts, &snes));
  PetscCall(ResidualFunction(snes, X, F, user));
  PetscCall(VecGetArray(F, &f));
  PetscCall(VecGetArrayRead(Xdot, &xdot));
  for (i = 0; i < ngen; i++) {
    f[9 * i] += xdot[9 * i];
    f[9 * i + 1] += xdot[9 * i + 1];
    f[9 * i + 2] += xdot[9 * i + 2];
    f[9 * i + 3] += xdot[9 * i + 3];
    f[9 * i + 6] += xdot[9 * i + 6];
    f[9 * i + 7] += xdot[9 * i + 7];
    f[9 * i + 8] += xdot[9 * i + 8];
  }
  PetscCall(VecRestoreArray(F, &f));
  PetscCall(VecRestoreArrayRead(Xdot, &xdot));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* This function is used for solving the algebraic system only during fault on and
   off times. It computes the entire F and then zeros out the part corresponding to
   differential equations
 F = [0;g(y)];
*/
PetscErrorCode AlgFunction(SNES snes, Vec X, Vec F, void *ctx)
{
  Userctx     *user = (Userctx *)ctx;
  PetscScalar *f;
  PetscInt     i;

  PetscFunctionBegin;
  PetscCall(ResidualFunction(snes, X, F, user));
  PetscCall(VecGetArray(F, &f));
  for (i = 0; i < ngen; i++) {
    f[9 * i]     = 0;
    f[9 * i + 1] = 0;
    f[9 * i + 2] = 0;
    f[9 * i + 3] = 0;
    f[9 * i + 6] = 0;
    f[9 * i + 7] = 0;
    f[9 * i + 8] = 0;
  }
  PetscCall(VecRestoreArray(F, &f));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode PreallocateJacobian(Mat J, Userctx *user)
{
  PetscInt *d_nnz;
  PetscInt  i, idx = 0, start = 0;
  PetscInt  ncols;

  PetscFunctionBegin;
  PetscCall(PetscMalloc1(user->neqs_pgrid, &d_nnz));
  for (i = 0; i < user->neqs_pgrid; i++) d_nnz[i] = 0;
  /* Generator subsystem */
  for (i = 0; i < ngen; i++) {
    d_nnz[idx] += 3;
    d_nnz[idx + 1] += 2;
    d_nnz[idx + 2] += 2;
    d_nnz[idx + 3] += 5;
    d_nnz[idx + 4] += 6;
    d_nnz[idx + 5] += 6;

    d_nnz[user->neqs_gen + 2 * gbus[i]] += 3;
    d_nnz[user->neqs_gen + 2 * gbus[i] + 1] += 3;

    d_nnz[idx + 6] += 2;
    d_nnz[idx + 7] += 2;
    d_nnz[idx + 8] += 5;

    idx = idx + 9;
  }

  start = user->neqs_gen;
  for (i = 0; i < nbus; i++) {
    PetscCall(MatGetRow(user->Ybus, 2 * i, &ncols, NULL, NULL));
    d_nnz[start + 2 * i] += ncols;
    d_nnz[start + 2 * i + 1] += ncols;
    PetscCall(MatRestoreRow(user->Ybus, 2 * i, &ncols, NULL, NULL));
  }

  PetscCall(MatSeqAIJSetPreallocation(J, 0, d_nnz));
  PetscCall(PetscFree(d_nnz));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
   J = [-df_dx, -df_dy
        dg_dx, dg_dy]
*/
PetscErrorCode ResidualJacobian(SNES snes, Vec X, Mat J, Mat B, void *ctx)
{
  Userctx           *user = (Userctx *)ctx;
  Vec                Xgen, Xnet;
  PetscScalar       *xgen, *xnet;
  PetscInt           i, idx = 0;
  PetscScalar        Vr, Vi, Vm, Vm2;
  PetscScalar        Eqp, Edp, delta; /* Generator variables */
  PetscScalar        Efd;             /* Exciter variables */
  PetscScalar        Id, Iq;          /* Generator dq axis currents */
  PetscScalar        Vd, Vq;
  PetscScalar        val[10];
  PetscInt           row[2], col[10];
  PetscInt           net_start = user->neqs_gen;
  PetscInt           ncols;
  const PetscInt    *cols;
  const PetscScalar *yvals;
  PetscInt           k;
  PetscScalar        Zdq_inv[4], det;
  PetscScalar        dVd_dVr, dVd_dVi, dVq_dVr, dVq_dVi, dVd_ddelta, dVq_ddelta;
  PetscScalar        dIGr_ddelta, dIGi_ddelta, dIGr_dId, dIGr_dIq, dIGi_dId, dIGi_dIq;
  PetscScalar        dSE_dEfd;
  PetscScalar        dVm_dVd, dVm_dVq, dVm_dVr, dVm_dVi;
  PetscScalar        PD, QD, Vm0, *v0, Vm4;
  PetscScalar        dPD_dVr, dPD_dVi, dQD_dVr, dQD_dVi;
  PetscScalar        dIDr_dVr, dIDr_dVi, dIDi_dVr, dIDi_dVi;

  PetscFunctionBegin;
  PetscCall(MatZeroEntries(B));
  PetscCall(DMCompositeGetLocalVectors(user->dmpgrid, &Xgen, &Xnet));
  PetscCall(DMCompositeScatter(user->dmpgrid, X, Xgen, Xnet));

  PetscCall(VecGetArray(Xgen, &xgen));
  PetscCall(VecGetArray(Xnet, &xnet));

  /* Generator subsystem */
  for (i = 0; i < ngen; i++) {
    Eqp   = xgen[idx];
    Edp   = xgen[idx + 1];
    delta = xgen[idx + 2];
    Id    = xgen[idx + 4];
    Iq    = xgen[idx + 5];
    Efd   = xgen[idx + 6];

    /*    fgen[idx]   = (Eqp + (Xd[i] - Xdp[i])*Id - Efd)/Td0p[i]; */
    row[0] = idx;
    col[0] = idx;
    col[1] = idx + 4;
    col[2] = idx + 6;
    val[0] = 1 / Td0p[i];
    val[1] = (Xd[i] - Xdp[i]) / Td0p[i];
    val[2] = -1 / Td0p[i];

    PetscCall(MatSetValues(J, 1, row, 3, col, val, INSERT_VALUES));

    /*    fgen[idx+1] = (Edp - (Xq[i] - Xqp[i])*Iq)/Tq0p[i]; */
    row[0] = idx + 1;
    col[0] = idx + 1;
    col[1] = idx + 5;
    val[0] = 1 / Tq0p[i];
    val[1] = -(Xq[i] - Xqp[i]) / Tq0p[i];
    PetscCall(MatSetValues(J, 1, row, 2, col, val, INSERT_VALUES));

    /*    fgen[idx+2] = - w + w_s; */
    row[0] = idx + 2;
    col[0] = idx + 2;
    col[1] = idx + 3;
    val[0] = 0;
    val[1] = -1;
    PetscCall(MatSetValues(J, 1, row, 2, col, val, INSERT_VALUES));

    /*    fgen[idx+3] = (-TM[i] + Edp*Id + Eqp*Iq + (Xqp[i] - Xdp[i])*Id*Iq + D[i]*(w - w_s))/M[i]; */
    row[0] = idx + 3;
    col[0] = idx;
    col[1] = idx + 1;
    col[2] = idx + 3;
    col[3] = idx + 4;
    col[4] = idx + 5;
    val[0] = Iq / M[i];
    val[1] = Id / M[i];
    val[2] = D[i] / M[i];
    val[3] = (Edp + (Xqp[i] - Xdp[i]) * Iq) / M[i];
    val[4] = (Eqp + (Xqp[i] - Xdp[i]) * Id) / M[i];
    PetscCall(MatSetValues(J, 1, row, 5, col, val, INSERT_VALUES));

    Vr = xnet[2 * gbus[i]];     /* Real part of generator terminal voltage */
    Vi = xnet[2 * gbus[i] + 1]; /* Imaginary part of the generator terminal voltage */
    PetscCall(ri2dq(Vr, Vi, delta, &Vd, &Vq));

    det = Rs[i] * Rs[i] + Xdp[i] * Xqp[i];

    Zdq_inv[0] = Rs[i] / det;
    Zdq_inv[1] = Xqp[i] / det;
    Zdq_inv[2] = -Xdp[i] / det;
    Zdq_inv[3] = Rs[i] / det;

    dVd_dVr    = PetscSinScalar(delta);
    dVd_dVi    = -PetscCosScalar(delta);
    dVq_dVr    = PetscCosScalar(delta);
    dVq_dVi    = PetscSinScalar(delta);
    dVd_ddelta = Vr * PetscCosScalar(delta) + Vi * PetscSinScalar(delta);
    dVq_ddelta = -Vr * PetscSinScalar(delta) + Vi * PetscCosScalar(delta);

    /*    fgen[idx+4] = Zdq_inv[0]*(-Edp + Vd) + Zdq_inv[1]*(-Eqp + Vq) + Id; */
    row[0] = idx + 4;
    col[0] = idx;
    col[1] = idx + 1;
    col[2] = idx + 2;
    val[0] = -Zdq_inv[1];
    val[1] = -Zdq_inv[0];
    val[2] = Zdq_inv[0] * dVd_ddelta + Zdq_inv[1] * dVq_ddelta;
    col[3] = idx + 4;
    col[4] = net_start + 2 * gbus[i];
    col[5] = net_start + 2 * gbus[i] + 1;
    val[3] = 1;
    val[4] = Zdq_inv[0] * dVd_dVr + Zdq_inv[1] * dVq_dVr;
    val[5] = Zdq_inv[0] * dVd_dVi + Zdq_inv[1] * dVq_dVi;
    PetscCall(MatSetValues(J, 1, row, 6, col, val, INSERT_VALUES));

    /*  fgen[idx+5] = Zdq_inv[2]*(-Edp + Vd) + Zdq_inv[3]*(-Eqp + Vq) + Iq; */
    row[0] = idx + 5;
    col[0] = idx;
    col[1] = idx + 1;
    col[2] = idx + 2;
    val[0] = -Zdq_inv[3];
    val[1] = -Zdq_inv[2];
    val[2] = Zdq_inv[2] * dVd_ddelta + Zdq_inv[3] * dVq_ddelta;
    col[3] = idx + 5;
    col[4] = net_start + 2 * gbus[i];
    col[5] = net_start + 2 * gbus[i] + 1;
    val[3] = 1;
    val[4] = Zdq_inv[2] * dVd_dVr + Zdq_inv[3] * dVq_dVr;
    val[5] = Zdq_inv[2] * dVd_dVi + Zdq_inv[3] * dVq_dVi;
    PetscCall(MatSetValues(J, 1, row, 6, col, val, INSERT_VALUES));

    dIGr_ddelta = Id * PetscCosScalar(delta) - Iq * PetscSinScalar(delta);
    dIGi_ddelta = Id * PetscSinScalar(delta) + Iq * PetscCosScalar(delta);
    dIGr_dId    = PetscSinScalar(delta);
    dIGr_dIq    = PetscCosScalar(delta);
    dIGi_dId    = -PetscCosScalar(delta);
    dIGi_dIq    = PetscSinScalar(delta);

    /* fnet[2*gbus[i]]   -= IGi; */
    row[0] = net_start + 2 * gbus[i];
    col[0] = idx + 2;
    col[1] = idx + 4;
    col[2] = idx + 5;
    val[0] = -dIGi_ddelta;
    val[1] = -dIGi_dId;
    val[2] = -dIGi_dIq;
    PetscCall(MatSetValues(J, 1, row, 3, col, val, INSERT_VALUES));

    /* fnet[2*gbus[i]+1]   -= IGr; */
    row[0] = net_start + 2 * gbus[i] + 1;
    col[0] = idx + 2;
    col[1] = idx + 4;
    col[2] = idx + 5;
    val[0] = -dIGr_ddelta;
    val[1] = -dIGr_dId;
    val[2] = -dIGr_dIq;
    PetscCall(MatSetValues(J, 1, row, 3, col, val, INSERT_VALUES));

    Vm = PetscSqrtScalar(Vd * Vd + Vq * Vq);

    /*    fgen[idx+6] = (KE[i]*Efd + SE - VR)/TE[i]; */
    /*    SE  = k1[i]*PetscExpScalar(k2[i]*Efd); */
    dSE_dEfd = k1[i] * k2[i] * PetscExpScalar(k2[i] * Efd);

    row[0] = idx + 6;
    col[0] = idx + 6;
    col[1] = idx + 8;
    val[0] = (KE[i] + dSE_dEfd) / TE[i];
    val[1] = -1 / TE[i];
    PetscCall(MatSetValues(J, 1, row, 2, col, val, INSERT_VALUES));

    /* Exciter differential equations */

    /*    fgen[idx+7] = (RF - KF[i]*Efd/TF[i])/TF[i]; */
    row[0] = idx + 7;
    col[0] = idx + 6;
    col[1] = idx + 7;
    val[0] = (-KF[i] / TF[i]) / TF[i];
    val[1] = 1 / TF[i];
    PetscCall(MatSetValues(J, 1, row, 2, col, val, INSERT_VALUES));

    /*    fgen[idx+8] = (VR - KA[i]*RF + KA[i]*KF[i]*Efd/TF[i] - KA[i]*(Vref[i] - Vm))/TA[i]; */
    /* Vm = (Vd^2 + Vq^2)^0.5; */
    dVm_dVd = Vd / Vm;
    dVm_dVq = Vq / Vm;
    dVm_dVr = dVm_dVd * dVd_dVr + dVm_dVq * dVq_dVr;
    dVm_dVi = dVm_dVd * dVd_dVi + dVm_dVq * dVq_dVi;
    row[0]  = idx + 8;
    col[0]  = idx + 6;
    col[1]  = idx + 7;
    col[2]  = idx + 8;
    val[0]  = (KA[i] * KF[i] / TF[i]) / TA[i];
    val[1]  = -KA[i] / TA[i];
    val[2]  = 1 / TA[i];
    col[3]  = net_start + 2 * gbus[i];
    col[4]  = net_start + 2 * gbus[i] + 1;
    val[3]  = KA[i] * dVm_dVr / TA[i];
    val[4]  = KA[i] * dVm_dVi / TA[i];
    PetscCall(MatSetValues(J, 1, row, 5, col, val, INSERT_VALUES));
    idx = idx + 9;
  }

  for (i = 0; i < nbus; i++) {
    PetscCall(MatGetRow(user->Ybus, 2 * i, &ncols, &cols, &yvals));
    row[0] = net_start + 2 * i;
    for (k = 0; k < ncols; k++) {
      col[k] = net_start + cols[k];
      val[k] = yvals[k];
    }
    PetscCall(MatSetValues(J, 1, row, ncols, col, val, INSERT_VALUES));
    PetscCall(MatRestoreRow(user->Ybus, 2 * i, &ncols, &cols, &yvals));

    PetscCall(MatGetRow(user->Ybus, 2 * i + 1, &ncols, &cols, &yvals));
    row[0] = net_start + 2 * i + 1;
    for (k = 0; k < ncols; k++) {
      col[k] = net_start + cols[k];
      val[k] = yvals[k];
    }
    PetscCall(MatSetValues(J, 1, row, ncols, col, val, INSERT_VALUES));
    PetscCall(MatRestoreRow(user->Ybus, 2 * i + 1, &ncols, &cols, &yvals));
  }

  PetscCall(MatAssemblyBegin(J, MAT_FLUSH_ASSEMBLY));
  PetscCall(MatAssemblyEnd(J, MAT_FLUSH_ASSEMBLY));

  PetscCall(VecGetArray(user->V0, &v0));
  for (i = 0; i < nload; i++) {
    Vr  = xnet[2 * lbus[i]];     /* Real part of load bus voltage */
    Vi  = xnet[2 * lbus[i] + 1]; /* Imaginary part of the load bus voltage */
    Vm  = PetscSqrtScalar(Vr * Vr + Vi * Vi);
    Vm2 = Vm * Vm;
    Vm4 = Vm2 * Vm2;
    Vm0 = PetscSqrtScalar(v0[2 * lbus[i]] * v0[2 * lbus[i]] + v0[2 * lbus[i] + 1] * v0[2 * lbus[i] + 1]);
    PD = QD = 0.0;
    dPD_dVr = dPD_dVi = dQD_dVr = dQD_dVi = 0.0;
    for (k = 0; k < ld_nsegsp[i]; k++) {
      PD += ld_alphap[k] * PD0[i] * PetscPowScalar((Vm / Vm0), ld_betap[k]);
      dPD_dVr += ld_alphap[k] * ld_betap[k] * PD0[i] * PetscPowScalar((1 / Vm0), ld_betap[k]) * Vr * PetscPowScalar(Vm, (ld_betap[k] - 2));
      dPD_dVi += ld_alphap[k] * ld_betap[k] * PD0[i] * PetscPowScalar((1 / Vm0), ld_betap[k]) * Vi * PetscPowScalar(Vm, (ld_betap[k] - 2));
    }
    for (k = 0; k < ld_nsegsq[i]; k++) {
      QD += ld_alphaq[k] * QD0[i] * PetscPowScalar((Vm / Vm0), ld_betaq[k]);
      dQD_dVr += ld_alphaq[k] * ld_betaq[k] * QD0[i] * PetscPowScalar((1 / Vm0), ld_betaq[k]) * Vr * PetscPowScalar(Vm, (ld_betaq[k] - 2));
      dQD_dVi += ld_alphaq[k] * ld_betaq[k] * QD0[i] * PetscPowScalar((1 / Vm0), ld_betaq[k]) * Vi * PetscPowScalar(Vm, (ld_betaq[k] - 2));
    }

    /*    IDr = (PD*Vr + QD*Vi)/Vm2; */
    /*    IDi = (-QD*Vr + PD*Vi)/Vm2; */

    dIDr_dVr = (dPD_dVr * Vr + dQD_dVr * Vi + PD) / Vm2 - ((PD * Vr + QD * Vi) * 2 * Vr) / Vm4;
    dIDr_dVi = (dPD_dVi * Vr + dQD_dVi * Vi + QD) / Vm2 - ((PD * Vr + QD * Vi) * 2 * Vi) / Vm4;

    dIDi_dVr = (-dQD_dVr * Vr + dPD_dVr * Vi - QD) / Vm2 - ((-QD * Vr + PD * Vi) * 2 * Vr) / Vm4;
    dIDi_dVi = (-dQD_dVi * Vr + dPD_dVi * Vi + PD) / Vm2 - ((-QD * Vr + PD * Vi) * 2 * Vi) / Vm4;

    /*    fnet[2*lbus[i]]   += IDi; */
    row[0] = net_start + 2 * lbus[i];
    col[0] = net_start + 2 * lbus[i];
    col[1] = net_start + 2 * lbus[i] + 1;
    val[0] = dIDi_dVr;
    val[1] = dIDi_dVi;
    PetscCall(MatSetValues(J, 1, row, 2, col, val, ADD_VALUES));
    /*    fnet[2*lbus[i]+1] += IDr; */
    row[0] = net_start + 2 * lbus[i] + 1;
    col[0] = net_start + 2 * lbus[i];
    col[1] = net_start + 2 * lbus[i] + 1;
    val[0] = dIDr_dVr;
    val[1] = dIDr_dVi;
    PetscCall(MatSetValues(J, 1, row, 2, col, val, ADD_VALUES));
  }
  PetscCall(VecRestoreArray(user->V0, &v0));

  PetscCall(VecRestoreArray(Xgen, &xgen));
  PetscCall(VecRestoreArray(Xnet, &xnet));

  PetscCall(DMCompositeRestoreLocalVectors(user->dmpgrid, &Xgen, &Xnet));

  PetscCall(MatAssemblyBegin(J, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(J, MAT_FINAL_ASSEMBLY));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
   J = [I, 0
        dg_dx, dg_dy]
*/
PetscErrorCode AlgJacobian(SNES snes, Vec X, Mat A, Mat B, void *ctx)
{
  Userctx *user = (Userctx *)ctx;

  PetscFunctionBegin;
  PetscCall(ResidualJacobian(snes, X, A, B, ctx));
  PetscCall(MatSetOption(A, MAT_KEEP_NONZERO_PATTERN, PETSC_TRUE));
  PetscCall(MatZeroRowsIS(A, user->is_diff, 1.0, NULL, NULL));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
   J = [a*I-df_dx, -df_dy
        dg_dx, dg_dy]
*/

PetscErrorCode IJacobian(TS ts, PetscReal t, Vec X, Vec Xdot, PetscReal a, Mat A, Mat B, Userctx *user)
{
  SNES        snes;
  PetscScalar atmp = (PetscScalar)a;
  PetscInt    i, row;

  PetscFunctionBegin;
  user->t = t;

  PetscCall(TSGetSNES(ts, &snes));
  PetscCall(ResidualJacobian(snes, X, A, B, user));
  for (i = 0; i < ngen; i++) {
    row = 9 * i;
    PetscCall(MatSetValues(A, 1, &row, 1, &row, &atmp, ADD_VALUES));
    row = 9 * i + 1;
    PetscCall(MatSetValues(A, 1, &row, 1, &row, &atmp, ADD_VALUES));
    row = 9 * i + 2;
    PetscCall(MatSetValues(A, 1, &row, 1, &row, &atmp, ADD_VALUES));
    row = 9 * i + 3;
    PetscCall(MatSetValues(A, 1, &row, 1, &row, &atmp, ADD_VALUES));
    row = 9 * i + 6;
    PetscCall(MatSetValues(A, 1, &row, 1, &row, &atmp, ADD_VALUES));
    row = 9 * i + 7;
    PetscCall(MatSetValues(A, 1, &row, 1, &row, &atmp, ADD_VALUES));
    row = 9 * i + 8;
    PetscCall(MatSetValues(A, 1, &row, 1, &row, &atmp, ADD_VALUES));
  }
  PetscCall(MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* Matrix JacobianP is constant so that it only needs to be evaluated once */
static PetscErrorCode RHSJacobianP(TS ts, PetscReal t, Vec X, Mat A, void *ctx0)
{
  PetscScalar a;
  PetscInt    row, col;
  Userctx    *ctx = (Userctx *)ctx0;

  PetscFunctionBeginUser;

  if (ctx->jacp_flg) {
    PetscCall(MatZeroEntries(A));

    for (col = 0; col < 3; col++) {
      a   = 1.0 / M[col];
      row = 9 * col + 3;
      PetscCall(MatSetValues(A, 1, &row, 1, &col, &a, INSERT_VALUES));
    }

    ctx->jacp_flg = PETSC_FALSE;

    PetscCall(MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY));
    PetscCall(MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode CostIntegrand(TS ts, PetscReal t, Vec U, Vec R, Userctx *user)
{
  const PetscScalar *u;
  PetscInt           idx;
  Vec                Xgen, Xnet;
  PetscScalar       *r, *xgen;
  PetscInt           i;

  PetscFunctionBegin;
  PetscCall(DMCompositeGetLocalVectors(user->dmpgrid, &Xgen, &Xnet));
  PetscCall(DMCompositeScatter(user->dmpgrid, U, Xgen, Xnet));

  PetscCall(VecGetArray(Xgen, &xgen));

  PetscCall(VecGetArrayRead(U, &u));
  PetscCall(VecGetArray(R, &r));
  r[0] = 0.;
  idx  = 0;
  for (i = 0; i < ngen; i++) {
    r[0] += PetscPowScalarInt(PetscMax(0., PetscMax(xgen[idx + 3] / (2. * PETSC_PI) - user->freq_u, user->freq_l - xgen[idx + 3] / (2. * PETSC_PI))), user->pow);
    idx += 9;
  }
  PetscCall(VecRestoreArrayRead(U, &u));
  PetscCall(VecRestoreArray(R, &r));
  PetscCall(DMCompositeRestoreLocalVectors(user->dmpgrid, &Xgen, &Xnet));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode DRDUJacobianTranspose(TS ts, PetscReal t, Vec U, Mat DRDU, Mat B, Userctx *user)
{
  Vec          Xgen, Xnet, Dgen, Dnet;
  PetscScalar *xgen, *dgen;
  PetscInt     i;
  PetscInt     idx;
  Vec          drdu_col;
  PetscScalar *xarr;

  PetscFunctionBegin;
  PetscCall(VecDuplicate(U, &drdu_col));
  PetscCall(MatDenseGetColumn(DRDU, 0, &xarr));
  PetscCall(VecPlaceArray(drdu_col, xarr));
  PetscCall(DMCompositeGetLocalVectors(user->dmpgrid, &Xgen, &Xnet));
  PetscCall(DMCompositeGetLocalVectors(user->dmpgrid, &Dgen, &Dnet));
  PetscCall(DMCompositeScatter(user->dmpgrid, U, Xgen, Xnet));
  PetscCall(DMCompositeScatter(user->dmpgrid, drdu_col, Dgen, Dnet));

  PetscCall(VecGetArray(Xgen, &xgen));
  PetscCall(VecGetArray(Dgen, &dgen));

  idx = 0;
  for (i = 0; i < ngen; i++) {
    dgen[idx + 3] = 0.;
    if (xgen[idx + 3] / (2. * PETSC_PI) > user->freq_u) dgen[idx + 3] = user->pow * PetscPowScalarInt(xgen[idx + 3] / (2. * PETSC_PI) - user->freq_u, user->pow - 1) / (2. * PETSC_PI);
    if (xgen[idx + 3] / (2. * PETSC_PI) < user->freq_l) dgen[idx + 3] = user->pow * PetscPowScalarInt(user->freq_l - xgen[idx + 3] / (2. * PETSC_PI), user->pow - 1) / (-2. * PETSC_PI);
    idx += 9;
  }

  PetscCall(VecRestoreArray(Dgen, &dgen));
  PetscCall(VecRestoreArray(Xgen, &xgen));
  PetscCall(DMCompositeGather(user->dmpgrid, INSERT_VALUES, drdu_col, Dgen, Dnet));
  PetscCall(DMCompositeRestoreLocalVectors(user->dmpgrid, &Dgen, &Dnet));
  PetscCall(DMCompositeRestoreLocalVectors(user->dmpgrid, &Xgen, &Xnet));
  PetscCall(VecResetArray(drdu_col));
  PetscCall(MatDenseRestoreColumn(DRDU, &xarr));
  PetscCall(VecDestroy(&drdu_col));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode DRDPJacobianTranspose(TS ts, PetscReal t, Vec U, Mat drdp, Userctx *user)
{
  PetscFunctionBegin;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode ComputeSensiP(Vec lambda, Vec mu, Vec *DICDP, Userctx *user)
{
  PetscScalar *x, *y, sensip;
  PetscInt     i;

  PetscFunctionBegin;
  PetscCall(VecGetArray(lambda, &x));
  PetscCall(VecGetArray(mu, &y));

  for (i = 0; i < 3; i++) {
    PetscCall(VecDot(lambda, DICDP[i], &sensip));
    sensip = sensip + y[i];
    /* PetscCall(PetscPrintf(PETSC_COMM_WORLD,"\n sensitivity wrt %" PetscInt_FMT " th parameter: %g \n",i,(double)sensip)); */
    y[i] = sensip;
  }
  PetscCall(VecRestoreArray(mu, &y));
  PetscFunctionReturn(PETSC_SUCCESS);
}

int main(int argc, char **argv)
{
  Userctx      user;
  Vec          p;
  PetscScalar *x_ptr;
  PetscMPIInt  size;
  PetscInt     i;
  PetscViewer  Xview, Ybusview;
  PetscInt    *idx2;
  Tao          tao;
  KSP          ksp;
  PC           pc;
  Vec          lowerb, upperb;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, "petscoptions", help));
  PetscCallMPI(MPI_Comm_size(PETSC_COMM_WORLD, &size));
  PetscCheck(size == 1, PETSC_COMM_WORLD, PETSC_ERR_WRONG_MPI_SIZE, "Only for sequential runs");

  user.jacp_flg   = PETSC_TRUE;
  user.neqs_gen   = 9 * ngen; /* # eqs. for generator subsystem */
  user.neqs_net   = 2 * nbus; /* # eqs. for network subsystem   */
  user.neqs_pgrid = user.neqs_gen + user.neqs_net;

  /* Create indices for differential and algebraic equations */
  PetscCall(PetscMalloc1(7 * ngen, &idx2));
  for (i = 0; i < ngen; i++) {
    idx2[7 * i]     = 9 * i;
    idx2[7 * i + 1] = 9 * i + 1;
    idx2[7 * i + 2] = 9 * i + 2;
    idx2[7 * i + 3] = 9 * i + 3;
    idx2[7 * i + 4] = 9 * i + 6;
    idx2[7 * i + 5] = 9 * i + 7;
    idx2[7 * i + 6] = 9 * i + 8;
  }
  PetscCall(ISCreateGeneral(PETSC_COMM_WORLD, 7 * ngen, idx2, PETSC_COPY_VALUES, &user.is_diff));
  PetscCall(ISComplement(user.is_diff, 0, user.neqs_pgrid, &user.is_alg));
  PetscCall(PetscFree(idx2));

  /* Set run time options */
  PetscOptionsBegin(PETSC_COMM_WORLD, NULL, "Transient stability fault options", "");
  {
    user.tfaulton  = 1.0;
    user.tfaultoff = 1.2;
    user.Rfault    = 0.0001;
    user.faultbus  = 8;
    PetscCall(PetscOptionsReal("-tfaulton", "", "", user.tfaulton, &user.tfaulton, NULL));
    PetscCall(PetscOptionsReal("-tfaultoff", "", "", user.tfaultoff, &user.tfaultoff, NULL));
    PetscCall(PetscOptionsInt("-faultbus", "", "", user.faultbus, &user.faultbus, NULL));
    user.t0   = 0.0;
    user.tmax = 1.3;
    PetscCall(PetscOptionsReal("-t0", "", "", user.t0, &user.t0, NULL));
    PetscCall(PetscOptionsReal("-tmax", "", "", user.tmax, &user.tmax, NULL));
    user.freq_u = 61.0;
    user.freq_l = 59.0;
    user.pow    = 2;
    PetscCall(PetscOptionsReal("-frequ", "", "", user.freq_u, &user.freq_u, NULL));
    PetscCall(PetscOptionsReal("-freql", "", "", user.freq_l, &user.freq_l, NULL));
    PetscCall(PetscOptionsInt("-pow", "", "", user.pow, &user.pow, NULL));
  }
  PetscOptionsEnd();

  /* Create DMs for generator and network subsystems */
  PetscCall(DMDACreate1d(PETSC_COMM_WORLD, DM_BOUNDARY_NONE, user.neqs_gen, 1, 1, NULL, &user.dmgen));
  PetscCall(DMSetOptionsPrefix(user.dmgen, "dmgen_"));
  PetscCall(DMDACreate1d(PETSC_COMM_WORLD, DM_BOUNDARY_NONE, user.neqs_net, 1, 1, NULL, &user.dmnet));
  PetscCall(DMSetOptionsPrefix(user.dmnet, "dmnet_"));
  PetscCall(DMSetFromOptions(user.dmnet));
  PetscCall(DMSetUp(user.dmnet));
  /* Create a composite DM packer and add the two DMs */
  PetscCall(DMCompositeCreate(PETSC_COMM_WORLD, &user.dmpgrid));
  PetscCall(DMSetOptionsPrefix(user.dmpgrid, "pgrid_"));
  PetscCall(DMSetFromOptions(user.dmgen));
  PetscCall(DMSetUp(user.dmgen));
  PetscCall(DMCompositeAddDM(user.dmpgrid, user.dmgen));
  PetscCall(DMCompositeAddDM(user.dmpgrid, user.dmnet));

  /* Read initial voltage vector and Ybus */
  PetscCall(PetscViewerBinaryOpen(PETSC_COMM_WORLD, "X.bin", FILE_MODE_READ, &Xview));
  PetscCall(PetscViewerBinaryOpen(PETSC_COMM_WORLD, "Ybus.bin", FILE_MODE_READ, &Ybusview));

  PetscCall(VecCreate(PETSC_COMM_WORLD, &user.V0));
  PetscCall(VecSetSizes(user.V0, PETSC_DECIDE, user.neqs_net));
  PetscCall(VecLoad(user.V0, Xview));

  PetscCall(MatCreate(PETSC_COMM_WORLD, &user.Ybus));
  PetscCall(MatSetSizes(user.Ybus, PETSC_DECIDE, PETSC_DECIDE, user.neqs_net, user.neqs_net));
  PetscCall(MatSetType(user.Ybus, MATBAIJ));
  /*  PetscCall(MatSetBlockSize(ctx->Ybus,2)); */
  PetscCall(MatLoad(user.Ybus, Ybusview));

  PetscCall(PetscViewerDestroy(&Xview));
  PetscCall(PetscViewerDestroy(&Ybusview));

  /* Allocate space for Jacobians */
  PetscCall(MatCreate(PETSC_COMM_WORLD, &user.J));
  PetscCall(MatSetSizes(user.J, PETSC_DECIDE, PETSC_DECIDE, user.neqs_pgrid, user.neqs_pgrid));
  PetscCall(MatSetFromOptions(user.J));
  PetscCall(PreallocateJacobian(user.J, &user));

  PetscCall(MatCreate(PETSC_COMM_WORLD, &user.Jacp));
  PetscCall(MatSetSizes(user.Jacp, PETSC_DECIDE, PETSC_DECIDE, user.neqs_pgrid, 3));
  PetscCall(MatSetFromOptions(user.Jacp));
  PetscCall(MatSetUp(user.Jacp));
  PetscCall(MatZeroEntries(user.Jacp)); /* initialize to zeros */

  PetscCall(MatCreateDense(PETSC_COMM_WORLD, PETSC_DECIDE, PETSC_DECIDE, 3, 1, NULL, &user.DRDP));
  PetscCall(MatSetUp(user.DRDP));
  PetscCall(MatCreateDense(PETSC_COMM_WORLD, PETSC_DECIDE, PETSC_DECIDE, user.neqs_pgrid, 1, NULL, &user.DRDU));
  PetscCall(MatSetUp(user.DRDU));

  /* Create TAO solver and set desired solution method */
  PetscCall(TaoCreate(PETSC_COMM_WORLD, &tao));
  PetscCall(TaoSetType(tao, TAOBLMVM));
  /*
     Optimization starts
  */
  /* Set initial solution guess */
  PetscCall(VecCreateSeq(PETSC_COMM_WORLD, 3, &p));
  PetscCall(VecGetArray(p, &x_ptr));
  x_ptr[0] = PG[0];
  x_ptr[1] = PG[1];
  x_ptr[2] = PG[2];
  PetscCall(VecRestoreArray(p, &x_ptr));

  PetscCall(TaoSetSolution(tao, p));
  /* Set routine for function and gradient evaluation */
  PetscCall(TaoSetObjectiveAndGradient(tao, NULL, FormFunctionGradient, &user));

  /* Set bounds for the optimization */
  PetscCall(VecDuplicate(p, &lowerb));
  PetscCall(VecDuplicate(p, &upperb));
  PetscCall(VecGetArray(lowerb, &x_ptr));
  x_ptr[0] = 0.5;
  x_ptr[1] = 0.5;
  x_ptr[2] = 0.5;
  PetscCall(VecRestoreArray(lowerb, &x_ptr));
  PetscCall(VecGetArray(upperb, &x_ptr));
  x_ptr[0] = 2.0;
  x_ptr[1] = 2.0;
  x_ptr[2] = 2.0;
  PetscCall(VecRestoreArray(upperb, &x_ptr));
  PetscCall(TaoSetVariableBounds(tao, lowerb, upperb));

  /* Check for any TAO command line options */
  PetscCall(TaoSetFromOptions(tao));
  PetscCall(TaoGetKSP(tao, &ksp));
  if (ksp) {
    PetscCall(KSPGetPC(ksp, &pc));
    PetscCall(PCSetType(pc, PCNONE));
  }

  /* SOLVE THE APPLICATION */
  PetscCall(TaoSolve(tao));

  PetscCall(VecView(p, PETSC_VIEWER_STDOUT_WORLD));
  /* Free TAO data structures */
  PetscCall(TaoDestroy(&tao));

  PetscCall(DMDestroy(&user.dmgen));
  PetscCall(DMDestroy(&user.dmnet));
  PetscCall(DMDestroy(&user.dmpgrid));
  PetscCall(ISDestroy(&user.is_diff));
  PetscCall(ISDestroy(&user.is_alg));

  PetscCall(MatDestroy(&user.J));
  PetscCall(MatDestroy(&user.Jacp));
  PetscCall(MatDestroy(&user.Ybus));
  /* PetscCall(MatDestroy(&user.Sol)); */
  PetscCall(VecDestroy(&user.V0));
  PetscCall(VecDestroy(&p));
  PetscCall(VecDestroy(&lowerb));
  PetscCall(VecDestroy(&upperb));
  PetscCall(MatDestroy(&user.DRDU));
  PetscCall(MatDestroy(&user.DRDP));
  PetscCall(PetscFinalize());
  return 0;
}

/* ------------------------------------------------------------------ */
/*
   FormFunction - Evaluates the function and corresponding gradient.

   Input Parameters:
   tao - the Tao context
   X   - the input vector
   ptr - optional user-defined context, as set by TaoSetObjectiveAndGradient()

   Output Parameters:
   f   - the newly evaluated function
   G   - the newly evaluated gradient
*/
PetscErrorCode FormFunctionGradient(Tao tao, Vec P, PetscReal *f, Vec G, void *ctx0)
{
  TS       ts, quadts;
  SNES     snes_alg;
  Userctx *ctx = (Userctx *)ctx0;
  Vec      X;
  PetscInt i;
  /* sensitivity context */
  PetscScalar *x_ptr;
  Vec          lambda[1], q;
  Vec          mu[1];
  PetscInt     steps1, steps2, steps3;
  Vec          DICDP[3];
  Vec          F_alg;
  PetscInt     row_loc, col_loc;
  PetscScalar  val;
  Vec          Xdot;

  PetscFunctionBegin;
  PetscCall(VecGetArrayRead(P, (const PetscScalar **)&x_ptr));
  PG[0] = x_ptr[0];
  PG[1] = x_ptr[1];
  PG[2] = x_ptr[2];
  PetscCall(VecRestoreArrayRead(P, (const PetscScalar **)&x_ptr));

  ctx->stepnum = 0;

  PetscCall(DMCreateGlobalVector(ctx->dmpgrid, &X));

  /* Create matrix to save solutions at each time step */
  /* PetscCall(MatCreateSeqDense(PETSC_COMM_SELF,ctx->neqs_pgrid+1,1002,NULL,&ctx->Sol)); */
  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Create timestepping solver context
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  PetscCall(TSCreate(PETSC_COMM_WORLD, &ts));
  PetscCall(TSSetProblemType(ts, TS_NONLINEAR));
  PetscCall(TSSetType(ts, TSCN));
  PetscCall(TSSetIFunction(ts, NULL, (TSIFunction)IFunction, ctx));
  PetscCall(TSSetIJacobian(ts, ctx->J, ctx->J, (TSIJacobian)IJacobian, ctx));
  PetscCall(TSSetApplicationContext(ts, ctx));
  /*   Set RHS JacobianP */
  PetscCall(TSSetRHSJacobianP(ts, ctx->Jacp, RHSJacobianP, ctx));

  PetscCall(TSCreateQuadratureTS(ts, PETSC_FALSE, &quadts));
  PetscCall(TSSetRHSFunction(quadts, NULL, (TSRHSFunction)CostIntegrand, ctx));
  PetscCall(TSSetRHSJacobian(quadts, ctx->DRDU, ctx->DRDU, (TSRHSJacobian)DRDUJacobianTranspose, ctx));
  PetscCall(TSSetRHSJacobianP(quadts, ctx->DRDP, (TSRHSJacobianP)DRDPJacobianTranspose, ctx));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Set initial conditions
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  PetscCall(SetInitialGuess(X, ctx));

  /* Approximate DICDP with finite difference, we want to zero out network variables */
  for (i = 0; i < 3; i++) PetscCall(VecDuplicate(X, &DICDP[i]));
  PetscCall(DICDPFiniteDifference(X, DICDP, ctx));

  PetscCall(VecDuplicate(X, &F_alg));
  PetscCall(SNESCreate(PETSC_COMM_WORLD, &snes_alg));
  PetscCall(SNESSetFunction(snes_alg, F_alg, AlgFunction, ctx));
  PetscCall(MatZeroEntries(ctx->J));
  PetscCall(SNESSetJacobian(snes_alg, ctx->J, ctx->J, AlgJacobian, ctx));
  PetscCall(SNESSetOptionsPrefix(snes_alg, "alg_"));
  PetscCall(SNESSetFromOptions(snes_alg));
  ctx->alg_flg = PETSC_TRUE;
  /* Solve the algebraic equations */
  PetscCall(SNESSolve(snes_alg, NULL, X));

  /* Just to set up the Jacobian structure */
  PetscCall(VecDuplicate(X, &Xdot));
  PetscCall(IJacobian(ts, 0.0, X, Xdot, 0.0, ctx->J, ctx->J, ctx));
  PetscCall(VecDestroy(&Xdot));

  ctx->stepnum++;

  /*
    Save trajectory of solution so that TSAdjointSolve() may be used
  */
  PetscCall(TSSetSaveTrajectory(ts));

  PetscCall(TSSetTimeStep(ts, 0.01));
  PetscCall(TSSetExactFinalTime(ts, TS_EXACTFINALTIME_MATCHSTEP));
  PetscCall(TSSetFromOptions(ts));
  /* PetscCall(TSSetPostStep(ts,SaveSolution)); */

  /* Prefault period */
  ctx->alg_flg = PETSC_FALSE;
  PetscCall(TSSetTime(ts, 0.0));
  PetscCall(TSSetMaxTime(ts, ctx->tfaulton));
  PetscCall(TSSolve(ts, X));
  PetscCall(TSGetStepNumber(ts, &steps1));

  /* Create the nonlinear solver for solving the algebraic system */
  /* Note that although the algebraic system needs to be solved only for
     Idq and V, we reuse the entire system including xgen. The xgen
     variables are held constant by setting their residuals to 0 and
     putting a 1 on the Jacobian diagonal for xgen rows
  */
  PetscCall(MatZeroEntries(ctx->J));

  /* Apply disturbance - resistive fault at ctx->faultbus */
  /* This is done by adding shunt conductance to the diagonal location
     in the Ybus matrix */
  row_loc = 2 * ctx->faultbus;
  col_loc = 2 * ctx->faultbus + 1; /* Location for G */
  val     = 1 / ctx->Rfault;
  PetscCall(MatSetValues(ctx->Ybus, 1, &row_loc, 1, &col_loc, &val, ADD_VALUES));
  row_loc = 2 * ctx->faultbus + 1;
  col_loc = 2 * ctx->faultbus; /* Location for G */
  val     = 1 / ctx->Rfault;
  PetscCall(MatSetValues(ctx->Ybus, 1, &row_loc, 1, &col_loc, &val, ADD_VALUES));

  PetscCall(MatAssemblyBegin(ctx->Ybus, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(ctx->Ybus, MAT_FINAL_ASSEMBLY));

  ctx->alg_flg = PETSC_TRUE;
  /* Solve the algebraic equations */
  PetscCall(SNESSolve(snes_alg, NULL, X));

  ctx->stepnum++;

  /* Disturbance period */
  ctx->alg_flg = PETSC_FALSE;
  PetscCall(TSSetTime(ts, ctx->tfaulton));
  PetscCall(TSSetMaxTime(ts, ctx->tfaultoff));
  PetscCall(TSSolve(ts, X));
  PetscCall(TSGetStepNumber(ts, &steps2));
  steps2 -= steps1;

  /* Remove the fault */
  row_loc = 2 * ctx->faultbus;
  col_loc = 2 * ctx->faultbus + 1;
  val     = -1 / ctx->Rfault;
  PetscCall(MatSetValues(ctx->Ybus, 1, &row_loc, 1, &col_loc, &val, ADD_VALUES));
  row_loc = 2 * ctx->faultbus + 1;
  col_loc = 2 * ctx->faultbus;
  val     = -1 / ctx->Rfault;
  PetscCall(MatSetValues(ctx->Ybus, 1, &row_loc, 1, &col_loc, &val, ADD_VALUES));

  PetscCall(MatAssemblyBegin(ctx->Ybus, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(ctx->Ybus, MAT_FINAL_ASSEMBLY));

  PetscCall(MatZeroEntries(ctx->J));

  ctx->alg_flg = PETSC_TRUE;

  /* Solve the algebraic equations */
  PetscCall(SNESSolve(snes_alg, NULL, X));

  ctx->stepnum++;

  /* Post-disturbance period */
  ctx->alg_flg = PETSC_TRUE;
  PetscCall(TSSetTime(ts, ctx->tfaultoff));
  PetscCall(TSSetMaxTime(ts, ctx->tmax));
  PetscCall(TSSolve(ts, X));
  PetscCall(TSGetStepNumber(ts, &steps3));
  steps3 -= steps2;
  steps3 -= steps1;

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Adjoint model starts here
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  PetscCall(TSSetPostStep(ts, NULL));
  PetscCall(MatCreateVecs(ctx->J, &lambda[0], NULL));
  /*   Set initial conditions for the adjoint integration */
  PetscCall(VecZeroEntries(lambda[0]));

  PetscCall(MatCreateVecs(ctx->Jacp, &mu[0], NULL));
  PetscCall(VecZeroEntries(mu[0]));
  PetscCall(TSSetCostGradients(ts, 1, lambda, mu));

  PetscCall(TSAdjointSetSteps(ts, steps3));
  PetscCall(TSAdjointSolve(ts));

  PetscCall(MatZeroEntries(ctx->J));
  /* Applying disturbance - resistive fault at ctx->faultbus */
  /* This is done by deducting shunt conductance to the diagonal location
     in the Ybus matrix */
  row_loc = 2 * ctx->faultbus;
  col_loc = 2 * ctx->faultbus + 1; /* Location for G */
  val     = 1. / ctx->Rfault;
  PetscCall(MatSetValues(ctx->Ybus, 1, &row_loc, 1, &col_loc, &val, ADD_VALUES));
  row_loc = 2 * ctx->faultbus + 1;
  col_loc = 2 * ctx->faultbus; /* Location for G */
  val     = 1. / ctx->Rfault;
  PetscCall(MatSetValues(ctx->Ybus, 1, &row_loc, 1, &col_loc, &val, ADD_VALUES));

  PetscCall(MatAssemblyBegin(ctx->Ybus, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(ctx->Ybus, MAT_FINAL_ASSEMBLY));

  /*   Set number of steps for the adjoint integration */
  PetscCall(TSAdjointSetSteps(ts, steps2));
  PetscCall(TSAdjointSolve(ts));

  PetscCall(MatZeroEntries(ctx->J));
  /* remove the fault */
  row_loc = 2 * ctx->faultbus;
  col_loc = 2 * ctx->faultbus + 1; /* Location for G */
  val     = -1. / ctx->Rfault;
  PetscCall(MatSetValues(ctx->Ybus, 1, &row_loc, 1, &col_loc, &val, ADD_VALUES));
  row_loc = 2 * ctx->faultbus + 1;
  col_loc = 2 * ctx->faultbus; /* Location for G */
  val     = -1. / ctx->Rfault;
  PetscCall(MatSetValues(ctx->Ybus, 1, &row_loc, 1, &col_loc, &val, ADD_VALUES));

  PetscCall(MatAssemblyBegin(ctx->Ybus, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(ctx->Ybus, MAT_FINAL_ASSEMBLY));

  /*   Set number of steps for the adjoint integration */
  PetscCall(TSAdjointSetSteps(ts, steps1));
  PetscCall(TSAdjointSolve(ts));

  PetscCall(ComputeSensiP(lambda[0], mu[0], DICDP, ctx));
  PetscCall(VecCopy(mu[0], G));

  PetscCall(TSGetQuadratureTS(ts, NULL, &quadts));
  PetscCall(TSGetSolution(quadts, &q));
  PetscCall(VecGetArray(q, &x_ptr));
  *f       = x_ptr[0];
  x_ptr[0] = 0;
  PetscCall(VecRestoreArray(q, &x_ptr));

  PetscCall(VecDestroy(&lambda[0]));
  PetscCall(VecDestroy(&mu[0]));

  PetscCall(SNESDestroy(&snes_alg));
  PetscCall(VecDestroy(&F_alg));
  PetscCall(VecDestroy(&X));
  PetscCall(TSDestroy(&ts));
  for (i = 0; i < 3; i++) PetscCall(VecDestroy(&DICDP[i]));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*TEST

   build:
      requires: double !complex !defined(PETSC_USE_64BIT_INDICES)

   test:
      args: -viewer_binary_skip_info -tao_monitor -tao_gttol .2
      localrunfiles: petscoptions X.bin Ybus.bin

TEST*/
