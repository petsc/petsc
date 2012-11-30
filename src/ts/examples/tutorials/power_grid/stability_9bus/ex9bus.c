
static char help[] = "Power grid stability analysis of WECC 9 bus system.\n\
This example is based on the 9-bus (node) example given in the book Power\n\
Systems Dynamics and Stability (Chapter 7) by P. Sauer and M. A. Pai.\n\
The power grid in this example consists of 9 buses (nodes), 3 generators,\n\
3 loads, and 9 transmission lines. The network equations are written\n\
in current balance form using rectangular coordiantes.\n\n";

/*
   The equations for the stability analysis are described by the DAE
   
   \dot{x} = f(x,y,t)
     0     = g(x,y,t)

   where the generators are described by differential equations, while the algebraic
   constraints define the network equations.

   The generators are modeled with a 4th order differential equation describing the electrical
   and mechanical dynamics. Each generator also has an exciter system modeled by 3rd order
   diff. eqns. describing the exciter, voltage regulator, and the feedback stabilizer
   mechanism.

   The network equations are described by nodal current balance equations.
    I(x,y) - Y*V = 0

   where:
    I(x,y) is the current injected from generators and loads.
      Y    is the admittance matrix, and
      V    is the voltage vector
*/

/*
   Include "petscts.h" so that we can use TS solvers.  Note that this
   file automatically includes:
     petscsys.h       - base PETSc routines   petscvec.h - vectors
     petscmat.h - matrices
     petscis.h     - index sets            petscksp.h - Krylov subspace methods
     petscviewer.h - viewers               petscpc.h  - preconditioners
     petscksp.h   - linear solvers
*/
#include <petscts.h>
#include <petscdmda.h>
#include <petscdmcomposite.h>

#define freq 60
#define w_s (2*PETSC_PI*freq)

/* Sizes and indices */
const PetscInt nbus = 9; /* Number of network buses */
const PetscInt ngen = 3; /* Number of generators */
const PetscInt nload = 3; /* Number of loads */
const PetscInt gbus[3] = {0,1,2}; /* Buses at which generators are incident */
const PetscInt lbus[3] = {4,5,7}; /* Buses at which loads are incident */

/* Generator real and reactive powers (found via loadflow) */
const PetscScalar PG[3] = {0.716786142395021,1.630000000000000,0.850000000000000};
const PetscScalar QG[3] = {0.270702180178785,0.066120127797275,-0.108402221791588};
/* Generator constants */
const PetscScalar H[3]   = {23.64,6.4,3.01}; /* Inertia constant */
const PetscScalar Rs[3] = {0.0,0.0,0.0}; /* Stator Resistance */
const PetscScalar Xd[3]  = {0.146,0.8958,1.3125}; /* d-axis reactance */
const PetscScalar Xdp[3] = {0.0608,0.1198,0.1813}; /* d-axis transient reactance */
const PetscScalar Xq[3]  = {0.4360,0.8645,1.2578}; /* q-axis reactance Xq(1) set to 0.4360, value given in text 0.0969 */   
const PetscScalar Xqp[3] = {0.0969,0.1969,0.25}; /* q-axis transient reactance */
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

typedef struct{
  DM dmgen, dmnet; /* DMs to manage generator and network subsystem */
  DM dmpgrid;        /* Composite DM to manage the entire power grid */
  Mat Ybus;        /* Network admittance matrix */
  Vec V0;          /* Initial voltage vector (Power flow solution) */
  PetscReal tfaulton,tfaultoff; /* Fault on and off times */
  PetscInt  faultbus;  /* Fault bus */
  PetscReal t0,tmax;
}Userctx;


/* Converts from machine frame (dq) to network (phase a real,imag) reference frame */
#undef __FUNCT__
#define __FUNCT__ "dq2ri"
PetscErrorCode dq2ri(PetscScalar Fd,PetscScalar Fq,PetscScalar delta,PetscScalar *Fr, PetscScalar *Fi)
{
  PetscFunctionBegin;
  *Fr =  Fd*sin(delta) + Fq*cos(delta);
  *Fi = -Fd*cos(delta) + Fq*sin(delta);
  PetscFunctionReturn(0);
}

/* Converts from network frame ([phase a real,imag) to machine (dq) reference frame */
#undef __FUNCT__
#define __FUNCT__ "ri2dq"
PetscErrorCode ri2dq(PetscScalar Fr,PetscScalar Fi,PetscScalar delta,PetscScalar *Fd, PetscScalar *Fq)
{
  PetscFunctionBegin;
  *Fd =  Fr*sin(delta) - Fi*cos(delta);
  *Fq =  Fr*cos(delta) + Fi*sin(delta);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SetInitialGuess"
PetscErrorCode SetInitialGuess(Vec X,Userctx* user)
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

  for(i=0; i < ngen; i++) {
    Vr = xnet[2*gbus[i]]; /* Real part of generator terminal voltage */
    Vi = xnet[2*gbus[i]+1]; /* Imaginary part of the generator terminal voltage */
    Vm = PetscSqrtScalar(Vr*Vr + Vi*Vi); Vm2 = Vm*Vm;
    IGr = (Vr*PG[i] + Vi*QG[i])/Vm2;
    IGi = (Vi*PG[i] - Vr*QG[i])/Vm2;

    delta = atan2(Vi+Xq[i]*IGr,Vr-Xq[i]*IGi); /* Machine angle */

    theta = PETSC_PI/2.0 - delta;

    Id = IGr*cos(theta) - IGi*sin(theta); /* d-axis stator current */
    Iq = IGr*sin(theta) + IGi*cos(theta); /* q-axis stator current */

    Vd = Vr*cos(theta) - Vi*sin(theta);
    Vq = Vr*sin(theta) + Vi*cos(theta);

    Edp = Vd + Rs[i]*Id - Xqp[i]*Iq; /* d-axis transient EMF */
    Eqp = Vq + Rs[i]*Iq + Xdp[i]*Id; /* q-axis transient EMF */

    TM[i] = PG[i];

    /* The generator variables are ordered as [Eqp,Edp,delta,w,Id,Iq] */
    xgen[idx]   = Eqp;
    xgen[idx+1] = Edp;
    xgen[idx+2] = delta;
    xgen[idx+3] = w_s;

    idx = idx + 4;

    xgen[idx] = Id;
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
  ierr = DMCompositeGather(user->dmpgrid,X,INSERT_VALUES,Xgen,Xnet);CHKERRQ(ierr);
  ierr = DMCompositeRestoreLocalVectors(user->dmpgrid,&Xgen,&Xnet);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "IFunction"
PetscErrorCode IFunction(TS ts,PetscReal t,Vec X, Vec Xdot, Vec F, Userctx* user)
{
  PetscErrorCode ierr;
  Vec            Xgen,Xnet,Xdotgen,Xdotnet,Fgen,Fnet;
  PetscScalar    *xgen,*xnet,*xdotgen,*xdotnet,*fgen,*fnet;
  PetscInt       i,idx=0;
  PetscScalar    Vr,Vi,Vm,Vm2;
  PetscScalar    Eqp,Edp,delta,w; /* Generator variables */
  PetscScalar    Efd,RF,VR; /* Exciter variables */
  PetscScalar    Id,Iq;  /* Generator dq axis currents */
  PetscScalar    Vd,Vq,SE;
  PetscScalar    IGr,IGi,IDr,IDi;

  PetscFunctionBegin;

  ierr = VecZeroEntries(F);CHKERRQ(ierr);
  ierr = DMCompositeGetLocalVectors(user->dmpgrid,&Xgen,&Xnet);CHKERRQ(ierr);
  ierr = DMCompositeGetLocalVectors(user->dmpgrid,&Xdotgen,&Xdotnet);CHKERRQ(ierr);
  ierr = DMCompositeGetLocalVectors(user->dmpgrid,&Fgen,&Fnet);CHKERRQ(ierr);
  ierr = DMCompositeScatter(user->dmpgrid,X,Xgen,Xnet);CHKERRQ(ierr);
  ierr = DMCompositeScatter(user->dmpgrid,Xdot,Xdotgen,Xdotnet);CHKERRQ(ierr);

  /* Network current balance residual Xdotnet - IG + Y*V + IL = 0. Only Xdotnet + YV is added here. 
     The generator current injection, IG, and load current injection, ID are added later
  */
  /* Note that the values in Ybus are stored assuming the imaginary current balance
     equation is ordered first followed by real current balance equation for each bus.
     Thus imaginary current contribution goes in location 2*i, and
     real current contribution in 2*i+1
  */
  ierr = MatMultAdd(user->Ybus,Xnet,Xdotnet,Fnet);

  ierr = VecGetArray(Xgen,&xgen);CHKERRQ(ierr);
  ierr = VecGetArray(Xnet,&xnet);CHKERRQ(ierr);
  ierr = VecGetArray(Xdotgen,&xdotgen);CHKERRQ(ierr);
  ierr = VecGetArray(Xdotnet,&xdotnet);CHKERRQ(ierr);
  ierr = VecGetArray(Fgen,&fgen);CHKERRQ(ierr);
  ierr = VecGetArray(Fnet,&fnet);CHKERRQ(ierr);
  

  /* Generator subsystem */
  for(i=0; i < ngen; i++) {
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
    fgen[idx]   = Td0p[i]*xdotgen[idx] + Eqp + (Xd[i] - Xdp[i])*Id - Efd;
    fgen[idx+1] = Tq0p[i]*xdotgen[idx+1] - Edp + (Xq[i] - Xqp[i])*Iq;
    fgen[idx+2] = xdotgen[idx+2] - w + w_s;
    fgen[idx+3] = M[i]*xdotgen[idx+3] - TM[i] + Edp*Id + Eqp*Iq + (Xqp[i] - Xdp[i])*Id*Iq + D[i]*(w - w_s);

    Vr = xnet[2*gbus[i]]; /* Real part of generator terminal voltage */
    Vi = xnet[2*gbus[i]+1]; /* Imaginary part of the generator terminal voltage */

    ierr = ri2dq(Vr,Vi,delta,&Vd,&Vq);CHKERRQ(ierr);
    /* Algebraic equations for stator currents */
    //    fgen[idx+4] = xdotgen[idx+4] - Edp + Vd + Rs[i]*Id - Xqp[i]*Iq;
    // fgen[idx+5] = xdotgen[idx+5] - Eqp + Vq + Rs[i]*Iq + Xdp[i]*Id;
    PetscScalar Zdq_inv[4],det;
    det = Rs[i]*Rs[i] + Xdp[i]*Xqp[i];
    Zdq_inv[0] = Rs[i]/det;
    Zdq_inv[1] = Xqp[i]/det;
    Zdq_inv[2] = -Xdp[i]/det;
    Zdq_inv[3] = Rs[i]/det;

    fgen[idx+4] = xdotgen[idx+4] + Zdq_inv[0]*(-Edp + Vd) + Zdq_inv[1]*(-Eqp + Vq) + Id;
    fgen[idx+5] = xdotgen[idx+5] + Zdq_inv[2]*(-Edp + Vd) + Zdq_inv[3]*(-Eqp + Vq) + Iq;

    /* Add generator current injection to network */
    ierr = dq2ri(Id,Iq,delta,&IGr,&IGi);CHKERRQ(ierr);
    fnet[2*gbus[i]]   -= IGi;
    fnet[2*gbus[i]+1] -= IGr;

    Vm = PetscSqrtScalar(Vr*Vr + Vi*Vi); Vm2 = Vm*Vm;

    SE  = k1[i]*PetscExpScalar(k2[i]*Efd);

    /* Exciter differential equations */
    fgen[idx+6] = TE[i]*xdotgen[idx+6] + KE[i]*Efd + SE - VR;
    fgen[idx+7] = TF[i]*xdotgen[idx+7] + RF - KF[i]*Efd/TF[i];
    fgen[idx+8] = TA[i]*xdotgen[idx+8] + VR - KA[i]*RF + KA[i]*KF[i]*Efd/TF[i] - KA[i]*(Vref[i] - Vm);

    idx = idx + 9;
  }

  PetscScalar PD,QD,Vm0,*v0;
  PetscInt    k;
  ierr = VecGetArray(user->V0,&v0);CHKERRQ(ierr);
  for(i=0; i < nload; i++) {
    Vr = xnet[2*lbus[i]]; /* Real part of load bus voltage */
    Vi = xnet[2*lbus[i]+1]; /* Imaginary part of the load bus voltage */
    Vm = PetscSqrtScalar(Vr*Vr + Vi*Vi); Vm2 = Vm*Vm;
    Vm0 = PetscSqrtScalar(v0[2*lbus[i]]*v0[2*lbus[i]] + v0[2*lbus[i]+1]*v0[2*lbus[i]+1]);
    PD = QD = 0.0;
    for(k=0; k < ld_nsegsp[i];k++) PD += ld_alphap[k]*PD0[i]*PetscPowScalar((Vm/Vm0),ld_betap[k]);
    for(k=0; k < ld_nsegsq[i];k++) QD += ld_alphaq[k]*QD0[i]*PetscPowScalar((Vm/Vm0),ld_betaq[k]);
    
    /* Load currents */
    IDr = (PD*Vr + QD*Vi)/Vm2;
    IDi = (-QD*Vr + PD*Vi)/Vm2;
    
    fnet[2*lbus[i]]   += IDi;
    fnet[2*lbus[i]+1] += IDr;
  }
  ierr = VecRestoreArray(user->V0,&v0);CHKERRQ(ierr);

  ierr = VecRestoreArray(Xgen,&xgen);CHKERRQ(ierr);
  ierr = VecRestoreArray(Xnet,&xnet);CHKERRQ(ierr);
  ierr = VecRestoreArray(Xdotgen,&xdotgen);CHKERRQ(ierr);
  ierr = VecRestoreArray(Xdotnet,&xdotnet);CHKERRQ(ierr);
  ierr = VecRestoreArray(Fgen,&fgen);CHKERRQ(ierr);
  ierr = VecRestoreArray(Fnet,&fnet);CHKERRQ(ierr);

  ierr = DMCompositeGather(user->dmpgrid,F,INSERT_VALUES,Fgen,Fnet);CHKERRQ(ierr);
  ierr = DMCompositeRestoreLocalVectors(user->dmpgrid,&Xgen,&Xnet);CHKERRQ(ierr);
  ierr = DMCompositeRestoreLocalVectors(user->dmpgrid,&Xdotgen,&Xdotnet);CHKERRQ(ierr);
  ierr = DMCompositeRestoreLocalVectors(user->dmpgrid,&Fgen,&Fnet);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc,char **argv)
{
  TS             ts;
  PetscErrorCode ierr;
  PetscMPIInt    size;
  PetscInt       neqs_gen,neqs_net,neqs_pgrid;
  Userctx        user;
  PetscViewer    Xview,Ybusview;
  Vec            X;
  Mat            J;

  ierr = PetscInitialize(&argc,&argv,(char*)0,help);CHKERRQ(ierr);
  ierr = MPI_Comm_size(PETSC_COMM_WORLD,&size);CHKERRQ(ierr);
  if (size > 1) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_SUP,"Only for sequential runs");

  neqs_gen   = 9*ngen; /* # eqs. for generator subsystem */
  neqs_net   = 2*nbus; /* # eqs. for network subsystem   */ 
  neqs_pgrid = neqs_gen + neqs_net; 

  /* Read initial voltage vector and Ybus */
  ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,"X.bin",FILE_MODE_READ,&Xview);CHKERRQ(ierr);
  ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,"Ybus.bin",FILE_MODE_READ,&Ybusview);CHKERRQ(ierr);

  ierr = VecCreate(PETSC_COMM_WORLD,&user.V0);CHKERRQ(ierr);
  ierr = VecSetSizes(user.V0,PETSC_DECIDE,neqs_net);CHKERRQ(ierr);
  ierr = VecLoad(user.V0,Xview);CHKERRQ(ierr);

  ierr = MatCreate(PETSC_COMM_WORLD,&user.Ybus);CHKERRQ(ierr);
  ierr = MatSetSizes(user.Ybus,PETSC_DECIDE,PETSC_DECIDE,neqs_net,neqs_net);CHKERRQ(ierr);
  ierr = MatSetType(user.Ybus,MATBAIJ);CHKERRQ(ierr);
  /*  ierr = MatSetBlockSize(user.Ybus,2);CHKERRQ(ierr); */
  ierr = MatLoad(user.Ybus,Ybusview);CHKERRQ(ierr);

  /* Set run time options */
  ierr = PetscOptionsBegin(PETSC_COMM_WORLD,PETSC_NULL,"Transient stability fault options","");CHKERRQ(ierr);
  {
    user.tfaulton = 0.1;
    user.tfaultoff = 0.2;
    user.faultbus = 4;
    ierr  = PetscOptionsReal("-tfaulton","","",user.tfaulton,&user.tfaulton,PETSC_NULL);CHKERRQ(ierr);
    ierr  = PetscOptionsReal("-tfaultoff","","",user.tfaulton,&user.tfaulton,PETSC_NULL);CHKERRQ(ierr);
    ierr  = PetscOptionsInt("-faultbus","","",user.faultbus,&user.faultbus,PETSC_NULL);CHKERRQ(ierr);
    user.t0 = 0.0;
    user.tmax = 10.0;
    ierr  = PetscOptionsReal("-t0","","",user.t0,&user.t0,PETSC_NULL);CHKERRQ(ierr);
    ierr  = PetscOptionsReal("-tmax","","",user.tmax,&user.tmax,PETSC_NULL);CHKERRQ(ierr);
  }
  ierr = PetscOptionsEnd();CHKERRQ(ierr);

  ierr = PetscViewerDestroy(&Xview);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&Ybusview);CHKERRQ(ierr);

  /* Create DMs for generator and network subsystems */
  ierr = DMDACreate1d(PETSC_COMM_WORLD,DMDA_BOUNDARY_NONE,neqs_gen,1,1,PETSC_NULL,&user.dmgen);CHKERRQ(ierr);
  ierr = DMSetOptionsPrefix(user.dmgen,"dmgen_");CHKERRQ(ierr);
  ierr = DMDACreate1d(PETSC_COMM_WORLD,DMDA_BOUNDARY_NONE,neqs_net,1,1,PETSC_NULL,&user.dmnet);CHKERRQ(ierr);
  ierr = DMSetOptionsPrefix(user.dmnet,"dmnet_");CHKERRQ(ierr);
  /* Create a composite DM packer and add the two DMs */
  ierr = DMCompositeCreate(PETSC_COMM_WORLD,&user.dmpgrid);CHKERRQ(ierr);
  ierr = DMSetOptionsPrefix(user.dmpgrid,"pgrid_");CHKERRQ(ierr);
  ierr = DMCompositeAddDM(user.dmpgrid,user.dmgen);CHKERRQ(ierr);
  ierr = DMCompositeAddDM(user.dmpgrid,user.dmnet);CHKERRQ(ierr);

  ierr = DMCreateGlobalVector(user.dmpgrid,&X);CHKERRQ(ierr);

  ierr = MatCreate(PETSC_COMM_WORLD,&J);CHKERRQ(ierr);
  ierr = MatSetSizes(J,PETSC_DECIDE,PETSC_DECIDE,neqs_pgrid,neqs_pgrid);CHKERRQ(ierr);
  ierr = MatSetFromOptions(J);CHKERRQ(ierr);
  ierr = MatSetUp(J);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Create timestepping solver context
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = TSCreate(PETSC_COMM_WORLD,&ts);CHKERRQ(ierr);
  ierr = TSSetProblemType(ts,TS_NONLINEAR);CHKERRQ(ierr);
  ierr = TSSetType(ts,TSROSW);CHKERRQ(ierr);
  ierr = TSSetIFunction(ts,PETSC_NULL,(TSIFunction) IFunction,&user);CHKERRQ(ierr);
  /* ierr = TSSetIJacobian(ts,J,J,(TSIJacobian)IJacobian,&user);CHKERRQ(ierr); */

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Set initial conditions
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = SetInitialGuess(X,&user);CHKERRQ(ierr);
  ierr = TSSetSolution(ts,X);CHKERRQ(ierr);

  ierr = TSSetDuration(ts,1000,user.tmax);CHKERRQ(ierr);
  ierr = TSSetInitialTimeStep(ts,0.0,0.01);CHKERRQ(ierr);
  ierr = TSSetFromOptions(ts);CHKERRQ(ierr);

  /* Finite difference Jacobian */
  SNES snes;
  ierr = TSGetSNES(ts,&snes);CHKERRQ(ierr);
  ierr = SNESSetJacobian(snes,J,J,SNESDefaultComputeJacobian,ts);CHKERRQ(ierr);

  /* Prefault period */
  ierr = TSSolve(ts,X);CHKERRQ(ierr);

  /* Fault-on period */
  /*  ierr = TSSetDuration(ts,1000,user.tfaultoff);CHKERRQ(ierr);
  ierr = TSSetInitialTimeStep(ts,user.tfaulton,.01);CHKERRQ(ierr);
  ierr = TSSolve(ts,X);CHKERRQ(ierr);
  */
  /* Post-fault period */
  /*
  ierr = TSSetDuration(ts,1000,user.tmax);CHKERRQ(ierr);
  ierr = TSSetInitialTimeStep(ts,user.tfaultoff,.01);CHKERRQ(ierr);
  ierr = TSSolve(ts,X);CHKERRQ(ierr);
  */
  ierr = PetscFinalize();
  return(0);
}
