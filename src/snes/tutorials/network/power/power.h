#ifndef PF_H
#define PF_H

#include <petscsnes.h>
#include <petscdmnetwork.h>

#define MAXLINE 1000
#define REF_BUS 3
#define PV_BUS 2
#define PQ_BUS 1
#define ISOLATED_BUS 4
#define NGEN_AT_BUS_MAX 15
#define NLOAD_AT_BUS_MAX 1

struct _p_UserCtx_Power{
  PetscScalar  Sbase;
  PetscBool    jac_error; /* introduce error in the jacobian */
  PetscInt     compkey_branch;
  PetscInt     compkey_bus;
  PetscInt     compkey_gen;
  PetscInt     compkey_load;
} PETSC_ATTRIBUTEALIGNED(PetscMax(sizeof(double),sizeof(PetscScalar)));

typedef struct _p_UserCtx_Power UserCtx_Power;

/* 2. Bus data */
/* 11 columns */
struct _p_VERTEX_Power{
  PetscInt      bus_i; /* Integer bus number .. used by some formats like Matpower */
  char          i[20]; /* Bus Number */
  char          name[20]; /* Bus Name */
  PetscScalar   basekV; /* Bus Base kV */
  PetscInt      ide; /* Bus type code */
  PetscScalar   gl; /* Active component of shunt admittance to ground */
  PetscScalar   bl; /* Reactive component of shunt admittance to ground */
  PetscInt      area; /* Area number */
  PetscInt      zone; /* Zone number */
  PetscScalar   vm; /* Bus voltage magnitude; in pu */
  PetscScalar   va; /* Bus voltage phase angle */
  PetscInt      owner; /* Owner number */
  PetscInt      internal_i; /* Internal Bus Number */
  PetscInt      ngen; /* Number of generators incident at this bus */
  PetscInt      gidx[NGEN_AT_BUS_MAX]; /* list of inndices for accessing the generator data in GEN structure */
  PetscInt      nload;
  PetscInt      lidx[NLOAD_AT_BUS_MAX];
} PETSC_ATTRIBUTEALIGNED(PetscMax(sizeof(double),sizeof(PetscScalar)));

typedef struct _p_VERTEX_Power *VERTEX_Power;

/* 3. Load data */
/* 12 columns */
struct _p_LOAD{
  PetscInt      bus_i; /* Bus number */
  char          i[20]; /* Bus Number or extended bus name*/
  char          id[20]; /* Load identifier, in case of multiple loads. 1 by default */
  PetscInt      status; /* Load status */
  PetscInt      area; /* Area to which load is assigned */
  PetscInt      zone; /* Zone to which load is assigned */
  PetscScalar   pl; /* Active power component of constant MVA load */
  PetscScalar   ql; /* Reactive power component of constant MVA load */
  PetscScalar   ip; /* Active power component of constant current load: MW pu V */
  PetscScalar   iq; /* Reactive power component of constant current load: Mvar pu V */
  PetscScalar   yp; /* Active power component of constant admittance load: MW pu V */
  PetscScalar   yq; /* Reactive power component of constant admittance load: Mvar pu V */
  PetscScalar   scale_load;
  PetscInt      owner; /* Owner number */
  PetscInt      internal_i; /* Internal Bus Number */
} PETSC_ATTRIBUTEALIGNED(PetscMax(sizeof(double),sizeof(PetscScalar)));

typedef struct _p_LOAD *LOAD;

/* 4. Generator data */
/* 20+ columns */
/******* 20, USING ONLY 1 OWNER's WORTH OF DATA. COME BACK TO THIS LATER, if necessary ******/
struct _p_GEN{
  PetscInt      bus_i;
  char          i[20]; /* Bus Number or extended bus name*/
  char          id[20]; /* Generator identifier, in case of multiple generators at same bus. 1 by default */
  PetscScalar   pg; /* Generator active power output */
  PetscScalar   qg; /* Generator reactive power output */
  PetscScalar   qt; /* Maximum reactive power output: Mvar */
  PetscScalar   qb; /* Minimum reactive power output: Mvar */
  PetscScalar   vs; /* Regulated voltage setpoint: pu */
  PetscInt      ireg; /* Remote bus number/identifier */
  PetscScalar   mbase; /* MVA base of the machine */
  PetscScalar   zr; /* Complex machine impedance ZSOURCE in pu on mbase */
  PetscScalar   zx; /* ----------------------"------------------------- */
  PetscScalar   rt; /* Step-up transformer impedance XTRAN in pu on mbase */
  PetscScalar   xt; /* -----------------------"-------------------------- */
  PetscScalar   gtap; /* Step-up transformer turns ratio */
  PetscInt      status; /* Machine status */
  PetscScalar   rmpct; /* Mvar % required to hold voltage at remote bus */
  PetscScalar   pt; /* Gen max active power output: MW */
  PetscScalar   pb; /* Gen min active power output: MW */
  PetscInt      o1; /* Owner number */
  PetscScalar   f1; /* Fraction of ownership */
  PetscScalar   scale_gen;
  PetscInt      internal_i; /* Internal Bus Number */
} PETSC_ATTRIBUTEALIGNED(PetscMax(sizeof(double),sizeof(PetscScalar)));

typedef struct _p_GEN *GEN;

/* 17+ columns */
struct _p_EDGE_Power{
  PetscInt      fbus;
  PetscInt      tbus;
  char          i[20]; /* Bus Number or extended bus name*/
  char          j[20]; /* Bus Number or extended bus name*/
  char          ckt[20]; /* Circuit identifier. 1 by default */
  PetscScalar   r; /* Branch resistance: pu */
  PetscScalar   x; /* Branch reactance: pu */
  PetscScalar   b; /* Branch charging susceptance: pu */
  PetscScalar   rateA; /* rate A in MVA */
  PetscScalar   rateB; /* rate B in MVA */
  PetscScalar   rateC; /* rate C in MVA */
  PetscScalar   tapratio;
  PetscScalar   phaseshift;
  PetscScalar   gi; /* Complex admittance at 'i' end: pu */
  PetscScalar   bi; /* Complex admittance at 'i' end: pu */
  PetscScalar   gj; /* Complex admittance at 'j' end: pu */
  PetscScalar   bj; /* Complex admittance at 'j' end: pu */
  PetscInt      status; /* Service status */
  PetscScalar   length; /* Line length */
  PetscInt      o1; /* Owner number */
  PetscScalar   f1; /* Fraction of ownership */
  PetscScalar   yff[2],yft[2],ytf[2],ytt[2]; /* [G,B] */
  PetscInt      internal_i; /* Internal From Bus Number */
  PetscInt      internal_j; /* Internal To Bus Number */
} PETSC_ATTRIBUTEALIGNED(PetscMax(sizeof(double),sizeof(PetscScalar)));

typedef struct _p_EDGE_Power *EDGE_Power;

/* PTI format data structure */
typedef struct{
  PetscScalar sbase; /* System base MVA */
  PetscInt    nbus,ngen,nbranch,nload; /* # of buses,gens,branches, and loads (includes elements which are
                                          out of service */
  VERTEX_Power bus;
  LOAD         load;
  GEN          gen;
  EDGE_Power   branch;
} PFDATA PETSC_ATTRIBUTEALIGNED(PetscMax(sizeof(double),sizeof(PetscScalar)));

extern PetscErrorCode PFReadMatPowerData(PFDATA*,char*);
extern PetscErrorCode GetListofEdges_Power(PFDATA*,PetscInt*);
extern PetscErrorCode FormJacobian_Power(SNES,Vec, Mat,Mat,void*);
extern PetscErrorCode FormJacobian_Power_private(DM,Vec,Mat,PetscInt,PetscInt,const PetscInt*,const PetscInt*,void*);
extern PetscErrorCode FormFunction_Power(DM,Vec,Vec,PetscInt,PetscInt,const PetscInt*,const PetscInt*,void*);
extern PetscErrorCode SetInitialGuess_Power(DM,Vec,PetscInt,PetscInt,const PetscInt *,const PetscInt *,void*);
#endif
