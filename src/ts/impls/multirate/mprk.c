/*
  Code for time stepping with the Multirate Partitioned Runge-Kutta method

  Notes:
  1) The general system is written as
     Udot = F(t,U)
     if one does not split the RHS function, but gives the indexes for both slow and fast components;
  2) The general system is written as
     Usdot = Fs(t,Us,Uf)
     Ufdot = Ff(t,Us,Uf)
     for component-wise partitioned system,
     users should split the RHS function themselves and also provide the indexes for both slow and fast components.
  3) To correct The confusing terminology in the paper, we use 'slow method', 'slow buffer method' and 'fast method' to denote the methods applied to 'slow region', 'slow buffer region' and 'fast region' respectively. The 'slow method' in the original paper actually means the 'slow buffer method'.
  4) Why does the buffer region have to be inside the slow region? The buffer region is treated with a slow method essentially. Applying the slow method to a region with a fast characteristic time scale is apparently not a good choice.

  Reference:
  Emil M. Constantinescu, Adrian Sandu, Multirate Timestepping Methods for Hyperbolic Conservation Laws, Journal of Scientific Computing 2007
*/

#include <petsc/private/tsimpl.h> /*I   "petscts.h"   I*/
#include <petscdm.h>

static TSMPRKType TSMPRKDefault = TSMPRK2A22;
static PetscBool  TSMPRKRegisterAllCalled;
static PetscBool  TSMPRKPackageInitialized;

typedef struct _MPRKTableau *MPRKTableau;
struct _MPRKTableau {
  char      *name;
  PetscInt   order;           /* Classical approximation order of the method i */
  PetscInt   sbase;           /* Number of stages in the base method*/
  PetscInt   s;               /* Number of stages */
  PetscInt   np;              /* Number of partitions */
  PetscReal *Af, *bf, *cf;    /* Tableau for fast components */
  PetscReal *Amb, *bmb, *cmb; /* Tableau for medium components */
  PetscInt  *rmb;             /* Array of flags for repeated stages in medium method */
  PetscReal *Asb, *bsb, *csb; /* Tableau for slow components */
  PetscInt  *rsb;             /* Array of flags for repeated staged in slow method*/
};
typedef struct _MPRKTableauLink *MPRKTableauLink;
struct _MPRKTableauLink {
  struct _MPRKTableau tab;
  MPRKTableauLink     next;
};
static MPRKTableauLink MPRKTableauList;

typedef struct {
  MPRKTableau  tableau;
  Vec         *Y; /* States computed during the step                           */
  Vec         *YdotRHS;
  Vec         *YdotRHS_slow;         /* Function evaluations by slow tableau for slow components  */
  Vec         *YdotRHS_slowbuffer;   /* Function evaluations by slow tableau for slow components  */
  Vec         *YdotRHS_medium;       /* Function evaluations by slow tableau for slow components  */
  Vec         *YdotRHS_mediumbuffer; /* Function evaluations by slow tableau for slow components  */
  Vec         *YdotRHS_fast;         /* Function evaluations by fast tableau for fast components  */
  PetscScalar *work_slow;            /* Scalar work_slow by slow tableau                          */
  PetscScalar *work_slowbuffer;      /* Scalar work_slow by slow tableau                          */
  PetscScalar *work_medium;          /* Scalar work_slow by medium tableau                        */
  PetscScalar *work_mediumbuffer;    /* Scalar work_slow by medium tableau                        */
  PetscScalar *work_fast;            /* Scalar work_fast by fast tableau                          */
  PetscReal    stage_time;
  TSStepStatus status;
  PetscReal    ptime;
  PetscReal    time_step;
  IS           is_slow, is_slowbuffer, is_medium, is_mediumbuffer, is_fast;
  TS           subts_slow, subts_slowbuffer, subts_medium, subts_mediumbuffer, subts_fast;
} TS_MPRK;

static PetscErrorCode TSMPRKGenerateTableau2(PetscInt ratio, PetscInt s, const PetscReal Abase[], const PetscReal bbase[], PetscReal A1[], PetscReal b1[], PetscReal A2[], PetscReal b2[])
{
  PetscInt i, j, k, l;

  PetscFunctionBegin;
  for (k = 0; k < ratio; k++) {
    /* diagonal blocks */
    for (i = 0; i < s; i++)
      for (j = 0; j < s; j++) {
        A1[(k * s + i) * ratio * s + k * s + j] = Abase[i * s + j];
        A2[(k * s + i) * ratio * s + k * s + j] = Abase[i * s + j] / ratio;
      }
    /* off diagonal blocks */
    for (l = 0; l < k; l++)
      for (i = 0; i < s; i++)
        for (j = 0; j < s; j++) A2[(k * s + i) * ratio * s + l * s + j] = bbase[j] / ratio;
    for (j = 0; j < s; j++) {
      b1[k * s + j] = bbase[j] / ratio;
      b2[k * s + j] = bbase[j] / ratio;
    }
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TSMPRKGenerateTableau3(PetscInt ratio, PetscInt s, const PetscReal Abase[], const PetscReal bbase[], PetscReal A1[], PetscReal b1[], PetscReal A2[], PetscReal b2[], PetscReal A3[], PetscReal b3[])
{
  PetscInt i, j, k, l, m, n;

  PetscFunctionBegin;
  for (k = 0; k < ratio; k++) { /* diagonal blocks of size ratio*s by ratio*s */
    for (l = 0; l < ratio; l++) /* diagonal sub-blocks of size s by s */
      for (i = 0; i < s; i++)
        for (j = 0; j < s; j++) {
          A1[((k * ratio + l) * s + i) * ratio * ratio * s + (k * ratio + l) * s + j] = Abase[i * s + j];
          A2[((k * ratio + l) * s + i) * ratio * ratio * s + (k * ratio + l) * s + j] = Abase[i * s + j] / ratio;
          A3[((k * ratio + l) * s + i) * ratio * ratio * s + (k * ratio + l) * s + j] = Abase[i * s + j] / ratio / ratio;
        }
    for (l = 0; l < k; l++) /* off-diagonal blocks of size ratio*s by ratio*s */
      for (m = 0; m < ratio; m++)
        for (n = 0; n < ratio; n++)
          for (i = 0; i < s; i++)
            for (j = 0; j < s; j++) {
              A2[((k * ratio + m) * s + i) * ratio * ratio * s + (l * ratio + n) * s + j] = bbase[j] / ratio / ratio;
              A3[((k * ratio + m) * s + i) * ratio * ratio * s + (l * ratio + n) * s + j] = bbase[j] / ratio / ratio;
            }
    for (m = 0; m < ratio; m++)
      for (n = 0; n < m; n++) /* off-diagonal sub-blocks of size s by s in the diagonal blocks */
        for (i = 0; i < s; i++)
          for (j = 0; j < s; j++) A3[((k * ratio + m) * s + i) * ratio * ratio * s + (k * ratio + n) * s + j] = bbase[j] / ratio / ratio;
    for (n = 0; n < ratio; n++)
      for (j = 0; j < s; j++) {
        b1[(k * ratio + n) * s + j] = bbase[j] / ratio / ratio;
        b2[(k * ratio + n) * s + j] = bbase[j] / ratio / ratio;
        b3[(k * ratio + n) * s + j] = bbase[j] / ratio / ratio;
      }
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*MC
     TSMPRK2A22 - Second Order Multirate Partitioned Runge Kutta scheme based on RK2A.

     This method has four stages for slow and fast parts. The refinement factor of the stepsize is 2.
     r = 2, np = 2

     Options Database Key:
.     -ts_mprk_type 2a22 - select this scheme

     Level: advanced

.seealso: [](chapter_ts), `TSMPRK`, `TSMPRKType`, `TSMPRKSetType()`
M*/
/*MC
     TSMPRK2A23 - Second Order Multirate Partitioned Runge-Kutta scheme based on RK2A.

     This method has eight stages for slow and medium and fast parts. The refinement factor of the stepsize is 2.
     r = 2, np = 3

     Options Database Key:
.     -ts_mprk_type 2a23 - select this scheme

     Level: advanced

.seealso: [](chapter_ts), `TSMPRK`, `TSMPRKType`, `TSMPRKSetType()`
M*/
/*MC
     TSMPRK2A32 - Second Order Multirate Partitioned Runge-Kutta scheme based on RK2A.

     This method has four stages for slow and fast parts. The refinement factor of the stepsize is 3.
     r = 3, np = 2

     Options Database Key:
.     -ts_mprk_type 2a32 - select this scheme

     Level: advanced

.seealso: [](chapter_ts), `TSMPRK`, `TSMPRKType`, `TSMPRKSetType()`
M*/
/*MC
     TSMPRK2A33 - Second Order Multirate Partitioned Runge-Kutta scheme based on RK2A.

     This method has eight stages for slow and medium and fast parts. The refinement factor of the stepsize is 3.
     r = 3, np = 3

     Options Database Key:
.     -ts_mprk_type 2a33- select this scheme

     Level: advanced

.seealso: [](chapter_ts), `TSMPRK`, `TSMPRKType`, `TSMPRKSetType()`
M*/
/*MC
     TSMPRK3P2M - Third Order Multirate Partitioned Runge-Kutta scheme.

     This method has eight stages for both slow and fast parts.

     Options Database Key:
.     -ts_mprk_type pm3 - select this scheme

     Level: advanced

.seealso: [](chapter_ts), `TSMPRK`, `TSMPRKType`, `TSMPRKSetType()`
M*/
/*MC
     TSMPRKP2 - Second Order Multirate Partitioned Runge-Kutta scheme.

     This method has five stages for both slow and fast parts.

     Options Database Key:
.     -ts_mprk_type p2 - select this scheme

     Level: advanced

.seealso: [](chapter_ts), `TSMPRK`, `TSMPRKType`, `TSMPRKSetType()`
M*/
/*MC
     TSMPRKP3 - Third Order Multirate Partitioned Runge-Kutta scheme.

     This method has ten stages for both slow and fast parts.

     Options Database Key:
.     -ts_mprk_type p3 - select this scheme

     Level: advanced

.seealso: [](chapter_ts), `TSMPRK`, `TSMPRKType`, `TSMPRKSetType()`
M*/

/*@C
  TSMPRKRegisterAll - Registers all of the Partitioned Runge-Kutta explicit methods in `TSMPRK`

  Not Collective, but should be called by all processes which will need the schemes to be registered

  Level: advanced

.seealso: [](chapter_ts), `TSMPRK`, `TSMPRKRegisterDestroy()`
@*/
PetscErrorCode TSMPRKRegisterAll(void)
{
  PetscFunctionBegin;
  if (TSMPRKRegisterAllCalled) PetscFunctionReturn(PETSC_SUCCESS);
  TSMPRKRegisterAllCalled = PETSC_TRUE;

#define RC PetscRealConstant
  {
    const PetscReal Abase[2][2] =
      {
        {0,       0},
        {RC(1.0), 0}
    },
                    bbase[2] = {RC(0.5), RC(0.5)};
    PetscReal Asb[4][4] = {{0}}, Af[4][4] = {{0}}, bsb[4] = {0}, bf[4] = {0};
    PetscInt  rsb[4] = {0, 0, 1, 2};
    PetscCall(TSMPRKGenerateTableau2(2, 2, &Abase[0][0], bbase, &Asb[0][0], bsb, &Af[0][0], bf));
    PetscCall(TSMPRKRegister(TSMPRK2A22, 2, 2, 2, 1, &Asb[0][0], bsb, NULL, rsb, NULL, NULL, NULL, NULL, &Af[0][0], bf, NULL));
  }
  {
    const PetscReal Abase[2][2] =
      {
        {0,       0},
        {RC(1.0), 0}
    },
                    bbase[2] = {RC(0.5), RC(0.5)};
    PetscReal Asb[8][8] = {{0}}, Amb[8][8] = {{0}}, Af[8][8] = {{0}}, bsb[8] = {0}, bmb[8] = {0}, bf[8] = {0};
    PetscInt  rsb[8] = {0, 0, 1, 2, 1, 2, 1, 2}, rmb[8] = {0, 0, 1, 2, 0, 0, 5, 6};
    PetscCall(TSMPRKGenerateTableau3(2, 2, &Abase[0][0], bbase, &Asb[0][0], bsb, &Amb[0][0], bmb, &Af[0][0], bf));
    PetscCall(TSMPRKRegister(TSMPRK2A23, 2, 2, 2, 2, &Asb[0][0], bsb, NULL, rsb, &Amb[0][0], bmb, NULL, rmb, &Af[0][0], bf, NULL));
  }
  {
    const PetscReal Abase[2][2] =
      {
        {0,       0},
        {RC(1.0), 0}
    },
                    bbase[2] = {RC(0.5), RC(0.5)};
    PetscReal Asb[6][6] = {{0}}, Af[6][6] = {{0}}, bsb[6] = {0}, bf[6] = {0};
    PetscInt  rsb[6] = {0, 0, 1, 2, 1, 2};
    PetscCall(TSMPRKGenerateTableau2(3, 2, &Abase[0][0], bbase, &Asb[0][0], bsb, &Af[0][0], bf));
    PetscCall(TSMPRKRegister(TSMPRK2A32, 2, 2, 3, 1, &Asb[0][0], bsb, NULL, rsb, NULL, NULL, NULL, NULL, &Af[0][0], bf, NULL));
  }
  {
    const PetscReal Abase[2][2] =
      {
        {0,       0},
        {RC(1.0), 0}
    },
                    bbase[2] = {RC(0.5), RC(0.5)};
    PetscReal Asb[18][18] = {{0}}, Amb[18][18] = {{0}}, Af[18][18] = {{0}}, bsb[18] = {0}, bmb[18] = {0}, bf[18] = {0};
    PetscInt  rsb[18] = {0, 0, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2}, rmb[18] = {0, 0, 1, 2, 1, 2, 0, 0, 7, 8, 7, 8, 0, 0, 13, 14, 13, 14};
    PetscCall(TSMPRKGenerateTableau3(3, 2, &Abase[0][0], bbase, &Asb[0][0], bsb, &Amb[0][0], bmb, &Af[0][0], bf));
    PetscCall(TSMPRKRegister(TSMPRK2A33, 2, 2, 3, 3, &Asb[0][0], bsb, NULL, rsb, &Amb[0][0], bmb, NULL, rmb, &Af[0][0], bf, NULL));
  }
  /*
    PetscReal
      Asb[8][8] = {{Abase[0][0],Abase[0][1],0,0,0,0,0,0},
                   {Abase[1][0],Abase[1][1],0,0,0,0,0,0},
                   {0,0,Abase[0][0],Abase[0][1],0,0,0,0},
                   {0,0,Abase[1][0],Abase[1][1],0,0,0,0},
                   {0,0,0,0,Abase[0][0],Abase[0][1],0,0},
                   {0,0,0,0,Abase[1][0],Abase[1][1],0,0},
                   {0,0,0,0,0,0,Abase[0][0],Abase[0][1]},
                   {0,0,0,0,0,0,Abase[1][0],Abase[1][1]}},
      Amb[8][8] = {{Abase[0][0]/m,Abase[0][1]/m,0,0,0,0,0,0},
                   {Abase[1][0]/m,Abase[1][1]/m,0,0,0,0,0,0},
                   {0,0,Abase[0][0]/m,Abase[0][1]/m,0,0,0,0},
                   {0,0,Abase[1][0]/m,Abase[1][1]/m,0,0,0,0},
                   {bbase[0]/m,bbase[1]/m,bbase[0]/m,bbase[1]/m,Abase[0][0]/m,Abase[0][1]/m,0,0},
                   {bbase[0]/m,bbase[1]/m,bbase[0]/m,bbase[1]/m,Abase[1][0]/m,Abase[1][1]/m,0,0},
                   {bbase[0]/m,bbase[1]/m,bbase[0]/m,bbase[1]/m,0,0,Abase[0][0]/m,Abase[0][1]/m},
                   {bbase[0]/m,bbase[1]/m,bbase[0]/m,bbase[1]/m,0,0,Abase[1][0]/m,Abase[1][1]/m}},
      Af[8][8] = {{Abase[0][0]/m/m,Abase[0][1]/m/m,0,0,0,0,0,0},
                   {Abase[1][0]/m/m,Abase[1][1]/m/m,0,0,0,0,0,0},
                   {0,0,Abase[0][0]/m/m,Abase[0][1]/m/m,0,0,0,0},
                   {0,0,Abase[1][0]/m/m,Abase[1][1]/m/m,0,0,0,0},
                   {bbase[0]/m/m,bbase[1]/m/m,bbase[0]/m/m,bbase[1]/m/m,Abase[0][0]/m/m,Abase[0][1]/m/m,0,0},
                   {bbase[0]/m/m,bbase[1]/m/m,bbase[0]/m/m,bbase[1]/m/m,Abase[1][0]/m/m,Abase[1][1]/m/m,0,0},
                   {bbase[0]/m/m,bbase[1]/m/m,bbase[0]/m/m,bbase[1]/m/m,bbase[0]/m/m,bbase[1]/m/m,Abase[0][0]/m,Abase[0][1]/m},
                   {bbase[0]/m/m,bbase[1]/m/m,bbase[0]/m/m,bbase[1]/m/m,bbase[0]/m/m,bbase[1]/m/m,Abase[1][0]/m,Abase[1][1]/m}},
      bsb[8]    = {bbase[0]/m/m,bbase[1]/m/m,bbase[0]/m/m,bbase[1]/m/m,bbase[0]/m/m,bbase[1]/m/m,bbase[0]/m/m,bbase[1]/m/m},
      bmb[8]    = {bbase[0]/m/m,bbase[1]/m/m,bbase[0]/m/m,bbase[1]/m/m,bbase[0]/m/m,bbase[1]/m/m,bbase[0]/m,bbase[1]/m/m},
      bf[8]     = {bbase[0]/m/m,bbase[1]/m/m,bbase[0]/m/m,bbase[1]/m/m,bbase[0]/m/m,bbase[1]/m,bbase[0]/m/m,bbase[1]/m/m},
*/
  /*{
      const PetscReal
        As[8][8] = {{0,0,0,0,0,0,0,0},
                    {RC(1.0)/RC(2.0),0,0,0,0,0,0,0},
                    {RC(-1.0)/RC(6.0),RC(2.0)/RC(3.0),0,0,0,0,0,0},
                    {RC(1.0)/RC(3.0),RC(-1.0)/RC(3.0),RC(1.0),0,0,0,0,0},
                    {0,0,0,0,0,0,0,0},
                    {0,0,0,0,RC(1.0)/RC(2.0),0,0,0},
                    {0,0,0,0,RC(-1.0)/RC(6.0),RC(2.0)/RC(3.0),0,0},
                    {0,0,0,0,RC(1.0)/RC(3.0),RC(-1.0)/RC(3.0),RC(1.0),0}},
         A[8][8] = {{0,0,0,0,0,0,0,0},
                    {RC(1.0)/RC(4.0),0,0,0,0,0,0,0},
                    {RC(-1.0)/RC(12.0),RC(1.0)/RC(3.0),0,0,0,0,0,0},
                    {RC(1.0)/RC(6.0),RC(-1.0)/RC(6.0),RC(1.0)/RC(2.0),0,0,0,0,0},
                    {RC(1.0)/RC(12.0),RC(1.0)/RC(6.0),RC(1.0)/RC(6.0),RC(1.0)/RC(12.0),0,0,0,0},
                    {RC(1.0)/RC(12.0),RC(1.0)/RC(6.0),RC(1.0)/RC(6.0),RC(1.0)/RC(12.0),RC(1.0)/RC(4.0),0,0,0},
                    {RC(1.0)/RC(12.0),RC(1.0)/RC(6.0),RC(1.0)/RC(6.0),RC(1.0)/RC(12.0),RC(-1.0)/RC(12.0),RC(1.0)/RC(3.0),0,0},
                    {RC(1.0)/RC(12.0),RC(1.0)/RC(6.0),RC(1.0)/RC(6.0),RC(1.0)/RC(12.0),RC(1.0)/RC(6.0),RC(-1.0)/RC(6.0),RC(1.0)/RC(2.0),0}},
          bs[8] = {RC(1.0)/RC(12.0),RC(1.0)/RC(6.0),RC(1.0)/RC(6.0),RC(1.0)/RC(12.0),RC(1.0)/RC(12.0),RC(1.0)/RC(6.0),RC(1.0)/RC(6.0),RC(1.0)/RC(12.0)},
           b[8] = {RC(1.0)/RC(12.0),RC(1.0)/RC(6.0),RC(1.0)/RC(6.0),RC(1.0)/RC(12.0),RC(1.0)/RC(12.0),RC(1.0)/RC(6.0),RC(1.0)/RC(6.0),RC(1.0)/RC(12.0)};
           PetscCall(TSMPRKRegister(TSMPRKPM3,3,8,&As[0][0],bs,NULL,&A[0][0],b,NULL));
  }*/

  {
    const PetscReal Asb[5][5] =
      {
        {0,                 0, 0, 0, 0},
        {RC(1.0) / RC(2.0), 0, 0, 0, 0},
        {RC(1.0) / RC(2.0), 0, 0, 0, 0},
        {RC(1.0),           0, 0, 0, 0},
        {RC(1.0),           0, 0, 0, 0}
    },
                    Af[5][5] = {{0, 0, 0, 0, 0}, {RC(1.0) / RC(2.0), 0, 0, 0, 0}, {RC(1.0) / RC(4.0), RC(1.0) / RC(4.0), 0, 0, 0}, {RC(1.0) / RC(4.0), RC(1.0) / RC(4.0), RC(1.0) / RC(2.0), 0, 0}, {RC(1.0) / RC(4.0), RC(1.0) / RC(4.0), RC(1.0) / RC(4.0), RC(1.0) / RC(4.0), 0}}, bsb[5] = {RC(1.0) / RC(2.0), 0, 0, 0, RC(1.0) / RC(2.0)}, bf[5] = {RC(1.0) / RC(4.0), RC(1.0) / RC(4.0), RC(1.0) / RC(4.0), RC(1.0) / RC(4.0), 0};
    const PetscInt rsb[5] = {0, 0, 2, 0, 4};
    PetscCall(TSMPRKRegister(TSMPRKP2, 2, 5, 1, 1, &Asb[0][0], bsb, NULL, rsb, NULL, NULL, NULL, NULL, &Af[0][0], bf, NULL));
  }

  {
    const PetscReal Asb[10][10] =
      {
        {0,                  0, 0, 0, 0,                  0,                 0, 0, 0, 0},
        {RC(1.0) / RC(4.0),  0, 0, 0, 0,                  0,                 0, 0, 0, 0},
        {RC(1.0) / RC(4.0),  0, 0, 0, 0,                  0,                 0, 0, 0, 0},
        {RC(1.0) / RC(2.0),  0, 0, 0, 0,                  0,                 0, 0, 0, 0},
        {RC(1.0) / RC(2.0),  0, 0, 0, 0,                  0,                 0, 0, 0, 0},
        {RC(-1.0) / RC(6.0), 0, 0, 0, RC(2.0) / RC(3.0),  0,                 0, 0, 0, 0},
        {RC(1.0) / RC(12.0), 0, 0, 0, RC(1.0) / RC(6.0),  RC(1.0) / RC(2.0), 0, 0, 0, 0},
        {RC(1.0) / RC(12.0), 0, 0, 0, RC(1.0) / RC(6.0),  RC(1.0) / RC(2.0), 0, 0, 0, 0},
        {RC(1.0) / RC(3.0),  0, 0, 0, RC(-1.0) / RC(3.0), RC(1.0),           0, 0, 0, 0},
        {RC(1.0) / RC(3.0),  0, 0, 0, RC(-1.0) / RC(3.0), RC(1.0),           0, 0, 0, 0}
    },
                    Af[10][10] = {{0, 0, 0, 0, 0, 0, 0, 0, 0, 0}, {RC(1.0) / RC(4.0), 0, 0, 0, 0, 0, 0, 0, 0, 0}, {RC(-1.0) / RC(12.0), RC(1.0) / RC(3.0), 0, 0, 0, 0, 0, 0, 0, 0}, {RC(1.0) / RC(6.0), RC(-1.0) / RC(6.0), RC(1.0) / RC(2.0), 0, 0, 0, 0, 0, 0, 0}, {RC(1.0) / RC(12.0), RC(1.0) / RC(6.0), RC(1.0) / RC(6.0), RC(1.0) / RC(12.0), 0, 0, 0, 0, 0, 0}, {RC(1.0) / RC(12.0), RC(1.0) / RC(6.0), RC(1.0) / RC(6.0), RC(1.0) / RC(12.0), 0, 0, 0, 0, 0, 0}, {RC(1.0) / RC(12.0), RC(1.0) / RC(6.0), RC(1.0) / RC(6.0), RC(1.0) / RC(12.0), 0, RC(1.0) / RC(4.0), 0, 0, 0, 0}, {RC(1.0) / RC(12.0), RC(1.0) / RC(6.0), RC(1.0) / RC(6.0), RC(1.0) / RC(12.0), 0, RC(-1.0) / RC(12.0), RC(1.0) / RC(3.0), 0, 0, 0}, {RC(1.0) / RC(12.0), RC(1.0) / RC(6.0), RC(1.0) / RC(6.0), RC(1.0) / RC(12.0), 0, RC(1.0) / RC(6.0), RC(-1.0) / RC(6.0), RC(1.0) / RC(2.0), 0, 0}, {RC(1.0) / RC(12.0), RC(1.0) / RC(6.0), RC(1.0) / RC(6.0), RC(1.0) / RC(12.0), 0, RC(1.0) / RC(12.0), RC(1.0) / RC(6.0), RC(1.0) / RC(6.0), RC(1.0) / RC(12.0), 0}}, bsb[10] = {RC(1.0) / RC(6.0), 0, 0, 0, RC(1.0) / RC(3.0), RC(1.0) / RC(3.0), 0, 0, 0, RC(1.0) / RC(6.0)}, bf[10] = {RC(1.0) / RC(12.0), RC(1.0) / RC(6.0), RC(1.0) / RC(6.0), RC(1.0) / RC(12.0), 0, RC(1.0) / RC(12.0), RC(1.0) / RC(6.0), RC(1.0) / RC(6.0), RC(1.0) / RC(12.0), 0};
    const PetscInt rsb[10] = {0, 0, 2, 0, 4, 0, 0, 7, 0, 9};
    PetscCall(TSMPRKRegister(TSMPRKP3, 3, 5, 2, 1, &Asb[0][0], bsb, NULL, rsb, NULL, NULL, NULL, NULL, &Af[0][0], bf, NULL));
  }
#undef RC
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
   TSMPRKRegisterDestroy - Frees the list of schemes that were registered by `TSMPRKRegister()`.

   Not Collective

   Level: advanced

.seealso: [](chapter_ts), `TSMPRK`, `TSMPRKRegister()`, `TSMPRKRegisterAll()`
@*/
PetscErrorCode TSMPRKRegisterDestroy(void)
{
  MPRKTableauLink link;

  PetscFunctionBegin;
  while ((link = MPRKTableauList)) {
    MPRKTableau t   = &link->tab;
    MPRKTableauList = link->next;
    PetscCall(PetscFree3(t->Asb, t->bsb, t->csb));
    PetscCall(PetscFree3(t->Amb, t->bmb, t->cmb));
    PetscCall(PetscFree3(t->Af, t->bf, t->cf));
    PetscCall(PetscFree(t->rsb));
    PetscCall(PetscFree(t->rmb));
    PetscCall(PetscFree(t->name));
    PetscCall(PetscFree(link));
  }
  TSMPRKRegisterAllCalled = PETSC_FALSE;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  TSMPRKInitializePackage - This function initializes everything in the `TSMPRK` package. It is called
  from `PetscDLLibraryRegister()` when using dynamic libraries, and on the first call to `TSCreate_MPRK()`
  when using static libraries.

  Level: developer

.seealso: [](chapter_ts), `TSMPRK`, `PetscInitialize()`
@*/
PetscErrorCode TSMPRKInitializePackage(void)
{
  PetscFunctionBegin;
  if (TSMPRKPackageInitialized) PetscFunctionReturn(PETSC_SUCCESS);
  TSMPRKPackageInitialized = PETSC_TRUE;
  PetscCall(TSMPRKRegisterAll());
  PetscCall(PetscRegisterFinalize(TSMPRKFinalizePackage));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  TSMPRKFinalizePackage - This function destroys everything in the `TSMPRK` package. It is
  called from `PetscFinalize()`.

  Level: developer

.seealso: [](chapter_ts), `TSMPRK`, `PetscFinalize()`
@*/
PetscErrorCode TSMPRKFinalizePackage(void)
{
  PetscFunctionBegin;
  TSMPRKPackageInitialized = PETSC_FALSE;
  PetscCall(TSMPRKRegisterDestroy());
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
   TSMPRKRegister - register a `TSMPRK` scheme by providing the entries in the Butcher tableau

   Not Collective, but the same schemes should be registered on all processes on which they will be used

   Input Parameters:
+  name - identifier for method
.  order - approximation order of method
.  sbase  - number of stages in the base methods
.  ratio1 - stepsize ratio at 1st level (e.g. slow/medium)
.  ratio2 - stepsize ratio at 2nd level (e.g. medium/fast)
.  Af - stage coefficients for fast components(dimension s*s, row-major)
.  bf - step completion table for fast components(dimension s)
.  cf - abscissa for fast components(dimension s)
.  As - stage coefficients for slow components(dimension s*s, row-major)
.  bs - step completion table for slow components(dimension s)
-  cs - abscissa for slow components(dimension s)

   Level: advanced

   Note:
   Several `TSMPRK` methods are provided, this function is only needed to create new methods.

.seealso: [](chapter_ts), `TSMPRK`
@*/
PetscErrorCode TSMPRKRegister(TSMPRKType name, PetscInt order, PetscInt sbase, PetscInt ratio1, PetscInt ratio2, const PetscReal Asb[], const PetscReal bsb[], const PetscReal csb[], const PetscInt rsb[], const PetscReal Amb[], const PetscReal bmb[], const PetscReal cmb[], const PetscInt rmb[], const PetscReal Af[], const PetscReal bf[], const PetscReal cf[])
{
  MPRKTableauLink link;
  MPRKTableau     t;
  PetscInt        s, i, j;

  PetscFunctionBegin;
  PetscValidCharPointer(name, 1);
  PetscValidRealPointer(Asb, 6);
  if (bsb) PetscValidRealPointer(bsb, 7);
  if (csb) PetscValidRealPointer(csb, 8);
  if (rsb) PetscValidIntPointer(rsb, 9);
  if (Amb) PetscValidRealPointer(Amb, 10);
  if (bmb) PetscValidRealPointer(bmb, 11);
  if (cmb) PetscValidRealPointer(cmb, 12);
  if (rmb) PetscValidIntPointer(rmb, 13);
  PetscValidRealPointer(Af, 14);
  if (bf) PetscValidRealPointer(bf, 15);
  if (cf) PetscValidRealPointer(cf, 16);

  PetscCall(PetscNew(&link));
  t = &link->tab;

  PetscCall(PetscStrallocpy(name, &t->name));
  s        = sbase * ratio1 * ratio2; /*  this is the dimension of the matrices below */
  t->order = order;
  t->sbase = sbase;
  t->s     = s;
  t->np    = 2;

  PetscCall(PetscMalloc3(s * s, &t->Af, s, &t->bf, s, &t->cf));
  PetscCall(PetscArraycpy(t->Af, Af, s * s));
  if (bf) {
    PetscCall(PetscArraycpy(t->bf, bf, s));
  } else
    for (i = 0; i < s; i++) t->bf[i] = Af[(s - 1) * s + i];
  if (cf) {
    PetscCall(PetscArraycpy(t->cf, cf, s));
  } else {
    for (i = 0; i < s; i++)
      for (j = 0, t->cf[i] = 0; j < s; j++) t->cf[i] += Af[i * s + j];
  }

  if (Amb) {
    t->np = 3;
    PetscCall(PetscMalloc3(s * s, &t->Amb, s, &t->bmb, s, &t->cmb));
    PetscCall(PetscArraycpy(t->Amb, Amb, s * s));
    if (bmb) {
      PetscCall(PetscArraycpy(t->bmb, bmb, s));
    } else {
      for (i = 0; i < s; i++) t->bmb[i] = Amb[(s - 1) * s + i];
    }
    if (cmb) {
      PetscCall(PetscArraycpy(t->cmb, cmb, s));
    } else {
      for (i = 0; i < s; i++)
        for (j = 0, t->cmb[i] = 0; j < s; j++) t->cmb[i] += Amb[i * s + j];
    }
    if (rmb) {
      PetscCall(PetscMalloc1(s, &t->rmb));
      PetscCall(PetscArraycpy(t->rmb, rmb, s));
    } else {
      PetscCall(PetscCalloc1(s, &t->rmb));
    }
  }

  PetscCall(PetscMalloc3(s * s, &t->Asb, s, &t->bsb, s, &t->csb));
  PetscCall(PetscArraycpy(t->Asb, Asb, s * s));
  if (bsb) {
    PetscCall(PetscArraycpy(t->bsb, bsb, s));
  } else
    for (i = 0; i < s; i++) t->bsb[i] = Asb[(s - 1) * s + i];
  if (csb) {
    PetscCall(PetscArraycpy(t->csb, csb, s));
  } else {
    for (i = 0; i < s; i++)
      for (j = 0, t->csb[i] = 0; j < s; j++) t->csb[i] += Asb[i * s + j];
  }
  if (rsb) {
    PetscCall(PetscMalloc1(s, &t->rsb));
    PetscCall(PetscArraycpy(t->rsb, rsb, s));
  } else {
    PetscCall(PetscCalloc1(s, &t->rsb));
  }
  link->next      = MPRKTableauList;
  MPRKTableauList = link;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TSMPRKSetSplits(TS ts)
{
  TS_MPRK    *mprk = (TS_MPRK *)ts->data;
  MPRKTableau tab  = mprk->tableau;
  DM          dm, subdm, newdm;

  PetscFunctionBegin;
  PetscCall(TSRHSSplitGetSubTS(ts, "slow", &mprk->subts_slow));
  PetscCall(TSRHSSplitGetSubTS(ts, "fast", &mprk->subts_fast));
  PetscCheck(mprk->subts_slow && mprk->subts_fast, PetscObjectComm((PetscObject)ts), PETSC_ERR_USER, "Must set up the RHSFunctions for 'slow' and 'fast' components using TSRHSSplitSetRHSFunction() or calling TSSetRHSFunction() for each sub-TS");

  /* Only copy the DM */
  PetscCall(TSGetDM(ts, &dm));

  PetscCall(TSRHSSplitGetSubTS(ts, "slowbuffer", &mprk->subts_slowbuffer));
  if (!mprk->subts_slowbuffer) {
    mprk->subts_slowbuffer = mprk->subts_slow;
    mprk->subts_slow       = NULL;
  }
  if (mprk->subts_slow) {
    PetscCall(DMClone(dm, &newdm));
    PetscCall(TSGetDM(mprk->subts_slow, &subdm));
    PetscCall(DMCopyDMTS(subdm, newdm));
    PetscCall(DMCopyDMSNES(subdm, newdm));
    PetscCall(TSSetDM(mprk->subts_slow, newdm));
    PetscCall(DMDestroy(&newdm));
  }
  PetscCall(DMClone(dm, &newdm));
  PetscCall(TSGetDM(mprk->subts_slowbuffer, &subdm));
  PetscCall(DMCopyDMTS(subdm, newdm));
  PetscCall(DMCopyDMSNES(subdm, newdm));
  PetscCall(TSSetDM(mprk->subts_slowbuffer, newdm));
  PetscCall(DMDestroy(&newdm));

  PetscCall(DMClone(dm, &newdm));
  PetscCall(TSGetDM(mprk->subts_fast, &subdm));
  PetscCall(DMCopyDMTS(subdm, newdm));
  PetscCall(DMCopyDMSNES(subdm, newdm));
  PetscCall(TSSetDM(mprk->subts_fast, newdm));
  PetscCall(DMDestroy(&newdm));

  if (tab->np == 3) {
    PetscCall(TSRHSSplitGetSubTS(ts, "medium", &mprk->subts_medium));
    PetscCall(TSRHSSplitGetSubTS(ts, "mediumbuffer", &mprk->subts_mediumbuffer));
    if (mprk->subts_medium && !mprk->subts_mediumbuffer) {
      mprk->subts_mediumbuffer = mprk->subts_medium;
      mprk->subts_medium       = NULL;
    }
    if (mprk->subts_medium) {
      PetscCall(DMClone(dm, &newdm));
      PetscCall(TSGetDM(mprk->subts_medium, &subdm));
      PetscCall(DMCopyDMTS(subdm, newdm));
      PetscCall(DMCopyDMSNES(subdm, newdm));
      PetscCall(TSSetDM(mprk->subts_medium, newdm));
      PetscCall(DMDestroy(&newdm));
    }
    PetscCall(DMClone(dm, &newdm));
    PetscCall(TSGetDM(mprk->subts_mediumbuffer, &subdm));
    PetscCall(DMCopyDMTS(subdm, newdm));
    PetscCall(DMCopyDMSNES(subdm, newdm));
    PetscCall(TSSetDM(mprk->subts_mediumbuffer, newdm));
    PetscCall(DMDestroy(&newdm));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
 This if for nonsplit RHS MPRK
 The step completion formula is

 x1 = x0 + h b^T YdotRHS

*/
static PetscErrorCode TSEvaluateStep_MPRK(TS ts, PetscInt order, Vec X, PetscBool *done)
{
  TS_MPRK     *mprk = (TS_MPRK *)ts->data;
  MPRKTableau  tab  = mprk->tableau;
  PetscScalar *wf   = mprk->work_fast;
  PetscReal    h    = ts->time_step;
  PetscInt     s    = tab->s, j;

  PetscFunctionBegin;
  for (j = 0; j < s; j++) wf[j] = h * tab->bf[j];
  PetscCall(VecCopy(ts->vec_sol, X));
  PetscCall(VecMAXPY(X, s, wf, mprk->YdotRHS));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TSStep_MPRK(TS ts)
{
  TS_MPRK         *mprk = (TS_MPRK *)ts->data;
  Vec             *Y = mprk->Y, *YdotRHS = mprk->YdotRHS, *YdotRHS_fast = mprk->YdotRHS_fast, *YdotRHS_slow = mprk->YdotRHS_slow, *YdotRHS_slowbuffer = mprk->YdotRHS_slowbuffer;
  Vec              Yslow, Yslowbuffer, Yfast;
  MPRKTableau      tab = mprk->tableau;
  const PetscInt   s   = tab->s;
  const PetscReal *Af = tab->Af, *cf = tab->cf, *Asb = tab->Asb, *csb = tab->csb;
  PetscScalar     *wf = mprk->work_fast, *wsb = mprk->work_slowbuffer;
  PetscInt         i, j;
  PetscReal        next_time_step = ts->time_step, t = ts->ptime, h = ts->time_step;

  PetscFunctionBegin;
  for (i = 0; i < s; i++) {
    mprk->stage_time = t + h * cf[i];
    PetscCall(TSPreStage(ts, mprk->stage_time));
    PetscCall(VecCopy(ts->vec_sol, Y[i]));

    /* slow buffer region */
    for (j = 0; j < i; j++) wsb[j] = h * Asb[i * s + j];
    for (j = 0; j < i; j++) PetscCall(VecGetSubVector(YdotRHS[j], mprk->is_slowbuffer, &YdotRHS_slowbuffer[j]));
    PetscCall(VecGetSubVector(Y[i], mprk->is_slowbuffer, &Yslowbuffer));
    PetscCall(VecMAXPY(Yslowbuffer, i, wsb, mprk->YdotRHS_slowbuffer));
    PetscCall(VecRestoreSubVector(Y[i], mprk->is_slowbuffer, &Yslowbuffer));
    for (j = 0; j < i; j++) PetscCall(VecRestoreSubVector(YdotRHS[j], mprk->is_slowbuffer, &YdotRHS_slowbuffer[j]));
    /* slow region */
    if (mprk->is_slow) {
      for (j = 0; j < i; j++) PetscCall(VecGetSubVector(YdotRHS[j], mprk->is_slow, &YdotRHS_slow[j]));
      PetscCall(VecGetSubVector(Y[i], mprk->is_slow, &Yslow));
      PetscCall(VecMAXPY(Yslow, i, wsb, mprk->YdotRHS_slow));
      PetscCall(VecRestoreSubVector(Y[i], mprk->is_slow, &Yslow));
      for (j = 0; j < i; j++) PetscCall(VecRestoreSubVector(YdotRHS[j], mprk->is_slow, &YdotRHS_slow[j]));
    }

    /* fast region */
    for (j = 0; j < i; j++) wf[j] = h * Af[i * s + j];
    for (j = 0; j < i; j++) PetscCall(VecGetSubVector(YdotRHS[j], mprk->is_fast, &YdotRHS_fast[j]));
    PetscCall(VecGetSubVector(Y[i], mprk->is_fast, &Yfast));
    PetscCall(VecMAXPY(Yfast, i, wf, mprk->YdotRHS_fast));
    PetscCall(VecRestoreSubVector(Y[i], mprk->is_fast, &Yfast));
    for (j = 0; j < i; j++) PetscCall(VecRestoreSubVector(YdotRHS[j], mprk->is_fast, &YdotRHS_fast[j]));
    if (tab->np == 3) {
      Vec         *YdotRHS_medium = mprk->YdotRHS_medium, *YdotRHS_mediumbuffer = mprk->YdotRHS_mediumbuffer;
      Vec          Ymedium, Ymediumbuffer;
      PetscScalar *wmb = mprk->work_mediumbuffer;

      for (j = 0; j < i; j++) wmb[j] = h * tab->Amb[i * s + j];
      /* medium region */
      if (mprk->is_medium) {
        for (j = 0; j < i; j++) PetscCall(VecGetSubVector(YdotRHS[j], mprk->is_medium, &YdotRHS_medium[j]));
        PetscCall(VecGetSubVector(Y[i], mprk->is_medium, &Ymedium));
        PetscCall(VecMAXPY(Ymedium, i, wmb, mprk->YdotRHS_medium));
        PetscCall(VecRestoreSubVector(Y[i], mprk->is_medium, &Ymedium));
        for (j = 0; j < i; j++) PetscCall(VecRestoreSubVector(YdotRHS[j], mprk->is_medium, &YdotRHS_medium[j]));
      }
      /* medium buffer region */
      for (j = 0; j < i; j++) PetscCall(VecGetSubVector(YdotRHS[j], mprk->is_mediumbuffer, &YdotRHS_mediumbuffer[j]));
      PetscCall(VecGetSubVector(Y[i], mprk->is_mediumbuffer, &Ymediumbuffer));
      PetscCall(VecMAXPY(Ymediumbuffer, i, wmb, mprk->YdotRHS_mediumbuffer));
      PetscCall(VecRestoreSubVector(Y[i], mprk->is_mediumbuffer, &Ymediumbuffer));
      for (j = 0; j < i; j++) PetscCall(VecRestoreSubVector(YdotRHS[j], mprk->is_mediumbuffer, &YdotRHS_mediumbuffer[j]));
    }
    PetscCall(TSPostStage(ts, mprk->stage_time, i, Y));
    /* compute the stage RHS by fast and slow tableau respectively */
    PetscCall(TSComputeRHSFunction(ts, t + h * csb[i], Y[i], YdotRHS[i]));
  }
  PetscCall(TSEvaluateStep(ts, tab->order, ts->vec_sol, NULL));
  ts->ptime += ts->time_step;
  ts->time_step = next_time_step;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
 This if for the case when split RHS is used
 The step completion formula is
 x1 = x0 + h b^T YdotRHS
*/
static PetscErrorCode TSEvaluateStep_MPRKSPLIT(TS ts, PetscInt order, Vec X, PetscBool *done)
{
  TS_MPRK     *mprk = (TS_MPRK *)ts->data;
  MPRKTableau  tab  = mprk->tableau;
  Vec          Xslow, Xfast, Xslowbuffer; /* subvectors for slow and fast components in X respectively */
  PetscScalar *wf = mprk->work_fast, *ws = mprk->work_slow, *wsb = mprk->work_slowbuffer;
  PetscReal    h = ts->time_step;
  PetscInt     s = tab->s, j, computedstages;

  PetscFunctionBegin;
  PetscCall(VecCopy(ts->vec_sol, X));

  /* slow region */
  if (mprk->is_slow) {
    computedstages = 0;
    for (j = 0; j < s; j++) {
      if (tab->rsb[j]) ws[tab->rsb[j] - 1] += h * tab->bsb[j];
      else ws[computedstages++] = h * tab->bsb[j];
    }
    PetscCall(VecGetSubVector(X, mprk->is_slow, &Xslow));
    PetscCall(VecMAXPY(Xslow, computedstages, ws, mprk->YdotRHS_slow));
    PetscCall(VecRestoreSubVector(X, mprk->is_slow, &Xslow));
  }

  if (tab->np == 3 && mprk->is_medium) {
    computedstages = 0;
    for (j = 0; j < s; j++) {
      if (tab->rmb[j]) wsb[computedstages - tab->sbase + (tab->rmb[j] - 1) % tab->sbase] += h * tab->bsb[j];
      else wsb[computedstages++] = h * tab->bsb[j];
    }
    PetscCall(VecGetSubVector(X, mprk->is_slowbuffer, &Xslowbuffer));
    PetscCall(VecMAXPY(Xslowbuffer, computedstages, wsb, mprk->YdotRHS_slowbuffer));
    PetscCall(VecRestoreSubVector(X, mprk->is_slowbuffer, &Xslowbuffer));
  } else {
    /* slow buffer region */
    for (j = 0; j < s; j++) wsb[j] = h * tab->bsb[j];
    PetscCall(VecGetSubVector(X, mprk->is_slowbuffer, &Xslowbuffer));
    PetscCall(VecMAXPY(Xslowbuffer, s, wsb, mprk->YdotRHS_slowbuffer));
    PetscCall(VecRestoreSubVector(X, mprk->is_slowbuffer, &Xslowbuffer));
  }
  if (tab->np == 3) {
    Vec          Xmedium, Xmediumbuffer;
    PetscScalar *wm = mprk->work_medium, *wmb = mprk->work_mediumbuffer;
    /* medium region and slow buffer region */
    if (mprk->is_medium) {
      computedstages = 0;
      for (j = 0; j < s; j++) {
        if (tab->rmb[j]) wm[computedstages - tab->sbase + (tab->rmb[j] - 1) % tab->sbase] += h * tab->bmb[j];
        else wm[computedstages++] = h * tab->bmb[j];
      }
      PetscCall(VecGetSubVector(X, mprk->is_medium, &Xmedium));
      PetscCall(VecMAXPY(Xmedium, computedstages, wm, mprk->YdotRHS_medium));
      PetscCall(VecRestoreSubVector(X, mprk->is_medium, &Xmedium));
    }
    /* medium buffer region */
    for (j = 0; j < s; j++) wmb[j] = h * tab->bmb[j];
    PetscCall(VecGetSubVector(X, mprk->is_mediumbuffer, &Xmediumbuffer));
    PetscCall(VecMAXPY(Xmediumbuffer, s, wmb, mprk->YdotRHS_mediumbuffer));
    PetscCall(VecRestoreSubVector(X, mprk->is_mediumbuffer, &Xmediumbuffer));
  }
  /* fast region */
  for (j = 0; j < s; j++) wf[j] = h * tab->bf[j];
  PetscCall(VecGetSubVector(X, mprk->is_fast, &Xfast));
  PetscCall(VecMAXPY(Xfast, s, wf, mprk->YdotRHS_fast));
  PetscCall(VecRestoreSubVector(X, mprk->is_fast, &Xfast));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TSStep_MPRKSPLIT(TS ts)
{
  TS_MPRK         *mprk = (TS_MPRK *)ts->data;
  MPRKTableau      tab  = mprk->tableau;
  Vec             *Y = mprk->Y, *YdotRHS_fast = mprk->YdotRHS_fast, *YdotRHS_slow = mprk->YdotRHS_slow, *YdotRHS_slowbuffer = mprk->YdotRHS_slowbuffer;
  Vec              Yslow, Yslowbuffer, Yfast; /* subvectors for slow and fast components in Y[i] respectively */
  PetscInt         s  = tab->s;
  const PetscReal *Af = tab->Af, *cf = tab->cf, *Asb = tab->Asb, *csb = tab->csb;
  PetscScalar     *wf = mprk->work_fast, *ws = mprk->work_slow, *wsb = mprk->work_slowbuffer;
  PetscInt         i, j, computedstages;
  PetscReal        next_time_step = ts->time_step, t = ts->ptime, h = ts->time_step;

  PetscFunctionBegin;
  for (i = 0; i < s; i++) {
    mprk->stage_time = t + h * cf[i];
    PetscCall(TSPreStage(ts, mprk->stage_time));
    /* calculate the stage value for fast and slow components respectively */
    PetscCall(VecCopy(ts->vec_sol, Y[i]));
    for (j = 0; j < i; j++) wsb[j] = h * Asb[i * s + j];

    /* slow buffer region */
    if (tab->np == 3 && mprk->is_medium) {
      if (tab->rmb[i]) {
        PetscCall(VecGetSubVector(Y[i], mprk->is_slowbuffer, &Yslowbuffer));
        PetscCall(VecISCopy(Y[tab->rmb[i] - 1], mprk->is_slowbuffer, SCATTER_REVERSE, Yslowbuffer));
        PetscCall(VecRestoreSubVector(Y[i], mprk->is_slowbuffer, &Yslowbuffer));
      } else {
        PetscScalar *wm = mprk->work_medium;
        computedstages  = 0;
        for (j = 0; j < i; j++) {
          if (tab->rmb[j]) wm[computedstages - tab->sbase + (tab->rmb[j] - 1) % tab->sbase] += wsb[j];
          else wm[computedstages++] = wsb[j];
        }
        PetscCall(VecGetSubVector(Y[i], mprk->is_slowbuffer, &Yslowbuffer));
        PetscCall(VecMAXPY(Yslowbuffer, computedstages, wm, YdotRHS_slowbuffer));
        PetscCall(VecRestoreSubVector(Y[i], mprk->is_slowbuffer, &Yslowbuffer));
      }
    } else {
      PetscCall(VecGetSubVector(Y[i], mprk->is_slowbuffer, &Yslowbuffer));
      PetscCall(VecMAXPY(Yslowbuffer, i, wsb, YdotRHS_slowbuffer));
      PetscCall(VecRestoreSubVector(Y[i], mprk->is_slowbuffer, &Yslowbuffer));
    }

    /* slow region */
    if (mprk->is_slow) {
      if (tab->rsb[i]) { /* repeat previous stage */
        PetscCall(VecGetSubVector(Y[i], mprk->is_slow, &Yslow));
        PetscCall(VecISCopy(Y[tab->rsb[i] - 1], mprk->is_slow, SCATTER_REVERSE, Yslow));
        PetscCall(VecRestoreSubVector(Y[i], mprk->is_slow, &Yslow));
      } else {
        computedstages = 0;
        for (j = 0; j < i; j++) {
          if (tab->rsb[j]) ws[tab->rsb[j] - 1] += wsb[j];
          else ws[computedstages++] = wsb[j];
        }
        PetscCall(VecGetSubVector(Y[i], mprk->is_slow, &Yslow));
        PetscCall(VecMAXPY(Yslow, computedstages, ws, YdotRHS_slow));
        PetscCall(VecRestoreSubVector(Y[i], mprk->is_slow, &Yslow));
        /* only depends on the slow buffer region */
        PetscCall(TSComputeRHSFunction(mprk->subts_slow, t + h * csb[i], Y[i], YdotRHS_slow[computedstages]));
      }
    }

    /* fast region */
    for (j = 0; j < i; j++) wf[j] = h * Af[i * s + j];
    PetscCall(VecGetSubVector(Y[i], mprk->is_fast, &Yfast));
    PetscCall(VecMAXPY(Yfast, i, wf, YdotRHS_fast));
    PetscCall(VecRestoreSubVector(Y[i], mprk->is_fast, &Yfast));

    if (tab->np == 3) {
      Vec             *YdotRHS_medium = mprk->YdotRHS_medium, *YdotRHS_mediumbuffer = mprk->YdotRHS_mediumbuffer;
      Vec              Ymedium, Ymediumbuffer;
      const PetscReal *Amb = tab->Amb, *cmb = tab->cmb;
      PetscScalar     *wm = mprk->work_medium, *wmb = mprk->work_mediumbuffer;

      for (j = 0; j < i; j++) wmb[j] = h * Amb[i * s + j];
      /* medium buffer region */
      PetscCall(VecGetSubVector(Y[i], mprk->is_mediumbuffer, &Ymediumbuffer));
      PetscCall(VecMAXPY(Ymediumbuffer, i, wmb, YdotRHS_mediumbuffer));
      PetscCall(VecRestoreSubVector(Y[i], mprk->is_mediumbuffer, &Ymediumbuffer));
      /* medium region */
      if (mprk->is_medium) {
        if (tab->rmb[i]) { /* repeat previous stage */
          PetscCall(VecGetSubVector(Y[i], mprk->is_medium, &Ymedium));
          PetscCall(VecISCopy(Y[tab->rmb[i] - 1], mprk->is_medium, SCATTER_REVERSE, Ymedium));
          PetscCall(VecRestoreSubVector(Y[i], mprk->is_medium, &Ymedium));
        } else {
          computedstages = 0;
          for (j = 0; j < i; j++) {
            if (tab->rmb[j]) wm[computedstages - tab->sbase + (tab->rmb[j] - 1) % tab->sbase] += wmb[j];
            else wm[computedstages++] = wmb[j];
          }
          PetscCall(VecGetSubVector(Y[i], mprk->is_medium, &Ymedium));
          PetscCall(VecMAXPY(Ymedium, computedstages, wm, YdotRHS_medium));
          PetscCall(VecRestoreSubVector(Y[i], mprk->is_medium, &Ymedium));
          /* only depends on the medium buffer region */
          PetscCall(TSComputeRHSFunction(mprk->subts_medium, t + h * cmb[i], Y[i], YdotRHS_medium[computedstages]));
          /* must be computed after all regions are updated in Y */
          PetscCall(TSComputeRHSFunction(mprk->subts_slowbuffer, t + h * csb[i], Y[i], YdotRHS_slowbuffer[computedstages]));
        }
      }
      /* must be computed after fast region and slow region are updated in Y */
      PetscCall(TSComputeRHSFunction(mprk->subts_mediumbuffer, t + h * cmb[i], Y[i], YdotRHS_mediumbuffer[i]));
    }
    if (!(tab->np == 3 && mprk->is_medium)) PetscCall(TSComputeRHSFunction(mprk->subts_slowbuffer, t + h * csb[i], Y[i], YdotRHS_slowbuffer[i]));
    PetscCall(TSComputeRHSFunction(mprk->subts_fast, t + h * cf[i], Y[i], YdotRHS_fast[i]));
  }

  PetscCall(TSEvaluateStep(ts, tab->order, ts->vec_sol, NULL));
  ts->ptime += ts->time_step;
  ts->time_step = next_time_step;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TSMPRKTableauReset(TS ts)
{
  TS_MPRK    *mprk = (TS_MPRK *)ts->data;
  MPRKTableau tab  = mprk->tableau;

  PetscFunctionBegin;
  if (!tab) PetscFunctionReturn(PETSC_SUCCESS);
  PetscCall(PetscFree(mprk->work_fast));
  PetscCall(PetscFree(mprk->work_slow));
  PetscCall(PetscFree(mprk->work_slowbuffer));
  PetscCall(PetscFree(mprk->work_medium));
  PetscCall(PetscFree(mprk->work_mediumbuffer));
  PetscCall(VecDestroyVecs(tab->s, &mprk->Y));
  if (ts->use_splitrhsfunction) {
    PetscCall(VecDestroyVecs(tab->s, &mprk->YdotRHS_fast));
    PetscCall(VecDestroyVecs(tab->s, &mprk->YdotRHS_slow));
    PetscCall(VecDestroyVecs(tab->s, &mprk->YdotRHS_slowbuffer));
    PetscCall(VecDestroyVecs(tab->s, &mprk->YdotRHS_medium));
    PetscCall(VecDestroyVecs(tab->s, &mprk->YdotRHS_mediumbuffer));
  } else {
    PetscCall(VecDestroyVecs(tab->s, &mprk->YdotRHS));
    if (mprk->is_slow) PetscCall(PetscFree(mprk->YdotRHS_slow));
    PetscCall(PetscFree(mprk->YdotRHS_slowbuffer));
    if (tab->np == 3) {
      if (mprk->is_medium) PetscCall(PetscFree(mprk->YdotRHS_medium));
      PetscCall(PetscFree(mprk->YdotRHS_mediumbuffer));
    }
    PetscCall(PetscFree(mprk->YdotRHS_fast));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TSReset_MPRK(TS ts)
{
  PetscFunctionBegin;
  PetscCall(TSMPRKTableauReset(ts));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode DMCoarsenHook_TSMPRK(DM fine, DM coarse, void *ctx)
{
  PetscFunctionBegin;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode DMRestrictHook_TSMPRK(DM fine, Mat restrct, Vec rscale, Mat inject, DM coarse, void *ctx)
{
  PetscFunctionBegin;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode DMSubDomainHook_TSMPRK(DM dm, DM subdm, void *ctx)
{
  PetscFunctionBegin;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode DMSubDomainRestrictHook_TSMPRK(DM dm, VecScatter gscat, VecScatter lscat, DM subdm, void *ctx)
{
  PetscFunctionBegin;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TSMPRKTableauSetUp(TS ts)
{
  TS_MPRK    *mprk = (TS_MPRK *)ts->data;
  MPRKTableau tab  = mprk->tableau;
  Vec         YdotRHS_slow, YdotRHS_slowbuffer, YdotRHS_medium, YdotRHS_mediumbuffer, YdotRHS_fast;

  PetscFunctionBegin;
  PetscCall(VecDuplicateVecs(ts->vec_sol, tab->s, &mprk->Y));
  if (mprk->is_slow) PetscCall(PetscMalloc1(tab->s, &mprk->work_slow));
  PetscCall(PetscMalloc1(tab->s, &mprk->work_slowbuffer));
  if (tab->np == 3) {
    if (mprk->is_medium) PetscCall(PetscMalloc1(tab->s, &mprk->work_medium));
    PetscCall(PetscMalloc1(tab->s, &mprk->work_mediumbuffer));
  }
  PetscCall(PetscMalloc1(tab->s, &mprk->work_fast));

  if (ts->use_splitrhsfunction) {
    if (mprk->is_slow) {
      PetscCall(VecGetSubVector(ts->vec_sol, mprk->is_slow, &YdotRHS_slow));
      PetscCall(VecDuplicateVecs(YdotRHS_slow, tab->s, &mprk->YdotRHS_slow));
      PetscCall(VecRestoreSubVector(ts->vec_sol, mprk->is_slow, &YdotRHS_slow));
    }
    PetscCall(VecGetSubVector(ts->vec_sol, mprk->is_slowbuffer, &YdotRHS_slowbuffer));
    PetscCall(VecDuplicateVecs(YdotRHS_slowbuffer, tab->s, &mprk->YdotRHS_slowbuffer));
    PetscCall(VecRestoreSubVector(ts->vec_sol, mprk->is_slowbuffer, &YdotRHS_slowbuffer));
    if (tab->np == 3) {
      if (mprk->is_medium) {
        PetscCall(VecGetSubVector(ts->vec_sol, mprk->is_medium, &YdotRHS_medium));
        PetscCall(VecDuplicateVecs(YdotRHS_medium, tab->s, &mprk->YdotRHS_medium));
        PetscCall(VecRestoreSubVector(ts->vec_sol, mprk->is_medium, &YdotRHS_medium));
      }
      PetscCall(VecGetSubVector(ts->vec_sol, mprk->is_mediumbuffer, &YdotRHS_mediumbuffer));
      PetscCall(VecDuplicateVecs(YdotRHS_mediumbuffer, tab->s, &mprk->YdotRHS_mediumbuffer));
      PetscCall(VecRestoreSubVector(ts->vec_sol, mprk->is_mediumbuffer, &YdotRHS_mediumbuffer));
    }
    PetscCall(VecGetSubVector(ts->vec_sol, mprk->is_fast, &YdotRHS_fast));
    PetscCall(VecDuplicateVecs(YdotRHS_fast, tab->s, &mprk->YdotRHS_fast));
    PetscCall(VecRestoreSubVector(ts->vec_sol, mprk->is_fast, &YdotRHS_fast));
  } else {
    PetscCall(VecDuplicateVecs(ts->vec_sol, tab->s, &mprk->YdotRHS));
    if (mprk->is_slow) PetscCall(PetscMalloc1(tab->s, &mprk->YdotRHS_slow));
    PetscCall(PetscMalloc1(tab->s, &mprk->YdotRHS_slowbuffer));
    if (tab->np == 3) {
      if (mprk->is_medium) PetscCall(PetscMalloc1(tab->s, &mprk->YdotRHS_medium));
      PetscCall(PetscMalloc1(tab->s, &mprk->YdotRHS_mediumbuffer));
    }
    PetscCall(PetscMalloc1(tab->s, &mprk->YdotRHS_fast));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TSSetUp_MPRK(TS ts)
{
  TS_MPRK    *mprk = (TS_MPRK *)ts->data;
  MPRKTableau tab  = mprk->tableau;
  DM          dm;

  PetscFunctionBegin;
  PetscCall(TSRHSSplitGetIS(ts, "slow", &mprk->is_slow));
  PetscCall(TSRHSSplitGetIS(ts, "fast", &mprk->is_fast));
  PetscCheck(mprk->is_slow && mprk->is_fast, PetscObjectComm((PetscObject)ts), PETSC_ERR_USER, "Must set up RHSSplits with TSRHSSplitSetIS() using split names 'slow' and 'fast' respectively in order to use the method '%s'", tab->name);

  if (tab->np == 3) {
    PetscCall(TSRHSSplitGetIS(ts, "medium", &mprk->is_medium));
    PetscCheck(mprk->is_medium, PetscObjectComm((PetscObject)ts), PETSC_ERR_USER, "Must set up RHSSplits with TSRHSSplitSetIS() using split names 'slow' and 'medium' and 'fast' respectively in order to use the method '%s'", tab->name);
    PetscCall(TSRHSSplitGetIS(ts, "mediumbuffer", &mprk->is_mediumbuffer));
    if (!mprk->is_mediumbuffer) { /* let medium buffer cover whole medium region */
      mprk->is_mediumbuffer = mprk->is_medium;
      mprk->is_medium       = NULL;
    }
  }

  /* If users do not provide buffer region settings, the solver will do them automatically, but with a performance penalty */
  PetscCall(TSRHSSplitGetIS(ts, "slowbuffer", &mprk->is_slowbuffer));
  if (!mprk->is_slowbuffer) { /* let slow buffer cover whole slow region */
    mprk->is_slowbuffer = mprk->is_slow;
    mprk->is_slow       = NULL;
  }
  PetscCall(TSCheckImplicitTerm(ts));
  PetscCall(TSMPRKTableauSetUp(ts));
  PetscCall(TSGetDM(ts, &dm));
  PetscCall(DMCoarsenHookAdd(dm, DMCoarsenHook_TSMPRK, DMRestrictHook_TSMPRK, ts));
  PetscCall(DMSubDomainHookAdd(dm, DMSubDomainHook_TSMPRK, DMSubDomainRestrictHook_TSMPRK, ts));
  if (ts->use_splitrhsfunction) {
    ts->ops->step         = TSStep_MPRKSPLIT;
    ts->ops->evaluatestep = TSEvaluateStep_MPRKSPLIT;
    PetscCall(TSMPRKSetSplits(ts));
  } else {
    ts->ops->step         = TSStep_MPRK;
    ts->ops->evaluatestep = TSEvaluateStep_MPRK;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TSSetFromOptions_MPRK(TS ts, PetscOptionItems *PetscOptionsObject)
{
  TS_MPRK *mprk = (TS_MPRK *)ts->data;

  PetscFunctionBegin;
  PetscOptionsHeadBegin(PetscOptionsObject, "PRK ODE solver options");
  {
    MPRKTableauLink link;
    PetscInt        count, choice;
    PetscBool       flg;
    const char    **namelist;
    for (link = MPRKTableauList, count = 0; link; link = link->next, count++)
      ;
    PetscCall(PetscMalloc1(count, (char ***)&namelist));
    for (link = MPRKTableauList, count = 0; link; link = link->next, count++) namelist[count] = link->tab.name;
    PetscCall(PetscOptionsEList("-ts_mprk_type", "Family of MPRK method", "TSMPRKSetType", (const char *const *)namelist, count, mprk->tableau->name, &choice, &flg));
    if (flg) PetscCall(TSMPRKSetType(ts, namelist[choice]));
    PetscCall(PetscFree(namelist));
  }
  PetscOptionsHeadEnd();
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TSView_MPRK(TS ts, PetscViewer viewer)
{
  TS_MPRK  *mprk = (TS_MPRK *)ts->data;
  PetscBool iascii;

  PetscFunctionBegin;
  PetscCall(PetscObjectTypeCompare((PetscObject)viewer, PETSCVIEWERASCII, &iascii));
  if (iascii) {
    MPRKTableau tab = mprk->tableau;
    TSMPRKType  mprktype;
    char        fbuf[512];
    char        sbuf[512];
    PetscInt    i;
    PetscCall(TSMPRKGetType(ts, &mprktype));
    PetscCall(PetscViewerASCIIPrintf(viewer, "  MPRK type %s\n", mprktype));
    PetscCall(PetscViewerASCIIPrintf(viewer, "  Order: %" PetscInt_FMT "\n", tab->order));

    PetscCall(PetscFormatRealArray(fbuf, sizeof(fbuf), "% 8.6f", tab->s, tab->cf));
    PetscCall(PetscViewerASCIIPrintf(viewer, "  Abscissa cf = %s\n", fbuf));
    PetscCall(PetscViewerASCIIPrintf(viewer, "  Af = \n"));
    for (i = 0; i < tab->s; i++) {
      PetscCall(PetscFormatRealArray(fbuf, sizeof(fbuf), "% 8.6f", tab->s, &tab->Af[i * tab->s]));
      PetscCall(PetscViewerASCIIPrintf(viewer, "    %s\n", fbuf));
    }
    PetscCall(PetscFormatRealArray(fbuf, sizeof(fbuf), "% 8.6f", tab->s, tab->bf));
    PetscCall(PetscViewerASCIIPrintf(viewer, "  bf = %s\n", fbuf));

    PetscCall(PetscFormatRealArray(sbuf, sizeof(sbuf), "% 8.6f", tab->s, tab->csb));
    PetscCall(PetscViewerASCIIPrintf(viewer, "  Abscissa csb = %s\n", sbuf));
    PetscCall(PetscViewerASCIIPrintf(viewer, "  Asb = \n"));
    for (i = 0; i < tab->s; i++) {
      PetscCall(PetscFormatRealArray(sbuf, sizeof(sbuf), "% 8.6f", tab->s, &tab->Asb[i * tab->s]));
      PetscCall(PetscViewerASCIIPrintf(viewer, "    %s\n", sbuf));
    }
    PetscCall(PetscFormatRealArray(sbuf, sizeof(sbuf), "% 8.6f", tab->s, tab->bsb));
    PetscCall(PetscViewerASCIIPrintf(viewer, "  bsb = %s\n", sbuf));

    if (tab->np == 3) {
      char mbuf[512];
      PetscCall(PetscFormatRealArray(mbuf, sizeof(mbuf), "% 8.6f", tab->s, tab->cmb));
      PetscCall(PetscViewerASCIIPrintf(viewer, "  Abscissa cmb = %s\n", mbuf));
      PetscCall(PetscViewerASCIIPrintf(viewer, "  Amb = \n"));
      for (i = 0; i < tab->s; i++) {
        PetscCall(PetscFormatRealArray(mbuf, sizeof(mbuf), "% 8.6f", tab->s, &tab->Amb[i * tab->s]));
        PetscCall(PetscViewerASCIIPrintf(viewer, "    %s\n", mbuf));
      }
      PetscCall(PetscFormatRealArray(mbuf, sizeof(mbuf), "% 8.6f", tab->s, tab->bmb));
      PetscCall(PetscViewerASCIIPrintf(viewer, "  bmb = %s\n", mbuf));
    }
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TSLoad_MPRK(TS ts, PetscViewer viewer)
{
  TSAdapt adapt;

  PetscFunctionBegin;
  PetscCall(TSGetAdapt(ts, &adapt));
  PetscCall(TSAdaptLoad(adapt, viewer));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  TSMPRKSetType - Set the type of `TSMPRK` scheme

  Not Collective

  Input Parameters:
+  ts - timestepping context
-  mprktype - type of `TSMPRK` scheme

  Options Database Key:
.   -ts_mprk_type - <pm2,p2,p3> - select the specific scheme

  Level: intermediate

.seealso: [](chapter_ts), `TSMPRKGetType()`, `TSMPRK`, `TSMPRKType`
@*/
PetscErrorCode TSMPRKSetType(TS ts, TSMPRKType mprktype)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts, TS_CLASSID, 1);
  PetscValidCharPointer(mprktype, 2);
  PetscTryMethod(ts, "TSMPRKSetType_C", (TS, TSMPRKType), (ts, mprktype));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  TSMPRKGetType - Get the type of `TSMPRK` scheme

  Not Collective

  Input Parameter:
.  ts - timestepping context

  Output Parameter:
.  mprktype - type of `TSMPRK` scheme

  Level: intermediate

.seealso: [](chapter_ts), `TSMPRKGetType()`, `TSMPRK`
@*/
PetscErrorCode TSMPRKGetType(TS ts, TSMPRKType *mprktype)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts, TS_CLASSID, 1);
  PetscUseMethod(ts, "TSMPRKGetType_C", (TS, TSMPRKType *), (ts, mprktype));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TSMPRKGetType_MPRK(TS ts, TSMPRKType *mprktype)
{
  TS_MPRK *mprk = (TS_MPRK *)ts->data;

  PetscFunctionBegin;
  *mprktype = mprk->tableau->name;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TSMPRKSetType_MPRK(TS ts, TSMPRKType mprktype)
{
  TS_MPRK        *mprk = (TS_MPRK *)ts->data;
  PetscBool       match;
  MPRKTableauLink link;

  PetscFunctionBegin;
  if (mprk->tableau) {
    PetscCall(PetscStrcmp(mprk->tableau->name, mprktype, &match));
    if (match) PetscFunctionReturn(PETSC_SUCCESS);
  }
  for (link = MPRKTableauList; link; link = link->next) {
    PetscCall(PetscStrcmp(link->tab.name, mprktype, &match));
    if (match) {
      if (ts->setupcalled) PetscCall(TSMPRKTableauReset(ts));
      mprk->tableau = &link->tab;
      if (ts->setupcalled) PetscCall(TSMPRKTableauSetUp(ts));
      PetscFunctionReturn(PETSC_SUCCESS);
    }
  }
  SETERRQ(PetscObjectComm((PetscObject)ts), PETSC_ERR_ARG_UNKNOWN_TYPE, "Could not find '%s'", mprktype);
}

static PetscErrorCode TSGetStages_MPRK(TS ts, PetscInt *ns, Vec **Y)
{
  TS_MPRK *mprk = (TS_MPRK *)ts->data;

  PetscFunctionBegin;
  *ns = mprk->tableau->s;
  if (Y) *Y = mprk->Y;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TSDestroy_MPRK(TS ts)
{
  PetscFunctionBegin;
  PetscCall(TSReset_MPRK(ts));
  if (ts->dm) {
    PetscCall(DMCoarsenHookRemove(ts->dm, DMCoarsenHook_TSMPRK, DMRestrictHook_TSMPRK, ts));
    PetscCall(DMSubDomainHookRemove(ts->dm, DMSubDomainHook_TSMPRK, DMSubDomainRestrictHook_TSMPRK, ts));
  }
  PetscCall(PetscFree(ts->data));
  PetscCall(PetscObjectComposeFunction((PetscObject)ts, "TSMPRKGetType_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)ts, "TSMPRKSetType_C", NULL));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*MC
      TSMPRK - ODE solver using Multirate Partitioned Runge-Kutta schemes

  The user should provide the right hand side of the equation using `TSSetRHSFunction()`.

  Level: beginner

  Note:
  The default is `TSMPRKPM2`, it can be changed with `TSMPRKSetType()` or -ts_mprk_type

.seealso: [](chapter_ts), `TSCreate()`, `TS`, `TSSetType()`, `TSMPRKSetType()`, `TSMPRKGetType()`, `TSMPRKType`, `TSMPRKRegister()`, `TSMPRKSetMultirateType()`
          `TSMPRKM2`, `TSMPRKM3`, `TSMPRKRFSMR3`, `TSMPRKRFSMR2`, `TSType`
M*/
PETSC_EXTERN PetscErrorCode TSCreate_MPRK(TS ts)
{
  TS_MPRK *mprk;

  PetscFunctionBegin;
  PetscCall(TSMPRKInitializePackage());

  ts->ops->reset          = TSReset_MPRK;
  ts->ops->destroy        = TSDestroy_MPRK;
  ts->ops->view           = TSView_MPRK;
  ts->ops->load           = TSLoad_MPRK;
  ts->ops->setup          = TSSetUp_MPRK;
  ts->ops->step           = TSStep_MPRK;
  ts->ops->evaluatestep   = TSEvaluateStep_MPRK;
  ts->ops->setfromoptions = TSSetFromOptions_MPRK;
  ts->ops->getstages      = TSGetStages_MPRK;

  PetscCall(PetscNew(&mprk));
  ts->data = (void *)mprk;

  PetscCall(PetscObjectComposeFunction((PetscObject)ts, "TSMPRKGetType_C", TSMPRKGetType_MPRK));
  PetscCall(PetscObjectComposeFunction((PetscObject)ts, "TSMPRKSetType_C", TSMPRKSetType_MPRK));

  PetscCall(TSMPRKSetType(ts, TSMPRKDefault));
  PetscFunctionReturn(PETSC_SUCCESS);
}
