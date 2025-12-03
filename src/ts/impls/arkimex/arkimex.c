/*
  Code for timestepping with additive Runge-Kutta IMEX method or Diagonally Implicit Runge-Kutta methods.

  Notes:
  For ARK, the general system is written as

  F(t,U,Udot) = G(t,U)

  where F represents the stiff part of the physics and G represents the non-stiff part.

*/
#include <petsc/private/tsimpl.h> /*I   "petscts.h"   I*/
#include <petscdm.h>
#include <../src/ts/impls/arkimex/arkimex.h>
#include <../src/ts/impls/arkimex/fsarkimex.h>

static ARKTableauLink ARKTableauList;
static TSARKIMEXType  TSARKIMEXDefault = TSARKIMEX3;
static TSDIRKType     TSDIRKDefault    = TSDIRKES213SAL;
static PetscBool      TSARKIMEXRegisterAllCalled;
static PetscBool      TSARKIMEXPackageInitialized;
static PetscErrorCode TSExtrapolate_ARKIMEX(TS, PetscReal, Vec);

/*MC
     TSARKIMEXARS122 - Second order ARK IMEX scheme, {cite}`ascher_1997`

     This method has one explicit stage and one implicit stage.

     Options Database Key:
.      -ts_arkimex_type ars122 - set arkimex type to ars122

     Level: advanced

.seealso: [](ch_ts), `TSARKIMEX`, `TSARKIMEXType`, `TSARKIMEXSetType()`
M*/

/*MC
     TSARKIMEXA2 - Second order ARK IMEX scheme with A-stable implicit part.

     This method has an explicit stage and one implicit stage, and has an A-stable implicit scheme. This method was provided by Emil Constantinescu.

     Options Database Key:
.      -ts_arkimex_type a2 - set arkimex type to a2

     Level: advanced

.seealso: [](ch_ts), `TSARKIMEX`, `TSARKIMEXType`, `TSARKIMEXSetType()`
M*/

/*MC
     TSARKIMEXL2 - Second order ARK IMEX scheme with L-stable implicit part, {cite}`pareschi_2005`

     This method has two implicit stages, and L-stable implicit scheme.

     Options Database Key:
.      -ts_arkimex_type l2 - set arkimex type to l2

     Level: advanced

.seealso: [](ch_ts), `TSARKIMEX`, `TSARKIMEXType`, `TSARKIMEXSetType()`
M*/

/*MC
     TSARKIMEX1BEE - First order backward Euler represented as an ARK IMEX scheme with extrapolation as error estimator. This is a 3-stage method.

     This method is aimed at starting the integration of implicit DAEs when explicit first-stage ARK methods are used.

     Options Database Key:
.      -ts_arkimex_type 1bee - set arkimex type to 1bee

     Level: advanced

.seealso: [](ch_ts), `TSARKIMEX`, `TSARKIMEXType`, `TSARKIMEXSetType()`
M*/

/*MC
     TSARKIMEX2C - Second order ARK IMEX scheme with L-stable implicit part.

     This method has one explicit stage and two implicit stages. The implicit part is the same as in TSARKIMEX2D and TSARKIMEX2E, but the explicit part has a larger stability region on the negative real axis. This method was provided by Emil Constantinescu.

     Options Database Key:
.      -ts_arkimex_type 2c - set arkimex type to 2c

     Level: advanced

.seealso: [](ch_ts), `TSARKIMEX`, `TSARKIMEXType`, `TSARKIMEXSetType()`
M*/

/*MC
     TSARKIMEX2D - Second order ARK IMEX scheme with L-stable implicit part.

     This method has one explicit stage and two implicit stages. The stability function is independent of the explicit part in the infinity limit of the implicit component. This method was provided by Emil Constantinescu.

     Options Database Key:
.      -ts_arkimex_type 2d - set arkimex type to 2d

     Level: advanced

.seealso: [](ch_ts), `TSARKIMEX`, `TSARKIMEXType`, `TSARKIMEXSetType()`
M*/

/*MC
     TSARKIMEX2E - Second order ARK IMEX scheme with L-stable implicit part.

     This method has one explicit stage and two implicit stages. It is an optimal method developed by Emil Constantinescu.

     Options Database Key:
.      -ts_arkimex_type 2e - set arkimex type to 2e

    Level: advanced

.seealso: [](ch_ts), `TSARKIMEX`, `TSARKIMEXType`, `TSARKIMEXSetType()`
M*/

/*MC
     TSARKIMEXPRSSP2 - Second order SSP ARK IMEX scheme, {cite}`pareschi_2005`

     This method has three implicit stages.

     This method is referred to as SSP2-(3,3,2) in <https://arxiv.org/abs/1110.4375>

     Options Database Key:
.      -ts_arkimex_type prssp2 - set arkimex type to prssp2

     Level: advanced

.seealso: [](ch_ts), `TSARKIMEX`, `TSARKIMEXType`, `TSARKIMEXSetType()`
M*/

/*MC
     TSARKIMEX3 - Third order ARK IMEX scheme with L-stable implicit part, {cite}`kennedy_2003`

     This method has one explicit stage and three implicit stages.

     Options Database Key:
.      -ts_arkimex_type 3 - set arkimex type to 3

     Level: advanced

.seealso: [](ch_ts), `TSARKIMEX`, `TSARKIMEXType`, `TSARKIMEXSetType()`
M*/

/*MC
     TSARKIMEXARS443 - Third order ARK IMEX scheme, {cite}`ascher_1997`

     This method has one explicit stage and four implicit stages.

     Options Database Key:
.      -ts_arkimex_type ars443 - set arkimex type to ars443

     Level: advanced

     Notes:
     This method is referred to as ARS(4,4,3) in <https://arxiv.org/abs/1110.4375>

.seealso: [](ch_ts), `TSARKIMEX`, `TSARKIMEXType`, `TSARKIMEXSetType()`
M*/

/*MC
     TSARKIMEXBPR3 - Third order ARK IMEX scheme. Referred to as ARK3 in <https://arxiv.org/abs/1110.4375>

     This method has one explicit stage and four implicit stages.

     Options Database Key:
.      -ts_arkimex_type bpr3 - set arkimex type to bpr3

     Level: advanced

.seealso: [](ch_ts), `TSARKIMEX`, `TSARKIMEXType`, `TSARKIMEXSetType()`
M*/

/*MC
     TSARKIMEX4 - Fourth order ARK IMEX scheme with L-stable implicit part, {cite}`kennedy_2003`.

     This method has one explicit stage and four implicit stages.

     Options Database Key:
.      -ts_arkimex_type 4 - set arkimex type to4

     Level: advanced

.seealso: [](ch_ts), `TSARKIMEX`, `TSARKIMEXType`, `TSARKIMEXSetType()`
M*/

/*MC
     TSARKIMEX5 - Fifth order ARK IMEX scheme with L-stable implicit part, {cite}`kennedy_2003`.

     This method has one explicit stage and five implicit stages.

     Options Database Key:
.      -ts_arkimex_type 5 - set arkimex type to 5

     Level: advanced

.seealso: [](ch_ts), `TSARKIMEX`, `TSARKIMEXType`, `TSARKIMEXSetType()`
M*/

/*MC
     TSDIRKS212 - Second order DIRK scheme.

     This method has two implicit stages with an embedded method of other 1.
     See `TSDIRK` for additional details.

     Options Database Key:
.      -ts_dirk_type s212 - select this method.

     Level: advanced

     Note:
     This is the default DIRK scheme in SUNDIALS.

.seealso: [](ch_ts), `TSDIRK`, `TSDIRKType`, `TSDIRKSetType()`
M*/

/*MC
     TSDIRKES122SAL - First order DIRK scheme <https://arxiv.org/abs/1803.01613>

     Uses backward Euler as advancing method and trapezoidal rule as embedded method. See `TSDIRK` for additional details.

     Options Database Key:
.      -ts_dirk_type es122sal - select this method.

     Level: advanced

.seealso: [](ch_ts), `TSDIRK`, `TSDIRKType`, `TSDIRKSetType()`
M*/

/*MC
     TSDIRKES213SAL - Second order DIRK scheme {cite}`kennedy2019diagonally`. Also known as TR-BDF2, see{cite}`hosea1996analysis`

     See `TSDIRK` for additional details.

     Options Database Key:
.      -ts_dirk_type es213sal - select this method.

     Level: advanced

     Note:
     This is the default DIRK scheme used in PETSc.

.seealso: [](ch_ts), `TSDIRK`, `TSDIRKType`, `TSDIRKSetType()`
M*/

/*MC
     TSDIRKES324SAL - Third order DIRK scheme, {cite}`kennedy2019diagonally`

     See `TSDIRK` for additional details.

     Options Database Key:
.      -ts_dirk_type es324sal - select this method.

     Level: advanced

.seealso: [](ch_ts), `TSDIRK`, `TSDIRKType`, `TSDIRKSetType()`
M*/

/*MC
     TSDIRKES325SAL - Third order DIRK scheme {cite}`kennedy2019diagonally`.

     See `TSDIRK` for additional details.

     Options Database Key:
.      -ts_dirk_type es325sal - select this method.

     Level: advanced

.seealso: [](ch_ts), `TSDIRK`, `TSDIRKType`, `TSDIRKSetType()`
M*/

/*MC
     TSDIRK657A - Sixth order DIRK scheme <https://github.com/yousefalamri55/High_Order_DIRK_Methods_Coeffs>

     See `TSDIRK` for additional details.

     Options Database Key:
.      -ts_dirk_type 657a - select this method.

     Level: advanced

.seealso: [](ch_ts), `TSDIRK`, `TSDIRKType`, `TSDIRKSetType()`
M*/

/*MC
     TSDIRKES648SA - Sixth order DIRK scheme <https://github.com/yousefalamri55/High_Order_DIRK_Methods_Coeffs>

     See `TSDIRK` for additional details.

     Options Database Key:
.      -ts_dirk_type es648sa - select this method.

     Level: advanced

.seealso: [](ch_ts), `TSDIRK`, `TSDIRKType`, `TSDIRKSetType()`
M*/

/*MC
     TSDIRK658A - Sixth order DIRK scheme  <https://github.com/yousefalamri55/High_Order_DIRK_Methods_Coeffs>

     See `TSDIRK` for additional details.

     Options Database Key:
.      -ts_dirk_type 658a - select this method.

     Level: advanced

.seealso: [](ch_ts), `TSDIRK`, `TSDIRKType`, `TSDIRKSetType()`
M*/

/*MC
     TSDIRKS659A - Sixth order DIRK scheme  <https://github.com/yousefalamri55/High_Order_DIRK_Methods_Coeffs>

     See `TSDIRK` for additional details.

     Options Database Key:
.      -ts_dirk_type s659a - select this method.

     Level: advanced

.seealso: [](ch_ts), `TSDIRK`, `TSDIRKType`, `TSDIRKSetType()`
M*/

/*MC
     TSDIRK7510SAL - Seventh order DIRK scheme <https://github.com/yousefalamri55/High_Order_DIRK_Methods_Coeffs>

     See `TSDIRK` for additional details.

     Options Database Key:
.      -ts_dirk_type 7510sal - select this method.

     Level: advanced

.seealso: [](ch_ts), `TSDIRK`, `TSDIRKType`, `TSDIRKSetType()`
M*/

/*MC
     TSDIRKES7510SA - Seventh order DIRK scheme <https://github.com/yousefalamri55/High_Order_DIRK_Methods_Coeffs>

     See `TSDIRK` for additional details.

     Options Database Key:
.      -ts_dirk_type es7510sa - select this method.

     Level: advanced

.seealso: [](ch_ts), `TSDIRK`, `TSDIRKType`, `TSDIRKSetType()`
M*/

/*MC
     TSDIRK759A - Seventh order DIRK scheme <https://github.com/yousefalamri55/High_Order_DIRK_Methods_Coeffs>

     See `TSDIRK` for additional details.

     Options Database Key:
.      -ts_dirk_type 759a - select this method.

     Level: advanced

.seealso: [](ch_ts), `TSDIRK`, `TSDIRKType`, `TSDIRKSetType()`
M*/

/*MC
     TSDIRKS7511SAL - Seventh order DIRK scheme <https://github.com/yousefalamri55/High_Order_DIRK_Methods_Coeffs>

     See `TSDIRK` for additional details.

     Options Database Key:
.      -ts_dirk_type s7511sal - select this method.

     Level: advanced

.seealso: [](ch_ts), `TSDIRK`, `TSDIRKType`, `TSDIRKSetType()`
M*/

/*MC
     TSDIRK8614A - Eighth order DIRK scheme <https://github.com/yousefalamri55/High_Order_DIRK_Methods_Coeffs>

     See `TSDIRK` for additional details.

     Options Database Key:
.      -ts_dirk_type 8614a - select this method.

     Level: advanced

.seealso: [](ch_ts), `TSDIRK`, `TSDIRKType`, `TSDIRKSetType()`
M*/

/*MC
     TSDIRK8616SAL - Eighth order DIRK scheme <https://github.com/yousefalamri55/High_Order_DIRK_Methods_Coeffs>

     See `TSDIRK` for additional details.

     Options Database Key:
.      -ts_dirk_type 8616sal - select this method.

     Level: advanced

.seealso: [](ch_ts), `TSDIRK`, `TSDIRKType`, `TSDIRKSetType()`
M*/

/*MC
     TSDIRKES8516SAL - Eighth order DIRK scheme <https://github.com/yousefalamri55/High_Order_DIRK_Methods_Coeffs>

     See `TSDIRK` for additional details.

     Options Database Key:
.      -ts_dirk_type es8516sal - select this method.

     Level: advanced

.seealso: [](ch_ts), `TSDIRK`, `TSDIRKType`, `TSDIRKSetType()`
M*/

PetscErrorCode TSHasRHSFunction(TS ts, PetscBool *has)
{
  TSRHSFunctionFn *func;

  PetscFunctionBegin;
  *has = PETSC_FALSE;
  PetscCall(DMTSGetRHSFunction(ts->dm, &func, NULL));
  if (func) *has = PETSC_TRUE;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  TSARKIMEXRegisterAll - Registers all of the additive Runge-Kutta implicit-explicit methods in `TSARKIMEX`

  Not Collective, but should be called by all processes which will need the schemes to be registered

  Level: advanced

.seealso: [](ch_ts), `TS`, `TSARKIMEX`, `TSARKIMEXRegisterDestroy()`
@*/
PetscErrorCode TSARKIMEXRegisterAll(void)
{
  PetscFunctionBegin;
  if (TSARKIMEXRegisterAllCalled) PetscFunctionReturn(PETSC_SUCCESS);
  TSARKIMEXRegisterAllCalled = PETSC_TRUE;

#define RC  PetscRealConstant
#define s2  RC(1.414213562373095048802)  /* PetscSqrtReal((PetscReal)2.0) */
#define us2 RC(0.2928932188134524755992) /* 1.0-1.0/PetscSqrtReal((PetscReal)2.0); */

  /* Diagonally implicit methods */
  {
    /* DIRK212, default of SUNDIALS */
    const PetscReal A[2][2] = {
      {RC(1.0),  RC(0.0)},
      {RC(-1.0), RC(1.0)}
    };
    const PetscReal b[2]      = {RC(0.5), RC(0.5)};
    const PetscReal bembed[2] = {RC(1.0), RC(0.0)};
    PetscCall(TSDIRKRegister(TSDIRKS212, 2, 2, &A[0][0], b, NULL, bembed, 1, b));
  }

  {
    /* ESDIRK12 from https://arxiv.org/pdf/1803.01613.pdf */
    const PetscReal A[2][2] = {
      {RC(0.0), RC(0.0)},
      {RC(0.0), RC(1.0)}
    };
    const PetscReal b[2]      = {RC(0.0), RC(1.0)};
    const PetscReal bembed[2] = {RC(0.5), RC(0.5)};
    PetscCall(TSDIRKRegister(TSDIRKES122SAL, 1, 2, &A[0][0], b, NULL, bembed, 1, b));
  }

  {
    /* ESDIRK213L[2]SA from KC16.
       TR-BDF2 from Hosea and Shampine
       ESDIRK23 in https://arxiv.org/pdf/1803.01613.pdf */
    const PetscReal A[3][3] = {
      {RC(0.0),      RC(0.0),      RC(0.0)},
      {us2,          us2,          RC(0.0)},
      {s2 / RC(4.0), s2 / RC(4.0), us2    },
    };
    const PetscReal b[3]      = {s2 / RC(4.0), s2 / RC(4.0), us2};
    const PetscReal bembed[3] = {(RC(1.0) - s2 / RC(4.0)) / RC(3.0), (RC(3.0) * s2 / RC(4.0) + RC(1.0)) / RC(3.0), us2 / RC(3.0)};
    PetscCall(TSDIRKRegister(TSDIRKES213SAL, 2, 3, &A[0][0], b, NULL, bembed, 1, b));
  }

  {
#define g   RC(0.43586652150845899941601945)
#define g2  PetscSqr(g)
#define g3  g *g2
#define g4  PetscSqr(g2)
#define g5  g *g4
#define c3  RC(1.0)
#define a32 c3 *(c3 - RC(2.0) * g) / (RC(4.0) * g)
#define b2  (-RC(2.0) + RC(3.0) * c3 + RC(6.0) * g * (RC(1.0) - c3)) / (RC(12.0) * g * (c3 - RC(2.0) * g))
#define b3  (RC(1.0) - RC(6.0) * g + RC(6.0) * g2) / (RC(3.0) * c3 * (c3 - RC(2.0) * g))
#if 0
/* This is for c3 = 3/5 */
  #define bh2 \
    c3 * (-RC(1.0) + RC(6.0) * g - RC(23.0) * g3 + RC(12.0) * g4 - RC(6.0) * g5) / (RC(4.0) * (RC(2.0) * g - c3) * (RC(1.0) - RC(6.0) * g + RC(6.0) * g2)) + (RC(3.0) - RC(27.0) * g + RC(68.0) * g2 - RC(55.0) * g3 + RC(21.0) * g4 - RC(6.0) * g5) / (RC(2.0) * (RC(2.0) * g - c3) * (RC(1.0) - RC(6.0) * g + RC(6.0) * g2))
  #define bh3 -g * (-RC(2.0) + RC(21.0) * g - RC(68.0) * g2 + RC(79.0) * g3 - RC(33.0) * g4 + RC(12.0) * g5) / (RC(2.0) * (RC(2.0) * g - c3) * (RC(1.0) - RC(6.0) * g + RC(6.0) * g2))
  #define bh4 -RC(3.0) * g2 * (-RC(1.0) + RC(4.0) * g - RC(2.0) * g2 + g3) / (RC(1.0) - RC(6.0) * g + RC(6.0) * g2)
#else
  /* This is for c3 = 1.0 */
  #define bh2 a32
  #define bh3 g
  #define bh4 RC(0.0)
#endif
    /* ESDIRK3(2I)4L[2]SA from KC16 with c3 = 1.0 */
    /* Given by Kvaerno https://link.springer.com/article/10.1023/b:bitn.0000046811.70614.38 */
    const PetscReal A[4][4] = {
      {RC(0.0),               RC(0.0), RC(0.0), RC(0.0)},
      {g,                     g,       RC(0.0), RC(0.0)},
      {c3 - a32 - g,          a32,     g,       RC(0.0)},
      {RC(1.0) - b2 - b3 - g, b2,      b3,      g      },
    };
    const PetscReal b[4]      = {RC(1.0) - b2 - b3 - g, b2, b3, g};
    const PetscReal bembed[4] = {RC(1.0) - bh2 - bh3 - bh4, bh2, bh3, bh4};
    PetscCall(TSDIRKRegister(TSDIRKES324SAL, 3, 4, &A[0][0], b, NULL, bembed, 1, b));
#undef g
#undef g2
#undef g3
#undef c3
#undef a32
#undef b2
#undef b3
#undef bh2
#undef bh3
#undef bh4
  }

  {
    /* ESDIRK3(2I)5L[2]SA from KC16 */
    const PetscReal A[5][5] = {
      {RC(0.0),                  RC(0.0),                  RC(0.0),                 RC(0.0),                   RC(0.0)           },
      {RC(9.0) / RC(40.0),       RC(9.0) / RC(40.0),       RC(0.0),                 RC(0.0),                   RC(0.0)           },
      {RC(19.0) / RC(72.0),      RC(14.0) / RC(45.0),      RC(9.0) / RC(40.0),      RC(0.0),                   RC(0.0)           },
      {RC(3337.0) / RC(11520.0), RC(233.0) / RC(720.0),    RC(207.0) / RC(1280.0),  RC(9.0) / RC(40.0),        RC(0.0)           },
      {RC(7415.0) / RC(34776.0), RC(9920.0) / RC(30429.0), RC(4845.0) / RC(9016.0), -RC(5827.0) / RC(19320.0), RC(9.0) / RC(40.0)},
    };
    const PetscReal b[5]      = {RC(7415.0) / RC(34776.0), RC(9920.0) / RC(30429.0), RC(4845.0) / RC(9016.0), -RC(5827.0) / RC(19320.0), RC(9.0) / RC(40.0)};
    const PetscReal bembed[5] = {RC(23705.0) / RC(104328.0), RC(29720.0) / RC(91287.0), RC(4225.0) / RC(9016.0), -RC(69304987.0) / RC(337732920.0), RC(42843.0) / RC(233080.0)};
    PetscCall(TSDIRKRegister(TSDIRKES325SAL, 3, 5, &A[0][0], b, NULL, bembed, 1, b));
  }

  {
    // DIRK(6,6)[1]A[(7,5)A] from https://github.com/yousefalamri55/High_Order_DIRK_Methods_Coeffs
    const PetscReal A[7][7] = {
      {RC(0.303487844706747),    RC(0.0),                RC(0.0),                   RC(0.0),                   RC(0.0),                 RC(0.0),                RC(0.0)              },
      {RC(-0.279756492709814),   RC(0.500032236020747),  RC(0.0),                   RC(0.0),                   RC(0.0),                 RC(0.0),                RC(0.0)              },
      {RC(0.280583215743895),    RC(-0.438560061586751), RC(0.217250734515736),     RC(0.0),                   RC(0.0),                 RC(0.0),                RC(0.0)              },
      {RC(-0.0677678738539846),  RC(0.984312781232293),  RC(-0.266720192540149),    RC(0.2476680834526),       RC(0.0),                 RC(0.0),                RC(0.0)              },
      {RC(0.125671616147993),    RC(-0.995401751002415), RC(0.761333109549059),     RC(-0.210281837202208),    RC(0.866743712636936),   RC(0.0),                RC(0.0)              },
      {RC(-0.368056238801488),   RC(-0.999928082701516), RC(0.534734253232519),     RC(-0.174856916279082),    RC(0.615007160285509),   RC(0.696549912132029),  RC(0.0)              },
      {RC(-0.00570546839653984), RC(-0.113110431835656), RC(-0.000965563207671587), RC(-0.000130490084629567), RC(0.00111737736895673), RC(-0.279385587378871), RC(0.618455906845342)}
    };
    const PetscReal b[7]      = {RC(0.257561510484877), RC(0.234281287047716), RC(0.126658904241469), RC(0.252363215441784), RC(0.396701083526306), RC(-0.267566000742152), RC(0.0)};
    const PetscReal bembed[7] = {RC(0.257561510484945), RC(0.387312822934391), RC(0.126658904241468), RC(0.252363215441784), RC(0.396701083526306), RC(-0.267566000742225), RC(-0.153031535886669)};
    PetscCall(TSDIRKRegister(TSDIRK657A, 6, 7, &A[0][0], b, NULL, bembed, 1, b));
  }
  {
    // ESDIRK(8,6)[2]SA[(8,4)] from https://github.com/yousefalamri55/High_Order_DIRK_Methods_Coeffs
    const PetscReal A[8][8] = {
      {RC(0.0),                RC(0.0),                 RC(0.0),                 RC(0.0),               RC(0.0),                RC(0.0),                RC(0.0),               RC(0.0)              },
      {RC(0.333222149217725),  RC(0.333222149217725),   RC(0.0),                 RC(0.0),               RC(0.0),                RC(0.0),                RC(0.0),               RC(0.0)              },
      {RC(0.0639743773182214), RC(-0.0830330224410214), RC(0.333222149217725),   RC(0.0),               RC(0.0),                RC(0.0),                RC(0.0),               RC(0.0)              },
      {RC(-0.728522201369326), RC(-0.210414479522485),  RC(0.532519916559342),   RC(0.333222149217725), RC(0.0),                RC(0.0),                RC(0.0),               RC(0.0)              },
      {RC(-0.175135269272067), RC(0.666675582067552),   RC(-0.304400907370867),  RC(0.656797712445756), RC(0.333222149217725),  RC(0.0),                RC(0.0),               RC(0.0)              },
      {RC(0.222695802705462),  RC(-0.0948971794681061), RC(-0.0234336346686545), RC(-0.45385925012042), RC(0.0283910313826958), RC(0.333222149217725),  RC(0.0),               RC(0.0)              },
      {RC(-0.132534078051299), RC(0.702597935004879),   RC(-0.433316453128078),  RC(0.893717488547587), RC(0.057381454791406),  RC(-0.207798411552402), RC(0.333222149217725), RC(0.0)              },
      {RC(0.0802253121418085), RC(0.281196044671022),   RC(0.406758926172157),   RC(-0.01945708512416), RC(-0.41785600088526),  RC(0.0545342658870322), RC(0.281376387919675), RC(0.333222149217725)}
    };
    const PetscReal b[8]      = {RC(0.0802253121418085), RC(0.281196044671022), RC(0.406758926172157), RC(-0.01945708512416), RC(-0.41785600088526), RC(0.0545342658870322), RC(0.281376387919675), RC(0.333222149217725)};
    const PetscReal bembed[8] = {RC(0.0), RC(0.292331064554014), RC(0.409676102283681), RC(-0.002094718084982), RC(-0.282771520835975), RC(0.113862336644901), RC(0.181973572260693), RC(0.287023163177669)};
    PetscCall(TSDIRKRegister(TSDIRKES648SA, 6, 8, &A[0][0], b, NULL, bembed, 1, b));
  }
  {
    // DIRK(8,6)[1]SAL[(8,5)A] from https://github.com/yousefalamri55/High_Order_DIRK_Methods_Coeffs
    const PetscReal A[8][8] = {
      {RC(0.477264457385826),    RC(0.0),                RC(0.0),                   RC(0.0),                 RC(0.0),                RC(0.0),                RC(0.0),                 RC(0.0)              },
      {RC(-0.197052588415002),   RC(0.476363428459584),  RC(0.0),                   RC(0.0),                 RC(0.0),                RC(0.0),                RC(0.0),                 RC(0.0)              },
      {RC(-0.0347674430372966),  RC(0.633051807335483),  RC(0.193634310075028),     RC(0.0),                 RC(0.0),                RC(0.0),                RC(0.0),                 RC(0.0)              },
      {RC(0.0967797668578702),   RC(-0.193533526466535), RC(-0.000207622945800473), RC(0.159572204849431),   RC(0.0),                RC(0.0),                RC(0.0),                 RC(0.0)              },
      {RC(0.162527231819875),    RC(-0.249672513547382), RC(-0.0459079972041795),   RC(0.36579476400859),    RC(0.255752838307699),  RC(0.0),                RC(0.0),                 RC(0.0)              },
      {RC(-0.00707603197171262), RC(0.846299854860295),  RC(0.344020016925018),     RC(-0.0720926054548865), RC(-0.215492331980875), RC(0.104341097622161),  RC(0.0),                 RC(0.0)              },
      {RC(0.00176857935179744),  RC(0.0779960013127515), RC(0.303333277564557),     RC(0.213160806732836),   RC(0.351769320319038),  RC(-0.381545894386538), RC(0.433517909105558),   RC(0.0)              },
      {RC(0.0),                  RC(0.22732353410559),   RC(0.308415837980118),     RC(0.157263419573007),   RC(0.243551137152275),  RC(-0.120953626732831), RC(-0.0802678473399899), RC(0.264667545261832)}
    };
    const PetscReal b[8]      = {RC(0.0), RC(0.22732353410559), RC(0.308415837980118), RC(0.157263419573007), RC(0.243551137152275), RC(-0.120953626732831), RC(-0.0802678473399899), RC(0.264667545261832)};
    const PetscReal bembed[8] = {RC(0.0), RC(0.22732353410559), RC(0.308415837980118), RC(0.157263419573007), RC(0.243551137152275), RC(-0.103483943222765), RC(-0.0103721771642262), RC(0.177302191576001)};
    PetscCall(TSDIRKRegister(TSDIRK658A, 6, 8, &A[0][0], b, NULL, bembed, 1, b));
  }
  {
    // SDIRK(9,6)[1]SAL[(9,5)A] from https://github.com/yousefalamri55/High_Order_DIRK_Methods_Coeffs
    const PetscReal A[9][9] = {
      {RC(0.218127781944908),   RC(0.0),                RC(0.0),                 RC(0.0),                 RC(0.0),                RC(0.0),                RC(0.0),                RC(0.0),                RC(0.0)              },
      {RC(-0.0903514856119419), RC(0.218127781944908),  RC(0.0),                 RC(0.0),                 RC(0.0),                RC(0.0),                RC(0.0),                RC(0.0),                RC(0.0)              },
      {RC(0.172952039138937),   RC(-0.35365501036282),  RC(0.218127781944908),   RC(0.0),                 RC(0.0),                RC(0.0),                RC(0.0),                RC(0.0),                RC(0.0)              },
      {RC(0.511999875919193),   RC(0.0289640332201925), RC(-0.0144030945657094), RC(0.218127781944908),   RC(0.0),                RC(0.0),                RC(0.0),                RC(0.0),                RC(0.0)              },
      {RC(0.00465303495506782), RC(-0.075635818766597), RC(0.217273030786712),   RC(-0.0206519428725472), RC(0.218127781944908),  RC(0.0),                RC(0.0),                RC(0.0),                RC(0.0)              },
      {RC(0.896145501762472),   RC(0.139267327700498),  RC(-0.186920979752805),  RC(0.0672971012371723),  RC(-0.350891963442176), RC(0.218127781944908),  RC(0.0),                RC(0.0),                RC(0.0)              },
      {RC(0.552959701885751),   RC(-0.439360579793662), RC(0.333704002325091),   RC(-0.0339426520778416), RC(-0.151947445912595), RC(0.0213825661026943), RC(0.218127781944908),  RC(0.0),                RC(0.0)              },
      {RC(0.631360374036476),   RC(0.724733619641466),  RC(-0.432170625425258),  RC(0.598611382182477),   RC(-0.709087197034345), RC(-0.483986685696934), RC(0.378391562905131),  RC(0.218127781944908),  RC(0.0)              },
      {RC(0.0),                 RC(-0.15504452530869),  RC(0.194518478660789),   RC(0.63515640279203),    RC(0.81172278664173),   RC(0.110736108691585),  RC(-0.495304692414479), RC(-0.319912341007872), RC(0.218127781944908)}
    };
    const PetscReal b[9]      = {RC(0.0), RC(-0.15504452530869), RC(0.194518478660789), RC(0.63515640279203), RC(0.81172278664173), RC(0.110736108691585), RC(-0.495304692414479), RC(-0.319912341007872), RC(0.218127781944908)};
    const PetscReal bembed[9] = {RC(3.62671059311602e-16), RC(0.0736615558278942), RC(0.103527397262229), RC(1.00247481935499), RC(0.361377289250057), RC(-0.785425929961365), RC(-0.0170499047960784), RC(0.296321252214769), RC(-0.0348864791524953)};
    PetscCall(TSDIRKRegister(TSDIRKS659A, 6, 9, &A[0][0], b, NULL, bembed, 1, b));
  }
  {
    // DIRK(10,7)[1]SAL[(10,5)A] from https://github.com/yousefalamri55/High_Order_DIRK_Methods_Coeffs
    const PetscReal A[10][10] = {
      {RC(0.233704632125264),   RC(0.0),                RC(0.0),                  RC(0.0),                  RC(0.0),                   RC(0.0),                  RC(0.0),                RC(0.0),                 RC(0.0),               RC(0.0)              },
      {RC(-0.0739324813149407), RC(0.200056838146104),  RC(0.0),                  RC(0.0),                  RC(0.0),                   RC(0.0),                  RC(0.0),                RC(0.0),                 RC(0.0),               RC(0.0)              },
      {RC(0.0943790344044812),  RC(0.264056067701605),  RC(0.133245202456465),    RC(0.0),                  RC(0.0),                   RC(0.0),                  RC(0.0),                RC(0.0),                 RC(0.0),               RC(0.0)              },
      {RC(0.269084810601201),   RC(-0.503479002548384), RC(-0.00486736469695022), RC(0.251518716213569),    RC(0.0),                   RC(0.0),                  RC(0.0),                RC(0.0),                 RC(0.0),               RC(0.0)              },
      {RC(0.145665801918423),   RC(0.204983170463176),  RC(0.407154634069484),    RC(-0.0121039135200389),  RC(0.190243622486334),     RC(0.0),                  RC(0.0),                RC(0.0),                 RC(0.0),               RC(0.0)              },
      {RC(0.985450198547345),   RC(0.806942652811456),  RC(-0.808130934167263),   RC(-0.669035819439391),   RC(0.0269384406756128),    RC(0.462144080607327),    RC(0.0),                RC(0.0),                 RC(0.0),               RC(0.0)              },
      {RC(0.163902957809563),   RC(0.228315094960095),  RC(0.0745971021260249),   RC(0.000509793400156559), RC(0.0166533681378294),    RC(-0.0229383879045797),  RC(0.103505486637336),  RC(0.0),                 RC(0.0),               RC(0.0)              },
      {RC(-0.162694156858437),  RC(0.0453478837428434), RC(0.997443481211424),    RC(0.200251514941093),    RC(-0.000161755458839048), RC(-0.0848134335980281),  RC(-0.36438666566666),  RC(0.158604420136055),   RC(0.0),               RC(0.0)              },
      {RC(0.200733156477425),   RC(0.239686443444433),  RC(0.303837014418929),    RC(-0.0534390596279896),  RC(0.0314067599640569),    RC(-0.00764032790448536), RC(0.0609191260198661), RC(-0.0736319201590642), RC(0.204602530607021), RC(0.0)              },
      {RC(0.0),                 RC(0.235563761744267),  RC(0.658651488684319),    RC(0.0308877804992098),   RC(-0.906514945595336),    RC(-0.0248488551739974),  RC(-0.309967582365257), RC(0.191663316925525),   RC(0.923933712199542), RC(0.200631323081727)}
    };
    const PetscReal b[10] = {RC(0.0), RC(0.235563761744267), RC(0.658651488684319), RC(0.0308877804992098), RC(-0.906514945595336), RC(-0.0248488551739974), RC(-0.309967582365257), RC(0.191663316925525), RC(0.923933712199542), RC(0.200631323081727)};
    const PetscReal bembed[10] =
      {RC(0.0), RC(0.222929376486581), RC(0.950668440138169), RC(0.0342694607044032), RC(0.362875840545746), RC(0.223572979288581), RC(-0.764361723526727), RC(0.563476909230026), RC(-0.690896961894185), RC(0.0974656790270323)};
    PetscCall(TSDIRKRegister(TSDIRK7510SAL, 7, 10, &A[0][0], b, NULL, bembed, 1, b));
  }
  {
    // ESDIRK(10,7)[2]SA[(10,5)] from https://github.com/yousefalamri55/High_Order_DIRK_Methods_Coeffs
    const PetscReal A[10][10] = {
      {RC(0.0),                 RC(0.0),                 RC(0.0),                 RC(0.0),                  RC(0.0),                  RC(0.0),                  RC(0.0),                RC(0.0),                 RC(0.0),               RC(0.0)              },
      {RC(0.210055790203419),   RC(0.210055790203419),   RC(0.0),                 RC(0.0),                  RC(0.0),                  RC(0.0),                  RC(0.0),                RC(0.0),                 RC(0.0),               RC(0.0)              },
      {RC(0.255781739921086),   RC(0.239850916980976),   RC(0.210055790203419),   RC(0.0),                  RC(0.0),                  RC(0.0),                  RC(0.0),                RC(0.0),                 RC(0.0),               RC(0.0)              },
      {RC(0.286789624880437),   RC(0.230494748834778),   RC(0.263925149885491),   RC(0.210055790203419),    RC(0.0),                  RC(0.0),                  RC(0.0),                RC(0.0),                 RC(0.0),               RC(0.0)              },
      {RC(-0.0219118128774335), RC(0.897684380345907),   RC(-0.657954605498907),  RC(0.124962304722633),    RC(0.210055790203419),    RC(0.0),                  RC(0.0),                RC(0.0),                 RC(0.0),               RC(0.0)              },
      {RC(-0.065614879584776),  RC(-0.0565630711859497), RC(0.0254881105065311),  RC(-0.00368981790650006), RC(-0.0115178258446329),  RC(0.210055790203419),    RC(0.0),                RC(0.0),                 RC(0.0),               RC(0.0)              },
      {RC(0.399860851232098),   RC(0.915588469718705),   RC(-0.0758429094934412), RC(-0.263369154872759),   RC(0.719687583564526),    RC(-0.787410407015369),   RC(0.210055790203419),  RC(0.0),                 RC(0.0),               RC(0.0)              },
      {RC(0.51693616104628),    RC(1.00000540846973),    RC(-0.0485110663289207), RC(-0.315208041581942),   RC(0.749742806451587),    RC(-0.990975090921248),   RC(0.0159279583407308), RC(0.210055790203419),   RC(0.0),               RC(0.0)              },
      {RC(-0.0303062129076945), RC(-0.297035174659034),  RC(0.184724697462164),   RC(-0.0351876079516183),  RC(-0.00324668230690761), RC(0.216151004053531),    RC(-0.126676252098317), RC(0.114040254365262),   RC(0.210055790203419), RC(0.0)              },
      {RC(0.0705997961586714),  RC(-0.0281516061956374), RC(0.314600470734633),   RC(-0.0907057557963371),  RC(0.168078953957742),    RC(-0.00655694984590575), RC(0.0505384497804303), RC(-0.0569572058725042), RC(0.368498056875488), RC(0.210055790203419)}
    };
    const PetscReal b[10]      = {RC(0.0705997961586714),   RC(-0.0281516061956374), RC(0.314600470734633),   RC(-0.0907057557963371), RC(0.168078953957742),
                                  RC(-0.00655694984590575), RC(0.0505384497804303),  RC(-0.0569572058725042), RC(0.368498056875488),   RC(0.210055790203419)};
    const PetscReal bembed[10] = {RC(-0.015494246543626), RC(0.167657963820093), RC(0.269858958144236),  RC(-0.0443258997755156), RC(0.150049236875266),
                                  RC(0.259452082755846),  RC(0.244624573502521), RC(-0.215528446920284), RC(0.0487601760292619),  RC(0.134945602112201)};
    PetscCall(TSDIRKRegister(TSDIRKES7510SA, 7, 10, &A[0][0], b, NULL, bembed, 1, b));
  }
  {
    // DIRK(9,7)[1]A[(9,5)A] from https://github.com/yousefalamri55/High_Order_DIRK_Methods_Coeffs
    const PetscReal A[9][9] = {
      {RC(0.179877789855839),   RC(0.0),                 RC(0.0),                RC(0.0),                  RC(0.0),                RC(0.0),                RC(0.0),                RC(0.0),                RC(0.0)               },
      {RC(-0.100405844885157),  RC(0.214948590644819),   RC(0.0),                RC(0.0),                  RC(0.0),                RC(0.0),                RC(0.0),                RC(0.0),                RC(0.0)               },
      {RC(0.112251360198995),   RC(-0.206162139150298),  RC(0.125159642941958),  RC(0.0),                  RC(0.0),                RC(0.0),                RC(0.0),                RC(0.0),                RC(0.0)               },
      {RC(-0.0335164000768257), RC(0.999942349946143),   RC(-0.491470853833294), RC(0.19820086325566),     RC(0.0),                RC(0.0),                RC(0.0),                RC(0.0),                RC(0.0)               },
      {RC(-0.0417345265478321), RC(0.187864510308215),   RC(0.0533789224305102), RC(-0.00822060284862916), RC(0.127670843671646),  RC(0.0),                RC(0.0),                RC(0.0),                RC(0.0)               },
      {RC(-0.0278257925239257), RC(0.600979340683382),   RC(-0.242632273241134), RC(-0.11318753652081),    RC(0.164326917632931),  RC(0.284116597781395),  RC(0.0),                RC(0.0),                RC(0.0)               },
      {RC(0.041465583858922),   RC(0.429657872601836),   RC(-0.381323410582524), RC(0.391934277498434),    RC(-0.245918275501241), RC(-0.35960669741231),  RC(0.184000022289158),  RC(0.0),                RC(0.0)               },
      {RC(-0.105565651574538),  RC(-0.0557833155018609), RC(0.358967568942643),  RC(-0.13489263413921),    RC(0.129553247260677),  RC(0.0992493795371489), RC(-0.15716610563461),  RC(0.17918862279814),   RC(0.0)               },
      {RC(0.00439696079965225), RC(0.960250486570491),   RC(0.143558372286706),  RC(0.0819015241056593),   RC(0.999562318563625),  RC(0.325203439314358),  RC(-0.679013149331228), RC(-0.990589559837246), RC(0.0773648037639896)}
    };

    const PetscReal b[9]      = {RC(0.0), RC(0.179291520437966), RC(0.115310295273026), RC(-0.857943261453138), RC(0.654911318641998), RC(1.18713633508094), RC(-0.0949482361570542), RC(-0.37661430946407), RC(0.19285633764033)};
    const PetscReal bembed[9] = {RC(0.0), RC(0.1897135479408), RC(0.127461414808862), RC(-0.835810807663404), RC(0.665114177777166), RC(1.16481046518346), RC(-0.11661858889792), RC(-0.387303251022099), RC(0.192633041873135)};
    PetscCall(TSDIRKRegister(TSDIRK759A, 7, 9, &A[0][0], b, NULL, bembed, 1, b));
  }
  {
    // SDIRK(11,7)[1]SAL[(11,5)A] from https://github.com/yousefalamri55/High_Order_DIRK_Methods_Coeffs
    const PetscReal A[11][11] = {
      {RC(0.200252661187742),  RC(0.0),                 RC(0.0),                  RC(0.0),                 RC(0.0),                 RC(0.0),                 RC(0.0),                 RC(0.0),                 RC(0.0),               RC(0.0),               RC(0.0)              },
      {RC(-0.082947368165267), RC(0.200252661187742),   RC(0.0),                  RC(0.0),                 RC(0.0),                 RC(0.0),                 RC(0.0),                 RC(0.0),                 RC(0.0),               RC(0.0),               RC(0.0)              },
      {RC(0.483452690540751),  RC(0.0),                 RC(0.200252661187742),    RC(0.0),                 RC(0.0),                 RC(0.0),                 RC(0.0),                 RC(0.0),                 RC(0.0),               RC(0.0),               RC(0.0)              },
      {RC(0.771076453481321),  RC(-0.22936926341842),   RC(0.289733373208823),    RC(0.200252661187742),   RC(0.0),                 RC(0.0),                 RC(0.0),                 RC(0.0),                 RC(0.0),               RC(0.0),               RC(0.0)              },
      {RC(0.0329683054968892), RC(-0.162397421903366),  RC(0.000951777538562805), RC(0.0),                 RC(0.200252661187742),   RC(0.0),                 RC(0.0),                 RC(0.0),                 RC(0.0),               RC(0.0),               RC(0.0)              },
      {RC(0.265888743485945),  RC(0.606743151103931),   RC(0.173443800537369),    RC(-0.0433968261546912), RC(-0.385211017224481),  RC(0.200252661187742),   RC(0.0),                 RC(0.0),                 RC(0.0),               RC(0.0),               RC(0.0)              },
      {RC(0.220662294551146),  RC(-0.0465078507657608), RC(-0.0333111995282464),  RC(0.011801580836998),   RC(0.169480801030105),   RC(-0.0167974432139385), RC(0.200252661187742),   RC(0.0),                 RC(0.0),               RC(0.0),               RC(0.0)              },
      {RC(0.323099728365267),  RC(0.0288371831672575),  RC(-0.0543404318773196),  RC(0.0137765831431662),  RC(0.0516799019060702),  RC(-0.0421359763835713), RC(0.181297932037826),   RC(0.200252661187742),   RC(0.0),               RC(0.0),               RC(0.0)              },
      {RC(-0.164226696476538), RC(0.187552004946792),   RC(0.0628674420973025),   RC(-0.0108886582703428), RC(-0.0117628641717889), RC(0.0432176880867965),  RC(-0.0315206836275473), RC(-0.0846007021638797), RC(0.200252661187742), RC(0.0),               RC(0.0)              },
      {RC(0.651428598623771),  RC(-0.10208078475356),   RC(0.198305701801888),    RC(-0.0117354096673789), RC(-0.0440385966743686), RC(-0.0358364455795087), RC(-0.0075408087654097), RC(0.160320941654639),   RC(0.017940248694499), RC(0.200252661187742), RC(0.0)              },
      {RC(0.0),                RC(-0.266259448580236),  RC(-0.615982357748271),   RC(0.561474126687165),   RC(0.266911112787025),   RC(0.219775952207137),   RC(0.387847665451514),   RC(0.612483137773236),   RC(0.330027015806089), RC(-0.6965298655714),  RC(0.200252661187742)}
    };
    const PetscReal b[11] =
      {RC(0.0), RC(-0.266259448580236), RC(-0.615982357748271), RC(0.561474126687165), RC(0.266911112787025), RC(0.219775952207137), RC(0.387847665451514), RC(0.612483137773236), RC(0.330027015806089), RC(-0.6965298655714), RC(0.200252661187742)};
    const PetscReal bembed[11] =
      {RC(0.0), RC(0.180185524442613), RC(-0.628869710835338), RC(0.186185675988647), RC(0.0484716652630425), RC(0.203927720607141), RC(0.44041662512573), RC(0.615710527731245), RC(0.0689648839032607), RC(-0.253599870605903), RC(0.138606958379488)};
    PetscCall(TSDIRKRegister(TSDIRKS7511SAL, 7, 11, &A[0][0], b, NULL, bembed, 1, b));
  }
  {
    // DIRK(13,8)[1]A[(14,6)A] from https://github.com/yousefalamri55/High_Order_DIRK_Methods_Coeffs
    const PetscReal A[14][14] = {
      {RC(0.421050745442291),   RC(0.0),                RC(0.0),                 RC(0.0),                  RC(0.0),                  RC(0.0),                RC(0.0),                 RC(0.0),                  RC(0.0),                RC(0.0),                 RC(0.0),                RC(0.0),                RC(0.0),                RC(0.0)               },
      {RC(-0.0761079419591268), RC(0.264353986580857),  RC(0.0),                 RC(0.0),                  RC(0.0),                  RC(0.0),                RC(0.0),                 RC(0.0),                  RC(0.0),                RC(0.0),                 RC(0.0),                RC(0.0),                RC(0.0),                RC(0.0)               },
      {RC(0.0727106904170694),  RC(-0.204265976977285), RC(0.181608196544136),   RC(0.0),                  RC(0.0),                  RC(0.0),                RC(0.0),                 RC(0.0),                  RC(0.0),                RC(0.0),                 RC(0.0),                RC(0.0),                RC(0.0),                RC(0.0)               },
      {RC(0.55763054816611),    RC(-0.409773579543499), RC(0.510926516886944),   RC(0.259892204518476),    RC(0.0),                  RC(0.0),                RC(0.0),                 RC(0.0),                  RC(0.0),                RC(0.0),                 RC(0.0),                RC(0.0),                RC(0.0),                RC(0.0)               },
      {RC(0.0228083864844437),  RC(-0.445569051836454), RC(-0.0915242778636248), RC(0.00450055909321655),  RC(0.6397807199983),      RC(0.0),                RC(0.0),                 RC(0.0),                  RC(0.0),                RC(0.0),                 RC(0.0),                RC(0.0),                RC(0.0),                RC(0.0)               },
      {RC(-0.135945849505152),  RC(0.0946509646963754), RC(-0.236110197279175),  RC(0.00318944206456517),  RC(0.255453021028118),    RC(0.174805219173446),  RC(0.0),                 RC(0.0),                  RC(0.0),                RC(0.0),                 RC(0.0),                RC(0.0),                RC(0.0),                RC(0.0)               },
      {RC(-0.147960260670772),  RC(-0.402188192230535), RC(-0.703014530043888),  RC(0.00941974677418186),  RC(0.885747111289207),    RC(0.261314066449028),  RC(0.16307697503668),    RC(0.0),                  RC(0.0),                RC(0.0),                 RC(0.0),                RC(0.0),                RC(0.0),                RC(0.0)               },
      {RC(0.165597241042244),   RC(0.824182962188923),  RC(-0.0280136160783609), RC(0.282372386631758),    RC(-0.957721354131182),   RC(0.489439550159977),  RC(0.170094415598103),   RC(0.0522519785718563),   RC(0.0),                RC(0.0),                 RC(0.0),                RC(0.0),                RC(0.0),                RC(0.0)               },
      {RC(0.0335292011495618),  RC(0.575750388029166),  RC(0.223289855356637),   RC(-0.00317458833242804), RC(-0.112890382135193),   RC(-0.419809267954284), RC(0.0466136902102104),  RC(-0.00115413813041085), RC(0.109685363692383),  RC(0.0),                 RC(0.0),                RC(0.0),                RC(0.0),                RC(0.0)               },
      {RC(-0.0512616878252355), RC(0.699261265830807),  RC(-0.117939611738769),  RC(0.0021745241931243),   RC(-0.00932826702640947), RC(-0.267575057469428), RC(0.126949139814065),   RC(0.00330353204502163),  RC(0.185949445053766),  RC(0.0938215615963721),  RC(0.0),                RC(0.0),                RC(0.0),                RC(0.0)               },
      {RC(-0.106521517960343),  RC(0.41835889096168),   RC(0.353585905881916),   RC(-0.0746474161579599),  RC(-0.015450626460289),   RC(-0.46224659192275),  RC(-0.0576406327329181), RC(-0.00712066942504018), RC(0.377776558014452),  RC(0.36890054338294),    RC(0.0618488746331837), RC(0.0),                RC(0.0),                RC(0.0)               },
      {RC(-0.163079104890997),  RC(0.644561721693806),  RC(0.636968661639572),   RC(-0.122346720085377),   RC(-0.333062564990312),   RC(-0.3054226490478),   RC(-0.357820712828352),  RC(-0.0125510510334706),  RC(0.371263681186311),  RC(0.371979640363694),   RC(0.0531090658708968), RC(0.0518279459132049), RC(0.0),                RC(0.0)               },
      {RC(0.579993784455521),   RC(-0.188833728676494), RC(0.999975696843775),   RC(0.0572810855901161),   RC(-0.264374735003671),   RC(0.165091739976854),  RC(-0.546675809010452),  RC(-0.0283821822291982),  RC(-0.102639860418374), RC(-0.0343251040446405), RC(0.4762598462591),    RC(-0.304153104931261), RC(0.0953911855943621), RC(0.0)               },
      {RC(0.0848552694007844),  RC(0.287193912340074),  RC(0.543683503004232),   RC(-0.081311059300692),   RC(-0.0328661289388557),  RC(-0.323456834372922), RC(-0.240378871658975),  RC(-0.0189913019930369),  RC(0.220663114082036),  RC(0.253029984360864),   RC(0.252011799370563),  RC(-0.154882222605423), RC(0.0315202264687415), RC(0.0514095812104714)}
    };
    const PetscReal b[14] = {RC(0.0), RC(0.516650324205117), RC(0.0773227217357826), RC(-0.12474204666975), RC(-0.0241052115180679), RC(-0.325821145180359), RC(0.0907237460123951), RC(0.0459271880596652), RC(0.221012259404702), RC(0.235510906761942), RC(0.491109674204385), RC(-0.323506525837343), RC(0.119918108821531), RC(0.0)};
    const PetscReal bembed[14] = {RC(2.32345691433618e-16), RC(0.499150900944401), RC(0.080991997189243), RC(-0.0359440417166322), RC(-0.0258910397441454), RC(-0.304540350278636),  RC(0.0836627473632563),
                                  RC(0.0417664613347638),   RC(0.223636394275293), RC(0.231569156867596), RC(0.240526201277663),   RC(-0.222933582911926),  RC(-0.0111479879597561), RC(0.19915314335888)};
    PetscCall(TSDIRKRegister(TSDIRK8614A, 8, 14, &A[0][0], b, NULL, bembed, 1, b));
  }
  {
    // DIRK(15,8)[1]SAL[(16,6)A] from https://github.com/yousefalamri55/High_Order_DIRK_Methods_Coeffs
    const PetscReal A[16][16] = {
      {RC(0.498904981271193),   RC(0.0),                  RC(0.0),                 RC(0.0),                  RC(0.0),                RC(0.0),                 RC(0.0),                RC(0.0),                 RC(0.0),                   RC(0.0),                 RC(0.0),                  RC(0.0),                 RC(0.0),               RC(0.0),                 RC(0.0),                 RC(0.0)              },
      {RC(-0.303806037341816),  RC(0.886299445992379),    RC(0.0),                 RC(0.0),                  RC(0.0),                RC(0.0),                 RC(0.0),                RC(0.0),                 RC(0.0),                   RC(0.0),                 RC(0.0),                  RC(0.0),                 RC(0.0),               RC(0.0),                 RC(0.0),                 RC(0.0)              },
      {RC(-0.581440223471476),  RC(0.371003719460259),    RC(0.43844717752802),    RC(0.0),                  RC(0.0),                RC(0.0),                 RC(0.0),                RC(0.0),                 RC(0.0),                   RC(0.0),                 RC(0.0),                  RC(0.0),                 RC(0.0),               RC(0.0),                 RC(0.0),                 RC(0.0)              },
      {RC(0.531852638870051),   RC(-0.339363014907108),   RC(0.422373239795441),   RC(0.223854203543397),    RC(0.0),                RC(0.0),                 RC(0.0),                RC(0.0),                 RC(0.0),                   RC(0.0),                 RC(0.0),                  RC(0.0),                 RC(0.0),               RC(0.0),                 RC(0.0),                 RC(0.0)              },
      {RC(0.118517891868867),   RC(-0.0756235584174296),  RC(-0.0864284870668712), RC(0.000536692838658312), RC(0.10101418329932),   RC(0.0),                 RC(0.0),                RC(0.0),                 RC(0.0),                   RC(0.0),                 RC(0.0),                  RC(0.0),                 RC(0.0),               RC(0.0),                 RC(0.0),                 RC(0.0)              },
      {RC(0.218733626116401),   RC(-0.139568928299635),   RC(0.30473612813488),    RC(0.00354038623073564),  RC(0.0932085751160559), RC(0.140161806097591),   RC(0.0),                RC(0.0),                 RC(0.0),                   RC(0.0),                 RC(0.0),                  RC(0.0),                 RC(0.0),               RC(0.0),                 RC(0.0),                 RC(0.0)              },
      {RC(0.0692944686081835),  RC(-0.0442152168939502),  RC(-0.0903375348855603), RC(0.00259030241156141),  RC(0.204514233679515),  RC(-0.0245383758960002), RC(0.199289437094059),  RC(0.0),                 RC(0.0),                   RC(0.0),                 RC(0.0),                  RC(0.0),                 RC(0.0),               RC(0.0),                 RC(0.0),                 RC(0.0)              },
      {RC(0.990640016505571),   RC(-0.632104756315967),   RC(0.856971425234221),   RC(0.174494099232246),    RC(-0.113715829680145), RC(-0.151494045307366),  RC(-0.438268629569005), RC(0.120578398912139),   RC(0.0),                   RC(0.0),                 RC(0.0),                  RC(0.0),                 RC(0.0),               RC(0.0),                 RC(0.0),                 RC(0.0)              },
      {RC(-0.099415677713136),  RC(0.211832014309207),    RC(-0.245998265866888),  RC(-0.182249672235861),   RC(0.167897635713799),  RC(0.212850335030069),   RC(-0.391739299440123), RC(-0.0118718506876767), RC(0.526293701659093),     RC(0.0),                 RC(0.0),                  RC(0.0),                 RC(0.0),               RC(0.0),                 RC(0.0),                 RC(0.0)              },
      {RC(0.383983914845461),   RC(-0.245011361219604),   RC(0.46717278554955),    RC(-0.0361272447593202),  RC(0.0742234660511333), RC(-0.0474816271948766), RC(-0.229859978525756), RC(0.0516283729206322),  RC(0.0),                   RC(0.193823890777594),   RC(0.0),                  RC(0.0),                 RC(0.0),               RC(0.0),                 RC(0.0),                 RC(0.0)              },
      {RC(0.0967855003180134),  RC(-0.0481037037916184),  RC(0.191268138832434),   RC(0.234977164564126),    RC(0.0620265921753097), RC(0.403432826534738),   RC(0.152403846687238),  RC(-0.118420429237746),  RC(0.0582141598685892),    RC(-0.13924540906863),   RC(0.106661313117545),    RC(0.0),                 RC(0.0),               RC(0.0),                 RC(0.0),                 RC(0.0)              },
      {RC(0.133941307432055),   RC(-0.0722076602896254),  RC(0.217086297689275),   RC(0.00495499602192887),  RC(0.0306090174933995), RC(0.26483526755746),    RC(0.204442440745605),  RC(0.196883395136708),   RC(0.056527012583996),     RC(-0.150216381356784),  RC(-0.217209415757333),   RC(0.330353722743315),   RC(0.0),               RC(0.0),                 RC(0.0),                 RC(0.0)              },
      {RC(0.157014274561299),   RC(-0.0883810256381874),  RC(0.117193033885034),   RC(-0.0362304243769466),  RC(0.0169030211466111), RC(-0.169835753576141),  RC(0.399749979234113),  RC(0.31806704093008),    RC(0.050340008347693),     RC(0.120284837472214),   RC(-0.235313193645423),   RC(0.232488522208926),   RC(0.117719679450729), RC(0.0),                 RC(0.0),                 RC(0.0)              },
      {RC(0.00276453816875833), RC(-0.00366028255231782), RC(-0.331078914515559),  RC(0.623377549031949),    RC(0.167618142989491),  RC(0.0748467945312516),  RC(0.797629286699677),  RC(-0.390714256799583),  RC(-0.00808553925131555),  RC(0.014840324980952),   RC(-0.0856180410248133),  RC(0.602943304937827),   RC(-0.5771359338496),  RC(0.112273026653282),   RC(0.0),                 RC(0.0)              },
      {RC(0.0),                 RC(0.0),                  RC(0.085283971980307),   RC(0.51334393454179),     RC(0.144355978013514),  RC(0.255379109487853),   RC(0.225075750790524),  RC(-0.343241323394982),  RC(0.0),                   RC(0.0798250392218852),  RC(0.0528824734082655),   RC(-0.0830350888900362), RC(0.022567388707279), RC(-0.0592631119040204), RC(0.106825878037621),   RC(0.0)              },
      {RC(0.173784481207652),   RC(-0.110887906116241),   RC(0.190052513365204),   RC(-0.0688345422674029),  RC(0.10326505079603),   RC(0.267127097115219),   RC(0.141703423176897),  RC(0.0117966866651728),  RC(-6.65650091812762e-15), RC(-0.0213725083662519), RC(-0.00931148598712566), RC(-0.10007679077114),   RC(0.123471797451553), RC(0.00203684241073055), RC(-0.0294320891781173), RC(0.195746619921528)}
    };
    const PetscReal b[16] = {RC(0.0), RC(0.0), RC(0.085283971980307), RC(0.51334393454179), RC(0.144355978013514), RC(0.255379109487853), RC(0.225075750790524), RC(-0.343241323394982), RC(0.0), RC(0.0798250392218852), RC(0.0528824734082655), RC(-0.0830350888900362), RC(0.022567388707279), RC(-0.0592631119040204), RC(0.106825878037621), RC(0.0)};
    const PetscReal bembed[16] = {RC(-1.31988512519898e-15), RC(7.53606601764004e-16), RC(0.0886789133915965),   RC(0.0968726531622137),  RC(0.143815375874267),     RC(0.335214773313601),  RC(0.221862366978063),  RC(-0.147408947987273),
                                  RC(4.16297599203445e-16),  RC(0.000727276166520566), RC(-0.00284892677941246), RC(0.00512492274297611), RC(-0.000275595071215218), RC(0.0136014719350733), RC(0.0165190013607726), RC(0.228116714912817)};
    PetscCall(TSDIRKRegister(TSDIRK8616SAL, 8, 16, &A[0][0], b, NULL, bembed, 1, b));
  }
  {
    // ESDIRK(16,8)[2]SAL[(16,5)] from https://github.com/yousefalamri55/High_Order_DIRK_Methods_Coeffs
    const PetscReal A[16][16] = {
      {RC(0.0),                  RC(0.0),                 RC(0.0),                  RC(0.0),                   RC(0.0),                  RC(0.0),                  RC(0.0),                  RC(0.0),                RC(0.0),                 RC(0.0),                 RC(0.0),                RC(0.0),                 RC(0.0),                 RC(0.0),                  RC(0.0),               RC(0.0)              },
      {RC(0.117318819358521),    RC(0.117318819358521),   RC(0.0),                  RC(0.0),                   RC(0.0),                  RC(0.0),                  RC(0.0),                  RC(0.0),                RC(0.0),                 RC(0.0),                 RC(0.0),                RC(0.0),                 RC(0.0),                 RC(0.0),                  RC(0.0),               RC(0.0)              },
      {RC(0.0557014605974616),   RC(0.385525646638742),   RC(0.117318819358521),    RC(0.0),                   RC(0.0),                  RC(0.0),                  RC(0.0),                  RC(0.0),                RC(0.0),                 RC(0.0),                 RC(0.0),                RC(0.0),                 RC(0.0),                 RC(0.0),                  RC(0.0),               RC(0.0)              },
      {RC(0.063493276428895),    RC(0.373556126263681),   RC(0.0082994166438953),   RC(0.117318819358521),     RC(0.0),                  RC(0.0),                  RC(0.0),                  RC(0.0),                RC(0.0),                 RC(0.0),                 RC(0.0),                RC(0.0),                 RC(0.0),                 RC(0.0),                  RC(0.0),               RC(0.0)              },
      {RC(0.0961351856230088),   RC(0.335558324517178),   RC(0.207077765910132),    RC(-0.0581917140797146),   RC(0.117318819358521),    RC(0.0),                  RC(0.0),                  RC(0.0),                RC(0.0),                 RC(0.0),                 RC(0.0),                RC(0.0),                 RC(0.0),                 RC(0.0),                  RC(0.0),               RC(0.0)              },
      {RC(0.0497669214238319),   RC(0.384288616546039),   RC(0.0821728117583936),   RC(0.120337007107103),     RC(0.202262782645888),    RC(0.117318819358521),    RC(0.0),                  RC(0.0),                RC(0.0),                 RC(0.0),                 RC(0.0),                RC(0.0),                 RC(0.0),                 RC(0.0),                  RC(0.0),               RC(0.0)              },
      {RC(0.00626710666809847),  RC(0.496491452640725),   RC(-0.111303249827358),   RC(0.170478821683603),     RC(0.166517073971103),    RC(-0.0328669811542241),  RC(0.117318819358521),    RC(0.0),                RC(0.0),                 RC(0.0),                 RC(0.0),                RC(0.0),                 RC(0.0),                 RC(0.0),                  RC(0.0),               RC(0.0)              },
      {RC(0.0463439767281591),   RC(0.00306724391019652), RC(-0.00816305222386205), RC(-0.0353302599538294),   RC(0.0139313601702569),   RC(-0.00992014507967429), RC(0.0210087909090165),   RC(0.117318819358521),  RC(0.0),                 RC(0.0),                 RC(0.0),                RC(0.0),                 RC(0.0),                 RC(0.0),                  RC(0.0),               RC(0.0)              },
      {RC(0.111574049232048),    RC(0.467639166482209),   RC(0.237773114804619),    RC(0.0798895699267508),    RC(0.109580615914593),    RC(0.0307353103825936),   RC(-0.0404391509541147),  RC(-0.16942110744293),  RC(0.117318819358521),   RC(0.0),                 RC(0.0),                RC(0.0),                 RC(0.0),                 RC(0.0),                  RC(0.0),               RC(0.0)              },
      {RC(-0.0107072484863877),  RC(-0.231376703354252),  RC(0.017541113036611),    RC(0.144871527682418),     RC(-0.041855459769806),   RC(0.0841832168332261),   RC(-0.0850020937282192),  RC(0.486170343825899),  RC(-0.0526717116822739), RC(0.117318819358521),   RC(0.0),                RC(0.0),                 RC(0.0),                 RC(0.0),                  RC(0.0),               RC(0.0)              },
      {RC(-0.0142238262314935),  RC(0.14752923682514),    RC(0.238235830732566),    RC(0.037950291904103),     RC(0.252075123381518),    RC(0.0474266904224567),   RC(-0.00363139069342027), RC(0.274081442388563),  RC(-0.0599166970745255), RC(-0.0527138812389185), RC(0.117318819358521),  RC(0.0),                 RC(0.0),                 RC(0.0),                  RC(0.0),               RC(0.0)              },
      {RC(-0.11837020183211),    RC(-0.635712481821264),  RC(0.239738832602538),    RC(0.330058936651707),     RC(-0.325784087988237),   RC(-0.0506514314589253),  RC(-0.281914404487009),   RC(0.852596345144291),  RC(0.651444614298805),   RC(-0.103476387303591),  RC(-0.354835880209975), RC(0.117318819358521),   RC(0.0),                 RC(0.0),                  RC(0.0),               RC(0.0)              },
      {RC(-0.00458164025442349), RC(0.296219694015248),   RC(0.322146049419995),    RC(0.15917778285238),      RC(0.284864871688843),    RC(0.185509526463076),    RC(-0.0784621067883274),  RC(0.166312223692047),  RC(-0.284152486083397),  RC(-0.357125104338944),  RC(0.078437074055306),  RC(0.0884129667114481),  RC(0.117318819358521),   RC(0.0),                  RC(0.0),               RC(0.0)              },
      {RC(-0.0545561913848106),  RC(0.675785423442753),   RC(0.423066443201941),    RC(-0.000165300126841193), RC(0.104252994793763),    RC(-0.105763019303021),   RC(-0.15988308809318),    RC(0.0515050001032011), RC(0.56013979290924),    RC(-0.45781539708603),   RC(-0.255870699752664), RC(0.026960254296416),   RC(-0.0721245985053681), RC(0.117318819358521),    RC(0.0),               RC(0.0)              },
      {RC(0.0649253995775223),   RC(-0.0216056457922249), RC(-0.073738139377975),   RC(0.0931033310077225),    RC(-0.0194339577299149),  RC(-0.0879623837313009),  RC(0.057125517179467),    RC(0.205120850488097),  RC(0.132576503537441),   RC(0.489416890627328),   RC(-0.1106765720501),   RC(-0.081038793996096),  RC(0.0606031613503788),  RC(-0.00241467937442272), RC(0.117318819358521), RC(0.0)              },
      {RC(0.0459979286336779),   RC(0.0780075394482806),  RC(0.015021874148058),    RC(0.195180277284195),     RC(-0.00246643310153235), RC(0.0473977117068314),   RC(-0.0682773558610363),  RC(0.19568019123878),   RC(-0.0876765449323747), RC(0.177874852409192),   RC(-0.337519251582222), RC(-0.0123255553640736), RC(0.311573291192553),   RC(0.0458604327754991),   RC(0.278352222645651), RC(0.117318819358521)}
    };
    const PetscReal b[16]      = {RC(0.0459979286336779),  RC(0.0780075394482806), RC(0.015021874148058),  RC(0.195180277284195),   RC(-0.00246643310153235), RC(0.0473977117068314), RC(-0.0682773558610363), RC(0.19568019123878),
                                  RC(-0.0876765449323747), RC(0.177874852409192),  RC(-0.337519251582222), RC(-0.0123255553640736), RC(0.311573291192553),    RC(0.0458604327754991), RC(0.278352222645651),   RC(0.117318819358521)};
    const PetscReal bembed[16] = {RC(0.0603373529853206),   RC(0.175453809423998),  RC(0.0537707777611352), RC(0.195309248607308),  RC(0.0135893741970232), RC(-0.0221160259296707), RC(-0.00726526156430691), RC(0.102961059369124),
                                  RC(0.000900215457460583), RC(0.0547959465692338), RC(-0.334995726863153), RC(0.0464409662093384), RC(0.301388101652194),  RC(0.00524851570622031), RC(0.229538601845236),    RC(0.124643044573514)};
    PetscCall(TSDIRKRegister(TSDIRKES8516SAL, 8, 16, &A[0][0], b, NULL, bembed, 1, b));
  }

  /* Additive methods */
  {
    const PetscReal A[3][3] = {
      {0.0, 0.0, 0.0},
      {0.0, 0.0, 0.0},
      {0.0, 0.5, 0.0}
    };
    const PetscReal At[3][3] = {
      {1.0, 0.0, 0.0},
      {0.0, 0.5, 0.0},
      {0.0, 0.5, 0.5}
    };
    const PetscReal b[3]       = {0.0, 0.5, 0.5};
    const PetscReal bembedt[3] = {1.0, 0.0, 0.0};
    PetscCall(TSARKIMEXRegister(TSARKIMEX1BEE, 2, 3, &At[0][0], b, NULL, &A[0][0], b, NULL, bembedt, bembedt, 1, b, NULL));
  }
  {
    const PetscReal A[2][2] = {
      {0.0, 0.0},
      {0.5, 0.0}
    };
    const PetscReal At[2][2] = {
      {0.0, 0.0},
      {0.0, 0.5}
    };
    const PetscReal b[2]       = {0.0, 1.0};
    const PetscReal bembedt[2] = {0.5, 0.5};
    /* binterpt[2][2] = {{1.0,-1.0},{0.0,1.0}};  second order dense output has poor stability properties and hence it is not currently in use */
    PetscCall(TSARKIMEXRegister(TSARKIMEXARS122, 2, 2, &At[0][0], b, NULL, &A[0][0], b, NULL, bembedt, bembedt, 1, b, NULL));
  }
  {
    const PetscReal A[2][2] = {
      {0.0, 0.0},
      {1.0, 0.0}
    };
    const PetscReal At[2][2] = {
      {0.0, 0.0},
      {0.5, 0.5}
    };
    const PetscReal b[2]       = {0.5, 0.5};
    const PetscReal bembedt[2] = {0.0, 1.0};
    /* binterpt[2][2] = {{1.0,-0.5},{0.0,0.5}}  second order dense output has poor stability properties and hence it is not currently in use */
    PetscCall(TSARKIMEXRegister(TSARKIMEXA2, 2, 2, &At[0][0], b, NULL, &A[0][0], b, NULL, bembedt, bembedt, 1, b, NULL));
  }
  {
    const PetscReal A[2][2] = {
      {0.0, 0.0},
      {1.0, 0.0}
    };
    const PetscReal At[2][2] = {
      {us2,             0.0},
      {1.0 - 2.0 * us2, us2}
    };
    const PetscReal b[2]           = {0.5, 0.5};
    const PetscReal bembedt[2]     = {0.0, 1.0};
    const PetscReal binterpt[2][2] = {
      {(us2 - 1.0) / (2.0 * us2 - 1.0),     -1 / (2.0 * (1.0 - 2.0 * us2))},
      {1 - (us2 - 1.0) / (2.0 * us2 - 1.0), -1 / (2.0 * (1.0 - 2.0 * us2))}
    };
    const PetscReal binterp[2][2] = {
      {1.0, -0.5},
      {0.0, 0.5 }
    };
    PetscCall(TSARKIMEXRegister(TSARKIMEXL2, 2, 2, &At[0][0], b, NULL, &A[0][0], b, NULL, bembedt, bembedt, 2, binterpt[0], binterp[0]));
  }
  {
    const PetscReal A[3][3] = {
      {0,      0,   0},
      {2 - s2, 0,   0},
      {0.5,    0.5, 0}
    };
    const PetscReal At[3][3] = {
      {0,            0,            0         },
      {1 - 1 / s2,   1 - 1 / s2,   0         },
      {1 / (2 * s2), 1 / (2 * s2), 1 - 1 / s2}
    };
    const PetscReal bembedt[3]     = {(4. - s2) / 8., (4. - s2) / 8., 1 / (2. * s2)};
    const PetscReal binterpt[3][2] = {
      {1.0 / s2, -1.0 / (2.0 * s2)},
      {1.0 / s2, -1.0 / (2.0 * s2)},
      {1.0 - s2, 1.0 / s2         }
    };
    PetscCall(TSARKIMEXRegister(TSARKIMEX2C, 2, 3, &At[0][0], NULL, NULL, &A[0][0], NULL, NULL, bembedt, bembedt, 2, binterpt[0], NULL));
  }
  {
    const PetscReal A[3][3] = {
      {0,      0,    0},
      {2 - s2, 0,    0},
      {0.75,   0.25, 0}
    };
    const PetscReal At[3][3] = {
      {0,            0,            0         },
      {1 - 1 / s2,   1 - 1 / s2,   0         },
      {1 / (2 * s2), 1 / (2 * s2), 1 - 1 / s2}
    };
    const PetscReal bembedt[3]     = {(4. - s2) / 8., (4. - s2) / 8., 1 / (2. * s2)};
    const PetscReal binterpt[3][2] = {
      {1.0 / s2, -1.0 / (2.0 * s2)},
      {1.0 / s2, -1.0 / (2.0 * s2)},
      {1.0 - s2, 1.0 / s2         }
    };
    PetscCall(TSARKIMEXRegister(TSARKIMEX2D, 2, 3, &At[0][0], NULL, NULL, &A[0][0], NULL, NULL, bembedt, bembedt, 2, binterpt[0], NULL));
  }
  { /* Optimal for linear implicit part */
    const PetscReal A[3][3] = {
      {0,                0,                0},
      {2 - s2,           0,                0},
      {(3 - 2 * s2) / 6, (3 + 2 * s2) / 6, 0}
    };
    const PetscReal At[3][3] = {
      {0,            0,            0         },
      {1 - 1 / s2,   1 - 1 / s2,   0         },
      {1 / (2 * s2), 1 / (2 * s2), 1 - 1 / s2}
    };
    const PetscReal bembedt[3]     = {(4. - s2) / 8., (4. - s2) / 8., 1 / (2. * s2)};
    const PetscReal binterpt[3][2] = {
      {1.0 / s2, -1.0 / (2.0 * s2)},
      {1.0 / s2, -1.0 / (2.0 * s2)},
      {1.0 - s2, 1.0 / s2         }
    };
    PetscCall(TSARKIMEXRegister(TSARKIMEX2E, 2, 3, &At[0][0], NULL, NULL, &A[0][0], NULL, NULL, bembedt, bembedt, 2, binterpt[0], NULL));
  }
  { /* Optimal for linear implicit part */
    const PetscReal A[3][3] = {
      {0,   0,   0},
      {0.5, 0,   0},
      {0.5, 0.5, 0}
    };
    const PetscReal At[3][3] = {
      {0.25,   0,      0     },
      {0,      0.25,   0     },
      {1. / 3, 1. / 3, 1. / 3}
    };
    PetscCall(TSARKIMEXRegister(TSARKIMEXPRSSP2, 2, 3, &At[0][0], NULL, NULL, &A[0][0], NULL, NULL, NULL, NULL, 0, NULL, NULL));
  }
  {
    const PetscReal A[4][4] = {
      {0,                                0,                                0,                                 0},
      {1767732205903. / 2027836641118.,  0,                                0,                                 0},
      {5535828885825. / 10492691773637., 788022342437. / 10882634858940.,  0,                                 0},
      {6485989280629. / 16251701735622., -4246266847089. / 9704473918619., 10755448449292. / 10357097424841., 0}
    };
    const PetscReal At[4][4] = {
      {0,                                0,                                0,                                 0                              },
      {1767732205903. / 4055673282236.,  1767732205903. / 4055673282236.,  0,                                 0                              },
      {2746238789719. / 10658868560708., -640167445237. / 6845629431997.,  1767732205903. / 4055673282236.,   0                              },
      {1471266399579. / 7840856788654.,  -4482444167858. / 7529755066697., 11266239266428. / 11593286722821., 1767732205903. / 4055673282236.}
    };
    const PetscReal bembedt[4]     = {2756255671327. / 12835298489170., -10771552573575. / 22201958757719., 9247589265047. / 10645013368117., 2193209047091. / 5459859503100.};
    const PetscReal binterpt[4][2] = {
      {4655552711362. / 22874653954995.,  -215264564351. / 13552729205753.  },
      {-18682724506714. / 9892148508045., 17870216137069. / 13817060693119. },
      {34259539580243. / 13192909600954., -28141676662227. / 17317692491321.},
      {584795268549. / 6622622206610.,    2508943948391. / 7218656332882.   }
    };
    PetscCall(TSARKIMEXRegister(TSARKIMEX3, 3, 4, &At[0][0], NULL, NULL, &A[0][0], NULL, NULL, bembedt, bembedt, 2, binterpt[0], NULL));
  }
  {
    const PetscReal A[5][5] = {
      {0,        0,       0,      0,       0},
      {1. / 2,   0,       0,      0,       0},
      {11. / 18, 1. / 18, 0,      0,       0},
      {5. / 6,   -5. / 6, .5,     0,       0},
      {1. / 4,   7. / 4,  3. / 4, -7. / 4, 0}
    };
    const PetscReal At[5][5] = {
      {0, 0,       0,       0,      0     },
      {0, 1. / 2,  0,       0,      0     },
      {0, 1. / 6,  1. / 2,  0,      0     },
      {0, -1. / 2, 1. / 2,  1. / 2, 0     },
      {0, 3. / 2,  -3. / 2, 1. / 2, 1. / 2}
    };
    PetscCall(TSARKIMEXRegister(TSARKIMEXARS443, 3, 5, &At[0][0], NULL, NULL, &A[0][0], NULL, NULL, NULL, NULL, 0, NULL, NULL));
  }
  {
    const PetscReal A[5][5] = {
      {0,      0,      0,      0, 0},
      {1,      0,      0,      0, 0},
      {4. / 9, 2. / 9, 0,      0, 0},
      {1. / 4, 0,      3. / 4, 0, 0},
      {1. / 4, 0,      3. / 5, 0, 0}
    };
    const PetscReal At[5][5] = {
      {0,       0,       0,   0,   0 },
      {.5,      .5,      0,   0,   0 },
      {5. / 18, -1. / 9, .5,  0,   0 },
      {.5,      0,       0,   .5,  0 },
      {.25,     0,       .75, -.5, .5}
    };
    PetscCall(TSARKIMEXRegister(TSARKIMEXBPR3, 3, 5, &At[0][0], NULL, NULL, &A[0][0], NULL, NULL, NULL, NULL, 0, NULL, NULL));
  }
  {
    const PetscReal A[6][6] = {
      {0,                               0,                                 0,                                 0,                                0,              0},
      {1. / 2,                          0,                                 0,                                 0,                                0,              0},
      {13861. / 62500.,                 6889. / 62500.,                    0,                                 0,                                0,              0},
      {-116923316275. / 2393684061468., -2731218467317. / 15368042101831., 9408046702089. / 11113171139209.,  0,                                0,              0},
      {-451086348788. / 2902428689909., -2682348792572. / 7519795681897.,  12662868775082. / 11960479115383., 3355817975965. / 11060851509271., 0,              0},
      {647845179188. / 3216320057751.,  73281519250. / 8382639484533.,     552539513391. / 3454668386233.,    3354512671639. / 8306763924573.,  4040. / 17871., 0}
    };
    const PetscReal At[6][6] = {
      {0,                            0,                       0,                       0,                   0,             0     },
      {1. / 4,                       1. / 4,                  0,                       0,                   0,             0     },
      {8611. / 62500.,               -1743. / 31250.,         1. / 4,                  0,                   0,             0     },
      {5012029. / 34652500.,         -654441. / 2922500.,     174375. / 388108.,       1. / 4,              0,             0     },
      {15267082809. / 155376265600., -71443401. / 120774400., 730878875. / 902184768., 2285395. / 8070912., 1. / 4,        0     },
      {82889. / 524892.,             0,                       15625. / 83664.,         69875. / 102672.,    -2260. / 8211, 1. / 4}
    };
    const PetscReal bembedt[6]     = {4586570599. / 29645900160., 0, 178811875. / 945068544., 814220225. / 1159782912., -3700637. / 11593932., 61727. / 225920.};
    const PetscReal binterpt[6][3] = {
      {6943876665148. / 7220017795957.,   -54480133. / 30881146., 6818779379841. / 7100303317025.  },
      {0,                                 0,                      0                                },
      {7640104374378. / 9702883013639.,   -11436875. / 14766696., 2173542590792. / 12501825683035. },
      {-20649996744609. / 7521556579894., 174696575. / 18121608., -31592104683404. / 5083833661969.},
      {8854892464581. / 2390941311638.,   -12120380. / 966161.,   61146701046299. / 7138195549469. },
      {-11397109935349. / 6675773540249., 3843. / 706.,           -17219254887155. / 4939391667607.}
    };
    PetscCall(TSARKIMEXRegister(TSARKIMEX4, 4, 6, &At[0][0], NULL, NULL, &A[0][0], NULL, NULL, bembedt, bembedt, 3, binterpt[0], NULL));
  }
  {
    const PetscReal A[8][8] = {
      {0,                                  0,                              0,                                 0,                                  0,                               0,                                 0,                               0},
      {41. / 100,                          0,                              0,                                 0,                                  0,                               0,                                 0,                               0},
      {367902744464. / 2072280473677.,     677623207551. / 8224143866563., 0,                                 0,                                  0,                               0,                                 0,                               0},
      {1268023523408. / 10340822734521.,   0,                              1029933939417. / 13636558850479.,  0,                                  0,                               0,                                 0,                               0},
      {14463281900351. / 6315353703477.,   0,                              66114435211212. / 5879490589093.,  -54053170152839. / 4284798021562.,  0,                               0,                                 0,                               0},
      {14090043504691. / 34967701212078.,  0,                              15191511035443. / 11219624916014., -18461159152457. / 12425892160975., -281667163811. / 9011619295870., 0,                                 0,                               0},
      {19230459214898. / 13134317526959.,  0,                              21275331358303. / 2942455364971.,  -38145345988419. / 4862620318723.,  -1. / 8,                         -1. / 8,                           0,                               0},
      {-19977161125411. / 11928030595625., 0,                              -40795976796054. / 6384907823539., 177454434618887. / 12078138498510., 782672205425. / 8267701900261.,  -69563011059811. / 9646580694205., 7356628210526. / 4942186776405., 0}
    };
    const PetscReal At[8][8] = {
      {0,                                0,                                0,                                 0,                                  0,                                0,                                  0,                                 0         },
      {41. / 200.,                       41. / 200.,                       0,                                 0,                                  0,                                0,                                  0,                                 0         },
      {41. / 400.,                       -567603406766. / 11931857230679., 41. / 200.,                        0,                                  0,                                0,                                  0,                                 0         },
      {683785636431. / 9252920307686.,   0,                                -110385047103. / 1367015193373.,   41. / 200.,                         0,                                0,                                  0,                                 0         },
      {3016520224154. / 10081342136671., 0,                                30586259806659. / 12414158314087., -22760509404356. / 11113319521817., 41. / 200.,                       0,                                  0,                                 0         },
      {218866479029. / 1489978393911.,   0,                                638256894668. / 5436446318841.,    -1179710474555. / 5321154724896.,   -60928119172. / 8023461067671.,   41. / 200.,                         0,                                 0         },
      {1020004230633. / 5715676835656.,  0,                                25762820946817. / 25263940353407., -2161375909145. / 9755907335909.,   -211217309593. / 5846859502534.,  -4269925059573. / 7827059040749.,   41. / 200,                         0         },
      {-872700587467. / 9133579230613.,  0,                                0,                                 22348218063261. / 9555858737531.,   -1143369518992. / 8141816002931., -39379526789629. / 19018526304540., 32727382324388. / 42900044865799., 41. / 200.}
    };
    const PetscReal bembedt[8]     = {-975461918565. / 9796059967033., 0, 0, 78070527104295. / 32432590147079., -548382580838. / 3424219808633., -33438840321285. / 15594753105479., 3629800801594. / 4656183773603., 4035322873751. / 18575991585200.};
    const PetscReal binterpt[8][3] = {
      {-17674230611817. / 10670229744614., 43486358583215. / 12773830924787.,  -9257016797708. / 5021505065439. },
      {0,                                  0,                                  0                                },
      {0,                                  0,                                  0                                },
      {65168852399939. / 7868540260826.,   -91478233927265. / 11067650958493., 26096422576131. / 11239449250142.},
      {15494834004392. / 5936557850923.,   -79368583304911. / 10890268929626., 92396832856987. / 20362823103730.},
      {-99329723586156. / 26959484932159., -12239297817655. / 9152339842473.,  30029262896817. / 10175596800299.},
      {-19024464361622. / 5461577185407.,  115839755401235. / 10719374521269., -26136350496073. / 3983972220547.},
      {-6511271360970. / 6095937251113.,   5843115559534. / 2180450260947.,    -5289405421727. / 3760307252460. }
    };
    PetscCall(TSARKIMEXRegister(TSARKIMEX5, 5, 8, &At[0][0], NULL, NULL, &A[0][0], NULL, NULL, bembedt, bembedt, 3, binterpt[0], NULL));
  }
#undef RC
#undef us2
#undef s2
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  TSARKIMEXRegisterDestroy - Frees the list of schemes that were registered by `TSARKIMEXRegister()`.

  Not Collective

  Level: advanced

.seealso: [](ch_ts), `TSARKIMEX`, `TSARKIMEXRegister()`, `TSARKIMEXRegisterAll()`
@*/
PetscErrorCode TSARKIMEXRegisterDestroy(void)
{
  ARKTableauLink link;

  PetscFunctionBegin;
  while ((link = ARKTableauList)) {
    ARKTableau t   = &link->tab;
    ARKTableauList = link->next;
    PetscCall(PetscFree6(t->At, t->bt, t->ct, t->A, t->b, t->c));
    PetscCall(PetscFree2(t->bembedt, t->bembed));
    PetscCall(PetscFree2(t->binterpt, t->binterp));
    PetscCall(PetscFree(t->name));
    PetscCall(PetscFree(link));
  }
  TSARKIMEXRegisterAllCalled = PETSC_FALSE;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  TSARKIMEXInitializePackage - This function initializes everything in the `TSARKIMEX` package. It is called
  from `TSInitializePackage()`.

  Level: developer

.seealso: [](ch_ts), `PetscInitialize()`, `TSARKIMEXFinalizePackage()`
@*/
PetscErrorCode TSARKIMEXInitializePackage(void)
{
  PetscFunctionBegin;
  if (TSARKIMEXPackageInitialized) PetscFunctionReturn(PETSC_SUCCESS);
  TSARKIMEXPackageInitialized = PETSC_TRUE;
  PetscCall(TSARKIMEXRegisterAll());
  PetscCall(PetscRegisterFinalize(TSARKIMEXFinalizePackage));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  TSARKIMEXFinalizePackage - This function destroys everything in the `TSARKIMEX` package. It is
  called from `PetscFinalize()`.

  Level: developer

.seealso: [](ch_ts), `PetscFinalize()`, `TSARKIMEXInitializePackage()`
@*/
PetscErrorCode TSARKIMEXFinalizePackage(void)
{
  PetscFunctionBegin;
  TSARKIMEXPackageInitialized = PETSC_FALSE;
  PetscCall(TSARKIMEXRegisterDestroy());
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  TSARKIMEXRegister - register a `TSARKIMEX` scheme by providing the entries in the Butcher tableau and optionally embedded approximations and interpolation

  Logically Collective

  Input Parameters:
+ name     - identifier for method
. order    - approximation order of method
. s        - number of stages, this is the dimension of the matrices below
. At       - Butcher table of stage coefficients for stiff part (dimension s*s, row-major)
. bt       - Butcher table for completing the stiff part of the step (dimension s; NULL to use the last row of At)
. ct       - Abscissa of each stiff stage (dimension s, NULL to use row sums of At)
. A        - Non-stiff stage coefficients (dimension s*s, row-major)
. b        - Non-stiff step completion table (dimension s; NULL to use last row of At)
. c        - Non-stiff abscissa (dimension s; NULL to use row sums of A)
. bembedt  - Stiff part of completion table for embedded method (dimension s; NULL if not available)
. bembed   - Non-stiff part of completion table for embedded method (dimension s; NULL to use bembedt if provided)
. pinterp  - Order of the interpolation scheme, equal to the number of columns of binterpt and binterp
. binterpt - Coefficients of the interpolation formula for the stiff part (dimension s*pinterp)
- binterp  - Coefficients of the interpolation formula for the non-stiff part (dimension s*pinterp; NULL to reuse binterpt)

  Level: advanced

  Note:
  Several `TSARKIMEX` methods are provided, this function is only needed to create new methods.

.seealso: [](ch_ts), `TSARKIMEX`, `TSType`, `TS`
@*/
PetscErrorCode TSARKIMEXRegister(TSARKIMEXType name, PetscInt order, PetscInt s, const PetscReal At[], const PetscReal bt[], const PetscReal ct[], const PetscReal A[], const PetscReal b[], const PetscReal c[], const PetscReal bembedt[], const PetscReal bembed[], PetscInt pinterp, const PetscReal binterpt[], const PetscReal binterp[])
{
  ARKTableauLink link;
  ARKTableau     t;
  PetscInt       i, j;

  PetscFunctionBegin;
  PetscCheck(s > 0, PETSC_COMM_SELF, PETSC_ERR_ARG_INCOMP, "Expected number of stages s %" PetscInt_FMT " > 0", s);
  PetscCall(TSARKIMEXInitializePackage());
  for (link = ARKTableauList; link; link = link->next) {
    PetscBool match;

    PetscCall(PetscStrcmp(link->tab.name, name, &match));
    PetscCheck(!match, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Method with name \"%s\" already registered", name);
  }
  PetscCall(PetscNew(&link));
  t = &link->tab;
  PetscCall(PetscStrallocpy(name, &t->name));
  t->order = order;
  t->s     = s;
  PetscCall(PetscMalloc6(s * s, &t->At, s, &t->bt, s, &t->ct, s * s, &t->A, s, &t->b, s, &t->c));
  PetscCall(PetscArraycpy(t->At, At, s * s));
  if (A) {
    PetscCall(PetscArraycpy(t->A, A, s * s));
    t->additive = PETSC_TRUE;
  }

  if (bt) PetscCall(PetscArraycpy(t->bt, bt, s));
  else
    for (i = 0; i < s; i++) t->bt[i] = At[(s - 1) * s + i];

  if (t->additive) {
    if (b) PetscCall(PetscArraycpy(t->b, b, s));
    else
      for (i = 0; i < s; i++) t->b[i] = t->bt[i];
  } else PetscCall(PetscArrayzero(t->b, s));

  if (ct) PetscCall(PetscArraycpy(t->ct, ct, s));
  else
    for (i = 0; i < s; i++)
      for (j = 0, t->ct[i] = 0; j < s; j++) t->ct[i] += At[i * s + j];

  if (t->additive) {
    if (c) PetscCall(PetscArraycpy(t->c, c, s));
    else
      for (i = 0; i < s; i++)
        for (j = 0, t->c[i] = 0; j < s; j++) t->c[i] += A[i * s + j];
  } else PetscCall(PetscArrayzero(t->c, s));

  t->stiffly_accurate = PETSC_TRUE;
  for (i = 0; i < s; i++)
    if (t->At[(s - 1) * s + i] != t->bt[i]) t->stiffly_accurate = PETSC_FALSE;

  t->explicit_first_stage = PETSC_TRUE;
  for (i = 0; i < s; i++)
    if (t->At[i] != 0.0) t->explicit_first_stage = PETSC_FALSE;

  /* def of FSAL can be made more precise */
  t->FSAL_implicit = (PetscBool)(t->explicit_first_stage && t->stiffly_accurate);

  if (bembedt) {
    PetscCall(PetscMalloc2(s, &t->bembedt, s, &t->bembed));
    PetscCall(PetscArraycpy(t->bembedt, bembedt, s));
    PetscCall(PetscArraycpy(t->bembed, bembed ? bembed : bembedt, s));
  }

  t->pinterp = pinterp;
  PetscCall(PetscMalloc2(s * pinterp, &t->binterpt, s * pinterp, &t->binterp));
  PetscCall(PetscArraycpy(t->binterpt, binterpt, s * pinterp));
  PetscCall(PetscArraycpy(t->binterp, binterp ? binterp : binterpt, s * pinterp));

  link->next     = ARKTableauList;
  ARKTableauList = link;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  TSDIRKRegister - register a `TSDIRK` scheme by providing the entries in its Butcher tableau and, optionally, embedded approximations and interpolation

  Logically Collective.

  Input Parameters:
+ name     - identifier for method
. order    - approximation order of method
. s        - number of stages, this is the dimension of the matrices below
. At       - Butcher table of stage coefficients (dimension `s`*`s`, row-major order)
. bt       - Butcher table for completing the step (dimension `s`; pass `NULL` to use the last row of `At`)
. ct       - Abscissa of each stage (dimension s, NULL to use row sums of At)
. bembedt  - Stiff part of completion table for embedded method (dimension s; `NULL` if not available)
. pinterp  - Order of the interpolation scheme, equal to the number of columns of `binterpt` and `binterp`
- binterpt - Coefficients of the interpolation formula (dimension s*pinterp)

  Level: advanced

  Note:
  Several `TSDIRK` methods are provided, the use of this function is only needed to create new methods.

.seealso: [](ch_ts), `TSDIRK`, `TSType`, `TS`
@*/
PetscErrorCode TSDIRKRegister(TSDIRKType name, PetscInt order, PetscInt s, const PetscReal At[], const PetscReal bt[], const PetscReal ct[], const PetscReal bembedt[], PetscInt pinterp, const PetscReal binterpt[])
{
  PetscFunctionBegin;
  PetscCall(TSARKIMEXRegister(name, order, s, At, bt, ct, NULL, NULL, NULL, bembedt, NULL, pinterp, binterpt, NULL));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
 The step completion formula is

 x1 = x0 - h bt^T YdotI + h b^T YdotRHS

 This function can be called before or after ts->vec_sol has been updated.
 Suppose we have a completion formula (bt,b) and an embedded formula (bet,be) of different order.
 We can write

 x1e = x0 - h bet^T YdotI + h be^T YdotRHS
     = x1 + h bt^T YdotI - h b^T YdotRHS - h bet^T YdotI + h be^T YdotRHS
     = x1 - h (bet - bt)^T YdotI + h (be - b)^T YdotRHS

 so we can evaluate the method with different order even after the step has been optimistically completed.
*/
static PetscErrorCode TSEvaluateStep_ARKIMEX(TS ts, PetscInt order, Vec X, PetscBool *done)
{
  TS_ARKIMEX  *ark = (TS_ARKIMEX *)ts->data;
  ARKTableau   tab = ark->tableau;
  PetscScalar *w   = ark->work;
  PetscReal    h;
  PetscInt     s = tab->s, j;
  PetscBool    hasE;

  PetscFunctionBegin;
  switch (ark->status) {
  case TS_STEP_INCOMPLETE:
  case TS_STEP_PENDING:
    h = ts->time_step;
    break;
  case TS_STEP_COMPLETE:
    h = ts->ptime - ts->ptime_prev;
    break;
  default:
    SETERRQ(PetscObjectComm((PetscObject)ts), PETSC_ERR_PLIB, "Invalid TSStepStatus");
  }
  if (order == tab->order) {
    if (ark->status == TS_STEP_INCOMPLETE) {
      if (!ark->imex && tab->stiffly_accurate) { /* Only the stiffly accurate implicit formula is used */
        PetscCall(VecCopy(ark->Y[s - 1], X));
      } else { /* Use the standard completion formula (bt,b) */
        PetscCall(VecCopy(ts->vec_sol, X));
        for (j = 0; j < s; j++) w[j] = h * tab->bt[j];
        PetscCall(VecMAXPY(X, s, w, ark->YdotI));
        if (tab->additive && ark->imex) { /* Method is IMEX, complete the explicit formula */
          PetscCall(TSHasRHSFunction(ts, &hasE));
          if (hasE) {
            for (j = 0; j < s; j++) w[j] = h * tab->b[j];
            PetscCall(VecMAXPY(X, s, w, ark->YdotRHS));
          }
        }
      }
    } else PetscCall(VecCopy(ts->vec_sol, X));
    if (done) *done = PETSC_TRUE;
    PetscFunctionReturn(PETSC_SUCCESS);
  } else if (order == tab->order - 1) {
    if (!tab->bembedt) goto unavailable;
    if (ark->status == TS_STEP_INCOMPLETE) { /* Complete with the embedded method (bet,be) */
      PetscCall(VecCopy(ts->vec_sol, X));
      for (j = 0; j < s; j++) w[j] = h * tab->bembedt[j];
      PetscCall(VecMAXPY(X, s, w, ark->YdotI));
      if (tab->additive) {
        PetscCall(TSHasRHSFunction(ts, &hasE));
        if (hasE) {
          for (j = 0; j < s; j++) w[j] = h * tab->bembed[j];
          PetscCall(VecMAXPY(X, s, w, ark->YdotRHS));
        }
      }
    } else { /* Rollback and re-complete using (bet-be,be-b) */
      PetscCall(VecCopy(ts->vec_sol, X));
      for (j = 0; j < s; j++) w[j] = h * (tab->bembedt[j] - tab->bt[j]);
      PetscCall(VecMAXPY(X, tab->s, w, ark->YdotI));
      if (tab->additive) {
        PetscCall(TSHasRHSFunction(ts, &hasE));
        if (hasE) {
          for (j = 0; j < s; j++) w[j] = h * (tab->bembed[j] - tab->b[j]);
          PetscCall(VecMAXPY(X, s, w, ark->YdotRHS));
        }
      }
    }
    if (done) *done = PETSC_TRUE;
    PetscFunctionReturn(PETSC_SUCCESS);
  }
unavailable:
  PetscCheck(done, PetscObjectComm((PetscObject)ts), PETSC_ERR_SUP, "ARKIMEX '%s' of order %" PetscInt_FMT " cannot evaluate step at order %" PetscInt_FMT ". Consider using -ts_adapt_type none or a different method that has an embedded estimate.",
             tab->name, tab->order, order);
  *done = PETSC_FALSE;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TSARKIMEXTestMassIdentity(TS ts, PetscBool *id)
{
  Vec         Udot, Y1, Y2;
  TS_ARKIMEX *ark = (TS_ARKIMEX *)ts->data;
  PetscReal   norm;

  PetscFunctionBegin;
  PetscCall(VecDuplicate(ts->vec_sol, &Udot));
  PetscCall(VecDuplicate(ts->vec_sol, &Y1));
  PetscCall(VecDuplicate(ts->vec_sol, &Y2));
  PetscCall(TSComputeIFunction(ts, ts->ptime, ts->vec_sol, Udot, Y1, ark->imex));
  PetscCall(VecSetRandom(Udot, NULL));
  PetscCall(TSComputeIFunction(ts, ts->ptime, ts->vec_sol, Udot, Y2, ark->imex));
  PetscCall(VecAXPY(Y2, -1.0, Y1));
  PetscCall(VecAXPY(Y2, -1.0, Udot));
  PetscCall(VecNorm(Y2, NORM_2, &norm));
  if (norm < 100.0 * PETSC_MACHINE_EPSILON) {
    *id = PETSC_TRUE;
  } else {
    *id = PETSC_FALSE;
    PetscCall(PetscInfo(ts, "IFunction(Udot = random) - IFunction(Udot = 0) is not near Udot, %g, suspect mass matrix implied in IFunction() is not the identity as required\n", (double)norm));
  }
  PetscCall(VecDestroy(&Udot));
  PetscCall(VecDestroy(&Y1));
  PetscCall(VecDestroy(&Y2));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TSARKIMEXComputeAlgebraicIS(TS, PetscReal, Vec, IS *);

static PetscErrorCode TSStep_ARKIMEX(TS ts)
{
  TS_ARKIMEX      *ark = (TS_ARKIMEX *)ts->data;
  ARKTableau       tab = ark->tableau;
  const PetscInt   s   = tab->s;
  const PetscReal *At = tab->At, *A = tab->A, *ct = tab->ct, *c = tab->c;
  PetscScalar     *w = ark->work;
  Vec             *Y = ark->Y, *YdotI = ark->YdotI, *YdotRHS = ark->YdotRHS, Ydot = ark->Ydot, Ydot0 = ark->Ydot0, Z = ark->Z;
  PetscBool        extrapolate = ark->extrapolate;
  TSAdapt          adapt;
  SNES             snes;
  PetscInt         i, j, its, lits;
  PetscInt         rejections = 0;
  PetscBool        hasE = PETSC_FALSE, dirk = (PetscBool)(!tab->additive), stageok, accept = PETSC_TRUE;
  PetscReal        next_time_step = ts->time_step;

  PetscFunctionBegin;
  if (ark->extrapolate && !ark->Y_prev) {
    PetscCall(VecDuplicateVecs(ts->vec_sol, tab->s, &ark->Y_prev));
    PetscCall(VecDuplicateVecs(ts->vec_sol, tab->s, &ark->YdotI_prev));
    if (tab->additive) PetscCall(VecDuplicateVecs(ts->vec_sol, tab->s, &ark->YdotRHS_prev));
  }

  if (!dirk) PetscCall(TSHasRHSFunction(ts, &hasE));
  if (!hasE) dirk = PETSC_TRUE;

  if (!ts->steprollback) {
    if (dirk || ts->equation_type >= TS_EQ_IMPLICIT) { /* Save the initial slope for the next step */
      PetscCall(VecCopy(YdotI[s - 1], Ydot0));
    }
    if (ark->extrapolate && !ts->steprestart) { /* Save the Y, YdotI, YdotRHS for extrapolation initial guess */
      for (i = 0; i < s; i++) {
        PetscCall(VecCopy(Y[i], ark->Y_prev[i]));
        PetscCall(VecCopy(YdotI[i], ark->YdotI_prev[i]));
        if (tab->additive && hasE) PetscCall(VecCopy(YdotRHS[i], ark->YdotRHS_prev[i]));
      }
    }
  }

  /*
     For fully implicit formulations we must solve the equations

       F(t_n,x_n,xdot) = 0

     for the explicit first stage.
     Here we call SNESSolve using PETSC_MAX_REAL as shift to flag it.
     Special handling is inside SNESTSFormFunction_ARKIMEX and SNESTSFormJacobian_ARKIMEX
     We compute Ydot0 if we restart the step or if we resized the problem after remeshing
  */
  if (dirk && tab->explicit_first_stage && (ts->steprestart || ts->stepresize)) {
    ark->scoeff = PETSC_MAX_REAL;
    PetscCall(VecCopy(ts->vec_sol, Z));
    if (!ark->alg_is) {
      PetscCall(TSARKIMEXComputeAlgebraicIS(ts, ts->ptime, Z, &ark->alg_is));
      PetscCall(ISViewFromOptions(ark->alg_is, (PetscObject)ts, "-ts_arkimex_algebraic_is_view"));
    }
    PetscCall(TSGetSNES(ts, &snes));
    PetscCall(PetscObjectIncrementTabLevel((PetscObject)snes, (PetscObject)snes, 1));
    PetscCall(SNESSolve(snes, NULL, Ydot0));
    if (ark->alg_is) PetscCall(VecISSet(Ydot0, ark->alg_is, 0.0));
    PetscCall(PetscObjectIncrementTabLevel((PetscObject)snes, (PetscObject)snes, -1));
  }

  /* For IMEX we compute a step */
  if (!dirk && ts->equation_type >= TS_EQ_IMPLICIT && tab->explicit_first_stage && ts->steprestart) {
    TS ts_start;
    if (PetscDefined(USE_DEBUG) && hasE) {
      PetscBool id = PETSC_FALSE;
      PetscCall(TSARKIMEXTestMassIdentity(ts, &id));
      PetscCheck(id, PetscObjectComm((PetscObject)ts), PETSC_ERR_ARG_INCOMP, "This scheme requires an identity mass matrix, however the TSIFunctionFn you provided does not utilize an identity mass matrix");
    }
    PetscCall(TSClone(ts, &ts_start));
    PetscCall(TSSetSolution(ts_start, ts->vec_sol));
    PetscCall(TSSetTime(ts_start, ts->ptime));
    PetscCall(TSSetMaxSteps(ts_start, ts->steps + 1));
    PetscCall(TSSetMaxTime(ts_start, ts->ptime + ts->time_step));
    PetscCall(TSSetExactFinalTime(ts_start, TS_EXACTFINALTIME_STEPOVER));
    PetscCall(TSSetTimeStep(ts_start, ts->time_step));
    PetscCall(TSSetType(ts_start, TSARKIMEX));
    PetscCall(TSARKIMEXSetFullyImplicit(ts_start, PETSC_TRUE));
    PetscCall(TSARKIMEXSetType(ts_start, TSARKIMEX1BEE));

    PetscCall(TSRestartStep(ts_start));
    PetscCall(TSSolve(ts_start, ts->vec_sol));
    PetscCall(TSGetTime(ts_start, &ts->ptime));
    PetscCall(TSGetTimeStep(ts_start, &ts->time_step));

    { /* Save the initial slope for the next step */
      TS_ARKIMEX *ark_start = (TS_ARKIMEX *)ts_start->data;
      PetscCall(VecCopy(ark_start->YdotI[ark_start->tableau->s - 1], Ydot0));
    }
    ts->steps++;

    /* Set the correct TS in SNES */
    /* We'll try to bypass this by changing the method on the fly */
    {
      PetscCall(TSGetSNES(ts, &snes));
      PetscCall(TSSetSNES(ts, snes));
    }
    PetscCall(TSDestroy(&ts_start));
  }

  ark->status = TS_STEP_INCOMPLETE;
  while (!ts->reason && ark->status != TS_STEP_COMPLETE) {
    PetscReal t = ts->ptime;
    PetscReal h = ts->time_step;
    for (i = 0; i < s; i++) {
      ark->stage_time = t + h * ct[i];
      PetscCall(TSPreStage(ts, ark->stage_time));
      if (At[i * s + i] == 0) { /* This stage is explicit */
        PetscCheck(i == 0 || ts->equation_type < TS_EQ_IMPLICIT, PetscObjectComm((PetscObject)ts), PETSC_ERR_SUP, "Explicit stages other than the first one are not supported for implicit problems");
        PetscCall(VecCopy(ts->vec_sol, Y[i]));
        for (j = 0; j < i; j++) w[j] = h * At[i * s + j];
        PetscCall(VecMAXPY(Y[i], i, w, YdotI));
        if (tab->additive && hasE) {
          for (j = 0; j < i; j++) w[j] = h * A[i * s + j];
          PetscCall(VecMAXPY(Y[i], i, w, YdotRHS));
        }
        PetscCall(TSGetSNES(ts, &snes));
        PetscCall(SNESResetCounters(snes));
      } else {
        ark->scoeff = 1. / At[i * s + i];
        /* Ydot = shift*(Y-Z) */
        PetscCall(VecCopy(ts->vec_sol, Z));
        for (j = 0; j < i; j++) w[j] = h * At[i * s + j];
        PetscCall(VecMAXPY(Z, i, w, YdotI));
        if (tab->additive && hasE) {
          for (j = 0; j < i; j++) w[j] = h * A[i * s + j];
          PetscCall(VecMAXPY(Z, i, w, YdotRHS));
        }
        if (extrapolate && !ts->steprestart) {
          /* Initial guess extrapolated from previous time step stage values */
          PetscCall(TSExtrapolate_ARKIMEX(ts, c[i], Y[i]));
        } else {
          /* Initial guess taken from last stage */
          PetscCall(VecCopy(i > 0 ? Y[i - 1] : ts->vec_sol, Y[i]));
        }
        PetscCall(TSGetSNES(ts, &snes));
        PetscCall(SNESSolve(snes, NULL, Y[i]));
        PetscCall(SNESGetIterationNumber(snes, &its));
        PetscCall(SNESGetLinearSolveIterations(snes, &lits));
        ts->snes_its += its;
        ts->ksp_its += lits;
        PetscCall(TSGetAdapt(ts, &adapt));
        PetscCall(TSAdaptCheckStage(adapt, ts, ark->stage_time, Y[i], &stageok));
        if (!stageok) {
          /* We are likely rejecting the step because of solver or function domain problems so we should not attempt to
           * use extrapolation to initialize the solves on the next attempt. */
          extrapolate = PETSC_FALSE;
          goto reject_step;
        }
      }
      if (dirk || ts->equation_type >= TS_EQ_IMPLICIT) {
        if (i == 0 && tab->explicit_first_stage) {
          PetscCheck(tab->stiffly_accurate, PetscObjectComm((PetscObject)ts), PETSC_ERR_SUP, "%s %s is not stiffly accurate and therefore explicit-first stage methods cannot be used if the equation is implicit because the slope cannot be evaluated",
                     ((PetscObject)ts)->type_name, ark->tableau->name);
          PetscCall(VecCopy(Ydot0, YdotI[0])); /* YdotI = YdotI(tn-1) */
        } else {
          PetscCall(VecAXPBYPCZ(YdotI[i], -ark->scoeff / h, ark->scoeff / h, 0, Z, Y[i])); /* YdotI = shift*(X-Z) */
        }
      } else {
        if (i == 0 && tab->explicit_first_stage) {
          PetscCall(VecZeroEntries(Ydot));
          PetscCall(TSComputeIFunction(ts, t + h * ct[i], Y[i], Ydot, YdotI[i], ark->imex)); /* YdotI = -G(t,Y,0)   */
          PetscCall(VecScale(YdotI[i], -1.0));
        } else {
          PetscCall(VecAXPBYPCZ(YdotI[i], -ark->scoeff / h, ark->scoeff / h, 0, Z, Y[i])); /* YdotI = shift*(X-Z) */
        }
        if (hasE) {
          if (ark->imex) {
            PetscCall(TSComputeRHSFunction(ts, t + h * c[i], Y[i], YdotRHS[i]));
          } else {
            PetscCall(VecZeroEntries(YdotRHS[i]));
          }
        }
      }
      PetscCall(TSPostStage(ts, ark->stage_time, i, Y));
    }

    ark->status = TS_STEP_INCOMPLETE;
    PetscCall(TSEvaluateStep_ARKIMEX(ts, tab->order, ts->vec_sol, NULL));
    ark->status = TS_STEP_PENDING;
    PetscCall(TSGetAdapt(ts, &adapt));
    PetscCall(TSAdaptCandidatesClear(adapt));
    PetscCall(TSAdaptCandidateAdd(adapt, tab->name, tab->order, 1, tab->ccfl, (PetscReal)tab->s, PETSC_TRUE));
    PetscCall(TSAdaptChoose(adapt, ts, ts->time_step, NULL, &next_time_step, &accept));
    ark->status = accept ? TS_STEP_COMPLETE : TS_STEP_INCOMPLETE;
    if (!accept) { /* Roll back the current step */
      PetscCall(VecCopy(ts->vec_sol0, ts->vec_sol));
      ts->time_step = next_time_step;
      goto reject_step;
    }

    ts->ptime += ts->time_step;
    ts->time_step = next_time_step;
    break;

  reject_step:
    ts->reject++;
    accept = PETSC_FALSE;
    if (!ts->reason && ++rejections > ts->max_reject && ts->max_reject >= 0) {
      ts->reason = TS_DIVERGED_STEP_REJECTED;
      PetscCall(PetscInfo(ts, "Step=%" PetscInt_FMT ", step rejections %" PetscInt_FMT " greater than current TS allowed, stopping solve\n", ts->steps, rejections));
    }
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
  This adjoint step function assumes the partitioned ODE system has an identity mass matrix and thus can be represented in the form
  Udot = H(t,U) + G(t,U)
  This corresponds to F(t,U,Udot) = Udot-H(t,U).

  The complete adjoint equations are
  (shift*I - dHdu) lambda_s[i]   = 1/at[i][i] (
    dGdU (b_i lambda_{n+1} + \sum_{j=i+1}^s a[j][i] lambda_s[j])
    + dHdU (bt[i] lambda_{n+1} +  \sum_{j=i+1}^s at[j][i] lambda_s[j])), i = s-1,...,0
  lambda_n = lambda_{n+1} + \sum_{j=1}^s lambda_s[j]
  mu_{n+1}[i]   = h (at[i][i] dHdP lambda_s[i]
    + dGdP (b_i lambda_{n+1} + \sum_{j=i+1}^s a[j][i] lambda_s[j])
    + dHdP (bt[i] lambda_{n+1} + \sum_{j=i+1}^s at[j][i] lambda_s[j])), i = s-1,...,0
  mu_n = mu_{n+1} + \sum_{j=1}^s mu_{n+1}[j]
  where shift = 1/(at[i][i]*h)

  If at[i][i] is 0, the first equation falls back to
  lambda_s[i] = h (
    (b_i dGdU + bt[i] dHdU) lambda_{n+1} + dGdU \sum_{j=i+1}^s a[j][i] lambda_s[j]
    + dHdU \sum_{j=i+1}^s at[j][i] lambda_s[j])

*/
static PetscErrorCode TSAdjointStep_ARKIMEX(TS ts)
{
  TS_ARKIMEX      *ark = (TS_ARKIMEX *)ts->data;
  ARKTableau       tab = ark->tableau;
  const PetscInt   s   = tab->s;
  const PetscReal *At = tab->At, *A = tab->A, *ct = tab->ct, *c = tab->c, *b = tab->b, *bt = tab->bt;
  PetscScalar     *w = ark->work;
  Vec             *Y = ark->Y, Ydot = ark->Ydot, *VecsDeltaLam = ark->VecsDeltaLam, *VecsSensiTemp = ark->VecsSensiTemp, *VecsSensiPTemp = ark->VecsSensiPTemp;
  Mat              Jex, Jim, Jimpre;
  PetscInt         i, j, nadj;
  PetscReal        t                 = ts->ptime, stage_time_ex;
  PetscReal        adjoint_time_step = -ts->time_step; /* always positive since ts->time_step is negative */
  KSP              ksp;

  PetscFunctionBegin;
  ark->status = TS_STEP_INCOMPLETE;
  PetscCall(SNESGetKSP(ts->snes, &ksp));
  PetscCall(TSGetRHSMats_Private(ts, &Jex, NULL));
  PetscCall(TSGetIJacobian(ts, &Jim, &Jimpre, NULL, NULL));

  for (i = s - 1; i >= 0; i--) {
    ark->stage_time = t - adjoint_time_step * (1.0 - ct[i]);
    stage_time_ex   = t - adjoint_time_step * (1.0 - c[i]);
    if (At[i * s + i] == 0) { // This stage is explicit
      ark->scoeff = 0.;
    } else {
      ark->scoeff = -1. / At[i * s + i]; // this makes shift=ark->scoeff/ts->time_step positive since ts->time_step is negative
    }
    PetscCall(TSComputeSNESJacobian(ts, Y[i], Jim, Jimpre));
    PetscCall(TSComputeRHSJacobian(ts, stage_time_ex, Y[i], Jex, Jex));
    if (ts->vecs_sensip) {
      PetscCall(TSComputeIJacobianP(ts, ark->stage_time, Y[i], Ydot, ark->scoeff / adjoint_time_step, ts->Jacp, PETSC_TRUE)); // get dFdP (-dHdP), Ydot not really used since mass matrix is identity
      PetscCall(TSComputeRHSJacobianP(ts, stage_time_ex, Y[i], ts->Jacprhs));                                                 // get dGdP
    }
    /* Build RHS (stored in VecsDeltaLam) for first-order adjoint */
    for (nadj = 0; nadj < ts->numcost; nadj++) {
      /* build implicit part */
      PetscCall(VecSet(VecsSensiTemp[nadj], 0));
      if (s - i - 1 > 0) {
        /* Temp = -\sum_{j=i+1}^s at[j][i] lambda_s[j] */
        for (j = i + 1; j < s; j++) w[j - i - 1] = -At[j * s + i];
        PetscCall(VecMAXPY(VecsSensiTemp[nadj], s - i - 1, w, &VecsDeltaLam[nadj * s + i + 1]));
      }
      /* Temp = Temp - bt[i] lambda_{n+1} */
      PetscCall(VecAXPY(VecsSensiTemp[nadj], -bt[i], ts->vecs_sensi[nadj]));
      if (bt[i] || s - i - 1 > 0) {
        /* (shift I - dHdU) Temp */
        PetscCall(MatMultTranspose(Jim, VecsSensiTemp[nadj], VecsDeltaLam[nadj * s + i]));
        /* cancel out shift Temp where shift=-scoeff/h */
        PetscCall(VecAXPY(VecsDeltaLam[nadj * s + i], ark->scoeff / adjoint_time_step, VecsSensiTemp[nadj]));
        if (ts->vecs_sensip) {
          /* - dHdP Temp */
          PetscCall(MatMultTranspose(ts->Jacp, VecsSensiTemp[nadj], VecsSensiPTemp[nadj]));
          /* mu_n += -h dHdP Temp */
          PetscCall(VecAXPY(ts->vecs_sensip[nadj], adjoint_time_step, VecsSensiPTemp[nadj]));
        }
      } else {
        PetscCall(VecSet(VecsDeltaLam[nadj * s + i], 0)); // make sure it is initialized
      }
      /* build explicit part */
      PetscCall(VecSet(VecsSensiTemp[nadj], 0));
      if (s - i - 1 > 0) {
        /* Temp = \sum_{j=i+1}^s a[j][i] lambda_s[j] */
        for (j = i + 1; j < s; j++) w[j - i - 1] = A[j * s + i];
        PetscCall(VecMAXPY(VecsSensiTemp[nadj], s - i - 1, w, &VecsDeltaLam[nadj * s + i + 1]));
      }
      /* Temp = Temp + b[i] lambda_{n+1} */
      PetscCall(VecAXPY(VecsSensiTemp[nadj], b[i], ts->vecs_sensi[nadj]));
      if (b[i] || s - i - 1 > 0) {
        /* dGdU Temp */
        PetscCall(MatMultTransposeAdd(Jex, VecsSensiTemp[nadj], VecsDeltaLam[nadj * s + i], VecsDeltaLam[nadj * s + i]));
        if (ts->vecs_sensip) {
          /* dGdP Temp */
          PetscCall(MatMultTranspose(ts->Jacprhs, VecsSensiTemp[nadj], VecsSensiPTemp[nadj]));
          /* mu_n += h dGdP Temp */
          PetscCall(VecAXPY(ts->vecs_sensip[nadj], adjoint_time_step, VecsSensiPTemp[nadj]));
        }
      }
      /* Build LHS for first-order adjoint */
      if (At[i * s + i] == 0) { // This stage is explicit
        PetscCall(VecScale(VecsDeltaLam[nadj * s + i], adjoint_time_step));
      } else {
        KSP                ksp;
        KSPConvergedReason kspreason;
        PetscCall(SNESGetKSP(ts->snes, &ksp));
        PetscCall(KSPSetOperators(ksp, Jim, Jimpre));
        PetscCall(VecScale(VecsDeltaLam[nadj * s + i], 1. / At[i * s + i]));
        PetscCall(KSPSolveTranspose(ksp, VecsDeltaLam[nadj * s + i], VecsDeltaLam[nadj * s + i]));
        PetscCall(KSPGetConvergedReason(ksp, &kspreason));
        if (kspreason < 0) {
          ts->reason = TSADJOINT_DIVERGED_LINEAR_SOLVE;
          PetscCall(PetscInfo(ts, "Step=%" PetscInt_FMT ", %" PetscInt_FMT "th cost function, transposed linear solve fails, stopping 1st-order adjoint solve\n", ts->steps, nadj));
        }
        if (ts->vecs_sensip) {
          /* -dHdP lambda_s[i] */
          PetscCall(MatMultTranspose(ts->Jacp, VecsDeltaLam[nadj * s + i], VecsSensiPTemp[nadj]));
          /* mu_n += h at[i][i] dHdP lambda_s[i] */
          PetscCall(VecAXPY(ts->vecs_sensip[nadj], -At[i * s + i] * adjoint_time_step, VecsSensiPTemp[nadj]));
        }
      }
    }
  }
  for (j = 0; j < s; j++) w[j] = 1.0;
  for (nadj = 0; nadj < ts->numcost; nadj++) // no need to do this for mu's
    PetscCall(VecMAXPY(ts->vecs_sensi[nadj], s, w, &VecsDeltaLam[nadj * s]));
  ark->status = TS_STEP_COMPLETE;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TSInterpolate_ARKIMEX(TS ts, PetscReal itime, Vec X)
{
  TS_ARKIMEX      *ark = (TS_ARKIMEX *)ts->data;
  ARKTableau       tab = ark->tableau;
  PetscInt         s = tab->s, pinterp = tab->pinterp, i, j;
  PetscReal        h;
  PetscReal        tt, t;
  PetscScalar     *bt = ark->work, *b = ark->work + s;
  const PetscReal *Bt = tab->binterpt, *B = tab->binterp;

  PetscFunctionBegin;
  PetscCheck(Bt && B, PetscObjectComm((PetscObject)ts), PETSC_ERR_SUP, "%s %s does not have an interpolation formula", ((PetscObject)ts)->type_name, ark->tableau->name);
  switch (ark->status) {
  case TS_STEP_INCOMPLETE:
  case TS_STEP_PENDING:
    h = ts->time_step;
    t = (itime - ts->ptime) / h;
    break;
  case TS_STEP_COMPLETE:
    h = ts->ptime - ts->ptime_prev;
    t = (itime - ts->ptime) / h + 1; /* In the interval [0,1] */
    break;
  default:
    SETERRQ(PetscObjectComm((PetscObject)ts), PETSC_ERR_PLIB, "Invalid TSStepStatus");
  }
  for (i = 0; i < s; i++) bt[i] = b[i] = 0;
  for (j = 0, tt = t; j < pinterp; j++, tt *= t) {
    for (i = 0; i < s; i++) {
      bt[i] += h * Bt[i * pinterp + j] * tt;
      b[i] += h * B[i * pinterp + j] * tt;
    }
  }
  PetscCall(VecCopy(ark->Y[0], X));
  PetscCall(VecMAXPY(X, s, bt, ark->YdotI));
  if (tab->additive) {
    PetscBool hasE;
    PetscCall(TSHasRHSFunction(ts, &hasE));
    if (hasE) PetscCall(VecMAXPY(X, s, b, ark->YdotRHS));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TSExtrapolate_ARKIMEX(TS ts, PetscReal c, Vec X)
{
  TS_ARKIMEX      *ark = (TS_ARKIMEX *)ts->data;
  ARKTableau       tab = ark->tableau;
  PetscInt         s = tab->s, pinterp = tab->pinterp, i, j;
  PetscReal        h, h_prev, t, tt;
  PetscScalar     *bt = ark->work, *b = ark->work + s;
  const PetscReal *Bt = tab->binterpt, *B = tab->binterp;

  PetscFunctionBegin;
  PetscCheck(Bt && B, PetscObjectComm((PetscObject)ts), PETSC_ERR_SUP, "TSARKIMEX %s does not have an interpolation formula", ark->tableau->name);
  h      = ts->time_step;
  h_prev = ts->ptime - ts->ptime_prev;
  t      = 1 + h / h_prev * c;
  for (i = 0; i < s; i++) bt[i] = b[i] = 0;
  for (j = 0, tt = t; j < pinterp; j++, tt *= t) {
    for (i = 0; i < s; i++) {
      bt[i] += h * Bt[i * pinterp + j] * tt;
      b[i] += h * B[i * pinterp + j] * tt;
    }
  }
  PetscCheck(ark->Y_prev, PetscObjectComm((PetscObject)ts), PETSC_ERR_SUP, "Stages from previous step have not been stored");
  PetscCall(VecCopy(ark->Y_prev[0], X));
  PetscCall(VecMAXPY(X, s, bt, ark->YdotI_prev));
  if (tab->additive) {
    PetscBool hasE;
    PetscCall(TSHasRHSFunction(ts, &hasE));
    if (hasE) PetscCall(VecMAXPY(X, s, b, ark->YdotRHS_prev));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TSARKIMEXTableauReset(TS ts)
{
  TS_ARKIMEX *ark = (TS_ARKIMEX *)ts->data;
  ARKTableau  tab = ark->tableau;

  PetscFunctionBegin;
  if (!tab) PetscFunctionReturn(PETSC_SUCCESS);
  PetscCall(PetscFree(ark->work));
  PetscCall(VecDestroyVecs(tab->s, &ark->Y));
  PetscCall(VecDestroyVecs(tab->s, &ark->YdotI));
  PetscCall(VecDestroyVecs(tab->s, &ark->YdotRHS));
  PetscCall(VecDestroyVecs(tab->s, &ark->Y_prev));
  PetscCall(VecDestroyVecs(tab->s, &ark->YdotI_prev));
  PetscCall(VecDestroyVecs(tab->s, &ark->YdotRHS_prev));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TSReset_ARKIMEX(TS ts)
{
  TS_ARKIMEX *ark = (TS_ARKIMEX *)ts->data;

  PetscFunctionBegin;
  if (ark->fastslowsplit) {
    PetscTryMethod(ts, "TSReset_ARKIMEX_FastSlowSplit_C", (TS), (ts));
  } else {
    PetscCall(TSARKIMEXTableauReset(ts));
    PetscCall(VecDestroy(&ark->Ydot));
    PetscCall(VecDestroy(&ark->Ydot0));
    PetscCall(VecDestroy(&ark->Z));
    PetscCall(ISDestroy(&ark->alg_is));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TSAdjointReset_ARKIMEX(TS ts)
{
  TS_ARKIMEX *ark = (TS_ARKIMEX *)ts->data;
  ARKTableau  tab = ark->tableau;

  PetscFunctionBegin;
  PetscCall(VecDestroyVecs(tab->s * ts->numcost, &ark->VecsDeltaLam));
  PetscCall(VecDestroyVecs(ts->numcost, &ark->VecsSensiTemp));
  PetscCall(VecDestroyVecs(ts->numcost, &ark->VecsSensiPTemp));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TSARKIMEXGetVecs(TS ts, DM dm, Vec *Z, Vec *Ydot)
{
  TS_ARKIMEX *ax = (TS_ARKIMEX *)ts->data;

  PetscFunctionBegin;
  if (Z) {
    if (dm && dm != ts->dm) PetscCall(DMGetNamedGlobalVector(dm, "TSARKIMEX_Z", Z));
    else *Z = ax->Z;
  }
  if (Ydot) {
    if (dm && dm != ts->dm) PetscCall(DMGetNamedGlobalVector(dm, "TSARKIMEX_Ydot", Ydot));
    else *Ydot = ax->Ydot;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TSARKIMEXRestoreVecs(TS ts, DM dm, Vec *Z, Vec *Ydot)
{
  PetscFunctionBegin;
  if (Z) {
    if (dm && dm != ts->dm) PetscCall(DMRestoreNamedGlobalVector(dm, "TSARKIMEX_Z", Z));
  }
  if (Ydot) {
    if (dm && dm != ts->dm) PetscCall(DMRestoreNamedGlobalVector(dm, "TSARKIMEX_Ydot", Ydot));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
  DAEs need special handling for algebraic variables when restarting DIRK methods with explicit
  first stage. In particular, we need:
     - to zero the nonlinear function (in case the dual variables are not consistent in the first step)
     - to modify the matrix by calling MatZeroRows with identity on these variables.
*/
static PetscErrorCode TSARKIMEXComputeAlgebraicIS(TS ts, PetscReal time, Vec X, IS *alg_is)
{
  TS_ARKIMEX        *ark = (TS_ARKIMEX *)ts->data;
  DM                 dm;
  Vec                F, W, Xdot;
  const PetscScalar *w;
  PetscInt           nz = 0, n, st;
  PetscInt          *nzr;

  PetscFunctionBegin;
  PetscCall(TSGetDM(ts, &dm)); /* may be already from SNES */
  PetscCall(DMGetGlobalVector(dm, &Xdot));
  PetscCall(DMGetGlobalVector(dm, &F));
  PetscCall(DMGetGlobalVector(dm, &W));
  PetscCall(VecSet(Xdot, 0.0));
  PetscCall(TSComputeIFunction(ts, time, X, Xdot, F, ark->imex));
  PetscCall(VecSetRandom(Xdot, NULL));
  PetscCall(TSComputeIFunction(ts, time, X, Xdot, W, ark->imex));
  PetscCall(VecAXPY(W, -1.0, F));
  PetscCall(VecGetOwnershipRange(W, &st, NULL));
  PetscCall(VecGetLocalSize(W, &n));
  PetscCall(VecGetArrayRead(W, &w));
  for (PetscInt i = 0; i < n; i++)
    if (w[i] == 0.0) nz++;
  PetscCall(PetscMalloc1(nz, &nzr));
  nz = 0;
  for (PetscInt i = 0; i < n; i++)
    if (w[i] == 0.0) nzr[nz++] = i + st;
  PetscCall(VecRestoreArrayRead(W, &w));
  PetscCall(ISCreateGeneral(PetscObjectComm((PetscObject)dm), nz, nzr, PETSC_OWN_POINTER, alg_is));
  PetscCall(DMRestoreGlobalVector(dm, &Xdot));
  PetscCall(DMRestoreGlobalVector(dm, &F));
  PetscCall(DMRestoreGlobalVector(dm, &W));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* As for the method specific Z and Ydot, we store the algebraic IS in the ARKIMEX data structure
   at the finest level, in the DM for coarser solves. */
static PetscErrorCode TSARKIMEXGetAlgebraicIS(TS ts, DM dm, IS *alg_is)
{
  TS_ARKIMEX *ax = (TS_ARKIMEX *)ts->data;

  PetscFunctionBegin;
  if (dm && dm != ts->dm) PetscCall(PetscObjectQuery((PetscObject)dm, "TSARKIMEX_ALG_IS", (PetscObject *)alg_is));
  else *alg_is = ax->alg_is;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* This defines the nonlinear equation that is to be solved with SNES */
static PetscErrorCode SNESTSFormFunction_ARKIMEX(SNES snes, Vec X, Vec F, TS ts)
{
  TS_ARKIMEX *ark = (TS_ARKIMEX *)ts->data;
  DM          dm, dmsave;
  Vec         Z, Ydot;
  IS          alg_is;

  PetscFunctionBegin;
  PetscCall(SNESGetDM(snes, &dm));
  PetscCall(TSARKIMEXGetVecs(ts, dm, &Z, &Ydot));
  if (ark->scoeff == PETSC_MAX_REAL) PetscCall(TSARKIMEXGetAlgebraicIS(ts, dm, &alg_is));

  dmsave = ts->dm;
  ts->dm = dm;

  if (ark->scoeff == PETSC_MAX_REAL) {
    /* We are solving F(t_n,x_n,xdot) = 0 to start the method */
    if (!alg_is) {
      PetscCheck(dmsave != ts->dm, PetscObjectComm((PetscObject)dm), PETSC_ERR_PLIB, "Missing algebraic IS");
      PetscCall(TSARKIMEXComputeAlgebraicIS(ts, ark->stage_time, Z, &alg_is));
      PetscCall(PetscObjectCompose((PetscObject)dm, "TSARKIMEX_ALG_IS", (PetscObject)alg_is));
      PetscCall(PetscObjectDereference((PetscObject)alg_is));
      PetscCall(ISViewFromOptions(alg_is, (PetscObject)snes, "-ts_arkimex_algebraic_is_view"));
    }
    PetscCall(TSComputeIFunction(ts, ark->stage_time, Z, X, F, ark->imex));
    PetscCall(VecISSet(F, alg_is, 0.0));
  } else {
    PetscReal shift = ark->scoeff / ts->time_step;
    PetscCall(VecAXPBYPCZ(Ydot, -shift, shift, 0, Z, X)); /* Ydot = shift*(X-Z) */
    PetscCall(TSComputeIFunction(ts, ark->stage_time, X, Ydot, F, ark->imex));
  }

  ts->dm = dmsave;
  PetscCall(TSARKIMEXRestoreVecs(ts, dm, &Z, &Ydot));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode SNESTSFormJacobian_ARKIMEX(SNES snes, Vec X, Mat A, Mat B, TS ts)
{
  TS_ARKIMEX *ark = (TS_ARKIMEX *)ts->data;
  DM          dm, dmsave;
  Vec         Ydot, Z;
  PetscReal   shift;
  IS          alg_is;

  PetscFunctionBegin;
  PetscCall(SNESGetDM(snes, &dm));
  /* ark->Ydot has already been computed in SNESTSFormFunction_ARKIMEX (SNES guarantees this) */
  PetscCall(TSARKIMEXGetVecs(ts, dm, &Z, &Ydot));
  /* alg_is has been computed in SNESTSFormFunction_ARKIMEX */
  if (ark->scoeff == PETSC_MAX_REAL) PetscCall(TSARKIMEXGetAlgebraicIS(ts, dm, &alg_is));

  dmsave = ts->dm;
  ts->dm = dm;

  if (ark->scoeff == PETSC_MAX_REAL) {
    PetscBool hasZeroRows;

    /* We are solving F(t_n,x_n,xdot) = 0 to start the method
       We compute with a very large shift and then scale back the matrix */
    shift = 1.0 / PETSC_MACHINE_EPSILON;
    PetscCall(TSComputeIJacobian(ts, ark->stage_time, Z, X, shift, A, B, ark->imex));
    PetscCall(MatScale(B, PETSC_MACHINE_EPSILON));
    PetscCall(MatHasOperation(B, MATOP_ZERO_ROWS, &hasZeroRows));
    if (hasZeroRows) {
      PetscCheck(alg_is, PetscObjectComm((PetscObject)dm), PETSC_ERR_PLIB, "Missing algebraic IS");
      /* the default of AIJ is to not keep the pattern! We should probably change it someday */
      PetscCall(MatSetOption(B, MAT_KEEP_NONZERO_PATTERN, PETSC_TRUE));
      PetscCall(MatZeroRowsIS(B, alg_is, 1.0, NULL, NULL));
    }
    PetscCall(MatViewFromOptions(B, (PetscObject)snes, "-ts_arkimex_alg_mat_view"));
    if (A != B) PetscCall(MatScale(A, PETSC_MACHINE_EPSILON));
  } else {
    shift = ark->scoeff / ts->time_step;
    PetscCall(TSComputeIJacobian(ts, ark->stage_time, X, Ydot, shift, A, B, ark->imex));
  }
  ts->dm = dmsave;
  PetscCall(TSARKIMEXRestoreVecs(ts, dm, &Z, &Ydot));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TSGetStages_ARKIMEX(TS ts, PetscInt *ns, Vec *Y[])
{
  TS_ARKIMEX *ark = (TS_ARKIMEX *)ts->data;

  PetscFunctionBegin;
  if (ns) *ns = ark->tableau->s;
  if (Y) *Y = ark->Y;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode DMCoarsenHook_TSARKIMEX(DM fine, DM coarse, void *ctx)
{
  PetscFunctionBegin;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode DMRestrictHook_TSARKIMEX(DM fine, Mat restrct, Vec rscale, Mat inject, DM coarse, void *ctx)
{
  TS  ts = (TS)ctx;
  Vec Z, Z_c;

  PetscFunctionBegin;
  PetscCall(TSARKIMEXGetVecs(ts, fine, &Z, NULL));
  PetscCall(TSARKIMEXGetVecs(ts, coarse, &Z_c, NULL));
  PetscCall(MatRestrict(restrct, Z, Z_c));
  PetscCall(VecPointwiseMult(Z_c, rscale, Z_c));
  PetscCall(TSARKIMEXRestoreVecs(ts, fine, &Z, NULL));
  PetscCall(TSARKIMEXRestoreVecs(ts, coarse, &Z_c, NULL));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode DMSubDomainHook_TSARKIMEX(DM dm, DM subdm, void *ctx)
{
  PetscFunctionBegin;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode DMSubDomainRestrictHook_TSARKIMEX(DM dm, VecScatter gscat, VecScatter lscat, DM subdm, void *ctx)
{
  TS  ts = (TS)ctx;
  Vec Z, Z_c;

  PetscFunctionBegin;
  PetscCall(TSARKIMEXGetVecs(ts, dm, &Z, NULL));
  PetscCall(TSARKIMEXGetVecs(ts, subdm, &Z_c, NULL));

  PetscCall(VecScatterBegin(gscat, Z, Z_c, INSERT_VALUES, SCATTER_FORWARD));
  PetscCall(VecScatterEnd(gscat, Z, Z_c, INSERT_VALUES, SCATTER_FORWARD));

  PetscCall(TSARKIMEXRestoreVecs(ts, dm, &Z, NULL));
  PetscCall(TSARKIMEXRestoreVecs(ts, subdm, &Z_c, NULL));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TSARKIMEXTableauSetUp(TS ts)
{
  TS_ARKIMEX *ark = (TS_ARKIMEX *)ts->data;
  ARKTableau  tab = ark->tableau;

  PetscFunctionBegin;
  PetscCall(PetscMalloc1(2 * tab->s, &ark->work));
  PetscCall(VecDuplicateVecs(ts->vec_sol, tab->s, &ark->Y));
  PetscCall(VecDuplicateVecs(ts->vec_sol, tab->s, &ark->YdotI));
  if (tab->additive) PetscCall(VecDuplicateVecs(ts->vec_sol, tab->s, &ark->YdotRHS));
  if (ark->extrapolate) {
    PetscCall(VecDuplicateVecs(ts->vec_sol, tab->s, &ark->Y_prev));
    PetscCall(VecDuplicateVecs(ts->vec_sol, tab->s, &ark->YdotI_prev));
    if (tab->additive) PetscCall(VecDuplicateVecs(ts->vec_sol, tab->s, &ark->YdotRHS_prev));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TSSetUp_ARKIMEX(TS ts)
{
  TS_ARKIMEX *ark = (TS_ARKIMEX *)ts->data;
  DM          dm;
  SNES        snes;

  PetscFunctionBegin;
  if (ark->fastslowsplit) {
    PetscTryMethod(ts, "TSSetUp_ARKIMEX_FastSlowSplit_C", (TS), (ts));
  } else {
    PetscCall(TSARKIMEXTableauSetUp(ts));
    PetscCall(VecDuplicate(ts->vec_sol, &ark->Ydot));
    PetscCall(VecDuplicate(ts->vec_sol, &ark->Ydot0));
    PetscCall(VecDuplicate(ts->vec_sol, &ark->Z));
    PetscCall(TSGetDM(ts, &dm));
    PetscCall(DMCoarsenHookAdd(dm, DMCoarsenHook_TSARKIMEX, DMRestrictHook_TSARKIMEX, ts));
    PetscCall(DMSubDomainHookAdd(dm, DMSubDomainHook_TSARKIMEX, DMSubDomainRestrictHook_TSARKIMEX, ts));
    PetscCall(TSGetSNES(ts, &snes));
    PetscCall(SNESSetDM(snes, dm));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TSAdjointSetUp_ARKIMEX(TS ts)
{
  TS_ARKIMEX *ark = (TS_ARKIMEX *)ts->data;
  ARKTableau  tab = ark->tableau;

  PetscFunctionBegin;
  PetscCall(VecDuplicateVecs(ts->vecs_sensi[0], tab->s * ts->numcost, &ark->VecsDeltaLam));
  PetscCall(VecDuplicateVecs(ts->vecs_sensi[0], ts->numcost, &ark->VecsSensiTemp));
  if (ts->vecs_sensip) PetscCall(VecDuplicateVecs(ts->vecs_sensip[0], ts->numcost, &ark->VecsSensiPTemp));
  if (PetscDefined(USE_DEBUG)) {
    PetscBool id = PETSC_FALSE;
    PetscCall(TSARKIMEXTestMassIdentity(ts, &id));
    PetscCheck(id, PetscObjectComm((PetscObject)ts), PETSC_ERR_ARG_INCOMP, "Adjoint ARKIMEX requires an identity mass matrix, however the TSIFunctionFn you provided does not utilize an identity mass matrix");
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TSSetFromOptions_ARKIMEX(TS ts, PetscOptionItems PetscOptionsObject)
{
  TS_ARKIMEX *ark = (TS_ARKIMEX *)ts->data;
  PetscBool   dirk;

  PetscFunctionBegin;
  PetscCall(PetscObjectTypeCompare((PetscObject)ts, TSDIRK, &dirk));
  PetscOptionsHeadBegin(PetscOptionsObject, dirk ? "DIRK ODE solver options" : "ARKIMEX ODE solver options");
  {
    ARKTableauLink link;
    PetscInt       count, choice;
    PetscBool      flg;
    const char   **namelist;
    for (link = ARKTableauList, count = 0; link; link = link->next) {
      if (!dirk && link->tab.additive) count++;
      if (dirk && !link->tab.additive) count++;
    }
    PetscCall(PetscMalloc1(count, (char ***)&namelist));
    for (link = ARKTableauList, count = 0; link; link = link->next) {
      if (!dirk && link->tab.additive) namelist[count++] = link->tab.name;
      if (dirk && !link->tab.additive) namelist[count++] = link->tab.name;
    }
    if (dirk) {
      PetscCall(PetscOptionsEList("-ts_dirk_type", "Family of DIRK method", "TSDIRKSetType", (const char *const *)namelist, count, ark->tableau->name, &choice, &flg));
      if (flg) PetscCall(TSDIRKSetType(ts, namelist[choice]));
    } else {
      PetscBool fastslowsplit;
      PetscCall(PetscOptionsEList("-ts_arkimex_type", "Family of ARK IMEX method", "TSARKIMEXSetType", (const char *const *)namelist, count, ark->tableau->name, &choice, &flg));
      if (flg) PetscCall(TSARKIMEXSetType(ts, namelist[choice]));
      flg = (PetscBool)!ark->imex;
      PetscCall(PetscOptionsBool("-ts_arkimex_fully_implicit", "Solve the problem fully implicitly", "TSARKIMEXSetFullyImplicit", flg, &flg, NULL));
      ark->imex = (PetscBool)!flg;
      PetscCall(PetscOptionsBool("-ts_arkimex_fastslowsplit", "Use ARK IMEX for fast-slow systems", "TSARKIMEXSetFastSlowSplit", ark->fastslowsplit, &fastslowsplit, &flg));
      if (flg) PetscCall(TSARKIMEXSetFastSlowSplit(ts, fastslowsplit));
      PetscCall(TSARKIMEXGetFastSlowSplit(ts, &fastslowsplit));
      if (fastslowsplit) {
        SNES snes;

        PetscCall(TSRHSSplitGetSNES(ts, &snes));
        PetscCall(SNESSetFromOptions(snes));
      }
    }
    PetscCall(PetscFree(namelist));
    PetscCall(PetscOptionsBool("-ts_arkimex_initial_guess_extrapolate", "Extrapolate the initial guess for the stage solution from stage values of the previous time step", "", ark->extrapolate, &ark->extrapolate, NULL));
  }
  PetscOptionsHeadEnd();
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TSView_ARKIMEX(TS ts, PetscViewer viewer)
{
  TS_ARKIMEX *ark = (TS_ARKIMEX *)ts->data;
  PetscBool   isascii, dirk;

  PetscFunctionBegin;
  PetscCall(PetscObjectTypeCompare((PetscObject)ts, TSDIRK, &dirk));
  PetscCall(PetscObjectTypeCompare((PetscObject)viewer, PETSCVIEWERASCII, &isascii));
  if (isascii) {
    PetscViewerFormat format;
    ARKTableau        tab = ark->tableau;
    TSARKIMEXType     arktype;
    char              buf[2048];
    PetscBool         flg;

    PetscCall(TSARKIMEXGetType(ts, &arktype));
    PetscCall(TSARKIMEXGetFullyImplicit(ts, &flg));
    PetscCall(PetscViewerASCIIPrintf(viewer, "  %s %s\n", dirk ? "DIRK" : "ARK IMEX", arktype));
    PetscCall(PetscFormatRealArray(buf, sizeof(buf), "% 8.6f", tab->s, tab->ct));
    PetscCall(PetscViewerASCIIPrintf(viewer, "  %sabscissa       ct = %s\n", dirk ? "" : "Stiff ", buf));
    PetscCall(PetscViewerGetFormat(viewer, &format));
    if (format == PETSC_VIEWER_ASCII_INFO_DETAIL) {
      PetscCall(PetscViewerASCIIPrintf(viewer, "  %sAt =\n", dirk ? "" : "Stiff "));
      for (PetscInt i = 0; i < tab->s; i++) {
        PetscCall(PetscFormatRealArray(buf, sizeof(buf), "% 8.6f", tab->s, tab->At + i * tab->s));
        PetscCall(PetscViewerASCIIPrintf(viewer, "    %s\n", buf));
      }
      PetscCall(PetscFormatRealArray(buf, sizeof(buf), "% 8.6f", tab->s, tab->bt));
      PetscCall(PetscViewerASCIIPrintf(viewer, "  %sbt = %s\n", dirk ? "" : "Stiff ", buf));
      PetscCall(PetscFormatRealArray(buf, sizeof(buf), "% 8.6f", tab->s, tab->bembedt));
      PetscCall(PetscViewerASCIIPrintf(viewer, "  %sbet = %s\n", dirk ? "" : "Stiff ", buf));
    }
    PetscCall(PetscViewerASCIIPrintf(viewer, "Fully implicit: %s\n", flg ? "yes" : "no"));
    PetscCall(PetscViewerASCIIPrintf(viewer, "Stiffly accurate: %s\n", tab->stiffly_accurate ? "yes" : "no"));
    PetscCall(PetscViewerASCIIPrintf(viewer, "Explicit first stage: %s\n", tab->explicit_first_stage ? "yes" : "no"));
    PetscCall(PetscViewerASCIIPrintf(viewer, "FSAL property: %s\n", tab->FSAL_implicit ? "yes" : "no"));
    if (!dirk) {
      PetscCall(PetscFormatRealArray(buf, sizeof(buf), "% 8.6f", tab->s, tab->c));
      PetscCall(PetscViewerASCIIPrintf(viewer, "  Nonstiff abscissa     c = %s\n", buf));
    }
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TSLoad_ARKIMEX(TS ts, PetscViewer viewer)
{
  SNES    snes;
  TSAdapt adapt;

  PetscFunctionBegin;
  PetscCall(TSGetAdapt(ts, &adapt));
  PetscCall(TSAdaptLoad(adapt, viewer));
  PetscCall(TSGetSNES(ts, &snes));
  PetscCall(SNESLoad(snes, viewer));
  /* function and Jacobian context for SNES when used with TS is always ts object */
  PetscCall(SNESSetFunction(snes, NULL, NULL, ts));
  PetscCall(SNESSetJacobian(snes, NULL, NULL, NULL, ts));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  TSARKIMEXSetType - Set the type of `TSARKIMEX` scheme

  Logically Collective

  Input Parameters:
+ ts      - timestepping context
- arktype - type of `TSARKIMEX` scheme

  Options Database Key:
. -ts_arkimex_type <1bee,a2,l2,ars122,2c,2d,2e,prssp2,3,bpr3,ars443,4,5> - set `TSARKIMEX` scheme type

  Level: intermediate

.seealso: [](ch_ts), `TSARKIMEXGetType()`, `TSARKIMEX`, `TSARKIMEXType`, `TSARKIMEX1BEE`, `TSARKIMEXA2`, `TSARKIMEXL2`, `TSARKIMEXARS122`, `TSARKIMEX2C`, `TSARKIMEX2D`,
          `TSARKIMEX2E`, `TSARKIMEXPRSSP2`, `TSARKIMEX3`, `TSARKIMEXBPR3`, `TSARKIMEXARS443`, `TSARKIMEX4`, `TSARKIMEX5`
@*/
PetscErrorCode TSARKIMEXSetType(TS ts, TSARKIMEXType arktype)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts, TS_CLASSID, 1);
  PetscAssertPointer(arktype, 2);
  PetscTryMethod(ts, "TSARKIMEXSetType_C", (TS, TSARKIMEXType), (ts, arktype));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  TSARKIMEXGetType - Get the type of `TSARKIMEX` scheme

  Logically Collective

  Input Parameter:
. ts - timestepping context

  Output Parameter:
. arktype - type of `TSARKIMEX` scheme

  Level: intermediate

.seealso: [](ch_ts), `TSARKIMEX`
@*/
PetscErrorCode TSARKIMEXGetType(TS ts, TSARKIMEXType *arktype)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts, TS_CLASSID, 1);
  PetscUseMethod(ts, "TSARKIMEXGetType_C", (TS, TSARKIMEXType *), (ts, arktype));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  TSARKIMEXSetFullyImplicit - Solve both parts of the equation implicitly, including the part that is normally solved explicitly

  Logically Collective

  Input Parameters:
+ ts  - timestepping context
- flg - `PETSC_TRUE` for fully implicit

  Options Database Key:
. -ts_arkimex_fully_implicit <true,false> - Solve both parts of the equation implicitly

  Level: intermediate

.seealso: [](ch_ts), `TSARKIMEX`, `TSARKIMEXGetType()`, `TSARKIMEXGetFullyImplicit()`
@*/
PetscErrorCode TSARKIMEXSetFullyImplicit(TS ts, PetscBool flg)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts, TS_CLASSID, 1);
  PetscValidLogicalCollectiveBool(ts, flg, 2);
  PetscTryMethod(ts, "TSARKIMEXSetFullyImplicit_C", (TS, PetscBool), (ts, flg));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  TSARKIMEXGetFullyImplicit - Inquires if both parts of the equation are solved implicitly

  Logically Collective

  Input Parameter:
. ts - timestepping context

  Output Parameter:
. flg - `PETSC_TRUE` for fully implicit

  Level: intermediate

.seealso: [](ch_ts), `TSARKIMEXGetType()`, `TSARKIMEXSetFullyImplicit()`
@*/
PetscErrorCode TSARKIMEXGetFullyImplicit(TS ts, PetscBool *flg)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts, TS_CLASSID, 1);
  PetscAssertPointer(flg, 2);
  PetscUseMethod(ts, "TSARKIMEXGetFullyImplicit_C", (TS, PetscBool *), (ts, flg));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TSARKIMEXGetType_ARKIMEX(TS ts, TSARKIMEXType *arktype)
{
  TS_ARKIMEX *ark = (TS_ARKIMEX *)ts->data;

  PetscFunctionBegin;
  *arktype = ark->tableau->name;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TSARKIMEXSetType_ARKIMEX(TS ts, TSARKIMEXType arktype)
{
  TS_ARKIMEX    *ark = (TS_ARKIMEX *)ts->data;
  PetscBool      match;
  ARKTableauLink link;

  PetscFunctionBegin;
  if (ark->tableau) {
    PetscCall(PetscStrcmp(ark->tableau->name, arktype, &match));
    if (match) PetscFunctionReturn(PETSC_SUCCESS);
  }
  for (link = ARKTableauList; link; link = link->next) {
    PetscCall(PetscStrcmp(link->tab.name, arktype, &match));
    if (match) {
      if (ts->setupcalled) PetscCall(TSARKIMEXTableauReset(ts));
      ark->tableau = &link->tab;
      if (ts->setupcalled) PetscCall(TSARKIMEXTableauSetUp(ts));
      ts->default_adapt_type = ark->tableau->bembed ? TSADAPTBASIC : TSADAPTNONE;
      PetscFunctionReturn(PETSC_SUCCESS);
    }
  }
  SETERRQ(PetscObjectComm((PetscObject)ts), PETSC_ERR_ARG_UNKNOWN_TYPE, "Could not find '%s'", arktype);
}

static PetscErrorCode TSARKIMEXSetFullyImplicit_ARKIMEX(TS ts, PetscBool flg)
{
  TS_ARKIMEX *ark = (TS_ARKIMEX *)ts->data;

  PetscFunctionBegin;
  ark->imex = (PetscBool)!flg;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TSARKIMEXGetFullyImplicit_ARKIMEX(TS ts, PetscBool *flg)
{
  TS_ARKIMEX *ark = (TS_ARKIMEX *)ts->data;

  PetscFunctionBegin;
  *flg = (PetscBool)!ark->imex;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TSDestroy_ARKIMEX(TS ts)
{
  PetscFunctionBegin;
  PetscCall(TSReset_ARKIMEX(ts));
  if (ts->dm) {
    PetscCall(DMCoarsenHookRemove(ts->dm, DMCoarsenHook_TSARKIMEX, DMRestrictHook_TSARKIMEX, ts));
    PetscCall(DMSubDomainHookRemove(ts->dm, DMSubDomainHook_TSARKIMEX, DMSubDomainRestrictHook_TSARKIMEX, ts));
  }
  PetscCall(PetscFree(ts->data));
  PetscCall(PetscObjectComposeFunction((PetscObject)ts, "TSDIRKGetType_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)ts, "TSDIRKSetType_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)ts, "TSARKIMEXGetType_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)ts, "TSARKIMEXSetType_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)ts, "TSARKIMEXSetFullyImplicit_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)ts, "TSARKIMEXGetFullyImplicit_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)ts, "TSARKIMEXSetFastSlowSplit_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)ts, "TSARKIMEXGetFastSlowSplit_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)ts, "TSSetUp_ARKIMEX_FastSlowSplit_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)ts, "TSReset_ARKIMEX_FastSlowSplit_C", NULL));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*MC
  TSARKIMEX - ODE and DAE solver using additive Runge-Kutta IMEX schemes

  These methods are intended for problems with well-separated time scales, especially when a slow scale is strongly
  nonlinear such that it is expensive to solve with a fully implicit method. The user should provide the stiff part
  of the equation using `TSSetIFunction()` and the non-stiff part with `TSSetRHSFunction()`.

  Options Database Keys:
+ -ts_arkimex_type <1bee,a2,l2,ars122,2c,2d,2e,prssp2,3,bpr3,ars443,4,5> - Set `TSARKIMEX` scheme type
. -ts_dirk_type <type>                                                   - Set `TSDIRK` scheme type
. -ts_arkimex_fully_implicit <true,false>                                - Solve both parts of the equation implicitly
. -ts_arkimex_fastslowsplit <true,false>                                 - Enables the `TSARKIMEX` solver for a fast-slow system where the RHS is split component-wise,
                                                                           see `TSRHSSplitSetIS()`
- -ts_arkimex_initial_guess_extrapolate                                  - Extrapolate the initial guess for the stage solution from stage values of the previous time step

  Level: beginner

  Notes:
  The default is `TSARKIMEX3`, it can be changed with `TSARKIMEXSetType()` or `-ts_arkimex_type`

  If the equation is implicit or a DAE, then `TSSetEquationType()` needs to be set accordingly. Refer to the manual for further information.

  Methods with an explicit stage can only be used with ODE in which the stiff part $ G(t,X,\dot{X}) $ has the form $ \dot{X} + \hat{G}(t,X)$.

  Consider trying `TSROSW` if the stiff part is linear or weakly nonlinear.

.seealso: [](ch_ts), `TSCreate()`, `TS`, `TSSetType()`, `TSARKIMEXSetType()`, `TSARKIMEXGetType()`, `TSARKIMEXSetFullyImplicit()`, `TSARKIMEXGetFullyImplicit()`,
          `TSARKIMEX1BEE`, `TSARKIMEX2C`, `TSARKIMEX2D`, `TSARKIMEX2E`, `TSARKIMEX3`, `TSARKIMEXL2`, `TSARKIMEXA2`, `TSARKIMEXARS122`,
          `TSARKIMEX4`, `TSARKIMEX5`, `TSARKIMEXPRSSP2`, `TSARKIMEXARS443`, `TSARKIMEXBPR3`, `TSARKIMEXType`, `TSARKIMEXRegister()`, `TSType`
M*/
PETSC_EXTERN PetscErrorCode TSCreate_ARKIMEX(TS ts)
{
  TS_ARKIMEX *ark;
  PetscBool   dirk;

  PetscFunctionBegin;
  PetscCall(TSARKIMEXInitializePackage());
  PetscCall(PetscObjectTypeCompare((PetscObject)ts, TSDIRK, &dirk));

  ts->ops->reset          = TSReset_ARKIMEX;
  ts->ops->adjointreset   = TSAdjointReset_ARKIMEX;
  ts->ops->destroy        = TSDestroy_ARKIMEX;
  ts->ops->view           = TSView_ARKIMEX;
  ts->ops->load           = TSLoad_ARKIMEX;
  ts->ops->setup          = TSSetUp_ARKIMEX;
  ts->ops->adjointsetup   = TSAdjointSetUp_ARKIMEX;
  ts->ops->step           = TSStep_ARKIMEX;
  ts->ops->interpolate    = TSInterpolate_ARKIMEX;
  ts->ops->evaluatestep   = TSEvaluateStep_ARKIMEX;
  ts->ops->setfromoptions = TSSetFromOptions_ARKIMEX;
  ts->ops->snesfunction   = SNESTSFormFunction_ARKIMEX;
  ts->ops->snesjacobian   = SNESTSFormJacobian_ARKIMEX;
  ts->ops->getstages      = TSGetStages_ARKIMEX;
  ts->ops->adjointstep    = TSAdjointStep_ARKIMEX;

  ts->usessnes = PETSC_TRUE;

  PetscCall(PetscNew(&ark));
  ts->data  = (void *)ark;
  ark->imex = dirk ? PETSC_FALSE : PETSC_TRUE;

  ark->VecsDeltaLam   = NULL;
  ark->VecsSensiTemp  = NULL;
  ark->VecsSensiPTemp = NULL;

  PetscCall(PetscObjectComposeFunction((PetscObject)ts, "TSARKIMEXGetType_C", TSARKIMEXGetType_ARKIMEX));
  PetscCall(PetscObjectComposeFunction((PetscObject)ts, "TSARKIMEXGetFullyImplicit_C", TSARKIMEXGetFullyImplicit_ARKIMEX));
  if (!dirk) {
    PetscCall(PetscObjectComposeFunction((PetscObject)ts, "TSARKIMEXSetType_C", TSARKIMEXSetType_ARKIMEX));
    PetscCall(PetscObjectComposeFunction((PetscObject)ts, "TSARKIMEXSetFullyImplicit_C", TSARKIMEXSetFullyImplicit_ARKIMEX));
    PetscCall(PetscObjectComposeFunction((PetscObject)ts, "TSARKIMEXSetFastSlowSplit_C", TSARKIMEXSetFastSlowSplit_ARKIMEX));
    PetscCall(PetscObjectComposeFunction((PetscObject)ts, "TSARKIMEXGetFastSlowSplit_C", TSARKIMEXGetFastSlowSplit_ARKIMEX));
    PetscCall(TSARKIMEXSetType(ts, TSARKIMEXDefault));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TSDIRKSetType_DIRK(TS ts, TSDIRKType dirktype)
{
  TS_ARKIMEX *ark = (TS_ARKIMEX *)ts->data;

  PetscFunctionBegin;
  PetscCall(TSARKIMEXSetType_ARKIMEX(ts, dirktype));
  PetscCheck(!ark->tableau->additive, PetscObjectComm((PetscObject)ts), PETSC_ERR_ARG_WRONG, "Method \"%s\" is not DIRK", dirktype);
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  TSDIRKSetType - Set the type of `TSDIRK` scheme

  Logically Collective

  Input Parameters:
+ ts       - timestepping context
- dirktype - type of `TSDIRK` scheme

  Options Database Key:
. -ts_dirkimex_type - set `TSDIRK` scheme type

  Level: intermediate

.seealso: [](ch_ts), `TSDIRKGetType()`, `TSDIRK`, `TSDIRKType`
@*/
PetscErrorCode TSDIRKSetType(TS ts, TSDIRKType dirktype)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts, TS_CLASSID, 1);
  PetscAssertPointer(dirktype, 2);
  PetscTryMethod(ts, "TSDIRKSetType_C", (TS, TSDIRKType), (ts, dirktype));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  TSDIRKGetType - Get the type of `TSDIRK` scheme

  Logically Collective

  Input Parameter:
. ts - timestepping context

  Output Parameter:
. dirktype - type of `TSDIRK` scheme

  Level: intermediate

.seealso: [](ch_ts), `TSDIRKSetType()`
@*/
PetscErrorCode TSDIRKGetType(TS ts, TSDIRKType *dirktype)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts, TS_CLASSID, 1);
  PetscUseMethod(ts, "TSDIRKGetType_C", (TS, TSDIRKType *), (ts, dirktype));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*MC
  TSDIRK - ODE and DAE solver using Diagonally implicit Runge-Kutta schemes.

  Level: beginner

  Notes:
  The default is `TSDIRKES213SAL`, it can be changed with `TSDIRKSetType()` or `-ts_dirk_type`.
  The convention used in PETSc to name the DIRK methods is TSDIRK[E][S]PQS[SA][L][A] with:
+ E - whether the method has an explicit first stage
. S - whether the method is single diagonal
. P - order of the advancing method
. Q - order of the embedded method
. S - number of stages
. SA - whether the method is stiffly accurate
. L - whether the method is L-stable
- A - whether the method is A-stable

.seealso: [](ch_ts), `TSCreate()`, `TS`, `TSSetType()`, `TSDIRKSetType()`, `TSDIRKGetType()`, `TSDIRKRegister()`.
M*/
PETSC_EXTERN PetscErrorCode TSCreate_DIRK(TS ts)
{
  PetscFunctionBegin;
  PetscCall(TSCreate_ARKIMEX(ts));
  PetscCall(PetscObjectComposeFunction((PetscObject)ts, "TSDIRKGetType_C", TSARKIMEXGetType_ARKIMEX));
  PetscCall(PetscObjectComposeFunction((PetscObject)ts, "TSDIRKSetType_C", TSDIRKSetType_DIRK));
  PetscCall(TSDIRKSetType(ts, TSDIRKDefault));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  TSARKIMEXSetFastSlowSplit - Use `TSARKIMEX` for solving a fast-slow system

  Logically Collective

  Input Parameters:
+ ts       - timestepping context
- fastslow - `PETSC_TRUE` enables the `TSARKIMEX` solver for a fast-slow system where the RHS is split component-wise.

  Options Database Key:
. -ts_arkimex_fastslowsplit <true,false> - enables the `TSARKIMEX` solver for a fast-slow system where the RHS is split component-wise

  Level: intermediate

.seealso: [](ch_ts), `TSARKIMEX`, `TSARKIMEXGetFastSlowSplit()`, `TSRHSSplitSetIS()`
@*/
PetscErrorCode TSARKIMEXSetFastSlowSplit(TS ts, PetscBool fastslow)
{
  PetscFunctionBegin;
  PetscTryMethod(ts, "TSARKIMEXSetFastSlowSplit_C", (TS, PetscBool), (ts, fastslow));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  TSARKIMEXGetFastSlowSplit - Gets whether to use `TSARKIMEX` for a fast-slow system

  Not Collective

  Input Parameter:
. ts - timestepping context

  Output Parameter:
. fastslow - `PETSC_TRUE` if `TSARKIMEX` will be used for solving a fast-slow system, `PETSC_FALSE` otherwise

  Level: intermediate

.seealso: [](ch_ts), `TSARKIMEX`, `TSARKIMEXSetFastSlowSplit()`
@*/
PetscErrorCode TSARKIMEXGetFastSlowSplit(TS ts, PetscBool *fastslow)
{
  PetscFunctionBegin;
  PetscUseMethod(ts, "TSARKIMEXGetFastSlowSplit_C", (TS, PetscBool *), (ts, fastslow));
  PetscFunctionReturn(PETSC_SUCCESS);
}
