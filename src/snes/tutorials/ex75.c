static char help[] = "Variable-Viscosity Stokes Problem in 2d.\n\
Exact solutions provided by Mirko Velic.\n\n\n";

#include<petsc.h>

#include "ex75.h"

typedef struct {
  PetscBool fem; /* Flag for FEM tests */
} AppCtx;

PetscErrorCode ProcessOptions(MPI_Comm comm, AppCtx *options)
{
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  options->fem = PETSC_FALSE;

  ierr = PetscOptionsBegin(comm, "", "Stokes Problem Options", "DMPLEX");PetscCall(ierr);
  PetscCall(PetscOptionsBool("-fem", "Run FEM tests", "ex75.c", options->fem, &options->fem, NULL));
  ierr = PetscOptionsEnd();PetscCall(ierr);
  PetscFunctionReturn(0);
}

/*
  SolKxSolution - Exact Stokes solutions for exponentially varying viscosity

 Input Parameters:
+ x  - The x coordinate at which to evaluate the solution
. z  - The z coordinate at which to evaluate the solution
. kn - The constant defining the x-dependence of the forcing function
. km - The constant defining the z-dependence of the forcing function
- B  - The viscosity coefficient

  Output Parameters:
+ vx - The x-velocity at (x,z)
. vz - The z-velocity at (x,z)
. p - The pressure at (x,z)
. sxx - The stress sigma_xx at (x,z)
. sxz - The stress sigma_xz at (x,z)
- szz - The stress sigma_zz at (x,z)

  Note:
$  The domain is the square 0 <= x,z <= 1. We solve the Stokes equation for incompressible flow with free-slip boundary
$  conditions everywhere. The forcing term f is given by
$
$    fx = 0
$    fz = sigma*sin(km*z)*cos(kn*x)
$
$  where
$
$    km = m*Pi (m may be non-integral)
$    kn = n*Pi
$
$  meaning that the density rho is -sigma*sin(km*z)*cos(kn*x). The viscosity eta is exp(2*B*x).
*/
PetscErrorCode SolKxSolution(PetscReal x, PetscReal z, PetscReal kn, PetscReal km, PetscReal B, PetscScalar *vx, PetscScalar *vz, PetscScalar *p, PetscScalar *sxx, PetscScalar *sxz, PetscScalar *szz)
{
  PetscScalar sigma;
  PetscScalar _C1,_C2,_C3,_C4;
  PetscScalar Rp, UU, VV;
  PetscScalar a,b,r,_aa,_bb,AA,BB,Rm;
  PetscScalar num1,num2,num3,num4,den1;

  PetscScalar t1,t2,t3,t4,t5,t6,t7,t8,t9,t10;
  PetscScalar t11,t12,t13,t14,t15,t16,t17,t18,t19,t20,t21;
  PetscScalar t22,t23,t24,t25,t26,t28,t29,t30,t31,t32;
  PetscScalar t33,t34,t35,t36,t37,t38,t39,t40,t41,t42;
  PetscScalar t44,t45,t46,t47,t48,t49,t51,t52,t53,t54;
  PetscScalar t56,t58,t61,t62,t63,t64,t65,t66,t67,t68;
  PetscScalar t69,t70,t71,t72,t73,t74,t75,t76,t77,t78;
  PetscScalar t79,t80,t81,t82,t83,t84,t85,t86,t87,t88;
  PetscScalar t89,t90,t91,t92,t93,t94,t95,t96,t97,t98;
  PetscScalar t99,t100,t101,t103,t104,t105,t106,t107,t108,t109;
  PetscScalar t110,t111,t112,t113,t114,t115,t116,t117,t118,t119;
  PetscScalar t120,t121,t123,t125,t127,t128,t130,t131,t132,t133;
  PetscScalar t135,t136,t138,t140,t141,t142,t143,t152,t160,t162;

  PetscFunctionBegin;
  /*************************************************************************/
  /*************************************************************************/
  /* rho = -sin(km*z)*cos(kn*x) */
  /* viscosity  Z= exp(2*B*z)  */
  /* solution valid for km not zero -- should get trivial solution if km=0 */
  sigma = 1.0;
  /*************************************************************************/
  /*************************************************************************/
  a = B*B + km*km;
  b = 2.0*km*B;
  r = sqrt(a*a + b*b);
  Rp = sqrt( (r+a)/2.0);
  Rm  = sqrt( (r-a)/2.0);
  UU  = Rp - B;
  VV = Rp + B;

  /*******************************************/
  /*         calculate the constants         */
  /*******************************************/
  t1 = kn * kn;
  t4 = km * km;
  t6 = t4 * t4;
  t7 = B * B;
  t9 = 0.4e1 * t7 * t4;
  t12 = 0.8e1 * t7 * kn * km;
  t14 = 0.4e1 * t7 * t1;
  t16 = 0.2e1 * t4 * t1;
  t17 = t1 * t1;
  _aa = -0.4e1 * B * t1 * sigma * (t4 + t1) / (t6 + t9 + t12 + t14 + t16 + t17) / (t6 + t9 - t12 + t14 + t16 + t17);

  t2 = kn * kn;
  t3 = t2 * t2;
  t4 = B * B;
  t6 = 0.4e1 * t4 * t2;
  t7 = km * km;
  t9 = 0.4e1 * t7 * t4;
  t10 = t7 * t7;
  t12 = 0.2e1 * t7 * t2;
  t16 = 0.8e1 * t4 * kn * km;
  _bb = sigma * kn * (t3 - t6 + t9 + t10 + t12) / (t10 + t9 + t16 + t6 + t12 + t3) / (t10 + t9 - t16 + t6 + t12 + t3);

  AA = _aa;
  BB = _bb;

  t1 = Rm * Rm;
  t2 = B - Rp;
  t4 = Rp + B;
  t6 = UU * x;
  t9 = exp(t6 - 0.4e1 * Rp);
  t13 = kn * kn;
  t15 = B * B;
  t18 = Rp * Rp;
  t19 = t18 * B;
  t20 = t15 * Rp;
  t22 = t1 * Rp;
  t24 = B * t1;
  t32 = 0.8e1 * t15 * BB * kn * Rp;
  t34 = 0.2e1 * Rm;
  t35 = cos(t34);
  t37 = Rm * Rp;
  t49 = sin(t34);
  t63 = exp(t6 - 0.2e1 * Rp);
  t65 = Rm * t2;
  t67 = 0.2e1 * B * kn;
  t68 = B * Rm;
  t69 = t67 + t68 + t37;
  t73 = 0.3e1 * t15;
  t75 = 0.2e1 * B * Rp;
  t76 = t73 - t75 + t1 - t13 - t18;
  t78 = t65 * t76 * BB;
  t80 = Rm - kn;
  t81 = cos(t80);
  t83 = t68 - t67 + t37;
  t88 = Rm + kn;
  t89 = cos(t88);
  t92 = t65 * t76 * AA;
  t97 = sin(t80);
  t103 = sin(t88);
  t108 = exp(t6 - 0.3e1 * Rp - B);
  t110 = Rm * t4;
  t111 = t67 + t68 - t37;
  t115 = t73 + t75 + t1 - t13 - t18;
  t117 = t110 * t115 * BB;
  t120 = -t67 + t68 - t37;
  t127 = t110 * t115 * AA;
  t140 = exp(t6 - Rp - B);
  num1 = -0.4e1 * t1 * t2 * t4 * AA * t9 + ((0.2e1 * Rp * (-B * t13 + 0.3e1 * t15 * B - t19 - 0.2e1 * t20 - 0.2e1 * t22 - t24) * AA - t32) * t35 + (0.2e1 * t37 * (t1 - t13 + 0.5e1 * t15 - t18) * AA - 0.8e1 * B * BB * kn * Rm * Rp) * t49 - 0.2e1 * B * (0.3e1 * t20 - t18 * Rp - 0.2e1 * t19 - Rp * t13 - t22 - 0.2e1 * t24) * AA + t32) * t63 + ((0.2e1 * t65 * t69 * AA + t78) * t81 + (0.2e1 * t65 * t83 * AA - t78) * t89 + (t92 - 0.2e1 * t65 * t69 * BB) * t97 + (t92 + 0.2e1 * t65 * t83 * BB) * t103) * t108 + ((-0.2e1 * t110 * t111 * AA - t117) * t81 + (-0.2e1 * t110 * t120 * AA + t117) * t89 + (-t127 + 0.2e1 * t110 * t111 * BB) * t97 + (-t127 - 0.2e1 * t110 * t120 * BB) * t103) * t140;

  t1 = Rp + B;
  t2 = Rm * t1;
  t3 = B * B;
  t4 = 0.3e1 * t3;
  t5 = B * Rp;
  t7 = Rm * Rm;
  t8 = kn * kn;
  t9 = Rp * Rp;
  t10 = t4 + 0.2e1 * t5 + t7 - t8 - t9;
  t12 = t2 * t10 * AA;
  t14 = B * Rm;
  t20 = UU * x;
  t23 = exp(t20 - 0.4e1 * Rp);
  t25 = Rm * Rp;
  t32 = Rm * kn;
  t37 = 0.2e1 * Rm;
  t38 = cos(t37);
  t41 = t3 * B;
  t44 = t3 * Rp;
  t48 = B * t7;
  t53 = t3 * BB;
  t54 = kn * Rp;
  t58 = sin(t37);
  t69 = exp(t20 - 0.2e1 * Rp);
  t71 = t9 * Rp;
  t72 = Rm * t71;
  t73 = t3 * Rm;
  t75 = 0.5e1 * t73 * Rp;
  t77 = 0.8e1 * t44 * kn;
  t78 = t25 * t8;
  t79 = t7 * Rm;
  t80 = B * t79;
  t81 = t14 * t8;
  t82 = t79 * Rp;
  t84 = 0.3e1 * t41 * Rm;
  t85 = t14 * t9;
  t86 = -t72 + t75 + t77 - t78 + t80 - t81 + t82 + t84 + t85;
  t88 = t7 * t9;
  t89 = t5 * t8;
  t90 = t7 * t3;
  t91 = B * t71;
  t92 = t48 * Rp;
  t94 = 0.2e1 * t14 * t54;
  t96 = 0.3e1 * Rp * t41;
  t98 = 0.2e1 * t73 * kn;
  t100 = 0.2e1 * t9 * t3;
  t101 = -t88 - t89 - t90 - t91 - t92 - t94 + t96 - t98 - t100;
  t105 = Rm - kn;
  t106 = cos(t105);
  t108 = t75 - t77 - t78 + t85 - t72 - t81 + t80 + t84 + t82;
  t110 = -t100 + t96 - t91 + t94 + t98 - t92 - t89 - t88 - t90;
  t114 = Rm + kn;
  t115 = cos(t114);
  t121 = sin(t105);
  t127 = sin(t114);
  t132 = exp(t20 - 0.3e1 * Rp - B);
  t135 = 0.2e1 * B * kn;
  t136 = t135 + t14 - t25;
  t142 = -t135 + t14 - t25;
  t152 = t2 * t10 * BB;
  t162 = exp(t20 - Rp - B);
  num2 = (0.2e1 * t12 - 0.8e1 * t14 * kn * t1 * BB) * t23 + ((-0.2e1 * t25 * (t7 - t8 + 0.5e1 * t3 - t9) * AA + 0.8e1 * B * BB * t32 * Rp) * t38 + (0.2e1 * Rp * (-B * t8 + 0.3e1 * t41 - t9 * B - 0.2e1 * t44 - 0.2e1 * t7 * Rp - t48) * AA - 0.8e1 * t53 * t54) * t58 - 0.2e1 * t14 * (t4 + t9 - t8 + t7) * AA + 0.8e1 * t53 * t32) * t69 + ((-t86 * AA - 0.2e1 * t101 * BB) * t106 + (-t108 * AA + 0.2e1 * t110 * BB) * t115 + (-0.2e1 * t101 * AA + t86 * BB) * t121 + (-0.2e1 * t110 * AA - t108 * BB) * t127) * t132 + ((t12 - 0.2e1 * t2 * t136 * BB) * t106 + (t12 + 0.2e1 * t2 * t142 * BB) * t115 + (-0.2e1 * t2 * t136 * AA - t152) * t121 + (-0.2e1 * t2 * t142 * AA + t152) * t127) * t162;

  t1 = Rm * Rm;
  t2 = B - Rp;
  t4 = Rp + B;
  t6 = VV * x;
  t7 = exp(-t6);
  t11 = B * t1;
  t12 = Rp * Rp;
  t13 = t12 * B;
  t14 = B * B;
  t15 = t14 * Rp;
  t19 = kn * kn;
  t21 = t1 * Rp;
  t30 = 0.8e1 * t14 * BB * kn * Rp;
  t32 = 0.2e1 * Rm;
  t33 = cos(t32);
  t35 = Rm * Rp;
  t47 = sin(t32);
  t61 = exp(-t6 - 0.2e1 * Rp);
  t63 = Rm * t2;
  t65 = 0.2e1 * B * kn;
  t66 = B * Rm;
  t67 = t65 + t66 + t35;
  t71 = 0.3e1 * t14;
  t73 = 0.2e1 * B * Rp;
  t74 = t71 - t73 + t1 - t19 - t12;
  t76 = t63 * t74 * BB;
  t78 = Rm - kn;
  t79 = cos(t78);
  t81 = t66 - t65 + t35;
  t86 = Rm + kn;
  t87 = cos(t86);
  t90 = t63 * t74 * AA;
  t95 = sin(t78);
  t101 = sin(t86);
  t106 = exp(-t6 - 0.3e1 * Rp - B);
  t108 = Rm * t4;
  t109 = t65 + t66 - t35;
  t113 = t71 + t73 + t1 - t19 - t12;
  t115 = t108 * t113 * BB;
  t118 = -t65 + t66 - t35;
  t125 = t108 * t113 * AA;
  t138 = exp(-t6 - Rp - B);
  num3 = -0.4e1 * t1 * t2 * t4 * AA * t7 + ((-0.2e1 * Rp * (-t11 - t13 + 0.2e1 * t15 + 0.3e1 * t14 * B - B * t19 + 0.2e1 * t21) * AA + t30) * t33 + (-0.2e1 * t35 * (t1 - t19 + 0.5e1 * t14 - t12) * AA + 0.8e1 * B * BB * kn * Rm * Rp) * t47 + 0.2e1 * B * (-t12 * Rp + 0.2e1 * t11 + 0.3e1 * t15 + 0.2e1 * t13 - t21 - Rp * t19) * AA - t30) * t61 + ((-0.2e1 * t63 * t67 * AA - t76) * t79 + (-0.2e1 * t63 * t81 * AA + t76) * t87 + (-t90 + 0.2e1 * t63 * t67 * BB) * t95 + (-t90 - 0.2e1 * t63 * t81 * BB) * t101) * t106 + ((0.2e1 * t108 * t109 * AA + t115) * t79 + (0.2e1 * t108 * t118 * AA - t115) * t87 + (t125 - 0.2e1 * t108 * t109 * BB) * t95 + (t125 + 0.2e1 * t108 * t118 * BB) * t101) * t138;

  t1 = B - Rp;
  t2 = Rm * t1;
  t3 = B * B;
  t4 = 0.3e1 * t3;
  t5 = B * Rp;
  t7 = Rm * Rm;
  t8 = kn * kn;
  t9 = Rp * Rp;
  t10 = t4 - 0.2e1 * t5 + t7 - t8 - t9;
  t12 = t2 * t10 * AA;
  t14 = B * Rm;
  t20 = VV * x;
  t21 = exp(-t20);
  t23 = Rm * Rp;
  t30 = Rm * kn;
  t35 = 0.2e1 * Rm;
  t36 = cos(t35);
  t38 = B * t7;
  t40 = t3 * Rp;
  t42 = t3 * B;
  t51 = t3 * BB;
  t52 = kn * Rp;
  t56 = sin(t35);
  t67 = exp(-t20 - 0.2e1 * Rp);
  t70 = 0.2e1 * B * kn;
  t71 = t70 + t14 + t23;
  t76 = Rm - kn;
  t77 = cos(t76);
  t79 = t14 - t70 + t23;
  t84 = Rm + kn;
  t85 = cos(t84);
  t91 = t2 * t10 * BB;
  t93 = sin(t76);
  t99 = sin(t84);
  t104 = exp(-t20 - 0.3e1 * Rp - B);
  t106 = t9 * Rp;
  t107 = Rm * t106;
  t108 = t3 * Rm;
  t110 = 0.5e1 * t108 * Rp;
  t112 = 0.8e1 * t40 * kn;
  t113 = t23 * t8;
  t114 = t7 * Rm;
  t115 = B * t114;
  t116 = t14 * t8;
  t117 = t114 * Rp;
  t119 = 0.3e1 * t42 * Rm;
  t120 = t14 * t9;
  t121 = t107 - t110 - t112 + t113 + t115 - t116 - t117 + t119 + t120;
  t123 = t38 * Rp;
  t125 = 0.2e1 * t14 * t52;
  t127 = 0.3e1 * Rp * t42;
  t128 = t7 * t3;
  t130 = 0.2e1 * t9 * t3;
  t131 = t7 * t9;
  t132 = B * t106;
  t133 = t5 * t8;
  t135 = 0.2e1 * t108 * kn;
  t136 = -t123 - t125 + t127 + t128 + t130 + t131 - t132 - t133 + t135;
  t141 = -t110 + t112 + t113 + t120 + t107 - t116 + t115 + t119 - t117;
  t143 = t125 - t132 + t130 - t135 + t127 + t131 - t123 + t128 - t133;
  t160 = exp(-t20 - Rp - B);
  num4 = (0.2e1 * t12 - 0.8e1 * t14 * kn * t1 * BB) * t21 + ((0.2e1 * t23 * (t7 - t8 + 0.5e1 * t3 - t9) * AA - 0.8e1 * B * BB * t30 * Rp) * t36 + (-0.2e1 * Rp * (-t38 - t9 * B + 0.2e1 * t40 + 0.3e1 * t42 - B * t8 + 0.2e1 * t7 * Rp) * AA + 0.8e1 * t51 * t52) * t56 - 0.2e1 * t14 * (t4 + t9 - t8 + t7) * AA + 0.8e1 * t51 * t30) * t67 + ((t12 - 0.2e1 * t2 * t71 * BB) * t77 + (t12 + 0.2e1 * t2 * t79 * BB) * t85 + (-0.2e1 * t2 * t71 * AA - t91) * t93 + (-0.2e1 * t2 * t79 * AA + t91) * t99) * t104 + ((-t121 * AA + 0.2e1 * t136 * BB) * t77 + (-t141 * AA - 0.2e1 * t143 * BB) * t85 + (0.2e1 * t136 * AA + t121 * BB) * t93 + (0.2e1 * t143 * AA - t141 * BB) * t99) * t160;

  t1 = Rm * Rm;
  t2 = Rp * Rp;
  t3 = t1 * t2;
  t4 = B * B;
  t5 = t1 * t4;
  t9 = exp(-0.4e1 * Rp);
  t15 = cos(0.2e1 * Rm);
  t22 = exp(-0.2e1 * Rp);
  den1 = (-0.4e1 * t3 + 0.4e1 * t5) * t9 + ((0.8e1 * t1 + 0.8e1 * t4) * t2 * t15 - 0.8e1 * t5 - 0.8e1 * t2 * t4) * t22 - 0.4e1 * t3 + 0.4e1 * t5;

  _C1=num1/den1; _C2=num2/den1; _C3=num3/den1; _C4=num4/den1;

  /*******************************************/
  /*         calculate solution         */
  /*******************************************/
  t1 = Rm * x;
  t2 = cos(t1);
  t4 = sin(t1);
  t10 = exp(-0.2e1 * x * B);
  t12 = kn * x;
  t13 = cos(t12);
  t16 = sin(t12);
  *vx = -km * (_C1 * t2 + _C2 * t4 + _C3 * t2 + _C4 * t4 + t10 * AA * t13 + t10 * BB * t16);

  t2 = Rm * x;
  t3 = cos(t2);
  t6 = sin(t2);
  t22 = exp(-0.2e1 * x * B);
  t23 = B * t22;
  t24 = kn * x;
  t25 = cos(t24);
  t29 = sin(t24);
  *vz = UU * _C1 * t3 + UU * _C2 * t6 - _C1 * t6 * Rm + _C2 * t3 * Rm - VV * _C3 * t3 - VV * _C4 * t6 - _C3 * t6 * Rm + _C4 * t3 * Rm - 0.2e1 * t23 * AA * t25 - 0.2e1 * t23 * BB * t29 - t22 * AA * t29 * kn + t22 * BB * t25 * kn;

  t3 = exp(0.2e1 * x * B);
  t4 = t3 * B;
  t8 = km * km;
  t9 = t3 * t8;
  t11 = 0.3e1 * t9 * Rm;
  t12 = Rm * Rm;
  t14 = t3 * t12 * Rm;
  t15 = UU * UU;
  t19 = 0.4e1 * t4 * UU * Rm - t11 - t14 + 0.3e1 * t3 * t15 * Rm;
  t20 = Rm * x;
  t21 = sin(t20);
  t26 = 0.2e1 * t9 * B;
  t33 = 0.2e1 * t4 * t12;
  t36 = -t3 * t15 * UU - t26 + 0.3e1 * t9 * UU + 0.3e1 * t3 * UU * t12 + t33 - 0.2e1 * t4 * t15;
  t37 = cos(t20);
  t46 = VV * VV;
  t53 = -t11 - t14 + 0.3e1 * t3 * t46 * Rm - 0.4e1 * t4 * VV * Rm;
  t64 = -t26 + t33 + t3 * t46 * VV - 0.3e1 * t9 * VV - 0.2e1 * t4 * t46 - 0.3e1 * t3 * VV * t12;
  t73 = kn * kn;
  t74 = t73 * kn;
  t79 = B * B;
  t86 = B * t8;
  t90 = kn * x;
  t91 = sin(t90);
  t106 = cos(t90);
  *sxx = -((t19 * t21 + t36 * t37) * _C1 + (t36 * t21 - t19 * t37) * _C2 + (t53 * t21 + t64 * t37) * _C3 + (t64 * t21 - t53 * t37) * _C4 + (-AA * t74 - 0.4e1 * BB * t73 * B + 0.4e1 * t79 * AA * kn - 0.3e1 * t8 * AA * kn - 0.8e1 * t86 * BB) * t91 + (-0.8e1 * t86 * AA - 0.4e1 * AA * t73 * B - 0.4e1 * t79 * BB * kn + 0.3e1 * t8 * BB * kn + BB * t74) * t106) / km;

  t3 = exp(0.2e1 * x * B);
  t4 = km * km;
  t5 = t3 * t4;
  t6 = Rm * x;
  t7 = cos(t6);
  t8 = _C1 * t7;
  t10 = sin(t6);
  t11 = _C2 * t10;
  t13 = _C3 * t7;
  t15 = _C4 * t10;
  t18 = kn * x;
  t19 = cos(t18);
  t22 = sin(t18);
  t24 = UU * UU;
  t25 = t3 * t24;
  t28 = t3 * UU;
  t38 = Rm * Rm;
  t39 = t7 * t38;
  t42 = t10 * t38;
  t44 = t5 * t8 + t5 * t11 + t5 * t13 + t5 * t15 + t4 * AA * t19 + t4 * BB * t22 + t25 * t8 + t25 * t11 - 0.2e1 * t28 * _C1 * t10 * Rm + 0.2e1 * t28 * _C2 * t7 * Rm - t3 * _C1 * t39 - t3 * _C2 * t42;
  t45 = VV * VV;
  t46 = t3 * t45;
  t49 = t3 * VV;
  t62 = B * B;
  t78 = kn * kn;
  t82 = t46 * t13 + t46 * t15 + 0.2e1 * t49 * _C3 * t10 * Rm - 0.2e1 * t49 * _C4 * t7 * Rm - t3 * _C3 * t39 - t3 * _C4 * t42 + 0.4e1 * t62 * AA * t19 + 0.4e1 * t62 * BB * t22 + 0.4e1 * B * AA * t22 * kn - 0.4e1 * B * BB * t19 * kn - AA * t19 * t78 - BB * t22 * t78;
  *sxz = t44 + t82;

  t3 = exp(0.2e1 * x * B);
  t4 = t3 * B;
  t8 = km * km;
  t9 = t3 * t8;
  t10 = t9 * Rm;
  t11 = Rm * Rm;
  t13 = t3 * t11 * Rm;
  t14 = UU * UU;
  t18 = 0.4e1 * t4 * UU * Rm - t10 - t13 + 0.3e1 * t3 * t14 * Rm;
  t19 = Rm * x;
  t20 = sin(t19);
  t25 = 0.2e1 * t9 * B;
  t31 = 0.2e1 * t4 * t11;
  t34 = -t3 * t14 * UU - t25 + t9 * UU + 0.3e1 * t3 * UU * t11 + t31 - 0.2e1 * t4 * t14;
  t35 = cos(t19);
  t44 = VV * VV;
  t51 = -t10 - t13 + 0.3e1 * t3 * t44 * Rm - 0.4e1 * t4 * VV * Rm;
  t61 = -t25 + t31 + t3 * t44 * VV - t9 * VV - 0.2e1 * t4 * t44 - 0.3e1 * t3 * VV * t11;
  t70 = kn * kn;
  t71 = t70 * kn;
  t76 = B * B;
  t82 = B * t8;
  t86 = kn * x;
  t87 = sin(t86);
  t101 = cos(t86);
  *p = ((t18 * t20 + t34 * t35) * _C1 + (t34 * t20 - t18 * t35) * _C2 + (t51 * t20 + t61 * t35) * _C3 + (t61 * t20 - t51 * t35) * _C4 + (-AA * t71 - 0.4e1 * BB * t70 * B + 0.4e1 * t76 * AA * kn - t8 * AA * kn - 0.4e1 * t82 * BB) * t87 + (-0.4e1 * t82 * AA - 0.4e1 * AA * t70 * B - 0.4e1 * t76 * BB * kn + t8 * BB * kn + BB * t71) * t101) / km;

  t3 = exp(0.2e1 * x * B);
  t4 = UU * UU;
  t8 = km * km;
  t9 = t3 * t8;
  t10 = t9 * Rm;
  t11 = Rm * Rm;
  t13 = t3 * t11 * Rm;
  t14 = t3 * B;
  t18 = 0.3e1 * t3 * t4 * Rm + t10 - t13 + 0.4e1 * t14 * UU * Rm;
  t19 = Rm * x;
  t20 = sin(t19);
  t23 = 0.2e1 * t9 * B;
  t33 = 0.2e1 * t14 * t11;
  t34 = -t23 + 0.3e1 * t3 * UU * t11 - t9 * UU - t3 * t4 * UU - 0.2e1 * t4 * t14 + t33;
  t35 = cos(t19);
  t47 = VV * VV;
  t51 = t10 - 0.4e1 * t14 * VV * Rm + 0.3e1 * t3 * t47 * Rm - t13;
  t61 = t9 * VV - t23 + t3 * t47 * VV - 0.2e1 * t14 * t47 + t33 - 0.3e1 * t3 * VV * t11;
  t70 = B * B;
  t74 = kn * kn;
  t75 = t74 * kn;
  t83 = kn * x;
  t84 = sin(t83);
  t96 = cos(t83);
  *szz = -((t18 * t20 + t34 * t35) * _C1 + (t34 * t20 - t18 * t35) * _C2 + (t51 * t20 + t61 * t35) * _C3 + (t61 * t20 - t51 * t35) * _C4 + (0.4e1 * t70 * AA * kn - AA * t75 - 0.4e1 * BB * t74 * B + t8 * AA * kn) * t84 + (-t8 * BB * kn - 0.4e1 * AA * t74 * B - 0.4e1 * t70 * BB * kn + BB * t75) * t96) / km;

  /* vx = Vx, vz = Vz, sxx = xx-component of stress tensor, sxz = xz-component of stress tensor, p = pressure, szz = zz-component of stress tensor */
  *vx  *= cos(km*z); /* Vx */
  *vz  *= sin(km*z); /* Vz */
  *p   *= cos(km*z); /* p */
  *sxx *= cos(km*z); /* sxx total stress */
  *sxz *= sin(km*z); /* tzx stress */
  *szz *= cos(km*z); /* szz total stress */

  /* rho = -sigma*sin(km*z)*cos(kn*x); */ /* density */
  PetscFunctionReturn(0);
}

PetscErrorCode SolKxWrapperV(PetscInt dim, const PetscReal x[], PetscInt Nf, PetscScalar v[], void *ctx)
{
  PetscReal   B  = 100.0;
  PetscReal   kn = 100*M_PI;
  PetscReal   km = 100*M_PI;
  PetscScalar p, sxx, sxz, szz;

  PetscFunctionBeginUser;
  SolKxSolution(x[0], x[1], kn, km, B, &v[0], &v[1], &p, &sxx, &sxz, &szz);
  PetscFunctionReturn(0);
}

PetscErrorCode SolKxWrapperP(PetscInt dim, const PetscReal x[], PetscInt Nf, PetscScalar v[], void *ctx)
{
  PetscReal   B  = 100.0;
  PetscReal   kn = 100*M_PI;
  PetscReal   km = 100*M_PI;
  PetscScalar vx, vz, sxx, sxz, szz;

  PetscFunctionBeginUser;
  SolKxSolution(x[0], x[1], kn, km, B, &vx, &vz, &v[0], &sxx, &sxz, &szz);
  PetscFunctionReturn(0);
}

/*
  Compare the C implementation with generated data from Maple
*/
PetscErrorCode MapleTest(MPI_Comm comm, AppCtx *ctx)
{
  const PetscInt n = 41;
  PetscScalar    vxMaple[41][41], vzMaple[41][41], pMaple[41][41], sxxMaple[41][41], sxzMaple[41][41], szzMaple[41][41];
  PetscReal      x[41], z[41];
  PetscReal      kn, km, B;
  PetscInt       i, j;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscCall(SolKxData5(x, z, &kn, &km, &B, vxMaple, vzMaple, pMaple, sxxMaple, sxzMaple, szzMaple));
  for (i = 0; i < n; ++i) {
    for (j = 0; j < n; ++j) {
      PetscScalar vx, vz, p, sxx, sxz, szz;
      PetscReal   norm;

      PetscCall(SolKxSolution(x[i], z[j], kn, km, B, &vx, &vz, &p, &sxx, &sxz, &szz));
      norm = sqrt(PetscSqr(PetscAbsScalar(vx - vxMaple[i][j])) + PetscSqr(PetscAbsScalar(vz - vzMaple[i][j])));
      if (norm > 1.0e-10) {
        ierr = PetscPrintf(PETSC_COMM_SELF, "%0.17e %0.17e %0.17e %0.17e %0.17e %0.17e %0.17e %0.17e %0.17e\n",
                           (double)x[i], (double)z[j], (double)PetscAbsScalar(vx - vxMaple[i][j]), (double)PetscAbsScalar(vz - vzMaple[i][j]), (double)PetscAbsScalar(p - pMaple[i][j]),
                           (double)PetscAbsScalar(sxx - sxxMaple[i][j]), (double)PetscAbsScalar(sxz - sxzMaple[i][j]), (double)PetscAbsScalar(szz - szzMaple[i][j]), (double)norm);
        SETERRQ(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Invalid solution, error %g", (double)norm);
      }
    }
  }
  PetscCall(PetscPrintf(comm, "Verified Maple test 5\n"));
  PetscFunctionReturn(0);
}

PetscErrorCode FEMTest(MPI_Comm comm, AppCtx *ctx)
{
  DM               dm;
  Vec              u;
  PetscErrorCode (*funcs[2])(PetscInt, const PetscReal [], PetscInt, PetscScalar *, void *) = {SolKxWrapperV, SolKxWrapperP};
  PetscReal        discError;
  PetscErrorCode   ierr;

  PetscFunctionBegin;
  if (!ctx->fem) PetscFunctionReturn(0);
  /* Create DM */
  PetscCall(DMPlexCreateBoxMesh(comm, 2, PETSC_TRUE, NULL, NULL, NULL, NULL, PETSC_FALSE, &dm));
  PetscCall(DMSetFromOptions(dm));
  /* Project solution into FE space */
  PetscCall(DMGetGlobalVector(dm, &u));
  PetscCall(DMProjectFunction(dm, 0.0, funcs, NULL, INSERT_VALUES, u));
  PetscCall(DMComputeL2Diff(dm, 0.0, funcs, NULL, u, &discError));
  PetscCall(VecViewFromOptions(u, NULL, "-vec_view"));
  /* Cleanup */
  PetscCall(DMRestoreGlobalVector(dm, &u));
  PetscCall(DMDestroy(&dm));
  PetscFunctionReturn(0);
}

int main(int argc, char **argv)
{
  AppCtx         user;                 /* user-defined work context */
  PetscErrorCode ierr;

  PetscCall(PetscInitialize(&argc, &argv, NULL,help));
  PetscCall(ProcessOptions(PETSC_COMM_WORLD, &user));
  PetscCall(MapleTest(PETSC_COMM_WORLD, &user));
  PetscCall(FEMTest(PETSC_COMM_WORLD, &user));
  PetscCall(PetscFinalize());
  return 0;
}
