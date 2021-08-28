#define LANDAU_INVSQRT(q) (1./PetscSqrtReal(q))

#if defined(__CUDA_ARCH__)
#define PETSC_DEVICE_FUNC_DECL __device__
#elif defined(KOKKOS_INLINE_FUNCTION)
#define PETSC_DEVICE_FUNC_DECL KOKKOS_INLINE_FUNCTION
#else
#define PETSC_DEVICE_FUNC_DECL static
#endif

#if LANDAU_DIM==2
/* elliptic functions
 */
PETSC_DEVICE_FUNC_DECL PetscReal polevl_10(PetscReal x, const PetscReal coef[])
{
  PetscReal ans;
  PetscInt  i;
  ans = coef[0];
  for (i=1; i<11; i++) ans = ans * x + coef[i];
  return(ans);
}
PETSC_DEVICE_FUNC_DECL PetscReal polevl_9(PetscReal x, const PetscReal coef[])
{
  PetscReal ans;
  PetscInt  i;
  ans = coef[0];
  for (i=1; i<10; i++) ans = ans * x + coef[i];
  return(ans);
}
/*
 *      Complete elliptic integral of the second kind
 */
PETSC_DEVICE_FUNC_DECL void ellipticE(PetscReal x,PetscReal *ret)
{
#if defined(PETSC_USE_REAL_SINGLE)
  static const PetscReal P2[] = {
    1.53552577301013293365E-4F,
    2.50888492163602060990E-3F,
    8.68786816565889628429E-3F,
    1.07350949056076193403E-2F,
    7.77395492516787092951E-3F,
    7.58395289413514708519E-3F,
    1.15688436810574127319E-2F,
    2.18317996015557253103E-2F,
    5.68051945617860553470E-2F,
    4.43147180560990850618E-1F,
    1.00000000000000000299E0F
  };
  static const PetscReal Q2[] = {
    3.27954898576485872656E-5F,
    1.00962792679356715133E-3F,
    6.50609489976927491433E-3F,
    1.68862163993311317300E-2F,
    2.61769742454493659583E-2F,
    3.34833904888224918614E-2F,
    4.27180926518931511717E-2F,
    5.85936634471101055642E-2F,
    9.37499997197644278445E-2F,
    2.49999999999888314361E-1F
  };
#else
  static const PetscReal P2[] = {
    1.53552577301013293365E-4,
    2.50888492163602060990E-3,
    8.68786816565889628429E-3,
    1.07350949056076193403E-2,
    7.77395492516787092951E-3,
    7.58395289413514708519E-3,
    1.15688436810574127319E-2,
    2.18317996015557253103E-2,
    5.68051945617860553470E-2,
    4.43147180560990850618E-1,
    1.00000000000000000299E0
  };
  static const PetscReal Q2[] = {
    3.27954898576485872656E-5,
    1.00962792679356715133E-3,
    6.50609489976927491433E-3,
    1.68862163993311317300E-2,
    2.61769742454493659583E-2,
    3.34833904888224918614E-2,
    4.27180926518931511717E-2,
    5.85936634471101055642E-2,
    9.37499997197644278445E-2,
    2.49999999999888314361E-1
  };
#endif
  x = 1 - x; /* where m = 1 - m1 */
  *ret = polevl_10(x,P2) - PetscLogReal(x) * (x * polevl_9(x,Q2));
}
/*
 *      Complete elliptic integral of the first kind
 */
PETSC_DEVICE_FUNC_DECL void ellipticK(PetscReal x,PetscReal *ret)
{
#if defined(PETSC_USE_REAL_SINGLE)
  static const PetscReal P1[] =
    {
      1.37982864606273237150E-4F,
      2.28025724005875567385E-3F,
      7.97404013220415179367E-3F,
      9.85821379021226008714E-3F,
      6.87489687449949877925E-3F,
      6.18901033637687613229E-3F,
      8.79078273952743772254E-3F,
      1.49380448916805252718E-2F,
      3.08851465246711995998E-2F,
      9.65735902811690126535E-2F,
      1.38629436111989062502E0F
    };
  static const PetscReal Q1[] =
    {
      2.94078955048598507511E-5F,
      9.14184723865917226571E-4F,
      5.94058303753167793257E-3F,
      1.54850516649762399335E-2F,
      2.39089602715924892727E-2F,
      3.01204715227604046988E-2F,
      3.73774314173823228969E-2F,
      4.88280347570998239232E-2F,
      7.03124996963957469739E-2F,
      1.24999999999870820058E-1F,
      4.99999999999999999821E-1F
    };
#else
  static const PetscReal P1[] =
    {
      1.37982864606273237150E-4,
      2.28025724005875567385E-3,
      7.97404013220415179367E-3,
      9.85821379021226008714E-3,
      6.87489687449949877925E-3,
      6.18901033637687613229E-3,
      8.79078273952743772254E-3,
      1.49380448916805252718E-2,
      3.08851465246711995998E-2,
      9.65735902811690126535E-2,
      1.38629436111989062502E0
    };
  static const PetscReal Q1[] =
    {
      2.94078955048598507511E-5,
      9.14184723865917226571E-4,
      5.94058303753167793257E-3,
      1.54850516649762399335E-2,
      2.39089602715924892727E-2,
      3.01204715227604046988E-2,
      3.73774314173823228969E-2,
      4.88280347570998239232E-2,
      7.03124996963957469739E-2,
      1.24999999999870820058E-1,
      4.99999999999999999821E-1
    };
#endif
  x = 1 - x; /* where m = 1 - m1 */
  *ret = polevl_10(x,P1) - PetscLogReal(x) * polevl_10(x,Q1);
}
/* flip sign. papers use du/dt = C, PETSc uses form G(u) = du/dt - C(u) = 0 */
PETSC_DEVICE_FUNC_DECL void LandauTensor2D(const PetscReal x[], const PetscReal rp, const PetscReal zp, PetscReal Ud[][2], PetscReal Uk[][2], const PetscReal mask)
{
  PetscReal l,s,r=x[0],z=x[1],i1func,i2func,i3func,ks,es,pi4pow,sqrt_1s,r2,rp2,r2prp2,zmzp,zmzp2,tt;
  //PetscReal mask /* = !!(r!=rp || z!=zp) */;
  /* !!(zmzp2 > 1.e-12 || (r-rp) >  1.e-12 || (r-rp) < -1.e-12); */
  r2=PetscSqr(r);
  zmzp=z-zp;
  rp2=PetscSqr(rp);
  zmzp2=PetscSqr(zmzp);
  r2prp2=r2+rp2;
  l = r2 + rp2 + zmzp2;
  /* if      (zmzp2 >  PETSC_SMALL) mask = 1; */
  /* else if ((tt=(r-rp)) >  PETSC_SMALL) mask = 1; */
  /* else if  (tt         < -PETSC_SMALL) mask = 1; */
  /* else mask = 0; */
  s = mask*2*r*rp/l; /* mask for vectorization */
  tt = 1./(1+s);
  pi4pow = 4*PETSC_PI*LANDAU_INVSQRT(PetscSqr(l)*l);
  sqrt_1s = PetscSqrtReal(1.+s);
  /* sp.ellipe(2.*s/(1.+s)) */
  ellipticE(2*s*tt,&es); /* 44 flops * 2 + 75 = 163 flops including 2 logs, 1 sqrt, 1 pow, 21 mult */
  /* sp.ellipk(2.*s/(1.+s)) */
  ellipticK(2*s*tt,&ks); /* 44 flops + 75 in rest, 21 mult */
  /* mask is needed here just for single precision */
  i2func = 2./((1-s)*sqrt_1s) * es;
  i1func = 4./(PetscSqr(s)*sqrt_1s + PETSC_MACHINE_EPSILON) * mask * (ks - (1.+s) * es);
  i3func = 2./((1-s)*(s)*sqrt_1s + PETSC_MACHINE_EPSILON) * (es - (1-s) * ks);
  Ud[0][0]=                   -pi4pow*(rp2*i1func+PetscSqr(zmzp)*i2func);
  Ud[0][1]=Ud[1][0]=Uk[0][1]=  pi4pow*(zmzp)*(r*i2func-rp*i3func);
  Uk[1][1]=Ud[1][1]=          -pi4pow*((r2prp2)*i2func-2*r*rp*i3func)*mask;
  Uk[0][0]=                   -pi4pow*(zmzp2*i3func+r*rp*i1func);
  Uk[1][0]=                    pi4pow*(zmzp)*(r*i3func-rp*i2func); /* 48 mults + 21 + 21 = 90 mults and divs */
}
#else
/* integration point functions */
/* Evaluates the tensor U=(I-(x-y)(x-y)/(x-y)^2)/|x-y| at point x,y */
/* if x==y we will return zero. This is not the correct result */
/* since the tensor diverges for x==y but when integrated */
/* the divergent part is antisymmetric and vanishes. This is not  */
/* trivial, but can be proven. */
PETSC_DEVICE_FUNC_DECL void LandauTensor3D(const PetscReal x1[], const PetscReal xp, const PetscReal yp, const PetscReal zp, PetscReal U[][3], PetscReal mask)
{
  PetscReal dx[3],inorm3,inorm,inorm2,norm2,x2[] = {xp,yp,zp};
  PetscInt  d;
  for (d = 0, norm2 = PETSC_MACHINE_EPSILON; d < 3; ++d) {
    dx[d] = x2[d] - x1[d];
    norm2 += dx[d] * dx[d];
  }
  inorm2 = mask/norm2;
  inorm = PetscSqrtReal(inorm2);
  inorm3 = inorm2*inorm;
  for (d = 0; d < 3; ++d) U[d][d] = -(inorm - inorm3 * dx[d] * dx[d]);
  U[1][0] = U[0][1] = inorm3 * dx[0] * dx[1];
  U[1][2] = U[2][1] = inorm3 * dx[2] * dx[1];
  U[2][0] = U[0][2] = inorm3 * dx[0] * dx[2];
}
/* Relativistic form */
#define GAMMA3(_x,_c02) PetscSqrtReal(1.0 + ((_x[0]*_x[0]) + (_x[1]*_x[1]) + (_x[2]*_x[2]))/(_c02))
PETSC_DEVICE_FUNC_DECL void LandauTensor3DRelativistic(const PetscReal a_x1[], const PetscReal xp, const PetscReal yp, const PetscReal zp, PetscReal U[][3], PetscReal mask, PetscReal c0)
{
  const PetscReal x2[3] = {xp,yp,zp}, x1[3] = {a_x1[0],a_x1[1],a_x1[2]}, c02 = c0*c0, g1 = GAMMA3(x1,c02), g2 = GAMMA3(x2,c02), g1_eps = g1 - 1., g2_eps = g2 - 1., gg_eps = g1_eps + g2_eps + g1_eps*g2_eps;
  PetscReal       fact, u1u2, diff[3], udiff2,u12,u22,wsq,rsq, tt;
  PetscInt        i,j;

  if (mask==0.0) {
    for (i = 0; i < 3; ++i) {
      for (j = 0; j < 3; ++j) {
        U[i][j] = 0;
      }
    }
  } else {
    for (i = 0, u1u2 = u12 = u22 = udiff2 = 0; i < 3; ++i) {
      diff[i] = x1[i] - x2[i];
      udiff2 += diff[i] * diff[i];
      u12 += x1[i]*x1[i];
      u22 += x2[i]*x2[i];
      u1u2 += x1[i]*x2[i];
    }
    tt = 2.*u1u2*(1.-g1*g2) + (u12*u22 + u1u2*u1u2)/c02; // these two terms are about the same with opposite sign
    wsq = udiff2 + tt;
    //wsq = udiff2 + 2.*u1u2*(1.-g1*g2) + (u12*u22 + u1u2*u1u2)/c02;
    rsq = 1.+wsq/c02;
    fact = -rsq/(g1*g2*PetscSqrtReal(wsq)); /* flip sign. papers use du/dt = C, PETSc uses form G(u) = du/dt - C(u) = 0 */
    for (i = 0; i < 3; ++i) {
      for (j = 0; j < 3; ++j) {
        U[i][j] = fact * ( -diff[i]*diff[j]/wsq + (PetscSqrtReal(rsq)-1.)*(x1[i]*x2[j] + x1[j]*x2[i])/wsq);
      }
      U[i][i] += fact;
    }
#if defined(PETSC_USE_DEBUG)
    {
      PetscReal diff_g[3], udiff = sqrt(udiff2), err, err2;
      for (i = 0; i < 3; ++i) diff_g[i] = x1[i]/g1 - x2[i]/g2;
      for (i = 0, err = 0; i < 3; ++i) {
        double tmp=0;
        for (j = 0; j < 3; ++j) {
          tmp += U[i][j]*diff_g[j];
        }
        err += tmp * tmp;
      }
      err = sqrt(err);
      err2 = udiff2*(err)/(g1*g2);
#if defined(PETSC_USE_REAL_SINGLE)
      if (err>1.e-6 || err!=err) exit(11);
#else
      if (err>1.e-13 || err!=err) exit(12);
#endif
    }
#endif
  }
}

#endif
