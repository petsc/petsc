/* $Id: vmult.h,v 1.1 1994/10/02 02:07:37 bsmith Exp bsmith $ */


/*
   Routines for vector - vector products.  Includes sums with vectors
 */
#ifndef VMULTADD1
/* VMULTADD(n+1) = vo += *d0++ * *v0 + ... *dn++ * *vn; v[o0-n] += vinc */
/* These all need inline_while, inline_for versions */
#ifdef UNROLL_VMULT

#else
#define VMULTADD1(vo,d0,v0,vinc,nnr)\
while(nnr--){*vo += *d0++ * *v0; v0 += vinc; vo += vinc;}
#define VMULTADD2(vo,d0,d1,v0,v1,vinc,nnr)\
while(nnr--){*vo += *d0++ * *v0 + *d1++ * *v1; v0 += vinc; v1 += vinc; \
 vo += vinc;}
#define VMULTADD3(vo,d0,d1,d2,v0,v1,v2,vinc,nnr)\
while(nnr--){*vo += *d0++ * *v0 + *d1++ * *v1 + *d2++ * *v2;\
 v0 += vinc; v1 += vinc; v2 += vinc; vo += vinc;}
#define VMULTADD4(vo,d0,d1,d2,d3,v0,v1,v2,v3,vinc,nnr)\
while(nnr--){*vo += *d0++ * *v0 + *d1++ * *v1 + *d2++ * *v2 + *d3++ * *v3;\
 v0 += vinc; v1 += vinc; v2 += vinc; v3 += vinc; vo += vinc;}
#define VMULTADD5(vo,d0,d1,d2,d3,d4,v0,v1,v2,v3,v4,vinc,nnr)\
while(nnr--){*vo += *d0++ * *v0 + *d1++ * *v1 + *d2++ * *v2 + *d3++ * *v3 \
  + *d4++ * *v4;\
 v0 += vinc; v1 += vinc; v2 += vinc; v3 += vinc; v4 += vinc; vo += vinc;}
#endif

/* VMULTADDSET(n+1) - like VMULTADD, but vo = , not += */
#ifdef UNROLL_VMULT

#else
#define VMULTSET1(vo,d0,v0,vinc,nnr)\
while(nnr--){*vo = *d0++ * *v0; v0 += vinc; vo += vinc;}
#define VMULTADDSET2(vo,d0,d1,v0,v1,vinc,nnr)\
while(nnr--){*vo = *d0++ * *v0 + *d1++ * *v1; v0 += vinc; v1 += vinc; vo+=vinc;}
#define VMULTADDSET3(vo,d0,d1,d2,v0,v1,v2,vinc,nnr)\
while(nnr--){*vo = *d0++ * *v0 + *d1++ * *v1 + *d2++ * *v2;\
 v0 += vinc; v1 += vinc; v2 += vinc; vo += vinc;}
#define VMULTADDSET4(vo,d0,d1,d2,d3,v0,v1,v2,v3,vinc,nnr)\
while(nnr--){*vo = *d0++ * *v0 + *d1++ * *v1 + *d2++ * *v2 + *d3++ * *v3;\
 v0 += vinc; v1 += vinc; v2 += vinc; v3 += vinc; vo += vinc;}
#define VMULTADDSET5(vo,d0,d1,d2,d3,d4,v0,v1,v2,v3,v4,vinc,nnr)\
while(nnr--){*vo = *d0++ * *v0 + *d1++ * *v1 + *d2++ * *v2 + *d3++ * *v3\
  + *d4++ * *v4;\
 v0 += vinc; v1 += vinc; v2 += vinc; v3 += vinc; v4 += vinc; vo += vinc;}
#endif

/* VMULTADDINC(n+1) = vo += *d0 * *v0 + ... *dn * *vn; v[o0-n] += vinc, 
   d? += dinc */

#ifdef UNROLL_VMULT

#else
#define VMULTADDINC1(vo,d0,dinc,v0,vinc,nnr)\
while(nnr--){*vo += *d0 * *v0; v0 += vinc; vo += vinc; d0 += dinc;}
#define VMULTADDINC2(vo,d0,d1,dinc,v0,v1,vinc,nnr)\
while(nnr--){*vo += *d0 * *v0 + *d1 * *v1; v0 += vinc; v1 += vinc; vo += vinc;\
d0 += dinc; d1 += dinc;}
#define VMULTADDINC3(vo,d0,d1,d2,dinc,v0,v1,v2,vinc,nnr)\
while(nnr--){*vo += *d0 * *v0 + *d1 * *v1 + *d2 * *v2;\
 v0 += vinc; v1 += vinc; v2 += vinc; vo += vinc; d0 += dinc;d1 += dinc; \
 d2 += dinc;}
#define VMULTADDINC4(vo,d0,d1,d2,d3,dinc,v0,v1,v2,v3,vinc,nnr)\
while(nnr--){*vo += *d0 * *v0 + *d1 * *v1 + *d2 * *v2 + *d3 * *v3;\
  v0 += vinc; v1 += vinc; v2 += vinc; v3 += vinc; vo += vinc;d0 += dinc;\
  d1 += dinc; d2 += dinc;d3 += dinc;}
#define VMULTADDINC5(vo,d0,d1,d2,d3,d4,dinc,v0,v1,v2,v3,v4,vinc,nnr)\
while(nnr--){*vo += *d0 * *v0 + *d1 * *v1 + *d2 * *v2 + *d3 * *v3 + *d4 * *v4;\
 v0 += vinc; v1 += vinc; v2 += vinc; v3 += vinc; v4 += vinc; vo += vinc;\
d0 += dinc;d1 += dinc;d2 += dinc;d3 += dinc;d4 += dinc;}
#endif

/* VMULTADDINCSET(n+1) - like VMULTADD, but vo = , not += (d += dinc) */
#ifdef UNROLL_VMULT

#else
#define VMULTINCSET1(vo,d0,dinc,v0,vinc,nnr)\
while(nnr--){*vo = *d0 * *v0; v0 += vinc; vo += vinc; d0 += dinc;}
#define VMULTADDINCSET2(vo,d0,d1,dinc,v0,v1,vinc,nnr)\
while(nnr--){*vo = *d0 * *v0 + *d1 * *v1; v0 += vinc; v1 += vinc; vo += vinc;\
d0 += dinc;d1 += dinc;}
#define VMULTADDINCSET3(vo,d0,d1,d2,dinc,v0,v1,v2,vinc,nnr)\
while(nnr--){*vo= *d0 * *v0 + *d1 * *v1 + *d2 * *v2;\
 v0 += vinc; v1 += vinc; v2 += vinc; vo += vinc;d0 += dinc;d1 += dinc;\
 d2 += dinc;}
#define VMULTADDINCSET4(vo,d0,d1,d2,d3,dinc,v0,v1,v2,v3,vinc,nnr)\
while(nnr--){*vo = *d0 * *v0 + *d1 * *v1 + *d2 * *v2 + *d3 * *v3;\
 v0 += vinc; v1 += vinc; v2 += vinc; v3 += vinc; vo += vinc;\
 d0 += dinc;d1 += dinc;d2 += dinc;d3 += dinc;}
#define VMULTADDINCSET5(vo,d0,d1,d2,d3,d4,dinc,v0,v1,v2,v3,v4,vinc,nnr)\
while(nnr--){*vo= *d0 * *v0 + *d1 * *v1 + *d2 * *v2 + *d3 * *v3 + *d4 * *v4;\
 v0 += vinc; v1 += vinc; v2 += vinc; v3 += vinc; v4 += vinc; vo += vinc;\
 d0 += dinc;d1 += dinc;d2 += dinc;d3 += dinc;d4 += dinc;}
#endif

#define VSCALINC(vo,a0,v0,vinc,nnr)\
while (nnr--) {*vo = a0 * *v0;  vo += vinc; v0 += vinc;}

#define VSCAL(vo,a0,v0,nnr)while(nnr--)*vo++ = a0 * *v0++;

#ifdef UNROLL
#define VSCALIP(v0,a0,n)\
switch (n&0x3){\
case 3:*v0 = a0 * *v0; v0++;\
case 2:*v0 = a0 * *v0; v0++;\
case 1:*v0 = a0 * *v0; v0++;n-=4;case 0:break;}\
while (n>0){v0[0] *= a0;v0[1] *=a0;v0[2] *= a0; v0[3] *= a0; v0 += 4; n -= 4;}

#elif defined(INLINE_WHILE)
#define VSCALIP(v0,a0,n)while(n--){*v0 = a0 * *v0; v0++;}

#elif defined(INLINE_BLAS)
#define VSCALIP(v0,a0,n) {int one=1;dscal_(&n,&a0,v0,&one);}

#else
#define VSCALIP(v0,a0,n){int __i;for(__i=0;__i<n;__i++)v0[__i] *= a0;}
#endif

#endif
