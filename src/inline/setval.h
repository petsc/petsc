
/* This file contains definitions for INLINING some popular operations
   All arguments should be simple and in register if possible.
 */

#ifndef SET

#ifdef PETSC_USE_UNROLL_KERNELS
#define SET(v,n,val) \
switch (n&0x3) { \
case 3: *v++ = val;\
case 2: *v++ = val;\
case 1: *v++ = val;n-=4;\
case 0: while (n>0) {v[0]=val;v[1]=val;v[2]=val;v[3]=val;v+=4;n-=4;}}

#elif defined(PETSC_USE_WHILE_KERNELS)
#define SET(v,n,val) while (n--) *v++ = val;

#else
#define SET(v,n,val) {int __i;for(__i=0;__i<n;__i++)v[__i] = val;}
#endif

#endif



