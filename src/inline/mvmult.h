/* $Id: mvmult.h,v 1.1 1994/10/02 02:07:34 bsmith Exp bsmith $ */

/*
   This file contains routines for matrix-vector products.

   The current definitions are for matrices stored in a diagonal
   format.  This generates a number of "AXPY" like operations, though
   with multiple AXPY's for improved memory referencing (or versions
   with Alpha == 1)
 */

#ifndef DIAGVMULT

/*
   DIAGVMULT(vout,vin,vinc,nd,dv,nr)

   vout += dv * vin; vin += vinc, nr entries, nd diagonals.
   These ASSUME the definitions:
   double *d0, *d1, *d2, *d3, *v0, *v1, *v2, *v3, *vo;
   int    nnr;
 */
#ifdef UNROLL
#define DIAGVMULT(vout,vin,vinc,nd,dv,nr) \
while (nd > 0) { nnr = nr; d0  = dv; dv += nr; d1  = dv; dv += nr;\
d2  = dv; dv += nr; d3  = dv; dv += nr; vo  = vout + sr; v0  = vin  + *doff++;\
v1  = vin  + *doff++; v2  = vin  + *doff++; v3  = vin  + *doff++;\
switch ((nd > 4) ? 4 : nd) {
case 1: VMULT1(vo,d0,v0,vinc,nnr);break;\
case 2: VMULT2(vo,d0,d1,v0,v1,vinc,nnr);break;\
case 3: VMULT3(vo,d0,d1,d2,v0,v1,v2,vinc,nnr);break;\
case 4: VMULT4(vo,d0,d1,d2,d3,v0,v1,v2,v3,vinc,nnr);break;\
}nd -= 4;}

#else
#define DIAGVMULT(vout,vin,vinc,nd,dv,nr) \
while (nd > 0) {nnr = nr; d0 = dv; dv += nr; vo=vout; v0=vin;\
while (nnr--){*vo += *v0 * *d0++; v0+=vinv;vo+=vinc;}}
#endif

#endif
