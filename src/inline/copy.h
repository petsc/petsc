/* $Id: copy.h,v 1.1 1994/10/02 02:07:33 bsmith Exp bsmith $ */

/* This file contains definitions for INLINING some popular operations
   All arguments should be simple and in register if possible.
 */

/* x <- y */
#ifndef COPY 

#ifdef UNROLL
#define COPY(x,y,n) \
switch (n & 0x3) {\
case 3: *x++ = *y++;\
case 2: *x++ = *y++;\
case 1: *x++ = *y++;\
n -= 4;case 0:break;}\
while (n>0) {x[0]=y[0];x[1]=y[1];x[2]=y[2];x[3]=y[3];y+=4;x+=4;\
n -= 4;}

#elif defined(INLINE_WHILE)
#define COPY(x,y,n) while(n--) *x++ = *y++;

#elif defined(INLINE_BLAS)
#define COPY(x,y,n) {MEMCPY(y,x,n*sizeof(double));}

#elif defined(INLINE_FOR)
#define COPY(x,y,n) {int __i;register double __s1, __s2;\
for(__i=0;__i<n-1;__i+=2){__s1=(y)[__i];__s2=(y)[__i+1];(x)[__i]=__s1;\
(x)[__i+1]=__s2;}if (n & 0x1)(x)[__i]=(y)[__i];}

#else
#define COPY(x,y,n) {int __i;for(__i=0;__i<n;__i++)(x)[__i]=(y)[__i];}
#endif

/* x <- y; y += inc */
#ifdef UNROLL
#define COPYINC1(x,y,yinc,n) \
if (n & 0x1) {\
*x++ = *y; y+=yinc; n-=1; }\
while (n>0) {x[0]=y[0];x[1]=y[yinc];n -= 2;x+=2;y+=(yinc+yinc);}

#elif defined(INLINE_WHILE)
#define COPYINC1(x,y,yinc,n) while(n--) {*x++ = *y; y+=yinc; }

#elif defined(INLINE_BLAS)
#define COPYINC1(x,y,yinc,n) {int one=1;dcopy_(&n,x,&one,y,&yinc);}

#else
#define COPYINC1(x,y,yinc,n) {int __i,__j=0; for(__i=0;__i<n;__i++){\
(x)[__i]=(y)[__j];__j+=yinc;}}
#endif

/* x <- y; x += inc */
#ifdef UNROLL
#define COPYINC2(x,y,xinc,n) \
if (n & 0x1) {\
*x = *y++; x+=xinc; n-=1; }\
while (n>0) {x[0]=y[0];x[xinc]=y[1];n -= 2;x+=(xinc+xinc);y+=2;}

#elif defined(INLINE_WHILE)
#define COPYINC2(x,y,xinc,n) while(n--) {*x = *y++; x+=xinc; }

#elif defined(INLINE_BLAS)
#define COPYINC2(x,y,xinc,n) {int one=1;dcopy_(&n,x,&xinc,y,&one);}

#else
#define COPYINC2(x,y,xinc,n) {int __i,__j=0;for(__i=0;__i<n;__i++){\
(x)[__j]=(y)[__i];__j+=xinc;}}
#endif

/* x <- y; x += xinc; y += yinc; */
#ifdef UNROLL
#define COPYINC3(x,y,xinc,yinc,n) \
if (n & 0x1) {\
*x = *y; x+=xinc; n-=1; y+=yinc;}\
while (n>0) {x[0]=y[0];x[xinc]=y[yinc];n -= 2;x+=(xinc+xinc);y+=(yinc+yinc);}

#elif defined(INLINE_WHILE)
#define COPYINC3(x,y,xinc,yinc,n) while(n--) {*x = *y; x+=xinc; y += yinc;}

#elif defined(INLINE_BLAS)
#define COPYINC3(x,y,xinc,yinc,n) {dcopy_(&n,x,&xinc,y,&yinc);}

#else
#define COPYINC3(x,y,xinc,yinc,n) {int __i,__j=0,__k=0;\
for(__i=0;__i<n;__i++){(x)[__j]=(y)[__k];__j+=xinc;__k+=yinc;}}
#endif

/* This is ok unless copy is a routine */
#if defined(INLINE_BLAS)
#define ICOPY(a,b,c) {MEMCPY(b,a,c*sizeof(int));}
#else
#define ICOPY(a,b,c) COPY(a,b,c)
#endif

#endif
