/* $Id: dot.h,v 1.1 1994/03/18 00:23:57 gropp Exp $ */

#ifndef DOT
#include "system/flog.h"

#ifdef UNROLL
#define DOT(sum,x,y,n) {LOGFLOAT(2*n);LOGLOAD(2*n);LOGSTORE(1);\
switch (n & 0x3) {\
case 3: sum += *x++ * *y++;\
case 2: sum += *x++ * *y++;\
case 1: sum += *x++ * *y++;\
n -= 4;case 0:break;}\
while (n>0) {sum += x[0]*y[0]+x[1]*y[1]+x[2]*y[2]+x[3]*y[3];x+=4;y+=4;\
n -= 4;}}
#define DOT2(sum1,sum2,x,y1,y2,n) {LOGFLOAT(4*n);LOGLOAD(3*n);LOGSTORE(2);\
if(n&0x1){sum1+=*x**y1++;sum2+=*x++**y2++;n--;}\
while (n>0) {sum1+=x[0]*y1[0]+x[1]*y1[1];sum2+=x[0]*y2[0]+x[1]*y2[1];x+=2;\
y1+=2;y2+=2;n -= 2;}}
#define SQR(sum,x,n) {LOGFLOAT(2*n);LOGLOAD(n);LOGSTORE(1);\
switch (n & 0x3) {\
case 3: sum += *x * *x;x++;\
case 2: sum += *x * *x;x++;\
case 1: sum += *x * *x;x++;\
n -= 4;case 0:break;}\
while (n>0) {sum += x[0]*x[0]+x[1]*x[1]+x[2]*x[2]+x[3]*x[3];x+=4;\
n -= 4;}}

#elif defined(INLINE_WHILE)
#define DOT(sum,x,y,n) {LOGFLOAT(2*n);LOGLOAD(2*n);LOGSTORE(1);\
while(n--) sum+= *x++ * *y++;}
#define DOT2(sum1,sum2,x,y1,y2,n) {LOGFLOAT(4*n);LOGLOAD(3*n);LOGSTORE(2);\
while(n--){sum1+= *x**y1++;sum2+=*x++**y2++;}}
#define SQR(sum,x,n)   {LOGFLOAT(2*n);LOGLOAD(n);LOGSTORE(1);\
while(n--) {sum+= *x * *x; x++;}}

#elif defined(INLINE_BLAS)
extern double ddot_();
#define DOT(sum,x,y,n) {int one=1;LOGFLOAT(2*n);LOGLOAD(2*n);LOGSTORE(1);\
sum=ddot_(&n,x,&one,y,&one);}
#define DOT2(sum1,sum2,x,y1,y2,n) {int __i;LOGFLOAT(4*n);LOGLOAD(3*n);\
LOGSTORE(2);\
for(__i=0;__i<n;__i++){sum1+=x[__i]*y1[__i];sum2+=x[__i]*y2[__i];}}
#define SQR(sum,x,n)   {int one=1;LOGFLOAT(2*n);LOGLOAD(n);LOGSTORE(1);\
sum=ddot_(&n,x,&one,x,&one);}

#else
#define DOT(sum,x,y,n) {int __i;LOGFLOAT(2*n);LOGLOAD(2*n);LOGSTORE(1);\
for(__i=0;__i<n;__i++)sum+=x[__i]*y[__i];}
#define DOT2(sum1,sum2,x,y1,y2,n) {int __i;LOGFLOAT(4*n);LOGLOAD(3*n);\
LOGSTORE(2);\
for(__i=0;__i<n;__i++){sum1+=x[__i]*y1[__i];sum2+=x[__i]*y2[__i];}}
#define SQR(sum,x,n)   {int __i;LOGFLOAT(2*n);LOGLOAD(n);LOGSTORE(1);\
for(__i=0;__i<n;__i++)sum+=x[__i]*x[__i];}
#endif

#endif
