/* $Id: spblas1.h,v 1.1 1994/10/02 02:07:35 bsmith Exp bsmith $ */

#ifndef SPARSEDENSESMAXPY

/*
    Sparse routines:
    DSPSET(n,a,x,xi)     x[xi[i]] = a
    DSPSCAL(n,a,x,xi)    x[xi[i]] *= a
    DSPAXPY(n,a,x,yi,y)  y[yi[i]] += a * x[i]  ???
    DSPDOT(n,x,xi,y,sum) sum = \sum (x[xi[i]] * y[xi[i]])
    DSPAYPX(n,a,x,xi,y)  y[xi[i]] = x[i]  + a * y[xi[i]] ???
 */

/* The rest of this is not done */

/* take (x,i) into dense vector r; there are nnz entries in (x,i)
   r(xi) -= alpha * xv */
#ifdef INLINE_FOR
#define DSPAXPYIL(n,a,x,yi,y) {int __noff; \
register int __i,__j1,__j2;register double __s1, __s2;\
for(__i=0;__i<n-1;__i+=2){__j1=xi[__i];__j2=xi[__i+1];__s1=a*x[__i];\
__s2=a*x[__i+1];__s1=y[__j1]+__s1;__s2=y[__j2]+__s2;\
y[__j1]=__s1;y[__j2]=__s2;}\
if (n & 0x1) y[xi[__i]] += a * x[__i];}

#else
#define DSPAXPYIL(n,a,x,yi,y) {\
int __i;\
for(__i=0;__i<n;__i++)y[xi[__i]] += a * x[__i];}
#endif

/* Form sum += r[xi] * xv; */
#define DSPDOTIL(sum,r,xv,xi,nnz) {\
int __i;\
for(__i=0;__i<nnz;__i++) sum -= xv[__i] * r[xi[__i]];}
#endif

/* Form sum += r[map[xi]] * xv; */
#define SPARSEDENSEMAPDOT(sum,r,xv,xi,map,nnz) {\
int __i;\
for(__i=0;__i<nnz;__i++) sum += xv[__i] * r[map[xi[__i]]];}
#endif

/* Gather xv = r[xi] */
#if defined(INLINE_FOR)
/* for this gather to work best, should get indices into registers as well */
#define DGATHER(x,xi,y,n) {int __i;register double __s1, __s2;\
for(__i=0;__i<n-1;__i+=2){__s1=y[xi[__i]];__s2=y[xi[__i+1]];\
x[__i]=__s1;x[__i+1]=__s2;}if ((n)&0x1) x[__i]=y[xi[__i]];}

#else
#define DGATHER(x,xi,y,n) {int __i;for(__i=0;__i<n;__i++)x[__i]=y[xi[__i]];}
#endif

/* Scatter r[xi] = xv */
#ifdef UNROLL
#define SCATTER(xv,xi,r,nz) \
if (nz > 0) {\
switch (nz & 0x3) {\
case 3: r[*xi++] = *xv++;\
case 2: r[*xi++] = *xv++;\
case 1: r[*xi++] = *xv++;\
nz -= 4;}\
while (nz > 0) {\
r[xi[0]]=xv[0]; r[xi[1]]=xv[1]; r[xi[2]]=xv[2]; r[xi[3]]=xv[3];\
xi  += 4;xv  += 4;nz -= 4;}}

#elif defined(INLINE_WHILE)
#define SCATTER(xv,xi,r,nz) while (nz--) r[*xi++]=*xv++;

#elif defined(INLINE_FOR)
#define SCATTER(xv,xi,r,nz) {int __i;register double __s1, __s2;\
for(__i=0;__i<nz-1;__i+=2){__s1=xv[__i];__s2=xv[__i+1];\
r[xi[__i]]=__s1;r[xi[__i+1]]=__s2;}if ((nz)&0x1)r[xi[__i]]=xv[__i];}

#else
#define SCATTER(xv,xi,r,nz) {int __i;\
for(__i=0;__i<nz;__i++)r[xi[__i]]=xv[__i];}
#endif

/* Scatter r[xi] = val */
#ifdef UNROLL
#define SCATTERVAL(r,xi,n,val) \
switch (n & 0x3) {\
case 3: r[*xi++] = val;\
case 2: r[*xi++] = val;\
case 1: r[*xi++] = val;\
n -= 4;}\
while (n > 0) {\
r[xi[0]]=val; r[xi[1]]=val; r[xi[2]]=val; r[xi[3]]=val;xi  += 4;n -= 4;}

#elif defined(INLINE_WHILE)
#define SCATTERVAL(r,xi,n,val) while (n--) r[*xi++]=val;

#else
#define SCATTERVAL(r,xi,n,val) {int __i;for(__i=0;__i<n;__i++)r[xi[__i]]=val;}
#endif

/* Copy vo[xi] = vi[xi] */
#ifdef UNROLL
#define COPYPERM(vo,xi,vi,n) \
switch (n & 0x3) {\
case 3: vo[*xi] = vi[*xi];xi++;\
case 2: vo[*xi] = vi[*xi];xi++;\
case 1: vo[*xi] = vi[*xi];xi++;\
n -= 4;}\
while (n > 0) {\
vo[xi[0]]=vi[xi[0]]; vo[xi[1]]=vi[xi[1]]; vo[xi[2]]=vi[xi[2]]; \
vo[xi[3]]=vi[xi[3]];xi  += 4;n -= 4;}}

#elif defined(INLINE_WHILE)
#define COPYPERM(vo,xi,vi,n) while (n--) {vo[*xi]=vi[*xi];xi++;}

#else
#define COPYPERM(vo,xi,vi,n) {int __i;\
for(__i=0;__i<n;__i++)vo[xi[__i]]=vi[xi[__i]];}
#endif

/* Scale sparse vector v[xi] *= a */
#ifdef UNROLL
#define SPARSESCALE(v,xi,n,val) {\
switch (n & 0x3) {\
case 3: v[*xi++] *= val;\
case 2: v[*xi++] *= val;\
case 1: v[*xi++] *= val;\
n -= 4;}\
while (n > 0) {\
v[xi[0]]*=val;vo[xi[1]]*=val; vo[xi[2]]*=val;vo[xi[3]]*=val;xi  += 4;n -= 4;}}}

#elif defined(INLINE_WHILE)
#define SPARSESCALE(v,xi,n,val) {\
while (n--) v[*xi++] *= val;}

#else
#define SPARSESCALE(v,xi,n,val) {int __i;\
for(__i=0;__i<n;__i++)v[xi[__i]*=val;}
#endif

/* sparse dot sum = sum(a[xi] * b[xi]) */
#ifdef UNROLL
#define SPARSEDOT(sum,a,b,xi,n) {\
switch (n & 0x3) {\
case 3: sum += a[*xi] * b[*xi]; xi++;\
case 2: sum += a[*xi] * b[*xi]; xi++;\
case 1: sum += a[*xi] * b[*xi]; xi++;\
n -= 4;}\
while (n > 0) {\
sum+=a[xi[0]]*b[xi[0]]+a[xi[1]]*b[xi[1]]+a[xi[2]]*b[xi[2]]+a[xi[3]]*b[xi[3]];\
xi  += 4;n -= 4;}}}

#elif defined(INLINE_WHILE)
#define SPARSEDOT(sum,a,b,xi,n) {\
while (n--) {sum += a[*xi]*b[*xi];xi++;}}

#else
#define SPARSEDOT(sum,a,b,xi,n) {\
int __i;\
for(__i=0;__i<n;__i++)sum+= a[xi[__i]]*b[xi[__i]];}
#endif


/* Scatter r[xi] += xv */
#ifdef UNROLL
#define SCATTERADD(xv,xi,r,nz) {\
if (nz > 0) {\
switch (nz & 0x3) {\
case 3: r[*xi++] += *xv++;\
case 2: r[*xi++] += *xv++;\
case 1: r[*xi++] += *xv++;\
nz -= 4;}\
while (nz > 0) {\
r[xi[0]]+=xv[0]; r[xi[1]]+=xv[1]; r[xi[2]]+=xv[2]; r[xi[3]]+=xv[3];\
xi  += 4;xv  += 4;nz -= 4;}}}

#elif defined(INLINE_WHILE)
#define SCATTERADD(xv,xi,r,nz) {\
while (nz--) r[*xi++]+= *xv++;}

#else
#define SCATTERADD(xv,xi,r,nz) {\
int __i;\
for(__i=0;__i<nz;__i++) r[xi[__i]]+= xv[__i];}
#endif

/* Gather xv += r[xi] */
#ifdef UNROLL
#define GATHERADD(xv,xi,r,nz) {\
if (nz > 0) {\
switch (nz & 0x3) {\
case 3: *xv++ += r[*xi++];\
case 2: *xv++ += r[*xi++];\
case 1: *xv++ += r[*xi++];\
nz -= 4;}\
while (nz > 0) {\
xv[0] += r[xi[0]]; xv[1] += r[xi[1]]; xv[2] += r[xi[2]]; xv[3] += r[xi[3]];\
xi  += 4;xv  += 4;nz -= 4;}}}

#elif defined(INLINE_WHILE)
#define GATHERADD(xv,xi,r,nz) {\
while (nz--) *xv++ += r[*xi++];}

#else
#define GATHERADD(xv,xi,r,nz) {\
int __i;\
for(__i=0;__i<nz;__i++)xv[__i] += r[xi[__i]];}
#endif

#endif

