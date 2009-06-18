
#ifndef SPARSEDENSESMAXPY


/* take (x,i) into dense vector r; there are nnz entries in (x,i)
   r(xi) -= alpha * xv */
#ifdef PETSC_USE_UNROLL_KERNELS
#define SPARSEDENSESMAXPY(r,alpha,xv,xi,nnz) {PetscInt __noff;\
__noff = nnz & 0x3;\
switch (__noff) {\
case 3: r[xi[2]] -= alpha * xv[2];\
case 2: r[xi[1]] -= alpha * xv[1];\
case 1: r[xi[0]] -= alpha * xv[0];\
nnz -= 4;xi+=__noff;xv+=__noff;\
}\
while (nnz > 0) {\
r[xi[0]] -= alpha * xv[0];r[xi[1]] -= alpha * xv[1];\
r[xi[2]] -= alpha * xv[2];r[xi[3]] -= alpha * xv[3];\
xi  += 4;xv  += 4;nnz -= 4;}}/*}*/

#elif defined(PETSC_USE_WHILE_KERNELS)
#define SPARSEDENSESMAXPY(r,alpha,xv,xi,nnz) {\
while (nnz-->0) r[*xi++] -= alpha * *xv++;}

#elif defined(PETSC_USE_FOR_KERNELS)
#define SPARSEDENSESMAXPY(r,alpha,xv,xi,nnz) {\
 PetscInt __i,__j1,__j2; PetscScalar__s1,__s2;\
for(__i=0;__i<nnz-1;__i+=2){__j1=xi[__i];__j2=xi[__i+1];__s1=alpha*xv[__i];\
__s2=alpha*xv[__i+1];__s1=r[__j1]-__s1;__s2=r[__j2]-__s2;\
r[__j1]=__s1;r[__j2]=__s2;}\
if (nnz & 0x1) r[xi[__i]] -= alpha * xv[__i];}

#else
#define SPARSEDENSESMAXPY(r,alpha,xv,xi,nnz) {\
PetscInt __i;\
for(__i=0;__i<nnz;__i++)r[xi[__i]] -= alpha * xv[__i];}
#endif


/* Form sum += r[map[xi]] * xv; */
#ifdef PETSC_USE_UNROLL_KERNELS
#define SPARSEDENSEMAPDOT(sum,r,xv,xi,map,nnz) {\
if (nnz > 0) {\
switch (nnz & 0x3) {\
case 3: sum += *xv++ * r[map[*xi++]];\
case 2: sum += *xv++ * r[map[*xi++]];\
case 1: sum += *xv++ * r[map[*xi++]];\
nnz -= 4;}\
while (nnz > 0) {\
sum = sum + xv[0] * r[map[xi[0]]] + xv[1] * r[map[xi[1]]] +\
	xv[2] * r[map[xi[2]]] + xv[3] * r[map[xi[3]]];\
xv  += 4; xi += 4; nnz -= 4; }}}

#elif defined(PETSC_USE_WHILE_KERNELS)
#define SPARSEDENSEMAPDOT(sum,r,xv,xi,map,nnz) {\
while (nnz--) sum += *xv++ * r[map[*xi++]];}

#else
#define SPARSEDENSEMAPDOT(sum,r,xv,xi,map,nnz) {\
PetscInt __i;\
for(__i=0;__i<nnz;__i++) sum += xv[__i] * r[map[xi[__i]]];}
#endif

/* Gather xv = r[xi] */
#ifdef PETSC_USE_UNROLL_KERNELS
#define GATHER(xv,xi,r,nz) {PetscInt __noff;\
if (nz > 0) {\
__noff = nz & 0x3;\
switch (nz & 0x3) {\
case 3: xv[2] = r[xi[2]];\
case 2: xv[1] = r[xi[1]];\
case 1: xv[0] = r[xi[0]];\
nz -= 4;xv+=__noff;xi+=__noff;}\
while (nz > 0) {\
xv[0] = r[xi[0]]; xv[1] = r[xi[1]]; xv[2] = r[xi[2]]; xv[3] = r[xi[3]];\
xi  += 4;xv  += 4;nz -= 4;}}}

#elif defined(PETSC_USE_WHILE_KERNELS)
#define GATHER(xv,xi,r,nz) while (nz--) *xv++ = r[*xi++];

#elif defined(PETSC_USE_FOR_KERNELS)
#define GATHER(xv,xi,r,nz) {PetscInt __i; PetscScalar__s1,__s2;\
for(__i=0;__i<nz-1;__i+=2){__s1=r[xi[__i]];__s2=r[xi[__i+1]];\
xv[__i]=__s1;xv[__i+1]=__s2;}if ((nz)&0x1) xv[__i]=r[xi[__i]];}

#else
#define GATHER(xv,xi,r,nz) {PetscInt __i;for(__i=0;__i<nz;__i++)xv[__i]=r[xi[__i]];}
#endif

/* Scatter r[xi] = xv */
#ifdef PETSC_USE_UNROLL_KERNELS
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

#elif defined(PETSC_USE_WHILE_KERNELS)
#define SCATTER(xv,xi,r,nz) while (nz--) r[*xi++]=*xv++;

#elif defined(PETSC_USE_FOR_KERNELS)
#define SCATTER(xv,xi,r,nz) {PetscInt __i; PetscScalar__s1,__s2;\
for(__i=0;__i<nz-1;__i+=2){__s1=xv[__i];__s2=xv[__i+1];\
r[xi[__i]]=__s1;r[xi[__i+1]]=__s2;}if ((nz)&0x1)r[xi[__i]]=xv[__i];}

#else
#define SCATTER(xv,xi,r,nz) {PetscInt __i;\
for(__i=0;__i<nz;__i++)r[xi[__i]]=xv[__i];}
#endif

/* Scatter r[xi] = val */
#ifdef PETSC_USE_UNROLL_KERNELS
#define SCATTERVAL(r,xi,n,val) \
switch (n & 0x3) {\
case 3: r[*xi++] = val;\
case 2: r[*xi++] = val;\
case 1: r[*xi++] = val;\
n -= 4;}\
while (n > 0) {\
r[xi[0]]=val; r[xi[1]]=val; r[xi[2]]=val; r[xi[3]]=val;xi  += 4;n -= 4;}

#elif defined(PETSC_USE_WHILE_KERNELS)
#define SCATTERVAL(r,xi,n,val) while (n--) r[*xi++]=val;

#else
#define SCATTERVAL(r,xi,n,val) {PetscInt __i;for(__i=0;__i<n;__i++)r[xi[__i]]=val;}
#endif

/* Copy vo[xi] = vi[xi] */
#ifdef PETSC_USE_UNROLL_KERNELS
#define COPYPERM(vo,xi,vi,n) \
switch (n & 0x3) {\
case 3: vo[*xi] = vi[*xi];xi++;\
case 2: vo[*xi] = vi[*xi];xi++;\
case 1: vo[*xi] = vi[*xi];xi++;\
n -= 4;}\
while (n > 0) {\
vo[xi[0]]=vi[xi[0]]; vo[xi[1]]=vi[xi[1]]; vo[xi[2]]=vi[xi[2]]; \
vo[xi[3]]=vi[xi[3]];xi  += 4;n -= 4;}}

#elif defined(PETSC_USE_WHILE_KERNELS)
#define COPYPERM(vo,xi,vi,n) while (n--) {vo[*xi]=vi[*xi];xi++;}

#else
#define COPYPERM(vo,xi,vi,n) {PetscInt __i;\
for(__i=0;__i<n;__i++)vo[xi[__i]]=vi[xi[__i]];}
#endif

/* Scale sparse vector v[xi] *= a */
#ifdef PETSC_USE_UNROLL_KERNELS
#define SPARSESCALE(v,xi,n,val) {\
switch (n & 0x3) {\
case 3: v[*xi++] *= val;\
case 2: v[*xi++] *= val;\
case 1: v[*xi++] *= val;\
n -= 4;}\
while (n > 0) {\
v[xi[0]]*=val;vo[xi[1]]*=val; vo[xi[2]]*=val;vo[xi[3]]*=val;xi  += 4;n -= 4;}}}

#elif defined(PETSC_USE_WHILE_KERNELS)
#define SPARSESCALE(v,xi,n,val) {\
while (n--) v[*xi++] *= val;}

#else
#define SPARSESCALE(v,xi,n,val) {PetscInt __i;\
for(__i=0;__i<n;__i++)v[xi[__i]*=val;}
#endif

/* sparse dot sum = sum(a[xi] * b[xi]) */
#ifdef PETSC_USE_UNROLL_KERNELS
#define SPARSEDOT(sum,a,b,xi,n) {\
switch (n & 0x3) {\
case 3: sum += a[*xi] * b[*xi]; xi++;\
case 2: sum += a[*xi] * b[*xi]; xi++;\
case 1: sum += a[*xi] * b[*xi]; xi++;\
n -= 4;}\
while (n > 0) {\
sum+=a[xi[0]]*b[xi[0]]+a[xi[1]]*b[xi[1]]+a[xi[2]]*b[xi[2]]+a[xi[3]]*b[xi[3]];\
xi  += 4;n -= 4;}}}

#elif defined(PETSC_USE_WHILE_KERNELS)
#define SPARSEDOT(sum,a,b,xi,n) {\
while (n--) {sum += a[*xi]*b[*xi];xi++;}}

#else
#define SPARSEDOT(sum,a,b,xi,n) {\
PetscInt __i;\
for(__i=0;__i<n;__i++)sum+= a[xi[__i]]*b[xi[__i]];}
#endif


/* Scatter r[xi] += xv */
#ifdef PETSC_USE_UNROLL_KERNELS
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

#elif defined(PETSC_USE_WHILE_KERNELS)
#define SCATTERADD(xv,xi,r,nz) {\
while (nz--) r[*xi++]+= *xv++;}

#elif defined(PETSC_USE_FOR_KERNELS)
#define SCATTERADD(xv,xi,r,nz) { PetscScalar__s1,__s2;\
 PetscInt __i,__i1,__i2;\
for(__i=0;__i<nz-1;__i+=2){__i1 = xi[__i]; __i2 = xi[__i+1];\
__s1 = r[__i1]; __s2 = r[__i2]; __s1 += xv[__i]; __s2 += xv[__i+1];\
r[__i1]=__s1;r[__i2]=__s2;}if ((nz)&0x1)r[xi[__i]]+=xv[__i];}

#else
#define SCATTERADD(xv,xi,r,nz) {\
PetscInt __i;\
for(__i=0;__i<nz;__i++) r[xi[__i]]+= xv[__i];}
#endif

/* Gather xv += r[xi] */
#ifdef PETSC_USE_UNROLL_KERNELS
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

#elif defined(PETSC_USE_WHILE_KERNELS)
#define GATHERADD(xv,xi,r,nz) {\
while (nz--) *xv++ += r[*xi++];}

#else
#define GATHERADD(xv,xi,r,nz) {\
PetscInt __i;\
for(__i=0;__i<nz;__i++)xv[__i] += r[xi[__i]];}
#endif

#endif
