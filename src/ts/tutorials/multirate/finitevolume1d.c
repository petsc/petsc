#include "finitevolume1d.h"
#include <petscdmda.h>
#include <petscdraw.h>
#include <petsc/private/tsimpl.h>

#include <petsc/private/kernels/blockinvert.h> /* For the Kernel_*_gets_* stuff for BAIJ */
const char *FVBCTypes[] = {"PERIODIC","OUTFLOW","INFLOW","FVBCType","FVBC_",0};

PETSC_STATIC_INLINE PetscReal Sgn(PetscReal a) { return (a<0) ? -1 : 1; }
PETSC_STATIC_INLINE PetscReal Abs(PetscReal a) { return (a<0) ? 0 : a; }
PETSC_STATIC_INLINE PetscReal Sqr(PetscReal a) { return a*a; }

PETSC_UNUSED PETSC_STATIC_INLINE PetscReal MinAbs(PetscReal a,PetscReal b) { return (PetscAbs(a) < PetscAbs(b)) ? a : b; }
PETSC_STATIC_INLINE PetscReal MinMod2(PetscReal a,PetscReal b) { return (a*b<0) ? 0 : Sgn(a)*PetscMin(PetscAbs(a),PetscAbs(b)); }
PETSC_STATIC_INLINE PetscReal MaxMod2(PetscReal a,PetscReal b) { return (a*b<0) ? 0 : Sgn(a)*PetscMax(PetscAbs(a),PetscAbs(b)); }
PETSC_STATIC_INLINE PetscReal MinMod3(PetscReal a,PetscReal b,PetscReal c) {return (a*b<0 || a*c<0) ? 0 : Sgn(a)*PetscMin(PetscAbs(a),PetscMin(PetscAbs(b),PetscAbs(c))); }

/* ----------------------- Lots of limiters, these could go in a separate library ------------------------- */
void Limit_Upwind(LimitInfo info,const PetscScalar *jL,const PetscScalar *jR,PetscScalar *lmt)
{
  PetscInt i;
  for (i=0; i<info->m; i++) lmt[i] = 0;
}
void Limit_LaxWendroff(LimitInfo info,const PetscScalar *jL,const PetscScalar *jR,PetscScalar *lmt)
{
  PetscInt i;
  for (i=0; i<info->m; i++) lmt[i] = jR[i];
}
void Limit_BeamWarming(LimitInfo info,const PetscScalar *jL,const PetscScalar *jR,PetscScalar *lmt)
{
  PetscInt i;
  for (i=0; i<info->m; i++) lmt[i] = jL[i];
}
void Limit_Fromm(LimitInfo info,const PetscScalar *jL,const PetscScalar *jR,PetscScalar *lmt)
{
  PetscInt i;
  for (i=0; i<info->m; i++) lmt[i] = 0.5*(jL[i]+jR[i]);
}
void Limit_Minmod(LimitInfo info,const PetscScalar *jL,const PetscScalar *jR,PetscScalar *lmt)
{
  PetscInt i;
  for (i=0; i<info->m; i++) lmt[i] = MinMod2(jL[i],jR[i]);
}
void Limit_Superbee(LimitInfo info,const PetscScalar *jL,const PetscScalar *jR,PetscScalar *lmt)
{
  PetscInt i;
  for (i=0; i<info->m; i++) lmt[i] = MaxMod2(MinMod2(jL[i],2*jR[i]),MinMod2(2*jL[i],jR[i]));
}
void Limit_MC(LimitInfo info,const PetscScalar *jL,const PetscScalar *jR,PetscScalar *lmt)
{
  PetscInt i;
  for (i=0; i<info->m; i++) lmt[i] = MinMod3(2*jL[i],0.5*(jL[i]+jR[i]),2*jR[i]);
}
void Limit_VanLeer(LimitInfo info,const PetscScalar *jL,const PetscScalar *jR,PetscScalar *lmt)
{ /* phi = (t + abs(t)) / (1 + abs(t)) */
  PetscInt i;
  for (i=0; i<info->m; i++) lmt[i] = (jL[i]*Abs(jR[i])+Abs(jL[i])*jR[i])/(Abs(jL[i])+Abs(jR[i])+1e-15);
}
void Limit_VanAlbada(LimitInfo info,const PetscScalar *jL,const PetscScalar *jR,PetscScalar *lmt) /* differentiable */
{ /* phi = (t + t^2) / (1 + t^2) */
  PetscInt i;
  for (i=0; i<info->m; i++) lmt[i] = (jL[i]*Sqr(jR[i])+Sqr(jL[i])*jR[i])/(Sqr(jL[i])+Sqr(jR[i])+1e-15);
}
void Limit_VanAlbadaTVD(LimitInfo info,const PetscScalar *jL,const PetscScalar *jR,PetscScalar *lmt)
{ /* phi = (t + t^2) / (1 + t^2) */
  PetscInt i;
  for (i=0; i<info->m; i++) lmt[i] = (jL[i]*jR[i]<0) ? 0 : (jL[i]*Sqr(jR[i])+Sqr(jL[i])*jR[i])/(Sqr(jL[i])+Sqr(jR[i])+1e-15);
}
void Limit_Koren(LimitInfo info,const PetscScalar *jL,const PetscScalar *jR,PetscScalar *lmt) /* differentiable */
{ /* phi = (t + 2*t^2) / (2 - t + 2*t^2) */
  PetscInt i;
  for (i=0; i<info->m; i++) lmt[i] = ((jL[i]*Sqr(jR[i])+2*Sqr(jL[i])*jR[i])/(2*Sqr(jL[i])-jL[i]*jR[i]+2*Sqr(jR[i])+1e-15));
}
void Limit_KorenSym(LimitInfo info,const PetscScalar *jL,const PetscScalar *jR,PetscScalar *lmt) /* differentiable */
{ /* Symmetric version of above */
  PetscInt i;
  for (i=0; i<info->m; i++) lmt[i] = (1.5*(jL[i]*Sqr(jR[i])+Sqr(jL[i])*jR[i])/(2*Sqr(jL[i])-jL[i]*jR[i]+2*Sqr(jR[i])+1e-15));
}
void Limit_Koren3(LimitInfo info,const PetscScalar *jL,const PetscScalar *jR,PetscScalar *lmt)
{ /* Eq 11 of Cada-Torrilhon 2009 */
  PetscInt i;
  for (i=0; i<info->m; i++) lmt[i] = MinMod3(2*jL[i],(jL[i]+2*jR[i])/3,2*jR[i]);
}
static PetscReal CadaTorrilhonPhiHatR_Eq13(PetscReal L,PetscReal R)
{
  return PetscMax(0,PetscMin((L+2*R)/3,PetscMax(-0.5*L,PetscMin(2*L,PetscMin((L+2*R)/3,1.6*R)))));
}
void Limit_CadaTorrilhon2(LimitInfo info,const PetscScalar *jL,const PetscScalar *jR,PetscScalar *lmt)
{ /* Cada-Torrilhon 2009, Eq 13 */
  PetscInt i;
  for (i=0; i<info->m; i++) lmt[i] = CadaTorrilhonPhiHatR_Eq13(jL[i],jR[i]);
}
void Limit_CadaTorrilhon3R(PetscReal r,LimitInfo info,const PetscScalar *jL,const PetscScalar *jR,PetscScalar *lmt)
{ /* Cada-Torrilhon 2009, Eq 22 */
  /* They recommend 0.001 < r < 1, but larger values are more accurate in smooth regions */
  const PetscReal eps = 1e-7,hx = info->hx;
  PetscInt        i;
  for (i=0; i<info->m; i++) {
    const PetscReal eta = (Sqr(jL[i])+Sqr(jR[i]))/Sqr(r*hx);
    lmt[i] = ((eta < 1-eps) ? (jL[i]+2*jR[i])/3 : ((eta>1+eps) ? CadaTorrilhonPhiHatR_Eq13(jL[i],jR[i]) : 0.5*((1-(eta-1)/eps)*(jL[i]+2*jR[i])/3+(1+(eta+1)/eps)*CadaTorrilhonPhiHatR_Eq13(jL[i],jR[i]))));
  }
}
void Limit_CadaTorrilhon3R0p1(LimitInfo info,const PetscScalar *jL,const PetscScalar *jR,PetscScalar *lmt)
{
  Limit_CadaTorrilhon3R(0.1,info,jL,jR,lmt);
}
void Limit_CadaTorrilhon3R1(LimitInfo info,const PetscScalar *jL,const PetscScalar *jR,PetscScalar *lmt)
{
  Limit_CadaTorrilhon3R(1,info,jL,jR,lmt);
}
void Limit_CadaTorrilhon3R10(LimitInfo info,const PetscScalar *jL,const PetscScalar *jR,PetscScalar *lmt)
{
  Limit_CadaTorrilhon3R(10,info,jL,jR,lmt);
}
void Limit_CadaTorrilhon3R100(LimitInfo info,const PetscScalar *jL,const PetscScalar *jR,PetscScalar *lmt)
{
  Limit_CadaTorrilhon3R(100,info,jL,jR,lmt);
}

/* ----------------------- Limiters for split systems ------------------------- */

void Limit2_Upwind(LimitInfo info,const PetscScalar *jL,const PetscScalar *jR,const PetscInt sf,const PetscInt fs,PetscInt n,PetscScalar *lmt)
{
  PetscInt i;
  for (i=0; i<info->m; i++) lmt[i] = 0;
}
void Limit2_LaxWendroff(LimitInfo info,const PetscScalar *jL,const PetscScalar *jR,const PetscInt sf,const PetscInt fs,PetscInt n,PetscScalar *lmt)
{
  PetscInt i;
  if (n < sf-1) {                                 /* slow components */
    for (i=0; i<info->m; i++) lmt[i] = jR[i]/info->hxs;
  } else if (n == sf-1) {                         /* slow component which is next to fast components */
    for (i=0; i<info->m; i++) lmt[i] = jR[i]/(info->hxs/2.0+info->hxf/2.0);
  } else if (n == sf) {                            /* fast component which is next to slow components */
    for (i=0; i<info->m; i++) lmt[i] = jR[i]/info->hxf;
  } else if (n > sf && n < fs-1) { /* fast components */
    for (i=0; i<info->m; i++) lmt[i] = jR[i]/info->hxf;
  } else if (n == fs-1) {                /* fast component next to slow components */
    for (i=0; i<info->m; i++) lmt[i] = jR[i]/(info->hxf/2.0+info->hxs/2.0);
  } else if (n == fs) {                  /* slow component next to fast components */
    for (i=0; i<info->m; i++) lmt[i] = jR[i]/info->hxs;
  } else {                                              /* slow components */
    for (i=0; i<info->m; i++) lmt[i] = jR[i]/info->hxs;
  }
}
void Limit2_BeamWarming(LimitInfo info,const PetscScalar *jL,const PetscScalar *jR,const PetscInt sf,const PetscInt fs,PetscInt n,PetscScalar *lmt)
{
  PetscInt i;
  if (n < sf-1) {
    for (i=0; i<info->m; i++) lmt[i] = jL[i]/info->hxs;
  } else if (n == sf-1) {
    for (i=0; i<info->m; i++) lmt[i] = jL[i]/info->hxs;
  } else if (n == sf) {
    for (i=0; i<info->m; i++) lmt[i] = jL[i]/(info->hxs/2.0+info->hxf/2.0);
  } else if (n > sf && n < fs-1) {
    for (i=0; i<info->m; i++) lmt[i] = jL[i]/info->hxf;
  } else if (n == fs-1) {
    for (i=0; i<info->m; i++) lmt[i] = jL[i]/info->hxf;
  } else if (n == fs) {
    for (i=0; i<info->m; i++) lmt[i] = jL[i]/(info->hxf/2.0+info->hxs/2.0);
  } else {
    for (i=0; i<info->m; i++) lmt[i] = jL[i]/info->hxs;
  }
}
void Limit2_Fromm(LimitInfo info,const PetscScalar *jL,const PetscScalar *jR,const PetscInt sf,const PetscInt fs,PetscInt n,PetscScalar *lmt)
{
  PetscInt i;
  if (n < sf-1) {
    for (i=0; i<info->m; i++) lmt[i] = 0.5*(jL[i]+jR[i])/info->hxs;
  } else if (n == sf-1) {
    for (i=0; i<info->m; i++) lmt[i] = 0.5*(jL[i]/info->hxs+jR[i]/(info->hxs/2.0+info->hxf/2.0));
  } else if (n == sf) {
    for (i=0; i<info->m; i++) lmt[i] = 0.5*(jL[i]/(info->hxs/2.0+info->hxf/2.0)+jR[i]/info->hxf);
  } else if (n > sf && n < fs-1) {
    for (i=0; i<info->m; i++) lmt[i] = 0.5*(jL[i]+jR[i])/info->hxf;
  } else if (n == fs-1) {
    for (i=0; i<info->m; i++) lmt[i] = 0.5*(jL[i]/info->hxf+jR[i]/(info->hxf/2.0+info->hxs/2.0));
  } else if (n == fs) {
    for (i=0; i<info->m; i++) lmt[i] = 0.5*(jL[i]/(info->hxf/2.0+info->hxs/2.0)+jR[i]/info->hxs);
  } else {
    for (i=0; i<info->m; i++) lmt[i] = 0.5*(jL[i]+jR[i])/info->hxs;
  }
}
void Limit2_Minmod(LimitInfo info,const PetscScalar *jL,const PetscScalar *jR,const PetscInt sf,const PetscInt fs,PetscInt n,PetscScalar *lmt)
{
  PetscInt i;
  if (n < sf-1) {
    for (i=0; i<info->m; i++) lmt[i] = MinMod2(jL[i],jR[i])/info->hxs;
  } else if (n == sf-1) {
    for (i=0; i<info->m; i++) lmt[i] = MinMod2(jL[i]/info->hxs,jR[i]/(info->hxs/2.0+info->hxf/2.0));
  } else if (n == sf) {
    for (i=0; i<info->m; i++) lmt[i] = MinMod2(jL[i]/(info->hxs/2.0+info->hxf/2.0),jR[i]/info->hxf);
  } else if (n > sf && n < fs-1) {
    for (i=0; i<info->m; i++) lmt[i] = MinMod2(jL[i],jR[i])/info->hxf;
  } else if (n == fs-1) {
    for (i=0; i<info->m; i++) lmt[i] = MinMod2(jL[i]/info->hxf,jR[i]/(info->hxf/2.0+info->hxs/2.0));
  } else if (n == fs) {
    for (i=0; i<info->m; i++) lmt[i] = MinMod2(jL[i]/(info->hxf/2.0+info->hxs/2.0),jR[i]/info->hxs);
  } else {
    for (i=0; i<info->m; i++) lmt[i] = MinMod2(jL[i],jR[i])/info->hxs;
  }
}
void Limit2_Superbee(LimitInfo info,const PetscScalar *jL,const PetscScalar *jR,const PetscInt sf,const PetscInt fs,PetscInt n,PetscScalar *lmt)
{
  PetscInt i;
  if (n < sf-1) {
    for (i=0; i<info->m; i++) lmt[i] = MaxMod2(MinMod2(jL[i],2*jR[i]),MinMod2(2*jL[i],jR[i]))/info->hxs;
  } else if (n == sf-1) {
    for (i=0; i<info->m; i++) lmt[i] = MaxMod2(MinMod2(jL[i]/info->hxs,2*jR[i]/(info->hxs/2.0+info->hxf/2.0)),MinMod2(2*jL[i]/info->hxs,jR[i]/(info->hxs/2.0+info->hxf/2.0)));
  } else if (n == sf) {
    for (i=0; i<info->m; i++) lmt[i] = MaxMod2(MinMod2(jL[i]/(info->hxs/2.0+info->hxf/2.0),2*jR[i]/info->hxf),MinMod2(2*jL[i]/(info->hxs/2.0+info->hxf/2.0),jR[i]/info->hxf));
  } else if (n > sf && n < fs-1) {
    for (i=0; i<info->m; i++) lmt[i] = MinMod2(MinMod2(jL[i],2*jR[i]),MinMod2(2*jL[i],jR[i]))/info->hxf;
  } else if (n == fs-1) {
    for (i=0; i<info->m; i++) lmt[i] = MinMod2(MinMod2(jL[i]/info->hxf,2*jR[i]/(info->hxf/2.0+info->hxs/2.0)),MinMod2(2*jL[i]/info->hxf,jR[i]/(info->hxf/2.0+info->hxs/2.0)));
  } else if (n == fs) {
    for (i=0; i<info->m; i++) lmt[i] = MinMod2(MinMod2(jL[i]/(info->hxf/2.0+info->hxs/2.0),2*jR[i]/info->hxs),MinMod2(2*jL[i]/(info->hxf/2.0+info->hxs/2.0),jR[i]/info->hxs));
  } else {
    for (i=0; i<info->m; i++) lmt[i] = MinMod2(MinMod2(jL[i],2*jR[i]),MinMod2(2*jL[i],jR[i]))/info->hxs;
  }
}
void Limit2_MC(LimitInfo info,const PetscScalar *jL,const PetscScalar *jR,const PetscInt sf,const PetscInt fs,PetscInt n,PetscScalar *lmt)
{
  PetscInt i;
  if (n < sf-1) {
    for (i=0; i<info->m; i++) lmt[i] = MinMod3(2*jL[i],0.5*(jL[i]+jR[i]),2*jR[i])/info->hxs;
  } else if (n == sf-1) {
    for (i=0; i<info->m; i++) lmt[i] = MinMod3(2*jL[i]/info->hxs,0.5*(jL[i]/info->hxs+jR[i]/(info->hxf/2.0+info->hxs/2.0)),2*jR[i]/(info->hxf/2.0+info->hxs/2.0));
  } else if (n == sf) {
    for (i=0; i<info->m; i++) lmt[i] = MinMod3(2*jL[i]/(info->hxs/2.0+info->hxf/2.0),0.5*(jL[i]/(info->hxs/2.0+info->hxf/2.0)+jR[i]/info->hxf),2*jR[i]/info->hxf);
  } else if (n > sf && n < fs-1) {
    for (i=0; i<info->m; i++) lmt[i] = MinMod3(2*jL[i],0.5*(jL[i]+jR[i]),2*jR[i])/info->hxf;
  } else if (n == fs-1) {
    for (i=0; i<info->m; i++) lmt[i] = MinMod3(2*jL[i]/info->hxf,0.5*(jL[i]/info->hxf+jR[i]/(info->hxf/2.0+info->hxs/2.0)),2*jR[i]/(info->hxf/2.0+info->hxs/2.0));
  } else if (n == fs) {
    for (i=0; i<info->m; i++) lmt[i] = MinMod3(2*jL[i]/(info->hxf/2.0+info->hxs/2.0),0.5*(jL[i]/(info->hxf/2.0+info->hxs/2.0)+jR[i]/info->hxs),2*jR[i]/info->hxs);
  } else {
    for (i=0; i<info->m; i++) lmt[i] = MinMod3(2*jL[i],0.5*(jL[i]+jR[i]),2*jR[i])/info->hxs;
  }
}
void Limit2_Koren3(LimitInfo info,const PetscScalar *jL,const PetscScalar *jR,const PetscInt sf,const PetscInt fs,PetscInt n,PetscScalar *lmt)
{ /* Eq 11 of Cada-Torrilhon 2009 */
  PetscInt i;
  if (n < sf-1) {
    for (i=0; i<info->m; i++) lmt[i] = MinMod3(2*jL[i],(jL[i]+2*jR[i])/3,2*jR[i])/info->hxs;
  } else if (n == sf-1) {
    for (i=0; i<info->m; i++) lmt[i] = MinMod3(2*jL[i]/info->hxs,(jL[i]/info->hxs+2*jR[i]/(info->hxf/2.0+info->hxs/2.0))/3,2*jR[i]/(info->hxf/2.0+info->hxs/2.0));
  } else if (n == sf) {
    for (i=0; i<info->m; i++) lmt[i] = MinMod3(2*jL[i]/(info->hxs/2.0+info->hxf/2.0),(jL[i]/(info->hxs/2.0+info->hxf/2.0)+2*jR[i]/info->hxf)/3,2*jR[i]/info->hxf);
  } else if (n > sf && n < fs-1) {
    for (i=0; i<info->m; i++) lmt[i] = MinMod3(2*jL[i],(jL[i]+2*jR[i])/3,2*jR[i])/info->hxf;
  } else if (n == fs-1) {
    for (i=0; i<info->m; i++) lmt[i] = MinMod3(2*jL[i]/info->hxf,(jL[i]/info->hxf+2*jR[i]/(info->hxf/2.0+info->hxs/2.0))/3,2*jR[i]/(info->hxf/2.0+info->hxs/2.0));
  } else if (n == fs) {
    for (i=0; i<info->m; i++) lmt[i] = MinMod3(2*jL[i]/(info->hxf/2.0+info->hxs/2.0),(jL[i]/(info->hxf/2.0+info->hxs/2.0)+2*jR[i]/info->hxs)/3,2*jR[i]/info->hxs);
  } else {
    for (i=0; i<info->m; i++) lmt[i] = MinMod3(2*jL[i],(jL[i]+2*jR[i])/3,2*jR[i])/info->hxs;
  }
}

/* ---- Three-way splitting ---- */
void Limit3_Upwind(LimitInfo info,const PetscScalar *jL,const PetscScalar *jR,const PetscInt sm,const PetscInt mf,const PetscInt fm,const PetscInt ms,PetscInt n,PetscScalar *lmt)
{
  PetscInt i;
  for (i=0; i<info->m; i++) lmt[i] = 0;
}
void Limit3_LaxWendroff(LimitInfo info,const PetscScalar *jL,const PetscScalar *jR,const PetscInt sm,const PetscInt mf,const PetscInt fm,const PetscInt ms,PetscInt n,PetscScalar *lmt)
{
  PetscInt i;
  if (n < sm-1 || n > ms) {                                 /* slow components */
    for (i=0; i<info->m; i++) lmt[i] = jR[i]/info->hxs;
  } else if (n == sm-1 || n == ms-1) {                         /* slow component which is next to medium components */
    for (i=0; i<info->m; i++) lmt[i] = jR[i]/(info->hxs/2.0+info->hxm/2.0);
  } else if (n < mf-1 || n > fm) { /* medium components */
    for (i=0; i<info->m; i++) lmt[i] = jR[i]/info->hxm;
  } else if (n == mf-1 || n == fm) { /* medium component next to fast components */
    for (i=0; i<info->m; i++) lmt[i] = jR[i]/(info->hxm/2.0+info->hxf/2.0);
  } else { /* fast components */
    for (i=0; i<info->m; i++) lmt[i] = jR[i]/info->hxf;
  }
}
void Limit3_BeamWarming(LimitInfo info,const PetscScalar *jL,const PetscScalar *jR,const PetscInt sm,const PetscInt mf,const PetscInt fm,const PetscInt ms,PetscInt n,PetscScalar *lmt)
{
  PetscInt i;
  if (n < sm || n > ms) {
    for (i=0; i<info->m; i++) lmt[i] = jL[i]/info->hxs;
  } else if (n == sm || n == ms) {
    for (i=0; i<info->m; i++) lmt[i] = jL[i]/(info->hxs/2.0+info->hxf/2.0);
  }else if (n < mf || n > fm) {
    for (i=0; i<info->m; i++) lmt[i] = jL[i]/info->hxm;
  } else if (n == mf || n == fm) {
    for (i=0; i<info->m; i++) lmt[i] = jL[i]/(info->hxm/2.0+info->hxf/2.0);
  } else {
    for (i=0; i<info->m; i++) lmt[i] = jL[i]/info->hxf;
  }
}
void Limit3_Fromm(LimitInfo info,const PetscScalar *jL,const PetscScalar *jR,const PetscInt sm,const PetscInt mf,const PetscInt fm,const PetscInt ms,PetscInt n,PetscScalar *lmt)
{
  PetscInt i;
  if (n < sm-1 || n > ms) {
    for (i=0; i<info->m; i++) lmt[i] = 0.5*(jL[i]+jR[i])/info->hxs;
  } else if (n == sm-1) {
    for (i=0; i<info->m; i++) lmt[i] = 0.5*(jL[i]/info->hxs+jR[i]/(info->hxs/2.0+info->hxf/2.0));
  } else if (n == sm) {
    for (i=0; i<info->m; i++) lmt[i] = 0.5*(jL[i]/(info->hxs/2.0+info->hxm/2.0)+jR[i]/info->hxm);
  } else if (n == ms-1) {
    for (i=0; i<info->m; i++) lmt[i] = 0.5*(jL[i]/info->hxm+jR[i]/(info->hxs/2.0+info->hxf/2.0));
  } else if (n == ms) {
    for (i=0; i<info->m; i++) lmt[i] = 0.5*(jL[i]/(info->hxm/2.0+info->hxs/2.0)+jR[i]/info->hxs);
  } else if (n < mf-1 || n > fm) {
    for (i=0; i<info->m; i++) lmt[i] = 0.5*(jL[i]+jR[i])/info->hxm;
  } else if (n == mf-1) {
    for (i=0; i<info->m; i++) lmt[i] = 0.5*(jL[i]/info->hxm+jR[i]/(info->hxm/2.0+info->hxf/2.0));
  } else if (n == mf) {
    for (i=0; i<info->m; i++) lmt[i] = 0.5*(jL[i]/(info->hxm/2.0+info->hxf/2.0)+jR[i]/info->hxf);
  } else if (n == fm-1) {
    for (i=0; i<info->m; i++) lmt[i] = 0.5*(jL[i]/info->hxf+jR[i]/(info->hxf/2.0+info->hxm/2.0));
  } else if (n == fm) {
    for (i=0; i<info->m; i++) lmt[i] = 0.5*(jL[i]/(info->hxf/2.0+info->hxm/2.0)+jR[i]/info->hxm);
  } else {
    for (i=0; i<info->m; i++) lmt[i] = 0.5*(jL[i]+jR[i])/info->hxf;
  }
}
void Limit3_Minmod(LimitInfo info,const PetscScalar *jL,const PetscScalar *jR,const PetscInt sm,const PetscInt mf,const PetscInt fm,const PetscInt ms,PetscInt n,PetscScalar *lmt)
{
  PetscInt i;
  if (n < sm-1 || n > ms) {
    for (i=0; i<info->m; i++) lmt[i] = MinMod2(jL[i],jR[i])/info->hxs;
  } else if (n == sm-1) {
    for (i=0; i<info->m; i++) lmt[i] = MinMod2(jL[i]/info->hxs,jR[i]/(info->hxs/2.0+info->hxf/2.0));
  } else if (n == sm) {
    for (i=0; i<info->m; i++) lmt[i] = MinMod2(jL[i]/(info->hxs/2.0+info->hxf/2.0),jR[i]/info->hxf);
  } else if (n == ms-1) {
    for (i=0; i<info->m; i++) lmt[i] = MinMod2(jL[i]/info->hxm,jR[i]/(info->hxm/2.0+info->hxs/2.0));
  } else if (n == ms) {
    for (i=0; i<info->m; i++) lmt[i] = MinMod2(jL[i]/(info->hxm/2.0+info->hxs/2.0),jR[i]/info->hxs);
  } else if (n < mf-1 || n > fm) {
    for (i=0; i<info->m; i++) lmt[i] = MinMod2(jL[i],jR[i])/info->hxm;
  } else if (n == mf-1) {
    for (i=0; i<info->m; i++) lmt[i] = MinMod2(jL[i]/info->hxm,jR[i]/(info->hxm/2.0+info->hxf/2.0));
  } else if (n == mf) {
    for (i=0; i<info->m; i++) lmt[i] = MinMod2(jL[i]/(info->hxm/2.0+info->hxf/2.0),jR[i]/info->hxf);
  } else if (n == fm-1) {
    for (i=0; i<info->m; i++) lmt[i] = MinMod2(jL[i]/info->hxf,jR[i]/(info->hxf/2.0+info->hxm/2.0));
  } else if (n == fm) {
    for (i=0; i<info->m; i++) lmt[i] = MinMod2(jL[i]/(info->hxf/2.0+info->hxm/2.0),jR[i]/info->hxm);
  } else {
    for (i=0; i<info->m; i++) lmt[i] = 0.5*(jL[i]+jR[i])/info->hxf;
  }
}
void Limit3_Superbee(LimitInfo info,const PetscScalar *jL,const PetscScalar *jR,const PetscInt sm,const PetscInt mf,const PetscInt fm,const PetscInt ms,PetscInt n,PetscScalar *lmt)
{
  PetscInt i;
  if (n < sm-1 || n > ms) {
    for (i=0; i<info->m; i++) lmt[i] = MaxMod2(MinMod2(jL[i],2*jR[i]),MinMod2(2*jL[i],jR[i]))/info->hxs;
  } else if (n == sm-1) {
    for (i=0; i<info->m; i++) lmt[i] = MaxMod2(MinMod2(jL[i]/info->hxs,2*jR[i]/(info->hxs/2.0+info->hxm/2.0)),MinMod2(2*jL[i]/info->hxs,jR[i]/(info->hxs/2.0+info->hxm/2.0)));
  } else if (n == sm) {
    for (i=0; i<info->m; i++) lmt[i] = MaxMod2(MinMod2(jL[i]/(info->hxs/2.0+info->hxm/2.0),2*jR[i]/info->hxm),MinMod2(2*jL[i]/(info->hxs/2.0+info->hxm/2.0),jR[i]/info->hxm));
  } else if (n == ms-1) {
    for (i=0; i<info->m; i++) lmt[i] = MinMod2(MinMod2(jL[i]/info->hxm,2*jR[i]/(info->hxm/2.0+info->hxs/2.0)),MinMod2(2*jL[i]/info->hxm,jR[i]/(info->hxm/2.0+info->hxs/2.0)));
  } else if (n == ms) {
    for (i=0; i<info->m; i++) lmt[i] = MinMod2(MinMod2(jL[i]/(info->hxm/2.0+info->hxs/2.0),2*jR[i]/info->hxs),MinMod2(2*jL[i]/(info->hxm/2.0+info->hxs/2.0),jR[i]/info->hxs));
  } else if (n < mf-1 || n > fm) {
    for (i=0; i<info->m; i++) lmt[i] = MaxMod2(MinMod2(jL[i],2*jR[i]),MinMod2(2*jL[i],jR[i]))/info->hxm;
  } else if (n == mf-1) {
    for (i=0; i<info->m; i++) lmt[i] = MaxMod2(MinMod2(jL[i]/info->hxm,2*jR[i]/(info->hxm/2.0+info->hxf/2.0)),MinMod2(2*jL[i]/info->hxm,jR[i]/(info->hxm/2.0+info->hxf/2.0)));
  } else if (n == mf) {
    for (i=0; i<info->m; i++) lmt[i] = MaxMod2(MinMod2(jL[i]/(info->hxm/2.0+info->hxf/2.0),2*jR[i]/info->hxf),MinMod2(2*jL[i]/(info->hxm/2.0+info->hxf/2.0),jR[i]/info->hxf));
  } else if (n == fm-1) {
    for (i=0; i<info->m; i++) lmt[i] = MinMod2(MinMod2(jL[i]/info->hxf,2*jR[i]/(info->hxf/2.0+info->hxm/2.0)),MinMod2(2*jL[i]/info->hxf,jR[i]/(info->hxf/2.0+info->hxm/2.0)));
  } else if (n == fm) {
    for (i=0; i<info->m; i++) lmt[i] = MinMod2(MinMod2(jL[i]/(info->hxf/2.0+info->hxm/2.0),2*jR[i]/info->hxm),MinMod2(2*jL[i]/(info->hxf/2.0+info->hxm/2.0),jR[i]/info->hxm));
  } else {
    for (i=0; i<info->m; i++) lmt[i] = MaxMod2(MinMod2(jL[i],2*jR[i]),MinMod2(2*jL[i],jR[i]))/info->hxf;
  }
}
void Limit3_MC(LimitInfo info,const PetscScalar *jL,const PetscScalar *jR,const PetscInt sm,const PetscInt mf,const PetscInt fm,const PetscInt ms,PetscInt n,PetscScalar *lmt)
{
  PetscInt i;
  if (n < sm-1 || n > ms) {
    for (i=0; i<info->m; i++) lmt[i] = MinMod3(2*jL[i],0.5*(jL[i]+jR[i]),2*jR[i])/info->hxs;
  } else if (n == sm-1) {
    for (i=0; i<info->m; i++) lmt[i] = MinMod3(2*jL[i]/info->hxs,0.5*(jL[i]/info->hxs+jR[i]/(info->hxs/2.0+info->hxm/2.0)),2*jR[i]/(info->hxs/2.0+info->hxm/2.0));
  } else if (n == sm) {
    for (i=0; i<info->m; i++) lmt[i] = MinMod3(2*jL[i]/(info->hxs/2.0+info->hxm/2.0),0.5*(jL[i]/(info->hxs/2.0+info->hxm/2.0)+jR[i]/info->hxm),2*jR[i]/info->hxm);
  } else if (n == ms-1) {
    for (i=0; i<info->m; i++) lmt[i] = MinMod3(2*jL[i]/info->hxm,0.5*(jL[i]/info->hxm+jR[i]/(info->hxm/2.0+info->hxs/2.0)),2*jR[i]/(info->hxm/2.0+info->hxs/2.0));
  } else if (n == ms) {
    for (i=0; i<info->m; i++) lmt[i] = MinMod3(2*jL[i]/(info->hxm/2.0+info->hxs/2.0),0.5*(jL[i]/(info->hxm/2.0+info->hxs/2.0)+jR[i]/info->hxs),2*jR[i]/info->hxs);
  } else if (n < mf-1 || n > fm) {
    for (i=0; i<info->m; i++) lmt[i] = MinMod3(2*jL[i],0.5*(jL[i]+jR[i]),2*jR[i])/info->hxm;
  } else if (n == mf-1) {
    for (i=0; i<info->m; i++) lmt[i] = MinMod3(2*jL[i]/info->hxm,0.5*(jL[i]/info->hxm+jR[i]/(info->hxm/2.0+info->hxf/2.0)),2*jR[i]/(info->hxm/2.0+info->hxf/2.0));
  } else if (n == mf) {
    for (i=0; i<info->m; i++) lmt[i] = MinMod3(2*jL[i]/(info->hxm/2.0+info->hxf/2.0),0.5*(jL[i]/(info->hxm/2.0+info->hxf/2.0)+jR[i]/info->hxf),2*jR[i]/info->hxf);
  } else if (n == fm-1) {
    for (i=0; i<info->m; i++) lmt[i] = MinMod3(2*jL[i]/info->hxf,0.5*(jL[i]/info->hxf+jR[i]/(info->hxf/2.0+info->hxm/2.0)),2*jR[i]/(info->hxf/2.0+info->hxm/2.0));
  } else if (n == fm) {
    for (i=0; i<info->m; i++) lmt[i] = MinMod3(2*jL[i]/(info->hxf/2.0+info->hxm/2.0),0.5*(jL[i]/(info->hxf/2.0+info->hxm/2.0)+jR[i]/info->hxm),2*jR[i]/info->hxm);
  } else {
    for (i=0; i<info->m; i++) lmt[i] = MinMod3(2*jL[i],0.5*(jL[i]+jR[i]),2*jR[i])/info->hxf;
  }
}
void Limit3_Koren3(LimitInfo info,const PetscScalar *jL,const PetscScalar *jR,const PetscInt sm,const PetscInt mf,const PetscInt fm,const PetscInt ms,PetscInt n,PetscScalar *lmt)
{ /* Eq 11 of Cada-Torrilhon 2009 */
  PetscInt i;
  if (n < sm-1 || n > ms) {
    for (i=0; i<info->m; i++) lmt[i] = MinMod3(2*jL[i],(jL[i]+2*jR[i])/3,2*jR[i])/info->hxs;
  } else if (n == sm-1) {
    for (i=0; i<info->m; i++) lmt[i] = MinMod3(2*jL[i]/info->hxs,(jL[i]/info->hxs+2*jR[i]/(info->hxm/2.0+info->hxs/2.0))/3,2*jR[i]/(info->hxm/2.0+info->hxs/2.0));
  } else if (n == sm) {
    for (i=0; i<info->m; i++) lmt[i] = MinMod3(2*jL[i]/(info->hxs/2.0+info->hxm/2.0),(jL[i]/(info->hxs/2.0+info->hxm/2.0)+2*jR[i]/info->hxm)/3,2*jR[i]/info->hxm);
  } else if (n == ms-1) {
    for (i=0; i<info->m; i++) lmt[i] = MinMod3(2*jL[i]/info->hxm,(jL[i]/info->hxm+2*jR[i]/(info->hxm/2.0+info->hxs/2.0))/3,2*jR[i]/(info->hxm/2.0+info->hxs/2.0));
  } else if (n == ms) {
    for (i=0; i<info->m; i++) lmt[i] = MinMod3(2*jL[i]/(info->hxm/2.0+info->hxs/2.0),(jL[i]/(info->hxm/2.0+info->hxs/2.0)+2*jR[i]/info->hxs)/3,2*jR[i]/info->hxs);
  } else if (n < mf-1 || n > fm) {
    for (i=0; i<info->m; i++) lmt[i] = MinMod3(2*jL[i],(jL[i]+2*jR[i])/3,2*jR[i])/info->hxm;
  } else if (n == mf-1) {
    for (i=0; i<info->m; i++) lmt[i] = MinMod3(2*jL[i]/info->hxm,(jL[i]/info->hxm+2*jR[i]/(info->hxm/2.0+info->hxf/2.0))/3,2*jR[i]/(info->hxm/2.0+info->hxf/2.0));
  } else if (n == mf) {
    for (i=0; i<info->m; i++) lmt[i] = MinMod3(2*jL[i]/(info->hxm/2.0+info->hxf/2.0),(jL[i]/(info->hxm/2.0+info->hxf/2.0)+2*jR[i]/info->hxf)/3,2*jR[i]/info->hxf);
  } else if (n == fm-1) {
    for (i=0; i<info->m; i++) lmt[i] = MinMod3(2*jL[i]/info->hxf,(jL[i]/info->hxf+2*jR[i]/(info->hxf/2.0+info->hxm/2.0))/3,2*jR[i]/(info->hxf/2.0+info->hxm/2.0));
  } else if (n == fm) {
    for (i=0; i<info->m; i++) lmt[i] = MinMod3(2*jL[i]/(info->hxf/2.0+info->hxm/2.0),(jL[i]/(info->hxf/2.0+info->hxm/2.0)+2*jR[i]/info->hxm)/3,2*jR[i]/info->hxm);
  } else {
    for (i=0; i<info->m; i++) lmt[i] = MinMod3(2*jL[i],(jL[i]+2*jR[i])/3,2*jR[i])/info->hxs;
  }
}

PetscErrorCode RiemannListAdd(PetscFunctionList *flist,const char *name,RiemannFunction rsolve)
{
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  ierr = PetscFunctionListAdd(flist,name,rsolve);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode RiemannListFind(PetscFunctionList flist,const char *name,RiemannFunction *rsolve)
{
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  ierr = PetscFunctionListFind(flist,name,rsolve);CHKERRQ(ierr);
  PetscAssertFalse(!*rsolve,PETSC_COMM_SELF,PETSC_ERR_ARG_UNKNOWN_TYPE,"Riemann solver \"%s\" could not be found",name);
  PetscFunctionReturn(0);
}

PetscErrorCode ReconstructListAdd(PetscFunctionList *flist,const char *name,ReconstructFunction r)
{
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  ierr = PetscFunctionListAdd(flist,name,r);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode ReconstructListFind(PetscFunctionList flist,const char *name,ReconstructFunction *r)
{
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  ierr = PetscFunctionListFind(flist,name,r);CHKERRQ(ierr);
  PetscAssertFalse(!*r,PETSC_COMM_SELF,PETSC_ERR_ARG_UNKNOWN_TYPE,"Reconstruction \"%s\" could not be found",name);
  PetscFunctionReturn(0);
}

PetscErrorCode RiemannListAdd_2WaySplit(PetscFunctionList *flist,const char *name,RiemannFunction_2WaySplit rsolve)
{
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  ierr = PetscFunctionListAdd(flist,name,rsolve);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode RiemannListFind_2WaySplit(PetscFunctionList flist,const char *name,RiemannFunction_2WaySplit *rsolve)
{
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  ierr = PetscFunctionListFind(flist,name,rsolve);CHKERRQ(ierr);
  PetscAssertFalse(!*rsolve,PETSC_COMM_SELF,PETSC_ERR_ARG_UNKNOWN_TYPE,"Riemann solver \"%s\" could not be found",name);
  PetscFunctionReturn(0);
}

PetscErrorCode ReconstructListAdd_2WaySplit(PetscFunctionList *flist,const char *name,ReconstructFunction_2WaySplit r)
{
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  ierr = PetscFunctionListAdd(flist,name,r);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode ReconstructListFind_2WaySplit(PetscFunctionList flist,const char *name,ReconstructFunction_2WaySplit *r)
{
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  ierr = PetscFunctionListFind(flist,name,r);CHKERRQ(ierr);
  PetscAssertFalse(!*r,PETSC_COMM_SELF,PETSC_ERR_ARG_UNKNOWN_TYPE,"Reconstruction \"%s\" could not be found",name);
  PetscFunctionReturn(0);
}

/* --------------------------------- Physics ------- */
PetscErrorCode PhysicsDestroy_SimpleFree(void *vctx)
{
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  ierr = PetscFree(vctx);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* --------------------------------- Finite Volume Solver --------------- */
PetscErrorCode FVRHSFunction(TS ts,PetscReal time,Vec X,Vec F,void *vctx)
{
  FVCtx          *ctx = (FVCtx*)vctx;
  PetscErrorCode ierr;
  PetscInt       i,j,k,Mx,dof,xs,xm;
  PetscReal      hx,cfl_idt = 0;
  PetscScalar    *x,*f,*slope;
  Vec            Xloc;
  DM             da;

  PetscFunctionBeginUser;
  ierr = TSGetDM(ts,&da);CHKERRQ(ierr);
  ierr = DMGetLocalVector(da,&Xloc);CHKERRQ(ierr);                          /* Xloc contains ghost points */
  ierr = DMDAGetInfo(da,0, &Mx,0,0, 0,0,0, &dof,0,0,0,0,0);CHKERRQ(ierr);   /* Mx is the number of center points */
  hx   = (ctx->xmax-ctx->xmin)/Mx;
  ierr = DMGlobalToLocalBegin(da,X,INSERT_VALUES,Xloc);CHKERRQ(ierr);       /* X is solution vector which does not contain ghost points */
  ierr = DMGlobalToLocalEnd(da,X,INSERT_VALUES,Xloc);CHKERRQ(ierr);
  ierr = VecZeroEntries(F);CHKERRQ(ierr);                                   /* F is the right hand side function corresponds to center points */
  ierr = DMDAVecGetArray(da,Xloc,&x);CHKERRQ(ierr);
  ierr = DMDAVecGetArray(da,F,&f);CHKERRQ(ierr);
  ierr = DMDAGetArray(da,PETSC_TRUE,&slope);CHKERRQ(ierr);                  /* contains ghost points */
  ierr = DMDAGetCorners(da,&xs,0,0,&xm,0,0);CHKERRQ(ierr);

  if (ctx->bctype == FVBC_OUTFLOW) {
    for (i=xs-2; i<0; i++) {
      for (j=0; j<dof; j++) x[i*dof+j] = x[j];
    }
    for (i=Mx; i<xs+xm+2; i++) {
      for (j=0; j<dof; j++) x[i*dof+j] = x[(xs+xm-1)*dof+j];
    }
  }

  for (i=xs-1; i<xs+xm+1; i++) {
    struct _LimitInfo info;
    PetscScalar       *cjmpL,*cjmpR;
    /* Determine the right eigenvectors R, where A = R \Lambda R^{-1} */
    ierr = (*ctx->physics.characteristic)(ctx->physics.user,dof,&x[i*dof],ctx->R,ctx->Rinv,ctx->speeds,ctx->xmin+hx*i);CHKERRQ(ierr);
    /* Evaluate jumps across interfaces (i-1, i) and (i, i+1), put in characteristic basis */
    ierr  = PetscArrayzero(ctx->cjmpLR,2*dof);CHKERRQ(ierr);
    cjmpL = &ctx->cjmpLR[0];
    cjmpR = &ctx->cjmpLR[dof];
    for (j=0; j<dof; j++) {
      PetscScalar jmpL,jmpR;
      jmpL = x[(i+0)*dof+j]-x[(i-1)*dof+j];
      jmpR = x[(i+1)*dof+j]-x[(i+0)*dof+j];
      for (k=0; k<dof; k++) {
        cjmpL[k] += ctx->Rinv[k+j*dof]*jmpL;
        cjmpR[k] += ctx->Rinv[k+j*dof]*jmpR;
      }
    }
    /* Apply limiter to the left and right characteristic jumps */
    info.m  = dof;
    info.hx = hx;
    (*ctx->limit)(&info,cjmpL,cjmpR,ctx->cslope);
    for (j=0; j<dof; j++) ctx->cslope[j] /= hx; /* rescale to a slope */
    for (j=0; j<dof; j++) {
      PetscScalar tmp = 0;
      for (k=0; k<dof; k++) tmp += ctx->R[j+k*dof]*ctx->cslope[k];
      slope[i*dof+j] = tmp;
    }
  }

  for (i=xs; i<xs+xm+1; i++) {
    PetscReal   maxspeed;
    PetscScalar *uL,*uR;
    uL = &ctx->uLR[0];
    uR = &ctx->uLR[dof];
    for (j=0; j<dof; j++) {
      uL[j] = x[(i-1)*dof+j]+slope[(i-1)*dof+j]*hx/2;
      uR[j] = x[(i-0)*dof+j]-slope[(i-0)*dof+j]*hx/2;
    }
    ierr    = (*ctx->physics.riemann)(ctx->physics.user,dof,uL,uR,ctx->flux,&maxspeed,ctx->xmin+hx*i,ctx->xmin,ctx->xmax);CHKERRQ(ierr);
    cfl_idt = PetscMax(cfl_idt,PetscAbsScalar(maxspeed/hx)); /* Max allowable value of 1/Delta t */
    if (i > xs) {
      for (j=0; j<dof; j++) f[(i-1)*dof+j] -= ctx->flux[j]/hx;
    }
    if (i < xs+xm) {
      for (j=0; j<dof; j++) f[i*dof+j] += ctx->flux[j]/hx;
    }
  }
  ierr = DMDAVecRestoreArray(da,Xloc,&x);CHKERRQ(ierr);
  ierr = DMDAVecRestoreArray(da,F,&f);CHKERRQ(ierr);
  ierr = DMDARestoreArray(da,PETSC_TRUE,&slope);CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(da,&Xloc);CHKERRQ(ierr);
  ierr = MPI_Allreduce(&cfl_idt,&ctx->cfl_idt,1,MPIU_REAL,MPIU_MAX,PetscObjectComm((PetscObject)da));CHKERRMPI(ierr);
  if (0) {
    /* We need to a way to inform the TS of a CFL constraint, this is a debugging fragment */
    PetscReal dt,tnow;
    ierr = TSGetTimeStep(ts,&dt);CHKERRQ(ierr);
    ierr = TSGetTime(ts,&tnow);CHKERRQ(ierr);
    if (dt > 0.5/ctx->cfl_idt) {
      if (1) {
        ierr = PetscPrintf(ctx->comm,"Stability constraint exceeded at t=%g, dt %g > %g\n",(double)tnow,(double)dt,(double)(0.5/ctx->cfl_idt));CHKERRQ(ierr);
      } else SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Stability constraint exceeded, %g > %g",(double)dt,(double)(ctx->cfl/ctx->cfl_idt));
    }
  }
  PetscFunctionReturn(0);
}

PetscErrorCode FVSample(FVCtx *ctx,DM da,PetscReal time,Vec U)
{
  PetscErrorCode ierr;
  PetscScalar    *u,*uj;
  PetscInt       i,j,k,dof,xs,xm,Mx;

  PetscFunctionBeginUser;
  PetscAssertFalse(!ctx->physics.sample,PETSC_COMM_SELF,PETSC_ERR_SUP,"Physics has not provided a sampling function");
  ierr = DMDAGetInfo(da,0, &Mx,0,0, 0,0,0, &dof,0,0,0,0,0);CHKERRQ(ierr);
  ierr = DMDAGetCorners(da,&xs,0,0,&xm,0,0);CHKERRQ(ierr);
  ierr = DMDAVecGetArray(da,U,&u);CHKERRQ(ierr);
  ierr = PetscMalloc1(dof,&uj);CHKERRQ(ierr);
  for (i=xs; i<xs+xm; i++) {
    const PetscReal h = (ctx->xmax-ctx->xmin)/Mx,xi = ctx->xmin+h/2+i*h;
    const PetscInt  N = 200;
    /* Integrate over cell i using trapezoid rule with N points. */
    for (k=0; k<dof; k++) u[i*dof+k] = 0;
    for (j=0; j<N+1; j++) {
      PetscScalar xj = xi+h*(j-N/2)/(PetscReal)N;
      ierr = (*ctx->physics.sample)(ctx->physics.user,ctx->initial,ctx->bctype,ctx->xmin,ctx->xmax,time,xj,uj);CHKERRQ(ierr);
      for (k=0; k<dof; k++) u[i*dof+k] += ((j==0 || j==N) ? 0.5 : 1.0)*uj[k]/N;
    }
  }
  ierr = DMDAVecRestoreArray(da,U,&u);CHKERRQ(ierr);
  ierr = PetscFree(uj);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode SolutionStatsView(DM da,Vec X,PetscViewer viewer)
{
  PetscErrorCode    ierr;
  PetscReal         xmin,xmax;
  PetscScalar       sum,tvsum,tvgsum;
  const PetscScalar *x;
  PetscInt          imin,imax,Mx,i,j,xs,xm,dof;
  Vec               Xloc;
  PetscBool         iascii;

  PetscFunctionBeginUser;
  ierr = PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&iascii);CHKERRQ(ierr);
  if (iascii) {
    /* PETSc lacks a function to compute total variation norm (difficult in multiple dimensions), we do it here */
    ierr  = DMGetLocalVector(da,&Xloc);CHKERRQ(ierr);
    ierr  = DMGlobalToLocalBegin(da,X,INSERT_VALUES,Xloc);CHKERRQ(ierr);
    ierr  = DMGlobalToLocalEnd(da,X,INSERT_VALUES,Xloc);CHKERRQ(ierr);
    ierr  = DMDAVecGetArrayRead(da,Xloc,(void*)&x);CHKERRQ(ierr);
    ierr  = DMDAGetCorners(da,&xs,0,0,&xm,0,0);CHKERRQ(ierr);
    ierr  = DMDAGetInfo(da,0, &Mx,0,0, 0,0,0, &dof,0,0,0,0,0);CHKERRQ(ierr);
    tvsum = 0;
    for (i=xs; i<xs+xm; i++) {
      for (j=0; j<dof; j++) tvsum += PetscAbsScalar(x[i*dof+j]-x[(i-1)*dof+j]);
    }
    ierr = MPI_Allreduce(&tvsum,&tvgsum,1,MPIU_REAL,MPIU_SUM,PetscObjectComm((PetscObject)da));CHKERRMPI(ierr);
    ierr = DMDAVecRestoreArrayRead(da,Xloc,(void*)&x);CHKERRQ(ierr);
    ierr = DMRestoreLocalVector(da,&Xloc);CHKERRQ(ierr);
    ierr = VecMin(X,&imin,&xmin);CHKERRQ(ierr);
    ierr = VecMax(X,&imax,&xmax);CHKERRQ(ierr);
    ierr = VecSum(X,&sum);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"Solution range [%8.5f,%8.5f] with minimum at %D, mean %8.5f, ||x||_TV %8.5f\n",(double)xmin,(double)xmax,imin,(double)(sum/Mx),(double)(tvgsum/Mx));CHKERRQ(ierr);
  } else SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"Viewer type not supported");
  PetscFunctionReturn(0);
}
