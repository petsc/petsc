/*
   Private include for tcqmr package
 */

#include <petsc/private/kspimpl.h>

/* vector names */
#define VEC_OFFSET 0
#define VEC_SOLN   ksp->vec_sol
#define VEC_RHS    ksp->vec_rhs
#define b          VEC_RHS
#define x          VEC_SOLN
#define r          ksp->work[VEC_OFFSET + 1]
#define um1        ksp->work[VEC_OFFSET + 2]
#define u          ksp->work[VEC_OFFSET + 3]
#define vm1        ksp->work[VEC_OFFSET + 4]
#define v          ksp->work[VEC_OFFSET + 5]
#define v0         ksp->work[VEC_OFFSET + 6]
#define pvec1      ksp->work[VEC_OFFSET + 7]
#define pvec2      ksp->work[VEC_OFFSET + 8]
#define p          ksp->work[VEC_OFFSET + 9]
#define y          ksp->work[VEC_OFFSET + 10]
#define z          ksp->work[VEC_OFFSET + 11]
#define utmp       ksp->work[VEC_OFFSET + 12]
#define up1        ksp->work[VEC_OFFSET + 13]
#define vp1        ksp->work[VEC_OFFSET + 14]
#define pvec       ksp->work[VEC_OFFSET + 15]
#define vtmp       ksp->work[VEC_OFFSET + 16]
#define TCQMR_VECS 17
