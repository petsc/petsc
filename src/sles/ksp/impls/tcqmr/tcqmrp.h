/* $Id: tcqmrp.h,v 1.2 1994/08/21 23:56:49 bsmith Exp $

   Private include for tcqmr package
 */

/* vector names */
#define VEC_OFFSET 0
#define VEC_SOLN itP->vec_sol
#define VEC_RHS  itP->vec_rhs
#define b       VEC_RHS
#define x  	VEC_SOLN
#define r   	itP->work[VEC_OFFSET+1]
#define um1 	itP->work[VEC_OFFSET+2]
#define u       itP->work[VEC_OFFSET+3]
#define vm1	itP->work[VEC_OFFSET+4]
#define v	itP->work[VEC_OFFSET+5]
#define v0	itP->work[VEC_OFFSET+6]
#define pvec1	itP->work[VEC_OFFSET+7]
#define pvec2   itP->work[VEC_OFFSET+8]
#define p	itP->work[VEC_OFFSET+9]
#define y	itP->work[VEC_OFFSET+10]
#define z	itP->work[VEC_OFFSET+11]
#define utmp	itP->work[VEC_OFFSET+12]
#define up1	itP->work[VEC_OFFSET+13]
#define vp1     itP->work[VEC_OFFSET+14]
#define pvec    itP->work[VEC_OFFSET+15]
#define vtmp    itP->work[VEC_OFFSET+16]
#define TCQMR_VECS 17


