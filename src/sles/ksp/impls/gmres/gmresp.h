/* $Id: gmresp.h,v 1.1 1994/03/18 00:24:26 gropp Exp $

   Private include for gmres package
 */

#define HH(a,b)  (gmresP->hh_origin + (b)*(gmresP->max_k+2)+(a))
#define HES(a,b) (gmresP->hes_origin + (b)*(gmresP->max_k+1)+(a))
#define CC(a)    (gmresP->cc_origin + (a))
#define SS(a)    (gmresP->ss_origin + (a))
#define RS(a)    (gmresP->rs_origin + (a))

/* vector names */
#define VEC_OFFSET 3
#define VEC_SOLN itP->vec_sol
#define VEC_RHS  itP->vec_rhs
#define VEC_TEMP gmresP->vecs[0]
#define VEC_TEMP_MATOP gmresP->vecs[1]
#define VEC_BINVF gmresP->vecs[2]
#define VEC_VV(i) gmresP->vecs[VEC_OFFSET+i]
