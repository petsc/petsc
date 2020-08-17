#if !defined(PIPEFGMRES_H_)
#define PIPEFGMRES_H_

#define KSPGMRES_NO_MACROS
#include <../src/ksp/ksp/impls/gmres/gmresimpl.h>

typedef struct {
  KSPGMRESHEADER

   /* new storage for explicit storage of preconditioned basis vectors */
  Vec *prevecs;                  /* holds the preconditioned basis vectors for fgmres.
                                    We will allocate these at the same time as vecs
                                    above (and in the same "chunks". */
  Vec **prevecs_user_work;       /* same purpose as user_work above, but this one is
                                    for our preconditioned vectors */

  /* new storage for explicit storage of pipelining quantities */
  Vec *zvecs;
  Vec **zvecs_user_work;

  /* A shift parameter */
  PetscScalar shift;

  /* Work space to allow all reductions in a single call */
  Vec         *redux;

} KSP_PIPEFGMRES;

#define HH(a,b)  (pipefgmres->hh_origin + (b)*(pipefgmres->max_k+2)+(a))
/* HH will be size (max_k+2)*(max_k+1)  -  think of HH as
   being stored columnwise for access purposes. */
#define HES(a,b) (pipefgmres->hes_origin + (b)*(pipefgmres->max_k+1)+(a))
/* HES will be size (max_k + 1) * (max_k + 1) -
   again, think of HES as being stored columnwise */
#define CC(a)    (pipefgmres->cc_origin + (a)) /* CC will be length (max_k+1) - cosines */
#define SS(a)    (pipefgmres->ss_origin + (a)) /* SS will be length (max_k+1) - sines */
#define RS(a)    (pipefgmres->rs_origin + (a)) /* RS will be length (max_k+2) - rt side */

/* vector names */
#define VEC_OFFSET     4
#define VEC_TEMP       pipefgmres->vecs[0]               /* work space */
#define VEC_TEMP_MATOP pipefgmres->vecs[1]               /* work space */
#define VEC_Q          pipefgmres->vecs[2]               /* work space - Q pipelining var */
#define VEC_W          pipefgmres->vecs[3]               /* work space - W pipelining var */

#define VEC_VV(i)      pipefgmres->vecs[VEC_OFFSET+i]    /* use to access othog basis vectors
                                                            Note the offset, since we use
                                                            the first few as workspace */

#define PREVEC(i)      pipefgmres->prevecs[i]            /* use to access preconditioned basis */
#define ZVEC(i)        pipefgmres->zvecs[i]
#endif
