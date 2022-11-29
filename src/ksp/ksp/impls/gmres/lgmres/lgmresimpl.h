/* A. Baker */
/*
   Private data structure used by the LGMRES method.
*/

#ifndef PETSC_LGMRESIMPL_H
#define PETSC_LGMRESIMPL_H

#define KSPGMRES_NO_MACROS
#include <../src/ksp/ksp/impls/gmres/gmresimpl.h>

typedef struct {
  KSPGMRESHEADER

  /* LGMRES_MOD - make these for the z vectors - new storage for lgmres */
  Vec  *augvecs;           /* holds the error approximation vectors for lgmres. */
  Vec **augvecs_user_work; /* same purpose as user_work above, but this one is
                                         for our error approx vectors */
  /* currently only augvecs_user_work[0] is used, not sure if this will be */
  /* extended in the future to use more, or if this is a design bug */
  PetscInt     aug_vv_allocated;   /* aug_vv_allocated is the number of allocated lgmres
                                          augmentation vectors */
  PetscInt     aug_vecs_allocated; /* aug_vecs_allocated is the total number of augmentation vecs
                                          available - used to simplify the dynamic
                                       allocation of vectors */
  PetscScalar *hwork;              /* work array to hold Hessenberg product */

  PetscInt augwork_alloc; /*size of chunk allocated for augmentation vectors */

  PetscInt aug_dim; /* max number of augmented directions to add */

  PetscInt aug_ct; /* number of aug. vectors available */

  PetscInt *aug_order; /*keeps track of order to use aug. vectors*/

  PetscBool approx_constant; /* = 1 then the approx space at each restart will
                                  be  size max_k .  Therefore, more than (max_k - aug_dim)
                                  krylov vectors may be used if less than aug_dim error
                                  approximations are available (in the first few restarts,
                                  for example) to keep the space a constant size. */

  PetscInt matvecs; /*keep track of matvecs */
} KSP_LGMRES;

#define HH(a, b) (lgmres->hh_origin + (b) * (lgmres->max_k + 2) + (a))
/* HH will be size (max_k+2)*(max_k+1)  -  think of HH as
   being stored columnwise (inc. zeros) for access purposes. */
#define HES(a, b) (lgmres->hes_origin + (b) * (lgmres->max_k + 1) + (a))
/* HES will be size (max_k + 1) * (max_k + 1) -
   again, think of HES as being stored columnwise */
#define CC(a)  (lgmres->cc_origin + (a)) /* CC will be length (max_k+1) - cosines */
#define SS(a)  (lgmres->ss_origin + (a)) /* SS will be length (max_k+1) - sines */
#define GRS(a) (lgmres->rs_origin + (a)) /* GRS will be length (max_k+2) - rt side */

/* vector names */
#define VEC_OFFSET     2
#define VEC_TEMP       lgmres->vecs[0] /* work space */
#define VEC_TEMP_MATOP lgmres->vecs[1] /* work space */
#define VEC_VV(i) \
  lgmres->vecs[VEC_OFFSET + i] /* use to access
                                                        othog basis vectors */
/*LGMRES_MOD */
#define AUG_OFFSET   1
#define AUGVEC(i)    lgmres->augvecs[AUG_OFFSET + i]                   /*error approx vectors */
#define AUG_ORDER(i) lgmres->aug_order[i]                              /*order in which to augment */
#define A_AUGVEC(i)  lgmres->augvecs[AUG_OFFSET + i + lgmres->aug_dim] /*A times error vector */
#define AUG_TEMP     lgmres->augvecs[0]                                /* work vector */

#endif // PETSC_LGMRESIMPL_H
