
#if !defined(__SCHUR_H)
#define __SCHUR_H

#include "private/pcimpl.h"

typedef struct {
  
  PetscTruth     seq;    /* flag for the uniprocessor case*/

  PetscInt       blocks, /* maximum subpartitions in local subdomain*/
                 ccsize, /* maximum subpartition size in local subdomain */
                 layers; /* number of layers for local strip problem */

  PetscTruth     sp_stats; /* sub-partitioning statistics */
  PetscInt       sp_count,
                 sp_minsz,
                 sp_maxsz,
                 sp_nseps;

  /* local interior problem */
  PetscInt       n,      /* number of nodes in this subdomain           */
                 n_I,    /* number of interior nodes in this subdomain  */
                 n_B;    /* number of interface nodes in this subdomain */
  IS             is_I,   /* local (sequential) index sets for interior nodes  */
                 is_B;   /* local (sequential) index sets for interface nodes */
  Vec            vec1_I, /* local (sequential) interior work vectors  */
                 vec2_I,
                 vec3_I, 
                 vec1_B, /* local (sequential) interface work vectors */
                 vec2_B;
  VecScatter     G_to_I, /* scattering context from all global to local interior nodes  */
                 G_to_B; /* scattering context from all global to local interface nodes */
  Mat            A,      /* local (sequential) subdomain matrix      */
                 A_II,   /* local (sequential) subdomain submatrices */
                 A_IB,
                 A_BI,
                 A_BB;
  Vec            D;      /* local diagonal scaling "matrix" (stored as a vector) */
  KSP            ksp_I;  /* local linear solver for interior problem  */
  
  /* global strip problem */
  PetscInt       n_L;    /* local size of global strip problem  */
  PetscInt       N_L;    /* global size of global strip problem  */
  IS             is_L;   /* local (sequential) index sets for strip nodes */
  Vec            vec1_L, /* global strip work vectors  */
                 vec2_L; 
  Mat            A_LL;   /* local (sequential) strip submatrix */
  Mat            mat_L;  /* global strip operator */
  KSP            ksp_L;  /* global linear solver for strip problem */


  /* global interface problem */
  PetscInt       n_S,    /* local size of global interface problem  */
                 N_S;    /* global size of global interface problem */
  Vec            vec1_S, /* global interface work vectors  */
                 vec2_S; 
  VecScatter     S_to_B; /* scattering context from global interface to local interface nodes */
  VecScatter     S_to_L;
  Mat            mat_S;  /* global Schur complement operator */
  KSP            ksp_S;  /* global linear solver for interface problem */
  PetscTruth     outer_ksp_tols;

  KSP            subksp[3];
  
} PC_Schur;


#endif /* __SCHUR_H */
