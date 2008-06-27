
#if !defined(__SCHUR_H)
#define __SCHUR_H

#include "private/pcimpl.h"

typedef struct {
  
  PetscTruth     seq;    /* flag for the uniprocessor case*/

  PetscInt       blocks, /* maximum subpartitions in local subdomain*/
                 ccsize; /* maximum subpartition size in local subdomain */

  PetscTruth     sp_stats; /* sub-partitioning statistics */
  PetscInt       sp_count,
                 sp_minsz,
                 sp_maxsz,
                 sp_nseps;


  Mat            A, /* local (sequential) subdomain matrix (diagonal part) */
                 B; /* local (sequential) subdomain matrix (off-diagonal part) */
  Vec            lvec;
  VecScatter     Mvctx;
  Vec            vec_G;

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
                 vec2_B,
                 vec3_B;
  VecScatter     G_to_I, /* scattering context from all global to local interior nodes  */
                 G_to_B; /* scattering context from all global to local interface nodes */
  Mat            A_II,   /* local (sequential) subdomain submatrices */
                 A_IB,
                 A_BI,
                 A_BB;
  KSP            ksp_I;  /* local linear solver for interior problem  */
  
  /* global interface problem */
  PetscInt       n_S,    /* local size of global interface problem  */
                 N_S;    /* global size of global interface problem */
  Vec            vec1_S, /* global interface work vectors  */
                 vec2_S; 
  VecScatter     S_to_B; /* scattering context from global interface to local interface nodes */
  Mat            mat_S;  /* global Schur complement operator (shell matrix) */
  KSP            ksp_S;  /* global linear solver for interface problem */
  
  KSP            subksp[3];
  
} PC_Schur;


#endif /* __SCHUR_H */
