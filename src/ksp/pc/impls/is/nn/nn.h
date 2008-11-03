
#if !defined(__pcnn_h)
#define __pcnn_h

#include "../src/ksp/pc/impls/is/pcis.h"

/* 
   Private context (data structure) for the NN preconditioner.  
*/
typedef struct {
  /* First MUST come the folowing line, for the stuff that is common to FETI and Neumann-Neumann. */
  PC_IS         pcis;
  /* Then, everything else. */
  Mat           coarse_mat;
  Vec           coarse_x;
  Vec           coarse_b;
  KSP          ksp_coarse;
  PetscScalar   **DZ_IN;          /* proc[k].DZ_IN[i][] = bit of vector to be received from processor i by proc. k  */
  PetscScalar   factor_coarse_rhs;
} PC_NN;

EXTERN PetscErrorCode PCNNCreateCoarseMatrix (PC);
EXTERN PetscErrorCode PCNNApplySchurToChunk(PC pc,PetscInt n,PetscInt* idx,PetscScalar *chunk,PetscScalar* array_N,Vec vec1_B,Vec vec2_B,Vec vec1_D,Vec vec2_D);
EXTERN PetscErrorCode PCNNApplyInterfacePreconditioner (PC pc,Vec r,Vec z,PetscScalar* work_N,Vec vec1_B,Vec vec2_B,
                                             Vec vec3_B,Vec vec1_D,Vec vec2_D,Vec vec1_N,Vec vec2_N);
EXTERN PetscErrorCode PCNNBalancing(PC pc,Vec r,Vec u,Vec z,Vec vec1_B,Vec vec2_B,Vec vec3_B,Vec vec1_D,Vec vec2_D,PetscScalar *work_N); 

#endif /* __pcnn_h */






