/*
   Provides an interface to the LLNL package hypre
*/
#include <petsc/private/petscimpl.h>
#include <petscpc.h>

/*@
  PCHYPRESetDiscreteGradient - Set the discrete gradient matrix for `PCHYPRE` type of AMS or ADS

  Collective

  Input Parameters:
+ pc - the preconditioning context
- G  - the discrete gradient

  Level: intermediate

  Notes:
  `G` should have as many rows as the number of edges and as many columns as the number of vertices in the mesh

  Each row of `G` has 2 nonzeros, with column indexes being the global indexes of edge's endpoints: matrix entries are +1 and -1 depending on edge orientation

  Developer Note:
  This automatically converts the matrix to `MATHYPRE` if it is not already of that type

.seealso: [](ch_ksp), `PCHYPRE`, `PCHYPRESetDiscreteCurl()`
@*/
PetscErrorCode PCHYPRESetDiscreteGradient(PC pc, Mat G)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc, PC_CLASSID, 1);
  PetscValidHeaderSpecific(G, MAT_CLASSID, 2);
  PetscCheckSameComm(pc, 1, G, 2);
  PetscTryMethod(pc, "PCHYPRESetDiscreteGradient_C", (PC, Mat), (pc, G));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PCHYPRESetDiscreteCurl - Set the discrete curl matrix for `PCHYPRE` type of ADS

  Collective

  Input Parameters:
+ pc - the preconditioning context
- C  - the discrete curl

  Level: intermediate

  Notes:
  `C` should have as many rows as the number of faces and as many columns as the number of edges in the mesh

  Each row of `C` has as many nonzeros as the number of edges of a face, with column indexes being the global indexes of the corresponding edge.
  Matrix entries are +1 and -1 depending on edge orientation with respect to the face orientation

  Developer Notes:
  This automatically converts the matrix to `MATHYPRE` if it is not already of that type

  If this is only for  `PCHYPRE` type of ADS it should be called `PCHYPREADSSetDiscreteCurl()`

.seealso: [](ch_ksp), `PCHYPRE`, `PCHYPRESetDiscreteGradient()`
@*/
PetscErrorCode PCHYPRESetDiscreteCurl(PC pc, Mat C)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc, PC_CLASSID, 1);
  PetscValidHeaderSpecific(C, MAT_CLASSID, 2);
  PetscCheckSameComm(pc, 1, C, 2);
  PetscTryMethod(pc, "PCHYPRESetDiscreteCurl_C", (PC, Mat), (pc, C));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PCHYPRESetInterpolations - Set the interpolation matrices for `PCHYPRE` type of AMS or ADS

  Collective

  Input Parameters:
+ pc        - the preconditioning context
. dim       - the dimension of the problem, only used in AMS
. RT_PiFull - Raviart-Thomas interpolation matrix
. RT_Pi     - x/y/z component of Raviart-Thomas interpolation matrix
. ND_PiFull - Nedelec interpolation matrix
- ND_Pi     - x/y/z component of Nedelec interpolation matrix

  Level: intermediate

  Notes:
  For AMS, only Nedelec interpolation matrices are needed, the Raviart-Thomas interpolation matrices can be set to `NULL`.

  For ADS, both type of interpolation matrices are needed.

  Developer Note:
  This automatically converts the matrix to `MATHYPRE` if it is not already of that type

.seealso: [](ch_ksp), `PCHYPRE`
@*/
PetscErrorCode PCHYPRESetInterpolations(PC pc, PetscInt dim, Mat RT_PiFull, Mat RT_Pi[], Mat ND_PiFull, Mat ND_Pi[])
{
  PetscInt i;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc, PC_CLASSID, 1);
  if (RT_PiFull) {
    PetscValidHeaderSpecific(RT_PiFull, MAT_CLASSID, 3);
    PetscCheckSameComm(pc, 1, RT_PiFull, 3);
  }
  if (RT_Pi) {
    PetscAssertPointer(RT_Pi, 4);
    for (i = 0; i < dim; ++i) {
      if (RT_Pi[i]) {
        PetscValidHeaderSpecific(RT_Pi[i], MAT_CLASSID, 4);
        PetscCheckSameComm(pc, 1, RT_Pi[i], 4);
      }
    }
  }
  if (ND_PiFull) {
    PetscValidHeaderSpecific(ND_PiFull, MAT_CLASSID, 5);
    PetscCheckSameComm(pc, 1, ND_PiFull, 5);
  }
  if (ND_Pi) {
    PetscAssertPointer(ND_Pi, 6);
    for (i = 0; i < dim; ++i) {
      if (ND_Pi[i]) {
        PetscValidHeaderSpecific(ND_Pi[i], MAT_CLASSID, 6);
        PetscCheckSameComm(pc, 1, ND_Pi[i], 6);
      }
    }
  }
  PetscTryMethod(pc, "PCHYPRESetInterpolations_C", (PC, PetscInt, Mat, Mat[], Mat, Mat[]), (pc, dim, RT_PiFull, RT_Pi, ND_PiFull, ND_Pi));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PCHYPRESetAlphaPoissonMatrix - Set the vector Poisson matrix for `PCHYPRE` of type AMS

  Collective

  Input Parameters:
+ pc - the preconditioning context
- A  - the matrix

  Level: intermediate

  Note:
  `A` should be obtained by discretizing the vector valued Poisson problem with linear finite elements

  Developer Notes:
  This automatically converts the matrix to `MATHYPRE` if it is not already of that type

  If this is only for  `PCHYPRE` type of AMS it should be called `PCHYPREAMSSetAlphaPoissonMatrix()`

.seealso: [](ch_ksp), `PCHYPRE`, `PCHYPRESetDiscreteGradient()`, `PCHYPRESetDiscreteCurl()`, `PCHYPRESetBetaPoissonMatrix()`
@*/
PetscErrorCode PCHYPRESetAlphaPoissonMatrix(PC pc, Mat A)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc, PC_CLASSID, 1);
  PetscValidHeaderSpecific(A, MAT_CLASSID, 2);
  PetscCheckSameComm(pc, 1, A, 2);
  PetscTryMethod(pc, "PCHYPRESetPoissonMatrix_C", (PC, Mat, PetscBool), (pc, A, PETSC_TRUE));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PCHYPRESetBetaPoissonMatrix - Set the Poisson matrix for `PCHYPRE` of type AMS

  Collective

  Input Parameters:
+ pc - the preconditioning context
- A  - the matrix, or `NULL` to turn it off

  Level: intermediate

  Note:
  `A` should be obtained by discretizing the Poisson problem with linear finite elements.

  Developer Notes:
  This automatically converts the matrix to `MATHYPRE` if it is not already of that type

  If this is only for  `PCHYPRE` type of AMS it should be called `PCHYPREAMSPCHYPRESetBetaPoissonMatrix()`

.seealso: [](ch_ksp), `PCHYPRE`, `PCHYPRESetDiscreteGradient()`, `PCHYPRESetDiscreteCurl()`, `PCHYPRESetAlphaPoissonMatrix()`
@*/
PetscErrorCode PCHYPRESetBetaPoissonMatrix(PC pc, Mat A)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc, PC_CLASSID, 1);
  if (A) {
    PetscValidHeaderSpecific(A, MAT_CLASSID, 2);
    PetscCheckSameComm(pc, 1, A, 2);
  }
  PetscTryMethod(pc, "PCHYPRESetPoissonMatrix_C", (PC, Mat, PetscBool), (pc, A, PETSC_FALSE));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PCHYPRESetEdgeConstantVectors - Set the representation of the constant vector fields in the edge element basis for `PCHYPRE` of type AMS

  Collective

  Input Parameters:
+ pc  - the preconditioning context
. ozz - vector representing (1,0,0) (or (1,0) in 2D)
. zoz - vector representing (0,1,0) (or (0,1) in 2D)
- zzo - vector representing (0,0,1) (use NULL in 2D)

  Level: intermediate

  Developer Note:
  If this is only for  `PCHYPRE` type of AMS it should be called `PCHYPREAMSSetEdgeConstantVectors()`

.seealso: [](ch_ksp), `PCHYPRE`, `PCHYPRESetDiscreteGradient()`, `PCHYPRESetDiscreteCurl()`, `PCHYPRESetAlphaPoissonMatrix()`
@*/
PetscErrorCode PCHYPRESetEdgeConstantVectors(PC pc, Vec ozz, Vec zoz, Vec zzo)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc, PC_CLASSID, 1);
  PetscValidHeaderSpecific(ozz, VEC_CLASSID, 2);
  PetscValidHeaderSpecific(zoz, VEC_CLASSID, 3);
  if (zzo) PetscValidHeaderSpecific(zzo, VEC_CLASSID, 4);
  PetscCheckSameComm(pc, 1, ozz, 2);
  PetscCheckSameComm(pc, 1, zoz, 3);
  if (zzo) PetscCheckSameComm(pc, 1, zzo, 4);
  PetscTryMethod(pc, "PCHYPRESetEdgeConstantVectors_C", (PC, Vec, Vec, Vec), (pc, ozz, zoz, zzo));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PCHYPREAMSSetInteriorNodes - Set the list of interior nodes to a zero-conductivity region for `PCHYPRE` of type AMS

  Collective

  Input Parameters:
+ pc       - the preconditioning context
- interior - vector. node is interior if its entry in the array is 1.0.

  Level: intermediate

  Note:
  This calls `HYPRE_AMSSetInteriorNodes()`

.seealso: [](ch_ksp), `PCHYPRE`, `PCHYPRESetDiscreteGradient()`, `PCHYPRESetDiscreteCurl()`, `PCHYPRESetAlphaPoissonMatrix()`
@*/
PetscErrorCode PCHYPREAMSSetInteriorNodes(PC pc, Vec interior)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc, PC_CLASSID, 1);
  PetscValidHeaderSpecific(interior, VEC_CLASSID, 2);
  PetscCheckSameComm(pc, 1, interior, 2);
  PetscTryMethod(pc, "PCHYPREAMSSetInteriorNodes_C", (PC, Vec), (pc, interior));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PCHYPRESetType - Sets which hypre preconditioner you wish to use

  Input Parameters:
+ pc   - the preconditioner context
- name - either euclid, ilu, pilut, parasails, boomeramg, ams, or ads

  Options Database Key:
. pc_hypre_type - One of euclid, ilu, pilut, parasails, boomeramg, ams, or ads

  Level: intermediate

.seealso: [](ch_ksp), `PCCreate()`, `PCSetType()`, `PCType`, `PC`, `PCHYPRE`
@*/
PetscErrorCode PCHYPRESetType(PC pc, const char name[])
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc, PC_CLASSID, 1);
  PetscAssertPointer(name, 2);
  PetscTryMethod(pc, "PCHYPRESetType_C", (PC, const char[]), (pc, name));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  PCHYPREGetCFMarkers - Gets CF marker arrays for all levels (except the finest level)

  Logically Collective

  Input Parameter:
. pc - the preconditioner context

  Output Parameters:
+ n_per_level - the number of nodes per level (size of `num_levels`)
- CFMarkers   - the Coarse/Fine Boolean arrays (size of `num_levels` - 1)

  Level: advanced

  Note:
  Caller is responsible for memory management of `n_per_level` and `CFMarkers` pointers. That is they should free them with `PetscFree()` when no longer needed.

.seealso: [](ch_ksp), `PC`, `PCMG`, `PCMGGetRestriction()`, `PCMGSetInterpolation()`, `PCMGGetRScale()`, `PCMGGetInterpolation()`, `PCGetInterpolations()`
@*/
PetscErrorCode PCHYPREGetCFMarkers(PC pc, PetscInt *n_per_level[], PetscBT *CFMarkers[])
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc, PC_CLASSID, 1);
  PetscAssertPointer(n_per_level, 2);
  PetscAssertPointer(CFMarkers, 3);
  PetscUseMethod(pc, "PCHYPREGetCFMarkers_C", (PC, PetscInt *[], PetscBT *[]), (pc, n_per_level, CFMarkers));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PCHYPREGetType - Gets which hypre preconditioner you are using

  Input Parameter:
. pc - the preconditioner context

  Output Parameter:
. name - either euclid, ilu, pilut, parasails, boomeramg, ams, or ads

  Level: intermediate

.seealso: [](ch_ksp), `PCCreate()`, `PCHYPRESetType()`, `PCType`, `PC`, `PCHYPRE`
@*/
PetscErrorCode PCHYPREGetType(PC pc, const char *name[])
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc, PC_CLASSID, 1);
  PetscAssertPointer(name, 2);
  PetscTryMethod(pc, "PCHYPREGetType_C", (PC, const char *[]), (pc, name));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PCMGGalerkinSetMatProductAlgorithm - Set type of sparse matrix-matrix product for hypre's BoomerAMG to use on GPUs

  Logically Collective

  Input Parameters:
+ pc   - the hypre context
- name - one of 'cusparse', 'hypre'

  Options Database Key:
. -pc_mg_galerkin_mat_product_algorithm <cusparse,hypre> - Type of sparse matrix-matrix product to use in hypre

  Level: intermediate

  Developer Note:
  How the name starts with `PCMG`, should it not be `PCHYPREBoomerAMG`?

.seealso: [](ch_ksp), `PCHYPRE`, `PCMGGalerkinGetMatProductAlgorithm()`
@*/
PetscErrorCode PCMGGalerkinSetMatProductAlgorithm(PC pc, const char name[])
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc, PC_CLASSID, 1);
  PetscTryMethod(pc, "PCMGGalerkinSetMatProductAlgorithm_C", (PC, const char[]), (pc, name));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PCMGGalerkinGetMatProductAlgorithm - Get type of sparse matrix-matrix product for hypre's BoomerAMG to use on GPUs

  Not Collective

  Input Parameter:
. pc - the multigrid context

  Output Parameter:
. name - one of 'cusparse', 'hypre'

  Level: intermediate

.seealso: [](ch_ksp), `PCHYPRE`, `PCMGGalerkinSetMatProductAlgorithm()`
@*/
PetscErrorCode PCMGGalerkinGetMatProductAlgorithm(PC pc, const char *name[])
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc, PC_CLASSID, 1);
  PetscTryMethod(pc, "PCMGGalerkinGetMatProductAlgorithm_C", (PC, const char *[]), (pc, name));
  PetscFunctionReturn(PETSC_SUCCESS);
}
