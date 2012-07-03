/*
  DMIGA, for NURBS finite elements used in isogeometric analysis.
*/
#if !defined(__PETSCDMIGA_H)
#define __PETSCDMIGA_H

/*S
  DMIGA - DM object that encapsulates Iso-Geometric Analysis over a Cartesian mesh, which is represented using a DMDA.

  Level: intermediate

  Concepts: grids, grid refinement

.seealso:  DM, DMIGACreate(), DMDA
S*/
#include <petscdm.h>
#include <petscdmda.h>

PETSC_EXTERN PetscErrorCode DMIGACreate(MPI_Comm,DM*);
PETSC_EXTERN PetscErrorCode DMIGAGetPolynomialOrder(DM,PetscInt*,PetscInt*,PetscInt*);
PETSC_EXTERN PetscErrorCode DMIGAGetNumQuadraturePoints(DM,PetscInt*,PetscInt*,PetscInt*);

PETSC_EXTERN PetscErrorCode DMIGASetFieldName(DM,PetscInt,const char[]);
PETSC_EXTERN PetscErrorCode DMIGAGetFieldName(DM,PetscInt,const char**);
PETSC_EXTERN PetscErrorCode DMIGAVecGetArray(DM,Vec,void *);
PETSC_EXTERN PetscErrorCode DMIGAVecRestoreArray(DM,Vec,void *);
PETSC_EXTERN PetscErrorCode DMIGAGetLocalInfo(DM,DMDALocalInfo*);

PETSC_EXTERN PetscErrorCode DMIGAInitializeUniform1d(DM dm,PetscBool IsRational,PetscInt NumDerivatives,PetscInt ndof,
                                               PetscInt px,PetscInt Nx,PetscInt Cx,PetscReal Ux0, PetscReal Uxf,PetscBool IsPeriodicX,PetscInt ngx);
PETSC_EXTERN PetscErrorCode DMIGAInitializeUniform2d(DM dm,PetscBool IsRational,PetscInt NumDerivatives,PetscInt ndof,
                                               PetscInt px,PetscInt Nx,PetscInt Cx,PetscReal Ux0, PetscReal Uxf,PetscBool IsPeriodicX,PetscInt ngx,
                                               PetscInt py,PetscInt Ny,PetscInt Cy,PetscReal Uy0, PetscReal Uyf,PetscBool IsPeriodicY,PetscInt ngy);
PETSC_EXTERN PetscErrorCode DMIGAInitializeSymmetricTaper2d(DM dm,PetscBool IsRational,PetscInt NumDerivatives,PetscInt ndof,
                                                      PetscInt px,PetscInt Nx,PetscInt Cx,PetscReal fx,
                                                      PetscReal Ux0, PetscReal Uxf,PetscBool IsPeriodicX,PetscInt ngx,
                                                      PetscInt py,PetscInt Ny,PetscInt Cy,PetscReal fy,
                                                      PetscReal Uy0, PetscReal Uyf,PetscBool IsPeriodicY,PetscInt ngy)
;
PETSC_EXTERN PetscErrorCode DMIGAInitializeUniform3d(DM dm,PetscBool IsRational,PetscInt NumDerivatives,PetscInt ndof,
                                               PetscInt px,PetscInt Nx,PetscInt Cx,PetscReal Ux0, PetscReal Uxf,PetscBool IsPeriodicX,PetscInt ngx,
                                               PetscInt py,PetscInt Ny,PetscInt Cy,PetscReal Uy0, PetscReal Uyf,PetscBool IsPeriodicY,PetscInt ngy,
                                               PetscInt pz,PetscInt Nz,PetscInt Cz,PetscReal Uz0, PetscReal Uzf,PetscBool IsPeriodicZ,PetscInt ngz);
PETSC_EXTERN PetscErrorCode DMIGAInitializeGeometry3d(DM dm,PetscInt ndof,PetscInt NumDerivatives,char *FunctionSpaceFile,char *GeomFile);
PETSC_EXTERN PetscErrorCode DMIGAKnotRefine2d(DM dm,PetscInt kx,PetscReal *Ux,PetscInt ky,PetscReal *Uy,DM iga_new);
PETSC_EXTERN PetscErrorCode DMIGAKnotRefine3d(DM dm,PetscInt kx,PetscReal *Ux,PetscInt ky,PetscReal *Uy,PetscInt kz,PetscReal *Uz,DM iga_new);
typedef struct {
  PetscReal *basis; /* (p+1)x(numD+1) */
  PetscReal gx,gw;
  PetscInt offset;
} GP;

typedef struct {
  int numD,p,numGP,numEl;
  int own_b, own_e;   /* beginning/end of elements I 'own' */
  int cont_b, cont_e; /* beginning/end of elements that contribute to dofs I own */
  GP  *data;
} BasisData1D;

typedef BasisData1D* BD;

PETSC_EXTERN PetscErrorCode DMIGAGetBasisData(DM,BD*,BD*,BD*);

PETSC_EXTERN PetscErrorCode BDCreate(BD *bd,PetscInt numD,PetscInt p,PetscInt numGP,PetscInt numEl);
PETSC_EXTERN PetscErrorCode BDDestroy(BD *bd);
PETSC_EXTERN PetscErrorCode BDGetBasis(BD bd, PetscInt iel, PetscInt igp, PetscInt ib, PetscInt ider, PetscReal *basis);
PETSC_EXTERN PetscErrorCode BDSetBasis(BD bd, PetscInt iel, PetscInt igp, PetscInt ib, PetscInt ider, PetscReal basis);
PETSC_EXTERN PetscErrorCode BDGetGaussPt(BD bd, PetscInt iel, PetscInt igp, PetscReal *gp);
PETSC_EXTERN PetscErrorCode BDSetGaussPt(BD bd, PetscInt iel, PetscInt igp, PetscReal gp);
PETSC_EXTERN PetscErrorCode BDGetGaussWt(BD bd, PetscInt iel, PetscInt igp, PetscReal *gw);
PETSC_EXTERN PetscErrorCode BDSetGaussWt(BD bd, PetscInt iel, PetscInt igp, PetscReal gw);
PETSC_EXTERN PetscErrorCode BDGetBasisOffset(BD bd, PetscInt iel, PetscInt *bOffset);
PETSC_EXTERN PetscErrorCode BDSetBasisOffset(BD bd, PetscInt iel, PetscInt bOffset);
PETSC_EXTERN PetscErrorCode BDSetElementOwnership(BD bd,PetscInt nel,PetscInt dof_b,PetscInt dof_e,PetscInt p);

#endif /* __PETSCDMIGA_H */
