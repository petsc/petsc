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

extern PetscErrorCode DMIGACreate(MPI_Comm,DM*);
extern PetscErrorCode DMIGAGetPolynomialOrder(DM,PetscInt*,PetscInt*,PetscInt*);
extern PetscErrorCode DMIGAGetNumQuadraturePoints(DM,PetscInt*,PetscInt*,PetscInt*);

extern PetscErrorCode DMIGASetFieldName(DM,PetscInt,const char[]);
extern PetscErrorCode DMIGAGetFieldName(DM,PetscInt,const char**);
extern PetscErrorCode DMIGAVecGetArray(DM,Vec,void *);
extern PetscErrorCode DMIGAVecRestoreArray(DM,Vec,void *);
extern PetscErrorCode DMIGAGetLocalInfo(DM,DMDALocalInfo*);

extern PetscErrorCode DMIGAInitializeUniform1d(DM dm,PetscBool IsRational,PetscInt NumDerivatives,PetscInt ndof,
                                               PetscInt px,PetscInt Nx,PetscInt Cx,PetscReal Ux0, PetscReal Uxf,PetscBool IsPeriodicX,PetscInt ngx);
extern PetscErrorCode DMIGAInitializeUniform2d(DM dm,PetscBool IsRational,PetscInt NumDerivatives,PetscInt ndof,
                                               PetscInt px,PetscInt Nx,PetscInt Cx,PetscReal Ux0, PetscReal Uxf,PetscBool IsPeriodicX,PetscInt ngx,
                                               PetscInt py,PetscInt Ny,PetscInt Cy,PetscReal Uy0, PetscReal Uyf,PetscBool IsPeriodicY,PetscInt ngy);
extern PetscErrorCode DMIGAInitializeSymmetricTaper2d(DM dm,PetscBool IsRational,PetscInt NumDerivatives,PetscInt ndof,
                                                      PetscInt px,PetscInt Nx,PetscInt Cx,PetscReal fx,
                                                      PetscReal Ux0, PetscReal Uxf,PetscBool IsPeriodicX,PetscInt ngx,
                                                      PetscInt py,PetscInt Ny,PetscInt Cy,PetscReal fy,
                                                      PetscReal Uy0, PetscReal Uyf,PetscBool IsPeriodicY,PetscInt ngy)
;
extern PetscErrorCode DMIGAInitializeUniform3d(DM dm,PetscBool IsRational,PetscInt NumDerivatives,PetscInt ndof,
                                               PetscInt px,PetscInt Nx,PetscInt Cx,PetscReal Ux0, PetscReal Uxf,PetscBool IsPeriodicX,PetscInt ngx,
                                               PetscInt py,PetscInt Ny,PetscInt Cy,PetscReal Uy0, PetscReal Uyf,PetscBool IsPeriodicY,PetscInt ngy,
                                               PetscInt pz,PetscInt Nz,PetscInt Cz,PetscReal Uz0, PetscReal Uzf,PetscBool IsPeriodicZ,PetscInt ngz);
extern PetscErrorCode DMIGAInitializeGeometry3d(DM dm,PetscInt ndof,PetscInt NumDerivatives,char *FunctionSpaceFile,char *GeomFile);
extern PetscErrorCode DMIGAKnotRefine2d(DM dm,PetscInt kx,PetscReal *Ux,PetscInt ky,PetscReal *Uy,DM iga_new);
extern PetscErrorCode DMIGAKnotRefine3d(DM dm,PetscInt kx,PetscReal *Ux,PetscInt ky,PetscReal *Uy,PetscInt kz,PetscReal *Uz,DM iga_new);
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

extern PetscErrorCode DMIGAGetBasisData(DM,BD*,BD*,BD*);

extern PetscErrorCode BDCreate(BD *bd,PetscInt numD,PetscInt p,PetscInt numGP,PetscInt numEl);
extern PetscErrorCode BDDestroy(BD *bd);
extern PetscErrorCode BDGetBasis(BD bd, PetscInt iel, PetscInt igp, PetscInt ib, PetscInt ider, PetscReal *basis);
extern PetscErrorCode BDSetBasis(BD bd, PetscInt iel, PetscInt igp, PetscInt ib, PetscInt ider, PetscReal basis);
extern PetscErrorCode BDGetGaussPt(BD bd, PetscInt iel, PetscInt igp, PetscReal *gp);
extern PetscErrorCode BDSetGaussPt(BD bd, PetscInt iel, PetscInt igp, PetscReal gp);
extern PetscErrorCode BDGetGaussWt(BD bd, PetscInt iel, PetscInt igp, PetscReal *gw);
extern PetscErrorCode BDSetGaussWt(BD bd, PetscInt iel, PetscInt igp, PetscReal gw);
extern PetscErrorCode BDGetBasisOffset(BD bd, PetscInt iel, PetscInt *bOffset);
extern PetscErrorCode BDSetBasisOffset(BD bd, PetscInt iel, PetscInt bOffset);
extern PetscErrorCode BDSetElementOwnership(BD bd,PetscInt nel,PetscInt dof_b,PetscInt dof_e,PetscInt p);

#endif /* __PETSCDMIGA_H */
