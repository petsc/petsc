/*
  DMIGA, for NURBS finite elements used in isogeometric analysis.
*/
#if !defined(__PETSCDMIGA_H)
#define __PETSCDMIGA_H

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
                                               PetscInt px,PetscInt Nx,PetscInt Cx,PetscScalar Ux0, PetscScalar Uxf,PetscBool IsPeriodicX,PetscInt ngx);
extern PetscErrorCode DMIGAInitializeUniform2d(DM dm,PetscBool IsRational,PetscInt NumDerivatives,PetscInt ndof,
                                               PetscInt px,PetscInt Nx,PetscInt Cx,PetscScalar Ux0, PetscScalar Uxf,PetscBool IsPeriodicX,PetscInt ngx,
                                               PetscInt py,PetscInt Ny,PetscInt Cy,PetscScalar Uy0, PetscScalar Uyf,PetscBool IsPeriodicY,PetscInt ngy);
extern PetscErrorCode DMIGAInitializeSymmetricTaper2d(DM dm,PetscBool IsRational,PetscInt NumDerivatives,PetscInt ndof,
                                                      PetscInt px,PetscInt Nx,PetscInt Cx,PetscScalar fx,
                                                      PetscScalar Ux0, PetscScalar Uxf,PetscBool IsPeriodicX,PetscInt ngx,
                                                      PetscInt py,PetscInt Ny,PetscInt Cy,PetscScalar fy,
                                                      PetscScalar Uy0, PetscScalar Uyf,PetscBool IsPeriodicY,PetscInt ngy);
extern PetscErrorCode DMIGAInitializeUniform3d(DM dm,PetscBool IsRational,PetscInt NumDerivatives,PetscInt ndof,
                                               PetscInt px,PetscInt Nx,PetscInt Cx,PetscScalar Ux0, PetscScalar Uxf,PetscBool IsPeriodicX,PetscInt ngx,
                                               PetscInt py,PetscInt Ny,PetscInt Cy,PetscScalar Uy0, PetscScalar Uyf,PetscBool IsPeriodicY,PetscInt ngy,
                                               PetscInt pz,PetscInt Nz,PetscInt Cz,PetscScalar Uz0, PetscScalar Uzf,PetscBool IsPeriodicZ,PetscInt ngz);
extern PetscErrorCode DMIGAInitializeGeometry3d(DM dm,PetscInt ndof,PetscInt NumDerivatives,char *FunctionSpaceFile,char *GeomFile);
extern PetscErrorCode DMIGAKnotRefine2d(DM dm,PetscInt kx,PetscScalar *Ux,PetscInt ky,PetscScalar *Uy,DM iga_new);
extern PetscErrorCode DMIGAKnotRefine3d(DM dm,PetscInt kx,PetscScalar *Ux,PetscInt ky,PetscScalar *Uy,PetscInt kz,PetscScalar *Uz,DM iga_new);
typedef struct {
  PetscScalar *basis; // (p+1)x(numD+1)
  PetscScalar gx,gw;
  PetscInt offset;
} GP;

typedef struct {
  int numD,p,numGP,numEl;
  int own_b, own_e;   // beginning/end of elements I 'own'
  int cont_b, cont_e; // beginning/end of elements that contribute to dofs I own
  GP  *data;
} BasisData1D;

typedef BasisData1D* BD;

extern PetscErrorCode DMIGAGetBasisData(DM,BD*,BD*,BD*);

extern PetscErrorCode BDCreate(BD *bd,int numD,int p,int numGP,int numEl);
extern PetscErrorCode BDDestroy(BD bd);
extern PetscErrorCode BDGetBasis(BD bd, int iel, int igp, int ib, int ider, double *basis);
extern PetscErrorCode BDSetBasis(BD bd, int iel, int igp, int ib, int ider, double basis);
extern PetscErrorCode BDGetGaussPt(BD bd, int iel, int igp, double *gp);
extern PetscErrorCode BDSetGaussPt(BD bd, int iel, int igp, double gp);
extern PetscErrorCode BDGetGaussWt(BD bd, int iel, int igp, double *gw);
extern PetscErrorCode BDSetGaussWt(BD bd, int iel, int igp, double gw);
extern PetscErrorCode BDGetBasisOffset(BD bd, int iel, int *bOffset);
extern PetscErrorCode BDSetBasisOffset(BD bd, int iel, int bOffset);
extern PetscErrorCode BDSetElementOwnership(BD bd,int nel,int dof_b,int dof_e,int p);

#endif /* __PETSCDMIGA_H */
