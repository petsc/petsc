#if !defined(_IGAIMPL_H)
#define _IGAIMPL_H

#include <petscdmiga.h> /*I      "petscdmmesh.h"    I*/
#include <petscdmda.h>
#include "private/dmimpl.h"

typedef struct {
  PetscScalar *basis; // (p+1)x(numD+1)
  PetscScalar gx,gw;
  PetscInt offset;
} GP;

typedef struct {
  PetscScalar x,y,z,w;
} GeometryPoint;

typedef struct {
  int numD,p,numGP,numEl;
  int own_b, own_e;   // beginning/end of elements I 'own'
  int cont_b, cont_e; // beginning/end of elements that contribute to dofs I own
  GP  *data;
} BasisData1D;

typedef BasisData1D* BD;

PetscErrorCode BDCreate(BD *bd,int numD,int p,int numGP,int numEl);
PetscErrorCode BDDestroy(BD bd);
PetscErrorCode BDGetBasis(BD bd, int iel, int igp, int ib, int ider, double *basis);
PetscErrorCode BDSetBasis(BD bd, int iel, int igp, int ib, int ider, double basis);
PetscErrorCode BDGetGaussPt(BD bd, int iel, int igp, double *gp);
PetscErrorCode BDSetGaussPt(BD bd, int iel, int igp, double gp);
PetscErrorCode BDGetGaussWt(BD bd, int iel, int igp, double *gw);
PetscErrorCode BDSetGaussWt(BD bd, int iel, int igp, double gw);
PetscErrorCode BDGetBasisOffset(BD bd, int iel, int *bOffset);
PetscErrorCode BDSetBasisOffset(BD bd, int iel, int bOffset);
PetscErrorCode BDSetElementOwnership(BD bd,int nel,int dof_b,int dof_e,int p);

PetscErrorCode Compute1DBasisFunctions(PetscInt numGP, PetscInt numD, double *U, PetscScalar m, PetscInt porder, BD *bd1D);
PetscErrorCode GetDersBasisFuns(int i,double u,int p,double *U, double **N,int nd);
PetscInt       FindSpan(double *U,int m,int j,int porder);
PetscErrorCode SetupGauss1D(int n,double *X,double *W);
PetscErrorCode CreateKnotVector(int N,int p,int C,int m, PetscScalar *U,PetscScalar U0,PetscScalar Uf);
PetscErrorCode CreateKnotVectorFromMesh(int N,int p,int C,int m, PetscScalar *U,PetscScalar *X,PetscInt nX);
PetscErrorCode CreatePeriodicKnotVector(int N,int p,int C,int m, PetscScalar *U,PetscScalar U0,PetscScalar Uf);
PetscErrorCode CreateTaperSetOfPoints(PetscScalar Xbegin,PetscScalar Xend,PetscScalar f,PetscInt N,PetscScalar *X);
PetscErrorCode CheckKnots(PetscInt m,PetscScalar *U,PetscInt k,PetscScalar *Uadd);

typedef struct {
  PetscInt   px,py,pz;     // polynomial order
  PetscInt   ngx,ngy,ngz;  // number of gauss per element
  PetscInt   nbx,nby,nbz;  // number of basis
  PetscInt   Nx,Ny,Nz;     // number of elements
  PetscInt   Cx,Cy,Cz;     // global continuity level (optional)
  PetscInt   mx,my,mz;     // size of knot vectors
  PetscReal *Ux, *Uy, *Uz; // knot vectors
  BD         bdX,bdY,bdZ;  // stores precomputed 1D basis functions
  PetscBool  IsPeriodicX,IsPeriodicY,IsPeriodicZ; // periodicity of knot vectors

  PetscInt   numD;         // number of 1D derivatives needed
  PetscBool  IsRational,IsMapped;

  DM da_dof,da_geometry;

  Mat K;
  Vec F;
  Vec G;
} DM_IGA;

#endif /* _IGAIMPL_H */
