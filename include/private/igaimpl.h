#if !defined(_IGAIMPL_H)
#define _IGAIMPL_H

#include <petscdmiga.h> /*I      "petscdmmesh.h"    I*/
#include <petscdmda.h>
#include "private/dmimpl.h"

typedef struct {
  PetscScalar x,y,z,w;
} GeometryPoint;

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
