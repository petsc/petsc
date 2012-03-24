#if !defined(_IGAIMPL_H)
#define _IGAIMPL_H

#include <petscdmiga.h> /*I      "petscdmmesh.h"    I*/
#include <petscdmda.h>
#include "petsc-private/dmimpl.h"

typedef struct {
  PetscScalar x,y,z,w;
} GeometryPoint;

PetscErrorCode Compute1DBasisFunctions(PetscInt numGP, PetscInt numD, PetscReal *U, PetscInt m, PetscInt porder, BD *bd1D);
PetscErrorCode GetDersBasisFuns(PetscInt i,PetscReal u,PetscInt p,PetscReal *U, PetscReal **N,PetscInt nd);
PetscInt            FindSpan(PetscReal *U,PetscInt m,PetscInt j,PetscInt porder);
PetscErrorCode SetupGauss1D(PetscInt n,PetscReal *X,PetscReal *W);
PetscErrorCode CreateKnotVector(PetscInt N,PetscInt p,PetscInt C,PetscInt m, PetscReal *U,PetscReal U0,PetscReal Uf);
PetscErrorCode CreateKnotVectorFromMesh(PetscInt N,PetscInt p,PetscInt C,PetscInt m, PetscReal *U,PetscReal *X,PetscInt nX);
PetscErrorCode CreatePeriodicKnotVector(PetscInt N,PetscInt p,PetscInt C,PetscInt m, PetscReal *U,PetscReal U0,PetscReal Uf);
PetscErrorCode CreateTaperSetOfPoints(PetscReal Xbegin,PetscReal Xend,PetscReal f,PetscInt N,PetscReal *X);
PetscErrorCode CheckKnots(PetscInt m,PetscReal *U,PetscInt k,PetscReal *Uadd);

typedef struct {
  PetscInt   px,py,pz;     /* polynomial order */
  PetscInt   ngx,ngy,ngz;  /* number of gauss per element */
  PetscInt   nbx,nby,nbz;  /* number of basis */
  PetscInt   Nx,Ny,Nz;     /* number of elements */
  PetscInt   Cx,Cy,Cz;     /* global continuity level (optional) */
  PetscInt   mx,my,mz;     /* size of knot vectors */
  PetscReal *Ux, *Uy, *Uz; /* knot vectors */
  BD         bdX,bdY,bdZ;  /* stores precomputed 1D basis functions */
  PetscBool  IsPeriodicX,IsPeriodicY,IsPeriodicZ; /* periodicity of knot vectors */

  PetscInt   numD;         /* number of 1D derivatives needed */
  PetscBool  IsRational,IsMapped;

  DM da_dof,da_geometry;

  Mat K;
  Vec F;
  Vec G;
} DM_IGA;

#endif /* _IGAIMPL_H */
