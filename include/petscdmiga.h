/*
  DMIGA, for NURBS finite elements used in isogeometric analysis.
*/
#if !defined(__PETSCDMIGA_H)
#define __PETSCDMIGA_H

#include <petscdm.h>

PetscErrorCode DMIGAInitializeUniform1d(DM dm,PetscBool IsRational,PetscInt NumDerivatives,PetscInt ndof,
                                        PetscInt px,PetscInt Nx,PetscInt Cx,PetscScalar Ux0, PetscScalar Uxf,PetscBool IsPeriodicX,PetscInt ngx);
PetscErrorCode DMIGAInitializeUniform2d(DM dm,PetscBool IsRational,PetscInt NumDerivatives,PetscInt ndof,
                                        PetscInt px,PetscInt Nx,PetscInt Cx,PetscScalar Ux0, PetscScalar Uxf,PetscBool IsPeriodicX,PetscInt ngx,
                                        PetscInt py,PetscInt Ny,PetscInt Cy,PetscScalar Uy0, PetscScalar Uyf,PetscBool IsPeriodicY,PetscInt ngy);
PetscErrorCode DMIGAInitializeSymmetricTaper2d(DM dm,PetscBool IsRational,PetscInt NumDerivatives,PetscInt ndof,
                                               PetscInt px,PetscInt Nx,PetscInt Cx,PetscScalar fx,
                                               PetscScalar Ux0, PetscScalar Uxf,PetscBool IsPeriodicX,PetscInt ngx,
                                               PetscInt py,PetscInt Ny,PetscInt Cy,PetscScalar fy,
                                               PetscScalar Uy0, PetscScalar Uyf,PetscBool IsPeriodicY,PetscInt ngy);
PetscErrorCode DMIGAInitializeUniform3d(DM dm,PetscBool IsRational,PetscInt NumDerivatives,PetscInt ndof,
                                        PetscInt px,PetscInt Nx,PetscInt Cx,PetscScalar Ux0, PetscScalar Uxf,PetscBool IsPeriodicX,PetscInt ngx,
                                        PetscInt py,PetscInt Ny,PetscInt Cy,PetscScalar Uy0, PetscScalar Uyf,PetscBool IsPeriodicY,PetscInt ngy,
                                        PetscInt pz,PetscInt Nz,PetscInt Cz,PetscScalar Uz0, PetscScalar Uzf,PetscBool IsPeriodicZ,PetscInt ngz);
PetscErrorCode DMIGAInitializeGeometry3d(DM dm,PetscInt ndof,PetscInt NumDerivatives,char *FunctionSpaceFile,char *GeomFile);
PetscErrorCode DMIGAKnotRefine2d(DM dm,PetscInt kx,PetscScalar *Ux,PetscInt ky,PetscScalar *Uy,DM iga_new);
PetscErrorCode DMIGAKnotRefine3d(DM dm,PetscInt kx,PetscScalar *Ux,PetscInt ky,PetscScalar *Uy,PetscInt kz,PetscScalar *Uz,DM iga_new);

#endif /* __PETSCDMIGA_H */
