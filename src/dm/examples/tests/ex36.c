
static char help[] = "Checks the functionality of DMGetInterpolation() on deformed grids.\n\n";

#include <petsc.h>
#include <petscvec.h>
#include <petscmat.h>
#include <petscdmda.h>

typedef struct _n_CCmplx CCmplx;
struct _n_CCmplx {
  double real;
  double imag;
};

CCmplx CCmplxPow(CCmplx a,double n)
{
  CCmplx b;
  double r,theta;
  r = sqrt(a.real*a.real + a.imag*a.imag);
  theta = atan2(a.imag,a.real);
  b.real = pow(r,n) * cos(n*theta);
  b.imag = pow(r,n) * sin(n*theta);
  return b;
}
CCmplx CCmplxExp(CCmplx a)
{
  CCmplx b;
  b.real = exp(a.real) * cos(a.imag);
  b.imag = exp(a.real) * sin(a.imag);
  return b;
}
CCmplx CCmplxSqrt(CCmplx a)
{
  CCmplx b;
  double r,theta;
  r = sqrt(a.real*a.real + a.imag*a.imag);
  theta = atan2(a.imag,a.real);
  b.real = sqrt(r) * cos(0.5*theta);
  b.imag = sqrt(r) * sin(0.5*theta);
  return b;
}
CCmplx CCmplxAdd(CCmplx a,CCmplx c)
{
  CCmplx b;
  b.real = a.real +c.real;
  b.imag = a.imag +c.imag;
  return b;
}
PetscScalar CCmplxRe(CCmplx a)
{
  return (PetscScalar)a.real;
}
PetscScalar CCmplxIm(CCmplx a)
{
  return (PetscScalar)a.imag;
}

#undef __FUNCT__
#define __FUNCT__ "DAApplyConformalMapping"
PetscErrorCode DAApplyConformalMapping(DM da,PetscInt idx)
{
  PetscErrorCode ierr;
  PetscInt       i,n;
  PetscInt       sx,nx,sy,ny,sz,nz,dim;
  Vec            Gcoords;
  PetscScalar    *XX;
  PetscScalar    xx,yy,zz;
  DM             cda;

  PetscFunctionBegin;
  if (idx==0) {
    PetscFunctionReturn(0);
  } else if (idx==1) { /* dam break */
    ierr = DMDASetUniformCoordinates(da, -1.0,1.0, -1.0,1.0, -1.0,1.0 );CHKERRQ(ierr);
  } else if (idx==2) { /* stagnation in a corner */
    ierr = DMDASetUniformCoordinates(da, -1.0,1.0, 0.0,1.0, -1.0,1.0 );CHKERRQ(ierr);
  } else if (idx==3) { /* nautilis */
    ierr = DMDASetUniformCoordinates(da, -1.0,1.0, -1.0,1.0, -1.0,1.0 );CHKERRQ(ierr);
  } else if (idx==4) {
    ierr = DMDASetUniformCoordinates(da, -1.0,1.0, -1.0,1.0, -1.0,1.0 );CHKERRQ(ierr);
  }

  ierr = DMGetCoordinateDM(da,&cda);CHKERRQ(ierr);
  ierr = DMGetCoordinates(da,&Gcoords);CHKERRQ(ierr);

  ierr = VecGetArray(Gcoords,&XX);CHKERRQ(ierr);
  ierr = DMDAGetCorners(da,&sx,&sy,&sz,&nx,&ny,&nz);CHKERRQ(ierr);
  ierr = DMDAGetInfo(da, &dim, 0,0,0, 0,0,0, 0,0,0,0,0,0);CHKERRQ(ierr);
  ierr = VecGetLocalSize(Gcoords,&n);CHKERRQ(ierr);
  n = n / dim;

  for (i=0; i<n; i++) {
    if ( (dim==3) && (idx!=2) ) {
      PetscScalar Ni[8];
      PetscScalar xi   = XX[dim*i  ];
      PetscScalar eta  = XX[dim*i+1];
      PetscScalar zeta = XX[dim*i+2];
      PetscScalar xn[] = {-1.0,1.0,-1.0,1.0,   -1.0,1.0,-1.0,1.0  };
      PetscScalar yn[] = {-1.0,-1.0,1.0,1.0,   -1.0,-1.0,1.0,1.0  };
      PetscScalar zn[] = {-0.1,-4.0,-0.2,-1.0,  0.1,4.0,0.2,1.0  };
      PetscInt p;

      Ni[0] = 0.125*(1.0-xi)*(1.0-eta)*(1.0-zeta);
      Ni[1] = 0.125*(1.0+xi)*(1.0-eta)*(1.0-zeta);
      Ni[2] = 0.125*(1.0-xi)*(1.0+eta)*(1.0-zeta);
      Ni[3] = 0.125*(1.0+xi)*(1.0+eta)*(1.0-zeta);

      Ni[4] = 0.125*(1.0-xi)*(1.0-eta)*(1.0+zeta);
      Ni[5] = 0.125*(1.0+xi)*(1.0-eta)*(1.0+zeta);
      Ni[6] = 0.125*(1.0-xi)*(1.0+eta)*(1.0+zeta);
      Ni[7] = 0.125*(1.0+xi)*(1.0+eta)*(1.0+zeta);
      xx = yy = zz = 0.0;
      for (p=0; p<8; p++ ) {
        xx += Ni[p]*xn[p];
        yy += Ni[p]*yn[p];
        zz += Ni[p]*zn[p];
      }
      XX[dim*i  ] = xx;
      XX[dim*i+1] = yy;
      XX[dim*i+2] = zz;
    }

    if (idx==1) {
      CCmplx zeta,t1,t2;

      xx = XX[dim*i  ] - 0.8;
      yy = XX[dim*i+1] + 1.5;
      zeta.real = PetscRealPart(xx);
      zeta.imag = PetscRealPart(yy);

      t1 = CCmplxPow(zeta,-1.0);
      t2 = CCmplxAdd(zeta,t1);
      XX[dim*i  ] = CCmplxRe(t2);
      XX[dim*i+1] = CCmplxIm(t2);
    } else if (idx==2) {
      CCmplx zeta,t1;

      xx = XX[dim*i  ];
      yy = XX[dim*i+1];
      zeta.real = PetscRealPart(xx);
      zeta.imag = PetscRealPart(yy);

      t1 = CCmplxSqrt(zeta);
      XX[dim*i  ] = CCmplxRe(t1);
      XX[dim*i+1] = CCmplxIm(t1);
    } else if (idx==3) {
      CCmplx zeta,t1,t2;

      xx = XX[dim*i  ] - 0.8;
      yy = XX[dim*i+1] + 1.5;

      zeta.real = PetscRealPart(xx);
      zeta.imag = PetscRealPart(yy);
      t1 = CCmplxPow(zeta,-1.0);
      t2 = CCmplxAdd(zeta,t1);
      XX[dim*i  ] = CCmplxRe(t2);
      XX[dim*i+1] = CCmplxIm(t2);

      xx = XX[dim*i  ];
      yy = XX[dim*i+1];
      zeta.real = PetscRealPart(xx);
      zeta.imag = PetscRealPart(yy);
      t1 = CCmplxExp(zeta);
      XX[dim*i  ] = CCmplxRe(t1);
      XX[dim*i+1] = CCmplxIm(t1);

      xx = XX[dim*i  ] + 0.4;
      yy = XX[dim*i+1];
      zeta.real = PetscRealPart(xx);
      zeta.imag = PetscRealPart(yy);
      t1 = CCmplxPow(zeta,2.0);
      XX[dim*i  ] = CCmplxRe(t1);
      XX[dim*i+1] = CCmplxIm(t1);
    }
    else if (idx==4) {
      PetscScalar Ni[4];
      PetscScalar xi  = XX[dim*i  ];
      PetscScalar eta = XX[dim*i+1];
      PetscScalar xn[] = {0.0,2.0,0.2,3.5};
      PetscScalar yn[] = {-1.3,0.0,2.0,4.0};
      PetscInt p;

      Ni[0] = 0.25*(1.0-xi)*(1.0-eta);
      Ni[1] = 0.25*(1.0+xi)*(1.0-eta);
      Ni[2] = 0.25*(1.0-xi)*(1.0+eta);
      Ni[3] = 0.25*(1.0+xi)*(1.0+eta);
      xx = yy = 0.0;
      for (p=0; p<4; p++ ) {
        xx += Ni[p]*xn[p];
        yy += Ni[p]*yn[p];
      }
      XX[dim*i  ] = xx;
      XX[dim*i+1] = yy;
    }
  }
  ierr = VecRestoreArray(Gcoords,&XX);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DAApplyTrilinearMapping"
PetscErrorCode DAApplyTrilinearMapping(DM da)
{
  PetscErrorCode ierr;
  PetscInt       i,j,k;
  PetscInt       sx,nx,sy,ny,sz,nz;
  Vec            Gcoords;
  DMDACoor3d     ***XX;
  PetscScalar    xx,yy,zz;
  DM             cda;

  PetscFunctionBegin;
  ierr = DMDASetUniformCoordinates(da, -1.0,1.0, -1.0,1.0, -1.0,1.0 );CHKERRQ(ierr);
  ierr = DMGetCoordinateDM(da,&cda);CHKERRQ(ierr);
  ierr = DMGetCoordinates(da,&Gcoords);CHKERRQ(ierr);

  ierr = DMDAVecGetArray(cda,Gcoords,&XX);CHKERRQ(ierr);
  ierr = DMDAGetCorners(da,&sx,&sy,&sz,&nx,&ny,&nz);CHKERRQ(ierr);

  for (i=sx; i<sx+nx; i++) {
    for (j=sy; j<sy+ny; j++ ) {
      for (k=sz; k<sz+nz; k++ ) {
        PetscScalar Ni[8];
        PetscScalar xi   = XX[k][j][i].x;
        PetscScalar eta  = XX[k][j][i].y;
        PetscScalar zeta = XX[k][j][i].z;
        PetscScalar xn[] = {0.0,2.0,0.2,3.5,   0.0,2.1,0.23,3.125  };
        PetscScalar yn[] = {-1.3,0.0,2.0,4.0,  -1.45,-0.1,2.24,3.79  };
        PetscScalar zn[] = {0.0,0.3,-0.1,0.123,  0.956,1.32,1.12,0.798  };
        PetscInt    p;

        Ni[0] = 0.125*(1.0-xi)*(1.0-eta)*(1.0-zeta);
        Ni[1] = 0.125*(1.0+xi)*(1.0-eta)*(1.0-zeta);
        Ni[2] = 0.125*(1.0-xi)*(1.0+eta)*(1.0-zeta);
        Ni[3] = 0.125*(1.0+xi)*(1.0+eta)*(1.0-zeta);

        Ni[4] = 0.125*(1.0-xi)*(1.0-eta)*(1.0+zeta);
        Ni[5] = 0.125*(1.0+xi)*(1.0-eta)*(1.0+zeta);
        Ni[6] = 0.125*(1.0-xi)*(1.0+eta)*(1.0+zeta);
        Ni[7] = 0.125*(1.0+xi)*(1.0+eta)*(1.0+zeta);
        xx = yy = zz = 0.0;
        for (p=0; p<8; p++ ) {
          xx += Ni[p]*xn[p];
          yy += Ni[p]*yn[p];
          zz += Ni[p]*zn[p];
        }
        XX[k][j][i].x = xx;
        XX[k][j][i].y = yy;
        XX[k][j][i].z = zz;
      }
    }
  }
  ierr = DMDAVecRestoreArray(cda,Gcoords,&XX);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DADefineXLinearField2D"
PetscErrorCode DADefineXLinearField2D(DM da,Vec field)
{
  PetscErrorCode ierr;
  PetscInt       i,j;
  PetscInt       sx,nx,sy,ny;
  Vec            Gcoords;
  DMDACoor2d     **XX;
  PetscScalar    **FF;
  DM             cda;

  PetscFunctionBegin;
  ierr = DMGetCoordinateDM(da,&cda);CHKERRQ(ierr);
  ierr = DMGetCoordinates(da,&Gcoords);CHKERRQ(ierr);

  ierr = DMDAVecGetArray(cda,Gcoords,&XX);CHKERRQ(ierr);
  ierr = DMDAVecGetArray(da,field,&FF);CHKERRQ(ierr);

  ierr = DMDAGetCorners(da,&sx,&sy,0,&nx,&ny,0);CHKERRQ(ierr);

  for (i=sx; i<sx+nx; i++) {
    for (j=sy; j<sy+ny; j++ ) {
      FF[j][i] = 10.0 + 3.0 * XX[j][i].x + 5.5 * XX[j][i].y + 8.003 * XX[j][i].x * XX[j][i].y;
    }
  }

  ierr = DMDAVecRestoreArray(da,field,&FF);CHKERRQ(ierr);
  ierr = DMDAVecRestoreArray(cda,Gcoords,&XX);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DADefineXLinearField3D"
PetscErrorCode DADefineXLinearField3D(DM da,Vec field)
{
  PetscErrorCode ierr;
  PetscInt       i,j,k;
  PetscInt       sx,nx,sy,ny,sz,nz;
  Vec            Gcoords;
  DMDACoor3d     ***XX;
  PetscScalar    ***FF;
  DM             cda;

  PetscFunctionBegin;
  ierr = DMGetCoordinateDM(da,&cda);CHKERRQ(ierr);
  ierr = DMGetCoordinates(da,&Gcoords);CHKERRQ(ierr);

  ierr = DMDAVecGetArray(cda,Gcoords,&XX);CHKERRQ(ierr);
  ierr = DMDAVecGetArray(da,field,&FF);CHKERRQ(ierr);

  ierr = DMDAGetCorners(da,&sx,&sy,&sz,&nx,&ny,&nz);CHKERRQ(ierr);

  for (k=sz; k<sz+nz; k++) {
    for (j=sy; j<sy+ny; j++ ) {
      for (i=sx; i<sx+nx; i++) {
        FF[k][j][i] = 10.0
                + 4.05 * XX[k][j][i].x
                + 5.50 * XX[k][j][i].y
                + 1.33 * XX[k][j][i].z
                + 2.03 * XX[k][j][i].x * XX[k][j][i].y
                + 0.03 * XX[k][j][i].x * XX[k][j][i].z
                + 0.83 * XX[k][j][i].y * XX[k][j][i].z
                + 3.79 * XX[k][j][i].x * XX[k][j][i].y * XX[k][j][i].z;
      }
    }
  }

  ierr = DMDAVecRestoreArray(da,field,&FF);CHKERRQ(ierr);
  ierr = DMDAVecRestoreArray(cda,Gcoords,&XX);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "da_test_RefineCoords1D"
PetscErrorCode da_test_RefineCoords1D(PetscInt mx)
{
  PetscErrorCode ierr;
  DM             dac,daf;
  PetscViewer    vv;
  Vec            ac,af;
  PetscInt       Mx;
  Mat            II,INTERP;
  Vec            scale;
  PetscBool      output = PETSC_FALSE;

  PetscFunctionBegin;
  ierr = DMDACreate1d( PETSC_COMM_WORLD, DMDA_BOUNDARY_NONE,
                      mx+1,
                      1, /* 1 dof */
                      1, /* stencil = 1 */
                      PETSC_NULL,
                      &dac ); CHKERRQ(ierr);
  ierr = DMSetFromOptions(dac);CHKERRQ(ierr);

  ierr = DMRefine(dac,MPI_COMM_NULL,&daf);CHKERRQ(ierr);
  ierr = DMDAGetInfo(daf,0,&Mx,0,0,0,0,0,0,0,0,0,0,0);CHKERRQ(ierr);
  Mx--;

  ierr = DMDASetUniformCoordinates(dac, -1.0,1.0, PETSC_DECIDE,PETSC_DECIDE, PETSC_DECIDE,PETSC_DECIDE );CHKERRQ(ierr);
  ierr = DMDASetUniformCoordinates(daf, -1.0,1.0, PETSC_DECIDE,PETSC_DECIDE, PETSC_DECIDE,PETSC_DECIDE );CHKERRQ(ierr);

  {
    DM  cdaf,cdac;
    Vec coordsc,coordsf;

    ierr = DMGetCoordinateDM(dac,&cdac);CHKERRQ(ierr);
    ierr = DMGetCoordinateDM(daf,&cdaf);CHKERRQ(ierr);

    ierr = DMGetCoordinates(dac,&coordsc);CHKERRQ(ierr);
    ierr = DMGetCoordinates(daf,&coordsf);CHKERRQ(ierr);

    ierr = DMCreateInterpolation(cdac,cdaf,&II,&scale);CHKERRQ(ierr);
    ierr = MatInterpolate(II,coordsc,coordsf);CHKERRQ(ierr);
    ierr = MatDestroy(&II);CHKERRQ(ierr);
    ierr = VecDestroy(&scale);CHKERRQ(ierr);
  }

  ierr = DMCreateInterpolation(dac,daf,&INTERP,PETSC_NULL);CHKERRQ(ierr);

  ierr = DMCreateGlobalVector(dac,&ac); CHKERRQ(ierr);
  ierr = VecSet(ac,66.99);CHKERRQ(ierr);

  ierr = DMCreateGlobalVector(daf,&af); CHKERRQ(ierr);

  ierr = MatMult(INTERP,ac, af); CHKERRQ(ierr);

  {
    Vec       afexact;
    PetscReal nrm;
    PetscInt  N;

    ierr = DMCreateGlobalVector(daf,&afexact); CHKERRQ(ierr);
    ierr = VecSet(afexact,66.99);CHKERRQ(ierr);
    ierr = VecAXPY(afexact,-1.0,af);CHKERRQ(ierr); /* af <= af - afinterp */
    ierr = VecNorm(afexact,NORM_2,&nrm);CHKERRQ(ierr);
    ierr = VecGetSize(afexact,&N);CHKERRQ(ierr);
    PetscPrintf(PETSC_COMM_WORLD,"%D=>%D, interp err = %1.4e\n",mx,Mx,nrm/sqrt((PetscReal)N) );
    ierr = VecDestroy(&afexact);CHKERRQ(ierr);
  }

  PetscOptionsGetBool(PETSC_NULL,"-output",&output,PETSC_NULL);CHKERRQ(ierr);
  if (output) {
    ierr = PetscViewerASCIIOpen(PETSC_COMM_WORLD, "dac_1D.vtk", &vv);CHKERRQ(ierr);
    ierr = PetscViewerSetFormat(vv, PETSC_VIEWER_ASCII_VTK);CHKERRQ(ierr);
    ierr = DMView(dac, vv);CHKERRQ(ierr);
    ierr = VecView(ac, vv);CHKERRQ(ierr);
    ierr = PetscViewerDestroy(&vv);CHKERRQ(ierr);

    ierr = PetscViewerASCIIOpen(PETSC_COMM_WORLD, "daf_1D.vtk", &vv);CHKERRQ(ierr);
    ierr = PetscViewerSetFormat(vv, PETSC_VIEWER_ASCII_VTK);CHKERRQ(ierr);
    ierr = DMView(daf, vv);CHKERRQ(ierr);
    ierr = VecView(af, vv);CHKERRQ(ierr);
    ierr = PetscViewerDestroy(&vv);CHKERRQ(ierr);
  }

  ierr = MatDestroy(&INTERP);CHKERRQ(ierr);
  ierr = DMDestroy(&dac);CHKERRQ(ierr);
  ierr = DMDestroy(&daf);CHKERRQ(ierr);
  ierr = VecDestroy(&ac);CHKERRQ(ierr);
  ierr = VecDestroy(&af);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "da_test_RefineCoords2D"
PetscErrorCode da_test_RefineCoords2D(PetscInt mx,PetscInt my)
{
  PetscErrorCode ierr;
  DM             dac,daf;
  PetscViewer    vv;
  Vec            ac,af;
  PetscInt       map_id,Mx,My;
  Mat            II,INTERP;
  Vec            scale;
  PetscBool      output = PETSC_FALSE;

  PetscFunctionBegin;
  ierr = DMDACreate2d(PETSC_COMM_WORLD, DMDA_BOUNDARY_NONE, DMDA_BOUNDARY_NONE, DMDA_STENCIL_BOX,
                      mx+1, my+1,
                      PETSC_DECIDE, PETSC_DECIDE,
                      1, /* 1 dof */
                      1, /* stencil = 1 */
                      PETSC_NULL, PETSC_NULL,
                      &dac ); CHKERRQ(ierr);
  ierr = DMSetFromOptions(dac);CHKERRQ(ierr);

  ierr = DMRefine(dac,MPI_COMM_NULL,&daf);CHKERRQ(ierr);
  ierr = DMDAGetInfo(daf,0,&Mx,&My,0,0,0,0,0,0,0,0,0,0);CHKERRQ(ierr);
  Mx--; My--;

  ierr = DMDASetUniformCoordinates(dac, -1.0,1.0, -1.0,1.0, PETSC_DECIDE,PETSC_DECIDE );CHKERRQ(ierr);
  ierr = DMDASetUniformCoordinates(daf, -1.0,1.0, -1.0,1.0, PETSC_DECIDE,PETSC_DECIDE );CHKERRQ(ierr);

  /* apply conformal mappings */
  map_id = 0;
  ierr = PetscOptionsGetInt( PETSC_NULL,"-cmap", &map_id,PETSC_NULL ); CHKERRQ(ierr);
  if (map_id >= 1) {
    ierr = DAApplyConformalMapping(dac,map_id);CHKERRQ(ierr);
  }

  {
    DM  cdaf,cdac;
    Vec coordsc,coordsf;

    ierr = DMGetCoordinateDM(dac,&cdac);CHKERRQ(ierr);
    ierr = DMGetCoordinateDM(daf,&cdaf);CHKERRQ(ierr);

    ierr = DMGetCoordinates(dac,&coordsc);CHKERRQ(ierr);
    ierr = DMGetCoordinates(daf,&coordsf);CHKERRQ(ierr);

    ierr = DMCreateInterpolation(cdac,cdaf,&II,&scale);CHKERRQ(ierr);
    ierr = MatInterpolate(II,coordsc,coordsf);CHKERRQ(ierr);
    ierr = MatDestroy(&II);CHKERRQ(ierr);
    ierr = VecDestroy(&scale);CHKERRQ(ierr);
  }


  ierr = DMCreateInterpolation(dac,daf,&INTERP,PETSC_NULL);CHKERRQ(ierr);

  ierr = DMCreateGlobalVector(dac,&ac); CHKERRQ(ierr);
  ierr = DADefineXLinearField2D(dac,ac);CHKERRQ(ierr);

  ierr = DMCreateGlobalVector(daf,&af); CHKERRQ(ierr);
  ierr = MatMult(INTERP,ac, af); CHKERRQ(ierr);

  {
    Vec       afexact;
    PetscReal nrm;
    PetscInt  N;

    ierr = DMCreateGlobalVector(daf,&afexact); CHKERRQ(ierr);
    ierr = VecZeroEntries(afexact);CHKERRQ(ierr);
    ierr = DADefineXLinearField2D(daf,afexact);CHKERRQ(ierr);
    ierr = VecAXPY(afexact,-1.0,af);CHKERRQ(ierr); /* af <= af - afinterp */
    ierr = VecNorm(afexact,NORM_2,&nrm);CHKERRQ(ierr);
    ierr = VecGetSize(afexact,&N);CHKERRQ(ierr);
    PetscPrintf(PETSC_COMM_WORLD,"[%D x %D]=>[%D x %D], interp err = %1.4e\n",mx,my,Mx,My,nrm/sqrt((PetscReal)N) );
    ierr = VecDestroy(&afexact);CHKERRQ(ierr);
  }

  PetscOptionsGetBool(PETSC_NULL,"-output",&output,PETSC_NULL);CHKERRQ(ierr);
  if (output) {
    ierr = PetscViewerASCIIOpen(PETSC_COMM_WORLD, "dac_2D.vtk", &vv);CHKERRQ(ierr);
    ierr = PetscViewerSetFormat(vv, PETSC_VIEWER_ASCII_VTK);CHKERRQ(ierr);
    ierr = DMView(dac, vv);CHKERRQ(ierr);
    ierr = VecView(ac, vv);CHKERRQ(ierr);
    ierr = PetscViewerDestroy(&vv);CHKERRQ(ierr);

    ierr = PetscViewerASCIIOpen(PETSC_COMM_WORLD, "daf_2D.vtk", &vv);CHKERRQ(ierr);
    ierr = PetscViewerSetFormat(vv, PETSC_VIEWER_ASCII_VTK);CHKERRQ(ierr);
    ierr = DMView(daf, vv);CHKERRQ(ierr);
    ierr = VecView(af, vv);CHKERRQ(ierr);
    ierr = PetscViewerDestroy(&vv);CHKERRQ(ierr);
  }

  ierr = MatDestroy(&INTERP);CHKERRQ(ierr);
  ierr = DMDestroy(&dac);CHKERRQ(ierr);
  ierr = DMDestroy(&daf);CHKERRQ(ierr);
  ierr = VecDestroy(&ac);CHKERRQ(ierr);
  ierr = VecDestroy(&af);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "da_test_RefineCoords3D"
PetscErrorCode da_test_RefineCoords3D(PetscInt mx,PetscInt my,PetscInt mz)
{
  PetscErrorCode ierr;
  DM             dac,daf;
  PetscViewer    vv;
  Vec            ac,af;
  PetscInt       map_id,Mx,My,Mz;
  Mat            II,INTERP;
  Vec            scale;
  PetscBool      output = PETSC_FALSE;

  PetscFunctionBegin;
  ierr = DMDACreate3d(PETSC_COMM_WORLD,DMDA_BOUNDARY_NONE,DMDA_BOUNDARY_NONE,DMDA_BOUNDARY_NONE,DMDA_STENCIL_BOX,
                      mx+1, my+1,mz+1,
                      PETSC_DECIDE, PETSC_DECIDE,PETSC_DECIDE,
                      1, /* 1 dof */
                      1, /* stencil = 1 */
                      PETSC_NULL,PETSC_NULL,PETSC_NULL,
                      &dac ); CHKERRQ(ierr);
  ierr = DMSetFromOptions(dac);CHKERRQ(ierr);

  ierr = DMRefine(dac,MPI_COMM_NULL,&daf);CHKERRQ(ierr);
  ierr = DMDAGetInfo(daf,0,&Mx,&My,&Mz,0,0,0,0,0,0,0,0,0);CHKERRQ(ierr);
  Mx--; My--; Mz--;

  ierr = DMDASetUniformCoordinates(dac, -1.0,1.0, -1.0,1.0, -1.0,1.0 );CHKERRQ(ierr);
  ierr = DMDASetUniformCoordinates(daf, -1.0,1.0, -1.0,1.0, -1.0,1.0 );CHKERRQ(ierr);

  /* apply trilinear mappings */
  /*ierr = DAApplyTrilinearMapping(dac);CHKERRQ(ierr);*/
  /* apply conformal mappings */
  map_id = 0;
  ierr = PetscOptionsGetInt( PETSC_NULL,"-cmap", &map_id,PETSC_NULL ); CHKERRQ(ierr);
  if (map_id >= 1) {
    ierr = DAApplyConformalMapping(dac,map_id);CHKERRQ(ierr);
  }

  {
    DM  cdaf,cdac;
    Vec coordsc,coordsf;

    ierr = DMGetCoordinateDM(dac,&cdac);CHKERRQ(ierr);
    ierr = DMGetCoordinateDM(daf,&cdaf);CHKERRQ(ierr);

    ierr = DMGetCoordinates(dac,&coordsc);CHKERRQ(ierr);
    ierr = DMGetCoordinates(daf,&coordsf);CHKERRQ(ierr);

    ierr = DMCreateInterpolation(cdac,cdaf,&II,&scale);CHKERRQ(ierr);
    ierr = MatInterpolate(II,coordsc,coordsf);CHKERRQ(ierr);
    ierr = MatDestroy(&II);CHKERRQ(ierr);
    ierr = VecDestroy(&scale);CHKERRQ(ierr);
  }

  ierr = DMCreateInterpolation(dac,daf,&INTERP,PETSC_NULL);CHKERRQ(ierr);

  ierr = DMCreateGlobalVector(dac,&ac); CHKERRQ(ierr);
  ierr = VecZeroEntries(ac);CHKERRQ(ierr);
  ierr = DADefineXLinearField3D(dac,ac);CHKERRQ(ierr);

  ierr = DMCreateGlobalVector(daf,&af); CHKERRQ(ierr);
  ierr = VecZeroEntries(af);CHKERRQ(ierr);

  ierr = MatMult(INTERP,ac, af); CHKERRQ(ierr);

  {
    Vec       afexact;
    PetscReal nrm;
    PetscInt  N;

    ierr = DMCreateGlobalVector(daf,&afexact); CHKERRQ(ierr);
    ierr = VecZeroEntries(afexact);CHKERRQ(ierr);
    ierr = DADefineXLinearField3D(daf,afexact);CHKERRQ(ierr);
    ierr = VecAXPY(afexact,-1.0,af);CHKERRQ(ierr); /* af <= af - afinterp */
    ierr = VecNorm(afexact,NORM_2,&nrm);CHKERRQ(ierr);
    ierr = VecGetSize(afexact,&N);CHKERRQ(ierr);
    PetscPrintf(PETSC_COMM_WORLD,"[%D x %D x %D]=>[%D x %D x %D], interp err = %1.4e\n",mx,my,mz,Mx,My,Mz,nrm/sqrt((PetscReal)N) );
    ierr = VecDestroy(&afexact);CHKERRQ(ierr);
  }

  PetscOptionsGetBool(PETSC_NULL,"-output",&output,PETSC_NULL);CHKERRQ(ierr);
  if (output) {
    ierr = PetscViewerASCIIOpen(PETSC_COMM_WORLD, "dac_3D.vtk", &vv);CHKERRQ(ierr);
    ierr = PetscViewerSetFormat(vv, PETSC_VIEWER_ASCII_VTK);CHKERRQ(ierr);
    ierr = DMView(dac, vv);CHKERRQ(ierr);
    ierr = VecView(ac, vv);CHKERRQ(ierr);
    ierr = PetscViewerDestroy(&vv);CHKERRQ(ierr);

    ierr = PetscViewerASCIIOpen(PETSC_COMM_WORLD, "daf_3D.vtk", &vv);CHKERRQ(ierr);
    ierr = PetscViewerSetFormat(vv, PETSC_VIEWER_ASCII_VTK);CHKERRQ(ierr);
    ierr = DMView(daf, vv);CHKERRQ(ierr);
    ierr = VecView(af, vv);CHKERRQ(ierr);
    ierr = PetscViewerDestroy(&vv);CHKERRQ(ierr);
  }

  ierr = MatDestroy(&INTERP);CHKERRQ(ierr);
  ierr = DMDestroy(&dac);CHKERRQ(ierr);
  ierr = DMDestroy(&daf);CHKERRQ(ierr);
  ierr = VecDestroy(&ac);CHKERRQ(ierr);
  ierr = VecDestroy(&af);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc,char **argv)
{
  PetscErrorCode ierr;
  PetscInt       mx,my,mz,l,nl,dim;

  ierr = PetscInitialize(&argc,&argv,0,help);

  mx = my = mz = 2;
  PetscOptionsGetInt(PETSC_NULL,"-mx", &mx, 0 );
  PetscOptionsGetInt(PETSC_NULL,"-my", &my, 0 );
  PetscOptionsGetInt(PETSC_NULL,"-mz", &mz, 0 );
  nl = 1;
  PetscOptionsGetInt(PETSC_NULL,"-nl", &nl, 0 );
  dim = 2;
  PetscOptionsGetInt(PETSC_NULL,"-dim", &dim, 0 );

  for (l=0; l<nl; l++) {
    if      (dim==1) { da_test_RefineCoords1D(mx); }
    else if (dim==2) { da_test_RefineCoords2D(mx,my); }
    else if (dim==3) { da_test_RefineCoords3D(mx,my,mz); }
    mx = mx * 2;
    my = my * 2;
    mz = mz * 2;
  }

  ierr = PetscFinalize();CHKERRQ(ierr);
  return 0;
}
