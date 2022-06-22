static char help[] ="Use DMDACreatePatchIS  to extract a slice from a vector, Command line options :\n\
mx/my/mz - set the dimensions of the parent vector\n\
dim - set the dimensionality of the parent vector (2,3)\n\
sliceaxis - Integer describing the axis along which the sice will be selected (0-X, 1-Y, 2-Z)\n\
sliceid - set the location where the slice will be extraced from the parent vector\n";

/*
   This test checks the functionality of DMDACreatePatchIS when
   extracting a 2D vector from a 3D vector and 1D vector from a
   2D vector.
   */

#include <petscdmda.h>

int main(int argc,char **argv)
{
  PetscMPIInt    rank, size;                    /* MPI rank and size */
  PetscInt       mx=4,my=4,mz=4;                /* Dimensions of parent vector */
  PetscInt       sliceid=2;                     /* k (z) index to pick the slice */
  PetscInt       sliceaxis=2;                   /* Select axis along which the slice will be extracted */
  PetscInt       dim=3;                         /* Dimension of the parent vector */
  PetscInt       i,j,k;                         /* Iteration indices */
  PetscInt       ixs,iys,izs;                   /* Corner indices for 3D vector */
  PetscInt       ixm,iym,izm;                   /* Widths of parent vector */
  PetscScalar    ***vecdata3d;                  /* Pointer to access 3d parent vector */
  PetscScalar    **vecdata2d;                   /* Pointer to access 2d parent vector */
  DM             da;                            /* 2D/3D DMDA object */
  Vec            vec_full;                      /* Parent vector */
  Vec            vec_slice;                     /* Slice vector */
  MatStencil     lower, upper;                  /* Stencils to select slice */
  IS             selectis;                      /* IS to select slice and extract subvector */
  PetscBool      patchis_offproc = PETSC_FALSE; /* flag to DMDACreatePatchIS indicating that off-proc values are to be ignored */

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Initialize program and set problem parameters
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  PetscCall(PetscInitialize(&argc,&argv,(char*)0,help));
  PetscCallMPI(MPI_Comm_size(PETSC_COMM_WORLD,&size));
  PetscCallMPI(MPI_Comm_rank(PETSC_COMM_WORLD,&rank));

  PetscCall(PetscOptionsGetInt(NULL, NULL, "-mx", &mx, NULL));
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-my", &my, NULL));
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-mz", &mz, NULL));
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-dim", &dim, NULL));
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-sliceid", &sliceid, NULL));
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-sliceaxis", &sliceaxis, NULL));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Create DMDA object.
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -*/
  if (dim==3) {
    PetscCall(DMDACreate3d(PETSC_COMM_WORLD,DM_BOUNDARY_NONE, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE,DMDA_STENCIL_STAR,mx, my, mz,
                           PETSC_DECIDE, PETSC_DECIDE, PETSC_DECIDE,1, 1,NULL, NULL, NULL,&da));
    PetscCall(DMSetFromOptions(da));
    PetscCall(DMSetUp(da));
  } else {
    PetscCall(DMDACreate2d(PETSC_COMM_WORLD,DM_BOUNDARY_NONE, DM_BOUNDARY_NONE,DMDA_STENCIL_STAR,mx, my,PETSC_DECIDE, PETSC_DECIDE,
                           1, 1,NULL, NULL,&da));
    PetscCall(DMSetFromOptions(da));
    PetscCall(DMSetUp(da));
  }

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Create the parent vector
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -*/
  PetscCall(DMCreateGlobalVector(da, &vec_full));
  PetscCall(PetscObjectSetName((PetscObject) vec_full, "full_vector"));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Populate the 3D vector
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  PetscCall(DMDAGetCorners(da, &ixs, &iys, &izs, &ixm, &iym, &izm));
  if (dim==3) {
    PetscCall(DMDAVecGetArray(da, vec_full, &vecdata3d));
    for (k=izs; k<izs+izm; k++) {
      for (j=iys; j<iys+iym; j++) {
        for (i=ixs; i<ixs+ixm; i++) {
          vecdata3d[k][j][i] = ((i-mx/2)*(j+mx/2))+k*100;
        }
      }
    }
    PetscCall(DMDAVecRestoreArray(da, vec_full, &vecdata3d));
  } else {
    PetscCall(DMDAVecGetArray(da, vec_full, &vecdata2d));
    for (j=iys; j<iys+iym; j++) {
      for (i=ixs; i<ixs+ixm; i++) {
        vecdata2d[j][i] = ((i-mx/2)*(j+mx/2));
      }
    }
    PetscCall(DMDAVecRestoreArray(da, vec_full, &vecdata2d));
  }

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Get an IS corresponding to a 2D slice
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  if (sliceaxis==0) {
    lower.i = sliceid; lower.j = 0;  lower.k = 0;  lower.c = 1;
    upper.i = sliceid; upper.j = my; upper.k = mz; upper.c = 1;
  } else if (sliceaxis==1) {
    lower.i = 0;  lower.j = sliceid; lower.k = 0;  lower.c = 1;
    upper.i = mx; upper.j = sliceid; upper.k = mz; upper.c = 1;
  } else if (sliceaxis==2 && dim==3) {
    lower.i = 0;  lower.j = 0;  lower.k = sliceid; lower.c = 1;
    upper.i = mx; upper.j = my; upper.k = sliceid; upper.c = 1;
  }
  PetscCall(DMDACreatePatchIS(da, &lower, &upper, &selectis, patchis_offproc));
  PetscCall(ISView(selectis, PETSC_VIEWER_STDOUT_WORLD));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Use the obtained IS to extract the slice as a subvector
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  PetscCall(VecGetSubVector(vec_full, selectis, &vec_slice));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     View the extracted subvector
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  PetscCall(PetscViewerPushFormat(PETSC_VIEWER_STDOUT_WORLD,PETSC_VIEWER_ASCII_DENSE));
  PetscCall(VecView(vec_slice, PETSC_VIEWER_STDOUT_WORLD));
  PetscCall(PetscViewerPopFormat(PETSC_VIEWER_STDOUT_WORLD));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Restore subvector, destroy data structures and exit.
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  PetscCall(VecRestoreSubVector(vec_full, selectis, &vec_slice));

  PetscCall(ISDestroy(&selectis));
  PetscCall(DMDestroy(&da));
  PetscCall(VecDestroy(&vec_full));

  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

    test:
      nsize: 1
      args: -sliceaxis 0

    test:
      suffix: 2
      nsize:  2
      args: -sliceaxis 1

    test:
      suffix: 3
      nsize:  4
      args:  -sliceaxis 2

    test:
      suffix: 4
      nsize:  2
      args: -sliceaxis 1 -dim 2

    test:
      suffix: 5
      nsize:  3
      args: -sliceaxis 0 -dim 2

TEST*/
