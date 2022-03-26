static char help[] ="Extract a 2D slice in natural ordering from a 3D vector, Command line options : \n\
Mx/My/Mz - set the dimensions of the parent vector \n\
sliceaxis - string describing the axis along which the sice will be selected : X, Y, Z \n\
gp - global grid point number along the sliceaxis direction where the slice will be extracted from the parent vector \n";

/*
  This example shows to extract a 2D slice in natural ordering
  from a 3D DMDA vector (first by extracting the slice and then
  by converting it to natural ordering)
*/

#include <petscdmda.h>

const char *const sliceaxes[] = {"X","Y","Z","sliceaxis","DM_",NULL};

int main(int argc,char **argv)
{
  DM                da3D;                          /* 3D DMDA object */
  DM                da2D;                          /* 2D DMDA object */
  Vec               vec_full;                      /* Parent vector */
  Vec               vec_extracted;                 /* Extracted slice vector (in DMDA ordering) */
  Vec               vec_slice;                     /* vector in natural ordering */
  Vec               vec_slice_g;                   /* aliased vector in natural ordering */
  IS                patchis_3d;                    /* IS to select slice and extract subvector */
  IS                patchis_2d;                    /* Patch IS for 2D vector, will be converted to application ordering */
  IS                scatis_extracted_slice;        /* PETSc indexed IS for extracted slice */
  IS                scatis_natural_slice;          /* natural/application ordered IS for slice*/
  IS                scatis_natural_slice_g;        /* aliased natural/application ordered IS  for slice */
  VecScatter        vscat;                         /* scatter slice in DMDA ordering <-> slice in column major ordering */
  AO                da2D_ao;                       /* AO associated with 2D DMDA */
  MPI_Comm          subset_mpi_comm=MPI_COMM_NULL; /* MPI communicator where the slice lives */
  PetscScalar       ***vecdata3d;                  /* Pointer to access 3d parent vector */
  const PetscScalar *array;                        /* pointer to create aliased Vec */
  PetscInt          Mx=4,My=4,Mz=4;                /* Dimensions for 3D DMDA */
  const PetscInt    *l1,*l2;                       /* 3D DMDA layout */
  PetscInt          M1=-1,M2=-1;                   /* Dimensions for 2D DMDA */
  PetscInt          m1=-1,m2=-1;                   /* Layouts for 2D DMDA */
  PetscInt          gp=2;                          /* grid point along sliceaxis to pick the slice */
  DMDirection       sliceaxis=DM_X;                /* Select axis along which the slice will be extracted */
  PetscInt          i,j,k;                         /* Iteration indices */
  PetscInt          ixs,iys,izs;                   /* Corner indices for 3D vector */
  PetscInt          ixm,iym,izm;                   /* Widths of parent vector */
  PetscInt          low, high;                     /* ownership range indices */
  PetscInt          in;                            /* local size index for IS*/
  PetscInt          vn;                            /* local size index */
  const PetscInt    *is_array;                     /* pointer to create aliased IS */
  MatStencil        lower, upper;                  /* Stencils to select slice for Vec */
  PetscBool         patchis_offproc = PETSC_FALSE; /* flag to DMDACreatePatchIS indicating that off-proc values are to be ignored */
  PetscMPIInt       rank,size;                     /* MPI rank and size */
  PetscErrorCode    ierr;                          /* error checking */

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Initialize program and set problem parameters
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  PetscCall(PetscInitialize(&argc, &argv, (char*)0, help));
  PetscCallMPI(MPI_Comm_size(PETSC_COMM_WORLD,&size));
  PetscCallMPI(MPI_Comm_rank(PETSC_COMM_WORLD,&rank));

  ierr = PetscOptionsBegin(PETSC_COMM_WORLD, "", "ex22 DMDA tutorial example options", "DMDA");PetscCall(ierr);
  PetscCall(PetscOptionsRangeInt("-Mx", "dimension along x-axis", "ex22.c", Mx, &Mx, NULL, 0, PETSC_MAX_INT));
  PetscCall(PetscOptionsRangeInt("-My", "dimension along y-axis", "ex22.c", My, &My, NULL, 0, PETSC_MAX_INT));
  PetscCall(PetscOptionsRangeInt("-Mz", "dimension along z-axis", "ex22.c", Mz, &Mz, NULL, 0, PETSC_MAX_INT));
  PetscCall(PetscOptionsEnum("-sliceaxis","axis along which 2D slice is extracted from : X, Y, Z","",sliceaxes,(PetscEnum)sliceaxis,(PetscEnum*)&sliceaxis,NULL));
  PetscCall(PetscOptionsRangeInt("-gp", "index along sliceaxis at which 2D slice is extracted", "ex22.c", gp, &gp, NULL, 0, PETSC_MAX_INT));
  ierr = PetscOptionsEnd();PetscCall(ierr);

  /* Ensure that the requested slice is not out of bounds for the selected axis */
  if (sliceaxis==DM_X) {
    PetscCheckFalse(gp>Mx,PETSC_COMM_WORLD, PETSC_ERR_USER_INPUT, "grid point along sliceaxis is larger than largest index!");
  } else if (sliceaxis==DM_Y) {
    PetscCheckFalse(gp>My,PETSC_COMM_WORLD, PETSC_ERR_USER_INPUT, "grid point along sliceaxis is larger than largest index!");
  } else if (sliceaxis==DM_Z) {
    PetscCheckFalse(gp>Mz,PETSC_COMM_WORLD, PETSC_ERR_USER_INPUT, "grid point along sliceaxis is larger than largest index!");
  }

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Create 3D DMDA object.
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -*/
  ierr = DMDACreate3d(PETSC_COMM_WORLD,
                      DM_BOUNDARY_NONE, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE,
                      DMDA_STENCIL_STAR,
                      Mx, My, Mz,
                      PETSC_DECIDE, PETSC_DECIDE, PETSC_DECIDE,
                      1, 1,
                      NULL, NULL, NULL,
                      &da3D);PetscCall(ierr);
  PetscCall(DMSetFromOptions(da3D));
  PetscCall(DMSetUp(da3D));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Create the parent vector
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -*/
  PetscCall(DMCreateGlobalVector(da3D, &vec_full));
  PetscCall(PetscObjectSetName((PetscObject) vec_full, "full_vector"));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Populate the 3D vector
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  PetscCall(DMDAGetCorners(da3D, &ixs, &iys, &izs, &ixm, &iym, &izm));
  PetscCall(DMDAVecGetArray(da3D, vec_full, &vecdata3d));
  for (k=izs; k<izs+izm; k++) {
    for (j=iys; j<iys+iym; j++) {
      for (i=ixs; i<ixs+ixm; i++) {
        vecdata3d[k][j][i] = ((i-Mx/2.0)*(j+Mx/2.0))+k*100;
      }
    }
  }
  PetscCall(DMDAVecRestoreArray(da3D, vec_full, &vecdata3d));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Get an IS corresponding to a 2D slice
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  if (sliceaxis==DM_X) {
    lower.i = gp; lower.j = 0;  lower.k = 0;
    upper.i = gp; upper.j = My; upper.k = Mz;
  } else if (sliceaxis==DM_Y) {
    lower.i = 0;  lower.j = gp; lower.k = 0;
    upper.i = Mx; upper.j = gp; upper.k = Mz;
  } else if (sliceaxis==DM_Z) {
    lower.i = 0;  lower.j = 0;  lower.k = gp;
    upper.i = Mx; upper.j = My; upper.k = gp;
  }
  PetscCall(DMDACreatePatchIS(da3D, &lower, &upper, &patchis_3d, patchis_offproc));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "\n IS to select slice from 3D DMDA vector : \n"));
  PetscCall(ISView(patchis_3d, PETSC_VIEWER_STDOUT_WORLD));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Use the obtained IS to extract the slice as a subvector
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  PetscCall(VecGetSubVector(vec_full, patchis_3d, &vec_extracted));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     View the extracted subvector
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  PetscCall(PetscViewerPushFormat(PETSC_VIEWER_STDOUT_WORLD,PETSC_VIEWER_ASCII_DENSE));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "\n Extracted slice vector, in DMDA ordering : \n"));
  PetscCall(VecView(vec_extracted, PETSC_VIEWER_STDOUT_WORLD));
  PetscCall(PetscViewerPopFormat(PETSC_VIEWER_STDOUT_WORLD));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Query 3D DMDA layout, get the subset MPI communicator
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -*/
  if (sliceaxis==DM_X) {
    PetscCall(DMDAGetInfo(da3D, NULL, NULL, NULL, NULL, NULL, &m1, &m2, NULL, NULL, NULL, NULL, NULL, NULL));
    PetscCall(DMDAGetOwnershipRanges(da3D, NULL, &l1, &l2));
    M1 = My; M2 = Mz;
    PetscCall(DMDAGetProcessorSubset(da3D, DM_X, gp, &subset_mpi_comm));
  } else if (sliceaxis==DM_Y) {
    PetscCall(DMDAGetInfo(da3D, NULL, NULL, NULL, NULL, &m1, NULL, &m2, NULL, NULL, NULL, NULL, NULL, NULL));
    PetscCall(DMDAGetOwnershipRanges(da3D, &l1, NULL, &l2));
    M1 = Mx; M2 = Mz;
    PetscCall(DMDAGetProcessorSubset(da3D, DM_Y, gp, &subset_mpi_comm));
  } else if (sliceaxis==DM_Z) {
    PetscCall(DMDAGetInfo(da3D, NULL, NULL, NULL, NULL, &m1, &m2, NULL, NULL, NULL, NULL, NULL, NULL, NULL));
    PetscCall(DMDAGetOwnershipRanges(da3D, &l1, &l2, NULL));
    M1 = Mx; M2 = My;
    PetscCall(DMDAGetProcessorSubset(da3D, DM_Z, gp, &subset_mpi_comm));
  }

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Create 2D DMDA object,
     vector (that will hold the slice as a column major flattened array) &
     index set (that will be used for scattering to the column major
     indexed slice vector)
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -*/
  if (subset_mpi_comm != MPI_COMM_NULL) {
    PetscCallMPI(MPI_Comm_size(subset_mpi_comm, &size));
    PetscCall(PetscSynchronizedPrintf(subset_mpi_comm, "subset MPI subcomm size is : %d, includes global rank : %d \n", size, rank));
    PetscCall(PetscSynchronizedFlush(subset_mpi_comm, PETSC_STDOUT));
    ierr = DMDACreate2d(subset_mpi_comm,
                        DM_BOUNDARY_NONE, DM_BOUNDARY_NONE,
                        DMDA_STENCIL_STAR,
                        M1, M2,
                        m1, m2,
                        1, 1,
                        l1, l2,
                        &da2D);PetscCall(ierr);
    PetscCall(DMSetFromOptions(da2D));
    PetscCall(DMSetUp(da2D));

    /* Create a 2D patch IS for the slice */
    lower.i = 0;  lower.j = 0;
    upper.i = M1; upper.j = M2;
    PetscCall(DMDACreatePatchIS(da2D, &lower, &upper, &patchis_2d, patchis_offproc));

    /* Convert the 2D patch IS to natural indexing (column major flattened) */
    PetscCall(ISDuplicate(patchis_2d, &scatis_natural_slice));
    PetscCall(DMDAGetAO(da2D, &da2D_ao));
    PetscCall(AOPetscToApplicationIS(da2D_ao, scatis_natural_slice));
    PetscCall(ISGetIndices(scatis_natural_slice, &is_array));
    PetscCall(ISGetLocalSize(scatis_natural_slice, &in));

    /* Create an aliased IS on the 3D DMDA's communicator */
    PetscCall(ISCreateGeneral(PETSC_COMM_WORLD, in, is_array, PETSC_USE_POINTER, &scatis_natural_slice_g));
    PetscCall(ISRestoreIndices(scatis_natural_slice, &is_array));

    /* Create a 2D DMDA global vector */
    PetscCall(DMCreateGlobalVector(da2D, &vec_slice));
    PetscCall(PetscObjectSetName((PetscObject) vec_slice, "slice_vector_natural"));
    PetscCall(VecGetLocalSize(vec_slice ,&vn));
    PetscCall(VecGetArrayRead(vec_slice, &array));

    /* Create an aliased version of the above on the 3D DMDA's communicator */
    PetscCall(VecCreateMPIWithArray(PETSC_COMM_WORLD, 1, vn, M1*M2, array, &vec_slice_g));
    PetscCall(VecRestoreArrayRead(vec_slice, &array));
  } else {
    /* Ranks not part of the subset MPI communicator provide no entries, but the routines for creating
       the IS and Vec on the 3D DMDA's communicator still need to called, since they are collective routines */
    PetscCall(ISCreateGeneral(PETSC_COMM_WORLD, 0, NULL, PETSC_USE_POINTER, &scatis_natural_slice_g));
    PetscCall(VecCreateMPIWithArray(PETSC_COMM_WORLD, 1, 0, M1*M2, NULL, &vec_slice_g));
  }
  PetscCall(PetscBarrier(NULL));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Create IS that maps from the extracted slice vector
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  PetscCall(VecGetOwnershipRange(vec_extracted, &low, &high));
  PetscCall(ISCreateStride(PETSC_COMM_WORLD, high-low, low, 1, &scatis_extracted_slice));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Scatter extracted subvector -> natural 2D slice vector
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  PetscCall(VecScatterCreate(vec_extracted, scatis_extracted_slice, vec_slice_g, scatis_natural_slice_g, &vscat));
  PetscCall(VecScatterBegin(vscat, vec_extracted, vec_slice_g, INSERT_VALUES, SCATTER_FORWARD));
  PetscCall(VecScatterEnd(vscat, vec_extracted, vec_slice_g, INSERT_VALUES, SCATTER_FORWARD));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     View the natural 2D slice vector
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  PetscCall(PetscViewerPushFormat(PETSC_VIEWER_STDOUT_WORLD,PETSC_VIEWER_ASCII_DENSE));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "\n Extracted slice vector, in natural ordering : \n"));
  PetscCall(VecView(vec_slice_g, PETSC_VIEWER_STDOUT_WORLD));
  PetscCall(PetscViewerPopFormat(PETSC_VIEWER_STDOUT_WORLD));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Restore subvector
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  PetscCall(VecRestoreSubVector(vec_full, patchis_3d, &vec_extracted));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Destroy data structures and exit.
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  PetscCall(VecDestroy(&vec_full));
  PetscCall(VecScatterDestroy(&vscat));
  PetscCall(ISDestroy(&scatis_extracted_slice));
  PetscCall(ISDestroy(&scatis_natural_slice_g));
  PetscCall(VecDestroy(&vec_slice_g));
  PetscCall(ISDestroy(&patchis_3d));
  PetscCall(DMDestroy(&da3D));

  if (subset_mpi_comm != MPI_COMM_NULL) {
    PetscCall(ISDestroy(&patchis_2d));
    PetscCall(ISDestroy(&scatis_natural_slice));
    PetscCall(VecDestroy(&vec_slice));
    PetscCall(DMDestroy(&da2D));
    PetscCallMPI(MPI_Comm_free(&subset_mpi_comm));
  }

  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

    test:
      nsize: 1
      args: -sliceaxis X -gp 0

    test:
      suffix: 2
      nsize:  2
      args: -sliceaxis Y -gp 1
      filter: grep -v "subset MPI subcomm"

    test:
      suffix: 3
      nsize:  3
      args:  -sliceaxis Z -gp 2
      filter: grep -v "subset MPI subcomm"

    test:
      suffix: 4
      nsize:  4
      args: -sliceaxis X -gp 2
      filter: grep -v "subset MPI subcomm"

    test:
      suffix: 5
      nsize:  4
      args: -sliceaxis Z -gp 1
      filter: grep -v "subset MPI subcomm"

TEST*/
