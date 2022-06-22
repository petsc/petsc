
static char help[] = "Tests I/O of vectors for different data formats (binary,HDF5)\n\n";

#include <petscvec.h>
#include <petscdevice.h>
#include <petscviewerhdf5.h>

/* Note:  Most applications would not read and write a vector within
  the same program.  This example is intended only to demonstrate
  both input and output and is written for use with either 1,2,or 4 processors. */

int main(int argc,char **args)
{
  PetscMPIInt       rank,size;
  PetscInt          i,m = 20,low,high,ldim,iglobal,lsize;
  PetscScalar       v;
  Vec               u;
  PetscViewer       viewer;
  PetscBool         vstage2,vstage3,mpiio_use,isbinary = PETSC_FALSE;
  VecType           vectype;
#if defined(PETSC_HAVE_HDF5)
  PetscBool         ishdf5 = PETSC_FALSE;
#endif
#if defined(PETSC_HAVE_ADIOS)
  PetscBool         isadios = PETSC_FALSE;
#endif
  PetscScalar const *values;

  PetscCall(PetscInitialize(&argc,&args,(char*)0,help));
  {
    PetscDeviceContext dctx; /* unused, only there to force initialization of device */

    PetscCall(PetscDeviceContextGetCurrentContext(&dctx));
  }

  mpiio_use = vstage2 = vstage3 = PETSC_FALSE;

  PetscCall(PetscOptionsGetBool(NULL,NULL,"-binary",&isbinary,NULL));
#if defined(PETSC_HAVE_HDF5)
  PetscCall(PetscOptionsGetBool(NULL,NULL,"-hdf5",&ishdf5,NULL));
#endif
#if defined(PETSC_HAVE_ADIOS)
  PetscCall(PetscOptionsGetBool(NULL,NULL,"-adios",&isadios,NULL));
#endif
  PetscCall(PetscOptionsGetBool(NULL,NULL,"-mpiio",&mpiio_use,NULL));
  PetscCall(PetscOptionsGetBool(NULL,NULL,"-sizes_set",&vstage2,NULL));
  PetscCall(PetscOptionsGetBool(NULL,NULL,"-type_set",&vstage3,NULL));

  PetscCallMPI(MPI_Comm_rank(PETSC_COMM_WORLD,&rank));
  PetscCallMPI(MPI_Comm_size(PETSC_COMM_WORLD,&size));
  PetscCall(PetscOptionsGetInt(NULL,NULL,"-m",&m,NULL));

  /* PART 1:  Generate vector, then write it in the given data format */

  /* Generate vector */
  PetscCall(VecCreate(PETSC_COMM_WORLD,&u));
  PetscCall(VecSetType(u, VECCUDA));
  PetscCall(PetscObjectSetName((PetscObject)u, "Test_Vec"));
  PetscCall(VecSetSizes(u,PETSC_DECIDE,m));
  PetscCall(VecSetFromOptions(u));
  PetscCall(VecGetOwnershipRange(u,&low,&high));
  PetscCall(VecGetLocalSize(u,&ldim));
  for (i=0; i<ldim; i++) {
    iglobal = i + low;
    v       = (PetscScalar)(i + low);
    PetscCall(VecSetValues(u,1,&iglobal,&v,INSERT_VALUES));
  }
  PetscCall(VecAssemblyBegin(u));
  PetscCall(VecAssemblyEnd(u));
  PetscCall(VecView(u,PETSC_VIEWER_STDOUT_WORLD));

  if (isbinary) {
    PetscCall(PetscPrintf(PETSC_COMM_WORLD,"writing vector in binary to vector.dat ...\n"));
    PetscCall(PetscViewerBinaryOpen(PETSC_COMM_WORLD,"vector.dat",FILE_MODE_WRITE,&viewer));
#if defined(PETSC_HAVE_HDF5)
  } else if (ishdf5) {
    PetscCall(PetscPrintf(PETSC_COMM_WORLD,"writing vector in hdf5 to vector.dat ...\n"));
    PetscCall(PetscViewerHDF5Open(PETSC_COMM_WORLD,"vector.dat",FILE_MODE_WRITE,&viewer));
#endif
#if defined(PETSC_HAVE_ADIOS)
  } else if (isadios) {
    PetscCall(PetscPrintf(PETSC_COMM_WORLD,"writing vector in adios to vector.dat ...\n"));
    PetscCall(PetscViewerADIOSOpen(PETSC_COMM_WORLD,"vector.dat",FILE_MODE_WRITE,&viewer));
#endif
  } else SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_SUP,"No data format specified, run with one of -binary -hdf5 -adios options");
  PetscCall(VecView(u,viewer));
  PetscCall(PetscViewerDestroy(&viewer));
  PetscCall(VecDestroy(&u));

  /* PART 2:  Read in vector in binary format */
  /* Read new vector in binary format */
  if (mpiio_use) {
    PetscCall(PetscPrintf(PETSC_COMM_WORLD,"Using MPI IO for reading the vector\n"));
    PetscCall(PetscOptionsSetValue(NULL,"-viewer_binary_mpiio",""));
  }
  if (isbinary) {
    PetscCall(PetscPrintf(PETSC_COMM_WORLD,"reading vector in binary from vector.dat ...\n"));
    PetscCall(PetscViewerBinaryOpen(PETSC_COMM_WORLD,"vector.dat",FILE_MODE_READ,&viewer));
    PetscCall(PetscViewerBinarySetFlowControl(viewer,2));
#if defined(PETSC_HAVE_HDF5)
  } else if (ishdf5) {
    PetscCall(PetscPrintf(PETSC_COMM_WORLD,"reading vector in hdf5 from vector.dat ...\n"));
    PetscCall(PetscViewerHDF5Open(PETSC_COMM_WORLD,"vector.dat",FILE_MODE_READ,&viewer));
#endif
#if defined(PETSC_HAVE_ADIOS)
  } else if (isadios) {
    PetscCall(PetscPrintf(PETSC_COMM_WORLD,"reading vector in adios from vector.dat ...\n"));
    PetscCall(PetscViewerADIOSOpen(PETSC_COMM_WORLD,"vector.dat",FILE_MODE_READ,&viewer));
#endif
  }
  PetscCall(VecCreate(PETSC_COMM_WORLD,&u));
  PetscCall(PetscObjectSetName((PetscObject) u,"Test_Vec"));
  if (vstage2) {
    PetscCall(PetscPrintf(PETSC_COMM_WORLD,"Setting vector sizes...\n"));
    if (size > 1) {
      if (rank == 0) {
        lsize = m/size + size;
        PetscCall(VecSetSizes(u,lsize,m));
      } else if (rank == size-1) {
        lsize = PetscMax(m/size - size,0);
        PetscCall(VecSetSizes(u,lsize,m));
      } else {
        lsize = m/size;
        PetscCall(VecSetSizes(u,lsize,m));
      }
    } else {
      PetscCall(VecSetSizes(u,m,m));
    }
  }

  PetscCall(PetscPrintf(PETSC_COMM_WORLD,"Setting vector type...\n"));
  PetscCall(VecSetType(u, VECCUDA));
  PetscCall(VecGetType(u, &vectype));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Before load, vectype is : %s\n", (char*)vectype));
  PetscCall(VecLoad(u,viewer));
  PetscCall(VecGetType(u, &vectype));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "After load, vectype is : %s\n", (char*)vectype));
  PetscCall(PetscViewerDestroy(&viewer));
  PetscCall(VecView(u,PETSC_VIEWER_STDOUT_WORLD));
  PetscCall(VecGetArrayRead(u,&values));
  PetscCall(VecGetLocalSize(u,&ldim));
  PetscCall(VecGetOwnershipRange(u,&low,NULL));
  for (i=0; i<ldim; i++) {
    PetscCheck(values[i] == (PetscScalar)(i + low),PETSC_COMM_WORLD,PETSC_ERR_SUP,"Data check failed!");
  }
  PetscCall(VecRestoreArrayRead(u,&values));

  /* Free data structures */
  PetscCall(VecDestroy(&u));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

     build:
       requires: cuda

     test:
       nsize: 2
       args: -binary

     test:
       suffix: 2
       nsize: 3
       args: -binary

     test:
       suffix: 3
       nsize: 5
       args: -binary

     test:
       suffix: 4
       requires: hdf5
       nsize: 2
       args: -hdf5

     test:
       suffix: 5
       nsize: 4
       args: -binary -sizes_set

     test:
       suffix: 6
       requires: hdf5
       nsize: 4
       args: -hdf5 -sizes_set

TEST*/
