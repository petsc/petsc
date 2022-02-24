
static char help[] = "Tests I/O of vectors for different data formats (binary,HDF5) and illustrates the use of user-defined event logging\n\n";

#include <petscvec.h>
#include <petscviewerhdf5.h>

/* Note:  Most applications would not read and write a vector within
  the same program.  This example is intended only to demonstrate
  both input and output and is written for use with either 1,2,or 4 processors. */

int main(int argc,char **args)
{
  PetscErrorCode    ierr;
  PetscMPIInt       rank,size;
  PetscInt          i,m = 20,low,high,ldim,iglobal,lsize;
  PetscScalar       v;
  Vec               u;
  PetscViewer       viewer;
  PetscBool         vstage2,vstage3,mpiio_use,isbinary = PETSC_FALSE;
#if defined(PETSC_HAVE_HDF5)
  PetscBool         ishdf5 = PETSC_FALSE;
#endif
#if defined(PETSC_HAVE_ADIOS)
  PetscBool         isadios = PETSC_FALSE;
#endif
  PetscScalar const *values;
#if defined(PETSC_USE_LOG)
  PetscLogEvent  VECTOR_GENERATE,VECTOR_READ;
#endif

  ierr = PetscInitialize(&argc,&args,(char*)0,help);if (ierr) return ierr;
  mpiio_use = vstage2 = vstage3 = PETSC_FALSE;

  CHKERRQ(PetscOptionsGetBool(NULL,NULL,"-binary",&isbinary,NULL));
#if defined(PETSC_HAVE_HDF5)
  CHKERRQ(PetscOptionsGetBool(NULL,NULL,"-hdf5",&ishdf5,NULL));
#endif
#if defined(PETSC_HAVE_ADIOS)
  CHKERRQ(PetscOptionsGetBool(NULL,NULL,"-adios",&isadios,NULL));
#endif
  CHKERRQ(PetscOptionsGetBool(NULL,NULL,"-mpiio",&mpiio_use,NULL));
  CHKERRQ(PetscOptionsGetBool(NULL,NULL,"-sizes_set",&vstage2,NULL));
  CHKERRQ(PetscOptionsGetBool(NULL,NULL,"-type_set",&vstage3,NULL));

  CHKERRMPI(MPI_Comm_rank(PETSC_COMM_WORLD,&rank));
  CHKERRMPI(MPI_Comm_size(PETSC_COMM_WORLD,&size));
  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-m",&m,NULL));

  /* PART 1:  Generate vector, then write it in the given data format */

  CHKERRQ(PetscLogEventRegister("Generate Vector",VEC_CLASSID,&VECTOR_GENERATE));
  CHKERRQ(PetscLogEventBegin(VECTOR_GENERATE,0,0,0,0));
  /* Generate vector */
  CHKERRQ(VecCreate(PETSC_COMM_WORLD,&u));
  CHKERRQ(PetscObjectSetName((PetscObject)u, "Test_Vec"));
  CHKERRQ(VecSetSizes(u,PETSC_DECIDE,m));
  CHKERRQ(VecSetFromOptions(u));
  CHKERRQ(VecGetOwnershipRange(u,&low,&high));
  CHKERRQ(VecGetLocalSize(u,&ldim));
  for (i=0; i<ldim; i++) {
    iglobal = i + low;
    v       = (PetscScalar)(i + low);
    CHKERRQ(VecSetValues(u,1,&iglobal,&v,INSERT_VALUES));
  }
  CHKERRQ(VecAssemblyBegin(u));
  CHKERRQ(VecAssemblyEnd(u));
  CHKERRQ(VecView(u,PETSC_VIEWER_STDOUT_WORLD));

  if (isbinary) {
    CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"writing vector in binary to vector.dat ...\n"));
    CHKERRQ(PetscViewerBinaryOpen(PETSC_COMM_WORLD,"vector.dat",FILE_MODE_WRITE,&viewer));
#if defined(PETSC_HAVE_HDF5)
  } else if (ishdf5) {
    CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"writing vector in hdf5 to vector.dat ...\n"));
    CHKERRQ(PetscViewerHDF5Open(PETSC_COMM_WORLD,"vector.dat",FILE_MODE_WRITE,&viewer));
#endif
#if defined(PETSC_HAVE_ADIOS)
  } else if (isadios) {
    CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"writing vector in adios to vector.dat ...\n"));
    CHKERRQ(PetscViewerADIOSOpen(PETSC_COMM_WORLD,"vector.dat",FILE_MODE_WRITE,&viewer));
#endif
  } else SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_SUP,"No data format specified, run with one of -binary -hdf5 -adios options");
  CHKERRQ(VecView(u,viewer));
  CHKERRQ(PetscViewerDestroy(&viewer));
  CHKERRQ(VecDestroy(&u));

  CHKERRQ(PetscLogEventEnd(VECTOR_GENERATE,0,0,0,0));

  /* PART 2:  Read in vector in binary format */

  /* Read new vector in binary format */
  CHKERRQ(PetscLogEventRegister("Read Vector",VEC_CLASSID,&VECTOR_READ));
  CHKERRQ(PetscLogEventBegin(VECTOR_READ,0,0,0,0));
  if (mpiio_use) {
    CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"Using MPI IO for reading the vector\n"));
    CHKERRQ(PetscOptionsSetValue(NULL,"-viewer_binary_mpiio",""));
  }
  if (isbinary) {
    CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"reading vector in binary from vector.dat ...\n"));
    CHKERRQ(PetscViewerBinaryOpen(PETSC_COMM_WORLD,"vector.dat",FILE_MODE_READ,&viewer));
    CHKERRQ(PetscViewerBinarySetFlowControl(viewer,2));
#if defined(PETSC_HAVE_HDF5)
  } else if (ishdf5) {
    CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"reading vector in hdf5 from vector.dat ...\n"));
    CHKERRQ(PetscViewerHDF5Open(PETSC_COMM_WORLD,"vector.dat",FILE_MODE_READ,&viewer));
#endif
#if defined(PETSC_HAVE_ADIOS)
  } else if (isadios) {
    CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"reading vector in adios from vector.dat ...\n"));
    CHKERRQ(PetscViewerADIOSOpen(PETSC_COMM_WORLD,"vector.dat",FILE_MODE_READ,&viewer));
#endif
  }
  CHKERRQ(VecCreate(PETSC_COMM_WORLD,&u));
  CHKERRQ(PetscObjectSetName((PetscObject) u,"Test_Vec"));

  if (vstage2) {
    CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"Setting vector sizes...\n"));
    if (size > 1) {
      if (rank == 0) {
        lsize = m/size + size;
        CHKERRQ(VecSetSizes(u,lsize,m));
      } else if (rank == size-1) {
        lsize = m/size - size;
        CHKERRQ(VecSetSizes(u,lsize,m));
      } else {
        lsize = m/size;
        CHKERRQ(VecSetSizes(u,lsize,m));
      }
    } else {
      CHKERRQ(VecSetSizes(u,m,m));
    }
  }

  if (vstage3) {
    CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"Setting vector type...\n"));
    CHKERRQ(VecSetType(u, VECMPI));
  }
  CHKERRQ(VecLoad(u,viewer));
  CHKERRQ(PetscViewerDestroy(&viewer));
  CHKERRQ(PetscLogEventEnd(VECTOR_READ,0,0,0,0));
  CHKERRQ(VecView(u,PETSC_VIEWER_STDOUT_WORLD));
  CHKERRQ(VecGetArrayRead(u,&values));
  CHKERRQ(VecGetLocalSize(u,&ldim));
  CHKERRQ(VecGetOwnershipRange(u,&low,NULL));
  for (i=0; i<ldim; i++) {
    PetscCheckFalse(values[i] != (PetscScalar)(i + low),PETSC_COMM_WORLD,PETSC_ERR_SUP,"Data check failed!");
  }
  CHKERRQ(VecRestoreArrayRead(u,&values));

  /* Free data structures */
  CHKERRQ(VecDestroy(&u));
  ierr = PetscFinalize();
  return ierr;
}

/*TEST

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
