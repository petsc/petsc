
static char help[] = "Tests I/O of vectors for different data formats (binary,HDF5,NetCDF) and illustrates the use of user-defined event logging\n\n";

#include <petscvec.h>

/* Note:  Most applications would not read and write a vector within
  the same program.  This example is intended only to demonstrate
  both input and output and is written for use with either 1,2,or 4 processors. */

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc,char **args)
{
  PetscErrorCode ierr;
  PetscMPIInt    rank,size;
  PetscInt       i,m = 20,low,high,ldim,iglobal,lsize;
  PetscScalar    v;
  Vec            u;
  PetscViewer    viewer;
  PetscBool      vstage2,vstage3,mpiio_use,isbinary,ishdf5;
#if defined(PETSC_USE_LOG)
  PetscLogEvent  VECTOR_GENERATE,VECTOR_READ;
#endif

  PetscInitialize(&argc,&args,(char *)0,help);
  isbinary = ishdf5 = PETSC_FALSE;
  mpiio_use = vstage2 = vstage3 = PETSC_FALSE;
  ierr = PetscOptionsGetBool(PETSC_NULL,"-binary",&isbinary,PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetBool(PETSC_NULL,"-hdf5",&ishdf5,PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetBool(PETSC_NULL,"-mpiio",&mpiio_use,PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetBool(PETSC_NULL,"-sizes_set",&vstage2,PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetBool(PETSC_NULL,"-type_set",&vstage3,PETSC_NULL);CHKERRQ(ierr);

  ierr = MPI_Comm_rank(PETSC_COMM_WORLD,&rank);CHKERRQ(ierr);
  ierr = MPI_Comm_size(PETSC_COMM_WORLD,&size);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(PETSC_NULL,"-m",&m,PETSC_NULL);CHKERRQ(ierr);

  /* PART 1:  Generate vector, then write it in the given data format */

  ierr = PetscLogEventRegister("Generate Vector",VEC_CLASSID,&VECTOR_GENERATE);CHKERRQ(ierr);
  ierr = PetscLogEventBegin(VECTOR_GENERATE,0,0,0,0);CHKERRQ(ierr);
  /* Generate vector */
  ierr = VecCreate(PETSC_COMM_WORLD,&u);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject)u, "Test_Vec");CHKERRQ(ierr);
  ierr = VecSetSizes(u,PETSC_DECIDE,m);CHKERRQ(ierr);
  ierr = VecSetFromOptions(u);CHKERRQ(ierr);
  ierr = VecGetOwnershipRange(u,&low,&high);CHKERRQ(ierr);
  ierr = VecGetLocalSize(u,&ldim);CHKERRQ(ierr);
  for (i=0; i<ldim; i++) {
    iglobal = i + low;
    v = (PetscScalar)(i + 100*rank);
    ierr = VecSetValues(u,1,&iglobal,&v,INSERT_VALUES);CHKERRQ(ierr);
  }
  ierr = VecAssemblyBegin(u);CHKERRQ(ierr);
  ierr = VecAssemblyEnd(u);CHKERRQ(ierr);
  ierr = VecView(u,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);

  if (isbinary) {
    ierr = PetscPrintf(PETSC_COMM_WORLD,"writing vector in binary to vector.dat ...\n");CHKERRQ(ierr);
    ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,"vector.dat",FILE_MODE_WRITE,&viewer);CHKERRQ(ierr);
#if defined(PETSC_HAVE_HDF5)
  } else if (ishdf5) {
    ierr = PetscPrintf(PETSC_COMM_WORLD,"writing vector in hdf5 to vector.dat ...\n");CHKERRQ(ierr);
    ierr = PetscViewerHDF5Open(PETSC_COMM_WORLD,"vector.dat",FILE_MODE_WRITE,&viewer);CHKERRQ(ierr);
#endif
  } else {
    SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_SUP,"No data format specified, run with either -binary or -hdf5 option\n");
  }
  ierr = VecView(u,viewer);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);
  ierr = VecDestroy(&u);CHKERRQ(ierr);
  /*  ierr = PetscOptionsClear();CHKERRQ(ierr); */


  ierr = PetscLogEventEnd(VECTOR_GENERATE,0,0,0,0);CHKERRQ(ierr);

  /* PART 2:  Read in vector in binary format */

  /* Read new vector in binary format */
  ierr = PetscLogEventRegister("Read Vector",VEC_CLASSID,&VECTOR_READ);CHKERRQ(ierr);
  ierr = PetscLogEventBegin(VECTOR_READ,0,0,0,0);CHKERRQ(ierr);
  if (mpiio_use) {
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Using MPI IO for reading the vector\n");CHKERRQ(ierr);
    ierr = PetscOptionsSetValue("-viewer_binary_mpiio","");CHKERRQ(ierr);
  }
  if (isbinary) {
    ierr = PetscPrintf(PETSC_COMM_WORLD,"reading vector in binary from vector.dat ...\n");CHKERRQ(ierr);
    ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,"vector.dat",FILE_MODE_READ,&viewer);CHKERRQ(ierr);
    ierr = PetscViewerBinarySetFlowControl(viewer,2);CHKERRQ(ierr);
#if defined(PETSC_HAVE_HDF5)
  } else if (ishdf5) {
    ierr = PetscPrintf(PETSC_COMM_WORLD,"reading vector in hdf5 from vector.dat ...\n");CHKERRQ(ierr);
    ierr = PetscViewerHDF5Open(PETSC_COMM_WORLD,"vector.dat",FILE_MODE_READ,&viewer);CHKERRQ(ierr);
#endif
  }
  ierr = VecCreate(PETSC_COMM_WORLD,&u);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) u,"Test_Vec");

  if (vstage2) {
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Setting vector sizes...\n");CHKERRQ(ierr);
    if (size > 1) {
      if (!rank) {
	lsize = m/size + size;
	ierr = VecSetSizes(u,lsize,m);CHKERRQ(ierr);
      }
      else if (rank == size-1) {
	lsize = m/size - size;
	ierr = VecSetSizes(u,lsize,m);CHKERRQ(ierr);
      }
      else {
	lsize = m/size;
	ierr = VecSetSizes(u,lsize,m);CHKERRQ(ierr);
      }
    } else {
      ierr = VecSetSizes(u,m,m);CHKERRQ(ierr);
    }
  }

  if (vstage3) {
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Setting vector type...\n");CHKERRQ(ierr);
    ierr = VecSetType(u, VECMPI);CHKERRQ(ierr);
  }
  ierr = VecLoad(u,viewer);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);
  ierr = PetscLogEventEnd(VECTOR_READ,0,0,0,0);CHKERRQ(ierr);
  ierr = VecView(u,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);

  /* Free data structures */
  ierr = VecDestroy(&u);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return 0;
}

