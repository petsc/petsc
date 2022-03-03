static const char help[] = "Test ParMETIS handling of negative weights.\n\n";

/* Test contributed by John Fettig */

/*
 * This file implements two tests for a bug reported in ParMETIS. These tests are not expected to pass without the
 * patches in the PETSc distribution of ParMetis. See parmetis.py
 *
 *
 * The bug was reported upstream, but has received no action so far.
 *
 * http://glaros.dtc.umn.edu/gkhome/node/837
 *
 */

#include <petscsys.h>
#include <parmetis.h>

#define CHKERRQPARMETIS(...) do {                                                              \
    int metis_ierr = __VA_ARGS__;                                                              \
    PetscCheck(metis_ierr != METIS_ERROR_INPUT,PETSC_COMM_SELF,PETSC_ERR_LIB,"ParMETIS error due to wrong inputs and/or options"); \
    PetscCheck(metis_ierr != METIS_ERROR_MEMORY,PETSC_COMM_SELF,PETSC_ERR_LIB,"ParMETIS error due to insufficient memory"); \
    PetscCheck(metis_ierr != METIS_ERROR,PETSC_COMM_SELF,PETSC_ERR_LIB,"ParMETIS general error"); \
  } while (0)

int main(int argc, char *argv[])
{
  PetscErrorCode ierr;
  PetscBool      flg;
  PetscMPIInt    rank, size;
  idx_t          ni,isize,*vtxdist, *xadj, *adjncy, *vwgt, *part;
  idx_t          wgtflag=0, numflag=0, ncon=1, ndims=3, edgecut=0;
  idx_t          options[5];
  PetscReal      *xyz;
  real_t         *sxyz, *tpwgts, ubvec[1];
  MPI_Comm       comm;
  FILE           *fp;
  char           fname[PETSC_MAX_PATH_LEN],prefix[PETSC_MAX_PATH_LEN] = "";
  size_t         red;

  ierr = PetscInitialize(&argc,&argv,NULL,help);if (ierr) return ierr;
#if defined(PETSC_USE_64BIT_INDICES)
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"This example only works with 32 bit indices\n"));
  ierr = PetscFinalize();
  return ierr;
#endif
  MPI_Comm_rank(PETSC_COMM_WORLD,&rank);
  MPI_Comm_size(PETSC_COMM_WORLD,&size);

  ierr = PetscOptionsBegin(PETSC_COMM_WORLD,NULL,"Parmetis test options","");CHKERRQ(ierr);
  CHKERRQ(PetscOptionsString("-prefix","Path and prefix of test file","",prefix,prefix,sizeof(prefix),&flg));
  PetscCheck(flg,PETSC_COMM_WORLD,PETSC_ERR_USER,"Must specify -prefix");
  ierr = PetscOptionsEnd();CHKERRQ(ierr);

  CHKERRQ(PetscMalloc1(size+1,&vtxdist));

  CHKERRQ(PetscSNPrintf(fname,sizeof(fname),"%s.%d.graph",prefix,rank));

  CHKERRQ(PetscFOpen(PETSC_COMM_SELF,fname,"r",&fp));

  red = fread(vtxdist, sizeof(idx_t), size+1, fp);PetscCheck(red == (size_t) (size+1),PETSC_COMM_SELF,PETSC_ERR_SYS,"Unable to read from data file");

  ni = vtxdist[rank+1]-vtxdist[rank];

  CHKERRQ(PetscMalloc1(ni+1,&xadj));

  red = fread(xadj, sizeof(idx_t), ni+1, fp);PetscCheck(red == (size_t) (ni+1),PETSC_COMM_SELF,PETSC_ERR_SYS,"Unable to read from data file");

  CHKERRQ(PetscMalloc1(xadj[ni],&adjncy));

  for (PetscInt i=0; i<ni; i++) {
    red = fread(&adjncy[xadj[i]], sizeof(idx_t), xadj[i+1]-xadj[i], fp);PetscCheck(red == (size_t) (xadj[i+1]-xadj[i]),PETSC_COMM_SELF,PETSC_ERR_SYS,"Unable to read from data file");
  }

  CHKERRQ(PetscFClose(PETSC_COMM_SELF,fp));

  CHKERRQ(PetscSNPrintf(fname,sizeof(fname),"%s.%d.graph.xyz",prefix,rank));
  CHKERRQ(PetscFOpen(PETSC_COMM_SELF,fname,"r",&fp));

  CHKERRQ(PetscMalloc3(ni*ndims,&xyz,ni,&part,size,&tpwgts));
  CHKERRQ(PetscMalloc1(ni*ndims,&sxyz));

  red = fread(xyz, sizeof(PetscReal), ndims*ni, fp);PetscCheck(red == (size_t) (ndims*ni),PETSC_COMM_SELF,PETSC_ERR_SYS,"Unable to read from data file");
  for (PetscInt i=0; i<ni*ndims; i++) sxyz[i] = (size_t) xyz[i];

  CHKERRQ(PetscFClose(PETSC_COMM_SELF,fp));

  vwgt = NULL;

  for (PetscInt i = 0; i < size; i++) tpwgts[i] = 1. / size;
  isize = size;

  ubvec[0]   = 1.05;
  options[0] = 0;
  options[1] = 2;
  options[2] = 15;
  options[3] = 0;
  options[4] = 0;

  CHKERRMPI(MPI_Comm_dup(MPI_COMM_WORLD, &comm));
  CHKERRQPARMETIS(ParMETIS_V3_PartGeomKway(vtxdist, xadj, adjncy, vwgt, NULL, &wgtflag, &numflag, &ndims, sxyz, &ncon, &isize, tpwgts, ubvec,options, &edgecut, part, &comm));
  CHKERRMPI(MPI_Comm_free(&comm));

  CHKERRQ(PetscFree(vtxdist));
  CHKERRQ(PetscFree(xadj));
  CHKERRQ(PetscFree(adjncy));
  CHKERRQ(PetscFree3(xyz,part,tpwgts));
  CHKERRQ(PetscFree(sxyz));
  ierr = PetscFinalize();
  return ierr;
}

/*TEST

   build:
      requires: parmetis

   test:
      nsize: 2
      requires: parmetis datafilespath !complex double !defined(PETSC_USE_64BIT_INDICES)
      args: -prefix ${DATAFILESPATH}/parmetis-test/testnp2

   test:
      suffix: 2
      nsize: 4
      requires: parmetis datafilespath !complex double !defined(PETSC_USE_64BIT_INDICES)
      args: -prefix ${DATAFILESPATH}/parmetis-test/testnp4

TEST*/
