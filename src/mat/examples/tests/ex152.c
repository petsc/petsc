static const char help[] = "Test ParMETIS handling of negative weights.\n\n";

/* Test contributed by John Fettig */

/*
 * This file implements two tests for a bug reported in ParMETIS. These tests are not expected to pass without the
 * following two patches.
 *
 * http://petsc.cs.iit.edu/petsc/externalpackages/parmetis-4.0.2/rev/2dd2eae596ac
 * http://petsc.cs.iit.edu/petsc/externalpackages/parmetis-4.0.2/rev/1c2b9fe39201
 *
 * The bug was reported upstream, but has received no action so far.
 *
 * http://glaros.dtc.umn.edu/gkhome/node/837
 *
 */

#include <petscsys.h>
#include <parmetis.h>

#define CHKERRQPARMETIS(n) \
  if (n == METIS_ERROR_INPUT) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_LIB,"ParMETIS error due to wrong inputs and/or options"); \
  else if (n == METIS_ERROR_MEMORY) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_LIB,"ParMETIS error due to insufficient memory"); \
  else if (n == METIS_ERROR) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_LIB,"ParMETIS general error"); \

int main(int argc, char* argv[])
{
  PetscErrorCode ierr;
  PetscBool flg;
  int rank, size;
  int i, ni, status;
  idx_t *vtxdist, *xadj, *adjncy, *vwgt, *part;
  idx_t wgtflag=0, numflag=0, ncon=1, ndims=3, edgecut=0;
  idx_t options[5];
  real_t *xyz, *tpwgts, ubvec[1];
  MPI_Comm comm;
  FILE *fp;
  char fname[PETSC_MAX_PATH_LEN],prefix[PETSC_MAX_PATH_LEN] = "";

  PetscInitialize(&argc,&argv,PETSC_NULL,help);
  MPI_Comm_rank(PETSC_COMM_WORLD,&rank);
  MPI_Comm_size(PETSC_COMM_WORLD,&size);

  ierr = PetscOptionsBegin(PETSC_COMM_WORLD,PETSC_NULL,"Parmetis test options","");CHKERRQ(ierr);
  ierr = PetscOptionsString("-prefix","Path and prefix of test file","",prefix,prefix,sizeof(prefix),&flg);CHKERRQ(ierr);
  if (!flg) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_USER,"Must specify -prefix");CHKERRQ(ierr);
  ierr = PetscOptionsEnd();CHKERRQ(ierr);

  ierr = PetscMalloc((size+1)*sizeof(idx_t),&vtxdist);CHKERRQ(ierr);

  ierr = PetscSNPrintf(fname,sizeof(fname),"%s.%d.graph",prefix,rank);CHKERRQ(ierr);

  ierr = PetscFOpen(PETSC_COMM_SELF,fname,"r",&fp);CHKERRQ(ierr);

  fread(vtxdist, sizeof(idx_t), size+1, fp);

  ni = vtxdist[rank+1]-vtxdist[rank];

  ierr = PetscMalloc((ni+1)*sizeof(idx_t),&xadj);CHKERRQ(ierr);

  fread(xadj, sizeof(idx_t), ni+1, fp);

  ierr = PetscMalloc(xadj[ni]*sizeof(idx_t),&adjncy);CHKERRQ(ierr);

  for (i=0; i<ni; i++)
    fread(&adjncy[xadj[i]], sizeof(idx_t), xadj[i+1]-xadj[i], fp);

  ierr = PetscFClose(PETSC_COMM_SELF,fp);CHKERRQ(ierr);

  ierr = PetscSNPrintf(fname,sizeof(fname),"%s.%d.graph.xyz",prefix,rank);CHKERRQ(ierr);
  ierr = PetscFOpen(PETSC_COMM_SELF,fname,"r",&fp);CHKERRQ(ierr);

  ierr = PetscMalloc3(ni*ndims,real_t,&xyz,ni,idx_t,&part,size,real_t,&tpwgts);CHKERRQ(ierr);

  fread(xyz, sizeof(real_t), ndims*ni, fp);
  ierr = PetscFClose(PETSC_COMM_SELF,fp);CHKERRQ(ierr);

  vwgt = NULL;

  for ( i = 0 ; i < size ; i++ )
    tpwgts[i] = 1. / size ;

  ubvec[0] = 1.05;
  options[0] = 1;
  options[1] = 2;
  options[2] = 15;
  options[3] = 0;
  options[4] = 0;

  ierr = MPI_Comm_dup(MPI_COMM_WORLD, &comm);CHKERRQ(ierr);
  status = ParMETIS_V3_PartGeomKway(vtxdist, xadj, adjncy, vwgt,
                                    PETSC_NULL, &wgtflag, &numflag, &ndims, xyz, &ncon, &size, tpwgts, ubvec,
                                    options, &edgecut, part, &comm);CHKERRQPARMETIS(status);
  ierr = MPI_Comm_free(&comm);CHKERRQ(ierr);

  ierr = PetscFree(vtxdist);CHKERRQ(ierr);
  ierr = PetscFree(xadj);CHKERRQ(ierr);
  ierr = PetscFree(adjncy);CHKERRQ(ierr);
  ierr = PetscFree3(xyz,part,tpwgts);CHKERRQ(ierr);
  PetscFinalize();
  return 0;
}
