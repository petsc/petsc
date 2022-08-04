
static char help[] = "Tests DMCreateDomainDecomposition.\n\n";

/*
Use the options
     -da_grid_x <nx> - number of grid points in x direction, if M < 0
     -da_grid_y <ny> - number of grid points in y direction, if N < 0
     -da_processors_x <MX> number of processors in x directio
     -da_processors_y <MY> number of processors in x direction
*/

#include <petscdm.h>
#include <petscdmda.h>

PetscErrorCode FillLocalSubdomain(DM da, Vec gvec)
{
  DMDALocalInfo  info;
  PetscMPIInt    rank;
  PetscInt       i,j,k,l;

  PetscFunctionBeginUser;
  PetscCallMPI(MPI_Comm_rank(PETSC_COMM_WORLD,&rank));
  PetscCall(DMDAGetLocalInfo(da,&info));

  if (info.dim == 3) {
    PetscScalar    ***g;
    PetscCall(DMDAVecGetArray(da,gvec,&g));
    /* loop over ghosts */
    for (k=info.zs; k<info.zs+info.zm; k++) {
      for (j=info.ys; j<info.ys+info.ym; j++) {
        for (i=info.xs; i<info.xs+info.xm; i++) {
          g[k][j][info.dof*i+0]   = i;
          g[k][j][info.dof*i+1]   = j;
          g[k][j][info.dof*i+2]   = k;
        }
      }
    }
    PetscCall(DMDAVecRestoreArray(da,gvec,&g));
  }
  if (info.dim == 2) {
    PetscScalar    **g;
    PetscCall(DMDAVecGetArray(da,gvec,&g));
    /* loop over ghosts */
    for (j=info.ys; j<info.ys+info.ym; j++) {
      for (i=info.xs; i<info.xs+info.xm; i++) {
        for (l = 0;l<info.dof;l++) {
          g[j][info.dof*i+0]   = i;
          g[j][info.dof*i+1]   = j;
          g[j][info.dof*i+2]   = rank;
        }
      }
    }
    PetscCall(DMDAVecRestoreArray(da,gvec,&g));
  }
  PetscFunctionReturn(0);
}

int main(int argc,char **argv)
{
  DM             da,*subda;
  PetscInt       i,dim = 3;
  PetscInt       M = 25, N = 25, P = 25;
  PetscMPIInt    size,rank;
  Vec            v;
  Vec            slvec,sgvec;
  IS             *ois,*iis;
  VecScatter     oscata;
  VecScatter     *iscat,*oscat,*gscat;
  DMDALocalInfo  info;
  PetscBool      patchis_offproc = PETSC_TRUE;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc,&argv,(char*)0,help));
  PetscCall(PetscOptionsGetInt(NULL,NULL,"-dim",&dim,NULL));

  /* Create distributed array and get vectors */
  PetscCallMPI(MPI_Comm_size(PETSC_COMM_WORLD,&size));
  PetscCallMPI(MPI_Comm_rank(PETSC_COMM_WORLD,&rank));
  if (dim == 2) {
    PetscCall(DMDACreate2d(PETSC_COMM_WORLD,DM_BOUNDARY_NONE,DM_BOUNDARY_NONE,DMDA_STENCIL_STAR,M,N,PETSC_DECIDE,PETSC_DECIDE,3,1,NULL,NULL,&da));
  } else if (dim == 3) {
    PetscCall(DMDACreate3d(PETSC_COMM_WORLD, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE,DM_BOUNDARY_NONE,DMDA_STENCIL_STAR,M,N,P,PETSC_DECIDE,PETSC_DECIDE,PETSC_DECIDE,3,1,NULL,NULL,NULL,&da));
  }
  PetscCall(DMSetFromOptions(da));
  PetscCall(DMSetUp(da));
  PetscCall(DMDAGetLocalInfo(da,&info));

  PetscCall(DMCreateDomainDecomposition(da,NULL,NULL,&iis,&ois,&subda));
  PetscCall(DMCreateDomainDecompositionScatters(da,1,subda,&iscat,&oscat,&gscat));

  {
    DMDALocalInfo subinfo;
    MatStencil    lower,upper;
    IS            patchis;
    Vec           smallvec;
    Vec           largevec;
    VecScatter    patchscat;

    PetscCall(DMDAGetLocalInfo(subda[0],&subinfo));

    lower.i = info.xs;
    lower.j = info.ys;
    lower.k = info.zs;
    upper.i = info.xs+info.xm;
    upper.j = info.ys+info.ym;
    upper.k = info.zs+info.zm;

    /* test the patch IS as a thing to scatter to/from */
    PetscCall(DMDACreatePatchIS(da,&lower,&upper,&patchis,patchis_offproc));
    PetscCall(DMGetGlobalVector(da,&largevec));

    PetscCall(VecCreate(PETSC_COMM_SELF,&smallvec));
    PetscCall(VecSetSizes(smallvec,info.dof*(upper.i - lower.i)*(upper.j - lower.j)*(upper.k - lower.k),PETSC_DECIDE));
    PetscCall(VecSetFromOptions(smallvec));
    PetscCall(VecScatterCreate(smallvec,NULL,largevec,patchis,&patchscat));

    PetscCall(FillLocalSubdomain(subda[0],smallvec));
    PetscCall(VecSet(largevec,0));

    PetscCall(VecScatterBegin(patchscat,smallvec,largevec,ADD_VALUES,SCATTER_FORWARD));
    PetscCall(VecScatterEnd(patchscat,smallvec,largevec,ADD_VALUES,SCATTER_FORWARD));
    PetscCall(ISView(patchis,PETSC_VIEWER_STDOUT_WORLD));
    PetscCall(VecScatterView(patchscat,PETSC_VIEWER_STDOUT_WORLD));

    for (i = 0; i < size; i++) {
      if (i == rank) {
        PetscCall(VecView(smallvec,PETSC_VIEWER_STDOUT_SELF));
      }
      PetscCallMPI(MPI_Barrier(PETSC_COMM_WORLD));
    }

    PetscCallMPI(MPI_Barrier(PETSC_COMM_WORLD));
    PetscCall(VecView(largevec,PETSC_VIEWER_STDOUT_WORLD));

    PetscCall(VecDestroy(&smallvec));
    PetscCall(DMRestoreGlobalVector(da,&largevec));
    PetscCall(ISDestroy(&patchis));
    PetscCall(VecScatterDestroy(&patchscat));
  }

  /* view the various parts */
  {
    for (i = 0; i < size; i++) {
      if (i == rank) {
        PetscCall(PetscPrintf(PETSC_COMM_SELF,"Processor %d: \n",i));
        PetscCall(DMView(subda[0],PETSC_VIEWER_STDOUT_SELF));
      }
      PetscCallMPI(MPI_Barrier(PETSC_COMM_WORLD));
    }

    PetscCall(DMGetLocalVector(subda[0],&slvec));
    PetscCall(DMGetGlobalVector(subda[0],&sgvec));
    PetscCall(DMGetGlobalVector(da,&v));

    /* test filling outer between the big DM and the small ones with the IS scatter*/
    PetscCall(VecScatterCreate(v,ois[0],sgvec,NULL,&oscata));

    PetscCall(FillLocalSubdomain(subda[0],sgvec));

    PetscCall(VecScatterBegin(oscata,sgvec,v,ADD_VALUES,SCATTER_REVERSE));
    PetscCall(VecScatterEnd(oscata,sgvec,v,ADD_VALUES,SCATTER_REVERSE));

    /* test the local-to-local scatter */

    /* fill up the local subdomain and then add them together */
    PetscCall(FillLocalSubdomain(da,v));

    PetscCall(VecScatterBegin(gscat[0],v,slvec,ADD_VALUES,SCATTER_FORWARD));
    PetscCall(VecScatterEnd(gscat[0],v,slvec,ADD_VALUES,SCATTER_FORWARD));

    PetscCall(VecView(v,PETSC_VIEWER_STDOUT_WORLD));

    /* test ghost scattering backwards */

    PetscCall(VecSet(v,0));

    PetscCall(VecScatterBegin(gscat[0],slvec,v,ADD_VALUES,SCATTER_REVERSE));
    PetscCall(VecScatterEnd(gscat[0],slvec,v,ADD_VALUES,SCATTER_REVERSE));

    PetscCall(VecView(v,PETSC_VIEWER_STDOUT_WORLD));

    /* test overlap scattering backwards */

    PetscCall(DMLocalToGlobalBegin(subda[0],slvec,ADD_VALUES,sgvec));
    PetscCall(DMLocalToGlobalEnd(subda[0],slvec,ADD_VALUES,sgvec));

    PetscCall(VecSet(v,0));

    PetscCall(VecScatterBegin(oscat[0],sgvec,v,ADD_VALUES,SCATTER_REVERSE));
    PetscCall(VecScatterEnd(oscat[0],sgvec,v,ADD_VALUES,SCATTER_REVERSE));

    PetscCall(VecView(v,PETSC_VIEWER_STDOUT_WORLD));

    /* test interior scattering backwards */

    PetscCall(VecSet(v,0));

    PetscCall(VecScatterBegin(iscat[0],sgvec,v,ADD_VALUES,SCATTER_REVERSE));
    PetscCall(VecScatterEnd(iscat[0],sgvec,v,ADD_VALUES,SCATTER_REVERSE));

    PetscCall(VecView(v,PETSC_VIEWER_STDOUT_WORLD));

    /* test matrix allocation */
    for (i = 0; i < size; i++) {
      if (i == rank) {
        Mat m;
        PetscCall(PetscPrintf(PETSC_COMM_SELF,"Processor %d: \n",i));
        PetscCall(DMSetMatType(subda[0],MATAIJ));
        PetscCall(DMCreateMatrix(subda[0],&m));
        PetscCall(MatView(m,PETSC_VIEWER_STDOUT_SELF));
        PetscCall(MatDestroy(&m));
      }
      PetscCallMPI(MPI_Barrier(PETSC_COMM_WORLD));
    }
    PetscCall(DMRestoreLocalVector(subda[0],&slvec));
    PetscCall(DMRestoreGlobalVector(subda[0],&sgvec));
    PetscCall(DMRestoreGlobalVector(da,&v));
  }

  PetscCall(DMDestroy(&subda[0]));
  PetscCall(ISDestroy(&ois[0]));
  PetscCall(ISDestroy(&iis[0]));

  PetscCall(VecScatterDestroy(&iscat[0]));
  PetscCall(VecScatterDestroy(&oscat[0]));
  PetscCall(VecScatterDestroy(&gscat[0]));
  PetscCall(VecScatterDestroy(&oscata));

  PetscCall(PetscFree(iscat));
  PetscCall(PetscFree(oscat));
  PetscCall(PetscFree(gscat));
  PetscCall(PetscFree(oscata));

  PetscCall(PetscFree(subda));
  PetscCall(PetscFree(ois));
  PetscCall(PetscFree(iis));

  PetscCall(DMDestroy(&da));
  PetscCall(PetscFinalize());
  return 0;
}
