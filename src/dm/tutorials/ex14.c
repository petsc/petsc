
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
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  CHKERRMPI(MPI_Comm_rank(PETSC_COMM_WORLD,&rank));
  CHKERRQ(DMDAGetLocalInfo(da,&info));

  if (info.dim == 3) {
    PetscScalar    ***g;
    CHKERRQ(DMDAVecGetArray(da,gvec,&g));
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
    CHKERRQ(DMDAVecRestoreArray(da,gvec,&g));
  }
  if (info.dim == 2) {
    PetscScalar    **g;
    CHKERRQ(DMDAVecGetArray(da,gvec,&g));
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
    CHKERRQ(DMDAVecRestoreArray(da,gvec,&g));
  }
  PetscFunctionReturn(0);
}

int main(int argc,char **argv)
{
  PetscErrorCode ierr;
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

  CHKERRQ(PetscInitialize(&argc,&argv,(char*)0,help));
  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-dim",&dim,NULL));

  /* Create distributed array and get vectors */
  CHKERRMPI(MPI_Comm_size(PETSC_COMM_WORLD,&size));
  CHKERRMPI(MPI_Comm_rank(PETSC_COMM_WORLD,&rank));
  if (dim == 2) {
    CHKERRQ(DMDACreate2d(PETSC_COMM_WORLD,DM_BOUNDARY_NONE,DM_BOUNDARY_NONE,DMDA_STENCIL_STAR,M,N,PETSC_DECIDE,PETSC_DECIDE,3,1,NULL,NULL,&da));
  } else if (dim == 3) {
    CHKERRQ(DMDACreate3d(PETSC_COMM_WORLD, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE,DM_BOUNDARY_NONE,DMDA_STENCIL_STAR,M,N,P,PETSC_DECIDE,PETSC_DECIDE,PETSC_DECIDE,3,1,NULL,NULL,NULL,&da));
  }
  CHKERRQ(DMSetFromOptions(da));
  CHKERRQ(DMSetUp(da));
  CHKERRQ(DMDAGetLocalInfo(da,&info));

  CHKERRQ(DMCreateDomainDecomposition(da,NULL,NULL,&iis,&ois,&subda));
  CHKERRQ(DMCreateDomainDecompositionScatters(da,1,subda,&iscat,&oscat,&gscat));

  {
    DMDALocalInfo subinfo;
    MatStencil    lower,upper;
    IS            patchis;
    Vec           smallvec;
    Vec           largevec;
    VecScatter    patchscat;

    CHKERRQ(DMDAGetLocalInfo(subda[0],&subinfo));

    lower.i = info.xs;
    lower.j = info.ys;
    lower.k = info.zs;
    upper.i = info.xs+info.xm;
    upper.j = info.ys+info.ym;
    upper.k = info.zs+info.zm;

    /* test the patch IS as a thing to scatter to/from */
    CHKERRQ(DMDACreatePatchIS(da,&lower,&upper,&patchis,patchis_offproc));
    CHKERRQ(DMGetGlobalVector(da,&largevec));

    CHKERRQ(VecCreate(PETSC_COMM_SELF,&smallvec));
    CHKERRQ(VecSetSizes(smallvec,info.dof*(upper.i - lower.i)*(upper.j - lower.j)*(upper.k - lower.k),PETSC_DECIDE));
    CHKERRQ(VecSetFromOptions(smallvec));
    CHKERRQ(VecScatterCreate(smallvec,NULL,largevec,patchis,&patchscat));

    CHKERRQ(FillLocalSubdomain(subda[0],smallvec));
    CHKERRQ(VecSet(largevec,0));

    CHKERRQ(VecScatterBegin(patchscat,smallvec,largevec,ADD_VALUES,SCATTER_FORWARD));
    CHKERRQ(VecScatterEnd(patchscat,smallvec,largevec,ADD_VALUES,SCATTER_FORWARD));
    CHKERRQ(ISView(patchis,PETSC_VIEWER_STDOUT_WORLD));
    CHKERRQ(VecScatterView(patchscat,PETSC_VIEWER_STDOUT_WORLD));

    for (i = 0; i < size; i++) {
      if (i == rank) {
        CHKERRQ(VecView(smallvec,PETSC_VIEWER_STDOUT_SELF));
      }
      CHKERRMPI(MPI_Barrier(PETSC_COMM_WORLD));
    }

    CHKERRMPI(MPI_Barrier(PETSC_COMM_WORLD));
    CHKERRQ(VecView(largevec,PETSC_VIEWER_STDOUT_WORLD));

    CHKERRQ(VecDestroy(&smallvec));
    CHKERRQ(DMRestoreGlobalVector(da,&largevec));
    CHKERRQ(ISDestroy(&patchis));
    CHKERRQ(VecScatterDestroy(&patchscat));
  }

  /* view the various parts */
  {
    for (i = 0; i < size; i++) {
      if (i == rank) {
        CHKERRQ(PetscPrintf(PETSC_COMM_SELF,"Processor %d: \n",i));
        CHKERRQ(DMView(subda[0],PETSC_VIEWER_STDOUT_SELF));
      }
      CHKERRMPI(MPI_Barrier(PETSC_COMM_WORLD));
    }

    CHKERRQ(DMGetLocalVector(subda[0],&slvec));
    CHKERRQ(DMGetGlobalVector(subda[0],&sgvec));
    CHKERRQ(DMGetGlobalVector(da,&v));

    /* test filling outer between the big DM and the small ones with the IS scatter*/
    CHKERRQ(VecScatterCreate(v,ois[0],sgvec,NULL,&oscata));

    CHKERRQ(FillLocalSubdomain(subda[0],sgvec));

    CHKERRQ(VecScatterBegin(oscata,sgvec,v,ADD_VALUES,SCATTER_REVERSE));
    CHKERRQ(VecScatterEnd(oscata,sgvec,v,ADD_VALUES,SCATTER_REVERSE));

    /* test the local-to-local scatter */

    /* fill up the local subdomain and then add them together */
    CHKERRQ(FillLocalSubdomain(da,v));

    CHKERRQ(VecScatterBegin(gscat[0],v,slvec,ADD_VALUES,SCATTER_FORWARD));
    CHKERRQ(VecScatterEnd(gscat[0],v,slvec,ADD_VALUES,SCATTER_FORWARD));

    CHKERRQ(VecView(v,PETSC_VIEWER_STDOUT_WORLD));

    /* test ghost scattering backwards */

    CHKERRQ(VecSet(v,0));

    CHKERRQ(VecScatterBegin(gscat[0],slvec,v,ADD_VALUES,SCATTER_REVERSE));
    CHKERRQ(VecScatterEnd(gscat[0],slvec,v,ADD_VALUES,SCATTER_REVERSE));

    CHKERRQ(VecView(v,PETSC_VIEWER_STDOUT_WORLD));

    /* test overlap scattering backwards */

    CHKERRQ(DMLocalToGlobalBegin(subda[0],slvec,ADD_VALUES,sgvec));
    CHKERRQ(DMLocalToGlobalEnd(subda[0],slvec,ADD_VALUES,sgvec));

    CHKERRQ(VecSet(v,0));

    CHKERRQ(VecScatterBegin(oscat[0],sgvec,v,ADD_VALUES,SCATTER_REVERSE));
    CHKERRQ(VecScatterEnd(oscat[0],sgvec,v,ADD_VALUES,SCATTER_REVERSE));

    CHKERRQ(VecView(v,PETSC_VIEWER_STDOUT_WORLD));

    /* test interior scattering backwards */

    CHKERRQ(VecSet(v,0));

    CHKERRQ(VecScatterBegin(iscat[0],sgvec,v,ADD_VALUES,SCATTER_REVERSE));
    CHKERRQ(VecScatterEnd(iscat[0],sgvec,v,ADD_VALUES,SCATTER_REVERSE));

    CHKERRQ(VecView(v,PETSC_VIEWER_STDOUT_WORLD));

    /* test matrix allocation */
    for (i = 0; i < size; i++) {
      if (i == rank) {
        Mat m;
        CHKERRQ(PetscPrintf(PETSC_COMM_SELF,"Processor %d: \n",i));
        CHKERRQ(DMSetMatType(subda[0],MATAIJ));
        CHKERRQ(DMCreateMatrix(subda[0],&m));
        CHKERRQ(MatView(m,PETSC_VIEWER_STDOUT_SELF));
        CHKERRQ(MatDestroy(&m));
      }
      CHKERRMPI(MPI_Barrier(PETSC_COMM_WORLD));
    }
    CHKERRQ(DMRestoreLocalVector(subda[0],&slvec));
    CHKERRQ(DMRestoreGlobalVector(subda[0],&sgvec));
    CHKERRQ(DMRestoreGlobalVector(da,&v));
  }

  CHKERRQ(DMDestroy(&subda[0]));
  CHKERRQ(ISDestroy(&ois[0]));
  CHKERRQ(ISDestroy(&iis[0]));

  CHKERRQ(VecScatterDestroy(&iscat[0]));
  CHKERRQ(VecScatterDestroy(&oscat[0]));
  CHKERRQ(VecScatterDestroy(&gscat[0]));
  CHKERRQ(VecScatterDestroy(&oscata));

  CHKERRQ(PetscFree(iscat));
  CHKERRQ(PetscFree(oscat));
  CHKERRQ(PetscFree(gscat));
  CHKERRQ(PetscFree(oscata));

  CHKERRQ(PetscFree(subda));
  CHKERRQ(PetscFree(ois));
  CHKERRQ(PetscFree(iis));

  CHKERRQ(DMDestroy(&da));
  CHKERRQ(PetscFinalize());
  return 0;
}
