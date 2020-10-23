
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
  ierr = MPI_Comm_rank(PETSC_COMM_WORLD,&rank);CHKERRQ(ierr);
  ierr = DMDAGetLocalInfo(da,&info);CHKERRQ(ierr);

  if (info.dim == 3) {
    PetscScalar    ***g;
    ierr = DMDAVecGetArray(da,gvec,&g);CHKERRQ(ierr);
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
    ierr = DMDAVecRestoreArray(da,gvec,&g);CHKERRQ(ierr);
  }
  if (info.dim == 2) {
    PetscScalar    **g;
    ierr = DMDAVecGetArray(da,gvec,&g);CHKERRQ(ierr);
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
    ierr = DMDAVecRestoreArray(da,gvec,&g);CHKERRQ(ierr);
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

  ierr = PetscInitialize(&argc,&argv,(char*)0,help);if (ierr) return ierr;
  ierr = PetscOptionsGetInt(NULL,NULL,"-dim",&dim,NULL);CHKERRQ(ierr);

  /* Create distributed array and get vectors */
  ierr = MPI_Comm_size(PETSC_COMM_WORLD,&size);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(PETSC_COMM_WORLD,&rank);CHKERRQ(ierr);
  if (dim == 2) {
    ierr = DMDACreate2d(PETSC_COMM_WORLD,DM_BOUNDARY_NONE,DM_BOUNDARY_NONE,DMDA_STENCIL_STAR,M,N,PETSC_DECIDE,PETSC_DECIDE,3,1,NULL,NULL,&da);CHKERRQ(ierr);
  } else if (dim == 3) {
    ierr = DMDACreate3d(PETSC_COMM_WORLD, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE,DM_BOUNDARY_NONE,DMDA_STENCIL_STAR,M,N,P,PETSC_DECIDE,PETSC_DECIDE,PETSC_DECIDE,3,1,NULL,NULL,NULL,&da);CHKERRQ(ierr);
  }
  ierr = DMSetFromOptions(da);CHKERRQ(ierr);
  ierr = DMSetUp(da);CHKERRQ(ierr);
  ierr = DMDAGetLocalInfo(da,&info);CHKERRQ(ierr);

  ierr = DMCreateDomainDecomposition(da,NULL,NULL,&iis,&ois,&subda);CHKERRQ(ierr);
  ierr = DMCreateDomainDecompositionScatters(da,1,subda,&iscat,&oscat,&gscat);CHKERRQ(ierr);

  {
    DMDALocalInfo subinfo;
    MatStencil    lower,upper;
    IS            patchis;
    Vec           smallvec;
    Vec           largevec;
    VecScatter    patchscat;

    ierr = DMDAGetLocalInfo(subda[0],&subinfo);CHKERRQ(ierr);

    lower.i = info.xs;
    lower.j = info.ys;
    lower.k = info.zs;
    upper.i = info.xs+info.xm;
    upper.j = info.ys+info.ym;
    upper.k = info.zs+info.zm;

    /* test the patch IS as a thing to scatter to/from */
    ierr = DMDACreatePatchIS(da,&lower,&upper,&patchis,patchis_offproc);CHKERRQ(ierr);
    ierr = DMGetGlobalVector(da,&largevec);CHKERRQ(ierr);

    ierr = VecCreate(PETSC_COMM_SELF,&smallvec);CHKERRQ(ierr);
    ierr = VecSetSizes(smallvec,info.dof*(upper.i - lower.i)*(upper.j - lower.j)*(upper.k - lower.k),PETSC_DECIDE);CHKERRQ(ierr);
    ierr = VecSetFromOptions(smallvec);CHKERRQ(ierr);
    ierr = VecScatterCreate(smallvec,NULL,largevec,patchis,&patchscat);CHKERRQ(ierr);

    ierr = FillLocalSubdomain(subda[0],smallvec);CHKERRQ(ierr);
    ierr = VecSet(largevec,0);CHKERRQ(ierr);

    ierr = VecScatterBegin(patchscat,smallvec,largevec,ADD_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
    ierr = VecScatterEnd(patchscat,smallvec,largevec,ADD_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
    ierr = ISView(patchis,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
    ierr = VecScatterView(patchscat,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);

    for (i = 0; i < size; i++) {
      if (i == rank) {
        ierr = VecView(smallvec,PETSC_VIEWER_STDOUT_SELF);CHKERRQ(ierr);
      }
      ierr = MPI_Barrier(PETSC_COMM_WORLD);CHKERRQ(ierr);
    }

    ierr = MPI_Barrier(PETSC_COMM_WORLD);CHKERRQ(ierr);
    ierr = VecView(largevec,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);

    ierr = VecDestroy(&smallvec);CHKERRQ(ierr);
    ierr = DMRestoreGlobalVector(da,&largevec);CHKERRQ(ierr);
    ierr = ISDestroy(&patchis);CHKERRQ(ierr);
    ierr = VecScatterDestroy(&patchscat);CHKERRQ(ierr);
  }

  /* view the various parts */
  {
    for (i = 0; i < size; i++) {
      if (i == rank) {
        ierr = PetscPrintf(PETSC_COMM_SELF,"Processor %d: \n",i);CHKERRQ(ierr);
        ierr = DMView(subda[0],PETSC_VIEWER_STDOUT_SELF);CHKERRQ(ierr);
      }
      ierr = MPI_Barrier(PETSC_COMM_WORLD);CHKERRQ(ierr);
    }

    ierr = DMGetLocalVector(subda[0],&slvec);CHKERRQ(ierr);
    ierr = DMGetGlobalVector(subda[0],&sgvec);CHKERRQ(ierr);
    ierr = DMGetGlobalVector(da,&v);CHKERRQ(ierr);

    /* test filling outer between the big DM and the small ones with the IS scatter*/
    ierr = VecScatterCreate(v,ois[0],sgvec,NULL,&oscata);CHKERRQ(ierr);

    ierr = FillLocalSubdomain(subda[0],sgvec);CHKERRQ(ierr);

    ierr = VecScatterBegin(oscata,sgvec,v,ADD_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
    ierr = VecScatterEnd(oscata,sgvec,v,ADD_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);

    /* test the local-to-local scatter */

    /* fill up the local subdomain and then add them together */
    ierr = FillLocalSubdomain(da,v);CHKERRQ(ierr);

    ierr = VecScatterBegin(gscat[0],v,slvec,ADD_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
    ierr = VecScatterEnd(gscat[0],v,slvec,ADD_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);

    ierr = VecView(v,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);

    /* test ghost scattering backwards */

    ierr = VecSet(v,0);CHKERRQ(ierr);

    ierr = VecScatterBegin(gscat[0],slvec,v,ADD_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
    ierr = VecScatterEnd(gscat[0],slvec,v,ADD_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);

    ierr = VecView(v,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);

    /* test overlap scattering backwards */

    ierr = DMLocalToGlobalBegin(subda[0],slvec,ADD_VALUES,sgvec);CHKERRQ(ierr);
    ierr = DMLocalToGlobalEnd(subda[0],slvec,ADD_VALUES,sgvec);CHKERRQ(ierr);

    ierr = VecSet(v,0);CHKERRQ(ierr);

    ierr = VecScatterBegin(oscat[0],sgvec,v,ADD_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
    ierr = VecScatterEnd(oscat[0],sgvec,v,ADD_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);

    ierr = VecView(v,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);

    /* test interior scattering backwards */

    ierr = VecSet(v,0);CHKERRQ(ierr);

    ierr = VecScatterBegin(iscat[0],sgvec,v,ADD_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
    ierr = VecScatterEnd(iscat[0],sgvec,v,ADD_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);

    ierr = VecView(v,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);

    /* test matrix allocation */
    for (i = 0; i < size; i++) {
      if (i == rank) {
        Mat m;
        ierr = PetscPrintf(PETSC_COMM_SELF,"Processor %d: \n",i);CHKERRQ(ierr);
        ierr = DMSetMatType(subda[0],MATAIJ);CHKERRQ(ierr);
        ierr = DMCreateMatrix(subda[0],&m);CHKERRQ(ierr);
        ierr = MatView(m,PETSC_VIEWER_STDOUT_SELF);CHKERRQ(ierr);
        ierr = MatDestroy(&m);CHKERRQ(ierr);
      }
      ierr = MPI_Barrier(PETSC_COMM_WORLD);CHKERRQ(ierr);
    }
    ierr = DMRestoreLocalVector(subda[0],&slvec);CHKERRQ(ierr);
    ierr = DMRestoreGlobalVector(subda[0],&sgvec);CHKERRQ(ierr);
    ierr = DMRestoreGlobalVector(da,&v);CHKERRQ(ierr);
  }


  ierr = DMDestroy(&subda[0]);CHKERRQ(ierr);
  ierr = ISDestroy(&ois[0]);CHKERRQ(ierr);
  ierr = ISDestroy(&iis[0]);CHKERRQ(ierr);

  ierr = VecScatterDestroy(&iscat[0]);CHKERRQ(ierr);
  ierr = VecScatterDestroy(&oscat[0]);CHKERRQ(ierr);
  ierr = VecScatterDestroy(&gscat[0]);CHKERRQ(ierr);
  ierr = VecScatterDestroy(&oscata);CHKERRQ(ierr);

  ierr = PetscFree(iscat);CHKERRQ(ierr);
  ierr = PetscFree(oscat);CHKERRQ(ierr);
  ierr = PetscFree(gscat);CHKERRQ(ierr);
  ierr = PetscFree(oscata);CHKERRQ(ierr);

  ierr = PetscFree(subda);CHKERRQ(ierr);
  ierr = PetscFree(ois);CHKERRQ(ierr);
  ierr = PetscFree(iis);CHKERRQ(ierr);

  ierr = DMDestroy(&da);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
}

