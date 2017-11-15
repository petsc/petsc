
static char help[] = "Tests DMCreateDomainDecomposition.\n\n";

/*
Use the options
     -da_grid_x <nx> - number of grid points in x direction, if M < 0
     -da_grid_y <ny> - number of grid points in y direction, if N < 0
     -da_processors_x <MX> number of processors in x directio
     -da_processors_y <MY> number of processors in x direction
*/

#include <petscdmda.h>

#undef __FUNCT__
#define __FUNCT__ "FillLocalSubdomain"
PetscErrorCode FillLocalSubdomain(DM da, Vec gvec)
{
  PetscScalar    **g;
  DMDALocalInfo  info;
  PetscMPIInt    rank;
  PetscInt       i,j;
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  ierr = MPI_Comm_rank(PETSC_COMM_WORLD,&rank);CHKERRQ(ierr);
  ierr = DMDAGetLocalInfo(da,&info);CHKERRQ(ierr);

  ierr = DMDAVecGetArray(da,gvec,&g);CHKERRQ(ierr);
  /* loop over ghosts */
  for (j=info.ys; j<info.ys+info.ym; j++) {
    for (i=info.xs; i<info.xs+info.xm; i++) {
      if (i == 0 || j == 0 || i == info.mx-1 || j == info.my-1) {
        g[j][2*i]   = i*i + j*j;
        g[j][2*i+1] = rank;
      } else {
        g[j][2*i]   = i*i + j*j;
        g[j][2*i+1] = rank;
      }
    }
  }

  ierr = DMDAVecRestoreArray(da,gvec,&g);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc,char **argv)
{
  PetscErrorCode ierr;
  DM             da,*subda;

  PetscInt    i;
  PetscMPIInt size,rank;

  Vec v;
  Vec slvec,sgvec;

  IS         *ois,*iis;
  VecScatter oscata;
  VecScatter *iscat,*oscat,*gscat;

  DMDALocalInfo info;

  ierr = PetscInitialize(&argc,&argv,(char*)0,help);CHKERRQ(ierr);

  /* Create distributed array and get vectors */
  ierr = MPI_Comm_size(PETSC_COMM_WORLD,&size);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(PETSC_COMM_WORLD,&rank);CHKERRQ(ierr);

  ierr = DMDACreate2d(PETSC_COMM_WORLD, DMDA_BOUNDARY_NONE, DMDA_BOUNDARY_NONE,DMDA_STENCIL_STAR,-4,-4,PETSC_DECIDE,PETSC_DECIDE,2,1,NULL,NULL,&da);CHKERRQ(ierr);

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

    lower.i = subinfo.xs;
    lower.j = subinfo.ys;
    lower.k = subinfo.zs;
    upper.i = subinfo.xs + subinfo.xm;
    upper.j = subinfo.ys + subinfo.ym;
    upper.k = subinfo.zs + subinfo.zm;

    /* test the patch IS as a thing to scatter to/from */
    ierr = DMDACreatePatchIS(da,&lower,&upper,&patchis);CHKERRQ(ierr);
    ierr = ISView(patchis,PETSC_VIEWER_STDOUT_SELF);CHKERRQ(ierr);
    ierr = DMGetGlobalVector(da,&largevec);CHKERRQ(ierr);

    ierr = VecCreate(PETSC_COMM_SELF,&smallvec);CHKERRQ(ierr);
    ierr = VecSetSizes(smallvec,2*(upper.i - lower.i)*(upper.j - lower.j),PETSC_DECIDE);CHKERRQ(ierr);
    ierr = VecSetFromOptions(smallvec);CHKERRQ(ierr);
    ierr = VecScatterCreate(largevec,patchis,smallvec,NULL,&patchscat);CHKERRQ(ierr);

    ierr = VecSet(smallvec,1.);CHKERRQ(ierr);

    ierr = VecScatterBegin(patchscat,smallvec,largevec,ADD_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
    ierr = VecScatterEnd(patchscat,smallvec,largevec,ADD_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);

    ierr = VecView(largevec,PETSC_VIEWER_DRAW_WORLD);CHKERRQ(ierr);

    ierr = VecDestroy(&smallvec);CHKERRQ(ierr);
    ierr = DMRestoreGlobalVector(da,&largevec);CHKERRQ(ierr);
    ierr = ISDestroy(&patchis);CHKERRQ(ierr);
    ierr = VecScatterDestroy(&patchscat);CHKERRQ(ierr);
  }

  /* view the various parts */
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

  ierr = VecView(v,PETSC_VIEWER_DRAW_WORLD);CHKERRQ(ierr);

  /* test ghost scattering backwards */

  ierr = VecSet(v,0);CHKERRQ(ierr);

  ierr = VecScatterBegin(gscat[0],slvec,v,ADD_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
  ierr = VecScatterEnd(gscat[0],slvec,v,ADD_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);

  ierr = VecView(v,PETSC_VIEWER_DRAW_WORLD);CHKERRQ(ierr);

  /* test overlap scattering backwards */

  ierr = DMLocalToGlobalBegin(subda[0],slvec,ADD_VALUES,sgvec);CHKERRQ(ierr);
  ierr = DMLocalToGlobalEnd(subda[0],slvec,ADD_VALUES,sgvec);CHKERRQ(ierr);

  ierr = VecSet(v,0);CHKERRQ(ierr);

  ierr = VecScatterBegin(oscat[0],sgvec,v,ADD_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
  ierr = VecScatterEnd(oscat[0],sgvec,v,ADD_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);

  ierr = VecView(v,PETSC_VIEWER_DRAW_WORLD);CHKERRQ(ierr);

  /* test interior scattering backwards */

  ierr = VecSet(v,0);CHKERRQ(ierr);

  ierr = VecScatterBegin(iscat[0],sgvec,v,ADD_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
  ierr = VecScatterEnd(iscat[0],sgvec,v,ADD_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);

  ierr = VecView(v,PETSC_VIEWER_DRAW_WORLD);CHKERRQ(ierr);

  /* test matrix allocation */
  for (i = 0; i < size; i++) {
    if (i == rank) {
      Mat m;
      ierr = PetscPrintf(PETSC_COMM_SELF,"Processor %d: \n",i);CHKERRQ(ierr);
      ierr = DMCreateMatrix(subda[0],"mpiaij",&m);CHKERRQ(ierr);
      ierr = MatView(m,PETSC_VIEWER_STDOUT_SELF);CHKERRQ(ierr);
      ierr = MatDestroy(&m);CHKERRQ(ierr);
    }
    ierr = MPI_Barrier(PETSC_COMM_WORLD);CHKERRQ(ierr);
  }

  ierr = DMRestoreLocalVector(subda[0],&slvec);CHKERRQ(ierr);
  ierr = DMRestoreGlobalVector(subda[0],&sgvec);CHKERRQ(ierr);
  ierr = DMRestoreGlobalVector(da,&v);CHKERRQ(ierr);

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
  return 0;
}

