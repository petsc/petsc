
static char help[] = "Tests DMSwarm with DMShell\n\n";

#include <petscsf.h>
#include <petscdm.h>
#include <petscdmshell.h>
#include <petscdmda.h>
#include <petscdmswarm.h>
#include <petsc/private/dmimpl.h>


PetscErrorCode _DMLocatePoints_DMDARegular_IS(DM dm,Vec pos,IS *iscell)
{
  PetscInt       p,n,bs,npoints,si,sj,milocal,mjlocal,mx,my;
  DM             dmregular;
  PetscInt       *cellidx;
  const PetscScalar *coor;
  PetscReal      dx,dy;
  PetscErrorCode ierr;
  PetscMPIInt    rank;

  ierr = MPI_Comm_rank(PETSC_COMM_WORLD,&rank);CHKERRMPI(ierr);
  ierr = VecGetLocalSize(pos,&n);CHKERRQ(ierr);
  ierr = VecGetBlockSize(pos,&bs);CHKERRQ(ierr);
  npoints = n/bs;

  ierr = PetscMalloc1(npoints,&cellidx);CHKERRQ(ierr);
  ierr = DMGetApplicationContext(dm,(void**)&dmregular);CHKERRQ(ierr);
  ierr = DMDAGetCorners(dmregular,&si,&sj,NULL,&milocal,&mjlocal,NULL);CHKERRQ(ierr);
  ierr = DMDAGetInfo(dmregular,NULL,&mx,&my,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL);CHKERRQ(ierr);

  dx = 2.0/((PetscReal)mx);
  dy = 2.0/((PetscReal)my);

  ierr = VecGetArrayRead(pos,&coor);CHKERRQ(ierr);
  for (p=0; p<npoints; p++) {
    PetscReal coorx,coory;
    PetscInt  mi,mj;

    coorx = PetscRealPart(coor[2*p]);
    coory = PetscRealPart(coor[2*p+1]);

    mi = (PetscInt)( (coorx - (-1.0))/dx);
    mj = (PetscInt)( (coory - (-1.0))/dy);

    cellidx[p] = DMLOCATEPOINT_POINT_NOT_FOUND;

    if ((mj >= sj) && (mj < sj + mjlocal)) {
      if ((mi >= si) && (mi < si + milocal)) {
        cellidx[p] = (mi-si) + (mj-sj) * milocal;
      }
    }
    if (coorx < -1.0) cellidx[p] = DMLOCATEPOINT_POINT_NOT_FOUND;
    if (coorx >  1.0) cellidx[p] = DMLOCATEPOINT_POINT_NOT_FOUND;
    if (coory < -1.0) cellidx[p] = DMLOCATEPOINT_POINT_NOT_FOUND;
    if (coory >  1.0) cellidx[p] = DMLOCATEPOINT_POINT_NOT_FOUND;
  }
  ierr = VecRestoreArrayRead(pos,&coor);CHKERRQ(ierr);
  ierr = ISCreateGeneral(PETSC_COMM_SELF,npoints,cellidx,PETSC_OWN_POINTER,iscell);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode DMLocatePoints_DMDARegular(DM dm,Vec pos,DMPointLocationType ltype, PetscSF cellSF)
{
  IS             iscell;
  PetscSFNode    *cells;
  PetscInt       p,bs,npoints,nfound;
  const PetscInt *boxCells;
  PetscErrorCode ierr;

  ierr = _DMLocatePoints_DMDARegular_IS(dm,pos,&iscell);CHKERRQ(ierr);
  ierr = VecGetLocalSize(pos,&npoints);CHKERRQ(ierr);
  ierr = VecGetBlockSize(pos,&bs);CHKERRQ(ierr);
  npoints = npoints / bs;

  ierr = PetscMalloc1(npoints, &cells);CHKERRQ(ierr);
  ierr = ISGetIndices(iscell, &boxCells);CHKERRQ(ierr);

  for (p=0; p<npoints; p++) {
    cells[p].rank  = 0;
    cells[p].index = DMLOCATEPOINT_POINT_NOT_FOUND;
    cells[p].index = boxCells[p];
  }
  ierr = ISRestoreIndices(iscell, &boxCells);CHKERRQ(ierr);
  ierr = ISDestroy(&iscell);CHKERRQ(ierr);
  nfound = npoints;
  ierr = PetscSFSetGraph(cellSF, npoints, nfound, NULL, PETSC_OWN_POINTER, cells, PETSC_OWN_POINTER);CHKERRQ(ierr);
  ierr = ISDestroy(&iscell);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode DMGetNeighbors_DMDARegular(DM dm,PetscInt *nneighbors,const PetscMPIInt **neighbors)
{
  DM             dmregular;
  PetscErrorCode ierr;

  ierr = DMGetApplicationContext(dm,(void**)&dmregular);CHKERRQ(ierr);
  ierr = DMGetNeighbors(dmregular,nneighbors,neighbors);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode SwarmViewGP(DM dms,const char prefix[])
{
  PetscReal      *array;
  PetscInt       *iarray;
  PetscInt       npoints,p,bs;
  FILE           *fp;
  char           name[PETSC_MAX_PATH_LEN];
  PetscMPIInt    rank;
  PetscErrorCode ierr;

  ierr = MPI_Comm_rank(PETSC_COMM_WORLD,&rank);CHKERRMPI(ierr);
  ierr = PetscSNPrintf(name,PETSC_MAX_PATH_LEN-1,"%s-rank%d.gp",prefix,rank);CHKERRQ(ierr);
  fp = fopen(name,"w");
  if (!fp) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_FILE_OPEN,"Cannot open file %s",name);
  ierr = DMSwarmGetLocalSize(dms,&npoints);CHKERRQ(ierr);
  ierr = DMSwarmGetField(dms,DMSwarmPICField_coor,&bs,NULL,(void**)&array);CHKERRQ(ierr);
  ierr = DMSwarmGetField(dms,"itag",NULL,NULL,(void**)&iarray);CHKERRQ(ierr);
  for (p=0; p<npoints; p++) {
    fprintf(fp,"%+1.4e %+1.4e %1.4e\n",array[2*p],array[2*p+1],(double)iarray[p]);
  }
  ierr = DMSwarmRestoreField(dms,"itag",NULL,NULL,(void**)&iarray);CHKERRQ(ierr);
  ierr = DMSwarmRestoreField(dms,DMSwarmPICField_coor,&bs,NULL,(void**)&array);CHKERRQ(ierr);
  fclose(fp);
  PetscFunctionReturn(0);
}

/*
 Create a DMShell and attach a regularly spaced DMDA for point location
 Override methods for point location
*/
PetscErrorCode ex3_1(void)
{
  DM             dms,dmcell,dmregular;
  PetscMPIInt    rank;
  PetscInt       p,bs,nlocal,overlap,mx,tk;
  PetscReal      dx;
  PetscReal      *array,dt;
  PetscInt       *iarray;
  PetscRandom    rand;
  PetscErrorCode ierr;


  ierr = MPI_Comm_rank(PETSC_COMM_WORLD,&rank);CHKERRMPI(ierr);

  /* Create a regularly spaced DMDA */
  mx = 40;
  overlap = 0;
  ierr = DMDACreate2d(PETSC_COMM_WORLD,DM_BOUNDARY_NONE,DM_BOUNDARY_NONE,DMDA_STENCIL_BOX,mx,mx,PETSC_DECIDE,PETSC_DECIDE,1,overlap,NULL,NULL,&dmregular);CHKERRQ(ierr);
  ierr = DMSetFromOptions(dmregular);CHKERRQ(ierr);
  ierr = DMSetUp(dmregular);CHKERRQ(ierr);

  dx = 2.0/((PetscReal)mx);
  ierr = DMDASetUniformCoordinates(dmregular,-1.0+0.5*dx,1.0-0.5*dx,-1.0+0.5*dx,1.0-0.5*dx,-1.0,1.0);CHKERRQ(ierr);

  /* Create a DMShell for point location purposes */
  ierr = DMShellCreate(PETSC_COMM_WORLD,&dmcell);CHKERRQ(ierr);
  ierr = DMSetApplicationContext(dmcell,(void*)dmregular);CHKERRQ(ierr);
  dmcell->ops->locatepoints = DMLocatePoints_DMDARegular;
  dmcell->ops->getneighbors = DMGetNeighbors_DMDARegular;

  /* Create the swarm */
  ierr = DMCreate(PETSC_COMM_WORLD,&dms);CHKERRQ(ierr);
  ierr = DMSetType(dms,DMSWARM);CHKERRQ(ierr);
  ierr = DMSetDimension(dms,2);CHKERRQ(ierr);

  ierr = DMSwarmSetType(dms,DMSWARM_PIC);CHKERRQ(ierr);
  ierr = DMSwarmSetCellDM(dms,dmcell);CHKERRQ(ierr);

  ierr = DMSwarmRegisterPetscDatatypeField(dms,"itag",1,PETSC_INT);CHKERRQ(ierr);
  ierr = DMSwarmFinalizeFieldRegister(dms);CHKERRQ(ierr);
  {
    PetscInt  si,sj,milocal,mjlocal;
    const PetscScalar *LA_coors;
    Vec       coors;
    PetscInt  cnt;

    ierr = DMDAGetCorners(dmregular,&si,&sj,NULL,&milocal,&mjlocal,NULL);CHKERRQ(ierr);
    ierr = DMGetCoordinates(dmregular,&coors);CHKERRQ(ierr);
    ierr = VecGetArrayRead(coors,&LA_coors);CHKERRQ(ierr);
    ierr = DMSwarmSetLocalSizes(dms,milocal*mjlocal,4);CHKERRQ(ierr);
    ierr = DMSwarmGetLocalSize(dms,&nlocal);CHKERRQ(ierr);
    ierr = DMSwarmGetField(dms,DMSwarmPICField_coor,&bs,NULL,(void**)&array);CHKERRQ(ierr);
    cnt = 0;
    ierr = PetscRandomCreate(PETSC_COMM_SELF,&rand);CHKERRQ(ierr);
    ierr = PetscRandomSetInterval(rand,-1.0,1.0);CHKERRQ(ierr);
    for (p=0; p<nlocal; p++) {
      PetscReal px,py,rx,ry,r2;

      ierr = PetscRandomGetValueReal(rand,&rx);CHKERRQ(ierr);
      ierr = PetscRandomGetValueReal(rand,&ry);CHKERRQ(ierr);

      px = PetscRealPart(LA_coors[2*p+0]) + 0.1*rx*dx;
      py = PetscRealPart(LA_coors[2*p+1]) + 0.1*ry*dx;

      r2 = px*px + py*py;
      if (r2 < 0.75*0.75) {
        array[bs*cnt+0] = px;
        array[bs*cnt+1] = py;
        cnt++;
      }
    }
    ierr = PetscRandomDestroy(&rand);CHKERRQ(ierr);
    ierr = DMSwarmRestoreField(dms,DMSwarmPICField_coor,&bs,NULL,(void**)&array);CHKERRQ(ierr);
    ierr = VecRestoreArrayRead(coors,&LA_coors);CHKERRQ(ierr);
    ierr = DMSwarmSetLocalSizes(dms,cnt,4);CHKERRQ(ierr);

    ierr = DMSwarmGetLocalSize(dms,&nlocal);CHKERRQ(ierr);
    ierr = DMSwarmGetField(dms,"itag",&bs,NULL,(void**)&iarray);CHKERRQ(ierr);
    for (p=0; p<nlocal; p++) {
      iarray[p] = (PetscInt)rank;
    }
    ierr = DMSwarmRestoreField(dms,"itag",&bs,NULL,(void**)&iarray);CHKERRQ(ierr);
  }

  ierr = DMView(dms,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  ierr = SwarmViewGP(dms,"step0");CHKERRQ(ierr);

  dt = 0.1;
  for (tk=1; tk<20; tk++) {
    char prefix[PETSC_MAX_PATH_LEN];
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Step %D \n",tk);CHKERRQ(ierr);
    /* push points */
    ierr = DMSwarmGetLocalSize(dms,&nlocal);CHKERRQ(ierr);
    ierr = DMSwarmGetField(dms,DMSwarmPICField_coor,&bs,NULL,(void**)&array);CHKERRQ(ierr);
    for (p=0; p<nlocal; p++) {
      PetscReal cx,cy,vx,vy;

      cx = array[2*p];
      cy = array[2*p+1];
      vx =  cy;
      vy = -cx;

      array[2*p  ] += dt * vx;
      array[2*p+1] += dt * vy;
    }
    ierr = DMSwarmRestoreField(dms,DMSwarmPICField_coor,&bs,NULL,(void**)&array);CHKERRQ(ierr);

    /* migrate points */
    ierr = DMSwarmMigrate(dms,PETSC_TRUE);CHKERRQ(ierr);
    /* view points */
    ierr = PetscSNPrintf(prefix,PETSC_MAX_PATH_LEN-1,"step%d",tk);CHKERRQ(ierr);
    /* should use the regular SwarmView() api, not one for a particular type */
    ierr = SwarmViewGP(dms,prefix);CHKERRQ(ierr);
  }
  ierr = DMDestroy(&dmregular);CHKERRQ(ierr);
  ierr = DMDestroy(&dmcell);CHKERRQ(ierr);
  ierr = DMDestroy(&dms);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

int main(int argc,char **argv)
{
  PetscErrorCode ierr;

  ierr = PetscInitialize(&argc,&argv,(char*)0,help);if (ierr) return ierr;
  ierr = ex3_1();CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
}

/*TEST

   build:
      requires: double !complex

   test:
      filter: grep -v atomic
      filter_output: grep -v atomic
TEST*/
