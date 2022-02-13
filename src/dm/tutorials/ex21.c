
static char help[] = "DMSwarm-PIC demonstator of advecting points within cell DM defined by a DA or PLEX object \n\
Options: \n\
-ppcell   : Number of times to sub-divide the reference cell when layout the initial particle coordinates \n\
-meshtype : 0 ==> DA , 1 ==> PLEX \n\
-nt       : Number of timestep to perform \n\
-view     : Write out initial condition and time dependent data \n";

#include <petsc.h>
#include <petscdm.h>
#include <petscdmda.h>
#include <petscdmswarm.h>

PetscErrorCode pic_advect(PetscInt ppcell,PetscInt meshtype)
{
  PetscErrorCode ierr;
  const PetscInt dim = 2;
  DM celldm,swarm;
  PetscInt tk,nt = 200;
  PetscBool view = PETSC_FALSE;
  Vec *pfields;
  PetscReal minradius;
  PetscReal dt;
  PetscReal vel[] = { 1.0, 0.16 };
  const char *fieldnames[] = { "phi" };
  PetscViewer viewer;

  PetscFunctionBegin;
  ierr = PetscOptionsGetBool(NULL,NULL,"-view",&view,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(NULL,NULL,"-nt",&nt,NULL);CHKERRQ(ierr);

  /* Create the background cell DM */
  if (meshtype == 0) { /* DA */
    PetscInt nxy;
    PetscInt dof = 1;
    PetscInt stencil_width = 1;

    ierr = PetscPrintf(PETSC_COMM_WORLD,"Mesh type: DMDA\n");CHKERRQ(ierr);
    nxy = 33;
    ierr = DMDACreate2d(PETSC_COMM_WORLD,DM_BOUNDARY_NONE,DM_BOUNDARY_NONE,DMDA_STENCIL_BOX,nxy,nxy,PETSC_DECIDE,PETSC_DECIDE,dof,stencil_width,NULL,NULL,&celldm);CHKERRQ(ierr);

    ierr = DMDASetElementType(celldm,DMDA_ELEMENT_Q1);CHKERRQ(ierr);

    ierr = DMSetFromOptions(celldm);CHKERRQ(ierr);

    ierr = DMSetUp(celldm);CHKERRQ(ierr);

    ierr = DMDASetUniformCoordinates(celldm,0.0,1.0,0.0,1.0,0.0,1.5);CHKERRQ(ierr);

    minradius = 1.0/((PetscReal)(nxy-1));

    ierr = PetscPrintf(PETSC_COMM_WORLD,"DA(minradius) %1.4e\n",(double)minradius);CHKERRQ(ierr);
  }

  if (meshtype == 1){ /* PLEX */
    DM distributedMesh = NULL;
    PetscInt numComp[] = {1};
    PetscInt numDof[] = {1,0,0}; /* vert, edge, cell */
    PetscInt faces[]  = {1,1,1};
    PetscInt numBC = 0;
    PetscSection section;
    Vec cellgeom = NULL;
    Vec facegeom = NULL;

    ierr = PetscPrintf(PETSC_COMM_WORLD,"Mesh type: DMPLEX\n");CHKERRQ(ierr);
    ierr = DMPlexCreateBoxMesh(PETSC_COMM_WORLD, dim, PETSC_TRUE, faces, NULL, NULL, PETSC_TRUE, &celldm);CHKERRQ(ierr);

    /* Distribute mesh over processes */
    ierr = DMPlexDistribute(celldm,0,NULL,&distributedMesh);CHKERRQ(ierr);
    if (distributedMesh) {
      ierr = DMDestroy(&celldm);CHKERRQ(ierr);
      celldm = distributedMesh;
    }

    ierr = DMSetFromOptions(celldm);CHKERRQ(ierr);

    ierr = DMPlexCreateSection(celldm,NULL,numComp,numDof,numBC,NULL,NULL,NULL,NULL,&section);CHKERRQ(ierr);
    ierr = DMSetLocalSection(celldm,section);CHKERRQ(ierr);

    ierr = DMSetUp(celldm);CHKERRQ(ierr);

    /* Calling DMPlexComputeGeometryFVM() generates the value returned by DMPlexGetMinRadius() */
    ierr = DMPlexComputeGeometryFVM(celldm,&cellgeom,&facegeom);CHKERRQ(ierr);
    ierr = DMPlexGetMinRadius(celldm,&minradius);CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_WORLD,"PLEX(minradius) %1.4e\n",(double)minradius);CHKERRQ(ierr);
    ierr = VecDestroy(&cellgeom);CHKERRQ(ierr);
    ierr = VecDestroy(&facegeom);CHKERRQ(ierr);
    ierr = PetscSectionDestroy(&section);CHKERRQ(ierr);
  }

  /* Create the DMSwarm */
  ierr = DMCreate(PETSC_COMM_WORLD,&swarm);CHKERRQ(ierr);
  ierr = DMSetType(swarm,DMSWARM);CHKERRQ(ierr);
  ierr = DMSetDimension(swarm,dim);CHKERRQ(ierr);

  /* Configure swarm to be of type PIC */
  ierr = DMSwarmSetType(swarm,DMSWARM_PIC);CHKERRQ(ierr);
  ierr = DMSwarmSetCellDM(swarm,celldm);CHKERRQ(ierr);

  /* Register two scalar fields within the DMSwarm */
  ierr = DMSwarmRegisterPetscDatatypeField(swarm,"phi",1,PETSC_REAL);CHKERRQ(ierr);
  ierr = DMSwarmRegisterPetscDatatypeField(swarm,"region",1,PETSC_REAL);CHKERRQ(ierr);
  ierr = DMSwarmFinalizeFieldRegister(swarm);CHKERRQ(ierr);

  /* Set initial local sizes of the DMSwarm with a buffer length of zero */
  ierr = DMSwarmSetLocalSizes(swarm,4,0);CHKERRQ(ierr);

  /* Insert swarm coordinates cell-wise */
  /*ierr = DMSwarmInsertPointsUsingCellDM(swarm,DMSWARMPIC_LAYOUT_REGULAR,ppcell);CHKERRQ(ierr);*/
  ierr = DMSwarmInsertPointsUsingCellDM(swarm,DMSWARMPIC_LAYOUT_SUBDIVISION,ppcell);CHKERRQ(ierr);

  /* Define initial conditions for th swarm fields "phi" and "region" */
  {
    PetscReal *s_coor,*s_phi,*s_region;
    PetscInt npoints,p;

    ierr = DMSwarmGetLocalSize(swarm,&npoints);CHKERRQ(ierr);
    ierr = DMSwarmGetField(swarm,DMSwarmPICField_coor,NULL,NULL,(void**)&s_coor);CHKERRQ(ierr);
    ierr = DMSwarmGetField(swarm,"phi",NULL,NULL,(void**)&s_phi);CHKERRQ(ierr);
    ierr = DMSwarmGetField(swarm,"region",NULL,NULL,(void**)&s_region);CHKERRQ(ierr);
    for (p=0; p<npoints; p++) {
      PetscReal pos[2];
      pos[0] = s_coor[2*p+0];
      pos[1] = s_coor[2*p+1];

      s_region[p] = 1.0;
      s_phi[p] = 1.0 + PetscExpReal(-200.0*((pos[0]-0.5)*(pos[0]-0.5) + (pos[1]-0.5)*(pos[1]-0.5)));
    }
    ierr = DMSwarmRestoreField(swarm,"region",NULL,NULL,(void**)&s_region);CHKERRQ(ierr);
    ierr = DMSwarmRestoreField(swarm,"phi",NULL,NULL,(void**)&s_phi);CHKERRQ(ierr);
    ierr = DMSwarmRestoreField(swarm,DMSwarmPICField_coor,NULL,NULL,(void**)&s_coor);CHKERRQ(ierr);
  }

  /* Project initial value of phi onto the mesh */
  ierr = DMSwarmProjectFields(swarm,1,fieldnames,&pfields,PETSC_FALSE);CHKERRQ(ierr);

  if (view) {
    /* View swarm all swarm fields using data type PETSC_REAL */
    ierr = DMSwarmViewXDMF(swarm,"ic_dms.xmf");CHKERRQ(ierr);

    /* View projected swarm field "phi" */
    ierr = PetscViewerCreate(PETSC_COMM_WORLD,&viewer);CHKERRQ(ierr);
    ierr = PetscViewerSetType(viewer,PETSCVIEWERVTK);CHKERRQ(ierr);
    ierr = PetscViewerFileSetMode(viewer,FILE_MODE_WRITE);CHKERRQ(ierr);
    if (meshtype == 0) { /* DA */
      ierr = PetscViewerFileSetName(viewer,"ic_dmda.vts");CHKERRQ(ierr);
      ierr = VecView(pfields[0],viewer);CHKERRQ(ierr);
    }
    if (meshtype == 1) { /* PLEX */
      ierr = PetscViewerFileSetName(viewer,"ic_dmplex.vtk");CHKERRQ(ierr);
      ierr = DMView(celldm,viewer);CHKERRQ(ierr);
      ierr = VecView(pfields[0],viewer);CHKERRQ(ierr);
    }
    ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);
  }

  ierr = DMView(celldm,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  ierr = DMView(swarm,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);

  dt = 0.5 * minradius / PetscSqrtReal(vel[0]*vel[0] + vel[1]*vel[1]);
  for (tk=1; tk<=nt; tk++) {
    PetscReal *s_coor;
    PetscInt npoints,p;

    ierr = PetscPrintf(PETSC_COMM_WORLD,"[step %D]\n",tk);CHKERRQ(ierr);
    /* advect with analytic prescribed (constant) velocity field */
    ierr = DMSwarmGetLocalSize(swarm,&npoints);CHKERRQ(ierr);
    ierr = DMSwarmGetField(swarm,DMSwarmPICField_coor,NULL,NULL,(void**)&s_coor);CHKERRQ(ierr);
    for (p=0; p<npoints; p++) {
      s_coor[2*p+0] += dt * vel[0];
      s_coor[2*p+1] += dt * vel[1];
    }
    ierr = DMSwarmRestoreField(swarm,DMSwarmPICField_coor,NULL,NULL,(void**)&s_coor);CHKERRQ(ierr);

    ierr = DMSwarmMigrate(swarm,PETSC_TRUE);CHKERRQ(ierr);

    /* Ad-hoc cell filling algorithm */
    /*
       The injection frequency is chosen for default DA case.
       They will likely not be appropriate for the general case using an unstructure PLEX mesh.
    */
    if (tk%10 == 0) {
      PetscReal dx = 1.0/32.0;
      PetscInt npoints_dir_x[] = { 32, 1 };
      PetscReal min[2],max[2];

      min[0] = 0.5 * dx;  max[0] = 0.5 * dx + 31.0 * dx;
      min[1] = 0.5 * dx;  max[1] = 0.5 * dx;
      ierr = DMSwarmSetPointsUniformCoordinates(swarm,min,max,npoints_dir_x,ADD_VALUES);CHKERRQ(ierr);
    }
    if (tk%2 == 0) {
      PetscReal dx = 1.0/32.0;
      PetscInt npoints_dir_y[] = { 2, 31 };
      PetscReal min[2],max[2];

      min[0] = 0.05 * dx; max[0] = 0.5 * dx;
      min[1] = 0.5 * dx;  max[1] = 0.5 * dx + 31.0 * dx;
      ierr = DMSwarmSetPointsUniformCoordinates(swarm,min,max,npoints_dir_y,ADD_VALUES);CHKERRQ(ierr);
    }

    /* Project swarm field "phi" onto the cell DM */
    ierr = DMSwarmProjectFields(swarm,1,fieldnames,&pfields,PETSC_TRUE);CHKERRQ(ierr);

    if (view) {
      PetscViewer viewer;
      char fname[PETSC_MAX_PATH_LEN];

      /* View swarm fields */
      ierr = PetscSNPrintf(fname,PETSC_MAX_PATH_LEN-1,"step%.4D_dms.xmf",tk);CHKERRQ(ierr);
      ierr = DMSwarmViewXDMF(swarm,fname);CHKERRQ(ierr);

      /* View projected field */
      ierr = PetscViewerCreate(PETSC_COMM_WORLD,&viewer);CHKERRQ(ierr);
      ierr = PetscViewerSetType(viewer,PETSCVIEWERVTK);CHKERRQ(ierr);
      ierr = PetscViewerFileSetMode(viewer,FILE_MODE_WRITE);CHKERRQ(ierr);

      if (meshtype == 0) { /* DA */
        ierr = PetscSNPrintf(fname,PETSC_MAX_PATH_LEN-1,"step%.4D_dmda.vts",tk);CHKERRQ(ierr);
        ierr = PetscViewerFileSetName(viewer,fname);CHKERRQ(ierr);
        ierr = VecView(pfields[0],viewer);CHKERRQ(ierr);
      }
      if (meshtype == 1) { /* PLEX */
        ierr = PetscSNPrintf(fname,PETSC_MAX_PATH_LEN-1,"step%.4D_dmplex.vtk",tk);CHKERRQ(ierr);
        ierr = PetscViewerFileSetName(viewer,fname);CHKERRQ(ierr);
        ierr = DMView(celldm,viewer);CHKERRQ(ierr);
        ierr = VecView(pfields[0],viewer);CHKERRQ(ierr);
      }
      ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);
    }

  }
  ierr = VecDestroy(&pfields[0]);CHKERRQ(ierr);
  ierr = PetscFree(pfields);CHKERRQ(ierr);
  ierr = DMDestroy(&celldm);CHKERRQ(ierr);
  ierr = DMDestroy(&swarm);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

int main(int argc,char **args)
{
  PetscErrorCode ierr;
  PetscInt ppcell = 1;
  PetscInt meshtype = 0;

  ierr = PetscInitialize(&argc,&args,(char*)0,help);if (ierr) return ierr;
  ierr = PetscOptionsGetInt(NULL,NULL,"-ppcell",&ppcell,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(NULL,NULL,"-meshtype",&meshtype,NULL);CHKERRQ(ierr);
  PetscCheckFalse(meshtype > 1,PETSC_COMM_WORLD,PETSC_ERR_USER,"-meshtype <value> must be 0 or 1");

  ierr = pic_advect(ppcell,meshtype);CHKERRQ(ierr);

  ierr = PetscFinalize();
  return ierr;
}
