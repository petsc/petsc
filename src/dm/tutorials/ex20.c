
static char help[] = "DMSwarm-PIC demonstator of inserting points into a cell DM \n\
Options: \n\
-mode {0,1} : 0 ==> DMDA, 1 ==> DMPLEX cell DM \n\
-dim {2,3}  : spatial dimension\n";

#include <petsc.h>
#include <petscdm.h>
#include <petscdmda.h>
#include <petscdmplex.h>
#include <petscdmswarm.h>

PetscErrorCode pic_insert_DMDA(PetscInt dim)
{
  PetscErrorCode ierr;
  DM             celldm = NULL,swarm;
  PetscInt       dof,stencil_width;
  PetscReal      min[3],max[3];
  PetscInt       ndir[3];

  PetscFunctionBegin;
  /* Create the background cell DM */
  dof = 1;
  stencil_width = 1;
  if (dim == 2) {
    ierr = DMDACreate2d(PETSC_COMM_WORLD,DM_BOUNDARY_NONE,DM_BOUNDARY_NONE,DMDA_STENCIL_BOX,25,13,PETSC_DECIDE,PETSC_DECIDE,dof,stencil_width,NULL,NULL,&celldm);CHKERRQ(ierr);
  }
  if (dim == 3) {
    ierr = DMDACreate3d(PETSC_COMM_WORLD,DM_BOUNDARY_NONE,DM_BOUNDARY_NONE,DM_BOUNDARY_NONE,DMDA_STENCIL_BOX,25,13,19,PETSC_DECIDE,PETSC_DECIDE,PETSC_DECIDE,dof,stencil_width,NULL,NULL,NULL,&celldm);CHKERRQ(ierr);
  }

  ierr = DMDASetElementType(celldm,DMDA_ELEMENT_Q1);CHKERRQ(ierr);
  ierr = DMSetFromOptions(celldm);CHKERRQ(ierr);
  ierr = DMSetUp(celldm);CHKERRQ(ierr);

  ierr = DMDASetUniformCoordinates(celldm,0.0,2.0,0.0,1.0,0.0,1.5);CHKERRQ(ierr);

  /* Create the DMSwarm */
  ierr = DMCreate(PETSC_COMM_WORLD,&swarm);CHKERRQ(ierr);
  ierr = DMSetType(swarm,DMSWARM);CHKERRQ(ierr);
  ierr = DMSetDimension(swarm,dim);CHKERRQ(ierr);

  /* Configure swarm to be of type PIC */
  ierr = DMSwarmSetType(swarm,DMSWARM_PIC);CHKERRQ(ierr);
  ierr = DMSwarmSetCellDM(swarm,celldm);CHKERRQ(ierr);

  /* Register two scalar fields within the DMSwarm */
  ierr = DMSwarmRegisterPetscDatatypeField(swarm,"viscosity",1,PETSC_DOUBLE);CHKERRQ(ierr);
  ierr = DMSwarmRegisterPetscDatatypeField(swarm,"density",1,PETSC_DOUBLE);CHKERRQ(ierr);
  ierr = DMSwarmFinalizeFieldRegister(swarm);CHKERRQ(ierr);

  /* Set initial local sizes of the DMSwarm with a buffer length of zero */
  ierr = DMSwarmSetLocalSizes(swarm,4,0);CHKERRQ(ierr);

  /* Insert swarm coordinates cell-wise */
  ierr = DMSwarmInsertPointsUsingCellDM(swarm,DMSWARMPIC_LAYOUT_REGULAR,3);CHKERRQ(ierr);
  min[0] = 0.5; max[0] = 0.7;
  min[1] = 0.5; max[1] = 0.8;
  min[2] = 0.5; max[2] = 0.9;
  ndir[0] = ndir[1] = ndir[2] = 30;
  ierr = DMSwarmSetPointsUniformCoordinates(swarm,min,max,ndir,ADD_VALUES);CHKERRQ(ierr);

  /* This should be dispatched from a regular DMView() */
  ierr = DMSwarmViewXDMF(swarm,"ex20.xmf");CHKERRQ(ierr);
  ierr = DMView(celldm,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  ierr = DMView(swarm,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);

  {
    PetscInt    npoints,*list;
    PetscMPIInt rank;

    ierr = MPI_Comm_rank(PETSC_COMM_WORLD,&rank);CHKERRQ(ierr);
    ierr = DMSwarmSortGetAccess(swarm);CHKERRQ(ierr);
    ierr = DMSwarmSortGetNumberOfPointsPerCell(swarm,0,&npoints);CHKERRQ(ierr);
    ierr = DMSwarmSortGetPointsPerCell(swarm,rank,&npoints,&list);CHKERRQ(ierr);
    ierr = PetscFree(list);CHKERRQ(ierr);
    ierr = DMSwarmSortRestoreAccess(swarm);CHKERRQ(ierr);
  }
  ierr = DMSwarmMigrate(swarm,PETSC_FALSE);CHKERRQ(ierr);
  ierr = DMDestroy(&celldm);CHKERRQ(ierr);
  ierr = DMDestroy(&swarm);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode pic_insert_DMPLEX_with_cell_list(PetscInt dim)
{
  PetscErrorCode ierr;
  DM             celldm = NULL,swarm,distributedMesh = NULL;
  const  char    *fieldnames[] = {"viscosity"};

  PetscFunctionBegin;
  /* Create the background cell DM */
  if (dim == 2) {
    PetscInt   cells_per_dim[2],nx[2];
    PetscInt   n_tricells;
    PetscInt   n_trivert;
    PetscInt   *tricells;
    PetscReal  *trivert,dx,dy;
    PetscInt   ii,jj,cnt;

    cells_per_dim[0] = 4;
    cells_per_dim[1] = 4;
    n_tricells = cells_per_dim[0] * cells_per_dim[1] * 2;
    nx[0] = cells_per_dim[0] + 1;
    nx[1] = cells_per_dim[1] + 1;
    n_trivert = nx[0] * nx[1];

    ierr = PetscMalloc1(n_tricells*3,&tricells);CHKERRQ(ierr);
    ierr = PetscMalloc1(nx[0]*nx[1]*2,&trivert);CHKERRQ(ierr);

    /* verts */
    cnt = 0;
    dx = 2.0/((PetscReal)cells_per_dim[0]);
    dy = 1.0/((PetscReal)cells_per_dim[1]);
    for (jj=0; jj<nx[1]; jj++) {
      for (ii=0; ii<nx[0]; ii++) {
        trivert[2*cnt+0] = 0.0 + ii * dx;
        trivert[2*cnt+1] = 0.0 + jj * dy;
        cnt++;
      }
    }

    /* connectivity */
    cnt = 0;
    for (jj=0; jj<cells_per_dim[1]; jj++) {
      for (ii=0; ii<cells_per_dim[0]; ii++) {
        PetscInt idx,idx0,idx1,idx2,idx3;

        idx = (ii) + (jj) * nx[0];
        idx0 = idx;
        idx1 = idx0 + 1;
        idx2 = idx1 + nx[0];
        idx3 = idx0 + nx[0];

        tricells[3*cnt+0] = idx0;
        tricells[3*cnt+1] = idx1;
        tricells[3*cnt+2] = idx2;
        cnt++;

        tricells[3*cnt+0] = idx0;
        tricells[3*cnt+1] = idx2;
        tricells[3*cnt+2] = idx3;
        cnt++;
      }
    }
    ierr = DMPlexCreateFromCellListPetsc(PETSC_COMM_WORLD,dim,n_tricells,n_trivert,3,PETSC_TRUE,tricells,dim,trivert,&celldm);CHKERRQ(ierr);
    ierr = PetscFree(trivert);CHKERRQ(ierr);
    ierr = PetscFree(tricells);CHKERRQ(ierr);
  }
  if (dim == 3) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_SUP,"Only 2D PLEX example supported");

  /* Distribute mesh over processes */
  ierr = DMPlexDistribute(celldm,0,NULL,&distributedMesh);CHKERRQ(ierr);
  if (distributedMesh) {
    ierr = DMDestroy(&celldm);CHKERRQ(ierr);
    celldm = distributedMesh;
  }
  ierr = DMSetFromOptions(celldm);CHKERRQ(ierr);
  {
    PetscInt     numComp[] = {1};
    PetscInt     numDof[] = {1,0,0}; /* vert, edge, cell */
    PetscInt     numBC = 0;
    PetscSection section;

    ierr = DMPlexCreateSection(celldm,NULL,numComp,numDof,numBC,NULL,NULL,NULL,NULL,&section);CHKERRQ(ierr);
    ierr = DMSetLocalSection(celldm,section);CHKERRQ(ierr);
    ierr = PetscSectionDestroy(&section);CHKERRQ(ierr);
  }
  ierr = DMSetUp(celldm);CHKERRQ(ierr);
  {
    PetscViewer viewer;

    ierr = PetscViewerCreate(PETSC_COMM_WORLD,&viewer);CHKERRQ(ierr);
    ierr = PetscViewerSetType(viewer,PETSCVIEWERVTK);CHKERRQ(ierr);
    ierr = PetscViewerFileSetMode(viewer,FILE_MODE_WRITE);CHKERRQ(ierr);
    ierr = PetscViewerFileSetName(viewer,"ex20plex.vtk");CHKERRQ(ierr);
    ierr = DMView(celldm,viewer);CHKERRQ(ierr);
    ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);
  }

  /* Create the DMSwarm */
  ierr = DMCreate(PETSC_COMM_WORLD,&swarm);CHKERRQ(ierr);
  ierr = DMSetType(swarm,DMSWARM);CHKERRQ(ierr);
  ierr = DMSetDimension(swarm,dim);CHKERRQ(ierr);

  ierr = DMSwarmSetType(swarm,DMSWARM_PIC);CHKERRQ(ierr);
  ierr = DMSwarmSetCellDM(swarm,celldm);CHKERRQ(ierr);

  /* Register two scalar fields within the DMSwarm */
  ierr = DMSwarmRegisterPetscDatatypeField(swarm,"viscosity",1,PETSC_DOUBLE);CHKERRQ(ierr);
  ierr = DMSwarmRegisterPetscDatatypeField(swarm,"density",1,PETSC_DOUBLE);CHKERRQ(ierr);
  ierr = DMSwarmFinalizeFieldRegister(swarm);CHKERRQ(ierr);

  /* Set initial local sizes of the DMSwarm with a buffer length of zero */
  ierr = DMSwarmSetLocalSizes(swarm,4,0);CHKERRQ(ierr);

  /* Insert swarm coordinates cell-wise */
  ierr = DMSwarmInsertPointsUsingCellDM(swarm,DMSWARMPIC_LAYOUT_SUBDIVISION,2);CHKERRQ(ierr);
  ierr = DMSwarmViewFieldsXDMF(swarm,"ex20.xmf",1,fieldnames);CHKERRQ(ierr);
  ierr = DMView(celldm,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  ierr = DMView(swarm,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  ierr = DMDestroy(&celldm);CHKERRQ(ierr);
  ierr = DMDestroy(&swarm);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode pic_insert_DMPLEX(PetscBool is_simplex,PetscInt dim)
{
  PetscErrorCode ierr;
  DM             celldm,swarm,distributedMesh = NULL;
  const char     *fieldnames[] = {"viscosity","DMSwarm_rank"};

  PetscFunctionBegin;

  /* Create the background cell DM */
  {
    PetscInt faces[3] = {4, 2, 4};
    ierr = DMPlexCreateBoxMesh(PETSC_COMM_WORLD, dim, is_simplex, faces, NULL, NULL, NULL, PETSC_TRUE, &celldm);CHKERRQ(ierr);
  }

  /* Distribute mesh over processes */
  ierr = DMPlexDistribute(celldm,0,NULL,&distributedMesh);CHKERRQ(ierr);
  if (distributedMesh) {
    ierr = DMDestroy(&celldm);CHKERRQ(ierr);
    celldm = distributedMesh;
  }
  ierr = DMSetFromOptions(celldm);CHKERRQ(ierr);
  {
    PetscInt     numComp[] = {1};
    PetscInt     numDof[] = {1,0,0}; /* vert, edge, cell */
    PetscInt     numBC = 0;
    PetscSection section;

    ierr = DMPlexCreateSection(celldm,NULL,numComp,numDof,numBC,NULL,NULL,NULL,NULL,&section);CHKERRQ(ierr);
    ierr = DMSetLocalSection(celldm,section);CHKERRQ(ierr);
    ierr = PetscSectionDestroy(&section);CHKERRQ(ierr);
  }
  ierr = DMSetUp(celldm);CHKERRQ(ierr);
  {
    PetscViewer viewer;

    ierr = PetscViewerCreate(PETSC_COMM_WORLD,&viewer);CHKERRQ(ierr);
    ierr = PetscViewerSetType(viewer,PETSCVIEWERVTK);CHKERRQ(ierr);
    ierr = PetscViewerFileSetMode(viewer,FILE_MODE_WRITE);CHKERRQ(ierr);
    ierr = PetscViewerFileSetName(viewer,"ex20plex.vtk");CHKERRQ(ierr);
    ierr = DMView(celldm,viewer);CHKERRQ(ierr);
    ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);
  }

  ierr = DMCreate(PETSC_COMM_WORLD,&swarm);CHKERRQ(ierr);
  ierr = DMSetType(swarm,DMSWARM);CHKERRQ(ierr);
  ierr = DMSetDimension(swarm,dim);CHKERRQ(ierr);

  ierr = DMSwarmSetType(swarm,DMSWARM_PIC);CHKERRQ(ierr);
  ierr = DMSwarmSetCellDM(swarm,celldm);CHKERRQ(ierr);

  /* Register two scalar fields within the DMSwarm */
  ierr = DMSwarmRegisterPetscDatatypeField(swarm,"viscosity",1,PETSC_DOUBLE);CHKERRQ(ierr);
  ierr = DMSwarmRegisterPetscDatatypeField(swarm,"density",1,PETSC_DOUBLE);CHKERRQ(ierr);
  ierr = DMSwarmFinalizeFieldRegister(swarm);CHKERRQ(ierr);

  /* Set initial local sizes of the DMSwarm with a buffer length of zero */
  ierr = DMSwarmSetLocalSizes(swarm,4,0);CHKERRQ(ierr);

  /* Insert swarm coordinates cell-wise */
  ierr = DMSwarmInsertPointsUsingCellDM(swarm,DMSWARMPIC_LAYOUT_GAUSS,3);CHKERRQ(ierr);
  ierr = DMSwarmViewFieldsXDMF(swarm,"ex20.xmf",2,fieldnames);CHKERRQ(ierr);
  ierr = DMView(celldm,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  ierr = DMView(swarm,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  ierr = DMDestroy(&celldm);CHKERRQ(ierr);
  ierr = DMDestroy(&swarm);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

int main(int argc,char **args)
{
  PetscErrorCode ierr;
  PetscInt       mode = 0;
  PetscInt       dim = 2;

  ierr = PetscInitialize(&argc,&args,(char*)0,help);if (ierr) return ierr;
  ierr = PetscOptionsGetInt(NULL,NULL,"-mode",&mode,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(NULL,NULL,"-dim",&dim,NULL);CHKERRQ(ierr);
  switch (mode) {
    case 0:
      ierr = pic_insert_DMDA(dim);CHKERRQ(ierr);
      break;
    case 1:
      /* tri / tet */
      ierr = pic_insert_DMPLEX(PETSC_TRUE,dim);CHKERRQ(ierr);
      break;
    case 2:
      /* quad / hex */
      ierr = pic_insert_DMPLEX(PETSC_FALSE,dim);CHKERRQ(ierr);
      break;
    default:
      ierr = pic_insert_DMDA(dim);CHKERRQ(ierr);
      break;
  }
  ierr = PetscFinalize();
  return ierr;
}

/*TEST

   test:
      args:
      requires: !complex double
      filter: grep -v DM_ | grep -v atomic
      filter_output: grep -v atomic

   test:
      suffix: 2
      requires: triangle double !complex
      args: -mode 1
      filter: grep -v DM_ | grep -v atomic
      filter_output: grep -v atomic

TEST*/
