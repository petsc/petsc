static char help[] = "Test GLVis high-order support with DMDAs\n\n";

#include <petscdm.h>
#include <petscdmda.h>
#include <petscdmplex.h>
#include <petscdt.h>

static PetscErrorCode MapPoint(PetscScalar xyz[],PetscScalar mxyz[])
{
  PetscScalar x,y,z,x2,y2,z2;

  x = xyz[0];
  y = xyz[1];
  z = xyz[2];
  x2 = x*x;
  y2 = y*y;
  z2 = z*z;
  mxyz[0] = x*PetscSqrtScalar(1.0-y2/2.0-z2/2.0 + y2*z2/3.0);
  mxyz[1] = y*PetscSqrtScalar(1.0-z2/2.0-x2/2.0 + z2*x2/3.0);
  mxyz[2] = z*PetscSqrtScalar(1.0-x2/2.0-y2/2.0 + x2*y2/3.0);
  return 0;
}

static PetscErrorCode test_3d(PetscInt cells[], PetscBool plex, PetscBool ho)
{
  DM             dm;
  Vec            v;
  PetscScalar    *c;
  PetscInt       nl,i;
  PetscReal      u[3] = {1.0,1.0,1.0}, l[3] = {-1.0,-1.0,-1.0};

  PetscFunctionBeginUser;
  if (ho) {
    u[0] = 2.*cells[0];
    u[1] = 2.*cells[1];
    u[2] = 2.*cells[2];
    l[0] = 0.;
    l[1] = 0.;
    l[2] = 0.;
  }
  if (plex) {
    CHKERRQ(DMCreate(PETSC_COMM_WORLD, &dm));
    CHKERRQ(DMSetType(dm, DMPLEX));
    CHKERRQ(DMSetFromOptions(dm));
  } else {
    CHKERRQ(DMDACreate3d(PETSC_COMM_WORLD,DM_BOUNDARY_NONE,DM_BOUNDARY_NONE,DM_BOUNDARY_NONE,DMDA_STENCIL_BOX,cells[0]+1,cells[1]+1,cells[2]+1,PETSC_DECIDE,PETSC_DECIDE,PETSC_DECIDE,1,1,NULL,NULL,NULL,&dm));
  }
  CHKERRQ(DMSetUp(dm));
  if (!plex) {
    CHKERRQ(DMDASetUniformCoordinates(dm,l[0],u[0],l[1],u[1],l[2],u[2]));
  }
  if (ho) { /* each element mapped to a sphere */
    DM           cdm;
    Vec          cv;
    PetscSection cSec;
    DMDACoor3d   ***_coords;
    PetscScalar  shift[3],*cptr;
    PetscInt     nel,dof = 3,nex,ney,nez,gx = 0,gy = 0,gz = 0;
    PetscInt     i,j,k,pi,pj,pk;
    PetscReal    *nodes,*weights;
    char         name[256];

    CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-order",&dof,NULL));
    dof += 1;

    CHKERRQ(PetscMalloc1(dof,&nodes));
    CHKERRQ(PetscMalloc1(dof,&weights));
    CHKERRQ(PetscDTGaussLobattoLegendreQuadrature(dof,PETSCGAUSSLOBATTOLEGENDRE_VIA_LINEAR_ALGEBRA,nodes,weights));
    CHKERRQ(DMGetCoordinatesLocal(dm,&cv));
    CHKERRQ(DMGetCoordinateDM(dm,&cdm));
    if (plex) {
      PetscInt cEnd,cStart;

      CHKERRQ(DMPlexGetHeightStratum(dm,0,&cStart,&cEnd));
      CHKERRQ(DMGetCoordinateSection(dm,&cSec));
      nel  = cEnd - cStart;
      nex  = nel;
      ney  = 1;
      nez  = 1;
    } else {
      CHKERRQ(DMDAVecGetArray(cdm,cv,&_coords));
      CHKERRQ(DMDAGetElementsCorners(dm,&gx,&gy,&gz));
      CHKERRQ(DMDAGetElementsSizes(dm,&nex,&ney,&nez));
      nel  = nex*ney*nez;
    }
    CHKERRQ(VecCreate(PETSC_COMM_WORLD,&v));
    CHKERRQ(VecSetSizes(v,3*dof*dof*dof*nel,PETSC_DECIDE));
    CHKERRQ(VecSetType(v,VECSTANDARD));
    CHKERRQ(VecGetArray(v,&c));
    cptr = c;
    for (k=gz;k<gz+nez;k++) {
      for (j=gy;j<gy+ney;j++) {
        for (i=gx;i<gx+nex;i++) {
          if (plex) {
            PetscScalar *t = NULL;

            CHKERRQ(DMPlexVecGetClosure(dm,cSec,cv,i,NULL,&t));
            shift[0] = t[0];
            shift[1] = t[1];
            shift[2] = t[2];
            CHKERRQ(DMPlexVecRestoreClosure(dm,cSec,cv,i,NULL,&t));
          } else {
            shift[0] = _coords[k][j][i].x;
            shift[1] = _coords[k][j][i].y;
            shift[2] = _coords[k][j][i].z;
          }
          for (pk=0;pk<dof;pk++) {
            PetscScalar xyz[3];

            xyz[2] = nodes[pk];
            for (pj=0;pj<dof;pj++) {
              xyz[1] = nodes[pj];
              for (pi=0;pi<dof;pi++) {
                xyz[0] = nodes[pi];
                CHKERRQ(MapPoint(xyz,cptr));
                cptr[0] += shift[0];
                cptr[1] += shift[1];
                cptr[2] += shift[2];
                cptr += 3;
              }
            }
          }
        }
      }
    }
    if (!plex) {
      CHKERRQ(DMDAVecRestoreArray(cdm,cv,&_coords));
    }
    CHKERRQ(VecRestoreArray(v,&c));
    CHKERRQ(PetscSNPrintf(name,sizeof(name),"FiniteElementCollection: L2_T1_3D_P%D",dof-1));
    CHKERRQ(PetscObjectSetName((PetscObject)v,name));
    CHKERRQ(PetscObjectCompose((PetscObject)dm,"_glvis_mesh_coords",(PetscObject)v));
    CHKERRQ(VecDestroy(&v));
    CHKERRQ(PetscFree(nodes));
    CHKERRQ(PetscFree(weights));
    CHKERRQ(DMViewFromOptions(dm,NULL,"-view"));
  } else { /* map the whole domain to a sphere */
    CHKERRQ(DMGetCoordinates(dm,&v));
    CHKERRQ(VecGetLocalSize(v,&nl));
    CHKERRQ(VecGetArray(v,&c));
    for (i=0;i<nl/3;i++) {
      CHKERRQ(MapPoint(c+3*i,c+3*i));
    }
    CHKERRQ(VecRestoreArray(v,&c));
    CHKERRQ(DMSetCoordinates(dm,v));
    CHKERRQ(DMViewFromOptions(dm,NULL,"-view"));
  }
  CHKERRQ(DMDestroy(&dm));
  PetscFunctionReturn(0);
}

int main(int argc, char *argv[])
{
  PetscErrorCode ierr;
  PetscBool      ho = PETSC_TRUE;
  PetscBool      plex = PETSC_FALSE;
  PetscInt       cells[3] = {2,2,2};

  ierr = PetscInitialize(&argc,&argv,0,help);if (ierr) return ierr;
  CHKERRQ(PetscOptionsGetBool(NULL,NULL,"-ho",&ho,NULL));
  CHKERRQ(PetscOptionsGetBool(NULL,NULL,"-plex",&plex,NULL));
  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-nex",&cells[0],NULL));
  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-ney",&cells[1],NULL));
  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-nez",&cells[2],NULL));
  CHKERRQ(test_3d(cells,plex,ho));
  ierr = PetscFinalize();
  return ierr;
}

/*TEST

   testset:
     nsize: 1
     args: -view glvis:

     test:
        suffix: dmda_glvis_ho
        args: -nex 1 -ney 1 -nez 1

     test:
        suffix: dmda_glvis_lo
        args: -ho 0

     test:
        suffix: dmplex_glvis_ho
        args: -nex 1 -ney 1 -nez 1

     test:
        suffix: dmplex_glvis_lo
        args: -ho 0

TEST*/
