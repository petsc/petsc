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
  PetscErrorCode ierr;

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
    DM               dm2;
    PetscPartitioner part;

    ierr = DMPlexCreateBoxMesh(PETSC_COMM_WORLD,3,PETSC_FALSE,cells,l,u,NULL,PETSC_FALSE,&dm);CHKERRQ(ierr);
    ierr = DMPlexGetPartitioner(dm,&part);CHKERRQ(ierr);
    ierr = PetscPartitionerSetType(part,PETSCPARTITIONERSIMPLE);CHKERRQ(ierr);
    ierr = DMPlexDistribute(dm,0,NULL,&dm2);CHKERRQ(ierr);
    if (dm2) {
      ierr = DMDestroy(&dm);CHKERRQ(ierr);
      dm   = dm2;
    }
  } else {
    ierr = DMDACreate3d(PETSC_COMM_WORLD,DM_BOUNDARY_NONE,DM_BOUNDARY_NONE,DM_BOUNDARY_NONE,DMDA_STENCIL_BOX,cells[0]+1,cells[1]+1,cells[2]+1,PETSC_DECIDE,PETSC_DECIDE,PETSC_DECIDE,1,1,NULL,NULL,NULL,&dm);CHKERRQ(ierr);
  }
  ierr = DMSetUp(dm);CHKERRQ(ierr);
  if (!plex) {
    ierr = DMDASetUniformCoordinates(dm,l[0],u[0],l[1],u[1],l[2],u[2]);CHKERRQ(ierr);
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

    ierr = PetscOptionsGetInt(NULL,NULL,"-order",&dof,NULL);CHKERRQ(ierr);
    dof += 1;

    ierr = PetscMalloc1(dof,&nodes);CHKERRQ(ierr);
    ierr = PetscMalloc1(dof,&weights);CHKERRQ(ierr);
    ierr = PetscDTGaussLobattoLegendreQuadrature(dof,PETSCGAUSSLOBATTOLEGENDRE_VIA_LINEAR_ALGEBRA,nodes,weights);CHKERRQ(ierr);
    ierr = DMGetCoordinatesLocal(dm,&cv);CHKERRQ(ierr);
    ierr = DMGetCoordinateDM(dm,&cdm);CHKERRQ(ierr);
    if (plex) {
      PetscInt cEnd,cStart;

      ierr = DMPlexGetHeightStratum(dm,0,&cStart,&cEnd);CHKERRQ(ierr);
      ierr = DMGetCoordinateSection(dm,&cSec);CHKERRQ(ierr);
      nel  = cEnd - cStart;
      nex  = nel;
      ney  = 1;
      nez  = 1;
    } else {
      ierr = DMDAVecGetArray(cdm,cv,&_coords);CHKERRQ(ierr);
      ierr = DMDAGetElementsCorners(dm,&gx,&gy,&gz);CHKERRQ(ierr);
      ierr = DMDAGetElementsSizes(dm,&nex,&ney,&nez);CHKERRQ(ierr);
      nel  = nex*ney*nez;
    }
    ierr = VecCreate(PETSC_COMM_WORLD,&v);CHKERRQ(ierr);
    ierr = VecSetSizes(v,3*dof*dof*dof*nel,PETSC_DECIDE);CHKERRQ(ierr);
    ierr = VecSetType(v,VECSTANDARD);CHKERRQ(ierr);
    ierr = VecGetArray(v,&c);CHKERRQ(ierr);
    cptr = c;
    for (k=gz;k<gz+nez;k++) {
      for (j=gy;j<gy+ney;j++) {
        for (i=gx;i<gx+nex;i++) {
          if (plex) {
            PetscScalar *t = NULL;

            ierr = DMPlexVecGetClosure(dm,cSec,cv,i,NULL,&t);CHKERRQ(ierr);
            shift[0] = t[0];
            shift[1] = t[1];
            shift[2] = t[2];
            ierr = DMPlexVecRestoreClosure(dm,cSec,cv,i,NULL,&t);CHKERRQ(ierr);
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
                ierr = MapPoint(xyz,cptr);CHKERRQ(ierr);
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
      ierr = DMDAVecRestoreArray(cdm,cv,&_coords);CHKERRQ(ierr);
    }
    ierr = VecRestoreArray(v,&c);CHKERRQ(ierr);
    ierr = PetscSNPrintf(name,sizeof(name),"FiniteElementCollection: L2_T1_3D_P%D",dof-1);CHKERRQ(ierr);
    ierr = PetscObjectSetName((PetscObject)v,name);CHKERRQ(ierr);
    ierr = PetscObjectCompose((PetscObject)dm,"_glvis_mesh_coords",(PetscObject)v);CHKERRQ(ierr);
    ierr = VecDestroy(&v);CHKERRQ(ierr);
    ierr = PetscFree(nodes);CHKERRQ(ierr);
    ierr = PetscFree(weights);CHKERRQ(ierr);
    ierr = DMViewFromOptions(dm,NULL,"-view");CHKERRQ(ierr);
  } else { /* map the whole domain to a sphere */
    ierr = DMGetCoordinates(dm,&v);CHKERRQ(ierr);
    ierr = VecGetLocalSize(v,&nl);CHKERRQ(ierr);
    ierr = VecGetArray(v,&c);CHKERRQ(ierr);
    for (i=0;i<nl/3;i++) {
      ierr = MapPoint(c+3*i,c+3*i);CHKERRQ(ierr);
    }
    ierr = VecRestoreArray(v,&c);CHKERRQ(ierr);
    ierr = DMSetCoordinates(dm,v);CHKERRQ(ierr);
    ierr = DMViewFromOptions(dm,NULL,"-view");CHKERRQ(ierr);
  }
  ierr = DMDestroy(&dm);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

int main(int argc, char *argv[])
{
  PetscErrorCode ierr;
  PetscBool      ho = PETSC_TRUE;
  PetscBool      plex = PETSC_FALSE;
  PetscInt       cells[3] = {2,2,2};

  ierr = PetscInitialize(&argc,&argv,0,help);if (ierr) return ierr;
  ierr = PetscOptionsGetBool(NULL,NULL,"-ho",&ho,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetBool(NULL,NULL,"-plex",&plex,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(NULL,NULL,"-nex",&cells[0],NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(NULL,NULL,"-ney",&cells[1],NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(NULL,NULL,"-nez",&cells[2],NULL);CHKERRQ(ierr);
  ierr = test_3d(cells,plex,ho);CHKERRQ(ierr);
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
