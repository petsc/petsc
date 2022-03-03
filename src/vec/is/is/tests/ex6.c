
static char help[] = "Tests ISComplement().\n\n";

#include <petscis.h>
#include <petscviewer.h>

int main(int argc,char **argv)
{
  PetscMPIInt    rank,size;
  PetscInt       i,j,n,cnt=0,rstart,rend;
  PetscBool      flg;
  IS             is[2],isc;
  PetscErrorCode ierr;

  ierr = PetscInitialize(&argc,&argv,(char*)0,help);if (ierr) return ierr;
  CHKERRMPI(MPI_Comm_rank(PETSC_COMM_WORLD,&rank));
  CHKERRMPI(MPI_Comm_size(PETSC_COMM_WORLD,&size));

  n      = 3*size;              /* Number of local indices, same on each process. */
  rstart = 3*(size+2)*rank;     /* start of local range */
  rend   = 3*(size+2)*(rank+1); /* end of local range */
  for (i=0; i<2; i++) {
    CHKERRQ(ISCreate(PETSC_COMM_WORLD,&is[i]));
    CHKERRQ(ISSetType(is[i],ISGENERAL));
  }
  {
    PetscBool *mask;

    CHKERRQ(PetscCalloc1(rend-rstart,&mask));
    for (i=0; i<3; i++) {
      for (j=0; j<size; j++) {
        mask[i*(size+2)+j] = PETSC_TRUE;
      }
    }
    CHKERRQ(ISGeneralSetIndicesFromMask(is[0],rstart,rend,mask));
    CHKERRQ(PetscFree(mask));
  }
  {
    PetscInt *indices;

    CHKERRQ(PetscMalloc1(n,&indices));
    for (i=0; i<3; i++) {
      for (j=0; j<size; j++) indices[cnt++] = rstart+i*(size+2)+j;
    }
    PetscCheckFalse(cnt != n,PETSC_COMM_SELF,PETSC_ERR_PLIB,"inconsistent count");
    CHKERRQ(ISGeneralSetIndices(is[1],n,indices,PETSC_COPY_VALUES));
    CHKERRQ(PetscFree(indices));
  }

  CHKERRQ(ISEqual(is[0],is[1],&flg));
  PetscCheck(flg,PETSC_COMM_SELF,PETSC_ERR_PLIB,"is[0] should be equal to is[1]");

  CHKERRQ(ISComplement(is[0],rstart,rend,&isc));
  CHKERRQ(ISView(is[0],PETSC_VIEWER_STDOUT_WORLD));
  CHKERRQ(ISView(isc,PETSC_VIEWER_STDOUT_WORLD));

  for (i=0; i<2; i++) CHKERRQ(ISDestroy(&is[i]));
  CHKERRQ(ISDestroy(&isc));
  ierr = PetscFinalize();
  return ierr;
}

/*TEST

   test:
      suffix: 3
      nsize: 3

TEST*/
