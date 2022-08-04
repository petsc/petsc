
static char help[] = "Tests ISComplement().\n\n";

#include <petscis.h>
#include <petscviewer.h>

int main(int argc,char **argv)
{
  PetscMPIInt    rank,size;
  PetscInt       i,j,n,cnt=0,rstart,rend;
  PetscBool      flg;
  IS             is[2],isc;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc,&argv,(char*)0,help));
  PetscCallMPI(MPI_Comm_rank(PETSC_COMM_WORLD,&rank));
  PetscCallMPI(MPI_Comm_size(PETSC_COMM_WORLD,&size));

  n      = 3*size;              /* Number of local indices, same on each process. */
  rstart = 3*(size+2)*rank;     /* start of local range */
  rend   = 3*(size+2)*(rank+1); /* end of local range */
  for (i=0; i<2; i++) {
    PetscCall(ISCreate(PETSC_COMM_WORLD,&is[i]));
    PetscCall(ISSetType(is[i],ISGENERAL));
  }
  {
    PetscBool *mask;

    PetscCall(PetscCalloc1(rend-rstart,&mask));
    for (i=0; i<3; i++) {
      for (j=0; j<size; j++) {
        mask[i*(size+2)+j] = PETSC_TRUE;
      }
    }
    PetscCall(ISGeneralSetIndicesFromMask(is[0],rstart,rend,mask));
    PetscCall(PetscFree(mask));
  }
  {
    PetscInt *indices;

    PetscCall(PetscMalloc1(n,&indices));
    for (i=0; i<3; i++) {
      for (j=0; j<size; j++) indices[cnt++] = rstart+i*(size+2)+j;
    }
    PetscCheck(cnt == n,PETSC_COMM_SELF,PETSC_ERR_PLIB,"inconsistent count");
    PetscCall(ISGeneralSetIndices(is[1],n,indices,PETSC_COPY_VALUES));
    PetscCall(PetscFree(indices));
  }

  PetscCall(ISEqual(is[0],is[1],&flg));
  PetscCheck(flg,PETSC_COMM_SELF,PETSC_ERR_PLIB,"is[0] should be equal to is[1]");

  PetscCall(ISComplement(is[0],rstart,rend,&isc));
  PetscCall(ISView(is[0],PETSC_VIEWER_STDOUT_WORLD));
  PetscCall(ISView(isc,PETSC_VIEWER_STDOUT_WORLD));

  for (i=0; i<2; i++) PetscCall(ISDestroy(&is[i]));
  PetscCall(ISDestroy(&isc));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

   test:
      suffix: 3
      nsize: 3

TEST*/
