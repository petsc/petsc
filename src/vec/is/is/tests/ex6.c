
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
  ierr = MPI_Comm_rank(PETSC_COMM_WORLD,&rank);CHKERRMPI(ierr);
  ierr = MPI_Comm_size(PETSC_COMM_WORLD,&size);CHKERRMPI(ierr);

  n      = 3*size;              /* Number of local indices, same on each process. */
  rstart = 3*(size+2)*rank;     /* start of local range */
  rend   = 3*(size+2)*(rank+1); /* end of local range */
  for (i=0; i<2; i++) {
    ierr = ISCreate(PETSC_COMM_WORLD,&is[i]);CHKERRQ(ierr);
    ierr = ISSetType(is[i],ISGENERAL);CHKERRQ(ierr);
  }
  {
    PetscBool *mask;

    ierr = PetscCalloc1(rend-rstart,&mask);CHKERRQ(ierr);
    for (i=0; i<3; i++) {
      for (j=0; j<size; j++) {
        mask[i*(size+2)+j] = PETSC_TRUE;
      }
    }
    ierr = ISGeneralSetIndicesFromMask(is[0],rstart,rend,mask);CHKERRQ(ierr);
    ierr = PetscFree(mask);CHKERRQ(ierr);
  }
  {
    PetscInt *indices;

    ierr = PetscMalloc1(n,&indices);CHKERRQ(ierr);
    for (i=0; i<3; i++) {
      for (j=0; j<size; j++) indices[cnt++] = rstart+i*(size+2)+j;
    }
    if (cnt != n) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_PLIB,"inconsistent count");
    ierr = ISGeneralSetIndices(is[1],n,indices,PETSC_COPY_VALUES);CHKERRQ(ierr);
    ierr = PetscFree(indices);CHKERRQ(ierr);
  }

  ierr = ISEqual(is[0],is[1],&flg);CHKERRQ(ierr);
  if (!flg) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_PLIB,"is[0] should be equal to is[1]");

  ierr = ISComplement(is[0],rstart,rend,&isc);CHKERRQ(ierr);
  ierr = ISView(is[0],PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  ierr = ISView(isc,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);

  for (i=0; i<2; i++) {ierr = ISDestroy(&is[i]);CHKERRQ(ierr);}
  ierr = ISDestroy(&isc);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
}

/*TEST

   test:
      suffix: 3
      nsize: 3

TEST*/
