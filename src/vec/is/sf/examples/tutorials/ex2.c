static const char help[] = "Test overlapped communication on a single star forest (PetscSF)\n\n";

#include <petscvec.h>
#include <petscsf.h>
#include <petscviewer.h>

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc, char **argv)
{
    PetscInt ierr;
    PetscSF sf;
    Vec A;
    Vec B;
    double *bufA;
    double *bufB;
    MPI_Comm c;
    PetscMPIInt rank, size;
    PetscInt nroots, nleaves;
    PetscInt i;
    PetscInt *ilocal;
    PetscSFNode *iremote;
    PetscInitialize(&argc,&argv,NULL,help);

    c = PETSC_COMM_WORLD;

    ierr = MPI_Comm_rank(c,&rank);CHKERRQ(ierr);
    ierr = MPI_Comm_size(c,&size);CHKERRQ(ierr);

    if (size != 2) {
        SETERRQ(c, PETSC_ERR_USER, "Only coded for two MPI processes\n");
    }

    ierr = PetscSFCreate(c,&sf);CHKERRQ(ierr);
    ierr = PetscSFSetFromOptions(sf);CHKERRQ(ierr);

    nleaves = 2;
    nroots = 1;

    ierr = PetscMalloc1(nleaves,&ilocal);CHKERRQ(ierr);

    for (i = 0; i<nleaves; i++) {
        ilocal[i] = i;
    }

    ierr = PetscMalloc1(nleaves,&iremote);CHKERRQ(ierr);
    if ( rank == 0 ) {
        iremote[0].rank = 0;
        iremote[0].index = 0;
        iremote[1].rank = 1;
        iremote[1].index = 0;
    } else {
        iremote[0].rank = 1;
        iremote[0].index = 0;
        iremote[1].rank = 0;
        iremote[1].index = 0;
    }
    ierr = PetscSFSetGraph(sf,nroots,nleaves,ilocal,PETSC_OWN_POINTER,
                           iremote,PETSC_OWN_POINTER);CHKERRQ(ierr);
    ierr = PetscSFSetUp(sf);CHKERRQ(ierr);
    ierr = PetscSFView(sf,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
    ierr = VecCreate(PETSC_COMM_WORLD,&A);CHKERRQ(ierr);
    ierr = VecSetSizes(A,2,PETSC_DETERMINE);CHKERRQ(ierr);
    ierr = VecSetFromOptions(A);CHKERRQ(ierr);
    ierr = VecSetUp(A);CHKERRQ(ierr);

    ierr = VecDuplicate(A,&B);CHKERRQ(ierr);

    ierr = VecGetArray(A,&bufA);CHKERRQ(ierr);
    ierr = VecGetArray(B,&bufB);CHKERRQ(ierr);
    for (i=0; i<2; i++) {
        bufA[i] = (double)rank;
        bufB[i] = (double)(rank) + 10.0;
    }
    ierr = PetscSFBcastBegin(sf,MPI_DOUBLE,(const void*)bufA,(void *)bufA);CHKERRQ(ierr);
    ierr = PetscSFBcastBegin(sf,MPI_DOUBLE,(const void*)bufB,(void *)bufB);CHKERRQ(ierr);
    ierr = PetscSFBcastEnd(sf,MPI_DOUBLE,(const void*)bufA,(void *)bufA);CHKERRQ(ierr);
    ierr = PetscSFBcastEnd(sf,MPI_DOUBLE,(const void*)bufB,(void *)bufB);CHKERRQ(ierr);

    ierr = VecRestoreArray(A,&bufA);CHKERRQ(ierr);
    ierr = VecRestoreArray(B,&bufB);CHKERRQ(ierr);

    ierr = VecView(A,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
    ierr = VecView(B,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
    ierr = VecDestroy(&A);CHKERRQ(ierr);
    ierr = VecDestroy(&B);CHKERRQ(ierr);
    ierr = PetscSFDestroy(&sf);CHKERRQ(ierr);

    PetscFinalize();

    return 0;
}
