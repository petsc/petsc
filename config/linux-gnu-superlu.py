#!/usr/bin/env python

if __name__ == '__main__':
    import configure

    configure_options = [
        '--with-mpi-include=/home/petsc/soft/linux-rh73/mpich-1.2.4/include',
        '--with-mpi-lib=[/home/petsc/soft/linux-rh73/mpich-1.2.4/lib/libmpich.a,/home/petsc/soft/linux-rh73/mpich-1.2.4/lib/libpmpich.a]',
        '--with-mpirun=mpirun -all-local',
        #'--with-superlu_dist-dir=/home/petsc/soft/linux-rh73/SuperLU_DIST_2.0',
        '--with-superlu_dist-include=/home/petsc/soft/linux-rh73/SuperLU_DIST_2.0/SRC',
        #'--with-superlu_dist-lib=/home/petsc/soft/linux-rh73/SuperLU_DIST_2.0/superlu_linux.a',
        '--with-cc=gcc'
        ]

    configure.petsc_configure(configure_options)
