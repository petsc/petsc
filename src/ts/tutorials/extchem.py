#
#    Reads in the output trajectory data from a run of extchem and
#  formats it for graphing
#
#  Make sure $PETSC_DIR/bin is in your PYTHONPATH
#
import sys
import PetscBinaryIOTrajectory

if __name__ == '__main__':
    directory = sys.argv[1] if len(sys.argv) > 1 else 'Visualization-data'
    (t, v, names) = PetscBinaryIOTrajectory.ReadTrajectory(directory)
    #  PetscBinaryIOTrajectory.PlotTrajectories(t,v,names,['Temp','CO','CO2','H2O','H2','O2','CH4','C2H2','N2'])
    #
    #  Code is currently hardwired to display certain species only, edit the list below to display the species you want displayed
    #
    from petsc4py import PETSc

    for i in range(len(t) - 1):
        PETSc.Sys.Print(
            t[i],
            v[i][names.index('Temp')],
            v[i][names.index('CH4')],
            v[i][names.index('O2')],
            v[i][names.index('N2')],
            v[i][names.index('CO')],
            v[i][names.index('CO2')],
            v[i][names.index('O')],
            v[i][names.index('OH')],
            v[i][names.index('H2O')],
        )
