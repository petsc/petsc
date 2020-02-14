#   Language is completely determined by the file prefix, .c .cxx .F
#     .F90 so this need not be in the requirements list
#
#   MPIUNI should work for all -n 1 examples so this need not be in the
#   requirements list
# 
#   DATAFILES are listed in the example arguments (e.g. -f
#   ${DATAFILES}/) so this is need not be in the requirement list
#
#   For packages, scalartypes and precisions:
#     ! => not
#     , => and
#
#   Precision types: single, double, quad, int32
#   Scalar types: complex  (and !complex)
#
#   Some examples:
#      requires:   x, superlu_dist, !single  
#      requires: !complex !single
#      requires: int32
#
#   buildrequires => file requires things just to build.  Usually
#      because of includes
#
#   There is some limited support for mapping args onto packages


makefileMap={}

# This looks for the pattern matching and then determines the
# requirement.  The distinction between buildrequires and requires is
# tricky.  I looked at the makefile's and files themselves to try and
# figure it out.
makefileMap["COMPLEX"]="buildrequires: complex"
makefileMap["NOCOMPLEX"]="buildrequires: !complex"
makefileMap["NOTSINGLE"]="buildrequires: !single"
makefileMap["NOSINGLE"]="buildrequires: !single"

makefileMap["DOUBLEINT32"]="buildrequires: !define(USE_64BIT_INDICES) define(PETSC_USE_REAL_DOUBLE)"  
makefileMap["THREADSAFETY"]="buildrequires: define(PETSC_USING_FREEFORM)"
makefileMap["F2003"]="buildrequires: define(PETSC_USING_FREEFORM) define(PETSC_USING_F2003)"
#makefileMap["F90_DATATYPES"]="" # ??

makefileMap["DATAFILESPATH"]="requires: datafilespath"
makefileMap['INFO']="requires: define(USE_INFO)"

# Typo
makefileMap["PARAMETIS"]="requires: parmetis"

# Some packages are runtime, but others are buildtime because of includes
reqpkgs=["HDF5", "HYPRE", "LUSOL","MKL_PARDISO", "ML", "MUMPS", "PARMETIS", "PARMS", "PASTIX", "PTSCOTCH", "REVOLVE", "SAWS", "SPAI", "STRUMPACK", "SUITESPARSE", "SUPERLU", "SUPERLU_DIST"]

bldpkgs=["CTETGEN", "EXODUSII", "CHOMBO","ELEMENTAL", "MATLAB", "MATLAB_ENGINE",  "MOAB", "FFTW", "TCHEM","VECCUDA","CUSP","CUSPARSE","TRILINOS", "X", "TRIANGLE", "YAML"]

for pkg in reqpkgs: makefileMap[pkg]="requires: "+ pkg.lower()
for pkg in bldpkgs: makefileMap[pkg]="buildrequires: "+ pkg.lower()

#  Map of "string" in arguments to package requirements; i.e.,
#    argMap[patternString]=packageRequired
#  I dislike this because it is such a crude hack.
#  But TESTEXAMPLES doesn't have all of the information for crufty tests.
#
argMap={}
# Things that are too short to do simple pattern search
reqpkgs.remove('ML') 
bldpkgs.remove('X') 
for pkg in reqpkgs+bldpkgs:
  argMap[pkg]="requires: "+pkg.lower()
argMap['DATAFILESPATH']='requires: datafilespath'
