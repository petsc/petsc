# $Id: base.S,v 1.3 1997/10/19 03:20:08 bsmith Exp $ 

#
PETSCFLAGS = -DUSE_PETSC_DEBUG -DUSE_PETSC_LOG  -DUSE_PETSC_BOPT_O
COPTFLAGS  = -O3  -S 
# For Fortran, there are lots of options
# -P is a preprocessor that seems to unroll loops.
# -Wp,-ew    invokes the most common optimizations = -ea2478
#            -e2 is no data dependencies in loops containing pointer-based
#            variables (noalias)
# The LINPACK tables used:
# -Wp,-ea478 -ea is associative transformations
#            -e4 is generate calls to BLAS library routines
#            -e7 is Automatically expand called routines inline
#            -e8 is search input file for expandable routines
#            -e9 is collapse array subscripts when inlining
#            -me is data alignment for COMMON blocks and arrays
#
# To see what the preprocessor does, use
# -Wp,-ew,-ofoo.f,-lfoo.lst 
# this generates the preprocessed Fortran in foo.f and a listing file.
# Also, there is another preprocessor, -Pk, which has a different syntax.
# You might want to try
# -r=1 (like -ea)
# -r=2 (-r=1 + reciprocal substitution)
# -o=5 (default, aggressive loop optimizations)
# -s=3 (default, aggressive scalar optimizations)
# -inl (inline)
# -ag=a (like -me)
# -lc=blas (like -e4)
# -f=file.f (like -ofoo.f)
# These should be used with the -Wp option:
#    -Wp,-r=2,-o=5,-s=3,-inl
# In OLDER systems, change -O3 to -O
#FOPTFLAGS = -O3 -P -Wp,-ew
#FOPTFLAGS = -O -P -Wp,-ew
# Further investigation has shown conclusively that the preprocessor WILL
# generate INCORRECT code for relatively simple doubly-nested loops, even
# when the only operations are ASSIGNMENTS!  See the ANL/IBM-SP1 User's
# guide for an example.
FOPTFLAGS = -O  -S
# For ANL, the following are also useful (cache is for RS6000/370)
#-qarch=pwr \
#-qhot  \
#-qcache:type=d:level=1:size=32:assoc=4:line=64 \
#-qcache:type=i:level=1:size=32:assoc=2:line=64 \
#-qcache:type=d:level=2:size=512:assoc=2:line=4096 \
#-qcache:type=i:level=2:size=512:assoc=2:line=4096
