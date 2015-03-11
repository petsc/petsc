function ex41(np,opt)
%
%  ex41(np,opt) - receives a matrix and vector from Matlab via socket
%  solves the linear system and returns the solution vector
%
% Run with option -on_error_attach_debugger to debug
%
%  Requires the Matlab mex routines in ${PETSC_DIR}/share/petsc/matlab and ${PETSC_DIR}/${PETSC_ARCH}/lib/matlab.
%  Make sure that ${PETSC_DIR}/share/petsc/matlab and ${PETSC_DIR}/${PETSC_ARCH}/lib/matlab is in your MATLABPATH or
%  $prefix/share/petsc/matlab and $prefix/lib/matlab if you ran ./configure with --prefix
%
if (nargin < 1)
  np = 1;
end
if (nargin < 2) 
  opt = ' ';
end
launch('./ex41  ',np,opt);

p = PetscOpenSocket;
b = [1 2 3];
A = sparse([3 2 1; 1 3 2; 1 2 3]);
PetscBinaryWrite(p,b);
PetscBinaryWrite(p,A);
x = PetscBinaryRead(p);
b' - A*x'
x' - A\b'
close(p);
