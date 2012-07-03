function ex41(np,opt)
%
%  ex41(np,opt) - receives a matrix and vector from Matlab via socket
%  solves the linear system and returns the solution vector
%
% Run with option -on_error_attach_debugger to debug
%
%  Requires the Matlab mex routines in ${PETSC_DIR}/bin/matlab.
%  Make sure that ${PETSC_DIR}/bin/matlab is in your Matlab PATH.
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
