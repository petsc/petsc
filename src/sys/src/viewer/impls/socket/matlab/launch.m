function error = launch(program,np)
% $Id: launch.m,v 1.3 1999/11/24 21:52:40 bsmith Exp bsmith $
%
%  error = launch(program,np)
%  Starts up PETSc program
% see openprot, closeport and receive
disp('You must build the launch mex file by doing: cd ${PETSC_DIR}; make BOPT=g matlabcodes')