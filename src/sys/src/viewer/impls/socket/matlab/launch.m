function error = launch(program,np)
% $Id: launch.m,v 1.5 2001/02/09 19:29:55 bsmith Exp $
%
%  error = launch(program,np)
%  Starts up PETSc program
% see openprot, closeport and receive
disp('You must build the launch mex file by doing: cd ${PETSC_DIR}/src/sys/src/viewer/impls/socket/matlab; make BOPT=g matlabcodes')
