function closeport(port)
% $Id: closeport.m,v 1.6 2000/02/02 20:07:58 bsmith Exp bsmith $
%
%   closeport(port)
%   Closes a PETSc port opened with openport()
%see openport, receive
disp('You must build the closeport mex file by doing: cd ${PETSC_DIR}/src/sys/src/viewer/impls/socket/matlab; make BOPT=g matlabcodes')
