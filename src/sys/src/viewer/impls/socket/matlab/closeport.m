function closeport(port)
% $Id: closeport.m,v 1.5 1999/11/24 21:52:40 bsmith Exp bsmith $
%
%   closeport(port)
%   Closes a PETSc port opened with openport()
%see openport, receive
disp('You must build the closeport mex file by doing: cd ${PETSC_DIR}; make BOPT=g matlabcodes')