function port = openport(number)
% $Id: openport.m,v 1.5 1999/11/24 21:52:40 bsmith Exp bsmith $
%
%  port = openport(number)
%  Opens a port to receive matrices from Petsc.
% see closeport and receive
disp('You must build the openport mex file by doing: cd ${PETSC_DIR}; make BOPT=g matlabcodes')