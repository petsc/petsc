function err = rhsfunction(dm,x,f)
%
%  Example of a function needed by PetscDMSetFunction
%  For linear problems x is ignored
%
err = 0;
f(:) = ones(length(f(:)));

