function bag = PetscBagRead(fd)
%
%  Reads in PETSc binary file bag object
%  emits as MATLAB struct.  Called from
%  PetscBinaryRead.m.
%

[name_len help_len] = ParsePetscBagDotH;

bagsize = read(fd,1,'int32');  %  no longer used after petsc-3.2 just here for backward compatibility of the binary files
count = read(fd,1,'int32');

bag.bag_name      = deblank(char(read(fd,name_len,'uchar')'));
bag.help.bag_help = deblank(char(read(fd,help_len,'uchar')'));

for lcv = 1:count
  offsetdtype = read(fd,2,'int32');
  dtype = offsetdtype(2);
  name  = strclean(deblank(char(read(fd,name_len,'uchar')')));
  help  = deblank(char(read(fd,help_len,'uchar')'));
  msize = read(fd,1,'int32');

  if dtype == 16     % integer
    val = read(fd,msize,'int32');
  elseif dtype == 1 % double
    val = read(fd,msize,'double');
  elseif dtype == 6 % char
    val = deblank(char(read(fd,msize,'uchar')'));
  elseif dtype == 9 % truth
    val = read(fd,1,'int32');
% PETSC_LOGICAL is a bit boolean and not currently handled
%  elseif dtype == 7 % boolean
%    val = read(fd,1,'bit1');
  elseif dtype == 8 % Enum
    val   = read(fd,1,'int32');
    n     = read(fd,1,'int32');
    sizes = read(fd,n,'int32');
    enumnames = {'  '};
    for i=1:n-2,
      enumnames{i} = deblank(char(read(fd,sizes(i),'uchar')));
    end
    val  = char(enumnames{val+1})';
    enumname   = deblank(char(read(fd,sizes(n-1),'uchar')));
    enumprefix = deblank(char(read(fd,sizes(n),'uchar')));
  else 
    val = [];
    warning('Bag entry %s could not be read',name);
  end 
  bag      = setfield(bag     ,name,val);
  bag.help = setfield(bag.help,name,help);
end
return

% ---------------------------------------------------- %
   
function [n, h] = ParsePetscBagDotH
   
   petscbagh = [GetPetscDir,'/include/petsc/private/bagimpl.h'];
   fid = fopen(petscbagh,'rt');
   if (fid<0)
      errstr = sprintf('Could not open %s.',petscbagh);
      error(errstr);
   end
   
   nametag = '#define PETSC_BAG_NAME_LENGTH';
   helptag = '#define PETSC_BAG_HELP_LENGTH';
   n = 0; h = 0;
   while ~feof(fid)
      lin = fgetl(fid);
      ni = strfind(lin,nametag);
      nh = strfind(lin,helptag);
      if ni
	 n = str2num(lin(ni+length(nametag):end));
      elseif nh
	 h = str2num(lin(nh+length(helptag):end));
      end   
      if (n>0 & h>0) break; end;
   end
   if (n==0 | h==0)
      errstr = sprintf('Could not parse %s.',petscbagh);
      error(errstr);
   end
   fclose(fid);
   return
   
% ---------------------------------------------------- %
   
function str = strclean(str)
   
   badchars = ' ()[]<>{}.-';
   for i=1:length(badchars);
      str(strfind(str,badchars(i))) = '_';
   end
   return
   
% ---------------------------------------------------- %
   
function dir = GetPetscDir
   
   dir = getenv('PETSC_DIR');
   if length(dir)==0
      error(['Please set environment variable PETSC_DIR' ...
	     ' and try again.'])
   end
   return
