function bag = PetscBagRead(fd)
%
%  Reads in PETSc binary file bag object
%  emits as Matlab struct.  Called from
%  PetscBinaryRead.m.
%

[name_len help_len] = ParsePetscBagDotH;

bagsizecount = fread(fd,2,'int32');
count        = bagsizecount(2);

bag.bag_name      = deblank(char(fread(fd,name_len,'uchar')'));
bag.help.bag_help = deblank(char(fread(fd,help_len,'uchar')'));

for lcv = 1:count
  offsetdtype = fread(fd,2,'int32');
  dtype = offsetdtype(2);
  name  = strclean(deblank(char(fread(fd,name_len,'uchar')')));
  help  = deblank(char(fread(fd,help_len,'uchar')'));
  msize = fread(fd,1,'int32');

  if dtype == 0     % integer
    val = fread(fd,1,'int32');
  elseif dtype == 1 % double
    val = fread(fd,1,'double');
  elseif dtype == 6 % char
    val = deblank(char(fread(fd,msize,'uchar')'));
  elseif dtype == 9 % truth
    val = fread(fd,1,'int32');
% PETSC_LOGICAL is a bit boolean and not currently handled
%  elseif dtype == 7 % boolean
%    val = fread(fd,1,'bit1');
  elseif dtype == 8 % Enum
    val   = fread(fd,1,'int32');
    n     = fread(fd,1,'int32');
    sizes = fread(fd,n,'int32');
    enumnames = {'  '};
    for i=1:n-2,
      enumnames{i} = deblank(char(fread(fd,sizes(i),'uchar')));
    end
    val  = char(enumnames{val+1})';
    enumname   = deblank(char(fread(fd,sizes(n-1),'uchar')));
    enumprefix = deblank(char(fread(fd,sizes(n),'uchar')));
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
   
   petscbagh = [GetPetscDir,'/src/sys/bag/bagimpl.h'];
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
