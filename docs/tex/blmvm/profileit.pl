#!/usr/local/bin/perl -w
# 
$count=0;
foreach $file (@ARGV)
{
    open(FIN1, "$file");
    open(FOUT1, ">t$count");
    @all = <FIN1>;
    close(FIN1);

    foreach $line (@all)
    {			       
	# 
	if ($line =~ m/Number of variables\s*(\d+.\d*)/){
	    $ng=0; $tt=0;
	} elsif ($line =~ m/Total execution time\s*(\d+.\d*)/){
	    $tt=$1;
	} elsif ($line =~ m/Number of function evaluations\s*(\d+)/){
	    $ng=$1;
	    if ($ng >= 10000){ $ng = 10000; } 
	} elsif ($line =~ m/Number of gradient evaluations\s*(\d+)/){
	    $ng=$1;
	    if ($ng >= 10000){ $ng = 10000; } 
	} elsif ($line =~ m/CONVERGENCE:\s/){
	    print FOUT1 "$ng\n";
	} elsif ($line =~ m/ERROR:\s/){
	    print FOUT1 "10000\n";
	} elsif ($line =~ m/ERROR\s/){
	    print FOUT1 "10000\n";
	}


    }				# 
	
    close(FOUT1);
    $count++;
}				# 
				# 
$inputs="";
for ($i=0;$i<$count;$i++)
{
    $inputs .= "t$i ";
}
system("perl perf $inputs -error 10000  >graph.m");
