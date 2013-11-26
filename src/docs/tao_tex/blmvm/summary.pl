#!/usr/local/bin/perl -w
# 
$count=0;
foreach $file (@ARGV)
{
    open(FIN1, "$file");
    open(FOUT1, ">t$count");
    @all = <FIN1>;
    close(FIN1);

    @pprob =();
    @nnn   =();
    @ttime =();
    @ffgg  =();

    foreach $line (@all)
    {			       
	# 
	$nprob=0;
	if ($line =~ m/Problem\s*(\S*)/){
	    $prob=$1;
	    $ng=0; $tt=0; $nn=0;
	} elsif ($line =~ m/Number of variables\s*(\d+)/){
	    $nn=$1;
	} elsif ($line =~ m/Total execution time\s*([\d|\+|-|\.|d|D|e|E]+)/){
	    $tt=$1;
	} elsif ($line =~ m/Number of function evaluations\s*(\d+)/){
	    $ng=$1;
	    if ($ng >= 10000){ $ng = 10000; } 
	} elsif ($line =~ m/Number of gradient evaluations\s*(\d+)/){
	    $ng=$1;
	    if ($ng >= 10000){ $ng = 10000; } 
	} elsif ($line =~ m/Exit message\s/){
	    @pprob=(@pprob,$prob);
	    @nnn   =(@nnn,$nn);
	    @ttime =(@ttime,$tt);
	    @ffgg  =(@ffgg,$ng);
	    $nprob++;
#	    print "$prob & $nn  $ng  $tt\n";
	} elsif ($line =~ m/ERROR\s/){
	} elsif ($line =~ m/CONVERGENCE:\s/){
	} elsif ($line =~ m/ERROR:\s/){
	    print FOUT1 "10000\n";
	}

    }				# 
	
    close(FOUT1);
    if ($count < 1){
	@prob1 = @pprob;
	@n1  = @nnn;
	@nfg1 = @ffgg;
	@tt1 = @ttime;
    } elsif ($count < 2){
	@prob2 = @pprob;
	@n2  = @nnn;
	@nfg2 = @ffgg;
	@tt2 = @ttime;
    }
    $count++;

}				# 

$nprobs = @prob1;

for ($i=0;$i<$nprobs;$i++){
    print "$prob1[$i] &  $n1[$i] &  $nfg1[$i] & $tt1[$i] &     ";
    if ($count > 1){
#	print "   $prob2[$i] &  $n2[$i] &";
	print "   $nfg2[$i] &  $tt2[$i] ";
    }
    print " \\\\ \n";
}



