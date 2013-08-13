#!/usr/bin/perl

@inputFiles = ("mcsfp", "mcdfp", "scsfp", "scdfp");

open($output, ">", "compiledData.csv") or die "cannot open < compiledData.csv: $!";

print $output "computer, size, run, open, setup, gpu, save, total\n";

foreach $comp (@inputFiles) {

	$file = do {
		local $/ = undef;
		$input = $comp.".txt";
		#open($fh, "<", "myComputer_float_output.txt") or die "cannot open < float_output.txt: $!";
		open($fh, "<", $input) or die "cannot open < $input: $!";
		<$fh>;
	};
	
	@matches = ($file =~ /Creating Image.*?Open MonkTimer Report:.*?Total/gs);
	
	foreach $match (@matches) {
		$match =~ /Saving Image to output(\d+)_(\d).png/;
		$size = $1;
		$run = $2;
	
		$match =~ /Save MonkTimer Report.*?Took.*?Took (\d+\.\d+).* Total/s;
		$saveTime = $1;
	
		$match =~ /Total MonkTimer Report.*?Took.*?Took (\d+\.\d+).* Total/s;
		$totalTime = $1;
	
		$match =~ /GPU MonkTimer Report.*?Took.*?Took (\d+\.\d+).* Total/s;
		$gpuTime = $1;
	
		$match =~ /Setup MonkTimer Report.*?Took.*?Took (\d+\.\d+).* Total/s;
		$setupTime = $1;
		
		$match =~ /Open MonkTimer Report.*?Took.*?Took (\d+\.\d+).* Total/s;
		$openTime = $1;
	
		print $output "$comp, $size, $run, $openTime, $setupTime, $gpuTime, $saveTime, $totalTime\n";
	}

}

close($output);
