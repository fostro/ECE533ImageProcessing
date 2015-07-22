#!/usr/bin/perl

$startVal = 3;
#$endVal = 21;
$endVal = 3;
$maxCount = 3;

for ($val = $startVal; $val <= $endVal; $val += 2) {
	for ($count = 1; $count <= $maxCount; ++$count) {
		$cmd = "../wand -i ../images/hs-2006-10-a-hires_jpg.jpg -s$val -d 1 -o output$val"."_$count".".png >> outputs.txt";
		print "$cmd\n";
		system($cmd);
	}
}
