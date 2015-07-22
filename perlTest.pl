#!/usr/bin/perl

$startVal = 3;
$endVal = 21;
$maxCount = 3;

for ($val = $startVal; $val <= $endVal; $val += 2) {
	for ($count = 1; $count <= $maxCount; ++$count) {
		$cmd = "./wand -i images/big_coral.jpg -s$val -o output$val"."_$count".".png >> outputs.txt";
		print "$cmd\n";
		system($cmd);
	}
}
