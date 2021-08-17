#!/usr/bin/perl -s
# usage: perl crossval.pl 10 (* or a different number depending what k for k-fold cross validation you want);
# usage: perl crossval.pl (not specifying number of folds will initialize default (10-fold));
# usage: perl crossval.pl 1 (1-fold means that training set will be the same as test set);
# usage: perl crossval.pl -loo ("-loo" parameter will initialize "leave one out" option; BTW, 
#	  "perl crossval.pl 4(any number) -loo" will also force "leave one out". When using this option make 
#	  sure both plus and minus sets are of the same size);
use List::Util 'shuffle';
use List::MoreUtils qw(uniq);

my $fold=$ARGV[0];
# print $fold;
if ($fold == 0){$fold=10;}

my @sentences;
open(FILE_1, "台風2次統合.txt") or die "Cannot open $file: $!";
@sentences = <FILE_1>;
foreach (@ainu) {
	utf8::decode($_);
	# chomp $_;
}
close(FILE_1);
chomp @sentences;


@sentences = shuffle(@sentences);
@original_sentences=@sentences;
# chomp @original_sentences;
if ($loo==1){$fold = @sentences;}
my $times = @sentences/$fold; 
$times = sprintf("%0.1f", $times);
if ($times =~ /(.+)\.(.+)/){
	$times_initial = $1;
	$times_rest = $2;
} else {$times_initial = $times;}

foreach $fold_initial (1 .. $times_initial) {
	foreach my $fold_no (1 .. $fold) {
		my $sentence = pop @sentences;
		push @$fold_no, $sentence;
	}
}

if ($times_rest > 0) {
	foreach my $fold_no (1 .. $times_rest) {
		my $sentence = pop @sentences;
		push @$fold_no, $sentence;
	}
}

foreach $set (1 .. $fold) {
	my @original_sentences_temp=@original_sentences;
	my @training_set;
	my @test_set = @$set;
	foreach $one_test_sentence (@test_set) {
		my $index = 0;
		$index++ until $original_sentences_temp[$index] eq $one_test_sentence;
		splice(@original_sentences_temp, $index, 1);
	}
	
	@training_set = @original_sentences_temp;
	# mkdir("crossval");
	# mkdir("crossval/$set");
	mkdir("$set");
	open(FILE_2, "+>>$set/plus_test.txt") or die "Cannot open $file: $!";
	foreach (@test_set){
		print FILE_2 "$_\n";
	}
	
	# open(FILE_3, "+>>$set/plus_training$set.txt") or die "Cannot open $file: $!";
	open(FILE_3, "+>>$set/input_file_plus.txt") or die "Cannot open $file: $!";
	if ($fold == 1) {
		foreach (@test_set){
			print FILE_3 "$_\n";
		}
	} else {
		foreach (@training_set){
			print FILE_3 "$_\n";
		}	
	}
	
	close FILE_2;
	close FILE_3;
}
	
