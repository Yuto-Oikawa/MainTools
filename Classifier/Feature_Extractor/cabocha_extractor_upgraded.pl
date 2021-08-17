#!/usr/bin/perl
#use threads;
#use threads::shared;
#use re::engine::RE2 -max_mem => 8<<23; #64MiB
use utf8;
# use MeCab;
binmode(STDOUT, ":utf8");
binmode(STDIN, ":utf8");
# use Encode;

my @input_chunks;
my @input_dependency;
my @input_chunksNER;
my @input_depNER;

my @input_tokenNER;
my @input_lemmaNER;

while (<>) {
utf8::decode($_);
my $input = $_;
$input =~ tr/\!/！/;
$input =~ tr/\?/？/;
$input =~ tr/\#/＃/;
$input =~ tr/\./．/;

chomp $input;
# my $input_mceab = $input;
# push(@final_output, $input);

my $input_chunks = $input;
my $input_dependency = $input;
my $input_chunksNER = $input;
my $input_depNER = $input;

my $input_tokenNER = $input;
my $input_lemmaNER = $input;

# only chunks
$input_chunks =~ s/^\* 0.+$//;
$input_chunks =~ s/^\*.+$/ /;
$input_chunks =~ s/^(.+?)(\s|\t).+$/$1/;
$input_chunks =~ s/^EOS$/\n/;
# $input_chunks = decode_utf8( $input_chunks );
push (@input_chunks, $input_chunks);

# dependency
$input_dependency =~ s/^\* 0 (\-?[0-9]{1,}D).+$/$1｜/;
$input_dependency =~ s/^\* [0-9]{1,} (\-?[0-9]{1,}D).+$/ $1｜/;
$input_dependency =~ s/^(.+?)(\s|\t).+$/$1/;
$input_dependency =~ s/^EOS$/\n/;
# $input_dependency = decode_utf8( $input_dependency );
push (@input_dependency, $input_dependency);

# chunks with named entities
$input_chunksNER =~ s/^\* 0.+$//;
$input_chunksNER =~ s/^\*.+$/ /;
$input_chunksNER =~ s/^(.+?)(\s|\t).+(\s|\t)B\-(.+)$/$4/;
$input_chunksNER =~ s/^(.+?)(\s|\t).+(\s|\t)I\-(.+)$//;
$input_chunksNER =~ s/^(.+?)(\s|\t).+(\s|\t)O$/$1/;
$input_chunksNER =~ s/^EOS$/\n/;
# $input_chunksNER = decode_utf8( $input_chunksNER );
push (@input_chunksNER, $input_chunksNER);

# dependency with named entities
$input_depNER =~ s/^\* 0 (\-?[0-9]{1,}D).+$/$1｜/;
$input_depNER =~ s/^\* [0-9]{1,} (\-?[0-9]{1,}D).+$/ $1｜/;
$input_depNER =~ s/^(.+?)(\s|\t).+(\s|\t)B\-(.+)$/$4/;
$input_depNER =~ s/^(.+?)(\s|\t).+(\s|\t)I\-(.+)$//;
$input_depNER =~ s/^(.+?)(\s|\t).+(\s|\t)O$/$1/;
$input_depNER =~ s/^EOS$/\n/;
# $input_depNER = decode_utf8( $input_depNER );
push (@input_depNER, $input_depNER);

# only tokens with NER
$input_tokenNER=~ s/^.+\t.+\t.+(ORGANIZATION|PERSON|LOCATION|ARTIFACT|DATE|TIME|MONEY|PERCENT)$/$1 /g;
$input_tokenNER =~ s/^\* [0-9].+$//g;
$input_tokenNER =~ s/^(.+?)\t.+\t.+$/$1 /g;
$input_tokenNER =~ s/\s\n//;
$input_tokenNER =~ s/^EOS$/\n/;
push (@input_tokenNER, $input_tokenNER);

# $input_lemmaNER = $input;
if ($input_lemmaNER =~ /EOS/g) {
	push (@input_lemmaNER, "\n");
}
$input_lemmaNER =~ s/	/,/;
@allelements = split(",", $input_lemmaNER);
my $ner = pop @allelements;
my $token = shift @allelements;
pop @allelements;
# pop @allelements;
my $lemma = pop @allelements;
$ner =~ s/.+(ORGANIZATION|PERSON|LOCATION|ARTIFACT|DATE|TIME|MONEY|PERCENT)/$1/g;
if ($ner =~ m/(ORGANIZATION|PERSON|LOCATION|ARTIFACT|DATE|TIME|MONEY|PERCENT)/g){
	push (@input_lemmaNER, $ner.' ');
} elsif ($lemma !~ m/\*/) {
	push (@input_lemmaNER, $lemma.' ');
} else {
	push (@input_lemmaNER, $token.' ');
}

}

foreach my $index (0 .. @input_tokenNER){
	if ($input_tokenNER[$index]=~m/(ORGANIZATION|PERSON|LOCATION|ARTIFACT|DATE|TIME|MONEY|PERCENT)/g){
		my $indy = $index+1;
		while ($input_tokenNER[$index] eq $input_tokenNER[$indy]){
			# print "gotcha!\n";
			splice @input_tokenNER, $index, 1;
		}
	}
	# print $input_tokenNER[$index];
	# $input_tokenNER[$index]　=~　s/  / /;
	# $input_tokenNER[$index]　=~　s/^ | $//;	
}

foreach my $index (0 .. @input_lemmaNER){
	if ($input_lemmaNER[$index]=~m/(ORGANIZATION|PERSON|LOCATION|ARTIFACT|DATE|TIME|MONEY|PERCENT)/g){
		my $indy = $index+1;
		while ($input_lemmaNER[$index] eq $input_lemmaNER[$indy]){
			# print "gotcha!\n";
			splice @input_lemmaNER, $index, 1;
		}
	}
	# $input_lemmaNER[$index]　=~　s/  / /;
	# $input_lemmaNER[$index]　=~　s/^ | $//;
}

# foreach (@input_tokenNER) {
	# $_ = $_;
	# }



open(FILE_1, "+>>:utf8", "06_chunks.txt") or die "Cannot open $file: $!";
print FILE_1 @input_chunks;
close FILE_1;

open(FILE_2, "+>>:utf8", "07_dependency.txt") or die "Cannot open $file: $!";
print FILE_2 @input_dependency;
close FILE_2;

open(FILE_3, "+>>:utf8", "08_chunksNER.txt") or die "Cannot open $file: $!";
print FILE_3 @input_chunksNER;
close FILE_3;

open(FILE_4, "+>>:utf8", "09_depNER.txt") or die "Cannot open $file: $!";
print FILE_4 @input_depNER;
close FILE_4;

open(FILE_5, "+>>:utf8", "10_tokenNER.txt") or die "Cannot open $file: $!";
print FILE_5 @input_tokenNER;
close FILE_5;

open(FILE_6, "+>>:utf8", "11_lemmaNER.txt") or die "Cannot open $file: $!";
print FILE_6 @input_lemmaNER;
close FILE_6;

__END__
