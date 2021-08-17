#!/usr/bin/perl
#use threads;
#use threads::shared;
#use re::engine::RE2 -max_mem => 8<<23; #64MiB
use utf8;
use MeCab;
binmode(STDOUT, ":utf8");
binmode(STDIN, ":utf8");

while (<>) {
utf8::decode($_);
my $input = $_;
$input =~ tr/\!/！/;
$input =~ tr/\?/？/;
chomp $input;
my $input_mecab = $input;
push(@final_output, $input);

#mecab trick.

my @input_lemmas;
my @found_interjections;
my @input_lemma_no_emo;
my @input_tokens;
my @input_pos;
my @input_tokenpos;
my @input_lemmapos;


my $mecab = MeCab::Tagger->new();#"-d/usr/lib/mecab/dic/ipadic");
my $node = $mecab->parseToNode($input_mecab);
for( ; $node; $node = $node->{next} ) {
	next unless defined $node->{surface};
	my $midasi = $node->{surface};
	my( $hinsi, $kijutsu, $genkei ) = (split( /,/, $node->{feature} ))[0,1,6];
	push (@input_tokens, $midasi);
	if ($genkei eq '*'){
		$genkei = $midasi;
		# push (@input_lemmas, $genkei);
	} 
	# else {
		# push (@input_lemmas, $genkei);
	# }
	push (@input_lemmas, $genkei);
	push (@input_pos, $hinsi);
	push (@input_tokenpos, $midasi.'|'.$hinsi);
	push (@input_lemmapos, $genkei.'|'.$hinsi);
}

shift @input_tokens; pop @input_tokens;
shift @input_lemmas; pop @input_lemmas;
shift @input_pos; pop @input_pos;
shift @input_tokenpos; pop @input_tokenpos;
shift @input_lemmapos; pop @input_lemmapos;

my $input_tokens = join (' ', @input_tokens);
my $input_lemmas = join (' ', @input_lemmas);
my $input_pos = join (' ', @input_pos);
my $input_tokenpos = join (' ', @input_tokenpos);
my $input_lemmapos = join (' ', @input_lemmapos);

open(FILE_1, "+>>01_tokenized.txt") or die "Cannot open $file: $!";
print FILE_1 "$input_tokens"."\n";
close FILE_1;

open(FILE_2, "+>>02_lemmatized.txt") or die "Cannot open $file: $!";
print FILE_2 "$input_lemmas"."\n";
close FILE_2;

open(FILE_3, "+>>03_pos.txt") or die "Cannot open $file: $!";
print FILE_3 "$input_pos"."\n";
close FILE_3;

open(FILE_4, "+>>04_tokenPOS.txt") or die "Cannot open $file: $!";
print FILE_4 "$input_tokenpos"."\n";
close FILE_4;

open(FILE_5, "+>>05_lemmaPOS.txt") or die "Cannot open $file: $!";
print FILE_5 "$input_lemmapos"."\n";
close FILE_5;
}

__END__
