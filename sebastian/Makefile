get_rost_data:
	mkdir -p data && cd data ; \
	wget -O nonRed_dataset.gz https://rostlab.org/~deepppi/PP2/nonRed_dataset.gz ; \
	wget -O annotations.gz https://rostlab.org/~deepppi/PP2/annotations.gz ; \
	tar xf nonRed_dataset.gz ; \
	tar xf annotations.gz ;

get_swiss_prot:
	mkdir -p data && cd data ; \
	mmseqs databases UniProtKB/Swiss-Prot swissprot tmp