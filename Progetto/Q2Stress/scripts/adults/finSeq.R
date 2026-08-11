# THE FUNCTION RETURNS ALL WORDS ENDING WITH THE SPECIFIED ORTHOGRAPHIC SEQUENCE AND STRESS PATTERN
# THE RESEARCH CAN BE REFINED BY SPEFICING NUMBER OF SYLLABLES, GRAMMATICAL CATEGORY.
# Parameter: a = word ending to look for (note that the sequence must be specified within quote; e.g., "ola")
# b = stress pattern to look for (form 0 to 3, with 0: final stress; 1: penultimate; 2: antepenultimate; 3: pre-antepenultimate);
# syll = number of syllables (from 2 to a maximum of 11); gCat = specifies the grammatical category of words we  look for 
# (note that gCat must be specified within quote; e.g., "S");
# example of application with all parameters: finSeq("olo",1, syll=3, gCat="S")
# running the example we look for all the words ending with olo, having penultimate stress, being of 3 syllable and being nouns,

finSeq = function(a, b, syll=NULL, gCat=NULL) {
	
	load("dbfSeq.RData")
	
	if (is.null(gCat)) {
		if (is.null(syll)) {
			if (a %in% poli$FinSeq) {	
				words=subset(poli, FinSeq==a & StressPattern==b)			
				if (!(nrow(words)==0)) {
					fname=paste('words_',a,"_",b,".txt",sep="")
					write.table(words, fname, sep="\t", row.names=F)
					} 
					else print(paste0("There are no words ending in ",a," with this stress pattern"))		
				} else print("This sequence is not in the list") 
		} 
		else {
			if (!(syll %in% poli$SumSylls))  {
					print("There are no words with this number of syllables")			
					}
				else {
				poli= poli[poli$SumSylls==syll,]
				if (a %in% poli$FinSeq) {	
					words=subset(poli, FinSeq==a & StressPattern==b)			
					if (!(nrow(words)==0)) {
						fname=paste('words_',a,"_",b,".txt",sep="")
						write.table(words, fname, sep="\t", row.names=F)
						} 
						else print(paste0("There are no words ending in ",a," with these properties"))		
					} else print("This sequence is not in the list") 
			}
		}
	}
	else {
		if (!(gCat %in% poli$gramCat))  {
				print("This grammatical category is not in the list")			
			}
		else {
			poli= poli[poli$gramCat==gCat,]
			if (is.null(syll)) {
			if (a %in% poli$FinSeq) {	
				words=subset(poli, FinSeq==a & StressPattern==b)			
				if (!(nrow(words)==0)) {
					fname=paste('words_',a,"_",b,".txt",sep="")
					write.table(words, fname, sep="\t", row.names=F)
					} 
					else print(paste0("There are no words ending in ",a," with these properties"))		
				} else print("This sequence is not in the list") 
		} 
		else {
			if (!(syll %in% poli$SumSylls))  {
					print("There are no words with this number of syllables")			
					}
				else {
				poli= poli[poli$SumSylls==syll,]
				if (a %in% poli$FinSeq) {	
					words=subset(poli, FinSeq==a & StressPattern==b)			
					if (!(nrow(words)==0)) {
						fname=paste('words_',a,"_",b,".txt",sep="")
						write.table(words, fname, sep="\t", row.names=F)
						} 
						else print(paste0("There are no words ending in ",a," with these properties"))		
					} else print("This sequence is not in the list") 
				}
			}
		}
	}
} 