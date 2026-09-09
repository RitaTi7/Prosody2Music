# THE FUNCTION RETURNS ALL WORDS STARTING WITH THE SPECIFIED ORTHOGRAPHIC SEQUENCE AND STRESS PATTERN
# THE RESEARCH CAN BE REFINED BY SPEFICING NUMBER OF SYLLABLES, GRAMMATICAL CATEGORY, AND CV-STRUCTURE OF FINAL SEQUENCE.
# Parameter: a = word beginning to look for (note that the sequence must be specified within quote; e.g., "ba"); b = stress pattern 
# to look for (for 0 to 3, with 0 = final stress; 1 = penultimate; 2 = antepenultimate; 3 = preantepenultimate); syll = number of
# syllables (from 2 to a maximum of 11); gCat = specifies the grammatical category of words we  look for (note that gCat must
# be specified within quote; e.g., "S"); cvStrFin= specifies the structure of the word ending of words we want to look for (with V = vowel, 
# C = consonant, S = stressed vowel, A = apostrophe) (note that the cvStrFin must be specified within quote; e.g., "VCV").
# example of application with all parameters: CHILD_beg("ba",1, syll=3, gCat="S", cvStrFin="VCV")
# running the example we look for all the words starting with ba, having penultimate stress, being of 3 syllable, being nouns, 
# and having a final sequence with a VCV structure

CHILD_beg = function(a, b, syll=NULL, gCat=NULL,cvStrFin=NULL) {
	
	l= nchar(a)
	fname= paste("db",l,".RData",sep="");
	load(fname)
	colnames(poli) [17] ="OrtFinSeq"

if (is.null(gCat)) {
	if (is.null(cvStrFin)) {
		if (is.null(syll)) {
			if (l==1) {
				if (a %in% poli$firstLet) {	
					words=subset(poli, firstLet==a & StressPattern==b)
					if (!(nrow(words)==0)) {
						fname2=paste('words_',a,"_",b,".txt",sep="")
						write.table(words, fname2, sep="\t", row.names=F)
					} 
					else print(paste0("There are no words starting in ",a," with this stress pattern"))	
				}
				else print("This sequence is not in the list") 
			}	
			else if (l==2) {
				if (a %in% poli$twoLet) {	
					words=subset(poli, twoLet==a & StressPattern==b)
					if (!(nrow(words)==0)) {
						fname2=paste('words_',a,"_",b,".txt",sep="")
						write.table(words, fname2, sep="\t", row.names=F)
					} 
					else print(paste0("There are no words starting in ",a," with this stress pattern"))	
				}
				else print("This sequence is not in the list") 
			}
			else if (l==3) {
				if (a %in% poli$threeLet) {	
					
					words=subset(poli, threeLet==a & StressPattern==b)
					if (!(nrow(words)==0)) {
						fname2=paste('words_',a,"_",b,".txt",sep="")
						write.table(words, fname2, sep="\t", row.names=F)
					} 
					else print(paste0("There are no words starting in ",a," with this stress pattern"))	
				}
				else print("This sequence is not in the list") 
			}	
			else if (l==4) {
				if (a %in% poli$fourLet) {	
				words=subset(poli, fourLet==a & StressPattern==b)
				if (!(nrow(words)==0)) {
						fname2=paste('words_',a,"_",b,".txt",sep="")
						write.table(words, fname2, sep="\t", row.names=F)
					} 
					else print(paste0("There are no words starting in ",a," with this stress pattern"))	
				}
				else print("This sequence is not in the list") 
			}	
		}
		else {
			if (!(syll %in% poli$SumSylls))  {
				print("There are no words with this number of syllables")			
				}
			else {
			poli= poli[poli$SumSylls==syll,]
				if (l==1) {
					if (a %in% poli$firstLet) {
						words=subset(poli, firstLet==a & StressPattern==b)
						if (!(nrow(words)==0)) {
							fname2=paste('words_',a,"_",b,".txt",sep="")
							write.table(words, fname2, sep="\t", row.names=F)
						} 
						else print(paste0("There are no words starting in ",a," with these properties"))	
					}
					else print("This sequence is not in the list") 
				}	
				else if (l==2) {
					if (a %in% poli$twoLet) {		
						words=subset(poli, twoLet==a & StressPattern==b)
						if (!(nrow(words)==0)) {
							fname2=paste('words_',a,"_",b,".txt",sep="")
							write.table(words, fname2, sep="\t", row.names=F)
						} 
						else print(paste0("There are no words starting in ",a," with these properties"))	
					}
					else print("This sequence is not in the list") 
				}
				else if (l==3) {
					if (a %in% poli$threeLet) {
						words=subset(poli, threeLet==a & StressPattern==b)
						if (!(nrow(words)==0)) {	
							fname2=paste('words_',a,"_",b,".txt",sep="")
							write.table(words, fname2, sep="\t", row.names=F)
						} 
						else print(paste0("There are no words starting in ",a," with these properties"))	
					}
					else print("This sequence is not in the list") 
				}	
				else if (l==4) {
					if (a %in% poli$fourLet) {							
						words=subset(poli, fourLet==a & StressPattern==b)
						if (!(nrow(words)==0)) {	
							fname2=paste('words_',a,"_",b,".txt",sep="")
							write.table(words, fname2, sep="\t", row.names=F)
						} 
						else print(paste0("There are no words starting in ",a," with these properties"))	
					}
					else print("This sequence is not in the list") 
				}
			}	
		}
	}		
	
	else {
		if (!(cvStrFin %in% poli$OrtFinSeq))  {
			print("This orthographic structure is not in the list")			
		}
		else {
			poli= poli[poli$OrtFinSeq==cvStrFin,]
			if (is.null(syll)) {
				if (l==1) {
					if (a %in% poli$firstLet) {						
						words=subset(poli, firstLet==a & StressPattern==b)
						if (!(nrow(words)==0)) {	
							fname2=paste('words_',a,"_",b,".txt",sep="")
							write.table(words, fname2, sep="\t", row.names=F)
						} 
						else print(paste0("There are no words starting in ",a," with these properties"))					
					}
					else print("This sequence is not in the list") 
				}	
				else if (l==2) {
					if (a %in% poli$twoLet) {							
						words=subset(poli, twoLet==a & StressPattern==b)
						if (!(nrow(words)==0)) {	
							fname2=paste('words_',a,"_",b,".txt",sep="")
							write.table(words, fname2, sep="\t", row.names=F)
						} 
						else print(paste0("There are no words starting in ",a," with these properties"))					
					}
					else print("This sequence is not in the list") 
				}
				else if (l==3) {
					if (a %in% poli$threeLet) {	
						words=subset(poli, threeLet==a & StressPattern==b)
						if (!(nrow(words)==0)) {	
							fname2=paste('words_',a,"_",b,".txt",sep="")
							write.table(words, fname2, sep="\t", row.names=F)
						} 
						else print(paste0("There are no words starting in ",a," with these properties"))					
				}
				else print("This sequence is not in the list") 
			}	
			else if (l==4) {
				if (a %in% poli$fourLet) {		
					words=subset(poli, fourLet==a & StressPattern==b)
					if (!(nrow(words)==0)) {	
						fname2=paste('words_',a,"_",b,".txt",sep="")
						write.table(words, fname2, sep="\t", row.names=F)
					} 
					else print(paste0("There are no words starting in ",a," with these properties"))					
				}
				else print("This sequence is not in the list") 
			}	
		}
		else {
			if (!(syll %in% poli$SumSylls))  {
				print("There are no words with this number of syllables")			
				}
			else {
			poli= poli[poli$SumSylls==syll,]
				if (l==1) {
					if (a %in% poli$firstLet) {							
						words=subset(poli, firstLet==a & StressPattern==b)
						if (!(nrow(words)==0)) {
							fname2=paste('words_',a,"_",b,".txt",sep="")
							write.table(words, fname2, sep="\t", row.names=F)
						} 
						else print(paste0("There are no words starting in ",a," with these properties"))					
					}
					else print("This sequence is not in the list") 
				}	
				else if (l==2) {
					if (a %in% poli$twoLet) {	
						words=subset(poli, twoLet==a & StressPattern==b)
						if (!(nrow(words)==0)) {	
							fname2=paste('words_',a,"_",b,".txt",sep="")
							write.table(words, fname2, sep="\t", row.names=F)
						} 
						else print(paste0("There are no words starting in ",a," with these properties"))					
					}
					else print("This sequence is not in the list") 
				}
				else if (l==3) {
					if (a %in% poli$threeLet) {		
						words=subset(poli, threeLet==a & StressPattern==b)
						if (!(nrow(words)==0)) {	
							fname2=paste('words_',a,"_",b,".txt",sep="")
							write.table(words, fname2, sep="\t", row.names=F)
						} 
						else print(paste0("There are no words starting in ",a," with these properties"))					
					}
					else print("This sequence is not in the list") 
				}	
				else if (l==4) {
					if (a %in% poli$fourLet) {	
						words=subset(poli, fourLet==a & StressPattern==b)
						if (!(nrow(words)==0)) {	
							fname2=paste('words_',a,"_",b,".txt",sep="")
							write.table(words, fname2, sep="\t", row.names=F)
						} 
						else print(paste0("There are no words starting in ",a," with these properties"))					
					}
					else print("This sequence is not in the list") 
				}
			}	
		}
	}
}
}

	else {
		if (!(gCat %in% poli$NewGramCat))  {
				print("This grammatical category is not in the list")			
			}
			else {
				poli= poli[poli$NewGramCat==gCat,]
			if (is.null(cvStrFin)) {
			if (is.null(syll)) {
				if (l==1) {
					if (a %in% poli$firstLet) {	
						words=subset(poli, firstLet==a & StressPattern==b)
						if (!(nrow(words)==0)) {	
							fname2=paste('words_',a,"_",b,".txt",sep="")
							write.table(words, fname2, sep="\t", row.names=F)
						} 
						else print(paste0("There are no words starting in ",a," with these properties"))
					}
					else print("This sequence is not in the list") 
				}	
				else if (l==2) {
					if (a %in% poli$twoLet) {	
						words=subset(poli, twoLet==a & StressPattern==b)
						if (!(nrow(words)==0)) {	
							fname2=paste('words_',a,"_",b,".txt",sep="")
							write.table(words, fname2, sep="\t", row.names=F)
						} 
						else print(paste0("There are no words starting in ",a," with these properties"))
					}
					else print("This sequence is not in the list") 
				}
				else if (l==3) {
					if (a %in% poli$threeLet) {		
						words=subset(poli, threeLet==a & StressPattern==b)
						if (!(nrow(words)==0)) {
							fname2=paste('words_',a,"_",b,".txt",sep="")
							write.table(words, fname2, sep="\t", row.names=F)
						} 
						else print(paste0("There are no words starting in ",a," with these properties"))
					}
					else print("This sequence is not in the list") 
				}	
				else if (l==4) {
					if (a %in% poli$fourLet) {	
						words=subset(poli, fourLet==a & StressPattern==b)
						if (!(nrow(words)==0)) {	
							fname2=paste('words_',a,"_",b,".txt",sep="")
							write.table(words, fname2, sep="\t", row.names=F)
						} 
						else print(paste0("There are no words starting in ",a," with these properties"))
					}
					else print("This sequence is not in the list") 
				}	
			}
			else {
				if (!(syll %in% poli$SumSylls))  {
					print("There are no words with this number of syllables")			
					}
				else {
				poli= poli[poli$SumSylls==syll,]
					if (l==1) {
						if (a %in% poli$firstLet) {
							words=subset(poli, firstLet==a & StressPattern==b)
							if (!(nrow(words)==0)) {	
								fname2=paste('words_',a,"_",b,".txt",sep="")
								write.table(words, fname2, sep="\t", row.names=F)
							} 
							else print(paste0("There are no words starting in ",a," with these properties"))
						}
						else print("This sequence is not in the list") 
					}	
					else if (l==2) {
						if (a %in% poli$twoLet) {
							words=subset(poli, twoLet==a & StressPattern==b)
							if (!(nrow(words)==0)) {	
								fname2=paste('words_',a,"_",b,".txt",sep="")
								write.table(words, fname2, sep="\t", row.names=F)
							} 
							else print(paste0("There are no words starting in ",a," with these properties"))
						}
						else print("This sequence is not in the list") 
					}
					else if (l==3) {
						if (a %in% poli$threeLet) {	
							words=subset(poli, threeLet==a & StressPattern==b)
							if (!(nrow(words)==0)) {	
								fname2=paste('words_',a,"_",b,".txt",sep="")
								write.table(words, fname2, sep="\t", row.names=F)
							} 
							else print(paste0("There are no words starting in ",a," with these properties"))
						}
						else print("This sequence is not in the list") 
					}	
					else if (l==4) {
						if (a %in% poli$fourLet) {	
							words=subset(poli, fourLet==a & StressPattern==b)
							if (!(nrow(words)==0)) {	
								fname2=paste('words_',a,"_",b,".txt",sep="")
								write.table(words, fname2, sep="\t", row.names=F)
							} 
							else print(paste0("There are no words starting in ",a," with these properties"))
						}
						else print("This sequence is not in the list") 
					}
				}	
			}
		}		
	
		else {
			if (!(cvStrFin %in% poli$OrtFinSeq))  {
				print("This orthographic structure is not in the list")			
			}
			else {
				poli= poli[poli$OrtFinSeq==cvStrFin,]
				if (is.null(syll)) {
					if (l==1) {
						if (a %in% poli$firstLet) {	
							words=subset(poli, firstLet==a & StressPattern==b)
							if (!(nrow(words)==0)) {	
								fname2=paste('words_',a,"_",b,".txt",sep="")
								write.table(words, fname2, sep="\t", row.names=F)
							} 
							else print(paste0("There are no words starting in ",a," with these properties"))
						}
						else print("This sequence is not in the list") 
					}	
					else if (l==2) {
						if (a %in% poli$twoLet) {	
							words=subset(poli, twoLet==a & StressPattern==b)
							if (!(nrow(words)==0)) {	
								fname2=paste('words_',a,"_",b,".txt",sep="")
								write.table(words, fname2, sep="\t", row.names=F)
							} 
							else print(paste0("There are no words starting in ",a," with these properties"))
						}
						else print("This sequence is not in the list") 
					}
					else if (l==3) {
						if (a %in% poli$threeLet) {	
							words=subset(poli, threeLet==a & StressPattern==b)
							if (!(nrow(words)==0)) {	
								fname2=paste('words_',a,"_",b,".txt",sep="")
								write.table(words, fname2, sep="\t", row.names=F)
							} 
							else print(paste0("There are no words starting in ",a," with these properties"))
					}
					else print("This sequence is not in the list") 
				}	
				else if (l==4) {
					if (a %in% poli$fourLet) {	
						words=subset(poli, fourLet==a & StressPattern==b)
						if (!(nrow(words)==0)) {	
							fname2=paste('words_',a,"_",b,".txt",sep="")
							write.table(words, fname2, sep="\t", row.names=F)
						} 
						else print(paste0("There are no words starting in ",a," with these properties"))
					}
					else print("This sequence is not in the list") 
				}	
			}
			else {
				if (!(syll %in% poli$SumSylls))  {
					print("There are no words with this number of syllables")			
					}
				else {
				poli= poli[poli$SumSylls==syll,]
					if (l==1) {
						if (a %in% poli$firstLet) {		
							words=subset(poli, firstLet==a & StressPattern==b)
							if (!(nrow(words)==0)) {
								fname2=paste('words_',a,"_",b,".txt",sep="")
								write.table(words, fname2, sep="\t", row.names=F)
							} 
							else print(paste0("There are no words starting in ",a," with these properties"))
						}
						else print("This sequence is not in the list") 
					}	
					else if (l==2) {
						if (a %in% poli$twoLet) {	
							words=subset(poli, twoLet==a & StressPattern==b)
							if (!(nrow(words)==0)) {	
								fname2=paste('words_',a,"_",b,".txt",sep="")
								write.table(words, fname2, sep="\t", row.names=F)
							} 
							else print(paste0("There are no words starting in ",a," with these properties"))
						}
						else print("This sequence is not in the list") 
					}
					else if (l==3) {
						if (a %in% poli$threeLet) {	
							words=subset(poli, threeLet==a & StressPattern==b)
							if (!(nrow(words)==0)) {	
								fname2=paste('words_',a,"_",b,".txt",sep="")
								write.table(words, fname2, sep="\t", row.names=F)
							} 
							else print(paste0("There are no words starting in ",a," with these properties"))
						}
						else print("This sequence is not in the list") 
					}	
					else if (l==4) {
						if (a %in% poli$fourLet) {	
							words=subset(poli, fourLet==a & StressPattern==b)
							if (!(nrow(words)==0)) {	
								fname2=paste('words_',a,"_",b,".txt",sep="")
								write.table(words, fname2, sep="\t", row.names=F)
							} 
							else print(paste0("There are no words starting in ",a," with these properties"))
						}
						else print("This sequence is not in the list") 
					}
				}	
			}
		}
	}
}
}
}
