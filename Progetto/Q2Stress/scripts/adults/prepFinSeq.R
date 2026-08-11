## PREPARE A DATABASE WITH FINAL SEQUENCES

d=read.delim('phonItalia 1.10.1 - word forms.txt',header=T)
poli = d[d$SumSylls > 1,]
poli$StressPattern = poli$SumSylls-poli$StressedSyllable

out=data.frame()
for (j in 1:length(poli$word)) {
			t = as.character(poli$word[j])
			s = strsplit(t,'')
			tmpb1=s[[1]][length(s[[1]])]
			tmpb2=s[[1]][length(s[[1]])-1]
			tmpb3=s[[1]][length(s[[1]])-2]			
			tmp=paste(tmpb3,tmpb2,tmpb1,sep="")
			out=append(out,tmp)
			cat(j,"of", length(poli$word), "\r")
			flush.console()	
		}
out=as.factor(as.character(out))
poli$FinSeq = out
save(poli, file="dbfSeq.RData")
