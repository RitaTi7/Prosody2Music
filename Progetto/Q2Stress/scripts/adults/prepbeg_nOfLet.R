

# the argument a must be a number from 1 to 4.
prepbeg_nOfLet = function (a) {

#read the file
d=read.delim('phonItalia 1.10.1 - word forms.txt',header=T)
poli = d[d$SumSylls > 1,]
poli$StressPattern = poli$SumSylls-poli$StressedSyllable

out3=data.frame()
for (j in 1:length(poli$word)) {
			t = as.character(poli$word[j])
			s = strsplit(t,'')
			tmpb1=s[[1]][length(s[[1]])]
			tmpb2=s[[1]][length(s[[1]])-1]
			tmpb3=s[[1]][length(s[[1]])-2]			
			tmp=paste(tmpb3,tmpb2,tmpb1,sep="")
			out3=append(out3,tmp)
			cat(j,"of", length(poli$word), "\r")
			flush.console()	
		}
out3=as.factor(as.character(out3))
poli$FinSeq = out3

out2=data.frame()
for (k in 1:length(poli$NewOrthVCV)) {
			t = as.character(poli$NewOrthVCV[k])
			s = strsplit(t,'')
			tmpb1=s[[1]][length(s[[1]])]
			tmpb2=s[[1]][length(s[[1]])-1]
			tmpb3=s[[1]][length(s[[1]])-2]			
			tmp=paste(tmpb3,tmpb2,tmpb1,sep="")
			out2=append(out2,tmp)
			cat(k,"of", length(poli$NewOrthVCV), "\r")
			flush.console()	
		}
out2=as.factor(as.character(out2))
poli$OrtFinSeq = out2

	if (a ==1) {
		out=data.frame()
		for (n in 1:length(poli$lemma)) {
			t = as.character(poli$lemma[n])
			s = strsplit(t,'')
			tmp=s[[1]][1]
			out=append(out,tmp)
			cat(n,"of", length(poli$lemma), "\r")
			flush.console()	
		}
		out=as.factor(as.character(out))
		poli$firstLet = out
		save(poli, file="db1.RData")
	}
	
	else if (a==2) {
		out=data.frame()
		for (n in 1:length(poli$lemma)) {
			t = as.character(poli$lemma[n])
			s = strsplit(t,'')
			tmpb=s[[1]][1:2]
			tmp=paste(tmpb[[1]][1],tmpb[[2]][1],sep="")
			out=append(out,tmp)
			cat(n,"of", length(poli$lemma), "\r")
			flush.console()	
		}
		out=as.factor(as.character(out))
		poli$twoLet = out
		save(poli, file="db2.RData")
	}
	
	else if (a==3) {
		out=data.frame()
		for (n in 1:length(poli$lemma)) {
			t = as.character(poli$lemma[n])
			s = strsplit(t,'')
			tmpb=s[[1]][1:3]
			tmp=paste(tmpb[[1]][1],tmpb[[2]][1],tmpb[[3]][1],sep="")
			out=append(out,tmp)
			cat(n,"of", length(poli$lemma), "\r")
			flush.console()	
		}
		out=as.factor(as.character(out))
		poli$threeLet = out
		save(poli, file="db3.RData")
	}
	
	else if (a==4) {
		out=data.frame()
		for (n in 1:length(poli$lemma)) {
			t = as.character(poli$lemma[n])
			s = strsplit(t,'')
			tmpb=s[[1]][1:4]
			tmp=paste(tmpb[[1]][1],tmpb[[2]][1],tmpb[[3]][1],tmpb[[4]][1],sep="")
			out=append(out,tmp)
			cat(n,"of", length(poli$lemma), "\r")
			flush.console()	
		}
		out=as.factor(as.character(out))
		poli$fourLet = out
		save(poli, file="db4.RData")
	}
}





