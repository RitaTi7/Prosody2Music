

# the argument a must be a number from 1 to 4.
CHILD_prepbeg_nOfLet = function (a) {

#read the file
load("lexeLem.RData")
d = p
poli = d[d$SumSylls > 1,]

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





