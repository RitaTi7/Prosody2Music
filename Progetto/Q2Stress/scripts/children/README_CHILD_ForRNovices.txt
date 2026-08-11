#####################################	
########### FOR R NOVICES ###########	
#####################################	
	
	
########### BEFORE YOU START	
	
1) Download R from http://www.r-project.org/	
	
2) Download the compressed folder with the functions and the database (for the folder location, see in the paper)	
	
3) create a folder (e.g., myFolder) in which you put all the files you have extracted from the compressed folder	
	
	
	
###########  WORD BEGINNING SEARCH	
	
There are 2 functions you have to use: CHILD_prepbeg_nOfLet.R and CHILD_beg.R	
	
1) Open R	
	
	
2) set the directory of the folder (myFolder in the example above) containing all the relevant files (functions and database). To do that, you have 2 ways: 	
	a) copy and paste the following line in the R command window and press Enter
	setwd("myPath"), where myPath indicates the path of your folder 
	[e.g., setwd("/Users/simone/Desktop/prova/myFolder")]
	b) use the software interface (if you use Windows, go to File-> Change Directory; 
	if you use Mac, go to Misc-> Change Working Directory)
	
	
3) read the function CHILD_prepbeg_nOfLet.R; To do that, write 	
	source("CHILD_prepbeg_nOfLet.R")
	and press Enter
Now the function is loaded and you can run it. 	
	
	
4) Run the function. The function CHILD_prepbeg_nOfLet.R has only one argument, i.e. a number from 1 to 4; the number specifies the length of word beginnings you are interested in. Therefore, if you are interested in looking at the word beginnings of 2 letters, you run the function with 2 as argument, i.e., CHILD_prepbeg_nOfLet(2). The function creates and saves an object called db2.RData (where the number 2 indicates the argument you have specified in the function; this means that if you run, e.g., CHILD_prepbeg_nOfLet(1), you will create an object called db1.RData). The object you create will be used later on. How to run the CHILD_prepbeg_nOfLet function? Just write	
	CHILD_prepbeg_nOfLet(n)
	and press Enter [NOTE that instead of n you have to write a number from 1 to 4; e.g., CHILD_prepbeg_nOfLet(3)]
	
The function takes a while (you can see on your screen some numbers indicating the processing). Before doing step 5, wait until the function ends. 	
	
Note that the CHILD_prepbeg_nOfLet function may be run only one time for each word beginning length. Once you have create the db object you are interested in, you can directly run the second function, i.e., CHILD_beg.R	
	
	
5) read the function CHILD_beg.R; To do that, write	
	source("CHILD_beg.R")
	and press Enter
Now the function is loaded and you can run it. 	
	
	
6) Run the function. The function CHILD_beg.R has 2 mandatory arguments (a, b) and 3 optional arguments (syll, gCat, cvStrFin). If an argument is mandatory means that you must specify it in the function; differently, if an argument is optional you can decide whether you want or do not want to specify it. Therefore, when you run the function, you must specify at least 2 arguments (a, b) and, if you want, you may refine your research by specifying also some (or all) of the optional arguments.	
The arguments are: 	
a = it specifies the word beginning you want to look for (note that the sequence must be specified within quotes; e.g., "ba"); the function will work on the RData object created by prepbeg_nOfLet with the corresponding number of letters (e.g., since "ba" has two letters, the function will work on the db2.RData object);
b = it specifies the stress pattern to look for (for 0 to 3, with 0 = final stress; 1 = penultimate; 2 = antepenultimate; 3 = preantepenultimate); 	
syll = it is the number of syllables of the words you look for (min 2 max 11; if the argument is not specified the function considers all the syllabic lengths); 	
gCat = it specifies the grammatical category of words you look for (if the argument is not specified the function considers all the grammatical categories); note that you have to use the symbols contained in the legend.txt file within quotes (e.g., "S" for nouns);	
cvStrFin = it specifies the structure of the word ending of words you look for (with V = vowel, C = consonant, S = stressed vowel, A = apostrophe). If the argument is not specified, the function considers all the cv-structures (note that the cvStrFin must be specified within quotes; e.g., "VCV").
How to run the functions? Below some examples; Write	
	
	CHILD_beg("ba",1)
	and press Enter
	
	CHILD_beg("ba",1, syll=3)
	and press Enter
	
	CHILD_beg("ba",1, syll=3, gCat="S")
	and press Enter
	
	CHILD_beg("ba",1, gCat="S")
	and press Enter
	
	CHILD_beg("ba",1, syll=3, gCat="S", cvStrFin="VCV")
	and press Enter
	
	CHILD_beg("ba",1, syll=3, cvStrFin="VCV")
	and press Enter
	
Note that you can freely decide whether you want to use no optional arguments, only one optional argument, two optional arguments, or all the 3 optional arguments.	
	
For each search, the function automatically saves the results in a txt file that specifies in the extension the word beginning and the stress pattern you looked for (e.g., words_ba_1.txt). Such file can be opened as any standard txt file (e.g., with excel). Note that optional arguments (i.e., number of syllables, grammatical category, and CV-structure) are not included in the name of the txt file. Each search replaces the txt file saved for searches with the same beginning and stress pattern regardless of which optional arguments are used.	
	
When you run the function, if by mistake you are looking for something not in the database (e.g., you want to look for all words starting with "bb" or the words of 15 syllables), the function returns a message indicating that the property is not in the database. In this case, simply re-run the function by changing the value of the argument.	
	
	
7) If you want to do multiple searches, repeat step 6) as many times as you want.	
	
	
	
###########  WORD ENDING SEARCH	
	
There is only one file you have to use: the function CHILD_finSeq.R	
	
1) Open R	
	
	
2) set the directory of the folder (myFolder in the example above) containing all the relevant files. To do that, you have 2 ways: 	
	a) copy and paste the following line in the R command window and press Enter
	setwd("myPath"), where myPath indicates the path of your folder 
	[e.g., setwd("/Users/simone/Desktop/prova/myFolder")]
	b) use the software interface (if you use Windows, go to File-> Change Directory; 
	if you use Mac, go to Misc-> Change Working Directory)
	
3) Run the function CHILD_finSeq.R; this function has 2 mandatory arguments (a, b) and 2 optional arguments (syll, gCat). If an argument is mandatory means that you must specify it in the function; differently, if an argument is optional you can decide whether you want or do not want to specify it. Therefore, when you run the function, you must specify at least 2 arguments (a,b) and, if you want, you may refine your research by specifying also some (or all) of the optional arguments.	
The arguments are:  	
a = it specifies the word ending you look for (note that the sequence must be specified within quotes; e.g., "ola")	
b = it specifies the stress pattern you look for (for 0 to 3, with 0 = final stress; 1 = penultimate; 2 = antepenultimate; 3 = preantepenultimate);	
syll = it is the number of syllables of the words you look for (min 2 max 11; if the argument is not specified the function considers all the syllabic lengths);  	
gCat = it specifies the grammatical category of words you look for (if the argument is not specified the function considers all the grammatical categories); note that you have to use the symbols contained in the legend.txt file within quotes (e.g., "S" for nouns);	
How to run the function? Below some examples; Write	
	CHILD_finSeq("olo",1)
	and press Enter
	
	CHILD_finSeq("olo",1,syll=3,gCat="S")
	and press Enter
	
	CHILD_finSeq("olo",1,syll=3)
	and press Enter
	
	CHILD_finSeq("olo",1,gCat="S")
	and press Enter
	
Note that you can freely decide whether you want to use no optional arguments, only one optional argument, or two optional arguments.	
	
For each search, the function automatically saves the results in a txt file that specifies in the extension the word beginning and the stress pattern you looked for (e.g., words_olo_1.txt). Such file can be opened as any standard txt file (e.g., with excel). Note that optional arguments (i.e., number of syllables and grammatical category) are not included in the name of the txt file. Each search replaces the txt file saved for searches with the same ending and stress pattern regardless of which optional arguments are used.
	
When you run the function, if you are looking for something not in the database (e.g., you want to look for all words ending with "tpy" or the words of 15 syllables), the function returns a message indicating that that property is not in the database. In this case, simply re-run the function by changing the value of the argument.	
	
	
4) If you want to do multiple searches, repeat step 3) as many times as you want.	
