#####################################
############ FOR R USERS ############
#####################################


###########  WORD BEGINNING SEARCH

There are 2 functions you have to use: prepbeg_nOfLet.R and beg.R

1) Run the function prepbeg_nOfLet.R; this function has only one argument, i.e. a number for 1 to 4; the number specifies the length of word beginning you are interested in. Therefore, if you are interested in looking at word beginnings of 2 letters, we run the function with 2 as argument, i.e., prepbeg_nOfLet(2). The function creates and saves an object called db2.RData (where the number 2 indicates the argument you have specified in the function; this means that if you run, e.g.,  prepbeg_nOfLet(1), the object you will create will be db1.RData). The object you create is used by the second function. 
Note that the prepbeg_nOfLet function may be run only one time for each word beginning length. Once you have created the db object you are interested in, you can directly run the second function, i.e., beg.R

2) Run the beg.R; this function has 2 mandatory arguments (a, b) and 3 optional arguments (syll, gCat, cvStrFin). 
The arguments are: 
a = the word beginning you want to look for (note that the sequence must be specified within quotes; e.g., "ba"); the function will work on the RData object created by prepbeg_nOfLet with the corresponding number of letters (e.g., since "ba" has two letters, the function will work on the db2.RData object);
b = the stress pattern you want to look for (for 0 to 3, with 0 = final stress; 1 = penultimate; 2 = antepenultimate; 3 = preantepenultimate); 
syll = the number of syllables of the words you look for (min 2 max 11; if the argument is not specified the function considers all the syllabic lengths); 
gCat = it specifies the grammatical category of words you look for (if the argument is not specified the function considers all the grammatical categories); note that you have to use the symbols contained in the legend.txt file within quotes (e.g., "S" for nouns);
cvStrFin = it specifies the structure of the word ending of words you look for (with V = vowel, C = consonant, S = stressed vowel, A = apostrophe). If the argument is not specified , the functions considers all the cv-structures(note that the cvStrFin must be specified within quotes; e.g., "VCV"). 
example of application with the mandatory arguments only: beg("ba",1).
example of application with all arguments: beg("ba",1,syll=3,gCat="S",cvStrFin="VCV")
For each search, the function automatically saves the results in a txt file that specifies in the extension the word beginning and the stress pattern you looked for (e.g., words_ba_1.txt). Note that optional arguments (i.e., number of syllables, grammatical category, and CV-structure) are not included in the name of the txt file. Each search replaces the txt file saved for searches with the same beginning and stress pattern regardless of which optional arguments are used.



###########  WORD ENDING SEARCH

There are 2 files you have to use: the script prepFinSeq.R and the function finSeq.R

1) read the script prepFinSeq.R (you can do with source("prepFinSeq.R") as for any other R script or function). The script creates an object (dbfSeq.RData) that is then used by the second function. 
Note that you can read and run this script only the first time; once you have created the dbfSeq.RData object you can directly run function finSeq.R
 
2) Run the function finSeq.R; this function has 2 mandatory arguments (a, b) and 2 optional arguments (syll, gCat)
The arguments are: 
a = the word ending to look for (note that the sequence must be specified within quotes; e.g., "ola")
b = the stress pattern to look for (for 0 to 3, with 0 = final stress; 1 = penultimate; 2 = antepenultimate; 3 = preantepenultimate);
syll = the number of syllables of the words you look for (min 2 max 11; if the argument is not specified the function considers all the syllabic lengths); 
gCat = it specifies the grammatical category of words you look for (if the argument is not specified the function considers all the grammatical categories); note that you have to use the symbols contained in the legend.txt file within quotes (e.g., "S" for nouns);
example of application with mandatory arguments only: finSeq("olo",1)
example of application with all arguments: finSeq("olo",1, syll=3, gCat="S")
For each search, the function automatically saves the results in a txt file that specifies in the extension the word ending and the stress pattern you looked for (e.g., words_ba_1.txt).  Note that optional arguments (i.e., number of syllables and grammatical category) are not included in the name of the txt file. Each search replaces the txt file saved for searches with the same ending and stress pattern regardless of which optional arguments are used.
