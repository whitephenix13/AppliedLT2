from collections import defaultdict
import sys

#HYPERPARAMETERS
MAX_SENTENCE_LENGTH= 6 # Do not consider sentence and english or german that have >MAX_SENTENCE_LENGTH words
TEST = False #set to True to read the test. files instead of the file.

#Small fix for python 3 code
python3Code = False;
if (sys.version_info > (3, 0)):
    python3Code = True

#Open the .en, .de and .aligned files
DATA_DIR  = 'data/'
#Loaded files
filename = 'file'
if(TEST):
    filename = 'test'
GLOBAL_f_en = open(DATA_DIR+filename +'.en', 'r')
GLOBAL_f_de = open(DATA_DIR+filename+'.de', 'r')
GLOBAL_f_align = open(DATA_DIR +filename+'.aligned', 'r')

####################### PHRASE EXTRACTION FUNCTIONS ####################################################################""


#words: List of the words of the considered phrase (in German)
#words_tgt: List of the words of the considered phrase (in English)
#alignments: List of pairs as follow : [[0,0],[0,1],[1,2]] that corresponds to the words alignement
#words_src_given_tgt: Used for lexical probabilities, dictionnary [German word] [English word] of the normalized word count
#words_tgt_given_src: Used for lexical probabilities, dictionnary [English word] [German word] of the normalized word count
#The idea of the algorithm is to check if each word alone can be a phrase and to extend those phrase by adding new words
#to get all possible combination (see report for more explanation)
def extractPhrases(words,words_tgt, alignments,words_src_given_tgt,words_tgt_given_src):

    currentPhrases = [] # Memory of all the currently build "phrases" (they can be incorrect)
    currentBoxes = [] # The corresponding boxes to the current phrases
    first_index=[] #keep track of the index of the first word of the currentPhrases

    allPhrases=[] # List of all correct phrases
    all_phrases_tgt = [] # all phrases in the target side

    lexical_src_given_tgt = [] # lexical probability of source given target
    lexical_tgt_given_src = [] # lexical probability of target given source

    #Loop over all the source words
    for word_ind in range(len(words)) :
        wordAlignments = findWordAlignments(word_ind,alignments)
        #Add a new box and a new phrase for the word
        currentBoxes.append([-1,-1,-1,-1])
        currentPhrases.append("")
        first_index.append(None)

        #Add this new word to all memorized phrase and see if they are correct
        for i in range(len(currentBoxes)):
            if currentPhrases[i] == "":
                currentPhrases[i] = words[word_ind]
                first_index[i]=word_ind
            else:
                currentPhrases[i] += " " + words[word_ind]
            correct, newBox = checkCorrectPhrase(currentBoxes[i],wordAlignments,alignments)
            #Add the new phrase to the final result if correct
            if correct :
                phrase_tgt = [None] * len(words_tgt)  # to generate phrase in the target side

                allPhrases.append(currentPhrases[i])
                target_index = []

                # getting the target side phrases
                for cpi in range(first_index[i],word_ind+1):
                    word_alignments = findWordAlignments(int(cpi),alignments)
                    for wa in word_alignments:
                        phrase_tgt[wa[1]] = words_tgt[wa[1]]
                        target_index.append(wa[1])

                phrase_tgt_new = " ".join([x for x in phrase_tgt if x != None])
                all_phrases_tgt.append(phrase_tgt_new)

                # getting the l(e|f) and l(f|e)
                lexval_temp = 1.0
                lexval_temp2 = 1.0

                for cpi in range(first_index[i],word_ind+1):
                    word_alignments = findWordAlignments(int(cpi),alignments)
                    temp = 0
                    if (len(word_alignments) > 0):
                        for wa in word_alignments:
                            temp += words_src_given_tgt[words[wa[0]]][words_tgt[wa[1]]]
                        temp /= len(word_alignments)
                        lexval_temp *= temp

                for ti in target_index:
                    word_alignments2 = findWordAlignments(int(ti),alignments, True)
                    temp2 = 0
                    if (len(word_alignments2) > 0):
                        for wa in word_alignments2:
                            temp2 += words_tgt_given_src[words_tgt[wa[1]]][words[wa[0]]]
                        temp2 /= len(word_alignments2)
                        lexval_temp2 *= temp2

                lexical_src_given_tgt.append(lexval_temp)
                lexical_tgt_given_src.append(lexval_temp2)
                #Update the boxes
            currentBoxes[i] = newBox

    return allPhrases, all_phrases_tgt, lexical_src_given_tgt, lexical_tgt_given_src

#prevBox: Current box of the phrases (without the new word)
#wordAlignments: alignments for the new word to add to the phrase
#alignements: all alignements
#Check if a phrase is correct by looking if no words are aligned to the outside
def checkCorrectPhrase(prevBox,wordAlignments,alignments):
    #Compute new box by the addition of the new word
    newBox = computePhraseBox(prevBox,wordAlignments)
    #test if there is no alignment of a word in the box (either en or de) to a word outside it
    #Maily, test if there is no alignement [x,y] such that x in the box but not y , or the opposite
    correct=True
    for pair in alignments :
        x_in_box = (newBox[0]<=pair[0]) and (pair[0]<=newBox[2])
        y_in_box = (newBox[1]<=pair[1]) and (pair[1]<=newBox[3])
        if( (x_in_box and not y_in_box) or (not x_in_box and y_in_box)):
            correct=False
            break
    return correct,newBox


#word_index: index of the word in the phrase
#alignments: All words alignment
#reverse: if alignments is de -> en , set reverse to true to have the corresponding en -> de alignments
#Given a word index, compute all the alignments from this word to other words. If reverse parameter is true, check the opposite alignments
def findWordAlignments(word_index,alignments,reverse=False):
    allAlignments = []
    for pair in alignments:
        if( (not reverse and (word_index==pair[0])) or (reverse and (word_index==pair[1]) ) ):
            allAlignments.append(pair)
    return allAlignments

# prevBox: Current box of the phrases
# newAlignment: alignement corresponding to the new word in the format [x,y] that has to be added to the prevBox
# Computes the box associated to the phrase. The computation is done by extending the box due to the addition of a word to the phrase.
#ie: box is [0,1,2,3] (alignments are within the points (0,1) and (2,3)) and we want to add a new word which has the following alignment:
#(2,4), the box will then be extended as [0,1,2,4]
def computePhraseBox(prevBox, newAlignment):
    box=prevBox
    for pair in newAlignment:
        x=pair[0]
        y=pair[1]
        if(box[0]==-1):
            box= [x,y,x,y]
        else:
            if(box[0]>x):
                box[0]=x
            if (box[1] > y):
                box[1] = y
            if (box[2] < x):
                box[2] = x
            if (box[3] < y):
                box[3] = y
    return box

####################### LEXICAL PROBABILITY COMPUTATION FUNCTION ####################################################################""

#Normalize the count of the words
def normalize(d, target=1.0):
    raw = sum(d.values())
    factor = target/raw
    if python3Code:
        return {key: value * factor for key, value in list(d.items())}
    else:
        return {key:value*factor for key,value in d.iteritems()}


####################### WRITING THE RESULT IN A FILE FUNCTION  ####################################################################""

#Function to write the results in a file given the following format :
#f ||| e ||| p(f|e) p(e|f) l(f|e) l(e|f) ||| freq(f) freq(e) freq(f, e)
def writeResults(filename, _phrases_src_given_tgt_counts, lexical_src_given_tgt, lexical_tgt_given_src ):
    f = open(filename, 'w')
    f.write("f ||| e ||| p(f|e) p(e|f) l(f|e) l(e|f) ||| freq(f) freq(e) freq(f, e)\n")
    for source_phrase, targetToCountDict in _phrases_src_given_tgt_counts.items():
        freq_f = 0
        #Compute frequencies here to avoid too much memory usage
        for key, value in _phrases_src_given_tgt_counts[source_phrase].items():
            freq_f+=value
        for target_phrase, freq_fe in targetToCountDict.items():
            if(freq_fe >0):
                freq_e = 0
                for key, value in _phrases_src_given_tgt_counts.items():
                    if _phrases_src_given_tgt_counts[key][target_phrase]:
                        freq_e+=_phrases_src_given_tgt_counts[key][target_phrase]
                lef = lexical_src_given_tgt[source_phrase][target_phrase]
                lfe = lexical_tgt_given_src[target_phrase][source_phrase]
                # f   |||e    |||p(f|e)      p(e|f) l(f|e) l(e|f) |||freq(f) freq(e) freq(f, e)
                f.write(str(source_phrase) \
                        +" ||| " + str(target_phrase)\
                        +" ||| "+str(freq_fe/freq_e)+ " "+str(freq_fe/freq_f) + " " + str(sum(lef) / float(len(lef))) + " " + str(sum(lfe) / float(len(lfe)))\
                        +" ||| "+ str(freq_f)+ " "+ str(freq_e)+ " "+ str(freq_fe)+"\n" )
    f.close()

############################            MAIN             ###########################################################################


### count word occurences ###
GLOBAL_words_src_given_tgt_counts = defaultdict(lambda : defaultdict(float))
GLOBAL_words_tgt_given_src_counts = defaultdict(lambda : defaultdict(float))
for sentence_en, sentence_de, line_aligned in zip(GLOBAL_f_en, GLOBAL_f_de, GLOBAL_f_align):
    words_de = sentence_de.replace("\n", "").split()
    words_de = list(filter(lambda x: x != '', words_de))
    words_en = sentence_en.replace("\n", "").split()
    words_en = list(filter(lambda x: x != '', words_en))

    if not len(words_de) > MAX_SENTENCE_LENGTH and not len(words_en) > MAX_SENTENCE_LENGTH:
        alignments = []
        alignments_temp = line_aligned.replace("\n","").split(" ")
        for al in alignments_temp:
            alignments.append(al.split("-"))
        alignments=[list(map(int,pair)) for pair in alignments]
        for alig in alignments:
            GLOBAL_words_src_given_tgt_counts[words_de[alig[0]]][words_en[alig[1]]] += 1.0
            GLOBAL_words_tgt_given_src_counts[words_en[alig[1]]][words_de[alig[0]]] += 1.0



### Normalize words occurence (for lexical) ###
for key in GLOBAL_words_src_given_tgt_counts:
    GLOBAL_words_src_given_tgt_counts[key] = normalize(GLOBAL_words_src_given_tgt_counts[key])
for key in GLOBAL_words_tgt_given_src_counts:
    GLOBAL_words_tgt_given_src_counts[key] = normalize(GLOBAL_words_tgt_given_src_counts[key])

### Define variables
GLOBAL_phrases_src_given_tgt_counts = defaultdict(lambda : defaultdict(float)) #dictionnary[German word][English word]
                                                                            # = count(German word,English word)
GLOBAL_lexical_src_given_tgt_prob = defaultdict(lambda : defaultdict(list)) #Dictionnary[German word][English word]  = lexical prob
GLOBAL_lexical_tgt_given_src_prob = defaultdict(lambda : defaultdict(list))  #Dictionnary[English word][German word]  = lexical prob

#Reset the files so that they can be read again
GLOBAL_f_en.seek(0)
GLOBAL_f_de.seek(0)
GLOBAL_f_align.seek(0)

#Counter to see the number of iteration
GLOBAL_count = 0
#Write the actual data used in a file : filtered by MAX_SENTENCE_LENGTH
GLOBAL_f_en_s = open(DATA_DIR +filename+'_simp.en', 'w')
GLOBAL_f_de_s = open(DATA_DIR +filename+'_simp.de', 'w')
GLOBAL_f_align_s = open(DATA_DIR +filename+'_simp.aligned', 'w')


#Main loop: extract phrases and count
for sentence_en, sentence_de, line_aligned in zip(GLOBAL_f_en, GLOBAL_f_de, GLOBAL_f_align):
    if GLOBAL_count%1000 == 0:
        print('count: '+ str(GLOBAL_count))
    #Extract the words from the phrase and remove the empty word
    words_de = sentence_de.replace("\n","").split(" ")
    words_de = list(filter(lambda x: x != '', words_de))
    words_en = sentence_en.replace("\n","").split(" ")
    words_en = list(filter(lambda x: x != '', words_en))
    #Filter sentence that are too long
    if not len(words_de) > MAX_SENTENCE_LENGTH and not len(words_en) > MAX_SENTENCE_LENGTH:
        GLOBAL_f_en_s.write(sentence_en)
        GLOBAL_f_de_s.write(sentence_de)
        GLOBAL_f_align_s.write(line_aligned)
        alignments = []
        #Store the alignments in the format : [[1,1][2,1][3,2]]
        alignments_temp = line_aligned.replace("\n","").split(" ")
        for al in alignments_temp:
            alignments.append(al.split("-"))
        alignments=[list(map(int,pair)) for pair in alignments]
        #Extracts phrases and count phrases occurences
        phrases_src, phrases_tgt, lexical_src_given_tgt, lexical_tgt_given_src = extractPhrases(words_de, words_en, alignments,\
                    GLOBAL_words_src_given_tgt_counts, GLOBAL_words_tgt_given_src_counts)
        for p_src, p_tgt, l_src_tgt, l_tgt_src in zip(phrases_src, phrases_tgt, lexical_src_given_tgt, lexical_tgt_given_src):
            GLOBAL_phrases_src_given_tgt_counts[p_src][p_tgt] += 1.0
            GLOBAL_lexical_src_given_tgt_prob[p_src][p_tgt].append(l_src_tgt)
            GLOBAL_lexical_tgt_given_src_prob[p_tgt][p_src].append(l_tgt_src)
    GLOBAL_count+=1
print(GLOBAL_count)

#Write the result, this operation is slow (~30 sec) to avoid too much memory usage
writeResults("test_big.txt", GLOBAL_phrases_src_given_tgt_counts, GLOBAL_lexical_src_given_tgt_prob, GLOBAL_lexical_tgt_given_src_prob)
print('done')

