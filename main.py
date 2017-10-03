from collections import defaultdict
import sys

#HYPERPARAMETERS
MAX_SENTENCE_LENGTH= 7 # Do not consider create phrase pair in english or german that have >MAX_SENTENCE_LENGTH words
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
#countDict: dict[(German word,English word)] = [p1(word),p2(word),...,p8(word),p1(phrase),p2(phrase),...,p8(phrase]
#The idea of the algorithm is to check if each word alone can be a phrase and to extend those phrase by adding new words
#to get all possible combination (see report for more explanation)
def extractPhrases(words,words_tgt,alignments,countDict):

    currentPhrases = [] # Memory of all the currently build "phrases" (they can be incorrect)
    currentBoxes = [] # The corresponding boxes to the current phrases
    first_index=[] #keep track of the index of the first word of the currentPhrases

    #Loop over all the source words
    for word_ind in range(len(words)) :
        wordAlignments = findWordAlignments(word_ind,alignments)
        #Add a new box and a new phrase for the word
        currentBoxes.append([-1,-1,-1,-1])
        currentPhrases.append("")
        first_index.append(None)

        #shorten the considered phrases to only create phrase of length at most MAX_SENTENCE_LENGTH
        final_i = len(currentBoxes)
        first_i = 0 if (len(currentBoxes)<MAX_SENTENCE_LENGTH) else (len(currentBoxes) - MAX_SENTENCE_LENGTH)

        #Add this new word to all memorized phrase and see if they are correct
        for i in range(first_i,final_i):
            if currentPhrases[i] == "":
                currentPhrases[i] = words[word_ind]
                first_index[i]=word_ind
            else:
                currentPhrases[i] += " " + words[word_ind]
            correct, newBox = checkCorrectPhrase(currentBoxes[i],wordAlignments,alignments)
            #Update current box
            currentBoxes[i] = newBox
            #Add the new phrase to the final result if correct
            if correct :
                phrase_tgt = [None] * len(words_tgt)  # to generate phrase in the target side

                target_index = []

                # getting the target side phrases
                for cpi in range(first_index[i],word_ind+1):
                    word_alignments = findWordAlignments(int(cpi),alignments)
                    for wa in word_alignments:
                        phrase_tgt[wa[1]] = words_tgt[wa[1]]
                        target_index.append(wa[1])

                phrase_tgt_new = " ".join([x for x in phrase_tgt if x != None])
                #TODO: do reordering count here
                countDict[(currentPhrases[i],phrase_tgt_new)][0]+=1 #Dummy count to test the dictionnary and writting

    return countDict

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

####################### WRITING THE RESULT IN A FILE FUNCTION  ####################################################################""

#countDict: dict[(German word,English word)] = [p1(word),p2(word),...,p8(word),p1(phrase),p2(phrase),...,p8(phrase]
#Function to write the results in a file given the following format :
#f ||| e ||| p1 p2 p3 p4 p5 p6 p7 p8
#with p1 = p1 = p l->r (m|(f,e)) // p2 = p l->r (s|(f,e)) //  p3 = p l->r (dl|(f,e)) //  p4 = p l->r (dr|(f,e))
#with p1 = p5 = p r->l (m|(f,e)) // p6 = p r->l (s|(f,e)) //  p7 = p r->l (dl|(f,e)) //  p8 = p r->l (dr|(f,e))

def writeResults(filename,countDict):
    f = open(filename, 'w')
    f.write("f ||| e ||| p_l->r (m|(f,e)) p_l->r (s|(f,e)) p_l->r (dl|(f,e)) p_l->r (dr|(f,e)) " + \
            "p_r->l (m|(f,e)) p_r->l (s|(f,e)) p_r->l (dl|(f,e)) p_r->l (dr|(f,e))\n")
    #for source_phrase, targetToCountDict in _phrases_src_given_tgt_counts.items():
        # f   |||e    |||p(f|e)      p(e|f) l(f|e) l(e|f) |||freq(f) freq(e) freq(f, e)
    for source_target_phrase,countList in countDict.items():
        #TODO: agregate the 16 values to 8
        f.write(str(source_target_phrase[0])+ " ||| "+ str(source_target_phrase[1])+ " ||| "+ str(countList[0])+" "+ str(countList[1])+" "+ \
                str(countList[2]) + " " + str(countList[3]) + " " + str(countList[4])+" "+ str(countList[5])+" "+ \
                str(countList[6]) + " " + str(countList[7]) + "\n")
    f.write("\n")
    f.close()

############################            MAIN             ###########################################################################


### Define variables
GLOBAL_countDict = defaultdict(lambda : [0]*16) #Dictionnary of list of 16 elements for all counts (p1->p8 and word/phrase based)
                                                #countDict: dict[(German word,English word)] = [p1(word),p2(word),...,p8(word),
                                                # p1(phrase),p2(phrase),...,p8(phrase]

GLOBAL_count = 0 #Counter to see the number of iteration

TEST_stop_after = -1 #set to negative value to consider all dataset or set to max number of phrase
#Main loop: extract phrases and count
for sentence_en, sentence_de, line_aligned in zip(GLOBAL_f_en, GLOBAL_f_de, GLOBAL_f_align):
    if GLOBAL_count%1000 == 0:
        print('count: '+ str(GLOBAL_count))
    if TEST_stop_after>0 and GLOBAL_count > TEST_stop_after:
        break
    #Extract the words from the phrase and remove the empty word
    words_de = sentence_de.replace("\n","").split(" ")
    words_de = list(filter(lambda x: x != '', words_de))
    words_en = sentence_en.replace("\n","").split(" ")
    words_en = list(filter(lambda x: x != '', words_en))
    alignments = []
    #Store the alignments in the format : [[1,1][2,1][3,2]]
    alignments_temp = line_aligned.replace("\n","").split(" ")
    for al in alignments_temp:
        alignments.append(al.split("-"))
    alignments=[list(map(int,pair)) for pair in alignments]
    #Extracts phrases and count phrases occurences
    GLOBAL_countDict = extractPhrases(words_de, words_en, alignments,GLOBAL_countDict)
    GLOBAL_count += 1
print(GLOBAL_count)

#Write the result
writeResults("final_result.txt", GLOBAL_countDict)
print('done')

