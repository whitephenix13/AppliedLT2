from collections import defaultdict
import sys
import time

#HYPERPARAMETERS
MAX_SENTENCE_LENGTH= 7 # Do not consider create phrase pair in english or german that have >MAX_SENTENCE_LENGTH words
TEST_stop_after = 50000 #set to negative value to consider all dataset or set to max number of phrase
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
def extractPhrases(words,words_tgt,alignments,countDict,event_proba):

    currentPhrases = [] # Memory of all the currently build "phrases" (they can be incorrect)
    currentBoxes = [] # The corresponding boxes to the current phrases
    first_index=[] #keep track of the index of the first word of the currentPhrases

    all_phrase_pair = []
    all_boxes = []
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
                if len(phrase_tgt)<=MAX_SENTENCE_LENGTH :
                    phrase_tgt_new = " ".join([x for x in phrase_tgt if x != None])

                    #word based reordering count
                    countDict,event_proba = addWordBasedEvent((currentPhrases[i],phrase_tgt_new),newBox,alignments,countDict,event_proba)
                    all_phrase_pair.append((currentPhrases[i],phrase_tgt_new))
                    all_boxes.append(newBox[:]) #Use this to make a true copy of the list

    #Update phrase based reordering counts
    for ind, phrase_pair in enumerate(all_phrase_pair):
        countDict, event_proba =addPhraseBasedEvent(phrase_pair, all_boxes[ind], all_boxes, countDict,event_proba)

    return countDict,event_proba

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

#Functions to ease the reading of the code
def left(box):
    return box[1]
def right(box):
    return box[3]
def bottom(box):
    return box[0]
def top(box):
    return box[2]

#Function to return the size of the box.
def size(box):
    if box==[-1,-1,-1,-1] :
        return -1
    return (box[2]-box[0]+1)*(box[3]-box[1]+1)
####################### REORDERING EVENTS COUNT ####################################################################

#phrase_pair: String tuple of (German phrase, English phrase)
#phrase_box :  Box corresponding of the current phrase of the form : [x min,y min,x max,y max] with an alignment defined as (x,y)
#sentence_alignments: list of pair of alignments (x,y) where x corresponds to the German word and y the English one
#countDict: dict[(German word,English word)] = [p1(word),p2(word),...,p8(word),p1(phrase),p2(phrase),...,p8(phrase]
#Returns the updated countDictionnary after addition of all events count (both left and right) in countDict
def addWordBasedEvent(phrase_pair, phrase_box,  sentence_alignments, countDict,event_proba):
    #Get rid of unaligned words
    if phrase_box==[-1,-1,-1,-1] :
        return countDict,event_proba
    lr_event = 3 #left right, 0=monotonic, 1=swap, 2= disc left, 3=disc right
    rl_event = 7 #right left, 4=monotonic, 5=swap, 6= disc left, 7=disc right

    left_alignment_found = False #set to true if the "best" event was found (wrt monotonic > swap > disc)
    right_alignment_found = False#set to true if the "best" event was found (wrt monotonic > swap > disc)
    shift = 0 # int to shift the english word we are looking at. Used to handle unaligned English words

    max_right_bound = -1
    left_bound_reached = False
    right_bound_reached = False

    #Loop until an non empty English word was found (so that a reordering event can be computed) or that the index is out
    #of bounds
    while not left_alignment_found or not right_alignment_found:
        some_right_align_found = False #set to true if some event was found (but not necessarily the best)
        some_left_align_found = False #set to true if some event was found (but not necessarily the best)

        for alignment_pair in sentence_alignments:
            x=alignment_pair[0]
            y=alignment_pair[1]
            #compute the maximum bound to the right
            if y > max_right_bound :
                max_right_bound=y
            #check left to right alignments if not found yet
            if not right_alignment_found and (right(phrase_box)+1+shift) == y:
                some_right_align_found = True
                #check for monotonic
                if (top(phrase_box)+1) == x:
                    lr_event = 0
                elif lr_event > 1:
                    #check for swap
                    if (bottom(phrase_box) - 1) == x:
                        lr_event = 1
                    else:
                        #check for left discontinuities
                        #NOTE: if phrases are correctly built, there is either a left or right discontinuities
                        #but their can't be both!
                        if (bottom(phrase_box)> x):
                            lr_event = 2
                        else:
                            lr_event = 3

            # check right to left alignments if not found yet
            if not left_alignment_found and (left(phrase_box) - 1-shift) == y:
                some_left_align_found=True
                # check for monotonic
                if (bottom(phrase_box) - 1) == x:
                    rl_event = 4
                elif rl_event > 5:
                    # check for swap
                    if (top(phrase_box) + 1) == x:
                        rl_event = 5
                    else:
                        # check for left discontinuities
                        # NOTE: if phrases are correctly built, there is either a left or right discontinuities
                        # but their can't be both!
                        if bottom(phrase_box) > x:
                            rl_event = 6
                        else:
                            rl_event = 7
        shift += 1
        #alignment is considered to be found if it was already found or it was found in this iteration or the boundary
        #is reached

        #Compute if an alignment was found
        left_alignment_found = left_alignment_found or some_left_align_found
        right_alignment_found = right_alignment_found or some_right_align_found

        #Check for out of bound and allow to know if the countDict has to be updated
        left_bound_reached = left_bound_reached or not left_alignment_found and (left(phrase_box) - 1 - shift) < 0
        right_bound_reached = right_bound_reached or not right_alignment_found and (right(phrase_box) + 1 + shift) > max_right_bound

        #Update wrt bound
        left_alignment_found = left_alignment_found or left_bound_reached
        right_alignment_found = right_alignment_found or right_bound_reached

    if not right_bound_reached:
        countDict[phrase_pair][lr_event]+=1
        event_proba[lr_event] +=1
    if not left_bound_reached:
        countDict[phrase_pair][rl_event]+=1
        event_proba[rl_event] += 1
    return countDict,event_proba

#phrase_pair: String tuple of (German phrase, English phrase)
#source_box :  Box corresponding of the current source phrase of the form : [x min,y min,x max,y max] with an alignment defined as (x,y)
#target_boxes: box associated to the target phrases
#countDict: dict[(German word,English word)] = [p1(word),p2(word),...,p8(word),p1(phrase),p2(phrase),...,p8(phrase]
#Returns the updated countDictionnary after addition of all events count (both left and right) in countDict
def addPhraseBasedEvent(phrase_pair, source_box, target_boxes, countDict,event_proba):
    # Get rid of unaligned words
    if source_box == [-1, -1, -1, -1]:
        return countDict,event_proba

    #Find the best box (ie the closest to the source and the biggest wrt its size) to the left of the source one
    best_left_box = [-1,-1,-1,-1]
    best_left_dist = 10000 #dist to source box
    best_left_size = -1

    #Find the best box (ie the closest to the source and the biggest wrt its size) to the right of the source one
    best_right_box = [-1,-1,-1,-1]
    best_right_dist = 10000 #dist to source box
    best_right_size = -1

    #Look for the biggest box that is the closest to the source_phrase's box
    for index, t_box in enumerate(target_boxes):
        if t_box == [-1,-1,-1,-1]:
            continue
        #left case
        left_dist = left(source_box) - right(t_box)
        right_dist = left(t_box) - right(source_box)

        if left_dist > 0:
            # look for the closest box to the source box
            if left_dist == best_left_dist :
                # look for the biggest box
                left_size = size(t_box)
                if left_size > best_left_size:
                    best_left_dist = left_dist
                    best_left_box = t_box
                    best_left_size = size(t_box)
            elif left_dist < best_left_dist  :
                best_left_dist=left_dist
                best_left_box=t_box
                best_left_size= size(t_box)

        elif right_dist >0:
             #look for the closest box to the source box

            if right_dist == best_right_dist :
                # look for the biggest box
                right_size = size(t_box)
                if right_size>best_right_size:
                    best_right_dist = right_dist
                    best_right_box = t_box
                    best_right_size = size(t_box)
            elif right_dist < best_right_dist :
                best_right_dist = right_dist
                best_right_box = t_box
                best_right_size = size(t_box)

    lr_event = 11  # left right, 8=monotonic, 9=swap, 10= disc left, 11=disc right
    rl_event = 15  # right left, 12=monotonic, 13=swap, 14= disc left, 15=disc right
    if best_right_box != [-1,-1,-1,-1]:
        if bottom(best_right_box)-1 == top(source_box):
            lr_event = 8
        elif top(best_right_box)+1 == bottom(source_box):
            lr_event = 9
        elif top(best_right_box) < bottom(source_box):
            lr_event = 10
        else:
            lr_event = 11
        countDict[phrase_pair][lr_event] += 1
        event_proba[lr_event] += 1
    if best_left_box != [-1, -1, -1, -1]:
        if top(best_left_box) +1 == bottom(source_box):
            rl_event = 12
        elif bottom(best_left_box) -1 == top(source_box):
            rl_event= 13
        elif top(best_left_box) < bottom(source_box):
            rl_event = 14
        else:
            rl_event = 15
        countDict[phrase_pair][rl_event] += 1
        event_proba[rl_event] += 1
    return countDict,event_proba

#According to formula slide 22 of Reordering
def smoothCount(count_list,event_proba):
    sigma = 0.5
    smoothed_list = []
    for j in range(4) :
        count_sum = sum(count_list[4*j:4*(j+1)])
        for i in range(4*j,4*(j+1)):
            smoothed_list.append((sigma * event_proba [i] + count_list[i]) / (sigma + count_sum))
    return smoothed_list
####################### WRITING THE RESULT IN A FILE FUNCTION  ####################################################################""

#countDict: dict[(German word,English word)] = [p1(word),p2(word),...,p8(word),p1(phrase),p2(phrase),...,p8(phrase]
#Function to write the results in a file given the following format :
#f ||| e ||| p1 p2 p3 p4 p5 p6 p7 p8
#with p1 = p1 = p l->r (m|(f,e)) // p2 = p l->r (s|(f,e)) //  p3 = p l->r (dl|(f,e)) //  p4 = p l->r (dr|(f,e))
#with p1 = p5 = p r->l (m|(f,e)) // p6 = p r->l (s|(f,e)) //  p7 = p r->l (dl|(f,e)) //  p8 = p r->l (dr|(f,e))

def writeResults(filename,countDict):
    f = open(filename+"_word.txt", 'w')
    f2 = open(filename+"_phrase.txt", 'w')
    info = "f ||| e ||| p_l->r (m|(f,e)) p_l->r (s|(f,e)) p_l->r (dl|(f,e)) p_l->r (dr|(f,e)) " + \
           "p_r->l (m|(f,e)) p_r->l (s|(f,e)) p_r->l (dl|(f,e)) p_r->l (dr|(f,e))\n"
    f.write(info)
    f2.write(info)
    #for source_phrase, targetToCountDict in _phrases_src_given_tgt_counts.items():
        # f   |||e    |||p(f|e)      p(e|f) l(f|e) l(e|f) |||freq(f) freq(e) freq(f, e)
    for source_target_phrase,countList in countDict.items():
        smoothed_list = smoothCount(countList,GLOBAL_event_proba)
        f.write(str(source_target_phrase[0])+ " ||| "+ str(source_target_phrase[1])+ " ||| "+ str(smoothed_list[0])+\
                " "+ str(smoothed_list[1])+" "+ str(smoothed_list[2]) + " " + str(smoothed_list[3]) + " " + str(smoothed_list[4])+\
                " "+ str(smoothed_list[5])+" "+ str(smoothed_list[6]) + " " + str(smoothed_list[7]) + "\n")
        f2.write(str(source_target_phrase[0])+ " ||| "+ str(source_target_phrase[1])+ " ||| "+ str(smoothed_list[8])+" "+\
                 str(smoothed_list[9])+" "+ str(smoothed_list[10]) + " " + str(smoothed_list[11]) + " " + str(smoothed_list[12])+" "+\
                 str(smoothed_list[13])+" "+ str(smoothed_list[14]) + " " + str(smoothed_list[15]) + "\n")
    f.write("\n")
    f2.write("\n")
    f.close()
    f2.close()

############################            MAIN             ###########################################################################


### Define variables
GLOBAL_countDict = defaultdict(lambda : [0]*16) #Dictionnary of list of 16 elements for all counts (p1->p8 and word/phrase based)
                                                #countDict: dict[(German word,English word)] = [p1(word),p2(word),...,p8(word),
                                                # p1(phrase),p2(phrase),...,p8(phrase]
GLOBAL_event_proba =[0] * 16 #list of p(o) where o is a reoredering event: [p l->r(m), ..., p r->l(dr)] for word and phrase based
GLOBAL_count = 0 #Counter to see the number of iteration
max_count = TEST_stop_after if TEST_stop_after>0 else 50000
start_time = time.time()
#Main loop: extract phrases and count
for sentence_en, sentence_de, line_aligned in zip(GLOBAL_f_en, GLOBAL_f_de, GLOBAL_f_align):
    if GLOBAL_count > 0 and GLOBAL_count%1000 == 0:
        print('count: '+ str(GLOBAL_count)+ ' estimated ends in '+ str(((max_count / GLOBAL_count)-1) *(time.time()-start_time)))
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
    GLOBAL_countDict,GLOBAL_event_proba = extractPhrases(words_de, words_en, alignments,GLOBAL_countDict,GLOBAL_event_proba)
    GLOBAL_count += 1
print(GLOBAL_count)

#Transform event proba to real proba :
for i in range(4):
    event_sum = sum(GLOBAL_event_proba[4*i:4*(i+1)])
    if event_sum>0 :
        for j in range(4*i,4*(i+1)):
            GLOBAL_event_proba[j] /= event_sum

#Write the result
writeResults("final_result", GLOBAL_countDict)


print('done')

