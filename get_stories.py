import os

"""This function creates from the original corpus a number of files where each file represents a separate story. The name of the file reflects the sequence of episodes. Each file contains six lines, one for each episode."""
def get_stories():
	f = open("original_corpus_with_deletion_of_repetitions.txt", 'r')
	corpus = []
	iterator = 0
	for line in f:
		elements = line.split('\t')
		#corpus.append([e for e in elements])
		filename = elements[0]+"_"+str(iterator) +"_"+elements[1]+"_"+elements[3]+"_"+elements[5]+"_"+elements[7]+"_"+elements[9]+"_"+elements[11]+".txt"
		story = open("narrative_corpus_story_per_file\\"+filename, 'w+')
		
		#Check if line ends with a punct mark. If not, add a full stop
		end_of_sent=['.', '?', '!', '"']
		for i in range(2,14,2):
			if elements[i].rstrip()[-1] not in end_of_sent:
				elements[i]=elements[i].rstrip()+'.'
		
		story.write(elements[2]+"\n"+elements[4]+"\n"+elements[6]+"\n"+elements[8]+"\n"+elements[10]+"\n"+elements[12]+"\n")
		iterator += 1
	#print(len(corpus))

import re
import xml.etree.ElementTree as etree

"""This function takes a generated xml file by CAEVO as input and adds the episode number to each sentence, i.e. to which episode the sentence belongs to in the original corpus. The episode number is added as a tag, e.g. "<episode number="1" />" at the end of each entry.
Output: the file named "output.xml".
"""
def add_episodes(orig_xmlfile):
	etree.register_namespace("","http://chambers.com/corpusinfo")
	tree = etree.parse(orig_xmlfile)
	root = tree.getroot()
	error=0
	sentence_count=0
	out_test=''
	for file in root:
		#Get the filename from xml
		filename = file.attrib['name']
		episode_sequence = filename[:-4].split('_')
		
		#Open the original txt file
		input_file = open('narrative_corpus_story_per_file\\'+filename, 'r')
		input_file_list = [line.rstrip() for line in input_file]
		for entry in file:
			if entry.tag == "{http://chambers.com/corpusinfo}entry":
				sentence = entry[0].text
				#Remove parentheses
				sentence = sentence.replace("-LRB-", "")
				sentence = sentence.replace("-RRB-", "")	
				
				sentence_count+=1
				#check which episode this sentence belongs to
				line_num = 0
				for line in input_file_list:
					separ_line = line.replace("n't", " n't") #separate negative particles
					separ_line = separ_line.replace("N'T", " N'T")
					separ_line = separ_line.replace("realise", "realize")
					separ_line = separ_line.replace("realising", "realizing")
					separ_line = separ_line.replace("grey", "gray")
					separ_line = separ_line.replace("favourite", "favorite")
					separ_line = separ_line.replace("centre", "center")
					separ_line = separ_line.replace("neighbours", "neighbors")
					separ_line = separ_line.replace("with a neighbour", "with a neighbor")
					separ_line = separ_line.replace("colour", "color")
					separ_line = separ_line.replace("cannot", "can not")
					separ_line = separ_line.replace("vigour", "vigor")
					line_list = re.findall(r"[\w]+", separ_line) #separate words and punct signs
					
					sentence_list = re.findall(r"[\w]+", sentence)
					
					if set(sentence_list).issubset(line_list):
						etree.SubElement(entry, '{http://chambers.com/corpusinfo}episode', attrib={'number':str(episode_sequence[line_num+2])})
						break
					line_num+=1
				'''#uncomment this to check if all episodes were added	
				sent_elements=''
								
				if entry.find('{http://chambers.com/corpusinfo}episode') == None:
					error+=1
					for element in sentence_list:
						sent_elements+=element+" "
					out_test+=filename+'\n'+sent_elements+'\n'+sentence+'\n\n'
	
	file_test=open('bugs.txt', "w+")
	file_test.write(out_test)
	print(error)
	print(sentence_count)'''
	tree.write('output.xml')
	
def get_patterns(xmlfile='output.xml'):
	tree = etree.parse(xmlfile)
	root = tree.getroot()
	patt_dict_overall={}
	timeline_rules={}
	for file in root:
		output=''
		episode_dict={}
		episode_dict_reduced={}
		episode_sequence=[]
		filename = file.attrib['name'][:-4]
		for entry in file:
			if entry.tag == "{http://chambers.com/corpusinfo}entry":
				sentence = entry[0].text
				deps = entry[3].text
				events = entry[4]
				timexes = entry[5]
				episode_number = entry[6].attrib['number']
				
				#tracking the episode sequence
				if episode_number not in episode_sequence:
					episode_sequence+=[episode_number]
				
				#Sentence consists of one word.
				if deps is None:
					continue
				
				#find root of the parse tree
				p = re.compile(r'root\(ROOT-0, (.+)-(\d+)\)')
				pattern = p.findall(deps)
				#print(pattern)
				dep_root=pattern[0][0]
				event_offset = int(pattern[0][1])
				
				pattern_dict = {} #store event and timexes in a dict with their offset as a key,
				#					e.g. {2:['PAST', 'NONE', 'wondered'], 21:['DURATION', 'PT1H', 'an hour']}
				
				#Identifying an event in a sentence: the root of a sentence tree must be listed in events
				main_event = ''
				for event in events:
					event_attr = event.attrib
					if event_attr['string'] == dep_root:
						main_event = dep_root
						tense = event_attr['tense']
						aspect = event_attr['aspect']
						event_offset = int(event_attr['offset'])
						break
						
				#If not found, take the first event
				if main_event == '':
					if events.text is not None:
						event_attr = events[0].attrib
						main_event = event_attr['string']
						tense = event_attr['tense']
						aspect = event_attr['aspect']
						event_offset = int(event_attr['offset'])
				#otherwise, the sentence root is an event without the attributes
					else:
						main_event = dep_root
						tense = 'Not available'
						aspect = 'Not available'
						#event_offset = 0
				pattern_dict[event_offset]=[tense, aspect, main_event]
				generalised_patt=''
				generalised_patt=tense+" "+aspect
				#all timexes are extracted and added to the pattern dictionary
				for timex in timexes:
					timex_attr = timex.attrib
					type = timex_attr['type']
					#value = timex_attr['value']
					value=''
					timex_string = timex_attr['text']
					offset = int(timex_attr['offset'])
					pattern_dict[offset]=[type, value, timex_string]
					
				pattern=''
				text_pattern=''
				for key in sorted(pattern_dict):
					value = pattern_dict[key]
					pattern+=value[0]+' '+value[1]+' '
					text_pattern+=value[2]+' '
				#write to file identified patterns in each sentence 
				output += episode_number+" "+pattern+'\n'+text_pattern+'\n'+sentence+'\n\n'
				episode_dict.setdefault(episode_number, []).append([pattern, text_pattern, sentence, generalised_patt])
				
				#reducing sentences in the episode to one (e.g. identifying the "main" event in the episode)
				for key in episode_dict:
					sentences = episode_dict[key]
					if len(sentences) == 1: #if there's only one sentence in the episode, it's the main event
						episode_dict_reduced[key] = sentences[0]
					else: 
						for sent in sentences:
							#the first sentence in order without "non-existent event" is the main event
							if sent[0] != "Not available Not available ": 
								episode_dict_reduced[key] = sent
								break
				
		#print(episode_dict)
		#print(episode_dict_reduced)
		#print(episode_sequence)
		f=open('results\\'+filename+'_patt_per_sentence.txt', 'w+')
		#f.write(output)
		out=''
		#text representation of the episode sequence
		timeline='&#8594;'.join(episode_sequence)
		if timeline not in timeline_rules:
			timeline_rules[timeline]={}
		
		#building pairwise patterns
		for i in range(0,5):
			current_epis = episode_sequence[i]
			next_epis = episode_sequence[i+1]
			current_patt=episode_dict_reduced[current_epis]
			next_patt=episode_dict_reduced[next_epis]
			out+=current_epis+" &#8594; "+next_epis+'\n'+current_patt[0]+" &#8594; "+next_patt[0]+'\n'+current_patt[1]+" &#8594; "+next_patt[1]+'\n'+current_patt[2]+" &#8594; "+next_patt[2]+'\n\n'
			
			#Creating pairwise patterns throughout the corpus
			ep_seq = current_epis+" &#8594; "+next_epis
			general_patt = current_patt[3]+" &#8594; "+next_patt[3] #PAST NONE  --> PAST NONE
			timeml_pattern = current_patt[0]+" &#8594; "+next_patt[0] #PAST NONE TIME  --> PAST NONE
			tokens = current_patt[1]+" &#8594; "+next_patt[1]
			sent = current_patt[2]+" &#8594; "+next_patt[2]
			if ep_seq not in patt_dict_overall:
				patt_dict_overall[ep_seq]={}
				#print(patt_dict_overall[ep_seq])
			if general_patt not in patt_dict_overall[ep_seq]:
				patt_dict_overall[ep_seq][general_patt]={}
				
			if timeml_pattern not in patt_dict_overall[ep_seq][general_patt]:
				patt_dict_overall[ep_seq].setdefault(general_patt, {}).update({timeml_pattern:[1, tokens, sent, filename]})
			else:
				patt_dict_overall[ep_seq][general_patt][timeml_pattern][0]+=1
				patt_dict_overall[ep_seq][general_patt][timeml_pattern].extend((tokens, sent, filename))
				
			#Creating patterns depending on the timeline
			if ep_seq not in timeline_rules[timeline]:
				timeline_rules[timeline][ep_seq]={}
			if general_patt not in timeline_rules[timeline][ep_seq]:
				timeline_rules[timeline][ep_seq][general_patt]={}
				
			if timeml_pattern not in timeline_rules[timeline][ep_seq][general_patt]:
				timeline_rules[timeline][ep_seq].setdefault(general_patt, {}).update({timeml_pattern:[1, tokens, sent, filename]})
			else:
				timeline_rules[timeline][ep_seq][general_patt][timeml_pattern][0]+=1
				timeline_rules[timeline][ep_seq][general_patt][timeml_pattern].extend((tokens, sent, filename))
			
		f1=open('results\\'+filename+'_patt_pairwised.txt', 'w+')
		f1.write(out)
	count = 0	
	out_all=''
	for epis_seq in sorted(patt_dict_overall):
		out_all+='\n'+epis_seq+"\n===============\n"
		gener_patterns=patt_dict_overall[epis_seq]
		for gener_patt in sorted(gener_patterns, key=lambda gener_patt: sum(value[0] for value in gener_patterns[gener_patt].values()), reverse=True): #sorted by frequency
			freq_patterns = gener_patterns[gener_patt]
			number = sum(value[0] for value in freq_patterns.values())
			out_all+="General pattern: "+gener_patt+" "+str(number)+'\n---------\n'
			for key in sorted(freq_patterns, key=freq_patterns.get, reverse=True):
				examples = '\n'.join(freq_patterns[key][1:])
				out_all+=key + " "+str(freq_patterns[key][0])+"\n"+examples+'\n\n'
				#count+=freq_patterns[key]
			out_all+='\n\n'
			
	f_all=open('all_patterns.txt', 'w+')
	f_all.write(out_all)
	
	out_tl=''
	for tl in sorted(timeline_rules):
		out_tl+='\n'+tl+'\n=====================\n'
		episodes = timeline_rules[tl]		
		for epis_seq in sorted(episodes):
			out_tl+='\n'+epis_seq+"\n=============\n"
			gener_patterns=episodes[epis_seq]
			for gener_patt in sorted(gener_patterns, key=lambda gener_patt: sum(value[0] for value in gener_patterns[gener_patt].values()), reverse=True): #sorted by frequency
				freq_patterns = gener_patterns[gener_patt]
				number = sum(value[0] for value in freq_patterns.values())
				out_tl+="General pattern: "+gener_patt+" "+str(number)+'\n---------\n'
				for key in sorted(freq_patterns, key=freq_patterns.get, reverse=True):
					examples = '\n'.join(freq_patterns[key][1:])
					out_tl+=key + " "+str(freq_patterns[key][0])+"\n"+examples+'\n\n'
					count+=freq_patterns[key][0]
				out_tl+='\n\n'
	
	f_all=open('timeline_patterns.txt', 'w+')
	f_all.write(out_tl)
	#html_out(timeline_rules)
	#html_episodes(patt_dict_overall, '<body>\n<div id="generated-toc"></div>')
	print(count)
	return timeline_rules
	
"""This function calculates the conditional probabilities of tenses in each episode
	and stores them in the following data structure:
	{timeline1: [{"0":{tense1:P, tense2:P, ...}},
				 {cond_tense1:{tense1:P, tense2:P, ...}, cond_tense2:{...}, ...},
				 {...}, 
				 {...},
				 {...},
				 {...}],
	 timeline2: [...], ...}, where
	each timeline maps to the list of six dictionaries, representing six episodes;
	"0" stands for the condition 'to be in the first episode';
	each dictionary has the condition as a key, which maps to several tenses with their probabilities P
	e.g. {Past None:{Infinitive None:0.2, Present None:0.8},...},
	which should be read as P(Infinitive None|Past None)=0.2, P(Present None|Past None)=0.8.
	
	Input: output of the function get_patterns
	Output: dictionary with conditional probabilities described above
"""
def create_cond_prob_data_structure():
	timeline_rules = get_patterns()
	cond_prob_all_timelines={}
	for sequence in sorted(timeline_rules):
		trans_count=0
		trie = []
		for transition in sorted(timeline_rules[sequence]):
			trans_count += 1
			number_of_stories = 0
			condit_prob = {}
			initial_prob = {}
			tense_freq = 0
			for pattern in sorted(timeline_rules[sequence][transition]):
				actual_tense = pattern.split(" &#8594; ")[1]
				condition_tense = pattern.split(" &#8594; ")[0]
				tense_freq = sum(item[0] for item in timeline_rules[sequence][transition][pattern].values())
				
				if trans_count == 1:
				#calculating probabilities at the beginning of a story: e.g. P(tense|0)
					initial_prob.setdefault("0", {})
					#if tense already in dict, implement its value
					initial_prob["0"][condition_tense] = initial_prob["0"].get(condition_tense, 0) + tense_freq
					
				#calculating conditional probabilities: e.g. P(tense|tense)
				condit_prob.setdefault(condition_tense, {}).update({actual_tense:tense_freq})
				number_of_stories += tense_freq
			#print(number_of_stories)
			#replacing frequencies by probabilities (i.e. dividing by the number of stories)
			for tense in condit_prob:
				for key in condit_prob[tense]:
					condit_prob[tense][key]=round(condit_prob[tense][key]/number_of_stories, 5)
				
			if initial_prob != {}:
				for tense in initial_prob:
					for key in initial_prob[tense]:
						initial_prob[tense][key]=round(initial_prob[tense][key]/number_of_stories, 5)
				trie+=[initial_prob]
				
			trie+=[condit_prob]
			
		cond_prob_all_timelines[sequence.replace("&#8594;","_")]=trie
	print(cond_prob_all_timelines)
	return cond_prob_all_timelines
	

create_cond_prob_data_structure()

def probability_model():
	cond_prob = create_cond_prob_data_structure()
	res={}
	for timeline in cond_prob:
		res[timeline]=f(timeline)
	return res
	
def calculate_paths():
	

def html_out(tl_rules):
	ex_count = 0
	out_tl=''
	out_tl+='<body>\n<div id="generated-toc"></div><h1>Patterns</h1>'
	for tl in sorted(tl_rules):
		out_tl+='<h2>Timeline: '+tl+'</h2>\n'
		episodes = tl_rules[tl]
		out_tl, ex_count = html_episodes(episodes, out_tl, ex_count)
		
	f_all=open('timeline_patterns.html', 'w+')
	f_all.write(out_tl)
		
def html_episodes(patt_dict, out_tl, ex_count=0):
	
	for epis_seq in sorted(patt_dict):
		out_tl+='\n<h3>Episode transition: '+epis_seq+'</h3><ol type="I">\n'
		gener_patterns=patt_dict[epis_seq]
		for gener_patt in sorted(gener_patterns, key=lambda gener_patt: sum(value[0] for value in gener_patterns[gener_patt].values()), reverse=True): #sorted by frequency
			freq_patterns = gener_patterns[gener_patt]
			number = sum(value[0] for value in freq_patterns.values())
			out_tl+='<li>'+gener_patt+"<br> Freq: "+str(number)+'<br></li><ol type="1">\n'
			for key in sorted(freq_patterns, key=freq_patterns.get, reverse=True):
				#examples = '\n'.join(freq_patterns[key][1:])
				out_tl+='<li>'+key + "<br> Freq: "+str(freq_patterns[key][0])+'<br><a href="#" id="example'+str(ex_count)+'-show" class="showLink" onclick="showHide(\'example'+str(ex_count)+'\');return false;">See examples</a></li>\n\n'
				out_tl+='<div id="example'+str(ex_count)+'" class="more"> <ul>\n'
				example = freq_patterns[key][1:]
				for i in range(0, len(example), 3):						
					out_tl+='<li><ul style="list-style-type:none"><li>'+example[i]+'</li>\n<li>'+example[i+1]+'<li><a href="html_stories/'+example[i+2]+'.html">Original text</a></li></ul></li><br>'
					
				out_tl+='</ul><p><a href="#" id="example'+str(ex_count)+'-hide" class="hideLink" onclick="showHide(\'example'+str(ex_count)+'\');return false;">Hide examples</a></p></div>\n\n'
				ex_count+=1
				
			out_tl+='</ol>\n'
		out_tl+='</ol>\n'
	out_tl+='</body></html>'
	
	f_all=open('patterns_all.html', 'w+')
	f_all.write(out_tl)
	return out_tl, ex_count
	
xml_file = 'output.xml'
original_xml_file = 'narrative_corpus_story_per_file-dir.info.xml'
#get_stories()
#get_patterns(xml_file)
#add_episodes(original_xml_file)