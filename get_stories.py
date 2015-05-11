import os
import operator
import re
import xml.etree.ElementTree as etree
from nltk.stem import PorterStemmer


def get_stories():
	"""Create from the original corpus a number of files
	where each file represents a separate story.
	The name of the file reflects the sequence of episodes.
	Each file contains six lines, one for each episode.
	"""
	f = open("original_corpus_with_deletion_of_repetitions.txt", 'r')
	corpus = []
	iterator = 0
	for line in f:
		elements = line.split('\t')
		# corpus.append([e for e in elements])
		filename = (
			elements[0] + "_" + str(iterator) + "_" + elements[1] +
			"_" + elements[3] + "_" + elements[5] + "_" + elements[7] +
			"_" + elements[9] + "_" + elements[11] + ".txt"
			)
		story = open("narrative_corpus_story_per_file\\" + filename, 'w+')

		# Check if line ends with a punct mark. If not, add a full stop
		end_of_sent = ['.', '?', '!', '"']
		for i in range(2, 14, 2):
			if elements[i].rstrip()[-1] not in end_of_sent:
				elements[i] = elements[i].rstrip() + '.'

		story.write(elements[2] + "\n" + elements[4] + "\n" + elements[6] + "\n" + elements[8] + "\n" + elements[10] + "\n" + elements[12] + "\n")
		iterator += 1
	# print(len(corpus))


def add_episodes(orig_xmlfile):
	"""Take a generated xml file by CAEVO as input and adds the episode number to each sentence, i.e. to which episode the sentence belongs to in the original corpus. The episode number is added as a tag, e.g. "<episode number="1" />" at the end of each entry.
	Output: the file named "output.xml".
	"""
	etree.register_namespace("", "http://chambers.com/corpusinfo")
	tree = etree.parse(orig_xmlfile)
	root = tree.getroot()
	error = 0
	sentence_count = 0
	out_test = ''
	for file in root:
		# Get the filename from xml
		filename = file.attrib['name']
		episode_sequence = filename[:-4].split('_')

		# Open the original txt file
		input_file = open('narrative_corpus_story_per_file\\' + filename, 'r')
		input_file_list = [line.rstrip() for line in input_file]
		for entry in file:
			if entry.tag == "{http://chambers.com/corpusinfo}entry":
				sentence = entry[0].text
				# Remove parentheses
				sentence = sentence.replace("-LRB-", "")
				sentence = sentence.replace("-RRB-", "")

				sentence_count += 1
				# Check which episode this sentence belongs to
				line_num = 0
				for line in input_file_list:
					separ_line = line.replace("n't", " n't")  # separate negative particles
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
					line_list = re.findall(r"[\w]+", separ_line)  # separate words and punct signs

					sentence_list = re.findall(r"[\w]+", sentence)

					if set(sentence_list).issubset(line_list):
						etree.SubElement(entry, '{http://chambers.com/corpusinfo}episode', attrib={'number': str(episode_sequence[line_num + 2])})
						break
					line_num += 1
				'''# uncomment this to check if all episodes were added
				sent_elements = ''

				if entry.find('{http://chambers.com/corpusinfo}episode') == None:
					error += 1
					for element in sentence_list:
						sent_elements += element + " "
					out_test += filename + '\n' + sent_elements + '\n' + sentence + '\n\n'

	file_test = open('bugs.txt', "w+")
	file_test.write(out_test)
	print(error)
	print(sentence_count)'''
	tree.write('output.xml')


def get_patterns(xmlfile='output.xml'):
	tree = etree.parse(xmlfile)
	root = tree.getroot()
	patt_dict_overall = {}
	timeline_rules = {}
	count_ep = 0  # number of episodes with more than 1 sent
	count_1sent = 0  # number of episodes with 1 sent
	count_NANA = 0  # number of key events with NANA
	for file in root:
		output = ''
		episode_dict = {}
		episode_dict_reduced = {}
		episode_sequence = []
		filename = file.attrib['name'][:-4]
		story = filename[0]
		for entry in file:
			if entry.tag == "{http://chambers.com/corpusinfo}entry":
				sentence = entry[0].text
				deps = entry[3].text
				events = entry[4]
				timexes = entry[5]
				episode_number = entry[6].attrib['number']

				# tracking the episode sequence
				if episode_number not in episode_sequence:
					episode_sequence += [episode_number]

				# Sentence consists of one word.
				if not deps:
					continue

				# Find root of the parse tree
				p = re.compile(r'root\(ROOT-0, (.+)-(\d+)\)')
				pattern = p.findall(deps)
				# print(pattern)
				dep_root = pattern[0][0]
				event_offset = int(pattern[0][1])

				# store event and timexes in a dict with their offset as a key,
				# e.g. {2:['PAST', 'NONE', 'wondered'], 21:['DURATION', 'PT1H', 'an hour']}
				pattern_dict = {}

				# Identifying an event in a sentence: the root of a sentence tree must be listed in events
				main_event = ''
				for event in events:
					event_attr = event.attrib
					if event_attr['string'] == dep_root:
						main_event = dep_root
						tense = event_attr['tense']
						aspect = event_attr['aspect']
						event_offset = int(event_attr['offset'])
						break

				# If not found, take the first event
				if main_event == '':
					if events.text:
						event_attr = events[0].attrib
						main_event = event_attr['string']
						tense = event_attr['tense']
						aspect = event_attr['aspect']
						event_offset = int(event_attr['offset'])
				# otherwise, the sentence root is an event without the attributes
					else:
						main_event = dep_root
						tense = 'Not available'
						aspect = 'Not available'
						# event_offset = 0
				pattern_dict[event_offset] = [tense, aspect, main_event]
				generalised_patt = ''
				generalised_patt = tense + " " + aspect
				# all timexes are extracted and added to the pattern dictionary
				for timex in timexes:
					timex_attr = timex.attrib
					type = timex_attr['type']
					# value = timex_attr['value']
					value = ''
					timex_string = timex_attr['text']
					offset = int(timex_attr['offset'])
					pattern_dict[offset] = [type, value, timex_string]

				pattern = ''
				text_pattern = ''
				for key in sorted(pattern_dict):
					value = pattern_dict[key]
					pattern += value[0] + ' ' + value[1] + ' '
					text_pattern += value[2] + ' '
				# write to file identified patterns in each sentence
				output += episode_number + " " + pattern + '\n' + text_pattern + '\n' + sentence + '\n\n'
				episode_dict.setdefault(episode_number, []).append([pattern, text_pattern, sentence, generalised_patt])

		# reducing sentences in the episode to one (e.g. identifying the key event in the episode)
		for key in episode_dict:
			sentences = episode_dict[key]
			if len(sentences) == 1:  # if there's only one sentence in the episode, it's the key event
				count_1sent += 1
				episode_dict_reduced[key] = sentences[0]
				if sentences[0][0] == "Not available Not available ":
					count_NANA += 1
			else:
				count_ep += 1
				key_event_number = identify_key_event(story, key, sentences)
				episode_dict_reduced[key] = sentences[key_event_number]
				if sentences[key_event_number][0] == "Not available Not available ":
					count_NANA += 1
		# print(episode_dict)
		# print(episode_dict_reduced)
		
		#with open('results\\' + filename + '_patt_per_sentence.txt', 'w+') as f:
		#	f.write(output)
		out = ''
		# text representation of the episode sequence
		timeline = '&#8594;'.join(episode_sequence)
		if timeline not in timeline_rules:
			timeline_rules[timeline] = {}
		# building pairwise patterns
		for i in range(0, 5):
			current_epis = episode_sequence[i]
			next_epis = episode_sequence[i + 1]
			current_patt = episode_dict_reduced[current_epis]
			next_patt = episode_dict_reduced[next_epis]
			out += current_epis + " &#8594; " + next_epis + '\n' + current_patt[0] + " &#8594; " + next_patt[0] + '\n' + current_patt[1] + " &#8594; " + next_patt[1] + '\n' + current_patt[2] + " &#8594; " + next_patt[2] + '\n\n'

			# Creating pairwise patterns throughout the corpus
			ep_seq = current_epis + " &#8594; " + next_epis
			general_patt = current_patt[3] + " &#8594; " + next_patt[3]  # PAST NONE  --> PAST NONE
			timeml_pattern = current_patt[0] + " &#8594; " + next_patt[0]  # PAST NONE TIME  --> PAST NONE
			tokens = current_patt[1] + " &#8594; " + next_patt[1]
			sent = current_patt[2] + " &#8594; " + next_patt[2]
			if ep_seq not in patt_dict_overall:
				patt_dict_overall[ep_seq] = {}
				# print(patt_dict_overall[ep_seq])
			if general_patt not in patt_dict_overall[ep_seq]:
				patt_dict_overall[ep_seq][general_patt] = {}

			if timeml_pattern not in patt_dict_overall[ep_seq][general_patt]:
				patt_dict_overall[ep_seq].setdefault(general_patt, {}).update({timeml_pattern: [1, tokens, sent, filename]})
			else:
				patt_dict_overall[ep_seq][general_patt][timeml_pattern][0] += 1
				patt_dict_overall[ep_seq][general_patt][timeml_pattern].extend((tokens, sent, filename))

			# Creating patterns depending on the timeline
			if ep_seq not in timeline_rules[timeline]:
				timeline_rules[timeline][ep_seq] = {}
			if general_patt not in timeline_rules[timeline][ep_seq]:
				timeline_rules[timeline][ep_seq][general_patt] = {}

			if timeml_pattern not in timeline_rules[timeline][ep_seq][general_patt]:
				timeline_rules[timeline][ep_seq].setdefault(general_patt, {}).update({timeml_pattern: [1, tokens, sent, filename]})
			else:
				timeline_rules[timeline][ep_seq][general_patt][timeml_pattern][0] += 1
				timeline_rules[timeline][ep_seq][general_patt][timeml_pattern].extend((tokens, sent, filename))

		#with open('results\\' + filename + '_patt_pairwised.txt', 'w+') as f1:
		#	f1.write(out)
		# Create the input for CRF++
		# crf_input(episode_dict_reduced, ''.join(episode_sequence))
		
	count = 0  # number of patterns
	out_all = ''
	for epis_seq in sorted(patt_dict_overall):
		out_all += '\n' + epis_seq + "\n===============\n"
		gener_patterns = patt_dict_overall[epis_seq]
		for gener_patt in sorted(gener_patterns, key=lambda gener_patt: sum(value[0] for value in gener_patterns[gener_patt].values()), reverse=True):  # sorted by frequency
			freq_patterns = gener_patterns[gener_patt]
			number = sum(value[0] for value in freq_patterns.values())
			out_all += "General pattern: " + gener_patt + " " + str(number) + '\n---------\n'
			for key in sorted(freq_patterns, key=freq_patterns.get, reverse=True):
				examples = '\n'.join(freq_patterns[key][1:])
				out_all += key + " " + str(freq_patterns[key][0]) + "\n" + examples + "\n\n"
				# count += freq_patterns[key]
			out_all += '\n\n'

	with open('all_patterns.txt', 'w+') as f_all:
		f_all.write(out_all)

	out_tl = ''
	for tl in sorted(timeline_rules):
		out_tl += '\n' + tl + '\n=====================\n'
		episodes = timeline_rules[tl]
		epis_sequence = tl.split('&#8594;')
		# sort by the episode sequence
		ep_layout_list = [epis_sequence[i] + ' &#8594; ' + epis_sequence[i+1] for i in range(len(epis_sequence)-1)]
		for epis_seq in sorted(episodes, key=lambda pair: ep_layout_list.index(pair)):
			out_tl += '\n' + epis_seq + "\n=============\n"
			gener_patterns = episodes[epis_seq]
			for gener_patt in sorted(gener_patterns, key=lambda gener_patt: sum(value[0] for value in gener_patterns[gener_patt].values()), reverse=True):  # sorted by frequency
				freq_patterns = gener_patterns[gener_patt]
				number = sum(value[0] for value in freq_patterns.values())
				out_tl += "General pattern: " + gener_patt + " " + str(number) + '\n---------\n'
				for key in sorted(freq_patterns, key=freq_patterns.get, reverse=True):
					examples = '\n'.join(freq_patterns[key][1:])
					out_tl += key + " " + str(freq_patterns[key][0]) + "\n" + examples + '\n\n'
					count += freq_patterns[key][0]
				out_tl += '\n\n'

	with open('timeline_patterns.txt', 'w+') as f_all:
		f_all.write(out_tl)

	print("Patterns: " + str(count))
	print("Episodes with > 1 sent: " + str(count_ep))
	print("Episodes with 1 sent: " + str(count_1sent))
	print("Key events with NA_NA: {}".format(count_NANA))
	return patt_dict_overall, timeline_rules

	
def get_captions():
	"""Read captions from file, store them in the dictionary with the story number as a key."""
	d = {}
	with open('captions.txt', 'r') as f:
		for line in f:
			story, *captions = line.split('\t')
			captions = [item.strip() for item in captions]
			d[story] = captions
	return d


captions = get_captions()
no_intersect = 0
unique_key = 0
mult_key = 0
first_sent = 0
stemmer = PorterStemmer()


def identify_key_event(story, episode, ep_sentences):
	"""Return the sequence number of the key event in the episode.
	Apply stemming to a story and the corresponding caption, find the word intersection.
	Input: story number, episode number, ep_sentences, where
	ep_sentences = [[pattern, text_pattern, sentence, generalised_patt], ...].
	"""
	global captions
	global no_intersect
	global unique_key
	global mult_key
	global first_sent
	caption = captions[story][int(episode) - 1]
	out = ''
	sent_intersect = {}
	count = 0

	# apply stemming
	for index, sentence in enumerate(ep_sentences):
		sent = sentence[2]
		caption_stem = [stemmer.stem(word) for word in caption.split()]  # stem captions
		sent_stem = [stemmer.stem(word) for word in sent.split()]  # stem sentences
		intersect = set(sent_stem).intersection(caption_stem)
		stopwords = ["and", "is", "at", "in", "of", "the", "a", "are", "to", "as", "be", "with"]
		intersect_clean = [word for word in intersect if word not in stopwords]  # remove stopwords
		if intersect_clean:
			out += episode + " " + sent + '\n' + caption + '\n' + " ".join(intersect_clean) + ' STEM\n'
			count += 1
			sent_intersect[index] = intersect_clean
		else:
			out += episode + " " + sent + '\n'
			
	out += '\n'
	'''with open('key_sentence.txt', 'a') as f:
		f.write(out)'''
		
	if count == 0:	
		no_intersect += 1
		# return the first sentence of the episode
		ep_number = 0
	elif count == 1:
		unique_key += 1
		# only one key event found, return it
		ep_number = list(sent_intersect.keys())[0]
		
		# check if the key event belongs to the 1st sentence of the episode
		if ep_number == 0:
			first_sent += 1

	else:
		mult_key += 1
		# compare key events, return with the largest amount of words in the intersection
		# if equal, return the first sent in the order
		for key, value in sent_intersect.items():
			sent_intersect[key] = len(value)
		ep_number = max(sent_intersect, key=lambda k: (sent_intersect[k]))
	return ep_number
	

def create_cond_prob_data_structure():
	"""Calculate the conditional probabilities of tenses in each episode
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

		Input: second variable of the return of the function get_patterns, i.e. timeline_rules
		Output: dictionary with conditional probabilities described above
	"""
	timeline_rules = get_patterns()[1]
	cond_prob_all_timelines = {}
	for sequence in sorted(timeline_rules):
		trans_count = 0
		trie = []
		# sort by the episode sequence
		epis_sequence = sequence.split('&#8594;')
		ep_layout_list = [epis_sequence[i] + ' &#8594; ' + epis_sequence[i+1] for i in range(len(epis_sequence)-1)]
		for transition in sorted(timeline_rules[sequence], key=lambda pair: ep_layout_list.index(pair)):
			trans_count += 1
			number_of_stories = 0
			condit_prob = {}
			initial_prob = {}
			tense_freq = 0
			for pattern in sorted(timeline_rules[sequence][transition]):
				actual_tense = pattern.split(" &#8594; ")[1]
				condition_tense = pattern.split(" &#8594; ")[0]
				tense_freq = sum(item[0] for item in timeline_rules[sequence][transition][pattern].values())

				# calculating probabilities at the beginning of a story: e.g. P(tense|0)
				if trans_count == 1:
					initial_prob.setdefault("0", {})
					# if tense already in dict, implement its value
					initial_prob["0"][condition_tense] = initial_prob["0"].get(condition_tense, 0) + tense_freq

				# calculating conditional probabilities: e.g. P(tense|tense)
				condit_prob.setdefault(condition_tense, {}).update({actual_tense: tense_freq})
				number_of_stories += tense_freq
			# print(number_of_stories)
			# replacing frequencies by probabilities (i.e. dividing by the number of stories)
			for tense in condit_prob:
				for key in condit_prob[tense]:
					condit_prob[tense][key] = round(condit_prob[tense][key] / number_of_stories, 5)

			if initial_prob != {}:
				for tense in initial_prob:
					for key in initial_prob[tense]:
						initial_prob[tense][key] = round(initial_prob[tense][key] / number_of_stories, 5)
				trie += [initial_prob]

			trie += [condit_prob]

		cond_prob_all_timelines[sequence.replace("&#8594;", ">>>")] = trie
	# print(cond_prob_all_timelines['2>>>3>>>4>>>5>>>1>>>6'])
	return cond_prob_all_timelines


def calculate_paths(probs, episode, tense, path, input):
	"""This is the bloody recursive function.
	It is going through a trie created in create_cond_prob_data_structure() and
	builds paths of tenses with their probabilities.
	Input: trie, number of episode, current tense, current path, dictionary of paths with probabilities.
	(All elements except the first one are changing with each recursion step.)
	Output: the dictionary {path:P, path2:P, ...}, some paths could have less than six elements
	NB: the first tense in a path will be the same for all paths, as it was defined when passing parameters to the function at its first call.
	"""
	res = {}
	# initialization with the first episode
	if episode == 0:
		curr_prob = probs[episode]["0"][tense]
		res[tense] = curr_prob
		episode += 1
		path = tense
		return calculate_paths(probs, episode, tense, path, res)
	else:
		if tense in probs[episode]:  # for some tenses path of 6 cannot be constructed
			prob_dict = probs[episode][tense]
			if episode != 5:
				for tense_cases in prob_dict:
					res[path + "||" + tense_cases] = round(input[path] * prob_dict[tense_cases], 7)
					path2 = path + "||" + tense_cases
					tense = tense_cases
					episode = len(path2.split("||"))
					res.update(calculate_paths(probs, episode, tense, path2, res))
			# recursion is finished with the last episode
			else:
				for tense_cases in prob_dict:
					res[path + "||" + tense_cases] = round(input[path] * prob_dict[tense_cases], 7)
	return res


def get_all_paths():
	"""Write to file sorted paths with their probabilities for each timeline."""
	cond_prob = create_cond_prob_data_structure()
	with open("conditional_prob.txt", "w+") as f_out:
		# calculate paths for each timeline
		for timeline in sorted(cond_prob):
			probabilities = cond_prob[timeline]
			result = {}
			# call recursive function for every starting tense, and update the dictionary
			for init_tenses in probabilities[0]["0"]:
				result.update(calculate_paths(probabilities, 0, init_tenses, init_tenses, {}))

			# write each path with its P to the file
			f_out.write(timeline + '\n')
			for key in sorted(result, key=result.get, reverse=True):  # sort by descending probabilities
				if len(key.split("||")) == 6:
					f_out.write(key + "\t" + str(result[key]) + "\n")
			f_out.write("\n\n")


f_crf = open('crf_data\\train_123456_adv_gener_withoutorder.txt', 'w+')


def crf_input(episode_d_red, timeline):
	global f_crf
	out = ''
	if timeline == '123456':
		for episode in timeline:
			tense_adv_l = []
			tense_adv = episode_d_red[episode][0].rstrip().replace('  ', ' ')
			tense_adv = tense_adv.replace('TIME', 'TIMEX')
			tense_adv = tense_adv.replace('DURATION', 'TIMEX')
			tense_adv = tense_adv.replace('SET', 'TIMEX')
			tense_adv = tense_adv.replace('DATE', 'TIMEX')
			tense_adv = tense_adv.replace(' ', '_')
			if 'TIMEX' in tense_adv.split('_'):
				tense_adv_l = [tense for tense in tense_adv.split('_') if tense != 'TIMEX']
				tense_adv_l.append('TIMEX')
			else:
				tense_adv_l = [tense for tense in tense_adv.split('_')]
			'''if tense_adv_l[0] == 'TIMEX':
				print(tense_adv)'''
			out += episode + '\t' + '_'.join(tense_adv_l) + '\n'
		out += '\n'
	f_crf.write(out)


def html_out(tl_rules, out_tl):
	ex_count = 0
		
	out_tl += '<h1>Patterns</h1>'
	for tl in sorted(tl_rules):
		out_tl += '<h2>Timeline: ' + tl + '</h2>\n'
		episodes = tl_rules[tl]
		out_tl, ex_count = html_episodes(episodes, out_tl, tl, ex_count)

	f_all = open('timeline_patterns.html', 'w+')
	f_all.write(out_tl)


def html_episodes(patt_dict, out_tl, timeline, ex_count=0):
	# uncomment this for the html_out function
	# epis_sequence = timeline.split('&#8594;')
	# ep_layout_list = [epis_sequence[i] + ' &#8594; ' + epis_sequence[i+1] for i in range(len(epis_sequence)-1)]
	for epis_seq in sorted(patt_dict):  # key=lambda pair: ep_layout_list.index(pair)
		out_tl += '\n<h3>Episode transition: ' + epis_seq + '</h3><ol type="I">\n'
		gener_patterns = patt_dict[epis_seq]
		for gener_patt in sorted(gener_patterns, key=lambda gener_patt: sum(value[0] for value in gener_patterns[gener_patt].values()), reverse=True):  # sorted by frequency
			freq_patterns = gener_patterns[gener_patt]
			number = sum(value[0] for value in freq_patterns.values())
			out_tl += '<li>' + gener_patt + "<br> Freq: " + str(number) + '<br></li><ol type="1">\n'
			for key in sorted(freq_patterns, key=freq_patterns.get, reverse=True):
				# examples = '\n'.join(freq_patterns[key][1:])
				out_tl += '<li>' + key + "<br> Freq: " + str(freq_patterns[key][0]) + '<br><a href="#" id="example' + str(ex_count) + '-show" class="showLink" onclick="showHide(\'example' + str(ex_count) + '\');return false;">See examples</a></li>\n\n'
				out_tl += '<div id="example' + str(ex_count) + '" class="more"> <ul>\n'
				example = freq_patterns[key][1:]
				for i in range(0, len(example), 3):
					out_tl += '<li><ul style="list-style-type:none"><li>' + example[i] + '</li>\n<li>' + example[i+1] + '<li><a href="html_stories/' + example[i+2] + '.html">Original text</a></li></ul></li><br>'

				out_tl += '</ul><p><a href="#" id="example' + str(ex_count) + '-hide" class="hideLink" onclick="showHide(\'example' + str(ex_count) + '\');return false;">Hide examples</a></p></div>\n\n'
				ex_count += 1

			out_tl += '</ol>\n'
		out_tl += '</ol>\n'
	out_tl += '</body></html>'

	f_all = open('patterns_all.html', 'w+')
	f_all.write(out_tl)
	return out_tl, ex_count

xml_file = 'output.xml'
original_xml_file = 'narrative_corpus_story_per_file-dir.info.xml'
# get_stories()
# add_episodes(original_xml_file)
# get_patterns(xml_file)
get_all_paths()

"""Generate html files."""
# add the head of the html-file
with open('head_html.txt', 'r') as f_head:
	html_head = ''.join(f_head)

# patt_dict_overall, timeline_rules = get_patterns(xml_file)
# html_out(timeline_rules, html_head)
# html_episodes(patt_dict_overall, html_head, timeline='')

print("No key events: {}".format(no_intersect))
print("One key events: {}".format(unique_key))
print("Multiple key events: {}".format(mult_key))
print("Key event is equal to the 1st sent in the episodes where one key event was detected: {}".format(first_sent))
f_crf.close()
