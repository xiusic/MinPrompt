#!/usr/bin/env python
# coding: utf-8

# In[1]:


import string
import copy
import unicodedata
import statistics
import re
import pickle

# from nltk.corpus import stopwords as nltk_stopwords
from utils import SpacyMagic
from itertools import combinations

STOPWORDS = {'ourselves', 'hers', 'between', 'yourself', 'but', 'again', 'there', 'about', 'once', 'during', 'out', 'very', 'having', 'with', 'they', 'own', 'an', 'be', 'some', 'for', 'do', 'its', 'yours', 'such', 'into', 'of', 'most', 'itself', 'other', 'off', 'is', 's', 'am', 'or', 'who', 'as', 'from', 'him', 'each', 'the', 'themselves', 'until', 'below', 'are', 'we', 'these', 'your', 'his', 'through', 'don', 'nor', 'me', 'were', 'her', 'more', 'himself', 'this', 'down', 'should', 'our', 'their', 'while', 'above', 'both', 'up', 'to', 'ours', 'had', 'she', 'all', 'no', 'when', 'at', 'any', 'before', 'them', 'same', 'and', 'been', 'have', 'in', 'will', 'on', 'does', 'yourselves', 'then', 'that', 'because', 'what', 'over', 'why', 'so', 'can', 'did', 'not', 'now', 'under', 'he', 'you', 'herself', 'has', 'just', 'where', 'too', 'only', 'myself', 'which', 'those', 'i', 'after', 'few', 'whom', 't', 'being', 'if', 'theirs', 'my', 'against', 'a', 'by', 'doing', 'it', 'how', 'further', 'was', 'here', 'than'}
PUNCTS = set(string.punctuation)
DISCARD_WORD_SET = STOPWORDS | PUNCTS | set([''])
ULIM_CHAR_PER_SENTENCE = 500

class TextPreprocessor:
    def __init__(self):
        pass

    @staticmethod
    def get_phrases(*, entities, noun_chunks):
        """
        :param entities: list of pairs (ent_str, ent_category)
        :param noun_chunks: list of pairs
        """
        phrases = copy.deepcopy(entities)

        ent_str_set = set([ent_str.lower() for ent_str, _ in entities])
        discard_set = ent_str_set | STOPWORDS

        for nc in noun_chunks:
            nc_str, _ = nc  # ensure it's in the correct format (i.e. pairs)
            nc_str_lower = nc_str.lower()

            if nc_str_lower not in discard_set:
                phrases.append(nc)

        return phrases

    @staticmethod
    def unicode_normalize(text):
        """
        Resolve different type of unicode encodings.

        e.g. unicodedata.normalize('NFKD', '\u00A0') will return ' '
        """
        return unicodedata.normalize('NFKD', text)

    @classmethod
    def clean_and_tokenize_str(cls, s):
        tokens = set(re.split(r'\W+', s.lower()))
        tokens = tokens - DISCARD_WORD_SET
        return tokens

    def findall_substr(self, substr, full_str):
        """
        Respect word boundaries
        """
        return re.findall(r'\b{}\b'.format(re.escape(substr)), full_str)

    def is_similar(self, sent1, sent2, f1_cutoff, *, discard_stopwords):
        """
        Based on bag of words.

        :param discard_stopwords: remove stopwords and lowercasing
        """
        if sent1 == sent2:
            return True

        if discard_stopwords:
            tokens1 = self.clean_and_tokenize_str(sent1)
            tokens2 = self.clean_and_tokenize_str(sent2)
        else:
            tokens1 = set(sent1.strip().split())
            tokens2 = set(sent2.strip().split())

        eps = 1e-100

        score1 = float(len(tokens1 & tokens2)) / (len(tokens1) + eps)
        score2 = float(len(tokens1 & tokens2)) / (len(tokens2) + eps)

        f1 = statistics.harmonic_mean([score1, score2])
        return f1 >= f1_cutoff

    def word_tokenize(self, raw_text):
        nlp = SpacyMagic.load_en_disable_all()
        return [w.text for w in nlp(raw_text)]

    def compute_ner(self, text):
        """
        :return: e.g. [('today', 'DATE'), ('Patrick', 'PERSON')]
        """
        nlp = SpacyMagic.load('my_english_ner', 'en_core_web_sm', disable=['tagger', 'parser'])
        ents = [(ent.text, ent.label_) for ent in nlp(text).ents]
        return sorted(set(ents))

    def compute_ner_and_noun_chunks(self, text):
        """
        https://spacy.io/usage/linguistic-features#noun-chunks

        ents: [('today', 'DATE'), ('Patrick', 'PERSON')]
        noun_chunks: e.g. [('Autonomous cars', 'nsubj'), ('insurance liability', 'dobj')]

        :return: (ents, noun_chunks)
        """
        if len(text) > ULIM_CHAR_PER_SENTENCE:
            return [], []

        # spacy has memory leaks: https://github.com/explosion/spaCy/issues/3618
        nlp = SpacyMagic.load('my_english', 'en_core_web_sm', disable=[])
        doc = nlp(text)

        ents = [(ent.text, ent.label_) for ent in doc.ents]
        chunks = [(nc.text, nc.root.dep_) for nc in doc.noun_chunks]

        ents = sorted(set(ents))
        chunks = sorted(set(chunks))

        return ents, chunks

    def normalize_basic(self, text):
        """
        :param text: tokenized text string
        """
        tokens = [w for w in text.lower().split() if w not in DISCARD_WORD_SET]
        return ' ' . join(tokens)

    def sent_tokenize(self, raw_text, title):
        """
        :return: a list of ...
        """
        # There are different types of sentence segmentation. See
        # https://spacy.io/usage/linguistic-features#sbd for more details
        # The sentencizer is much faster, but not as good as DependencyParser
        # Alternatively, nlp = SpacyMagic.load('en_core_web_sm')  # using DependencyParser
        nlp = SpacyMagic.load_en_sentencizer()

        text_lst = re.split(r'[\n\r]+', raw_text)
        if title and text_lst[0] == title:
            # remove the first element if is the same as the title
            text_lst = text_lst[1:]

        sentences_agg = []
        for text in text_lst:
            doc = nlp(text)
            sentences = [sent.string.strip() for sent in doc.sents]
            sentences_agg.extend(sentences)
        return sentences_agg


# In[2]:



from operator import add

# from text_preprocessor import TextPreprocessor


class SquadNerCreatorError(Exception):
    pass


"""
{
    "qid": "57277c965951b619008f8b2b",
    "question": "What do people engage in after they've disguised themselves?",
    "context": "In Greece Carnival is also ...",
    "answers": [
        {
            "answer_start": 677,
            "ner_category": "SOME_CATEGORY",
            "text": "pranks and revelry"
        }
    ],
    "article_title": "Carnival"
}
"""

class SquadNerCreator:

    def __init__(self, output_dir, *, debug_save, num_partitions):
        self.output_dir = output_dir
        self.debug_save = debug_save
        self.num_partitions = num_partitions

        self.text_preprocessor = TextPreprocessor()

    def _process_row(self, question):
        """
        1. Run NER on `context` field
        2. Set `ner_category` for answers
        3. Discard answers that are not entities
        """

        # this returns a list, e.g. [('today', 'DATE'), ('Patrick', 'PERSON')]
        ent_lst = self.text_preprocessor.compute_ner(question)

        ent_dct = {}
        for ent_str, ner_category in ent_lst:
            # perform uncasing
            ent_dct[ent_str.lower()] = ner_category

#         new_answers = []
#         for ans_struct in question.answers:
#             ans_text = ans_struct['text'].lower()  # perform uncasing
#             if ans_text not in ent_dct:
#                 continue

#             ans_struct['ner_category'] = ent_dct[ans_text]
#             new_answers.append(ans_struct)

#         question.answers = new_answers
        return question, ent_lst

    def _print_output_stats(self, question_rdd, metric_fptr):
        print('Count of new question_rdd: {}'.format(question_rdd.count()), file=metric_fptr)

        ner_category_counts = question_rdd.map(
            lambda x: (x.answers[0]['ner_category'], 1)
        ).reduceByKey(add).collectAsMap()

        total_count = sum(ner_category_counts.values())
        print(file=metric_fptr)
        for ner_category, count in sorted(ner_category_counts.items()):
            print('Number of samples with NER category "{}": {} / {} ({:.2f}%)'.format(
                ner_category,
                count,
                total_count,
                100.0 * count / total_count), file=metric_fptr)



    def run_job(self, sc, question_rdd, metric_fptr):
        print('Count of original question_rdd: {}'.format(question_rdd.count()), file=metric_fptr)

        question_rdd = question_rdd.map(lambda x: self._process_row(x)).filter(lambda x: len(x.answers) >= 1)

        final_output_dir = self.output_dir  # normally, we do os.path.join(self.output_dir, <...>)
        question_rdd.map(lambda x: x.jsonify()).saveAsTextFile(final_output_dir)

        self._print_output_stats(question_rdd, metric_fptr)


# In[3]:


import json
from tqdm import tqdm

contexts = []
with open("~/ptQA/mrqa-few-shot/squad/train-v2.0.json", "r") as fin, open("~/ptQA/mrqa-few-shot/squad/train-v2.0-onlycontext.txt", "w", encoding='utf8') as fout:
    line = fin.readline()
    line = json.loads(line)
    data = line['data']

    for wiki in data:
        paragraphs = wiki['paragraphs']
        for paragraph in paragraphs:
            context = paragraph['context']
#             print(context)
            fout.write(context + '\n')
            contexts.append(context)


# In[4]:


# contexts = \
# ["""Beyoncé Giselle Knowles-Carter (/biːˈjɒnseɪ/ bee-YON-say) (born September 4, 1981) is an American singer
# , songwriter, record producer and actress. Born and raised in Houston, Texas, she performed in various singi
# ng and dancing competitions as a child, and rose to fame in the late 1990s as lead singer of R&B girl-group
# Destiny's Child. Managed by her father, Mathew Knowles, the group became one of the world's best-selling girl
# groups of all time. Their hiatus saw the release of Beyoncé's debut album, Dangerously in Love (2003), which 
# established her as a solo artist worldwide, earned five Grammy Awards and featured the Billboard Hot 100 
# number-one singles \"Crazy in Love\" and \"Baby Boy\".""",\
# """Following the disbandment of Destiny's Child in June 2005, she released her second solo album, B'Day (2006), 
#  which contained hits \"Déjà Vu\", \"Irreplaceable\", and \"Beautiful Liar\". Beyoncé also ventured into acti
#  ng, with a Golden Globe-nominated performance in Dreamgirls (2006), and starring roles in The Pink Panther (2006)
#  and Obsessed (2009). Her marriage to rapper Jay Z and portrayal of Etta James in Cadillac Records (2008) 
#  influenced her third album, I Am... Sasha Fierce (2008), which saw the birth of her alter-ego Sasha Fierce 
#  and earned a record-setting six Grammy Awards in 2010, including Song of the Year for \"Single Ladies (Put a
#  Ring on It)\". Beyoncé took a hiatus from music in 2010 and took over management of her career; her fourth 
#  album 4 (2011) was subsequently mellower in tone, exploring 1970s funk, 1980s pop, and 1990s soul. Her 
#  critically acclaimed fifth studio album, Beyoncé (2013), was distinguished from previous releases by its 
#  experimental production and exploration of darker themes.""",\
# """A self-described \"modern-day feminist\", Beyoncé creates songs that are often characterized by themes of love, relationships, and monogamy, as well as female sexuality and empowerment. On sta
# ge, her dynamic, highly choreographed performances have led to critics hailing her as one of the best entertainers in contemporary popular music. Throughout a career spanning 19 years, she has sold over 118 million
#  records as a solo artist, and a further 60 million with Destiny's Child, making her one of the best-selling music artists of all time. She has won 20 Grammy Awards and is the most nominated woman in the award's hi
# story. The Recording Industry Association of America recognized her as the Top Certified Artist in America during the 2000s decade. In 2009, Billboard named her the Top Radio Songs Artist of the Decade, the Top Fem
# ale Artist of the 2000s and their Artist of the Millennium in 2011. Time listed her among the 100 most influential people in the world in 2013 and 2014. Forbes magazine also listed her as the most powerful female m
# usician of 2015."""]




d_zeroid2sentence = {}
d_sentence2zeroid = {}

d_zeroid2entity = {}
d_entity2zeroid = {}

d_sentence2context = {}

d_s2e = {}
d_e2s = {}
d_adj = {}
s_iso = set()
cnt_sentence = 0
snc = SquadNerCreator(output_dir = ".", num_partitions=1, debug_save = False)
for idx, context in enumerate(tqdm(contexts)):
    sentences = context.split('.')[:-1]
    for i, sentence in enumerate(sentences):
        if sentence:
            p, l_ent = snc._process_row(question = sentence)
            
            d_sentence2context[cnt_sentence] = context
            
            d_zeroid2sentence[cnt_sentence] = sentence
            d_sentence2zeroid[sentence] = cnt_sentence

            d_s2e[cnt_sentence] = l_ent

            for ent in l_ent:
                if d_e2s.get(ent):
                    d_e2s[ent].append(cnt_sentence)
                else:
                    d_e2s[ent] = [cnt_sentence]
            
            cnt_sentence += 1

len_sentences = len(d_s2e)
len_entities = len(d_e2s)
print(f"# sentences: {len_sentences}")
print(f"# entities: {len_entities}")

l_entity = sorted(d_e2s.keys())
for i, e in enumerate(l_entity):
    d_zeroid2entity[i] = e
    d_entity2zeroid[e] = i

for clique in d_e2s.values():
    if len(clique) >= 2:
        res = list(combinations(clique, 2))
#         print(res)
        for s, t in res:
            if d_adj.get(s):
                d_adj[s].append(t)
            else:
                d_adj[s] = [t]

            if d_adj.get(t):
                d_adj[t].append(s)
            else:
                d_adj[t] = [s]

for i in range(len_sentences):
    if i not in d_adj:
        s_iso.add(i)
        
print(f"# isolated vertices (sentences): {len(s_iso)}")
#     else:
#         print(clique)
#         s_iso.add(clique[0])
        # if an entity is merely covered by one sentence, then the sentence becomes a isolated vertex


# In[5]:


d_adj


# In[6]:


s_iso


# In[7]:


with open("~/ptQA/mrqa-few-shot/squad/adj_list.txt", "w") as fout:
    for i in range(len_sentences):
        if d_adj.get(i):
            output_line = ' '.join(map(str, d_adj[i])) + '\n'
            fout.write(output_line)
        else:
            output_line = '\n'
            fout.write(output_line)


# In[13]:


import pickle

with open("/efs/core-pecos/users/xiusi/minQG/mrqa-unsupervised/squad-d_sentence2zeroid.pkl", "wb") as fout:
    pickle.dump(d_sentence2zeroid, fout)
    
with open("/efs/core-pecos/users/xiusi/minQG/mrqa-unsupervised/squad-d_zeroid2sentence.pkl", "wb") as fout:
    pickle.dump(d_zeroid2sentence, fout)

with open("/efs/core-pecos/users/xiusi/minQG/mrqa-unsupervised/squad-d_entity2zeroid.pkl", "wb") as fout:
    pickle.dump(d_entity2zeroid, fout)

with open("/efs/core-pecos/users/xiusi/minQG/mrqa-unsupervised/squad-d_zeroid2entity.pkl", "wb") as fout:
    pickle.dump(d_zeroid2entity, fout)

with open("/efs/core-pecos/users/xiusi/minQG/mrqa-unsupervised/squad-d_e2s.pkl", "wb") as fout:
    pickle.dump(d_e2s, fout)
    
with open("/efs/core-pecos/users/xiusi/minQG/mrqa-unsupervised/squad-d_s2e.pkl", "wb") as fout:
    pickle.dump(d_s2e, fout)
    
with open("/efs/core-pecos/users/xiusi/minQG/mrqa-unsupervised/squad-d_adj.pkl", "wb") as fout:
    pickle.dump(d_adj, fout)
    
with open("/efs/core-pecos/users/xiusi/minQG/mrqa-unsupervised/squad-d_sentence2context.pkl", "wb") as fout:
    pickle.dump(d_sentence2context, fout)


# In[9]:


d_zeroid2sentence[0]


# In[10]:


d_s2e[0]


# In[11]:


len(d_sentence2context)


# In[12]:


d_sentence2context[0]


# In[ ]:




