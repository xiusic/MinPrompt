#!/usr/bin/env python
# coding: utf-8

# In[8]:


from data_models import QuestionStyle
from text_preprocessor import TextPreprocessor
import re
import utils
from whxx_ngram_table import WhxxNgramTable
import spacy
from spacy.matcher import Matcher


class QuestionGeneratorError(Exception):
    pass


class QuestionGenerator:
    def __init__(self, whxx_ngram_table, text_preprocessor):
        self.whxx_ngram_table = whxx_ngram_table
        self.text_preprocessor = text_preprocessor

    def make_cloze_style(self, text, answer_str, mask):
        # replace ALL occurrences
        new_text = text.replace(answer_str, mask)

        if new_text == text:
            raise QuestionGeneratorError(
                'Failed to convert cloze style, did not replace anything with answer="{}": {}'.format(
                    answer_str, text))

        return new_text

    def _replace_with_question_mark_ending(self, text):
        return re.sub(r'\W*$', '?', text)

    def _post_questionify(self, text):
        text = text.strip()
        # capitalize() does not work here, because it lowercase the rest of the sentence
        text = text[0].upper() + text[1:]
        return self._replace_with_question_mark_ending(text)

    def _generate_template_awb(self, text, answer_str, sampled_ngram):
        """
        If cloze-style is “[FragmentA] [PERSON] [FragmentB]”, then:

        "[FragmentA], who [FragmentB]?" - AWB
        """

        # need to use \W+ to ensure word boundaries and not part of a word
        template = re.sub(
            r'\W+{}\W+'.format(re.escape(answer_str)),
            ', {} '.format(sampled_ngram),
            ' ' + text + ' ')

        # remove leading comma if the replacement was the first word
        template = re.sub(r'^,\s*', '', template)
        template = self._post_questionify(template)

        return template

    def _generate_template_wba(self, text, answer_str, sampled_ngram):
        """
        If cloze-style is “[FragmentA] [PERSON] [FragmentB]”, then:

        "Who [FragmentB] [FragmentA]?" - WBA
        """
        # need to use \W+ to ensure word boundaries and not part of a word
        template = re.sub(
            r'^(.*?)\W+{}\W+(.*?)\W*$'.format(re.escape(answer_str)),
            r'{} \2, \1'.format(sampled_ngram),
            ' ' + text + ' ')

        template = re.sub(r'\s+', ' ', template)  # regex above may have created double spaces
        template = self._post_questionify(template)

        # self._check_template(template, answer_str, text)

        return template

    def make_template_qg_styles(self, text, answer_str, ner_category, rng):
        if not self.text_preprocessor.findall_substr(answer_str, text):
            raise QuestionGeneratorError(
                'Failed to convert template QG style, answer="{}" not in question-text: {}'.format(
                    answer_str, text))

        sampled_ngram = self.whxx_ngram_table.rand_sample_ngram(rng.np, ner_category)

        styles = {
            QuestionStyle.TEMPLATE_AWB: self._generate_template_awb(text, answer_str, sampled_ngram),
            QuestionStyle.TEMPLATE_WBA: self._generate_template_wba(text, answer_str, sampled_ngram),
        }

        return styles


# In[9]:


with open("~/ptQA/unsupervised-qa/resources/whxx_ngram_table.toml") as fptr:
        whxx_ngram_table = WhxxNgramTable.import_from_toml(fptr)


tp = TextPreprocessor()
qg = QuestionGenerator(whxx_ngram_table, tp)


# In[10]:


rng = utils.RandomNumberGenerator()

sampled_ngram = qg.whxx_ngram_table.rand_sample_ngram(rng.np, "DATE")
qg._generate_template_wba("Beyoncé Giselle Knowles-Carter (/biːˈjɒnseɪ/ bee-YON-say) (born September 4, 1981) is an American singer, songwriter, record producer and actress", "September 4, 1981", sampled_ngram)


# In[20]:


import pickle

# with open("/efs/core-pecos/users/xiusi/minQG/mrqa-unsupervised/squad-d_sentence2zeroid.pkl", "wb") as fin:
#     d_sentence2zeroid = pickle.load(fin)
    
with open("~/ptQA/mrqa-unsupervised/naturalquestions-d_zeroid2sentence.pkl", "rb") as fin:
    d_zeroid2sentence = pickle.load(fin)

# with open("/efs/core-pecos/users/xiusi/minQG/mrqa-unsupervised/squad-d_entity2zeroid.pkl", "wb") as fin:
#     pickle.dump(d_entity2zeroid, fin)
    
# with open("/efs/core-pecos/users/xiusi/minQG/mrqa-unsupervised/squad-d_zeroid2entity.pkl", "wb") as fout:
#     pickle.dump(d_zeroid2entity, fin)

# with open("/efs/core-pecos/users/xiusi/minQG/mrqa-unsupervised/squad-d_e2s.pkl", "wb") as fin:
#     pickle.dump(d_e2s, fin)
    
with open("~/ptQA/mrqa-unsupervised/naturalquestions-d_s2e.pkl", "rb") as fin:
    d_s2e = pickle.load(fin)
    
# with open("/efs/core-pecos/users/xiusi/minQG/mrqa-unsupervised/squad-d_adj.pkl", "wb") as fin:
#     pickle.dump(d_adj, fin)

with open("~/ptQA/mrqa-unsupervised/naturalquestions-d_sentence2context.pkl", "rb") as fin:
    d_sentence2context = pickle.load(fin)


# In[21]:


import json
from tqdm import tqdm
s_id = []
with open("~/ptQA/mrqa-unsupervised/naturalquestions-DS-id.txt", "r") as s_retrived:
    ss = s_retrived.readlines()
    for s in ss:
        s_id = s.strip().split(",")
        
# IndependentNodes_id = []
# with open("~/ptQA/mrqa-unsupervised/hotpotqa-IndependentNodes-id.txt", "r") as s_retrived:
#     ss = s_retrived.readlines()
#     for s in ss:
#         IndependentNodes_id = s.strip().split(",")
        

        
s_id = list(map(int, s_id))
# IndependentNodes_id = list(map(int, IndependentNodes_id))
# s_id += IndependentNodes_id
nlp = spacy.load('en_core_web_sm')
matcher = Matcher(nlp.vocab)
aug_question_text = []
mask_token = " <mask> "
with open("~/ptQA/mrqa-unsupervised/aug_naturalquestions_DS_for_splinter.json", "w") as f_train_X_src_trg:
    i = 0
    for sentence_id in tqdm(s_id):
        sentence_text = d_zeroid2sentence[sentence_id]
        entities = d_s2e[sentence_id]
        for entity, ent_type in entities:
            sampled_ngram = qg.whxx_ngram_table.rand_sample_ngram(rng.np, ent_type)
            generated_question_text = qg._generate_template_wba(sentence_text, entity, sampled_ngram)
            
            X_question = generated_question_text
            parsed_question = nlp(X_question)
            question_tokens = [[token.text,token.idx] for token in parsed_question]
            
            X_answer = entity
            
            # context and context_tokens
            X_context = d_sentence2context[sentence_id]
            parsed_context = nlp(X_context)
#             [[token.text,token.i] for token in parsed_context]
#             [(u'This', 0), (u'is', 1), (u'my', 2), (u'sentence', 3)]
            context_tokens = [[token.text,token.idx] for token in parsed_context]
#             [(u'This', 0), (u'is', 5), (u'my', 8), (u'sentence', 11)]
            
            found_same_ent = False
            for ent in parsed_context.ents:
                if ent.text == entity:
                    char_spans = [[ent.start_char, ent.end_char-1]]
                    token_spans = [[ent.start, ent.end-1]]
                    found_same_ent = True
                    break
            # Detected_answers
            if found_same_ent:
                X_detected_answers = [{"text": entity, "char_spans": char_spans, "token_spans": token_spans}]

                pattern = [{'TEXT': X_answer}]

                qas = []
                qas_dict = [{"answers": entity, "question": X_question, "id": i, "qid": i, "question_tokens": question_tokens, "detected_answers": X_detected_answers}]
                output_rec = {"id": "", "context": X_context, "qas": qas_dict, "context_tokens": context_tokens}
    #             X_bart_src = X_question + ". Answer: " + mask_token + ". " + X_context 
    #             X_bart_target = X_question + " " + X_answer
    #             X_bart_src_trg = {'src': X_bart_src, 'trg': X_bart_target}
                output_rec = json.dumps(output_rec)
                f_train_X_src_trg.write(output_rec + '\n')


                aug_question_text.append(generated_question_text)
        
        i += 1


# In[ ]:


len(aug_question_text)


# In[ ]:


len(IndependentNodes_id)


# In[ ]:


len(s_id)


# In[ ]:




