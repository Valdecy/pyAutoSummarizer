############################################################################

# Created by: Prof. Valdecy Pereira, D.Sc.
# UFF - Universidade Federal Fluminense (Brazil)
# email:  valdecy.pereira@gmail.com
#pyAutoSummarizer - An Extractive and Abstractive Summarization Library Powered with Artificial Intelligence

# Citation: 
# PEREIRA, V. (2023). Project: pyAutoSummarizer, GitHub repository: <https://github.com/Valdecy/pyAutoSummarizer>

############################################################################

# Required Libraries
import numpy as np
import openai 
import re 
import regex
import unicodedata 

try:
    import importlib.resources as pkg_resources
except ImportError:
    import importlib_resources as pkg_resources
from . import stws

from itertools import combinations
from sentence_transformers import SentenceTransformer 
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import T5ForConditionalGeneration, T5Tokenizer
from transformers import PegasusForConditionalGeneration, PegasusTokenizer

############################################################################

class summarization():
    def __init__(self, text, stop_words = ['en'], n_words = 0, n_chars = 0, lowercase = True, rmv_accents = True, rmv_special_chars = True, rmv_numbers = True, rmv_custom_words = [], verbose = False):
        self.p_sw     = stop_words
        self.p_lc     = lowercase
        self.p_ra     = rmv_accents
        self.p_rc     = rmv_special_chars 
        self.p_rn     = rmv_numbers
        self.p_rw     = rmv_custom_words
        self.p_vb     = verbose
        self.limit_w  = n_words
        self.limit_c  = n_chars
        self.full_txt = text
        pattern       = r'(?<!\b(?:Mr|Mrs|Ms|Dr|Prof|St|Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec|[A-Z]|[A-Z][a-z]|[a-z]))[.!?]\s+' # r'(?<=[.!?])\s+'
        if (self.limit_w <= 0 and self.limit_c <= 0):
            self.sentences = regex.split(pattern, text.strip()) 
            self.original  = regex.split(pattern, text.strip()) 
        else:
            self.sentences = self.split_sentences(words_per_sentence = self.limit_w, chars_per_sentence = self.limit_c)
            self.original  = self.split_sentences(words_per_sentence = self.limit_w, chars_per_sentence = self.limit_c)
        self.corpus = self.clear_text(self.sentences, stop_words = self.p_sw, lowercase = self.p_lc, rmv_accents = self.p_ra, rmv_special_chars = self.p_rc, rmv_numbers = self.p_rn, rmv_custom_words = self.p_rw)
        self.txt       = '. '.join(self.corpus)
        self.sentences = regex.split(pattern, self.txt)
        for i in range(len(self.sentences)-1, -1, -1):
            self.sentences[i] = self.sentences[i].replace('.', '')
            if (self.sentences[i] == ''):
                del self.sentences[i]
                del self.original[i]
        self.tokens        = []
        self.w_freq        = {}
        self.w_dist        = {}
        self.loaded_models = {}
        for i in range(0, len(self.sentences)):
            self.tokens.append(re.findall(r'\b\w+\b', self.sentences[i].lower()))
        self.vocabulary = set([token for s in self.tokens for token in s])
        self.all_tokens = [item for sublist in self.tokens for item in sublist]
        for token in self.vocabulary:
            self.w_freq[token] = self.all_tokens.count(token)
        total_w        = sum(self.w_freq.values()) 
        for token in self.vocabulary:
            self.w_dist[token] = self.w_freq[token]/total_w
   
    ##############################################################################

    # Function: Split Sentences
    def split_sentences(self, words_per_sentence = 7, chars_per_sentence = -1):
        words     = self.full_txt.split()
        num_words = len(words)
        if (chars_per_sentence > -1):
            words_per_sentence = num_words
        num_sentences = (num_words + words_per_sentence - 1) // words_per_sentence
        sentences     = []
        for i in range(0, num_sentences):
            start    = i * words_per_sentence
            end      = min((i + 1) * words_per_sentence, num_words)
            sentence = ' '.join(words[start:end])
            if (chars_per_sentence > -1 and len(sentence) > chars_per_sentence):
                sentence_parts = []
                part           = ''
                for word in sentence.split():
                    if (len(part) + len(word) + 1 > chars_per_sentence):
                        sentence_parts.append(part)
                        part = ''
                    part = part + (' ' if part else '') + word
                if (part):
                    sentence_parts.append(part)
                sentences.extend(sentence_parts)
            else:
                sentences.append(sentence)
        return sentences

    # Function: Show Summary
    def show_summary(self, rank, n = 3):
        idx       = np.argsort(rank)[::-1]
        self.summ = []
        if (n > len(rank)):
            n = len(rank)
        for i in range(0, n):
            sentence =  self.original[idx[i]]
            print(sentence)
            self.summ.append(sentence)
        self.summ = ' '.join(self.summ)
        return self.summ
    
    # Function: Text Pre-Processing
    def clear_text(self, corpus, stop_words = ['en'], lowercase = True, rmv_accents = True, rmv_special_chars = True, rmv_numbers = True, rmv_custom_words = [], verbose = False):
        sw_full = []
        # Lower Case
        if (lowercase == True):
            if (verbose == True):
                print('Lower Case: Working...')
            corpus = [str(x).lower().replace("’","'") for x in  corpus]
            if (verbose == True):
                print('Lower Case: Done!')
        # Remove Punctuation & Special Characters
        if (rmv_special_chars == True):
            if (verbose == True):
                print('Removing Special Characters: Working...')
            corpus = [re.sub(r"[^a-zA-Z0-9']+", ' ', i) for i in corpus]
            if (verbose == True):
                print('Removing Special Characters: Done!')
        # Remove Stopwords
        if (len(stop_words) > 0):
            for sw_ in stop_words: 
                if   (sw_ == 'ar' or sw_ == 'ara' or 'arabic'):
                    f_file = pkg_resources.open_text(stws, 'Stopwords-Arabic.txt', encoding = 'utf8')
                elif (sw_ == 'bn' or sw_ == 'ben' or 'bengali'):
                    f_file = pkg_resources.open_text(stws, 'Stopwords-Bengali.txt', encoding = 'utf8')
                elif (sw_ == 'bg' or sw_ == 'bul' or 'bulgarian'):
                    f_file = pkg_resources.open_text(stws, 'Stopwords-Bulgarian.txt', encoding = 'utf8')
                elif (sw_ == 'zh' or sw_ == 'chi' or 'chinese'):
                    f_file = pkg_resources.open_text(stws, 'Stopwords-Chinese.txt', encoding = 'utf8')
                elif (sw_ == 'cs' or sw_ == 'cze' or sw_ == 'ces' or 'czech'):
                    f_file = pkg_resources.open_text(stws, 'Stopwords-Czech.txt', encoding = 'utf8')
                elif (sw_ == 'en' or sw_ == 'eng' or 'english'):
                    f_file = pkg_resources.open_text(stws, 'Stopwords-English.txt', encoding = 'utf8')
                elif (sw_ == 'fi' or sw_ == 'fin' or 'finnish'):
                    f_file = pkg_resources.open_text(stws, 'Stopwords-Finnish.txt', encoding = 'utf8')
                elif (sw_ == 'fr' or sw_ == 'fre' or sw_ == 'fra' or 'french'):
                    f_file = pkg_resources.open_text(stws, 'Stopwords-French.txt', encoding = 'utf8')
                elif (sw_ == 'de' or sw_ == 'ger' or sw_ == 'deu' or 'german'):
                    f_file = pkg_resources.open_text(stws, 'Stopwords-German.txt', encoding = 'utf8')
                elif (sw_ == 'el' or sw_ == 'gre' or 'greek'):
                    f_file = pkg_resources.open_text(stws, 'Stopwords-Greek.txt', encoding = 'utf8')
                elif (sw_ == 'he' or sw_ == 'heb' or 'hebrew'):
                    f_file = pkg_resources.open_text(stws, 'Stopwords-Hebrew.txt', encoding = 'utf8')
                elif (sw_ == 'hi' or sw_ == 'hin' or 'hind'):
                    f_file = pkg_resources.open_text(stws, 'Stopwords-Hind.txt', encoding = 'utf8')
                elif (sw_ == 'hu' or sw_ == 'hun' or 'hungarian'):
                    f_file = pkg_resources.open_text(stws, 'Stopwords-Hungarian.txt', encoding = 'utf8')
                elif (sw_ == 'it' or sw_ == 'ita' or 'italian'):
                    f_file = pkg_resources.open_text(stws, 'Stopwords-Italian.txt', encoding = 'utf8')
                elif (sw_ == 'ja' or sw_ == 'jpn' or 'japanese'):
                    f_file = pkg_resources.open_text(stws, 'Stopwords-Japanese.txt', encoding = 'utf8')
                elif (sw_ == 'ko' or sw_ == 'kor' or 'korean'):
                    f_file = pkg_resources.open_text(stws, 'Stopwords-Korean.txt', encoding = 'utf8')
                elif (sw_ == 'mr' or sw_ == 'mar' or 'marathi'):
                    f_file = pkg_resources.open_text(stws, 'Stopwords-Marathi.txt', encoding = 'utf8')
                elif (sw_ == 'fa' or sw_ == 'per' or sw_ == 'fas' or 'persian'):
                    f_file = pkg_resources.open_text(stws, 'Stopwords-Persian.txt', encoding = 'utf8')
                elif (sw_ == 'pl' or sw_ == 'pol' or 'polish'):
                    f_file = pkg_resources.open_text(stws, 'Stopwords-Polish.txt', encoding = 'utf8')
                elif (sw_ == 'pt-br' or sw_ == 'por-br' or 'portuguese-br'):
                    f_file = pkg_resources.open_text(stws, 'Stopwords-Portuguese-br.txt', encoding = 'utf8')
                elif (sw_ == 'ro' or sw_ == 'rum' or sw_ == 'ron' or 'romanian'):
                    f_file = pkg_resources.open_text(stws, 'Stopwords-Romanian.txt', encoding = 'utf8')
                elif (sw_ == 'ru' or sw_ == 'rus' or 'russian'):
                    f_file = pkg_resources.open_text(stws, 'Stopwords-Russian.txt', encoding = 'utf8')
                elif (sw_ == 'sk' or sw_ == 'slo' or 'slovak'):
                    f_file = pkg_resources.open_text(stws, 'Stopwords-Slovak.txt', encoding = 'utf8')
                elif (sw_ == 'es' or sw_ == 'spa' or 'spanish'):
                    f_file = pkg_resources.open_text(stws, 'Stopwords-Spanish.txt', encoding = 'utf8')
                elif (sw_ == 'sv' or sw_ == 'swe' or 'swedish'):
                    f_file = pkg_resources.open_text(stws, 'Stopwords-Swedish.txt', encoding = 'utf8')
                elif (sw_ == 'th' or sw_ == 'tha' or 'thai'):
                    f_file = pkg_resources.open_text(stws, 'Stopwords-Thai.txt', encoding = 'utf8')
                elif (sw_ == 'uk' or sw_ == 'ukr' or 'ukrainian'):
                    f_file = pkg_resources.open_text(stws, 'Stopwords-Ukrainian.txt', encoding = 'utf8')
                f_lines = f_file.read()
                sw      = f_lines.split('\n')
                sw      = list(filter(None, sw))
                sw_full.extend(sw)
            if (verbose == True):
                print('Removing Stopwords: Working...')
            for i in range(0, len(corpus)):
               text      = corpus[i].split()
               text      = [x.replace(' ', '') for x in text if x.replace(' ', '') not in sw_full]
               corpus[i] = ' '.join(text) 
               if (verbose == True):
                   print('Removing Stopwords: ' + str(i + 1) +  ' of ' + str(len(corpus)) )
            if (verbose == True):
                print('Removing Stopwords: Done!')
        # Remove Custom Words
        if (len(rmv_custom_words) > 0):
            if (verbose == True):
                print('Removing Custom Words: Working...')
            for i in range(0, len(corpus)):
               text      = corpus[i].split()
               text      = [x.replace(' ', '') for x in text if x.replace(' ', '') not in rmv_custom_words]
               corpus[i] = ' '.join(text) 
               if (verbose == True):
                   print('Removing Custom Words: ' + str(i + 1) +  ' of ' + str(len(corpus)) )
            if (verbose == True):
                print('Removing Custom Word: Done!')
        # Replace Accents 
        if (rmv_accents == True):
            if (verbose == True):
                print('Removing Accents: Working...')
            for i in range(0, len(corpus)):
                text = corpus[i]
                try:
                    text = unicode(text, 'utf-8')
                except NameError: 
                    pass
                text      = unicodedata.normalize('NFD', text).encode('ascii', 'ignore').decode('utf-8')
                corpus[i] = str(text)
            if (verbose == True):
                print('Removing Accents: Done!')
        # Remove Numbers
        if (rmv_numbers == True):
            if (verbose == True):
                print('Removing Numbers: Working...')
            corpus = [re.sub('[0-9]', ' ', i) for i in corpus] 
            if (verbose == True):
                print('Removing Numbers: Done!')
        for i in range(0, len(corpus)):
            corpus[i] = ' '.join(corpus[i].split())
        return corpus
    
    ##############################################################################
    
    # Function: Invert Rank    
    def invert_rank(self, rank):
        inverted_rank = [0] * len(rank)
        for i, r in enumerate(rank):
            inverted_rank[i] = len(rank) - 1 - r
        return inverted_rank
    
    # Function: Page Rank
    def page_rank(self, M, iteration = 1000, D = 0.85):
        N   = len(M)
        V   = np.full(N, 1 / N)
        V_  = V.copy() 
        M_N = ( M - M.min() ) / ( M.max() - M.min() + 0.00000000000000001)
        M_N = M_N/( M_N.sum(axis = 0) + 0.00000000000000001)
        M_T = ( D * M_N + (1 - D) / N )
        for i in range(0, iteration):
            V = np.dot(M_T, V)
            if ( np.isnan(V).any() or np.isinf(V).any() ):
                break
            else:
                V_ = V.copy()
        return V_
    
    ##############################################################################
    
    # Function: TF
    def tf_matrix(self):
        vectorizer = CountVectorizer()
        vectorizer.fit(self.sentences)
        tf_matrix  = vectorizer.transform(self.sentences)
        tf_m       = tf_matrix.toarray()
        tf_m       = tf_m/np.sum(tf_m, axis = 1).reshape(-1, 1)
        #vectorizer.vocabulary_
        #vectorizer.get_feature_names()
        return tf_m
    
    # Function: IDF
    def idf_matrix(self):
        vectorizer = TfidfVectorizer(use_idf = True)
        vectorizer.fit(self.sentences)
        idf_m      = vectorizer.transform(self.sentences)
        idf_m      = vectorizer.idf_
        #vectorizer.vocabulary_
        return idf_m
    
    # Function: TF-IDF
    def tf_idf_matrix(self):
        vectorizer    = TfidfVectorizer()
        vectorizer.fit(self.sentences)
        tf_idf_matrix = vectorizer.transform(self.sentences)
        tf_idf_m      = tf_idf_matrix.toarray()
        #vectorizer.vocabulary_
        #vectorizer.get_feature_names()
        return tf_idf_m

    ##############################################################################
    
    # Function: Rouge N 
    def rouge_N(self, generated_summary, reference_summary, n = 1):
        generated_summary = self.clear_text([generated_summary], stop_words = self.p_sw, lowercase = self.p_lc, rmv_accents = self.p_ra, rmv_special_chars = self.p_rc, rmv_numbers = self.p_rn, rmv_custom_words = self.p_rw)
        reference_summary = self.clear_text([reference_summary], stop_words = self.p_sw, lowercase = self.p_lc, rmv_accents = self.p_ra, rmv_special_chars = self.p_rc, rmv_numbers = self.p_rn, rmv_custom_words = self.p_rw)
        gen_tokens        = generated_summary[0].split()
        ref_tokens        = reference_summary[0].split()
        max_n             = min(len(gen_tokens), len(ref_tokens))
        if (n > max_n):
            n = max_n
        if (n > 1):
            gen_ngrams = set([tuple(gen_tokens[i:i + n]) for i in range(len(gen_tokens) - n + 1)]) 
            ref_ngrams = set([tuple(ref_tokens[i:i + n]) for i in range(len(ref_tokens) - n + 1)])
        else:
            gen_ngrams = set(gen_tokens)
            ref_ngrams = set(ref_tokens)
        if (len(gen_ngrams) == 0 or len(ref_ngrams) == 0):
            return 0, 0, 0
        overlap   = len(set(gen_ngrams).intersection(set(ref_ngrams)))
        precision = overlap / len(gen_ngrams) if len(gen_ngrams) > 0 else 0
        recall    = overlap / len(ref_ngrams) if len(ref_ngrams) > 0 else 0
        f1        = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        return f1, precision, recall

    ##############################################################################
    
    # Function: LCS
    def LCS(self, gen_tokens, ref_tokens):
        matrix = np.zeros((len(ref_tokens) + 1, len(gen_tokens) + 1))
        for i in range(1, len(ref_tokens) + 1):
            for j in range(1, len(gen_tokens) + 1):
                if (ref_tokens[i-1] == gen_tokens[j-1] ):
                    matrix[i,j] = matrix[i-1, j-1] + 1
                else:
                    matrix[i,j] = max(matrix[i-1, j], matrix[i, j-1])
        long_c_s = int(matrix[-1,-1])
        return long_c_s
        
    # Function: Rouge LCS
    def rouge_L(self, generated_summary, reference_summary):
        generated_summary = self.clear_text([generated_summary], stop_words = self.p_sw, lowercase = self.p_lc, rmv_accents = self.p_ra, rmv_special_chars = self.p_rc, rmv_numbers = self.p_rn, rmv_custom_words = self.p_rw)
        reference_summary = self.clear_text([reference_summary], stop_words = self.p_sw, lowercase = self.p_lc, rmv_accents = self.p_ra, rmv_special_chars = self.p_rc, rmv_numbers = self.p_rn, rmv_custom_words = self.p_rw)
        gen_tokens        = generated_summary[0].split()
        ref_tokens        = reference_summary[0].split()
        lcs_length        = self.LCS(gen_tokens, ref_tokens)
        precision         = lcs_length / len(gen_tokens) if len(gen_tokens) > 0 else 0
        recall            = lcs_length / len(ref_tokens) if len(ref_tokens) > 0 else 0
        f1                = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        return f1, precision, recall
    
    ##############################################################################
    
    # Function: Generate Skip Gram
    def generate_skip_bigrams(self, tokens, skip_distance):
        skip_bigrams = set()
        tokens       = self.clear_text([tokens], stop_words = self.p_sw, lowercase = self.p_lc, rmv_accents = self.p_ra, rmv_special_chars = self.p_rc, rmv_numbers = self.p_rn, rmv_custom_words = self.p_rw)
        tokens       = tokens[0].split()
        for i in range(0, len(tokens)):
            for j in range(i+1, min(i+1+skip_distance, len(tokens))):
                skip_bigrams.add((tokens[i], tokens[j]))
        return skip_bigrams
    
    # Function: ROUGE S
    def rouge_S(self, generated_summary, reference_summary, skip_distance = 4):
        g_skip_bigrams = self.generate_skip_bigrams(generated_summary, skip_distance = skip_distance)
        r_skip_bigrams = self.generate_skip_bigrams(reference_summary, skip_distance = skip_distance)
        overlap        = len(set(g_skip_bigrams).intersection(set(r_skip_bigrams)))
        total_g        = len(g_skip_bigrams)
        total_r        = len(g_skip_bigrams)
        precision      = overlap / total_g if total_g > 0 else 0
        recall         = overlap / total_r if total_r > 0 else 0
        f1             = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        return f1, precision, recall

    ##############################################################################
    
    # Function: Count N-Grams
    def count_ngrams(self, tokens, n):
        ngrams = {}
        if (n == 1):
            for i in range(0, len(tokens) - n + 1):
                ngram         = tokens[i:i+n][0]
                ngrams[ngram] = ngrams.get(ngram, 0) + 1
        else:
            for i in range(0, len(tokens) - n + 1):
                ngram         = tuple(tokens[i:i+n])
                ngrams[ngram] = ngrams.get(ngram, 0) + 1
        return ngrams
    
    # Function: Compute BLEU
    def compute_bleu(self, gen_tokens, ref_tokens, n):
        generated_ngrams = self.count_ngrams(gen_tokens, n)
        reference_ngrams = self.count_ngrams(ref_tokens, n)
        numerator        = sum(min(generated_ngrams[ngram], reference_ngrams.get(ngram, 0)) for ngram in generated_ngrams)
        denominator      = sum(generated_ngrams.values())
        c_bleu           = numerator / max(denominator, 1)
        return c_bleu
    
    # Function: BLEU Score
    def bleu(self, generated_summary, reference_summary, n = 4):
        generated_summary = self.clear_text([generated_summary], stop_words = self.p_sw, lowercase = self.p_lc, rmv_accents = self.p_ra, rmv_special_chars = self.p_rc, rmv_numbers = self.p_rn, rmv_custom_words = self.p_rw)
        reference_summary = self.clear_text([reference_summary], stop_words = self.p_sw, lowercase = self.p_lc, rmv_accents = self.p_ra, rmv_special_chars = self.p_rc, rmv_numbers = self.p_rn, rmv_custom_words = self.p_rw)
        gen_tokens        = generated_summary[0].split()
        ref_tokens        = reference_summary[0].split()
        bleu_n            = [self.compute_bleu(gen_tokens, ref_tokens, n + 1) for n in range(0, n)]
        gm                = [p for p in bleu_n if p != 0]
        gm                = np.exp(np.mean(np.log(gm)))   
        rt                = len(ref_tokens) / len(gen_tokens) if  len(gen_tokens) > 0 else 0
        bp                = min(1, np.exp(1 - rt)) # brevity penalty
        bleu_s            = bp * gm
        return bleu_s

    ##############################################################################
    
    # Function: Match Tokens
    def match_tokens(self, gen_tokens, ref_tokens):
        matches  = 0
        ref_dict = {}
        for token in ref_tokens:
            if (token in ref_dict):
                ref_dict[token] = ref_dict[token] + 1
            else:
                ref_dict[token] = 1
        for token in gen_tokens:
            if (token in ref_dict and ref_dict[token] > 0):
                matches         = matches + 1
                ref_dict[token] = ref_dict[token] - 1
        return matches
    
    # Function: Chunck Penalty
    def calculate_chunk_penalty(self, gen_tokens, ref_tokens, matches):
        candidate_chunks = set()
        reference_chunks = set()
        chunk_start      = None
        ref_chunk_start  = None
        ref_chunk_end    = None
        for i, candidate_word in enumerate(gen_tokens):
            if (candidate_word in ref_tokens):
                ref_index = ref_tokens.index(candidate_word)
                if (chunk_start is None):
                    chunk_start = i
                if (ref_chunk_start is None):
                    ref_chunk_start = ref_index
                ref_chunk_end = ref_index
            else:
                if (chunk_start is not None ):
                    candidate_chunks.add(tuple(range(chunk_start, i)))
                    chunk_start = None
                if (ref_chunk_start is not None and ref_chunk_end is not None):
                    reference_chunks.add(tuple(range(ref_chunk_start, ref_chunk_end+1)))
                    ref_chunk_start = None
                    ref_chunk_end = None
        if (chunk_start is not None):
            candidate_chunks.add(tuple(range(chunk_start, len(gen_tokens))))
        if (ref_chunk_start is not None and ref_chunk_end is not None):
            reference_chunks.add(tuple(range(ref_chunk_start, ref_chunk_end+1)))
        num_chunks    = len(candidate_chunks.intersection(reference_chunks))
        chunk_penalty = 1 - (num_chunks / matches)**0.5 if matches > 0 else 1
        return chunk_penalty
    
    # Function: METEOR Score
    def meteor(self, generated_summary, reference_summary, alpha = 0.9, beta = 3):
        generated_summary = self.clear_text([generated_summary], stop_words = self.p_sw, lowercase = self.p_lc, rmv_accents = self.p_ra, rmv_special_chars = self.p_rc, rmv_numbers = self.p_rn, rmv_custom_words = self.p_rw)
        reference_summary = self.clear_text([reference_summary], stop_words = self.p_sw, lowercase = self.p_lc, rmv_accents = self.p_ra, rmv_special_chars = self.p_rc, rmv_numbers = self.p_rn, rmv_custom_words = self.p_rw)
        gen_tokens        = generated_summary[0].split()
        ref_tokens        = reference_summary[0].split()
        matches           = self.match_tokens(gen_tokens, ref_tokens)
        precision         = matches / len(gen_tokens) if len(gen_tokens)> 0 else 0
        recall            = matches / len(ref_tokens) if len(ref_tokens)> 0 else 0
        if (precision == 0 or recall == 0):
            return 0
        fmean    = (precision * recall) / (alpha * precision + (1 - alpha) * recall)
        penalty  = self.calculate_chunk_penalty(gen_tokens, ref_tokens, matches)
        meteor_s = fmean * (1 - penalty**beta)
        return  meteor_s
    
        ##############################################################################
    
    # TextRank
    
    # Function: Sentence Embeddings
    def create_embeddings(self, model = 'all-MiniLM-L6-v2'):
        if (model not in self.loaded_models):
            self.loaded_models[model] = SentenceTransformer(model)
        #model_ = SentenceTransformer(model)
        #embds  = model_.encode(self.sentences)
        embds = self.loaded_models[model].encode(self.sentences)
        return embds
    
    # Function: TextRank
    def summ_text_rank(self, iteration = 1000, D = 0.85, model = 'all-MiniLM-L6-v2'):
        embeddings = self.create_embeddings(model = model)
        sim_matrix = cosine_similarity(embeddings)
        rank       = self.page_rank(sim_matrix, iteration = iteration, D = D)
        return rank

    ##############################################################################
    
    # LexRank
    
    # Function: LexRank
    def summ_lex_rank(self, iteration = 1000, D = 0.85):
        tf_idf     = self.tf_idf_matrix()
        sim_matrix = cosine_similarity(tf_idf + 0.1) 
        rank       = self.page_rank(sim_matrix, iteration = iteration, D = D)
        return rank 
    
    ##############################################################################
    
    # LSA
    
    # Function: LSA
    def summ_ext_LSA(self, embeddings = True, model = 'all-MiniLM-L6-v2'):
        if (embeddings == True):
            X = self.create_embeddings(model = model)
        else:
            X = self.tf_idf_matrix()
        U, S, Vt = np.linalg.svd(X, full_matrices = True)
        rank     = np.copy(U[:, 0])
        return rank

    ##############################################################################
    
    # KL Divergence  
    
    # Function: Candidate Summary Distribution
    def cs_distribution(self, idx, vocab):
        tokens = []
        for i in idx:
            tokens.append(self.sentences[i].split())
        tokens = [word for sentence in tokens for word in sentence]
        q      = [tokens.count(word)/len(tokens) if word in tokens else 0 for word in vocab]
        return q 

    # Function: KL Divergence
    def KL_divergence(self, p, q):
        pq = [(p_i, q_i) for p_i, q_i in zip(p, q) if p_i > 0 and q_i > 0]
        KL = [p_i * np.log2(p_i / q_i) for p_i, q_i in pq]
        KL = sum(KL)
        return KL
    
    # Function: KL
    def summ_ext_KL(self, n = 3):  
        best_summary = None
        min_KL       = float('inf')
        vocab        = sorted(self.vocabulary)
        p            = [self.w_dist[word] for word in vocab]
        for candidate_summaries in combinations(range(0, len(self.sentences)), n):
            idx = [i for i in candidate_summaries]
            q   = self.cs_distribution(idx, vocab)
            KL  = self.KL_divergence(p, q)
            if (KL < min_KL):
                min_KL       = KL
                best_summary = [item for item in idx]
        rank     = [1 if x in best_summary else 0 for x in range(0, len(self.sentences))]
        rank_sum = sum(rank)
        for i in range(0, len(rank)):
            if (rank[i] == 1):
                rank[i]  = rank_sum
                rank_sum = rank_sum - 1
        rank = np.array(rank)
        return rank

    ##############################################################################   

    # Function: T5
    def summ_ext_t5(self, model = 't5-base', min_len = 30, max_len = 250):
        tokenizer = T5Tokenizer.from_pretrained(model)
        t5_model  = T5ForConditionalGeneration.from_pretrained(model)
        inputs    = tokenizer.encode('summarize: ' + self.full_txt, return_tensors = 'pt', max_length = 512, truncation = True)
        outputs   = t5_model.generate(inputs, min_length = min_len, max_length = max_len, num_beams = 4, no_repeat_ngram_size = 2)
        summary   = tokenizer.decode(outputs[0], skip_special_tokens = True)
        self.summ = summary 
        print(summary)
        return summary

    ##############################################################################
    
    # Function: chatGPT
    def summ_abst_chatgpt(self, api_key = 'your_api_key_here', query = 'make an abstratctive summarization', model = 'text-davinci-003', max_tokens = 250, n = 1, temperature = 0.8):
        def query_chatgpt(prompt, model = model, max_tokens = max_tokens, n = n, temperature = temperature):
            response = openai.Completion.create(
                                                engine      = model,
                                                prompt      = prompt,
                                                max_tokens  = max_tokens,
                                                n           = n,
                                                stop        = None,
                                                temperature = temperature
                                                )
            return response.choices[0].text.strip()
        openai.api_key = api_key   
        prompt         = query + ':\n\n' + f'{self.full_txt}\n'
        summary        = query_chatgpt(prompt)
        self.summ      = summary 
        print(summary)
        return summary 
    
    # Function: PEGASUS
    def summ_abst_pegasus(self, text, model_name = 'google/pegasus-xsum', min_L = 100, max_L = 150):
        tokenizer = PegasusTokenizer.from_pretrained(model_name)
        pegasus   = PegasusForConditionalGeneration.from_pretrained(model_name)
        tokens    = tokenizer.encode('summarize: ' + text, return_tensors = 'pt', max_length = 1024, truncation = True)
        summary   = pegasus.generate(tokens, min_length = min_L, max_length = max_L, length_penalty = 2.0, num_beams = 4, early_stopping = True)
        summary   = tokenizer.decode(summary[0], skip_special_tokens = True)
        self.summ = summary 
        print(summary)
        return summary

    ##############################################################################   
