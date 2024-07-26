from flask import Flask, render_template, request
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from difflib import SequenceMatcher
import html
from collections import Counter

app = Flask(__name__)

# Download required NLTK data
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)

def preprocess_text(text):
    # Tokenize the text into sentences
    sentences = sent_tokenize(text)
    
    # Initialize lemmatizer and stopwords
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
    
    # Preprocess each sentence
    processed_sentences = []
    for sentence in sentences:
        # Tokenize words and convert to lowercase
        words = word_tokenize(sentence.lower())
        
        # Remove stopwords and lemmatize
        words = [lemmatizer.lemmatize(word) for word in words if word.isalnum() and word not in stop_words]
        
        processed_sentences.append(' '.join(words))
    
    return processed_sentences, sentences

def get_synonyms(word):
    synonyms = set()
    for syn in wordnet.synsets(word):
        for lemma in syn.lemmas():
            synonyms.add(lemma.name().lower())
    return synonyms

def jaccard_similarity(set1, set2):
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    return intersection / union if union != 0 else 0

def lcs_similarity(s1, s2):
    matcher = SequenceMatcher(None, s1, s2)
    match = matcher.find_longest_match(0, len(s1), 0, len(s2))
    lcs_length = match.size
    max_length = max(len(s1), len(s2))
    return lcs_length / max_length if max_length > 0 else 0


def get_wordnet_pos(word):
    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dict = {"J": wordnet.ADJ, "N": wordnet.NOUN, "V": wordnet.VERB, "R": wordnet.ADV}
    return tag_dict.get(tag, wordnet.NOUN)

def preprocess_text(text):
    sentences = sent_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
    
    processed_sentences = []
    for sentence in sentences:
        words = word_tokenize(sentence.lower())
        words = [lemmatizer.lemmatize(word, get_wordnet_pos(word)) for word in words if word.isalnum() and word not in stop_words]
        processed_sentences.append(' '.join(words))
    
    return processed_sentences, sentences

def compare_texts(text1, text2):
    processed_sentences1, original_sentences1 = preprocess_text(text1)
    processed_sentences2, original_sentences2 = preprocess_text(text2)
    
    all_sentences = processed_sentences1 + processed_sentences2
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(all_sentences)
    
    similarity_matrix = cosine_similarity(tfidf_matrix, tfidf_matrix)
    
    matches = []
    total_chars = sum(len(sent) for sent in original_sentences1)
    matched_chars = 0
    
    for i, (proc_sent1, orig_sent1) in enumerate(zip(processed_sentences1, original_sentences1)):
        for j, (proc_sent2, orig_sent2) in enumerate(zip(processed_sentences2, original_sentences2)):
            cosine_sim = similarity_matrix[i][j + len(processed_sentences1)]
            jaccard_sim = jaccard_similarity(set(proc_sent1.split()), set(proc_sent2.split()))
            lcs_sim = lcs_similarity(proc_sent1, proc_sent2)
            
            # Check for synonym overlap
            words1 = set(proc_sent1.split())
            words2 = set(proc_sent2.split())
            synonym_overlap = sum(1 for w1 in words1 for w2 in words2 if w2 in get_synonyms(w1))
            synonym_sim = synonym_overlap / max(len(words1), len(words2)) if max(len(words1), len(words2)) > 0 else 0
            
            # Combine similarity scores
            combined_sim = (cosine_sim + jaccard_sim + lcs_sim + synonym_sim) / 4
            
            if combined_sim >= 0.36:
                if combined_sim >= 0.8:
                    color = 'dark_green'
                elif combined_sim >= 0.7:
                    color = 'medium_green'
                else:
                    color = 'light_green'
                matches.append((orig_sent1, orig_sent2, color, combined_sim))
                matched_chars += len(orig_sent1)
    
    similarity_percentage = (matched_chars / total_chars) * 100 if total_chars > 0 else 0
    
    return matches, similarity_percentage


def highlight_text_html(text, matches, is_text1=True):
    highlighted_text = text
    
    color_map = {
        'dark_green': '#00B050',
        'medium_green': '#92D050',
        'light_green': '#C6E0B4' 
    }
    
    for sent1, sent2, color, _ in matches:
        html_color = color_map[color]
        sent_to_replace = sent1 if is_text1 else sent2
        highlighted_text = highlighted_text.replace(
            sent_to_replace, 
            f'<span style="background-color: {html_color};">{html.escape(sent_to_replace)}</span>'
        )
    
    # Replace newlines with <br> tags to maintain line spacing
    highlighted_text = highlighted_text.replace('\n', '<br>')
    
    return highlighted_text

def generate_html_output(text1, text2, highlighted_text1, highlighted_text2, similarity_percentage, matches):
    html_content = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Plagiarism Detection Result</title>
        <style>
            body {{ font-family: Arial, sans-serif; line-height: 1.6; padding: 20px; }}
            h1 {{ color: #333; }}
            .text-container {{ display: flex; justify-content: space-between; }}
            .text-box {{ width: 48%; border: 1px solid #ccc; padding: 10px; margin-bottom: 20px; }}
            h2 {{ color: #444; }}
            .similarity {{ font-size: 1.2em; font-weight: bold; margin-bottom: 20px; }}
            .legend {{ margin-bottom: 20px; }}
            .legend-item {{ display: inline-block; margin-right: 20px; }}
            .color-box {{ display: inline-block; width: 20px; height: 20px; margin-right: 5px; vertical-align: middle; }}
        </style>
    </head>
    <body>
        <h1>Content Similarity Checker</h1>
        <div class="similarity">Similarity Percentage: {similarity_percentage:.2f}%</div>
        <div class="legend">
            <div class="legend-item">
                <span class="color-box" style="background-color: #00B050;"></span>
                High Similarity (>=0.8)
            </div>
            <div class="legend-item">
                <span class="color-box" style="background-color: #92D050;"></span>
                Medium Similarity (0.7-0.79)
            </div>
            <div class="legend-item">
                <span class="color-box" style="background-color: #C6E0B4;"></span>
                Low Similarity (0.5-0.69)
            </div>
        </div>
        <div class="text-container">
            <div class="text-box">
                <h2>Text 1</h2>
                <p>{highlighted_text1}</p>
            </div>
           <div class="text-box">
                <h2>Text 2</h2>
                <p>{highlighted_text2}</p>
            </div>
        </div>
        <h2>Detailed Matches</h2>
        <ul>
            {''.join(f'<li>Similarity: {sim:.2f} - Text 1: "{s1}" | Text 2: "{s2}"</li>' for s1, s2, _, sim in matches)}
        </ul>
    </body>
    </html>
    """
    return html_content






@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        text1 = request.form['text1']
        text2 = request.form['text2']
        
        matches, similarity_percentage = compare_texts(text1, text2)
        highlighted_text1 = highlight_text_html(text1, matches, is_text1=True)
        highlighted_text2 = highlight_text_html(text2, matches, is_text1=False)
        
        return render_template('result.html', 
                               similarity_percentage=similarity_percentage, 
                               highlighted_text1=highlighted_text1, 
                               highlighted_text2=highlighted_text2,
                               matches=matches)
    
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)