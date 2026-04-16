"""
╔══════════════════════════════════════════════════════════════╗
║         Automated Fake News Detection System                 ║
║         app.py  —  Flask Web Server (Final Version)         ║
╠══════════════════════════════════════════════════════════════╣
║  Run  : python app.py                                        ║
║  Open : http://127.0.0.1:5000                                ║
╚══════════════════════════════════════════════════════════════╝
"""

from flask import Flask, request, jsonify, render_template
from model import preprocess_text, extract_features, demo_predict, predict
import pickle
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)

# ══════════════════════════════════════════════════════════════
# LOAD TRAINED MODEL
# ══════════════════════════════════════════════════════════════

tfidf        = None
clf          = None
MODEL_LOADED = False

for prefix in ['fake_news_model', 'truthscan_model']:
    try:
        with open(f'{prefix}_tfidf.pkl', 'rb') as f:
            tfidf = pickle.load(f)
        with open(f'{prefix}_clf.pkl', 'rb') as f:
            clf = pickle.load(f)
        MODEL_LOADED = True
        print(f"  ✅ Model loaded: {prefix}_*.pkl")
        break
    except FileNotFoundError:
        continue

if not MODEL_LOADED:
    print("  ⚠️  No model found — using rule-based fallback")
    print("      Run: python model.py  to train first")


# ══════════════════════════════════════════════════════════════
# BUILD API RESPONSE
# ══════════════════════════════════════════════════════════════

def build_response(text, verdict, confidence, real_prob=None, fake_prob=None):
    try:
        feats = extract_features(text)
    except Exception:
        feats = {
            'caps_ratio': 0, 'exclamation_count': 0, 'question_count': 0,
            'word_count': 0, 'avg_word_len': 0, 'fake_keyword_count': 0,
            'credibility_keyword_count': 0, 'type_token_ratio': 0, 'sentence_count': 0
        }

    rp = real_prob if real_prob is not None else round(100 - confidence, 1)
    fp = fake_prob if fake_prob is not None else round(confidence, 1)

    # Build signals list
    signals, signal_types = [], []
    if feats.get('caps_ratio', 0) > 0.12:
        signals.append("High CAPS ratio ({:.1f}%)".format(feats['caps_ratio'] * 100))
        signal_types.append('negative')
    if feats.get('exclamation_count', 0) > 2:
        signals.append("{} exclamation marks".format(feats['exclamation_count']))
        signal_types.append('negative')
    if feats.get('question_count', 0) > 3:
        signals.append("{} rhetorical questions".format(feats['question_count']))
        signal_types.append('negative')
    if feats.get('fake_keyword_count', 0) > 0:
        signals.append("{} clickbait keyword(s) detected".format(feats['fake_keyword_count']))
        signal_types.append('negative')
    if feats.get('credibility_keyword_count', 0) > 1:
        signals.append("{} credibility indicator(s) found".format(feats['credibility_keyword_count']))
        signal_types.append('positive')
    if feats.get('word_count', 0) > 100:
        signals.append("Detailed article ({} words)".format(feats['word_count']))
        signal_types.append('positive')
    elif 0 < feats.get('word_count', 0) < 40:
        signals.append("Very short content ({} words)".format(feats['word_count']))
        signal_types.append('warning')
    if feats.get('avg_word_len', 0) > 5.5:
        signals.append("Formal academic vocabulary")
        signal_types.append('positive')
    if feats.get('type_token_ratio', 0) > 0.70:
        signals.append("Rich vocabulary diversity")
        signal_types.append('positive')
    if not signals:
        signals.append("No strong signals detected")
        signal_types.append('warning')

    # Build verdict-specific content
    if verdict == 'FAKE':
        risk          = 'CRITICAL' if confidence > 90 else ('HIGH' if confidence > 75 else 'MEDIUM')
        writing_style = 'Sensationalist & Manipulative'
        short_verdict = 'Strong misinformation signals detected.'
        explanation   = (
            "The Automated Fake News Detection system classified this as FAKE NEWS "
            "with {:.1f}% confidence using a hybrid ML + linguistic rules approach. "
            "Real probability: {:.1f}%  |  Fake probability: {:.1f}%. "
            "The content shows {} clickbait or conspiracy keyword(s), "
            "a CAPS ratio of {:.1f}%, and {} exclamation mark(s). "
            "These are strong indicators of misinformation and emotional manipulation."
        ).format(
            confidence, rp, fp,
            feats.get('fake_keyword_count', 0),
            feats.get('caps_ratio', 0) * 100,
            feats.get('exclamation_count', 0)
        )
        recommendation = (
            "⚠️ Do NOT share this content without verification. "
            "Search for the same story on BBC, Reuters, AP News, or FactCheck.org. "
            "If reputable outlets have not reported it, treat it as misinformation."
        )

    elif verdict == 'REAL':
        risk          = 'LOW' if confidence > 75 else 'MEDIUM'
        writing_style = 'Professional & Credible'
        short_verdict = 'Content shows strong credibility signals.'
        explanation   = (
            "The Automated Fake News Detection system classified this as REAL NEWS "
            "with {:.1f}% confidence using a hybrid ML + linguistic rules approach. "
            "Real probability: {:.1f}%  |  Fake probability: {:.1f}%. "
            "The article contains {} credibility indicator(s) such as named sources, "
            "statistics, institutional citations, or research references. "
            "The writing style is professional, measured, and evidence-based."
        ).format(
            confidence, rp, fp,
            feats.get('credibility_keyword_count', 0)
        )
        recommendation = (
            "✅ This content appears credible based on ML analysis. "
            "Always verify the author name and publication date. "
            "Trusted outlets: BBC, Reuters, AP News, The Hindu, The Guardian."
        )

    else:
        risk          = 'MEDIUM'
        writing_style = 'Mixed or Ambiguous'
        short_verdict = 'Cannot be confidently classified — verify independently.'
        explanation   = (
            "The system returned an UNCERTAIN verdict "
            "(Real: {:.1f}%  |  Fake: {:.1f}%). "
            "The content shows a mix of credibility indicators and potential red flags. "
            "This may be opinion content, satire, or a story lacking sufficient "
            "context for reliable automated classification."
        ).format(rp, fp)
        recommendation = (
            "🔍 Verify this story through at least two independent trusted sources. "
            "Check Snopes, PolitiFact, or FactCheck.org for related fact-checks."
        )

    return {
        'verdict':        str(verdict),
        'confidence':     float(confidence),
        'riskLevel':      str(risk),
        'writingStyle':   str(writing_style),
        'shortVerdict':   str(short_verdict),
        'signals':        list(signals),
        'signalTypes':    list(signal_types),
        'explanation':    str(explanation),
        'recommendation': str(recommendation),
        'features': {
            'wordCount':        int(feats.get('word_count', 0)),
            'capsRatio':        round(float(feats.get('caps_ratio', 0)) * 100, 1),
            'exclamations':     int(feats.get('exclamation_count', 0)),
            'fakeKeywords':     int(feats.get('fake_keyword_count', 0)),
            'credibilityScore': int(feats.get('credibility_keyword_count', 0)),
        },
        'modelType': 'Hybrid ML + Rules (LR + TF-IDF, WELFake 96%)' if MODEL_LOADED else 'Rule-Based Only'
    }


# ══════════════════════════════════════════════════════════════
# ROUTES
# ══════════════════════════════════════════════════════════════

@app.route('/')
def home():
    return render_template('index.html')


@app.route('/analyse', methods=['POST'])
def analyse():
    try:
        data       = request.get_json(force=True, silent=True) or {}
        text       = str(data.get('text', '')).strip()
        input_type = str(data.get('type', 'text'))

        if not text:
            return jsonify({'error': 'No input provided.'}), 400

        # URL mode — fetch article text
        if input_type == 'url':
            try:
                import requests as req
                from bs4 import BeautifulSoup
                resp       = req.get(text, headers={'User-Agent': 'Mozilla/5.0'}, timeout=8)
                soup       = BeautifulSoup(resp.text, 'html.parser')
                paragraphs = soup.find_all('p')
                fetched    = ' '.join(p.get_text() for p in paragraphs[:25])
                if len(fetched.strip()) > 100:
                    text = fetched
            except Exception as e:
                print(f"URL fetch failed: {e}")

        # Run prediction
        real_prob, fake_prob = None, None
        if MODEL_LOADED:
            result     = predict(text, tfidf, clf)
            verdict    = result['verdict']
            confidence = result['confidence']
            real_prob  = result.get('real_prob')
            fake_prob  = result.get('fake_prob')
        else:
            result     = demo_predict(text)
            verdict    = result['verdict']
            confidence = result['confidence']
            real_prob  = result.get('real_score')
            fake_prob  = result.get('fake_score')

        return jsonify(build_response(text, verdict, confidence, real_prob, fake_prob))

    except Exception as e:
        import traceback
        traceback.print_exc()
        # Never crash the UI — always return valid JSON
        return jsonify({
            'verdict':        'UNCERTAIN',
            'confidence':     0.0,
            'riskLevel':      'MEDIUM',
            'writingStyle':   'Unknown',
            'shortVerdict':   'Server error — please try again.',
            'signals':        ['An error occurred on the server'],
            'signalTypes':    ['warning'],
            'explanation':    f'Error: {str(e)}. Please restart the server.',
            'recommendation': 'Run python app.py again and retry.',
            'features':       {'wordCount':0,'capsRatio':0,'exclamations':0,'fakeKeywords':0,'credibilityScore':0},
            'modelType':      'Error'
        }), 200


@app.route('/status')
def status():
    return jsonify({
        'status':      'running',
        'modelLoaded': MODEL_LOADED,
        'modelType':   'Hybrid ML + Rules' if MODEL_LOADED else 'Rule-Based Only',
        'project':     'Automated Fake News Detection System'
    })


# ══════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════

if __name__ == '__main__':
    import os
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
    