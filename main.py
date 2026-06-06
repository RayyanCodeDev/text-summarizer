import os
import re
import time
import traceback
import threading
import requests
from flask import Flask, render_template, request, jsonify
app = Flask(__name__, static_folder='static', template_folder='templates')
print('Loading model...')
_model = None
_tokenizer = None
try:
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
    _tokenizer = AutoTokenizer.from_pretrained('t5-small')
    _model = AutoModelForSeq2SeqLM.from_pretrained('t5-small')
    print('Model loaded.')
except Exception as e:
    print('Model failed:', e)
    traceback.print_exc()
def count_words(t):
    return len(t.split()) if t.strip() else 0
def clean_summary(t):
    t=t.strip()
    return (t[0].upper()+t[1:] if t else t)
@app.route('/')
def home():
    return render_template('index.html')
@app.route('/health')
def health():
    return jsonify({'status':'ok','model':'loaded' if _model else 'failed'})
@app.route('/summarize',methods=['POST'])
def summarize():
    try:
        if _model is None:
            return jsonify({'error':'Model not loaded.','summary':''}),503
        data=request.get_json(silent=True) or request.form or {}
        text=(data.get('paragraph') or data.get('text') or '').strip()
        if not text:
            return jsonify({'error':'No text','summary':''}),400
        words=text.split()
        if len(words)>400:
            text=' '.join(words[:400])
        ow=count_words(text)
        inputs=_tokenizer('summarize: '+text,return_tensors='pt',max_length=512,truncation=True)
        outputs=_model.generate(inputs['input_ids'],min_length=40,max_new_tokens=150,do_sample=False)
        raw=_tokenizer.decode(outputs[0],skip_special_tokens=True).strip()
        if not raw:
            return jsonify({'error':'Empty result.','summary':''}),500
        raw=clean_summary(raw)
        sw=count_words(raw)
        red=round(((ow-sw)/ow)*100) if ow>0 else 0
        warn='<p style="color:#ff9800;font-weight:bold;">⚠️ Summarizer can make mistakes. Verify important content yourself.</p>'
        return jsonify({'summary':warn+raw,'original_word_count':ow,'summary_word_count':sw,'reduction_percentage':red})
    except Exception as e:
        traceback.print_exc()
        return jsonify({'error':'Internal server error','summary':''}),500
if __name__=='__main__':
    port=int(os.environ.get('PORT',8000))
    app.run(host='0.0.0.0',port=port,debug=False)
