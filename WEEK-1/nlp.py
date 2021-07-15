from flask import Flask, url_for, request, render_template, jsonify, send_file, redirect, session
from flask_bootstrap import Bootstrap
import json
from nltk import text
import pytesseract
import os
import pandas as pd
import base64
from textblob import TextBlob
import matplotlib.pyplot as plt
from io import BytesIO
import random
import time
from PIL import Image
import argparse
import cv2
import imutils
import datetime
import io
import spacy

nlp = spacy.load("en_core_web_sm")

# Initialize App
app = Flask(__name__)
Bootstrap(app)

app.secret_key = b'_5#y2L"F4Q8z\n\xec]/'


@app.route('/')
def index():
    im = Image.open("logo.png")
    data = io.BytesIO()
    im.save(data, "PNG")
    encoded_img_data = base64.b64encode(data.getvalue())
    return render_template('upload.html', img_data=encoded_img_data.decode('utf-8'))


@app.route('/scanner', methods=['GET', 'POST'])
def scan_file():
    if request.method == 'POST':
        start_time = datetime.datetime.now()
        image_data = request.files['file'].read()
        scanned_text = pytesseract.image_to_string(Image.open(io.BytesIO(image_data)))
        print("Found data:", scanned_text)
        session['data'] = {
            "text": scanned_text,
            "time": str((datetime.datetime.now() - start_time).total_seconds())
        }

        return redirect(url_for('analyze'))


@app.route('/process', methods=['GET', 'POST'])
def process():
    start = time.time()
    im = Image.open("logo.png")
    data = io.BytesIO()
    im.save(data, "PNG")
    encoded_img_data = base64.b64encode(data.getvalue())
    if request.method == 'POST':
        choice = request.form['taskoption']
        if "data" in session:
            data = session['data']
            rawtext = data["text"]
            docx = nlp(rawtext)
            custom_tokens = [token.text for token in docx]
            custom_wordinfo = [(token.text, token.lemma_, token.shape_, token.is_alpha, token.is_stop) for token in docx]
            custom_postagging = [(word.text, word.tag_, word.pos_, word.dep_) for word in docx]
            custom_namedentities = [(entity.text, entity.label_) for entity in docx.ents]
            lst = []
            for i in custom_namedentities:
                i = i[1]
                if i not in lst:
                    lst.append(i)
            custom = lst

            blob = TextBlob(rawtext)
            blob_sentiment, blob_subjectivity = blob.sentiment.polarity, blob.sentiment.subjectivity
            allData = [('"Token":"{}","Tag":"{}","POS":"{}","Dependency":"{}","Lemma":"{}","Shape":"{}","Alpha":"{}","IsStopword":"{}"'.format(
                        token.text, token.tag_, token.pos_, token.dep_, token.lemma_, token.shape_, token.is_alpha,
                        token.is_stop)) for token in docx]

            result_json = json.dumps(allData, sort_keys=False, indent=2)

        d = []
        
        for ent in docx.ents:
            d.append((ent.label_, ent.text))
            df = pd.DataFrame(d, columns=('named entity', 'output'))
            ORG_named_entity = df.loc[df['named entity'] == 'ORG']['output']
            PERSON_named_entity = df.loc[df['named entity'] == 'PERSON']['output']
            GPE_named_entity = df.loc[df['named entity'] == 'GPE']['output']
            MONEY_named_entity = df.loc[df['named entity'] == 'MONEY']['output']
            NATIONALITY_named_entity = df.loc[df['named entity'] == 'NORP']['output']
            BUILDING_named_entity = df.loc[df['named entity'] == 'FAC']['output']
            LOCATION_named_entity = df.loc[df['named entity'] == 'LOC']['output']
            PRODUCT_named_entity = df.loc[df['named entity'] == 'PRODUCT']['output']
            EVENT_named_entity = df.loc[df['named entity'] == 'EVENT']['output']
            WOA_named_entity = df.loc[df['named entity'] == 'WORK_OF_ART']['output']
            LAW_named_entity = df.loc[df['named entity'] == 'LAW']['output']
            LANGUAGE_named_entity = df.loc[df['named entity'] == 'LANGUAGE']['output']
            DATE_named_entity = df.loc[df['named entity'] == 'DATE']['output']
            TIME_named_entity = df.loc[df['named entity'] == 'TIME']['output']
            PERCENT_named_entity = df.loc[df['named entity'] == 'PERCENT']['output']
            CARDINAL_named_entity = df.loc[df['named entity'] == 'CARDINAL']['output']
            ORDINAL_named_entity = df.loc[df['named entity'] == 'ORDINAL']['output']
            QUANTITY_named_entity = df.loc[df['named entity'] == 'QUANTITY']['output']

        if choice == 'ORG':
            results = ORG_named_entity
            num_of_results = len(results)
        elif choice == 'PERSON':
            results = PERSON_named_entity
            num_of_results = len(results)
        elif choice == 'GPE':
            results = GPE_named_entity
            num_of_results = len(results)
        elif choice == 'MONEY':
            results = MONEY_named_entity
            num_of_results = len(results)
        elif choice == 'NORP':
            results = NATIONALITY_named_entity
            num_of_results = len(results)
        elif choice == 'FAC':
            results = BUILDING_named_entity
            num_of_results = len(results)
        elif choice == 'LOC':
            results = LOCATION_named_entity
            num_of_results = len(results)
        elif choice == 'PRODUCT':
            results = PRODUCT_named_entity
            num_of_results = len(results)
        elif choice == 'EVENT':
            results = EVENT_named_entity
            num_of_results = len(results)
        elif choice == 'WORK_OF_ART':
            results = WOA_named_entity
            num_of_results = len(results)
        elif choice == 'LAW':
            results = LAW_named_entity
            num_of_results = len(results)
        elif choice == 'LANGUAGE':
            results = LANGUAGE_named_entity
            num_of_results = len(results)
        elif choice == 'DATE':
            results = DATE_named_entity
            num_of_results = len(results)
        elif choice == 'TIME':
            results = TIME_named_entity
            num_of_results = len(results)
        elif choice == 'PERCENT':
            results = PERCENT_named_entity
            num_of_results = len(results)
        elif choice == 'CARDINAL':
            results = CARDINAL_named_entity
            num_of_results = len(results)
        elif choice == 'ORDINAL':
            results = ORDINAL_named_entity
            num_of_results = len(results)
        elif choice == 'QUANTITY':
            results = QUANTITY_named_entity
            num_of_results = len(results)
    end = time.time()
    final_time = end - start
    return render_template('index.html', num_of_results=num_of_results,results = results, choice = choice,ctext=rawtext, custom_tokens=custom_tokens,
                               custom_postagging=custom_postagging, custom_namedentities=custom_namedentities,
                               custom_wordinfo=custom_wordinfo, blob_sentiment=blob_sentiment,final_time=final_time,
                               blob_subjectivity=blob_subjectivity, result_json=result_json,
                               custom=custom,time=data["time"],text=data["text"],words=len(data["text"].split(" ")), img_data=encoded_img_data.decode('utf-8'))


@app.route('/analyze')#, methods=['GET', 'POST'])
def analyze():
    start = time.time()
    im = Image.open("logo.png")
    data = io.BytesIO()
    im.save(data, "PNG")
    encoded_img_data = base64.b64encode(data.getvalue())
    if "data" in session:
        data = session['data']
        rawtext = data["text"]
        docx = nlp(rawtext)
        custom_tokens = [token.text for token in docx]
        custom_wordinfo = [(token.text, token.lemma_, token.shape_, token.is_alpha, token.is_stop) for token in docx]
        custom_postagging = [(word.text, word.tag_, word.pos_, word.dep_) for word in docx]
        custom_namedentities = [(entity.text, entity.label_) for entity in docx.ents]
        lst = []
        for i in custom_namedentities:
            i = i[1]
            if i not in lst:
                lst.append(i)
        custom = lst

        blob = TextBlob(rawtext)
        blob_sentiment, blob_subjectivity = blob.sentiment.polarity, blob.sentiment.subjectivity
        allData = [('"Token":"{}","Tag":"{}","POS":"{}","Dependency":"{}","Lemma":"{}","Shape":"{}","Alpha":"{}","IsStopword":"{}"'.format(
                   token.text, token.tag_, token.pos_, token.dep_, token.lemma_, token.shape_, token.is_alpha,
                   token.is_stop)) for token in docx]

        result_json = json.dumps(allData, sort_keys=False, indent=2)

        end = time.time()
        final_time = end - start

        return render_template('index.html', ctext=rawtext, custom_tokens=custom_tokens,
                               custom_postagging=custom_postagging, custom_namedentities=custom_namedentities,
                               custom_wordinfo=custom_wordinfo, blob_sentiment=blob_sentiment,
                               blob_subjectivity=blob_subjectivity, final_time=final_time, result_json=result_json,
                               custom=custom,time=data["time"],text=data["text"],words=len(data["text"].split(" ")), img_data=encoded_img_data.decode('utf-8'))

if __name__ == '__main__':
    pytesseract.pytesseract.tesseract_cmd = r'D:\NCU\Danalitics\tesseract.exe'
    app.run(debug=True)