+++
title="Named Entity Recognition using Apache OpenNLP. Customization Model: Data Preparation."
date=2020-11-10

[taxonomies]
categories = ["Programming"]
tags = ["python", "static typing"]

[extra]
toc = true

+++

<h1>Named Entity Recognition using Apache OpenNLP. Customization Model: Data Preparation.</h1>
When we are talking about NER, we are talking about one sub-task of a NLP(Natural Language Processing). 
<br><br><b>Why NLP?</b><br>
 Natural language processing is a component of Text mining that performs a special kind of linguistic analysis that essentially helps a machine
“read” text. So this part of Data Science provides methods and approaches to mining patterns in texts.
Thus NER just one in the list of NLP tasks.<br><br>
NER has a lot of synonyms, such as Entity Identification, Entity Chunking and Entity Extraction. 
<br>
<b>Named Entity Recognition</b> is a process of finding named-entities (entities, valuable for some special domain) in the text.
Standard kinds of entities looking for (using NER) in texts are:
- Person;
- Organization;
- Location.
<br>
<i>Let's clarify some common terms using in NLP:</i><br>

* Gazetteres (List of Seeds, Seeds, Named Entity Lists): list of domain entities. 
* Corpus (Corpora): a huge natural language text representing special domain.
* Annotation (Markup process): process of entities marking-up in a corpus.
* OpenNLP learning process: training/testing process using annotated corpus.

<b>Apache OpenNLP has a standard for NLP learning process pipeline:</b>
1) Preparing List of Entities
2) Annotating corpus / Use OpenNLP corpus
3) Training OpenNLP Parameters
4) Training own model / Use pre-trained models for corresponding corpus
5) Testing model with relevant data

<b>1) Named-entity list preparations:</b>
There are a lot of examples how to use Apache OpenNLP used standard domainwith models by default. I would like to make the task more complex. In this case let's:
customize NER for medical domain and allow model to find medical entities in texts.
will try to lear model find special entities in German language. There is no pretrained German models for Apache OpenNL.

For the construction of a list of medical entities  I used MeSH open-sourced database (https://www.dimdi.de/dynamic/de/klassifikationen/weitere-klassifikationen-und-standards/mesh/index.html).
Medical Subject Headings (MeSH) DB – is a polyhierarchical, concept-based thesaurus for the cataloguing of book and media collections, indexing of databases and creation of search profiles in medical domain.
Terms are extracted in the separate file where each line contains just one seed.

<b>2) Corpus preparation:</b>
Corpus  is important step or NER is Prepare unlabelled corpus. For OpenNLP it has a strict syntax. 
Corpus Requierments:
- have lots of data. In OpenNLP documentation it is recommends a minimum of 15000 sentences.
- entities need to be surrounded by tags in format <START:entity-name> Entity <END>.  Entity name is label of entities you are learning model to looking for.
- add spaces between tags (START/END) and labeled data.
- if possible, use one label per model. Multiple labels are possible, but not recommende.
- each line contains just one sentence. 
- empty lines delimit documents.  

Next steps 3-5 will be described in the next article.
More detailed code you can found here: https://github.com/nnovakova/opennlp-annotator







