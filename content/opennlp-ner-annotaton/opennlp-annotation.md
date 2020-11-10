+++
title="Custom Named Entity Recognition using Apache OpenNLP. Data Preparaton."
date=2020-11-10

[taxonomies]
categories = ["Programming"]
tags = ["python", "static typing"]

[extra]
toc = true

<h1>Named Entity Recognition using Apache OpenNLP. Data Preparaton.</h1>

When we are talking about NER, we are talking about one sub-task of a NLP(Natural Language Processing). 
 
Why NLP? Natural language processing is a component of Text mining that performs a special kind of linguistic analysis that essentially helps a machine
“read” text. So this part of Data Science provides methods and approaches to mining patterns in texts.
Thus NER just one in the list of NLP tasks.

NER has a lot of synonyms, such as Entity Identification, Entity Chunking and Entity Extraction. 

<b>Named Entity Recognition</b> is a process of finding named-entities (entities, valuable for some special domain) in the text.

Standard kinds of entities looking for (using NER) in texts are:
- Person;
- Organisation;
- Location.

NER Terms:
* Gazetteres (List of Seeds, Seeds, Named Entity Lists): list of domain entities. 
* Corpus (Corpora): a huge natural language text representing special domain.
* Annotation (Markup process): process of entities marking-up in a corpus.
* OpenNLP learning process: training/testing process using annotated corpus.

Apache OpenNLP library.
Pros:
Cons:

Apache OpenNLP Learning process pipeline:
- Prepare List of Entities
- Annotate corpus / Use OpenNLP corpus
- Train OpenNLP Parametrs
- Train own model / Use pre-trained models for corresponding corpus
- Test model with relevant data

Let's customize NER for medical domain and allow model to find medical entities in texts.
Another aspect of our example that we will try to lear model find special entities in German language. 
For the formation of a list of medical entities used MeSH - 
Medical Subject Headings DB https://www.dimdi.de/dynamic/de/klassifikationen/weitere-klassifikationen-und-standards/mesh/  as polyhierarchical, concept-based thesaurus (keyword register) for the cataloguing of book and media collections, indexing of databases and creation of search profiles.

Input Data Preparation:

Named-entity list preparations. Lists of words/phrases from medical dictionaries (https://www.dimdi.de/dynamic/en/classifications/further-classifications-and-standards/mesh/index.html)
Prepare unlabelled corpus. This corpus  must provide a variety of medical texts: diversity must be obtained in addition to mere volume, since our specific aim is to represent the many different facets of medical language. We need to characterise this diversity by describing it along appropriate dimensions: origin, genre, domain, etc. These dimensions have to be documented precisely for each text sample. This documentation must be encoded formally, as meta-information included with each document, so that sub-corpora can be extracted as needed to study relevant families of document types. Finally, text contents must also be encoded in a uniform way independently of the many original formats of documents .
Train Specified model (German language+medical text). 
Validate trained model with MM DB cases.
Intersect classifier results with validated data. 

Corpus Requierments:

- entities need to be surrounded by tags in format <START:entity-name> Entity <END>. 
- add spaces between tags (START/END) and labeled data
- if possible, use one label per model. Multiple labels are possible, but not recommended
- have lots of data. Documentation recommends a minimum of 15000 sentences
- each line is a “sentence”. 
- empty lines delimit documents. 




