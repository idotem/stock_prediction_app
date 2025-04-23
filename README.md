# 10-K Report Analyzer with GraphRAG

![Project Screenshot](static/img.png)

## Overview

This application allows users to download and analyze SEC 10-K reports for public companies using GraphRAG from Microsoft. Users can ask natural language questions about financial reports and receive AI-generated responses powered by OpenAI's 4o-mini model (you can use other models of course).

## Features

- **Document Retrieval**: Download 10-K reports for any publicly traded company
- **GraphRAG Indexing**: Process documents using Graph Retrieval Augmented Generation for enhanced context awareness
- **Interactive Q&A**: Ask questions about company financials, risks, and business operations
- **AI-Powered Analysis**: Leverages OpenAI's 4o-mini model to generate comprehensive responses

## Technology Stack

- **Backend**: Django
- **NLP Processing**: GraphRAG for document indexing and context retrieval
- **AI Model**: OpenAI 4o-mini for response generation

## How It Works

1. Users select a company and download its 10-K report
2. After downloading all the wanted reports, the user clicks the index graphrag button, and the app indexes the newly downloaded reports.
3. Users can ask questions in natural language
4. The GraphRAG system retrieves relevant context from the indexed document
5. OpenAI's 4o-mini model generates comprehensive responses based on the retrieved context
6. Use the references with [graphrag-visualizer](https://github.com/noworneverev/graphrag-visualizer) integration to see the retrieved context in more detail in the graph's tables.

## Getting Started

### Prerequisites

- Python 3.11+
- Django
- Required Python packages (see requirements.txt)
- OpenAI API access (or you can configure graphrag yourself with whatever model you see fit)

### Installation

1. Clone this repository
2. Install dependencies:
   ```bash
   pip install -r static/requirements.txt
   ```
3. Configure your OpenAI API key
4. Start the Django server:
   ```bash
   python manage.py runserver
   ```