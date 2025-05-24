import os

from django.shortcuts import render

from .download_10k_from_ticker import download_10k_from_ticker
from .download_stock_tickers import download_nasdaq_symbols, parse_symbols, download_other_exchanges_symbols
from .get_indexed_graphrag_docs import get_indexed_graphrag_docs
from .graphrag_chat import answer_question
from .index_graphrag_from_web import move_downloaded_files_and_index
from .open_text_file import open_text_file


def default_graphrag(request):
    return render(request, "chat.html")


def ask_question(request):
    question = request.GET.get("question")
    if question:
        context_success = {"answer": answer_question(question)}
        return render(request, "answer.html", context_success)
    else:
        context_err = {"error": "You submitted an empty question. Submit a valid one!"}
        return render(request, "error.html", context_err)


def get_tickers_context():
    file_path = download_nasdaq_symbols()
    all_symbols = parse_symbols(file_path)
    file_path = download_other_exchanges_symbols()
    all_symbols.extend(parse_symbols(file_path))
    indexed_docs = get_indexed_graphrag_docs("graphrag-10k/input")
    next_docs_to_index = get_indexed_graphrag_docs("data/next_docs_to_index")
    context = {"tickers": all_symbols,
               "indexed_docs": indexed_docs,
               "next_docs_to_index": next_docs_to_index}
    return context


def get_tickers(request):
    context = get_tickers_context()
    return render(request, "side-bar.html", context)


def download_10k(request):
    if request.method == 'POST':
        ticker = request.POST.get('ticker')
        if ticker:
            download_10k_from_ticker(ticker)
            context = get_tickers_context()

            return render(request, 'side-bar.html', context)

    context = get_tickers_context()
    return render(request, "side-bar.html", context)


def index_graphrag(request):
    if request.method == 'POST':
        success = move_downloaded_files_and_index()
        # After processing, prepare data for the sidebar template
        print(f"Success index: {success}")
        context = get_tickers_context()
        # Return the rendered sidebar template
        return render(request, 'side-bar.html', context)
    context = get_tickers_context()
    return render(request, "side-bar.html", context)


def open_text_file_from_next_to_index(request, file_name):
    file_path = os.path.join('data', 'next_docs_to_index', file_name)
    return open_text_file(file_path, file_name)


def open_text_file_from_indexed(request, file_name):
    file_path = os.path.join('graphrag-10k', 'input', file_name)
    return open_text_file(file_path, file_name)
